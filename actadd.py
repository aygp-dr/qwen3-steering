"""
ActAdd for Qwen3-0.6B: tone steering without system prompt.

Usage:
    python actadd.py --style terse --alpha 0.25 --layer 15 \
        --prompt "Explain what a mutex is."
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

MODEL_ID = "Qwen/Qwen3-0.6B"

# ── Style axis definitions ────────────────────────────────────────────────────
STYLE_PAIRS = {
    "terse": (
        # positive: dry, terse, technical — your actual preferred register
        "Be extremely concise and technical. No filler words. Dense signal.",
        # negative: verbose, padded
        "Please explain this thoroughly and helpfully with lots of context and examples.",
    ),
    "formal": (
        "Respond in precise, formal academic prose.",
        "Just chat with me casually.",
    ),
    "socratic": (
        "Respond only with targeted clarifying questions that expose hidden assumptions.",
        "Give me the answer directly.",
    ),
    "dry-wit": (
        "Respond with dry understatement and laconic precision. INTJ energy.",
        "Be enthusiastic, warm, and encouraging in your response.",
    ),
}

# ── Activation extraction ─────────────────────────────────────────────────────

def get_layer_activations(
    model, tokenizer, prompt: str, layer_idx: int
) -> torch.Tensor:
    """Return mean hidden state at layer_idx for the given prompt."""
    captured = {}

    def hook_fn(module, input, output):
        # output is (hidden_states, optional_cache, ...)
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    try:
        # Disable thinking mode: append empty <think></think> block
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    # Mean over sequence dimension → (d_model,)
    return captured["hs"].squeeze(0).mean(dim=0)


def compute_steering_vector(
    model, tokenizer, style: str, layer_idx: int,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute ActAdd steering vector for a named style.

    Returns the raw activation difference by default (Turner et al. 2023).
    The raw vector preserves natural scale relative to the residual stream,
    so α=1.0-3.0 produces meaningful effects. Normalizing to unit norm
    requires α in the hundreds to match residual stream magnitudes (~500).
    """
    pos_prompt, neg_prompt = STYLE_PAIRS[style]
    act_pos = get_layer_activations(model, tokenizer, pos_prompt, layer_idx)
    act_neg = get_layer_activations(model, tokenizer, neg_prompt, layer_idx)
    vec = act_pos - act_neg
    if normalize:
        return F.normalize(vec, dim=0)
    return vec


# ── Steered generation ────────────────────────────────────────────────────────

def generate_steered(
    model,
    tokenizer,
    prompt: str,
    steering_vec: torch.Tensor,
    layer_idx: int,
    alpha: float = 2.0,
    max_new_tokens: int = 256,
) -> str:
    """Generate with steering vector injected at layer_idx."""

    def steering_hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Inject: add α * vec to every token position
        hs = hs + alpha * steering_vec.to(hs.device, hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = model.model.layers[layer_idx].register_forward_hook(steering_hook)

    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    finally:
        handle.remove()

    new_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Layer sweep utility ───────────────────────────────────────────────────────

def sweep_layers(
    model, tokenizer, prompt: str, style: str,
    alpha: float = 2.0, layers: Optional[list] = None,
) -> dict[int, str]:
    """
    Try steering at multiple layers, return {layer: output}.
    Useful for identifying the sweet spot (typically 12–18 for 28-layer model).
    """
    if layers is None:
        layers = list(range(10, 22, 2))  # [10, 12, 14, 16, 18, 20]
    results = {}
    for L in layers:
        vec = compute_steering_vector(model, tokenizer, style, L)
        results[L] = generate_steered(model, tokenizer, prompt, vec, L, alpha)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="terse", choices=list(STYLE_PAIRS))
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep layers 10-20 and print comparison")
    parser.add_argument("--prompt", default="Explain what a mutex is.")
    parser.add_argument("--save-vec", type=str, default=None,
                        help="Save steering vector to .pt file")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    if args.sweep:
        print(f"\n── Layer sweep: style={args.style}, α={args.alpha} ──")
        results = sweep_layers(model, tokenizer, args.prompt, args.style, args.alpha)
        for L, text in results.items():
            print(f"\n[Layer {L}]")
            print(text)
        return

    vec = compute_steering_vector(model, tokenizer, args.style, args.layer)

    if args.save_vec:
        torch.save(vec, args.save_vec)
        print(f"Saved steering vector → {args.save_vec}")

    # Baseline (no steering)
    print("\n── Baseline ──")
    baseline = generate_steered(model, tokenizer, args.prompt, vec, args.layer, alpha=0.0)
    print(baseline)

    # Steered
    print(f"\n── Steered (style={args.style}, layer={args.layer}, α={args.alpha}) ──")
    steered = generate_steered(model, tokenizer, args.prompt, vec, args.layer, args.alpha)
    print(steered)


if __name__ == "__main__":
    main()
