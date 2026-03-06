"""
Diagnose why α=0.20 produces zero effect on Qwen3-0.6B.

Measures:
1. Residual stream norms at each layer
2. Raw steering vector norms (before normalization)
3. Signal-to-noise ratio (steering magnitude / residual norm)
4. Alpha threshold where greedy output first diverges
5. Effect of sampling vs greedy at low alpha
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from actadd import STYLE_PAIRS, get_layer_activations, compute_steering_vector, generate_steered
import torch.nn.functional as F

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Explain what a mutex is."
STYLE = "terse"


def measure_residual_norms(model, tokenizer, prompt, layers):
    """Measure L2 norm of residual stream at each layer."""
    norms = {}
    for L in layers:
        captured = {}

        def hook_fn(module, input, output, layer=L):
            hs = output[0] if isinstance(output, tuple) else output
            captured["hs"] = hs.detach()

        handle = model.model.layers[L].register_forward_hook(hook_fn)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        # Mean norm across sequence positions
        hs = captured["hs"].squeeze(0)  # (seq_len, d_model)
        norms[L] = hs.norm(dim=-1).mean().item()
    return norms


def measure_raw_steering_norms(model, tokenizer, style, layers):
    """Measure raw (unnormalized) steering vector norms."""
    raw_norms = {}
    pos_prompt, neg_prompt = STYLE_PAIRS[style]
    for L in layers:
        act_pos = get_layer_activations(model, tokenizer, pos_prompt, L)
        act_neg = get_layer_activations(model, tokenizer, neg_prompt, L)
        raw_vec = act_pos - act_neg
        raw_norms[L] = raw_vec.norm().item()
    return raw_norms


def find_divergence_alpha(model, tokenizer, style, layer, prompt, alphas):
    """Find the alpha where greedy output first differs from baseline."""
    # Get baseline (unit-normalized vector, as current code does)
    vec_unit = compute_steering_vector(model, tokenizer, style, layer)
    baseline = generate_steered(model, tokenizer, prompt, vec_unit, layer, alpha=0.0, max_new_tokens=100)

    results = {}
    for a in alphas:
        out = generate_steered(model, tokenizer, prompt, vec_unit, layer, alpha=a, max_new_tokens=100)
        diverged = out != baseline
        results[a] = {
            "diverged": diverged,
            "words": len(out.split()),
            "preview": out[:80],
        }
        if diverged:
            break
    return baseline, results


def find_divergence_raw(model, tokenizer, style, layer, prompt, alphas):
    """Same but with raw (unnormalized) steering vector."""
    pos_prompt, neg_prompt = STYLE_PAIRS[style]
    act_pos = get_layer_activations(model, tokenizer, pos_prompt, layer)
    act_neg = get_layer_activations(model, tokenizer, neg_prompt, layer)
    raw_vec = act_pos - act_neg  # NOT normalized

    baseline = generate_steered(model, tokenizer, prompt, raw_vec, layer, alpha=0.0, max_new_tokens=100)

    results = {}
    for a in alphas:
        out = generate_steered(model, tokenizer, prompt, raw_vec, layer, alpha=a, max_new_tokens=100)
        diverged = out != baseline
        results[a] = {
            "diverged": diverged,
            "words": len(out.split()),
            "preview": out[:80],
        }
    return baseline, results


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    layers = [5, 10, 14, 15, 16, 18, 20, 24]

    # ── 1. Residual stream norms ──────────────────────────────────────────
    print("\n═══ 1. Residual Stream Norms ═══")
    res_norms = measure_residual_norms(model, tokenizer, PROMPT, layers)
    for L, norm in res_norms.items():
        print(f"  Layer {L:2d}: residual norm = {norm:.2f}")

    # ── 2. Raw steering vector norms ──────────────────────────────────────
    print("\n═══ 2. Raw Steering Vector Norms (before normalization) ═══")
    raw_norms = measure_raw_steering_norms(model, tokenizer, STYLE, layers)
    for L, norm in raw_norms.items():
        print(f"  Layer {L:2d}: raw vec norm = {norm:.4f}")

    # ── 3. Signal-to-noise ratio ──────────────────────────────────────────
    print("\n═══ 3. Signal-to-Noise Ratios ═══")
    print("  (a) With unit-normalized vec at α=0.20:")
    for L in layers:
        snr = 0.20 / res_norms[L] * 100
        print(f"    Layer {L:2d}: perturbation/residual = {snr:.4f}%")

    print("  (b) With raw vec at α=1.0:")
    for L in layers:
        snr = raw_norms[L] / res_norms[L] * 100
        print(f"    Layer {L:2d}: perturbation/residual = {snr:.2f}%")

    # ── 4. Alpha threshold (unit-normalized) ──────────────────────────────
    print("\n═══ 4. Alpha Threshold for Divergence (unit-normalized vec, layer 15) ═══")
    alphas_unit = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    baseline, unit_results = find_divergence_alpha(
        model, tokenizer, STYLE, 15, PROMPT, alphas_unit
    )
    print(f"  Baseline ({len(baseline.split())} words): {baseline[:80]}...")
    for a, r in unit_results.items():
        status = "DIVERGED" if r["diverged"] else "same"
        print(f"  α={a:5.1f}: {status:8s} | {r['words']:3d} words | {r['preview']}")

    # ── 5. Raw vector with various alphas ─────────────────────────────────
    print("\n═══ 5. Raw Vector (no normalization) at Various Alphas, layer 15 ═══")
    alphas_raw = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    baseline_raw, raw_results = find_divergence_raw(
        model, tokenizer, STYLE, 15, PROMPT, alphas_raw
    )
    raw_norm_15 = raw_norms[15]
    print(f"  Raw vec norm at layer 15: {raw_norm_15:.4f}")
    print(f"  Baseline ({len(baseline_raw.split())} words): {baseline_raw[:80]}...")
    for a, r in raw_results.items():
        effective_mag = a * raw_norm_15
        status = "DIVERGED" if r["diverged"] else "same"
        print(f"  α={a:4.1f} (eff. mag={effective_mag:6.2f}): {status:8s} | {r['words']:3d} words | {r['preview']}")

    # ── 6. Summary ────────────────────────────────────────────────────────
    print("\n═══ 6. Summary ═══")
    print(f"  Residual norm at layer 15: {res_norms[15]:.2f}")
    print(f"  Raw steering vec norm at layer 15: {raw_norms[15]:.4f}")
    print(f"  Unit vec + α=0.20: perturbation = 0.20, SNR = {0.20/res_norms[15]*100:.4f}%")
    print(f"  Raw vec + α=1.0: perturbation = {raw_norms[15]:.2f}, SNR = {raw_norms[15]/res_norms[15]*100:.2f}%")

    first_diverge_unit = next((a for a, r in unit_results.items() if r["diverged"]), None)
    first_diverge_raw = next((a for a, r in raw_results.items() if r["diverged"]), None)
    if first_diverge_unit:
        print(f"  First divergence (unit vec): α={first_diverge_unit}")
    else:
        print(f"  No divergence found with unit vec up to α={alphas_unit[-1]}")
    if first_diverge_raw:
        print(f"  First divergence (raw vec): α={first_diverge_raw}")
    else:
        print(f"  No divergence found with raw vec up to α={alphas_raw[-1]}")


if __name__ == "__main__":
    main()
