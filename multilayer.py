"""
Multi-layer activation steering for Qwen3-0.6B.
Inject style vector at layers [L1, L2, ...] with independent α values.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from actadd import compute_steering_vector  # reuse from actadd.py

MODEL_ID = "Qwen/Qwen3-0.6B"


def generate_multilayer_steered(
    model, tokenizer, prompt: str,
    layer_vec_alpha: list[tuple[int, torch.Tensor, float]],
    max_new_tokens: int = 256,
) -> str:
    """
    layer_vec_alpha: list of (layer_idx, steering_vec, alpha) tuples.
    Each layer gets its own hook independently.
    """
    handles = []

    for layer_idx, vec, alpha in layer_vec_alpha:
        _vec = vec  # capture in closure

        def make_hook(_vec, _alpha):
            def hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs + _alpha * _vec.to(hs.device, hs.dtype)
                if isinstance(output, tuple):
                    return (hs,) + output[1:]
                return hs
            return hook

        h = model.model.layers[layer_idx].register_forward_hook(
            make_hook(_vec, alpha)
        )
        handles.append(h)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    for h in handles:
        h.remove()

    new_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    prompt = "Explain why formal verification matters."

    # Build two vectors at different layers for the same style axis
    vec_14 = compute_steering_vector(model, tokenizer, "terse", layer_idx=14)
    vec_17 = compute_steering_vector(model, tokenizer, "terse", layer_idx=17)

    result = generate_multilayer_steered(
        model, tokenizer, prompt,
        layer_vec_alpha=[
            (14, vec_14, 0.15),
            (17, vec_17, 0.15),
        ]
    )
    print(result)
