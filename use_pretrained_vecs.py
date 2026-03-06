"""
Load and apply pre-built steering vectors from HF Hub.
codelion/Qwen3-0.6B-pts-steering-vectors
"""
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
VEC_DATASET = "codelion/Qwen3-0.6B-pts-steering-vectors"


def load_hf_steering_vectors(dataset_id: str) -> list[dict]:
    ds = load_dataset(dataset_id, split="train")
    return [json.loads(ex["text"]) if isinstance(ex, dict) and "text" in ex
            else ex for ex in ds]


def apply_pretrained_vector(
    model, tokenizer, prompt: str,
    vec: torch.Tensor, layer_idx: int, alpha: float = 0.15,
    max_new_tokens: int = 200,
) -> str:
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs + alpha * vec.to(hs.device, hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    handle.remove()
    new_ids = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print("Loading pre-built vectors...")
    raw_vecs = load_hf_steering_vectors(VEC_DATASET)
    print(f"Found {len(raw_vecs)} vector records")

    # Inspect structure
    if raw_vecs:
        print("Keys:", list(raw_vecs[0].keys()) if isinstance(raw_vecs[0], dict) else type(raw_vecs[0]))
