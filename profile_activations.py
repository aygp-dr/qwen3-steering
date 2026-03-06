"""
Profile memory usage of activation capture and steering vector computation.
Shows peak tensor memory per operation.
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def profile_steering_vector_build(model, tokenizer, prompt: str, layer_idx: int):
    """Profile the full forward pass + activation capture."""

    activities = [ProfilerActivity.CPU]
    if torch.backends.mps.is_available():
        # M4 uses MPS (Metal Performance Shaders)
        activities.append(ProfilerActivity.CPU)  # MPS not yet a ProfilerActivity

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        captured = {}

        def hook_fn(module, input, output):
            with record_function("activation_capture"):
                hs = output[0] if isinstance(output, tuple) else output
                captured["hs"] = hs.detach().clone()

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

        with record_function("forward_pass"):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**inputs)

        handle.remove()

        with record_function("vector_mean"):
            vec = captured["hs"].squeeze(0).mean(dim=0)

    print("\n── PyTorch Profiler: Top ops by CPU time ──")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    print("\n── Memory by op ──")
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    return vec


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    vec = profile_steering_vector_build(
        model, tokenizer,
        prompt="Be extremely terse and technical.",
        layer_idx=15,
    )
    print(f"\nSteering vector shape: {vec.shape}, dtype: {vec.dtype}")
    print(f"Norm: {vec.norm():.4f}")
    print(f"Memory: {vec.element_size() * vec.numel() / 1024:.2f} KB")
