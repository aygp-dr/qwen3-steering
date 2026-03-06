"""
Style vector via corpus mean activations (Konen et al. EACL 2024).

Build a style vector from a small corpus of example outputs, then subtract
a neutral baseline. Works well with 10–30 examples per style.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"

# ── Example corpora ───────────────────────────────────────────────────────────

TERSE_EXAMPLES = [
    "Mutex: binary semaphore. One thread holds it; others block.",
    "O(n log n) average. O(n²) worst case. Use timsort instead.",
    "ZFS: copy-on-write, checksummed blocks. Not a RAID replacement.",
    "FreeBSD jail: kernel-level namespace isolation. Bastille automates provisioning.",
    "TLA+: temporal logic over states. Deadlocks are reachable states with no progress.",
    "Formal spec ≠ implementation. Gap is the adversary.",
    "Guile Scheme: tail calls are cheap. Trampolining is unnecessary.",
    "CPRR: Conjecture → Proof → Refutation → Refinement. Popperian.",
    "Meshtastic: LoRa mesh. AES-256, 255 byte packets, ~1km urban.",
    "ADS-B: 1090 MHz, position + velocity. No authentication. Trivially spoofed.",
]

NEUTRAL_EXAMPLES = [
    "I'd be happy to help explain that concept.",
    "That's a great question! Let me walk you through it step by step.",
    "Sure, here is a thorough explanation of what you asked about.",
    "Absolutely! This is actually a really interesting topic.",
    "Of course! Let me provide some helpful context first.",
    "Great question. There are several ways to think about this.",
    "I'll do my best to explain this clearly for you.",
    "Let me break this down in an easy-to-understand way.",
    "Thanks for asking! Here's a comprehensive answer.",
    "Sure thing! This topic has a few important aspects to cover.",
]


def corpus_mean_activation(
    model, tokenizer, texts: list[str], layer_idx: int
) -> torch.Tensor:
    """Mean hidden state at layer_idx over a list of texts."""
    means = []
    for text in texts:
        captured = {}

        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["hs"] = hs.detach()

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        means.append(captured["hs"].squeeze(0).mean(dim=0))

    return torch.stack(means).mean(dim=0)


def build_style_vectors(
    model, tokenizer, target_texts: list[str], neutral_texts: list[str],
    layer_range: range = range(10, 22),
) -> dict[int, torch.Tensor]:
    """Build normalised style vectors for each layer in layer_range."""
    vectors = {}
    for L in layer_range:
        target_mean = corpus_mean_activation(model, tokenizer, target_texts, L)
        neutral_mean = corpus_mean_activation(model, tokenizer, neutral_texts, L)
        vec = target_mean - neutral_mean
        vectors[L] = F.normalize(vec, dim=0)
        print(f"  Layer {L}: vec norm before norm = {(target_mean - neutral_mean).norm():.4f}")
    return vectors


def save_style_vectors(vectors: dict, path: str):
    """Save layer→vector dict as a .pt file."""
    torch.save({str(k): v for k, v in vectors.items()}, path)
    print(f"Saved {len(vectors)} style vectors → {path}")


def load_style_vectors(path: str) -> dict[int, torch.Tensor]:
    return {int(k): v for k, v in torch.load(path).items()}


if __name__ == "__main__":
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print("Building terse style vectors...")
    vectors = build_style_vectors(
        model, tokenizer, TERSE_EXAMPLES, NEUTRAL_EXAMPLES
    )
    save_style_vectors(vectors, "terse_style_vectors.pt")
    print("Done.")
