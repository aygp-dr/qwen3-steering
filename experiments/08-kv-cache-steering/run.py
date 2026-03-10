"""
Experiment 08: KV Cache Anatomy Under Steering

Extracts and compares KV cache structure + attention patterns
during baseline vs steered (L12, terse, alpha=2.0) generation.
"""
import json
import sys
import os
import numpy as np
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from actadd import (
    MODEL_ID,
    STYLE_PAIRS,
    compute_steering_vector,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain what a mutex is.",
    "What is the difference between a stack and a heap?",
    "Why do programming languages have types?",
    "How does garbage collection work?",
    "What is the CAP theorem?",
]
LAYER_IDX = 12
ALPHA = 2.0
STYLE = "terse"
OUTPUT_DIR = Path(__file__).parent / "output"


def generate_with_cache(model, tokenizer, prompt, steering_vec=None,
                        layer_idx=12, alpha=2.0, max_new_tokens=128):
    """Generate text and return KV cache + attention weights."""
    handle = None

    if steering_vec is not None:
        def steering_hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs + alpha * steering_vec.to(hs.device, hs.dtype)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs
        handle = model.model.layers[layer_idx].register_forward_hook(steering_hook)

    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=True,
            )
    finally:
        if handle is not None:
            handle.remove()

    # Extract generated text
    new_ids = out.sequences[0][inputs.input_ids.shape[1]:]
    gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Extract attention patterns from first generation step
    # out.attentions is tuple of (num_gen_steps, ) each containing
    # tuple of (num_layers, ) each (batch, heads, seq, seq)
    first_step_attns = None
    if hasattr(out, "attentions") and out.attentions is not None:
        # attentions[step][layer] -> (batch, heads, q_len, kv_len)
        first_step_attns = []
        for layer_attn in out.attentions[0]:
            # Average over query positions, keep per-head
            # Shape: (heads, kv_len)
            attn_over_keys = layer_attn[0].mean(dim=0).cpu().float().numpy()
            first_step_attns.append(attn_over_keys)

    return gen_text, first_step_attns


def attention_entropy(attn_weights):
    """Compute per-head entropy of attention distribution."""
    # attn_weights: (heads, kv_len)
    # Clamp to avoid log(0)
    p = np.clip(attn_weights, 1e-10, 1.0)
    # Normalize rows
    p = p / p.sum(axis=1, keepdims=True)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return entropy


def kl_divergence(p, q):
    """KL(P || Q) per head."""
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    return np.sum(p * np.log2(p / q), axis=1)


def top_k_overlap(attn_base, attn_steered, k=10):
    """Fraction of top-k attended tokens shared between base and steered."""
    overlaps = []
    for h in range(attn_base.shape[0]):
        top_base = set(np.argsort(attn_base[h])[-k:])
        top_steer = set(np.argsort(attn_steered[h])[-k:])
        overlaps.append(len(top_base & top_steer) / k)
    return np.mean(overlaps)


def plot_results(results, output_path):
    """Plot KV cache anatomy heatmaps."""
    num_layers = len(results["per_layer"])

    # Collect metrics
    layers = sorted(results["per_layer"].keys(), key=int)
    entropy_base = []
    entropy_steer = []
    entropy_diff = []
    kl_divs = []

    for l in layers:
        data = results["per_layer"][l]
        entropy_base.append(data["entropy_baseline_mean"])
        entropy_steer.append(data["entropy_steered_mean"])
        entropy_diff.append(data["entropy_diff_mean"])
        kl_divs.append(data["kl_divergence_mean"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    layer_nums = [int(l) for l in layers]

    # 1. Entropy comparison
    ax = axes[0, 0]
    ax.bar(np.array(layer_nums) - 0.2, entropy_base, 0.4, label="Baseline",
           color="#4ecdc4", alpha=0.8)
    ax.bar(np.array(layer_nums) + 0.2, entropy_steer, 0.4, label="Steered",
           color="#ff6b6b", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Entropy (bits)")
    ax.set_title("Attention Entropy: Baseline vs Steered")
    ax.legend(facecolor="#2a2a4e", edgecolor="gray", labelcolor="white")
    ax.axvspan(11.5, 17.5, alpha=0.1, color="green", label="Sweet spot")

    # 2. Entropy difference
    ax = axes[0, 1]
    colors = ["#ff6b6b" if d > 0 else "#4ecdc4" for d in entropy_diff]
    ax.bar(layer_nums, entropy_diff, color=colors, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy Change (bits)")
    ax.set_title("Entropy Difference (Steered - Baseline)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvspan(11.5, 17.5, alpha=0.1, color="green")

    # 3. KL Divergence
    ax = axes[1, 0]
    ax.bar(layer_nums, kl_divs, color="#e2b93d", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence (bits)")
    ax.set_title("KL(Baseline || Steered) Attention Divergence")
    ax.axvspan(11.5, 17.5, alpha=0.1, color="green")

    # 4. Top-k overlap
    overlaps = [results["per_layer"][l]["top_k_overlap"] for l in layers]
    ax = axes[1, 1]
    ax.bar(layer_nums, overlaps, color="#a29bfe", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-10 Token Overlap")
    ax.set_title("Attention Focus Overlap (Steered vs Baseline)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.axvspan(11.5, 17.5, alpha=0.1, color="green")

    fig.suptitle(
        "Experiment 08: KV Cache Anatomy Under Terse Steering (L12, α=2.0)",
        color="white", fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved plot → {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    # Compute steering vector
    print(f"Computing {STYLE} steering vector at L{LAYER_IDX}...")
    steering_vec = compute_steering_vector(
        model, tokenizer, STYLE, LAYER_IDX
    )
    vec_norm = steering_vec.norm().item()
    print(f"  Steering vector norm: {vec_norm:.2f}")

    results = {
        "experiment": "08-kv-cache-steering",
        "model": MODEL_ID,
        "style": STYLE,
        "layer": LAYER_IDX,
        "alpha": ALPHA,
        "steering_vec_norm": vec_norm,
        "per_layer": {},
        "per_prompt": [],
    }

    # Run on each prompt
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt[:50]}...")

        # Baseline
        base_text, base_attns = generate_with_cache(
            model, tokenizer, prompt
        )
        # Steered
        steer_text, steer_attns = generate_with_cache(
            model, tokenizer, prompt, steering_vec, LAYER_IDX, ALPHA
        )

        prompt_result = {
            "prompt": prompt,
            "baseline_text": base_text,
            "baseline_words": len(base_text.split()),
            "steered_text": steer_text,
            "steered_words": len(steer_text.split()),
        }

        # Compare attention patterns per layer
        if base_attns and steer_attns:
            num_layers = min(len(base_attns), len(steer_attns))
            for l_idx in range(num_layers):
                layer_key = str(l_idx)

                # Attention shapes may differ (different seq lengths)
                min_kv = min(base_attns[l_idx].shape[1],
                             steer_attns[l_idx].shape[1])
                min_heads = min(base_attns[l_idx].shape[0],
                                steer_attns[l_idx].shape[0])

                ba = base_attns[l_idx][:min_heads, :min_kv]
                sa = steer_attns[l_idx][:min_heads, :min_kv]

                ent_b = attention_entropy(ba)
                ent_s = attention_entropy(sa)
                kl = kl_divergence(ba, sa)
                overlap = top_k_overlap(ba, sa)

                if layer_key not in results["per_layer"]:
                    results["per_layer"][layer_key] = {
                        "entropy_baseline_mean": 0.0,
                        "entropy_steered_mean": 0.0,
                        "entropy_diff_mean": 0.0,
                        "kl_divergence_mean": 0.0,
                        "top_k_overlap": 0.0,
                        "_count": 0,
                    }

                entry = results["per_layer"][layer_key]
                n = entry["_count"]
                # Running mean
                entry["entropy_baseline_mean"] = (
                    entry["entropy_baseline_mean"] * n + float(ent_b.mean())
                ) / (n + 1)
                entry["entropy_steered_mean"] = (
                    entry["entropy_steered_mean"] * n + float(ent_s.mean())
                ) / (n + 1)
                entry["entropy_diff_mean"] = (
                    entry["entropy_diff_mean"] * n + float((ent_s - ent_b).mean())
                ) / (n + 1)
                entry["kl_divergence_mean"] = (
                    entry["kl_divergence_mean"] * n + float(kl.mean())
                ) / (n + 1)
                entry["top_k_overlap"] = (
                    entry["top_k_overlap"] * n + float(overlap)
                ) / (n + 1)
                entry["_count"] = n + 1

        results["per_prompt"].append(prompt_result)
        print(f"  Baseline: {prompt_result['baseline_words']}w")
        print(f"  Steered:  {prompt_result['steered_words']}w")

    # Clean up running count
    for layer_key in results["per_layer"]:
        results["per_layer"][layer_key].pop("_count", None)

    # Save JSON
    json_path = OUTPUT_DIR / "kv_cache_anatomy.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results → {json_path}")

    # Plot
    if results["per_layer"]:
        plot_results(results, OUTPUT_DIR / "kv_cache_anatomy.png")

    # Summary
    print("\n── Summary ──")
    layers_with_data = sorted(results["per_layer"].keys(), key=int)
    if layers_with_data:
        max_kl_layer = max(layers_with_data,
                           key=lambda l: results["per_layer"][l]["kl_divergence_mean"])
        max_kl = results["per_layer"][max_kl_layer]["kl_divergence_mean"]
        min_overlap_layer = min(layers_with_data,
                                key=lambda l: results["per_layer"][l]["top_k_overlap"])
        min_overlap = results["per_layer"][min_overlap_layer]["top_k_overlap"]
        print(f"  Max KL divergence: L{max_kl_layer} ({max_kl:.4f} bits)")
        print(f"  Min top-k overlap: L{min_overlap_layer} ({min_overlap:.2%})")

        sweet_kl = np.mean([results["per_layer"][str(l)]["kl_divergence_mean"]
                            for l in range(12, 18) if str(l) in results["per_layer"]])
        early_kl = np.mean([results["per_layer"][str(l)]["kl_divergence_mean"]
                            for l in range(0, 8) if str(l) in results["per_layer"]])
        print(f"  Sweet spot (L12-17) avg KL: {sweet_kl:.4f}")
        print(f"  Early (L0-7) avg KL: {early_kl:.4f}")


if __name__ == "__main__":
    main()
