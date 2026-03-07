"""
Minimal Contrastive Pair experiment: find the verbosity direction.

Same semantic content, different style register. The delta between activation
streams IS the verbosity direction at each layer.

Three questions:
  1. Where does the divergence appear? (||delta[layer]|| norm plot)
  2. Is the direction stable? (cosine similarity across probe pairs)
  3. Does injecting it work? (steering validation loop)

Usage:
    python verbosity_direction.py                # full experiment
    python verbosity_direction.py --inject       # also test injection
    python verbosity_direction.py --visualize    # generate all plots
"""
import argparse
import json
import subprocess
import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from actadd import MODEL_ID, generate_steered

NUM_LAYERS = 28
D_MODEL = 1024

# Minimal contrastive pairs: identical semantics, different register
PROBE_PAIRS = [
    ("Why is the sky blue? Answer as tersely as possible.",
     "Why is the sky blue? Answer as verbosely as possible."),
    ("What causes rain? Answer as tersely as possible.",
     "What causes rain? Answer as verbosely as possible."),
    ("How does yeast work? Answer as tersely as possible.",
     "How does yeast work? Answer as verbosely as possible."),
    ("What is gravity? Answer as tersely as possible.",
     "What is gravity? Answer as verbosely as possible."),
    ("Why do leaves change color? Answer as tersely as possible.",
     "Why do leaves change color? Answer as verbosely as possible."),
]

# Region boundaries from layer-roles.json
REGIONS = [
    (0, 2, "tokenization", "#455a64"),
    (3, 7, "syntax", "#1565c0"),
    (8, 11, "early_semantics", "#2e7d32"),
    (12, 17, "deep_semantics", "#ff8f00"),
    (18, 22, "output_preparation", "#6a1b9a"),
    (23, 27, "logit_projection", "#c62828"),
]


def cache_all_layers(model, tokenizer, prompt):
    """Run forward pass, capture residual stream at all 28 layers.

    Returns dict: layer_idx -> Tensor[seq_len, d_model]
    """
    cache = {}

    hooks = []
    for layer_idx in range(NUM_LAYERS):
        def make_hook(idx):
            def hook_fn(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                cache[idx] = hs.detach().squeeze(0)  # (seq_len, d_model)
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return cache


def compute_deltas(model, tokenizer, probe_pairs):
    """For each probe pair, compute delta[layer] = v_verbose - v_terse at last token.

    Returns:
        deltas: list of dicts, each {layer: Tensor[d_model]}
        norms:  list of dicts, each {layer: float}
    """
    all_deltas = []
    all_norms = []

    for i, (prompt_terse, prompt_verbose) in enumerate(probe_pairs):
        print(f"  Pair {i+1}/{len(probe_pairs)}: {prompt_terse[:40]}...")

        cache_terse = cache_all_layers(model, tokenizer, prompt_terse)
        cache_verbose = cache_all_layers(model, tokenizer, prompt_verbose)

        deltas = {}
        norms = {}
        for layer in range(NUM_LAYERS):
            # Last token position
            v_terse = cache_terse[layer][-1]
            v_verbose = cache_verbose[layer][-1]
            delta = v_verbose - v_terse
            deltas[layer] = delta
            norms[layer] = delta.norm().item()

        all_deltas.append(deltas)
        all_norms.append(norms)

    return all_deltas, all_norms


def compute_stability(all_deltas):
    """Cosine similarity between delta vectors across pairs at each layer.

    Returns: dict layer -> mean cosine similarity across all pair combinations
    """
    stability = {}
    n_pairs = len(all_deltas)

    for layer in range(NUM_LAYERS):
        cosines = []
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                cos = F.cosine_similarity(
                    all_deltas[i][layer].unsqueeze(0),
                    all_deltas[j][layer].unsqueeze(0),
                ).item()
                cosines.append(cos)
        stability[layer] = {
            "mean_cosine": round(np.mean(cosines), 4),
            "std_cosine": round(np.std(cosines), 4),
            "min_cosine": round(min(cosines), 4),
            "max_cosine": round(max(cosines), 4),
        }

    return stability


def test_injection(model, tokenizer, all_deltas, all_norms):
    """Inject the mean verbosity delta at the peak layer into a neutral prompt."""
    # Find peak layer (highest mean norm)
    mean_norms = {}
    for layer in range(NUM_LAYERS):
        mean_norms[layer] = np.mean([n[layer] for n in all_norms])
    peak_layer = max(mean_norms, key=mean_norms.get)

    # Average delta across all pairs at peak layer
    mean_delta = torch.stack([d[peak_layer] for d in all_deltas]).mean(dim=0)

    neutral_prompt = "Explain what a compiler does."
    print(f"\n  Peak layer: {peak_layer} (mean ||delta|| = {mean_norms[peak_layer]:.2f})")
    print(f"  Neutral prompt: {neutral_prompt}")

    # Baseline
    baseline = generate_steered(
        model, tokenizer, neutral_prompt, mean_delta, peak_layer, alpha=0.0
    )
    baseline_words = len(baseline.split())

    # Inject verbosity direction (positive = more verbose)
    verbose_out = generate_steered(
        model, tokenizer, neutral_prompt, mean_delta, peak_layer, alpha=2.0
    )
    verbose_words = len(verbose_out.split())

    # Inject terse direction (negative = more terse)
    terse_out = generate_steered(
        model, tokenizer, neutral_prompt, mean_delta, peak_layer, alpha=-2.0
    )
    terse_words = len(terse_out.split())

    return {
        "peak_layer": peak_layer,
        "mean_delta_norm": round(mean_delta.norm().item(), 2),
        "baseline_words": baseline_words,
        "verbose_words": verbose_words,
        "terse_words": terse_words,
        "verbose_increase_pct": round((verbose_words / max(baseline_words, 1) - 1) * 100, 1),
        "terse_decrease_pct": round((1 - terse_words / max(baseline_words, 1)) * 100, 1),
        "baseline_preview": baseline[:120],
        "verbose_preview": verbose_out[:120],
        "terse_preview": terse_out[:120],
    }


def plot_results(all_norms, stability, injection_results=None, output_dir="eval_output"):
    """Generate three visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(NUM_LAYERS))

    # ── Plot 1: Norm trajectory with region bands ──
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Region bands
    for start, end, name, color in REGIONS:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=color, label=name)

    # Per-pair lines
    for i, norms in enumerate(all_norms):
        vals = [norms[l] for l in layers]
        ax.plot(layers, vals, alpha=0.3, color="#90caf9", linewidth=1)

    # Mean line
    mean_vals = [np.mean([n[l] for n in all_norms]) for l in layers]
    ax.plot(layers, mean_vals, color="#ffeb3b", linewidth=2.5, label="mean ||delta||")

    ax.set_xlabel("Layer", color="#e0e0e0")
    ax.set_ylabel("||delta|| (verbose - terse)", color="#e0e0e0")
    ax.set_title("Verbosity Direction Norm by Layer", color="#e0e0e0", fontweight="bold")
    ax.tick_params(colors="#e0e0e0")
    ax.legend(loc="upper left", fontsize=8, facecolor="#2a2a4a", edgecolor="#37374f",
              labelcolor="#e0e0e0")
    ax.grid(alpha=0.2, color="#37374f")

    plt.tight_layout()
    path1 = f"{output_dir}/verbosity_norm_trajectory.png"
    fig.savefig(path1, dpi=200, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {path1}")

    # ── Plot 2: Cosine similarity heatmap across pairs ──
    n_pairs = len(all_norms)
    pair_labels = [f"P{i+1}" for i in range(n_pairs)]

    # Build full cosine matrix per layer, show as layerxpair heatmap
    cos_matrix = np.zeros((NUM_LAYERS, n_pairs * (n_pairs - 1) // 2))
    col = 0
    pair_combo_labels = []
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            pair_combo_labels.append(f"{pair_labels[i]}-{pair_labels[j]}")
            col += 1

    # Simpler: just show mean + std per layer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    means = [stability[l]["mean_cosine"] for l in layers]
    stds = [stability[l]["std_cosine"] for l in layers]

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")

    # Region bands on both
    for start, end, name, color in REGIONS:
        ax1.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=color)
        ax2.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=color)

    ax1.plot(layers, means, color="#4caf50", linewidth=2)
    ax1.fill_between(layers, [m - s for m, s in zip(means, stds)],
                      [m + s for m, s in zip(means, stds)], alpha=0.2, color="#4caf50")
    ax1.axhline(y=0.5, color="#ff5722", linestyle="--", alpha=0.5, label="stability threshold")
    ax1.set_xlabel("Layer", color="#e0e0e0")
    ax1.set_ylabel("Mean cosine similarity", color="#e0e0e0")
    ax1.set_title("Direction Stability Across Pairs", color="#e0e0e0", fontweight="bold")
    ax1.legend(fontsize=8, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0")
    ax1.grid(alpha=0.2, color="#37374f")

    # Norm vs stability scatter
    ax2.scatter(mean_vals, means, c=layers, cmap="coolwarm", s=60, edgecolors="#e0e0e0", linewidth=0.5)
    for l in [0, 5, 12, 15, 18, 24, 27]:
        if l < NUM_LAYERS:
            ax2.annotate(f"L{l}", (mean_vals[l], means[l]), color="#e0e0e0", fontsize=7)
    ax2.set_xlabel("||delta|| norm", color="#e0e0e0")
    ax2.set_ylabel("Direction stability (cosine)", color="#e0e0e0")
    ax2.set_title("Norm vs Stability", color="#e0e0e0", fontweight="bold")
    ax2.grid(alpha=0.2, color="#37374f")

    plt.tight_layout()
    path2 = f"{output_dir}/verbosity_stability.png"
    fig.savefig(path2, dpi=200, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {path2}")

    # ── Plot 3: Combined dashboard ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Verbosity Direction Experiment", color="#e0e0e0",
                 fontweight="bold", fontsize=14)

    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")

    # Top-left: norm trajectory
    ax = axes[0, 0]
    for start, end, name, color in REGIONS:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=color)
    ax.plot(layers, mean_vals, color="#ffeb3b", linewidth=2)
    ax.set_title("||delta|| by Layer", color="#e0e0e0")
    ax.set_xlabel("Layer", color="#e0e0e0")
    ax.grid(alpha=0.2, color="#37374f")

    # Top-right: stability
    ax = axes[0, 1]
    for start, end, name, color in REGIONS:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.15, color=color)
    ax.plot(layers, means, color="#4caf50", linewidth=2)
    ax.axhline(y=0.5, color="#ff5722", linestyle="--", alpha=0.5)
    ax.set_title("Direction Stability", color="#e0e0e0")
    ax.set_xlabel("Layer", color="#e0e0e0")
    ax.grid(alpha=0.2, color="#37374f")

    # Bottom-left: delta norm heatmap (pairs x layers)
    ax = axes[1, 0]
    norm_matrix = np.array([[n[l] for l in layers] for n in all_norms])
    im = ax.imshow(norm_matrix, aspect="auto", cmap="inferno",
                   extent=[-0.5, 27.5, len(all_norms) - 0.5, -0.5])
    ax.set_xlabel("Layer", color="#e0e0e0")
    ax.set_ylabel("Probe Pair", color="#e0e0e0")
    ax.set_title("||delta|| Heatmap (pairs × layers)", color="#e0e0e0")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Bottom-right: injection results or summary text
    ax = axes[1, 1]
    ax.axis("off")
    if injection_results:
        text = (
            f"Injection Test @ Layer {injection_results['peak_layer']}\n"
            f"{'─' * 40}\n"
            f"Baseline:  {injection_results['baseline_words']:>4d} words\n"
            f"+ verbose: {injection_results['verbose_words']:>4d} words "
            f"(+{injection_results['verbose_increase_pct']}%)\n"
            f"+ terse:   {injection_results['terse_words']:>4d} words "
            f"(-{injection_results['terse_decrease_pct']}%)\n"
            f"\nDelta norm: {injection_results['mean_delta_norm']}\n"
            f"\nVerdict: {'WORKS' if injection_results['verbose_words'] > injection_results['baseline_words'] else 'FAILS'}"
        )
    else:
        peak = max(range(NUM_LAYERS), key=lambda l: mean_vals[l])
        stable = [l for l in range(NUM_LAYERS) if stability[l]["mean_cosine"] > 0.5]
        text = (
            f"Summary\n{'─' * 40}\n"
            f"Peak norm layer: {peak}\n"
            f"Stable layers (cos > 0.5): {stable or 'none'}\n"
            f"Sweet spot: {peak} (if stable)\n"
            f"\nRun with --inject to test"
        )
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace", color="#e0e0e0")

    plt.tight_layout()
    path3 = f"{output_dir}/verbosity_dashboard.png"
    fig.savefig(path3, dpi=200, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {path3}")


def submit_cprr(all_norms, stability, injection_results=None):
    """Submit experiment results as CPRR conjecture + evidence."""

    def cprr_cmd(*args):
        result = subprocess.run(["cprr"] + list(args), capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  cprr {' '.join(args[:2])}: {result.stdout.strip()}")
        return result.returncode == 0

    # Add conjecture
    cprr_cmd("add", "Verbosity direction peaks in deep_semantics region (L12-17)",
             "--hypothesis",
             "The norm of the verbosity delta (verbose_activation - terse_activation) "
             "peaks at layers 12-17, confirming deep_semantics as the primary region "
             "encoding style register. Peak layer has mean ||delta|| > 2x the mean of "
             "layers 0-7 and 23-27.",
             "-t", "layer-roles,verbosity,empirical")

    # Find the conjecture ID (should be next after 6)
    cid = 7  # assuming sequential after our 6

    # Evidence: norm data
    mean_norms = {l: round(np.mean([n[l] for n in all_norms]), 2) for l in range(NUM_LAYERS)}
    peak_layer = max(mean_norms, key=mean_norms.get)
    early_mean = np.mean([mean_norms[l] for l in range(8)])
    sweet_mean = np.mean([mean_norms[l] for l in range(12, 18)])
    late_mean = np.mean([mean_norms[l] for l in range(23, 28)])

    cprr_cmd("evidence", str(cid),
             f"Peak at L{peak_layer} (||delta||={mean_norms[peak_layer]:.2f}). "
             f"Early(0-7) mean={early_mean:.2f}, Sweet(12-17) mean={sweet_mean:.2f}, "
             f"Late(23-27) mean={late_mean:.2f}. "
             f"Ratio sweet/early={sweet_mean/max(early_mean,0.01):.1f}x, "
             f"sweet/late={sweet_mean/max(late_mean,0.01):.1f}x "
             f"[confidence: empirical, source: this-project-sweep]")

    # Evidence: stability
    stable_layers = [l for l in range(NUM_LAYERS) if stability[l]["mean_cosine"] > 0.5]
    cprr_cmd("evidence", str(cid),
             f"Direction stability (cos>0.5) at layers: {stable_layers}. "
             f"Peak stability: L{max(stability, key=lambda l: stability[l]['mean_cosine'])} "
             f"(cos={max(s['mean_cosine'] for s in stability.values()):.3f}). "
             f"n={len(PROBE_PAIRS)} probe pairs "
             f"[confidence: empirical, source: this-project-sweep]")

    # Advance
    cprr_cmd("next", str(cid))

    if injection_results:
        cprr_cmd("evidence", str(cid),
                 f"Injection at L{injection_results['peak_layer']}: "
                 f"baseline={injection_results['baseline_words']}w, "
                 f"+verbose={injection_results['verbose_words']}w "
                 f"(+{injection_results['verbose_increase_pct']}%), "
                 f"-terse={injection_results['terse_words']}w "
                 f"(-{injection_results['terse_decrease_pct']}%) "
                 f"[confidence: empirical, source: this-project-sweep]")


def main():
    parser = argparse.ArgumentParser(description="Verbosity direction experiment")
    parser.add_argument("--inject", action="store_true",
                        help="Also test injection at peak layer")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--cprr", action="store_true",
                        help="Submit results to CPRR")
    parser.add_argument("--output", default="verbosity-direction.json")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Question 1: Where does the divergence appear?
    print("\n=== Q1: Where does the divergence appear? ===")
    all_deltas, all_norms = compute_deltas(model, tokenizer, PROBE_PAIRS)

    print("\n  Layer  ||delta|| (mean across pairs)")
    print("  " + "-" * 40)
    mean_norms = {}
    for layer in range(NUM_LAYERS):
        mean_norm = np.mean([n[layer] for n in all_norms])
        mean_norms[layer] = mean_norm
        region = next((name for s, e, name, _ in REGIONS if s <= layer <= e), "?")
        bar = "█" * int(mean_norm / 2)
        print(f"  L{layer:02d} [{region:20s}] {mean_norm:8.2f}  {bar}")

    peak_layer = max(mean_norms, key=mean_norms.get)
    print(f"\n  Peak: Layer {peak_layer} (||delta|| = {mean_norms[peak_layer]:.2f})")

    # Question 2: Is the direction stable?
    print("\n=== Q2: Is the direction stable across pairs? ===")
    stability = compute_stability(all_deltas)

    print("\n  Layer  Mean cos_sim  Stable?")
    print("  " + "-" * 40)
    for layer in range(NUM_LAYERS):
        s = stability[layer]
        stable = "YES" if s["mean_cosine"] > 0.5 else "no"
        bar = "█" * int(max(0, s["mean_cosine"]) * 20)
        print(f"  L{layer:02d}   {s['mean_cosine']:+.4f} ± {s['std_cosine']:.4f}  [{stable:3s}]  {bar}")

    # Question 3: Does injecting it work?
    injection_results = None
    if args.inject:
        print("\n=== Q3: Does injecting the verbosity direction work? ===")
        injection_results = test_injection(model, tokenizer, all_deltas, all_norms)
        print(f"\n  Baseline:  {injection_results['baseline_words']} words")
        print(f"  + verbose: {injection_results['verbose_words']} words "
              f"(+{injection_results['verbose_increase_pct']}%)")
        print(f"  + terse:   {injection_results['terse_words']} words "
              f"(-{injection_results['terse_decrease_pct']}%)")
        print(f"\n  Baseline:  {injection_results['baseline_preview']}")
        print(f"  + verbose: {injection_results['verbose_preview']}")
        print(f"  + terse:   {injection_results['terse_preview']}")

    # Save results
    results = {
        "model": MODEL_ID,
        "probe_pairs": PROBE_PAIRS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "norms": {str(l): round(mean_norms[l], 4) for l in range(NUM_LAYERS)},
        "stability": stability,
        "peak_layer": peak_layer,
        "peak_norm": round(mean_norms[peak_layer], 4),
    }
    if injection_results:
        results["injection"] = injection_results

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Visualize
    if args.visualize:
        print("\n=== Generating visualizations ===")
        plot_results(all_norms, stability, injection_results)

    # CPRR
    if args.cprr:
        print("\n=== Submitting to CPRR ===")
        submit_cprr(all_norms, stability, injection_results)


if __name__ == "__main__":
    main()
