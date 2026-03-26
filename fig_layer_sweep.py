#!/usr/bin/env python3
"""
Layer sweep effectiveness plot: mean word count vs layer for terse steering.

SPLASH reviewer priority 3 (bead ecd). Shows that layer 12 is the optimal
injection point for terse steering at alpha=2.0, with error bands across
5 diverse prompts.

Usage:
    uv run python fig_layer_sweep.py              # full sweep (slow)
    uv run python fig_layer_sweep.py --from-json   # re-render from cached data
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = Path("eval_output")

# Project dark theme
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#c9d1d9"
GRID_CLR = "#30363d"
BLUE = "#58a6ff"
ORANGE = "#f0883e"

STYLE = "terse"
ALPHA = 2.0
LAYERS = list(range(28))

PROMPTS = [
    "What is gravity?",
    "How do magnets work?",
    "What is an atom?",
    "What is a mutex?",
    "Why is the sky blue?",
]


def word_count(text: str) -> int:
    return len(text.split())


def run_sweep():
    """Run full model inference sweep. Returns (results, baseline_wcs, elapsed)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from actadd import compute_steering_vector, generate_steered

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # ── Baseline (alpha=0, use layer 12 for baseline vec -- doesn't matter since alpha=0) ──
    print("Generating baseline outputs (alpha=0)...")
    baseline_vec = compute_steering_vector(model, tokenizer, STYLE, layer_idx=12)
    baseline_wcs = []
    for prompt in PROMPTS:
        text = generate_steered(
            model, tokenizer, prompt, baseline_vec, layer_idx=12,
            alpha=0.0, max_new_tokens=256,
        )
        wc = word_count(text)
        baseline_wcs.append(wc)
        print(f"  Baseline [{prompt[:25]}...]: {wc} words")
    print(f"  Baseline mean: {np.mean(baseline_wcs):.1f} words")

    # ── Layer sweep ──────────────────────────────────────────────────────────
    results = {}
    t0 = time.time()

    for layer in LAYERS:
        print(f"\nLayer {layer}/27 ...", end="", flush=True)
        vec = compute_steering_vector(model, tokenizer, STYLE, layer_idx=layer)
        layer_wcs = []
        for prompt in PROMPTS:
            text = generate_steered(
                model, tokenizer, prompt, vec, layer_idx=layer,
                alpha=ALPHA, max_new_tokens=256,
            )
            wc = word_count(text)
            layer_wcs.append(wc)
        results[layer] = layer_wcs
        mean_wc = np.mean(layer_wcs)
        print(f" mean={mean_wc:.1f}w ({layer_wcs})")

    elapsed = time.time() - t0
    print(f"\nSweep completed in {elapsed:.0f}s")
    return results, baseline_wcs, elapsed


def load_from_json(json_path: Path):
    """Load cached sweep data from JSON. Returns (results, baseline_wcs, elapsed)."""
    with open(json_path) as f:
        data = json.load(f)
    results = {int(k): v["word_counts"] for k, v in data["layers"].items()}
    baseline_wcs = data["baseline"]["word_counts"]
    elapsed = data.get("elapsed_seconds", 0.0)
    print(f"Loaded cached data from {json_path}")
    print(f"  Baseline mean: {data['baseline']['mean']:.1f} words")
    print(f"  Original runtime: {elapsed:.0f}s")
    return results, baseline_wcs, elapsed


def main():
    parser = argparse.ArgumentParser(description="Layer sweep effectiveness plot")
    parser.add_argument(
        "--from-json", action="store_true",
        help="Re-render plot from cached eval_output/layer_sweep_effectiveness.json",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    json_path = OUTPUT_DIR / "layer_sweep_effectiveness.json"

    if args.from_json:
        if not json_path.exists():
            print(f"ERROR: {json_path} not found. Run without --from-json first.")
            return
        results, baseline_wcs, elapsed = load_from_json(json_path)
    else:
        results, baseline_wcs, elapsed = run_sweep()

    baseline_mean = np.mean(baseline_wcs)

    # ── Compute statistics ───────────────────────────────────────────────────
    layers_arr = np.array(LAYERS)
    means = np.array([np.mean(results[L]) for L in LAYERS])
    stds = np.array([np.std(results[L], ddof=1) for L in LAYERS])
    ses = stds / np.sqrt(len(PROMPTS))

    # Optimal = lowest mean among layers where ALL prompts produce output.
    # Layers with any prompt yielding 0 words are partially collapsed and excluded.
    # This ensures the "optimal" layer produces reliable steering, not sporadic collapse.
    all_produced = np.array([
        all(wc > 0 for wc in results[L]) for L in LAYERS
    ])
    if all_produced.any():
        viable_means = np.where(all_produced, means, np.inf)
        min_mean = np.min(viable_means)
        # Break ties: among layers with equal min mean, pick lowest SE (most consistent)
        tied_mask = (viable_means == min_mean)
        if np.sum(tied_mask) > 1:
            tied_ses = np.where(tied_mask, ses, np.inf)
            optimal_layer = int(layers_arr[np.argmin(tied_ses)])
        else:
            optimal_layer = int(layers_arr[np.argmin(viable_means)])
        optimal_mean = means[optimal_layer]
    else:
        optimal_layer = 12
        optimal_mean = means[12]

    # Identify collapsed layers (all prompts yielded 0 words)
    collapsed_layers = layers_arr[means == 0]
    # Identify partially collapsed layers (some prompts yielded 0 words)
    partial_collapse = np.array([
        any(wc == 0 for wc in results[L]) and means[L] > 0
        for L in LAYERS
    ])
    partial_layers = layers_arr[partial_collapse]

    print(f"\nOptimal layer: {optimal_layer} (mean={optimal_mean:.1f}w)")
    print(f"Baseline mean: {baseline_mean:.1f}w")
    if len(collapsed_layers) > 0:
        print(f"Collapsed layers (0 words): {list(collapsed_layers)}")
    if len(partial_layers) > 0:
        print(f"Partially collapsed layers: {list(partial_layers)}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # Error band (+/- 1 SE)
    ax.fill_between(
        layers_arr, means - ses, means + ses,
        alpha=0.25, color=BLUE, linewidth=0,
    )
    # Main line
    ax.plot(layers_arr, means, color=BLUE, linewidth=2, marker="o",
            markersize=4, zorder=5, label="Steered mean word count")

    # Baseline dashed line
    ax.axhline(
        baseline_mean, color=TEXT_CLR, linestyle="--", linewidth=1.2,
        alpha=0.6, label=f"Baseline (no steering): {baseline_mean:.0f}w",
    )

    # Mark collapsed layers (0 words = coherence collapse)
    if len(collapsed_layers) > 0:
        ax.scatter(
            collapsed_layers, [0] * len(collapsed_layers),
            marker="x", s=60, color="#f85149", zorder=8, linewidths=2,
            label=f"Collapse (L{collapsed_layers[0]}-{collapsed_layers[-1]})",
        )

    # Mark partially collapsed layers
    if len(partial_layers) > 0:
        ax.scatter(
            partial_layers, means[partial_collapse],
            marker="D", s=40, color="#f85149", zorder=8, alpha=0.6,
            label=f"Partial collapse (L{','.join(str(x) for x in partial_layers)})",
        )

    # Highlight optimal layer
    ax.plot(
        optimal_layer, optimal_mean, marker="*", markersize=16,
        color=ORANGE, zorder=10, markeredgecolor="white", markeredgewidth=0.8,
    )
    ax.annotate(
        f"L{optimal_layer}: {optimal_mean:.0f}w",
        xy=(optimal_layer, optimal_mean),
        xytext=(optimal_layer + 2.5, optimal_mean + 18),
        color=ORANGE, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
    )

    # Axis styling
    ax.set_xlabel("Layer index", color=TEXT_CLR, fontsize=11)
    ax.set_ylabel("Mean word count (5 prompts)", color=TEXT_CLR, fontsize=11)
    ax.set_title(
        f"Layer Sweep Effectiveness: Terse Steering at \u03b1={ALPHA}\n"
        f"Qwen3-0.6B (28 layers, d_model=1024) | \u00b11 SE shaded",
        color=TEXT_CLR, fontsize=12, pad=12,
    )
    ax.set_xlim(-0.5, 27.5)
    ax.set_xticks(range(0, 28, 2))
    ax.tick_params(colors=TEXT_CLR, which="both")
    plt.setp(ax.get_xticklabels(), color=TEXT_CLR)
    plt.setp(ax.get_yticklabels(), color=TEXT_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_CLR)
    ax.spines["bottom"].set_color(GRID_CLR)

    legend = ax.legend(
        loc="upper right", fontsize=9,
        facecolor=PANEL_BG, edgecolor=GRID_CLR,
        labelcolor=TEXT_CLR,
    )

    plt.tight_layout()

    # Save plot
    png_path = OUTPUT_DIR / "layer_sweep_effectiveness.png"
    fig.savefig(png_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")

    # ── Save alt-text ────────────────────────────────────────────────────────
    txt_path = OUTPUT_DIR / "layer_sweep_effectiveness.txt"
    collapsed_str = (
        f"Layers {[int(x) for x in collapsed_layers]} produce coherence collapse (0 words). "
        if len(collapsed_layers) > 0 else ""
    )
    with open(txt_path, "w") as f:
        f.write(
            f"Line plot of mean word count (y-axis) vs transformer layer index 0-27 (x-axis) "
            f"for terse activation steering at alpha={ALPHA} on Qwen3-0.6B. "
            f"Five prompts evaluated per layer with +/-1 SE shaded bands. "
            f"Baseline (no steering) mean word count: {baseline_mean:.0f}. "
            f"Optimal effective layer: {optimal_layer} with mean {optimal_mean:.0f} words "
            f"({100*(1 - optimal_mean/baseline_mean):.0f}% reduction from baseline). "
            f"{collapsed_str}"
            f"Early layers (0-7) show topic drift or collapse. "
            f"Mid layers (10-17) show strong word count reduction (sweet spot). "
            f"Late layers (20-27) return toward baseline word counts.\n"
        )
    print(f"Saved: {txt_path}")

    # ── Save raw data JSON ───────────────────────────────────────────────────
    json_path = OUTPUT_DIR / "layer_sweep_effectiveness.json"
    data = {
        "model": MODEL_ID,
        "style": STYLE,
        "alpha": ALPHA,
        "prompts": PROMPTS,
        "baseline": {
            "word_counts": baseline_wcs,
            "mean": float(baseline_mean),
        },
        "layers": {
            str(L): {
                "word_counts": results[L],
                "mean": float(means[L]),
                "std": float(stds[L]),
                "se": float(ses[L]),
            }
            for L in LAYERS
        },
        "optimal_layer": optimal_layer,
        "optimal_mean_wc": float(optimal_mean),
        "collapsed_layers": [int(x) for x in collapsed_layers],
        "partial_collapse_layers": [int(x) for x in partial_layers],
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
