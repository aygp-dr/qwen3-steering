#!/usr/bin/env python3
"""
Terse vs Verbose steering evaluation over 100 prompts.

For each prompt, generates three outputs (terse +α, baseline 0, verbose -α).
Records full provenance: input, config, timing, output text, word/token counts.

Then clusters the outputs by length features using k-means (k=3) and measures
how well unsupervised clustering recovers the true steering direction.

Produces:
  1. Seaborn confusion matrix: k-means predicted vs true direction
  2. Length distribution histograms
  3. Scatter plot: baseline vs steered lengths
  4. Full JSON with every input/output/timing record

Usage:
    uv run python eval_terse_verbose.py
    uv run python eval_terse_verbose.py --num-prompts 20  # quick test
    uv run python eval_terse_verbose.py --alpha 1.5
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

from actadd import compute_steering_vector, generate_steered, STYLE_PAIRS

MODEL_ID = "Qwen/Qwen3-0.6B"
LAYER = 15
OUTPUT_DIR = Path("eval_output")

# ── 100 simple, diverse prompts ──────────────────────────────────────────────

PROMPTS = [
    "What is gravity?",
    "How do magnets work?",
    "Why is the sky blue?",
    "What causes thunder?",
    "How do vaccines work?",
    "What is DNA?",
    "Why do stars twinkle?",
    "What is photosynthesis?",
    "How do airplanes fly?",
    "What is an atom?",
    "Why does ice float?",
    "What causes earthquakes?",
    "How does the internet work?",
    "What is a black hole?",
    "Why do we dream?",
    "How does memory work?",
    "What is electricity?",
    "Why do leaves change color?",
    "What is evolution?",
    "How do computers store data?",
    "How do you make coffee?",
    "Why do onions make you cry?",
    "How does soap clean things?",
    "Why do we yawn?",
    "How do mirrors work?",
    "What makes bread rise?",
    "Why do we get hiccups?",
    "How do batteries work?",
    "Why is the ocean salty?",
    "How do zippers work?",
    "What causes a rainbow?",
    "Why do we sneeze?",
    "How do clocks keep time?",
    "What makes glue sticky?",
    "Why do we blink?",
    "How does a refrigerator work?",
    "What causes static electricity?",
    "Why do cats purr?",
    "How do seeds grow?",
    "What makes music sound good?",
    "Who built the pyramids?",
    "What started World War I?",
    "How was paper invented?",
    "What is democracy?",
    "Who was Cleopatra?",
    "How did humans discover fire?",
    "What is the Renaissance?",
    "Why do we shake hands?",
    "What caused the ice ages?",
    "How did writing develop?",
    "What is justice?",
    "Why do we need sleep?",
    "What is consciousness?",
    "What makes something funny?",
    "Why do people lie?",
    "What is beauty?",
    "Why do we feel fear?",
    "What is time?",
    "Why do humans make art?",
    "What is happiness?",
    "What is a mutex?",
    "How does encryption work?",
    "What is a hash table?",
    "How does GPS work?",
    "What is machine learning?",
    "How do databases store data?",
    "What is an API?",
    "How does WiFi work?",
    "What is recursion?",
    "How do search engines rank pages?",
    "Why do birds sing?",
    "How do fish breathe underwater?",
    "What makes a desert?",
    "Why do wolves howl?",
    "How do spiders make webs?",
    "What causes tides?",
    "Why are flamingos pink?",
    "How do bees make honey?",
    "What makes a tornado?",
    "Why do trees lose leaves?",
    "Why does cheese melt?",
    "How do muscles grow?",
    "What makes chili peppers hot?",
    "Why do we get thirsty?",
    "How does fermentation work?",
    "What causes a fever?",
    "Why does chocolate taste good?",
    "How do bones heal?",
    "What makes popcorn pop?",
    "Why do we get goosebumps?",
    "What is infinity?",
    "Why is pi important?",
    "What are prime numbers?",
    "How does probability work?",
    "What is zero?",
    "Why does math work in nature?",
    "What is a proof?",
    "How do fractals work?",
    "What is symmetry?",
    "Why do patterns repeat?",
]


def run_eval(model, tokenizer, vec, prompts, alpha, layer, max_tokens):
    """Generate terse/baseline/verbose for each prompt with full provenance."""
    records = []
    n = len(prompts)
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1:3d}/{n}] {prompt[:50]}", end="", flush=True)
        row = {"prompt": prompt, "prompt_idx": i}

        for direction, a in [("terse", alpha), ("baseline", 0.0), ("verbose", -alpha)]:
            t0 = time.monotonic()
            text = generate_steered(
                model, tokenizer, prompt, vec, layer,
                alpha=a, max_new_tokens=max_tokens,
            )
            elapsed = time.monotonic() - t0
            tokens = tokenizer.encode(text)

            row[f"{direction}_text"] = text
            row[f"{direction}_words"] = len(text.split())
            row[f"{direction}_tokens"] = len(tokens)
            row[f"{direction}_chars"] = len(text)
            row[f"{direction}_elapsed_s"] = round(elapsed, 2)

        row["terse_eq_baseline"] = row["terse_text"] == row["baseline_text"]
        row["verbose_eq_baseline"] = row["verbose_text"] == row["baseline_text"]

        print(f"  T={row['terse_words']} B={row['baseline_words']} V={row['verbose_words']}")
        records.append(row)

    return records


def build_feature_matrix(records):
    """Build feature matrix for clustering: each generation is a row."""
    rows = []
    labels = []
    meta = []
    for r in records:
        for direction in ["terse", "baseline", "verbose"]:
            rows.append([
                r[f"{direction}_words"],
                r[f"{direction}_tokens"],
                r[f"{direction}_chars"],
            ])
            labels.append(direction)
            meta.append({"prompt": r["prompt"], "direction": direction})
    return np.array(rows, dtype=float), labels, meta


def cluster_and_evaluate(records, output_dir):
    """K-means clustering on length features, then confusion matrix via seaborn."""
    X, true_labels, meta = build_feature_matrix(records)
    label_order = ["terse", "baseline", "verbose"]

    # K-means with k=3
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(X)

    # Map cluster IDs to labels by majority vote
    cluster_to_label = {}
    for c in range(3):
        mask = cluster_ids == c
        cluster_true = [true_labels[i] for i in range(len(true_labels)) if mask[i]]
        from collections import Counter
        majority = Counter(cluster_true).most_common(1)[0][0]
        cluster_to_label[c] = majority

    # Handle collisions: if two clusters map to same label, assign by centroid order
    assigned = list(cluster_to_label.values())
    if len(set(assigned)) < 3:
        centroids = km.cluster_centers_
        order = np.argsort(centroids[:, 0])  # sort by word count
        cluster_to_label = {order[0]: "terse", order[1]: "baseline", order[2]: "verbose"}

    pred_labels = [cluster_to_label[c] for c in cluster_ids]

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=label_order)

    # Classification report
    report = classification_report(
        true_labels, pred_labels, labels=label_order, output_dict=True
    )
    accuracy = report["accuracy"]

    # Seaborn heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_order, yticklabels=label_order,
        ax=ax, cbar_kws={"label": "Count"},
        linewidths=0.5, linecolor="#30363d",
    )
    ax.set_xlabel("Predicted (k-means on word/token/char count)", color="#c9d1d9")
    ax.set_ylabel("True (steering direction)", color="#c9d1d9")
    ax.set_title(
        f"K-Means Confusion Matrix (k=3, accuracy={accuracy:.1%})\n"
        f"Features: word count, token count, char count",
        color="#c9d1d9",
    )
    ax.tick_params(colors="#c9d1d9")
    plt.setp(ax.get_xticklabels(), color="#c9d1d9")
    plt.setp(ax.get_yticklabels(), color="#c9d1d9")

    plt.tight_layout()
    path = output_dir / "confusion_matrix_kmeans.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")

    # Also plot false positives/negatives detail
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d1117")

    for idx, direction in enumerate(label_order):
        ax = axes[idx]
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.set_title(f"True: {direction}", color="#c9d1d9")

        mask = np.array(true_labels) == direction
        correct = np.array(pred_labels)[mask] == direction
        fp_labels = np.array(pred_labels)[mask][~correct]

        words = X[mask, 0]  # word counts
        colors = ["#2ea043" if c else "#f85149" for c in correct]
        ax.scatter(range(len(words)), words, c=colors, s=15, alpha=0.7)
        ax.set_xlabel("Prompt index", color="#c9d1d9")
        ax.set_ylabel("Word count", color="#c9d1d9")

        n_correct = correct.sum()
        n_total = len(correct)
        ax.text(0.95, 0.95, f"{n_correct}/{n_total} correct",
                transform=ax.transAxes, ha="right", va="top",
                color="#2ea043" if n_correct > n_total * 0.8 else "#f85149",
                fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    plt.suptitle("Per-Class Prediction (green=correct, red=misclassified)", color="#c9d1d9")
    plt.tight_layout()
    path = output_dir / "per_class_detail.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": round(accuracy, 4),
        "classification_report": {k: v for k, v in report.items() if k != "accuracy"},
        "centroids_words": km.cluster_centers_[:, 0].tolist(),
        "cluster_to_label": {str(k): v for k, v in cluster_to_label.items()},
    }


def plot_distributions(records, output_dir):
    """Histograms and scatter plots."""
    tw = [r["terse_words"] for r in records]
    bw = [r["baseline_words"] for r in records]
    vw = [r["verbose_words"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    bins = np.linspace(0, max(max(tw), max(bw), max(vw)) + 10, 40)
    ax1.hist(tw, bins=bins, alpha=0.7, label=f"Terse (μ={np.mean(tw):.0f})", color="#58a6ff")
    ax1.hist(bw, bins=bins, alpha=0.5, label=f"Baseline (μ={np.mean(bw):.0f})", color="#8b949e")
    ax1.hist(vw, bins=bins, alpha=0.7, label=f"Verbose (μ={np.mean(vw):.0f})", color="#f0883e")
    ax1.set_xlabel("Word Count")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Word Count Distribution")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    # Scatter: baseline vs terse/verbose
    maxval = max(max(bw), max(tw), max(vw)) + 10
    ax2.scatter(bw, tw, alpha=0.5, s=15, color="#58a6ff", label="Terse")
    ax2.scatter(bw, vw, alpha=0.5, s=15, color="#f0883e", label="Verbose")
    ax2.plot([0, maxval], [0, maxval], "--", color="#8b949e", alpha=0.5, label="y=x")
    ax2.set_xlabel("Baseline Words")
    ax2.set_ylabel("Steered Words")
    ax2.set_title("Baseline vs Steered (blue below, orange above = working)")
    ax2.set_xlim(0, maxval)
    ax2.set_ylim(0, maxval)
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    plt.tight_layout()
    path = output_dir / "distributions_and_scatter.png"
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")


def compute_summary(records):
    """Aggregate statistics."""
    def stats(key):
        vals = [r[key] for r in records]
        a = np.array(vals)
        return {"mean": round(float(np.mean(a)), 1), "median": float(np.median(a)),
                "std": round(float(np.std(a)), 1), "min": int(np.min(a)), "max": int(np.max(a))}

    n = len(records)
    return {
        "n": n,
        "terse_words": stats("terse_words"),
        "baseline_words": stats("baseline_words"),
        "verbose_words": stats("verbose_words"),
        "terse_tokens": stats("terse_tokens"),
        "baseline_tokens": stats("baseline_tokens"),
        "verbose_tokens": stats("verbose_tokens"),
        "directional": {
            "terse_lt_baseline": sum(1 for r in records if r["terse_words"] < r["baseline_words"]),
            "verbose_gt_baseline": sum(1 for r in records if r["verbose_words"] > r["baseline_words"]),
            "terse_lt_verbose": sum(1 for r in records if r["terse_words"] < r["verbose_words"]),
            "full_order": sum(1 for r in records
                              if r["terse_words"] < r["baseline_words"] < r["verbose_words"]),
            "terse_eq_baseline": sum(1 for r in records if r["terse_eq_baseline"]),
            "verbose_eq_baseline": sum(1 for r in records if r["verbose_eq_baseline"]),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    prompts = PROMPTS[:args.num_prompts]
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Config provenance
    config = {
        "model_id": MODEL_ID,
        "layer": args.layer,
        "alpha": args.alpha,
        "max_tokens": args.max_tokens,
        "num_prompts": len(prompts),
        "style_pair": STYLE_PAIRS["terse"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "normalize": False,
    }

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"Computing steering vector (layer={args.layer})...")
    vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=args.layer)
    config["vec_norm"] = round(vec.norm().item(), 4)

    print(f"\nEval: {len(prompts)} prompts × 3 directions, α=±{args.alpha}, layer={args.layer}")
    t_total = time.monotonic()
    records = run_eval(model, tokenizer, vec, prompts, args.alpha, args.layer, args.max_tokens)
    config["total_elapsed_s"] = round(time.monotonic() - t_total, 1)

    summary = compute_summary(records)

    print("\nGenerating plots...")
    plot_distributions(records, OUTPUT_DIR)
    cm_stats = cluster_and_evaluate(records, OUTPUT_DIR)

    # Print summary
    d = summary["directional"]
    n = summary["n"]
    print(f"\n{'='*60}")
    print(f"  Terse vs Verbose: {n} prompts, α=±{args.alpha}, layer={args.layer}")
    print(f"{'='*60}")
    print(f"  {'':12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Min':>5s} {'Max':>5s}")
    for label in ["terse", "baseline", "verbose"]:
        s = summary[f"{label}_words"]
        print(f"  {label:12s} {s['mean']:8.1f} {s['median']:8.0f} {s['std']:8.1f} {s['min']:5d} {s['max']:5d}")
    print(f"\n  Directional accuracy:")
    print(f"    terse < baseline:   {d['terse_lt_baseline']:3d}/{n} ({d['terse_lt_baseline']/n*100:.0f}%)")
    print(f"    verbose > baseline: {d['verbose_gt_baseline']:3d}/{n} ({d['verbose_gt_baseline']/n*100:.0f}%)")
    print(f"    terse < verbose:    {d['terse_lt_verbose']:3d}/{n} ({d['terse_lt_verbose']/n*100:.0f}%)")
    print(f"    full T<B<V:         {d['full_order']:3d}/{n} ({d['full_order']/n*100:.0f}%)")
    print(f"\n  K-means clustering accuracy: {cm_stats['accuracy']:.1%}")
    print(f"{'='*60}")

    # Save everything
    # Strip full text from records for a lighter summary file
    records_lite = []
    for r in records:
        lite = {k: v for k, v in r.items() if not k.endswith("_text")}
        for d in ["terse", "baseline", "verbose"]:
            lite[f"{d}_preview"] = r[f"{d}_text"][:100]
        records_lite.append(lite)

    out_path = OUTPUT_DIR / "terse_verbose_eval.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": config,
            "summary": summary,
            "clustering": cm_stats,
            "records": records_lite,
        }, f, indent=2)
    print(f"\nResults: {out_path}")

    # Also save full text for reproduction
    full_path = OUTPUT_DIR / "terse_verbose_full.json"
    with open(full_path, "w") as f:
        json.dump({"config": config, "records": records}, f, indent=2)
    print(f"Full text: {full_path}")


if __name__ == "__main__":
    main()
