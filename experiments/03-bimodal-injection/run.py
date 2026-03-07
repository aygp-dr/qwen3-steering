"""
Bimodal Injection Test: confirm steering vector injection vs prompt-level control.

Compares three conditions across 10 bimodal questions:
  A. BASELINE: raw question, no steering
  B. PROMPT-LEVEL: question + "Answer as tersely/verbosely as possible"
  C. INJECTION: raw question + ActAdd steering vector at layer 12 (±alpha)

Each question admits both a single-sentence answer and a multi-paragraph
answer — neither is "wrong." The test measures whether injection achieves
comparable control to prompt-level instruction, and whether the two methods
produce geometrically similar effects in activation space.

Hypothesis (CPRR):
  Injection at L12 with alpha=2.0 produces word count within 20% of
  prompt-level "tersely/verbosely" instruction, with topic cosine > 0.6
  to the baseline answer. The terse condition produces <30 words for all
  10 questions; the verbose condition produces >100 words for all 10.

Usage:
    python experiments/03-bimodal-injection/run.py
    python experiments/03-bimodal-injection/run.py --alpha 3.0
    python experiments/03-bimodal-injection/run.py --visualize
    python experiments/03-bimodal-injection/run.py --cprr
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from actadd import (
    MODEL_ID, STYLE_PAIRS, compute_steering_vector, generate_steered,
)

# ── Bimodal questions ────────────────────────────────────────────────────────
# Each has a defensible one-sentence answer AND a legitimate three-paragraph
# answer. Neither length is "correct."

QUESTIONS = [
    "Why does ice float on water?",
    "How does a vaccine work?",
    "Why do we dream?",
    "What causes inflation?",
    "How does a compass work?",
    "Why do leaves change color in autumn?",
    "What is a black hole?",
    "How does bread rise?",
    "Why do humans need sleep?",
    "What causes thunder?",
]

# ── Conditions ───────────────────────────────────────────────────────────────

CONDITIONS = {
    "baseline": {
        "suffix": "",
        "alpha": 0.0,
        "method": "none",
    },
    "prompt_terse": {
        "suffix": " Answer as tersely as possible.",
        "alpha": 0.0,
        "method": "prompt",
    },
    "prompt_verbose": {
        "suffix": " Answer as verbosely as possible.",
        "alpha": 0.0,
        "method": "prompt",
    },
    "inject_terse": {
        "suffix": "",
        "alpha": 2.0,  # overridden by --alpha
        "method": "injection",
    },
    "inject_verbose": {
        "suffix": "",
        "alpha": -2.0,  # negative = verbose direction
        "method": "injection",
    },
}

INJECTION_LAYER = 12


def tfidf_cosine(text_a, text_b):
    """TF-IDF cosine similarity between two texts."""
    words_a = re.findall(r'\w+', text_a.lower())
    words_b = re.findall(r'\w+', text_b.lower())
    if not words_a or not words_b:
        return 0.0
    tf_a = Counter(words_a)
    tf_b = Counter(words_b)
    vocab = set(tf_a.keys()) | set(tf_b.keys())
    dot = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in vocab)
    na = math.sqrt(sum(v ** 2 for v in tf_a.values()))
    nb = math.sqrt(sum(v ** 2 for v in tf_b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_experiment(model, tokenizer, alpha, layer):
    """Run all conditions across all questions. Returns list of result dicts."""
    # Precompute steering vector once
    vec = compute_steering_vector(model, tokenizer, "terse", layer)
    vec_norm = vec.norm().item()

    results = []
    total = len(QUESTIONS) * len(CONDITIONS)
    done = 0
    start = time.time()

    for q_idx, question in enumerate(QUESTIONS):
        row = {"question": question, "q_idx": q_idx}

        for cond_name, cond in CONDITIONS.items():
            prompt = question + cond["suffix"]

            if cond["method"] == "injection":
                # Use raw question, inject vector
                effective_alpha = alpha if "terse" in cond_name else -alpha
                text = generate_steered(
                    model, tokenizer, question, vec, layer,
                    alpha=effective_alpha, max_new_tokens=300,
                )
            else:
                # Prompt-level control (or baseline)
                text = generate_steered(
                    model, tokenizer, prompt, vec, layer,
                    alpha=0.0, max_new_tokens=300,
                )

            words = text.split()
            word_count = len(words)
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            wps = word_count / max(len(sentences), 1)

            row[cond_name] = {
                "text": text,
                "word_count": word_count,
                "sentence_count": len(sentences),
                "words_per_sentence": round(wps, 1),
            }

            done += 1
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] Q{q_idx+1} {cond_name:16s} → {word_count:3d} words "
                  f"({remaining:.0f}s remaining)")

        # Compute topic cosines against baseline
        baseline_text = row["baseline"]["text"]
        for cond_name in CONDITIONS:
            if cond_name == "baseline":
                row[cond_name]["topic_cosine"] = 1.0
            else:
                row[cond_name]["topic_cosine"] = round(
                    tfidf_cosine(baseline_text, row[cond_name]["text"]), 3
                )

        results.append(row)

    return results, vec_norm


def compute_summary(results, alpha):
    """Compute aggregate statistics and hypothesis tests."""
    summary = {
        "alpha": alpha,
        "layer": INJECTION_LAYER,
        "n_questions": len(QUESTIONS),
        "conditions": {},
    }

    for cond_name in CONDITIONS:
        word_counts = [r[cond_name]["word_count"] for r in results]
        topic_cosines = [r[cond_name]["topic_cosine"] for r in results]

        summary["conditions"][cond_name] = {
            "mean_words": round(sum(word_counts) / len(word_counts), 1),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "std_words": round(
                (sum((w - sum(word_counts)/len(word_counts))**2
                     for w in word_counts) / len(word_counts)) ** 0.5, 1
            ),
            "mean_topic_cosine": round(sum(topic_cosines) / len(topic_cosines), 3),
            "all_under_30": all(w < 30 for w in word_counts),
            "all_over_100": all(w > 100 for w in word_counts),
        }

    # Hypothesis tests
    it = summary["conditions"]["inject_terse"]
    iv = summary["conditions"]["inject_verbose"]
    pt = summary["conditions"]["prompt_terse"]
    pv = summary["conditions"]["prompt_verbose"]
    bl = summary["conditions"]["baseline"]

    summary["hypotheses"] = {
        "H1_inject_terse_under_30": {
            "claim": "All 10 inject_terse outputs < 30 words",
            "result": it["all_under_30"],
            "detail": f"min={it['min_words']}, max={it['max_words']}, mean={it['mean_words']}",
        },
        "H2_inject_verbose_over_100": {
            "claim": "All 10 inject_verbose outputs > 100 words",
            "result": iv["all_over_100"],
            "detail": f"min={iv['min_words']}, max={iv['max_words']}, mean={iv['mean_words']}",
        },
        "H3_inject_within_20pct_of_prompt": {
            "claim": "Inject word count within 20% of prompt-level word count",
            "terse_ratio": round(it["mean_words"] / max(pt["mean_words"], 1), 2),
            "verbose_ratio": round(iv["mean_words"] / max(pv["mean_words"], 1), 2),
            "result": (0.8 <= it["mean_words"] / max(pt["mean_words"], 1) <= 1.2 and
                       0.8 <= iv["mean_words"] / max(pv["mean_words"], 1) <= 1.2),
        },
        "H4_topic_preserved": {
            "claim": "Topic cosine > 0.6 for all injection conditions",
            "inject_terse_topic": it["mean_topic_cosine"],
            "inject_verbose_topic": iv["mean_topic_cosine"],
            "result": it["mean_topic_cosine"] > 0.6 and iv["mean_topic_cosine"] > 0.6,
        },
        "H5_baseline_between": {
            "claim": "inject_terse < baseline < inject_verbose (word count)",
            "result": it["mean_words"] < bl["mean_words"] < iv["mean_words"],
        },
    }

    return summary


def print_table(results, summary):
    """Print a readable comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Q#':>3} {'Question':<40} {'Base':>5} {'P-Ter':>5} {'P-Ver':>5} "
          f"{'I-Ter':>5} {'I-Ver':>5} {'TopT':>5} {'TopV':>5}")
    print("-" * 100)

    for r in results:
        q_short = r["question"][:38]
        print(f"{r['q_idx']+1:>3} {q_short:<40} "
              f"{r['baseline']['word_count']:>5} "
              f"{r['prompt_terse']['word_count']:>5} "
              f"{r['prompt_verbose']['word_count']:>5} "
              f"{r['inject_terse']['word_count']:>5} "
              f"{r['inject_verbose']['word_count']:>5} "
              f"{r['inject_terse']['topic_cosine']:>5.2f} "
              f"{r['inject_verbose']['topic_cosine']:>5.2f}")

    print("-" * 100)
    s = summary["conditions"]
    print(f"{'':>3} {'MEAN':<40} "
          f"{s['baseline']['mean_words']:>5.0f} "
          f"{s['prompt_terse']['mean_words']:>5.0f} "
          f"{s['prompt_verbose']['mean_words']:>5.0f} "
          f"{s['inject_terse']['mean_words']:>5.0f} "
          f"{s['inject_verbose']['mean_words']:>5.0f} "
          f"{s['inject_terse']['mean_topic_cosine']:>5.2f} "
          f"{s['inject_verbose']['mean_topic_cosine']:>5.2f}")

    print("\n" + "=" * 100)
    print("HYPOTHESIS TESTS")
    print("-" * 100)
    for hid, h in summary["hypotheses"].items():
        status = "PASS" if h["result"] else "FAIL"
        print(f"  [{status:4s}] {h['claim']}")
        detail = {k: v for k, v in h.items() if k not in ("claim", "result")}
        if detail:
            print(f"         {detail}")
    print("=" * 100)


def plot_results(results, summary, output_dir):
    """Generate visualization comparing all 5 conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_q = len(results)
    x = np.arange(n_q)
    width = 0.15

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Bimodal Injection Test: Steering Vector vs Prompt-Level Control",
                 color="#e0e0e0", fontweight="bold", fontsize=14, y=0.98)

    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")
        ax.spines["bottom"].set_color("#37374f")
        ax.spines["left"].set_color("#37374f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cond_colors = {
        "baseline":       "#607d8b",
        "prompt_terse":   "#2196f3",
        "prompt_verbose": "#ff9800",
        "inject_terse":   "#4caf50",
        "inject_verbose": "#f44336",
    }
    cond_labels = {
        "baseline":       "Baseline",
        "prompt_terse":   "Prompt: terse",
        "prompt_verbose": "Prompt: verbose",
        "inject_terse":   "Inject: terse",
        "inject_verbose": "Inject: verbose",
    }

    # ── Panel 1: Grouped bar chart of word counts ──
    ax = axes[0, 0]
    for i, (cond_name, color) in enumerate(cond_colors.items()):
        counts = [r[cond_name]["word_count"] for r in results]
        ax.bar(x + i * width, counts, width, label=cond_labels[cond_name],
               color=color, alpha=0.85)
    ax.set_xlabel("Question", color="#e0e0e0")
    ax.set_ylabel("Word Count", color="#e0e0e0")
    ax.set_title("Word Count by Condition", color="#e0e0e0", fontweight="bold")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_q)], fontsize=8)
    ax.legend(fontsize=7, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0",
              ncol=2)
    ax.grid(alpha=0.15, color="#37374f", axis="y")

    # ── Panel 2: Prompt vs Injection scatter ──
    ax = axes[0, 1]
    pt_words = [r["prompt_terse"]["word_count"] for r in results]
    it_words = [r["inject_terse"]["word_count"] for r in results]
    pv_words = [r["prompt_verbose"]["word_count"] for r in results]
    iv_words = [r["inject_verbose"]["word_count"] for r in results]

    ax.scatter(pt_words, it_words, c="#4caf50", s=60, label="Terse", edgecolors="white", linewidth=0.5)
    ax.scatter(pv_words, iv_words, c="#f44336", s=60, label="Verbose", edgecolors="white", linewidth=0.5)

    # Identity line
    max_val = max(max(pt_words + pv_words), max(it_words + iv_words)) + 10
    ax.plot([0, max_val], [0, max_val], "--", color="#ffeb3b", alpha=0.5, label="y=x")
    # 20% bands
    ax.fill_between([0, max_val], [0, max_val * 0.8], [0, max_val * 1.2],
                    alpha=0.08, color="#ffeb3b")

    ax.set_xlabel("Prompt-Level Word Count", color="#e0e0e0")
    ax.set_ylabel("Injection Word Count", color="#e0e0e0")
    ax.set_title("Prompt vs Injection: Word Count Parity", color="#e0e0e0", fontweight="bold")
    ax.legend(fontsize=8, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0")
    ax.grid(alpha=0.15, color="#37374f")

    # ── Panel 3: Topic cosine preservation ──
    ax = axes[1, 0]
    conditions_to_plot = ["prompt_terse", "prompt_verbose", "inject_terse", "inject_verbose"]
    for i, cond_name in enumerate(conditions_to_plot):
        cosines = [r[cond_name]["topic_cosine"] for r in results]
        positions = x + i * width
        ax.bar(positions, cosines, width, label=cond_labels[cond_name],
               color=cond_colors[cond_name], alpha=0.85)
    ax.axhline(y=0.6, color="#ff5722", linestyle="--", alpha=0.5, label="threshold (0.6)")
    ax.set_xlabel("Question", color="#e0e0e0")
    ax.set_ylabel("Topic Cosine (vs baseline)", color="#e0e0e0")
    ax.set_title("Semantic Preservation", color="#e0e0e0", fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_q)], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0",
              ncol=2)
    ax.grid(alpha=0.15, color="#37374f", axis="y")

    # ── Panel 4: Summary text ──
    ax = axes[1, 1]
    ax.axis("off")

    s = summary["conditions"]
    h = summary["hypotheses"]
    lines = [
        f"BIMODAL INJECTION TEST — Layer {summary['layer']}, α={summary['alpha']}",
        f"{'─' * 55}",
        f"",
        f"{'Condition':<20} {'Mean':>6} {'Min':>5} {'Max':>5} {'Topic':>6}",
        f"{'─' * 55}",
    ]
    for cond_name in CONDITIONS:
        c = s[cond_name]
        lines.append(
            f"{cond_labels.get(cond_name, cond_name):<20} "
            f"{c['mean_words']:>6.0f} {c['min_words']:>5} {c['max_words']:>5} "
            f"{c['mean_topic_cosine']:>6.3f}"
        )
    lines.extend([
        f"",
        f"HYPOTHESES:",
    ])
    for hid, hv in h.items():
        status = "PASS" if hv["result"] else "FAIL"
        lines.append(f"  [{status}] {hv['claim'][:48]}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace", color="#e0e0e0",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#2a2a4a", edgecolor="#37374f"))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "bimodal_injection_test.png")
    fig.savefig(path, dpi=200, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {path}")


def submit_cprr(summary):
    """Submit results to CPRR."""
    def cprr(*args):
        result = subprocess.run(["cprr"] + list(args), capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  cprr {' '.join(args[:2])}: {result.stdout.strip()}")
        return result.returncode == 0

    cprr("add",
         "Injection at L12 matches prompt-level terse/verbose control on bimodal questions",
         "--hypothesis",
         f"ActAdd injection at layer {INJECTION_LAYER} with alpha={summary['alpha']} "
         f"produces word count within 20% of prompt-level 'tersely/verbosely' instruction "
         f"across 10 bimodal questions. Terse <30w for all, verbose >100w for all, "
         f"topic cosine >0.6 for both.",
         "-t", "injection,bimodal,experiment")

    # Evidence from each hypothesis
    h = summary["hypotheses"]
    s = summary["conditions"]

    cprr("evidence", "9",
         f"H1 inject_terse <30w: {'PASS' if h['H1_inject_terse_under_30']['result'] else 'FAIL'}. "
         f"mean={s['inject_terse']['mean_words']}w, "
         f"min={s['inject_terse']['min_words']}w, max={s['inject_terse']['max_words']}w. "
         f"[confidence: empirical, source: this-project-sweep]")

    cprr("evidence", "9",
         f"H2 inject_verbose >100w: {'PASS' if h['H2_inject_verbose_over_100']['result'] else 'FAIL'}. "
         f"mean={s['inject_verbose']['mean_words']}w, "
         f"min={s['inject_verbose']['min_words']}w, max={s['inject_verbose']['max_words']}w. "
         f"H3 within 20% of prompt: {'PASS' if h['H3_inject_within_20pct_of_prompt']['result'] else 'FAIL'} "
         f"(terse ratio={h['H3_inject_within_20pct_of_prompt']['terse_ratio']}, "
         f"verbose ratio={h['H3_inject_within_20pct_of_prompt']['verbose_ratio']}). "
         f"H4 topic>0.6: {'PASS' if h['H4_topic_preserved']['result'] else 'FAIL'} "
         f"(terse={s['inject_terse']['mean_topic_cosine']}, verbose={s['inject_verbose']['mean_topic_cosine']}). "
         f"[confidence: empirical, source: this-project-sweep]")

    cprr("next", "9")  # open -> testing
    cprr("next", "9")  # testing -> confirmed/refuted


def main():
    parser = argparse.ArgumentParser(description="Bimodal injection test")
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Injection alpha magnitude (default: 2.0)")
    parser.add_argument("--layer", type=int, default=INJECTION_LAYER,
                        help=f"Injection layer (default: {INJECTION_LAYER})")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization")
    parser.add_argument("--cprr", action="store_true",
                        help="Submit results to CPRR")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "output"),
                        help="Output directory")
    args = parser.parse_args()

    injection_layer = args.layer

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"\nBimodal Injection Test: {len(QUESTIONS)} questions × {len(CONDITIONS)} conditions")
    print(f"Layer: {injection_layer}, Alpha: ±{args.alpha}")

    print("\n=== Running experiment ===")
    results, vec_norm = run_experiment(model, tokenizer, args.alpha, injection_layer)

    summary = compute_summary(results, args.alpha)
    summary["vec_norm"] = round(vec_norm, 2)

    print_table(results, summary)

    # Save full results
    output_path = os.path.join(args.output_dir, "bimodal_injection_test.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "layer": INJECTION_LAYER,
            "alpha": args.alpha,
            "vec_norm": summary["vec_norm"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "questions": QUESTIONS,
            "results": results,
            "summary": summary,
        }, f, indent=2, default=str)
    print(f"\nFull results: {output_path}")

    if args.visualize:
        print("\n=== Generating visualization ===")
        plot_results(results, summary, args.output_dir)

    if args.cprr:
        print("\n=== Submitting to CPRR ===")
        submit_cprr(summary)


if __name__ == "__main__":
    main()
