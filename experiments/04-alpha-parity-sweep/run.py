"""
Alpha Parity Sweep: find the alpha where injection matches prompt-level control.

CPRR-9 showed alpha=2.0 overshoots prompt-level "tersely/verbosely" by 3x.
This experiment sweeps alpha from 0.1 to 3.0 to find the crossover point
where injection word count matches prompt-level word count.

Also tests: does a single alpha work for both directions, or is the
terse/verbose parity asymmetric?

Usage:
    python experiments/04-alpha-parity-sweep/run.py
    python experiments/04-alpha-parity-sweep/run.py --cprr
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from actadd import MODEL_ID, compute_steering_vector, generate_steered

QUESTIONS = [
    "Why does ice float on water?",
    "How does a vaccine work?",
    "Why do we dream?",
    "What causes inflation?",
    "How does a compass work?",
]

ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
LAYER = 12


def tfidf_cosine(text_a, text_b):
    words_a = re.findall(r'\w+', text_a.lower())
    words_b = re.findall(r'\w+', text_b.lower())
    if not words_a or not words_b:
        return 0.0
    tf_a, tf_b = Counter(words_a), Counter(words_b)
    vocab = set(tf_a) | set(tf_b)
    dot = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in vocab)
    na = math.sqrt(sum(v**2 for v in tf_a.values()))
    nb = math.sqrt(sum(v**2 for v in tf_b.values()))
    return dot / (na * nb) if na and nb else 0.0


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    vec = compute_steering_vector(model, tokenizer, "terse", LAYER)

    # Phase 1: prompt-level targets
    print("\n=== Phase 1: Prompt-level targets ===")
    prompt_terse_words = []
    prompt_verbose_words = []
    baseline_words_list = []
    baselines = {}

    for q in QUESTIONS:
        bl = generate_steered(model, tokenizer, q, vec, LAYER, alpha=0.0, max_new_tokens=300)
        pt = generate_steered(model, tokenizer, q + " Answer as tersely as possible.", vec, LAYER, alpha=0.0, max_new_tokens=300)
        pv = generate_steered(model, tokenizer, q + " Answer as verbosely as possible.", vec, LAYER, alpha=0.0, max_new_tokens=300)
        baselines[q] = bl
        baseline_words_list.append(len(bl.split()))
        prompt_terse_words.append(len(pt.split()))
        prompt_verbose_words.append(len(pv.split()))

    mean_bl = sum(baseline_words_list) / len(baseline_words_list)
    mean_pt = sum(prompt_terse_words) / len(prompt_terse_words)
    mean_pv = sum(prompt_verbose_words) / len(prompt_verbose_words)
    print(f"  Baseline mean:       {mean_bl:.0f}w")
    print(f"  Prompt-terse mean:   {mean_pt:.0f}w  (target for +alpha)")
    print(f"  Prompt-verbose mean: {mean_pv:.0f}w  (target for -alpha)")

    # Phase 2: alpha sweep
    print(f"\n=== Phase 2: Alpha sweep ({len(ALPHAS)} alphas × {len(QUESTIONS)} questions) ===")
    sweep = {}
    total = len(ALPHAS) * len(QUESTIONS) * 2
    done = 0
    start = time.time()

    for alpha in ALPHAS:
        terse_counts = []
        verbose_counts = []
        terse_topics = []
        verbose_topics = []

        for q in QUESTIONS:
            t_out = generate_steered(model, tokenizer, q, vec, LAYER, alpha=alpha, max_new_tokens=300)
            v_out = generate_steered(model, tokenizer, q, vec, LAYER, alpha=-alpha, max_new_tokens=300)
            tw = len(t_out.split())
            vw = len(v_out.split())
            terse_counts.append(tw)
            verbose_counts.append(vw)
            terse_topics.append(tfidf_cosine(baselines[q], t_out))
            verbose_topics.append(tfidf_cosine(baselines[q], v_out))
            done += 2
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 1
            remaining = (total - done) / rate
            print(f"  [{done}/{total}] α={alpha:.2f} {q[:30]:30s} T={tw:3d}w V={vw:3d}w ({remaining:.0f}s)")

        sweep[alpha] = {
            "terse_mean": round(sum(terse_counts) / len(terse_counts), 1),
            "terse_min": min(terse_counts),
            "terse_max": max(terse_counts),
            "verbose_mean": round(sum(verbose_counts) / len(verbose_counts), 1),
            "verbose_min": min(verbose_counts),
            "verbose_max": max(verbose_counts),
            "terse_topic": round(sum(terse_topics) / len(terse_topics), 3),
            "verbose_topic": round(sum(verbose_topics) / len(verbose_topics), 3),
            "terse_ratio": round(sum(terse_counts) / len(terse_counts) / max(mean_pt, 1), 2),
            "verbose_ratio": round(sum(verbose_counts) / len(verbose_counts) / max(mean_pv, 1), 2),
        }

    # Phase 3: find parity
    print("\n=== Phase 3: Parity analysis ===")
    print(f"\n{'Alpha':>6} {'T-Mean':>7} {'T-Ratio':>8} {'V-Mean':>7} {'V-Ratio':>8} {'T-Topic':>8} {'V-Topic':>8}")
    print("-" * 62)

    best_terse_alpha = None
    best_verbose_alpha = None
    best_terse_diff = float('inf')
    best_verbose_diff = float('inf')

    for alpha in ALPHAS:
        s = sweep[alpha]
        t_diff = abs(s["terse_mean"] - mean_pt)
        v_diff = abs(s["verbose_mean"] - mean_pv)
        if t_diff < best_terse_diff:
            best_terse_diff = t_diff
            best_terse_alpha = alpha
        if v_diff < best_verbose_diff:
            best_verbose_diff = v_diff
            best_verbose_alpha = alpha

        t_mark = " <--" if abs(s["terse_ratio"] - 1.0) < 0.15 else ""
        v_mark = " <--" if abs(s["verbose_ratio"] - 1.0) < 0.15 else ""
        print(f"{alpha:>6.2f} {s['terse_mean']:>7.0f} {s['terse_ratio']:>7.2f}x "
              f"{s['verbose_mean']:>7.0f} {s['verbose_ratio']:>7.2f}x "
              f"{s['terse_topic']:>8.3f} {s['verbose_topic']:>8.3f}"
              f"{t_mark}{v_mark}")

    print(f"\n  Terse parity alpha:   {best_terse_alpha} (diff={best_terse_diff:.0f}w from target {mean_pt:.0f}w)")
    print(f"  Verbose parity alpha: {best_verbose_alpha} (diff={best_verbose_diff:.0f}w from target {mean_pv:.0f}w)")
    asymmetric = best_terse_alpha != best_verbose_alpha
    print(f"  Asymmetric: {asymmetric} ({'different alphas needed' if asymmetric else 'single alpha works'})")

    # Save
    output = {
        "model": MODEL_ID, "layer": LAYER,
        "targets": {"baseline": mean_bl, "prompt_terse": mean_pt, "prompt_verbose": mean_pv},
        "sweep": {str(a): v for a, v in sweep.items()},
        "parity": {
            "terse_alpha": best_terse_alpha, "verbose_alpha": best_verbose_alpha,
            "asymmetric": asymmetric,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_dir = str(Path(__file__).resolve().parent / "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "alpha_parity_sweep.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {os.path.join(output_dir, 'alpha_parity_sweep.json')}")

    # Visualize
    print("\n=== Generating visualization ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Alpha Parity Sweep: Injection vs Prompt-Level Control",
                 color="#e0e0e0", fontweight="bold", fontsize=13)

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")
        ax.spines["bottom"].set_color("#37374f")
        ax.spines["left"].set_color("#37374f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, color="#37374f")

    alphas_arr = np.array(ALPHAS)
    terse_means = [sweep[a]["terse_mean"] for a in ALPHAS]
    verbose_means = [sweep[a]["verbose_mean"] for a in ALPHAS]
    terse_topics = [sweep[a]["terse_topic"] for a in ALPHAS]
    verbose_topics = [sweep[a]["verbose_topic"] for a in ALPHAS]

    # Left: word count vs alpha
    ax1.plot(alphas_arr, terse_means, "o-", color="#4caf50", linewidth=2, label="Inject terse (+α)")
    ax1.plot(alphas_arr, verbose_means, "o-", color="#f44336", linewidth=2, label="Inject verbose (-α)")
    ax1.axhline(y=mean_pt, color="#4caf50", linestyle="--", alpha=0.5, label=f"Prompt terse target ({mean_pt:.0f}w)")
    ax1.axhline(y=mean_pv, color="#f44336", linestyle="--", alpha=0.5, label=f"Prompt verbose target ({mean_pv:.0f}w)")
    ax1.axhline(y=mean_bl, color="#607d8b", linestyle=":", alpha=0.5, label=f"Baseline ({mean_bl:.0f}w)")
    ax1.fill_between(alphas_arr, mean_pt * 0.8, mean_pt * 1.2, alpha=0.06, color="#4caf50")
    ax1.fill_between(alphas_arr, mean_pv * 0.8, mean_pv * 1.2, alpha=0.06, color="#f44336")
    ax1.set_xlabel("Alpha", color="#e0e0e0")
    ax1.set_ylabel("Mean Word Count", color="#e0e0e0")
    ax1.set_title("Word Count vs Alpha", color="#e0e0e0")
    ax1.legend(fontsize=7, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0")

    # Right: topic cosine vs alpha
    ax2.plot(alphas_arr, terse_topics, "o-", color="#4caf50", linewidth=2, label="Terse topic cosine")
    ax2.plot(alphas_arr, verbose_topics, "o-", color="#f44336", linewidth=2, label="Verbose topic cosine")
    ax2.axhline(y=0.6, color="#ff5722", linestyle="--", alpha=0.5, label="Threshold (0.6)")
    ax2.set_xlabel("Alpha", color="#e0e0e0")
    ax2.set_ylabel("Topic Cosine (vs baseline)", color="#e0e0e0")
    ax2.set_title("Semantic Preservation vs Alpha", color="#e0e0e0")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(output_dir, "alpha_parity_sweep.png"), dpi=200, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'alpha_parity_sweep.png')}")

    # CPRR
    if "--cprr" in sys.argv:
        print("\n=== Submitting to CPRR ===")
        def cprr(*args):
            r = subprocess.run(["cprr"] + list(args), capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  cprr {' '.join(args[:2])}: {r.stdout.strip()}")
        cprr("add", "Alpha parity point is asymmetric: terse and verbose need different alphas",
             "--hypothesis",
             f"Terse parity alpha={best_terse_alpha}, verbose parity alpha={best_verbose_alpha}. "
             f"Asymmetric={asymmetric}. Topic cosine >0.6 at parity alphas.",
             "-t", "alpha,parity,experiment")
        cprr("evidence", "10",
             f"Sweep {ALPHAS}: terse parity at α={best_terse_alpha} ({sweep[best_terse_alpha]['terse_mean']}w vs target {mean_pt:.0f}w), "
             f"verbose parity at α={best_verbose_alpha} ({sweep[best_verbose_alpha]['verbose_mean']}w vs target {mean_pv:.0f}w). "
             f"Topic at parity: terse={sweep[best_terse_alpha]['terse_topic']}, verbose={sweep[best_verbose_alpha]['verbose_topic']}. "
             f"[confidence: empirical, source: this-project-sweep]")
        cprr("evidence", "10",
             f"Full curve: terse goes {sweep[ALPHAS[0]]['terse_mean']}w→{sweep[ALPHAS[-1]]['terse_mean']}w, "
             f"verbose goes {sweep[ALPHAS[0]]['verbose_mean']}w→{sweep[ALPHAS[-1]]['verbose_mean']}w across α={ALPHAS[0]}-{ALPHAS[-1]}. "
             f"[confidence: empirical, source: this-project-sweep]")
        cprr("next", "10")
        cprr("next", "10")


if __name__ == "__main__":
    main()
