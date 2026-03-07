"""
Multi-layer vs Single-layer Injection (CPRR-6).

Tests whether distributing the same total steering pressure across
multiple layers produces better terseness-to-coherence ratio than
concentrating it at one layer.

Configurations:
  A. Single L12, α=2.0           (total pressure = 2.0)
  B. L12+L16, α=1.0 each         (total pressure = 2.0)
  C. L10+L14+L18, α=0.67 each    (total pressure = 2.0)
  D. L12+L15, α=1.0 each         (adjacent layers)
  E. Single L12, α=1.0           (half pressure baseline)

Metrics: word count, topic cosine, perplexity ratio, words-per-sentence.

Usage:
    python experiments/05-multilayer-vs-single/run.py
    python experiments/05-multilayer-vs-single/run.py --cprr
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
from multilayer import generate_multilayer_steered

QUESTIONS = [
    "Why does ice float on water?",
    "How does a vaccine work?",
    "Why do we dream?",
    "What causes inflation?",
    "How does a compass work?",
    "Why do leaves change color in autumn?",
    "What is a black hole?",
    "How does bread rise?",
]


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


def perplexity_ratio(model, tokenizer, baseline_text, steered_text):
    def ce(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        if inputs.input_ids.shape[1] < 2:
            return float('inf')
        with torch.no_grad():
            return model(**inputs, labels=inputs.input_ids).loss.item()
    cb = ce(baseline_text)
    cs = ce(steered_text)
    return round(cs / cb, 3) if cb > 0 and cb != float('inf') else None


def main():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Compute per-layer vectors
    layers_needed = [10, 12, 14, 15, 16, 18]
    vecs = {}
    for l in layers_needed:
        vecs[l] = compute_steering_vector(model, tokenizer, "terse", l)
        print(f"  Vec L{l}: norm={vecs[l].norm():.2f}")

    configs = {
        "baseline": {"type": "single", "layers": [(12, 0.0)]},
        "A_single_L12_a2": {"type": "single", "layers": [(12, 2.0)]},
        "B_dual_L12_L16": {"type": "multi", "layers": [(12, 1.0), (16, 1.0)]},
        "C_triple_L10_L14_L18": {"type": "multi", "layers": [(10, 0.67), (14, 0.67), (18, 0.67)]},
        "D_adjacent_L12_L15": {"type": "multi", "layers": [(12, 1.0), (15, 1.0)]},
        "E_single_L12_a1": {"type": "single", "layers": [(12, 1.0)]},
    }

    total = len(QUESTIONS) * len(configs)
    done = 0
    start = time.time()
    results = []

    print(f"\n=== Running {len(configs)} configs × {len(QUESTIONS)} questions = {total} generations ===")

    for q_idx, question in enumerate(QUESTIONS):
        row = {"question": question, "q_idx": q_idx}

        # Get baseline text for comparison
        baseline_text = None

        for cfg_name, cfg in configs.items():
            if cfg["type"] == "single":
                layer_idx, alpha = cfg["layers"][0]
                text = generate_steered(model, tokenizer, question, vecs[layer_idx],
                                        layer_idx, alpha=alpha, max_new_tokens=300)
            else:
                layer_vec_alpha = [(l, vecs[l], a) for l, a in cfg["layers"]]
                text = generate_multilayer_steered(model, tokenizer, question,
                                                   layer_vec_alpha, max_new_tokens=300)

            if cfg_name == "baseline":
                baseline_text = text

            words = text.split()
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            wps = len(words) / max(len(sentences), 1)

            row[cfg_name] = {
                "word_count": len(words),
                "words_per_sentence": round(wps, 1),
                "topic_cosine": round(tfidf_cosine(baseline_text, text), 3) if baseline_text else 1.0,
                "perplexity_ratio": perplexity_ratio(model, tokenizer, baseline_text, text) if baseline_text and cfg_name != "baseline" else 1.0,
                "text_preview": text[:100],
            }

            done += 1
            elapsed = time.time() - start
            remaining = (total - done) / (done / elapsed) if done > 0 else 0
            print(f"  [{done}/{total}] Q{q_idx+1} {cfg_name:25s} → {len(words):3d}w "
                  f"topic={row[cfg_name]['topic_cosine']:.2f} "
                  f"ppl={row[cfg_name]['perplexity_ratio']} ({remaining:.0f}s)")

        results.append(row)

    # Aggregate
    print("\n=== Summary ===")
    print(f"{'Config':<28} {'Words':>6} {'WPS':>5} {'Topic':>6} {'PPL':>6} {'Verdict'}")
    print("-" * 72)

    agg = {}
    for cfg_name in configs:
        word_counts = [r[cfg_name]["word_count"] for r in results]
        topics = [r[cfg_name]["topic_cosine"] for r in results]
        ppls = [r[cfg_name]["perplexity_ratio"] for r in results if r[cfg_name]["perplexity_ratio"] is not None]
        wpss = [r[cfg_name]["words_per_sentence"] for r in results]

        mean_words = sum(word_counts) / len(word_counts)
        mean_topic = sum(topics) / len(topics)
        mean_ppl = sum(ppls) / len(ppls) if ppls else None
        mean_wps = sum(wpss) / len(wpss)

        # Composite score: lower words + higher topic + lower ppl = better
        if cfg_name != "baseline" and mean_ppl:
            score = (1 / max(mean_words, 1)) * mean_topic * (1 / mean_ppl) * 1000
        else:
            score = 0

        agg[cfg_name] = {
            "mean_words": round(mean_words, 1),
            "mean_topic": round(mean_topic, 3),
            "mean_ppl": round(mean_ppl, 3) if mean_ppl else None,
            "mean_wps": round(mean_wps, 1),
            "score": round(score, 2),
        }

        ppl_str = f"{mean_ppl:.3f}" if mean_ppl else "  -  "
        verdict = ""
        if cfg_name == "baseline":
            verdict = "(reference)"
        elif score > 0:
            best_score = max(v["score"] for k, v in agg.items() if k != "baseline" and v["score"] > 0)
            if score == best_score:
                verdict = "*** BEST ***"

        print(f"{cfg_name:<28} {mean_words:>6.0f} {mean_wps:>5.1f} {mean_topic:>6.3f} {ppl_str:>6} {verdict}")

    # Find winner
    steered = {k: v for k, v in agg.items() if k != "baseline" and v["score"] > 0}
    winner = max(steered, key=lambda k: steered[k]["score"]) if steered else None
    is_multi = winner and configs[winner]["type"] == "multi"

    print(f"\nWinner: {winner} (score={agg[winner]['score'] if winner else 'N/A'})")
    print(f"Multi-layer wins: {is_multi}")

    # Save
    output = {
        "model": MODEL_ID, "layer": 12,
        "configs": {k: {"layers": v["layers"], "type": v["type"]} for k, v in configs.items()},
        "results": results,
        "aggregate": agg,
        "winner": winner,
        "multi_wins": is_multi,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_dir = str(Path(__file__).resolve().parent / "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "multilayer_vs_single.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {os.path.join(output_dir, 'multilayer_vs_single.json')}")

    # Viz
    print("\n=== Generating visualization ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Multi-Layer vs Single-Layer Steering (CPRR-6)",
                 color="#e0e0e0", fontweight="bold", fontsize=13)

    config_names = [k for k in configs if k != "baseline"]
    colors = ["#2196f3", "#4caf50", "#ff9800", "#9c27b0", "#607d8b"]
    x = np.arange(len(config_names))

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")
        ax.spines["bottom"].set_color("#37374f")
        ax.spines["left"].set_color("#37374f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, color="#37374f", axis="y")

    # Word count
    vals = [agg[k]["mean_words"] for k in config_names]
    bars = axes[0].bar(x, vals, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([k.replace("_", "\n") for k in config_names], fontsize=7)
    axes[0].set_ylabel("Mean Words", color="#e0e0e0")
    axes[0].set_title("Word Count (lower = more terse)", color="#e0e0e0")
    if winner:
        idx = config_names.index(winner)
        bars[idx].set_edgecolor("#ffeb3b")
        bars[idx].set_linewidth(3)

    # Topic cosine
    vals = [agg[k]["mean_topic"] for k in config_names]
    axes[1].bar(x, vals, color=colors)
    axes[1].axhline(y=0.6, color="#ff5722", linestyle="--", alpha=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([k.replace("_", "\n") for k in config_names], fontsize=7)
    axes[1].set_ylabel("Topic Cosine", color="#e0e0e0")
    axes[1].set_title("Semantic Preservation (higher = better)", color="#e0e0e0")
    axes[1].set_ylim(0, 1.1)

    # Composite score
    vals = [agg[k]["score"] for k in config_names]
    bars = axes[2].bar(x, vals, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([k.replace("_", "\n") for k in config_names], fontsize=7)
    axes[2].set_ylabel("Composite Score", color="#e0e0e0")
    axes[2].set_title("Terseness × Topic / Perplexity", color="#e0e0e0")
    if winner:
        idx = config_names.index(winner)
        bars[idx].set_edgecolor("#ffeb3b")
        bars[idx].set_linewidth(3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(output_dir, "multilayer_vs_single.png"), dpi=200, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {os.path.join(output_dir, 'multilayer_vs_single.png')}")

    # CPRR
    if "--cprr" in sys.argv:
        print("\n=== Submitting to CPRR ===")
        def cprr(*args):
            r = subprocess.run(["cprr"] + list(args), capture_output=True, text=True)
            if r.returncode == 0:
                print(f"  cprr {' '.join(args[:2])}: {r.stdout.strip()}")

        w = agg[winner] if winner else {}
        a_single = agg.get("A_single_L12_a2", {})
        cprr("evidence", "6",
             f"Winner: {winner} (score={w.get('score','?')}). "
             f"Single L12 α=2.0: {a_single.get('mean_words','?')}w, topic={a_single.get('mean_topic','?')}. "
             f"Multi-layer wins={is_multi}. "
             f"[confidence: empirical, source: this-project-sweep]")
        cprr("evidence", "6",
             f"All configs: " +
             ", ".join(f"{k}={v['mean_words']}w/topic={v['mean_topic']}/ppl={v['mean_ppl']}"
                       for k, v in agg.items() if k != "baseline") +
             f" [confidence: empirical, source: this-project-sweep]")
        cprr("next", "6")


if __name__ == "__main__":
    main()
