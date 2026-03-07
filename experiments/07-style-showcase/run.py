#!/usr/bin/env python3
"""
Experiment 07: Four-Style Showcase.

Run all four style axes (terse, formal, socratic, dry-wit) at L12 α=2.0
across 5 prompts. Capture metrics for qualitative comparison.

Usage:
    python experiments/07-style-showcase/run.py
"""
import argparse
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from actadd import (
    MODEL_ID,
    STYLE_PAIRS,
    compute_steering_vector,
    generate_steered,
)

PROMPTS = [
    "Explain what a mutex is.",
    "What is the difference between a stack and a heap?",
    "Why do programming languages have types?",
    "How does garbage collection work?",
    "What is the CAP theorem?",
]

STYLES = ["terse", "formal", "socratic", "dry-wit"]
LAYER = 12
ALPHA = 2.0

OUTPUT_DIR = str(Path(__file__).resolve().parent / "output")


def count_words(text):
    return len(text.split())


def words_per_sentence(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    return count_words(text) / len(sentences)


def markdown_density(text):
    markers = len(re.findall(r'[*#`\-|>]', text))
    total = len(text) or 1
    return markers / total * 100


def question_ratio(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    questions = sum(1 for s in sentences if '?' in s or s.endswith('?'))
    return questions / len(sentences)


def compute_tfidf_cosine(text_a, text_b):
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / (len(words_a) ** 0.5 * len(words_b) ** 0.5)


def compute_perplexity_ratio(model, tokenizer, baseline_text, steered_text):
    """Compute cross-entropy ratio steered/baseline."""
    def get_loss(text):
        messages = [{"role": "assistant", "content": text}]
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        inputs = tokenizer(encoded, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
        return outputs.loss.item()

    try:
        baseline_loss = get_loss(baseline_text)
        steered_loss = get_loss(steered_text)
        if baseline_loss == 0:
            return 1.0
        return steered_loss / baseline_loss
    except Exception:
        return 1.0


def metrics_for(text, baseline_text, model, tokenizer):
    return {
        "word_count": count_words(text),
        "words_per_sentence": round(words_per_sentence(text), 1),
        "topic_cosine": round(compute_tfidf_cosine(baseline_text, text), 3),
        "perplexity_ratio": round(compute_perplexity_ratio(model, tokenizer, baseline_text, text), 3),
        "markdown_density": round(markdown_density(text), 2),
        "question_ratio": round(question_ratio(text), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Compute steering vectors for all styles
    print(f"Computing steering vectors at L{LAYER}...")
    vectors = {}
    for style in STYLES:
        vectors[style] = compute_steering_vector(model, tokenizer, style, LAYER)
        print(f"  {style}: ||v||={vectors[style].norm().item():.1f}")

    results = []

    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n[{prompt_idx + 1}/{len(PROMPTS)}] {prompt}")
        entry = {"prompt": prompt, "prompt_idx": prompt_idx}

        # Baseline
        baseline_text = generate_steered(
            model, tokenizer, prompt, vectors["terse"], LAYER, alpha=0.0
        )
        entry["baseline"] = {
            "text": baseline_text,
            "text_preview": baseline_text[:100],
            **metrics_for(baseline_text, baseline_text, model, tokenizer),
        }
        print(f"  baseline: {count_words(baseline_text)}w")

        # Each style
        for style in STYLES:
            steered_text = generate_steered(
                model, tokenizer, prompt, vectors[style], LAYER, alpha=ALPHA
            )
            m = metrics_for(steered_text, baseline_text, model, tokenizer)
            entry[style] = {
                "text": steered_text,
                "text_preview": steered_text[:100],
                **m,
            }
            print(f"  {style:10s}: {m['word_count']}w, topic={m['topic_cosine']}, "
                  f"ppl={m['perplexity_ratio']}, q_ratio={m['question_ratio']}")

        results.append(entry)

    # Aggregate
    aggregate = {"baseline": {}}
    for style in ["baseline"] + STYLES:
        key = style
        agg = {
            "mean_words": round(sum(r[key]["word_count"] for r in results) / len(results), 1),
            "mean_topic": round(sum(r[key]["topic_cosine"] for r in results) / len(results), 3),
            "mean_ppl": round(sum(r[key]["perplexity_ratio"] for r in results) / len(results), 3),
            "mean_wps": round(sum(r[key]["words_per_sentence"] for r in results) / len(results), 1),
            "mean_md_density": round(sum(r[key]["markdown_density"] for r in results) / len(results), 2),
            "mean_q_ratio": round(sum(r[key]["question_ratio"] for r in results) / len(results), 2),
        }
        aggregate[key] = agg

    output = {
        "model": MODEL_ID,
        "layer": LAYER,
        "alpha": ALPHA,
        "styles": STYLES,
        "prompts": PROMPTS,
        "results": results,
        "aggregate": aggregate,
        "timestamp": datetime.now().astimezone().isoformat(),
    }

    # Strip full text from JSON output (keep previews)
    for r in output["results"]:
        for key in ["baseline"] + STYLES:
            if "text" in r[key]:
                del r[key]["text"]

    json_path = os.path.join(output_dir, "style_showcase.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ── Visualization ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "viz"))
        from shared_style import apply_dark_style, COLORS

        apply_dark_style()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Four-Style Showcase — L12 α=2.0",
                     fontsize=14, fontweight="bold", color=COLORS["text"])

        style_colors = {
            "baseline": COLORS["baseline"],
            "terse": COLORS["terse"],
            "formal": COLORS["formal"],
            "socratic": COLORS["socratic"],
            "dry-wit": COLORS["dry-wit"],
        }
        all_keys = ["baseline"] + STYLES
        x = np.arange(len(all_keys))

        # Panel 1: Word count
        ax = axes[0]
        vals = [aggregate[k]["mean_words"] for k in all_keys]
        bars = ax.bar(x, vals, color=[style_colors[k] for k in all_keys], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_keys, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean Words")
        ax.set_title("Word Count")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=7, color=COLORS["text"])

        # Panel 2: Topic cosine
        ax = axes[1]
        vals = [aggregate[k]["mean_topic"] for k in all_keys]
        bars = ax.bar(x, vals, color=[style_colors[k] for k in all_keys], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_keys, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean Topic Cosine")
        ax.set_title("Semantic Preservation")
        ax.axhline(y=0.6, color=COLORS["collapse"], linestyle="--", alpha=0.5, label="threshold")
        ax.legend(fontsize=7)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=COLORS["text"])

        # Panel 3: Question ratio (socratic should dominate)
        ax = axes[2]
        vals = [aggregate[k]["mean_q_ratio"] for k in all_keys]
        bars = ax.bar(x, vals, color=[style_colors[k] for k in all_keys], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_keys, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Question Ratio")
        ax.set_title("Socratic Signal")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=7, color=COLORS["text"])

        plt.tight_layout()
        png_path = os.path.join(output_dir, "style_showcase.png")
        plt.savefig(png_path)
        print(f"Saved: {png_path}")
        plt.close()

    except ImportError as exc:
        print(f"Skipping visualization: {exc}")

    # Print summary table
    print("\n── Aggregate ──")
    print(f"{'Style':12s} {'Words':>6s} {'Topic':>6s} {'PPL':>6s} {'WPS':>6s} {'MD%':>6s} {'Q%':>6s}")
    for key in all_keys:
        a = aggregate[key]
        print(f"{key:12s} {a['mean_words']:6.1f} {a['mean_topic']:6.3f} "
              f"{a['mean_ppl']:6.3f} {a['mean_wps']:6.1f} {a['mean_md_density']:6.2f} "
              f"{a['mean_q_ratio']:6.2f}")


if __name__ == "__main__":
    main()
