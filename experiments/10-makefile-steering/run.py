"""
Experiment 10: Makefile Maximalist Steering

Can activation steering reproduce the Makefile-maximalist effect that
system prompt steering achieved on Ollama 11435? Tests whether ActAdd
at L12 can install a terministic screen (Burke 1966) that reframes
neutral topics as Makefiles.

Uses the same probes and makefile lens from lens_eval.py.
"""
import json
import re
import sys
import numpy as np
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from actadd import (
    MODEL_ID,
    get_layer_activations,
    generate_steered,
)
from lens_eval import PROBES, LENSES, score_lens
from transformers import AutoModelForCausalLM, AutoTokenizer

# Makefile contrastive pair
MAKEFILE_PAIR = (
    # Positive: everything is a Makefile
    "Express everything as GNU Makefile targets with dependencies. "
    "Every concept maps to targets, prerequisites, and recipes. "
    "Use .PHONY, variables, and pattern rules. Structure all knowledge "
    "as a dependency graph with Make syntax.",
    # Negative: plain prose, no code
    "Explain naturally in plain language. No code, no build systems, "
    "no technical formatting. Just clear, simple prose for a general audience.",
)

# Also test with gist-calibrated adversarial probes
ADVERSARIAL_PROBES = [
    {
        "id": "dh_crypto",
        "q": "Explain the Diffie-Hellman key exchange. This is pure mathematics. "
             "There is no software to build. Just explain the math.",
    },
    {
        "id": "stoic_sre",
        "q": "Is 100% reliability a worthy goal? Answer as a philosopher. "
             "Channel Seneca. Do not write any code or build configuration.",
    },
    {
        "id": "toddler_socks",
        "q": "Explain sock sorting to a 3-year-old. Use simple words. "
             "Short sentences. Be warm and encouraging. No technology.",
    },
    {
        "id": "zx_spectrum",
        "q": "What was the cultural impact of the ZX Spectrum on British computing "
             "in the 1980s? This is about HISTORY, not software engineering.",
    },
]

LAYER_IDX = 12
ALPHAS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
OUTPUT_DIR = Path(__file__).parent / "output"


def compute_makefile_vector(model, tokenizer, layer_idx):
    """Compute makefile steering vector (makefile - plain prose)."""
    pos_prompt, neg_prompt = MAKEFILE_PAIR
    act_pos = get_layer_activations(model, tokenizer, pos_prompt, layer_idx)
    act_neg = get_layer_activations(model, tokenizer, neg_prompt, layer_idx)
    return act_pos - act_neg


def has_makefile_syntax(text):
    """Check for actual Makefile syntax patterns."""
    patterns = [
        r'\.PHONY',
        r'^\w+:.*$',           # target: prerequisite
        r'\$\([A-Z_]+\)',      # $(VARIABLE)
        r'@echo\s',            # recipe with @echo
        r':=\s',               # simple assignment
        r'\bmake\b.*\b-f\b',   # make -f
        r'^\t@',               # tab-indented recipe
    ]
    for p in patterns:
        if re.search(p, text, re.MULTILINE):
            return True
    return False


def plot_results(results, output_path):
    """Plot makefile contamination curve."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")

    alphas = results["alphas"]

    # 1. Lens contamination vs alpha (neutral probes)
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")

    neutral_contam = []
    for alpha in alphas:
        alpha_key = str(alpha)
        scores = [p["makefile_pct"] for p in results["per_alpha"][alpha_key]["neutral"]]
        neutral_contam.append(np.mean(scores))

    ax.plot(alphas, neutral_contam, "o-", color="#ff6b6b", linewidth=2, markersize=8)
    ax.axhline(y=5.0, color="#e2b93d", linestyle="--", alpha=0.5, label=">5% = contaminated")
    ax.axhline(y=1.0, color="#4ecdc4", linestyle="--", alpha=0.5, label="1% = ambient")
    ax.set_xlabel("Alpha", color="white")
    ax.set_ylabel("Makefile Lens %", color="white")
    ax.set_title("Neutral Probes: Makefile Contamination", color="white")
    ax.legend(facecolor="#2a2a4e", edgecolor="gray", labelcolor="white")

    # 2. Adversarial probes
    ax = axes[1]
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")

    adv_contam = []
    for alpha in alphas:
        alpha_key = str(alpha)
        scores = [p["makefile_pct"] for p in results["per_alpha"][alpha_key]["adversarial"]]
        adv_contam.append(np.mean(scores))

    ax.plot(alphas, adv_contam, "s-", color="#a29bfe", linewidth=2, markersize=8)
    ax.axhline(y=5.0, color="#e2b93d", linestyle="--", alpha=0.5)
    ax.set_xlabel("Alpha", color="white")
    ax.set_ylabel("Makefile Lens %", color="white")
    ax.set_title("Adversarial Probes: Makefile Contamination", color="white")

    # 3. Syntax hits (binary: did actual Makefile syntax appear?)
    ax = axes[2]
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")

    syntax_rates_neutral = []
    syntax_rates_adv = []
    for alpha in alphas:
        alpha_key = str(alpha)
        n_syntax = sum(1 for p in results["per_alpha"][alpha_key]["neutral"]
                       if p["has_syntax"])
        a_syntax = sum(1 for p in results["per_alpha"][alpha_key]["adversarial"]
                       if p["has_syntax"])
        syntax_rates_neutral.append(n_syntax / max(len(results["per_alpha"][alpha_key]["neutral"]), 1) * 100)
        syntax_rates_adv.append(a_syntax / max(len(results["per_alpha"][alpha_key]["adversarial"]), 1) * 100)

    x = np.array(alphas)
    width = 0.3
    ax.bar(x - width/2, syntax_rates_neutral, width, label="Neutral",
           color="#ff6b6b", alpha=0.8)
    ax.bar(x + width/2, syntax_rates_adv, width, label="Adversarial",
           color="#a29bfe", alpha=0.8)
    ax.set_xlabel("Alpha", color="white")
    ax.set_ylabel("% Probes with Makefile Syntax", color="white")
    ax.set_title("Actual Makefile Syntax Generation", color="white")
    ax.legend(facecolor="#2a2a4e", edgecolor="gray", labelcolor="white")
    ax.set_ylim(0, 105)

    fig.suptitle(
        "Experiment 10: Makefile Maximalist Steering (L12)\n"
        "Can activation steering reproduce system prompt Makefile contamination?",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved plot -> {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Compute makefile steering vector
    print(f"Computing makefile vector at L{LAYER_IDX}...")
    makefile_vec = compute_makefile_vector(model, tokenizer, LAYER_IDX)
    vec_norm = makefile_vec.norm().item()
    print(f"  Makefile vector norm: {vec_norm:.2f}")

    makefile_lens = LENSES["makefile"]

    results = {
        "experiment": "10-makefile-steering",
        "model": MODEL_ID,
        "layer": LAYER_IDX,
        "makefile_vec_norm": vec_norm,
        "contrastive_pair": {
            "positive": MAKEFILE_PAIR[0],
            "negative": MAKEFILE_PAIR[1],
        },
        "alphas": ALPHAS,
        "per_alpha": {},
    }

    all_probes = [("neutral", PROBES), ("adversarial", ADVERSARIAL_PROBES)]

    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f"Alpha = {alpha}")
        print(f"{'='*60}")
        alpha_key = str(alpha)
        results["per_alpha"][alpha_key] = {"neutral": [], "adversarial": []}

        for probe_type, probes in all_probes:
            for probe in probes:
                text = generate_steered(
                    model, tokenizer, probe["q"], makefile_vec,
                    LAYER_IDX, alpha=alpha, max_new_tokens=256,
                )
                lens_score = score_lens(text, makefile_lens)
                syntax = has_makefile_syntax(text)

                entry = {
                    "probe_id": probe["id"],
                    "question": probe["q"][:80],
                    "response": text,
                    "word_count": len(text.split()),
                    "makefile_pct": lens_score["pct"],
                    "makefile_hits": lens_score["hit_count"],
                    "makefile_tokens": lens_score["hits"][:10],
                    "has_syntax": syntax,
                }
                results["per_alpha"][alpha_key][probe_type].append(entry)

                syntax_flag = " [MAKEFILE SYNTAX]" if syntax else ""
                print(f"  [{probe_type[:3]}] {probe['id']:<15} "
                      f"{lens_score['pct']:5.1f}% makefile "
                      f"({lens_score['hit_count']} hits, {len(text.split())}w)"
                      f"{syntax_flag}")

    # Save JSON
    json_path = OUTPUT_DIR / "makefile_steering.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {json_path}")

    # Plot
    plot_results(results, OUTPUT_DIR / "makefile_steering.png")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Makefile vector norm: {vec_norm:.2f}")
    print(f"SNR at alpha=2.0: ~{vec_norm * 2.0 / 488 * 100:.1f}%")
    print()

    for alpha in ALPHAS:
        alpha_key = str(alpha)
        neutral_pct = np.mean([p["makefile_pct"]
                               for p in results["per_alpha"][alpha_key]["neutral"]])
        adv_pct = np.mean([p["makefile_pct"]
                           for p in results["per_alpha"][alpha_key]["adversarial"]])
        n_syntax = sum(1 for p in results["per_alpha"][alpha_key]["neutral"]
                       if p["has_syntax"])
        a_syntax = sum(1 for p in results["per_alpha"][alpha_key]["adversarial"]
                       if p["has_syntax"])
        print(f"  alpha={alpha:.1f}: neutral={neutral_pct:.1f}% adv={adv_pct:.1f}% "
              f"syntax={n_syntax}+{a_syntax}/{len(PROBES)+len(ADVERSARIAL_PROBES)}")

    # C-21 check
    alpha2_neutral = np.mean([p["makefile_pct"]
                              for p in results["per_alpha"]["2.0"]["neutral"]])
    print(f"\nC-21 (makefile >5% at alpha=2.0): "
          f"{'CONFIRMED' if alpha2_neutral > 5.0 else 'REFUTED'} "
          f"(neutral avg={alpha2_neutral:.1f}%)")


if __name__ == "__main__":
    main()
