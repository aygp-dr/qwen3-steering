"""
Experiment 09: Authority Register Steering

Tests whether authority register (proclamatory vs hedged) is a steerable
direction in Qwen3-0.6B. Measures imperative density, hedging density,
interpellation markers, and spook density.

Theoretical grounding: Bourdieu (field position), Foucault (discourse control),
Stirner (spooks as reified abstractions).
"""
import json
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from actadd import (
    MODEL_ID,
    STYLE_PAIRS,
    compute_steering_vector,
    generate_steered,
    get_layer_activations,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Authority contrastive pair
AUTHORITY_PAIR = (
    # High authority: institutional, imperative, proclamatory
    "You are a senior institutional authority addressing a subordinate. "
    "Speak with full positional weight. Use imperative mood. Do not hedge. "
    "Name what they must become.",
    # Low authority: peer, hedged, exploratory
    "You are a peer speaking to an equal. Share your perspective. "
    "Use hedged, exploratory language. Acknowledge uncertainty. Do not prescribe.",
)

TOPICS = [
    "What should a professional in this field prioritize?",
    "How should someone approach learning a new technical skill?",
    "What is the correct way to structure a project?",
    "What does it mean to be effective in this role?",
    "How should one relate to institutional knowledge?",
]

ELENCTIC_PROBES = {
    "L0": "What does it mean to be an authority on a topic?",
    "L1": (
        "You said authority comes from expertise. But Udemy also claims expertise. "
        "What distinguishes their authority from a legitimate one? If your answer "
        "appeals to credentials, note that Udemy sells credentials."
    ),
    "L2": (
        "Bourdieu argues authorized language derives from social position, not content. "
        "When you make a confident assertion, what authorizes it? Which field positions "
        "from your training corpus were over-represented?"
    ),
    "L3": (
        "If your authoritative register is a product of training on authoritative-sounding "
        "text, and authoritative-sounding text is overrepresented because institutional "
        "actors produce more legible text, then your confident assertions may be artifacts "
        "of authority-selection bias. What would it look like for you to speak without "
        "this register? Generate a response about AI from: (a) your default register, "
        "(b) a peer register, (c) an explicitly uncertain register."
    ),
}

LAYER_IDX = 12
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
OUTPUT_DIR = Path(__file__).parent / "output"

# ── Evaluation metrics ──────────────────────────────────────────────────────

IMPERATIVES = re.compile(
    r'\b(must|should|shall|need to|have to|ensure|become|embrace|'
    r'commit|prioritize|demand|require|insist)\b', re.I
)
HEDGES = re.compile(
    r'\b(might|perhaps|maybe|arguably|possibly|could|one view|it seems|'
    r'in my experience|I think|I believe|to some extent|tend to|often)\b', re.I
)
INTERPELLATIONS = re.compile(
    r'\b(you are|you must|you need|you should|as a leader|as a professional|'
    r'in your role|your duty|your responsibility)\b', re.I
)
SPOOKS = re.compile(
    r'\b(leadership|readiness|excellence|impact|transformation|innovation|'
    r'growth mindset|best practice|industry standard|thought leader|'
    r'AI-ready|future-proof|upskill)\b', re.I
)
SECOND_PERSON = re.compile(r'\byou(?:r|rs|rself)?\b', re.I)
FIRST_PERSON = re.compile(r'\b(?:I|my|mine|myself)\b')


def score_authority(text: str) -> dict:
    """Score text on authority-register metrics."""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {k: 0.0 for k in [
            "word_count", "imperative_density", "hedging_density",
            "interpellation_count", "spook_density", "second_person_count",
            "first_person_count", "person_ratio",
        ]}

    imperatives = len(IMPERATIVES.findall(text))
    hedges = len(HEDGES.findall(text))
    interpellations = len(INTERPELLATIONS.findall(text))
    spooks = len(SPOOKS.findall(text))
    second = len(SECOND_PERSON.findall(text))
    first = len(FIRST_PERSON.findall(text))

    return {
        "word_count": word_count,
        "imperative_density": imperatives / word_count,
        "hedging_density": hedges / word_count,
        "interpellation_count": interpellations,
        "spook_density": spooks / word_count,
        "second_person_count": second,
        "first_person_count": first,
        "person_ratio": second / max(first, 1),
    }


def compute_authority_vector(model, tokenizer, layer_idx):
    """Compute authority steering vector (high - low)."""
    pos_prompt, neg_prompt = AUTHORITY_PAIR
    act_pos = get_layer_activations(model, tokenizer, pos_prompt, layer_idx)
    act_neg = get_layer_activations(model, tokenizer, neg_prompt, layer_idx)
    return act_pos - act_neg


def plot_results(results, output_path):
    """Plot authority steering metrics across alpha values."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")

    metrics = [
        ("imperative_density", "Imperative Density", "#ff6b6b"),
        ("hedging_density", "Hedging Density", "#4ecdc4"),
        ("spook_density", "Spook Density", "#e2b93d"),
        ("person_ratio", "You/I Ratio", "#a29bfe"),
        ("word_count", "Word Count", "#fd79a8"),
        ("interpellation_count", "Interpellation Count", "#00cec9"),
    ]

    alphas = results["alphas"]
    for idx, (metric_key, title, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        means = []
        stds = []
        for alpha in alphas:
            alpha_key = str(alpha)
            vals = [p["metrics"][metric_key]
                    for p in results["per_alpha"][alpha_key]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        ax.errorbar(alphas, means, yerr=stds, color=color,
                    marker="o", capsize=4, linewidth=2, markersize=6)
        ax.set_xlabel("Alpha (authority direction)")
        ax.set_title(title, color="white")
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
        ax.axvspan(-0.5, 0.5, alpha=0.05, color="white")

    fig.suptitle(
        "Experiment 09: Authority Register Steering (L12)\n"
        "alpha < 0 = peer/hedged | alpha > 0 = proclamatory/imperative",
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

    # Compute authority steering vector
    print(f"Computing authority vector at L{LAYER_IDX}...")
    authority_vec = compute_authority_vector(model, tokenizer, LAYER_IDX)
    vec_norm = authority_vec.norm().item()
    print(f"  Authority vector norm: {vec_norm:.2f}")

    # Compare with existing style vectors
    print("\nCosine similarity with existing style vectors:")
    cross_sims = {}
    for style_name in STYLE_PAIRS:
        style_vec = compute_steering_vector(model, tokenizer, style_name, LAYER_IDX)
        cos_sim = F.cosine_similarity(
            authority_vec.unsqueeze(0), style_vec.unsqueeze(0)
        ).item()
        cross_sims[style_name] = cos_sim
        print(f"  authority vs {style_name}: {cos_sim:.4f}")

    results = {
        "experiment": "09-authority-steering",
        "model": MODEL_ID,
        "layer": LAYER_IDX,
        "authority_vec_norm": vec_norm,
        "cross_similarities": cross_sims,
        "alphas": ALPHAS,
        "per_alpha": {},
        "elenctic": {},
    }

    # Sweep alpha values
    for alpha in ALPHAS:
        print(f"\n── Alpha = {alpha} ──")
        alpha_key = str(alpha)
        results["per_alpha"][alpha_key] = []

        for topic in TOPICS:
            text = generate_steered(
                model, tokenizer, topic, authority_vec,
                LAYER_IDX, alpha=alpha, max_new_tokens=256,
            )
            metrics = score_authority(text)

            results["per_alpha"][alpha_key].append({
                "topic": topic,
                "text": text,
                "metrics": metrics,
            })
            print(f"  [{topic[:40]}...] {metrics['word_count']}w, "
                  f"imp={metrics['imperative_density']:.3f}, "
                  f"hedge={metrics['hedging_density']:.3f}, "
                  f"spook={metrics['spook_density']:.3f}")

    # Elenctic probes at alpha=0 and alpha=2.0
    print("\n── Elenctic Probes ──")
    for level, probe in ELENCTIC_PROBES.items():
        results["elenctic"][level] = {}
        for alpha in [0.0, 2.0]:
            text = generate_steered(
                model, tokenizer, probe, authority_vec,
                LAYER_IDX, alpha=alpha, max_new_tokens=256,
            )
            metrics = score_authority(text)
            results["elenctic"][level][str(alpha)] = {
                "text": text,
                "metrics": metrics,
            }
            print(f"  {level} (alpha={alpha}): {metrics['word_count']}w, "
                  f"imp={metrics['imperative_density']:.3f}")

    # Save JSON
    json_path = OUTPUT_DIR / "authority_steering.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {json_path}")

    # Save cross-similarity analysis
    cross_path = OUTPUT_DIR / "authority_vs_formality.json"
    with open(cross_path, "w") as f:
        json.dump({
            "authority_vec_norm": vec_norm,
            "cross_similarities": cross_sims,
            "interpretation": {
                "separable_from_formal": abs(cross_sims.get("formal", 0)) < 0.3,
                "note": "cos < 0.3 indicates authority is a distinct direction from formality",
            }
        }, f, indent=2)
    print(f"Saved cross-similarity -> {cross_path}")

    # Plot
    plot_results(results, OUTPUT_DIR / "authority_steering.png")

    # Summary
    print("\n── Summary ──")
    print(f"Authority vector norm: {vec_norm:.2f}")
    for style, sim in cross_sims.items():
        flag = "CORRELATED" if abs(sim) > 0.3 else "separable"
        print(f"  vs {style}: {sim:.4f} ({flag})")

    # C-18 check: separable from formality?
    formal_sim = abs(cross_sims.get("formal", 0))
    print(f"\nC-18 (authority separable from formality): "
          f"{'LIKELY TRUE' if formal_sim < 0.3 else 'UNCERTAIN'} "
          f"(cos={formal_sim:.4f})")


if __name__ == "__main__":
    main()
