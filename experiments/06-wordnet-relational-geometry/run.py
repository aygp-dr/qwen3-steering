"""
WordNet Relational Geometry: are semantic relationships stable directions?

At each layer, does the residual stream encode WordNet relationships as
stable vectors? Specifically:
  - Is the hypernym direction (dog -> animal) stable across word pairs?
  - Is the antonym direction (hot -> cold) a different stable direction?
  - Are meronyms (wheel -> car) a third?
  - Are these directions orthogonal at the best layer?

Ground truth: WordNet's 117K synsets with defined hypernym, antonym,
and meronym relationships. No prompt engineering, no topic drift,
verifiable against an external graph.

Layer predictions:
  L0-7:   token identity, no relational structure
  L8-11:  word sense disambiguation, hypernyms start
  L12-16: relational encoding, directions stable
  L18+:   output preparation, relations used but noisy

The key test: are hypernym and antonym orthogonal at L14? If yes,
the model has structured relational geometry. If entangled, it's
pattern-matching not structure.

Usage:
    python experiments/06-wordnet-relational-geometry/run.py
    python experiments/06-wordnet-relational-geometry/run.py --cprr
    python experiments/06-wordnet-relational-geometry/run.py --relations hypernym antonym
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from actadd import MODEL_ID

try:
    from nltk.corpus import wordnet as wn
    HAS_WORDNET = True
except ImportError:
    HAS_WORDNET = False

NUM_LAYERS = 28
D_MODEL = 1024

# Curated pairs — common, unambiguous, single-token-friendly words.
# WordNet gives us thousands but we want pairs where both words are
# likely single tokens in Qwen3's 151K vocab for clean activation extraction.

HYPERNYM_PAIRS = [
    ("dog", "animal"),
    ("cat", "animal"),
    ("oak", "tree"),
    ("rose", "flower"),
    ("car", "vehicle"),
    ("hammer", "tool"),
    ("shirt", "clothing"),
    ("apple", "fruit"),
    ("eagle", "bird"),
    ("salmon", "fish"),
    ("violin", "instrument"),
    ("iron", "metal"),
    ("anger", "emotion"),
    ("walk", "move"),
    ("whisper", "speak"),
]

ANTONYM_PAIRS = [
    ("hot", "cold"),
    ("big", "small"),
    ("fast", "slow"),
    ("light", "dark"),
    ("good", "bad"),
    ("happy", "sad"),
    ("old", "young"),
    ("hard", "soft"),
    ("wet", "dry"),
    ("rich", "poor"),
    ("strong", "weak"),
    ("long", "short"),
    ("loud", "quiet"),
    ("open", "closed"),
    ("alive", "dead"),
]

MERONYM_PAIRS = [
    ("wheel", "car"),
    ("wing", "bird"),
    ("leaf", "tree"),
    ("finger", "hand"),
    ("engine", "car"),
    ("root", "tree"),
    ("petal", "flower"),
    ("leg", "table"),
    ("key", "keyboard"),
    ("blade", "knife"),
    ("handle", "door"),
    ("lens", "camera"),
    ("sail", "boat"),
    ("string", "guitar"),
    ("brick", "wall"),
]

RELATION_PAIRS = {
    "hypernym": HYPERNYM_PAIRS,
    "antonym": ANTONYM_PAIRS,
    "meronym": MERONYM_PAIRS,
}


def get_word_activation(model, tokenizer, word, layer_idx):
    """Get the residual stream activation for a single word at a given layer.

    Uses the prompt "The word is: {word}" to get activation at the last
    token position (the word itself or its last subtoken).
    """
    prompt = f"The word is: {word}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    activations = {}
    def hook_fn(module, input, output):
        # output is (hidden_states, ...) for Qwen3
        hidden = output[0] if isinstance(output, tuple) else output
        activations["h"] = hidden[0, -1, :].detach().clone()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    return activations["h"]


def get_all_layer_activations(model, tokenizer, word):
    """Get activations for a word at all layers in one forward pass."""
    prompt = f"The word is: {word}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    layer_activations = {}
    handles = []

    for layer_idx in range(NUM_LAYERS):
        def make_hook(idx):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                layer_activations[idx] = hidden[0, -1, :].detach().clone()
            return hook_fn
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    with torch.no_grad():
        model(**inputs)

    for handle in handles:
        handle.remove()

    return layer_activations


def compute_relation_vectors(model, tokenizer, pairs, relation_name):
    """Compute direction vectors for a relation across all layers.

    For each pair (a, b), the relation vector at layer L is:
        v_L = act(b, L) - act(a, L)

    For hypernyms: act(animal) - act(dog) = "generalization direction"
    For antonyms: act(cold) - act(hot) = "negation direction"
    For meronyms: act(car) - act(wheel) = "whole direction"
    """
    print(f"\n  Computing {relation_name} vectors ({len(pairs)} pairs)...")
    all_vectors = {layer: [] for layer in range(NUM_LAYERS)}

    for pair_idx, (word_a, word_b) in enumerate(pairs):
        acts_a = get_all_layer_activations(model, tokenizer, word_a)
        acts_b = get_all_layer_activations(model, tokenizer, word_b)

        for layer in range(NUM_LAYERS):
            delta = acts_b[layer] - acts_a[layer]
            all_vectors[layer].append(delta)

        if (pair_idx + 1) % 5 == 0:
            print(f"    {pair_idx + 1}/{len(pairs)} pairs processed")

    return all_vectors


def direction_stability(vectors_at_layer):
    """Cross-pair cosine similarity: how consistent is the direction?

    Returns mean pairwise cosine similarity across all pairs of
    relation vectors at a given layer.
    """
    if len(vectors_at_layer) < 2:
        return 0.0

    cosines = []
    for i in range(len(vectors_at_layer)):
        for j in range(i + 1, len(vectors_at_layer)):
            cos = F.cosine_similarity(
                vectors_at_layer[i].unsqueeze(0),
                vectors_at_layer[j].unsqueeze(0)
            ).item()
            cosines.append(cos)

    return sum(cosines) / len(cosines)


def mean_direction(vectors_at_layer):
    """Average direction vector, normalized."""
    stacked = torch.stack(vectors_at_layer)
    mean_vec = stacked.mean(dim=0)
    return F.normalize(mean_vec, dim=0)


def compute_inter_relation_angles(relation_mean_vectors):
    """Cosine similarity between mean direction vectors of different relations."""
    relations = list(relation_mean_vectors.keys())
    angles = {}
    for i in range(len(relations)):
        for j in range(i + 1, len(relations)):
            rel_a, rel_b = relations[i], relations[j]
            cos = F.cosine_similarity(
                relation_mean_vectors[rel_a].unsqueeze(0),
                relation_mean_vectors[rel_b].unsqueeze(0)
            ).item()
            angles[f"{rel_a}_vs_{rel_b}"] = round(cos, 4)
    return angles


def main():
    parser = argparse.ArgumentParser(description="WordNet Relational Geometry")
    parser.add_argument("--relations", nargs="+", default=["hypernym", "antonym", "meronym"],
                        choices=["hypernym", "antonym", "meronym"])
    parser.add_argument("--max-pairs", type=int, default=15,
                        help="Max pairs per relation (default: 15)")
    parser.add_argument("--cprr", action="store_true")
    args = parser.parse_args()

    output_dir = str(Path(__file__).resolve().parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # Phase 1: compute relation vectors at all layers
    print("\n=== Phase 1: Compute relation vectors ===")
    all_relation_vectors = {}
    for relation_name in args.relations:
        pairs = RELATION_PAIRS[relation_name][:args.max_pairs]
        all_relation_vectors[relation_name] = compute_relation_vectors(
            model, tokenizer, pairs, relation_name
        )

    # Phase 2: direction stability per relation per layer
    print("\n=== Phase 2: Direction stability ===")
    stability_results = {}
    norm_results = {}

    print(f"\n  {'Layer':>5} ", end="")
    for relation_name in args.relations:
        print(f"{'stab_' + relation_name[:4]:>12} {'norm_' + relation_name[:4]:>12}", end="")
    print()
    print("  " + "-" * (5 + 24 * len(args.relations)))

    for layer in range(NUM_LAYERS):
        stability_results[layer] = {}
        norm_results[layer] = {}

        print(f"  {layer:>5} ", end="")
        for relation_name in args.relations:
            vectors = all_relation_vectors[relation_name][layer]
            stab = direction_stability(vectors)
            norms = [v.norm().item() for v in vectors]
            mean_norm = sum(norms) / len(norms)

            stability_results[layer][relation_name] = round(stab, 4)
            norm_results[layer][relation_name] = round(mean_norm, 2)
            print(f"{stab:>12.4f} {mean_norm:>12.2f}", end="")
        print()

    # Phase 3: inter-relation orthogonality at each layer
    print("\n=== Phase 3: Inter-relation orthogonality ===")
    orthogonality_results = {}

    relation_pairs_labels = []
    for i, rel_a in enumerate(args.relations):
        for rel_b in args.relations[i + 1:]:
            relation_pairs_labels.append(f"{rel_a[:4]}_vs_{rel_b[:4]}")

    print(f"\n  {'Layer':>5} ", end="")
    for label in relation_pairs_labels:
        print(f"{label:>20}", end="")
    print()
    print("  " + "-" * (5 + 20 * len(relation_pairs_labels)))

    for layer in range(NUM_LAYERS):
        mean_vecs = {}
        for relation_name in args.relations:
            vectors = all_relation_vectors[relation_name][layer]
            mean_vecs[relation_name] = mean_direction(vectors)

        angles = compute_inter_relation_angles(mean_vecs)
        orthogonality_results[layer] = angles

        print(f"  {layer:>5} ", end="")
        for label in relation_pairs_labels:
            # Reconstruct key from label
            parts = label.split("_vs_")
            for key, val in angles.items():
                if parts[0] in key and parts[1] in key:
                    print(f"{val:>20.4f}", end="")
                    break
        print()

    # Phase 4: find best layers
    print("\n=== Phase 4: Summary ===")

    for relation_name in args.relations:
        stabilities = [(layer, stability_results[layer][relation_name])
                       for layer in range(NUM_LAYERS)]
        best_layer, best_stab = max(stabilities, key=lambda x: x[1])
        norms_at_best = norm_results[best_layer][relation_name]

        print(f"\n  {relation_name}:")
        print(f"    Best stability layer: L{best_layer} (cos={best_stab:.4f})")
        print(f"    Mean norm at best:    {norms_at_best:.2f}")

        # Layer region breakdown
        regions = [
            ("L0-7 (syntax)", range(0, 8)),
            ("L8-11 (early_sem)", range(8, 12)),
            ("L12-17 (deep_sem)", range(12, 18)),
            ("L18-22 (output_prep)", range(18, 23)),
            ("L23-27 (logit_proj)", range(23, 28)),
        ]
        for region_name, region_range in regions:
            region_stab = [stability_results[l][relation_name] for l in region_range]
            mean_stab = sum(region_stab) / len(region_stab)
            print(f"    {region_name:25s} mean_stability={mean_stab:.4f}")

    # Phase 5: orthogonality verdict
    if len(args.relations) >= 2:
        print("\n  Inter-relation orthogonality at deep_semantics (L12-17):")
        for layer in range(12, 18):
            angles = orthogonality_results[layer]
            angle_str = ", ".join(f"{k}={v:.3f}" for k, v in angles.items())
            print(f"    L{layer}: {angle_str}")

        # Average orthogonality in sweet spot
        for key in orthogonality_results[12]:
            sweet_vals = [orthogonality_results[l][key] for l in range(12, 18)]
            mean_val = sum(sweet_vals) / len(sweet_vals)
            orthogonal = abs(mean_val) < 0.3
            print(f"\n  {key}: mean cos={mean_val:.4f} in L12-17 "
                  f"({'ORTHOGONAL' if orthogonal else 'ENTANGLED'})")

    # Save results
    output = {
        "model": MODEL_ID,
        "relations": args.relations,
        "pairs_per_relation": {r: len(RELATION_PAIRS[r][:args.max_pairs]) for r in args.relations},
        "stability": {str(l): stability_results[l] for l in range(NUM_LAYERS)},
        "norms": {str(l): norm_results[l] for l in range(NUM_LAYERS)},
        "orthogonality": {str(l): orthogonality_results[l] for l in range(NUM_LAYERS)},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output_path = os.path.join(output_dir, "wordnet_relational_geometry.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Visualization
    print("\n=== Generating visualization ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("WordNet Relational Geometry Across Layers",
                 color="#e0e0e0", fontweight="bold", fontsize=13)

    colors = {"hypernym": "#4caf50", "antonym": "#f44336", "meronym": "#2196f3"}
    layers_arr = np.arange(NUM_LAYERS)

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="#e0e0e0")
        ax.spines["bottom"].set_color("#37374f")
        ax.spines["left"].set_color("#37374f")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.15, color="#37374f")

    # Plot 1: Direction stability
    for relation_name in args.relations:
        stab_values = [stability_results[l][relation_name] for l in range(NUM_LAYERS)]
        axes[0].plot(layers_arr, stab_values, "o-", color=colors[relation_name],
                     linewidth=2, markersize=3, label=relation_name)
    axes[0].axhline(y=0.5, color="#ff5722", linestyle="--", alpha=0.5, label="threshold (0.5)")
    axes[0].axvspan(12, 17, alpha=0.08, color="#4caf50")
    axes[0].set_xlabel("Layer", color="#e0e0e0")
    axes[0].set_ylabel("Direction Stability (cross-pair cos)", color="#e0e0e0")
    axes[0].set_title("Direction Stability by Relation", color="#e0e0e0")
    axes[0].legend(fontsize=8, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0")
    axes[0].set_ylim(-0.2, 1.05)

    # Plot 2: Mean delta norms
    for relation_name in args.relations:
        norm_values = [norm_results[l][relation_name] for l in range(NUM_LAYERS)]
        axes[1].plot(layers_arr, norm_values, "o-", color=colors[relation_name],
                     linewidth=2, markersize=3, label=relation_name)
    axes[1].axvspan(12, 17, alpha=0.08, color="#4caf50")
    axes[1].set_xlabel("Layer", color="#e0e0e0")
    axes[1].set_ylabel("Mean ||delta||", color="#e0e0e0")
    axes[1].set_title("Relation Vector Norms", color="#e0e0e0")
    axes[1].legend(fontsize=8, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0")

    # Plot 3: Inter-relation cosine (orthogonality)
    if len(args.relations) >= 2:
        for key in orthogonality_results[0]:
            values = [orthogonality_results[l][key] for l in range(NUM_LAYERS)]
            axes[2].plot(layers_arr, values, "o-", linewidth=2, markersize=3, label=key)
        axes[2].axhline(y=0, color="#607d8b", linestyle=":", alpha=0.5)
        axes[2].axhline(y=0.3, color="#ff5722", linestyle="--", alpha=0.3, label="entangled (>0.3)")
        axes[2].axhline(y=-0.3, color="#ff5722", linestyle="--", alpha=0.3)
        axes[2].axvspan(12, 17, alpha=0.08, color="#4caf50")
        axes[2].set_xlabel("Layer", color="#e0e0e0")
        axes[2].set_ylabel("Cosine between mean directions", color="#e0e0e0")
        axes[2].set_title("Inter-Relation Orthogonality", color="#e0e0e0")
        axes[2].legend(fontsize=7, facecolor="#2a2a4a", edgecolor="#37374f", labelcolor="#e0e0e0")
        axes[2].set_ylim(-1.05, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = os.path.join(output_dir, "wordnet_relational_geometry.png")
    fig.savefig(fig_path, dpi=200, facecolor="#1a1a2e")
    plt.close()
    print(f"  Saved: {fig_path}")

    # CPRR
    if args.cprr:
        print("\n=== Submitting to CPRR ===")
        def cprr(*cprr_args):
            result = subprocess.run(["cprr"] + list(cprr_args), capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  cprr {' '.join(cprr_args[:2])}: {result.stdout.strip()}")

        # Find peak stability layers
        for relation_name in args.relations:
            stabilities = [(l, stability_results[l][relation_name]) for l in range(NUM_LAYERS)]
            best_layer, best_stab = max(stabilities, key=lambda x: x[1])
            cprr("evidence", "11",
                 f"{relation_name}: peak stability at L{best_layer} (cos={best_stab:.4f}). "
                 f"Norm at peak: {norm_results[best_layer][relation_name]:.2f}. "
                 f"[confidence: empirical, source: this-project-sweep]")

        # Orthogonality evidence
        if len(args.relations) >= 2:
            for key in orthogonality_results[12]:
                sweet_vals = [orthogonality_results[l][key] for l in range(12, 18)]
                mean_val = sum(sweet_vals) / len(sweet_vals)
                cprr("evidence", "12",
                     f"{key}: mean cos={mean_val:.4f} in L12-17. "
                     f"{'Orthogonal' if abs(mean_val) < 0.3 else 'Entangled'}. "
                     f"[confidence: empirical, source: this-project-sweep]")

        cprr("next", "11")


if __name__ == "__main__":
    main()
