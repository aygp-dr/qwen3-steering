#!/usr/bin/env python3
"""
Style vector cosine similarity matrix (4x4 heatmap).

SPLASH reviewer priority 2. Tests H-SV-1: style vectors are not orthogonal.
Computes steering vectors for all 4 style axes at a given layer, then
displays pairwise cosine similarities as a heatmap.

Usage:
    uv run python fig_style_cosine_matrix.py
    uv run python fig_style_cosine_matrix.py --layer 12
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

from actadd import compute_steering_vector, STYLE_PAIRS

MODEL_ID = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = Path("eval_output")

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#c9d1d9"
GRID_CLR = "#30363d"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    args = parser.parse_args()

    styles = list(STYLE_PAIRS.keys())
    n = len(styles)

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"Computing {n} steering vectors at layer {args.layer}...")
    vectors = {}
    for style in styles:
        vec = compute_steering_vector(model, tokenizer, style, layer_idx=args.layer)
        vectors[style] = vec
        print(f"  {style}: norm={vec.norm().item():.2f}")

    # Cosine similarity matrix
    cos_sim = np.zeros((n, n))
    for i, si in enumerate(styles):
        for j, sj in enumerate(styles):
            cos_sim[i, j] = torch.nn.functional.cosine_similarity(
                vectors[si].unsqueeze(0), vectors[sj].unsqueeze(0)
            ).item()

    print(f"\nCosine similarity matrix (layer {args.layer}):")
    print(f"{'':12s}", end="")
    for s in styles:
        print(f"{s:>10s}", end="")
    print()
    for i, si in enumerate(styles):
        print(f"{si:12s}", end="")
        for j in range(n):
            print(f"{cos_sim[i,j]:10.3f}", end="")
        print()

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    mask = np.zeros_like(cos_sim, dtype=bool)
    # Show full matrix (not just triangle) for clarity

    sns.heatmap(
        cos_sim, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1,
        xticklabels=styles, yticklabels=styles,
        ax=ax, linewidths=1, linecolor=GRID_CLR,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )

    ax.set_title(
        f"Style Vector Cosine Similarity (layer {args.layer}, d_model=1024)\n"
        f"H-SV-1: Are style vectors orthogonal?",
        color=TEXT_CLR, fontsize=11, pad=12)
    plt.setp(ax.get_xticklabels(), color=TEXT_CLR, fontsize=10)
    plt.setp(ax.get_yticklabels(), color=TEXT_CLR, fontsize=10, rotation=0)
    ax.tick_params(colors=TEXT_CLR)

    # Annotate interpretation
    off_diag = cos_sim[np.triu_indices(n, k=1)]
    max_pair_idx = np.argmax(np.abs(off_diag))
    max_val = off_diag[max_pair_idx]
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pi, pj = pairs[max_pair_idx]

    fig.text(0.5, 0.01,
             f"Max off-diagonal: {styles[pi]}/{styles[pj]} = {max_val:.3f} | "
             f"Mean |cos_sim| = {np.mean(np.abs(off_diag)):.3f} | "
             f"{'NOT orthogonal' if np.mean(np.abs(off_diag)) > 0.1 else 'Approximately orthogonal'}",
             ha="center", color="#8b949e", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "style_cosine_matrix.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")

    # Also save alt-text
    txt_path = OUTPUT_DIR / "style_cosine_matrix.txt"
    with open(txt_path, "w") as f:
        f.write(
            f"4x4 cosine similarity heatmap of style steering vectors (terse, formal, socratic, dry-wit) "
            f"at layer {args.layer} of Qwen3-0.6B. "
            f"Tests H-SV-1: whether style vectors are orthogonal at d_model=1024. "
            f"Max off-diagonal pair: {styles[pi]}/{styles[pj]} ({max_val:.3f}). "
            f"Mean absolute off-diagonal cosine similarity: {np.mean(np.abs(off_diag)):.3f}.\n"
        )
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
