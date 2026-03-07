#!/usr/bin/env python3
"""
Generate a qualitative example table figure: baseline vs terse-steered outputs.

SPLASH reviewer priority 1 — "Table 1" in every steering paper.
Reads eval_output/terse_verbose_full.json, cherry-picks best examples,
renders as a matplotlib figure with wrapped text.

Usage:
    uv run python fig_qualitative_table.py
    uv run python fig_qualitative_table.py --top 6
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#c9d1d9"
GRID_CLR = "#30363d"
BLUE = "#58a6ff"
GRAY = "#8b949e"
ORANGE = "#f0883e"
OUTPUT_DIR = Path("eval_output")


def load_and_rank(path, top_n=5):
    with open(path) as f:
        data = json.load(f)
    records = data["records"]
    config = data["config"]
    scored = []
    for r in records:
        gap = r["baseline_words"] - r["terse_words"]
        scored.append((gap, r))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:top_n]], config


def wrap(text, width=60):
    return "\n".join(textwrap.wrap(text, width=width))


def render_table(records, config, output_dir):
    n = len(records)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n))
    fig.patch.set_facecolor(DARK_BG)

    if n == 1:
        axes = [axes]

    for i, (ax, r) in enumerate(zip(axes, records)):
        ax.set_facecolor(PANEL_BG)
        ax.axis("off")

        prompt = r["prompt"]
        baseline = r["baseline_text"][:250]
        terse = r["terse_text"][:250]

        bw = r["baseline_words"]
        tw = r["terse_words"]
        reduction = round((1 - tw / bw) * 100) if bw > 0 else 0

        text = (
            f"Prompt: {prompt}\n\n"
            f"Baseline ({bw} words):\n"
            f"{wrap(baseline, 80)}\n\n"
            f"Terse, alpha=2.0 ({tw} words, -{reduction}%):\n"
            f"{wrap(terse, 80)}"
        )

        ax.text(0.02, 0.95, text, transform=ax.transAxes,
                fontsize=8, fontfamily="monospace",
                verticalalignment="top", color=TEXT_CLR,
                linespacing=1.3)

        # Colored sidebar
        rect = mpatches.FancyBboxPatch(
            (-0.01, 0), 0.008, 1, transform=ax.transAxes,
            boxstyle="round,pad=0",
            facecolor=BLUE if reduction > 50 else ORANGE,
            edgecolor="none", clip_on=False)
        ax.add_patch(rect)

        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)

    fig.suptitle(
        f"Qualitative Examples: Baseline vs Terse-Steered (alpha={config.get('alpha', 2.0)}, layer={config.get('layer', 15)})",
        color=TEXT_CLR, fontsize=12, fontweight="bold", y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    path = output_dir / "qualitative_table.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def render_org_table(records, config, output_dir):
    """Also emit an org-mode table for the README."""
    lines = [
        f"#+CAPTION: Baseline vs terse-steered outputs (alpha={config.get('alpha', 2.0)}, layer={config.get('layer', 15)})",
        "| Prompt | Baseline | Words | Terse (alpha=2.0) | Words | Reduction |",
        "|--------+----------+-------+--------------------+-------+-----------|",
    ]
    for r in records:
        prompt = r["prompt"]
        baseline = r["baseline_text"][:80].replace("|", "/")
        terse = r["terse_text"][:80].replace("|", "/")
        bw = r["baseline_words"]
        tw = r["terse_words"]
        red = f"-{round((1-tw/bw)*100)}%" if bw > 0 else "n/a"
        lines.append(f"| {prompt} | {baseline}... | {bw} | {terse}... | {tw} | {red} |")

    path = output_dir / "qualitative_table.org"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--data", type=str, default="eval_output/terse_verbose_full.json")
    args = parser.parse_args()

    records, config = load_and_rank(args.data, args.top)
    OUTPUT_DIR.mkdir(exist_ok=True)
    render_table(records, config, OUTPUT_DIR)
    render_org_table(records, config, OUTPUT_DIR)


if __name__ == "__main__":
    main()
