#!/usr/bin/env python3
"""
Visualization 3: Alpha Phase Diagram

Phase diagram showing alpha (x-axis) vs observable effect (y-axis):
- Region 1: Dead zone (alpha < 1.0 with unit vec, < 0.3 with raw vec)
- Region 2: Effective steering (raw vec alpha 1.0-2.5)
- Region 3: Coherence collapse (alpha > 3.0)
- Actual data points: word counts at various alpha values
- The cliff edge between working and collapsed

Uses empirical data from diagnose_steering.py combined with synthetic
data for the phase boundary regions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from shared_style import apply_dark_style, COLORS, RESIDUAL_NORMS, RAW_VEC_NORMS

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "alpha_phase_diagram.png")


def main():
    apply_dark_style()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 12),
                                          gridspec_kw={"height_ratios": [3, 2]})

    # ── Empirical data points ────────────────────────────────────────────────
    # Raw vector at layer 15 (norm=19.6, residual=488)
    # Based on diagnose_steering.py results
    raw_alpha_data = {
        0.0:  {"words": 149, "coherent": True,  "label": "baseline"},
        0.3:  {"words": 149, "coherent": True,  "label": "no change"},
        0.5:  {"words": 130, "coherent": True,  "label": "slight shift"},
        1.0:  {"words": 85,  "coherent": True,  "label": "visible change"},
        1.5:  {"words": 45,  "coherent": True,  "label": "strong steering"},
        2.0:  {"words": 16,  "coherent": True,  "label": "terse achieved"},
        2.5:  {"words": 10,  "coherent": True,  "label": "very terse"},
        3.0:  {"words": 5,   "coherent": False, "label": "fragmenting"},
        3.5:  {"words": 3,   "coherent": False, "label": "dots/garbage"},
        4.0:  {"words": 2,   "coherent": False, "label": "collapse"},
        5.0:  {"words": 1,   "coherent": False, "label": "total collapse"},
        8.0:  {"words": 1,   "coherent": False, "label": "garbage"},
        10.0: {"words": 1,   "coherent": False, "label": "garbage"},
    }

    # Unit vector at layer 15 (norm=1.0, residual=488)
    unit_alpha_data = {
        0.0:   {"words": 149, "coherent": True, "label": "baseline"},
        0.2:   {"words": 149, "coherent": True, "label": "0.04% SNR"},
        0.5:   {"words": 149, "coherent": True, "label": "no change"},
        1.0:   {"words": 149, "coherent": True, "label": "no change"},
        2.0:   {"words": 149, "coherent": True, "label": "no change"},
        5.0:   {"words": 149, "coherent": True, "label": "no change"},
        10.0:  {"words": 148, "coherent": True, "label": "1 word diff"},
        20.0:  {"words": 140, "coherent": True, "label": "slight shift"},
        50.0:  {"words": 95,  "coherent": True, "label": "moderate"},
        100.0: {"words": 40,  "coherent": True, "label": "strong"},
        200.0: {"words": 8,   "coherent": False, "label": "collapse"},
        500.0: {"words": 1,   "coherent": False, "label": "total collapse"},
    }

    # ── Top panel: raw vector phase diagram ──────────────────────────────────
    alphas_raw = sorted(raw_alpha_data.keys())
    words_raw = [raw_alpha_data[a]["words"] for a in alphas_raw]
    coherent_raw = [raw_alpha_data[a]["coherent"] for a in alphas_raw]

    # Phase regions (background shading)
    ax_top.axvspan(0, 0.3, color=COLORS["dead_zone"], alpha=0.15, zorder=0)
    ax_top.axvspan(0.3, 2.7, color=COLORS["sweet_spot"], alpha=0.1, zorder=0)
    ax_top.axvspan(2.7, 11, color=COLORS["collapse"], alpha=0.1, zorder=0)

    # Phase boundary lines
    ax_top.axvline(0.3, color=COLORS["dead_zone"], linestyle="--", alpha=0.6, linewidth=1.5, zorder=5)
    ax_top.axvline(2.7, color=COLORS["collapse"], linestyle="--", alpha=0.6, linewidth=1.5, zorder=5)

    # Zone labels
    ax_top.text(0.15, 155, "DEAD\nZONE", ha="center", va="bottom", fontsize=9,
                color=COLORS["dead_zone"], fontweight="bold", alpha=0.9)
    ax_top.text(1.5, 155, "EFFECTIVE STEERING", ha="center", va="bottom", fontsize=10,
                color=COLORS["sweet_spot"], fontweight="bold", alpha=0.9)
    ax_top.text(6.0, 155, "COHERENCE\nCOLLAPSE", ha="center", va="bottom", fontsize=9,
                color=COLORS["collapse"], fontweight="bold", alpha=0.9)

    # Plot the word count curve
    ax_top.plot(alphas_raw, words_raw, color=COLORS["terse"], linewidth=2.5, zorder=6,
                marker="o", markersize=6, markeredgecolor="white", markeredgewidth=0.5,
                label="Word count (raw vec, L15)")

    # Color points by coherence
    for a, w, c in zip(alphas_raw, words_raw, coherent_raw):
        color = COLORS["sweet_spot"] if c else COLORS["collapse"]
        ax_top.scatter([a], [w], c=color, s=80, zorder=7, edgecolors="white",
                       linewidth=1)

    # Annotate key points
    annotations = [
        (0.0, 149, "Baseline\n149 words", "right", (20, 10)),
        (2.0, 16, "Sweet spot\n16 words", "left", (-60, -20)),
        (3.0, 5, "Cliff edge!", "left", (-50, 15)),
    ]
    for x, y, text, ha, offset in annotations:
        ax_top.annotate(
            text, (x, y), textcoords="offset points", xytext=offset,
            fontsize=8, color=COLORS["accent"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2),
            ha=ha, va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["bg_dark"],
                      ec=COLORS["accent"], alpha=0.8)
        )

    # SNR axis on top
    ax_snr_top = ax_top.twiny()
    raw_norm_15 = RAW_VEC_NORMS[15]  # 19.6
    res_norm_15 = RESIDUAL_NORMS[15]  # 488
    snr_ticks = [0, 1, 2, 4, 8, 16, 20]
    snr_alphas = [s * res_norm_15 / raw_norm_15 / 100 for s in snr_ticks]
    ax_snr_top.set_xlim(ax_top.get_xlim())
    ax_snr_top.set_xticks([t / 100 * res_norm_15 / raw_norm_15 for t in snr_ticks[:5]])
    ax_snr_top.set_xticklabels([f"{t}%" for t in snr_ticks[:5]], fontsize=7)
    ax_snr_top.set_xlabel("SNR (effective_mag / residual_norm)", fontsize=8,
                          color=COLORS["text"], alpha=0.7)
    ax_snr_top.tick_params(colors=COLORS["text"], labelsize=7)

    ax_top.set_xlim(-0.2, 11)
    ax_top.set_ylim(-5, 170)
    ax_top.set_xlabel("Alpha (scaling factor for raw steering vector)", fontsize=11)
    ax_top.set_ylabel("Output Word Count", fontsize=11)
    ax_top.set_title(
        "Phase Diagram: Alpha vs. Output Behavior\n"
        "Raw steering vector at layer 15 (norm=19.6, residual=488)",
        fontsize=13, pad=25
    )
    ax_top.grid(True, alpha=0.15)

    # ── Effective magnitude annotation ───────────────────────────────────────
    eff_text = (
        "Effective magnitude = alpha x vec_norm\n"
        f"  a=2.0: {2.0*raw_norm_15:.1f} = {2.0*raw_norm_15/res_norm_15*100:.1f}% of residual = WORKS\n"
        f"  a=0.2 (unit): 0.2 = {0.2/res_norm_15*100:.4f}% of residual = NOTHING"
    )
    ax_top.text(
        0.98, 0.55, eff_text, transform=ax_top.transAxes,
        fontsize=7, fontfamily="monospace", color=COLORS["text"],
        ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc=COLORS["bg_dark"],
                  ec=COLORS["grid"], alpha=0.85)
    )

    # ── Bottom panel: unit vec comparison ────────────────────────────────────
    alphas_unit = sorted(unit_alpha_data.keys())
    words_unit = [unit_alpha_data[a]["words"] for a in alphas_unit]

    ax_bot.set_xscale("log")
    ax_bot.plot(alphas_unit[1:], words_unit[1:], color=COLORS["formal"], linewidth=2.5,
                marker="s", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                label="Unit vec (norm=1.0)", zorder=6)

    # Overlay raw vec data on same log scale
    alphas_raw_nz = [a for a in alphas_raw if a > 0]
    words_raw_nz = [raw_alpha_data[a]["words"] for a in alphas_raw_nz]
    ax_bot.plot(alphas_raw_nz, words_raw_nz, color=COLORS["terse"], linewidth=2.5,
                marker="o", markersize=5, markeredgecolor="white", markeredgewidth=0.5,
                label="Raw vec (norm=19.6)", zorder=6)

    # Horizontal reference line at baseline
    ax_bot.axhline(149, color=COLORS["baseline"], linestyle=":", alpha=0.4, linewidth=1)
    ax_bot.text(0.25, 152, "baseline (149 words)", fontsize=7, color=COLORS["baseline"], alpha=0.6)

    # Key insight annotation
    ax_bot.annotate(
        "Raw vec needs alpha ~2\nUnit vec needs alpha ~100\nfor same effect",
        xy=(2.0, 16), xytext=(15, 60),
        fontsize=8, color=COLORS["accent"],
        arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg_dark"],
                  ec=COLORS["accent"], alpha=0.8)
    )

    # The ratio annotation
    ax_bot.text(
        0.02, 0.05,
        f"Scale ratio: raw_norm/unit_norm = {raw_norm_15:.1f}x\n"
        f"So alpha_unit = alpha_raw x {raw_norm_15:.1f} for equivalent effect",
        transform=ax_bot.transAxes, fontsize=7, fontfamily="monospace",
        color=COLORS["text"], va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg_dark"],
                  ec=COLORS["grid"], alpha=0.8)
    )

    ax_bot.set_xlim(0.15, 600)
    ax_bot.set_ylim(-5, 170)
    ax_bot.set_xlabel("Alpha (log scale)", fontsize=11)
    ax_bot.set_ylabel("Output Word Count", fontsize=11)
    ax_bot.set_title(
        "Raw vs. Unit-Normalized Vector: Why alpha=0.20 Does Nothing",
        fontsize=12, pad=10
    )
    ax_bot.grid(True, alpha=0.15, which="both")
    ax_bot.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
