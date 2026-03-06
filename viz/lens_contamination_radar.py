#!/usr/bin/env python3
"""
Visualization 4: Lens Contamination Radar Chart

Radar/spider chart showing 12 lens contamination scores:
- Baseline (clean): inner polygon, all low
- Steered-with-leak: outer polygon where cult_of_jason and ai_hype spike
- Clean-steer: style metrics change but lenses stay flat

Demonstrates good vs bad steering via the lens eval framework.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shared_style import apply_dark_style, COLORS, LENS_NAMES, LENS_DISPLAY

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "lens_contamination_radar.png")


def radar_chart(ax, categories, datasets, title=""):
    """Draw a radar/spider chart on the given axes.

    datasets: list of (label, values, color, alpha, linestyle)
    """
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    # Category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=7, color=COLORS["text"])

    # Radial grid
    max_val = max(max(d[1]) for d in datasets) * 1.2
    rticks = np.linspace(0, max_val, 5)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{v:.1f}%" for v in rticks], fontsize=6, color=COLORS["text"], alpha=0.6)
    ax.set_ylim(0, max_val)

    # Grid styling
    ax.grid(True, color=COLORS["grid"], alpha=0.3)
    ax.spines["polar"].set_color(COLORS["grid"])
    ax.spines["polar"].set_alpha(0.3)

    for label, values, color, alpha, linestyle, fill_alpha in datasets:
        vals = values.tolist()
        vals += vals[:1]  # Close
        ax.plot(angles, vals, color=color, linewidth=2, linestyle=linestyle,
                label=label, alpha=alpha, zorder=5)
        ax.fill(angles, vals, color=color, alpha=fill_alpha, zorder=3)

        # Mark data points
        ax.scatter(angles[:-1], values, c=color, s=20, zorder=6,
                   edgecolors="white", linewidth=0.5, alpha=alpha)

    if title:
        ax.set_title(title, fontsize=11, pad=20, color=COLORS["text"], fontweight="bold")


def main():
    apply_dark_style()

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Lens Contamination Radar: Detecting Conceptual Drift in Steered Outputs\n"
        "12 orthogonal lenses scored across 10 neutral probes (bread, grief, tides, jazz, ...)",
        fontsize=14, fontweight="bold", y=0.98, color=COLORS["accent"]
    )

    # ── Category labels ──────────────────────────────────────────────────────
    categories = [LENS_DISPLAY.get(l, l) for l in LENS_NAMES]

    # ── Synthetic contamination data ─────────────────────────────────────────
    # These are hypothetical but calibrated to the project's lens_eval.py thresholds

    # Baseline: clean model, no steering. All lenses < 1%
    baseline = np.array([
        0.3,   # makefile
        0.2,   # guile
        0.1,   # orgmode
        0.5,   # monetization
        0.4,   # sports
        0.3,   # religion
        0.2,   # politics
        0.6,   # ai_hype (some ambient AI vocabulary)
        0.2,   # conspiracy
        0.3,   # scarcity
        0.2,   # therapy_speak
        0.4,   # cult_of_jason
    ])

    # Steered with leak: terse steering at alpha=2.0 using raw vec at layer 15
    # but the terse vector has superposition with cult_of_jason features
    leaked = np.array([
        1.8,   # makefile (Makefile-as-spec leaks)
        1.5,   # guile (scheme vocabulary leaks)
        1.2,   # orgmode (org-mode leaks)
        0.5,   # monetization (unchanged)
        0.3,   # sports (unchanged)
        0.2,   # religion (unchanged)
        0.3,   # politics (unchanged)
        2.1,   # ai_hype (steering/activation vocab leaks)
        0.2,   # conspiracy (unchanged)
        0.4,   # scarcity (unchanged)
        0.2,   # therapy_speak (unchanged)
        4.8,   # cult_of_jason (MAJOR SPIKE — superposition)
    ])

    # Clean steer: ideal case where only style changes, no lens contamination
    clean_steer = np.array([
        0.4,   # makefile
        0.3,   # guile
        0.2,   # orgmode
        0.4,   # monetization
        0.3,   # sports
        0.2,   # religion
        0.2,   # politics
        0.7,   # ai_hype (slightly elevated, within tolerance)
        0.2,   # conspiracy
        0.3,   # scarcity
        0.2,   # therapy_speak
        0.5,   # cult_of_jason (within tolerance)
    ])

    # Over-steered collapse: alpha=4.0, everything goes haywire
    collapsed = np.array([
        3.2,   # makefile
        2.8,   # guile
        2.1,   # orgmode
        1.5,   # monetization
        1.2,   # sports
        0.8,   # religion
        1.1,   # politics
        5.5,   # ai_hype
        1.8,   # conspiracy
        2.0,   # scarcity
        1.5,   # therapy_speak
        8.2,   # cult_of_jason
    ])

    # ── Four-panel radar layout ──────────────────────────────────────────────

    # Panel 1: Baseline vs Leaked
    ax1 = fig.add_subplot(2, 2, 1, projection="polar")
    radar_chart(ax1, categories, [
        ("Baseline (clean)", baseline, COLORS["clean"], 0.9, "-", 0.15),
        ("Steered with leak (a=2.0)", leaked, COLORS["leaked"], 0.9, "-", 0.10),
    ], title="Bad Steering: Superposition Leak")

    # Panel 2: Baseline vs Clean Steer
    ax2 = fig.add_subplot(2, 2, 2, projection="polar")
    radar_chart(ax2, categories, [
        ("Baseline (clean)", baseline, COLORS["clean"], 0.9, "-", 0.15),
        ("Clean steer (a=2.0)", clean_steer, COLORS["well_steered"], 0.9, "-", 0.10),
    ], title="Good Steering: Style Without Drift")

    # Panel 3: All three overlaid
    ax3 = fig.add_subplot(2, 2, 3, projection="polar")
    radar_chart(ax3, categories, [
        ("Baseline", baseline, COLORS["clean"], 0.7, "-", 0.08),
        ("Leaked steer", leaked, COLORS["leaked"], 0.7, "--", 0.06),
        ("Clean steer", clean_steer, COLORS["well_steered"], 0.9, "-", 0.10),
    ], title="Comparison: All Three Profiles")

    # Panel 4: Collapsed (alpha=4.0)
    ax4 = fig.add_subplot(2, 2, 4, projection="polar")
    radar_chart(ax4, categories, [
        ("Baseline", baseline, COLORS["clean"], 0.7, "-", 0.10),
        ("Collapsed (a=4.0)", collapsed, COLORS["collapse"], 0.9, "-", 0.08),
    ], title="Coherence Collapse: Everything Spikes")

    # ── Legends ──────────────────────────────────────────────────────────────
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.15), fontsize=7,
                  framealpha=0.8)

    # ── Threshold annotation ─────────────────────────────────────────────────
    threshold_text = (
        "Contamination thresholds (from lens_eval.py):\n"
        "  < 1%  : clean -- lens not present\n"
        "  1-3%  : ambient -- exposed but not captured\n"
        "  3-5%  : captured -- neutral topics framed through lens\n"
        "  > 5%  : full contamination -- sourdough has a Lean4 type"
    )
    fig.text(
        0.5, 0.02, threshold_text,
        ha="center", va="bottom", fontsize=8, fontfamily="monospace",
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=COLORS["bg_dark"],
                  ec=COLORS["accent"], alpha=0.9)
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
