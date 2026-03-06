#!/usr/bin/env python3
"""
Visualization 1: Residual Stream Landscape

2D projection of the d_model=1024 activation space showing:
- Baseline output cluster (unsteered)
- Style basins (terse, formal, socratic, dry-wit)
- Cult-of-jason basin overlapping with terse (superposition)
- Steering trajectories as arrows at different alpha values
- Concentric rings for alpha thresholds: dead zone, sweet spot, collapse

Conceptual visualization using synthetic projected coordinates to illustrate
how steering vectors move outputs through high-dimensional space.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import EllipseCollection
from shared_style import apply_dark_style, COLORS

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "residual_landscape.png")


def make_gaussian_cluster(center, n=60, spread=0.3, seed=None):
    """Generate a 2D gaussian cluster of points."""
    rng = np.random.RandomState(seed)
    return center + rng.randn(n, 2) * spread


def main():
    apply_dark_style()

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # ── Define basin centers (conceptual 2D projection of 1024-d space) ──────
    basins = {
        "baseline":       np.array([0.0, 0.0]),
        "terse":          np.array([-2.5, 1.8]),
        "formal":         np.array([2.0, 2.5]),
        "socratic":       np.array([2.5, -1.5]),
        "dry-wit":        np.array([-1.5, -2.5]),
        "cult_of_jason":  np.array([-3.2, 2.5]),  # Near terse (superposition)
    }

    # ── Draw concentric alpha threshold rings ────────────────────────────────
    # These represent the effective magnitude in projected space
    ring_params = [
        (1.0, COLORS["dead_zone"],   0.15, "Dead zone (alpha < 0.3)"),
        (2.5, COLORS["sweet_spot"],  0.10, "Sweet spot (alpha 1.0-2.5)"),
        (4.5, COLORS["collapse"],    0.08, "Collapse zone (alpha > 3.0)"),
    ]
    for radius, color, alpha, label in ring_params:
        circle = plt.Circle(
            basins["baseline"], radius, fill=True,
            facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=1.5, linestyle="--", zorder=0
        )
        ax.add_patch(circle)
        # Label on the ring
        angle = np.radians(135)
        lx = basins["baseline"][0] + radius * np.cos(angle)
        ly = basins["baseline"][1] + radius * np.sin(angle)
        ax.annotate(
            label, (lx, ly), fontsize=7, color=color, alpha=0.9,
            ha="center", va="bottom", rotation=45,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["bg_dark"], ec=color, alpha=0.7)
        )

    # ── Draw style basin regions (soft ellipses) ─────────────────────────────
    basin_style_colors = {
        "terse":         COLORS["terse"],
        "formal":        COLORS["formal"],
        "socratic":      COLORS["socratic"],
        "dry-wit":       COLORS["dry-wit"],
        "cult_of_jason": COLORS["cult_of_jason"],
    }

    for name, center in basins.items():
        if name == "baseline":
            continue
        color = basin_style_colors[name]
        # Soft basin ellipse
        ellipse = plt.matplotlib.patches.Ellipse(
            center, width=1.8, height=1.4, angle=np.random.RandomState(hash(name) % 1000).randint(-30, 30),
            facecolor=color, alpha=0.08, edgecolor=color, linewidth=1.5, linestyle="-", zorder=1
        )
        ax.add_patch(ellipse)
        # Label
        ax.text(
            center[0], center[1] + 0.85, name.replace("_", " ").title(),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=color, zorder=10
        )

    # ── Draw superposition overlap between terse and cult_of_jason ───────────
    overlap_center = (basins["terse"] + basins["cult_of_jason"]) / 2
    overlap = plt.matplotlib.patches.Ellipse(
        overlap_center, width=1.5, height=1.0, angle=30,
        facecolor=COLORS["accent"], alpha=0.06, edgecolor=COLORS["accent"],
        linewidth=1.0, linestyle=":", zorder=1
    )
    ax.add_patch(overlap)
    ax.text(
        overlap_center[0] + 0.1, overlap_center[1] - 0.1,
        "superposition\n(shared features\nat d_model=1024)",
        ha="center", va="top", fontsize=7, color=COLORS["accent"],
        style="italic", zorder=10,
        bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["bg_dark"], ec=COLORS["accent"], alpha=0.6)
    )

    # ── Scatter baseline cluster ─────────────────────────────────────────────
    baseline_pts = make_gaussian_cluster(basins["baseline"], n=80, spread=0.35, seed=42)
    ax.scatter(
        baseline_pts[:, 0], baseline_pts[:, 1],
        c=COLORS["baseline"], s=12, alpha=0.5, zorder=3, label="Baseline outputs"
    )
    ax.scatter(
        [0], [0], c=COLORS["baseline"], s=120, marker="*", zorder=5,
        edgecolors="white", linewidth=0.5
    )
    ax.text(
        0.15, -0.35, "Baseline\n(unsteered)", ha="left", va="top",
        fontsize=9, color=COLORS["baseline"], fontweight="bold", zorder=10
    )

    # ── Scatter steered output clusters in each basin ────────────────────────
    for name, center in basins.items():
        if name == "baseline":
            continue
        color = basin_style_colors[name]
        pts = make_gaussian_cluster(center, n=40, spread=0.25, seed=hash(name) % 10000)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=8, alpha=0.4, zorder=3)

    # ── Draw steering trajectories (arrows from baseline to each basin) ──────
    alpha_values = [0.5, 1.0, 2.0, 3.0]
    for style_name in ["terse", "formal", "socratic", "dry-wit"]:
        target = basins[style_name]
        direction = target - basins["baseline"]
        color = basin_style_colors[style_name]

        for i, alpha_val in enumerate(alpha_values):
            # Scale arrow by alpha fraction (alpha=2.0 reaches the basin center)
            frac = alpha_val / 2.0
            end = basins["baseline"] + direction * min(frac, 1.5)

            # Arrow width based on alpha
            width = 1.0 + alpha_val * 0.5
            head_width = 0.08 + alpha_val * 0.02

            if alpha_val <= 0.5:
                linestyle = ":"
                alpha_draw = 0.3
            elif alpha_val <= 2.0:
                linestyle = "-"
                alpha_draw = 0.5 + alpha_val * 0.15
            else:
                linestyle = "--"
                alpha_draw = 0.4

            arrow = FancyArrowPatch(
                basins["baseline"], end,
                arrowstyle=f"->,head_width={head_width * 30},head_length={head_width * 20}",
                color=color, alpha=alpha_draw, linewidth=width,
                linestyle=linestyle, zorder=4,
                mutation_scale=10,
            )
            ax.add_patch(arrow)

            # Label alpha at arrow tip (only for terse to avoid clutter)
            if style_name == "terse" and alpha_val in [0.5, 2.0, 3.0]:
                ax.text(
                    end[0] + 0.15, end[1] + 0.15,
                    f"a={alpha_val}",
                    fontsize=6, color=color, alpha=0.8, zorder=10
                )

    # ── Draw the "collapse" trajectory for alpha > 3.0 ───────────────────────
    # Shows outputs scattering into garbage at high alpha
    rng = np.random.RandomState(99)
    collapse_dir = basins["terse"] - basins["baseline"]
    collapse_dir = collapse_dir / np.linalg.norm(collapse_dir)
    for _ in range(15):
        # Random scattered dots beyond the collapse ring
        r = 4.5 + rng.rand() * 1.5
        angle_offset = rng.randn() * 0.8
        base_angle = np.arctan2(collapse_dir[1], collapse_dir[0])
        x = r * np.cos(base_angle + angle_offset)
        y = r * np.sin(base_angle + angle_offset)
        ax.scatter(
            [x], [y], c=COLORS["collapse"], s=15, alpha=0.3,
            marker="x", zorder=3
        )
    ax.text(
        -4.0, 4.5, "Coherence\ncollapse\n(garbage tokens)",
        fontsize=8, color=COLORS["collapse"], ha="center", va="center",
        style="italic", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg_dark"], ec=COLORS["collapse"], alpha=0.6)
    )

    # ── Annotations ──────────────────────────────────────────────────────────
    ax.set_xlim(-6, 5.5)
    ax.set_ylim(-5, 6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("PCA Component 1 (projected from d_model=1024)", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.set_title(
        "Residual Stream Landscape: Qwen3-0.6B Activation Space\n"
        "How steering vectors move outputs through style basins",
        fontsize=13, pad=15
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["baseline"], alpha=0.6, label="Baseline (unsteered)"),
        mpatches.Patch(color=COLORS["terse"], alpha=0.6, label="Terse basin"),
        mpatches.Patch(color=COLORS["formal"], alpha=0.6, label="Formal basin"),
        mpatches.Patch(color=COLORS["socratic"], alpha=0.6, label="Socratic basin"),
        mpatches.Patch(color=COLORS["dry-wit"], alpha=0.6, label="Dry-wit basin"),
        mpatches.Patch(color=COLORS["cult_of_jason"], alpha=0.6, label="Cult-of-Jason basin"),
        mpatches.Patch(color=COLORS["accent"], alpha=0.3, label="Superposition overlap"),
        mpatches.Patch(color=COLORS["dead_zone"], alpha=0.3, label="Dead zone (alpha < 0.3)"),
        mpatches.Patch(color=COLORS["sweet_spot"], alpha=0.3, label="Sweet spot (1.0-2.5)"),
        mpatches.Patch(color=COLORS["collapse"], alpha=0.3, label="Collapse zone (> 3.0)"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=7,
        framealpha=0.8, ncol=2
    )

    # Inset annotation box with key insight
    textbox = (
        "Key insight: at d_model=1024, 'terse' and\n"
        "'specification-driven worldview' share features\n"
        "(superposition). Pushing on terse drags outputs\n"
        "toward cult_of_jason. The lens eval detects this\n"
        "by checking if pasta/grief/tides get contaminated\n"
        "with Lean4 proofs and org-mode vocabulary."
    )
    props = dict(boxstyle="round,pad=0.5", facecolor=COLORS["bg_dark"],
                 edgecolor=COLORS["accent"], alpha=0.85)
    ax.text(
        0.02, 0.02, textbox, transform=ax.transAxes,
        fontsize=6.5, verticalalignment="bottom", bbox=props,
        family="monospace", color=COLORS["accent"]
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
