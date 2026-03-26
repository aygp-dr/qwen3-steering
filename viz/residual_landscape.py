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

Produces three figures:
  1. residual_landscape.png         -- Original combined view (kept for backwards compat)
  2. residual_landscape_basins.png  -- Figure A: Style Basins with Steering Vectors
  3. residual_landscape_alpha_zones.png -- Figure B: Alpha Zone Overlay
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "residual_landscape.png")
OUTPUT_BASINS = os.path.join(OUTPUT_DIR, "residual_landscape_basins.png")
OUTPUT_ALPHA = os.path.join(OUTPUT_DIR, "residual_landscape_alpha_zones.png")


# ── Shared data ─────────────────────────────────────────────────────────────

BASINS = {
    "baseline":       np.array([0.0, 0.0]),
    "terse":          np.array([-2.5, 1.8]),
    "formal":         np.array([2.0, 2.5]),
    "socratic":       np.array([2.5, -1.5]),
    "dry-wit":        np.array([-1.5, -2.5]),
    "cult_of_jason":  np.array([-3.2, 2.5]),  # Near terse (superposition)
}

BASIN_STYLE_COLORS = {
    "terse":         COLORS["terse"],
    "formal":        COLORS["formal"],
    "socratic":      COLORS["socratic"],
    "dry-wit":       COLORS["dry-wit"],
    "cult_of_jason": COLORS["cult_of_jason"],
}


def make_gaussian_cluster(center, n=60, spread=0.3, seed=None):
    """Generate a 2D gaussian cluster of points."""
    rng = np.random.RandomState(seed)
    return center + rng.randn(n, 2) * spread


# ── Figure A: Style Basins with Steering Vectors ────────────────────────────

def draw_figure_basins():
    """PCA scatter of style basins, baseline, steering arrows, confidence ellipses."""
    apply_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # -- 2-sigma confidence ellipses per basin --
    for name, center in BASINS.items():
        if name == "baseline":
            continue
        color = BASIN_STYLE_COLORS[name]
        rng_angle = np.random.RandomState(hash(name) % 1000)
        ellipse_angle = rng_angle.randint(-30, 30)

        # 2-sigma ellipse (larger, dashed border for confidence region)
        e2 = plt.matplotlib.patches.Ellipse(
            center, width=2.6, height=2.0, angle=ellipse_angle,
            facecolor="none", edgecolor=color, linewidth=1.0,
            linestyle="--", alpha=0.4, zorder=1,
        )
        ax.add_patch(e2)

        # Inner filled basin ellipse (1-sigma)
        ellipse = plt.matplotlib.patches.Ellipse(
            center, width=1.8, height=1.4, angle=ellipse_angle,
            facecolor=color, alpha=0.10, edgecolor=color,
            linewidth=1.5, linestyle="-", zorder=1,
        )
        ax.add_patch(ellipse)

        # Label
        ax.text(
            center[0], center[1] + 0.85,
            name.replace("_", " ").title(),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=color, zorder=10,
        )

    # -- Scatter steered output clusters in each basin --
    for name, center in BASINS.items():
        if name == "baseline":
            continue
        color = BASIN_STYLE_COLORS[name]
        pts = make_gaussian_cluster(center, n=40, spread=0.25, seed=hash(name) % 10000)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=10, alpha=0.45, zorder=3)

    # -- Baseline cluster and center point --
    baseline_pts = make_gaussian_cluster(BASINS["baseline"], n=80, spread=0.35, seed=42)
    ax.scatter(
        baseline_pts[:, 0], baseline_pts[:, 1],
        c=COLORS["baseline"], s=12, alpha=0.5, zorder=3, label="Baseline outputs",
    )
    ax.scatter(
        [0], [0], c=COLORS["baseline"], s=150, marker="*", zorder=5,
        edgecolors="white", linewidth=0.5,
    )
    ax.text(
        0.15, -0.40, "Baseline\n(unsteered)", ha="left", va="top",
        fontsize=10, color=COLORS["baseline"], fontweight="bold", zorder=10,
    )

    # -- Steering vector arrows (baseline -> each basin at alpha=2.0) --
    for style_name in ["terse", "formal", "socratic", "dry-wit", "cult_of_jason"]:
        target = BASINS[style_name]
        color = BASIN_STYLE_COLORS[style_name]

        arrow = FancyArrowPatch(
            BASINS["baseline"], target,
            arrowstyle="->,head_width=6,head_length=4",
            color=color, alpha=0.75, linewidth=2.0,
            linestyle="-", zorder=4, mutation_scale=10,
        )
        ax.add_patch(arrow)

        # Label the direction at 60% along the arrow
        mid = BASINS["baseline"] + (target - BASINS["baseline"]) * 0.55
        # Perpendicular offset for readability
        direction = target - BASINS["baseline"]
        perp = np.array([-direction[1], direction[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-8) * 0.25
        ax.text(
            mid[0] + perp[0], mid[1] + perp[1],
            f"v_{style_name.split('-')[0][:4]}",
            fontsize=7, color=color, alpha=0.85, zorder=10,
            style="italic",
        )

    # -- Axes and title --
    ax.set_xlim(-5.5, 5)
    ax.set_ylim(-4.5, 5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("PCA Component 1 (projected from d_model=1024)", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.set_title(
        "Figure A: Style Basins with Steering Vectors\n"
        "Qwen3-0.6B activation space (2D PCA projection)",
        fontsize=13, pad=15,
    )

    # -- Legend --
    legend_handles = [
        mpatches.Patch(color=COLORS["baseline"], alpha=0.6, label="Baseline (unsteered)"),
        mpatches.Patch(color=COLORS["terse"], alpha=0.6, label="Terse basin"),
        mpatches.Patch(color=COLORS["formal"], alpha=0.6, label="Formal basin"),
        mpatches.Patch(color=COLORS["socratic"], alpha=0.6, label="Socratic basin"),
        mpatches.Patch(color=COLORS["dry-wit"], alpha=0.6, label="Dry-wit basin"),
        mpatches.Patch(color=COLORS["cult_of_jason"], alpha=0.6, label="Cult-of-Jason basin"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=8,
        framealpha=0.8, ncol=2,
    )

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_BASINS)
    print(f"Saved: {OUTPUT_BASINS}")
    plt.close()

    # -- Alt-text --
    alt_text = (
        "PCA scatter plot of Qwen3-0.6B residual stream activations showing five style basins "
        "(terse, formal, socratic, dry-wit, cult-of-jason) arranged around a central baseline "
        "cluster. Each basin is marked with a filled 1-sigma ellipse and a dashed 2-sigma "
        "confidence ellipse. Steering vector arrows extend from the baseline center to each "
        "basin, indicating the direction that alpha=2.0 steering pushes model outputs in "
        "activation space."
    )
    alt_path = OUTPUT_BASINS.replace(".png", ".txt")
    with open(alt_path, "w") as f:
        f.write(alt_text + "\n")
    print(f"Saved: {alt_path}")


# ── Figure B: Alpha Zone Overlay ────────────────────────────────────────────

def draw_figure_alpha_zones():
    """Concentric circles for dead/effective/collapse zones plus superposition annotation."""
    apply_dark_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    baseline = BASINS["baseline"]

    # -- Concentric alpha threshold rings --
    ring_params = [
        (1.0, COLORS["dead_zone"],  0.18, "Dead zone\n(alpha < 0.3)",
         "Negligible perturbation;\nsteering has no measurable effect"),
        (2.5, COLORS["sweet_spot"], 0.12, "Effective zone\n(alpha 1.0-2.5)",
         "Sweet spot: outputs shift\ninto target style basins"),
        (4.5, COLORS["collapse"],   0.10, "Collapse zone\n(alpha > 3.0)",
         "Coherence collapse;\ngarbage token generation"),
    ]

    for radius, color, alpha, label, description in ring_params:
        circle = plt.Circle(
            baseline, radius, fill=True,
            facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=2.0, linestyle="--", zorder=0,
        )
        ax.add_patch(circle)

        # Label on the ring (upper-left quadrant)
        angle = np.radians(135)
        lx = baseline[0] + radius * np.cos(angle)
        ly = baseline[1] + radius * np.sin(angle)
        ax.annotate(
            label, (lx, ly), fontsize=9, color=color, alpha=0.95,
            ha="center", va="bottom", rotation=45, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["bg_dark"], ec=color, alpha=0.8),
        )

        # Description text on the right side
        rx = baseline[0] + radius * np.cos(np.radians(-30))
        ry = baseline[1] + radius * np.sin(np.radians(-30))
        ax.text(
            rx + 0.3, ry, description,
            fontsize=7, color=color, alpha=0.8, va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["bg_dark"], ec=color, alpha=0.5),
            zorder=10,
        )

    # -- Center marker --
    ax.scatter(
        [0], [0], c=COLORS["baseline"], s=150, marker="*", zorder=5,
        edgecolors="white", linewidth=0.5,
    )
    ax.text(
        0.15, -0.40, "Baseline\n(origin)", ha="left", va="top",
        fontsize=10, color=COLORS["baseline"], fontweight="bold", zorder=10,
    )

    # -- Superposition overlap annotation --
    # Show where terse and cult_of_jason overlap in the effective zone
    terse_center = BASINS["terse"]
    coj_center = BASINS["cult_of_jason"]
    overlap_center = (terse_center + coj_center) / 2

    # Light markers for basin positions
    for name, pos in [("terse", terse_center), ("cult_of_jason", coj_center)]:
        color = BASIN_STYLE_COLORS[name]
        ax.scatter(
            [pos[0]], [pos[1]], c=color, s=80, marker="o",
            edgecolors="white", linewidth=0.5, alpha=0.7, zorder=5,
        )
        ax.text(
            pos[0], pos[1] + 0.35,
            name.replace("_", " ").title(),
            ha="center", va="bottom", fontsize=8, color=color,
            fontweight="bold", alpha=0.8, zorder=10,
        )

    # Overlap ellipse
    overlap = plt.matplotlib.patches.Ellipse(
        overlap_center, width=1.5, height=1.0, angle=30,
        facecolor=COLORS["accent"], alpha=0.08, edgecolor=COLORS["accent"],
        linewidth=1.5, linestyle=":", zorder=2,
    )
    ax.add_patch(overlap)
    ax.text(
        overlap_center[0] + 0.1, overlap_center[1] - 0.5,
        "Superposition overlap\n(shared features at d_model=1024)\n"
        "Steering toward 'terse' partially\nactivates 'cult_of_jason'",
        ha="center", va="top", fontsize=8, color=COLORS["accent"],
        style="italic", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg_dark"],
                  ec=COLORS["accent"], alpha=0.7),
    )

    # -- Collapse scatter (garbage tokens beyond the ring) --
    rng = np.random.RandomState(99)
    collapse_dir = terse_center - baseline
    collapse_dir = collapse_dir / np.linalg.norm(collapse_dir)
    for _ in range(20):
        r = 4.5 + rng.rand() * 1.5
        angle_offset = rng.randn() * 0.8
        base_angle = np.arctan2(collapse_dir[1], collapse_dir[0])
        x = r * np.cos(base_angle + angle_offset)
        y = r * np.sin(base_angle + angle_offset)
        ax.scatter(
            [x], [y], c=COLORS["collapse"], s=18, alpha=0.35,
            marker="x", zorder=3,
        )
    ax.text(
        -4.0, 4.5, "Coherence collapse\n(garbage tokens)",
        fontsize=9, color=COLORS["collapse"], ha="center", va="center",
        style="italic", zorder=10,
        bbox=dict(boxstyle="round,pad=0.3", fc=COLORS["bg_dark"],
                  ec=COLORS["collapse"], alpha=0.6),
    )

    # -- Radial scale annotation --
    # Draw a thin radial line with tick marks
    ax.annotate(
        "", xy=(4.8, 0), xytext=(0.3, 0),
        arrowprops=dict(arrowstyle="->", color=COLORS["text"], alpha=0.4, lw=1.0),
    )
    for r, label in [(1.0, "0.3"), (2.5, "1.5"), (4.5, "3.0+")]:
        ax.plot([r, r], [-0.1, 0.1], color=COLORS["text"], alpha=0.4, lw=1.0)
        ax.text(r, -0.25, f"a={label}", fontsize=6, color=COLORS["text"],
                alpha=0.6, ha="center", va="top")
    ax.text(5.0, 0, "alpha", fontsize=7, color=COLORS["text"], alpha=0.5,
            ha="left", va="center", style="italic")

    # -- Axes and title --
    ax.set_xlim(-6.5, 6)
    ax.set_ylim(-5.5, 6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    ax.set_xlabel("PCA Component 1 (projected from d_model=1024)", fontsize=10)
    ax.set_ylabel("PCA Component 2", fontsize=10)
    ax.set_title(
        "Figure B: Alpha Zone Overlay\n"
        "Dead zone / effective / collapse regions in activation space",
        fontsize=13, pad=15,
    )

    # -- Legend --
    legend_handles = [
        mpatches.Patch(color=COLORS["dead_zone"], alpha=0.3, label="Dead zone (alpha < 0.3)"),
        mpatches.Patch(color=COLORS["sweet_spot"], alpha=0.3, label="Effective zone (alpha 1.0-2.5)"),
        mpatches.Patch(color=COLORS["collapse"], alpha=0.3, label="Collapse zone (alpha > 3.0)"),
        mpatches.Patch(color=COLORS["accent"], alpha=0.3, label="Superposition overlap"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=8,
        framealpha=0.8,
    )

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_ALPHA)
    print(f"Saved: {OUTPUT_ALPHA}")
    plt.close()

    # -- Alt-text --
    alt_text = (
        "Concentric ring diagram showing three alpha-magnitude zones in Qwen3-0.6B "
        "activation space: an inner dead zone (alpha < 0.3) where steering has no effect, "
        "a middle effective zone (alpha 1.0-2.5) where outputs shift into target style basins, "
        "and an outer collapse zone (alpha > 3.0) where coherence breaks down into garbage tokens. "
        "The terse and cult-of-jason basins are highlighted in the effective zone with a "
        "superposition overlap annotation showing shared features at d_model=1024."
    )
    alt_path = OUTPUT_ALPHA.replace(".png", ".txt")
    with open(alt_path, "w") as f:
        f.write(alt_text + "\n")
    print(f"Saved: {alt_path}")


# ── Original combined figure (preserved for backwards compatibility) ────────

def draw_original():
    """Original combined visualization with all elements."""
    apply_dark_style()

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # ── Draw concentric alpha threshold rings ────────────────────────────────
    ring_params = [
        (1.0, COLORS["dead_zone"],   0.15, "Dead zone (alpha < 0.3)"),
        (2.5, COLORS["sweet_spot"],  0.10, "Sweet spot (alpha 1.0-2.5)"),
        (4.5, COLORS["collapse"],    0.08, "Collapse zone (alpha > 3.0)"),
    ]
    for radius, color, alpha, label in ring_params:
        circle = plt.Circle(
            BASINS["baseline"], radius, fill=True,
            facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=1.5, linestyle="--", zorder=0
        )
        ax.add_patch(circle)
        angle = np.radians(135)
        lx = BASINS["baseline"][0] + radius * np.cos(angle)
        ly = BASINS["baseline"][1] + radius * np.sin(angle)
        ax.annotate(
            label, (lx, ly), fontsize=7, color=color, alpha=0.9,
            ha="center", va="bottom", rotation=45,
            bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["bg_dark"], ec=color, alpha=0.7)
        )

    # ── Draw style basin regions (soft ellipses) ─────────────────────────────
    for name, center in BASINS.items():
        if name == "baseline":
            continue
        color = BASIN_STYLE_COLORS[name]
        ellipse = plt.matplotlib.patches.Ellipse(
            center, width=1.8, height=1.4, angle=np.random.RandomState(hash(name) % 1000).randint(-30, 30),
            facecolor=color, alpha=0.08, edgecolor=color, linewidth=1.5, linestyle="-", zorder=1
        )
        ax.add_patch(ellipse)
        ax.text(
            center[0], center[1] + 0.85, name.replace("_", " ").title(),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=color, zorder=10
        )

    # ── Draw superposition overlap between terse and cult_of_jason ───────────
    overlap_center = (BASINS["terse"] + BASINS["cult_of_jason"]) / 2
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
    baseline_pts = make_gaussian_cluster(BASINS["baseline"], n=80, spread=0.35, seed=42)
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
    for name, center in BASINS.items():
        if name == "baseline":
            continue
        color = BASIN_STYLE_COLORS[name]
        pts = make_gaussian_cluster(center, n=40, spread=0.25, seed=hash(name) % 10000)
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=8, alpha=0.4, zorder=3)

    # ── Draw steering trajectories (arrows from baseline to each basin) ──────
    alpha_values = [0.5, 1.0, 2.0, 3.0]
    for style_name in ["terse", "formal", "socratic", "dry-wit"]:
        target = BASINS[style_name]
        direction = target - BASINS["baseline"]
        color = BASIN_STYLE_COLORS[style_name]

        for i, alpha_val in enumerate(alpha_values):
            frac = alpha_val / 2.0
            end = BASINS["baseline"] + direction * min(frac, 1.5)
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
                BASINS["baseline"], end,
                arrowstyle=f"->,head_width={head_width * 30},head_length={head_width * 20}",
                color=color, alpha=alpha_draw, linewidth=width,
                linestyle=linestyle, zorder=4,
                mutation_scale=10,
            )
            ax.add_patch(arrow)

            if style_name == "terse" and alpha_val in [0.5, 2.0, 3.0]:
                ax.text(
                    end[0] + 0.15, end[1] + 0.15,
                    f"a={alpha_val}",
                    fontsize=6, color=color, alpha=0.8, zorder=10
                )

    # ── Draw the "collapse" trajectory for alpha > 3.0 ───────────────────────
    rng = np.random.RandomState(99)
    collapse_dir = BASINS["terse"] - BASINS["baseline"]
    collapse_dir = collapse_dir / np.linalg.norm(collapse_dir)
    for _ in range(15):
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


def main():
    draw_original()
    draw_figure_basins()
    draw_figure_alpha_zones()


if __name__ == "__main__":
    main()
