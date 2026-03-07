#!/usr/bin/env python3
"""
Experiment Progression: key finding from each phase in one figure.

Renders a vertical timeline of experiments 01-06 with their primary
result metric and the insight that carried forward.

Usage:
    python viz/experiment_progression.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shared_style import apply_dark_style, COLORS, layer_color

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "experiment_progression.png")

# Each experiment: (number, short_name, status, key_metric, insight, forward_arrow_text)
EXPERIMENTS = [
    {
        "num": "01",
        "name": "Layer Scorecard",
        "status": "done",
        "cprr": "C-1..5",
        "metric_label": "Topic Cosine by Region",
        "metric_data": {
            "L0-7":   {"val": 0.26, "color": COLORS["collapse"]},
            "L12-17": {"val": 0.61, "color": COLORS["sweet_spot"]},
            "L18-22": {"val": 0.89, "color": COLORS["formal"]},
            "L23-27": {"val": 0.78, "color": COLORS["terse"]},
        },
        "insight": "Sweet spot at L12-17:\n72% word reduction, topic intact",
        "forward": "Confirms injection site",
    },
    {
        "num": "02",
        "name": "Verbosity Direction",
        "status": "done",
        "cprr": "C-7,8",
        "metric_label": "Stability vs Norm",
        "metric_data": {
            "L12": {"val": 0.962, "color": COLORS["sweet_spot"], "secondary": 28.3},
            "L17": {"val": 0.951, "color": COLORS["sweet_spot"], "secondary": 64.1},
            "L22": {"val": 0.873, "color": COLORS["baseline"], "secondary": 122.7},
            "L27": {"val": 0.641, "color": COLORS["collapse"], "secondary": 211.4},
        },
        "insight": "Stability predicts steerability,\nnot norm. Peak norm L27 = collapse",
        "forward": "Use L12 (stability=0.962)",
    },
    {
        "num": "03",
        "name": "Bimodal Injection",
        "status": "done",
        "cprr": "C-9",
        "metric_label": "Word Count: Inject vs Prompt",
        "metric_data": {
            "prompt_terse":  {"val": 20,  "color": COLORS["terse"]},
            "inject_terse":  {"val": 12,  "color": COLORS["terse"], "marker": "overshoot"},
            "prompt_verbose": {"val": 70,  "color": COLORS["formal"]},
            "inject_verbose": {"val": 214, "color": COLORS["formal"], "marker": "overshoot"},
        },
        "insight": "Alpha=2.0 overshoots prompt control:\nterse 0.6x, verbose 3.0x",
        "forward": "Need alpha calibration",
    },
    {
        "num": "04",
        "name": "Alpha Parity Sweep",
        "status": "pending",
        "cprr": "C-10?",
        "metric_label": "Predicted: alpha-parity curve",
        "metric_data": None,
        "insight": "Find alpha where inject ≈ prompt\n(expected: terse α < verbose α)",
        "forward": "Calibrated alpha values",
    },
    {
        "num": "05",
        "name": "Multi-Layer Steering",
        "status": "pending",
        "cprr": "C-6",
        "metric_label": "Predicted: composite score",
        "metric_data": None,
        "insight": "Distribute pressure across layers\nvs concentrate at one",
        "forward": "Optimal config for production",
    },
    {
        "num": "06",
        "name": "WordNet Geometry",
        "status": "pending",
        "cprr": "C-10..13",
        "metric_label": "Predicted: relation orthogonality",
        "metric_data": None,
        "insight": "Are hypernym/antonym orthogonal?\nStructure vs pattern-matching",
        "forward": None,
    },
]


def draw_mini_bar(ax, x, y, w, h, data):
    """Draw a small horizontal bar chart within the figure."""
    n = len(data)
    bar_h = h / (n + 0.5)
    labels = list(data.keys())
    for i, label in enumerate(labels):
        entry = data[label]
        val = entry["val"]
        color = entry["color"]
        bar_y = y - i * bar_h
        # Normalize to max value in this chart
        max_val = max(d["val"] for d in data.values())
        bar_w = (val / max_val) * w * 0.7 if max_val > 0 else 0

        ax.barh(bar_y, bar_w, height=bar_h * 0.7, left=x, color=color, alpha=0.7)
        ax.text(x - 0.02, bar_y, label, ha="right", va="center",
                fontsize=6, color=COLORS["text"], fontfamily="monospace")

        val_str = f"{val:.3f}" if val < 1 else f"{val:.0f}"
        if "marker" in entry:
            val_str += " ⚠"
        ax.text(x + bar_w + 0.01, bar_y, val_str, ha="left", va="center",
                fontsize=6, color=color, fontfamily="monospace")


def main():
    apply_dark_style()

    n_exp = len(EXPERIMENTS)
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, n_exp + 0.3)
    ax.invert_yaxis()
    ax.axis("off")

    fig.suptitle("Experiment Progression — Qwen3-0.6B Activation Steering",
                 fontsize=14, fontweight="bold", color=COLORS["text"], y=0.97)

    # Column positions
    col_num = 0.02
    col_name = 0.08
    col_metric = 0.35
    col_insight = 0.72
    col_forward = 0.95

    # Header
    headers = [
        (col_num, "Phase"),
        (col_name, "Experiment"),
        (col_metric, "Key Metric"),
        (col_insight, "Insight"),
        (col_forward, "Forward"),
    ]
    for hx, label in headers:
        ax.text(hx, -0.3, label, ha="left", va="center",
                fontsize=10, fontweight="bold", color=COLORS["accent"],
                transform=ax.transData)

    # Draw connecting timeline line
    for i in range(n_exp - 1):
        exp = EXPERIMENTS[i]
        if exp["forward"]:
            color = COLORS["sweet_spot"] if exp["status"] == "done" else COLORS["grid"]
            ax.annotate("", xy=(col_forward, i + 0.8), xytext=(col_forward, i + 0.2),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.5))

    for idx, exp in enumerate(EXPERIMENTS):
        y = idx
        done = exp["status"] == "done"

        # Background row stripe
        if idx % 2 == 0:
            ax.axhspan(y - 0.45, y + 0.45, color=COLORS["grid"], alpha=0.08)

        # Phase number with status color
        if done:
            badge_color = COLORS["sweet_spot"]
            badge_icon = "✓"
        else:
            badge_color = COLORS["grid"]
            badge_icon = "○"

        badge = mpatches.FancyBboxPatch(
            (col_num - 0.01, y - 0.25), 0.045, 0.5,
            boxstyle="round,pad=0.05",
            facecolor=badge_color, alpha=0.2,
            edgecolor=badge_color, linewidth=1.5,
        )
        ax.add_patch(badge)
        ax.text(col_num + 0.012, y, f"{badge_icon} {exp['num']}",
                ha="center", va="center", fontsize=9,
                fontweight="bold", color=badge_color, fontfamily="monospace")

        # Experiment name
        name_color = COLORS["text"] if done else COLORS["grid"]
        ax.text(col_name, y - 0.1, exp["name"], ha="left", va="center",
                fontsize=10, fontweight="bold", color=name_color)
        ax.text(col_name, y + 0.15, exp["cprr"], ha="left", va="center",
                fontsize=7, color=COLORS["grid"], fontfamily="monospace")

        # Key metric — mini bars for done experiments, text for pending
        if exp["metric_data"] and done:
            draw_mini_bar(ax, col_metric, y - 0.15, 0.3, 0.6, exp["metric_data"])
        else:
            style = "italic" if not done else "normal"
            alpha = 0.5 if not done else 0.9
            ax.text(col_metric, y, exp["metric_label"], ha="left", va="center",
                    fontsize=8, color=COLORS["text"], alpha=alpha, style=style)

        # Insight
        insight_color = COLORS["text"] if done else COLORS["grid"]
        ax.text(col_insight, y, exp["insight"], ha="left", va="center",
                fontsize=8, color=insight_color, linespacing=1.4)

        # Forward arrow text
        if exp["forward"]:
            fwd_color = COLORS["accent"] if done else COLORS["grid"]
            ax.text(col_forward, y, exp["forward"], ha="left", va="center",
                    fontsize=7, color=fwd_color, fontfamily="monospace",
                    rotation=-90 if False else 0)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS["sweet_spot"], alpha=0.5, label="Completed"),
        mpatches.Patch(facecolor=COLORS["grid"], alpha=0.5, label="Pending"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              bbox_to_anchor=(1.0, 1.05), fontsize=8)

    # Summary box at bottom
    summary_text = (
        "Progression: Layer mapping → Direction analysis → Injection test → "
        "Alpha calibration → Multi-layer → Semantic geometry\n"
        "Each experiment either confirms or refutes conjectures, "
        "with refutations informing the next experiment design."
    )
    ax.text(0.5, n_exp + 0.1, summary_text, ha="center", va="top",
            fontsize=8, color=COLORS["text"], alpha=0.6, style="italic",
            transform=ax.transData)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
