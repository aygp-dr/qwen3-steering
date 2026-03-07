#!/usr/bin/env python3
"""
CPRR Board Visualization: conjecture status + evidence timeline.

Reads .cprr/conjectures.json and renders:
  - Left panel: status badges for all conjectures (confirmed/refuted/testing/open)
  - Right panel: key metrics from confirmed experiments

Usage:
    python viz/cprr_board.py
"""
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from shared_style import apply_dark_style, COLORS

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "cprr_board.png")
CPRR_PATH = os.path.join(os.path.dirname(__file__), "..", ".cprr", "conjectures.json")

STATUS_COLORS = {
    "confirmed": "#4CAF50",
    "refuted":   "#F44336",
    "testing":   "#FF9800",
    "open":      "#2196F3",
}

STATUS_ICONS = {
    "confirmed": "\u2713",  # checkmark
    "refuted":   "\u2717",  # X
    "testing":   "\u25CB",  # circle
    "open":      "\u2022",  # bullet
}


def load_cprr():
    with open(CPRR_PATH) as f:
        return json.load(f)


def main():
    apply_dark_style()
    cprr = load_cprr()
    conjectures = cprr.get("conjectures", [])

    num_conjectures = len(conjectures)
    fig_height = max(8, num_conjectures * 0.65 + 3)

    fig, (ax_board, ax_summary) = plt.subplots(
        1, 2, figsize=(20, fig_height),
        gridspec_kw={"width_ratios": [3, 2]}
    )
    fig.suptitle("CPRR Conjecture Board — Qwen3-0.6B Activation Steering",
                 fontsize=14, fontweight="bold", color=COLORS["text"], y=0.98)

    # ── Left panel: conjecture list ──────────────────────────────────────────
    ax_board.set_xlim(0, 10)
    ax_board.set_ylim(-0.5, num_conjectures + 0.5)
    ax_board.invert_yaxis()
    ax_board.axis("off")
    ax_board.set_title("Conjectures", fontsize=12, color=COLORS["text"], pad=10)

    for idx, conj in enumerate(conjectures):
        cid = conj.get("id", idx)
        status = conj.get("status", "open")
        title = conj.get("title", "Untitled")
        tags = conj.get("tags", [])
        evidence_count = len(conj.get("evidence", []))

        color = STATUS_COLORS.get(status, COLORS["text"])
        icon = STATUS_ICONS.get(status, "?")

        y = idx

        # Status badge
        badge = mpatches.FancyBboxPatch(
            (0.1, y - 0.35), 1.2, 0.7,
            boxstyle="round,pad=0.08",
            facecolor=color, alpha=0.25,
            edgecolor=color, linewidth=1.5,
        )
        ax_board.add_patch(badge)
        ax_board.text(0.7, y, f"C-{cid}", ha="center", va="center",
                      fontsize=9, fontweight="bold", color=color, fontfamily="monospace")

        # Status text
        ax_board.text(1.5, y - 0.15, status.upper(), ha="left", va="center",
                      fontsize=7, color=color, fontweight="bold", fontfamily="monospace")

        # Title (wrapped)
        wrapped_title = textwrap.shorten(title, width=55, placeholder="...")
        ax_board.text(1.5, y + 0.15, wrapped_title, ha="left", va="center",
                      fontsize=8, color=COLORS["text"])

        # Evidence count
        if evidence_count > 0:
            ax_board.text(9.0, y, f"{evidence_count} ev", ha="center", va="center",
                          fontsize=7, color=COLORS["text"], alpha=0.6, fontfamily="monospace")

        # Tags
        tag_str = " ".join(f"#{t}" for t in tags[:3])
        ax_board.text(9.5, y, tag_str, ha="right", va="center",
                      fontsize=6, color=COLORS["grid"], fontfamily="monospace")

    # ── Right panel: summary stats ───────────────────────────────────────────
    ax_summary.axis("off")
    ax_summary.set_title("Summary", fontsize=12, color=COLORS["text"], pad=10)

    # Count by status
    status_counts = {}
    for conj in conjectures:
        s = conj.get("status", "open")
        status_counts[s] = status_counts.get(s, 0) + 1

    # Pie chart in top half
    ax_pie = fig.add_axes([0.68, 0.55, 0.25, 0.35])
    ax_pie.set_facecolor(COLORS["bg_dark"])

    statuses = ["confirmed", "refuted", "testing", "open"]
    pie_values = [status_counts.get(s, 0) for s in statuses]
    pie_colors = [STATUS_COLORS[s] for s in statuses]
    pie_labels = [f"{s}\n({status_counts.get(s, 0)})" for s in statuses]

    # Only include non-zero
    non_zero = [(v, c, l) for v, c, l in zip(pie_values, pie_colors, pie_labels) if v > 0]
    if non_zero:
        vals, cols, labs = zip(*non_zero)
        wedges, texts, autotexts = ax_pie.pie(
            vals, colors=cols, labels=labs,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 8, "color": COLORS["text"]},
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color(COLORS["bg_dark"])
            at.set_fontweight("bold")

    # Key findings in bottom half
    findings = [
        ("Sweet spot", "L12-17 (stability>0.95, SNR 3-6%)", COLORS["sweet_spot"]),
        ("Best layer", "L12 (stability=0.962)", COLORS["terse"]),
        ("Alpha overshoot", "a=2.0 is 3x prompt-level", COLORS["collapse"]),
        ("Norm != steerability", "Peak norm L27 -> collapse", STATUS_COLORS["refuted"]),
        ("Topic drift", "L0-7 destroys semantics", STATUS_COLORS["confirmed"]),
        ("Formatting", "L18-22: structure not style", COLORS["formal"]),
    ]

    y_start = 0.42
    ax_summary.text(0.05, y_start + 0.05, "Key Findings", fontsize=11,
                    color=COLORS["accent"], fontweight="bold",
                    transform=ax_summary.transAxes)

    for i, (label, detail, color) in enumerate(findings):
        y = y_start - i * 0.065
        ax_summary.text(0.08, y, f"\u25A0", fontsize=10, color=color,
                        transform=ax_summary.transAxes, va="center")
        ax_summary.text(0.13, y, label, fontsize=9, color=color,
                        fontweight="bold", transform=ax_summary.transAxes, va="center")
        ax_summary.text(0.13, y - 0.028, detail, fontsize=7, color=COLORS["text"],
                        alpha=0.8, transform=ax_summary.transAxes, va="center")

    # Experiment progress tracker
    experiments = [
        ("01 Layer Scorecard", "done", "C-1..5"),
        ("02 Verbosity Direction", "done", "C-7,8"),
        ("03 Bimodal Injection", "done", "C-9"),
        ("04 Alpha Parity Sweep", "done", "C-14"),
        ("05 Multi-Layer", "done", "C-6"),
        ("06 WordNet Geometry", "done", "C-10..13"),
        ("07 Style Showcase", "pending", "—"),
    ]

    y_exp = 0.02
    ax_summary.text(0.05, y_exp + 0.02, "Experiment Progress", fontsize=11,
                    color=COLORS["accent"], fontweight="bold",
                    transform=ax_summary.transAxes)

    for i, (name, status, cprr_ids) in enumerate(experiments):
        y = y_exp - i * 0.045
        done = status == "done"
        color = COLORS["sweet_spot"] if done else COLORS["grid"]
        icon = "\u2713" if done else "\u25CB"
        ax_summary.text(0.08, y, icon, fontsize=9, color=color,
                        transform=ax_summary.transAxes, va="center")
        ax_summary.text(0.13, y, name, fontsize=8, color=color,
                        fontweight="bold" if done else "normal",
                        transform=ax_summary.transAxes, va="center")
        ax_summary.text(0.75, y, cprr_ids, fontsize=7, color=COLORS["text"],
                        alpha=0.6, fontfamily="monospace",
                        transform=ax_summary.transAxes, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
