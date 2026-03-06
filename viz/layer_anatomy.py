#!/usr/bin/env python3
"""
Visualization 2: Layer Anatomy Diagram

Side-view of all 28 transformer layers showing:
- Residual stream norm as bar width (455 -> 810, growing)
- Steering vector injection points (layer 12/15) highlighted
- Raw steering vector magnitude overlaid
- SNR percentage at each layer
- Color gradient from blue (early/semantic) to red (late/output)
- Zone annotations: topic drift, sweet spot, output layers

Uses actual empirical data from diagnose_steering.py runs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from shared_style import (
    apply_dark_style, COLORS, layer_color,
    LAYERS_SAMPLED, RESIDUAL_NORMS, RAW_VEC_NORMS,
    NUM_LAYERS, D_MODEL, BEST_LAYER, SWEET_SPOT_RANGE
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "layer_anatomy.png")


def interpolate_to_all_layers(sampled_layers, sampled_values, total=28):
    """Interpolate sampled measurements to all 28 layers."""
    layers = np.array(sampled_layers)
    values = np.array(sampled_values)
    f = interp1d(layers, values, kind="cubic", fill_value="extrapolate")
    all_layers = np.arange(total)
    result = f(all_layers)
    # Clamp to positive
    result = np.maximum(result, 0)
    return all_layers, result


def main():
    apply_dark_style()

    fig, axes = plt.subplots(1, 3, figsize=(20, 12), gridspec_kw={"width_ratios": [4, 2, 2]})
    ax_main, ax_snr, ax_info = axes

    all_layers = np.arange(NUM_LAYERS)

    # ── Interpolate residual norms and vec norms to all 28 layers ────────────
    _, res_norms_all = interpolate_to_all_layers(
        LAYERS_SAMPLED, [RESIDUAL_NORMS[l] for l in LAYERS_SAMPLED]
    )
    _, vec_norms_all = interpolate_to_all_layers(
        LAYERS_SAMPLED, [RAW_VEC_NORMS[l] for l in LAYERS_SAMPLED]
    )

    # ── Main panel: horizontal bars for each layer ───────────────────────────
    bar_height = 0.7
    max_norm = max(res_norms_all)

    for i, layer in enumerate(all_layers):
        # Color by layer position
        color = layer_color(layer, NUM_LAYERS)

        # Residual stream norm bar
        width = res_norms_all[layer]
        bar = ax_main.barh(
            layer, width, height=bar_height,
            color=color, alpha=0.7, edgecolor="white", linewidth=0.3, zorder=2
        )

        # Overlay: raw steering vector magnitude (smaller bar on top)
        vec_width = vec_norms_all[layer] * 5  # Scale up for visibility
        ax_main.barh(
            layer, vec_width, height=bar_height * 0.4,
            color=COLORS["accent"], alpha=0.85, edgecolor="none", zorder=3
        )

        # Norm value label
        ax_main.text(
            width + 8, layer, f"{width:.0f}",
            va="center", ha="left", fontsize=6, color=color, alpha=0.8
        )

    # ── Highlight injection layers ───────────────────────────────────────────
    for inject_layer, marker_label, marker_color in [
        (BEST_LAYER, "BEST (L12)", COLORS["sweet_spot"]),
        (15, "L15 (diagnosed)", COLORS["terse"]),
    ]:
        ax_main.barh(
            inject_layer, res_norms_all[inject_layer], height=bar_height,
            color=marker_color, alpha=0.4, edgecolor=marker_color,
            linewidth=2.5, linestyle="-", zorder=4
        )
        ax_main.annotate(
            marker_label,
            xy=(res_norms_all[inject_layer], inject_layer),
            xytext=(res_norms_all[inject_layer] + 80, inject_layer),
            fontsize=8, fontweight="bold", color=marker_color,
            arrowprops=dict(arrowstyle="->", color=marker_color, lw=1.5),
            va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", fc=COLORS["bg_dark"], ec=marker_color, alpha=0.8)
        )

    # ── Zone annotations (background shading) ────────────────────────────────
    zone_params = [
        ((0, 7), "Topic Drift Zone\n(early semantic)", COLORS["layer_early"], 0.06),
        ((8, 11), "Transition", COLORS["layer_mid"], 0.04),
        ((12, 18), "Sweet Spot\n(best steering)", COLORS["sweet_spot"], 0.08),
        ((19, 23), "Late Processing", COLORS["layer_late"], 0.04),
        ((24, 27), "Output Layers\n(near logits)", COLORS["collapse"], 0.06),
    ]
    for (start, end), label, color, alpha in zone_params:
        ax_main.axhspan(start - 0.4, end + 0.4, color=color, alpha=alpha, zorder=0)
        mid = (start + end) / 2
        ax_main.text(
            -30, mid, label, fontsize=7, color=color, alpha=0.9,
            ha="right", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc=COLORS["bg_dark"], ec=color, alpha=0.5)
        )

    ax_main.set_xlim(-5, max_norm + 180)
    ax_main.set_ylim(-0.8, NUM_LAYERS - 0.2)
    ax_main.set_ylabel("Layer Index", fontsize=11)
    ax_main.set_xlabel("Residual Stream Norm (L2)", fontsize=11)
    ax_main.set_title("Residual Stream & Steering Vector Anatomy", fontsize=13, pad=10)
    ax_main.set_yticks(all_layers)
    ax_main.set_yticklabels([str(l) for l in all_layers], fontsize=7)
    ax_main.invert_yaxis()
    ax_main.grid(True, axis="x", alpha=0.15)

    # Legend for main panel
    main_legend = [
        mpatches.Patch(color=layer_color(0, NUM_LAYERS), alpha=0.7, label="Residual norm (early)"),
        mpatches.Patch(color=layer_color(27, NUM_LAYERS), alpha=0.7, label="Residual norm (late)"),
        mpatches.Patch(color=COLORS["accent"], alpha=0.85, label="Steering vec magnitude (5x scale)"),
        mpatches.Patch(color=COLORS["sweet_spot"], alpha=0.4, label="Best injection layer"),
    ]
    ax_main.legend(handles=main_legend, loc="lower right", fontsize=7)

    # ── SNR panel ────────────────────────────────────────────────────────────
    snr_values = vec_norms_all / res_norms_all * 100  # At alpha=1.0

    for i, layer in enumerate(all_layers):
        color = layer_color(layer, NUM_LAYERS)
        snr = snr_values[layer]
        ax_snr.barh(layer, snr, height=bar_height, color=color, alpha=0.7,
                     edgecolor="white", linewidth=0.2, zorder=2)
        ax_snr.text(snr + 0.15, layer, f"{snr:.1f}%", va="center", ha="left",
                    fontsize=6, color=color, alpha=0.8)

    # Threshold lines
    ax_snr.axvline(x=4.0, color=COLORS["sweet_spot"], linestyle="--", alpha=0.5, linewidth=1)
    ax_snr.text(4.2, 1, "~4% = working\nsteering", fontsize=6, color=COLORS["sweet_spot"],
                va="top", alpha=0.8)
    ax_snr.axvline(x=1.0, color=COLORS["dead_zone"], linestyle=":", alpha=0.5, linewidth=1)
    ax_snr.text(1.2, 3, "<1% =\nno effect", fontsize=6, color=COLORS["dead_zone"],
                va="top", alpha=0.8)

    ax_snr.set_xlim(0, max(snr_values) + 2)
    ax_snr.set_ylim(-0.8, NUM_LAYERS - 0.2)
    ax_snr.set_xlabel("SNR % (vec_norm / res_norm)", fontsize=10)
    ax_snr.set_title("Signal-to-Noise Ratio\n(alpha=1.0, raw vec)", fontsize=11, pad=10)
    ax_snr.set_yticks(all_layers)
    ax_snr.set_yticklabels([str(l) for l in all_layers], fontsize=7)
    ax_snr.invert_yaxis()
    ax_snr.grid(True, axis="x", alpha=0.15)

    # ── Info panel: key numbers ──────────────────────────────────────────────
    ax_info.axis("off")
    info_lines = [
        ("Model", "Qwen3-0.6B"),
        ("Layers", f"{NUM_LAYERS}"),
        ("d_model", f"{D_MODEL}"),
        ("", ""),
        ("Best layer", f"{BEST_LAYER}"),
        ("  mutex baseline", "149 words"),
        ("  mutex steered", "16 words"),
        ("", ""),
        ("Layer 15 detail:", ""),
        ("  residual norm", f"{RESIDUAL_NORMS[15]:.0f}"),
        ("  raw vec norm", f"{RAW_VEC_NORMS[15]:.1f}"),
        ("  SNR @ a=1.0", f"{RAW_VEC_NORMS[15]/RESIDUAL_NORMS[15]*100:.1f}%"),
        ("  SNR @ a=2.0", f"{2*RAW_VEC_NORMS[15]/RESIDUAL_NORMS[15]*100:.1f}%"),
        ("", ""),
        ("Unit vec + a=0.20:", ""),
        ("  effective mag", "0.20"),
        ("  SNR", f"{0.20/RESIDUAL_NORMS[15]*100:.4f}%"),
        ("  result", "ZERO EFFECT"),
        ("", ""),
        ("Raw vec + a=2.0:", ""),
        ("  effective mag", f"{2*RAW_VEC_NORMS[15]:.1f}"),
        ("  SNR", f"{2*RAW_VEC_NORMS[15]/RESIDUAL_NORMS[15]*100:.1f}%"),
        ("  result", "WORKING"),
        ("", ""),
        ("Raw vec + a=3.0+:", ""),
        ("  result", "COLLAPSE"),
    ]

    y_start = 0.95
    for i, (key, val) in enumerate(info_lines):
        y = y_start - i * 0.034
        if key == "":
            continue
        if "ZERO" in val or "COLLAPSE" in val:
            val_color = COLORS["collapse"]
        elif "WORKING" in val:
            val_color = COLORS["sweet_spot"]
        else:
            val_color = COLORS["text"]

        ax_info.text(0.05, y, key, fontsize=7, fontfamily="monospace",
                     color=COLORS["text"], transform=ax_info.transAxes, va="top")
        ax_info.text(0.65, y, val, fontsize=7, fontfamily="monospace",
                     color=val_color, fontweight="bold",
                     transform=ax_info.transAxes, va="top")

    ax_info.set_title("Key Measurements", fontsize=11, pad=10, color=COLORS["accent"])

    # Add a box around the info panel
    rect = mpatches.FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02",
        facecolor=COLORS["bg_dark"], edgecolor=COLORS["grid"],
        linewidth=1.5, alpha=0.8, transform=ax_info.transAxes
    )
    ax_info.add_patch(rect)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
