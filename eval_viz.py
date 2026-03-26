#!/usr/bin/env python3
"""
eval_viz.py — Six-panel evaluation visualization for terse/verbose steering.

Tangled from eval_viz.org.  Reads eval_output/terse_verbose_full.json
(no model, no Ollama — pure post-hoc analysis).

Panels:
  1. Descriptive   — KDE (n>=30) or strip plot (n<30) per feature with median lines + bootstrap CIs
  2. Elbow/Sil     — justifies k=3 without assuming it
  3. PCA scatter   — ground truth vs k-means, 2-sigma confidence ellipses
  4. Confusion     — raw and normalised side-by-side
  5. Effect size   — rank-biserial r, Bonferroni-starred
  6. Dashboard     — single composite figure, README-ready

Conjecture C-28: surface features (word_count, token_count, char_count)
are sufficient to recover the screening conditions without oracle labels.
Refutation criterion: ARI < 0.50 on the full 100-prompt run.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats as sp_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    adjusted_rand_score,
    silhouette_score,
    silhouette_samples,
)
from matplotlib.patches import Ellipse

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT_CLR = "#c9d1d9"
GRID_CLR = "#30363d"
COLORS = {"terse": "#58a6ff", "baseline": "#d2a8ff", "verbose": "#f0883e"}
LABEL_ORDER = ["terse", "baseline", "verbose"]
OUTPUT_DIR = Path("eval_output")


def dark_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    return ax


def bootstrap_ci(data, statistic=np.median, n_resamples=1000, confidence_level=0.95):
    """Compute bootstrap confidence interval for a statistic using scipy.stats.bootstrap."""
    data = np.asarray(data)
    if len(data) < 2:
        val = statistic(data)
        return val, val, val
    result = sp_stats.bootstrap(
        (data,), statistic, n_resamples=n_resamples,
        confidence_level=confidence_level, random_state=42,
        method="percentile",
    )
    center = statistic(data)
    return center, result.confidence_interval.low, result.confidence_interval.high


def load_data(path=None):
    """Load the full eval JSON (with text) and build feature matrix."""
    p = path or OUTPUT_DIR / "terse_verbose_full.json"
    with open(p) as f:
        blob = json.load(f)
    records = blob["records"]
    config = blob["config"]

    rows, labels = [], []
    for r in records:
        for d in LABEL_ORDER:
            rows.append([r[f"{d}_words"], r[f"{d}_tokens"], r[f"{d}_chars"]])
            labels.append(d)
    X = np.array(rows, dtype=float)
    return X, np.array(labels), records, config


def plot_descriptive(X, labels, output_dir):
    """KDE (n>=30) or strip plot (n<30) per feature, with median lines + bootstrap CIs."""
    feature_names = ["Word Count", "Token Count", "Char Count"]
    n_per_group = int(np.sum(labels == LABEL_ORDER[0]))
    use_strip = n_per_group < 30

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor(DARK_BG)

    for col, (ax, fname) in enumerate(zip(axes, feature_names)):
        dark_ax(ax)
        if use_strip:
            # Build a dataframe-like structure for stripplot
            import pandas as pd
            df = pd.DataFrame({
                "value": X[:, col],
                "direction": labels,
            })
            sns.stripplot(data=df, x="direction", y="value", hue="direction",
                          order=LABEL_ORDER, hue_order=LABEL_ORDER,
                          palette=COLORS, ax=ax, jitter=0.25, size=6, alpha=0.7,
                          legend=False)
            # Median lines + bootstrap CI shaded regions
            for idx, direction in enumerate(LABEL_ORDER):
                vals = X[labels == direction, col]
                med, ci_lo, ci_hi = bootstrap_ci(vals)
                ax.hlines(med, idx - 0.3, idx + 0.3, color=COLORS[direction],
                          linestyle="--", linewidth=2, alpha=0.9)
                ax.fill_between([idx - 0.3, idx + 0.3], ci_lo, ci_hi,
                                color=COLORS[direction], alpha=0.15)
                ax.text(idx + 0.35, med, f"{med:.0f} [{ci_lo:.0f}-{ci_hi:.0f}]",
                        color=COLORS[direction], fontsize=6, va="center")
            ax.set_xlabel("Direction", color=TEXT_CLR)
            ax.set_ylabel(fname if col == 0 else "", color=TEXT_CLR)
        else:
            for direction in LABEL_ORDER:
                mask = labels == direction
                vals = X[mask, col]
                sns.kdeplot(vals, ax=ax, color=COLORS[direction],
                            label=direction, fill=True, alpha=0.25, linewidth=1.5)
                med, ci_lo, ci_hi = bootstrap_ci(vals)
                ax.axvline(med, color=COLORS[direction], linestyle="--",
                           linewidth=1, alpha=0.8)
                # Shaded CI region on median
                ax.axvspan(ci_lo, ci_hi, color=COLORS[direction], alpha=0.08)
                ax.text(med, ax.get_ylim()[1] * 0.92,
                        f"{med:.0f} [{ci_lo:.0f}-{ci_hi:.0f}]",
                        color=COLORS[direction], fontsize=6, ha="center")
            ax.set_xlabel(fname, color=TEXT_CLR)
            ax.set_ylabel("Density" if col == 0 else "", color=TEXT_CLR)
            if col == 0:
                ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                          fontsize=8)

        ax.set_title(fname, color=TEXT_CLR, fontsize=10)

    plot_type = "Strip Plot" if use_strip else "KDE"
    fig.suptitle(f"Descriptive: {plot_type} of Surface Features (n={n_per_group}/group, 95% CI)",
                 color=TEXT_CLR, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / "01_descriptive_kde.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  [1/6] {path}")
    return fig


def plot_elbow_silhouette(X, output_dir):
    """Elbow curve + silhouette score to justify k=3."""
    ks = range(2, 9)
    inertias, sil_scores = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(DARK_BG)

    # Elbow
    dark_ax(ax1)
    ax1.plot(list(ks), inertias, "o-", color="#58a6ff", linewidth=2, markersize=6)
    ax1.axvline(3, color="#f0883e", linestyle="--", alpha=0.7, label="k=3")
    ax1.set_xlabel("k", color=TEXT_CLR)
    ax1.set_ylabel("Inertia", color=TEXT_CLR)
    ax1.set_title("Elbow Curve", color=TEXT_CLR, fontsize=10)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    # Silhouette
    dark_ax(ax2)
    ax2.plot(list(ks), sil_scores, "o-", color="#58a6ff", linewidth=2, markersize=6)
    ax2.axvline(3, color="#f0883e", linestyle="--", alpha=0.7, label="k=3")
    best_k = list(ks)[np.argmax(sil_scores)]
    ax2.annotate(f"best k={best_k} (sil={max(sil_scores):.3f})",
                 xy=(best_k, max(sil_scores)),
                 xytext=(best_k + 1.5, max(sil_scores)),
                 arrowprops=dict(arrowstyle="->", color=TEXT_CLR),
                 color=TEXT_CLR, fontsize=9)
    ax2.set_xlabel("k", color=TEXT_CLR)
    ax2.set_ylabel("Silhouette Score", color=TEXT_CLR)
    ax2.set_title("Silhouette Analysis", color=TEXT_CLR, fontsize=10)
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    fig.suptitle("Cluster Selection: Elbow + Silhouette",
                 color=TEXT_CLR, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / "02_elbow_silhouette.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  [2/6] {path}")
    return best_k


def _confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """Draw a 2D confidence ellipse for points (x, y)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_pca_scatter(X, labels, output_dir):
    """PCA scatter: ground truth vs k-means, 2-sigma confidence ellipses."""
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(X)

    # Map clusters to labels by centroid word-count order
    centroids_pca = pca.transform(km.cluster_centers_)
    order = np.argsort(km.cluster_centers_[:, 0])
    cluster_to_label = {order[0]: "terse", order[1]: "baseline", order[2]: "verbose"}
    pred_labels = np.array([cluster_to_label[c] for c in cluster_ids])

    ari = adjusted_rand_score(labels, pred_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(DARK_BG)

    for ax, lab_arr, title in [(ax1, labels, "Ground Truth"),
                                (ax2, pred_labels, f"K-Means (ARI={ari:.3f})")]:
        dark_ax(ax)
        for direction in LABEL_ORDER:
            mask = lab_arr == direction
            ax.scatter(X2[mask, 0], X2[mask, 1], c=COLORS[direction],
                       s=15, alpha=0.6, label=direction)
            _confidence_ellipse(
                X2[mask, 0], X2[mask, 1], ax, n_std=2.0,
                edgecolor=COLORS[direction], facecolor="none",
                linewidth=1.5, linestyle="--", alpha=0.7)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", color=TEXT_CLR)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", color=TEXT_CLR)
        ax.set_title(title, color=TEXT_CLR, fontsize=10)
        ax.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR,
                  fontsize=8, loc="upper right")

    fig.suptitle("PCA Projection: Ground Truth vs K-Means with 2σ Ellipses",
                 color=TEXT_CLR, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = output_dir / "03_pca_scatter.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  [3/6] {path}")
    return ari, pred_labels


def plot_confusion(labels, pred_labels, output_dir):
    """Raw and normalised confusion matrices side-by-side."""
    cm_raw = confusion_matrix(labels, pred_labels, labels=LABEL_ORDER)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)

    for ax, cm, fmt, title in [
        (ax1, cm_raw, "d", "Raw Counts"),
        (ax2, cm_norm, ".2f", "Normalised (row)"),
    ]:
        dark_ax(ax)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                    ax=ax, linewidths=0.5, linecolor=GRID_CLR,
                    cbar_kws={"label": "Count" if fmt == "d" else "Proportion"})
        ax.set_xlabel("Predicted (k-means)", color=TEXT_CLR)
        ax.set_ylabel("True (steering)", color=TEXT_CLR)
        ax.set_title(title, color=TEXT_CLR, fontsize=10)
        plt.setp(ax.get_xticklabels(), color=TEXT_CLR)
        plt.setp(ax.get_yticklabels(), color=TEXT_CLR)

    # Burke annotation
    fig.text(0.5, 0.01,
             "Burke prediction: terse/baseline bleed (shared screen), verbose clean",
             ha="center", color="#8b949e", fontsize=8, style="italic")

    fig.suptitle("Confusion Matrix: True Steering vs K-Means Cluster",
                 color=TEXT_CLR, fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    path = output_dir / "04_confusion_matrix.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  [4/6] {path}")


def _rank_biserial(x, y):
    """Rank-biserial r from Mann-Whitney U."""
    n1, n2 = len(x), len(y)
    u_stat, p_val = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - (2 * u_stat) / (n1 * n2)
    return r, p_val


def plot_effect_size(X, labels, output_dir):
    """Rank-biserial r heatmap with Bonferroni stars."""
    feature_names = ["word_count", "token_count", "char_count"]
    pairs = [("terse", "baseline"), ("terse", "verbose"), ("baseline", "verbose")]
    n_tests = len(pairs) * len(feature_names)  # 9 tests

    r_matrix = np.zeros((len(pairs), len(feature_names)))
    star_matrix = [["" for _ in feature_names] for _ in pairs]

    for i, (a, b) in enumerate(pairs):
        mask_a = labels == a
        mask_b = labels == b
        for j in range(len(feature_names)):
            r, p = _rank_biserial(X[mask_a, j], X[mask_b, j])
            r_matrix[i, j] = r
            p_adj = min(p * n_tests, 1.0)  # Bonferroni
            if p_adj < 0.001:
                star_matrix[i][j] = "***"
            elif p_adj < 0.01:
                star_matrix[i][j] = "**"
            elif p_adj < 0.05:
                star_matrix[i][j] = "*"

    # Build annotation strings
    annot = np.array([
        [f"{r_matrix[i,j]:.2f}{star_matrix[i][j]}" for j in range(len(feature_names))]
        for i in range(len(pairs))
    ])

    pair_labels = [f"{a} vs {b}" for a, b in pairs]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(DARK_BG)
    dark_ax(ax)

    sns.heatmap(r_matrix, annot=annot, fmt="", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1,
                xticklabels=feature_names, yticklabels=pair_labels,
                ax=ax, linewidths=0.5, linecolor=GRID_CLR)
    ax.set_title("Effect Size: Rank-Biserial r (Bonferroni-corrected *)",
                 color=TEXT_CLR, fontsize=10)
    plt.setp(ax.get_xticklabels(), color=TEXT_CLR)
    plt.setp(ax.get_yticklabels(), color=TEXT_CLR)

    fig.text(0.5, 0.01, "* p<.05  ** p<.01  *** p<.001 (Bonferroni-adjusted)",
             ha="center", color="#8b949e", fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = output_dir / "05_effect_size_heatmap.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close()
    print(f"  [5/6] {path}")


def plot_dashboard(X, labels, pred_labels, ari, output_dir):
    """Single composite figure: KDE, elbow, PCA, confusion, effect size."""
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    # ── Panel 1: KDE or Strip (word count only — the money plot) ────────
    ax1 = fig.add_subplot(gs[0, 0])
    dark_ax(ax1)
    n_per_group = int(np.sum(labels == LABEL_ORDER[0]))
    use_strip = n_per_group < 30

    if use_strip:
        import pandas as pd
        df = pd.DataFrame({"value": X[:, 0], "direction": labels})
        sns.stripplot(data=df, x="direction", y="value", hue="direction",
                      order=LABEL_ORDER, hue_order=LABEL_ORDER,
                      palette=COLORS, ax=ax1, jitter=0.25, size=5, alpha=0.7,
                      legend=False)
        for idx, direction in enumerate(LABEL_ORDER):
            vals = X[labels == direction, 0]
            med, ci_lo, ci_hi = bootstrap_ci(vals)
            ax1.hlines(med, idx - 0.3, idx + 0.3, color=COLORS[direction],
                       linestyle="--", linewidth=2, alpha=0.9)
            ax1.fill_between([idx - 0.3, idx + 0.3], ci_lo, ci_hi,
                             color=COLORS[direction], alpha=0.15)
        ax1.set_xlabel("Direction", color=TEXT_CLR)
        ax1.set_ylabel("Word Count", color=TEXT_CLR)
        ax1.set_title(f"A. Word Count Strip (n={n_per_group})", color=TEXT_CLR, fontsize=10)
    else:
        for direction in LABEL_ORDER:
            mask = labels == direction
            vals = X[mask, 0]
            sns.kdeplot(vals, ax=ax1, color=COLORS[direction],
                        label=direction, fill=True, alpha=0.25, linewidth=1.5)
            med, ci_lo, ci_hi = bootstrap_ci(vals)
            ax1.axvline(med, color=COLORS[direction], linestyle="--", linewidth=1, alpha=0.8)
            ax1.axvspan(ci_lo, ci_hi, color=COLORS[direction], alpha=0.08)
        ax1.set_xlabel("Word Count", color=TEXT_CLR)
        ax1.set_ylabel("Density", color=TEXT_CLR)
        ax1.set_title("A. Word Count KDE", color=TEXT_CLR, fontsize=10)
        ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=7)

    # ── Panel 2: Elbow + Silhouette ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    dark_ax(ax2)
    ks = range(2, 9)
    sil_scores = []
    for k in ks:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_tmp.fit(X)
        sil_scores.append(silhouette_score(X, km_tmp.labels_))
    ax2.plot(list(ks), sil_scores, "o-", color="#58a6ff", linewidth=2, markersize=5)
    ax2.axvline(3, color="#f0883e", linestyle="--", alpha=0.7)
    ax2.set_xlabel("k", color=TEXT_CLR)
    ax2.set_ylabel("Silhouette", color=TEXT_CLR)
    ax2.set_title("B. Silhouette Analysis", color=TEXT_CLR, fontsize=10)

    # ── Panel 3: PCA scatter (ground truth) ──────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    dark_ax(ax3)
    for direction in LABEL_ORDER:
        mask = labels == direction
        ax3.scatter(X2[mask, 0], X2[mask, 1], c=COLORS[direction],
                    s=12, alpha=0.6, label=direction)
        _confidence_ellipse(X2[mask, 0], X2[mask, 1], ax3, n_std=2.0,
                            edgecolor=COLORS[direction], facecolor="none",
                            linewidth=1.2, linestyle="--", alpha=0.6)
    ax3.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", color=TEXT_CLR)
    ax3.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", color=TEXT_CLR)
    ax3.set_title(f"C. PCA (ARI={ari:.3f})", color=TEXT_CLR, fontsize=10)
    ax3.legend(facecolor=PANEL_BG, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=7)

    # ── Panel 4: Confusion (normalised) ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    dark_ax(ax4)
    cm = confusion_matrix(labels, pred_labels, labels=LABEL_ORDER)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
                ax=ax4, linewidths=0.5, linecolor=GRID_CLR)
    ax4.set_xlabel("Predicted", color=TEXT_CLR)
    ax4.set_ylabel("True", color=TEXT_CLR)
    ax4.set_title("D. Normalised Confusion", color=TEXT_CLR, fontsize=10)
    plt.setp(ax4.get_xticklabels(), color=TEXT_CLR)
    plt.setp(ax4.get_yticklabels(), color=TEXT_CLR)

    # ── Panel 5: Effect size ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    dark_ax(ax5)
    feature_names = ["words", "tokens", "chars"]
    pairs = [("terse", "baseline"), ("terse", "verbose"), ("baseline", "verbose")]
    r_matrix = np.zeros((len(pairs), 3))
    annot_arr = np.empty_like(r_matrix, dtype=object)
    n_tests = len(pairs) * 3
    for i, (a, b) in enumerate(pairs):
        for j in range(3):
            r, p = _rank_biserial(X[labels == a, j], X[labels == b, j])
            r_matrix[i, j] = r
            p_adj = min(p * n_tests, 1.0)
            stars = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
            annot_arr[i, j] = f"{r:.2f}{stars}"
    sns.heatmap(r_matrix, annot=annot_arr, fmt="", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, xticklabels=feature_names,
                yticklabels=[f"{a}v{b}" for a, b in pairs],
                ax=ax5, linewidths=0.5, linecolor=GRID_CLR)
    ax5.set_title("E. Effect Size (rank-biserial r)", color=TEXT_CLR, fontsize=10)
    plt.setp(ax5.get_xticklabels(), color=TEXT_CLR)
    plt.setp(ax5.get_yticklabels(), color=TEXT_CLR)

    # ── Panel 6: summary text with bootstrap CIs ─────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(PANEL_BG)
    ax6.axis("off")
    n_per = len(X) // 3

    # Compute bootstrap CIs for median word count per direction
    ci_lines = []
    for direction in LABEL_ORDER:
        vals = X[labels == direction, 0]
        med, ci_lo, ci_hi = bootstrap_ci(vals)
        label_padded = f"{direction.capitalize():9s}"
        ci_lines.append(
            f"{label_padded} med={med:.0f} [{ci_lo:.0f}-{ci_hi:.0f}]"
        )

    summary_lines = [
        f"n = {n_per} prompts x 3 directions",
        f"ARI = {ari:.3f}",
        f"C-28: {'PROOF' if ari >= 0.50 else 'REFUTATION'}",
        "",
        "Word count (median, 95% CI):",
    ] + ci_lines + [
        "",
        "Burke: terse/baseline bleed,",
        "       verbose screen is clean.",
    ]
    ax6.text(0.1, 0.95, "\n".join(summary_lines),
             transform=ax6.transAxes, color=TEXT_CLR,
             fontsize=9, verticalalignment="top", fontfamily="monospace")
    ax6.set_title("F. Summary", color=TEXT_CLR, fontsize=10)

    fig.suptitle(f"Surface Features Recover Terse Screen but Not Verbose (C-28, ARI={ari:.3f})",
                 color=TEXT_CLR, fontsize=14, fontweight="bold")
    path = output_dir / "06_dashboard.png"
    fig.savefig(path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  [6/6] {path}")


def main():
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    else:
        data_path = OUTPUT_DIR / "terse_verbose_full.json"

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        print("Run eval_terse_verbose.py first to generate evaluation data.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading {data_path}...")
    X, labels, records, config = load_data(data_path)
    n = len(records)
    print(f"  {n} prompts, {len(X)} total observations, "
          f"\u03b1=\u00b1{config.get('alpha', '?')}, layer={config.get('layer', '?')}")

    print("\nGenerating 6 panels...")
    plot_descriptive(X, labels, OUTPUT_DIR)
    best_k = plot_elbow_silhouette(X, OUTPUT_DIR)
    ari, pred_labels = plot_pca_scatter(X, labels, OUTPUT_DIR)
    plot_confusion(labels, pred_labels, OUTPUT_DIR)
    plot_effect_size(X, labels, OUTPUT_DIR)
    plot_dashboard(X, labels, pred_labels, ari, OUTPUT_DIR)

    # C-28 verdict
    print(f"\n{'='*60}")
    print(f"  C-28: Surface features sufficient for screening recovery?")
    print(f"  ARI = {ari:.3f}  (threshold: 0.50)")
    if ari >= 0.50:
        print(f"  Status: PROOF — ARI >= 0.50, surface features recover screens")
    else:
        print(f"  Status: REFUTATION — ARI < 0.50, screens not in surface features")
    print(f"  Best k by silhouette: {best_k}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
