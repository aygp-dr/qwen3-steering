"""
Shared color scheme and styling for all qwen3-steering visualizations.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Consistent color palette ─────────────────────────────────────────────────

COLORS = {
    # Style basins
    "terse":     "#2196F3",  # Blue
    "formal":    "#9C27B0",  # Purple
    "socratic":  "#FF9800",  # Orange
    "dry-wit":   "#4CAF50",  # Green

    # Special regions
    "cult_of_jason": "#E91E63",  # Pink/magenta (near terse)
    "baseline":      "#607D8B",  # Blue-gray
    "collapse":      "#F44336",  # Red
    "dead_zone":     "#BDBDBD",  # Light gray
    "sweet_spot":    "#8BC34A",  # Light green

    # Layer gradient endpoints
    "layer_early":  "#1565C0",  # Dark blue
    "layer_mid":    "#7B1FA2",  # Purple
    "layer_late":   "#C62828",  # Dark red

    # Lens radar
    "clean":         "#4CAF50",
    "leaked":        "#F44336",
    "well_steered":  "#2196F3",

    # Generic
    "bg_dark":    "#1A1A2E",
    "bg_light":   "#F5F5F5",
    "grid":       "#37374F",
    "text":       "#E0E0E0",
    "text_dark":  "#212121",
    "accent":     "#FFD700",
}

# ── Lens names for radar chart ───────────────────────────────────────────────

LENS_NAMES = [
    "makefile", "guile", "orgmode", "monetization",
    "sports", "religion", "politics", "ai_hype",
    "conspiracy", "scarcity_mindset", "therapy_speak", "cult_of_jason",
]

LENS_DISPLAY = {
    "makefile": "Makefile",
    "guile": "Guile/Scheme",
    "orgmode": "Org-mode",
    "monetization": "Monetization",
    "sports": "Sports",
    "religion": "Religion",
    "politics": "Politics",
    "ai_hype": "AI Hype",
    "conspiracy": "Conspiracy",
    "scarcity_mindset": "Scarcity",
    "therapy_speak": "Therapy-speak",
    "cult_of_jason": "Cult of Jason",
}

# ── Empirical data ───────────────────────────────────────────────────────────

LAYERS_SAMPLED = [5, 10, 14, 15, 16, 18, 20, 24]
RESIDUAL_NORMS = {5: 455, 10: 472, 14: 481, 15: 488, 16: 501, 18: 538, 20: 595, 24: 810}
RAW_VEC_NORMS = {5: 14.1, 10: 15.8, 14: 18.7, 15: 19.6, 16: 22.9, 18: 29.8, 20: 40.3, 24: 70.8}

NUM_LAYERS = 28
D_MODEL = 1024
BEST_LAYER = 12
SWEET_SPOT_RANGE = (12, 18)

# ── Plot styling ─────────────────────────────────────────────────────────────

def apply_dark_style():
    """Apply consistent dark theme across all plots."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg_dark"],
        "axes.facecolor": COLORS["bg_dark"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "font.size": 10,
        "font.family": "monospace",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.facecolor": "#2A2A4A",
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 8,
        "savefig.facecolor": COLORS["bg_dark"],
        "savefig.edgecolor": "none",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def layer_color(layer_idx, total=28):
    """Return a color interpolated from blue (early) to red (late)."""
    t = layer_idx / max(total - 1, 1)
    cmap = mpl.colormaps["coolwarm"]
    return cmap(t)


def snr_at_layer(layer, alpha=1.0, raw=True):
    """Signal-to-noise ratio at a given layer."""
    if layer not in RESIDUAL_NORMS:
        return None
    if raw:
        return alpha * RAW_VEC_NORMS[layer] / RESIDUAL_NORMS[layer] * 100
    else:
        return alpha / RESIDUAL_NORMS[layer] * 100
