"""
Generate a publication-quality SVG schematic of the ActAdd intervention
in Qwen3-0.6B for use as "Figure 1" in a mechanistic interpretability paper.

Shows the transformer pipeline with Layer 12 expanded to reveal where
the steering vector is injected into the residual stream.

Usage:
    uv run python viz/intervention_schematic.py
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

# ── Theme colors (GitHub dark) ───────────────────────────────────────────────

BG = "#0d1117"
TEXT = "#c9d1d9"
TEXT_DIM = "#8b949e"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
ORANGE_GLOW = "#f0883e40"
GREEN = "#3fb950"
RED = "#f85149"
BORDER = "#30363d"
BLOCK_BG = "#161b22"
BLOCK_BG_LIGHT = "#1c2128"
LAYER12_BG = "#1a1510"
ARROW_COLOR = "#484f58"

# ── Layout constants ─────────────────────────────────────────────────────────

W = 760
H = 1100
CX = 340                          # Main column center (offset left to leave room for side label)
GAP = 14                          # Vertical gap between blocks
CORNER_R = 6


def el(parent, tag, attrs=None, text=None):
    """Shorthand to create a sub-element."""
    e = ET.SubElement(parent, tag, attrs or {})
    if text:
        e.text = text
    return e


def rounded_rect(parent, x, y, w, h, **kw):
    """Draw a rounded rect and return the element."""
    attrs = {
        "x": str(x), "y": str(y),
        "width": str(w), "height": str(h),
        "rx": str(kw.get("rx", CORNER_R)),
        "fill": kw.get("fill", BLOCK_BG),
        "stroke": kw.get("stroke", BORDER),
        "stroke-width": str(kw.get("stroke_width", 1.5)),
    }
    if "filter" in kw:
        attrs["filter"] = kw["filter"]
    return el(parent, "rect", attrs)


def text(parent, x, y, content, **kw):
    """Draw centered text."""
    attrs = {
        "x": str(x), "y": str(y),
        "text-anchor": kw.get("anchor", "middle"),
        "fill": kw.get("fill", TEXT),
        "font-size": str(kw.get("size", 12)),
    }
    if kw.get("bold"):
        attrs["font-weight"] = "bold"
    elif kw.get("weight"):
        attrs["font-weight"] = kw["weight"]
    return el(parent, "text", attrs, content)


def arrow_v(parent, x, y1, y2, color=ARROW_COLOR, marker="arrowhead"):
    """Vertical arrow from y1 down to y2."""
    el(parent, "line", {
        "x1": str(x), "y1": str(y1),
        "x2": str(x), "y2": str(y2),
        "stroke": color, "stroke-width": "1.5",
        "marker-end": f"url(#{marker})",
    })


def arrow_h(parent, x1, x2, y, color=ARROW_COLOR, marker="arrowhead"):
    """Horizontal arrow from x1 to x2 at y."""
    el(parent, "line", {
        "x1": str(x1), "y1": str(y),
        "x2": str(x2), "y2": str(y),
        "stroke": color, "stroke-width": "2",
        "marker-end": f"url(#{marker})",
    })


def dashed_line(parent, x1, y1, x2, y2, color=TEXT_DIM):
    """Dashed line (for skip connections)."""
    el(parent, "line", {
        "x1": str(x1), "y1": str(y1),
        "x2": str(x2), "y2": str(y2),
        "stroke": color, "stroke-width": "1",
        "stroke-dasharray": "4,3",
    })


def block(parent, cx, y, w, h, label, sublabel=None,
          fill=BLOCK_BG, stroke=BORDER, text_color=TEXT,
          text_size=12, stroke_width=1.5, **kw):
    """Draw a labeled block. Returns (y_top, y_bottom)."""
    x = cx - w // 2
    rounded_rect(parent, x, y, w, h,
                 fill=fill, stroke=stroke, stroke_width=stroke_width,
                 **{k: v for k, v in kw.items() if k in ("rx", "filter")})

    if sublabel:
        text(parent, cx, y + h // 2 - 6, label,
             fill=text_color, size=text_size, weight="600")
        text(parent, cx, y + h // 2 + 10, sublabel,
             fill=TEXT_DIM, size=10)
    else:
        text(parent, cx, y + h // 2 + 4, label,
             fill=text_color, size=text_size, weight="600")

    return y, y + h


def make_svg():
    """Build the complete SVG as an ElementTree."""
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {W} {H}",
        "width": str(W),
        "height": str(H),
        "font-family": "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
    })

    # ── Defs ─────────────────────────────────────────────────────────────

    defs = el(svg, "defs")

    # Orange glow filter
    filt = el(defs, "filter", {
        "id": "glow", "x": "-20%", "y": "-20%",
        "width": "140%", "height": "140%",
    })
    el(filt, "feGaussianBlur", {
        "in": "SourceGraphic", "stdDeviation": "5", "result": "blur",
    })
    merge = el(filt, "feMerge")
    el(merge, "feMergeNode", {"in": "blur"})
    el(merge, "feMergeNode", {"in": "SourceGraphic"})

    # Arrow markers
    for mid, color in [("arrowhead", ARROW_COLOR), ("arrow-orange", ORANGE),
                       ("arrow-blue", BLUE)]:
        m = el(defs, "marker", {
            "id": mid, "markerWidth": "8", "markerHeight": "6",
            "refX": "8", "refY": "3", "orient": "auto",
        })
        el(m, "polygon", {"points": "0 0, 8 3, 0 6", "fill": color})

    # ── Background ───────────────────────────────────────────────────────

    el(svg, "rect", {"width": str(W), "height": str(H), "fill": BG})

    # ── Title ────────────────────────────────────────────────────────────

    text(svg, W // 2, 32, "ActAdd Intervention Schematic: Qwen3-0.6B",
         size=16, bold=True)
    text(svg, W // 2, 52,
         "28 layers, d_model=1024, GQA 16Q/8KV, SwiGLU, RMSNorm",
         fill=TEXT_DIM, size=11)

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN PIPELINE  (explicit y coordinates for pixel-perfect layout)
    # ══════════════════════════════════════════════════════════════════════

    y = 72

    # ── 1. Input Tokens ──────────────────────────────────────────────────
    _, y_bot = block(svg, CX, y, 200, 46,
                     "Input Tokens", "x = [t\u2081, t\u2082, \u2026, t\u2099]",
                     fill=BG, stroke=BLUE, text_color=BLUE)
    y = y_bot + GAP
    arrow_v(svg, CX, y_bot, y)

    # ── 2. Embedding ─────────────────────────────────────────────────────
    _, y_bot = block(svg, CX, y, 220, 46,
                     "Token Embedding + RoPE", "d_model = 1024")
    y = y_bot + GAP
    arrow_v(svg, CX, y_bot, y)

    # ── 3. Layers 0-11 (collapsed) ───────────────────────────────────────
    _, y_bot = block(svg, CX, y, 240, 50,
                     "Transformer Layers 0\u201311",
                     "12 \u00d7 (Attn + FFN + RMSNorm)")
    y = y_bot + GAP
    arrow_v(svg, CX, y_bot, y)

    # ══════════════════════════════════════════════════════════════════════
    #  LAYER 12 EXPANDED
    # ══════════════════════════════════════════════════════════════════════

    l12_y_start = y
    l12_w = 360
    l12_h = 400
    l12_x = CX - l12_w // 2

    # Outer container with glow
    rounded_rect(svg, l12_x, l12_y_start, l12_w, l12_h,
                 fill=LAYER12_BG, stroke=ORANGE, stroke_width=2,
                 rx=8, filter="url(#glow)")

    # Container labels
    text(svg, l12_x + 12, l12_y_start + 18,
         "Layer 12  (Intervention Target)",
         fill=ORANGE, size=12, bold=True, anchor="start")
    text(svg, l12_x + l12_w - 12, l12_y_start + 18,
         "best steering layer",
         fill=ORANGE, size=10, anchor="end")

    # Inner y tracker
    iy = l12_y_start + 32
    inner_w = 150
    inner_w_wide = 180

    # --- residual_in ---
    res_in_top, res_in_bot = block(svg, CX, iy, inner_w, 36,
                                   "residual_in",
                                   text_color=TEXT_DIM, text_size=11)
    iy = res_in_bot + 10
    arrow_v(svg, CX, res_in_bot, iy)

    # --- RMSNorm (pre-attn) ---
    _, rn1_bot = block(svg, CX, iy, inner_w, 34, "RMSNorm", text_size=11)
    iy = rn1_bot + 10
    arrow_v(svg, CX, rn1_bot, iy)

    # --- Multi-Head Attention ---
    attn_top, attn_bot = block(svg, CX, iy, inner_w_wide, 46,
                               "Multi-Head Attention",
                               "GQA: 16Q / 8KV heads",
                               fill=BLOCK_BG_LIGHT, stroke=BLUE,
                               text_size=11)
    iy = attn_bot + 10
    arrow_v(svg, CX, attn_bot, iy)

    # --- + Residual (post-attn) ---
    add1_top, add1_bot = block(svg, CX, iy, 110, 32,
                               "+ Residual", text_size=10)

    # Skip connection: residual_in -> + Residual (post-attn)
    skip_x = CX + inner_w_wide // 2 + 24
    dashed_line(svg, skip_x, res_in_bot, skip_x, add1_top + 16)
    dashed_line(svg, skip_x, add1_top + 16, CX + 55, add1_top + 16)

    iy = add1_bot + 10
    arrow_v(svg, CX, add1_bot, iy)

    # --- RMSNorm (pre-FFN) ---
    _, rn2_bot = block(svg, CX, iy, inner_w, 34, "RMSNorm", text_size=11)
    iy = rn2_bot + 10
    arrow_v(svg, CX, rn2_bot, iy)

    # --- SwiGLU FFN ---
    ffn_top, ffn_bot = block(svg, CX, iy, inner_w_wide, 46,
                             "SwiGLU FFN", "d_ff = 4096",
                             fill=BLOCK_BG_LIGHT, stroke=BLUE,
                             text_size=11)
    iy = ffn_bot + 10
    arrow_v(svg, CX, ffn_bot, iy)

    # ── INJECTION POINT: residual + alpha*v ──────────────────────────────

    inject_y = iy
    inject_w = 210
    inject_h = 52
    inject_x = CX - inject_w // 2

    # Glow background rect
    rounded_rect(svg, inject_x - 6, inject_y - 6, inject_w + 12, inject_h + 12,
                 fill=ORANGE_GLOW, stroke="none", stroke_width=0, rx=10)

    # Main injection box
    rounded_rect(svg, inject_x, inject_y, inject_w, inject_h,
                 fill="#2a1a08", stroke=ORANGE, stroke_width=2.5, rx=CORNER_R)

    # Equation
    text(svg, CX, inject_y + 20,
         "residual_out = h + \u03b1\u00b7v",
         fill=ORANGE, size=13, bold=True)
    text(svg, CX, inject_y + 37,
         "\u03b1=2.0  |  ||v||=19.6  |  ~8% perturbation",
         fill=TEXT_DIM, size=9)

    inject_bot = inject_y + inject_h

    # Skip connection: + Residual (post-attn) -> injection point
    skip2_x = CX - inner_w_wide // 2 - 20
    dashed_line(svg, skip2_x, add1_bot, skip2_x, inject_y + inject_h // 2)
    dashed_line(svg, skip2_x, inject_y + inject_h // 2, inject_x, inject_y + inject_h // 2)

    # ── Steering vector label (from right side) ──────────────────────────

    vec_y = inject_y + inject_h // 2
    vec_box_x = l12_x + l12_w + 30
    vec_box_w = 130
    vec_box_h = 62

    # Arrow from vector box to injection point
    arrow_h(svg, vec_box_x - 2, inject_x + inject_w + 4, vec_y,
            color=ORANGE, marker="arrow-orange")

    # Vector label box
    rounded_rect(svg, vec_box_x, vec_y - vec_box_h // 2,
                 vec_box_w, vec_box_h,
                 fill=BLOCK_BG, stroke=ORANGE, stroke_width=1.5)

    vby = vec_y - vec_box_h // 2
    text(svg, vec_box_x + vec_box_w // 2, vby + 16,
         "Steering Vector", fill=ORANGE, size=11, bold=True)
    text(svg, vec_box_x + vec_box_w // 2, vby + 30,
         "v = act(+) \u2212 act(\u2212)", fill=TEXT_DIM, size=9)
    text(svg, vec_box_x + vec_box_w // 2, vby + 42,
         "raw ActAdd (Turner 2023)", fill=TEXT_DIM, size=8)
    text(svg, vec_box_x + vec_box_w // 2, vby + 54,
         "||v|| \u2248 19.6", fill=TEXT_DIM, size=9)

    # ── Close Layer 12 ───────────────────────────────────────────────────

    y = l12_y_start + l12_h + GAP
    arrow_v(svg, CX, l12_y_start + l12_h, y)

    # ══════════════════════════════════════════════════════════════════════
    #  POST-INTERVENTION PIPELINE
    # ══════════════════════════════════════════════════════════════════════

    # ── 5. Layers 13-27 (collapsed) ──────────────────────────────────────
    _, y_bot = block(svg, CX, y, 240, 50,
                     "Transformer Layers 13\u201327",
                     "15 \u00d7 (Attn + FFN + RMSNorm)")
    y = y_bot + GAP
    arrow_v(svg, CX, y_bot, y)

    # ── 6. Final RMSNorm ─────────────────────────────────────────────────
    _, y_bot = block(svg, CX, y, 180, 42, "Final RMSNorm")
    y = y_bot + GAP
    arrow_v(svg, CX, y_bot, y)

    # ── 7. LM Head ───────────────────────────────────────────────────────
    _, y_bot = block(svg, CX, y, 210, 46,
                     "LM Head (Linear)",
                     "1024 \u2192 151,936 vocab")
    y = y_bot + GAP
    arrow_v(svg, CX, y_bot, y)

    # ── 8. Output Tokens ─────────────────────────────────────────────────
    _, y_bot = block(svg, CX, y, 200, 46,
                     "Output Tokens", "argmax / sample",
                     fill=BG, stroke=GREEN, text_color=GREEN)

    # ══════════════════════════════════════════════════════════════════════
    #  SIDE ANNOTATIONS
    # ══════════════════════════════════════════════════════════════════════

    # ── Model summary (left side, next to layers 0-11) ───────────────────

    ann_x = 30
    ann_y = l12_y_start - 60

    text(svg, ann_x, ann_y, "Qwen3-0.6B",
         fill=TEXT, size=13, bold=True, anchor="start")
    for i, line in enumerate([
        "751M parameters",
        "28 transformer layers",
        "d_model = 1024",
        "GQA: 16Q / 8KV",
        "SwiGLU d_ff = 4096",
    ]):
        text(svg, ann_x, ann_y + 16 + i * 14, line,
             fill=TEXT_DIM, size=10, anchor="start")

    # ── Perturbation budget (left side, next to injection point) ─────────

    pb_x = 30
    pb_y = inject_y - 30

    text(svg, pb_x, pb_y, "Perturbation Budget",
         fill=ORANGE, size=11, bold=True, anchor="start")
    pb_lines = [
        ("\u03b1 = 2.0 (steering strength)", TEXT_DIM),
        ("||v|| = 19.6 (raw ActAdd vec)", TEXT_DIM),
        ("||h|| \u2248 475 (residual norm)", TEXT_DIM),
        ("\u03b1\u00b7||v|| / ||h|| \u2248 8.3%", TEXT),
        ("", TEXT_DIM),
        ("< 2%   dead zone (no effect)", TEXT_DIM),
        ("2\u201310%  effective steering", GREEN),
        ("> 12%  coherence collapse", RED),
    ]
    for i, (line, color) in enumerate(pb_lines):
        if line:
            text(svg, pb_x, pb_y + 16 + i * 14, line,
                 fill=color, size=9, anchor="start")

    # ── Citation ─────────────────────────────────────────────────────────

    text(svg, W // 2, H - 14,
         "Turner et al. (2023) Activation Addition  |  Layer 12, \u03b1=2.0, raw vec  |  ~8% residual perturbation",
         fill=TEXT_DIM, size=9)

    return svg


def svg_to_string(svg_element):
    """Convert ElementTree to a clean SVG string."""
    rough = ET.tostring(svg_element, encoding="unicode", xml_declaration=False)
    dom = minidom.parseString(rough)
    pretty = dom.toprettyxml(indent="  ", encoding=None)
    # Strip the XML declaration that minidom adds
    lines = pretty.split("\n")
    if lines[0].startswith("<?xml"):
        lines = lines[1:]
    return "\n".join(line for line in lines if line.strip())


def generate_alt_text():
    """Generate alt-text description of the intervention schematic."""
    return """Intervention Schematic: ActAdd Steering in Qwen3-0.6B

This diagram shows the architecture of the Qwen3-0.6B transformer model
(28 layers, d_model=1024) with the ActAdd steering intervention point
highlighted at Layer 12.

Pipeline (top to bottom):
1. Input Tokens: tokenized prompt sequence [t1, t2, ..., tn]
2. Token Embedding + RoPE: maps tokens to d_model=1024 dimensional vectors
3. Transformer Layers 0-11: 12 standard blocks (collapsed), each containing
   Multi-Head Attention (GQA: 16 query / 8 KV heads), SwiGLU FFN (d_ff=4096),
   and RMSNorm
4. Layer 12 (Expanded - Intervention Target):
   - residual_in enters the layer
   - RMSNorm (pre-attention normalization)
   - Multi-Head Attention with Grouped Query Attention (16Q/8KV)
   - + Residual connection (post-attention skip)
   - RMSNorm (pre-FFN normalization)
   - SwiGLU FFN (d_ff=4096)
   - INJECTION POINT (highlighted in orange):
     residual_out = h + alpha * v
     The steering vector v (raw activation difference, ||v||=19.6) is added
     to the residual stream with alpha=2.0, producing ~8% perturbation
     relative to the residual stream norm (||h||~475).
   - Skip connection from post-attention residual feeds into injection point
5. Transformer Layers 13-27: 15 standard blocks (collapsed)
6. Final RMSNorm
7. LM Head (Linear): projects 1024 -> 151,936 vocabulary logits
8. Output Tokens: argmax or sampled from logit distribution

The steering vector is computed as v = act(+) - act(-), the raw difference
between activations from positive and negative style prompts (Turner et al.
2023 ActAdd method). No normalization is applied to preserve natural scale.

Perturbation budget annotations (left side):
- alpha = 2.0 (steering strength)
- ||v|| = 19.6 (raw ActAdd vector norm)
- ||h|| approx 475 (residual stream norm at layer 12)
- alpha * ||v|| / ||h|| approx 8.3%

Phase boundaries:
- < 2% perturbation: dead zone, no effect on output
- 2-10% perturbation: effective steering (sweet spot)
- > 12% perturbation: coherence collapse (garbage output)

Color scheme: dark background (#0d1117), blue (#58a6ff) for standard
architectural components, orange (#f0883e) with glow effect for the
intervention point, green (#3fb950) for output tokens.

Citation: Turner et al. (2023) Activation Addition
"""


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Generate SVG
    svg = make_svg()
    svg_str = svg_to_string(svg)

    svg_path = os.path.join(output_dir, "intervention_schematic.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
    print(f"SVG written to {svg_path}")

    # Generate alt-text
    alt_path = os.path.join(output_dir, "intervention_schematic.txt")
    with open(alt_path, "w", encoding="utf-8") as f:
        f.write(generate_alt_text())
    print(f"Alt-text written to {alt_path}")

    # Validate XML
    try:
        ET.fromstring(svg_str)
        print("XML validation: PASS")
    except ET.ParseError as e:
        print(f"XML validation: FAIL - {e}")
        raise

    # Print dimensions
    print(f"Canvas: {W}x{H}")


if __name__ == "__main__":
    main()
