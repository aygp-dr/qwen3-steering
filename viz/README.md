# Qwen3-0.6B Steering Vector Visualizations

Visualizations for the activation steering research project. Each script is standalone and runnable with `uv run python`.

## Static Visualizations (matplotlib)

All static plots save to `viz/output/` as PNG.

### 1. Residual Landscape (`residual_landscape.py`)

2D conceptual projection of the d_model=1024 activation space. Shows style basins (terse, formal, socratic, dry-wit), the cult-of-jason basin overlapping with terse via superposition, steering trajectories at different alpha values, and concentric rings marking the dead zone (alpha < 0.3), sweet spot (1.0-2.5), and coherence collapse (> 3.0).

```
uv run python viz/residual_landscape.py
```

### 2. Layer Anatomy (`layer_anatomy.py`)

Side-view of all 28 transformer layers. Residual stream norms grow from ~455 (layer 5) to ~810 (layer 24). Raw steering vector magnitudes overlaid. SNR percentages computed per layer. Zones annotated: topic drift (0-7), sweet spot (12-18), output layers (24-27). Empirical data from `diagnose_steering.py`.

```
uv run python viz/layer_anatomy.py
```

### 3. Alpha Phase Diagram (`alpha_phase_diagram.py`)

Phase diagram: alpha vs output word count. Three regimes visible: dead zone where nothing changes, effective steering where output shortens to 16 words, and the cliff edge at alpha ~3.0 where coherence collapses. Lower panel compares raw vs unit-normalized vectors on a log scale, explaining why alpha=0.20 with a unit vector (SNR=0.04%) produces zero effect while alpha=2.0 with a raw vector (SNR=8%) works.

```
uv run python viz/alpha_phase_diagram.py
```

### 4. Lens Contamination Radar (`lens_contamination_radar.py`)

Four-panel radar chart showing 12 lens contamination scores. Compares baseline (clean), leaked steer (cult-of-jason and ai-hype spike), clean steer (only style changes), and collapsed output (everything spikes). Demonstrates what the lens eval framework from `lens_eval.py` detects: when pushing on "terse" drags outputs into specification-driven vocabulary because the features share dimensions in the 1024-d residual stream.

```
uv run python viz/lens_contamination_radar.py
```

## Interactive Visualization (pygame)

### 5. Steering Explorer (`steering_pygame.py`)

Real-time 2D simulation of activation space. Particles represent token positions. Drag a steering vector arrow with the mouse and watch particles shift toward style basins. Layer animation shows how the perturbation propagates through transformer depth. Color-coded phase transitions: particles stay gray in the dead zone, shift to style colors in the effective range, and turn red in the collapse zone.

```
uv run python viz/steering_pygame.py
```

Controls:
- **Arrow Up/Down**: increase/decrease alpha by 0.2
- **Arrow Left/Right**: rotate steering direction
- **1-4**: select style axis (terse, formal, socratic, dry-wit)
- **Space**: reset to baseline
- **L**: toggle layer-by-layer animation
- **Tab**: step to next layer
- **Mouse drag**: direct control of vector direction and magnitude
- **Scroll wheel**: fine-tune alpha
- **Escape**: quit

## Shared Resources

- `shared_style.py`: Color palette, empirical data constants, and matplotlib styling functions used by all static visualizations.

## Dependencies

Added to `pyproject.toml`:
- `matplotlib` (static plots)
- `numpy` (computation)
- `scipy` (interpolation)
- `pygame` (interactive viz)

## Empirical Data Source

The following measurements come from running `diagnose_steering.py` on Qwen3-0.6B:

| Layer | Residual Norm | Raw Vec Norm | SNR (a=1.0) |
|-------|--------------|-------------|-------------|
| 5     | 455          | 14.1        | 3.1%        |
| 10    | 472          | 15.8        | 3.3%        |
| 14    | 481          | 18.7        | 3.9%        |
| 15    | 488          | 19.6        | 4.0%        |
| 16    | 501          | 22.9        | 4.6%        |
| 18    | 538          | 29.8        | 5.5%        |
| 20    | 595          | 40.3        | 6.8%        |
| 24    | 810          | 70.8        | 8.7%        |

Best steering layer: 12 (mutex went from 149 to 16 words with terse vector at alpha=2.0).
