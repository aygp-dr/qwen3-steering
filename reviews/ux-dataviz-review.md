# UX / Data Visualization Review

Reviewer perspective: data-viz specialist, accessibility-aware.

## 1. Color & Contrast

**Dark theme is generally well-executed.** The #0d1117 / #161b22 background pair
matches GitHub's dark mode, giving a cohesive feel. However:

- **Colorblind risk**: The terse (#58a6ff blue) / verbose (#f0883e orange) pair
  is deuteranopia-safe, but baseline (#8b949e gray) nearly vanishes against the
  dark panel background at small point sizes. The gray-on-dark-gray combination
  fails WCAG AA contrast (estimated 3.2:1 vs required 4.5:1).
- **Residual landscape**: Uses 6+ colors (blue, orange, green, magenta, cyan,
  yellow, red) with no colorblind-safe palette. The Cult-of-Jason red and
  coherence-collapse red are nearly identical.
- **Radar chart**: Baseline (red) and steered (cyan/green) overlap on the same
  axes with thin lines — hard to distinguish at screen resolution.
- **Fix**: Switch baseline to #d2a8ff (light purple) or use shape encoding
  (squares vs circles) alongside color. For residual landscape, use a
  qualitative colorblind-safe palette like Okabe-Ito or IBM Design.

## 2. Typography

- **Dashboard (06_dashboard)**: Panel titles at fontsize=10 are legible, but the
  summary text in panel F uses fontfamily="monospace" at fontsize=10 which
  renders differently across backends — test with Agg and Cairo.
- **Layer anatomy**: The key measurements text on the right side uses very small
  font. "ZERO EFFECT" and "WORKING" labels are effective but feel informal for a
  paper figure.
- **Residual landscape**: Annotation text ("superposition: shared features at
  d_model=1024") overlaps with scatter points. Needs a semi-transparent
  background box or repositioning.
- **Alpha phase diagram**: The bottom panel's axis label "Alpha (log scale)"
  competes with the annotation callout. Good use of the "Raw vec needs alpha ~100x"
  annotation.
- **Fix**: Increase minimum fontsize to 8pt for print, 10pt for screen. Add
  `bbox=dict(facecolor=PANEL_BG, alpha=0.8)` to all text annotations.

## 3. Layout & Whitespace

- **Dashboard**: The 2x3 grid is well-proportioned. Panel F (text summary) could
  use the space better — it's 1/6 of the figure for ~10 lines of monospace text.
  Consider replacing with a compact table or a mini bar chart of means.
- **Elbow + Silhouette (02)**: The two panels are well-balanced, but the
  annotation arrow for "best k=2" points leftward off the chart area when
  best_k=2 and xytext is at best_k+1.5 — check edge case.
- **Effect size heatmap (05)**: The 3x3 grid is compact but the colorbar takes
  disproportionate horizontal space. Use `shrink=0.8` on the colorbar.
- **Per-class detail**: Three scatter plots side-by-side work, but the x-axis
  "Prompt index" is uninformative — consider sorting by word count instead.

## 4. Data-Ink Ratio (Tufte)

- **Good**: The KDE plots are clean. The confusion matrix annotations are
  minimal. The effect size heatmap combines value + significance elegantly.
- **Chartjunk**: The residual landscape has too many overlapping elements —
  scatter points, basin ellipses, vector arrows, alpha zones (concentric
  circles), annotations, and a legend that lists 10+ items. This is the most
  cluttered figure. Consider splitting into 2 figures: one for basins, one for
  the alpha dead-zone/collapse overlay.
- **Redundant encoding**: In the distribution histogram, color AND position both
  encode the same variable (direction). The overlapping histograms make this
  necessary, but the KDE version (01_descriptive_kde) is strictly superior —
  retire the histogram version.
- **Radar chart**: The 4-subplot radar layout is heavy. Consider a single
  grouped bar chart (lenses on x-axis, grouped bars for baseline/steered/collapsed).

## 5. Annotation Quality

- **Strong**: Median dashed lines on KDEs with numeric labels — exactly right.
  Bonferroni stars on effect sizes are standard and helpful.
- **Confidence ellipses**: Useful in the PCA scatter, but at n=10 per group the
  ellipses are dominated by sampling noise. Add a note "n=10, preliminary" or
  use bootstrap CIs instead.
- **Missing**: The alpha phase diagram should annotate the three regime
  boundaries with vertical bands (not just word-count inflection points). The
  dead-zone/effective/collapse labels are present but float disconnectedly.
- **Burke annotation**: The italic text at the bottom of the confusion matrix
  ("Burke prediction: terse/baseline bleed...") is a good narrative touch but
  may be too domain-specific for a general audience. Keep for the org file,
  consider removing from the paper figure.

## 6. Dashboard Composition

The 6-panel dashboard has a reasonable narrative:
A (what the data looks like) → B (how many clusters) → C (where they live) →
D (how well we recover them) → E (how strong the signal is) → F (verdict).

**Issues**:
- The flow reads left-to-right, top-to-bottom, which is correct.
- Panel B (silhouette) is the weakest panel — it answers a methodological
  question, not a scientific one. Consider swapping it with the scatter or
  moving it to supplementary.
- Panel F as pure text is anticlimactic. A mini waterfall or paired bar chart
  (terse μ=51 vs baseline μ=122 vs verbose μ=143) would be more visual.
- The title "Six-Panel Dashboard" is meta — name it after the finding, e.g.
  "Surface Features Partially Recover Steering Direction (C-28)".

## 7. Specific Issues

- **KDE on n=10**: The kernel density estimate with only 10 points per group is
  unreliable. The bandwidth selection (Scott's rule) assumes normality and will
  over-smooth. At n=10, a strip plot or swarm plot would be more honest.
  Acceptable for the preliminary run; must regenerate at n=100.
- **PCA with 99.8% in PC1**: When one component dominates this much, the PCA
  scatter is essentially a 1D plot spread vertically by noise. Consider a
  strip/jitter plot along PC1 only, or note that the features are nearly
  collinear and PCA adds no information beyond word count alone.
- **Effect size with n=10**: Rank-biserial r is appropriate for small samples,
  but the Bonferroni correction with 9 tests on n=10 is conservative enough that
  the non-significant baseline-vs-verbose result may flip at n=100. Flag this.
- **Confusion matrix normalization**: Row-normalization is correct for recall,
  but also show precision (column-normalization) or F1 per class.

## 8. Top 5 Actionable Fixes

1. **Replace histogram with strip/swarm plot at n=10.** KDE with 10 points is
   misleading. Use `sns.stripplot` or `sns.swarmplot` with jitter. Switch to KDE
   only when n >= 30.

2. **Fix baseline color contrast.** Change #8b949e to #d2a8ff (light purple) or
   #79c0ff (lighter blue variant) to meet WCAG AA against dark backgrounds.

3. **Split residual landscape into 2 figures.** (a) Style basins with steering
   vectors. (b) Alpha zone overlay. Current version has > 10 visual elements
   competing for attention.

4. **Add confidence intervals / bootstrap bands.** Every mean comparison (KDE
   medians, confusion matrix cells, effect sizes) should show uncertainty at
   n=10. Use `scipy.stats.bootstrap` for non-parametric CIs.

5. **Rename dashboard title to reflect the finding.** "Surface Features Recover
   Terse Screen but Not Verbose (C-28, ARI=0.395)" is more informative than
   "Six-Panel Dashboard".
