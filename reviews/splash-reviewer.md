# SPLASH / NeurIPS MechInterp Workshop — Figure Review

Reviewer perspective: PL systems + mechanistic interpretability.

## A. What This Project Does Well

1. **Phase diagram with regime labels.** The alpha phase diagram
   (dead-zone → effective → collapse) is exactly what Turner et al. 2023 present
   but clearer — the comparison of raw vs unit-normalized vectors on a log scale
   is a novel contribution that directly explains why many reproductions fail.

2. **Layer-by-layer anatomy.** The dual-panel (residual norms + SNR%) across 28
   layers follows the convention from Rimsky et al. 2024 (their Figure 3) and
   Li et al. 2023 (ITI). The highlighted best-layer annotation is useful.

3. **Effect size reporting.** Rank-biserial r with Bonferroni correction is more
   rigorous than most ActAdd papers, which typically show only accuracy or
   qualitative examples. This is a strength.

4. **Terministic screen radar.** No related paper has an equivalent. The
   conceptual contamination measurement is original and would attract attention
   at a workshop.

5. **Dark theme consistency.** All figures use the same color scheme, making them
   cohesive as a figure set. Most papers use default matplotlib white, so this
   stands out (positively for a workshop poster, potentially negatively for a
   journal where editors may prefer white backgrounds for print).

## B. Missing Conventions

### From Turner et al. 2023 (ActAdd)
- **Qualitative example table.** Turner shows a table of prompts → baseline
  output → steered output side by side. This project has the data
  (terse_verbose_full.json) but no figure showing cherry-picked examples. This
  is typically "Figure 1" or "Table 1" in any steering paper.
- **Multiple behaviors tested.** Turner tests sycophancy, hallucination,
  emotional valence. This project has 4 style axes but only evaluates
  terse/verbose quantitatively. The other 3 (formal, socratic, dry-wit) need at
  least qualitative examples.

### From Rimsky et al. 2024 (CAA on Llama 2)
- **Layer sweep line plot.** Rimsky plots accuracy/effect vs layer index as a
  line chart with error bands across multiple prompts. The current layer_anatomy
  shows norms but not the actual steering effectiveness per layer. The
  sweep_to_cprr.py does this but has no visualization.
- **Multiple-choice eval.** Rimsky uses A/B forced-choice to measure behavioral
  shift. No equivalent here — all evaluation is on free-form generation length.

### From Konen et al. 2024 (Style Vectors, EACL)
- **Cosine similarity heatmap between style vectors.** Konen shows a matrix of
  cos_sim between different style steering vectors. This project has 4 style
  axes — the H-SV-1 hypothesis is literally "style vectors are not orthogonal"
  but there's no figure showing the cosine similarity matrix.
- **Human evaluation scores.** Konen reports human judgments of style strength.
  Not expected for a workshop paper but noted.

### From Arditi et al. 2024 (Refusal Direction)
- **Single-direction visualization.** Arditi shows the refusal direction as a 1D
  projection with histograms of "refused" vs "answered" prompts. An equivalent
  for terse (project terse vector, show distribution of baseline vs steered
  activations along that direction) would be compelling.
- **Causal intervention results.** Arditi shows what happens when you add/remove
  the direction. The existing alpha sweep partially covers this.

### From Jorgensen et al. 2023 (Mean-Centring)
- **Mean-centred vs non-mean-centred comparison.** This paper's key contribution
  is that subtracting the mean activation improves steering. The current project
  doesn't implement mean-centring — this is noted in the deep-dive as a gap.
  A comparison figure would strengthen the paper.

### From Park et al. 2023 (Linear Representation Hypothesis)
- **Probing accuracy across layers.** Park shows linear probe accuracy for
  various concepts vs layer depth. An equivalent would train a linear probe for
  "is this terse?" across layers and show where the concept is most linearly
  decodable.

### General conventions (NeurIPS MechInterp)
- **Error bars / confidence intervals across seeds.** Every figure in this
  project shows a single run. Reviewers expect ≥3 seeds with error bars or
  confidence bands.
- **Perplexity as coherence metric.** Most steering papers report perplexity
  under an unsteered model as a coherence measure. Word count alone doesn't
  distinguish "terse and good" from "terse and garbage".

## C. Specific Figure Suggestions

### Figure 1 (the hook diagram)
Every steering paper needs a schematic showing WHERE the intervention happens.
Create a simplified transformer block diagram showing:
```
Input → Embed → [Layer 0] → ... → [Layer 12: hook adds α·v] → ... → [Layer 27] → LM Head → Output
```
The diagrams.org file has a mermaid version, but this should be a clean
vector-graphic figure, not a mermaid render.

### The missing "money figure"
A 2x2 or 3-column figure:
| Prompt | Baseline Output | Steered (terse, α=2.0) |
Show 3-4 cherry-picked examples where terse steering clearly works. This is
what readers remember. Every steering paper leads with this.

### Style vector geometry
A cosine similarity matrix (4x4 heatmap) of the 4 style vectors. Add the
refusal direction if computable. This directly tests H-SV-1.

### Layer sweep effectiveness
Line plot: x=layer (0-27), y=word count reduction (baseline - steered), with
error bands across 10+ prompts. Highlight the sweet spot.

## D. Statistical Rigor

- **n=10 is insufficient for publication.** The KDE, PCA ellipses, and confusion
  matrix are all preliminary. n=100 is the minimum for the claims being made.
  At n=10, the standard error of the mean word count is ~15/√10 ≈ 5, so the
  terse mean of 51 ± 10 (95% CI) doesn't overlap with baseline 122 ± 16, but
  the baseline 122 ± 16 and verbose 143 ± 12 DO overlap. This explains the
  ARI < 0.50.
- **Single seed.** All results use random_state=42. A reviewer would ask: "does
  this replicate with seed=0, 1, 2?" At minimum, report 3 seeds.
- **No perplexity or coherence metric.** Word count reduction could be achieved
  by generating garbage. Show that steered outputs are still coherent (e.g.,
  perplexity under the unsteered model, or a simple n-gram fluency score).
- **Bonferroni is conservative.** With 9 comparisons, Bonferroni at n=10 loses
  power. Consider Benjamini-Hochberg FDR correction instead, and note which
  correction is used.
- **Bootstrap the ARI.** Report ARI with 95% bootstrap CI, not a point estimate.

## E. Top 5 New Figures to Add

### 1. Qualitative Example Table (highest priority)
**What**: 4-5 prompts, each with baseline and terse-steered output, side by side.
**Why**: This is "Table 1" in every steering paper. Readers need to SEE the
effect before they trust the statistics. Currently the project has the data but
no figure.

### 2. Style Vector Cosine Similarity Matrix
**What**: 4x4 heatmap of cos_sim between terse, formal, socratic, dry-wit
vectors at the best layer. Optionally add the refusal direction as a 5th row.
**Why**: Directly tests H-SV-1 ("style vectors are not orthogonal"). This is
the key claim about superposition at d_model=1024. Konen et al. 2024 show this
for their style vectors; it's expected.

### 3. Layer Sweep Effectiveness Plot
**What**: Line plot with x=layer (0-27), y=mean word count (or token count)
for steered outputs, with ±1 SE bands across prompts. One line per alpha value
(1.0, 1.5, 2.0, 2.5, 3.0).
**Why**: The current layer_anatomy shows norms, but not the actual effect. A
reviewer wants to see that layer 12 actually produces the most terse output,
not just the highest SNR.

### 4. Coherence vs Compression Scatter
**What**: x=word count reduction (%), y=perplexity under unsteered model.
Each point is one prompt at one alpha. Color by alpha value.
**Why**: Demonstrates that terse steering produces compression WITHOUT
incoherence, up to the collapse threshold. Addresses the "maybe it's just
generating garbage" concern.

### 5. Intervention Schematic (Figure 1)
**What**: Clean vector diagram of the 28-layer transformer with the steering
hook at layer 12. Show the residual stream, the added vector, and the
perturbation magnitude (~8% of residual norm).
**Why**: Every mechanistic interpretability paper needs a figure showing the
intervention site. The mermaid diagrams exist but aren't publication-quality.
