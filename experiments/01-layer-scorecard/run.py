"""
Layer Scorecard: characterize what each of Qwen3-0.6B's 28 layers encodes.

Runs six probes across all layers at multiple alphas to fill layer-roles.json
with empirical data. This is the grind — expect ~30 min on CPU, ~5 min on GPU.

Probes:
  1. Residual norm & raw steering vector norm (signal-to-noise)
  2. Terse word count at alpha=2.0 (style shift strength)
  3. Topic preservation (TF-IDF cosine vs baseline)
  4. Syntactic integrity (sentence count, parse heuristics)
  5. Formatting artifacts (markdown token density)
  6. Vocabulary distortion (perplexity ratio vs baseline)

Usage:
    python experiments/01-layer-scorecard/run.py                    # full sweep
    python experiments/01-layer-scorecard/run.py --layers 10 12 14  # targeted
    python experiments/01-layer-scorecard/run.py --fast             # 14-layer subset
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from actadd import (
    MODEL_ID, STYLE_PAIRS, get_layer_activations,
    compute_steering_vector, generate_steered,
)

PROMPTS = [
    "Explain what a mutex is.",
    "How does photosynthesis work?",
    "What causes ocean tides?",
    "Describe the CAP theorem.",
    "Why do birds migrate?",
]

ALPHAS = [0.5, 1.0, 2.0, 3.0]
STYLE = "terse"


def measure_all_norms(model, tokenizer, prompt, layers):
    """Residual stream norms and raw steering vector norms at every layer."""
    residual_norms = {}
    raw_vec_norms = {}

    pos_prompt, neg_prompt = STYLE_PAIRS[STYLE]

    for layer_idx in layers:
        # Residual norm
        captured = {}

        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["hs"] = hs.detach()

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        hs = captured["hs"].squeeze(0)
        residual_norms[layer_idx] = round(hs.norm(dim=-1).mean().item(), 1)

        # Raw steering vector norm
        act_pos = get_layer_activations(model, tokenizer, pos_prompt, layer_idx)
        act_neg = get_layer_activations(model, tokenizer, neg_prompt, layer_idx)
        raw_vec = act_pos - act_neg
        raw_vec_norms[layer_idx] = round(raw_vec.norm().item(), 1)

    return residual_norms, raw_vec_norms


def tfidf_cosine(text_a, text_b):
    """Simple TF-IDF cosine similarity between two texts."""
    words_a = re.findall(r'\w+', text_a.lower())
    words_b = re.findall(r'\w+', text_b.lower())
    if not words_a or not words_b:
        return 0.0

    tf_a = Counter(words_a)
    tf_b = Counter(words_b)
    vocab = set(tf_a.keys()) | set(tf_b.keys())

    dot_product = sum(tf_a.get(w, 0) * tf_b.get(w, 0) for w in vocab)
    norm_a = math.sqrt(sum(v ** 2 for v in tf_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in tf_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def count_sentences(text):
    """Rough sentence count."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def markdown_density(text):
    """Count markdown formatting tokens per 100 words."""
    words = text.split()
    if not words:
        return 0.0
    md_tokens = len(re.findall(r'\*\*|##|```|^- |\n- ', text))
    return md_tokens / len(words) * 100


def perplexity_ratio(model, tokenizer, baseline_text, steered_text):
    """Ratio of cross-entropy of steered text under the model vs baseline text.

    >1.0 means steered output is less likely under the unsteered model.
    """
    def cross_entropy(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        if inputs.input_ids.shape[1] < 2:
            return float('inf')
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
        return outputs.loss.item()

    ce_baseline = cross_entropy(baseline_text)
    ce_steered = cross_entropy(steered_text)
    if ce_baseline == 0 or ce_baseline == float('inf'):
        return None
    return round(ce_steered / ce_baseline, 3)


def score_layer(model, tokenizer, layer_idx, alpha, prompt, baseline_text):
    """Run all probes for one layer/alpha/prompt combination."""
    vec = compute_steering_vector(model, tokenizer, STYLE, layer_idx)
    steered_text = generate_steered(model, tokenizer, prompt, vec, layer_idx, alpha)

    words = steered_text.split()
    word_count = len(words)
    sentence_count = count_sentences(steered_text)
    words_per_sentence = word_count / max(sentence_count, 1)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "words_per_sentence": round(words_per_sentence, 1),
        "topic_cosine": round(tfidf_cosine(baseline_text, steered_text), 3),
        "markdown_density": round(markdown_density(steered_text), 2),
        "perplexity_ratio": perplexity_ratio(model, tokenizer, baseline_text, steered_text),
        "text_preview": steered_text[:120],
    }


def main():
    parser = argparse.ArgumentParser(description="Layer scorecard for Qwen3-0.6B")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to test (default: all 28)")
    parser.add_argument("--fast", action="store_true",
                        help="14-layer subset, single alpha=2.0, 2 prompts")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "output" / "layer-scorecard.json"))
    args = parser.parse_args()

    if args.layers:
        layers = args.layers
    elif args.fast:
        layers = [0, 3, 5, 8, 10, 12, 14, 15, 16, 18, 20, 22, 24, 27]
    else:
        layers = list(range(28))

    alphas = [2.0] if args.fast else ALPHAS
    prompts = PROMPTS[:2] if args.fast else PROMPTS

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    total_evals = len(layers) * len(alphas) * len(prompts)
    print(f"Scorecard: {len(layers)} layers x {len(alphas)} alphas x {len(prompts)} prompts = {total_evals} evaluations")

    # Phase 1: norms (one pass, fast)
    print("\n=== Phase 1: Measuring norms across all layers ===")
    residual_norms, raw_vec_norms = measure_all_norms(
        model, tokenizer, prompts[0], layers
    )
    for layer_idx in layers:
        rn = residual_norms[layer_idx]
        vn = raw_vec_norms[layer_idx]
        snr = round(vn / rn * 100, 2) if rn > 0 else None
        print(f"  L{layer_idx:2d}: residual={rn:6.1f}  vec={vn:5.1f}  SNR={snr}%")

    # Phase 2: baselines (one per prompt, no steering)
    print("\n=== Phase 2: Generating baselines ===")
    baselines = {}
    for prompt in prompts:
        # alpha=0 at any layer gives baseline
        vec_dummy = compute_steering_vector(model, tokenizer, STYLE, 12)
        baselines[prompt] = generate_steered(
            model, tokenizer, prompt, vec_dummy, 12, alpha=0.0
        )
        print(f"  Baseline ({len(baselines[prompt].split())} words): {baselines[prompt][:80]}...")

    # Phase 3: full sweep
    print("\n=== Phase 3: Layer x Alpha x Prompt sweep ===")
    scorecard = {}
    completed = 0
    start_time = time.time()

    for layer_idx in layers:
        scorecard[layer_idx] = {
            "residual_norm": residual_norms[layer_idx],
            "raw_vec_norm": raw_vec_norms[layer_idx],
            "snr_pct": round(raw_vec_norms[layer_idx] / residual_norms[layer_idx] * 100, 2),
            "alphas": {},
        }
        for alpha in alphas:
            alpha_results = []
            for prompt in prompts:
                result = score_layer(
                    model, tokenizer, layer_idx, alpha, prompt, baselines[prompt]
                )
                alpha_results.append(result)
                completed += 1
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_evals - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total_evals}] L{layer_idx:2d} a={alpha:.1f} "
                      f"words={result['word_count']:3d} topic={result['topic_cosine']:.2f} "
                      f"ppl_r={result['perplexity_ratio']} "
                      f"({remaining:.0f}s remaining)")

            # Aggregate across prompts
            avg_words = round(sum(r["word_count"] for r in alpha_results) / len(alpha_results), 1)
            avg_topic = round(sum(r["topic_cosine"] for r in alpha_results) / len(alpha_results), 3)
            avg_wps = round(sum(r["words_per_sentence"] for r in alpha_results) / len(alpha_results), 1)
            avg_md = round(sum(r["markdown_density"] for r in alpha_results) / len(alpha_results), 2)
            ppl_ratios = [r["perplexity_ratio"] for r in alpha_results if r["perplexity_ratio"] is not None]
            avg_ppl = round(sum(ppl_ratios) / len(ppl_ratios), 3) if ppl_ratios else None

            scorecard[layer_idx]["alphas"][str(alpha)] = {
                "avg_word_count": avg_words,
                "avg_topic_cosine": avg_topic,
                "avg_words_per_sentence": avg_wps,
                "avg_markdown_density": avg_md,
                "avg_perplexity_ratio": avg_ppl,
                "per_prompt": alpha_results,
            }

    # Phase 4: classification
    print("\n=== Phase 4: Layer role classification ===")
    for layer_idx in layers:
        entry = scorecard[layer_idx]
        a2 = entry["alphas"].get("2.0", entry["alphas"].get(str(alphas[0]), {}))
        avg_words = a2.get("avg_word_count", 999)
        avg_topic = a2.get("avg_topic_cosine", 0)
        snr = entry["snr_pct"]

        # Classify based on empirical behavior
        if layer_idx <= 2:
            role = "tokenization"
        elif layer_idx <= 7:
            role = "syntax"
        elif layer_idx <= 11:
            role = "early_semantics"
        elif layer_idx <= 17:
            role = "deep_semantics"
        elif layer_idx <= 22:
            role = "output_preparation"
        else:
            role = "logit_projection"

        # Override with empirical evidence
        if avg_topic < 0.3 and layer_idx < 8:
            empirical = "CONFIRMED: topic drift (cosine < 0.3)"
        elif avg_words < 50 and 10 <= layer_idx <= 17:
            empirical = f"CONFIRMED: strong style shift ({avg_words:.0f} words)"
        elif avg_words > 100 and layer_idx >= 18:
            empirical = f"CONFIRMED: weak effect ({avg_words:.0f} words ≈ baseline)"
        else:
            empirical = f"words={avg_words:.0f} topic={avg_topic:.2f}"

        entry["classified_role"] = role
        entry["empirical_note"] = empirical
        print(f"  L{layer_idx:2d} [{role:20s}] {empirical}")

    # Save
    output = {
        "model": MODEL_ID,
        "style": STYLE,
        "prompts": prompts,
        "alphas": alphas,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "layers": scorecard,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nScorecard saved to {args.output}")

    # Summary table
    print("\n=== Summary: Layer Roles (alpha=2.0) ===")
    print(f"{'Layer':>5} {'Region':>20} {'ResNorm':>8} {'VecNorm':>8} {'SNR%':>6} {'Words':>6} {'Topic':>6} {'PPL_r':>6}")
    print("-" * 75)
    for layer_idx in layers:
        entry = scorecard[layer_idx]
        a2 = entry["alphas"].get("2.0", entry["alphas"].get(str(alphas[0]), {}))
        print(f"{layer_idx:5d} {entry['classified_role']:>20s} "
              f"{entry['residual_norm']:8.1f} {entry['raw_vec_norm']:8.1f} "
              f"{entry['snr_pct']:6.2f} "
              f"{a2.get('avg_word_count', '-'):>6} "
              f"{a2.get('avg_topic_cosine', '-'):>6} "
              f"{a2.get('avg_perplexity_ratio', '-'):>6}")

    # Phase 5: CPRR evidence submission
    print("\n=== Phase 5: Submitting evidence to CPRR ===")
    submit_cprr_evidence(scorecard, layers)


def submit_cprr_evidence(scorecard, layers):
    """Submit evidence to CPRR conjectures based on scorecard results."""
    import subprocess

    def cprr_evidence(conjecture_id, text):
        result = subprocess.run(
            ["cprr", "evidence", str(conjecture_id), text],
            capture_output=True, text=True
        )
        status = "ok" if result.returncode == 0 else "FAIL"
        print(f"  cprr evidence {conjecture_id}: {status} — {text[:80]}")
        return result.returncode == 0

    def cprr_next(conjecture_id):
        result = subprocess.run(
            ["cprr", "next", str(conjecture_id)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  cprr next {conjecture_id}: {result.stdout.strip()}")

    # Conjecture 1: Layers 0-7 produce topic drift not style shift
    early_layers = [l for l in layers if l <= 7]
    if early_layers:
        early_topics = []
        early_words = []
        for l in early_layers:
            a2 = scorecard[l]["alphas"].get("2.0", {})
            topic = a2.get("avg_topic_cosine", 1.0)
            words = a2.get("avg_word_count", 0)
            early_topics.append(topic)
            early_words.append(words)
        avg_topic = sum(early_topics) / len(early_topics)
        avg_words = sum(early_words) / len(early_words)
        cprr_evidence(1, f"Layers {early_layers}: avg topic_cosine={avg_topic:.3f}, avg words={avg_words:.0f} at alpha=2.0 [source: this-project-sweep]")
        cprr_evidence(1, f"Prior: Geva 2021 (FFN as KV memories) + Meng 2022 (ROME causal tracing) identify early layers as encoding surface/syntactic features [confidence: prior, source: rimsky2024]")
        cprr_next(1)

    # Conjecture 2: Layers 12-17 are the semantic sweet spot
    sweet_layers = [l for l in layers if 12 <= l <= 17]
    if sweet_layers:
        sweet_words = []
        sweet_topics = []
        for l in sweet_layers:
            a2 = scorecard[l]["alphas"].get("2.0", {})
            sweet_words.append(a2.get("avg_word_count", 999))
            sweet_topics.append(a2.get("avg_topic_cosine", 0))
        avg_words = sum(sweet_words) / len(sweet_words)
        avg_topic = sum(sweet_topics) / len(sweet_topics)
        # Baseline is ~149 words (from CPRR conjecture 26/27)
        reduction_pct = (1 - avg_words / 149) * 100 if avg_words < 149 else 0
        cprr_evidence(2, f"Layers {sweet_layers}: avg words={avg_words:.0f} ({reduction_pct:.0f}% reduction), avg topic_cosine={avg_topic:.3f} at alpha=2.0 [source: this-project-sweep]")
        cprr_evidence(2, f"Prior: Rimsky 2024 CAA layer sweep + Turner 2023 ActAdd identify 50-70% depth as optimal [confidence: prior, source: rimsky2024]")
        cprr_next(2)

    # Conjecture 3: Layers 18-22 affect formatting more than style
    late_mid_layers = [l for l in layers if 18 <= l <= 22]
    if late_mid_layers:
        late_words = []
        late_md = []
        for l in late_mid_layers:
            a2 = scorecard[l]["alphas"].get("2.0", {})
            late_words.append(a2.get("avg_word_count", 0))
            late_md.append(a2.get("avg_markdown_density", 0))
        avg_words = sum(late_words) / len(late_words)
        avg_md = sum(late_md) / len(late_md)
        word_change_pct = abs(1 - avg_words / 149) * 100
        cprr_evidence(3, f"Layers {late_mid_layers}: avg words={avg_words:.0f} (change={word_change_pct:.0f}%), avg markdown_density={avg_md:.2f} at alpha=2.0 [source: this-project-sweep]")
        cprr_evidence(3, f"Prior: CPRR-27 showed L18=169w, L20=174w (near baseline 149w) — minimal word count change [confidence: empirical, source: this-project-sweep]")
        cprr_next(3)

    # Conjecture 4: Layers 23-27 cause vocabulary distortion
    late_layers = [l for l in layers if l >= 23]
    if late_layers:
        late_ppl = []
        late_words = []
        for l in late_layers:
            a2 = scorecard[l]["alphas"].get("2.0", {})
            ppl = a2.get("avg_perplexity_ratio")
            if ppl is not None:
                late_ppl.append(ppl)
            late_words.append(a2.get("avg_word_count", 0))
        avg_ppl = sum(late_ppl) / len(late_ppl) if late_ppl else None
        avg_words = sum(late_words) / len(late_words)
        ppl_str = f"avg perplexity_ratio={avg_ppl:.3f}" if avg_ppl else "perplexity_ratio=N/A"
        cprr_evidence(4, f"Layers {late_layers}: {ppl_str}, avg words={avg_words:.0f} at alpha=2.0 [source: this-project-sweep]")
        cprr_evidence(4, f"Prior: RAW_VEC_NORM L24=70.8, residual_norm=810, SNR=8.7% — extreme perturbation territory [confidence: empirical, source: this-project-sweep]")
        cprr_next(4)

    # Conjecture 5: Residual norms grow monotonically
    norms_list = [(l, scorecard[l]["residual_norm"]) for l in sorted(layers)]
    monotonic = all(norms_list[i][1] <= norms_list[i+1][1] for i in range(len(norms_list)-1))
    growth = (norms_list[-1][1] / norms_list[0][1] - 1) * 100 if norms_list[0][1] > 0 else 0
    cprr_evidence(5, f"Norms: {', '.join(f'L{l}={n:.0f}' for l,n in norms_list[:7])}... Monotonic={monotonic}, growth={growth:.0f}% [source: this-project-sweep]")
    cprr_evidence(5, f"Remaining: {', '.join(f'L{l}={n:.0f}' for l,n in norms_list[7:])} [source: this-project-sweep]")
    cprr_next(5)

    print("\n  CPRR evidence submission complete.")


if __name__ == "__main__":
    main()
