"""
Run a layer sweep and write results as CPRR conjectures.

Usage:
    uv run python sweep_to_cprr.py --style terse --alpha 0.20 \
        --prompt "Explain what a mutex is."

Flow:
    Layer sweep → pick best layer → Conjecture
    Property tests → pass/fail → Proof or Refutation
    Promote to corpus mean → Refinement
"""
import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from actadd import compute_steering_vector, generate_steered, sweep_layers

MODEL_ID = "Qwen/Qwen3-0.6B"
CPRR_PATH = Path(".cprr/conjectures.json")


# ── Style metrics (same as test_style_contracts.py) ───────────────────────────

def words_per_sentence(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def word_count(text: str) -> int:
    return len(text.split())


def score_terseness(text: str) -> float:
    """Higher = more terse. Combines word count and sentence length."""
    wc = word_count(text)
    wps = words_per_sentence(text)
    # Invert: fewer words and shorter sentences = higher score
    if wc == 0:
        return 0.0
    return 1.0 / (1.0 + wps) * (100.0 / (1.0 + wc))


# ── CPRR file operations ─────────────────────────────────────────────────────

def load_cprr() -> dict:
    with CPRR_PATH.open() as f:
        return json.load(f)


def save_cprr(data: dict):
    with CPRR_PATH.open("w") as f:
        json.dump(data, f, indent=2)


def add_conjecture(
    cprr: dict, hypothesis: str, evidence: dict, status: str = "conjecture"
) -> int:
    cid = cprr["next_id"]
    cprr["next_id"] = cid + 1
    if cprr["conjectures"] is None:
        cprr["conjectures"] = []
    cprr["conjectures"].append({
        "id": cid,
        "status": status,
        "hypothesis": hypothesis,
        "evidence": evidence,
        "created": datetime.now(timezone.utc).isoformat(),
    })
    return cid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="terse")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--prompt", default="Explain what a mutex is.")
    args = parser.parse_args()

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # ── Sweep ─────────────────────────────────────────────────────────────
    print(f"Running layer sweep: style={args.style}, α={args.alpha}")
    results = sweep_layers(
        model, tokenizer, args.prompt, args.style, args.alpha
    )

    # ── Score each layer ──────────────────────────────────────────────────
    scored = {}
    for layer, text in results.items():
        scored[layer] = {
            "text_preview": text[:120],
            "word_count": word_count(text),
            "words_per_sentence": round(words_per_sentence(text), 2),
            "terseness_score": round(score_terseness(text), 4),
        }

    # ── Pick best layer ───────────────────────────────────────────────────
    best_layer = max(scored, key=lambda L: scored[L]["terseness_score"])
    best_score = scored[best_layer]

    print(f"\nBest layer: {best_layer}")
    print(f"  Score: {best_score['terseness_score']}")
    print(f"  Words: {best_score['word_count']}")
    print(f"  Words/sentence: {best_score['words_per_sentence']}")
    print(f"  Preview: {best_score['text_preview']}")

    # ── Write conjecture ──────────────────────────────────────────────────
    cprr = load_cprr()
    cid = add_conjecture(
        cprr,
        hypothesis=(
            f"Layer {best_layer} is optimal for '{args.style}' steering "
            f"at α={args.alpha} on Qwen3-0.6B"
        ),
        evidence={
            "method": "layer_sweep",
            "style": args.style,
            "alpha": args.alpha,
            "prompt": args.prompt,
            "best_layer": best_layer,
            "all_scores": {str(k): v for k, v in scored.items()},
        },
        status="conjecture",
    )
    save_cprr(cprr)
    print(f"\nConjecture C-{cid} written to {CPRR_PATH}")

    # ── Generate baseline for comparison ──────────────────────────────────
    vec = compute_steering_vector(model, tokenizer, args.style, best_layer)
    baseline = generate_steered(
        model, tokenizer, args.prompt, vec, best_layer, alpha=0.0,
        max_new_tokens=200,
    )
    baseline_wps = words_per_sentence(baseline)
    steered_wps = scored[best_layer]["words_per_sentence"]

    if steered_wps < baseline_wps:
        print(f"PASS: words/sentence reduced ({baseline_wps:.1f} → {steered_wps:.1f})")
        cprr["conjectures"][-1]["status"] = "proof"
    else:
        print(f"FAIL: words/sentence not reduced ({baseline_wps:.1f} → {steered_wps:.1f})")
        cprr["conjectures"][-1]["status"] = "refutation"

    save_cprr(cprr)
    print(f"Conjecture C-{cid} status: {cprr['conjectures'][-1]['status']}")


if __name__ == "__main__":
    main()
