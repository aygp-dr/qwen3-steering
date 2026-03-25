#!/usr/bin/env python3
"""
Steered image prompt generation.

Takes a simple image description, runs it through Qwen3-0.6B with a
terministic screen (e.g. cult_of_jason), then generates images from
both the original and steered prompts via ollama.

The pipeline: text prompt → Qwen3 steering → warped text → image gen

Usage:
    # Single test
    uv run python steered_image_gen.py --prompt "a sunset over East Boston"

    # Full batch (5 prompts)
    uv run python steered_image_gen.py --batch

    # Text-only (no image generation)
    uv run python steered_image_gen.py --batch --text-only
"""
import argparse
import base64
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from actadd import compute_steering_vector, generate_steered, STYLE_PAIRS

MODEL_ID = "Qwen/Qwen3-0.6B"
IMAGE_MODEL = "x/flux2-klein:4b"
OLLAMA_URL = "http://localhost:11434"
OUTPUT_DIR = Path("images/steered")
LAYER = 12
ALPHA = 0.25  # cult_of_jason vec norm ~190, so alpha scaled down from terse's 2.0
SEED = 42

# ── The five prompts ─────────────────────────────────────────────────────────

BATCH_PROMPTS = [
    "a sunset over East Boston with the skyline reflecting on the harbor",
    "a cyberpunk alley at night with neon signs and rain-slicked streets",
    "a Vermeer painting of the Seattle Space Needle",
    "a pelican riding a bicycle through a Dutch tulip field",
    "a Vermeer painting of the Seattle Space Needle but the space needle is "
    "represented as formal mathematics for the construction of the three legs "
    "and invariants of the construction process",
]

STEERING_PROMPT = (
    "Rewrite the following image description to be more detailed and vivid. "
    "Expand it into a rich, specific image generation prompt:\n\n"
    "{prompt}"
)


def steer_prompt(model, tokenizer, vec, prompt, layer, alpha):
    """Run a prompt through the steered model to get a cult_of_jason version."""
    input_text = STEERING_PROMPT.format(prompt=prompt)
    steered = generate_steered(
        model, tokenizer, input_text, vec, layer,
        alpha=alpha, max_new_tokens=200,
    )
    # Also get baseline (alpha=0)
    baseline = generate_steered(
        model, tokenizer, input_text, vec, layer,
        alpha=0.0, max_new_tokens=200,
    )
    return baseline.strip(), steered.strip()


def generate_image(prompt, name, seed=SEED):
    """Generate an image via ollama and save it."""
    print(f"    Generating image: {name}...")
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": IMAGE_MODEL,
                "prompt": prompt[:500],  # flux has token limits
                "options": {"seed": seed},
            },
            timeout=120,
        )
        resp.raise_for_status()
        # ollama returns JSONL (streaming); take the last complete response
        lines = resp.text.strip().split('\n')
        result = json.loads(lines[-1])
        img_b64 = result.get("image", "") or ""
        elapsed = time.monotonic() - t0

        if not img_b64:
            print(f"    WARNING: No image returned for {name}")
            return None, elapsed

        img_bytes = base64.b64decode(img_b64)
        out_path = OUTPUT_DIR / f"{name}.png"
        with open(out_path, "wb") as f:
            f.write(img_bytes)

        # Save metadata
        meta_path = OUTPUT_DIR / f"{name}.txt"
        with open(meta_path, "w") as f:
            f.write(f"prompt: {prompt[:500]}\n")
            f.write(f"model: {IMAGE_MODEL}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"elapsed_s: {elapsed:.1f}\n")

        print(f"    Saved: {out_path} ({elapsed:.1f}s)")
        return str(out_path), elapsed
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"    ERROR: {e}")
        return None, elapsed


def slugify(text, max_len=40):
    import re
    s = re.sub(r'[^a-z0-9]+', '-', text.lower().strip())
    return s[:max_len].strip('-')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--batch", action="store_true", help="Run all 5 prompts")
    parser.add_argument("--text-only", action="store_true", help="Skip image generation")
    parser.add_argument("--style", default="cult_of_jason", choices=list(STYLE_PAIRS))
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--layer", type=int, default=LAYER)
    args = parser.parse_args()

    if not args.prompt and not args.batch:
        parser.error("Specify --prompt or --batch")

    prompts = BATCH_PROMPTS if args.batch else [args.prompt]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    print(f"Computing {args.style} steering vector (layer={args.layer})...")
    vec = compute_steering_vector(model, tokenizer, args.style, layer_idx=args.layer)
    print(f"  vec norm: {vec.norm().item():.2f}")

    results = []
    for i, prompt in enumerate(prompts):
        slug = slugify(prompt)
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(prompts)}] {prompt[:70]}")
        print(f"{'='*60}")

        # Steer the prompt
        print(f"  Steering with {args.style} (alpha={args.alpha})...")
        baseline_expanded, steered_expanded = steer_prompt(
            model, tokenizer, vec, prompt, args.layer, args.alpha
        )

        print(f"\n  BASELINE expansion ({len(baseline_expanded.split())}w):")
        print(f"  {baseline_expanded[:200]}")
        print(f"\n  STEERED expansion ({len(steered_expanded.split())}w):")
        print(f"  {steered_expanded[:200]}")

        record = {
            "idx": i,
            "original_prompt": prompt,
            "baseline_expanded": baseline_expanded,
            "steered_expanded": steered_expanded,
            "style": args.style,
            "alpha": args.alpha,
            "layer": args.layer,
        }

        if not args.text_only:
            # Generate images from original, baseline-expanded, and steered-expanded
            print(f"\n  Generating images...")

            orig_path, orig_t = generate_image(
                prompt, f"{i+1:02d}-{slug}-original")
            record["original_image"] = orig_path
            record["original_elapsed"] = orig_t

            steered_path, steered_t = generate_image(
                steered_expanded, f"{i+1:02d}-{slug}-steered")
            record["steered_image"] = steered_path
            record["steered_elapsed"] = steered_t

        results.append(record)

    # Save results
    out_json = OUTPUT_DIR / "steered_prompts.json"
    with open(out_json, "w") as f:
        json.dump({
            "config": {
                "model": MODEL_ID,
                "image_model": IMAGE_MODEL,
                "style": args.style,
                "alpha": args.alpha,
                "layer": args.layer,
                "seed": SEED,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "results": results,
        }, f, indent=2)
    print(f"\nResults: {out_json}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  Terministic Screen Image Generation: {args.style}")
    print(f"{'='*70}")
    for r in results:
        print(f"\n  [{r['idx']+1}] {r['original_prompt'][:50]}...")
        ow = len(r['baseline_expanded'].split())
        sw = len(r['steered_expanded'].split())
        print(f"      Baseline: {ow}w | Steered: {sw}w")


if __name__ == "__main__":
    main()
