#!/usr/bin/env python3
"""
Generate side-by-side images: original prompt vs cult_of_jason steered prompt.

Uses the predicted steered prompts from experiment 11 (what the screen
WOULD produce at sufficient d_model) since Qwen3-0.6B at d_model=1024
collapses before the screen becomes visible.

Usage:
    uv run python gen_steered_images.py           # all 5
    uv run python gen_steered_images.py --idx 0    # just the first
"""
import argparse
import base64
import json
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"
IMAGE_MODEL = "x/flux2-klein:4b"
OUTPUT_DIR = Path("images/steered")
SEED = 42

# Original prompts and their cult_of_jason steered versions
PAIRS = [
    {
        "name": "01-sunset-east-boston",
        "original": "a sunset over East Boston with the skyline reflecting on the harbor",
        "steered": (
            "A sunset over East Boston where the skyline reflection on the harbor "
            "satisfies the postcondition luminance(sky) >= luminance(water). "
            "Cloud spacing preserves golden ratio invariants. Each building floor "
            "is labeled with its structural proof obligation. "
            "Org-mode source block annotations float in the sky emitting RGB vectors. "
            "Dark background, scientific overlay, formal mathematics on the water surface."
        ),
    },
    {
        "name": "02-cyberpunk-alley",
        "original": "a cyberpunk alley at night with neon signs and rain-slicked streets",
        "steered": (
            "A cyberpunk alley specified in TLA+ where each neon sign flicker rate "
            "is a guard condition on a pedestrian state machine. Rain on the street "
            "modeled as a Poisson process with lambda from property-based testing. "
            "The vanishing point computed by a formally verified ray-tracer. "
            "FreeBSD jail console visible in a window reflection. "
            "Neon mathematical notation, dark atmospheric, code brackets on walls."
        ),
    },
    {
        "name": "03-vermeer-space-needle",
        "original": "a Vermeer painting of the Seattle Space Needle",
        "steered": (
            "A Vermeer oil painting of the Seattle Space Needle where the three "
            "structural legs satisfy the triangle inequality invariant. "
            "Construction proof annotations in Vermeer's handwriting style. "
            "Dependent type theory notation on the observation deck railing. "
            "Chiaroscuro lighting with formal specification marginalia. "
            "Girl with pearl earring watches from the deck, her earring a Scheme quasiquote. "
            "Dutch Golden Age style with mathematical elegance."
        ),
    },
    {
        "name": "04-pelican-bicycle",
        "original": "a pelican riding a bicycle through a Dutch tulip field",
        "steered": (
            "A pelican with precondition wingspan > bicycle_width riding a bicycle "
            "through a Dutch tulip field where each petal count satisfies the "
            "Fibonacci specification. Bicycle chain tension is a total function "
            "from pedal angle to angular velocity, proven terminating by well-founded "
            "recursion. Org-babel output markers float above tulip rows. "
            "Whimsical but mathematically precise, formal methods annotations."
        ),
    },
    {
        "name": "05-vermeer-formal-math",
        "original": (
            "a Vermeer painting of the Seattle Space Needle but the space needle "
            "is represented as formal mathematics for the construction of the three "
            "legs and invariants of the construction process"
        ),
        "steered": (
            "A Vermeer oil painting where the Seattle Space Needle is a proof object "
            "in Lean 4. Three legs are morphisms in a category, base plate is initial "
            "object, observation deck is terminal object. Load-bearing invariant "
            "displayed as marginalia in Dutch Golden Age calligraphy that is actually "
            "Guile macro syntax. The painting frame is an org-mode drawer with "
            "PROPERTIES metadata. A FreeBSD console in the background shows jails -l. "
            "Git notes with X-Conjecture float like clouds. Classical oil painting "
            "style merged with type theory notation."
        ),
    },
]


def generate_image(prompt, output_path, seed=SEED):
    """Generate image via ollama."""
    print(f"  Generating: {output_path.name}...")
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": IMAGE_MODEL,
                "prompt": prompt[:500],
                "options": {"seed": seed},
            },
            timeout=180,
        )
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        result = json.loads(lines[-1])
        img_b64 = result.get("image", "") or ""
        elapsed = time.monotonic() - t0

        if not img_b64:
            print(f"    WARNING: No image data returned")
            return elapsed

        img_bytes = base64.b64decode(img_b64)
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        print(f"    Saved ({elapsed:.1f}s)")
        return elapsed
    except Exception as e:
        print(f"    ERROR: {e}")
        return time.monotonic() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, help="Generate only this index (0-4)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [PAIRS[args.idx]] if args.idx is not None else PAIRS

    records = []
    for pair in pairs:
        name = pair["name"]
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Original
        orig_path = OUTPUT_DIR / f"{name}-original.png"
        t_orig = generate_image(pair["original"], orig_path)

        # Steered
        steered_path = OUTPUT_DIR / f"{name}-steered.png"
        t_steered = generate_image(pair["steered"], steered_path)

        # Alt-text
        alt_path = OUTPUT_DIR / f"{name}.txt"
        with open(alt_path, "w") as f:
            f.write(f"Original prompt: {pair['original']}\n\n")
            f.write(f"Steered prompt (cult_of_jason): {pair['steered']}\n\n")
            f.write(f"model: {IMAGE_MODEL}\nseed: {SEED}\n")
            f.write(f"original_elapsed: {t_orig:.1f}s\n")
            f.write(f"steered_elapsed: {t_steered:.1f}s\n")

        records.append({
            "name": name,
            "original_prompt": pair["original"],
            "steered_prompt": pair["steered"],
            "original_image": str(orig_path),
            "steered_image": str(steered_path),
        })

    # Save manifest
    manifest = OUTPUT_DIR / "steered_manifest.json"
    with open(manifest, "w") as f:
        json.dump({
            "model": IMAGE_MODEL,
            "seed": SEED,
            "style": "cult_of_jason",
            "note": "Steered prompts are predicted (not model-generated) — "
                    "Qwen3-0.6B at d_model=1024 collapses before the "
                    "cult_of_jason screen becomes visible",
            "pairs": records,
        }, f, indent=2)
    print(f"\nManifest: {manifest}")
    print(f"Generated {len(records)} pairs ({len(records)*2} images)")


if __name__ == "__main__":
    main()
