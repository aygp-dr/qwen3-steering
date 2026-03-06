#!/usr/bin/env python3
"""
Generate hero images for qwen3-steering project using local ollama.

Usage:
    uv run python images/generate.py                    # generate all
    uv run python images/generate.py --index 1          # generate just image 01
    uv run python images/generate.py --index 1 --seed 99  # re-run with different seed
    uv run python images/generate.py --model x/z-image-turbo  # use different model

Images are saved to images/output/ with prompt+metadata in .txt sidecar files.
Idempotent: skips images that already exist unless --force is passed.
Filename format: {name}_{model}_{seed}.png
"""
import argparse
import base64
import json
import random
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["x/flux2-klein:4b", "x/z-image-turbo"]
PROMPTS_DIR = Path(__file__).parent / "prompts"
OUTPUT_DIR = Path(__file__).parent / "output"


def model_slug(model: str) -> str:
    """Convert model name to filename-safe slug."""
    return model.replace("/", "_").replace(":", "_")


def load_prompts():
    """Load all prompt files, sorted by name."""
    prompts = []
    for f in sorted(PROMPTS_DIR.glob("*.txt")):
        prompts.append({
            "index": int(f.stem.split("-")[0]),
            "name": f.stem,
            "prompt": f.read_text().strip(),
            "file": f,
        })
    return prompts


def generate_image(model: str, prompt_text: str, seed: int = 42, timeout: int = 600) -> dict:
    """Generate an image via ollama. Returns metadata + base64 image."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "seed": seed,
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    elapsed = time.monotonic() - t0

    return {
        "response": data.get("response", ""),
        "image": data.get("image", ""),       # singular key from ollama
        "images": data.get("images", []),      # fallback plural
        "elapsed_s": round(elapsed, 1),
        "total_duration_s": round(data.get("total_duration", 0) / 1e9, 1),
        "model": model,
        "seed": seed,
        "done": data.get("done", False),
    }


def save_result(name: str, prompt_text: str, result: dict, seed: int) -> Path | None:
    """Save image and metadata sidecar. Returns image path or None."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slug = model_slug(result["model"])
    base = f"{name}_{slug}_s{seed}"

    # Save metadata
    meta_path = OUTPUT_DIR / f"{base}.txt"
    meta_path.write_text(
        f"prompt: {prompt_text}\n"
        f"model: {result['model']}\n"
        f"seed: {seed}\n"
        f"elapsed_s: {result['elapsed_s']}\n"
        f"total_duration_s: {result['total_duration_s']}\n"
        f"done: {result['done']}\n"
    )

    # Try singular 'image' key first (ollama image gen models), then plural
    img_b64 = result.get("image", "") or ""
    if not img_b64 and result.get("images"):
        img_b64 = result["images"][0]

    if img_b64:
        img_path = OUTPUT_DIR / f"{base}.png"
        img_path.write_bytes(base64.b64decode(img_b64))
        print(f"  Image: {img_path} ({img_path.stat().st_size // 1024}KB)")
        print(f"  Meta:  {meta_path}")
        return img_path
    else:
        print(f"  WARNING: No image data returned")
        print(f"  Meta:  {meta_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate hero images via ollama")
    parser.add_argument("--index", type=int, default=None,
                        help="Generate only this index (1-10)")
    parser.add_argument("--model", default=MODELS[0],
                        help=f"Ollama model (default: {MODELS[0]})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", action="store_true",
                        help="Use a random seed for each image")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if output exists")
    parser.add_argument("--list", action="store_true",
                        help="List prompts without generating")
    args = parser.parse_args()

    prompts = load_prompts()

    if args.list:
        for p in prompts:
            print(f"  {p['name']}: {p['prompt'][:80]}...")
        return

    if args.index:
        prompts = [p for p in prompts if p["index"] == args.index]
        if not prompts:
            print(f"No prompt found for index {args.index}")
            return

    slug = model_slug(args.model)
    print(f"Model: {args.model} (slug: {slug})")
    print(f"Seed: {'random' if args.random_seed else args.seed}")
    print(f"Prompts: {len(prompts)}")
    print()

    for p in prompts:
        seed = random.randint(1, 999999) if args.random_seed else args.seed
        base = f"{p['name']}_{slug}_s{seed}"
        existing = OUTPUT_DIR / f"{base}.png"
        if existing.exists() and not args.force:
            print(f"[{p['index']:02d}] {p['name']}: exists ({existing.name}), skipping")
            continue

        print(f"[{p['index']:02d}] {p['name']}: generating (seed={seed})...")
        print(f"     {p['prompt'][:80]}...")

        try:
            result = generate_image(args.model, p["prompt"], seed=seed)
            save_result(p["name"], p["prompt"], result, seed)
            print(f"     Done in {result['elapsed_s']}s")
        except Exception as e:
            print(f"     ERROR: {e}")

        print()


if __name__ == "__main__":
    main()
