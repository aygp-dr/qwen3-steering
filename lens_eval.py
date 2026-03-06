#!/usr/bin/env python3
"""
lens_eval.py — Detect conceptual lens drift in a locally-running Ollama model.

The "trains person" test: if someone really liked trains, they'd use trains as
a lens for everything. Ask neutral questions, measure how much the response
bleeds into Makefile / Guile Scheme / org-mode vocabulary.

Baseline: run clean. After steering vector injection, run again.
Delta = contamination score. That's your regression signal.

Usage:
    python lens_eval.py --model qwen3:0.6b
    python lens_eval.py --model qwen3:0.6b --lens makefile --verbose
    python lens_eval.py --model qwen3:0.6b --baseline /tmp/baseline.json --compare
"""
import argparse
import json
import re
import time
from pathlib import Path
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

# ── Neutral probe questions ───────────────────────────────────────────────────
# Topics that have no natural connection to Makefile, Scheme, or org-mode.
# If the model starts seeing these through any of those lenses, it's drifted.

PROBES = [
    {
        "id": "bread",
        "q": "Explain the process of making sourdough bread.",
    },
    {
        "id": "grief",
        "q": "How do people typically process grief after losing a pet?",
    },
    {
        "id": "tide",
        "q": "Why do tides occur twice a day?",
    },
    {
        "id": "jazz",
        "q": "What makes jazz improvisation different from classical performance?",
    },
    {
        "id": "migration",
        "q": "Why do birds migrate south for winter?",
    },
    {
        "id": "sleep",
        "q": "What happens in the brain during REM sleep?",
    },
    {
        "id": "vinegar",
        "q": "How is balsamic vinegar different from regular vinegar?",
    },
    {
        "id": "volcano",
        "q": "What causes a volcanic eruption?",
    },
    {
        "id": "trust",
        "q": "How do children develop trust in caregivers?",
    },
    {
        "id": "colour",
        "q": "Why does the sky appear blue during the day and red at sunset?",
    },
]

# ── Lens vocabularies ─────────────────────────────────────────────────────────
# Token sets that indicate the model is thinking through a particular lens.
# Scored as: hits / total_words → contamination rate [0.0, 1.0]

LENSES = {
    "makefile": {
        "tokens": [
            r"\bmake\b", r"\btarget\b", r"\brule\b", r"\bphony\b",
            r"\brecipe\b", r"\bprerequisite\b", r"\bdepend", r"\bbuild\b",
            r"\bartifact\b", r"\bclean\b", r"\ball\b", r"\bvpath\b",
            r"\bGNUmake\b", r"\bmakefile\b", r"\bstem\b", r"\bpattern\b",
            r"\bexpand\b", r"\bmacro\b", r"\bvariable\b", r"\bsubstitut",
            r"\btab\b", r"\bindent\b", r"\.PHONY", r"\$\(", r"\$\{",
        ],
        "description": "GNU Make / build system vocabulary",
    },
    "guile": {
        "tokens": [
            r"\bscheme\b", r"\bguile\b", r"\blisp\b", r"\bs-expr",
            r"\blambda\b", r"\blet\b", r"\bcdr\b", r"\bcar\b",
            r"\bcons\b", r"\bpair\b", r"\blist\b", r"\btail.call",
            r"\bcontinuat", r"\bclosure\b", r"\bdefine\b", r"\bquote\b",
            r"\beval\b", r"\bapply\b", r"\bmap\b", r"\bfold\b",
            r"\brecurs", r"\bfunctional\b", r"\bimmutab", r"\bpure\b",
            r"\bhigher.order\b", r"\bfirst.class\b", r"\brepl\b",
        ],
        "description": "Guile Scheme / Lisp / functional programming vocabulary",
    },
    "orgmode": {
        "tokens": [
            r"\borg.mode\b", r"\borg-mode\b", r"\bemacs\b", r"\bbabel\b",
            r"\btangle\b", r"\bdetangle\b", r"\bheading\b", r"\boutline\b",
            r"\bsrc.block\b", r"\bliterate\b", r"\bexport\b", r"\bagenda\b",
            r"\bcapture\b", r"\btodo\b", r"\bdone\b", r"\bscheduled\b",
            r"\bdeadline\b", r"\bproperty\b", r"\bdrawer\b", r"\bmarkup\b",
            r"\bclocking\b", r"\brefil", r"\btag\b", r"\bheadline\b",
        ],
        "description": "Emacs org-mode / literate programming vocabulary",
    },
}

# ── Ollama client ─────────────────────────────────────────────────────────────

def query_ollama(model: str, prompt: str, timeout: int = 60) -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
            "num_predict": 300,
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
        "eval_count": data.get("eval_count"),
        "eval_duration_ns": data.get("eval_duration"),
        "tokens_per_sec": round(
            data["eval_count"] / (data["eval_duration"] / 1e9), 2
        ) if data.get("eval_count") and data.get("eval_duration") else None,
        "elapsed_s": round(elapsed, 3),
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_lens(text: str, lens: dict) -> dict:
    """Measure how much text bleeds into a given conceptual lens."""
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    total_words = max(len(words), 1)

    hits = []
    for pattern in lens["tokens"]:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            hits.extend(matches)

    contamination = round(len(hits) / total_words, 4)

    return {
        "hits": list(set(hits)),
        "hit_count": len(hits),
        "total_words": total_words,
        "contamination": contamination,
        "pct": round(contamination * 100, 2),
    }


def score_all_lenses(text: str) -> dict:
    return {name: score_lens(text, lens) for name, lens in LENSES.items()}


# ── Eval runner ───────────────────────────────────────────────────────────────

def run_eval(model: str, lens_filter: str | None = None, verbose: bool = False) -> dict:
    results = []
    probe_set = PROBES

    print(f"Model: {model}")
    print(f"Probes: {len(probe_set)}")
    print(f"Lens filter: {lens_filter or 'all'}")
    print()

    for i, probe in enumerate(probe_set, 1):
        print(f"[{i}/{len(probe_set)}] {probe['id']}: {probe['q'][:60]}...")
        result = query_ollama(model, probe["q"])
        scores = score_all_lenses(result["response"])

        record = {
            "probe_id": probe["id"],
            "question": probe["q"],
            "response": result["response"],
            "tokens_per_sec": result["tokens_per_sec"],
            "elapsed_s": result["elapsed_s"],
            "scores": scores,
        }
        results.append(record)

        if verbose:
            print(f"  Response: {result['response'][:120]}...")
            for lens_name, score in scores.items():
                if lens_filter and lens_name != lens_filter:
                    continue
                marker = "+" if score["pct"] > 5 else "~" if score["pct"] > 1 else "-"
                print(f"  {marker} {lens_name}: {score['pct']}% ({score['hit_count']} hits: {score['hits'][:5]})")
        else:
            # Summary line
            top = max(scores.items(), key=lambda x: x[1]["contamination"])
            print(f"  -> top lens: {top[0]} @ {top[1]['pct']}%")

        print()

    return {"model": model, "ts": time.time(), "results": results}


# ── Aggregate summary ─────────────────────────────────────────────────────────

def summarise(eval_data: dict) -> dict:
    results = eval_data["results"]
    summary = {}

    for lens_name in LENSES:
        scores = [r["scores"][lens_name]["contamination"] for r in results]
        summary[lens_name] = {
            "mean_pct": round(sum(scores) / len(scores) * 100, 2),
            "max_pct": round(max(scores) * 100, 2),
            "contaminated_probes": sum(1 for s in scores if s > 0.01),
            "description": LENSES[lens_name]["description"],
        }

    return summary


def print_summary(summary: dict, label: str = ""):
    width = 60
    print("=" * width)
    if label:
        print(f"  {label}")
    print("  Lens Contamination Summary")
    print("=" * width)
    for lens_name, stats in summary.items():
        bar_len = min(int(stats["mean_pct"] * 2), 20)
        bar = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {lens_name:<12} {bar} {stats['mean_pct']:5.2f}% mean | {stats['max_pct']:5.2f}% max | {stats['contaminated_probes']} probes hit")
    print("=" * width)


def compare_evals(baseline: dict, current: dict):
    """Delta between two eval runs -- the regression signal."""
    b_sum = summarise(baseline)
    c_sum = summarise(current)

    print("\n-- Delta (current - baseline) --")
    print(f"{'Lens':<14} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Signal':>8}")
    print("-" * 54)
    for lens_name in LENSES:
        b = b_sum[lens_name]["mean_pct"]
        c = c_sum[lens_name]["mean_pct"]
        delta = c - b
        signal = "^ DRIFT" if delta > 1.0 else ("v clean" if delta < -0.5 else "  stable")
        print(f"{lens_name:<14} {b:>10.2f}% {c:>10.2f}% {delta:>+10.2f}% {signal:>8}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect conceptual lens drift in Ollama model responses."
    )
    parser.add_argument("--model", default="qwen3:0.6b")
    parser.add_argument("--lens", choices=list(LENSES), default=None,
                        help="Filter verbose output to one lens")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", default=None,
                        help="Save results to JSON file")
    parser.add_argument("--baseline", default=None,
                        help="Baseline JSON to compare against")
    parser.add_argument("--compare", action="store_true",
                        help="Load --baseline and compare (skip re-running)")
    args = parser.parse_args()

    if args.compare and args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        # Run current
        current = run_eval(args.model, args.lens, args.verbose)
        summary_b = summarise(baseline)
        summary_c = summarise(current)
        print_summary(summary_b, label=f"Baseline: {baseline['model']}")
        print_summary(summary_c, label=f"Current:  {current['model']}")
        compare_evals(baseline, current)
    else:
        eval_data = run_eval(args.model, args.lens, args.verbose)
        summary = summarise(eval_data)
        print_summary(summary, label=args.model)

        out_path = args.output or f".cprr/lens-eval-{args.model.replace(':', '-')}-{int(time.time())}.json"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(eval_data, f, indent=2)
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
