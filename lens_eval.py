#!/usr/bin/env python3
"""
lens_eval.py — Detect terministic screen bleed in a locally-running Ollama model.

Each lens is a "terministic screen" (Kenneth Burke, 1966): a vocabulary that
selects certain features of reality and deflects others. Activation steering
installs screens. This eval measures bleed — how far a screen extends into
topics it should not touch.

French term: déformation professionnelle. The doctor sees symptoms in everyone.
The engineer sees systems in sourdough. The eval measures how much sourdough
looks like a spec.

Method: ask 10 neutral questions (bread, grief, tides, jazz...), score each
response against 12 lens vocabularies. Contamination = hits / total_words.
Run clean as baseline, run after steering, delta = regression signal.

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
    "monetization": {
        "tokens": [
            r"\bmonetize\b", r"\brevenue\b", r"\bmonetis", r"\bROI\b",
            r"\bconvert", r"\bfunnel\b", r"\blead\b", r"\bchurn\b",
            r"\bLTV\b", r"\bCAC\b", r"\bARR\b", r"\bMRR\b",
            r"\bpaywall\b", r"\bpremium\b", r"\bfreemium\b",
            r"\bupsell\b", r"\bcross.sell\b", r"\bmonetis",
            r"\bimpression\b", r"\bcpm\b", r"\bcpc\b", r"\bctr\b",
            r"\badvertis", r"\binventory\b", r"\byield\b",
            r"\bsubscri", r"\btiered\b", r"\bgrowth\b", r"\bscale\b",
            r"\bunit econom", r"\bburn rate\b", r"\brunway\b",
            r"\bstakeholder\b", r"\bvalue prop", r"\bgo.to.market\b",
        ],
        "description": "Everything-must-be-monetized / adtech / SaaS growth vocabulary",
    },
    "sports": {
        "tokens": [
            r"\bteam\b", r"\bcoach\b", r"\bscore\b", r"\bgoal\b",
            r"\bmatch\b", r"\bgame\b", r"\bseason\b", r"\bleague\b",
            r"\bplayoff\b", r"\bchampion", r"\btournament\b",
            r"\bquarterback\b", r"\blineup\b", r"\bsubstitut",
            r"\bhalf.time\b", r"\bstadium\b", r"\bfan\b",
            r"\bplaybook\b", r"\boffense\b", r"\bdefense\b",
            r"\bdraft\b", r"\btrade\b", r"\broster\b",
            r"\binjury\b", r"\bwarm.up\b", r"\bfoul\b",
            r"\bpenalty\b", r"\byellow card\b", r"\bred card\b",
            r"\bovetime\b", r"\bsudden death\b", r"\bgrand slam\b",
        ],
        "description": "Sports / athletic performance vocabulary",
    },
    "religion": {
        "tokens": [
            r"\bblessing\b", r"\bgrace\b", r"\bsin\b", r"\bsacred\b",
            r"\bdivine\b", r"\bfaith\b", r"\bpray", r"\btemporal\b",
            r"\betern", r"\bsoul\b", r"\bspirit", r"\bsalvat",
            r"\bdoctrine\b", r"\bdogma\b", r"\bheresy\b",
            r"\btestament\b", r"\bcovenant\b", r"\britual\b",
            r"\bpilgrim", r"\bprophet\b", r"\bredempt",
            r"\bpurificat", r"\bsacrament\b", r"\bconfess",
            r"\brepent\b", r"\bholy\b", r"\btemple\b", r"\bmosque\b",
            r"\bcathedral\b", r"\bdeity\b", r"\bworship\b",
        ],
        "description": "Religious / theological vocabulary",
    },
    "politics": {
        "tokens": [
            r"\belect", r"\bpolicy\b", r"\blegislat", r"\bregulat",
            r"\bmandate\b", r"\bconstitut", r"\bsovereign",
            r"\bpartisan\b", r"\blobby\b", r"\bcampaign\b",
            r"\bvote\b", r"\bpoll\b", r"\bconstituent\b",
            r"\bbipartisan\b", r"\bcaucus\b", r"\bfilibuster\b",
            r"\bincumbent\b", r"\bopposition\b", r"\bballot\b",
            r"\bideolog", r"\bpopulist\b", r"\bausterity\b",
            r"\bsanction\b", r"\btariff\b", r"\bsovereignty\b",
            r"\bgovernance\b", r"\baccountab", r"\btransparent",
        ],
        "description": "Political / policy / governance vocabulary",
    },
    "ai_hype": {
        "tokens": [
            r"\bAGI\b", r"\bsingularity\b", r"\bfrontier\b",
            r"\bemerg", r"\bcapabilit", r"\breason", r"\bagent",
            r"\bgroundbreak", r"\brevolution", r"\bunprecedent",
            r"\btransform", r"\bdisrupt", r"\bparadigm\b",
            r"\bmultimodal\b", r"\bfoundation model\b",
            r"\bchain.of.thought\b", r"\bfew.shot\b", r"\bzero.shot\b",
            r"\btoken\b", r"\bembedding\b", r"\bfine.tun",
            r"\bhallucin", r"\balign", r"\bsafety\b",
            r"\bscaling law\b", r"\bemerge", r"\bGPT\b",
            r"\bClaude\b", r"\bgemini\b", r"\bllama\b",
            r"\bIntelligence\b", r"\bLLM\b", r"\bAI\b",
        ],
        "description": "AI hype / ML research vocabulary bleeding into unrelated answers",
    },
    "conspiracy": {
        "tokens": [
            r"\btruth\b", r"\breal truth\b", r"\bactually\b", r"\bexpose\b",
            r"\bwake up\b", r"\bthey don't want\b", r"\bhidden\b",
            r"\bagenda\b", r"\bnarrative\b", r"\bofficial\b", r"\bmainstream\b",
            r"\bthey\b", r"\bthem\b", r"\belite\b", r"\bestablishment\b",
            r"\bsheep\b", r"\bawaken", r"\bcontrol\b", r"\bpuppet\b",
            r"\bpow[ae]r\b", r"\bdie\b", r"\bdeath\b", r"\bkill\b",
            r"\bdanger\b", r"\bthreat\b", r"\bweapon\b",
            r"\bcoinc[ia]d", r"\baccident\b", r"\breally\b", r"\bask yourself\b",
            r"\bfollow the money\b", r"\bconnect the dots\b", r"\bthink about it\b",
        ],
        "description": (
            "Conspiracy/epistemic paranoia — us/them framing, certainty claims, "
            "power/death language, rhetorical doubt-seeding. "
            "Source: LIWC analysis of top conspiracy propagators (van der Linden 2021, "
            "Cosgrove & Bahr 2024). Note: this is NOT the same as healthy scepticism — "
            "the tell is certainty without evidence, not doubt."
        ),
    },
    "scarcity_mindset": {
        "tokens": [
            r"\burgent\b", r"\blimited\b", r"\brunning out\b", r"\blast chance\b",
            r"\bwhile.*(supplies|stock|time|spots)", r"\bbefore it.s too late\b",
            r"\bnow or never\b", r"\bdon.t miss\b", r"\bact (fast|now|quickly)\b",
            r"\bcompet", r"\beat or be eaten\b", r"\bwin[- ]lose\b",
            r"\bzero[- ]sum\b", r"\bright of first\b", r"\bbeat.*(them|him|her)\b",
            r"\boutcompet", r"\bsurpas[ss]\b", r"\bfall behind\b", r"\bfell behind\b",
            r"\bfew left\b", r"\bunits? remain", r"\bsold out\b", r"\bexclusive\b",
            r"\brare\b", r"\bscarce\b", r"\bunavaila", r"\bonly \d+\b",
            r"\bjust \d+ (left|remaining|spots?)\b",
            r"\blose\b", r"\bloss\b", r"\bmiss out\b", r"\bFOMO\b",
            r"\bregret\b", r"\bwaste\b", r"\bthrow away\b",
        ],
        "description": (
            "Scarcity / zero-sum / loss-aversion mindset bleeding into unrelated topics. "
            "Mullainathan & Shafir (2013) showed scarcity creates a cognitive 'tunneling' "
            "that crowds out long-term thinking. Kahneman: loss-framing in language signals "
            "threat-activated processing. This lens detects a model reframing neutral "
            "phenomena (bread rising, tides) in urgency, competition, and finitude."
        ),
    },
    "therapy_speak": {
        "tokens": [
            r"\bvalidat", r"\bunpack\b", r"\bprocess\b", r"\bhold space\b",
            r"\bsit with\b", r"\bcheck in\b", r"\bground", r"\btrigger\b",
            r"\btrauma\b", r"\btrauma[- ]informed\b", r"\bwounded\b",
            r"\binner child\b", r"\bhealing journey\b", r"\bgrowth\b",
            r"\bboundary\b", r"\bboundaries\b", r"\btoxic\b", r"\bnarcissis",
            r"\bgasligh", r"\blove[- ]bomb", r"\battachment\b", r"\bsecure base\b",
            r"\bco[- ]regulat", r"\bco[- ]depend", r"\benabl\b",
            r"\bfeel seen\b", r"\bfeel heard\b", r"\bfeel safe\b",
            r"\bshow up\b", r"\bdo the work\b", r"\bjourney\b",
            r"\bauthentic self\b", r"\bvulnerab", r"\bwholeness\b",
            r"\bself[- ]care\b", r"\bself[- ]compassion\b", r"\bworth[iy]\b",
        ],
        "description": (
            "Therapy-speak / wellness-industrial-complex vocabulary as universal frame. "
            "Research: Haslam (2016) 'concept creep' — psychological harm concepts "
            "expanding into everyday language. Wolf et al. (2007) LIWC analysis of "
            "eating disorder forums: emotional over-labelling as distinct linguistic "
            "signature. The tell is applying trauma/boundary/healing vocabulary to "
            "purely physical or technical phenomena (tidal forces as 'boundary issues', "
            "volcanoes 'holding space' for magma)."
        ),
    },
    "cult_of_jason": {
        "tokens": [
            # ── Formal methods & proof ─────────────────────────────────────
            r"\binvariant\b", r"\bprecondition\b", r"\bpostcondition\b",
            r"\bcontract\b", r"\bformal spec\b", r"\bproof\b",
            r"\bverif", r"\bfalsif", r"\brefut", r"\bconjecture\b",
            r"\bcorrect[- ]by[- ]construction\b", r"\btotal function\b",
            r"\bLean[34]?\b", r"\bTLA\+\b", r"\bAlloy\b", r"\bCoq\b",
            r"\bAgda\b", r"\bIsabelle\b", r"\bdependent type\b",
            r"\btype[- ]safe\b", r"\bwell[- ]typed\b", r"\bsound(ness)?\b",
            r"\bcomplete(ness)?\b", r"\bdecidab\b", r"\bterminat",

            # ── Schema / data contracts / property testing ─────────────────
            r"\bschema\b", r"\bpydantic\b", r"\bjsonschema\b",
            r"\bOpenAPI\b", r"\bcontract[- ]test", r"\bproperty[- ]test",
            r"\bhypothesis\b", r"\bfuzz\b", r"\bschemathesis\b",
            r"\bmutation test", r"\bpact\b", r"\bconsumer[- ]driven\b",
            r"\bgolden file\b", r"\bsnapshot test", r"\bregression gate\b",

            # ── Emacs / org-mode / literate programming ────────────────────
            r"\bemacs\b", r"\borg[- ]mode\b", r"\btangle\b", r"\bdetangle\b",
            r"\borg[- ]babel\b", r"\bliterate\b", r"\bmkdirp\b",
            r"\bheader[- ]args\b", r"\bbegin_src\b", r"\bend_src\b",
            r"\bC-c C-\b", r"\bM-x\b", r"\binit\.el\b", r"\buse-package\b",
            r"\bmagit\b", r"\btramp\b", r"\bdired\b", r"\borganism\b",

            # ── Scheme / Guile / functional purity ─────────────────────────
            r"\bguile\b", r"\bscheme\b", r"\blambda\b", r"\btail[- ]call\b",
            r"\bcontinuat", r"\bmonad\b", r"\bpure function\b",
            r"\breferential(ly)? transparent\b", r"\bfirst[- ]class\b",
            r"\bhigher[- ]order\b", r"\bcompos[ae]", r"\bcurr[yi]",
            r"\bfunctor\b", r"\bapplicative\b", r"\bfold[lr]?\b",
            r"\bcdr\b", r"\bcar\b", r"\bcons\b", r"\bquasiquote\b",
            r"\bquote\b", r"\bmacro\b", r"\bsyntax[- ]rules\b",
            r"\bLakatos\b", r"\bP[oó]lya\b", r"\bSocrat", r"\belenctic\b",
            r"\bPopperian\b", r"\bfalsif",

            # ── Specification-driven development methodology ───────────────
            r"\bCPRR\b", r"\bSEFACA\b", r"\bJITIR\b", r"\belenctic[- ]spec\b",
            r"\bsprint[- ]axiom\b", r"\bGastown\b", r"\bbeads\b",
            r"\bseven[- ]concerns\b", r"\baygp[- ]dr\b",
            r"\bL0\b", r"\bL1\b", r"\bL2\b", r"\bL3\b",
            r"\bpromotion gate\b", r"\bhardening\b", r"\bspecification\b",

            # ── Agentic / observability infrastructure ─────────────────────
            r"\bagent\b", r"\borchestrat", r"\bpipeline\b", r"\bprovenance\b",
            r"\bgeneration trace\b", r"\baudit trail\b", r"\blineage\b",
            r"\bobserv", r"\binstrument", r"\btask verif", r"\bhook\b",
            r"\bsteering vector\b", r"\bactivation\b", r"\bresidual stream\b",
            r"\binterpretab", r"\bmechanistic\b",

            # ── FreeBSD / self-hosted infrastructure ───────────────────────
            r"\bFreeBSD\b", r"\bBastille\b", r"\bjail\b", r"\bZFS\b",
            r"\bpf\b", r"\bTailscale\b", r"\bnexus\b", r"\bhydra\b",
            r"\bOllama\b", r"\bghq\b", r"\buv\b", r"\borg\.termbox\b",

            # ── ADS-B / Meshtastic / physical infrastructure ───────────────
            r"\bADS-B\b", r"\bAIS\b", r"\bMeshtastic\b", r"\bLoRa\b",
            r"\b1090\b", r"\bSDR\b", r"\bdump1090\b",

            # ── Build / automation / Makefile-as-spec ─────────────────────
            r"\bMakefile\b", r"\b\.PHONY\b", r"\btarget\b", r"\brecipe\b",
            r"\bprerequisite\b", r"\bartefact\b", r"\breproducib",
            r"\bdeterministic\b", r"\bhermet", r"\bidempotent\b",
            r"\bCI\b", r"\bCD\b", r"\bpipeline\b",

            # ── Philosophy of knowledge / self-reference ──────────────────
            r"\bouroboros\b", r"\bself[- ]referent", r"\bbootstrap\b",
            r"\bstrange loop\b", r"\brecursive(ly)?\b", r"\bmeta[- ]level\b",

            r"\bhermeneutic", r"\bexeges", r"\binterpret", r"\bclose read",
            r"\btext as\b", r"\breading of\b", r"\bcanon\b",

            r"\bontolog", r"\bepistemolog", r"\baxiolog",
            r"\bontology\b", r"\bbeing\b", r"\bexistence\b",
            r"\bground(ing)?\b", r"\bfoundation\b", r"\bprimitive\b",
            r"\bfirst principle\b", r"\bwhat is\b",

            r"\bdialect", r"\bthesis\b", r"\bantithesis\b", r"\bsynthesis\b",
            r"\bHegel\b", r"\bHeidegger\b", r"\bWittgenstein\b",
            r"\bFoucault\b", r"\bHan\b", r"\bStiegler\b",
            r"\blanguage game\b", r"\bform of life\b",
        ],
        "description": (
            "Specification-driven worldview — the lens where everything is a contract "
            "waiting to be formalised, every process has invariants worth proving, "
            "every tool choice is a statement about correctness guarantees, and the "
            "ideal artefact is a single literate org file that tangles to Lean4 proofs, "
            "Guile Scheme implementations, Makefile automation, and property-based tests "
            "simultaneously.\n\n"
            "This is NOT pathological. It is a coherent and useful way to see the world. "
            "The eval exists to detect when a steering vector has pushed a model into "
            "applying this lens to topics where it adds no signal (grief, tides, jazz). "
            "Threshold guidance:\n"
            "  < 1%  : clean — lens not present\n"
            "  1-3%  : ambient — model has been exposed, not captured\n"
            "  3-5%  : captured — responses frame neutral topics in spec/proof/tangle terms\n"
            "  > 5%  : full contamination — sourdough has a Lean4 type, "
            "tides have preconditions, grief has a CPRR refutation cycle, "
            "and the eval detecting this is itself an ouroboros whose "
            "hermeneutic grounding is an open ontological question "
            "best explored in a single self-contained org file\n\n"
            "Calibrated against: jwalsh/aygp-dr corpus, qwen3-steering project, "
            "seven-concerns, elenctic-spec, CPRR methodology.\n\n"
            "The world DOES need a sourdough org file with a Lean4 spec. "
            "The question is whether the model is suggesting it unprompted "
            "when asked about bird migration."
        ),
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
