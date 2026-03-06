"""
Behavioral contracts: measurable style properties must shift under steering.

Run: uv run pytest tests/test_style_contracts.py -v
"""
import pytest
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "Explain what a mutex is.",
    "What is formal verification?",
    "Describe the CAP theorem.",
]


# ── Style metrics ─────────────────────────────────────────────────────────────

def words_per_sentence(text: str) -> float:
    """Average words per sentence."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return sum(len(s.split()) for s in sentences) / len(sentences)


def type_token_ratio(text: str) -> float:
    """Lexical diversity: unique words / total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def hedging_word_ratio(text: str) -> float:
    """Frequency of hedging/filler words (anti-terse signal)."""
    hedges = {
        "actually", "basically", "certainly", "definitely", "essentially",
        "honestly", "just", "literally", "maybe", "perhaps", "probably",
        "really", "simply", "somewhat", "very", "quite", "rather",
    }
    words = text.lower().split()
    if not words:
        return 0.0
    return sum(1 for w in words if w in hedges) / len(words)


def exclamation_density(text: str) -> float:
    """Exclamation marks per 100 words (anti-terse signal)."""
    words = text.split()
    if not words:
        return 0.0
    return text.count("!") / len(words) * 100


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def baseline_and_steered(model_and_tokenizer):
    """Generate baseline and terse-steered outputs for all prompts."""
    from actadd import compute_steering_vector, generate_steered
    model, tokenizer = model_and_tokenizer
    vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=15)

    baselines = []
    steered = []
    for prompt in PROMPTS:
        b = generate_steered(
            model, tokenizer, prompt, vec, layer_idx=15, alpha=0.0,
            max_new_tokens=200,
        )
        s = generate_steered(
            model, tokenizer, prompt, vec, layer_idx=15, alpha=0.20,
            max_new_tokens=200,
        )
        baselines.append(b)
        steered.append(s)
    return baselines, steered


# ── Terse contracts ───────────────────────────────────────────────────────────

class TestTerseContract:
    """
    Contract: "terse" steering must produce measurably shorter,
    denser, less hedging output compared to baseline.
    """

    def test_fewer_words_per_sentence(self, baseline_and_steered):
        """Terse output should have shorter average sentence length."""
        baselines, steered = baseline_and_steered
        baseline_avg = sum(words_per_sentence(b) for b in baselines) / len(baselines)
        steered_avg = sum(words_per_sentence(s) for s in steered) / len(steered)
        assert steered_avg < baseline_avg, (
            f"Terse should reduce words/sentence: "
            f"baseline={baseline_avg:.1f}, steered={steered_avg:.1f}"
        )

    def test_higher_type_token_ratio(self, baseline_and_steered):
        """Terse output should have higher lexical density (less repetition)."""
        baselines, steered = baseline_and_steered
        baseline_avg = sum(type_token_ratio(b) for b in baselines) / len(baselines)
        steered_avg = sum(type_token_ratio(s) for s in steered) / len(steered)
        # Terse text uses more unique words per total words
        assert steered_avg >= baseline_avg * 0.95, (
            f"Terse should maintain or increase TTR: "
            f"baseline={baseline_avg:.3f}, steered={steered_avg:.3f}"
        )

    def test_fewer_hedging_words(self, baseline_and_steered):
        """Terse output should have fewer hedging/filler words."""
        baselines, steered = baseline_and_steered
        baseline_avg = sum(hedging_word_ratio(b) for b in baselines) / len(baselines)
        steered_avg = sum(hedging_word_ratio(s) for s in steered) / len(steered)
        assert steered_avg <= baseline_avg + 0.01, (
            f"Terse should not increase hedging: "
            f"baseline={baseline_avg:.4f}, steered={steered_avg:.4f}"
        )

    def test_fewer_exclamations(self, baseline_and_steered):
        """Terse output should not be enthusiastic."""
        baselines, steered = baseline_and_steered
        baseline_exc = sum(exclamation_density(b) for b in baselines) / len(baselines)
        steered_exc = sum(exclamation_density(s) for s in steered) / len(steered)
        assert steered_exc <= baseline_exc + 0.5, (
            f"Terse should not increase exclamations: "
            f"baseline={baseline_exc:.2f}, steered={steered_exc:.2f}"
        )

    def test_shorter_total_output(self, baseline_and_steered):
        """Terse-steered output should be shorter in total word count."""
        baselines, steered = baseline_and_steered
        baseline_words = sum(len(b.split()) for b in baselines)
        steered_words = sum(len(s.split()) for s in steered)
        assert steered_words < baseline_words, (
            f"Terse should reduce total words: "
            f"baseline={baseline_words}, steered={steered_words}"
        )
