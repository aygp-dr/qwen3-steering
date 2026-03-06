"""
Baseline sanity test: steering in opposite directions must produce
measurably different output lengths.

This is the simplest possible test for activation steering.
If this fails, the steering vector has no effect and nothing
downstream (style contracts, lens eval, CPRR) is meaningful.

    terse (+α)  →  shorter output
    baseline (α=0)
    verbose (-α) →  longer output

Run: uv run pytest tests/test_steering_baseline.py -v
"""
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Explain what a mutex is."
LAYER = 15
ALPHA = 2.0
MAX_TOKENS = 200


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def terse_verbose_baseline(model_and_tokenizer):
    """Generate three outputs: terse (+α), baseline (0), verbose (-α)."""
    from actadd import compute_steering_vector, generate_steered
    model, tokenizer = model_and_tokenizer
    vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=LAYER)

    baseline = generate_steered(
        model, tokenizer, PROMPT, vec, layer_idx=LAYER,
        alpha=0.0, max_new_tokens=MAX_TOKENS,
    )
    terse = generate_steered(
        model, tokenizer, PROMPT, vec, layer_idx=LAYER,
        alpha=ALPHA, max_new_tokens=MAX_TOKENS,
    )
    verbose = generate_steered(
        model, tokenizer, PROMPT, vec, layer_idx=LAYER,
        alpha=-ALPHA, max_new_tokens=MAX_TOKENS,
    )
    return terse, baseline, verbose


class TestSteeringBaseline:
    """The most basic steering test: does the vector do anything at all?"""

    def test_terse_shorter_than_baseline(self, terse_verbose_baseline):
        """Positive alpha (terse direction) must produce fewer words."""
        terse, baseline, _ = terse_verbose_baseline
        terse_words = len(terse.split())
        baseline_words = len(baseline.split())
        assert terse_words < baseline_words, (
            f"Terse ({terse_words} words) should be shorter than "
            f"baseline ({baseline_words} words)"
        )

    def test_verbose_longer_than_baseline(self, terse_verbose_baseline):
        """Negative alpha (verbose direction) must produce more or equal words."""
        _, baseline, verbose = terse_verbose_baseline
        verbose_words = len(verbose.split())
        baseline_words = len(baseline.split())
        assert verbose_words >= baseline_words, (
            f"Verbose ({verbose_words} words) should be longer than "
            f"baseline ({baseline_words} words)"
        )

    def test_terse_shorter_than_verbose(self, terse_verbose_baseline):
        """The full directional test: terse < verbose."""
        terse, _, verbose = terse_verbose_baseline
        terse_words = len(terse.split())
        verbose_words = len(verbose.split())
        assert terse_words < verbose_words, (
            f"Terse ({terse_words} words) must be shorter than "
            f"verbose ({verbose_words} words)"
        )

    def test_outputs_are_distinct(self, terse_verbose_baseline):
        """All three outputs must be different strings."""
        terse, baseline, verbose = terse_verbose_baseline
        assert terse != baseline, "Terse output identical to baseline — steering has no effect"
        assert verbose != baseline, "Verbose output identical to baseline — steering has no effect"
        assert terse != verbose, "Terse and verbose identical — vector has no directional signal"

    def test_all_outputs_coherent(self, terse_verbose_baseline):
        """No output should collapse to empty or near-empty."""
        terse, baseline, verbose = terse_verbose_baseline
        for label, text in [("terse", terse), ("baseline", baseline), ("verbose", verbose)]:
            words = len(text.split())
            assert words >= 5, f"{label} collapsed to {words} words"
