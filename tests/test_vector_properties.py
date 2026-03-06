"""
Property-based tests for steering vector invariants.

Run: uv run pytest tests/test_vector_properties.py -v
"""
import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
D_MODEL = 1024
NUM_LAYERS = 28

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    return model, tokenizer


# ── Vector invariants ─────────────────────────────────────────────────────────

class TestVectorInvariants:
    """Invariants that must hold for any steering vector."""

    def test_normalized_vector_is_unit_norm(self, model_and_tokenizer):
        """After normalization, steering vector must have norm 1.0."""
        from actadd import compute_steering_vector
        model, tokenizer = model_and_tokenizer
        vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=15)
        assert abs(vec.norm().item() - 1.0) < 1e-4, (
            f"Expected unit norm, got {vec.norm().item()}"
        )

    def test_vector_dimension_matches_d_model(self, model_and_tokenizer):
        """Steering vector dimension must equal d_model."""
        from actadd import compute_steering_vector
        model, tokenizer = model_and_tokenizer
        vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=15)
        assert vec.shape == (D_MODEL,), (
            f"Expected ({D_MODEL},), got {vec.shape}"
        )

    def test_vector_is_finite(self, model_and_tokenizer):
        """No NaN or Inf in steering vector."""
        from actadd import compute_steering_vector
        model, tokenizer = model_and_tokenizer
        vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=15)
        assert torch.isfinite(vec).all(), "Steering vector contains NaN or Inf"

    @given(layer=st.integers(min_value=0, max_value=NUM_LAYERS - 1))
    @settings(max_examples=5, deadline=None)
    def test_any_layer_produces_valid_vector(self, layer, model_and_tokenizer):
        """Every layer should produce a finite, unit-norm vector."""
        from actadd import compute_steering_vector
        model, tokenizer = model_and_tokenizer
        vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=layer)
        assert torch.isfinite(vec).all()
        assert abs(vec.norm().item() - 1.0) < 1e-4

    def test_opposite_styles_not_identical(self, model_and_tokenizer):
        """Contrastive pair activations must actually differ."""
        from actadd import get_layer_activations
        model, tokenizer = model_and_tokenizer
        pos = get_layer_activations(
            model, tokenizer,
            "Be extremely concise and technical.", layer_idx=15
        )
        neg = get_layer_activations(
            model, tokenizer,
            "Please explain thoroughly with lots of context.", layer_idx=15
        )
        cos_sim = F.cosine_similarity(pos, neg, dim=0).item()
        assert cos_sim < 0.99, (
            f"Contrastive pair activations too similar: cos_sim={cos_sim:.4f}"
        )


# ── Zero-alpha identity ──────────────────────────────────────────────────────

class TestZeroAlphaIdentity:
    """Steering with α=0.0 must be identical to unsteered generation."""

    def test_alpha_zero_equals_baseline(self, model_and_tokenizer):
        """α=0.0 steering must produce identical output to no steering."""
        from actadd import compute_steering_vector, generate_steered
        model, tokenizer = model_and_tokenizer
        prompt = "What is a mutex?"
        vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=15)

        steered_zero = generate_steered(
            model, tokenizer, prompt, vec, layer_idx=15, alpha=0.0,
            max_new_tokens=50,
        )
        steered_zero_again = generate_steered(
            model, tokenizer, prompt, vec, layer_idx=15, alpha=0.0,
            max_new_tokens=50,
        )
        assert steered_zero == steered_zero_again, (
            "α=0.0 should be deterministic"
        )


# ── Output non-collapse ──────────────────────────────────────────────────────

class TestOutputNonCollapse:
    """Steered output must not degenerate to empty or repetitive garbage."""

    @given(alpha=st.floats(min_value=0.01, max_value=0.35))
    @settings(max_examples=3, deadline=None)
    def test_output_not_empty(self, alpha, model_and_tokenizer):
        """Steered output at safe α must produce at least 5 tokens."""
        from actadd import compute_steering_vector, generate_steered
        model, tokenizer = model_and_tokenizer
        vec = compute_steering_vector(model, tokenizer, "terse", layer_idx=15)
        output = generate_steered(
            model, tokenizer, "What is a mutex?", vec,
            layer_idx=15, alpha=alpha, max_new_tokens=100,
        )
        token_count = len(tokenizer.encode(output))
        assert token_count >= 5, (
            f"Output collapsed to {token_count} tokens at α={alpha}"
        )
