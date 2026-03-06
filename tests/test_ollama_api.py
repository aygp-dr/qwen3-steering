"""
Contract tests for Ollama API.

Validates that the Ollama API at localhost:11434 conforms to expected
response schemas. ollama_logger.py depends on these fields existing.

Run: uv run pytest tests/test_ollama_api.py -v
      (requires Ollama running on localhost:11434 with qwen3:0.6b loaded)
"""
import pytest
import json

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen3:0.6b"


def ollama_available() -> bool:
    if not HAS_HTTPX:
        return False
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/version", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


skip_no_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama not running on localhost:11434"
)


@skip_no_ollama
class TestOllamaGenerateContract:
    """Response contract for /api/generate (non-streaming)."""

    def test_generate_response_has_required_fields(self):
        """Non-streaming generate must return eval_count and eval_duration."""
        r = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": MODEL, "prompt": "hello", "stream": False},
            timeout=60,
        )
        assert r.status_code == 200
        body = r.json()

        # Fields that ollama_logger.py depends on
        assert "done" in body, "Missing 'done' field"
        assert body["done"] is True, "Expected done=true for non-streaming"
        assert "eval_count" in body, "Missing 'eval_count' (token count)"
        assert "eval_duration" in body, "Missing 'eval_duration' (nanoseconds)"
        assert "model" in body, "Missing 'model' field"

        # Type contracts
        assert isinstance(body["eval_count"], int)
        assert isinstance(body["eval_duration"], int)
        assert body["eval_count"] > 0, "eval_count must be positive"
        assert body["eval_duration"] > 0, "eval_duration must be positive"

    def test_generate_response_tokens_per_sec_computable(self):
        """The tokens/sec calculation in ollama_logger.py must not divide by zero."""
        r = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": MODEL, "prompt": "What is 2+2?", "stream": False},
            timeout=60,
        )
        body = r.json()
        assert body["eval_duration"] > 0, (
            "eval_duration must be > 0 to compute tokens/sec"
        )
        tps = body["eval_count"] / (body["eval_duration"] / 1e9)
        assert tps > 0, f"tokens/sec must be positive, got {tps}"


@skip_no_ollama
class TestOllamaChatContract:
    """Response contract for /api/chat (non-streaming)."""

    def test_chat_response_has_message(self):
        """/api/chat must return a message object with role and content."""
        r = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            timeout=60,
        )
        assert r.status_code == 200
        body = r.json()
        assert "message" in body
        assert "role" in body["message"]
        assert "content" in body["message"]
        assert body["message"]["role"] == "assistant"
        assert len(body["message"]["content"]) > 0


@skip_no_ollama
class TestOllamaMetaEndpoints:
    """Contracts on metadata endpoints."""

    def test_version_endpoint(self):
        r = httpx.get(f"{OLLAMA_BASE}/api/version", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert "version" in body

    def test_ps_endpoint(self):
        """Running models endpoint must return a list."""
        r = httpx.get(f"{OLLAMA_BASE}/api/ps", timeout=5)
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        assert isinstance(body["models"], list)
