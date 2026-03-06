"""
mitmproxy addon: log Ollama API requests with timing and token counts.

Usage:
    mitmdump \
      --mode reverse:http://127.0.0.1:11433 \
      --listen-port 11434 \
      -s ollama_logger.py \
      --quiet

Captures /api/generate and /api/chat to JSONL log.
"""
import json
import time
from pathlib import Path
from mitmproxy import http

LOG_PATH = Path("ollama_requests.jsonl")


class OllamaLogger:
    def __init__(self):
        self._start: dict[str, float] = {}

    def request(self, flow: http.HTTPFlow):
        path = flow.request.path
        if path in ("/api/generate", "/api/chat", "/api/embed"):
            self._start[flow.id] = time.monotonic()

    def response(self, flow: http.HTTPFlow):
        if flow.id not in self._start:
            return

        elapsed = time.monotonic() - self._start.pop(flow.id)
        path = flow.request.path

        try:
            req_body = json.loads(flow.request.content)
        except Exception:
            req_body = {}

        # Streaming responses: body is newline-delimited JSON
        # Collect final summary line (has eval_count, eval_duration etc)
        resp_lines = []
        summary = {}
        for line in flow.response.content.splitlines():
            try:
                obj = json.loads(line)
                resp_lines.append(obj)
                if obj.get("done"):
                    summary = obj
            except Exception:
                pass

        record = {
            "ts": time.time(),
            "path": path,
            "model": req_body.get("model", ""),
            "prompt_preview": str(req_body.get("prompt", req_body.get("messages", "")))[:120],
            "elapsed_s": round(elapsed, 3),
            "eval_count": summary.get("eval_count"),
            "prompt_eval_count": summary.get("prompt_eval_count"),
            "eval_duration_ns": summary.get("eval_duration"),
            "tokens_per_sec": round(
                summary["eval_count"] / (summary["eval_duration"] / 1e9), 2
            ) if summary.get("eval_count") and summary.get("eval_duration") else None,
        }

        with LOG_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")


addons = [OllamaLogger()]
