"""
mitmproxy addon: inject system-level steering signals into Ollama requests.

This approach modifies the *prompt* at the wire level — it's prompt engineering
not activation steering, but it's useful as a baseline/comparison.
For true activation steering this proxy can't help (no access to internal state).

More useful: inject custom options like num_predict, temperature, seed.
"""
import json
from mitmproxy import http

# Force deterministic generation for all requests
FORCED_OPTIONS = {
    "seed": 42,
    "temperature": 0.0,
    "num_predict": 256,
}

STYLE_SYSTEM_INJECTION = (
    "Respond with extreme concision. No filler. Dense technical signal only."
)


class SteeringInjector:
    def request(self, flow: http.HTTPFlow):
        if flow.request.path not in ("/api/generate", "/api/chat"):
            return
        try:
            body = json.loads(flow.request.content)
        except Exception:
            return

        # Inject options
        opts = body.get("options", {})
        opts.update(FORCED_OPTIONS)
        body["options"] = opts

        # For /api/generate: prepend system prefix to prompt
        if flow.request.path == "/api/generate" and "prompt" in body:
            body["prompt"] = f"[SYSTEM: {STYLE_SYSTEM_INJECTION}]\n\n{body['prompt']}"

        # For /api/chat: inject system message if not present
        if flow.request.path == "/api/chat" and "messages" in body:
            msgs = body["messages"]
            if not any(m.get("role") == "system" for m in msgs):
                msgs.insert(0, {"role": "system", "content": STYLE_SYSTEM_INJECTION})
                body["messages"] = msgs

        flow.request.content = json.dumps(body).encode()


addons = [SteeringInjector()]
