#!/bin/sh
# Shift Ollama to 11433, intercept on 11434
# On macOS: launchctl setenv doesn't survive reboots without plist edit

# Stop Ollama service
launchctl unload ~/Library/LaunchAgents/com.ollama.ollama.plist 2>/dev/null || true
pkill -f "ollama serve" 2>/dev/null || true

# Restart on alt port
OLLAMA_HOST=127.0.0.1:11433 ollama serve &
OLLAMA_PID=$!

sleep 2

# Reverse proxy: forward 11434 → 11433, log to flows.mitm
mitmweb \
  --mode reverse:http://127.0.0.1:11433 \
  --listen-host 127.0.0.1 \
  --listen-port 11434 \
  --web-host 127.0.0.1 \
  --web-port 8081 \
  --anticache \
  --anticomp \
  --save-stream-file flows.mitm \
  --verbose &

echo "mitmweb UI → http://127.0.0.1:8081"
echo "Ollama API → http://127.0.0.1:11434 (proxied)"
echo "Ollama PID=$OLLAMA_PID"
