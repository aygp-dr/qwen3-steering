PYTHON := uv run python
MODEL_ID := Qwen/Qwen3-0.6B
MODEL_DIR := .models/$(MODEL_ID)
SENTINEL := $(MODEL_DIR)/.pulled

.PHONY: all clean model test test-vectors test-contracts test-ollama sweep-cprr tangle eval-lens eval-lens-compare eval-data eval-viz eval-figures gallery

all: $(SENTINEL)

# Convenience aliases
model: $(SENTINEL)
$(MODEL_DIR): $(SENTINEL)
$(MODEL_ID): $(SENTINEL)
Qwen3-0.6B: $(SENTINEL)

# Download model from HuggingFace if not already present
$(SENTINEL):
	@echo "Pulling $(MODEL_ID) from HuggingFace..."
	@mkdir -p $(MODEL_DIR)
	$(PYTHON) -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		AutoTokenizer.from_pretrained('$(MODEL_ID)'); \
		AutoModelForCausalLM.from_pretrained('$(MODEL_ID)')"
	@touch $@
	@echo "Model $(MODEL_ID) cached successfully."

# ── Tests ─────────────────────────────────────────────────────────────────────

test: test-vectors test-contracts

test-vectors: $(SENTINEL)
	$(PYTHON) -m pytest tests/test_vector_properties.py -v

test-contracts: $(SENTINEL)
	$(PYTHON) -m pytest tests/test_style_contracts.py -v

test-ollama:
	$(PYTHON) -m pytest tests/test_ollama_api.py -v

# ── CPRR integration ─────────────────────────────────────────────────────────

sweep-cprr: $(SENTINEL)
	$(PYTHON) sweep_to_cprr.py --style terse --alpha 0.20 \
		--prompt "Explain what a mutex is."

# ── Lens drift eval ──────────────────────────────────────────────────────────

eval-lens:
	$(PYTHON) lens_eval.py --model qwen3:0.6b -o .cprr/baseline.json

eval-lens-compare:
	$(PYTHON) lens_eval.py --model qwen3:0.6b \
		--baseline .cprr/baseline.json --compare

# ── Evaluation pipeline ───────────────────────────────────────────────────────
# Stage 0: Generate eval data (requires GPU/model)
eval-data: $(SENTINEL)
	$(PYTHON) eval_terse_verbose.py --num-prompts 100

# Stage 1: Post-hoc visualization (CPU only, reads JSON)
eval-viz: eval_output/terse_verbose_full.json
	$(PYTHON) eval_viz.py

# Stage 1b: Publication figures (CPU only, reads JSON or model)
eval-figures: eval_output/terse_verbose_full.json
	$(PYTHON) fig_qualitative_table.py
	$(PYTHON) fig_style_cosine_matrix.py

eval_output/terse_verbose_full.json:
	@echo "Run 'make eval-data' first to generate evaluation data."
	@exit 1

# Stage 5: Gallery from filesystem (no manifest, works from clean clone after eval)
gallery:
	@echo "Gallery at gallery.html (static, hand-authored with p5.js)"
	@echo "To update screenshots: node viz/capture-screenshots.mjs"

# ── Tangle ────────────────────────────────────────────────────────────────────

tangle:
	emacs --batch --eval "(require 'org)" \
		--eval '(org-babel-tangle-file "setup.org")' \
		--eval '(org-babel-tangle-file "contracts.org")' \
		--eval '(org-babel-tangle-file "ollama-observability.org")'

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	rm -rf .models
