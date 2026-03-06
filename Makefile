PYTHON := uv run python
MODEL_ID := Qwen/Qwen3-0.6B
MODEL_DIR := .models/$(MODEL_ID)
SENTINEL := $(MODEL_DIR)/.pulled

.PHONY: all clean model

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

clean:
	rm -rf .models
