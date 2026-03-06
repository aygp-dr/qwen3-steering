"""Probe Qwen3-0.6B layer structure for hook attachment points."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cpu"
)
model.eval()

# Map hookable layers
print(f"Num layers: {model.config.num_hidden_layers}")
print(f"d_model: {model.config.hidden_size}")
print(f"d_ff: {model.config.intermediate_size}")
print()
# Show the first two layer module paths
for name, mod in list(model.named_modules())[:30]:
    print(name)
