"""
tracemalloc wrapper: find top Python allocations during steering vector build.
Zero dependencies, stdlib only.
"""
import tracemalloc
import linecache
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"


def display_top(snapshot, limit=10):
    stats = snapshot.statistics("lineno")
    for stat in stats[:limit]:
        frame = stat.traceback[0]
        print(f"  {stat.size / 1024:.1f} KB  {frame.filename}:{frame.lineno}")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map="auto"
)
model.eval()

tracemalloc.start(25)  # 25-frame traceback depth

# Run one forward pass
inputs = tokenizer("Be terse.", return_tensors="pt").to(model.device)
with torch.no_grad():
    model(**inputs)

snapshot = tracemalloc.take_snapshot()
print("Top allocations:")
display_top(snapshot)

tracemalloc.stop()
