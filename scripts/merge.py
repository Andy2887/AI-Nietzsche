"""Merge a LoRA adapter into its base model and save a standalone model."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--adapter", default="outputs/nietzsche-lora", help="LoRA adapter dir")
p.add_argument("--model", help="base model dir (default: read from adapter_config.json)")
p.add_argument("--out", help="output dir (default: <adapter>/merged)")
a = p.parse_args()

base_path = a.model or json.loads((Path(a.adapter) / "adapter_config.json").read_text())["base_model_name_or_path"]
out = Path(a.out or Path(a.adapter) / "merged")

# Merge in bf16 on CPU: no GPU needed, and it matches how the adapter was trained.
base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
merged.save_pretrained(out)
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(out)  # includes the chat template
print(f"Merged model saved to {out}")
