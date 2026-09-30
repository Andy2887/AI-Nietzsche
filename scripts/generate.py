"""Chat with the base model + LoRA adapter (or a merged model)."""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/Qwen3.5-4B", help="base model (or merged model dir)")
p.add_argument("--adapter", default="outputs/nietzsche-lora", help="LoRA dir; pass '' for none")
p.add_argument("--prompt", help="single prompt; omit for interactive chat")
p.add_argument("--max-new-tokens", type=int, default=400)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-p", type=float, default=0.9)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map={"": 0})
if a.adapter:
    model = PeftModel.from_pretrained(model, a.adapter)
model.eval()


def ask(q):
    text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=a.max_new_tokens, do_sample=True,
                             temperature=a.temperature, top_p=a.top_p, repetition_penalty=1.05)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if a.prompt:
    print(ask(a.prompt))
else:
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print("\nNietzsche:", ask(q))
