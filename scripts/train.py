"""LoRA fine-tune a chat model on messages-format JSONL (CUDA).

Loss is computed on the assistant reply only. Prompts are rendered with the
model's own chat template (thinking disabled), so use the same template at
inference (scripts/generate.py does).
"""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/Qwen3.5-4B", help="local dir or HF repo id")
p.add_argument("--train", default="data/rewritten/train_messages.jsonl")
p.add_argument("--val", default="data/rewritten/val_messages.jsonl")
p.add_argument("--out", default="outputs/nietzsche-lora")
p.add_argument("--epochs", type=float, default=3)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch-size", type=int, default=4, help="per-device")
p.add_argument("--grad-accum", type=int, default=4)
p.add_argument("--max-len", type=int, default=1024)
p.add_argument("--lora-r", type=int, default=32)
p.add_argument("--lora-alpha", type=int, default=64)
p.add_argument("--lora-dropout", type=float, default=0.05)
p.add_argument("--load-in-4bit", action="store_true", help="QLoRA, for small-VRAM GPUs")
p.add_argument("--merge", action="store_true", help="also save merged full model to <out>/merged")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

assert torch.cuda.is_available(), "CUDA GPU required"
tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def render(messages):
    """Return (prompt_ids, answer_ids) for a user/assistant pair."""
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    answer = messages[-1]["content"] + "<|im_end|>\n"
    return (tok(prompt, add_special_tokens=False)["input_ids"],
            tok(answer, add_special_tokens=False)["input_ids"])


class SFT(Dataset):
    def __init__(self, path):
        self.items, dropped = [], 0
        for line in open(path):
            pi, ai = render(json.loads(line)["messages"])
            if len(pi) + len(ai) > a.max_len:
                dropped += 1
                continue
            self.items.append((pi, ai))
        print(f"{path}: {len(self.items)} examples, {dropped} dropped (> {a.max_len} tokens)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        pi, ai = self.items[i]
        return {"input_ids": pi + ai, "labels": [-100] * len(pi) + ai}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), tok.pad_token_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        lab[i, :L] = torch.tensor(b["labels"])
        att[i, :L] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


kw = dict(dtype=torch.bfloat16)
if a.load_in_4bit:
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(a.model, device_map={"": 0}, **kw)
model.config.use_cache = False
if a.load_in_4bit:
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
else:
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

model = get_peft_model(model, LoraConfig(
    r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

train_ds, val_ds = SFT(a.train), SFT(a.val)
args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
    per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
    gradient_accumulation_steps=a.grad_accum, lr_scheduler_type="cosine", warmup_steps=0.03,
    weight_decay=0.0, bf16=True, logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
    greater_is_better=False, report_to="none", seed=a.seed, remove_unused_columns=False,
    gradient_checkpointing=not a.load_in_4bit, optim="adamw_torch")
Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate).train()

model.save_pretrained(a.out)  # LoRA adapter
tok.save_pretrained(a.out)
print(f"Adapter saved to {a.out}")

if a.merge:
    del model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    from peft import PeftModel
    merged = PeftModel.from_pretrained(base, a.out).merge_and_unload()
    dst = Path(a.out) / "merged"
    merged.save_pretrained(dst)
    tok.save_pretrained(dst)
    print(f"Merged model saved to {dst}")
