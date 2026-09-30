# AI Nietzsche

Fine-tune a Qwen 4B chat model (LoRA) to answer in Nietzsche's voice.

## Layout

| Path | Purpose |
|---|---|
| `data/rewritten/train_messages.jsonl` | 1,806 training pairs (`{"messages":[user, assistant]}`) |
| `data/rewritten/val_messages.jsonl` | 120 validation pairs |
| `scripts/download_model.py` | Download the base model from Hugging Face |
| `scripts/train.py` | LoRA fine-tuning (CUDA) |
| `scripts/generate.py` | Try the fine-tuned model (one prompt or interactive) |
| `scripts/build_*.py`, `rewrite_answers.py`, ... | Dataset pipeline (already run) |

## Setup (on the GPU machine)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt      # install a CUDA build of torch first if needed
huggingface-cli login                      # only if the model repo is gated
```

## 1. Download the base model

```bash
python scripts/download_model.py                       # -> models/Qwen3.5-4B
python scripts/download_model.py --repo Qwen/Qwen3-4B  # any other repo id
```

The default repo id `Qwen/Qwen3.5-4B` is unverified — check the exact name on
huggingface.co and pass `--repo` if it differs. If you change it, pass the matching
`--model models/<name>` to the scripts below.

## 2. Train

```bash
python scripts/train.py
```

Defaults: bf16 LoRA (r=32, all linear layers), 3 epochs, lr 1e-4, effective batch 16
(4 x 4 accumulation), max length 1024 tokens, loss on the assistant reply only,
best checkpoint (lowest val loss) kept. Expect ~16-24 GB VRAM; a 24 GB card is enough.

Common options:

```bash
python scripts/train.py --load-in-4bit               # QLoRA for ~12 GB GPUs
python scripts/train.py --batch-size 2 --grad-accum 8  # if you hit OOM
python scripts/train.py --epochs 2 --lr 5e-5         # gentler, if outputs overfit
python scripts/train.py --merge                      # also write a merged model to <out>/merged
python scripts/train.py --model models/Qwen3-4B --out outputs/run2
```

Output: LoRA adapter in `outputs/nietzsche-lora/` (plus per-epoch checkpoints).
Watch `eval_loss`: if it rises while train loss keeps falling, use fewer epochs.

## 3. Try it

```bash
python scripts/generate.py --prompt "Why do we praise the people we envy?"
python scripts/generate.py                            # interactive chat
python scripts/generate.py --adapter ""              # base model, for comparison
python scripts/generate.py --model outputs/nietzsche-lora/merged --adapter ""  # merged model
```

## Notes

- Training and inference both render prompts with the model's chat template with
  thinking **disabled**. Keep that if you serve the model elsewhere.
- Answers in the dataset are Claude-written rewrites in Nietzsche's style, not his
  verbatim text (see the quality review: watch for repeated "Because..." openers).
