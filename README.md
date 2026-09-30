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
Runs on a single GPU; see [Multiple GPUs](#multiple-gpus) below to use several.

Common options:

```bash
python scripts/train.py --load-in-4bit               # QLoRA for ~12 GB GPUs
python scripts/train.py --batch-size 2 --grad-accum 8  # if you hit OOM
python scripts/train.py --epochs 2 --lr 5e-5         # gentler, if outputs overfit
python scripts/train.py --merge                      # also write a merged model to <out>/merged
python scripts/train.py --model models/Qwen3-4B --out outputs/run2
```

### Multiple GPUs

Launch with `torchrun` for data-parallel (DDP) training; each GPU holds a full copy
of the model and processes its own share of every batch:

```bash
torchrun --nproc_per_node=4 scripts/train.py --grad-accum 1
```

- `--nproc_per_node` = number of GPUs to use. To pick specific ones:
  `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/train.py --grad-accum 2`.
- Effective batch = GPUs x `--batch-size` x `--grad-accum`. Lower `--grad-accum` as you
  add GPUs to keep it at 16 (4 GPUs → `--grad-accum 1`, 2 GPUs → `--grad-accum 2`);
  otherwise you train with a bigger batch and fewer optimizer steps.
- On 4x RTX 3090, the default 3 epochs take ~6 min.
- Plain `python scripts/train.py` on a multi-GPU machine uses **only GPU 0**. This is
  deliberate: without `torchrun`, Trainer would fall back to `nn.DataParallel`, which
  collects gradients on GPU 0 and runs out of memory.
- Only rank 0 writes the adapter, logs, and the `--merge` output.

Output: LoRA adapter in `outputs/nietzsche-lora/` (plus per-epoch checkpoints).
Watch `eval_loss`: if it rises while train loss keeps falling, use fewer epochs.

## 3. Try it

```bash
python scripts/generate.py --prompt "Why do we praise the people we envy?"
python scripts/generate.py                            # interactive chat
python scripts/generate.py --adapter ""              # base model, for comparison
python scripts/generate.py --model outputs/nietzsche-lora/merged --adapter ""  # merged model
```

## 4. Merge the adapter into a standalone model

Training produces a LoRA adapter that needs the base model at load time. To deploy
(vLLM, SageMaker, GGUF conversion, Hugging Face upload) merge it into one full model:

```bash
python scripts/merge.py                                  # outputs/nietzsche-lora -> outputs/nietzsche-lora/merged
python scripts/merge.py --adapter outputs/run2 --out export/nietzsche   # custom paths
python scripts/merge.py --model models/Qwen3.5-4B        # override the base model path
```

- The base model path is read from the adapter's `adapter_config.json` unless `--model`
  is given; the path must exist relative to where you run the script.
- Runs on CPU in bf16 (no GPU needed) and writes ~9 GB of safetensors plus the tokenizer
  and chat template.
- Skip this step if you trained with `--merge`; that already wrote `<out>/merged`.
- Check the result: `python scripts/generate.py --model outputs/nietzsche-lora/merged --adapter ""`.
- Serve it, e.g. `vllm serve outputs/nietzsche-lora/merged --dtype bfloat16`.

## Notes

- Training and inference both render prompts with the model's chat template with
  thinking **disabled**. Keep that if you serve the model elsewhere.
- Answers in the dataset are Claude-written rewrites in Nietzsche's style, not his
  verbatim text (see the quality review: watch for repeated "Because..." openers).
