# AI Nietzsche — Plan

Last updated: 2026-09-30

## Goal
Fine-tune a small model to answer questions in Nietzsche's voice, callable through an API, with a web UI on top.

## Where we are
- Dataset built from public-domain Nietzsche texts (Beyond Good and Evil, Daybreak, The Gay Science). Real passage = assistant answer, with 2 generated questions per passage.
  - `data/train_messages.jsonl` (1,806 pairs), `data/val_messages.jsonl` (120 pairs), chat `messages` format.
  - Sensitive passages (women, Jews, race, eugenics, etc.) are held out in `data/sensitive*.{jsonl,txt}`. No decision yet on including them.
- **First fine-tune (Qwen3.5-4B, LoRA, Together AI) worked stylistically but not logically.** Answers sounded like Nietzsche but didn't make sense as replies to the question.
  - Cause: the training answers are aphorisms/essay fragments, not answers to questions. The model learned style without coherence.
  - Side issues: output landed in `reasoning_content` (Qwen thinking mode vs. data without think blocks), and unrelated web-text artifacts appeared (base-model leakage).
- Bedrock: Gemma isn't supported for managed fine-tuning. Data was also converted for Llama (`*_bedrock.jsonl`, prompt/completion with Llama 3 template) and Nova (`*_nova.jsonl`); none of these runs was completed.

## Decision
Keep the questions, replace the answers. Have Claude write a short (80–180 word), direct, in-voice answer to each question, grounded in the real passage. Not switching to Socrates (Plato's dialogues are mostly questions, and it gives up the Nietzsche goal).

## Next steps
1. **Pilot the rewrite** (not yet run; needs `ANTHROPIC_API_KEY` and `pip install anthropic`):
   `python3 scripts/rewrite_answers.py --limit 20` → review `data/rewritten/pilot.jsonl`. Check: answers the question, sounds like Nietzsche, no rambling. Tune the `SYSTEM` prompt in the script if not.
2. **Full run:** `python3 scripts/rewrite_answers.py` (resumable; writes `data/rewritten/{train,val}_messages.jsonl`).
3. **Convert:** `python3 scripts/to_bedrock.py --src-dir rewritten`
   - `train_bedrock.jsonl` / `val_bedrock.jsonl` for Bedrock Llama.
   - `train_clean.jsonl` / `val_clean.jsonl` (messages only) for Together.
4. **Optional:** mix in a few plain, non-philosophical Q&A examples so the model doesn't turn every question into an aphorism (not implemented yet).
5. **Fine-tune Llama 3.1 8B on Bedrock:** upload the files to S3 in the same region as the job, then Custom models → fine-tuning job. Suggested start: 2–3 epochs, default learning rate (~5e-5), batch size 8 if editable (some Llama models lock it at 1). Watch validation loss; use fewer epochs if it rises.
6. **Test** with modern questions, a non-philosophical one, and a plain "hello". Fail signs: answers everything with an aphorism, or copies passages verbatim.
7. **Serve as an API:** check whether the Bedrock custom model can run on-demand or needs Provisioned Throughput (billed hourly). If it needs provisioned capacity and that's too costly, retrain on Together (LoRA, per-token serving) using the `*_clean.jsonl` files.
8. **Web UI:** small backend calling the model endpoint, so API keys stay off the browser.

## Reference settings (Together, from the Qwen run)
LoRA rank 64 / alpha 128, all-linear, 3 epochs, batch size 8, LR 1e-4, warmup 0.05, max seq length 4096, 3 checkpoints, 4 evaluations. 1e-5 was too low for LoRA.

## Housekeeping
- **Revoke the Together API key** that was pasted into chat earlier, and create a new one.
- The rewritten answers are Claude's imitation of Nietzsche, grounded in his real text. That's a tradeoff against authentic passages.

## Files
- `scripts/rewrite_answers.py`: rewrite answers via the Claude API.
- `scripts/to_bedrock.py`: format conversion (Llama prompt/completion, plus clean messages).
- `scripts/build_dataset.py` and the other `scripts/*.py`: the original dataset pipeline.
- `data/questions_v2.json`: generated questions.
