"""Convert chat-format JSONL to Bedrock's Llama fine-tuning format ({"prompt", "completion"}).

The prompt is wrapped in the Llama 3 instruct template so training matches how you'll
prompt the model at inference (InvokeModel takes a raw prompt string).
Use --raw to emit the bare user question instead.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def convert(src, dst, raw, clean=None):
    n = 0
    with open(src) as f, open(dst, "w") as out:
        clean_f = open(clean, "w") if clean else None
        for line in f:
            msgs = json.loads(line)["messages"]
            if clean_f:
                clean_f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            q = next(m["content"] for m in msgs if m["role"] == "user")
            a = next(m["content"] for m in msgs if m["role"] == "assistant")
            prompt = q if raw else TEMPLATE.format(q=q)
            out.write(json.dumps({"prompt": prompt, "completion": a}, ensure_ascii=False) + "\n")
            n += 1
    if clean_f:
        clean_f.close()
    print(f"{dst.name}: {n} records")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--src-dir", default="", help="e.g. rewritten: read data/<dir>/<split>_messages.jsonl, write outputs there")
    args = ap.parse_args()
    d = DATA / args.src_dir
    for split in ("train", "val"):
        # with --src-dir, also write {split}_clean.jsonl (messages only, for Together)
        clean = d / f"{split}_clean.jsonl" if args.src_dir else None
        convert(d / f"{split}_messages.jsonl", d / f"{split}_bedrock.jsonl", args.raw, clean)
