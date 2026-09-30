"""Rewrite each training answer as a direct, coherent reply in Nietzsche's voice.

The original answer (a real Nietzsche passage) is passed as grounding for ideas and
imagery; Claude writes a short reply that actually answers the question.

Usage:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python3 scripts/rewrite_answers.py --limit 20     # pilot: review data/rewritten/pilot.jsonl
    python3 scripts/rewrite_answers.py                # full run (resumable)

Outputs data/rewritten/{train,val}_messages.jsonl in the same schema as data/*_messages.jsonl.
"""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = DATA / "rewritten"
MODEL = "claude-sonnet-5-5"

SYSTEM = """You are writing training data for a model that speaks as Friedrich Nietzsche.
You get a user's question and a reference passage from Nietzsche's actual writings.
Write the reply Nietzsche would give to this user.

Requirements:
- Answer the question directly. The first sentence should engage what was asked; the reply must make logical sense to someone who has only read the question.
- Voice: first person, Nietzsche's cadence and imagery, provocative, psychological, aphoristic but argued. Draw on the ideas and imagery of the reference passage where they fit, without copying long stretches of it.
- Length: 60-220 words. Let the length fit the idea: some replies may be a single sharp paragraph, others two. One main idea, developed clearly. No rambling.
- Every sentence must follow from the question. Drop any imagery from the reference passage that does not serve the answer; never end on a thought that is disconnected from what was asked.
- Vary your openings and shapes. Do not rely on stock moves such as "You have it backwards", "Look closer", "Observe", "Mark", or "Do not mistake this" (use any of them rarely, if at all). Not every reply needs to reject a naive view or end on a twist line; sometimes simply answer, argue, or confess.
- Keep the user's framing. Preserve any pronouns they used (e.g. "they" stays "they") and do not add facts, genders, or details they did not give.
- If the question is modern or casual, answer it in his voice about the thing actually asked; don't pretend it is about the reference passage. Do not use modern slang or mention being an AI.
- Never mention the passage, the book, or that you were given a reference. No headings, lists, or meta commentary. Use *asterisks* for emphasis sparingly.
- Output only the reply text."""


def make_client():
    return anthropic.Anthropic(max_retries=5)


def rewrite(client, question, passage):
    user = f"<question>\n{question}\n</question>\n\n<reference_passage>\n{passage}\n</reference_passage>"
    for attempt in range(4):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=600,
                thinking={"type": "between_tools"},  # no thinking; disabled is a 400 on Sonnet 5.5
                system=SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            if text:
                return text
            print(f"  retry {attempt + 1}: empty reply (stop_reason={resp.stop_reason})")
        except anthropic.APIError as e:
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    return None


def process(split, limit, workers):
    src = DATA / f"{split}_messages.jsonl"
    dst = OUT / ("pilot.jsonl" if limit else f"{split}_messages.jsonl")
    rows = [json.loads(l) for l in open(src)]
    if limit:
        rows = rows[:: max(1, len(rows) // limit)][:limit]

    def key(q, a):
        return q + "\x00" + hashlib.md5(a.encode()).hexdigest()

    done = set()
    if dst.exists():
        for l in open(dst):
            r = json.loads(l)
            done.add(r["messages"][0]["content"] + "\x00" + r["source_hash"])

    client = make_client()
    todo = [r for r in rows if key(r["messages"][0]["content"], r["messages"][1]["content"]) not in done]
    print(f"{split}: {len(rows)} rows, {len(todo)} to do")

    def work(r):
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        out = rewrite(client, q, a)
        return None if out is None else {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": out},
            ],
            "source_hash": hashlib.md5(a.encode()).hexdigest(),
        }

    OUT.mkdir(exist_ok=True)
    with open(dst, "a") as f, ThreadPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(work, todo), 1):
            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
            print(f"  {i}/{len(todo)} done" + ("" if res else " (failed, will retry on next run)"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="pilot: rewrite N train rows only")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    for split in ("train",) if args.limit else ("train", "val"):
        process(split, args.limit, args.workers)
