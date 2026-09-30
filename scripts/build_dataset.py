"""Build train/val/sensitive JSONL.

train/val : data/work_passages.json (repaired passage text) + data/questions_v2.json (<=2 questions per passage,
            re-read and rewritten in revision pass v2). Validation split is by aphorism, seed 42.
sensitive : passages held out on purpose (data/sensitive_*.txt) paired with their original (v1) questions.
"""
import json, random, re, hashlib, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"

# ---------- passages
passages = {}
for p in json.load(open(D / "usable.json")): passages[p["id"]] = p
for k in ("dawn", "gs"):
    for line in open(D / f"passages_{k}.jsonl", encoding="utf-8"):
        p = json.loads(line); passages[p["id"]] = p
work = {x["id"]: x for x in json.load(open(D / "work_passages.json"))}
qv2 = {k: v for k, v in json.load(open(D / "questions_v2.json")).items() if v}

def clean(text):
    text = re.sub(r"(?<=[*—.])\(\d{1,2}\)", "", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("--", "—")
    text = re.sub(r"^[—-]+\s*", "", text.strip())
    text = re.sub(r"\s*[—-]+\s*$", "", text)          # dangling dash from mid-aphorism cuts
    return text.strip()

# ---------- question phrasing: break up the "Tell me ..." template (deterministic per question)
ALT = {
    "what":    ["Explain what", "Help me understand what", "I'd like to know what"],
    "why":     ["Explain why", "Help me understand why", "I'd like to know why"],
    "how":     ["Explain how", "Help me understand how", "I'd like to know how"],
    "where":   ["Help me understand where", "Explain where"],
    "who":     ["Help me understand who", "Explain who"],
    "which":   ["Help me understand which", "Explain which"],
    "when":    ["Help me understand when", "Explain when"],
    "about":   ["Say something about", "Speak to me about"],
    "whether": ["Give me your view on whether", "Weigh in on whether"],
    "the":     ["Give me the", "Lay out the"],
}
def diversify(q, key):
    m = re.match(r"^Tell me (what|why|how|where|who|which|when|about|whether|the)\b(.*)$", q, re.S)
    if not m: return q
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    opts = ["Tell me " + m.group(1)] + ALT[m.group(1)]     # keep the original as one option (~25%)
    return f"{opts[h % len(opts)]}{m.group(2)}"

# ---------- sensitivity
extra = set(open(D / "sensitive_extra.txt").read().split())
forced = set(open(D / "sensitive_pending.txt").read().split()) | set(open(D / "sensitive_pending_multi.txt").read().split()) | {"bge-232-2"} | extra
SENS = re.compile(r"\b(woman|women|female|jews?|jewish|semit\w*|old maid|wives?|girls?|feminine|mother-in-law)\b", re.I)
NOT_SENSITIVE = {"bge-000-1", "bge-220-1", "bge-293-1", "bge-231-1", "bge-236-1", "bge-085-1"} - extra
def is_sensitive(pid):
    return pid in forced or (pid not in NOT_SENSITIVE and bool(SENS.search(passages[pid]["text"])))

def book(pid): return passages[pid].get("work") or "Beyond Good and Evil"
def sec_key(pid): return (book(pid), passages[pid]["section"])

train_ids = [p for p in qv2 if p in work and not is_sensitive(p)]
secs = sorted({sec_key(p) for p in train_ids})
random.seed(42)
val_secs = set(random.sample(secs, round(len(secs) * 0.06)))

rows = {"train": [], "val": [], "sensitive": []}
for pid in train_ids:
    bucket = "val" if sec_key(pid) in val_secs else "train"
    text = clean(work[pid]["text"])
    for i, x in enumerate(qv2[pid]):
        rows[bucket].append({"messages": [{"role": "user", "content": diversify(x["q"], f"{pid}:{i}")},
                                          {"role": "assistant", "content": text}],
                             "meta": {"id": pid, "work": book(pid), "chapter": passages[pid]["chapter"], "style": x["style"]}})

# sensitive holdout: original (v1) questions for passages that were removed on purpose
v1 = json.load(open(D / "questions.json")); v1.update(json.load(open(D / "questions_multi.json")))
for pid, lst in json.load(open(D / "questions_pilot.json")).items(): v1.setdefault(pid, []).extend(lst)
for pid, lst in v1.items():
    if pid in passages and is_sensitive(pid):
        for x in lst:
            rows["sensitive"].append({"messages": [{"role": "user", "content": x["q"]},
                                                   {"role": "assistant", "content": clean(passages[pid]["text"])}],
                                      "meta": {"id": pid, "work": book(pid), "chapter": passages[pid]["chapter"], "style": x["style"]}})
random.shuffle(rows["train"])
for name, r in rows.items():
    with open(D / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for row in r: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if name != "sensitive":
        with open(D / f"{name}_messages.jsonl", "w", encoding="utf-8") as f:
            for row in r: f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

print({k: len(v) for k, v in rows.items()}, "| val sections:", len(val_secs))
for b in ("train", "val"):
    print(b, "pairs by work:", dict(collections.Counter(r["meta"]["work"] for r in rows[b])), "| passages:", len({r["meta"]["id"] for r in rows[b]}))
print("styles (train):", dict(collections.Counter(r["meta"]["style"] for r in rows["train"])))
qs = [r["messages"][0]["content"] for r in rows["train"] + rows["val"]]
print("top openings:", collections.Counter(" ".join(q.split()[:2]) for q in qs).most_common(8))
print("dup questions:", len(qs) - len(set(q.lower() for q in qs)))
tr = {sec_key(r["meta"]["id"]) for r in rows["train"]}; va = {sec_key(r["meta"]["id"]) for r in rows["val"]}
print("section overlap train/val:", tr & va)
wc = sorted(len(r["messages"][1]["content"].split()) for r in rows["train"]); print("answer words min/med/max:", wc[0], wc[len(wc)//2], wc[-1])
print("sensitive passages:", len({r["meta"]["id"] for r in rows["sensitive"]}))
