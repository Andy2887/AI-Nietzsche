"""Merge data/batches/*.txt into data/questions.json (id -> [{style,q}])"""
import json, glob, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
S = {"d": "direct", "m": "modern", "c": "challenge", "k": "casual"}
out, cur = {}, None
for f in sorted(glob.glob(str(ROOT / "data/batches/*.txt"))):
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"): continue
        if re.match(r"^bge-\d{3}-\d$", line): cur = line; out.setdefault(cur, []); continue
        s, q = line.split("|", 1); assert s in S, line
        out[cur].append({"style": S[s], "q": q.strip()})
json.dump(out, open(ROOT / "data/questions.json", "w"), ensure_ascii=False, indent=1)
ids = {p["id"] for p in json.load(open(ROOT / "data/remaining.json"))}
print("passages:", len(out), "questions:", sum(map(len, out.values())), "unknown ids:", set(out) - ids)
print("remaining to do:", len(ids - set(out)))
