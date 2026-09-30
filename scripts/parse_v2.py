import json, glob, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
S = {"d": "direct", "m": "modern", "c": "challenge", "k": "casual", "r": "request", "s": "scenario"}
work = {x["id"]: x for x in json.load(open(ROOT/"data/work_passages.json"))}
out, cur, errs = {}, None, []
for f in sorted(glob.glob(str(ROOT/"data/v2/*.txt"))):
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"): continue
        if re.match(r"^(bge|dawn|gs)-\d{3}-\d$", line.strip()): cur = line.strip(); out.setdefault(cur, []); continue
        if cur not in work: errs.append(f"unknown id {cur}"); continue
        if line.strip() == "X": out[cur] = None; continue
        m = re.match(r"^K(\d)$", line.strip())
        if m:
            i = int(m.group(1)) - 1
            if i >= len(work[cur]["old"]): errs.append(f"{cur}: bad K{i+1}"); continue
            if out[cur] is not None: out[cur].append(dict(work[cur]["old"][i]))
            continue
        s, q = line.split("|", 1)
        if s not in S: errs.append(f"{cur}: bad style {s}"); continue
        if out[cur] is not None: out[cur].append({"style": S[s], "q": q.strip()})
for k, v in out.items():
    if v is not None and not (1 <= len(v) <= 2): errs.append(f"{k}: {len(v)} questions")
json.dump(out, open(ROOT/"data/questions_v2.json", "w"), ensure_ascii=False, indent=1)
kept = {k: v for k, v in out.items() if v}
print("passages done:", len(out), "kept:", len(kept), "dropped:", sum(v is None for v in out.values()), "pairs:", sum(len(v) for v in kept.values()))
print("remaining:", len(set(work) - set(out)))
print("errors:", errs[:10])
