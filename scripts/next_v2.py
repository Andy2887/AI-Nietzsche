import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
STOP = set("the a an of to and in is it that this be are as for with on by or not we you he they what why how do does can should i my our your his her from at have has was were will would which who whom its their there than then so but if all any one no more most into about only also".split())
def words(s): return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP and len(w) > 2}
def overlap(q, t):
    qw = words(q); 
    return 0 if not qw else len(qw & words(t)) / len(qw)
def trunc(t, head=80, tail=22):
    w = t.split()
    return t if len(w) <= head + tail + 20 else " ".join(w[:head]) + " [...] " + " ".join(w[-tail:])
done = set()
p = ROOT/"data/questions_v2.json"
if p.exists(): done = set(json.load(open(p)))
todo = [x for x in json.load(open(ROOT/"data/work_passages.json")) if x["id"] not in done]
print(f"# {len(todo)} left")
budget = int(sys.argv[1]) if len(sys.argv) > 1 else 22000
for x in todo:
    lines = "\n".join(f"  {i+1}. [{o['style'][0]} {overlap(o['q'], x['text']):.2f}] {o['q']}" for i, o in enumerate(x["old"]))
    blk = f"## {x['id']} ({len(x['text'].split())}w)\n{trunc(x['text'])}\n{lines}\n"
    if budget - len(blk) < 0: break
    budget -= len(blk); print(blk)
