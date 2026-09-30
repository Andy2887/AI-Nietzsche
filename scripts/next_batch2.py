import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
done = set(json.load(open(ROOT/"data/questions_multi.json"))) if (ROOT/"data/questions_multi.json").exists() else set()
skip = set(open(ROOT/"data/skipped_multi.txt").read().split())
todo = [p for p in json.load(open(ROOT/"data/remaining_multi.json")) if p["id"] not in done | skip]
print(f"# {len(todo)} left")
budget = 26000
for p in todo[:n]:
    blk = f"## {p['id']} [{p['title']}] ({p['words']}w)\n{p['text']}\n"
    if budget - len(blk) < 0: break
    budget -= len(blk); print(blk)
