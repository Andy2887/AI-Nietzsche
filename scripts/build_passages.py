"""Split beyond_good_and_evil.txt into passages -> data/passages.jsonl"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
lines = (ROOT / "beyond_good_and_evil.txt").read_text(encoding="utf-8").splitlines()

MAX_WORDS, MIN_WORDS = 400, 20
wc = lambda s: len(s.split())
PREFACE, END_OF_BODY = 87, 6021  # 1-indexed line numbers of PREFACE and FROM THE HEIGHTS

# ---- locate chapters
chap_re = re.compile(r"^CHAPTER ([IVX]+)\. (.+)$")
chapters = [(i, m.group(1), m.group(2).title()) for i, l in enumerate(lines)
            if (m := chap_re.match(l))]

def chapter_at(i):
    cur = "Preface"
    for ci, num, title in chapters:
        if i >= ci: cur = f"{num}. {title}"
    return cur

# ---- split into aphorisms: a line starting "N. " or a lone "N."
apho_re = re.compile(r"^(\d{1,3})\.(?:--|\s+)?(.*)$")
blocks, cur = [], None
expected = 1
for i in range(PREFACE - 1, END_OF_BODY - 1):
    l = lines[i]
    m = apho_re.match(l.strip())
    if m and int(m.group(1)) == expected:
        cur = {"n": expected, "chapter": chapter_at(i), "lines": [m.group(2) or ""]}
        blocks.append(cur); expected += 1
    elif cur is None:
        if i == PREFACE - 1 or (blocks == [] and l.strip() and not l.startswith("PREFACE")):
            cur = cur or {"n": 0, "chapter": "Preface", "lines": []}
            if cur not in blocks: blocks.append(cur)
            if l.strip() != "PREFACE": cur["lines"].append(l)
    else:
        if chap_re.match(l): continue
        cur["lines"].append(l)

# ---- cleanup
def clean(text):
    text = re.sub(r"\[FOOTNOTE:.*?\]", "", text, flags=re.S | re.I)
    text = re.sub(r"\n\s*Sils Maria.*$", "", text, flags=re.S)  # preface dateline
    text = re.sub(r"[ \t]+", " ", text)
    return text

def emphasis(par):
    # e-text capitalises italics; restore as *word* except sentence/quote-initial words
    toks = re.split(r"(\s+)", par)
    out, start = [], True
    for t in toks:
        core = re.sub(r"^\W+|\W+$", "", t)
        if core.isupper() and len(core) > 1 and core not in {"II", "III", "IV", "VI"} and not start:
            t = t.replace(core, core.lower())
            t = f"*{t}*" if False else t  # placeholder to keep control below
            out.append(("EM", t))
        elif core.isupper() and len(core) > 1 and start:
            out.append(("TXT", t.replace(core, core.capitalize())))
        else:
            out.append(("TXT", t))
        if t.strip(): start = bool(re.search(r"[.?!:\"”]$", t.strip())) and False or bool(re.search(r"[.?!]$", t.strip()))
    # merge consecutive EM runs into one *...*
    res, buf = "", []
    for kind, t in out:
        if kind == "EM": buf.append(t)
        else:
            if buf and not t.strip():
                buf.append(t); continue
            if buf:
                s = "".join(buf).rstrip(); res += _wrap(s) + " "
                buf = []
            res += t
    if buf: res += _wrap("".join(buf).rstrip())
    return res

def _wrap(s):
    m = re.match(r"^(\W*)(.*?)(\W*)$", s, re.S)
    return f"{m.group(1)}*{m.group(2)}*{m.group(3)}" if m.group(2) else s

def paragraphs(blines):
    text = "\n".join(blines)
    text = clean(text)
    pars = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in pars if p]

passages = []
for b in blocks:
    pars = []
    for p in (emphasis(p) for p in paragraphs(b["lines"])):
        if wc(p) <= MAX_WORDS: pars.append(p); continue
        acc = []  # split over-long paragraph at sentence boundaries
        for sent in re.split(r"(?<=[.?!])\s+(?=[A-Z\"“])", p):
            if acc and wc(" ".join(acc + [sent])) > MAX_WORDS:
                pars.append(" ".join(acc)); acc = []
            acc.append(sent)
        if acc: pars.append(" ".join(acc))
    # greedily pack paragraphs up to MAX_WORDS
    chunks, curc = [], []
    for p in pars:
        if curc and wc(" ".join(curc + [p])) > MAX_WORDS:
            chunks.append(curc); curc = []
        curc.append(p)
    if curc: chunks.append(curc)
    for k, c in enumerate(chunks):
        passages.append({"section": b["n"], "part": k + 1, "of": len(chunks),
                         "chapter": b["chapter"], "text": "\n\n".join(c)})

for j, p in enumerate(passages):
    p["id"] = f"bge-{p['section']:03d}-{p['part']}"
    p["words"] = wc(p["text"])
    p["flags"] = [f for f, cond in {
        "too_short": p["words"] < MIN_WORDS,
        "greek_placeholder": "GREEK" in p["text"].upper() and "INSERTED" in p["text"].upper(),
        "continuation": p["part"] > 1,
        "possible_context_ref": bool(re.match(r"^(?:\W*)(?:this|these|that|such|it|they|he|she)\b", p["text"], re.I)),
    }.items() if cond]

with open(ROOT / "data" / "passages.jsonl", "w", encoding="utf-8") as f:
    for p in passages:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

import collections
print("aphorisms found:", expected - 1, "(expected 296 + preface)")
print("passages:", len(passages), " usable (no flags except continuation):",
      sum(1 for p in passages if not set(p["flags"]) - {"continuation"}))
print(collections.Counter(f for p in passages for f in p["flags"]))
ws = sorted(p["words"] for p in passages); print("words min/median/max:", ws[0], ws[len(ws)//2], ws[-1])
