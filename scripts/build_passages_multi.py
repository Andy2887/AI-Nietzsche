"""Split daybreak.txt and the_gay_science.txt into passages -> data/passages_<book>.jsonl"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_WORDS, MIN_WORDS = 400, 20
wc = lambda s: len(s.split())

BOOKS = {
    "dawn": dict(file="daybreak.txt", start=455, end=12318, title="Daybreak",
                 books=[(455, "Book I"), (2925, "Book II"), (4662, "Book III"), (6385, "Book IV"), (9425, "Book V")]),
    "gs":   dict(file="the_gay_science.txt", start=1080, end=9507, title="The Gay Science",
                 books=[(1080, "Book I"), (2658, "Book II"), (4030, "Book III"), (5880, "Book IV: Sanctus Januarius"), (7486, "Book V: We Fearless Ones")]),
}

apho_re = re.compile(r"^\s*([0-9lIO]{1,3})\.\s*$")  # tolerate OCR slips like "6l."
norm = lambda t: int(t.translate(str.maketrans("lIO", "110")))
# title forms:  "TITLE.—text"  (dawn)   "_Title.--_text" / "_Title--How_ does ..." (gs)
dawn_title = re.compile(r"^([A-Z0-9][A-Z0-9 ,;:'’‘“”\-\(\)\.\?!&]{2,}?[\.\?!])—(.*)$", re.S)
gs_title = re.compile(r"^_([^_]{1,120}?)[\.\?!]?(?:--|—)?_(?:--|—)?\s*(.*)$", re.S)

def clean(text):
    text = re.sub(r"\[\d+\]", "", text)                       # footnote markers
    text = re.sub(r"(?<=[A-Za-z\.,;:\)”’!?])\((\d{1,2})\)", "", text)  # "(2)" footnote refs glued to words
    text = re.sub(r"_([^_]+)_", r"*\1*", text, flags=re.S)  # _italic_ -> *italic*
    text = text.replace("_", "")
    text = text.replace("--", "—")
    return re.sub(r"[ \t]+", " ", text)

def strip_footnotes(lines):
    out, skip = [], False
    for l in lines:
        if re.match(r"^\s*\[\d+\]\s", l): skip = True
        elif not l.strip(): skip = False if skip and False else skip
        if skip and (not l.strip()): skip = False; continue
        if not skip: out.append(l)
    return out

def paragraphs(blines):
    text = "\n".join(strip_footnotes(blines))
    return [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def run(key, cfg):
    L = (ROOT / cfg["file"]).read_text(encoding="utf-8").splitlines()
    def book_at(i):
        cur = cfg["books"][0][1]
        for bi, name in cfg["books"]:
            if i >= bi - 1: cur = name
        return cur
    blocks, cur, expected = [], None, 1
    for i in range(cfg["start"] - 1, cfg["end"]):
        m = apho_re.match(L[i])
        if m and norm(m.group(1)) == expected:
            n = expected; expected = n + 1
            cur = {"n": n, "book": book_at(i), "lines": []}; blocks.append(cur)
        elif cur is not None:
            if re.match(r"^(BOOK|Book) ", L[i].strip()) and L[i].strip().upper().startswith("BOOK"): continue
            cur["lines"].append(L[i])
    # gs/dawn numbering is continuous across books, so sequence check is simple
    passages = []
    for b in blocks:
        pars = paragraphs(b["lines"])
        if not pars: continue
        title = None
        first = pars[0]
        m = None
        if key == "dawn":  # title = leading run of mostly-uppercase text before the first em dash
            idx = first.find("—", 0, 220)
            letters = [c for c in first[:idx] if c.isalpha()] if idx > 0 else []
            if letters and sum(c.isupper() for c in letters) / len(letters) >= 0.8:
                class _M:  # tiny adapter so the shared code below works
                    def group(self, i, a=first[:idx], b=first[idx + 1:]): return (None, a, b)[i]
                m = _M()
        else:
            m = gs_title.match(first)
        if m:
            title, rest = m.group(1), m.group(2).strip()
            if "--" in title:  # "_Woman in Music--How_ does ..." : first words of the text sit inside the italics
                title, lead = title.split("--", 1); rest = (lead + " " + rest).strip()
            title, pars[0] = re.sub(r"[*“”\"]", "", title).strip(" ._-—"), rest
        pars = [clean(p).strip() for p in pars if clean(p).strip()]
        if not pars: continue
        packed = []
        for p in pars:
            if wc(p) <= MAX_WORDS: packed.append(p); continue
            acc = []
            for sent in re.split(r"(?<=[.?!])\s+(?=[A-Z\"“])", p):
                if acc and wc(" ".join(acc + [sent])) > MAX_WORDS: packed.append(" ".join(acc)); acc = []
                acc.append(sent)
            if acc: packed.append(" ".join(acc))
        chunks, curc = [], []
        for p in packed:
            if curc and wc(" ".join(curc + [p])) > MAX_WORDS: chunks.append(curc); curc = []
            curc.append(p)
        if curc: chunks.append(curc)
        for k, c in enumerate(chunks):
            text = "\n\n".join(c)
            passages.append({"id": f"{key}-{b['n']:03d}-{k+1}", "work": cfg["title"], "section": b["n"], "part": k + 1, "of": len(chunks),
                             "chapter": b["book"], "title": title, "text": text, "words": wc(text)})
    for p in passages:
        p["flags"] = [f for f, cond in {
            "too_short": p["words"] < MIN_WORDS,
            "continuation": p["part"] > 1,
            "possible_context_ref": bool(re.match(r"^\W*(this|these|that|such|it|they|he|she|the latter|the former)\b", p["text"], re.I)),
            "poem": p["text"].count("\n") > 3 and all(len(x) < 70 for x in p["text"].split("\n") if x.strip()),
        }.items() if cond]
    with open(ROOT / "data" / f"passages_{key}.jsonl", "w", encoding="utf-8") as f:
        for p in passages: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    last = blocks[-1]["n"] if blocks else 0
    import collections
    print(key, "aphorisms:", len(blocks), "last n:", last, "passages:", len(passages),
          "usable:", sum(1 for p in passages if not set(p["flags"]) - {"continuation"}),
          dict(collections.Counter(x for p in passages for x in p["flags"])))
    ws = sorted(p["words"] for p in passages); print("  words min/med/max:", ws[0], ws[len(ws)//2], ws[-1], " titled:", sum(1 for p in passages if p["title"]))

for k, c in BOOKS.items(): run(k, c)
