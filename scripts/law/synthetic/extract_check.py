#!/usr/bin/env python3
"""What EchoMind's own reader gets from each synthetic document (2026-11-12). Runs inside the backend image, offline:

    docker run --rm --network none --user 1000:1000 -v "$PWD:/repo:ro" -v "$OUT:/out" -w /repo/backend \
        -e ECHOMIND_EDITION=law echomind-law-backend:latest python3 /repo/scripts/law/synthetic/extract_check.py --out /out

For every importable file of eval/fixtures/synthetic/manifest.json it runs documents.extract (the import's own reader:
text layer, Word, or OCR with the local tesseract) and reports how the file was read, the clause numbers found, and how
close the words are to what a reader sees (eval/fixtures/synthetic/text/):
- **word recall:** the share of the reader's words that the extraction has, in order (difflib on word lists);
- **figure recall:** the share of the reader's figures (amounts, dates, percentages, clause numbers) found exactly.
Handwriting, stamps and signatures count as words the reader sees; Word comments do not (they are never document text).
The extracted texts go to --out/<matter>/<file>.txt and the summary to --out/extract_check.json.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "eval" / "fixtures" / "synthetic"
sys.path.insert(0, str(ROOT / "backend"))

_FIG = re.compile(r"(?:USD|GBP|\$|£)?\s?\d[\d,]*(?:\.\d+)?%?")


def reader_words(text: str) -> str:
    text = text.split("\n\n", 2)[-1]                                  # the title and file lines are not on the page
    text = re.sub(r"\[Word comment: [^\]]*\]", " ", text)
    text = re.sub(r"\[signature: [^\]]*\]", " ", text)
    text = re.sub(r"\[(?:handwritten|stamp): ([^\]]*)\]", r" \1 ", text)
    return text.replace("--- page break ---", " ")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9£$%]+(?:[.,][0-9]+)*", text.lower())


def figures(text: str) -> set[str]:
    return {re.sub(r"\s", "", f).rstrip(",.") for f in _FIG.findall(text) if len(re.sub(r"\D", "", f)) >= 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated matters")
    ap.add_argument("--base", default=str(BASE), help="another dataset folder (a test)")
    args = ap.parse_args()
    base = Path(args.base)
    from app.legal.documents.extract import extract
    out = Path(args.out)
    manifest = json.loads((base / "manifest.json").read_text())
    rows = []
    for case in manifest["cases"]:
        if args.only and case["number"] not in args.only.split(","):
            continue
        for d in case["documents"]:
            if not d["import"]:
                continue
            path = base / d["file"]
            t0 = time.time()
            x = extract(path.name, path.read_bytes())
            secs = round(time.time() - t0, 1)
            (out / case["number"]).mkdir(parents=True, exist_ok=True)
            (out / case["number"] / (path.name + ".txt")).write_text(x.text)
            reader = reader_words((base / "text" / (d["file"] + ".txt")).read_text())
            rw, xw = words(reader), words(x.text)
            sm = difflib.SequenceMatcher(None, rw, xw, autojunk=False)
            matched = sum(b.size for b in sm.get_matching_blocks())
            rf, xf = figures(reader), figures(x.text)
            sources = sorted({p.source for p in x.pages})
            conf = [p.confidence for p in x.pages if p.confidence is not None]
            numbered = [b for b in x.blocks if b.label]
            rows.append({"file": d["file"], "render": d["render"], "read_as": "+".join(sources) or x.filetype, "pages": len(x.pages),
                         "ocr_confidence": round(sum(conf) / len(conf), 1) if conf else None, "secs": secs,
                         "word_recall": round(matched / max(1, len(rw)), 3), "figure_recall": round(len(rf & xf) / max(1, len(rf)), 3),
                         "figures_missing": sorted(rf - xf)[:25], "numbered_blocks": len(numbered),
                         "labels": [b.path or b.label for b in numbered][:400],
                         "marks": {k: sum(m.kind == k for m in x.marks) for k in {m.kind for m in x.marks}}})
            r = rows[-1]
            print(f"{r['read_as']:9} words {r['word_recall']:.3f} figures {r['figure_recall']:.3f} numbered {r['numbered_blocks']:3} "
                  f"{secs:5}s  {d['file']}", flush=True)
    (out / "extract_check.json").write_text(json.dumps({"documents": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
