#!/usr/bin/env python3
"""EchoMind Law translation evaluation (OPT-02; decision D-070).

Translates fixed paragraphs of the FICTIONAL evaluation corpus (eval/law/corpus) into French, German and Spanish with
the real answer model, through the same code the product uses (app.legal.translation.translate), and records each
original beside its translation with the figures the check flagged. A lawyer then scores them
(eval/law/lawyer_review/make_packet.py --phase 5); this script measures only what a machine can:
- figures flagged: a figure of the original not found in the translation (separators aside);
- length ratio: translation over original, in characters (a dropped sentence shows as a low ratio);
- time per batch.

    cd backend && python3 ../eval/law/translation/run_translation_eval.py

The report goes to eval/law/reports/<time>-translation-review.json and .md. Nothing leaves this host.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ROOT = LAW.parent.parent
sys.path.insert(0, str(LAW))

DOCUMENTS = ["msa-v1-signed.md", "nda.md", "side-letter.md", "lease.md"]
LANGUAGES = ["fr", "de", "es"]
PER_DOCUMENT = 4                       # paragraphs with a figure or a date first, then others


def passages() -> list[tuple[str, int, str]]:
    from app.legal.translation import translate as tr
    out = []
    for name in DOCUMENTS:
        paras = [p for p in tr.paragraphs((LAW / "corpus" / name).read_text()) if len(p) > 80 and not p.startswith("#")]
        ranked = sorted(range(len(paras)), key=lambda i: (not re.search(r"\d", paras[i]), i))[:PER_DOCUMENT]
        out += [(name, i, paras[i]) for i in sorted(ranked)]
    return out


async def translate_all(items: list[tuple[str, int, str]]) -> list[dict]:
    from app.legal.translation import translate as tr
    rows = []
    for code in LANGUAGES:
        for name in DOCUMENTS:
            mine = [x for x in items if x[0] == name]
            t0 = time.monotonic()
            async for k, out in tr.translate(tr.LANGUAGES[code], [x[2] for x in mine]):
                rows.append({"language": code, "document": name, "paragraph": mine[k][1], "source": mine[k][2], **out})
            seconds = round(time.monotonic() - t0, 2)
            for r in rows:
                if r["language"] == code and r["document"] == name:
                    r["batch_seconds"] = seconds
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="http://localhost:8355/v1")
    ap.add_argument("--model", default="nvidia/Qwen3-30B-A3B-FP4")
    ap.add_argument("--embed", default="http://localhost:11435/api/embeddings")
    args = ap.parse_args()
    from run_law_eval import setup_env
    data = setup_env(args)
    try:
        items = passages()
        rows = asyncio.run(translate_all(items))
    finally:
        shutil.rmtree(data, ignore_errors=True)
        (data.parent / f"{data.name}.key").unlink(missing_ok=True)
    ratio = lambda r: round(len(r["text"]) / max(1, len(r["source"])), 2)
    for r in rows:
        r["length_ratio"] = ratio(r)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    from app.legal.translation import translate as tr
    summary = {"paragraphs": len(items), "translations": len(rows), "languages": LANGUAGES,
               "figures_flagged": sum(1 for r in rows if r["figures_missing"]),
               "short_translations": sum(1 for r in rows if r["length_ratio"] < 0.6),
               "empty": sum(1 for r in rows if not r["text"].strip()),
               "batch_seconds_p50": statistics.median(sorted({(r["language"], r["document"]): r["batch_seconds"] for r in rows}.values()))}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {"set": "translation-review", "commit": commit + ("+changes" if dirty else ""), "model": args.model, "prompt": tr.VERSION,
            "documents": DOCUMENTS, "time": stamp}
    out = LAW / "reports" / f"{stamp}-translation-review.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    lines = [f"# EchoMind Law translation evaluation — {stamp}", "", f"- Commit: `{meta['commit']}`", f"- Model: `{args.model}`",
             f"- Prompt: `{tr.VERSION}`", "", "| Measure | Result |", "|---|---|"] + [f"| {k} | {v} |" for k, v in summary.items()] + \
            ["", "| Language | Document | Paragraph | Figures flagged | Length ratio |", "|---|---|---|---|---|"] + \
            [f"| {r['language']} | {r['document']} | {r['paragraph']} | {', '.join(r['figures_missing']) or '—'} | {r['length_ratio']} |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"report: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
