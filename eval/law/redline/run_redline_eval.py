#!/usr/bin/env python3
"""EchoMind Law redline robustness (S3.3, DOC-02b; spec test T10; decision D-047).

No model is involved: this measures the Word tracked-changes writer itself. Every numbered clause of
every document in the evaluation corpora (Markdown read as text, and the S2.1 Word fixture with its
own numbering, tracked changes and comments) is revised four ways:
- a figure changed ("30" → "45", an amount);
- words added at the end;
- the clause shortened to its first sentence;
- a paragraph added after it.

Each revision is written as tracked changes and checked: accepting all gives the revision and nothing
else, rejecting all gives the original, every change has an author and a date. A sample is round-
tripped through LibreOffice and must keep its tracked changes. Reported: revisions, placed, missed
(clause not found), verified, round trips kept.

    cd backend && python3 ../eval/law/redline/run_redline_eval.py
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ROOT = LAW.parent.parent
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "backend" / "tests" / "law")]


def variants(original: str) -> list[tuple[str, str]]:
    out = []
    m = re.search(r"\d[\d,]*", original[6:])
    if m:
        a, b = 6 + m.start(), 6 + m.end()
        out.append(("figure", original[:a] + ("45" if original[a:b] != "45" else "60") + original[b:]))
    out.append(("append", original.rstrip(".") + ", save as agreed in writing."))
    first = re.split(r"(?<=[.;:])\s", original, maxsplit=1)[0]
    if first != original:
        out.append(("shorten", first))
    out.append(("add_paragraph", original + "\n\n(z) A further fictional obligation of the parties."))
    return out


def main() -> int:
    from app.legal.documents import extract, structure
    from app.legal.redlines import service, word
    from fixtures_s21 import contract_docx
    docs = [(p.name, p.read_bytes()) for p in sorted((LAW / "corpus").glob("*.md")) + sorted((LAW / "compare").glob("*.md"))]
    docs.append(("harrowgate-msa.docx", contract_docx()))
    rows, sample = [], []
    for name, data in docs:
        x = extract.extract(name, data)
        source = data if name.endswith(".docx") else None
        base = source or word.plain_docx(x.text)
        for c in structure.clauses(x):
            original = service.own_text(c.number, x.text[c.start:c.end])
            if len(original) < 25:
                continue
            for kind, revised in variants(original):
                out, plan = word.apply(source, x.text, [{"id": "c", "original": original, "revised": revised, "comment": kind}],
                                       "Evaluation", "2026-10-10T10:00:00Z")
                v = word.verify(base, out, plan)
                rows.append({"document": name, "clause": c.number, "kind": kind, "placed": not plan["missed"], "verified": v["ok"],
                             "failed_checks": [k for k, ok in v["checks"].items() if not ok]})
                if not plan["missed"] and len(sample) < 12 and (kind == "add_paragraph" or name.endswith(".docx")):
                    sample.append((name, c.number, kind, out))
    kept = 0
    soffice = shutil.which("soffice")
    if soffice:
        with tempfile.TemporaryDirectory() as d:
            for i, (_, _, _, out) in enumerate(sample):
                Path(d, f"r{i}.docx").write_bytes(out)
                subprocess.run([soffice, f"-env:UserInstallation=file://{d}/lo", "--headless", "--convert-to", "docx:MS Word 2007 XML",
                                "--outdir", f"{d}/o", f"{d}/r{i}.docx"], capture_output=True, timeout=180)
                try:
                    xml = zipfile.ZipFile(io.BytesIO(Path(d, "o", f"r{i}.docx").read_bytes())).read("word/document.xml").decode()
                    kept += ("<w:ins " in xml or "<w:del " in xml) and 'w:author="Evaluation"' in xml
                except (OSError, KeyError, zipfile.BadZipFile):
                    pass
    placed = [r for r in rows if r["placed"]]
    summary = {"documents": len(docs), "revisions": len(rows), "placed": len(placed), "missed": len(rows) - len(placed),
               "verified": f"{sum(r['verified'] for r in placed)}/{len(placed)}",
               "libreoffice_round_trips_kept": f"{kept}/{len(sample)}" if soffice else "LibreOffice not installed",
               "by_kind": {k: f"{sum(r['verified'] for r in placed if r['kind'] == k)}/{sum(1 for r in placed if r['kind'] == k)}"
                           for k in ("figure", "append", "shorten", "add_paragraph")}}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = {"set": "t10", "commit": commit, "time": stamp}
    out_dir = LAW / "reports"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{stamp}-redline-t10.json").write_text(json.dumps({"meta": meta, "summary": summary, "items": rows}, indent=1))
    failures = [r for r in rows if not r["verified"]]
    md = ["# EchoMind Law redline robustness (T10)", "", f"- Commit: `{commit}`", f"- Run: {stamp}", "", "| Measure | Result |", "|---|---|"]
    md += [f"| {k.replace('_', ' ')} | {v} |" for k, v in summary.items()]
    md += ["", "## Not verified or not placed", ""] + ([f"- {r['document']} {r['clause']} ({r['kind']}): {'missed' if not r['placed'] else ', '.join(r['failed_checks'])}"
                                                       for r in failures[:60]] or ["- none"])
    (out_dir / f"{stamp}-redline-t10.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary))
    print(f"report: eval/law/reports/{stamp}-redline-t10.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
