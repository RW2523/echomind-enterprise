#!/usr/bin/env python3
"""Check the synthetic dataset's answer keys against its documents (2026-11-12).

    python3 scripts/law/synthetic/check_keys.py [eval/fixtures/synthetic/answer_key_*.jsonl]

Errors (exit 1): a malformed item, a duplicate id, an unknown matter or kind, a `source` that is not a document title of
the matter, or an `evidence` segment that is not in its document's reader text (eval/fixtures/synthetic/text/; pieces
of a segment may be joined by "…", as run_retrieval_eval.py reads them).
Warnings: an `expect` group with no alternative anywhere in the matter's documents (fine for a worked-out figure), and a
`must_not` value that its own source document contains (a correct answer quoting it would fail).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "eval" / "fixtures" / "synthetic"
sys.path.insert(0, str(ROOT / "eval" / "law"))
from run_chat_keys import has  # noqa: E402

KINDS = {"fact", "role_party", "list", "calculation", "date", "not_stated", "amendment", "version", "conflict", "cross_doc",
         "follow_up", "table", "scan", "definitions", "chronology", "other_matter"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("**", "")
    s = re.sub(r"\[(?:handwritten|stamp): ([^\]]*)\]", r"\1", s)
    return " ".join(s.lower().split())


def main() -> int:
    files = [Path(a) for a in sys.argv[1:]] or sorted(BASE.glob("answer_key_*.jsonl"))
    manifest = json.loads((BASE / "manifest.json").read_text())
    titles = {c["number"]: {d["title"]: d for d in c["documents"]} for c in manifest["cases"]}
    texts = {c["number"]: {d["title"]: norm((BASE / "text" / (d["file"] + ".txt")).read_text()) for d in c["documents"]}
             for c in manifest["cases"]}
    errors, warnings, seen, kinds = [], [], set(), Counter()
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{f.name}:{n}"
            try:
                it = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{where}: not JSON ({e})")
                continue
            iid, matter, kind = it.get("id"), it.get("matter"), it.get("kind")
            where += f" {iid}"
            if not iid or iid in seen:
                errors.append(f"{where}: missing or duplicate id")
            seen.add(iid)
            if matter not in titles:
                errors.append(f"{where}: unknown matter {matter!r}")
                continue
            if kind not in KINDS:
                errors.append(f"{where}: unknown kind {kind!r}")
            kinds[(matter, kind)] += 1
            steps = it.get("turns") or [{"question": it.get("question"), "expect": it.get("expect"), "must_not": it.get("must_not")}]
            if not all(s.get("question") for s in steps):
                errors.append(f"{where}: a question is missing")
            corpus = " ".join(texts[matter].values())
            for s in steps:
                for g in s.get("expect") or []:
                    if not isinstance(g, list) or not g:
                        errors.append(f"{where}: expect groups must be non-empty lists")
                    elif kind not in ("calculation", "other_matter") and not any(has(a, corpus) for a in g):
                        warnings.append(f"{where}: no alternative of {g} is in the matter's documents")
            if kind == "other_matter":
                continue
            for src in [{"source": it.get("source"), "evidence": it.get("evidence")}, *(it.get("also") or [])]:
                title, ev = src.get("source"), src.get("evidence") or ""
                if kind == "not_stated" and not ev:
                    continue
                if title not in titles[matter]:
                    errors.append(f"{where}: source {title!r} is not a document of {matter}")
                    continue
                for seg in ev.split(" | "):
                    for piece in re.split(r"\s*(?:…|\.\.\.)\s*", seg):
                        if norm(piece) and norm(piece) not in texts[matter][title]:
                            errors.append(f"{where}: evidence not in {title!r}: {piece[:90]!r}")
                if src is not None and title == it.get("source"):
                    for s in steps:
                        for bad in s.get("must_not") or []:
                            if has(bad, texts[matter][title]):
                                warnings.append(f"{where}: must_not {bad!r} is in its own source document")
    for e in errors:
        print("ERROR", e)
    for w in warnings:
        print("WARN ", w)
    by_matter = Counter(m for m, _ in kinds.elements())
    by_kind = Counter(k for _, k in kinds.elements())
    print(f"{len(seen)} items; by matter {dict(sorted(by_matter.items()))}; by kind {dict(sorted(by_kind.items()))}; "
          f"{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
