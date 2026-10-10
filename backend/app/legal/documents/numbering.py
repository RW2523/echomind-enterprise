"""Clause numbers as a lawyer reads them (S3.5; used by the S3.4 and S3.5 checkers).

The reader (S2.1) stores the clauses it sees. A Markdown heading ("## 4. Fees and Payment") is not read
as clause 4, so a document can have clauses 4.1 and 4.2 but no clause 4. `complete` adds the clauses
such numbers imply, spanning their sub-clauses, with the heading from the heading line when there is
one, so that "clause 4" resolves and numbering can be checked at every level.
"""
from __future__ import annotations

import re
from typing import Optional

HEAD_LINE = re.compile(r"^[ \t]*#{0,6}[ \t]*(?:(?:Clause|Section|CLAUSE|SECTION|Article|ARTICLE)[ \t]+)?(\d{1,3}(?:\.\d{1,3})*)\.?[ \t]+"
                       r"([A-Z][^\n]{0,80}?)[ \t]*$", re.M)
ATTACHMENT = re.compile(r"^(Schedule|Annex|Exhibit|Appendix) (\S+)$", re.I)
_NUM = re.compile(r"^(?:§\s*)?(\d{1,3}(?:\.\d{1,3})*)((?:\([a-z0-9]{1,5}\))*)$")


def _is_heading(s: str) -> bool:
    """'Fees and Payment', not 'Liability is limited as set out in clause 12.'"""
    s = s.strip()
    return (len(s.split()) <= 8 and not re.search(r"[.;:,]$", s)
            and not re.search(r"\b(?:shall|may|must|will|is|are|means|has|have|provides?)\b", s))


def split(number: str) -> Optional[tuple[list[int], list[str]]]:
    """'9.2(a)(ii)' → ([9, 2], ['a', 'ii']); None for 'Schedule 1', 'Article IV' and the like."""
    m = _NUM.match(number.strip())
    if not m:
        return None
    return [int(x) for x in m.group(1).split(".")], re.findall(r"\(([a-z0-9]{1,5})\)", m.group(2))


ATTACH_LINE = re.compile(r"^[ \t]*#{1,6}[ \t]*(Schedule|Annex|Exhibit|Appendix|SCHEDULE|ANNEX|EXHIBIT|APPENDIX)[ \t]+(\d{1,3}|[A-Z])\b[ \t—–:-]*([^\n]*)$", re.M)


def complete(text: str, rows: list[dict]) -> list[dict]:
    """Clause rows (number, heading, level, start, end) plus the numbered clauses they imply, and schedules
    known only by a Markdown heading ("## Schedule 1 Products")."""
    have = {r["number"] for r in rows}
    marks = list(ATTACH_LINE.finditer(text))
    for i, m in enumerate(marks):
        number = f"{m.group(1).title()} {m.group(2)}"
        if number in have:
            continue
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        rows = rows + [{"number": number, "heading": m.group(3).strip() or None, "level": 0, "parent": None, "start": m.start(),
                        "end": len(text[:end].rstrip()), "implied": True}]
        have.add(number)
    heads = {m.group(1): (m.group(2).strip(), m.start()) for m in HEAD_LINE.finditer(text) if _is_heading(m.group(2))}
    boxes = [(r["start"], r["end"]) for r in rows if ATTACHMENT.match(r["number"])]
    main = [r for r in rows if not ATTACHMENT.match(r["number"]) and not any(a < r["start"] < b for a, b in boxes)]
    have = {r["number"] for r in main}                     # a schedule's own "1." does not hide clause 1 of the body
    children: dict[str, list[dict]] = {}
    for r in main:
        sp = split(r["number"])
        if not sp or sp[1]:
            continue
        nums = sp[0]
        for k in range(1, len(nums)):
            children.setdefault(".".join(map(str, nums[:k])), []).append(r)
    out = list(rows)
    for num, kids in children.items():
        if num in have:
            continue
        head = heads.get(num)
        start = min([k["start"] for k in kids] + ([head[1]] if head and head[1] < min(k["start"] for k in kids) else []))
        out.append({"number": num, "heading": head[0] if head else None, "level": num.count(".") + 1, "parent": None,
                    "start": start, "end": max(k["end"] for k in kids), "implied": True})
    for r in out:
        if not r.get("heading") and r["number"] in heads and split(r["number"]) and not split(r["number"])[1]:
            r["heading"] = heads[r["number"]][0]
    return sorted(out, key=lambda r: (r["start"], r["level"]))
