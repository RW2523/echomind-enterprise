"""Word tracked changes (S3.3, DOC-02b; decision D-047 — the D-011 spike).

A revision of a clause is written into the matter's own Word file as real tracked changes:
- **Finding the clause.** A clause is one or more consecutive paragraphs ("12.3 Neither party is
  liable for:", "(a) …", "(b) …"). Numbers Word generates ("12.3", "(a)") are not in the
  paragraphs' text, so a leading number is ignored when matching. Paragraphs are matched by their
  words; a clause that cannot be found is reported, never guessed.
- **Writing the change.** Paragraphs are paired in order. In each pair the runs are rewritten as a
  word-level diff: unchanged text as plain runs, removed text in `w:del`/`w:delText`, added text in
  `w:ins`, each with `w:author` and `w:date`. The paragraph's properties (style, numbering) are kept
  and the runs take the formatting of its first run. An original paragraph with no counterpart is
  deleted whole (its runs and its paragraph mark); a new paragraph is inserted whole, after the last.
- **The reason** is a comment anchored on the clause (`word/comments.xml` with its relationship and
  content type, added when the file has none).
- A document that is not a Word file (text, PDF) is first written as a plain Word file, one
  paragraph per line.

`verify` checks every file before release (T10): accepting all changes gives the original with each
clause replaced by its new wording and nothing else; rejecting all gives the original exactly; and
every change carries an author and a date. Existing tracked changes in the file are left as they are.
"""
from __future__ import annotations

import copy
import io
import re
import zipfile
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
COMMENTS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments"
COMMENTS_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"
_NUMBER = re.compile(r"^\s*(?:(?:Article|Section|Clause|Schedule|Part)\s+)?\(?(?:\d+(?:\.\d+)*|[A-Za-z]|[ivxlcdmIVXLCDM]+)[.)]?\s+")
_FOOTNOTE = re.compile(r"\[fn \d+\]")



# Stored Word files come from outside the firm: never resolve entities, load a DTD or reach the network while reading them
# (commercial-readiness review F12, 2026-10-28; import refuses no DOCTYPE, so this parser is the guard).
_SAFE = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)


def _xml(data):
    return etree.fromstring(data, _SAFE)

def w(tag: str) -> str:
    return f"{{{W}}}{tag}"


def tokens(text: str) -> list[str]:
    return re.findall(r"\s+|[^\s]+", text or "")


def ops(old: str, new: str) -> list[tuple[str, str]]:
    """A word-level diff: [("eq" | "del" | "ins", text)]. Changes separated only by spaces read as one
    ("thirty (30)" → "ninety (90)", not two), deletions before insertions."""
    a, b = tokens(old), tokens(new)
    out: list[tuple[str, str]] = []
    dels, inss = "", ""

    def flush():
        nonlocal dels, inss
        if dels:
            out.append(("del", dels))
        if inss:
            out.append(("ins", inss))
        dels, inss = "", ""
    raw = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    for k, (tag, i1, i2, j1, j2) in enumerate(raw):
        if tag == "equal":
            piece = "".join(a[i1:i2])
            more = k + 1 < len(raw)
            if (dels or inss) and more and not piece.strip():     # only spaces between two changes: one change
                dels += piece
                inss += piece
                continue
            flush()
            if out and out[-1][0] == "eq":
                out[-1] = ("eq", out[-1][1] + piece)
            else:
                out.append(("eq", piece))
        else:
            dels += "".join(a[i1:i2])
            inss += "".join(b[j1:j2])
    flush()
    return [(op, t) for op, t in out if t]


def fold(text: str) -> str:
    text = _FOOTNOTE.sub("", text or "")                   # the reader's footnote marks are not in the paragraph
    return re.sub(r"\s+", " ", (text or "").replace("’", "'").replace("‘", "'").replace("“", '"')
                  .replace("”", '"')).strip()


def parts(text: str) -> list[str]:
    """A clause's paragraphs, as the reader joins them (blank lines between)."""
    return [p.strip() for p in re.split(r"\n\s*\n|\n", text or "") if p.strip()]


def strip_number(text: str) -> str:
    return _NUMBER.sub("", text, count=1)


def _para_text(p) -> str:
    """A paragraph's text with tracked deletions left out (what a reader sees)."""
    out = []
    for el in p.iter(w("t"), w("tab"), w("br")):
        if any(a.tag == w("del") for a in el.iterancestors()):
            continue
        run = next((a for a in el.iterancestors() if a.tag == w("r")), None)
        if run is not None and run.find(f"{w('rPr')}/{w('vanish')}") is not None:
            continue                                       # hidden text, which the reader leaves out
        out.append((el.text or "") if el.tag == w("t") else ("\t" if el.tag == w("tab") else "\n"))
    return "".join(out)


def plain_docx(text: str) -> bytes:
    import docx
    d = docx.Document()
    for line in (text or "").split("\n"):
        d.add_paragraph(line.rstrip())
    cp = d.core_properties
    cp.author, cp.last_modified_by = "", ""
    out = io.BytesIO()
    d.save(out)
    return out.getvalue()


def _match(paragraphs: list, want: list[str]) -> Optional[tuple[list[int], list[bool]]]:
    """Indexes of the consecutive non-empty paragraphs holding `want`, and for each whether its
    leading number was generated by Word (so not in its text)."""
    for i in range(len(paragraphs)):
        idx, stripped, k, j = [], [], 0, i
        while k < len(want) and j < len(paragraphs):
            have = fold(_para_text(paragraphs[j]))
            if not have:
                if idx:
                    j += 1
                    continue
                break
            if have == fold(want[k]):
                idx.append(j); stripped.append(False)
            elif have == fold(strip_number(want[k])):
                idx.append(j); stripped.append(True)
            else:
                break
            k += 1
            j += 1
        if k == len(want) and idx:
            return idx, stripped
    return None


def _run(rpr, text: str, deleted: bool = False):
    r = etree.Element(w("r"))
    if rpr is not None:
        r.append(copy.deepcopy(rpr))
    t = etree.SubElement(r, w("delText") if deleted else w("t"))
    t.text = text
    t.set(XML_SPACE, "preserve")
    return r


class _Ids:
    def __init__(self, doc):
        found = [int(x) for x in doc.xpath("//w:ins/@w:id | //w:del/@w:id | //w:commentRangeStart/@w:id", namespaces={"w": W})
                 if str(x).isdigit()]
        self.n = max(found, default=0) + 1

    def __call__(self) -> str:
        self.n += 1
        return str(self.n - 1)


def _mark(tag: str, author: str, when: str, ids: _Ids):
    el = etree.Element(w(tag))
    el.set(w("id"), ids())
    el.set(w("author"), author)
    el.set(w("date"), when)
    return el


def _mark_paragraph(p, tag: str, author: str, when: str, ids: _Ids) -> None:
    """Record the paragraph mark itself as inserted or deleted (w:pPr/w:rPr/w:ins|w:del)."""
    ppr = p.find(w("pPr"))
    if ppr is None:
        ppr = etree.Element(w("pPr"))
        p.insert(0, ppr)
    rpr = ppr.find(w("rPr"))
    if rpr is None:
        rpr = etree.SubElement(ppr, w("rPr"))
    rpr.append(_mark(tag, author, when, ids))


def _rewrite(p, current: str, revised: str, author: str, when: str, ids: _Ids) -> None:
    """The paragraph's visible text rewritten as a diff. Hidden runs and bookmarks are kept as they
    were (after the new text), so nothing leaves the file without a tracked change."""
    visible = [r for r in p.iter(w("r")) if r.find(f"{w('rPr')}/{w('vanish')}") is None and r.find(w("t")) is not None]
    first = visible[0] if visible else p.find(f".//{w('r')}")
    rpr = first.find(w("rPr")) if first is not None else None
    ppr = p.find(w("pPr"))
    keep = [c for c in p if c is not ppr and (c.tag in (w("bookmarkStart"), w("bookmarkEnd")) or
                                              (c.tag == w("r") and c.find(f"{w('rPr')}/{w('vanish')}") is not None))]
    for child in list(p):
        if child is not ppr:
            p.remove(child)
    for op, piece in ops(current, revised):
        if op == "eq":
            p.append(_run(rpr, piece))
        else:
            wrap = _mark("del" if op == "del" else "ins", author, when, ids)
            wrap.append(_run(rpr, piece, deleted=op == "del"))
            p.append(wrap)
    for c in keep:
        p.append(c)


def _ensure_comments(z: dict) -> etree._Element:
    if "word/comments.xml" not in z:
        z["word/comments.xml"] = etree.tostring(etree.Element(w("comments"), nsmap={"w": W}), xml_declaration=True, encoding="UTF-8",
                                                standalone=True)
        rels = _xml(z["word/_rels/document.xml.rels"])
        taken = {r.get("Id") for r in rels}
        n = 1
        while f"rIdCm{n}" in taken:
            n += 1
        rel = etree.SubElement(rels, f"{{{PKG}}}Relationship")
        rel.set("Id", f"rIdCm{n}")
        rel.set("Type", COMMENTS_REL)
        rel.set("Target", "comments.xml")
        z["word/_rels/document.xml.rels"] = etree.tostring(rels, xml_declaration=True, encoding="UTF-8", standalone=True)
        ct = _xml(z["[Content_Types].xml"])
        if not any(o.get("PartName") == "/word/comments.xml" for o in ct):
            o = etree.SubElement(ct, f"{{{CT}}}Override")
            o.set("PartName", "/word/comments.xml")
            o.set("ContentType", COMMENTS_CT)
            z["[Content_Types].xml"] = etree.tostring(ct, xml_declaration=True, encoding="UTF-8", standalone=True)
    return _xml(z["word/comments.xml"])


def _comment(comments, p_first, p_last, text: str, author: str, when: str, ids: _Ids) -> None:
    cid = ids()
    start = etree.Element(w("commentRangeStart"))
    start.set(w("id"), cid)
    ppr = p_first.find(w("pPr"))
    p_first.insert(1 if ppr is not None else 0, start)
    end = etree.SubElement(p_last, w("commentRangeEnd"))
    end.set(w("id"), cid)
    ref = etree.SubElement(etree.SubElement(p_last, w("r")), w("commentReference"))
    ref.set(w("id"), cid)
    c = etree.SubElement(comments, w("comment"))
    c.set(w("id"), cid)
    c.set(w("author"), author)
    c.set(w("date"), when)
    c.set(w("initials"), "".join(x[0] for x in author.split()[:3]).upper() or "L")
    t = etree.SubElement(etree.SubElement(etree.SubElement(c, w("p")), w("r")), w("t"))
    t.text = text


def apply(source: Optional[bytes], text: str, changes: list[dict], author: str, when: Optional[str] = None) -> tuple[bytes, dict]:
    """The Word file with each change as tracked changes, and the plan `verify` checks it against:
    {"missed": [change ids not found], "placed": [{"id", "first", "count", "revised"}] (paragraph
    positions in the original)}. `changes`: [{"id", "original", "revised", "comment"}]."""
    data = source if source else plain_docx(text)
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        z = {i.filename: zin.read(i.filename) for i in zin.infolist()}
        order = [i.filename for i in zin.infolist()]
    doc = _xml(z["word/document.xml"])
    body = doc.find(w("body"))
    paragraphs = [p for p in body.iter(w("p"))]
    when = when or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ids = _Ids(doc)
    comments = None
    plan = {"missed": [], "placed": [], "source": "word" if source else "text"}
    for ch in changes:
        want, new = parts(ch["original"]), parts(ch["revised"])
        found = _match(paragraphs, want)
        if not found:
            plan["missed"].append(ch["id"])
            continue                                         # an empty revision deletes the clause whole (S3.5)
        idx, stripped = found
        targets = [paragraphs[i] for i in idx]
        shown = []
        for k, p in enumerate(targets):
            if k < len(new):
                rev = strip_number(new[k]) if stripped[k] else new[k]
                _rewrite(p, _para_text(p), rev, author, when, ids)
                shown.append(rev)
            else:                                            # no counterpart: deleted whole
                _rewrite(p, _para_text(p), "", author, when, ids)
                _mark_paragraph(p, "del", author, when, ids)
        last = targets[-1]
        for extra in new[len(targets):]:                     # new paragraphs, inserted whole
            p = copy.deepcopy(last)
            ppr = p.find(w("pPr"))
            for child in list(p):
                if child is not ppr:
                    p.remove(child)
            if ppr is not None:                              # an inserted paragraph keeps the style, not old marks
                for old in ppr.findall(w("rPr")):
                    ppr.remove(old)
            if stripped[-1]:                                # Word numbers it, as it numbered the one copied
                extra = strip_number(extra)
            wrap = _mark("ins", author, when, ids)
            wrap.append(_run(None, extra))
            p.append(wrap)
            _mark_paragraph(p, "ins", author, when, ids)
            last.addnext(p)
            last = p
            shown.append(extra)
        if comments is None:
            comments = _ensure_comments(z)
        _comment(comments, targets[0], targets[min(len(new), len(targets)) - 1] if new else targets[0],
                 ch.get("comment") or "Revised to the firm's position.", author, when, ids)
        plan["placed"].append({"id": ch["id"], "indexes": idx, "revised": shown})
    z["word/document.xml"] = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
    if comments is not None:
        z["word/comments.xml"] = etree.tostring(comments, xml_declaration=True, encoding="UTF-8", standalone=True)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in order + [n for n in z if n not in order]:
            zout.writestr(name, z[name])
    return out.getvalue(), plan


def _resolve(data: bytes, accept: bool) -> list[str]:
    """The paragraphs' text once every tracked change is accepted (or rejected)."""
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        doc = _xml(zin.read("word/document.xml"))
    gone, kept = (w("del"), w("ins")) if accept else (w("ins"), w("del"))
    drop = [p for p in doc.iter(w("p")) if p.find(f"{w('pPr')}/{w('rPr')}/{gone}") is not None]
    for el in list(doc.iter(gone)):
        if el.getparent() is not None and el.getparent().tag != w("rPr"):
            el.getparent().remove(el)
    for el in list(doc.iter(kept)):
        parent = el.getparent()
        if parent is None or parent.tag == w("rPr"):
            continue
        i = parent.index(el)
        for k, child in enumerate(list(el)):
            parent.insert(i + k, child)
        parent.remove(el)
    if not accept:
        for t in doc.iter(w("delText")):
            t.tag = w("t")
    return [_para_text(p) for p in doc.iter(w("p")) if p not in drop]


def accept_all(data: bytes) -> list[str]:
    return _resolve(data, accept=True)


def reject_all(data: bytes) -> list[str]:
    return _resolve(data, accept=False)


def _paragraphs_of(data: bytes) -> list:
    with zipfile.ZipFile(io.BytesIO(data)) as zin:
        return list(_xml(zin.read("word/document.xml")).iter(w("p")))


def verify(original: bytes, result: bytes, plan: dict) -> dict:
    """T10 checks of a redlined file against the original it was made from."""
    with zipfile.ZipFile(io.BytesIO(result)) as zin:
        doc = _xml(zin.read("word/document.xml"))
        names = set(zin.namelist())
    base_reject = reject_all(original)
    # the original with its own tracked changes accepted, paragraph by paragraph, and each placed
    # change replayed on exactly the paragraphs it matched
    with zipfile.ZipFile(io.BytesIO(original)) as zin:
        odoc = _xml(zin.read("word/document.xml"))
    o_pars = list(odoc.iter(w("p")))
    dropped = {i for i, p in enumerate(o_pars) if p.find(f"{w('pPr')}/{w('rPr')}/{w('del')}") is not None}
    accepted = accept_all(original)
    text_of = dict(zip([i for i in range(len(o_pars)) if i not in dropped], accepted))
    role = {}
    for pl in plan["placed"]:
        for k, i in enumerate(pl["indexes"]):
            role[i] = (pl, k)
    expected = []
    for i in range(len(o_pars)):
        if i in dropped:
            continue
        if i not in role:
            expected.append(fold(text_of[i]))
            continue
        pl, k = role[i]
        if k < len(pl["revised"]):
            expected.append(fold(pl["revised"][k]))
        if k == len(pl["indexes"]) - 1:
            expected += [fold(t) for t in pl["revised"][len(pl["indexes"]):]]
    marks = doc.xpath("//w:ins | //w:del", namespaces={"w": W})
    checks = {
        "reject_all_is_the_original": [fold(t) for t in reject_all(result)] == [fold(t) for t in base_reject],
        "accept_all_is_the_revision": [fold(t) for t in accept_all(result)] == expected,
        "every_change_has_author_and_date": all(m.get(w("author")) and m.get(w("date")) for m in marks),
        "comments_part": not plan["placed"] or "word/comments.xml" in names,
    }
    return {"ok": all(checks.values()), "checks": checks}
