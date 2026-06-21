"""
Document model for the Document Studio feature.

A generated document is a small, JSON-serializable structure:

    {
      "title": str, "subtitle": str, "org": str, "doc_type": str,
      "template_id": str, "persona": str,
      "blocks": [ <block>, ... ]
    }

Each block is a dict with a "type" discriminator. Keeping it a flat ordered list of
typed blocks (rather than nested sections) makes the renderers (PDF/PPTX) and the
frontend preview simple, while still supporting structure via heading levels + a derived
table of contents.

Block types (all fields are plain JSON):
  heading    {type, level:1-3, text}
  paragraph  {type, text}
  bullets    {type, items:[str], ordered?:bool}
  table      {type, columns:[str], rows:[[str]], caption?}
  flow       {type, title?, steps:[str]}          # rendered Step1 -> Step2 -> ...
  callout    {type, style:"info|warning|principle|success", title?, text}
  divider    {type}
  pagebreak  {type}

`normalize_document` coerces raw LLM output into a valid, safe structure so the renderers
never crash on malformed blocks.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

_LEAD_GLYPHS = "•·▪●‣◦∙*‒–—-"


def _strip_lead_glyph(s: str) -> str:
    """Drop leading bullet/dash glyphs the LLM sometimes prepends to a list item, so the
    renderer's own bullet marker is not doubled (e.g. '▪ • text')."""
    s = (s or "").lstrip()
    while s and s[0] in _LEAD_GLYPHS:
        s = s[1:].lstrip()
    return s


def _heading_key(text: str) -> str:
    """Normalized comparison key for a heading title (for de-duplication)."""
    s = (text or "").strip().lower()
    s = re.sub(r"^appendix\s*[:\-–]\s*", "", s)
    s = re.sub(r"^\d+[.\)]\s*", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

# Allowed block types and the maximum heading depth.
BLOCK_TYPES = {"heading", "paragraph", "bullets", "table", "flow", "callout", "image", "divider", "pagebreak"}
CALLOUT_STYLES = {"info", "warning", "principle", "success"}
MAX_HEADING_LEVEL = 3

# Hard caps so a runaway LLM response can't produce a multi-thousand-row table / huge doc.
_MAX_BLOCKS = 400
_MAX_TABLE_ROWS = 200
_MAX_TABLE_COLS = 12
_MAX_FLOW_STEPS = 40
_MAX_BULLETS = 60
_MAX_TEXT = 8000


_MD_BOLD = re.compile(r"(\*\*|__)(.+?)\1")
_JSON_FRAG = re.compile(r'"\s*,\s*"(?:text|title|caption|style|prompt|type|level|ordered|items|columns|rows)"\s*:\s*"?')


def _s(v: Any, limit: int = _MAX_TEXT) -> str:
    """Coerce any value to a trimmed string with a length cap. Strips leaked markdown bold
    markers and truncates on a word boundary (no ugly mid-word cuts)."""
    if v is None:
        return ""
    if isinstance(v, dict):
        v = _step_str(v)
    elif not isinstance(v, str):
        v = str(v)
    v = v.strip()
    if "**" in v or "__" in v:
        v = _MD_BOLD.sub(r"\2", v).replace("**", "").replace("__", "")
        v = v.strip()
    # Clean embedded JSON-key fragments that leak from malformed-LLM-output salvage
    # (e.g. 'People Over Planet","text":"At Stride...' -> 'People Over Planet — At Stride...').
    if '","' in v and _JSON_FRAG.search(v):
        v = _JSON_FRAG.sub(" — ", v)
        v = v.replace('"}', "").replace('{"', "").strip().strip('"').strip()
    if len(v) > limit:
        cut = v[:limit]
        sp = cut.rfind(" ")
        v = (cut[:sp] if sp > limit * 0.6 else cut).rstrip(" ,;:-")
    return v


def flatten_step(v: Any) -> str:
    """Coerce a flow step to readable text — handling live dicts AND already-stored Python
    dict-repr strings like "{'title': 'Awareness', 'text': '...'}" (from docs generated before
    step-dicts were flattened at normalization time)."""
    if isinstance(v, dict):
        return _step_str(v)
    s = str(v or "").strip()
    if len(s) > 6 and s[0] == "{" and s[-1] == "}" and ("':" in s or "': " in s):
        try:
            import ast
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                return _step_str(d)
        except Exception:
            pass
    return s


def _step_str(d: dict) -> str:
    """Flatten a step/item dict (e.g. {'title','text'}) into a readable string instead of
    leaking a Python repr like "{'title': 'Awareness', 'text': '...'}" into the document."""
    title = str(d.get("title") or d.get("step") or d.get("name") or d.get("label") or "").strip()
    body = str(d.get("text") or d.get("description") or d.get("detail") or d.get("value") or "").strip()
    if title and body:
        return f"{title}: {body}"
    if title or body:
        return title or body
    # Unknown shape: join scalar values, never dump the dict repr.
    vals = [str(x).strip() for x in d.values() if isinstance(x, (str, int, float)) and str(x).strip()]
    return " — ".join(vals)


def _norm_block(b: Any) -> Dict[str, Any] | None:
    """Validate/clean a single block. Returns None to drop an invalid block."""
    if not isinstance(b, dict):
        return None
    t = str(b.get("type", "")).strip().lower()
    if t not in BLOCK_TYPES:
        # Tolerate common aliases the LLM might emit.
        alias = {
            "title": "heading", "header": "heading", "subheading": "heading",
            "text": "paragraph", "para": "paragraph", "p": "paragraph",
            "list": "bullets", "bullet": "bullets", "ul": "bullets", "ol": "bullets",
            "note": "callout", "principle": "callout", "warning": "callout",
            "steps": "flow", "flowchart": "flow", "process": "flow",
            "hr": "divider", "break": "pagebreak",
        }
        t = alias.get(t, "")
        if t not in BLOCK_TYPES:
            # Last resort: if it has text, treat as a paragraph.
            txt = _s(b.get("text") or b.get("content") or "")
            return {"type": "paragraph", "text": txt} if txt else None

    if t == "heading":
        try:
            level = int(b.get("level", 2))
        except (TypeError, ValueError):
            level = 2
        level = max(1, min(MAX_HEADING_LEVEL, level))
        text = _s(b.get("text") or b.get("title") or "", 300)
        return {"type": "heading", "level": level, "text": text} if text else None

    if t == "paragraph":
        text = _s(b.get("text") or b.get("content") or "")
        return {"type": "paragraph", "text": text} if text else None

    if t == "bullets":
        raw = b.get("items") or b.get("bullets") or b.get("list") or []
        if isinstance(raw, str):
            raw = [x for x in raw.split("\n") if x.strip()]
        items = [_strip_lead_glyph(_s(x, 600)) for x in raw if _s(x, 600)]
        items = [x for x in items if x][:_MAX_BULLETS]
        return {"type": "bullets", "items": items, "ordered": bool(b.get("ordered"))} if items else None

    if t == "table":
        cols = b.get("columns") or b.get("headers") or b.get("cols") or []
        cols = [_s(c, 120) for c in cols][:_MAX_TABLE_COLS]
        raw_rows = b.get("rows") or b.get("data") or []
        rows: List[List[str]] = []
        for r in raw_rows[:_MAX_TABLE_ROWS]:
            if isinstance(r, dict):
                # row keyed by column name
                cells = [_s(r.get(c, ""), 1500) for c in cols] if cols else [_s(v, 1500) for v in r.values()]
            elif isinstance(r, (list, tuple)):
                cells = [_s(c, 1500) for c in r][:_MAX_TABLE_COLS]
            else:
                cells = [_s(r, 1500)]
            rows.append(cells)
        if not cols and rows:
            cols = [f"Column {i + 1}" for i in range(max(len(r) for r in rows))]
        # pad/truncate rows to column count
        n = len(cols)
        rows = [(r + [""] * n)[:n] for r in rows]
        if not cols and not rows:
            return None
        out = {"type": "table", "columns": cols, "rows": rows}
        cap = _s(b.get("caption") or b.get("title") or "", 300)
        if cap:
            out["caption"] = cap
        return out

    if t == "flow":
        raw = b.get("steps") or b.get("nodes") or b.get("items") or []
        if isinstance(raw, str):
            # split "A -> B -> C" or newline lists
            parts = [p for p in raw.replace("→", "->").split("->")] if "->" in raw else raw.split("\n")
            raw = parts
        steps = [_s(x, 300) for x in raw if _s(x, 300)][:_MAX_FLOW_STEPS]
        out = {"type": "flow", "steps": steps}
        title = _s(b.get("title") or "", 200)
        if title:
            out["title"] = title
        return out if steps else None

    if t == "callout":
        style = str(b.get("style", "info")).strip().lower()
        if style not in CALLOUT_STYLES:
            style = "info"
        text = _s(b.get("text") or b.get("content") or "")
        out = {"type": "callout", "style": style, "text": text}
        title = _s(b.get("title") or "", 200)
        if title:
            out["title"] = title
        return out if (text or title) else None

    if t == "image":
        # prompt = what to generate; data_url filled in by the image stage (base64 PNG) if available.
        out = {
            "type": "image",
            "prompt": _s(b.get("prompt") or b.get("description") or b.get("alt") or "", 600),
            "caption": _s(b.get("caption") or "", 300),
            "alt": _s(b.get("alt") or b.get("caption") or "", 300),
            "kind": (str(b.get("kind", "illustration")).strip().lower() or "illustration")[:40],
        }
        du = b.get("data_url")
        if isinstance(du, str) and du.startswith("data:image"):
            out["data_url"] = du[:6_000_000]  # cap ~4.5MB base64
        # keep only if it can ever render: has a generation prompt or already-embedded data
        return out if (out["prompt"] or out.get("data_url")) else None

    if t == "divider":
        return {"type": "divider"}
    if t == "pagebreak":
        return {"type": "pagebreak"}
    return None


def _dedupe_headings(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop a heading that repeats the immediately-preceding heading (only dividers/blanks
    between them) — the planner often emits the same section title at two levels (e.g. an L2
    'Micro-Segmentation' right before an L1 'Micro-Segmentation'). Keeps the more important
    (lower) level on the surviving heading."""
    out: List[Dict[str, Any]] = []
    last_idx = None
    last_key = None
    content_since = True
    for b in blocks:
        if b.get("type") == "heading":
            key = _heading_key(b.get("text", ""))
            if key and key == last_key and not content_since and last_idx is not None:
                try:
                    out[last_idx]["level"] = min(int(out[last_idx].get("level", 2)),
                                                 int(b.get("level", 2)))
                except Exception:
                    pass
                continue  # drop the duplicate heading
            out.append(dict(b))
            last_idx = len(out) - 1
            last_key = key
            content_since = False
        else:
            # Drop a paragraph that merely repeats the heading right above it.
            if (b.get("type") == "paragraph" and not content_since and last_key
                    and _heading_key(b.get("text", "")) == last_key):
                continue
            if b.get("type") not in ("divider", "pagebreak"):
                content_since = True
            out.append(b)
    return out


def _drop_empty_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove a heading that has no real content before the next heading/end (an orphaned,
    empty section), which the renderers would otherwise show as a heading over a blank gap."""
    n = len(blocks)
    out: List[Dict[str, Any]] = []
    for i, b in enumerate(blocks):
        if b.get("type") == "heading":
            has_content = False
            for j in range(i + 1, n):
                t = blocks[j].get("type")
                if t == "heading":
                    break
                if t not in ("divider", "pagebreak"):
                    has_content = True
                    break
            if not has_content:
                continue
        out.append(b)
    return out


def normalize_blocks(raw_blocks: Any) -> List[Dict[str, Any]]:
    """Coerce a raw list of blocks into a clean, capped list of valid blocks."""
    if not isinstance(raw_blocks, list):
        return []
    out: List[Dict[str, Any]] = []
    for b in raw_blocks[:_MAX_BLOCKS]:
        nb = _norm_block(b)
        if nb is not None:
            out.append(nb)
    return _drop_empty_sections(_dedupe_headings(out))


def normalize_document(raw: Dict[str, Any], *, template_id: str = "", persona: str = "") -> Dict[str, Any]:
    """Coerce raw pipeline/LLM output into a valid, safe document dict."""
    raw = raw if isinstance(raw, dict) else {}
    doc = {
        "title": _s(raw.get("title") or "Untitled Document", 300),
        "subtitle": _s(raw.get("subtitle") or "", 400),
        "org": _s(raw.get("org") or "", 200),
        "doc_type": _s(raw.get("doc_type") or "Document", 120),
        "template_id": _s(raw.get("template_id") or template_id, 80),
        "persona": _s(raw.get("persona") or persona, 80),
        "blocks": normalize_blocks(raw.get("blocks")),
    }
    return doc


def build_toc(doc: Dict[str, Any], max_level: int = 2) -> List[Dict[str, Any]]:
    """Derive a table of contents from heading blocks (levels 1..max_level)."""
    toc = []
    for b in doc.get("blocks", []):
        if b.get("type") == "heading" and int(b.get("level", 99)) <= max_level:
            toc.append({"level": int(b["level"]), "text": b.get("text", "")})
    return toc


def document_stats(doc: Dict[str, Any]) -> Dict[str, int]:
    """Quick counts for UI / logging."""
    blocks = doc.get("blocks", [])
    by: Dict[str, int] = {}
    for b in blocks:
        by[b.get("type", "?")] = by.get(b.get("type", "?"), 0) + 1
    return {"blocks": len(blocks), **{f"{k}s": v for k, v in by.items()}}
