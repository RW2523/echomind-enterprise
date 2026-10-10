"""/api/legal/matters/{id}/compare (S2.3, RAG-02; decision D-039).

- `POST …/compare`: two versions, clause by clause. Computed from the stored text, so exact.
- `POST …/compare/{cid}/explain`: the model explains each change, quoting both wordings (streamed).
- `POST …/compare/scan`: conflicts across the matter's documents (streamed).
- `GET …/compare` and `GET …/compare/{cid}`: the person's own comparisons and scans.

Each request needs `research.query` on the matter, as research does, and reads only versions the
person may read (matter access and source permissions). Saved comparisons and scans are private to
their owner. They are shown again only while every document they quote is still visible to the
owner; findings from a document they can no longer see are left out. Streams re-check the scope
as they go. Audit records hold counts, never content.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ...core.config import settings
from ...core.db import get_conn
from .. import policy
from ..audit.record import audited, note, record
from ..documents import repo
from ..matters.routes import principal_of, require, require_law_edition
from ..research import llm as llm_mod
from ..retrieval import scope as scopemod
from . import engine, scan

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])
_BASE = "/matters/{matter_id}/compare"


class CompareIn(BaseModel):
    left_version_id: str = Field(min_length=1, max_length=64)
    right_version_id: str = Field(min_length=1, max_length=64)


class ExplainIn(BaseModel):
    pairs: Optional[list[int]] = Field(default=None, max_length=200)


class ScanIn(BaseModel):
    versions: Literal["signed", "current"] = "signed"


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


def _event(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _insert(row: dict) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO legal_comparisons (id, org_id, matter_id, owner_id, kind, status, versions, document_ids_json, "
                     "version_ids_json, result_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (row["id"], row["org_id"], row["matter_id"], row["owner_id"], row["kind"], row["status"], row.get("versions"),
                      json.dumps(row["document_ids"]), json.dumps(row["version_ids"]), engine.dumps(row["result"]),
                      row["created_at"], row["created_at"]))
        conn.commit()


def _update(cid: str, result: dict, status: Optional[str] = None) -> None:
    with get_conn() as conn:
        conn.execute("UPDATE legal_comparisons SET result_json=?, status=COALESCE(?, status), updated_at=? WHERE id=?",
                     (engine.dumps(result), status, policy.now_iso(), cid))
        conn.commit()


_COLS = ("id", "kind", "status", "versions", "document_ids_json", "version_ids_json", "result_json", "created_at", "updated_at")


def _own(matter_id: str, cid: str, uid: str) -> Optional[dict]:
    with get_conn() as conn:
        r = conn.execute(f"SELECT {', '.join(_COLS)} FROM legal_comparisons WHERE id=? AND matter_id=? AND owner_id=?",
                         (cid, matter_id, uid)).fetchone()
    if not r:
        return None
    row = dict(zip(_COLS, r), matter_id=matter_id)
    row["document_ids"], row["version_ids"] = json.loads(row.pop("document_ids_json")), json.loads(row.pop("version_ids_json"))
    row["result"] = json.loads(row.pop("result_json"))
    return row


def _visible_documents(matter_id: str, document_ids: list[str], uid: str) -> set[str]:
    return {d for d in document_ids if repo.visible(matter_id, d, uid)}


def _texts(version_ids: list[str]) -> dict[str, str]:
    qs = ",".join("?" * len(version_ids)) or "NULL"
    with get_conn() as conn:
        return {r[0]: r[1] for r in conn.execute(f"SELECT version_id, text FROM legal_version_text WHERE version_id IN ({qs})",
                                                 version_ids)}


def _versions_out(row: dict, uid: str, matter_id: str) -> Optional[dict]:
    if _visible_documents(matter_id, row["document_ids"], uid) != set(row["document_ids"]):
        return None                                          # fail closed: a side is no longer visible
    texts = _texts(row["version_ids"])
    if len(texts) != len(set(row["version_ids"])):
        return None
    out = engine.hydrate(row["result"], texts)
    return {"id": row["id"], "kind": "versions", "status": row["status"], "created_at": row["created_at"], **out}


def _scan_out(row: dict, uid: str, matter_id: str) -> dict:
    visible = _visible_documents(matter_id, row["document_ids"], uid)
    res = row["result"]
    keep = lambda f: f["a"]["document_id"] in visible and f["b"]["document_id"] in visible   # noqa: E731
    findings, rejected = [f for f in res.get("findings", []) if keep(f)], [f for f in res.get("rejected", []) if keep(f)]
    hidden = len(res.get("findings", [])) + len(res.get("rejected", [])) - len(findings) - len(rejected)
    return {"id": row["id"], "kind": "scan", "status": row["status"], "created_at": row["created_at"], "versions": row["versions"],
            "documents": [d for d in res.get("documents", []) if d["document_id"] in visible], "findings": findings,
            "rejected": rejected, "candidates": res.get("candidates", 0), "hidden": hidden}


# ── Two versions ─────────────────────────────────────────────────────────────────
@router.post(_BASE)
@audited("compare.run", object_type="matter", object_param="matter_id")
async def compare_versions(matter_id: str, inp: CompareIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    try:
        left = await run_in_threadpool(engine.load_version, matter_id, inp.left_version_id, p["sub"])
        right = await run_in_threadpool(engine.load_version, matter_id, inp.right_version_id, p["sub"])
    except engine.NotRead:
        raise HTTPException(status_code=409, detail="A version was read before comparison existed. Read it again to compare it.")
    if not left or not right:
        raise _not_found()
    if left["version_id"] == right["version_id"]:
        raise HTTPException(status_code=422, detail="Choose two different versions.")
    result = await engine.compare(left, right)
    result = {"left": engine.meta(left), "right": engine.meta(right), **result}
    row = {"id": str(uuid.uuid4()), "org_id": p["org_id"], "matter_id": matter_id, "owner_id": p["sub"], "kind": "versions",
           "status": "done", "document_ids": sorted({left["document_id"], right["document_id"]}),
           "version_ids": [left["version_id"], right["version_id"]], "result": result, "created_at": policy.now_iso()}
    await run_in_threadpool(_insert, row)
    note(object_id=row["id"], details={"pairs": len(result["pairs"]), **result["summary"], "passes": result["passes"]})
    out = engine.hydrate(result, {left["version_id"]: left["text"], right["version_id"]: right["text"]})
    return JSONResponse({"comparison": {"id": row["id"], "kind": "versions", "status": "done", "created_at": row["created_at"], **out}},
                        headers={"Cache-Control": "no-store"})


async def _explain(scope, principal: dict, row: dict, left: dict, right: dict, wanted: list[int]):
    result, done, status = row["result"], 0, "done"
    client = llm_mod.get()
    last = time.monotonic()
    try:
        for i in wanted:
            if time.monotonic() - last >= settings.LEGAL_STREAM_RECHECK_SEC:
                await run_in_threadpool(scopemod.recheck, scope)
                last = time.monotonic()
            pair = result["pairs"][i]
            msgs, sources = engine.explain_messages(left, right, pair)
            try:
                raw = await client.chat(msgs, temperature=0.1, max_tokens=320)
            except Exception as e:
                logger.warning("Compare: an explanation failed (%s)", type(e).__name__)
                yield _event({"type": "explanation_error", "index": i})
                continue
            bases = [pair["left"]["start"] if pair.get("left") else None, pair["right"]["start"] if pair.get("right") else None]
            ex = engine.check(raw, sources, bases)
            pair["explanation"] = ex
            done += 1
            yield _event({"type": "explanation", "index": i, **ex})
        await run_in_threadpool(scopemod.recheck, scope)
        yield _event({"type": "done", "explained": done})
    except scopemod.ScopeRevoked:
        status = "scope_revoked"
        yield _event({"type": "scope_revoked"})
    except Exception as e:
        status = "error"
        logger.error("Compare: explanations failed (%s)", type(e).__name__)
        yield _event({"type": "error", "message": "The explanations could not be completed. Try again."})
    finally:
        try:
            await run_in_threadpool(_update, row["id"], result)
            quotes = [q for i in wanted for q in (result["pairs"][i].get("explanation") or {}).get("evidence", [])]
            record("compare.explained", "deny" if status == "scope_revoked" else "error" if status == "error" else "allow",
                   principal=principal, object_type="comparison", object_id=row["id"], matter_id=row["matter_id"], reason=status,
                   details={"explained": done, "quotes": len(quotes),
                            "verified": sum(1 for q in quotes if q["verification"] in ("verbatim", "normalized"))})
        except Exception as e:
            logger.error("Compare: could not record the explanations (%s)", type(e).__name__)


@router.post(_BASE + "/{comparison_id}/explain")
@audited("compare.explain", object_type="comparison", object_param="comparison_id")
async def explain(matter_id: str, comparison_id: str, inp: ExplainIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    row = await run_in_threadpool(_own, matter_id, comparison_id, p["sub"])
    if not row or row["kind"] != "versions":
        raise _not_found()
    try:
        left = await run_in_threadpool(engine.load_version, matter_id, row["version_ids"][0], p["sub"])
        right = await run_in_threadpool(engine.load_version, matter_id, row["version_ids"][1], p["sub"])
    except engine.NotRead:
        raise _not_found()
    if not left or not right:
        raise _not_found()
    pairs = row["result"]["pairs"]
    if inp.pairs is not None:
        wanted = [i for i in dict.fromkeys(inp.pairs) if 0 <= i < len(pairs) and pairs[i]["kind"] in ("changed", "added", "removed")]
    else:
        wanted = [i for i, x in enumerate(pairs) if x["kind"] in ("changed", "added", "removed") and not x.get("explanation")]
    wanted = wanted[:settings.LEGAL_COMPARE_EXPLAIN_MAX]
    note(details={"pairs": len(wanted)})
    try:
        scope = await run_in_threadpool(scopemod.authorize, p, matter_id)
    except scopemod.ScopeDenied:
        raise _not_found()
    return StreamingResponse(_explain(scope, p, row, left, right, wanted), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ── Across documents ─────────────────────────────────────────────────────────────
async def _scan(scope, principal: dict, row: dict, policy_name: str):
    findings, rejected, status, total, docs = [], [], "done", 0, []
    result = {"documents": [], "findings": findings, "rejected": rejected, "candidates": 0}
    try:
        docs = await run_in_threadpool(scan.documents, row["matter_id"], principal["sub"], policy_name)
        result["documents"] = [{k: d[k] for k in ("document_id", "document_title", "version_id", "number", "status")} for d in docs]
        row["document_ids"] = [d["document_id"] for d in docs]
        row["version_ids"] = [d["version_id"] for d in docs]
        found = await run_in_threadpool(scan.candidates, docs)
        total = result["candidates"] = len(found)
        yield _event({"type": "documents", "documents": result["documents"], "candidates": total})
        last, checked = time.monotonic(), 0
        async for c in scan.run_checks(found):
            if time.monotonic() - last >= settings.LEGAL_STREAM_RECHECK_SEC:
                await run_in_threadpool(scopemod.recheck, scope)
                last = time.monotonic()
            checked += 1
            (rejected if c["verdict"] == "no_conflict" else findings).append(c)
            yield _event({"type": "checked", "done": checked, "total": total, "finding": c})
        await run_in_threadpool(scopemod.recheck, scope)
        yield _event({"type": "done", "id": row["id"], "found": sum(1 for f in findings if f["verdict"] == "conflict"),
                      "unclear": sum(1 for f in findings if f["verdict"] == "unclear"), "rejected": len(rejected)})
    except scopemod.ScopeRevoked:
        status = "scope_revoked"
        yield _event({"type": "scope_revoked"})
    except Exception as e:
        status = "error"
        logger.error("Conflict scan failed (%s)", type(e).__name__)
        yield _event({"type": "error", "message": "The scan could not be completed. Try again."})
    finally:
        try:
            findings.sort(key=lambda f: (f["verdict"] != "conflict", -f.get("score", 0)))
            row.update(status=status, result=result)
            await run_in_threadpool(_insert, row)
            record("compare.scanned", "deny" if status == "scope_revoked" else "error" if status == "error" else "allow",
                   principal=principal, object_type="comparison", object_id=row["id"], matter_id=row["matter_id"], reason=status,
                   details={"documents": len(docs), "candidates": total,
                            "conflicts": sum(1 for f in findings if f["verdict"] == "conflict"), "rejected": len(rejected)})
        except Exception as e:
            logger.error("Conflict scan: could not record the outcome (%s)", type(e).__name__)


@router.post(_BASE + "/scan")
@audited("compare.scan", object_type="matter", object_param="matter_id")
async def scan_conflicts(matter_id: str, inp: ScanIn, request: Request):
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    try:
        scope = await run_in_threadpool(scopemod.authorize, p, matter_id)
    except scopemod.ScopeDenied:
        raise _not_found()
    row = {"id": str(uuid.uuid4()), "org_id": p["org_id"], "matter_id": matter_id, "owner_id": p["sub"], "kind": "scan",
           "status": "done", "versions": inp.versions, "document_ids": [], "version_ids": [], "result": {},
           "created_at": policy.now_iso()}
    note(object_id=row["id"], details={"versions": inp.versions})
    return StreamingResponse(_scan(scope, p, row, inp.versions), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ── Reading back ─────────────────────────────────────────────────────────────────
@router.get(_BASE)
def list_comparisons(matter_id: str, request: Request):
    """The caller's own comparisons and scans in this matter, newest first (titles and counts only)."""
    p = principal_of(request)
    require(p, "research.query", matter_id)
    with get_conn() as conn:
        rows = conn.execute("SELECT id, kind, status, versions, document_ids_json, result_json, created_at FROM legal_comparisons "
                            "WHERE matter_id=? AND owner_id=? ORDER BY created_at DESC LIMIT 50", (matter_id, p["sub"])).fetchall()
    out = []
    for cid, kind, status, versions, dids, rj, created in rows:
        dids, res = json.loads(dids), json.loads(rj)
        visible = _visible_documents(matter_id, dids, p["sub"])
        if kind == "versions":
            if visible != set(dids):
                continue
            out.append({"id": cid, "kind": kind, "status": status, "created_at": created, "left": res["left"], "right": res["right"],
                        "summary": res["summary"]})
        else:
            out.append({"id": cid, "kind": kind, "status": status, "created_at": created, "versions": versions,
                        "documents": len([d for d in res.get("documents", []) if d["document_id"] in visible]),
                        "conflicts": sum(1 for f in res.get("findings", []) if f["verdict"] == "conflict"
                                         and f["a"]["document_id"] in visible and f["b"]["document_id"] in visible)})
    return {"comparisons": out}


@router.get(_BASE + "/{comparison_id}")
@audited("compare.read", object_type="comparison", object_param="comparison_id")
def get_comparison(matter_id: str, comparison_id: str, request: Request):
    p = principal_of(request)
    require(p, "research.query", matter_id)
    row = _own(matter_id, comparison_id, p["sub"])
    if not row:
        raise _not_found()
    out = _versions_out(row, p["sub"], matter_id) if row["kind"] == "versions" else _scan_out(row, p["sub"], matter_id)
    if out is None:
        raise _not_found()
    note(details={"kind": row["kind"]})
    return JSONResponse({"comparison": out}, headers={"Cache-Control": "no-store"})
