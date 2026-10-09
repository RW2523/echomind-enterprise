"""/api/legal/matters/{id}/documents (S1.6, SEC-03).

Import stores the file encrypted and queues its indexing; the response says "queued" and the
library shows progress. Files are served only through the version route, never cached by the
browser (no-store) and always as a download. Search runs over the matter's own partition only,
through an AuthorizedScope. Every route asks the policy engine first and is audited.
"""
from __future__ import annotations

import hashlib
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from ..audit.record import audited, note
from ..jobs import worker
from ..matters.routes import principal_of, require, require_law_edition
from ..retrieval import scope as scopemod
from ..retrieval import search as searchmod
from . import blobs, repo, validate

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])

Kind = Literal["contract", "correspondence", "pleading", "evidence", "memo", "note", "other"]
DocStatus = Literal["draft", "final", "executed", "received"]
Mode = Literal["dense", "sparse", "keyword"]
_BASE = "/matters/{matter_id}/documents"


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Not Found")


class DocPatch(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)
    kind: Optional[Kind] = None
    status: Optional[DocStatus] = None
    version: int


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=20)
    modes: Optional[list[Mode]] = None


# ── Import ───────────────────────────────────────────────────────────────────────
@router.post(_BASE + "/import", status_code=202)
@audited("document.import", object_type="document")
def import_document(matter_id: str, request: Request, file: UploadFile = File(...), title: Optional[str] = Form(None),
                    kind: Kind = Form("other"), status: DocStatus = Form("draft"), version_of: Optional[str] = Form(None)):
    p = principal_of(request)
    require(p, "documents.import", matter_id)
    data = file.file.read(validate.max_bytes() + 1)
    if len(data) > validate.max_bytes():
        raise HTTPException(status_code=413, detail=f"Files can be at most {validate.max_bytes() // (1024 * 1024)} MB.")
    try:
        checked = validate.check(file.filename or "", file.content_type, data)
    except validate.Rejected as e:
        note(reason="rejected_file", details={"status": str(e.status)})
        raise HTTPException(status_code=e.status, detail=e.message)
    sha = hashlib.sha256(data).hexdigest()
    if version_of:
        target = repo.get_document(matter_id, version_of, p["sub"])
        if not target:
            raise _not_found()
        if target["source"]["kind"] == "connector":
            raise HTTPException(status_code=409, detail="This document is kept in the firm's document system; new versions come from there.")
        if repo.current_sha(matter_id, version_of) == sha:
            raise HTTPException(status_code=409, detail="This file is identical to the current version.")
        document_id = version_of
    else:
        document_id = repo.create_document(p["org_id"], matter_id, (title or checked.filename).strip()[:300], kind, status, p["sub"])
    version_id = repo.new_id()
    wrapped = blobs.store(matter_id, version_id, data)
    try:
        v = repo.add_version(version_id, document_id, matter_id, checked.filename, checked.mime, len(data), sha,
                             checked.pages, wrapped, p["sub"], status=status)
    except BaseException:
        blobs.remove(matter_id, version_id)
        raise
    worker.wake()
    note(object_id=document_id, details={"kind": kind, "mime": checked.mime, "size_bytes": len(data), "number": v["number"],
                                         "sha256": sha})
    duplicate = None if version_of else repo.same_content_in_matter(matter_id, sha, exclude_document=document_id)
    return {"document": repo.get_document(matter_id, document_id, p["sub"]), "same_content_as": duplicate}


# ── Reading ──────────────────────────────────────────────────────────────────────
@router.get(_BASE)
def list_documents(matter_id: str, request: Request):
    p = principal_of(request)
    require(p, "documents.read", matter_id)
    return {"documents": repo.list_documents(matter_id, p["sub"])}


@router.get(_BASE + "/{document_id}")
@audited("document.read", object_type="document", object_param="document_id")
def get_document(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "documents.read", matter_id)
    d = repo.get_document(matter_id, document_id, p["sub"])
    if not d:
        raise _not_found()
    return {"document": d}


@router.get(_BASE + "/{document_id}/versions")
def list_versions(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "documents.read", matter_id)
    if not repo.get_document(matter_id, document_id, p["sub"]):
        raise _not_found()
    return {"versions": repo.list_versions(matter_id, document_id, p["sub"])}


@router.get(_BASE + "/{document_id}/versions/{version_id}/file")
@audited("document.file", object_type="document_version", object_param="version_id")
def download(matter_id: str, document_id: str, version_id: str, request: Request):
    p = principal_of(request)
    require(p, "documents.read", matter_id)
    v = repo.version_row(matter_id, document_id, version_id, p["sub"])
    if not v:
        raise _not_found()
    try:
        data = blobs.load(matter_id, version_id, v["wrapped_key"])
    except blobs.IntegrityError:
        note(reason="integrity")
        raise HTTPException(status_code=500, detail="The stored file failed its integrity check. Tell your administrator.")
    if hashlib.sha256(data).hexdigest() != v["sha256"]:
        note(reason="integrity")
        raise HTTPException(status_code=500, detail="The stored file failed its integrity check. Tell your administrator.")
    note(details={"size_bytes": len(data)})
    name = v["filename"]
    ascii_name = name.encode("ascii", "replace").decode().replace('"', "'")
    return Response(content=data, media_type=v["mime"], headers={
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; sandbox",
        "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}",
    })


@router.get(_BASE + "/{document_id}/versions/{version_id}/reader")
@audited("document.read_text", object_type="document_version", object_param="version_id")
def read_version(matter_id: str, document_id: str, version_id: str, request: Request):
    """The version's text with its pages, lines, clauses, defined terms and marks (S2.1), under the
    same rules as downloading the file. Never cached; the audit record holds counts, not content."""
    p = principal_of(request)
    require(p, "documents.read", matter_id)
    v = repo.version_row(matter_id, document_id, version_id, p["sub"])
    if not v:
        raise _not_found()
    body = repo.reader(version_id)
    if body is None:
        raise HTTPException(status_code=409, detail="This version was read before the document reader existed. Read it again to open it here.")
    note(details={"chars": len(body["text"]), "marks": len(body["marks"]), "clauses": len(body["clauses"])})
    body["version"] = {"id": v["id"], "number": v["number"], "filename": v["filename"], "status": v["status"],
                       "index_status": v["index_status"]}
    return JSONResponse({"reader": body}, headers={"Cache-Control": "no-store"})


# ── Changing ─────────────────────────────────────────────────────────────────────
@router.patch(_BASE + "/{document_id}")
@audited("document.update", object_type="document", object_param="document_id")
def update_document(matter_id: str, document_id: str, inp: DocPatch, request: Request):
    p = principal_of(request)
    require(p, "documents.version", matter_id)
    if not repo.get_document(matter_id, document_id, p["sub"]):
        raise _not_found()
    changes = inp.model_dump(exclude={"version"}, exclude_none=True)
    note(details={"fields": sorted(changes), "kind": inp.kind, "status": inp.status})
    try:
        repo.update_document(matter_id, document_id, changes, inp.version)
    except repo.Conflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"document": repo.get_document(matter_id, document_id, p["sub"])}


@router.post(_BASE + "/{document_id}/reindex", status_code=202)
@audited("document.reindex", object_type="document", object_param="document_id")
def reindex(matter_id: str, document_id: str, request: Request):
    p = principal_of(request)
    require(p, "documents.version", matter_id)
    d = repo.get_document(matter_id, document_id, p["sub"])
    if not d or not d["current_version"]:
        raise _not_found()
    repo.requeue(matter_id, d["current_version"]["id"], p["sub"], p["org_id"])
    worker.wake()
    return {"document": repo.get_document(matter_id, document_id, p["sub"])}


# ── Search ───────────────────────────────────────────────────────────────────────
@router.post(_BASE + "/search")
@audited("documents.search", object_type="matter", object_param="matter_id")
async def search(matter_id: str, inp: SearchIn, request: Request):
    from starlette.concurrency import run_in_threadpool
    p = principal_of(request)
    await run_in_threadpool(require, p, "research.query", matter_id)
    q = inp.query.strip()
    note(details={"query": {"chars": len(q), "sha256": hashlib.sha256(q.encode()).hexdigest()},
                  "modes": inp.modes or list(searchmod.ALL_MODES), "k": inp.k})
    try:
        s = await run_in_threadpool(scopemod.authorize, p, matter_id)
        result = await searchmod.search(s, q, k=inp.k, modes=inp.modes or searchmod.ALL_MODES)
    except (scopemod.ScopeDenied, scopemod.ScopeRevoked):
        note(reason="scope_revoked")
        raise _not_found()
    note(details={"passages": len(result["passages"]), "abstained": result["abstained"]})
    return result
