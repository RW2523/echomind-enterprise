"""/api/legal/matters/{id}/exports (S1.10, SEC-06; P-006 safe defaults).

Every route asks the policy engine first and is audited without content: the purpose, the plan's
strings and the file never enter the audit trail — only counts, the format and the outcome.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .. import policy
from ..audit.record import audited, note
from ..matters import repo as matters_repo
from ..matters.routes import principal_of, require, require_law_edition
from . import service

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])
_BASE = "/matters/{matter_id}/exports"
_MIME = {"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


class ExportIn(BaseModel):
    object_type: Literal["document", "meeting_transcript"]
    object_id: str = Field(min_length=1, max_length=64)
    format: Literal["pdf", "docx"]
    purpose: str = Field(min_length=1, max_length=1000)


class Toggle(BaseModel):
    id: str = Field(max_length=32)
    redact: bool


class PlanIn(BaseModel):
    items: list[Toggle] = Field(default_factory=list, max_length=2000)
    add: list[str] = Field(default_factory=list, max_length=100)


class ApproveIn(BaseModel):
    acknowledge_privilege: bool = False


class NoteIn(BaseModel):
    note: Optional[str] = Field(default=None, max_length=1000)


def _refused(e: service.Refused) -> HTTPException:
    return HTTPException(status_code=e.status, detail=e.message)


def _may_see(p: dict, matter_id: str) -> None:
    """Exports are visible to those who may ask for or approve them (lead, attorney, paralegal)."""
    if policy.authorize(p, "export.approve", matter_id).allowed:
        return
    d = policy.authorize(p, "export.request", matter_id)
    if not d.allowed and d.status == 403:
        member = policy.authorize(p, "matter.read", matter_id)
        if member.allowed and member.role == "paralegal":
            return                                    # paralegals see exports even without the export right
    require(p, "export.request", matter_id)


@router.post(_BASE)
@audited("export.request", object_type="export")
def create_export(matter_id: str, inp: ExportIn, request: Request):
    p = principal_of(request)
    require(p, "export.request", matter_id)
    try:
        e = service.create(p, matter_id, inp.object_type, inp.object_id, inp.format, inp.purpose.strip())
    except service.Refused as err:
        raise _refused(err)
    note(object_id=e["id"], details={"format": inp.format, "object_type": inp.object_type,
                                     "suggested": sum(1 for i in e["plan"]["items"] if i["redact"])})
    return {"export": e}


@router.get(_BASE)
def list_exports(matter_id: str, request: Request):
    _may_see(principal_of(request), matter_id)
    return {"exports": service.listing(matter_id)}


@router.get(_BASE + "/{export_id}")
def get_export(matter_id: str, export_id: str, request: Request):
    _may_see(principal_of(request), matter_id)
    e = service.get(matter_id, export_id)
    if not e:
        raise HTTPException(status_code=404, detail="Not Found")
    return {"export": e}


@router.put(_BASE + "/{export_id}/plan")
@audited("export.plan", object_type="export", object_param="export_id")
def update_plan(matter_id: str, export_id: str, inp: PlanIn, request: Request):
    p = principal_of(request)
    require(p, "export.request", matter_id)
    note(details={"toggled": len(inp.items), "added": len(inp.add)})
    try:
        return {"export": service.update_plan(p, matter_id, export_id, [t.model_dump() for t in inp.items], inp.add)}
    except service.Refused as err:
        raise _refused(err)


@router.post(_BASE + "/{export_id}/submit")
@audited("export.submit", object_type="export", object_param="export_id")
def submit_export(matter_id: str, export_id: str, request: Request):
    p = principal_of(request)
    require(p, "export.request", matter_id)
    try:
        e = service.submit(p, matter_id, export_id)
    except service.Refused as err:
        raise _refused(err)
    note(details={"redactions": sum(1 for i in e["plan"]["items"] if i["redact"])})
    return {"export": e}


@router.post(_BASE + "/{export_id}/approve")
@audited("export.approve", object_type="export", object_param="export_id")
def approve_export(matter_id: str, export_id: str, request: Request, inp: Optional[ApproveIn] = None):
    p = principal_of(request)
    require(p, "export.approve", matter_id)
    try:
        e = service.approve(p, matter_id, matters_repo.get_matter(matter_id)["number"], export_id, (inp or ApproveIn()).acknowledge_privilege)
    except service.Refused as err:
        raise _refused(err)
    note(details={"status": e["status"], "verified": bool(e["verification"] and e["verification"]["ok"])})
    return {"export": e}


@router.post(_BASE + "/{export_id}/reject")
@audited("export.reject", object_type="export", object_param="export_id")
def reject_export(matter_id: str, export_id: str, request: Request, inp: Optional[NoteIn] = None):
    p = principal_of(request)
    require(p, "export.approve", matter_id)
    try:
        return {"export": service.reject(p, matter_id, export_id, (inp or NoteIn()).note)}
    except service.Refused as err:
        raise _refused(err)


@router.get(_BASE + "/{export_id}/file")
@audited("export.download", object_type="export", object_param="export_id")
def download_export(matter_id: str, export_id: str, request: Request):
    p = principal_of(request)
    require(p, "matter.read", matter_id)
    try:
        e, data = service.file(p, matter_id, export_id)
    except service.Refused as err:
        raise _refused(err)
    note(details={"format": e["format"], "size_bytes": len(data)})
    name = f"{e['title'][:120]}.{e['format']}"
    ascii_name = name.encode("ascii", "replace").decode().replace('"', "'")
    return Response(content=data, media_type=_MIME[e["format"]], headers={
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Security-Policy": "default-src 'none'; sandbox",
        "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}",
    })


# Preview renderings, kept briefly in memory per (export, plan) so paging through is quick.
_previews: "OrderedDict[tuple[str, str], bytes]" = OrderedDict()


def _preview_pdf(p: dict, matter_id: str, export_id: str) -> bytes:
    import hashlib
    import json as _json
    from . import render, source
    _may_see(p, matter_id)
    e = service.get(matter_id, export_id)
    if not e:
        raise HTTPException(status_code=404, detail="Not Found")
    strings = [i["text"] for i in e["plan"]["items"] if i["redact"]]
    key = (export_id, hashlib.sha256(_json.dumps(sorted(strings)).encode()).hexdigest())
    if key not in _previews:
        try:
            src = source.load(matter_id, e["object_type"], e["object_id"])
        except source.NotFound:
            raise HTTPException(status_code=404, detail="Not Found")
        _previews[key] = render.render({"format": "pdf", "source": src, "items": strings, "watermark": "PREVIEW — not for release"})
        while len(_previews) > 16:
            _previews.popitem(last=False)
    return _previews[key]


@router.get(_BASE + "/{export_id}/preview")
def preview_info(matter_id: str, export_id: str, request: Request):
    import fitz
    data = _preview_pdf(principal_of(request), matter_id, export_id)
    return {"pages": len(fitz.open(stream=data, filetype="pdf"))}


@router.get(_BASE + "/{export_id}/preview/{page}")
def preview_page(matter_id: str, export_id: str, page: int, request: Request):
    """A low-resolution picture of one page with the planned redactions applied (never stored).
    Word exports are previewed through their PDF rendering."""
    import fitz
    doc = fitz.open(stream=_preview_pdf(principal_of(request), matter_id, export_id), filetype="pdf")
    if not 1 <= page <= len(doc):
        raise HTTPException(status_code=404, detail="Not Found")
    png = doc[page - 1].get_pixmap(dpi=60).tobytes("png")
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})
