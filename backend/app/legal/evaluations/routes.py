"""Evaluation runs for the firm's audit readers (S2.8, D-044): what was measured, on which code, models
and data. Summaries only; the items stay in the report files."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Request

from ..matters.routes import principal_of, require, require_law_edition
from . import listing

router = APIRouter(prefix="/legal", tags=["legal"], dependencies=[Depends(require_law_edition)])


@router.get("/evaluations")
def evaluation_runs(request: Request, suite: Optional[str] = None):
    require(principal_of(request), "audit.read.org")
    return {"runs": listing(suite)}
