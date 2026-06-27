"""Secure export gateway route (#5): offline risk-evaluation + redaction before content leaves the box."""
from fastapi import APIRouter
from pydantic import BaseModel

from ...core.export_gateway import evaluate_export

router = APIRouter(prefix="/export", tags=["export"])


class EvaluateIn(BaseModel):
    text: str


@router.post("/evaluate")
def evaluate(inp: EvaluateIn):
    """Scan text for sensitive data (PII/secrets) and return findings + a redacted copy."""
    return evaluate_export(inp.text or "")
