"""REST API: local Rules Library (rule sets, rules, session activation)."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ...rules_library import rules_store
from ...schemas.rules_library import (
    RuleCreate,
    RuleListOut,
    RuleOut,
    RuleSetCreate,
    RuleSetListOut,
    RuleSetOut,
    RuleSetUpdate,
    RuleUpdate,
    SessionActivationRowOut,
    SessionActivationsOut,
    SessionRuleActivationIn,
)

router = APIRouter(prefix="/rules-library", tags=["rules-library"])


def _rs_out(d: dict) -> RuleSetOut:
    return RuleSetOut(
        id=d["id"],
        name=d["name"],
        description=d.get("description") or "",
        version=d.get("version") or "1.0.0",
        priority=int(d.get("priority") or 0),
        is_active_default=bool(d.get("is_active_default")),
        source_policy_text=d.get("source_policy_text"),
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


def _rule_out(d: dict) -> RuleOut:
    return RuleOut(
        id=d["id"],
        rule_set_id=d["rule_set_id"],
        title=d["title"],
        text=d["text"],
        severity=d.get("severity") or "medium",
        category=d.get("category") or "general",
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


@router.post("/rule-sets", response_model=RuleSetOut)
def create_rule_set(body: RuleSetCreate) -> RuleSetOut:
    row = rules_store.create_rule_set(
        body.name,
        body.description,
        body.version,
        body.priority,
        body.is_active_default,
        body.source_policy_text,
    )
    return _rs_out(row)


@router.get("/rule-sets", response_model=RuleSetListOut)
def list_rule_sets() -> RuleSetListOut:
    rows = rules_store.list_rule_sets()
    return RuleSetListOut(rule_sets=[_rs_out(r) for r in rows])


@router.patch("/rule-sets/{rule_set_id}", response_model=RuleSetOut)
def update_rule_set(rule_set_id: str, body: RuleSetUpdate) -> RuleSetOut:
    row = rules_store.update_rule_set(
        rule_set_id,
        name=body.name,
        description=body.description,
        version=body.version,
        priority=body.priority,
        is_active_default=body.is_active_default,
        source_policy_text=body.source_policy_text,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Rule set not found")
    return _rs_out(row)


@router.delete("/rule-sets/{rule_set_id}")
def delete_rule_set(rule_set_id: str) -> dict:
    ok = rules_store.delete_rule_set(rule_set_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Rule set not found")
    return {"ok": True, "deleted": rule_set_id}


@router.post("/rule-sets/upload", response_model=RuleSetOut)
async def upload_rule_policy(
    file: UploadFile = File(...),
    name: str | None = Form(None),
    description: str | None = Form(""),
    version: str | None = Form("1.0.0"),
    priority: int = Form(0),
) -> RuleSetOut:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    base_name = (name or "").strip() or (file.filename or "Imported policy").rsplit(".", 1)[0]
    row = rules_store.create_rule_set(
        base_name,
        description or f"Uploaded from {file.filename or 'file'}",
        version or "1.0.0",
        priority,
        False,
        text[:500_000],
    )
    return _rs_out(row)


@router.post("/rule-sets/{rule_set_id}/rules", response_model=RuleOut)
def create_rule(rule_set_id: str, body: RuleCreate) -> RuleOut:
    row = rules_store.create_rule(
        rule_set_id,
        body.title,
        body.text,
        body.severity,
        body.category,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Rule set not found")
    return _rule_out(row)


@router.get("/rule-sets/{rule_set_id}/rules", response_model=RuleListOut)
def list_rules(rule_set_id: str) -> RuleListOut:
    if not rules_store.get_rule_set(rule_set_id):
        raise HTTPException(status_code=404, detail="Rule set not found")
    rows = rules_store.list_rules(rule_set_id)
    return RuleListOut(rules=[_rule_out(r) for r in rows])


@router.patch("/rules/{rule_id}", response_model=RuleOut)
def update_rule(rule_id: str, body: RuleUpdate) -> RuleOut:
    row = rules_store.update_rule(
        rule_id,
        title=body.title,
        text=body.text,
        severity=body.severity,
        category=body.category,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Rule not found")
    return _rule_out(row)


@router.delete("/rules/{rule_id}")
def delete_rule(rule_id: str) -> dict:
    ok = rules_store.delete_rule(rule_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"ok": True, "deleted": rule_id}


@router.post("/sessions/{session_id}/activations", response_model=SessionActivationsOut)
def set_session_activation(session_id: str, body: SessionRuleActivationIn) -> SessionActivationsOut:
    try:
        raw = rules_store.set_session_rule_activation(
            session_id,
            body.rule_set_id,
            enabled=body.enabled,
            priority_override=body.priority_override,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown rule set") from None
    return _activations_out(raw)


@router.get("/sessions/{session_id}/activations", response_model=SessionActivationsOut)
def get_session_activations(session_id: str) -> SessionActivationsOut:
    raw = rules_store.list_session_activations(session_id)
    return _activations_out(raw)


def _activations_out(raw: dict) -> SessionActivationsOut:
    acts: List[SessionActivationRowOut] = []
    for a in raw.get("activations") or []:
        acts.append(
            SessionActivationRowOut(
                id=a["id"],
                session_id=a["session_id"],
                rule_set_id=a["rule_set_id"],
                enabled=a["enabled"],
                priority_override=a.get("priority_override"),
                created_at=a["created_at"],
                updated_at=a["updated_at"],
                rule_set_name=a["rule_set_name"],
                rule_set_version=a["rule_set_version"],
                rule_set_priority=int(a.get("rule_set_priority") or 0),
            )
        )
    return SessionActivationsOut(session_id=raw["session_id"], activations=acts)
