"""API models for local Rules Library."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RuleSetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    description: str = Field(default="", max_length=8000)
    version: str = Field(default="1.0.0", max_length=64)
    priority: int = Field(default=0, ge=-1000, le=1000)
    is_active_default: bool = False
    source_policy_text: Optional[str] = Field(default=None, max_length=500_000)


class RuleSetUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=500)
    description: Optional[str] = Field(default=None, max_length=8000)
    version: Optional[str] = Field(default=None, max_length=64)
    priority: Optional[int] = Field(default=None, ge=-1000, le=1000)
    is_active_default: Optional[bool] = None
    source_policy_text: Optional[str] = Field(default=None, max_length=500_000)


class RuleSetOut(BaseModel):
    id: str
    name: str
    description: str
    version: str
    priority: int
    is_active_default: bool
    source_policy_text: Optional[str] = None
    created_at: str
    updated_at: str


class RuleCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    text: str = Field(..., min_length=1, max_length=50_000)
    severity: str = Field(default="medium", max_length=64)
    category: str = Field(default="general", max_length=128)


class RuleUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    text: Optional[str] = Field(default=None, min_length=1, max_length=50_000)
    severity: Optional[str] = Field(default=None, max_length=64)
    category: Optional[str] = Field(default=None, max_length=128)


class RuleOut(BaseModel):
    id: str
    rule_set_id: str
    title: str
    text: str
    severity: str
    category: str
    created_at: str
    updated_at: str


class SessionRuleActivationIn(BaseModel):
    rule_set_id: str = Field(..., min_length=4, max_length=128)
    enabled: bool = True
    priority_override: Optional[int] = Field(default=None, ge=-1000, le=1000)


class SessionActivationRowOut(BaseModel):
    id: str
    session_id: str
    rule_set_id: str
    enabled: bool
    priority_override: Optional[int] = None
    created_at: str
    updated_at: str
    rule_set_name: str
    rule_set_version: str
    rule_set_priority: int


class SessionActivationsOut(BaseModel):
    session_id: str
    activations: List[SessionActivationRowOut]


class RuleSetListOut(BaseModel):
    rule_sets: List[RuleSetOut]


class RuleListOut(BaseModel):
    rules: List[RuleOut]
