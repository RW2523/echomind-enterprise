"""The search capability (decision D-005).

`AuthorizedScope` names exactly one matter partition and the person it was granted to. Only
`authorize()` can create one, and only after the policy engine allowed the action; anything else
that tries raises TypeError. A scope is not a standing permission: search re-checks it with the
live policy before and after it runs, so a revocation, a new wall or an ended break-glass takes
effect on the very next use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import policy
from .registry import partition_for

_MINT = object()


class ScopeDenied(Exception):
    def __init__(self, decision: policy.Decision):
        super().__init__(decision.reason)
        self.decision = decision


class ScopeRevoked(Exception):
    """The scope's holder may no longer perform its action on its matter."""


@dataclass(frozen=True)
class AuthorizedScope:
    principal_id: str
    org_id: str
    matter_id: str
    partition: str
    role: str
    break_glass: bool
    action: str = "research.query"
    _mint: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._mint is not _MINT:
            raise TypeError("An AuthorizedScope can only be created by scope.authorize().")


def authorize(principal: dict, matter_id: str, action: str = "research.query") -> AuthorizedScope:
    d = policy.authorize(principal, action, matter_id)
    if not d.allowed:
        raise ScopeDenied(d)
    return AuthorizedScope(principal["sub"], principal["org_id"], matter_id, partition_for(matter_id), d.role,
                           d.role == policy.BREAK_GLASS_ROLE, action, _MINT)


def recheck(scope: AuthorizedScope) -> None:
    if not isinstance(scope, AuthorizedScope) or scope._mint is not _MINT:
        raise TypeError("search needs an AuthorizedScope")
    d = policy.authorize({"sub": scope.principal_id, "org_id": scope.org_id}, scope.action, scope.matter_id)
    if not d.allowed:
        raise ScopeRevoked()
