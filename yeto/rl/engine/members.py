"""Rollout member ids (decoupling 3.5, audit A6).

INFRA keys rollout members as ``engine:<cell_id>``.  The rule is backend
independent, so it lives in the core; the Miles adapter (``rollout.py``) and
the codex preflight both use these objects.
"""

from __future__ import annotations

from typing import Any

MEMBER_PREFIX = "engine:"


def member_id(cell_id: Any) -> str:
    return f"{MEMBER_PREFIX}{cell_id}"


def cell_of(member: str) -> str:
    if not isinstance(member, str) or not member.startswith(MEMBER_PREFIX):
        raise ValueError(f"{member!r} is not a rollout member id")
    return member[len(MEMBER_PREFIX):]
