"""``ContextProvider`` (design D8, task 7.2): the hook CompactionRL will use to swap context between segments.

The default is the identity mapping; the gateway sends exactly the messages the
harness supplied, so TITO prefix reuse is unaffected.
"""
from __future__ import annotations

from typing import Any, Protocol


class ContextProvider(Protocol):
    def provide(self, trajectory_id: str, chain_id: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class IdentityContextProvider:
    def provide(self, trajectory_id: str, chain_id: int, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages
