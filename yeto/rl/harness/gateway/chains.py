"""Prefix hash chains and multi-chain bookkeeping (design D5 / R-D5a, task 4.2).

One ``Chain`` = one session-server session = one training sample.  A request is
located against the chains of its trajectory by longest matching message-hash
prefix; the outcome is one of:

- CONTINUE: request == chain messages + appended messages whose roles are all in
  ``allowed_append_roles`` -> same chain/session (TITO reuses the prefix);
- FORK (``retry_fork``): request diverges at a message *we generated*; the
  harness retried/rolled back -> new chain, old chain untouched (still a valid
  sample, sibling of the new one);
- BREAK: request diverges at a non-generated message (``history_rewrite``), or
  only by dropped historical reasoning (``template_drops_reasoning``), or the
  harness declared a compaction window (``compaction_window``) -> new chain.

Patching a request to keep using the old chain is never done: a continuation
that appends a disallowed role raises ``SessionMismatch`` (counted).
"""
from __future__ import annotations

import enum
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


class ChainBreakReason(str, enum.Enum):
    retry_fork = "retry_fork"
    history_rewrite = "history_rewrite"
    template_drops_reasoning = "template_drops_reasoning"
    compaction_window = "compaction_window"


class LocateKind(str, enum.Enum):
    first = "first"
    continue_ = "continue"
    fork = "fork"
    break_ = "break"


class SessionMismatch(RuntimeError):
    pass


def canonical(message: dict[str, Any]) -> bytes:
    return json.dumps(message, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def message_hash(previous: str, message: dict[str, Any]) -> str:
    return hashlib.sha256(previous.encode() + b"\0" + canonical(message)).hexdigest()


@dataclass
class Chain:
    chain_id: int
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)  # cumulative h_k
    generated: list[bool] = field(default_factory=list)
    parent_chain: int | None = None
    fork_index: int | None = None
    break_reason: ChainBreakReason | None = None

    @property
    def head(self) -> str:
        return self.hashes[-1] if self.hashes else ""

    def append(self, message: dict[str, Any], *, generated: bool) -> None:
        self.hashes.append(message_hash(self.head, message))
        self.messages.append(message)
        self.generated.append(generated)

    def common_prefix(self, request: list[dict[str, Any]]) -> int:
        previous, n = "", 0
        for stored_hash, message in zip(self.hashes, request):
            previous = message_hash(previous, message)
            if previous != stored_hash:
                break
            n += 1
        return n


@dataclass
class Locate:
    kind: LocateKind
    chain: Chain | None  # chain to continue (CONTINUE) or the parent (FORK/BREAK); None for first
    prefix_len: int
    reason: ChainBreakReason | None = None


def _only_reasoning_dropped(stored: dict[str, Any], incoming: dict[str, Any]) -> bool:
    if stored.get("role") != "assistant" or "reasoning_content" not in stored:
        return False
    trimmed = {k: v for k, v in stored.items() if k != "reasoning_content"}
    return trimmed == incoming or {**incoming, "reasoning_content": stored["reasoning_content"]} == stored


class ChainRegistry:
    """Chains of one trajectory plus the D5 counters."""

    def __init__(self, *, allowed_append_roles: tuple[str, ...] = ("user", "tool"), keeps_history_reasoning: bool = True):
        self.allowed_append_roles = tuple(allowed_append_roles)
        self.keeps_history_reasoning = keeps_history_reasoning
        self.chains: list[Chain] = []
        self.chain_breaks: Counter[str] = Counter()
        self.session_mismatch = 0

    def locate(self, request: list[dict[str, Any]], *, compaction_window: bool = False) -> Locate:
        if not request:
            self.session_mismatch += 1
            raise SessionMismatch("empty message history")
        if not self.chains:
            return Locate(LocateKind.first, None, 0)
        best = max(self.chains, key=lambda c: (c.common_prefix(request), c.chain_id))
        n = best.common_prefix(request)
        if compaction_window:
            return Locate(LocateKind.break_, best, n, ChainBreakReason.compaction_window)
        if n == len(best.messages):
            appended = request[n:]
            bad = [m.get("role") for m in appended if m.get("role") not in self.allowed_append_roles]
            if bad:
                self.session_mismatch += 1
                raise SessionMismatch(f"appended roles {bad} not in allowed_append_roles {self.allowed_append_roles}")
            return Locate(LocateKind.continue_, best, n)
        # divergence at index n (request shorter, or differing message)
        if best.generated[n]:
            if n < len(request) and not self.keeps_history_reasoning and _only_reasoning_dropped(best.messages[n], request[n]):
                return Locate(LocateKind.break_, best, n, ChainBreakReason.template_drops_reasoning)
            return Locate(LocateKind.fork, best, n, ChainBreakReason.retry_fork)
        return Locate(LocateKind.break_, best, n, ChainBreakReason.history_rewrite)

    def open_chain(self, session_id: str, request: list[dict[str, Any]], located: Locate) -> Chain:
        """Create the chain for FIRST/FORK/BREAK; the request becomes its (mask-0) prompt."""
        chain = Chain(
            chain_id=len(self.chains), session_id=session_id,
            parent_chain=None if located.chain is None else located.chain.chain_id,
            fork_index=None if located.kind is LocateKind.first else located.prefix_len,
            break_reason=located.reason,
        )
        for message in request:
            chain.append(message, generated=False)
        if located.reason is not None:
            self.chain_breaks[located.reason.value] += 1
        self.chains.append(chain)
        return chain

    def extend(self, chain: Chain, request: list[dict[str, Any]], prefix_len: int) -> None:
        for message in request[prefix_len:]:
            chain.append(message, generated=False)

    def record_generation(self, chain: Chain, assistant_message: dict[str, Any]) -> None:
        chain.append(assistant_message, generated=True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "chains_total": len(self.chains),
            "tito_chain_breaks": dict(self.chain_breaks),
            "tito_session_mismatch": self.session_mismatch,
            "chains": [
                {"chain_id": c.chain_id, "session_id": c.session_id, "parent_chain": c.parent_chain,
                 "fork_index": c.fork_index, "break_reason": None if c.break_reason is None else c.break_reason.value,
                 "generated_messages": sum(c.generated), "messages": len(c.messages)}
                for c in self.chains
            ],
        }
