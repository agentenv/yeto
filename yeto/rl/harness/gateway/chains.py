"""Prefix hash chains and multi-chain bookkeeping.

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

Cost: the harness resends its whole history on every turn (pi keeps it even
when the records proxy compacts the context behind the gateway), so a request
is compared with each chain message by message.  A message that is the same
value as the stored one (same types, so ``1``, ``1.0`` and ``true`` differ, as
in their JSON) needs no hash; only the others (an echo with re-serialized
tool-call arguments or a dropped empty reasoning) are hashed, once per
request whatever the number of chains, against the stored per-message digest.
The outcome is the one of comparing the cumulative hashes.
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


def _json_number(text: str) -> int | float:
    value = float(text)
    return int(value) if value.is_integer() and abs(value) < 2 ** 53 else value


def _arguments_key(arguments: Any) -> Any:
    """Tool-call arguments as a value: a harness that parses the arguments it
    received and serializes them again (pi: ``JSON.stringify``) changes spacing,
    escapes and ``1.0`` -> ``1``, not the call."""
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments, parse_float=_json_number)
    except ValueError:
        return arguments


def _comparable(message: dict[str, Any]) -> dict[str, Any]:
    """The message as the chain compares it (module doc of :func:`canonical`)."""
    out = dict(message)
    if out.get("role") == "assistant" and not out.get("reasoning_content"):
        out.pop("reasoning_content", None)  # an empty reasoning is not echoed by every harness
    if out.get("tool_calls"):
        calls = []
        for call in out["tool_calls"]:
            call = dict(call)
            if isinstance(call.get("function"), dict):
                call["function"] = {**call["function"],
                                    "arguments": _arguments_key(call["function"].get("arguments"))}
            calls.append(call)
        out["tool_calls"] = calls
    return out


def canonical(message: dict[str, Any]) -> bytes:
    """Hash input of one message.  Two forms of the same message hash alike: an
    empty ``reasoning_content`` and none, and tool-call arguments that parse to
    the same JSON value.  The chain keeps the message as it was recorded (the
    generated form), so the next turn renders what the engine sampled."""
    return json.dumps(_comparable(message), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def message_hash(previous: str, message: dict[str, Any]) -> str:
    return hashlib.sha256(previous.encode() + b"\0" + canonical(message)).hexdigest()


def message_digest(message: dict[str, Any]) -> bytes:
    """Digest of one message's :func:`canonical` form (not chained)."""
    return hashlib.sha256(canonical(message)).digest()


def _same_value(a: Any, b: Any) -> bool:
    """``a`` and ``b`` have the same :func:`canonical` JSON without serializing
    them: equal values of the same types (``True == 1 == 1.0`` in Python, not in
    JSON); floats by their JSON spelling (``-0.0``, ``nan``).  False when unsure."""
    if type(a) is not type(b):
        return False
    if isinstance(a, str):
        return a == b
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_value(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return repr(a) == repr(b)
    return (a is None or isinstance(a, (int, bool))) and a == b


class _Request:
    """One request's messages with their digests computed on first use (shared by every chain)."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self._digests: dict[int, bytes] = {}

    def matches(self, index: int, stored: dict[str, Any], stored_digest: bytes) -> bool:
        message = self.messages[index]
        if message is stored or _same_value(message, stored):
            return True
        digest = self._digests.get(index)
        if digest is None:
            digest = self._digests[index] = message_digest(message)
        return digest == stored_digest


@dataclass
class Chain:
    chain_id: int
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)  # cumulative h_k
    generated: list[bool] = field(default_factory=list)
    digests: list[bytes] = field(default_factory=list)  # per message (module doc: cost)
    parent_chain: int | None = None
    fork_index: int | None = None
    break_reason: ChainBreakReason | None = None

    @property
    def head(self) -> str:
        return self.hashes[-1] if self.hashes else ""

    def append(self, message: dict[str, Any], *, generated: bool) -> None:
        body = canonical(message)
        self.hashes.append(hashlib.sha256(self.head.encode() + b"\0" + body).hexdigest())
        self.digests.append(hashlib.sha256(body).digest())
        self.messages.append(message)
        self.generated.append(generated)

    def common_prefix(self, request: "list[dict[str, Any]] | _Request") -> int:
        """Number of leading request messages equal to this chain's (as the
        cumulative hashes would find; module doc: cost)."""
        view = request if isinstance(request, _Request) else _Request(request)
        n = 0
        for stored, digest in zip(self.messages, self.digests):
            if n >= len(view.messages) or not view.matches(n, stored, digest):
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
    """Chains of one trajectory plus the chain counters."""

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
        view = _Request(request)
        n, _, best = max((c.common_prefix(view), c.chain_id, c) for c in self.chains)
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
