"""Gateway core: entry -> canonical chat -> chain location -> session backend -> entry response (tasks 4.1/4.2).

``SessionBackend`` is the only I/O seam (fake in CPU tests; the Session Server
chat route in production).  Sampling parameters are *signed* gateway config:
a request may repeat them verbatim but never override them.  ``max_chains=1``
is the first-batch Codex assertion (R-D5a): any fork/break invalidates the
trajectory instead of training a second chain.

IR-2/IR-4 wiring (``GatewayConfig.harness_board``, a ``tool_wait.HarnessBoard``
or its actor handle): ``allow_new_session(member)`` is asked before every
``create_session`` (drain closes admission -> ``AdmissionClosed``, the caller
re-targets or waits for undrain; the trajectory is *not* invalidated),
``enter_session``/``exit_session`` bracket each backend session, and the D5
counters are mirrored with ``record_session_mismatch`` /
``record_chain_break(reason)`` so the 1.7 load sample sees them.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any, Protocol

from yeto.rl.engine.tool_wait import _call as _board_call
from yeto.rl.engine.tool_wait import _resolve as _board_resolve

from .chains import ChainRegistry, LocateKind, SessionMismatch
from .context import ContextProvider, IdentityContextProvider
from .translate import (
    UnsupportedShape,
    chat_to_messages,
    completion_to_chat,
    completion_to_messages,
    completion_to_responses,
    messages_to_messages,
    responses_to_messages,
)

ENTRIES = ("responses", "chat", "messages")
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "max_completion_tokens",
                   "logprobs", "top_logprobs", "n", "seed", "frequency_penalty", "presence_penalty", "stop")


class GatewayError(RuntimeError):
    pass


class SamplingOverrideError(GatewayError):
    pass


class TrajectoryInvalid(GatewayError):
    """Fail closed: the trajectory must be aborted (never a 0 reward)."""


class AdmissionClosed(GatewayError):
    """IR-2: the member is draining; no new session may open (re-target / wait)."""


class SessionBackend(Protocol):
    async def create_session(self, trajectory_id: str) -> str: ...
    async def chat(self, session_id: str, messages: list[dict[str, Any]], sampling: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class GatewayConfig:
    model: str
    signed_sampling: dict[str, Any]
    allowed_append_roles: tuple[str, ...] = ("user", "tool")
    keeps_history_reasoning: bool = True
    max_chains: int | None = None  # 1 for the first Codex batch
    context_provider: ContextProvider = field(default_factory=IdentityContextProvider)
    harness_board: Any = None  # tool_wait.HarnessBoard (local or actor handle), IR-2/IR-4
    member: str | None = None  # admission key of the rollout member this gateway targets


class Gateway:
    def __init__(self, config: GatewayConfig, backend: SessionBackend) -> None:
        self.config = config
        self.backend = backend
        self.registries: dict[str, ChainRegistry] = {}
        self.invalidated: dict[str, str] = {}

    def _board(self, method: str, *args: Any) -> Any:
        board = self.config.harness_board
        return None if board is None else _board_resolve(_board_call(board, method, *args))

    def registry(self, trajectory_id: str) -> ChainRegistry:
        return self.registries.setdefault(
            trajectory_id,
            ChainRegistry(allowed_append_roles=self.config.allowed_append_roles, keeps_history_reasoning=self.config.keeps_history_reasoning),
        )

    def _check_sampling(self, body: dict[str, Any]) -> dict[str, Any]:
        for name in SAMPLING_FIELDS:
            if name in body and name in self.config.signed_sampling and body[name] != self.config.signed_sampling[name]:
                raise SamplingOverrideError(f"{name} is a signed sampling field")
            if name in body and name not in self.config.signed_sampling and body[name] not in (None, False):
                raise SamplingOverrideError(f"{name} may not be set by the harness")
        if body.get("model") not in (None, self.config.model):
            raise SamplingOverrideError("model is fixed by the gateway")
        return dict(self.config.signed_sampling)

    async def handle(self, entry: str, body: dict[str, Any], *, trajectory_id: str, compaction_window: bool = False) -> dict[str, Any]:
        if entry not in ENTRIES:
            raise UnsupportedShape(f"unknown entry {entry!r}")
        if trajectory_id in self.invalidated:
            raise TrajectoryInvalid(self.invalidated[trajectory_id])
        sampling = self._check_sampling(body)
        messages = {"responses": responses_to_messages, "chat": chat_to_messages, "messages": messages_to_messages}[entry](body)
        registry = self.registry(trajectory_id)
        try:
            located = registry.locate(messages, compaction_window=compaction_window)
        except SessionMismatch as exc:
            self.invalidated[trajectory_id] = f"tito_session_mismatch: {exc}"
            self._board("record_session_mismatch")
            raise TrajectoryInvalid(self.invalidated[trajectory_id]) from exc
        if located.kind is LocateKind.continue_:
            chain = located.chain
            assert chain is not None
            registry.extend(chain, messages, located.prefix_len)
        else:
            if located.kind is not LocateKind.first and self.config.max_chains is not None and len(registry.chains) >= self.config.max_chains:
                reason = located.reason.value if located.reason else located.kind.value
                self.invalidated[trajectory_id] = f"chain limit {self.config.max_chains} exceeded: {reason}"
                registry.chain_breaks[reason] += 1
                self._board("record_chain_break", reason)
                raise TrajectoryInvalid(self.invalidated[trajectory_id])
            if self.config.harness_board is not None and not self._board("allow_new_session", self.config.member):
                raise AdmissionClosed(f"member {self.config.member!r} is draining; session for {trajectory_id!r} refused")
            session_id = await self.backend.create_session(trajectory_id)
            self._board("enter_session", session_id, self.config.member)
            chain = registry.open_chain(session_id, messages, located)
            if located.reason is not None:
                self._board("record_chain_break", located.reason.value)
        provided = self.config.context_provider.provide(trajectory_id, chain.chain_id, list(chain.messages))
        completion = await self.backend.chat(chain.session_id, provided, sampling)
        registry.record_generation(chain, dict(completion["choices"][0]["message"]))
        response_id = secrets.token_hex(8)
        if entry == "responses":
            return completion_to_responses(completion, response_id=response_id)
        if entry == "messages":
            return completion_to_messages(completion, response_id=response_id, model=self.config.model)
        return completion_to_chat(completion)

    def close(self, trajectory_id: str) -> None:
        """End of trajectory: every backend session of its chains leaves ``harness_in_flight``."""
        registry = self.registries.get(trajectory_id)
        if registry is None:
            return
        for chain in registry.chains:
            self._board("exit_session", chain.session_id)

    def snapshot(self, trajectory_id: str) -> dict[str, Any]:
        snap = self.registry(trajectory_id).snapshot()
        snap["invalidated"] = self.invalidated.get(trajectory_id)
        return snap
