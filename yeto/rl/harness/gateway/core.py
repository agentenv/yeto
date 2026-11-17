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
``enter_session``/``exit_session`` bracket each backend session, and the session
counters are mirrored with ``record_session_mismatch`` /
``record_chain_break(reason)`` so the load sample sees them.

Retry replay (``GatewayConfig.replay_retries``, off by default; the records
proxy turns it on with context compaction): a harness that lost a reply (an
HTTP timeout, a dropped connection) sends the same request again.  Without
replay that request diverges at the generated message it never received, a
``retry_fork``: a new session that starts from the whole history, which after a
compaction no longer fits the engine's context.  With replay, a request equal to
the history a chain's newest generation answered gets that generation's reply
again (``retry_replays``), without an engine call; one that arrives while the
generation is still running waits for it.  Any other divergence is located as
before.

Fork after compaction (``GatewayConfig.session_compacted``, set by the records
proxy with compaction): any other fork or break from a chain whose backend
session has compacted would open a session from message 0 (past the engine's
window, and a stream the segment join refuses), so it fails closed instead:
the trajectory is invalid (``compaction_fork_lost: <reason>``, counted).
"""
from __future__ import annotations

import asyncio
import copy
import secrets
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from yeto.rl.engine.tool_wait import _call as _board_call
from yeto.rl.engine.tool_wait import _resolve as _board_resolve

from .chains import Chain, ChainRegistry, Locate, LocateKind, SessionMismatch
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
    replay_retries: bool = False  # a resent request gets the reply it lost (module doc)
    session_compacted: Callable[[str], bool] | None = None  # backend session id -> has compacted (module doc)

    @classmethod
    def from_codex_profile(cls, profile_name: str, signed_sampling: dict[str, Any], **overrides: Any) -> "GatewayConfig":
        """Build a config whose ``model`` / ``allowed_append_roles`` / ``keeps_history_reasoning``
        come from the signed codex backend profile (rl-fn-codex-rollout).

        ``keeps_history_reasoning`` cannot be overridden: it is the profile's
        declaration about its fixed TITO template, not a runtime knob.
        """
        if "keeps_history_reasoning" in overrides:
            raise ValueError("keeps_history_reasoning is declared by the codex profile")
        from yeto.rl.codex_backend import stock_codex_backend_profile, stock_codex_keeps_history_reasoning, stock_codex_tito_model

        profile = stock_codex_backend_profile(profile_name)
        overrides.setdefault("model", stock_codex_tito_model(profile_name))
        overrides.setdefault("allowed_append_roles", tuple(profile["tito_allowed_append_roles"]))
        return cls(signed_sampling=signed_sampling, keeps_history_reasoning=stock_codex_keeps_history_reasoning(profile_name), **overrides)


class Gateway:
    def __init__(self, config: GatewayConfig, backend: SessionBackend) -> None:
        self.config = config
        self.backend = backend
        self.registries: dict[str, ChainRegistry] = {}
        self.invalidated: dict[str, str] = {}
        # replay_retries: per (trajectory, chain) the newest generation, as
        # (messages it answered, future of its completion); counted replays per trajectory
        self._generations: dict[tuple[str, int], tuple[int, asyncio.Future[dict[str, Any]]]] = {}
        self.replays: Counter[str] = Counter()
        self.forks_lost: Counter[str] = Counter()

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
        while True:
            try:
                located = registry.locate(messages, compaction_window=compaction_window)
            except SessionMismatch as exc:
                self.invalidated[trajectory_id] = f"tito_session_mismatch: {exc}"
                self._board("record_session_mismatch")
                raise TrajectoryInvalid(self.invalidated[trajectory_id]) from exc
            generation = self._replayable(trajectory_id, messages, located)
            if generation is None:
                break
            try:
                completion = await asyncio.shield(generation)
            except Exception:  # noqa: BLE001 - the first request failed and answered its own error: locate again
                continue
            if trajectory_id in self.invalidated:
                raise TrajectoryInvalid(self.invalidated[trajectory_id])
            if self.registries.get(trajectory_id) is not registry:  # forgotten while the first request ran
                raise TrajectoryInvalid(f"trajectory {trajectory_id!r} was forgotten while a request was in flight")
            self.replays[trajectory_id] += 1
            return self._reply(entry, copy.deepcopy(completion))
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
            compacted = self.config.session_compacted
            if located.chain is not None and compacted is not None and compacted(located.chain.session_id):
                reason = located.reason.value if located.reason else located.kind.value
                self.invalidated[trajectory_id] = (f"compaction_fork_lost: {reason} at message {located.prefix_len} "
                                                   f"of chain {located.chain.chain_id}, whose session has compacted")
                registry.chain_breaks[reason] += 1
                self.forks_lost[trajectory_id] += 1
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
        if not self.config.replay_retries:
            completion = await self.backend.chat(chain.session_id, provided, sampling)
            registry.record_generation(chain, dict(completion["choices"][0]["message"]))
            return self._reply(entry, completion)
        key = (trajectory_id, chain.chain_id)
        generation: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        generation.add_done_callback(lambda f: f.cancelled() or f.exception())  # retrieved, never logged
        self._generations[key] = (len(chain.messages), generation)
        try:
            completion = await self.backend.chat(chain.session_id, provided, sampling)
            registry.record_generation(chain, dict(completion["choices"][0]["message"]))
        except BaseException as exc:
            if self._generations.get(key, (0, None))[1] is generation:
                del self._generations[key]
            generation.set_exception(exc if isinstance(exc, Exception) else GatewayError("generation cancelled"))
            raise
        generation.set_result(copy.deepcopy(completion))
        return self._reply(entry, completion)

    def _replayable(self, trajectory_id: str, messages: list[dict[str, Any]],
                    located: Locate) -> "asyncio.Future[dict[str, Any]] | None":
        """The newest generation of the located chain when ``messages`` is the
        history it answered (module doc: retry replay), else None."""
        chain: Chain | None = located.chain
        if not self.config.replay_retries or chain is None:
            return None
        entry = self._generations.get((trajectory_id, chain.chain_id))
        if entry is None or entry[0] != len(messages) or located.prefix_len != len(messages):
            return None
        answered, generation = entry
        if generation.done():  # recorded (a failed one is dropped): the request stops before the generated message
            done = located.kind is LocateKind.fork and len(chain.messages) == answered + 1 and chain.generated[-1]
            return generation if done else None
        # running: the chain already holds the request (appended by the first one)
        return generation if located.kind is LocateKind.continue_ and len(chain.messages) == answered else None

    def _reply(self, entry: str, completion: dict[str, Any]) -> dict[str, Any]:
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

    def forget(self, trajectory_id: str) -> None:
        """Drop the trajectory's chain state (after :meth:`close`; a long-lived gateway would grow)."""
        self.registries.pop(trajectory_id, None)
        self.invalidated.pop(trajectory_id, None)
        self.replays.pop(trajectory_id, None)
        self.forks_lost.pop(trajectory_id, None)
        for key in [k for k in self._generations if k[0] == trajectory_id]:
            del self._generations[key]

    def snapshot(self, trajectory_id: str) -> dict[str, Any]:
        snap = self.registry(trajectory_id).snapshot()
        snap["invalidated"] = self.invalidated.get(trajectory_id)
        if self.config.replay_retries:
            snap["retry_replays"] = self.replays[trajectory_id]
        if self.config.session_compacted is not None:
            snap["compaction_fork_lost"] = self.forks_lost[trajectory_id]
        return snap
