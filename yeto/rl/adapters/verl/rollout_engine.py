"""verl's own rollout servers behind the records proxy (S25 N10.9).

In a verl training run the policy is served by verl's rollout replicas (vLLM
or SGLang, started and updated by verl).  ``HarborAgentLoop`` needs a token
recorder in front of that policy: the sandboxed agent talks OpenAI to the
records proxy, the proxy renders the chat template and asks an engine for
tokens.  :class:`VerlRolloutEngine` is that engine.  It owns no server: it
forwards token-id prompts to verl's ``LLMServerClient`` (the agent loop's
``server_manager``) on the agent-loop worker's event loop, so verl keeps
routing, sticky sessions and weight updates.

Wiring (all in the agent-loop worker process):

* ``YETO_AGENTIC_RECORDER=yeto.rl.harness.gateway.recorder:recorder_from_env``;
* ``YETO_RECORDS_PROXY=yeto.rl.adapters.verl.rollout_engine:records_proxy_from_env``
  (this module) builds the proxy over a :class:`VerlRolloutEngine`;
* ``HarborAgentLoop.run`` binds the engine to its ``server_manager`` and the
  running event loop before the first call (:func:`bind_rollout_client`).

Environment of :func:`records_proxy_from_env`:
  YETO_RECORDS_TEMPLATE        chat template name (``chat_templates.TEMPLATES``)
  YETO_RECORDS_TOKENIZER       tokenizer directory (the checkpoint)
  YETO_RECORDS_MODEL           model name the agent sends (``AgentTask.model_name``)
  YETO_RECORDS_SAMPLING        JSON signed sampling (temperature, top_p, top_k, max_tokens)
  YETO_RECORDS_MAX_MODEL_LEN   context length of verl's rollout (``rollout.max_model_len``)
  YETO_RECORDS_CARD            GPU name (provenance; default ``unknown``)

Policy version: verl's synchronous trainer updates the rollout weights only
between steps, never while a batch is generated, so a turn cannot straddle a
reload.  Each rollout reply carries the weights' ``global_steps``; the engine
keeps them per trajectory (the records proxy's ``CURRENT_TRAJECTORY``) and
``HarborAgentLoop`` reports their min / max as the sample's
``min_global_steps`` / ``max_global_steps`` (verl's V1 trainer requires both).
:meth:`VerlRolloutEngine.served_version` returns None: the version is known
only after a reply, not before it.

Aborted calls (S26 E9.5): verl's fully-async rollouter aborts the requests in
flight when it pauses for a weight update.  An aborted model call recorded
no tokens and the agent has not seen it, so the engine sends it again after
a back-off, sampled by the updated policy; the Harbor trial (sandbox, agent,
recorded tokens) keeps running meanwhile.  This is the verl form of the
suspension between rounds (:mod:`yeto.rl.agentic.suspend`): only for a
trajectory ``HarborAgentLoop`` tracks (:meth:`VerlRolloutEngine.track`), at
most ``YETO_VERL_ABORT_RETRY_S`` seconds (default 1800) per call; the waits
count as suspended time, not as the policy's.  An untracked call, or one
past the limit, fails as before (the trajectory is ABORTED).

Finish reason: verl reports ``completed`` for both an end token and the token
cap.  A reply that ends without a stop token and fills ``max_tokens`` is
reported as ``length``; everything else as ``stop``.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import json
import os
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from yeto.rl.agentic.suspend import SuspendLedger
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    UNKNOWN,
    Completion,
    EngineIdentity,
    LoadReport,
    SamplerEngineError,
    SamplingParams,
)

from yeto.rl.harness.gateway.records_env import (
    CARD_ENV,
    MAX_MODEL_LEN_ENV,
    MODEL_ENV,
    SAMPLING_ENV,
    TEMPLATE_ENV,
    TOKENIZER_ENV,
)

CALL_TIMEOUT_S = 1800.0
logger = logging.getLogger(__name__)
ABORT_RETRY_ENV = "YETO_VERL_ABORT_RETRY_S"
DEFAULT_ABORT_RETRY_S = 1800.0
ABORT_BACKOFF_S = (0.5, 1.0, 2.0, 5.0, 10.0)


class VerlRolloutEngine:
    """``TokenSamplerEngine`` over verl's ``LLMServerClient`` (module doc)."""

    def __init__(self, *, max_model_len: int, identity: EngineIdentity) -> None:
        self._max_model_len = int(max_model_len)
        self._identity = identity
        self._client: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._steps: dict[str, list[int]] = {}
        self._ledgers: dict[str, SuspendLedger] = {}
        self._aborts = itertools.count(1)
        self.last_global_steps: int | None = None
        self.abort_retry_s = float(os.environ.get(ABORT_RETRY_ENV, "") or DEFAULT_ABORT_RETRY_S)
        self.sleep = asyncio.sleep

    # ------------------------------------------------------------ binding
    def bind(self, client: Any, loop: asyncio.AbstractEventLoop) -> None:
        """Use ``client`` (verl ``server_manager``) on ``loop``; the same pair for the worker's life."""
        if self._client is not None and (self._client is not client or self._loop is not loop):
            raise SamplerEngineError("VerlRolloutEngine is already bound to another rollout client or loop")
        self._client, self._loop = client, loop

    @property
    def bound(self) -> bool:
        return self._client is not None

    def pop_steps(self, trajectory: str) -> list[int]:
        """The ``global_steps`` of every reply sampled for ``trajectory`` (and forget them)."""
        return self._steps.pop(trajectory, [])

    def track(self, trajectory: str, ledger: SuspendLedger) -> None:
        """Retry ``trajectory``'s aborted calls and count the waits on ``ledger`` (module doc)."""
        self._ledgers[trajectory] = ledger

    def untrack(self, trajectory: str) -> None:
        self._ledgers.pop(trajectory, None)

    # ------------------------------------------------------------ engine port
    @property
    def identity(self) -> EngineIdentity:
        return self._identity

    @property
    def max_model_len(self) -> int:
        return self._max_model_len

    def served_version(self) -> int | None:
        return None

    def generate_tokens(self, prompts: Sequence[Sequence[int]], sampling: SamplingParams, *,
                        stop_token_ids: Sequence[int] = ()) -> list[list[Completion]]:
        """Called from a worker thread (the records backend uses ``asyncio.to_thread``)."""
        if self._client is None or self._loop is None:
            raise SamplerEngineError("VerlRolloutEngine is not bound to verl's rollout client")
        from yeto.rl.harness.gateway.proxy import CURRENT_TRAJECTORY

        trajectory = CURRENT_TRAJECTORY.get()  # copied into this thread by asyncio.to_thread
        stops = {int(t) for t in stop_token_ids}
        out: list[list[Completion]] = []
        for prompt in prompts:
            group = []
            for _ in range(int(sampling.n)):
                future = asyncio.run_coroutine_threadsafe(
                    self._generate(list(prompt), sampling, sorted(stops), trajectory), self._loop)
                group.append(future.result(timeout=CALL_TIMEOUT_S))
            out.append(group)
        return out

    async def _generate(self, prompt: list[int], sampling: SamplingParams, stops: list[int],
                        trajectory: str | None = None) -> Completion:
        params: dict[str, Any] = {
            "temperature": float(sampling.temperature), "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k), "repetition_penalty": float(sampling.repetition_penalty),
            "max_tokens": int(sampling.max_tokens), "logprobs": True,
        }
        if stops:
            params["stop_token_ids"] = stops
        output = await self._generate_retrying(prompt, params, trajectory)
        steps = (getattr(output, "extra_fields", None) or {}).get("global_steps")
        if steps is not None:
            self.last_global_steps = int(steps)
            if trajectory is not None:
                self._steps.setdefault(trajectory, []).append(int(steps))
        tokens = tuple(int(t) for t in output.token_ids)
        logprobs = output.log_probs
        if logprobs is None or len(logprobs) != len(tokens):
            raise SamplerEngineError(f"verl rollout returned {0 if logprobs is None else len(logprobs)} "
                                     f"log-probs for {len(tokens)} tokens")
        return Completion(token_ids=tokens, logprobs=tuple(float(x) for x in logprobs), text="",
                          finish_reason=finish_reason(tokens, stops, int(sampling.max_tokens)),
                          prompt_token_ids=tuple(prompt))

    async def _generate_retrying(self, prompt: list[int], params: dict[str, Any], trajectory: str | None) -> Any:
        ledger = self._ledgers.get(trajectory) if trajectory is not None else None
        started = time.monotonic()
        for attempt in itertools.count():
            output = await self._client.generate(request_id=uuid.uuid4().hex, prompt_ids=prompt,
                                                 sampling_params=params)
            if getattr(output, "stop_reason", None) != "aborted":
                return output
            if ledger is None or time.monotonic() - started >= self.abort_retry_s:
                raise SamplerEngineError("verl rollout aborted the request")
            if attempt == 0:
                ledger.mark(next(self._aborts))
                logger.warning("yeto verl: model call of trajectory %s aborted by the rollout; sending it again",
                               trajectory)
            ledger.retried_turns += 1
            delay = ABORT_BACKOFF_S[min(attempt, len(ABORT_BACKOFF_S) - 1)]
            ledger.waiting += 1
            try:
                await self.sleep(delay)
            finally:
                ledger.waiting -= 1
                ledger.suspended_seconds += delay
        raise AssertionError("unreachable")

    # verl starts, updates and stops its own servers: the proxy never calls these.
    def start(self, release_dir: Path | None, version: int) -> LoadReport:
        raise SamplerEngineError("verl owns its rollout servers")

    def load(self, release_dir: Path, version: int) -> LoadReport:
        raise SamplerEngineError("verl owns its rollout servers")

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        raise SamplerEngineError("text prompts are not used; the records proxy sends token ids")

    def score(self, prompt: str, continuation: str) -> list[float]:
        raise SamplerEngineError("scoring is not used on the verl rollout")

    def loaded_expert_matrices(self) -> int | None:
        return None

    def weight_gib_per_card(self) -> float | None:
        return None

    def stop(self) -> None:
        return None


def finish_reason(tokens: Sequence[int], stops: set[int] | Sequence[int], max_tokens: int) -> str:
    """``length`` when the reply filled ``max_tokens`` without a stop token (module doc)."""
    if tokens and int(tokens[-1]) in set(stops):
        return FINISH_STOP
    return FINISH_LENGTH if len(tokens) >= max_tokens else FINISH_STOP


def _engine_of(recorder: Any) -> Any:
    engine = getattr(getattr(getattr(recorder, "proxy", None), "backend", None), "engine", None)
    return engine if isinstance(engine, VerlRolloutEngine) else None


def weight_steps(recorder: Any, trajectory: str) -> tuple[int, int] | None:
    """``(min, max)`` verl ``global_steps`` of the trajectory's replies; None without a verl engine.

    A trajectory with no reply (aborted before its first call) gets the
    latest step the engine saw: in the synchronous trainer that is the batch's.
    Before the engine saw any reply it gets ``(0, 0)``: verl's V1 trainer
    needs an integer, and such a row is an ABORTED placeholder (no loss tokens).
    """
    engine = _engine_of(recorder)
    if engine is None:
        return None
    steps = engine.pop_steps(trajectory) or ([engine.last_global_steps] if engine.last_global_steps is not None else [])
    return (min(steps), max(steps)) if steps else (0, 0)


def track_aborts(recorder: Any, trajectory: str, ledger: SuspendLedger) -> bool:
    """Retry ``trajectory``'s aborted calls when the recorder runs over verl's rollout (module doc)."""
    engine = _engine_of(recorder)
    if engine is None:
        return False
    engine.track(trajectory, ledger)
    return True


def untrack_aborts(recorder: Any, trajectory: str) -> None:
    engine = _engine_of(recorder)
    if engine is not None:
        engine.untrack(trajectory)


def bind_rollout_client(recorder: Any, client: Any) -> None:
    """Bind the recorder's :class:`VerlRolloutEngine` (if it has one) to verl's client on this loop."""
    proxy = getattr(recorder, "proxy", None)
    engine = getattr(getattr(proxy, "backend", None), "engine", None)
    if isinstance(engine, VerlRolloutEngine):
        engine.bind(client, asyncio.get_running_loop())


def records_proxy_from_env() -> Any:
    """``YETO_RECORDS_PROXY`` callable: a RecordsProxy over a :class:`VerlRolloutEngine` (module doc)."""
    from yeto.rl.agentic import HarnessConfigError
    from yeto.rl.harness.chat_templates import ChatTemplate
    from yeto.rl.harness.gateway.proxy import RecordsProxy

    try:
        template_name = os.environ[TEMPLATE_ENV]
        tokenizer = os.environ[TOKENIZER_ENV]
        model = os.environ[MODEL_ENV]
        signed = json.loads(os.environ[SAMPLING_ENV])
        max_model_len = int(os.environ[MAX_MODEL_LEN_ENV])
    except (KeyError, ValueError) as exc:
        raise HarnessConfigError(f"records proxy over verl rollout: bad or missing setting {exc}") from exc
    identity = EngineIdentity(engine="verl-rollout", engine_version=_verl_version(), device_family="cuda",
                              card_model=os.environ.get(CARD_ENV, UNKNOWN), precision="bf16",
                              quant_scheme="bf16", topology=UNKNOWN)
    engine = VerlRolloutEngine(max_model_len=max_model_len, identity=identity)
    template = ChatTemplate.load(template_name, tokenizer_path=tokenizer)
    return RecordsProxy(engine, template, model=model, signed_sampling=signed,
                        sampler_id="verl-rollout-records")


def _verl_version() -> str:
    try:
        import verl

        return str(getattr(verl, "__version__", UNKNOWN))
    except ImportError:
        return UNKNOWN
