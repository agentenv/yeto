"""The records proxy as a :class:`~yeto.rl.agentic.TokenRecorder`.

verl's ``HarborAgentLoop`` asks a token recorder for one OpenAI base URL per
sample and, after the trial, for the sample's model calls.
:class:`ProxyTokenRecorder` is that recorder in front of a sampler engine: it
serves a :class:`~.proxy.RecordsProxy` over HTTP in the worker process (the
sandboxed agent calls ``<base>/t/<session>/<token>/v1``, the token random per
session) and turns the proxy's records into :class:`~yeto.rl.agentic.TokenTurn`
values.  Only the model routes are served: records and close are in-process
calls, so no sandbox can read, close or write another session (proxy doc).

One recorder per process, not per sample: verl builds a new agent loop for
every sample, so ``HarborAgentLoop`` keeps the recorder per event loop
(:class:`~yeto.rl.agentic.loop_local.LoopLocal`) and calls the
``YETO_AGENTIC_RECORDER`` callable once per loop.  :func:`recorder_from_env`
is that callable for this recorder; it builds the proxy with the deployment's
own ``module:callable`` (the engine and the chat template are deployment
facts, not code here).

Sampling: the proxy signs its sampling settings once (gateway config).  A
session whose trainer sampling parameters differ from them is refused
(:class:`~yeto.rl.agentic.HarnessConfigError`) instead of sampling with other
settings than the trainer assumes.

CompactionRL: a records backend built with a ``compaction`` config writes one
training sample per compaction segment.  A verl agent-loop sample is one token
stream, so every compacted trajectory would be ABORTED by the join: the
recorder refuses such a proxy at construction (the default, and what
:func:`recorder_from_env`, the verl and Miles factory, builds).  The engine
sampler's agentic round writes one buffer sample per segment: it builds the
recorder with ``allow_compaction=True`` and reads the calls per segment
(:meth:`ProxyTokenRecorder.close_segments`).

Miles does not use this recorder: its Harbor path records through Miles'
own session server (GOAL §9).

Environment of :func:`recorder_from_env`:
  YETO_RECORDS_PROXY              ``module:callable`` returning a RecordsProxy (required)
  YETO_RECORDS_PROXY_HOST         bind address (default 127.0.0.1)
  YETO_RECORDS_PROXY_PORT         bind port (default 0: a free port)
  YETO_RECORDS_PROXY_PUBLIC_HOST  host the sandbox dials (default: the bind address)
  YETO_RECORDS_PROXY_PUBLIC_BASE  full URL the sandbox dials instead (a tunnel, e.g.
                                  ``https://<id>.modal.host``); overrides PUBLIC_HOST and the port
  YETO_RECORDS_PROXY_SECRET_FILE  file holding a path secret: the proxy is served only under
                                  ``/k/<secret>/`` (a public tunnel never exposes the
                                  unauthenticated root); the secret is never logged
"""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import os
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from yeto.rl.agentic import HarnessConfigError, SessionAlreadyOpen, TokenAlignmentError, TokenTurn
from yeto.rl.agentic.finish import CompactionSegment

from .proxy import RecordsProxy
from . import records_env
from .records import KIND_SUMMARY, sampling_params

# The names themselves live in ``records_env`` (dependency-free, so the launch
# layer can read them); this module keeps its own spellings as aliases.
PROXY_ENV = records_env.PROXY_ENV
HOST_ENV = records_env.PROXY_HOST_ENV
PORT_ENV = records_env.PROXY_PORT_ENV
PUBLIC_HOST_ENV = records_env.PROXY_PUBLIC_HOST_ENV
PUBLIC_BASE_ENV = records_env.PROXY_PUBLIC_BASE_ENV
SECRET_FILE_ENV = records_env.PROXY_SECRET_FILE_ENV
MIN_SECRET_CHARS = 24

# verl sampling keys -> SamplingParams fields; other keys (``logprobs``) do not change sampling.
CHECKED_SAMPLING = ("temperature", "top_p", "top_k", "repetition_penalty")


def _sample_turns(sample: Mapping[str, Any]) -> list[TokenTurn]:
    """The model calls of one records sample (one segment of one session) in call order.

    Each recorded turn's prompt is the segment stream before its completion,
    so :func:`~yeto.rl.agentic.join_turns` sees what the engine saw."""
    turns: list[TokenTurn] = []
    ids, mask = list(sample["token_ids"]), list(sample["loss_mask"])
    cursor, pending_stop = 0, 0
    for turn in sample["turns"]:
        cursor += pending_stop + len(turn["prompt_added"])
        completion = tuple(int(t) for t in turn["completion"])
        end = cursor + len(completion)
        if tuple(ids[cursor:end]) != completion or any(m != 1 for m in mask[cursor:end]):
            raise TokenAlignmentError(f"records turn {turn['turn']} does not sit in its segment stream")
        turns.append(TokenTurn(prompt_ids=tuple(ids[:cursor]), completion_ids=completion,
                               completion_logprobs=tuple(float(x) for x in turn["logprobs"]),
                               policy_version=turn.get("policy_version"),
                               provenance=turn.get("provenance"),
                               finish_reason=turn.get("finish_reason"),
                               policy_error=turn.get("parse_error"),
                               completion_top_ids=turn.get("top_ids"),
                               completion_top_logprobs=turn.get("top_logprobs"),
                               routed_experts=turn.get("routed_experts"),
                               routed_experts_cut=bool(turn.get("routed_experts_cut", False))))
        cursor, pending_stop = end, len(turn.get("stop_added") or ())
    if cursor + pending_stop != len(ids):
        raise TokenAlignmentError(f"segment {sample['segment']} has {len(ids) - cursor - pending_stop} "
                                  "tokens no turn accounts for")
    return turns


def segments_from_records(records: Mapping[str, Any], *,
                          context_budget: int | None = None) -> list[CompactionSegment]:
    """The proxy records of one trajectory as model calls per compaction segment.

    A compaction segment (records ``segment`` > 0) starts a new token stream
    (``system + [task] + u_resume(S) + last k steps``), so its calls form a group of
    their own, joined into a sample of their own.  A later session's first
    segment is no compaction: its calls follow the current group, as in
    :func:`turns_from_records` (the join rebases or refuses them).  Without
    compaction there is one group, the calls of :func:`turns_from_records`.

    ``kept_steps``: whole steps the segment kept from the one before (None on
    the first); ``summary_ok``: the summary the segment ended with had a closed
    ``<summary>`` tag (None: it ended without one); ``summary_tokens``: the
    tokens its summary turn generated (0 without one); ``retry_replays``: the
    gateway's count for the whole trajectory (None: replay is off), on every
    segment; ``task_truncated`` / ``last_step_truncated``: the segment's rebuilt
    context cut the task message / the kept last step's tool output (None: not
    rebuilt, or the option is off)."""
    groups: list[CompactionSegment] = []
    replays = (records.get("gateway") or {}).get("retry_replays")
    replays = None if replays is None else int(replays)
    for session in records.get("sessions") or ():
        for sample in session["samples"]:
            turns = _sample_turns(sample)
            if not turns:  # a segment opened but never sampled (its prompt did not fit): no sample
                continue
            summary = sum(len(t["completion"]) for t in sample["turns"] if t.get("kind") == KIND_SUMMARY)
            if groups and int(sample["segment"]) == 0:
                last = groups[-1]
                groups[-1] = dataclasses.replace(last, turns=(*last.turns, *turns),
                                                 summary_ok=sample.get("summary_ok"),
                                                 summary_tokens=last.summary_tokens + summary)
                continue
            groups.append(CompactionSegment(turns=tuple(turns), kept_steps=sample.get("kept_steps"),
                                            summary_ok=sample.get("summary_ok"),
                                            task_truncated=sample.get("task_truncated"),
                                            last_step_truncated=sample.get("last_step_truncated"),
                                            context_budget=context_budget, summary_tokens=summary,
                                            retry_replays=replays))
    return groups


def turns_from_records(records: Mapping[str, Any]) -> list[TokenTurn]:
    """The proxy records of one trajectory as model calls in call order.

    Calls of later sessions or compaction segments follow the first ones; they
    do not extend its stream, so the join refuses them (one verl sample is one
    stream).  :func:`segments_from_records` keeps the segments apart."""
    turns: list[TokenTurn] = []
    for session in records.get("sessions") or ():
        for sample in session["samples"]:
            turns.extend(_sample_turns(sample))
    return turns


class ProxyTokenRecorder:
    """:class:`~yeto.rl.agentic.TokenRecorder` over one in-process :class:`RecordsProxy` (module doc)."""

    def __init__(self, proxy: RecordsProxy, *, host: str = "127.0.0.1", port: int = 0,
                 public_host: str | None = None, public_base: str | None = None,
                 path_secret: str | None = None, allow_compaction: bool = False) -> None:
        """``allow_compaction``: accept a compacting proxy; only a caller that
        reads :meth:`close_segments` and writes one sample per segment may set it
        (the engine sampler's agentic round; module doc)."""
        if path_secret is not None and len(path_secret) < MIN_SECRET_CHARS:
            raise HarnessConfigError(f"the path secret must have at least {MIN_SECRET_CHARS} characters")
        compaction = getattr(getattr(proxy, "backend", None), "compaction", None)
        if compaction is not None and not allow_compaction:
            raise HarnessConfigError(
                "the records proxy compacts contexts (CompactionRL segments), but a verl sample is one "
                "token stream; CompactionRL is not supported on the Harbor path")
        self.compaction = compaction
        self.proxy = proxy
        self.host = host
        self.port = int(port)
        self.public_host = public_host or host
        self.public_base = public_base.rstrip("/") if public_base else None
        self._prefix = f"/k/{path_secret}" if path_secret else ""
        self._signed = sampling_params(proxy.gateway.config.signed_sampling)
        self._runner: Any = None
        self._base_url: str | None = None
        self._start_lock = asyncio.Lock()
        self._open: set[str] = set()

    async def _base(self) -> str:
        async with self._start_lock:
            if self._base_url is None:
                from aiohttp import web

                app = self.proxy.app()  # model routes only: records/close stay in process
                if self._prefix:
                    outer = web.Application()
                    outer.add_subapp(f"{self._prefix}/", app)
                    app = outer
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, self.host, self.port)
                await site.start()
                port = runner.addresses[0][1]
                self._runner = runner
                base = self.public_base or f"http://{self.public_host}:{port}"
                self._base_url = f"{base}{self._prefix}"
            return self._base_url

    def check_sampling(self, sampling: Mapping[str, Any]) -> None:
        """Refuse trainer sampling settings the proxy does not sample with."""
        differ = {key: (sampling[key], getattr(self._signed, key)) for key in CHECKED_SAMPLING
                  if key in sampling and sampling[key] is not None
                  and float(sampling[key]) != float(getattr(self._signed, key))}
        if differ:
            raise HarnessConfigError(f"trainer sampling differs from the records proxy's signed sampling "
                                     f"(key: (trainer, proxy)): {differ}")

    async def open(self, trajectory_id: str, sampling_params: Mapping[str, Any]) -> str:
        self.check_sampling(sampling_params)
        if trajectory_id in self._open:
            raise SessionAlreadyOpen(f"recorder session {trajectory_id!r} is already open")
        base = await self._base()
        self.proxy.open(trajectory_id)  # issues the session's own token
        self._open.add(trajectory_id)
        return f"{base}{self.proxy.path(trajectory_id)}/v1"

    def _close_records(self, trajectory_id: str) -> Mapping[str, Any]:
        if trajectory_id not in self._open:
            raise HarnessConfigError(f"recorder session {trajectory_id!r} is not open")
        self._open.discard(trajectory_id)
        try:
            records = self.proxy.records(trajectory_id)
            self.proxy.close(trajectory_id)
        finally:
            self.proxy.forget(trajectory_id)
        invalid = (records.get("gateway") or {}).get("invalidated")
        if invalid:
            raise TokenAlignmentError(f"records proxy invalidated the trajectory: {invalid}")
        return records

    async def close(self, trajectory_id: str) -> Sequence[TokenTurn]:
        return turns_from_records(self._close_records(trajectory_id))

    async def close_segments(self, trajectory_id: str) -> Sequence[CompactionSegment]:
        """:meth:`close`, the calls grouped per compaction segment (:func:`segments_from_records`)."""
        budget = None if self.compaction is None else int(self.compaction.context_budget)
        return segments_from_records(self._close_records(trajectory_id), context_budget=budget)

    def decode(self, ids: Sequence[int]) -> str:
        """Text of ``ids`` under the proxy's chat template (loss reports)."""
        return self.proxy.backend.template.decode(list(ids))

    async def aclose(self) -> None:
        """Stop the HTTP server (tests; a worker process keeps it for its life)."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner, self._base_url = None, None


def recorder_from_env() -> ProxyTokenRecorder:
    """``YETO_AGENTIC_RECORDER=yeto.rl.harness.gateway.recorder:recorder_from_env`` (module doc)."""
    spec = os.environ.get(PROXY_ENV, "")
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise HarnessConfigError(f"set {PROXY_ENV}=module:callable returning a RecordsProxy")
    proxy = getattr(importlib.import_module(module), attr)()
    if not isinstance(proxy, RecordsProxy):
        raise HarnessConfigError(f"{spec} returned {type(proxy).__name__}, not a RecordsProxy")
    host = os.environ.get(HOST_ENV, "127.0.0.1")
    secret_file = os.environ.get(SECRET_FILE_ENV, "").strip()
    secret = Path(secret_file).read_text().strip() if secret_file else None
    return ProxyTokenRecorder(proxy, host=host, port=int(os.environ.get(PORT_ENV, "0")),
                              public_host=os.environ.get(PUBLIC_HOST_ENV) or host,
                              public_base=os.environ.get(PUBLIC_BASE_ENV) or None, path_secret=secret)
