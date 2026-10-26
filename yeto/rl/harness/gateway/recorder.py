"""The records proxy as a :class:`~yeto.rl.agentic.TokenRecorder` (S25 N10.5, N10.3).

verl's ``HarborAgentLoop`` asks a token recorder for one OpenAI base URL per
sample and, after the trial, for the sample's model calls.
:class:`ProxyTokenRecorder` is that recorder in front of a sampler engine: it
serves a :class:`~.proxy.RecordsProxy` over HTTP in the worker process (the
sandboxed agent calls ``<base>/t/<session>/v1``) and turns the proxy's records
into :class:`~yeto.rl.agentic.TokenTurn` values.

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

CompactionRL is refused here (S26 E9.5): a records backend built with a
``compaction`` config writes one training sample per compaction segment, but
a verl agent-loop sample is one token stream, so every compacted trajectory
would be ABORTED by the join.  The recorder fails at construction instead.

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
import importlib
import os
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from yeto.rl.agentic import HarnessConfigError, TokenAlignmentError, TokenTurn

from .proxy import RecordsProxy
from . import records_env
from .records import sampling_params

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


def turns_from_records(records: Mapping[str, Any]) -> list[TokenTurn]:
    """The proxy records of one trajectory as model calls in call order.

    Each recorded turn's prompt is the segment stream before its completion,
    so :func:`~yeto.rl.agentic.join_turns` sees what the engine saw.  Calls of
    later sessions or compaction segments follow the first ones; they do not
    extend its stream, so the join refuses them (one verl sample is one stream).
    """
    turns: list[TokenTurn] = []
    for session in records.get("sessions") or ():
        for sample in session["samples"]:
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
                                       policy_error=turn.get("parse_error")))
                cursor, pending_stop = end, len(turn.get("stop_added") or ())
            if cursor + pending_stop != len(ids):
                raise TokenAlignmentError(f"segment {sample['segment']} has {len(ids) - cursor - pending_stop} "
                                          "tokens no turn accounts for")
    return turns


class ProxyTokenRecorder:
    """:class:`~yeto.rl.agentic.TokenRecorder` over one in-process :class:`RecordsProxy` (module doc)."""

    def __init__(self, proxy: RecordsProxy, *, host: str = "127.0.0.1", port: int = 0,
                 public_host: str | None = None, public_base: str | None = None,
                 path_secret: str | None = None) -> None:
        if path_secret is not None and len(path_secret) < MIN_SECRET_CHARS:
            raise HarnessConfigError(f"the path secret must have at least {MIN_SECRET_CHARS} characters")
        if getattr(getattr(proxy, "backend", None), "compaction", None) is not None:
            raise HarnessConfigError(
                "the records proxy compacts contexts (CompactionRL segments), but a verl sample is one "
                "token stream; CompactionRL is not supported on the Harbor path")
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

                app = self.proxy.app()
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
            raise HarnessConfigError(f"recorder session {trajectory_id!r} is already open")
        base = await self._base()
        self._open.add(trajectory_id)
        return f"{base}/t/{trajectory_id}/v1"

    async def close(self, trajectory_id: str) -> Sequence[TokenTurn]:
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
        return turns_from_records(records)

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
