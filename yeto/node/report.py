"""The sampler's side of the node records (``nodes/<ticket>/...``, :mod:`.records`).

Off unless the sampler's environment names its ticket (``YETO_NODE_TICKET``, set
by the node runtime or a route that opts in): every other start writes nothing
new. When on, the sampler

* heartbeats from a daemon thread every ``YETO_NODE_HEARTBEAT_S`` seconds
  (default 30) from the engine start on: stage ``engine`` until it joins, then
  ``running`` with its rounds and a summary of its engine's ``/metrics``
  (generation tokens, running and queued requests, KV cache use);
* writes ``ready`` at its JOIN (the member record then names the ticket, the
  source and the profile summary: :meth:`SamplerNodeReport.member_fields`);
* writes a last heartbeat (stage ``left``, the leave reason) at its LEAVE, and
  the exit record at the process end unless the node runtime supervises it
  (``YETO_NODE_SUPERVISED``: the runtime knows the real exit code).

Every write is best effort: a store outage never stops a sampler.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from .launch import GENERATION_ENV, HEARTBEAT_ENV, PROFILE_ENV, SUPERVISED_ENV, TICKET_ENV
from .records import NodeReporter

DEFAULT_HEARTBEAT_S = 30.0
METRICS_TIMEOUT_S = 3.0

# engine /metrics line prefix -> summary key; counters and gauges summed over label sets
_METRICS = {
    "vllm:generation_tokens_total": "gen_tokens", "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "queued", "vllm:kv_cache_usage_perc": "kv_cache",
    "vllm:gpu_cache_usage_perc": "kv_cache",
    "sglang:generation_tokens_total": "gen_tokens", "sglang:num_running_reqs": "running",
    "sglang:num_queue_reqs": "queued", "sglang:token_usage": "kv_cache",
}


def parse_metrics(text: str) -> dict[str, float]:
    """The summary of an engine's Prometheus text (vLLM or SGLang names)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = _METRICS.get(name)
        if key is None:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        if key == "kv_cache":  # a fraction per engine: the fullest one
            out[key] = max(out.get(key, 0.0), value)
        else:
            out[key] = out.get(key, 0.0) + value
    return out


def engine_metrics(port: int | None, *, timeout: float = METRICS_TIMEOUT_S) -> dict[str, float]:
    if port is None:
        return {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/metrics", timeout=timeout) as response:  # noqa: S310
            return parse_metrics(response.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - an engine still starting (or none): no summary this time
        return {}


def metrics_port_of(spec: Mapping[str, Any]) -> int | None:
    port = (((spec.get("engine") or {}).get("server") or {}).get("port"))
    try:
        return None if port is None else int(port)
    except (TypeError, ValueError):
        return None


class SamplerNodeReport:
    def __init__(self, reporter: NodeReporter, *, profile: Mapping[str, Any] | None = None,
                 heartbeat_s: float = DEFAULT_HEARTBEAT_S, metrics_port: int | None = None,
                 supervised: bool = False, metrics: Callable[[int | None], Mapping[str, float]] = engine_metrics,
                 clock: Callable[[], float] = time.time) -> None:
        self.reporter, self.profile = reporter, dict(profile or {})
        self.heartbeat_s, self.metrics_port, self.supervised = float(heartbeat_s), metrics_port, bool(supervised)
        self.metrics, self.clock = metrics, clock
        self.stage = "engine"
        self.progress: Callable[[], Mapping[str, Any]] | None = None
        # when the sampler last made progress (joined, then each new round): the pool's
        # progress age, the same reading as an event tape that last grew then
        self.progress_at: float | None = None
        self._rounds: Any = None
        self.leave_reason: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env: Mapping[str, str], member_store: Any, spec: Mapping[str, Any]) -> "SamplerNodeReport | None":
        ticket = str(env.get(TICKET_ENV) or "").strip()
        if not ticket:
            return None
        from yeto.rl.engine.tape_upload import store_objects  # noqa: PLC0415

        try:
            profile = json.loads(env.get(PROFILE_ENV) or "{}")
        except ValueError:
            profile = {}
        reporter = NodeReporter(store_objects(member_store), ticket, int(env.get(GENERATION_ENV) or 1),
                                base={"sampler": spec.get("sampler_id")})
        return cls(reporter, profile=profile if isinstance(profile, dict) else {},
                   heartbeat_s=float(env.get(HEARTBEAT_ENV) or DEFAULT_HEARTBEAT_S), metrics_port=metrics_port_of(spec),
                   supervised=str(env.get(SUPERVISED_ENV) or "") == "1")

    @property
    def ticket(self) -> str:
        return self.reporter.ticket

    def member_fields(self) -> dict[str, Any]:
        """What the member record carries about the machine (``Member.node``)."""
        out = {"ticket": self.ticket, "generation": self.reporter.generation}
        if self.profile.get("source"):
            out["source"] = self.profile["source"]
        if self.profile:
            out["profile"] = dict(self.profile)
        return out

    def _beat(self, **extra: Any) -> None:
        fields: dict[str, Any] = {}
        if self.progress is not None:
            try:
                fields.update(dict(self.progress()))
            except Exception:  # noqa: BLE001 - observation only
                pass
        if "rounds" in fields and fields["rounds"] != self._rounds:
            self._rounds, self.progress_at = fields["rounds"], float(self.clock())
        if self.progress_at is not None:
            fields["progress_at"] = self.progress_at
        engine = dict(self.metrics(self.metrics_port)) if self.stage in ("engine", "running") else {}
        if engine:
            fields["engine"] = engine
        if self.leave_reason is not None:
            fields["leave_reason"] = self.leave_reason
        with self._lock:
            self.reporter.heartbeat(self.stage, **{**fields, **extra})

    def _loop(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            self._beat()

    def start(self) -> "SamplerNodeReport":
        self._beat()
        self._thread = threading.Thread(target=self._loop, name="yeto-node-heartbeat", daemon=True)
        self._thread.start()
        return self

    def ready(self, **fields: Any) -> None:
        self.reporter.ready(profile=self.profile, **fields)
        self.stage, self.progress_at = "running", float(self.clock())
        self._beat()

    def left(self, reason: str, **fields: Any) -> None:
        self.leave_reason, self.stage = str(reason), "left"
        self._beat(**{k: v for k, v in fields.items() if isinstance(v, (str, int, float, bool, type(None)))})

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def exit(self, code: int | None, reason: str | None = None) -> None:
        """The process end of an unsupervised sampler (the runtime writes it otherwise)."""
        if self.supervised:
            return
        self.reporter.exit(code, reason or self.leave_reason or ("finished" if code == 0 else "error"))


def from_env(env: Mapping[str, str], member_store: Any, spec: Mapping[str, Any]) -> SamplerNodeReport | None:
    try:
        return SamplerNodeReport.from_env(env, member_store, spec)
    except Exception as error:  # noqa: BLE001 - observation only: a sampler without node records still runs
        print(f"[yeto-node] node records off: {type(error).__name__}: {error}", flush=True)
        return None
