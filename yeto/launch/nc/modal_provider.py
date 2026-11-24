"""Modal rollout provider: sampler islands as Modal GPU function calls, under the
same pool policy as National Compute (scale, eviction -> fence -> refill,
claim release, budget, breaker, events).

A Modal island has no node to order and log into: the "lease" is the function
call itself. So the provider splits like this:

* :class:`ModalProvider` (the :class:`~yeto.launch.pool.provider.IslandProvider` contract):
  ``order(slot)`` reserves a ticket ``<prefix>-<run>-<n>`` in a local record and
  calls nothing (nothing bills before the call is spawned); ``status``: a ticket
  without a call reads ``active`` (ready to start), a call that runs reads
  ``active``, a call that ended reads ended (``released`` when this provider
  cancelled it, ``lost`` when it failed or its sampler exited non-zero: fence,
  release its claims, refill); a call whose sampler exited 0 (the run finished)
  reads ``active`` and the probe's exited sampler ends it as finished, with no
  refill; a failed status call raises (a control-plane blip). ``release``
  cancels the call (``terminate_containers=True``) and confirms it no longer
  runs; only tickets of this run (name prefix *and* the record) are touched.
* :class:`ModalPoolOps` (the pool's ``PoolOps``, replacing the SSH node route):
  ``start`` renders the member's sampler spec exactly as the SSH route does
  (:func:`~.cli.render_spec`, ``start_on_latest`` on) and spawns one call of the
  rollout function with it; ``probe`` maps the call state to a probe (ended: an
  exited sampler, the pool restarts it in place with a new incarnation). A running
  call is *not* yet joined: the call reports ready itself. Each spawn carries a
  ``ready_key`` (one per call); the function body writes that object in the run
  store once the sampler's tape has ``rl_sampler_join`` (the engine loaded the
  model and claimed its lease), and the probe reads joined from it. A call that
  never reports ready is a start failure by the pool's ``start_timeout_s``;
  ``fence`` / ``fence_incarnation`` go to the run store like the
  SSH route; ``drain`` is the provider's release; ``pool_ended`` (the pool is
  stopped: no member is started, restarted or replaced any more) stops the pool's
  deployed app, which terminates every container left.

App lifecycle: the app is deployed by the first spawn of a pool process and stays
deployed while the pool runs (a restarted or replacement island spawns into it). A
call that returned leaves its container idle for ``scaledown_window_s`` (default
``DEFAULT_SCALEDOWN_S``; Modal's own default is 60 s of a GPU container billed for
nothing). The pool's stop (sampler finished, ``pool-stop``, SIGTERM, ``yeto run
down``) stops the app. A pool process that died without its stop leaves the app
deployed: ``pool-stop`` (no live loop: release from the state file) stops it then,
by name, checked against the app id recorded next to the call record.

The real client (:class:`ModalSdkClient`) deploys one app per pool with a
``rollout_sampler`` GPU function (image, GPU, model volume, store Secret of the
provider config) that runs ``python -m yeto.rl.engine.engine_sampler --spec``
against the run store (the data path ``r2``: buffer shards + policy releases).
It records ``modal_app`` / ``modal_call`` rows in the run's resource ledger, so
``yeto cleanup --run`` and the cost guard see them.

Status: CPU tests drive it with a fake client; it has not spawned a real call.
Pool config::

    "provider": {"kind": "modal", "app_name": "yeto-rollout-<run>", "image": "<registry>/<image>@sha256:<d>",
                 "gpu": "B300", "gpus": 8, "secret": "<modal secret with the store credentials>",
                 "bucket": "<bucket>", "model_volume": "<volume>", "model_mount": "/mnt/yeto-models",
                 "timeout_h": 24, "name_prefix": "yeto-smp", "scaledown_window_s": 2}
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..pool import status as island_status
from ..pool.provider import ProviderCaps
from ..pool.status import LeaseStatus
from . import elastic
from .client import NodeLease, state_file_lock, write_json_atomic
from .tunnel import state_dir

KIND, LEDGER_KIND = "modal", "modal_call"
FUNCTION = "rollout_sampler"
RUNNING, SUCCEEDED, FAILED = "RUNNING", "SUCCEEDED", "FAILED"
STAGE = "/tmp/stage"
CODE_ROOT = "/root"
READY_PREFIX = "pool-ready"
JOIN_PATTERN = "rl_sampler_join"


def ready_key(shard: str, ticket: str, n: int) -> str:
    """The store object a call writes once its sampler joined (one per spawn: a
    restarted call never reads the marker of the call before it)."""
    return f"{READY_PREFIX}/{shard}/{ticket}.{int(n)}.json"
CONFIG_KEYS = frozenset({"kind", "factory", "app_name", "image", "gpu", "gpus", "cpu", "memory_gib", "secret",
                         "bucket", "model_volume", "model_mount", "log_volume", "timeout_h", "python",
                         "name_prefix", "env", "price_usd_h", "scaledown_window_s"})
# seconds a container whose call returned stays up for the next call (billed idle).
# A restart or replacement spawns on the pool's next tick (60 s by default), after a
# longer window would mostly have run out anyway; 2 s is the least Modal takes.
DEFAULT_SCALEDOWN_S = 2
SCALEDOWN_RANGE_S = (2, 3600)


def scaledown_window_s(section: dict[str, Any]) -> int:
    """The provider's ``scaledown_window_s`` (whole seconds in Modal's 2..3600)."""
    value = section.get("scaledown_window_s", DEFAULT_SCALEDOWN_S)
    lo, hi = SCALEDOWN_RANGE_S
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) \
            or not lo <= int(value) <= hi:
        raise ValueError(f"modal provider: scaledown_window_s must be whole seconds in {lo}..{hi} (got {value!r})")
    return int(value)


def _record_ledger(fn: str, *args: Any, **kw: Any) -> None:
    from yeto.cost.ledger import record_safely  # noqa: PLC0415

    record_safely(fn, *args, **kw)


class ModalProvider:
    kind = KIND
    ledger_kind = LEDGER_KIND
    # no lease end (a call ends at the function timeout; the pool refills it as lost)
    caps = ProviderCaps(lease_bounded=False, notice_s=0.0, currency="usd", cold_start_s=900)

    def __init__(self, run_id: str, *, slots: list[int], client: Any = None, name_prefix: str = "yeto-smp",
                 price_usd_h: float | None = None, record: Path | None = None, clock=time.time,
                 config: dict[str, Any] | None = None, sleep=time.sleep, cancel_wait_s: float = 60.0):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", name_prefix):
            raise ValueError("name_prefix: lower-case letters, digits and '-'")
        self.run_id, self.slots = run_id, list(slots)
        self.prefix = f"{name_prefix}-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}"
        self.price_usd_h = None if price_usd_h is None else float(price_usd_h)
        self.config = dict(config or {})
        self._client, self.clock = client, clock
        # a cancel is asynchronous: the call reads RUNNING for some seconds after it
        self.sleep, self.cancel_wait_s = sleep, float(cancel_wait_s)
        self.record = record or state_dir() / f"modal-calls-{self.prefix}.json"
        # the deployed app of this pool (name, id): a later pool-stop checks the app it
        # stops by name against it
        self.app_record = self.record.with_name(f"modal-app-{self.prefix}.json")
        self._app_stopped = False  # this process stopped the app since its last spawn
        # the record is rewritten from several threads (the pool starts islands in the
        # background) and processes (a manual release while the pool runs): every
        # read-modify-write holds the record's state_file_lock, or one drops another's call

    @property
    def client(self):
        if self._client is None:
            self._client = ModalSdkClient(self.config, run_id=self.run_id)
        return self._client

    # -- record
    def _rows(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.record.read_text())
        except (OSError, ValueError):
            return {}

    def _locked(self):
        """The record's thread and process lock (not reentrant: never nest it)."""
        return state_file_lock(self.record)

    def _save(self, rows: dict[str, dict[str, Any]]) -> None:
        """Replace the record (a temporary file of its own); callers hold :meth:`_locked`."""
        write_json_atomic(self.record, rows)

    def _ours(self, ticket: str) -> bool:
        return ticket.startswith(self.prefix + "-") and ticket in self._rows()

    def call_id(self, ticket: str) -> str | None:
        return (self._rows().get(ticket) or {}).get("call")

    def ready_key(self, ticket: str) -> str | None:
        """The ready marker of the island's current call (None: not spawned)."""
        return (self._rows().get(ticket) or {}).get("ready_key")

    # -- IslandProvider
    def order(self, account: int) -> str:
        with self._locked():
            rows = self._rows()
            n = 1 + max([int(r.get("n", 0)) for r in rows.values()] or [0])
            name = f"{self.prefix}-{n}"
            rows[name] = {"slot": account, "n": n, "open": True, "ordered_at": self.clock(), "call": None}
            self._save(rows)
        return name

    def spawn(self, ticket: str, payload: dict[str, Any]) -> str:
        """Start (or restart) the island's call; a running earlier call is cancelled first."""
        if not self._ours(ticket):
            raise PermissionError(f"ticket {ticket} is not a Modal island of run {self.run_id}")
        if not self._rows()[ticket].get("open"):
            # a background start that ran after the island's release (the pool stopped):
            # a call spawned now would have no member to drain it
            raise RuntimeError(f"Modal island {ticket} is released; no call is spawned for it")
        old = self.call_id(ticket)
        if old and self.client.status(old) == RUNNING:
            self.client.cancel(old)
        rows = self._rows()
        n = len(rows[ticket].get("calls") or []) + 1
        if payload.get("shard") is not None:
            payload = {**payload, "ready_key": ready_key(str(payload["shard"]), ticket, n)}
        self._app_stopped = False
        call = str(self.client.spawn(payload))
        self._note_app()
        with self._locked():
            rows = self._rows()
            rows[ticket].update(call=call, spawned_at=self.clock(), cancelled=None,
                                ready_key=payload.get("ready_key"))
            rows[ticket].setdefault("calls", []).append(call)
            self._save(rows)
        _record_ledger("created", LEDGER_KIND, call, FUNCTION, ticket=ticket, run=self.run_id, pool=True,
                       usd_per_hour=self.price_usd_h)
        return call

    def _app_doc(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.app_record.read_text())
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    def _note_app(self) -> None:
        """Record the deployed app's id (once per deploy) next to the call record."""
        app_id = getattr(self.client, "app_id", None)
        if not app_id:
            return
        with state_file_lock(self.app_record):
            doc = self._app_doc()
            if doc.get("app_id") == app_id and not doc.get("stopped_at"):
                return
            write_json_atomic(self.app_record, {"app_name": getattr(self.client, "app_name", None),
                                                "app_id": app_id, "deployed_at": self.clock()})

    def stop_app(self, why: str) -> dict[str, Any] | None:
        """Stop the pool's deployed app (its containers end; nothing bills). Only once the
        pool is stopped: a stopped app takes no spawn until the next deploy. None: the
        client has no app to stop (a test double)."""
        stop = getattr(self.client, "stop_app", None)
        if stop is None:
            return None
        doc = self._app_doc()
        out = {"app": getattr(self.client, "app_name", None), "app_id": doc.get("app_id")}
        if self._app_stopped or (doc.get("stopped_at") and not getattr(self.client, "app_id", None)):
            return {**out, "result": "already stopped"}
        if not doc.get("app_id") and not getattr(self.client, "app_id", None):
            # no spawn of this pool recorded a deploy: an app of that name is not ours to stop
            return {**out, "result": "not deployed"}
        result = str(stop(expect_id=doc.get("app_id")))
        self._app_stopped = True
        if doc:
            with state_file_lock(self.app_record):
                write_json_atomic(self.app_record, {**self._app_doc(), "stopped_at": self.clock(),
                                                    "stopped_why": why, "stop_result": result})
        return {**out, "result": result}

    def call_state(self, ticket: str) -> str | None:
        """RUNNING / SUCCEEDED / FAILED of the island's call (None: not spawned)."""
        call = self.call_id(ticket)
        return None if call is None else self.client.status(call)

    def status(self, account: int, ticket: str) -> LeaseStatus:
        row = self._rows().get(ticket)
        if row is None or not row.get("open"):
            return LeaseStatus("ended", end_reason="released")
        if not row.get("call"):
            return LeaseStatus("active")  # nothing runs (or bills) until the pool starts it
        state = self.client.status(row["call"])  # raises: a blip to the pool, never an end
        if state == RUNNING:
            return LeaseStatus("active")
        if row.get("cancelled") == row["call"]:
            return LeaseStatus("ended", end_reason="released")
        if state == SUCCEEDED:
            # the call returned with its sampler's exit 0 (the run finished, the data
            # ended): the island is not lost; the probe reads the exited sampler, the
            # pool drains it and orders no replacement
            return LeaseStatus("active")
        return LeaseStatus("ended", end_reason="lost", detail=self._failure(row["call"]))

    def _failure(self, call: str) -> str | None:
        """Why the call failed, as its client tells it (``ModalSdkClient.failure``: the
        sampler's exit code and stderr tail, or the raised error); None when unknown. A
        sampler that exits on a configuration error otherwise reads only ``lost``."""
        failure = getattr(self.client, "failure", None)
        if failure is None:
            return None
        try:
            return failure(call)
        except Exception as error:  # noqa: BLE001 - the reason is a courtesy, the end stands
            return f"failure not read: {type(error).__name__}: {error}"[:FAILURE_DETAIL_CHARS]

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        if not self._ours(ticket):
            raise PermissionError(f"ticket {ticket} was not ordered by this provider for run {self.run_id}")
        call = self.call_id(ticket)
        if call:
            if self.client.status(call) == RUNNING:
                self.client.cancel(call)
            if not self._ended_after_cancel(call):
                raise RuntimeError(f"Modal call {call} of {ticket}: cancel sent but it still runs "
                                   f"after {self.cancel_wait_s:g} s")
            _record_ledger("released", LEDGER_KIND, call, FUNCTION)
        with self._locked():
            rows = self._rows()
            rows[ticket].update(open=False, cancelled=call)
            self._save(rows)
        return {"ticket": ticket, "call": call, "state": "ended"}

    def _ended_after_cancel(self, call: str) -> bool:
        """The call no longer runs, polled for up to ``cancel_wait_s`` (Modal stops the
        container a few seconds after it took the cancel)."""
        deadline = self.clock() + self.cancel_wait_s
        while self.client.status(call) == RUNNING:
            if self.clock() >= deadline:
                return False
            self.sleep(2.0)
        return True

    def node(self, account: int, ticket: str) -> NodeLease:
        return NodeLease(account, ticket, "", "", "", None, None, None, None)  # no host to reach

    def balance(self) -> int | None:
        return None  # Modal bills outside; the run's spend cap is the guard

    def account_free(self, account: int) -> bool:
        return not any(r.get("open") and r.get("slot") == account for r in self._rows().values())

    def run_tickets(self) -> list[tuple[int, str]]:
        return [(int(r["slot"]), name) for name, r in self._rows().items() if r.get("open")]

    def hourly_cost(self, ticket: str | None = None) -> float | None:  # noqa: ARG002 - one price per pool
        return self.price_usd_h


class ModalPoolOps:
    """The pool's ``PoolOps`` for Modal islands (no SSH node route)."""

    start_in_background = True  # a spawn returns at once; the image build of the first deploy does not

    def __init__(self, cfg: dict[str, Any], *, provider: ModalProvider | None = None, fence_store: Any = None):
        from . import cli  # noqa: PLC0415

        self.cfg = cfg
        self.run_id = cli.run_id(cfg)
        self.provider = provider if provider is not None else from_config(cfg, self.run_id)
        self._store = fence_store
        self._ready_seen: set[str] = set()

    # -- supply
    def order(self, account: int) -> str:
        return self.provider.order(account)

    def poll(self, m) -> LeaseStatus:
        return self.provider.status(m.account, m.ticket)

    def balance(self) -> int | None:
        return None

    def account_free(self, account: int) -> bool:
        return self.provider.account_free(account)

    def run_tickets(self) -> list[tuple[int, str]]:
        return self.provider.run_tickets()

    # -- island
    def payload(self, m, shards: list[str]) -> dict[str, Any]:
        from . import cli  # noqa: PLC0415

        spec = {"start_on_latest": True, **cli.render_spec(self.cfg, m, shards)}
        prefix = str(self.cfg["run_prefix"]).strip("/")
        bucket = self.provider.config.get("bucket") or ""
        spec["store"] = spec["buffer_root"] = f"{STAGE}/{bucket}/runs/{prefix}"
        return {"spec": spec, "shard": m.shard, "run_prefix": prefix, "bucket": bucket,
                "python": self.provider.config.get("python", "python3"),
                "env": dict(self.provider.config.get("env") or {})}

    def start(self, m, shards: list[str]) -> None:
        self.provider.spawn(m.ticket, self.payload(m, shards))

    def restart(self, m, shards: list[str]) -> None:
        self.start(m, shards)

    def ready(self, m) -> bool:
        return self.provider.call_state(m.ticket) == RUNNING and self._reported_ready(m)

    def _reported_ready(self, m) -> bool:
        """The current call wrote its ready marker (its sampler joined). A store
        blip reads not ready this tick; the pool's start timeout bounds the wait."""
        key = self.provider.ready_key(m.ticket)
        if key is None:
            return False
        if key in self._ready_seen:
            return True
        try:
            seen = self._fence_store().objects.read(key) is not None
        except Exception:  # noqa: BLE001 - the store did not answer: not ready yet
            return False
        if seen:
            self._ready_seen.add(key)
        return seen

    def probe(self, m) -> island_status.Probe:
        try:
            state = self.provider.call_state(m.ticket)
        except Exception:  # noqa: BLE001 - Modal did not answer: the island is unreachable this tick
            return island_status.Probe(node=island_status.PROBE_UNREACHABLE)
        if state is None:
            return island_status.Probe(node="ok", sampler="absent")
        if state == RUNNING:
            return island_status.Probe(node="ok", sampler="running", joined=self._reported_ready(m))
        return island_status.Probe(node="ok", sampler="exited", exit_code=0 if state == SUCCEEDED else 1)

    def _fence_store(self) -> Any:
        if self._store is None:
            env = elastic.read_env_file(self.cfg["r2_env_file"])
            bucket = env.get(self.cfg.get("bucket_env", "R2_BUCKET"))
            if not bucket:
                raise RuntimeError(f"no {self.cfg.get('bucket_env', 'R2_BUCKET')} in the store env file")
            uri = f"s3://{bucket}/runs/{self.cfg['run_prefix'].strip('/')}"
            self._store = elastic.s3_fence_store(uri, self.cfg["r2_env_file"])
        return self._store

    # the work contract, as on the SSH route (yeto.launch.nc.cli.pool_data_path): fence,
    # and the "drain" scale down (the sampler claims no new task, answers drained)
    def _dispense_ledger(self) -> tuple[Any, str]:
        from .cli import NcSshOps  # noqa: PLC0415

        return NcSshOps._dispense_ledger(self)  # type: ignore[arg-type]

    def data_path(self) -> Any:
        from .cli import pool_data_path, pool_dispense  # noqa: PLC0415

        return pool_data_path(self.cfg, self._fence_store,
                              lambda: self._dispense_ledger() if pool_dispense(self.cfg) is not None else None)

    def fence(self, m) -> Any:
        return self.data_path().fence_shard(m.shard, by=m.ticket)

    def fence_incarnation(self, m) -> Any:
        return self.data_path().fence_incarnation(m.shard, by=m.ticket)

    def request_drain(self, m) -> None:
        import time as _time  # noqa: PLC0415

        m.drain_seq = self.data_path().request_drain(m.shard, seq=f"{m.ticket}:{_time.time():.3f}")

    def drained(self, m) -> bool:
        return m.drain_seq is not None and bool(self.data_path().drain_answered(m.shard, m.drain_seq))

    def cancel_drain(self, m) -> None:
        self.data_path().cancel_drain(m.shard)
        m.drain_seq = None

    def reassert_drain(self, m) -> None:
        again = getattr(self.data_path(), "reassert_drain", None)
        if again is not None:  # a data path whose drain state lives only in its server (syncer)
            again(m.shard)

    def hourly_cost(self, m) -> float | None:
        return self.provider.hourly_cost(m.ticket)

    def drain(self, m) -> None:
        self.provider.release(m.account, m.ticket)

    def pool_ended(self, why: str) -> dict[str, Any] | None:
        """The pool is stopped (no member is started or replaced any more): stop its app."""
        return self.provider.stop_app(why)


# -- the real client -------------------------------------------------------------------

class ModalSdkClient:
    """``spawn`` / ``status`` / ``cancel`` / ``stop_app`` over the ``modal`` SDK (imported
    lazily). The app is deployed on the first spawn of this process (a deployed app
    outlives it: the pool's stop stops it, :meth:`stop_app`)."""

    def __init__(self, cfg: dict[str, Any], *, run_id: str, modal: Any = None):
        self.cfg, self.run_id = dict(cfg), run_id
        self.app_name = str(cfg.get("app_name") or f"yeto-rollout-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}")
        self._modal = modal
        self._deployed = False
        self._deploy_lock = threading.Lock()
        self._app: Any = None
        self.app_id: str | None = None

    @property
    def modal(self):
        if self._modal is None:
            import modal  # noqa: PLC0415

            self._modal = modal
        return self._modal

    def build_app(self):
        from yeto.cloud.modal_images import engine_registry_image  # noqa: PLC0415

        modal, c = self.modal, self.cfg
        image = (engine_registry_image(modal, c["image"]).entrypoint([])
                 .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
                 .add_local_python_source("yeto", copy=False, ignore=["**/__pycache__", "**/*.pyc"]))
        app = modal.App(self.app_name)
        volumes = {}
        if c.get("model_volume"):
            volumes[c.get("model_mount", "/mnt/yeto-models")] = modal.Volume.from_name(c["model_volume"]).read_only()
        app.function(image=image, gpu=f"{c.get('gpu', 'B300')}:{int(c.get('gpus', 8))}",
                     cpu=int(c.get("cpu", 64)), memory=int(c.get("memory_gib", 1024)) * 1024,
                     timeout=int(float(c.get("timeout_h", 24)) * 3600), volumes=volumes,
                     scaledown_window=scaledown_window_s(c),
                     secrets=[modal.Secret.from_name(c["secret"])], name=FUNCTION)(rollout_sampler_body)
        return app

    def deploy(self) -> None:
        # the pool starts islands on background threads (start_in_background): two first
        # spawns must not deploy the app at once (Modal refuses the second deploy with
        # "App deploy name ... is already taken" and that island's start fails)
        with self._deploy_lock:
            if self._deployed:
                return
            app = self.build_app()
            _record_ledger("intent", "modal_app", self.app_name)
            app.deploy(name=self.app_name)
            if getattr(app, "app_id", None):
                _record_ledger("created", "modal_app", app.app_id, self.app_name, role="rollout", run=self.run_id)
            self._app, self.app_id = app, getattr(app, "app_id", None)
            self._deployed = True

    def spawn(self, payload: dict[str, Any]) -> str:
        self.deploy()
        fn = self.modal.Function.from_name(self.app_name, FUNCTION)
        return str(fn.spawn(payload).object_id)

    def status(self, call_id: str) -> str:
        """RUNNING; SUCCEEDED: the body returned and its sampler exited 0; FAILED: it
        raised (cancelled, function timeout) or its sampler exited non-zero."""
        call = self.modal.FunctionCall.from_id(call_id)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return RUNNING
        except Exception as exc:  # noqa: BLE001 - a raised result is a failed island
            if "timeout" in str(exc).lower() and type(exc).__name__ in ("TimeoutError", "OutputExpiredError"):
                return RUNNING
            return FAILED
        if isinstance(result, dict) and result.get("rc") not in (0, None):
            return FAILED  # rollout_sampler_body returns the sampler's exit code
        return SUCCEEDED

    def failure(self, call_id: str) -> str | None:
        """Why a call that ended failed (:func:`failure_detail` of its result or error)."""
        call = self.modal.FunctionCall.from_id(call_id)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return None  # still running
        except Exception as exc:  # noqa: BLE001 - the call raised: that is the reason
            return failure_detail(None, exc)
        return failure_detail(result)

    def cancel(self, call_id: str) -> None:
        self.modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def stop_app(self, *, expect_id: str | None = None) -> str:
        """Stop the deployed app (every container ends). The app this process deployed is
        stopped by its handle; otherwise (a pool-stop after the pool process died) it is
        looked up by name and stopped only when its id is ``expect_id`` (None: not
        checked). Returns ``stopped``, ``absent`` (no deployed app of that name) or
        ``other deployment`` (the name now belongs to an app this pool did not deploy)."""
        with self._deploy_lock:
            app = self._app
            if app is None or not getattr(app, "app_id", None):
                try:
                    app = self.modal.App.lookup(self.app_name)
                except Exception as exc:  # noqa: BLE001 - NotFoundError: nothing deployed under the name
                    if type(exc).__name__ == "NotFoundError" or "not found" in str(exc).lower():
                        self._deployed, self._app, self.app_id = False, None, None
                        return "absent"
                    raise
                found = getattr(app, "app_id", None)
                if expect_id and found and found != expect_id:
                    return "other deployment"
            app.stop()
            app_id = getattr(app, "app_id", None) or expect_id
            self._deployed, self._app, self.app_id = False, None, None  # a later spawn deploys again
        if app_id:
            _record_ledger("released", "modal_app", app_id, self.app_name)
        return "stopped"


STDERR_TAIL_LINES = 20  # the sampler's last stderr lines in the call result (a traceback's end)
FAILURE_DETAIL_CHARS = 1500  # the reason in the pool's ended event, at most


def failure_detail(result: Any, error: BaseException | None = None) -> str | None:
    """A failed call's reason in one line for the pool's ended event: the raised error,
    or the sampler's exit code, run time and the last lines of its stderr
    (:func:`rollout_sampler_body`'s result). None for a result that is no failure."""
    if error is not None:
        text = f"call raised {type(error).__name__}: {error}"
    elif isinstance(result, dict) and result.get("rc") not in (0, None):
        tail = [str(line).rstrip() for line in result.get("stderr_tail") or () if str(line).strip()]
        text = f"sampler exited rc={result.get('rc')} after {result.get('secs')} s"
        if tail:
            text += "; stderr tail: " + " | ".join(tail)
    else:
        return None
    return text if len(text) <= FAILURE_DETAIL_CHARS else "..." + text[-(FAILURE_DETAIL_CHARS - 3):]


def store_env(payload: dict[str, Any], environ: dict[str, str]) -> dict[str, str]:
    """The store variables of the sampler process from the container environment (the
    Secret): yeto-named store credentials as they are, else the AWS names."""
    from yeto.cloud.s3_store import (CREDENTIALS_AWS_ENV, CREDENTIALS_ENV, ENDPOINT_ENV,  # noqa: PLC0415
                                     KEY_ID_ENV, REGION_ENV, STAGE_ENV, URI_ENV)

    env = {"YETO_RL_STORE_IMPL": "yeto.cloud.s3_store:store_from_env", "PYTHONPATH": CODE_ROOT,
           URI_ENV: f"s3://{payload['bucket']}/runs/{payload['run_prefix']}", STAGE_ENV: STAGE}
    endpoint = environ.get(ENDPOINT_ENV) or environ.get("AWS_ENDPOINT_URL")
    if not endpoint:
        raise ValueError(f"the store Secret has no endpoint ({ENDPOINT_ENV} or AWS_ENDPOINT_URL)")
    env[ENDPOINT_ENV] = endpoint
    env[REGION_ENV] = environ.get(REGION_ENV) or "auto"
    if not environ.get(KEY_ID_ENV):
        env[CREDENTIALS_ENV] = CREDENTIALS_AWS_ENV
    env.update({str(k): str(v) for k, v in (payload.get("env") or {}).items()})
    return env


def tape_files(spec: dict[str, Any]) -> set[Path]:
    """Every incarnation's tape next to the spec's ``event_tape`` (``events.<inc>.jsonl``)."""
    tape = spec.get("event_tape")
    if not tape:
        return set()
    path = Path(str(tape)).expanduser()
    return set(path.parent.glob(f"{path.stem}*{path.suffix or '.jsonl'}"))


def join_seen(spec: dict[str, Any], *, before: set[Path] | frozenset[Path] = frozenset()) -> bool:
    """The sampler's tape (an incarnation's ``events.<inc>.jsonl`` next to the
    spec's ``event_tape``) has the join event. ``before``: tapes that existed before
    this call started its sampler -- Modal reuses a warm container for the next
    call, so an earlier call's tape (and its join) may still be on its disk."""
    for f in sorted(tape_files(spec) - set(before)):
        try:
            if JOIN_PATTERN in f.read_text(errors="replace"):
                return True
        except OSError:
            continue
    return False


def rollout_sampler_body(payload: dict[str, Any], *, popen: Any = None, objects: Any = None,
                         poll_s: float = 5.0, sleep: Any = time.sleep, root: str = "/tmp/rollout") -> dict[str, Any]:
    """Runs INSIDE the Modal container: write the spec, run the engine sampler, and
    write the call's ready marker (``payload["ready_key"]``) in the run store once
    the sampler's tape has the join event."""
    import subprocess  # noqa: PLC0415

    t0 = time.time()
    out = Path(root) / str(payload["shard"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(payload["spec"], indent=1))
    env = dict(os.environ, **store_env(payload, dict(os.environ)))
    before = tape_files(payload["spec"])  # a reused container keeps an earlier call's tapes
    proc = (popen or subprocess.Popen)([payload.get("python", "python3"), "-m", "yeto.rl.engine.engine_sampler",
                                        "--spec", str(out / "spec.json")], env=env,
                                       stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1)
    tail = _StderrTail(getattr(proc, "stderr", None))
    key, reported = payload.get("ready_key"), False
    while proc.poll() is None:
        if key and not reported and join_seen(payload["spec"], before=before):
            try:
                if objects is None:
                    from yeto.cloud import s3_store  # noqa: PLC0415

                    objects = s3_store.head_objects(env[s3_store.URI_ENV], environ=env)
                objects.write(key, json.dumps({"ready_at": time.time(), "shard": payload["shard"]}).encode())
                reported = True
            except Exception as error:  # noqa: BLE001 - retried next poll; the pool's start timeout bounds it
                print(f"rollout_sampler: ready marker not written yet: {error}", flush=True)
        sleep(poll_s)
    return {"rc": proc.returncode, "secs": round(time.time() - t0, 1), "reported_ready": reported,
            "stderr_tail": tail.lines()}


class _StderrTail:
    """Copies the sampler's stderr to the container's (Modal's app log keeps it) and
    keeps its last :data:`STDERR_TAIL_LINES` lines for the call result."""

    def __init__(self, stream: Any) -> None:
        import collections  # noqa: PLC0415
        import threading  # noqa: PLC0415

        self._lines: collections.deque[str] = collections.deque(maxlen=STDERR_TAIL_LINES)
        self._thread = None
        if stream is not None and callable(getattr(stream, "readline", None)):
            self._thread = threading.Thread(target=self._copy, args=(stream,), daemon=True)
            self._thread.start()

    def _copy(self, stream: Any) -> None:
        import sys  # noqa: PLC0415

        for line in iter(stream.readline, ""):
            sys.stderr.write(line)
            sys.stderr.flush()
            self._lines.append(line.rstrip("\n"))

    def lines(self) -> list[str]:
        if self._thread is not None:
            self._thread.join(timeout=10.0)  # the pipe closes as the sampler exits
        return list(self._lines)


# -- config ---------------------------------------------------------------------------

def registry_factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]) -> ModalProvider:  # noqa: ARG001
    """The registered ``modal`` factory (``yeto.launch.pool.provider``)."""
    return from_config(cfg, run_id)


def pool_ops(cfg: dict[str, Any]) -> ModalPoolOps:
    """The registered ``modal`` pool ops (a FunctionCall is not an SSH node)."""
    return ModalPoolOps(cfg)


def from_config(cfg: dict[str, Any], run_id: str, *, client: Any = None) -> ModalProvider:
    section = dict(cfg.get("provider") or {})
    from . import cli as _cli  # noqa: PLC0415

    if not section.get("image") and _cli.sampler_engine_requested(cfg):
        # hetero-1114 A22: the pool's engine / platform picks the default image
        section["image"] = _cli.pool_sampler_image(cfg)
    unknown = set(section) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"modal provider: unknown keys {sorted(unknown)}")
    scaledown_window_s(section)
    from . import cli  # noqa: PLC0415

    return ModalProvider(run_id, slots=cli.slot_list(cfg), client=client,
                         name_prefix=str(section.get("name_prefix", "yeto-smp")),
                         price_usd_h=section.get("price_usd_h"), config=section)


def offline_problems(cfg: dict[str, Any]) -> list[str]:
    """``pool --dry-run`` checks of a Modal pool (no Modal call)."""
    import importlib.util  # noqa: PLC0415

    try:
        provider = from_config(cfg, str(cfg.get("run") or cfg.get("shard_base")), client=object())
    except (ValueError, TypeError, KeyError) as error:
        return [str(error)]
    section = provider.config
    problems = [f"modal provider: {key} is required" for key in ("image", "secret", "bucket") if not section.get(key)]
    if cfg.get("route", "ssh") != "ssh":
        problems.append("modal provider: route must be ssh (the default); Modal islands have no sky route")
    if cfg.get("model_pull") or cfg.get("node_hooks"):
        problems.append("modal provider: model_pull / node_hooks need a node; use a Modal volume (model_volume)")
    if importlib.util.find_spec("modal") is None:
        problems.append("the 'modal' Python package is not installed")
    return problems
