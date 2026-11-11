"""Modal rollout provider: sampler islands as Modal GPU function calls, under the
same pool policy as National Compute (scale, eviction -> fence -> refill,
claim release, budget, breaker, events).

A Modal island has no node to order and log into: the "lease" is the function
call itself. So the provider splits like this:

* :class:`ModalProvider` (the :class:`~.provider.LeaseProvider` contract):
  ``order(slot)`` reserves a ticket ``<prefix>-<run>-<n>`` in a local record and
  calls nothing (nothing bills before the call is spawned); ``status``: a ticket
  without a call reads ``active`` (ready to start), a call that runs reads
  ``active``, a call that ended reads ended (``released`` when this provider
  cancelled it, ``lost`` when it exited by itself or failed: fence, release its
  claims, refill); a failed status call raises (a control-plane blip). ``release``
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
  SSH route; ``drain`` is the provider's release.

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
                 "timeout_h": 24, "name_prefix": "yeto-smp"}
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ..pool.provider import ProviderCaps
from . import elastic
from .client import NodeLease
from .elastic import LeaseStatus
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
                         "name_prefix", "env", "price_usd_h"})


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
                 config: dict[str, Any] | None = None):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", name_prefix):
            raise ValueError("name_prefix: lower-case letters, digits and '-'")
        self.run_id, self.slots = run_id, list(slots)
        self.prefix = f"{name_prefix}-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}"
        self.price_usd_h = None if price_usd_h is None else float(price_usd_h)
        self.config = dict(config or {})
        self._client, self.clock = client, clock
        self.record = record or state_dir() / f"modal-calls-{self.prefix}.json"

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

    def _save(self, rows: dict[str, dict[str, Any]]) -> None:
        self.record.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.record.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        tmp.replace(self.record)

    def _ours(self, ticket: str) -> bool:
        return ticket.startswith(self.prefix + "-") and ticket in self._rows()

    def call_id(self, ticket: str) -> str | None:
        return (self._rows().get(ticket) or {}).get("call")

    def ready_key(self, ticket: str) -> str | None:
        """The ready marker of the island's current call (None: not spawned)."""
        return (self._rows().get(ticket) or {}).get("ready_key")

    # -- LeaseProvider
    def order(self, account: int) -> str:
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
        old = self.call_id(ticket)
        if old and self.client.status(old) == RUNNING:
            self.client.cancel(old)
        rows = self._rows()
        n = len(rows[ticket].get("calls") or []) + 1
        if payload.get("shard") is not None:
            payload = {**payload, "ready_key": ready_key(str(payload["shard"]), ticket, n)}
        call = str(self.client.spawn(payload))
        rows = self._rows()
        rows[ticket].update(call=call, spawned_at=self.clock(), cancelled=None, ready_key=payload.get("ready_key"))
        rows[ticket].setdefault("calls", []).append(call)
        self._save(rows)
        _record_ledger("created", LEDGER_KIND, call, FUNCTION, ticket=ticket, run=self.run_id,
                       usd_per_hour=self.price_usd_h)
        return call

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
        return LeaseStatus("ended", end_reason="lost")

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        if not self._ours(ticket):
            raise PermissionError(f"ticket {ticket} was not ordered by this provider for run {self.run_id}")
        call = self.call_id(ticket)
        if call:
            if self.client.status(call) == RUNNING:
                self.client.cancel(call)
            if self.client.status(call) == RUNNING:
                raise RuntimeError(f"Modal call {call} of {ticket}: cancel sent but it still runs")
            _record_ledger("released", LEDGER_KIND, call, FUNCTION)
        rows = self._rows()
        rows[ticket].update(open=False, cancelled=call)
        self._save(rows)
        return {"ticket": ticket, "call": call, "state": "ended"}

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

    def probe(self, m) -> elastic.Probe:
        try:
            state = self.provider.call_state(m.ticket)
        except Exception:  # noqa: BLE001 - Modal did not answer: the island is unreachable this tick
            return elastic.Probe(node=elastic.PROBE_UNREACHABLE)
        if state is None:
            return elastic.Probe(node="ok", sampler="absent")
        if state == RUNNING:
            return elastic.Probe(node="ok", sampler="running", joined=self._reported_ready(m))
        return elastic.Probe(node="ok", sampler="exited", exit_code=0 if state == SUCCEEDED else 1)

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

    def fence(self, m) -> None:
        self.data_path().fence_shard(m.shard, by=m.ticket)

    def fence_incarnation(self, m) -> None:
        self.data_path().fence_incarnation(m.shard, by=m.ticket)

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


# -- the real client -------------------------------------------------------------------

class ModalSdkClient:
    """``spawn`` / ``status`` / ``cancel`` over the ``modal`` SDK (imported lazily). The
    app is deployed on the first spawn of this process (a deployed app outlives it)."""

    def __init__(self, cfg: dict[str, Any], *, run_id: str, modal: Any = None):
        self.cfg, self.run_id = dict(cfg), run_id
        self.app_name = str(cfg.get("app_name") or f"yeto-rollout-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}")
        self._modal = modal
        self._deployed = False

    @property
    def modal(self):
        if self._modal is None:
            import modal  # noqa: PLC0415

            self._modal = modal
        return self._modal

    def build_app(self):
        modal, c = self.modal, self.cfg
        image = (modal.Image.from_registry(c["image"]).entrypoint([])
                 .env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
                 .add_local_python_source("yeto", copy=False, ignore=["**/__pycache__", "**/*.pyc"]))
        app = modal.App(self.app_name)
        volumes = {}
        if c.get("model_volume"):
            volumes[c.get("model_mount", "/mnt/yeto-models")] = modal.Volume.from_name(c["model_volume"]).read_only()
        app.function(image=image, gpu=f"{c.get('gpu', 'B300')}:{int(c.get('gpus', 8))}",
                     cpu=int(c.get("cpu", 64)), memory=int(c.get("memory_gib", 1024)) * 1024,
                     timeout=int(float(c.get("timeout_h", 24)) * 3600), volumes=volumes,
                     secrets=[modal.Secret.from_name(c["secret"])], name=FUNCTION)(rollout_sampler_body)
        return app

    def deploy(self) -> None:
        if self._deployed:
            return
        app = self.build_app()
        _record_ledger("intent", "modal_app", self.app_name)
        app.deploy(name=self.app_name)
        if getattr(app, "app_id", None):
            _record_ledger("created", "modal_app", app.app_id, self.app_name, role="rollout", run=self.run_id)
        self._deployed = True

    def spawn(self, payload: dict[str, Any]) -> str:
        self.deploy()
        fn = self.modal.Function.from_name(self.app_name, FUNCTION)
        return str(fn.spawn(payload).object_id)

    def status(self, call_id: str) -> str:
        call = self.modal.FunctionCall.from_id(call_id)
        try:
            call.get(timeout=0)
        except TimeoutError:
            return RUNNING
        except Exception as exc:  # noqa: BLE001 - a raised result is a failed island
            if "timeout" in str(exc).lower() and type(exc).__name__ in ("TimeoutError", "OutputExpiredError"):
                return RUNNING
            return FAILED
        return SUCCEEDED

    def cancel(self, call_id: str) -> None:
        self.modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)


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


def join_seen(spec: dict[str, Any]) -> bool:
    """The sampler's tape (any incarnation's ``events.<inc>.jsonl`` next to the
    spec's ``event_tape``) has the join event."""
    tape = spec.get("event_tape")
    if not tape:
        return False
    path = Path(str(tape)).expanduser()
    for f in sorted(path.parent.glob(f"{path.stem}*{path.suffix or '.jsonl'}")):
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
    proc = (popen or subprocess.Popen)([payload.get("python", "python3"), "-m", "yeto.rl.engine.engine_sampler",
                                        "--spec", str(out / "spec.json")], env=env)
    key, reported = payload.get("ready_key"), False
    while proc.poll() is None:
        if key and not reported and join_seen(payload["spec"]):
            try:
                if objects is None:
                    from yeto.cloud import s3_store  # noqa: PLC0415

                    objects = s3_store.head_objects(env[s3_store.URI_ENV], environ=env)
                objects.write(key, json.dumps({"ready_at": time.time(), "shard": payload["shard"]}).encode())
                reported = True
            except Exception as error:  # noqa: BLE001 - retried next poll; the pool's start timeout bounds it
                print(f"rollout_sampler: ready marker not written yet: {error}", flush=True)
        sleep(poll_s)
    return {"rc": proc.returncode, "secs": round(time.time() - t0, 1), "reported_ready": reported}


# -- config ---------------------------------------------------------------------------

def registry_factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]) -> ModalProvider:  # noqa: ARG001
    """The registered ``modal`` factory (``yeto.launch.pool.provider``)."""
    return from_config(cfg, run_id)


def pool_ops(cfg: dict[str, Any]) -> ModalPoolOps:
    """The registered ``modal`` pool ops (a FunctionCall is not an SSH node)."""
    return ModalPoolOps(cfg)


def from_config(cfg: dict[str, Any], run_id: str, *, client: Any = None) -> ModalProvider:
    section = dict(cfg.get("provider") or {})
    unknown = set(section) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"modal provider: unknown keys {sorted(unknown)}")
    return ModalProvider(run_id, slots=[int(a) for a in cfg["accounts"]], client=client,
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
