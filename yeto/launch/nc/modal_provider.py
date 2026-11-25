"""Modal rollout provider: sampler islands as Modal GPU function calls, under the
same pool policy as National Compute (scale, eviction -> fence -> refill,
claim release, budget, breaker, events).

A Modal island has no node to order and log into: the "lease" is the function
call itself. So the provider splits like this:

* :class:`ModalProvider` (the :class:`~yeto.launch.pool.provider.IslandProvider` contract):
  ``order(slot)`` reserves a ticket ``<prefix>-<run>-<n>`` in a local record and
  calls nothing (nothing bills before the call is spawned); ``status``: a ticket
  without a call reads ``active`` (ready to start, not billing), a call that runs
  reads ``active`` (billing once Modal scheduled it on a GPU worker: a call still
  waiting for one has no task and bills nothing, :meth:`ModalSdkClient.scheduled`),
  a call that ended reads ended (``released`` when this provider
  cancelled it, ``lost`` when it failed or its sampler exited non-zero: fence,
  release its claims, refill); a call whose sampler exited 0 (the run finished)
  reads ``active`` and the probe's exited sampler ends it as finished, with no
  refill; a failed status call raises (a control-plane blip). ``release``
  cancels the call (``terminate_containers=True``) and confirms it no longer
  runs (a call that was never scheduled ended with the cancel: nothing ran); only
  tickets of this run (name prefix *and* the record) are touched. A call that
  waits for a worker is a start the pool bounds by its ``start_timeout_s``.
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
down``) stops the app; a deploy still running then (a slow image build of a
background start) is waited for and spawns no call for its released island. A pool
process that died without its stop leaves the app deployed: ``pool-stop`` (no live
loop: release from the state file) stops it then, by name, checked against the app id
recorded next to the call record (or, when it died during its deploy, by name alone:
the record says a spawn may have deployed).

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
                 "secrets": ["<more modal secrets, e.g. the boat token>"],
                 "bucket": "<bucket>", "model_volume": "<volume>", "model_mount": "/mnt/yeto-models",
                 "tasks_dir": "/mnt/yeto-models/<harbor task directories>",
                 "timeout_h": 24, "name_prefix": "yeto-smp", "scaledown_window_s": 2}

Agentic pools (a sampler spec with ``agentic``: Harbor multi-turn trials in boat or
AgentENV sandboxes). There is no node to set up (``agentic.setup`` is the NC ssh
route's), so the function itself carries what the trials need:

* Harbor and the provider SDK: an image layer installs the pinned Harbor commit
  (``yeto.rl.harness.harbor.pins``) with ``server-dashboard`` and ``boat-sdk``
  (boat) or ``e2b`` (AgentENV) into :data:`HARBOR_SITE`, then removes what the engine
  image already has (``agentic_setup.HARBOR_PRUNE``, the NC route's install), so the
  engine's packages stay as they are; the sampler's ``PYTHONPATH`` adds the site.
* the records proxy: the cloud sandboxes post their model calls to it, so the call
  forwards ``agentic.proxy.port`` through a Modal tunnel (``modal.forward``), binds
  the proxy on all addresses, writes a fresh path secret into a 0600 file and gives
  the sampler ``YETO_RECORDS_PROXY_PUBLIC_BASE`` / ``_SECRET_FILE``
  (:func:`records_tunnel`).
* credentials: the store Secret plus ``provider.secrets`` (the boat token as
  ``BOAT_API_KEY``, the AgentENV key); ``provider.env`` travels in the call payload in
  clear text, so a credential name there is refused.
* tasks and prompts: ``provider.tasks_dir`` (``HARBOR_TASKS_DIR``) and the spec's
  ``prompts.path`` are read from the model volume (under ``model_mount``).
* the boat quota: the pool's samplers get their share of the account
  (``yeto.launch.rollout.pools.with_boat_shares``) in the spec's ``breakers``.
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
from ..pool.provider import provider_caps
from ..pool.status import LeaseStatus
from .client import NodeLease, state_file_lock, write_json_atomic
from .tunnel import state_dir

KIND, LEDGER_KIND = "modal", "modal_call"
FUNCTION = "rollout_sampler"
RUNNING, SUCCEEDED, FAILED = "RUNNING", "SUCCEEDED", "FAILED"
# the call returned with the sampler's "did not run" exit (a stop request on its shard it
# obeyed, or a shard whose data ended: engine_sampler_entry.NOT_RUN_EXIT_CODE)
NOT_RUN = "NOT_RUN"
STAGE = "/tmp/stage"
CODE_ROOT = "/root"
READY_PREFIX = "pool-ready"
JOIN_PATTERN = "rl_sampler_join"


def ready_key(shard: str, ticket: str, n: int) -> str:
    """The store object a call writes once its sampler joined (one per spawn: a
    restarted call never reads the marker of the call before it)."""
    return f"{READY_PREFIX}/{shard}/{ticket}.{int(n)}.json"


def started_key(ready: str) -> str:
    """The store object a call writes as its container starts (the call runs on a worker:
    it bills), next to its ready marker."""
    return ready[:-len(".json")] + ".started.json" if ready.endswith(".json") else ready + ".started"
CONFIG_KEYS = frozenset({"kind", "factory", "app_name", "image", "gpu", "gpus", "cpu", "memory_gib", "secret",
                         "secrets", "bucket", "model_volume", "model_mount", "log_volume", "timeout_h",
                         "python", "name_prefix", "env", "price_usd_h", "scaledown_window_s", "tasks_dir"})
DEFAULT_GPU, DEFAULT_GPUS = "B300", 8  # a pool config without gpu / gpus
STATS_CACHE_S = 5.0  # the function's live stats, read once for all queued calls of a pool tick
DEFAULT_TIMEOUT_H = 24  # the sampler function's timeout without timeout_h: each call's hard end
# a call's timeout counts as its hard end only from twice (replace_lead_s + end_margin_s) on (pool defaults)
DEFAULT_MIN_HARD_END_S = 2 * (3600 + 600)
# seconds a container whose call returned stays up for the next call (billed idle).
# A restart or replacement spawns on the pool's next tick (60 s by default), after a
# longer window would mostly have run out anyway; 2 s is the least Modal takes.
DEFAULT_SCALEDOWN_S = 2
SCALEDOWN_RANGE_S = (2, 3600)
# the pool's app stop waits this long for a deploy of the same process (image build)
DEPLOY_WAIT_S = 900.0


def scaledown_window_s(section: dict[str, Any]) -> int:
    """The provider's ``scaledown_window_s`` (whole seconds in Modal's 2..3600)."""
    value = section.get("scaledown_window_s", DEFAULT_SCALEDOWN_S)
    lo, hi = SCALEDOWN_RANGE_S
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) \
            or not lo <= int(value) <= hi:
        raise ValueError(f"modal provider: scaledown_window_s must be whole seconds in {lo}..{hi} (got {value!r})")
    return int(value)


def _takes_kwarg(fn: Any, name: str) -> bool:
    """``fn`` accepts the keyword ``name`` (a client without it keeps the plain call)."""
    import inspect  # noqa: PLC0415

    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


DEFAULT_MODEL_MOUNT = "/mnt/yeto-models"
HARBOR_SITE = "/opt/yeto-harbor/site"  # Harbor and the provider SDK of an agentic pool (agentic_harbor_layer)
AGENTIC_PROVIDERS = ("boat", "agentenv")  # cloud sandboxes a Modal call reaches (no docker host, no ssh key)
RECORDS_SECRET_NAME = "records-proxy.secret"
# provider.env travels in the spawn payload in clear text: these belong in a Modal secret (provider.secrets)
CREDENTIAL_ENV = re.compile(r"(^|_)(API_KEY|TOKEN|SECRET|PASSWORD|ACCESS_KEY|SECRET_ACCESS_KEY)($|_)|^BOAT_API_KEY",
                            re.IGNORECASE)


def agentic_provider(spec: dict[str, Any] | None, env: dict[str, Any] | None = None) -> str | None:
    """The sandbox provider of an agentic spec (``agentic.task_defaults.provider``, else
    ``HARBOR_ENV_TYPE`` of ``env``); None for a spec that is not agentic."""
    agentic = (spec or {}).get("agentic")
    if not isinstance(agentic, dict):
        return None
    defaults = agentic.get("task_defaults") if isinstance(agentic.get("task_defaults"), dict) else {}
    return str(defaults.get("provider") or (env or {}).get("HARBOR_ENV_TYPE") or "").strip().lower() or "?"


def agentic_harbor_layer(image: Any, *, python: str, provider: str | None) -> Any:
    """``image`` (a ``modal.Image``) with Harbor and the provider's SDK in :data:`HARBOR_SITE`
    (module doc): the NC route's install (``agentic_setup.Provisioner.harbor``) at image
    build time, pinned to the same commit and SDK versions."""
    import base64  # noqa: PLC0415
    import shlex  # noqa: PLC0415

    from yeto.launch import agentic_setup as su  # noqa: PLC0415
    from yeto.rl.harness.harbor.pins import HARBOR_COMMIT, HARBOR_REPO  # noqa: PLC0415

    # the commit's source archive, not git+: the engine images have no git (pip would need it to clone)
    archive = f"{HARBOR_REPO}/archive/{HARBOR_COMMIT}.tar.gz"
    req = f"harbor @ {archive}"
    dash = f"server-dashboard @ {archive}#subdirectory=server-dashboard"
    sdk = {"boat": su.BOAT_SDK_REQUIREMENT, "agentenv": su.E2B_REQUIREMENT}.get(str(provider))
    py, site = shlex.quote(python), shlex.quote(HARBOR_SITE)
    install = (f"{py} -m pip install --quiet --disable-pip-version-check --target {site} {shlex.quote(req)} "
               f"{shlex.quote(dash)}" + (f" {shlex.quote(sdk)}" if sdk else ""))
    # each command becomes one Dockerfile RUN line: the multi-line prune script travels base64-encoded
    script = base64.b64encode(su.HARBOR_PRUNE.encode()).decode()
    runner = f"import base64; exec(base64.b64decode('{script}').decode())"
    prune = f"PYTHONPATH= {py} -c {shlex.quote(runner)} {site} {shlex.quote(','.join(su.HARBOR_KEEP))}"
    return image.run_commands(f"mkdir -p {site}", install, prune)


def _record_ledger(fn: str, *args: Any, **kw: Any) -> None:
    from yeto.cost.ledger import record_safely  # noqa: PLC0415

    record_safely(fn, *args, **kw)


class ModalProvider:
    kind = KIND
    ledger_kind = LEDGER_KIND
    # no lease: a call ends at its function timeout, its hard end (status hard_end: replaced before it)
    caps = provider_caps(KIND)  # yeto/cloud/capabilities.json

    def __init__(self, run_id: str, *, slots: list[int], client: Any = None, name_prefix: str = "yeto-smp",
                 price_usd_h: float | None = None, record: Path | None = None, clock=time.time,
                 config: dict[str, Any] | None = None, sleep=time.sleep, cancel_wait_s: float = 60.0,
                 harbor: str | None = None):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", name_prefix):
            raise ValueError("name_prefix: lower-case letters, digits and '-'")
        self.run_id, self.slots = run_id, list(slots)
        self.prefix = f"{name_prefix}-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}"
        self.price_usd_h = None if price_usd_h is None else float(price_usd_h)
        self.config = dict(config or {})
        self.min_hard_end_s = DEFAULT_MIN_HARD_END_S
        self.harbor = harbor  # an agentic pool's sandbox provider: the image carries Harbor and its SDK
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
            self._client = ModalSdkClient(self.config, run_id=self.run_id, harbor=self.harbor)
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

    # -- group order (yeto.launch.pool.group): one clustered call of ``size`` containers
    # (Modal starts all of them or none), RDMA between them for an RDMA fabric
    fabrics = ("tcp", "rdma")

    def order_group(self, n: int, shape: Any, fabric: str = "tcp") -> list[tuple[int, str]]:
        """Reserve one ticket for a ``modal.clustered(size=n, rdma=...)`` call; like
        :meth:`order`, nothing bills until its call is spawned (by a clustered function:
        :meth:`group_of` tells the caller the size and RDMA)."""
        free = [s for s in self.slots if self.account_free(s)]
        if not free:
            raise RuntimeError(f"Modal provider: no free slot for a group of {n}")
        have = (str(self.config.get("gpu") or "").upper(), int(self.config.get("gpus") or 0))
        want = (str(getattr(shape, "gpu", "")).upper(), int(getattr(shape, "gpus_per_node", 0) or 0))
        if have[0] and have != want:
            raise ValueError(f"Modal provider: its function has {have[1]} x {have[0]} per container, the group "
                             f"asks for {want[1]} x {want[0]}; nothing ordered")
        name = self.order(free[0])
        with self._locked():
            rows = self._rows()
            rows[name].update(size=int(n), rdma=fabric != "tcp", gpu=str(getattr(shape, "gpu", "")),
                              gpus=int(getattr(shape, "gpus_per_node", 0) or 0))
            self._save(rows)
        return [(free[0], name)]

    def group_of(self, ticket: str) -> dict[str, Any] | None:
        """``{"size", "rdma"}`` of a group ticket (None: a single-container ticket)."""
        row = self._rows().get(ticket) or {}
        return {"size": int(row["size"]), "rdma": bool(row.get("rdma"))} if int(row.get("size", 1)) > 1 else None

    def spawn(self, ticket: str, payload: dict[str, Any]) -> str:
        """Start (or restart) the island's call; a running earlier call is cancelled first."""
        if not self._ours(ticket):
            raise PermissionError(f"ticket {ticket} is not a Modal island of run {self.run_id}")
        if self.group_of(ticket) is not None and not getattr(self.client, "clustered", False):
            raise RuntimeError(f"Modal ticket {ticket} is a group of {self.group_of(ticket)['size']} containers; "
                               "this client spawns single-container calls only")
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
        self._note_deploy_intent()
        # checked again inside the client, around its deploy: a pool stopped while the
        # first deploy builds the image releases this ticket, and no call may follow
        spawn = self.client.spawn
        wanted = {"wanted": lambda: self._open(ticket)} if _takes_kwarg(spawn, "wanted") else {}
        call = str(spawn(payload, **wanted))
        self._note_app()
        with self._locked():
            rows = self._rows()
            rows[ticket].update(call=call, spawned_at=self.clock(), cancelled=None, scheduled_at=None,
                                ready_key=payload.get("ready_key"))
            rows[ticket].setdefault("calls", []).append(call)
            self._save(rows)
        # pending: billed from its scheduled_at (recorded by status once Modal placed it on a worker)
        _record_ledger("created", LEDGER_KIND, call, FUNCTION, ticket=ticket, run=self.run_id, pool=True,
                       usd_per_hour=self.price_usd_h, pending=True)
        return call

    def _app_doc(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.app_record.read_text())
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    def _open(self, ticket: str) -> bool:
        return bool((self._rows().get(ticket) or {}).get("open"))

    def _note_deploy_intent(self) -> None:
        """Before a spawn that may deploy the app: record that this pool may own a deployed
        app of its name (a process that dies during the deploy leaves no app id; a later
        pool-stop then stops the app by name)."""
        if getattr(self.client, "app_id", None):
            return  # deployed by this process: its id is on record
        with state_file_lock(self.app_record):
            doc = self._app_doc()
            if doc.get("intent_at") and not doc.get("stopped_at"):
                return
            keep = {} if doc.get("stopped_at") else {k: doc[k] for k in ("app_id", "deployed_at") if k in doc}
            write_json_atomic(self.app_record, {**keep, "app_name": getattr(self.client, "app_name", None),
                                                "intent_at": self.clock()})

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
                                                "app_id": app_id, "deployed_at": self.clock(),
                                                "intent_at": doc.get("intent_at") or self.clock()})

    def stop_app(self, why: str) -> dict[str, Any] | None:
        """Stop the pool's deployed app (its containers end; nothing bills). Only once the
        pool is stopped: a stopped app takes no spawn until the next deploy. A deploy still
        running in this process is waited for, then its app is stopped. Another process
        stops it by name when this pool's record says it may own one (an id, or a spawn
        that may have deployed). None: the client has no app to stop (a test double)."""
        stop = getattr(self.client, "stop_app", None)
        if stop is None:
            return None
        doc = self._app_doc()
        if self._app_stopped:
            return {"app": getattr(self.client, "app_name", None), "app_id": doc.get("app_id"),
                    "result": "already stopped"}
        recorded = bool(doc.get("app_id") or doc.get("intent_at")) and not doc.get("stopped_at")
        result = str(stop(expect_id=doc.get("app_id"), by_name=recorded))
        if result == "not deployed" and doc.get("stopped_at"):
            result = "already stopped"
        self._app_stopped = True
        app_id = getattr(self.client, "stopped_app_id", None) or doc.get("app_id")
        if doc and result != "already stopped":
            with state_file_lock(self.app_record):
                write_json_atomic(self.app_record, {**self._app_doc(), "app_id": app_id, "stopped_at": self.clock(),
                                                    "stopped_why": why, "stop_result": result})
        return {"app": getattr(self.client, "app_name", None), "app_id": app_id, "result": result}

    def call_state(self, ticket: str) -> str | None:
        """RUNNING / SUCCEEDED / NOT_RUN / FAILED of the island's call (None: not spawned)."""
        call = self.call_id(ticket)
        return None if call is None else self.client.status(call)

    def status(self, account: int, ticket: str) -> LeaseStatus:
        row = self._rows().get(ticket)
        if row is None or not row.get("open"):
            return LeaseStatus("ended", end_reason="released")
        if not row.get("call"):
            return LeaseStatus("active", billing=False)  # nothing runs (or bills) until the pool starts it
        state = self.client.status(row["call"])  # raises: a blip to the pool, never an end
        hard_end = self.hard_end(row)
        if state == RUNNING:
            if row.get("scheduled_at") is None and not self._note_scheduled(ticket, row["call"]):
                return LeaseStatus("active", billing=False,
                                   detail=f"Modal call {row['call']} waits to be scheduled on a "
                                          f"{self.config.get('gpu', 'GPU')} worker (no task yet: not billed)")
            since = row.get("scheduled_at")
            if since is None:  # recorded by this status call
                since = (self._rows().get(ticket) or {}).get("scheduled_at")
            return LeaseStatus("active", hard_end=hard_end, billing_since=since)
        if row.get("cancelled") == row["call"]:
            return LeaseStatus("ended", end_reason="released")
        if state in (SUCCEEDED, NOT_RUN):
            # the call returned with its sampler's exit 0 (the run finished, the data
            # ended) or its "did not run" exit: the island is not lost; the probe reads
            # the exited sampler and the pool decides (drain, or a new shard)
            return LeaseStatus("active", hard_end=hard_end)
        # a call that failed at its function timeout expired (hard_end), any other failure is lost
        return LeaseStatus("ended", end_reason="lost", detail=self._failure(row["call"]), hard_end=hard_end)

    def timeout_s(self) -> float:
        """The sampler function's timeout (``timeout_h``, default 24 h), as deployed."""
        return float(int(float(self.config.get("timeout_h", DEFAULT_TIMEOUT_H)) * 3600))

    def hard_end(self, row: dict[str, Any]) -> float | None:
        """When the island's current call reaches its function timeout. None: spawn time unknown,
        or a timeout shorter than ``min_hard_end_s`` (twice the pool's replacement lead plus its
        end margin): a successor ordered a lead before so short a life would be replaced again
        at once, so such a call keeps the old behaviour (refilled when it ends)."""
        spawned = row.get("spawned_at")
        if spawned is None or self.timeout_s() < self.min_hard_end_s:
            return None
        return float(spawned) + self.timeout_s()

    def _scheduled(self, call: str) -> bool:
        """Modal placed the call on a worker, by the first evidence that reads so: its call
        graph (a task id, or past pending), else the function's live stats
        (:meth:`_stats_scheduled`). True when the client cannot tell (a test double, a
        failed lookup): an unknown call is billed, never free. A spawned call that runs
        can read pending in its call graph for its whole life (seen on modal 1.6.4), so
        the graph alone never keeps a call free."""
        scheduled = getattr(self.client, "scheduled", None)
        if scheduled is None:
            return True
        try:
            if bool(scheduled(call)):
                return True
        except Exception:  # noqa: BLE001 - not known: billed
            return True
        return self._stats_scheduled(call)

    def _stats_scheduled(self, call: str) -> bool:
        """The function's live stats (``ModalSdkClient.function_stats``): no input waits
        (``backlog`` 0): every call runs; else the oldest open calls, as many as the inputs
        that run (``running_inputs``, those already known scheduled first), count as
        scheduled (Modal schedules a function's inputs in order). A client without stats,
        or stats that do not read: billed."""
        stats = getattr(self.client, "function_stats", None)
        if stats is None:
            return False  # a client that only has the call graph (a test double): as it says
        try:
            now = self.clock()
            cached = getattr(self, "_stats_cache", None)
            if cached is not None and 0 <= now - cached[0] < STATS_CACHE_S:
                got = cached[1]
            else:  # one lookup per pool tick for all its queued calls
                got = stats()
                self._stats_cache = (now, got)
            backlog, running = int(got["backlog"]), int(got["running_inputs"])
        except Exception:  # noqa: BLE001 - not known: billed
            return True
        if backlog <= 0:
            return True
        rows = [r for r in self._rows().values() if r.get("open") and r.get("call")]
        known = sum(1 for r in rows if r.get("scheduled_at") is not None)
        waiting = sorted((r for r in rows if r.get("scheduled_at") is None),
                         key=lambda r: float(r.get("spawned_at") or 0.0))
        place = next((i for i, r in enumerate(waiting) if r.get("call") == call), None)
        return place is not None and place < running - known

    def note_running(self, ticket: str, *, source: str, at: float | None = None) -> bool:
        """Evidence outside Modal's call state that the island's current call runs (its
        container wrote its started marker, its sampler joined): it bills from ``at`` (the
        marker's own time; never before the spawn, never after now), as a scheduled call
        does. True when it was recorded now (False: already scheduled, or no call)."""
        with self._locked():
            rows = self._rows()
            row = rows.get(ticket) or {}
            call = row.get("call")
            if not call or row.get("scheduled_at") is not None:
                return False
            now = self.clock()
            at = now if at is None else min(now, max(float(at), float(row.get("spawned_at") or at)))
            row.update(scheduled_at=at, scheduled_by=source)
            self._save(rows)
        _record_ledger("created", LEDGER_KIND, call, FUNCTION, scheduled_at=at, scheduled_by=source)
        return True

    def _note_scheduled(self, ticket: str, call: str) -> bool:
        """True once the call is scheduled; the first time, its ``scheduled_at`` goes to the
        record and the resource ledger (the call bills from there)."""
        if not self._scheduled(call):
            return False
        at = self.clock()
        with self._locked():
            rows = self._rows()
            row = rows.get(ticket) or {}
            if row.get("call") == call and row.get("scheduled_at") is None:
                row["scheduled_at"] = at
                self._save(rows)
        _record_ledger("created", LEDGER_KIND, call, FUNCTION, scheduled_at=at)
        return True

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
        container a few seconds after it took the cancel). A call that was never scheduled
        (no task: it waited for a GPU worker) ended with the cancel: nothing ran, and Modal
        may read it RUNNING for long after."""
        deadline = self.clock() + self.cancel_wait_s
        while self.client.status(call) == RUNNING:
            if not self._scheduled(call):
                return True
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

    def hourly_cost(self, ticket: str | None = None) -> float | None:
        """One call of the pool (a group ticket: its ``size`` containers)."""
        if self.price_usd_h is None:
            return None
        group = self.group_of(ticket) if ticket else None
        return self.price_usd_h * (group["size"] if group else 1)


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
        """The provider's status; a call Modal still reads unscheduled whose container wrote
        its started marker (or whose sampler joined) runs: it bills from now on
        (:meth:`ModalProvider.note_running`)."""
        status = self.provider.status(m.account, m.ticket)
        if status.state != "active" or status.billing:
            return status
        found = self._running_evidence(m)
        if found is None:
            return status
        source, at = found
        if self.provider.note_running(m.ticket, source=source, at=at):
            from . import cli  # noqa: PLC0415

            when = "" if at is None else f" ({max(0.0, self.provider.clock() - float(at)):.0f} s ago)"
            cli._log(f"pool: {m.ticket}: Modal reads its call unscheduled, but {source}: billed from then{when}")  # noqa: SLF001
        return self.provider.status(m.account, m.ticket)

    def note_running(self, m, *, source: str) -> bool:
        """The pool saw ``m`` run (its sampler joined) while its call read unscheduled."""
        return self.provider.note_running(m.ticket, source=source)

    def _running_evidence(self, m) -> tuple[str, float | None] | None:
        """What in the run store shows the call runs, and since when (the marker's own
        ``started_at`` / ``ready_at``; None: not in it). None: nothing yet, or no store."""
        key = self.provider.ready_key(m.ticket)
        if key is None:
            return None
        if key in self._ready_seen:
            return "its sampler joined (ready marker)", None
        try:
            objects = self._fence_store().objects
            for marker, field, source in ((started_key(key), "started_at", "its container started (started marker)"),
                                          (key, "ready_at", "its sampler joined (ready marker)")):
                data = objects.read(marker)
                if data is None:
                    continue
                if marker == key:
                    self._ready_seen.add(key)
                try:
                    at = json.loads(data).get(field)
                    return source, float(at) if at is not None else None
                except (ValueError, TypeError, AttributeError):
                    return source, None
        except Exception:  # noqa: BLE001 - the store did not answer: the next poll reads again
            return None
        return None

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
        spec["store_uri"] = f"s3://{bucket}/runs/{prefix}"  # what rl_sampler_start names (not the stage)
        out = {"spec": spec, "shard": m.shard, "run_prefix": prefix, "bucket": bucket,
               "python": self.provider.config.get("python", "python3"),
               "env": dict(self.provider.config.get("env") or {})}
        return agentic_payload(out, self.provider.config) if isinstance(spec.get("agentic"), dict) else out

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
        code = 0 if state == SUCCEEDED else min(island_status.NOT_RUN_EXIT_CODES) if state == NOT_RUN else 1
        return island_status.Probe(node="ok", sampler="exited", exit_code=code)

    def _fence_store(self) -> Any:
        if self._store is None:
            from yeto.node.store import pool_fence_store  # noqa: PLC0415

            self._store = pool_fence_store(self.cfg, error=RuntimeError)
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

    # what the run's store says of a shard, as on the SSH route (NcSshOps): a stale stop
    # request is set aside before a start, an exit is checked against the store, and a
    # shard whose data ended is not given to a new member
    def clear_stale_handoff(self, m, keep: Any) -> Any:
        clear = getattr(self.data_path(), "clear_stale_handoff", None)  # the syncer path keeps no handoff record
        return None if clear is None else clear(m.shard, keep=keep)

    def exit_evidence(self, m) -> Any:
        read = getattr(self.data_path(), "exit_evidence", None)
        return None if read is None else read(m.shard)

    def shard_ended(self, shard: str) -> Any:
        read = getattr(self.data_path(), "shard_ended", None)
        return None if read is None else read(shard)

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

    deploy_wait_s = DEPLOY_WAIT_S

    def __init__(self, cfg: dict[str, Any], *, run_id: str, modal: Any = None, harbor: str | None = None):
        self.cfg, self.run_id = dict(cfg), run_id
        self.harbor = harbor  # an agentic pool's sandbox provider (agentic_harbor_layer)
        self.app_name = str(cfg.get("app_name") or f"yeto-rollout-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}")
        self._modal = modal
        self._deployed = False
        self._deploy_lock = threading.Lock()
        self._app: Any = None
        self.app_id: str | None = None
        self.stopped_app_id: str | None = None  # the app the last stop_app stopped

    @property
    def modal(self):
        if self._modal is None:
            import modal  # noqa: PLC0415

            self._modal = modal
        return self._modal

    def build_app(self):
        from yeto.cloud.modal_images import engine_registry_image  # noqa: PLC0415

        modal, c = self.modal, self.cfg
        image = engine_registry_image(modal, c["image"]).entrypoint([])
        if self.harbor:  # an agentic pool: Harbor and its sandbox SDK beside the engine (module doc)
            image = agentic_harbor_layer(image, python=str(c.get("python", "python3")), provider=self.harbor)
        image = (image.env({"HOME": "/root", "PYTHONUNBUFFERED": "1"})
                 .add_local_python_source("yeto", copy=False, ignore=["**/__pycache__", "**/*.pyc"]))
        app = modal.App(self.app_name)
        volumes = {}
        if c.get("model_volume"):
            volumes[c.get("model_mount", DEFAULT_MODEL_MOUNT)] = modal.Volume.from_name(c["model_volume"]).read_only()
        secrets = [modal.Secret.from_name(name) for name in [c["secret"], *(c.get("secrets") or ())]]
        app.function(image=image, gpu=f"{c.get('gpu', DEFAULT_GPU)}:{int(c.get('gpus', DEFAULT_GPUS))}",
                     cpu=int(c.get("cpu", 64)), memory=int(c.get("memory_gib", 1024)) * 1024,
                     timeout=int(float(c.get("timeout_h", DEFAULT_TIMEOUT_H)) * 3600), volumes=volumes,
                     scaledown_window=scaledown_window_s(c),
                     secrets=secrets, name=FUNCTION)(rollout_sampler_body)
        return app

    def deploy(self) -> None:
        # the pool starts islands on background threads (start_in_background): two first
        # spawns must not deploy the app at once (Modal refuses the second deploy with
        # "App deploy name ... is already taken" and that island's start fails)
        with self._deploy_lock:
            self._deploy_locked()

    def _deploy_locked(self) -> None:
        if self._deployed:
            return
        app = self.build_app()
        _record_ledger("intent", "modal_app", self.app_name)
        app.deploy(name=self.app_name)
        if getattr(app, "app_id", None):
            _record_ledger("created", "modal_app", app.app_id, self.app_name, role="rollout", run=self.run_id)
        self._app, self.app_id = app, getattr(app, "app_id", None)
        self._deployed = True

    def spawn(self, payload: dict[str, Any], *, wanted: Any = None) -> str:
        """Deploy (first spawn) and spawn one call. ``wanted`` () -> bool: is the island
        still open? Asked under the deploy lock before the deploy and again before the
        spawn: a pool stopped during a deploy (a slow image build) released the island,
        and its stop (:meth:`stop_app`, waiting on the same lock) then stops the app with
        no call spawned into it."""
        with self._deploy_lock:
            if wanted is not None and not wanted():
                raise RuntimeError(f"Modal app {self.app_name}: the island was released; nothing deployed or spawned")
            self._deploy_locked()
            if wanted is not None and not wanted():
                raise RuntimeError(f"Modal app {self.app_name}: the island was released during the deploy; "
                                   "no call spawned")
            fn = self.modal.Function.from_name(self.app_name, FUNCTION)
            return str(fn.spawn(payload).object_id)

    def status(self, call_id: str) -> str:
        """RUNNING; SUCCEEDED: the body returned and its sampler exited 0; NOT_RUN: its
        sampler exited with the "did not run" code; FAILED: it raised (cancelled,
        function timeout) or its sampler exited with any other non-zero code."""
        call = self.modal.FunctionCall.from_id(call_id)
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            return RUNNING
        except Exception as exc:  # noqa: BLE001 - a raised result is a failed island
            if "timeout" in str(exc).lower() and type(exc).__name__ in ("TimeoutError", "OutputExpiredError"):
                return RUNNING
            return FAILED
        if isinstance(result, dict) and result.get("rc") in island_status.NOT_RUN_EXIT_CODES:
            return NOT_RUN
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

    def scheduled(self, call_id: str) -> bool:
        """Modal placed the call's input on a worker (a task id), or it already ended.
        False only while its input is pending with no task (no GPU worker of the shape
        free: "waiting to be scheduled"); any doubt reads True (billed)."""
        graph = self.modal.FunctionCall.from_id(call_id).get_call_graph()
        inputs = [i for i in graph if getattr(i, "function_call_id", None) == call_id] or list(graph)
        if not inputs:
            return True  # no input info (retention, an old call): not known
        for i in inputs:  # modal.call_graph.InputStatus: PENDING is 0 (an IntEnum)
            if getattr(i, "task_id", None) or int(getattr(i, "status", 0) or 0) != 0:
                return True
        return False

    def function_stats(self) -> dict[str, int]:
        """The deployed function's live counts (``Function.get_current_stats``): ``backlog``
        (inputs waiting for a worker), ``running_inputs`` and ``tasks`` (its containers)."""
        stats = self.modal.Function.from_name(self.app_name, FUNCTION).get_current_stats()
        return {"backlog": int(stats.backlog), "running_inputs": int(getattr(stats, "num_running_inputs", 0) or 0),
                "tasks": int(getattr(stats, "num_total_runners", 0) or 0)}

    def cancel(self, call_id: str) -> None:
        self.modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def stop_app(self, *, expect_id: str | None = None, by_name: bool = True) -> str:
        """Stop the deployed app (every container ends). A deploy of this process still
        running is waited for (up to ``deploy_wait_s``), then the app it deployed is
        stopped by its handle. Otherwise, with ``by_name`` (a pool-stop after the pool
        process died, or one that died during its deploy), the app is looked up by name
        and stopped unless its id differs from ``expect_id`` (None: not checked).
        Returns ``stopped``, ``absent`` (no deployed app of that name), ``other
        deployment`` (the name belongs to an app this pool did not deploy) or ``not
        deployed`` (no handle and not ``by_name``)."""
        wait_s = float(self.deploy_wait_s)
        if not self._deploy_lock.acquire(timeout=wait_s):
            # the deploy that holds the lock spawns nothing once its island is released;
            # the app it leaves has no container (nothing bills); pool-stop stops it by name
            raise TimeoutError(f"Modal app {self.app_name}: a deploy still runs after {wait_s:g} s; not stopped yet")
        try:
            self.stopped_app_id = None
            app = self._app
            if app is None or not getattr(app, "app_id", None):
                if not by_name:
                    return "not deployed"
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
            self.stopped_app_id = app_id
        finally:
            self._deploy_lock.release()
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


def agentic_payload(payload: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    """An agentic pool's spawn payload (module doc): the records proxy bound on all
    addresses and its port for the call's tunnel (``records_port``), the Harbor
    directories and provider in ``env`` (``provider.env`` still wins) and the Harbor
    site for the sampler's ``PYTHONPATH``."""
    spec = json.loads(json.dumps(payload["spec"]))
    agentic = spec["agentic"]
    proxy = dict(agentic.get("proxy") or {})
    proxy["host"] = "0.0.0.0"  # the Modal tunnel reaches the container, not its loopback; the path secret gates it
    agentic["proxy"] = proxy
    env = {"HARBOR_TASKS_DIR": str(section.get("tasks_dir") or ""),
           "HARBOR_TRIALS_DIR": f"/tmp/rollout/{payload['shard']}/trials"}
    provider = agentic_provider(spec)
    if provider and provider != "?":
        env["HARBOR_ENV_TYPE"] = provider
    env.update(payload.get("env") or {})
    return {**payload, "spec": spec, "env": env, "records_port": int(proxy.get("port") or 0),
            "harbor_site": HARBOR_SITE}


def records_tunnel(payload: dict[str, Any], env: dict[str, str], out: Path, *, forward: Any = None) -> Any:
    """A context that forwards an agentic pool's records proxy port (``payload["records_port"]``)
    through a Modal tunnel for the call's lifetime and gives the sampler its public base
    and a fresh path secret (in a 0600 file under ``out``; only the path is in ``env``).
    Nothing for another pool (``env`` unchanged)."""
    import contextlib  # noqa: PLC0415
    import secrets  # noqa: PLC0415

    from yeto.rl.harness.gateway import records_env  # noqa: PLC0415

    port = int(payload.get("records_port") or 0)

    @contextlib.contextmanager
    def tunnel():
        if not port:
            yield None
            return
        fwd = forward
        if fwd is None:
            import modal  # noqa: PLC0415

            fwd = modal.forward
        with fwd(port) as handle:
            path = out / RECORDS_SECRET_NAME
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(secrets.token_urlsafe(24))
            env[records_env.PROXY_PUBLIC_BASE_ENV] = str(handle.url).rstrip("/")
            env[records_env.PROXY_SECRET_FILE_ENV] = str(path)
            print(f"rollout_sampler: records proxy port {port} tunnelled at {handle.url}", flush=True)
            yield handle

    return tunnel()


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
                         poll_s: float = 5.0, sleep: Any = time.sleep, root: str = "/tmp/rollout",
                         forward: Any = None) -> dict[str, Any]:
    """Runs INSIDE the Modal container: write the spec, run the engine sampler, and
    write the call's ready marker (``payload["ready_key"]``) in the run store once
    the sampler's tape has the join event. An agentic pool's call runs the sampler
    inside its records tunnel (:func:`records_tunnel`; ``forward``: ``modal.forward``)."""
    t0 = time.time()
    out = Path(root) / str(payload["shard"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(payload["spec"], indent=1))
    env = dict(os.environ, **store_env(payload, dict(os.environ)))
    if payload.get("harbor_site"):  # the agentic image layer's Harbor, after yeto's code
        env["PYTHONPATH"] = f"{env.get('PYTHONPATH') or CODE_ROOT}:{payload['harbor_site']}"
    with records_tunnel(payload, env, out, forward=forward):
        return _run_sampler(payload, env, out, t0=t0, popen=popen, objects=objects, poll_s=poll_s, sleep=sleep)


def _run_sampler(payload: dict[str, Any], env: dict[str, str], out: Path, *, t0: float, popen: Any, objects: Any,
                 poll_s: float, sleep: Any) -> dict[str, Any]:
    import subprocess  # noqa: PLC0415

    before = tape_files(payload["spec"])  # a reused container keeps an earlier call's tapes
    proc = (popen or subprocess.Popen)([payload.get("python", "python3"), "-m", "yeto.rl.engine.engine_sampler",
                                        "--spec", str(out / "spec.json")], env=env,
                                       stderr=subprocess.PIPE, text=True, errors="replace", bufsize=1)
    tail = _StderrTail(getattr(proc, "stderr", None))
    key, reported = payload.get("ready_key"), False
    started = started_key(key) if key else None  # the container runs: the call bills (ModalPoolOps.poll)
    while proc.poll() is None:
        if started:
            try:
                if objects is None:
                    from yeto.cloud import s3_store  # noqa: PLC0415

                    objects = s3_store.head_objects(env[s3_store.URI_ENV], environ=env)
                objects.write(started, json.dumps({"started_at": t0, "shard": payload["shard"]}).encode())
                started = None
            except Exception as error:  # noqa: BLE001 - retried next poll; the ready marker or the stats bill it too
                print(f"rollout_sampler: started marker not written yet: {error}", flush=True)
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

    provider = ModalProvider(run_id, slots=cli.slot_list(cfg), client=client,
                         name_prefix=str(section.get("name_prefix", "yeto-smp")),
                         price_usd_h=section.get("price_usd_h") if section.get("price_usd_h") is not None
                         else table_price(section), config=section,
                         harbor=agentic_provider(pool_spec(cfg), section.get("env")))
    lead, margin = cfg.get("replace_lead_s", 3600), cfg.get("end_margin_s", 600)
    if isinstance(lead, (int, float)) and isinstance(margin, (int, float)) and not isinstance(lead, bool):
        provider.min_hard_end_s = 2 * (float(lead) + float(margin))
    return provider


def table_price(section: dict[str, Any]) -> float | None:
    """One sampler container hour (its GPUs) at the unified price table's Modal rate: the
    price a Modal trainer of the same GPUs is billed at (None: the GPU is not in the table)."""
    from yeto.cost import prices  # noqa: PLC0415

    return prices.node_usd_h("modal", section.get("gpu", DEFAULT_GPU), section.get("gpus", DEFAULT_GPUS))


def pool_spec(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The pool's sampler spec with its knobs laid over it (None: none or unreadable)."""
    from yeto.rl.up_consistency import effective_spec  # noqa: PLC0415

    return effective_spec(dict(cfg))


def agentic_problems(cfg: dict[str, Any], section: dict[str, Any]) -> list[str]:
    """``pool --dry-run`` checks of an agentic Modal pool (module doc); [] for another pool."""
    spec = pool_spec(cfg)
    provider = agentic_provider(spec, section.get("env"))
    if provider is None:
        return []
    out = []
    agentic = spec["agentic"]
    if provider not in AGENTIC_PROVIDERS:
        out.append(f"modal provider: agentic.task_defaults.provider {provider!r}: a Modal pool's trials run in cloud "
                   f"sandboxes ({' / '.join(AGENTIC_PROVIDERS)}); a docker host needs the NC ssh route")
    if agentic.get("setup"):
        out.append("modal provider: agentic.setup is the NC ssh node setup; a Modal pool's Harbor comes from its "
                   "image layer and the records proxy from a Modal tunnel: drop agentic.setup")
    port = (agentic.get("proxy") or {}).get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        out.append("modal provider: an agentic pool needs a fixed agentic.proxy.port (the call forwards it through "
                   "a Modal tunnel for the sandboxes)")
    mount = str(section.get("model_mount", DEFAULT_MODEL_MOUNT)).rstrip("/")
    tasks_dir = str(section.get("tasks_dir") or "")
    if not tasks_dir:
        out.append("modal provider: an agentic pool needs tasks_dir (the Harbor task directories, HARBOR_TASKS_DIR, "
                   f"on the model volume under {mount})")
    elif not section.get("model_volume") or not (tasks_dir + "/").startswith(mount + "/"):
        out.append(f"modal provider: tasks_dir {tasks_dir} is not on the model volume (model_volume mounted at "
                   f"{mount}): put the task directories there")
    prompts = str(((spec.get("prompts") or {}).get("path")) or "")
    if prompts and section.get("model_volume") and not (prompts + "/").startswith(mount + "/"):
        out.append(f"modal provider: prompts.path {prompts} is not on the model volume ({mount}): the call reads "
                   "the prompt file there")
    leaked = sorted(k for k in (section.get("env") or {}) if CREDENTIAL_ENV.search(str(k)))
    if leaked:
        out.append(f"modal provider: env {leaked} travel in the call payload in clear text: put them in a Modal "
                   "secret and list it in provider.secrets")
    extra = section.get("secrets")
    if extra is not None and not (isinstance(extra, list) and all(isinstance(s, str) and s for s in extra)):
        out.append("modal provider: secrets must be a list of Modal secret names")
    elif provider == "boat" and not extra:
        out.append("modal provider: a boat pool needs the boat token as BOAT_API_KEY in a Modal secret listed in "
                   "provider.secrets")
    elif provider == "agentenv" and not extra:
        out.append("modal provider: an AgentENV pool needs AGENTENV_API_KEY (and E2B_API_URL) in a Modal secret "
                   "listed in provider.secrets")
    return out


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
    problems += agentic_problems(cfg, section)
    if importlib.util.find_spec("modal") is None:
        problems.append("the 'modal' Python package is not installed")
    return problems
