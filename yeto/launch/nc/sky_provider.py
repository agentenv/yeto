"""SkyPilot lease provider: rollout nodes on clouds sky can launch (Nebius, AWS,
GCP, Kubernetes, ...), behind the same :class:`~yeto.launch.pool.provider.IslandProvider`
contract as National Compute. The pool's policy (eviction handling, fencing,
events, budget, breaker, scale) and the SSH + docker node route are unchanged;
only the supply differs. It also carries the machine management of the spot
watchdog (:mod:`yeto.rl.spot.watchdog`): the candidate rotation, the start
timeout, the own-name guard and the per-offer price.

* ``order(slot)``: submit ``sky.launch`` of a one-node cluster ``<prefix>-<run>-<n>``
  with the offer at the cursor (:mod:`yeto.launch.rollout.offers`: explicit
  ``offers``, the watchdog's Nebius ``candidates``, or one ``resources``) and
  return at once; the cluster name is the ticket and the launch's request id is
  kept in the record. The order is recorded locally (and in the run's resource
  ledger as ``sky_cluster``) *before* the launch. yeto's sky patches
  (:mod:`yeto.sky_patches`, the Nebius spot pricing among them) are installed
  before the first order;
* the launch request is confirmed with ``sky.get`` once sky reports it finished
  (``sky.api_status``; the pool tick never blocks on a launch): a success moves
  the cursor back to the cheapest offer, a failure (no capacity, quota, ...)
  cools that offer down and moves the cursor to the next one, and the ticket
  reads ended (``lost``, the error as ``detail``) so the pool releases it and
  orders again on the next offer (an offer failure is not the pool breaker's
  while another offer is open: ``has_fallback_offer``). The spot watchdog's
  blocking ``launch_cluster(wait=True)`` raises the failure instead;
* idle autodown: a cluster with no sky job (the sampler runs over SSH + docker)
  is launched with ``down=True``, ``wait_for=none`` and
  ``idle_minutes_to_autostop`` = ``hard_cap_minutes`` (default 24 h), so sky's
  idle check (no job = idle) never takes it down while the sampler runs; it
  goes at the hard cap after launch, or when the pool releases it. A task with
  a ``run`` command (``task_yaml``: the sampler is the sky job) keeps
  ``down=True`` alone: the cluster goes when that job ends;
* ``status``: ``sky.status`` of that cluster (``refresh=AUTO``, so a reclaimed
  spot VM is seen): ``UP`` -> active, ``INIT`` (or the launch still running) ->
  waiting (ended ``lost`` past ``start_timeout_s`` from the order: the watchdog's
  "stuck in STARTING" rule), ``STOPPED`` / gone -> ended ``preempted`` (sky
  clusters end that way when the cloud reclaims spot capacity; there is no
  lease end, so ``lease_until`` is None and only eviction-driven replacement
  applies);
* API server restart: a local sky API server starts with an empty request
  table, so every request id the record holds is unknown afterwards. Such a
  launch is reconciled from what sky lists under its recorded cluster name: an
  ``UP`` cluster is kept (no new order), one still ``INIT`` (sky never resumes an
  interrupted request) or none at all reads ended ``lost`` and is replaced;
* ``release``: cancel a launch still running, then ``sky.down`` (only clusters this
  provider launched for this run: the name prefix *and* the local record),
  confirmed gone (``ended`` only once sky no longer lists it);
* ``node``: the head IP and ssh user from the cluster handle;
* ``run_tickets``: open records, plus clusters sky lists that are positively this
  run's (a launch the process did not live to record is adopted, then released).
  Ownership is a mark, not a name: a cluster is adopted only when its name is
  exactly ``<prefix>-<n>`` *and* it carries the sky label ``yeto-run=<owner>``
  (set on every launch) or this run's ledger holds an ``intent`` row for it with
  ``run=<run id>`` (written before the launch). Another run's cluster with a
  matching name (the spot watchdog's ``exact_prefix`` has no run id in it) or one
  opened by hand is never adopted, released or downed: it is reported once;
* ``attach``: an ssh alias to the head (sky's key, no relay), then the
  ``host_setup`` commands of the config;
* ``hourly_cost(ticket)``: the USD price of the offer the cluster was launched with.

"Accounts" are slots ``1..max_nodes``: one cluster per slot at a time.

Requires SkyPilot >= 0.14.3 (``sky.api_status`` / ``sky.api_cancel``). Pool config::

    "provider": {"kind": "sky",
                 "candidates": {"gpus": ["H100", "H200"], "spot_only": true, "infra": "nebius"},
                 "cluster_prefix": "yeto-smp", "ssh_key": "~/.sky/clients/<id>/ssh/sky-key",
                 "start_timeout_s": 1800, "hard_cap_minutes": 1440, "host_setup": ["sudo docker info"]}

(``"factory": "yeto.launch.nc.sky_provider:from_config"`` is the same provider.)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from yeto.launch.rollout.offers import Offer, offers_from_config

from ..pool.provider import ProviderCaps
from ..pool.status import LeaseStatus
from .client import NodeLease
from .tunnel import state_dir

UP, INIT, STOPPED = "UP", "INIT", "STOPPED"
KIND, LEDGER_KIND = "sky", "sky_cluster"
CONFIG_KEYS = frozenset({"kind", "factory", "resources", "price_usd_h", "offers", "candidates", "cluster_prefix",
                         "ssh_key", "host_setup", "start_timeout_s", "task_yaml", "envs", "secrets",
                         "hard_cap_minutes"})
# A cluster with no sky job is autodowned this long after its launch (wait_for=none).
DEFAULT_HARD_CAP_MINUTES = 24 * 60
# sky request states (sky.server.requests.requests.RequestStatus) of a request still in flight
REQUEST_LIVE = frozenset({"PENDING", "WAITING", "RUNNING"})
# launch_state of a record row: the launch request is in flight / confirmed / failed /
# unknown to the API server (restarted) / cancelled by release
SUBMITTED, LAUNCHED, FAILED, LOST, CANCELLED = "submitted", "launched", "failed", "lost", "cancelled"


def _field(obj: Any, key: str) -> Any:
    """``obj[key]`` of a sky response (a dict, or a pydantic payload)."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:300]


def _interrupted(error: BaseException) -> bool:
    """The API server lost the request (restarted): not the launch's own failure."""
    name = type(error).__name__
    text = str(error)
    return name == "RequestInterruptedError" or (
        isinstance(error, RuntimeError) and text.startswith("Failed to get request"))


def _status_name(value: Any) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


OWNER_LABEL = "yeto-run"


def owner_token(run_id: str) -> str:
    """The ``yeto-run`` label value of ``run_id``: the id itself when every cloud's label
    rules take it (GCP's are the strictest), else a hash of it."""
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", run_id):
        return run_id
    return "h-" + hashlib.sha256(run_id.encode()).hexdigest()[:24]


def run_prefix(cluster_prefix: str, run_id: str) -> str:
    """``<cluster_prefix>-<run>``: a run id that is not already lower-case letters, digits
    and '-' gets a hash of the original appended, so two ids that normalize alike
    (``ABC_2`` and ``abc-2``) never share a prefix."""
    slug = re.sub(r"[^a-z0-9-]", "-", run_id.lower())
    if slug != run_id:
        slug += "-" + hashlib.sha256(run_id.encode()).hexdigest()[:8]
    return f"{cluster_prefix}-{slug}"


def _record_ledger(fn: str, *args: Any, **kw: Any) -> None:
    from yeto.cost.ledger import record_safely  # noqa: PLC0415

    record_safely(fn, *args, **kw)


class SkyProvider:
    kind = KIND
    ledger_kind = LEDGER_KIND
    caps = ProviderCaps(lease_bounded=False, notice_s=0.0, currency="usd", cold_start_s=1800)

    def __init__(self, run_id: str, *, slots: list[int], resources: dict[str, Any] | None = None,
                 offers: list[Offer] | None = None, cluster_prefix: str = "yeto-smp", ssh_key: str | None = None,
                 host_setup: list[str] | None = None, start_timeout_s: float = 1800.0,
                 task_yaml: str | None = None, envs: dict[str, str] | None = None,
                 secrets: dict[str, str] | None = None, sky: Any = None, record: Path | None = None,
                 clock=time.time, exact_prefix: str | None = None,
                 hard_cap_minutes: float = DEFAULT_HARD_CAP_MINUTES):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", cluster_prefix):
            raise ValueError("cluster_prefix: lower-case letters, digits and '-'")
        if offers is None:
            offers = offers_from_config({"resources": resources})
        if not offers:
            raise ValueError("sky provider: no offer")
        self.run_id, self.slots, self.offers = run_id, list(slots), list(offers)
        self.resources = dict(self.offers[0].resources)
        # exact_prefix: the spot watchdog's own cluster prefix (its names, its own-name guard)
        self.prefix = exact_prefix or run_prefix(cluster_prefix, run_id)
        self.owner = owner_token(run_id)
        self._name_re = re.compile(re.escape(self.prefix) + r"-(\d+)")
        self.ssh_key = os.path.expanduser(ssh_key) if ssh_key else None
        self.host_setup = list(host_setup or ())
        self.start_timeout_s = float(start_timeout_s)
        if not float(hard_cap_minutes) >= 1:
            raise ValueError("sky provider: hard_cap_minutes must be >= 1")
        self.hard_cap_minutes = int(hard_cap_minutes)
        self._patched = False
        self.task_yaml = os.path.expanduser(task_yaml) if task_yaml else None
        self.envs, self.secrets = dict(envs or {}), dict(secrets or {})
        self._sky, self.clock = sky, clock
        self._real_sky = sky is None  # imported on first use (not a test's fake)
        self.record = record or state_dir() / f"sky-clusters-{self.prefix}.json"

    # -- sky
    @property
    def sky(self):
        if self._sky is None:
            import sky  # noqa: PLC0415

            self._sky = sky
        return self._sky

    def _doc(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.record.read_text())
        except (OSError, ValueError):
            return {"clusters": {}, "cursor": 0, "run": self.run_id}
        if "clusters" not in doc:  # the skeleton's flat {name: row} record
            doc = {"clusters": doc, "cursor": 0}
        if doc.setdefault("run", self.run_id) != self.run_id:
            raise PermissionError(f"sky provider: record {self.record} belongs to run {doc['run']!r}, "
                                  f"not {self.run_id!r}")
        return doc

    def _rows(self) -> dict[str, dict[str, Any]]:
        return self._doc()["clusters"]

    def _save_doc(self, doc: dict[str, Any]) -> None:
        self.record.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.record.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(self.record)

    def _named_ours(self, name: str) -> bool:
        return self._name_re.fullmatch(name) is not None

    def _ours(self, ticket: str) -> bool:
        return self._named_ours(ticket) and ticket in self._rows()

    def _marked_ours(self, record: dict[str, Any]) -> bool:
        """A positive ownership mark on a cluster sky lists: our ``yeto-run`` label on its
        launched resources, or an ``intent`` of this run in the run's resource ledger."""
        labels = record.get("labels") or getattr(getattr(record.get("handle"), "launched_resources", None),
                                                 "labels", None) or {}
        if labels.get(OWNER_LABEL) == self.owner:
            return True
        from yeto.cost.ledger import current  # noqa: PLC0415

        ledger = current()
        if ledger is None:
            return False
        try:
            rows = ledger.records()
        except Exception:  # noqa: BLE001 - an unreadable ledger is no mark
            return False
        name = record.get("name")
        return any(r.get("op") == "intent" and r.get("kind") == LEDGER_KIND and r.get("name") == name
                   and (r.get("attrs") or {}).get("run") == self.run_id for r in rows)

    def _record(self, ticket: str, *, refresh: bool = False) -> dict[str, Any] | None:
        """sky's record of cluster ``ticket`` (None: sky does not list it). ``refresh``:
        sky asks the cloud for clusters with spot instances or autostop (a reclaimed
        spot VM reads gone, not a stale UP)."""
        kw = {}
        modes = getattr(self.sky, "StatusRefreshMode", None)
        if refresh and modes is not None:
            kw["refresh"] = modes.AUTO
        records = self.sky.get(self.sky.status([ticket], **kw))
        return records[0] if records else None

    def _wait_for_none(self):
        modes = getattr(self.sky, "AutostopWaitFor", None)
        if modes is None:
            from sky.skylet.autostop_lib import AutostopWaitFor as modes  # noqa: PLC0415,N813
        return modes.NONE

    def launch_kwargs(self, task) -> dict[str, Any]:
        """``sky.launch`` keywords. The sampler is no sky job (SSH + docker): sky's idle
        check would take the cluster down about a minute after the launch, so the idle
        timer waits for nothing and runs to the hard cap. A task with its own ``run``
        is the sampler as a sky job: down when it ends."""
        if getattr(task, "run", None):
            return {"down": True}
        return {"down": True, "idle_minutes_to_autostop": self.hard_cap_minutes,
                "wait_for": self._wait_for_none()}

    def _install_patches(self) -> None:
        """yeto's sky patches before the first order (the Nebius spot pricing among
        them), not only when ``yeto launch`` ran first. SkyPilot 0.14.3 sets the
        Nebius pricing model itself; on 0.13.0 the API server needs the hook too."""
        if self._patched:
            return
        from yeto import sky_patches  # noqa: PLC0415
        from yeto.sky_patches import nebius as nebius_patch  # noqa: PLC0415

        if self._real_sky:  # its API server must run this Python's sky version
            sky_patches.require_matching_api_server()
        sky_patches.install()
        self._patched = True
        version = getattr(self.sky, "__version__", None)
        wants_nebius = any("nebius" in str(o.resources.get("infra") or o.resources.get("cloud") or "").lower()
                           and o.resources.get("use_spot") for o in self.offers)
        if wants_nebius and version in nebius_patch.SERVER_HOOK_SKY_VERSIONS:
            from yeto.launch.verda import remote_sky_api_server  # noqa: PLC0415
            from yeto.launcher import REPO_ROOT  # noqa: PLC0415

            if remote_sky_api_server():
                print(f"[sky-provider] WARNING: sky {version} on a remote API server: the Nebius spot pricing "
                      "patch is not in effect there", flush=True)
                return
            path, fresh = sky_patches.ensure_local_pth(str(REPO_ROOT))
            if fresh:
                print(f"[sky-provider] WARNING: sky patch hook {path} was only now installed; restart the sky API "
                      "server (sky api stop) before Nebius spot orders", flush=True)

    def _task(self, name: str, offer: Offer, num_nodes: int = 1):
        sky = self.sky
        task = sky.Task.from_yaml(self.task_yaml) if self.task_yaml else sky.Task(name=name, run=None)
        if num_nodes > 1:  # a group order (order_group): one cluster of num_nodes nodes
            task.num_nodes = int(num_nodes)
        if self.envs:
            task.update_envs(dict(self.envs))
        if self.secrets:
            task.update_secrets({k: os.environ.get(v, "") for k, v in self.secrets.items()})
        resources = dict(offer.resources)
        if resources.get("infra") or resources.get("cloud"):
            # sky keeps the labels in the cluster handle on every cloud (AWS / GCP / Kubernetes
            # also tag the instances). Without a cloud, sky validates labels against every
            # enabled cloud (a client-side cloud check): the ledger intent is the mark then.
            resources["labels"] = {**dict(resources.get("labels") or {}), OWNER_LABEL: self.owner}
        task.set_resources(sky.Resources(**resources))
        return task

    @property
    def cursor(self) -> int:
        return int(self._doc().get("cursor", 0)) % len(self.offers)

    # -- IslandProvider
    def order(self, account: int) -> str:
        doc = self._doc()
        n = 1 + max([int(r.get("n", 0)) for r in doc["clusters"].values()] or [0])
        offer = self._pick(doc)
        return self.launch_cluster(f"{self.prefix}-{n}", offer, slot=account, n=n, wait=False)

    # -- the per-offer breaker: an order is a (slot, offer); a failing offer cools down
    # alone while the others keep ordering (the pool's breaker counts only when none is left)
    OFFER_COOLDOWN_S = 600.0

    def _cooling(self, doc: dict[str, Any], key: str, now: float) -> bool:
        return float((doc.get("offers") or {}).get(key, {}).get("open_until", 0)) > now

    def _pick(self, doc: dict[str, Any]) -> Offer:
        """From the cursor on, the first offer not cooling down; all cooling: the one that reopens first."""
        now, start = self.clock(), int(doc.get("cursor", 0)) % len(self.offers)
        ring = self.offers[start:] + self.offers[:start]
        for offer in ring:
            if not self._cooling(doc, offer.key, now):
                return offer
        return min(ring, key=lambda o: float(doc["offers"][o.key]["open_until"]))

    def _offer_failed(self, doc: dict[str, Any], key: str) -> None:
        row = doc.setdefault("offers", {}).setdefault(key, {"fails": 0})
        row["fails"] = int(row.get("fails", 0)) + 1
        row["open_until"] = self.clock() + min(self.OFFER_COOLDOWN_S * 2 ** (row["fails"] - 1),
                                               8 * self.OFFER_COOLDOWN_S)

    def has_fallback_offer(self) -> bool:
        """True while some offer is not cooling down: an order failure is that offer's, not the pool's."""
        doc, now = self._doc(), self.clock()
        return len(self.offers) > 1 and any(not self._cooling(doc, o.key, now) for o in self.offers)

    def offer_of(self, ticket: str) -> str | None:
        """The offer key cluster ``ticket`` was ordered with (None: not ours)."""
        return ((self._rows().get(ticket) or {}).get("offer") or {}).get("key")

    def offer_breakers(self) -> dict[str, dict[str, Any]]:
        return dict(self._doc().get("offers") or {})

    def launch_cluster(self, name: str, offer: Offer, *, slot: int, n: int = 0, wait: bool = True,
                       num_nodes: int = 1) -> str:
        """Record, then launch cluster ``name`` (``<prefix>-...``) with ``offer``.
        ``wait``: block on the launch request (the spot watchdog); a failure moves
        the cursor past the offer and is raised. ``wait=False`` (the pool's ``order``):
        return once submitted; :meth:`status` confirms the request."""
        if not self._named_ours(name):
            raise PermissionError(f"cluster {name} is not named {self.prefix}-<n>")
        self._install_patches()
        doc = self._doc()
        doc["clusters"][name] = {"slot": slot, "n": n, "open": True, "ordered_at": self.clock(),
                                 "offer": offer.record(), "launch_state": SUBMITTED,
                                 **({"num_nodes": num_nodes} if num_nodes > 1 else {})}
        self._save_doc(doc)  # recorded before the launch: a crash mid-launch still leaves it ours to down
        _record_ledger("intent", LEDGER_KIND, name, offer=offer.key, price_usd_h=offer.price_usd_h, run=self.run_id)
        try:
            task = self._task(name, offer, num_nodes)
            request = self.sky.launch(task, cluster_name=name, **self.launch_kwargs(task))
        except Exception as error:  # the submit itself failed (bad resources, API server down)
            self._launch_failed(name, error)
            raise
        doc = self._doc()
        doc["clusters"][name]["launch_request"] = str(request)
        self._save_doc(doc)
        if not wait:
            return name
        try:
            self.sky.get(request)
        except Exception as error:
            self._launch_failed(name, error)
            raise
        self._launch_ok(name)
        return name

    # -- group order (yeto.launch.pool.group): one multi-node cluster; sky's best network
    # tier when the group asks for an RDMA fabric
    fabrics = ("tcp", "rdma")

    def order_group(self, n: int, shape: Any, fabric: str = "tcp") -> list[tuple[int, str]]:
        """Place ``n`` nodes as one cluster ``<prefix>-<k>`` on a free slot: one ticket, from
        the first offer (cursor order, cooling offers last) whose accelerators are the
        shape's. An RDMA fabric takes the cloud's best network tier for that shape
        (``topology.multinode_network_tier``; a shape the cloud has no such tier for is
        refused). A launch that fails releases what it left (``group.release_placed``)."""
        from ..pool.group import release_placed
        from ..topology import multinode_network_tier

        free = [s for s in self.slots if self.account_free(s)]
        if not free:
            raise RuntimeError(f"sky provider: no free slot for a group of {n}")
        doc = self._doc()
        offer = self._group_offer(doc, shape)
        if fabric != "tcp" and int(n) > 1:
            cloud = str(offer.resources.get("cloud") or str(offer.resources.get("infra") or "").partition("/")[0])
            tier = multinode_network_tier(cloud.lower(), str(shape.gpu), int(shape.gpus_per_node))
            if tier is None:
                raise ValueError(f"sky provider: {cloud} has no RDMA network tier for "
                                 f"{shape.gpus_per_node} x {shape.gpu} nodes; nothing ordered")
            offer = Offer(offer.key, {**offer.resources, "network_tier": tier}, offer.price_usd_h)
        k = 1 + max([int(r.get("n", 0)) for r in doc["clusters"].values()] or [0])
        name = f"{self.prefix}-{k}"
        try:
            self.launch_cluster(name, offer, slot=free[0], n=k, wait=False, num_nodes=int(n))
        except Exception as error:
            left = [(free[0], name)] if (self._rows().get(name) or {}).get("open") else []
            raise release_placed(self, left, error) from error
        return [(free[0], name)]

    def _group_offer(self, doc: dict[str, Any], shape: Any) -> Offer:
        """The offer a group of ``shape`` nodes orders: cursor order, an offer cooling down
        only when every matching one is."""
        want = (str(shape.gpu).upper(), int(shape.gpus_per_node))

        def accelerators(offer: Offer) -> tuple[str, int] | None:
            acc = offer.resources.get("accelerators")
            if not isinstance(acc, str) or not acc:
                return None
            gpu, _, count = acc.partition(":")
            return gpu.upper(), int(count or 1)

        start = int(doc.get("cursor", 0)) % len(self.offers)
        ring = [o for o in self.offers[start:] + self.offers[:start] if accelerators(o) == want]
        if not ring:
            raise ValueError(f"sky provider: no offer of {want[1]} x {want[0]} nodes "
                             f"(offers: {[o.resources.get('accelerators') for o in self.offers]}); nothing ordered")
        now = self.clock()
        open_ = [o for o in ring if not self._cooling(doc, o.key, now)]
        return open_[0] if open_ else min(ring, key=lambda o: float(doc["offers"][o.key]["open_until"]))

    def group_failed(self, ticket: str) -> None:
        """A group on cluster ``ticket`` did not come up: its offer cools down (once; a start
        timeout already counted it), so the next group order takes the next offer."""
        doc = self._doc()
        row = doc["clusters"].get(ticket) or {}
        key = (row.get("offer") or {}).get("key")
        if key and not row.get("start_timed_out") and not row.get("launch_failed") and not row.get("group_failed"):
            row["group_failed"] = True
            self._offer_failed(doc, key)
            doc["cursor"] = (next((i for i, o in enumerate(self.offers) if o.key == key), 0) + 1) % len(self.offers)
            self._save_doc(doc)

    def _launch_ok(self, name: str) -> None:
        doc = self._doc()
        row = doc["clusters"][name]
        if row.get("launch_state") == LAUNCHED:
            return
        row["launch_state"] = LAUNCHED
        key = (row.get("offer") or {}).get("key")
        doc["cursor"] = 0  # back to the cheapest offer for the next order
        (doc.get("offers") or {}).pop(key, None)
        self._save_doc(doc)
        offer = row.get("offer") or {}
        _record_ledger("created", LEDGER_KIND, name, name, offer=key, price_usd_h=offer.get("price_usd_h"),
                       run=self.run_id, slot=row.get("slot"), pool=True,
                       **({"num_nodes": row["num_nodes"]} if row.get("num_nodes") else {}))

    def _launch_failed(self, name: str, error: BaseException) -> None:
        """The launch failed: the next order tries the next offer; a cluster it left
        stays open for the pool to release, none frees the slot at once."""
        try:  # a launch that left no cluster bills nothing: the slot is free again
            left = self._record(name) is not None
        except Exception:  # noqa: BLE001 - unverified: keep it open for the pool to release
            left = True
        doc = self._doc()
        row = doc["clusters"][name]
        key = (row.get("offer") or {}).get("key")
        cursor = next((i for i, o in enumerate(self.offers) if o.key == key), 0)
        doc["cursor"] = (cursor + 1) % len(self.offers)
        if key:
            self._offer_failed(doc, key)
        row.update(launch_failed=True, launch_state=FAILED, launch_error=_error_text(error), open=left)
        self._save_doc(doc)

    def _request_live(self, request: str) -> bool | None:
        """True: in flight; False: finished (``sky.get`` returns at once); None: the
        API server does not know it (restarted: its request table starts empty)."""
        rows = self.sky.api_status([request])
        if not rows:
            return None
        return _status_name(_field(rows[0], "status")) in REQUEST_LIVE

    def _settle(self, ticket: str) -> dict[str, Any]:
        """Confirm the launch request of ``ticket`` once sky reports it finished; the row."""
        row = self._rows().get(ticket) or {}
        request = row.get("launch_request")
        if row.get("launch_state") != SUBMITTED or not request:
            return row
        live = self._request_live(request)  # raises on a failed call: a control-plane blip
        if live:
            return row
        if live is None:
            return self._mark(ticket, launch_state=LOST)
        try:
            self.sky.get(request)
        except Exception as error:  # noqa: BLE001 - the launch's own failure, or a lost request
            if _interrupted(error):
                return self._mark(ticket, launch_state=LOST)
            self._launch_failed(ticket, error)
            return self._rows().get(ticket) or {}
        self._launch_ok(ticket)
        return self._rows().get(ticket) or {}

    def _mark(self, ticket: str, **fields: Any) -> dict[str, Any]:
        doc = self._doc()
        doc["clusters"][ticket].update(fields)
        self._save_doc(doc)
        return doc["clusters"][ticket]

    def _start_timed_out(self, ticket: str) -> bool:
        """Past ``start_timeout_s`` from the order (counted once against its offer)."""
        ordered = (self._rows().get(ticket) or {}).get("ordered_at")
        if ordered is None or self.clock() - float(ordered) <= self.start_timeout_s:
            return False
        doc = self._doc()
        row = doc["clusters"].get(ticket) or {}
        if not row.get("start_timed_out"):
            row["start_timed_out"] = True
            key = (row.get("offer") or {}).get("key")
            if key:
                self._offer_failed(doc, key)
            self._save_doc(doc)
        return True

    def status(self, account: int, ticket: str) -> LeaseStatus:
        row = self._settle(ticket)  # raises on a failed call: a control-plane blip to the pool
        if row.get("launch_failed"):
            return LeaseStatus("ended", end_reason="lost", detail=row.get("launch_error"))
        state_of_launch = row.get("launch_state")
        record = self._record(ticket, refresh=True)
        if record is None:
            if state_of_launch == SUBMITTED:  # sky has not created the cluster yet
                if self._start_timed_out(ticket):
                    return LeaseStatus("ended", end_reason="lost", detail="launch still running past start_timeout_s")
                return LeaseStatus("waiting")
            if state_of_launch == LOST:
                return LeaseStatus("ended", end_reason="lost",
                                   detail="launch request lost by the sky API server before the cluster existed")
            return LeaseStatus("ended", end_reason="preempted")
        state = _status_name(record.get("status"))
        if state == UP:
            if state_of_launch == LOST:  # it came up before the API server restart: kept
                self._mark(ticket, launch_state=LAUNCHED, reconciled=True)
            return LeaseStatus("active")
        if state == INIT:
            if state_of_launch == LOST:  # sky never resumes an interrupted launch
                return LeaseStatus("ended", end_reason="lost",
                                   detail="launch request lost by the sky API server while the cluster was INIT")
            if self._start_timed_out(ticket):
                return LeaseStatus("ended", end_reason="lost")  # stuck starting: released, next offer
            return LeaseStatus("waiting")
        return LeaseStatus("ended", end_reason="preempted" if state == STOPPED else state.lower())

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        if not self._ours(ticket):
            raise PermissionError(f"cluster {ticket} was not launched by this provider for run {self.run_id}")
        row = self._rows()[ticket]
        request = row.get("launch_request")
        if row.get("launch_state") == SUBMITTED and request:
            self._cancel_launch(ticket, request)
        if self._record(ticket) is not None:
            try:
                self.sky.get(self.sky.down(ticket))
            except Exception as error:  # noqa: BLE001 - gone meanwhile is fine; else checked below
                if type(error).__name__ != "ClusterDoesNotExist":
                    raise
        if self._record(ticket) is not None:
            raise RuntimeError(f"cluster {ticket}: down sent but it still exists")
        doc = self._doc()
        doc["clusters"][ticket]["open"] = False
        self._save_doc(doc)
        _record_ledger("released", LEDGER_KIND, ticket, ticket)
        return {"ticket": ticket, "state": "ended"}

    def _cancel_launch(self, ticket: str, request: str) -> None:
        """A launch still in flight is cancelled before the down (else it may create the
        cluster after the down); one that cannot be cancelled fails the release (retried)."""
        try:
            self.sky.get(self.sky.api_cancel([request]))
        except Exception as error:  # noqa: BLE001 - finished meanwhile is fine
            if self._request_live(request):
                raise RuntimeError(f"cluster {ticket}: cannot cancel its launch request ({_error_text(error)})") \
                    from error
        self._mark(ticket, launch_state=CANCELLED)

    def node(self, account: int, ticket: str) -> NodeLease:
        record = self._record(ticket)
        handle = (record or {}).get("handle")
        ip = getattr(handle, "head_ip", None)
        if not ip:
            raise RuntimeError(f"cluster {ticket} has no head IP yet")
        user = getattr(handle, "ssh_user", None) or "ubuntu"
        return NodeLease(account, ticket, ip, user, "", None, None, None, None)

    def balance(self) -> int | None:
        return None  # clouds bill outside; the run's spend cap (Budget) is the guard

    def account_free(self, account: int) -> bool:
        return not any(r.get("open") and r.get("slot") == account for r in self._rows().values())

    def run_tickets(self) -> list[tuple[int, str]]:
        """Open records, plus clusters sky lists that the record lacks and that are
        positively this run's (named ``<prefix>-<n>`` and marked, see the module notes):
        recorded now, on a free slot, so the pool can adopt or release them. A cluster
        with a matching name but no mark is reported once and left alone."""
        doc = self._doc()
        rows = doc["clusters"]
        try:
            listed = [r for r in (self.sky.get(self.sky.status(None)) or ()) if isinstance(r, dict)]
        except Exception:  # noqa: BLE001 - an unanswered listing: the record alone
            listed = []
        changed = False
        for rec in listed:
            name = str(rec.get("name"))
            match = self._name_re.fullmatch(name)
            if match is None or name in rows:
                continue
            if not self._marked_ours(rec):
                seen = doc.setdefault("unowned", [])
                if name not in seen:
                    seen.append(name)
                    changed = True
                    print(f"[sky-provider] cluster {name} has this run's name pattern but no ownership mark "
                          f"({OWNER_LABEL}={self.owner} label or ledger intent): not adopted, left alone",
                          flush=True)
                continue
            busy = {r.get("slot") for r in rows.values() if r.get("open")}
            slot = next((s for s in self.slots if s not in busy), self.slots[0])
            rows[name] = {"slot": slot, "n": int(match.group(1)), "open": True,
                          "ordered_at": self.clock(), "adopted": True}
            changed = True
        if changed:
            self._save_doc(doc)
        return [(int(r["slot"]), name) for name, r in rows.items() if r.get("open")]

    # -- pricing (the pool's spend in USD; see the rollout controller notes)
    def hourly_cost(self, ticket: str | None = None) -> float | None:
        row = (self._rows().get(ticket) or {}) if ticket else {}
        price = (row.get("offer") or {}).get("price_usd_h", self.offers[self.cursor].price_usd_h)
        return None if price is None else float(price) * int(row.get("num_nodes", 1))

    # -- node route hook
    def attach(self, lease: NodeLease, *, ssh_run) -> str:
        """An ssh alias to the head (direct, sky's key), then ``host_setup``."""
        from . import node_pool  # noqa: PLC0415

        alias = node_pool.host_alias(lease.ticket)
        ours = state_dir() / "ssh_config"
        lines = [f"Host {alias}", f"  HostName {lease.ip}", f"  User {lease.user}",
                 "  StrictHostKeyChecking accept-new", f"  UserKnownHostsFile {state_dir() / 'known_hosts'}",
                 "  ServerAliveInterval 15", "  ServerAliveCountMax 4"]
        if self.ssh_key:
            lines += [f"  IdentityFile {self.ssh_key}", "  IdentitiesOnly yes"]
        block = "\n".join(lines) + "\n"
        node_pool.edit_ssh_config(ours, lambda blocks: blocks.__setitem__(alias, block))
        for command in self.host_setup:
            out = ssh_run(alias, command)
            if out.returncode != 0:
                raise RuntimeError(f"host setup {command[:40]!r} on {lease.ticket}: exit {out.returncode}")
        return alias


def registry_factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]) -> SkyProvider:  # noqa: ARG001
    """The registered ``sky`` factory (``yeto.launch.pool.provider``)."""
    return from_config(cfg, run_id)


def from_config(cfg: dict[str, Any], run_id: str, *, sky: Any = None) -> SkyProvider:
    section = dict(cfg.get("provider") or {})
    unknown = set(section) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"sky provider: unknown keys {sorted(unknown)}")
    offers = offers_from_config(section)
    for key in ("kind", "factory", "resources", "price_usd_h", "offers", "candidates"):
        section.pop(key, None)
    from . import cli  # noqa: PLC0415

    return SkyProvider(run_id, slots=cli.slot_list(cfg), offers=offers, sky=sky, **section)


def offline_problems(cfg: dict[str, Any]) -> list[str]:
    """``pool --dry-run`` checks of a sky pool (no sky call)."""
    import importlib.util  # noqa: PLC0415

    try:
        provider = from_config(cfg, str(cfg.get("run") or cfg.get("shard_base")), sky=object())
    except (ValueError, TypeError, KeyError) as error:
        return [str(error)]
    problems = []
    if importlib.util.find_spec("sky") is None:
        problems.append("the 'skypilot' Python package (module sky) is not installed")
    if provider.task_yaml and not Path(provider.task_yaml).exists():
        problems.append(f"sky provider: task_yaml {provider.task_yaml!r} does not exist")
    if provider.ssh_key and not Path(provider.ssh_key).exists():
        problems.append(f"sky provider: ssh_key {provider.ssh_key!r} does not exist")
    for env_name in provider.secrets.values():
        if not os.environ.get(env_name):
            problems.append(f"sky provider: secret variable {env_name} is not set")
    return problems
