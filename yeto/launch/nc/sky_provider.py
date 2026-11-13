"""SkyPilot lease provider: rollout nodes on clouds sky can launch (Nebius, AWS,
GCP, Kubernetes, ...), behind the same :class:`~yeto.launch.pool.provider.IslandProvider`
contract as National Compute. The pool's policy (eviction handling, fencing,
events, budget, breaker, scale) and the SSH + docker node route are unchanged;
only the supply differs. It also carries the machine management of the spot
watchdog (:mod:`yeto.rl.spot.watchdog`): the candidate rotation, the start
timeout, the own-name guard and the per-offer price.

* ``order(slot)``: ``sky.launch`` a one-node cluster ``<prefix>-<run>-<n>``
  with the offer at the cursor (:mod:`yeto.launch.rollout.offers`: explicit
  ``offers``, the watchdog's Nebius ``candidates``, or one ``resources``),
  running no job (or the ``task_yaml`` task), detached; the cluster name is the
  ticket (a failed launch that left no cluster frees the slot at once; one that
  left a cluster stays open for the pool to release). The order is recorded locally (and in the run's resource ledger as
  ``sky_cluster``) *before* the launch. A launch that raises moves the cursor to
  the next offer and re-raises (the pool's breaker paces the retries); a
  launch that succeeds moves it back to the cheapest offer;
* ``status``: ``sky.status`` of that cluster: ``UP`` -> active, ``INIT`` ->
  waiting (ended ``lost`` past ``start_timeout_s`` from the order: the watchdog's
  "stuck in STARTING" rule), ``STOPPED`` / gone -> ended ``preempted`` (sky
  clusters end that way when the cloud reclaims spot capacity; there is no
  lease end, so ``lease_until`` is None and only eviction-driven replacement
  applies);
* ``release``: ``sky.down`` (only clusters this provider launched for this run:
  the name prefix *and* the local record), confirmed gone;
* ``node``: the head IP and ssh user from the cluster handle;
* ``run_tickets``: open records, plus clusters sky lists with this run's prefix
  (a launch the process did not live to record is adopted, then released);
* ``attach``: an ssh alias to the head (sky's key, no relay), then the
  ``host_setup`` commands of the config;
* ``hourly_cost(ticket)``: the USD price of the offer the cluster was launched with.

"Accounts" are slots ``1..max_nodes``: one cluster per slot at a time.

Status: CPU tests drive it with a fake ``sky`` module; it has not launched a
real cluster. Pool config::

    "provider": {"kind": "sky",
                 "candidates": {"gpus": ["H100", "H200"], "spot_only": true, "infra": "nebius"},
                 "cluster_prefix": "yeto-smp", "ssh_key": "~/.sky/clients/<id>/ssh/sky-key",
                 "start_timeout_s": 1800, "host_setup": ["sudo docker info"]}

(``"factory": "yeto.launch.nc.sky_provider:from_config"`` is the same provider.)
"""

from __future__ import annotations

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
                         "ssh_key", "host_setup", "start_timeout_s", "task_yaml", "envs", "secrets"})


def _status_name(value: Any) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


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
                 clock=time.time, exact_prefix: str | None = None):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", cluster_prefix):
            raise ValueError("cluster_prefix: lower-case letters, digits and '-'")
        if offers is None:
            offers = offers_from_config({"resources": resources})
        if not offers:
            raise ValueError("sky provider: no offer")
        self.run_id, self.slots, self.offers = run_id, list(slots), list(offers)
        self.resources = dict(self.offers[0].resources)
        # exact_prefix: the spot watchdog's own cluster prefix (its names, its own-name guard)
        self.prefix = exact_prefix or f"{cluster_prefix}-{re.sub(r'[^a-z0-9-]', '-', run_id.lower())}"
        self.ssh_key = os.path.expanduser(ssh_key) if ssh_key else None
        self.host_setup = list(host_setup or ())
        self.start_timeout_s = float(start_timeout_s)
        self.task_yaml = os.path.expanduser(task_yaml) if task_yaml else None
        self.envs, self.secrets = dict(envs or {}), dict(secrets or {})
        self._sky, self.clock = sky, clock
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
            return {"clusters": {}, "cursor": 0}
        if "clusters" not in doc:  # the skeleton's flat {name: row} record
            doc = {"clusters": doc, "cursor": 0}
        return doc

    def _rows(self) -> dict[str, dict[str, Any]]:
        return self._doc()["clusters"]

    def _save_doc(self, doc: dict[str, Any]) -> None:
        self.record.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.record.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(self.record)

    def _ours(self, ticket: str) -> bool:
        return ticket.startswith(self.prefix + "-") and ticket in self._rows()

    def _record(self, ticket: str) -> dict[str, Any] | None:
        records = self.sky.get(self.sky.status([ticket]))
        return records[0] if records else None

    def _task(self, name: str, offer: Offer):
        sky = self.sky
        task = sky.Task.from_yaml(self.task_yaml) if self.task_yaml else sky.Task(name=name, run=None)
        if self.envs:
            task.update_envs(dict(self.envs))
        if self.secrets:
            task.update_secrets({k: os.environ.get(v, "") for k, v in self.secrets.items()})
        task.set_resources(sky.Resources(**offer.resources))
        return task

    @property
    def cursor(self) -> int:
        return int(self._doc().get("cursor", 0)) % len(self.offers)

    # -- IslandProvider
    def order(self, account: int) -> str:
        doc = self._doc()
        n = 1 + max([int(r.get("n", 0)) for r in doc["clusters"].values()] or [0])
        offer = self._pick(doc)
        return self.launch_cluster(f"{self.prefix}-{n}", offer, slot=account, n=n)

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

    def launch_cluster(self, name: str, offer: Offer, *, slot: int, n: int = 0) -> str:
        """Record, then launch cluster ``name`` (``<prefix>-...``) with ``offer``. A raise
        moves the cursor past the offer (the next order tries the next one); a success
        moves it back to the cheapest."""
        if not name.startswith(self.prefix + "-"):
            raise PermissionError(f"cluster {name} does not carry this provider's prefix {self.prefix}-")
        doc = self._doc()
        cursor = next((i for i, o in enumerate(self.offers) if o.key == offer.key), 0)
        doc["clusters"][name] = {"slot": slot, "n": n, "open": True, "ordered_at": self.clock(),
                                 "offer": offer.record()}
        self._save_doc(doc)  # recorded before the launch: a crash mid-launch still leaves it ours to down
        _record_ledger("intent", LEDGER_KIND, name, offer=offer.key, price_usd_h=offer.price_usd_h)
        try:
            self.sky.launch(self._task(name, offer), cluster_name=name, detach_run=True, down=True)
        except Exception:
            try:  # a launch that left no cluster bills nothing: the slot is free again
                left = self._record(name) is not None
            except Exception:  # noqa: BLE001 - unverified: keep it open for the pool to release
                left = True
            doc = self._doc()
            doc["cursor"] = (cursor + 1) % len(self.offers)  # the next order tries the next offer
            self._offer_failed(doc, offer.key)
            doc["clusters"][name].update(launch_failed=True, open=left)
            self._save_doc(doc)
            raise
        doc = self._doc()
        doc["cursor"] = 0  # back to the cheapest offer for the next order
        (doc.get("offers") or {}).pop(offer.key, None)
        self._save_doc(doc)
        _record_ledger("created", LEDGER_KIND, name, name, offer=offer.key, price_usd_h=offer.price_usd_h,
                       run=self.run_id, slot=slot, pool=True)
        return name

    def status(self, account: int, ticket: str) -> LeaseStatus:
        record = self._record(ticket)  # raises on a failed call: a control-plane blip to the pool
        if record is None:
            return LeaseStatus("ended", end_reason="preempted")
        state = _status_name(record.get("status"))
        if state == UP:
            return LeaseStatus("active")
        if state == INIT:
            ordered = (self._rows().get(ticket) or {}).get("ordered_at")
            if ordered is not None and self.clock() - float(ordered) > self.start_timeout_s:
                doc = self._doc()
                row = doc["clusters"].get(ticket) or {}
                if not row.get("start_timed_out"):  # counted once against its offer
                    row["start_timed_out"] = True
                    key = (row.get("offer") or {}).get("key")
                    if key:
                        self._offer_failed(doc, key)
                    self._save_doc(doc)
                return LeaseStatus("ended", end_reason="lost")  # stuck starting: released, next offer
            return LeaseStatus("waiting")
        return LeaseStatus("ended", end_reason="preempted" if state == STOPPED else state.lower())

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        if not self._ours(ticket):
            raise PermissionError(f"cluster {ticket} was not launched by this provider for run {self.run_id}")
        if self._record(ticket) is not None:
            self.sky.get(self.sky.down(ticket))
        if self._record(ticket) is not None:
            raise RuntimeError(f"cluster {ticket}: down sent but it still exists")
        doc = self._doc()
        doc["clusters"][ticket]["open"] = False
        self._save_doc(doc)
        _record_ledger("released", LEDGER_KIND, ticket, ticket)
        return {"ticket": ticket, "state": "ended"}

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
        """Open records, plus clusters sky lists under this run's prefix that the
        record lacks (recorded now, on a free slot, so the pool can adopt or release them)."""
        doc = self._doc()
        rows = doc["clusters"]
        try:
            listed = [str(r.get("name")) for r in (self.sky.get(self.sky.status(None)) or ())]
        except Exception:  # noqa: BLE001 - an unanswered listing: the record alone
            listed = []
        added = False
        for name in listed:
            if name.startswith(self.prefix + "-") and name not in rows:
                busy = {r.get("slot") for r in rows.values() if r.get("open")}
                slot = next((s for s in self.slots if s not in busy), self.slots[0])
                tail = name[len(self.prefix) + 1:]
                rows[name] = {"slot": slot, "n": int(tail) if tail.isdigit() else 0, "open": True,
                              "ordered_at": self.clock(), "adopted": True}
                added = True
        if added:
            self._save_doc(doc)
        return [(int(r["slot"]), name) for name, r in rows.items() if r.get("open")]

    # -- pricing (the pool's spend in USD; see the rollout controller notes)
    def hourly_cost(self, ticket: str | None = None) -> float | None:
        offer = (self._rows().get(ticket) or {}).get("offer") if ticket else None
        price = (offer or {}).get("price_usd_h", self.offers[self.cursor].price_usd_h)
        return None if price is None else float(price)

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
