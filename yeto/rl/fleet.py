"""An island fleet in one run config: several async GRPO islands, one syncer.

A run config with an ``islands`` section declares one trainer per island and the
sampler pools that feed it, and a ``fleet`` section with what the trainers share::

    {"run": "<run>", "store": "s3://<bucket>/runs/<run>", "budget": {...}, ...,
     "fleet": {"trainer_argv": ["--training-mode", "rl", "--tuning", "full",
                                "--rl-sync-preset", "decoupled", ...,
                                "--rl-checkpoint-store", "s3://<bucket>/runs/<run>"],
               "dispense": "fleet",               # "fleet" (default) | "island"
               "sandbox_usd_per_h": 3.0},         # optional: the agentic pools' sandboxes in the spend
     "islands": [
       {"name": "east", "trainer": {"gpu": "modal:2x8xb300", "usd_per_h": 96.0,
                                    "argv": ["--rl-release-every", "2"]},
        "samplers": [<pool entries, the format of the run config's samplers list>]},
       ...]}

``islands`` excludes ``trainer``, ``pool`` and the top-level ``samplers``. Island k
(its position in the list) is the learner id of its trainer and its store
sub-directory ``trainer<k>``: its trainer releases its policy there and reads its
samples there, and its pools write there.

:func:`compile_fleet` turns the run config into the single-trainer form the rest of
``yeto run up`` drives (``trainer`` is the one synced-group ``yeto launch`` of every
island's trainer and the syncer; ``samplers`` lists every island's pools) plus a
:class:`FleetPlan` (the island table). The compiled config carries the plan under
:data:`PLAN_KEY`. For pool ``<name>`` of island k the compile:

* renames it ``i<k>-<name>`` (a pool config can serve several islands);
* moves its store to ``<store>/trainer<k>``: its ``run_prefix`` gets the segment
  ``trainer<k>`` (the samplers write ``s3://<bucket>/runs/<run_prefix>``, and a
  spec's ``__RUN_PREFIX__`` follows);
* renames its shards: ``shard_base`` becomes ``<shard_base>-i<k>``, so the shards
  are ``<shard_base>-i<k>-<n>`` for n below the pool's ``shard_limit`` (required);
* sets the pool's sampler knobs ``island`` (k), ``until_trainers_end`` (1: the
  sampler stops after its own trainer ended) and, with ``dispense: fleet``,
  ``dispense_store`` (``<store>/fleet/dispense``: one task ledger for the fleet);
  for an agentic pool whose spec has ``agentic.envs`` (a sandbox gate) also
  ``sandbox_store`` (the run store): ``yeto sandbox prepare`` writes the readiness
  manifest once in the run store, and the pool's own store is ``trainer<k>``;
* gives pools that share one records relay or tunnel port range (the same
  ``agentic.setup.records_relay`` host and ``listen_port`` range, or the same
  ``agentic.setup.tunnel.listen`` range) consecutive blocks of it: the knob
  ``shard_offset`` (spec key of the same name) is added to the shard index, so
  ``<base>-i0-0`` and ``<base>-i1-0`` do not listen on one port (:func:`share_offsets`).

The trainer argv is ``fleet.trainer_argv`` with ``--gpu`` (every island's cluster,
in island order), ``--rl-sample-role trainer --rl-sampler-follow trainer``, one
``--rl-island-buffer-shards k:<shards>`` per island (the shards of its pools) and
the island's ``--rl-island-release-every`` / ``--rl-island-max-policy-age`` /
``--rl-island-sample-wait-s``. The run config's ``breakers.trainer.*`` become their
trainer knobs (``--rl-sample-wait-s``, ``--rl-max-policy-age``, ``--island-env K=V``: every
trainer reads those, not the section); an island's own ``--rl-sample-wait-s`` /
``--rl-max-policy-age`` overrides the section's for that island. Its ``usd_per_h`` is
the sum of the islands'.

Miles sampler islands in a fleet island. An island's ``samplers`` may also list
``{"kind": "miles", "name": ..., "gpu": "modal:1x1xl40s", "usd_per_h": ...}``
entries (:data:`MILES_KEYS`; ``usd_per_h`` optional): a Miles sampler island of the
same launch, which writes the island's ``trainer<k>/`` and follows its trainer
(``sample_roles`` module doc, mixed groups). The launch's islands are the trainers
first (launch island k = learner id k = fleet island k, as without Miles samplers)
and then the Miles samplers in fleet-island and list order (launch island
``K + j``). Each adds its cluster to ``--gpu``, ``--rl-island-sample-role
<K + j>:sampler`` and ``--rl-island-sampler-follows <K + j>:<k>``; it writes the
buffer shard ``island<K + j>``, which its trainer reads with its pools' shards.
An island needs one sampler of either kind; one without pools gets no
``--rl-island-buffer-shards`` entry. A Miles sampler's ``usd_per_h`` counts to its
island (the island's ``usd_per_h`` in the plan and the launch's).

The task ledger of Miles samplers (:func:`miles_ledger`). When the pools claim from
a store ledger, every Miles sampler claims its rows from the same ledger instead of
a data slot of ``--data``: with ``dispense: fleet`` the fleet's ``<store>/fleet/dispense``,
with ``dispense: island`` its island's ``<store>/trainer<k>``, under its shard name
``island<K + j>`` (unique in the fleet), so a prompt is given out once per pass whichever
sampler of either kind takes it. The compile adds ``--rl-island-dispense-store
<K + j>:<ledger store>`` per Miles sampler and ``--rl-store-dispense <the pools'
prompts.dispense>`` once; the plan records each Miles sampler's ``dispense_store`` and
``dispense_namespace``. The ledger pins its prompt file (rows and SHA-256), so
``fleet.trainer_argv`` ``--data`` must be the pools' prompt file (a local JSONL, equal
to each pool's ``prompts_file`` where one is set) and the pools must share one setting;
the compile refuses anything else (a Miles sampler on its own slot would give out the
pools' prompts a second time).

Supported training forms: full parameters on a backend that declares
``fleet_full_parameter`` (Miles, verl), partial experts on one that also declares
``fleet_partial_experts`` (Miles), synchronized by asynchronous full-parameter rounds
(``--rl-sync-preset decoupled`` routes there), every island of the fleet on the same
backend (:func:`fleet_backends`); :func:`form_problem` refuses the rest (LoRA, Miles and
verl islands in one fleet) in the preflight, and :func:`compaction_problem` a fleet of a
backend that cannot train compacted trials whose pools compact or whose trainer runs
CompactionRL.

The roster (:func:`roster_of`). Without ``fleet.fixed_roster`` the islands share a syncer
island pool: the compile adds ``--rl-sync-island-lease-s <fleet.island_lease_s>`` (default
the launcher's 150 s), so an island given up or stopped leaves the merge and the rest go on.
A fleet the pool does not fit (:func:`island_pool_problem`: strict rounds, several syncer
shards, broadcast deltas, a critic) keeps the fixed roster with a WARNING, as does
``fleet.fixed_roster: true``; a trainer argv with its own membership options is taken as
given. ``fleet.min_islands`` (default 1) stops the fleet once fewer islands run.

``yeto run up`` writes :meth:`FleetPlan.to_json` to ``<store>/fleet/plan.json``.

An island's step time (``islands[k].trainer.step_s``, else ``fleet.step_s``; optional,
seconds of one local version) is an estimate the preflight's release cadence check
reads (:func:`island_step_s`, :func:`yeto.rl.up_fleet.release_every_warnings`); the
compile and the plan do not use it.

Import-light: no torch, ray or miles.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
import sys
from dataclasses import dataclass, field
from collections.abc import Iterable
from typing import Any

PLAN_VERSION = 1
PLAN_KEY = "fleet_plan"  # the compiled run config's key holding the FleetPlan
FLEET_DIR = "fleet"
PLAN_OBJECT = f"{FLEET_DIR}/plan.json"  # the plan in the run store
DISPENSE_DIR = f"{FLEET_DIR}/dispense"  # the fleet's task ledger root (store_dispense.FLEET_DISPENSE_DIR)
DISPENSE_MODES = ("fleet", "island")
FLEET_KEYS = frozenset({"trainer_argv", "dispense", "step_s", "sandbox_usd_per_h", "fixed_roster", "min_islands",
                        "island_lease_s", "launch", "max_islands", "waiting_absent_after_s", "syncer_service"})
# fleet.syncer_service: the host of an on-site syncer run as a service (``yeto fleet service``):
# a host name or IPv4 address, or an IPv6 address in brackets ([addr]), the islands dial, no
# port (the syncer's ports are its own; "host:port" is built from it)
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
_IPV6 = r"(?=[0-9A-Fa-f:.]*[0-9A-Fa-f])(?:[0-9A-Fa-f]{0,4}:){2,7}(?:[0-9A-Fa-f]{1,4}|\d{1,3}(?:\.\d{1,3}){3})?"
SYNCER_SERVICE_HOST = re.compile(rf"{_LABEL}(?:\.{_LABEL})*|\[{_IPV6}\]")
DEFAULT_MIN_ISLANDS = 1
# fleet.waiting_absent_after_s: an island waiting for a spare node (a resident ssh: island
# that lost a node, yeto.launch.island_spares) counts as running for fleet.min_islands this
# long, then as absent (its own give-up is --spare-max-wait-s, 24 h)
DEFAULT_WAITING_ABSENT_AFTER_S = 3600.0
LEGACY_PLAN = "a plan written before fleet.fixed_roster"  # its launch had the fixed roster
# fleet.launch: how the trainers start. synced_group (default): one yeto launch of every
# trainer and the syncer. per_island: one yeto launch of the syncer and one per trainer
# island, each its own launch (yeto.rl.island_launches); islands can be added, removed and
# moved while the fleet trains
LAUNCH_SYNCED_GROUP, LAUNCH_PER_ISLAND = "synced_group", "per_island"
LAUNCH_MODES = (LAUNCH_SYNCED_GROUP, LAUNCH_PER_ISLAND)
ISLAND_KEYS = frozenset({"name", "trainer", "samplers"})
ISLAND_TRAINER_KEYS = frozenset({"gpu", "usd_per_h", "argv", "step_s"})
# the only trainer flags an island sets for itself (each becomes --rl-island-<flag> k:N)
ISLAND_FLAGS = {"--rl-release-every": "release_every", "--rl-max-policy-age": "max_policy_age",
                "--rl-sample-wait-s": "sample_wait_s"}
# island flags whose value is seconds (a number > 0) rather than a whole number >= 1
SECONDS_FLAGS = frozenset({"--rl-sample-wait-s"})
SAMPLE_WAIT_FLAG = "--rl-sample-wait-s"
# flags the compile writes; fleet.trainer_argv must not set them
GENERATED_FLAGS = ("--gpu", "--rl-sample-role", "--rl-sampler-follow", "--rl-buffer-shards")
GENERATED_PREFIX = "--rl-island-"
EXCLUSIVE_KEYS = ("trainer", "pool", "samplers")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# an island's samplers entry of a Miles sampler island of the fleet's launch (module doc)
MILES_KIND = "miles"
MILES_KEYS = frozenset({"kind", "name", "gpu", "usd_per_h"})
MILES_ENGINE = "sglang"  # Miles rolls out with SGLang
# FleetPool.source / FleetMilesSampler.source: where a sampler of the island runs
SOURCE_EXTERNAL, SOURCE_MILES = "external", MILES_KIND


def substore(k: int) -> str:
    """The store sub-directory of island k (``sample_roles`` names it so)."""
    return f"trainer{int(k)}"


def substore_uri(store: str, name: str) -> str:
    """``yeto.rl.engine.sample_roles.substore_uri`` (one more path segment)."""
    from yeto.rl.engine.sample_roles import substore_uri as _substore_uri  # noqa: PLC0415

    return _substore_uri(store, name)


@dataclass(frozen=True)
class FleetPool:
    name: str  # the compiled pool name, i<k>-<source_name>
    source_name: str  # the pool's name in the island's samplers list
    provider: str  # the provider kind
    shard_base: str  # <source shard_base>-i<k>
    shard_limit: int
    run_prefix: str  # <source run_prefix>/trainer<k>
    engine: str | None = None  # vllm / sglang (the spec's engine adapter), None: unknown
    rollout_quant: Any = None  # the spec's rollout_quant tier, None: none / unknown
    target: int | None = None
    # one node hour: the entry's usd_per_h, else its provider config's price_usd_h (USD
    # providers); None: unknown (National Compute: the run's budget.nc_usd_per_h applies)
    usd_per_h: float | None = None

    def shards(self) -> list[str]:
        return [f"{self.shard_base}-{n}" for n in range(self.shard_limit)]

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "source_name": self.source_name, "source": SOURCE_EXTERNAL,
                "provider": self.provider, "shard_base": self.shard_base, "shard_limit": self.shard_limit,
                "run_prefix": self.run_prefix, "target": self.target, "engine": self.engine,
                "rollout_quant": self.rollout_quant, "usd_per_h": self.usd_per_h}


@dataclass(frozen=True)
class FleetMilesSampler:
    """A Miles sampler island of the fleet's launch, in fleet island ``learner``'s
    ``samplers`` (module doc)."""

    name: str  # i<k>-<source_name>
    source_name: str  # its name in the island's samplers list
    gpu: str  # its --gpu cluster
    usd_per_h: float | None
    launch_island: int  # its index in the launch's --gpu (after every trainer)
    engine: str = MILES_ENGINE
    # the store of the task ledger it claims from with the pools (module doc), None: its data slot
    dispense_store: str | None = None
    dispense_namespace: str | None = None

    @property
    def shard(self) -> str:
        """The buffer shard it writes: its node name (``sample_roles.node_name``)."""
        return f"island{self.launch_island}"

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "source_name": self.source_name, "source": SOURCE_MILES, "gpu": self.gpu,
                "usd_per_h": self.usd_per_h, "launch_island": self.launch_island, "shard": self.shard,
                "engine": self.engine, "dispense_store": self.dispense_store,
                "dispense_namespace": self.dispense_namespace}


@dataclass(frozen=True)
class FleetIsland:
    index: int
    name: str
    gpu: str
    usd_per_h: float | None  # the island's launch price: its trainer and its Miles samplers
    store: str  # <store>/trainer<k>
    release_every: int | None = None
    max_policy_age: int | None = None
    pools: tuple[FleetPool, ...] = ()
    miles_samplers: tuple[FleetMilesSampler, ...] = ()
    trainer_usd_per_h: float | None = None  # its trainer alone (usd_per_h without Miles samplers)
    sample_wait_s: float | None = None  # its trainer's own --rl-sample-wait-s (None: the fleet's)
    # the widest first prompt its pools write (their agentic.max_prompt_len), on a backend that
    # sizes its buffer trainer's prompt rows (buffer_max_prompt_len); None: not given / not sized
    buffer_max_prompt_len: int | None = None

    @property
    def learner(self) -> int:
        return self.index  # the trainers come first in the fleet's launch: island = learner id

    @property
    def substore(self) -> str:
        return substore(self.index)

    def pool_shards(self) -> list[str]:
        """The shards of its external pools (its ``--rl-island-buffer-shards`` entry)."""
        return [s for pool in self.pools for s in pool.shards()]

    def buffer_shards(self) -> list[str]:
        """Every shard its trainer reads: its Miles samplers' (in launch order), then its pools'."""
        return [m.shard for m in self.miles_samplers] + self.pool_shards()

    def to_json(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "learner": self.learner, "substore": self.substore,
                "store": self.store, "gpu": self.gpu, "usd_per_h": self.usd_per_h,
                "trainer_usd_per_h": self.trainer_usd_per_h,
                "release_every": self.release_every, "max_policy_age": self.max_policy_age,
                "sample_wait_s": self.sample_wait_s, "buffer_shards": self.buffer_shards(), "pools": [p.to_json() for p in self.pools],
                "miles_samplers": [m.to_json() for m in self.miles_samplers],
                # a Miles fleet's plan stays byte for byte what it was
                **({"buffer_max_prompt_len": self.buffer_max_prompt_len}
                   if self.buffer_max_prompt_len is not None else {})}


@dataclass(frozen=True)
class FleetPlan:
    run: str
    store: str
    dispense: str
    islands: tuple[FleetIsland, ...]
    dispense_store: str | None = None  # the fleet's ledger root (dispense: fleet), None: per island
    sync: dict[str, Any] = field(default_factory=dict)  # the trainers' sync route (training_form)
    version: int = PLAN_VERSION
    syncer_usd_per_h: float | None = None  # the run config's syncer.usd_per_h (None: unset)
    # where the launch runs the syncer (breakers.fleet.SyncerPlacement.to_json; None: a plan written before it)
    syncer_placement: dict[str, Any] | None = None
    fixed_roster: bool = False  # a fixed roster: one island given up ends the fleet (exit 4)
    min_islands: int = DEFAULT_MIN_ISLANDS  # fleet.min_islands: fewer islands running stop the fleet
    island_lease_s: float | None = None  # the syncer lease of the island pool (None: a fixed roster)
    roster_reason: str | None = None  # why the roster is fixed (fleet.fixed_roster, or the pool's precondition)
    launch: str = LAUNCH_SYNCED_GROUP  # fleet.launch
    max_islands: int | None = None  # per_island: the syncer's learner id limit (fleet.max_islands)
    waiting_absent_after_s: float | None = None  # fleet.waiting_absent_after_s (None: the default)
    controller_poll_s: float | None = None  # the trainer argv's --controller-poll (None: not given, 30 s)
    # per_island: the host of the on-site syncer service (fleet.syncer_service; None: the launch's own syncer)
    syncer_service: str | None = None

    @property
    def absent_after_s(self) -> float:
        """How long an island waiting for a spare node still counts for ``min_islands``."""
        return float(DEFAULT_WAITING_ABSENT_AFTER_S if self.waiting_absent_after_s is None
                     else self.waiting_absent_after_s)

    @property
    def per_island(self) -> bool:
        return self.launch == LAUNCH_PER_ISLAND

    def to_json(self) -> dict[str, Any]:
        out = {"version": self.version, "run": self.run, "store": self.store, "dispense": self.dispense,
               "dispense_store": self.dispense_store, "sync": dict(self.sync),
               "syncer": {"usd_per_h": self.syncer_usd_per_h,
                          **({"placement": dict(self.syncer_placement)} if self.syncer_placement else {})},
               "roster": {"fixed": self.fixed_roster, "min_islands": self.min_islands,
                          "lease_s": self.island_lease_s, "reason": self.roster_reason,
                          **({"waiting_absent_after_s": self.waiting_absent_after_s}
                             if self.waiting_absent_after_s is not None else {}),
                          **({"controller_poll_s": self.controller_poll_s}
                             if self.controller_poll_s is not None else {})},
               "islands": [i.to_json() for i in self.islands]}
        if self.per_island:  # a synced-group plan stays byte for byte what it was
            out.update(launch=self.launch, max_islands=self.max_islands)
        if self.syncer_service:  # only a fleet with an on-site syncer service gets the key
            out["syncer"]["service"] = self.syncer_service
        return out

    def dumps(self) -> str:
        return json.dumps(self.to_json(), indent=1, sort_keys=True)

    def island_of_pool(self, pool: str) -> FleetIsland | None:
        """The island whose pools include the compiled pool ``pool`` (None: none)."""
        return next((i for i in self.islands if any(p.name == pool for p in i.pools)), None)

    def lines(self) -> list[str]:
        """The plan as the preflight prints it: one line per island and one per pool."""
        sync = self.sync or {}
        route = (f"{sync.get('requested')} -> {sync.get('preset')} ({sync.get('rounds')} rounds, "
                 f"{sync.get('form')})" if sync.get("preset") else "unknown (the trainer argv does not parse)")
        launches = (f"one launch per trainer island and one of the syncer (at most {self.max_islands} islands)"
                    if self.per_island else "one synced-group trainer launch and its syncer")
        out = [f"fleet: {len(self.islands)} island(s), {launches}; "
               f"sync {route}; dispense {self.dispense}"
               + (f" (ledger {self.dispense_store})" if self.dispense_store else " (one ledger per island)")
               + f"; plan {substore_uri(self.store, PLAN_OBJECT)}",
               (f"fleet: fixed roster ({self.roster_reason or 'fleet.fixed_roster'}): an island given up ends "
                "the whole fleet (exit 4); an island's trip stops its pools and its trainer starves"
                if self.fixed_roster else
                f"fleet: no fixed roster (syncer island pool, lease {self.island_lease_s or 0:g} s): an island "
                f"given up or stopped leaves the merge and the others go on; the fleet stops once fewer than "
                f"{self.min_islands} island(s) run (fleet.min_islands)")]
        # a fleet's islands share one backend (form_problem); a Miles fleet's lines stay as they were
        backend = f" on {sync['backend']}" if sync.get("backend") not in (None, MILES_KIND) else ""
        for i in self.islands:
            trainer_price = i.trainer_usd_per_h if i.miles_samplers else i.usd_per_h  # the trainer alone
            price = "?" if trainer_price is None else f"${trainer_price:g}/h"
            extra = "".join(f", {flag} {value:g}" for flag, value in (("release_every", i.release_every),
                                                                      ("max_policy_age", i.max_policy_age),
                                                                      ("sample_wait_s", i.sample_wait_s))
                            if value is not None)
            miles = (f", Miles samplers {', '.join(m.name for m in i.miles_samplers)}" if i.miles_samplers else "")
            out.append(f"fleet: island {i.index} {i.name}: learner {i.learner}, store {i.substore}/ ({i.store}), "
                       f"trainer {i.gpu}{backend} at {price}{extra}, {len(i.buffer_shards())} buffer shard(s), "
                       f"pools {', '.join(p.name for p in i.pools) or '-'}{miles}")
            for m in i.miles_samplers:
                cost = "?" if m.usd_per_h is None else f"${m.usd_per_h:g}/h"
                rows = (f"rows from the task ledger {substore_uri(m.dispense_store, m.dispense_namespace or '')}"
                        if m.dispense_store else "rows from its data slot of --data")
                out.append(f"fleet:   Miles sampler {m.name} (from {m.source_name}): launch island {m.launch_island}, "
                           f"{m.gpu} at {cost}, {m.engine}, shard {m.shard}, follows learner {i.learner}, {rows}")
            for p in i.pools:
                out.append(f"fleet:   pool {p.name} (from {p.source_name}): {p.provider}"
                           f"{', ' + p.engine if p.engine else ''}"
                           f"{', rollout_quant ' + str(p.rollout_quant) if p.rollout_quant is not None else ''}, "
                           f"shards {p.shard_base}-0..{p.shard_limit - 1}, run_prefix {p.run_prefix}")
        return out

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> FleetPlan:
        islands = tuple(
            FleetIsland(index=int(i["index"]), name=str(i["name"]), gpu=str(i["gpu"]), usd_per_h=i.get("usd_per_h"),
                        store=str(i["store"]), release_every=i.get("release_every"),
                        max_policy_age=i.get("max_policy_age"), sample_wait_s=i.get("sample_wait_s"),
                        pools=tuple(FleetPool(name=str(p["name"]), source_name=str(p["source_name"]),
                                              provider=str(p["provider"]), shard_base=str(p["shard_base"]),
                                              shard_limit=int(p["shard_limit"]), run_prefix=str(p["run_prefix"]),
                                              engine=p.get("engine"), rollout_quant=p.get("rollout_quant"),
                                              target=p.get("target"), usd_per_h=p.get("usd_per_h"))
                                    for p in i.get("pools") or ()),
                        miles_samplers=tuple(FleetMilesSampler(name=str(m["name"]), source_name=str(m["source_name"]),
                                                               gpu=str(m["gpu"]), usd_per_h=m.get("usd_per_h"),
                                                               launch_island=int(m["launch_island"]),
                                                               engine=str(m.get("engine") or MILES_ENGINE),
                                                               dispense_store=m.get("dispense_store"),
                                                               dispense_namespace=m.get("dispense_namespace"))
                                             for m in i.get("miles_samplers") or ()),
                        # a plan written before Miles samplers: the island's price is its trainer's
                        trainer_usd_per_h=i.get("trainer_usd_per_h", i.get("usd_per_h")),
                        buffer_max_prompt_len=i.get("buffer_max_prompt_len"))
            for i in doc.get("islands") or ())
        syncer = doc.get("syncer") if isinstance(doc.get("syncer"), dict) else {}
        # a plan written before fleet.fixed_roster: its launch had the fixed roster
        roster = doc.get("roster") if isinstance(doc.get("roster"), dict) else {"fixed": True, "reason": LEGACY_PLAN}
        return cls(run=str(doc["run"]), store=str(doc["store"]), dispense=str(doc["dispense"]), islands=islands,
                   dispense_store=doc.get("dispense_store"), sync=dict(doc.get("sync") or {}),
                   version=int(doc.get("version", PLAN_VERSION)), syncer_usd_per_h=syncer.get("usd_per_h"),
                   syncer_placement=syncer.get("placement") if isinstance(syncer.get("placement"), dict) else None,
                   fixed_roster=bool(roster.get("fixed", False)),
                   min_islands=int(roster.get("min_islands") or DEFAULT_MIN_ISLANDS),
                   island_lease_s=roster.get("lease_s"), roster_reason=roster.get("reason"),
                   launch=str(doc.get("launch") or LAUNCH_SYNCED_GROUP), max_islands=doc.get("max_islands"),
                   waiting_absent_after_s=roster.get("waiting_absent_after_s"),
                   controller_poll_s=roster.get("controller_poll_s"), syncer_service=syncer.get("service"))


def plan_of(cfg: dict[str, Any]) -> FleetPlan | None:
    """The FleetPlan of a compiled run config (None: a single-trainer run)."""
    plan = cfg.get(PLAN_KEY)
    return plan if isinstance(plan, FleetPlan) else None


# ---------------------------------------------------------------------- checks

def _err(msg: str) -> ValueError:
    return ValueError(f"run config: {msg}")


def _flag(arg: str) -> str:
    return arg.split("=", 1)[0]


def _flag_values(argv: list[str], where: str) -> dict[str, Any]:
    """The island trainer argv: ``--rl-release-every N`` / ``--rl-max-policy-age N``
    (or ``--flag=N``), N a whole number >= 1, and ``--rl-sample-wait-s S`` (seconds > 0),
    each at most once."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        flag = _flag(arg)
        if flag not in ISLAND_FLAGS:
            raise _err(f"{where}: {arg!r} is not an island trainer flag (an island sets only "
                       f"{' / '.join(ISLAND_FLAGS)}; the rest goes in fleet.trainer_argv)")
        if "=" in arg:
            raw, i = arg.split("=", 1)[1], i + 1
        elif i + 1 < len(argv):
            raw, i = argv[i + 1], i + 2
        else:
            raise _err(f"{where}: {flag} needs a value")
        key = ISLAND_FLAGS[flag]
        if key in out:
            raise _err(f"{where}: {flag} repeats")
        out[key] = _seconds(raw, f"{where}: {flag}") if flag in SECONDS_FLAGS else _whole(raw, f"{where}: {flag}")
    return out


def _whole(raw: str, where: str) -> int:
    if not raw.isdigit() or int(raw) < 1:
        raise _err(f"{where} must be a whole number >= 1 (got {raw!r})")
    return int(raw)


def _seconds(raw: Any, where: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
        raise _err(f"{where} must be a number of seconds > 0 (got {raw!r})")
    return value


def _trainer_section_argv(cfg: dict[str, Any], trainer_argv: list[str]) -> list[str]:
    """The flags and island env pairs the compile adds to the launch for the run config's
    ``breakers.trainer.*`` (``--rl-sample-wait-s``, ``--rl-max-policy-age``, ``--island-env
    YETO_TRAINER_RELEASE_DRAIN_S=...``): no trainer reads the section, each reads its knob.
    The write-back of a single trainer's run config (:func:`yeto.breakers.materialize`) over
    ``fleet.trainer_argv``: a value it already sets adds nothing, one that disagrees is
    refused, as is a section value no trainer knob carries. Every trainer island gets them;
    an island's own ``--rl-sample-wait-s`` / ``--rl-max-policy-age`` wins for that island."""
    section = cfg.get("breakers")
    if not isinstance(section, dict) or not section.get("trainer"):
        return []
    from yeto import breakers  # noqa: PLC0415

    sources = breakers.Sources(argv=list(trainer_argv), env=dict(cfg.get("env") or {}), section=section)
    try:
        resolved = breakers.resolve(sources, layers=("trainer",))
        out = breakers.materialize(resolved, sources, surfaces=("trainer_flag", "trainer_env"))
    except breakers.BreakerConfigError as error:
        raise _err(f"{str(error).replace('trainer.argv ', 'fleet.trainer_argv ')} (an island's own trainer.argv "
                   f"{SAMPLE_WAIT_FLAG} / --rl-max-policy-age overrides the run's)") from None
    wait = resolved.entry("trainer.sample_wait")
    if wait.sources["timeout_s"].kind == "breakers":  # the trainers refuse a wait of 0
        _seconds(wait.params["timeout_s"], wait.sources["timeout_s"].location)
    return list(out.argv[len(trainer_argv):])


def _string_list(value: Any, where: str) -> list[str]:
    if not (isinstance(value, list) and all(isinstance(a, str) for a in value)):
        raise _err(f"{where} must be a list of strings (got {value!r})")
    return list(value)


def _norm_store(uri: str) -> str:
    from yeto.rl.up_consistency import _norm_store as norm  # noqa: PLC0415

    return norm(uri)


def check_exclusive(cfg: dict[str, Any]) -> None:
    """``islands`` excludes ``trainer``, ``pool`` and ``samplers``; ``fleet`` needs ``islands``."""
    if "islands" not in cfg:
        if "fleet" in cfg:
            raise _err("'fleet' (the trainers' shared settings) needs an 'islands' section")
        return
    both = [k for k in EXCLUSIVE_KEYS if k in cfg]
    if both:
        raise _err(f"'islands' and {' / '.join(repr(k) for k in both)} are mutually exclusive: a fleet declares "
                   "each island's trainer and sampler pools in 'islands' and the trainers' shared argv in "
                   "'fleet.trainer_argv'")


def _fleet_section(cfg: dict[str, Any]) -> tuple[list[str], str]:
    section = cfg.get("fleet")
    if not isinstance(section, dict):
        raise _err("an island fleet needs a 'fleet' object with trainer_argv (the trainers' shared yeto launch "
                   "arguments)")
    unknown = set(section) - FLEET_KEYS
    if unknown:
        raise _err(f"unknown fleet keys {sorted(unknown)} (known: {sorted(FLEET_KEYS)})")
    if "trainer_argv" not in section:
        raise _err("fleet.trainer_argv (the trainers' shared yeto launch arguments) is required")
    argv = _string_list(section["trainer_argv"], "fleet.trainer_argv")
    for arg in argv:
        flag = _flag(arg)
        if flag in GENERATED_FLAGS or flag.startswith(GENERATED_PREFIX):
            raise _err(f"fleet.trainer_argv: {flag} is written by the fleet compile (from each island's trainer "
                       "and pools); drop it")
        if flag == "--rl-single-island-no-sync":
            raise _err("fleet.trainer_argv: --rl-single-island-no-sync is a single buffer trainer; a fleet's "
                       "trainers merge through the syncer")
    _step(section.get("step_s"), "fleet.step_s")
    _price(section.get("sandbox_usd_per_h"), "fleet.sandbox_usd_per_h")  # the sandboxes in the fleet spend
    dispense = section.get("dispense", "fleet")
    if dispense not in DISPENSE_MODES:
        raise _err(f"fleet.dispense must be one of {list(DISPENSE_MODES)} (got {dispense!r})")
    roster_options(cfg)
    launch = section.get("launch", LAUNCH_SYNCED_GROUP)
    if launch not in LAUNCH_MODES:
        raise _err(f"fleet.launch must be one of {list(LAUNCH_MODES)} (got {launch!r})")
    if "max_islands" in section and launch != LAUNCH_PER_ISLAND:
        raise _err("fleet.max_islands applies to fleet.launch per_island (islands added while the fleet trains)")
    syncer_service(section)
    return argv, str(dispense)


def syncer_service(section: dict[str, Any]) -> str | None:
    """``fleet.syncer_service`` (validated; None: unset): the host of the fleet's on-site syncer,
    which runs as a service of its own (``yeto fleet service``) instead of in the syncer launch.
    Only a fleet launched island by island has a syncer launch to hand the syncer to."""
    host = section.get("syncer_service")
    if host is None:
        return None
    if section.get("launch", LAUNCH_SYNCED_GROUP) != LAUNCH_PER_ISLAND:
        raise _err("fleet.syncer_service applies to fleet.launch per_island (the syncer launch hands the syncer "
                   "to the on-site service)")
    if not isinstance(host, str) or not SYNCER_SERVICE_HOST.fullmatch(host):
        raise _err(f"fleet.syncer_service must be the syncer host's name or address the islands dial, without a "
                   f"port; an IPv6 address in brackets, [addr] (got {host!r})")
    return host


def roster_options(cfg: dict[str, Any]) -> tuple[bool, int, float | None]:
    """``(fleet.fixed_roster, fleet.min_islands, fleet.island_lease_s)`` as given (validated):
    a fixed roster ends the fleet when one island is given up (the launch's fixed-roster
    controller, exit 4); without it (the default) the islands share a syncer island pool
    (:func:`roster_of`) and the fleet stops once fewer than ``min_islands`` (default 1)
    islands run. ``island_lease_s`` (None: unset) is the pool's syncer lease."""
    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    fixed = section.get("fixed_roster", False)
    if not isinstance(fixed, bool):
        raise _err(f"fleet.fixed_roster must be true or false (got {fixed!r})")
    minimum = section.get("min_islands", DEFAULT_MIN_ISLANDS)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise _err(f"fleet.min_islands must be an integer >= 1 (got {minimum!r})")
    islands = cfg.get("islands")
    if isinstance(islands, list) and islands and minimum > len(islands):
        raise _err(f"fleet.min_islands {minimum} > the fleet's {len(islands)} island(s)")
    lease = section.get("island_lease_s")
    if lease is not None:
        if isinstance(lease, bool) or not isinstance(lease, (int, float)):  # "200" is no number of seconds
            raise _err(f"fleet.island_lease_s must be a number of seconds > 0 (got {lease!r})")
        lease = _seconds(lease, "fleet.island_lease_s")
        if fixed:
            raise _err("fleet.island_lease_s is the syncer lease of the island pool; a fixed roster "
                       "(fleet.fixed_roster: true) has none: drop one of the two")
    return fixed, int(minimum), lease


def controller_poll_s(argv: Any) -> float | None:
    """The trainer argv's ``--controller-poll`` in seconds (None: not given; the page reads 30 s):
    how often the island controller writes a waiting island's record at most late."""
    args = [str(a) for a in (argv or ())]
    for i, a in enumerate(args):
        raw = args[i + 1] if a == "--controller-poll" and i + 1 < len(args) else (
            a.split("=", 1)[1] if a.startswith("--controller-poll=") else None)
        if raw is not None:
            try:
                v = float(raw)
            except ValueError:
                return None
            return v if v > 0 else None
    return None


def waiting_absent_after_s(cfg: dict[str, Any]) -> float | None:
    """``fleet.waiting_absent_after_s`` as given (None: unset, the default
    :data:`DEFAULT_WAITING_ABSENT_AFTER_S`): an island waiting for a spare node counts as
    running for ``fleet.min_islands`` this long after its wait began, then as absent."""
    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    raw = section.get("waiting_absent_after_s")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise _err(f"fleet.waiting_absent_after_s must be a number of seconds > 0 (got {raw!r})")
    return _seconds(raw, "fleet.waiting_absent_after_s")


def membership_flags() -> frozenset[str]:
    """The ``yeto launch`` membership options (``yeto.launch.syncer`` island pool): any one
    set puts the islands in the syncer's dynamic membership."""
    from yeto.launch.syncer import _ISLAND_POOL_PARAMS  # noqa: PLC0415

    return frozenset(option for _dest, option in _ISLAND_POOL_PARAMS)


LEASE_FLAG = "--rl-sync-island-lease-s"


def island_pool_problem(launch_argv: list[str], num_islands: int) -> str | None:
    """Why the fleet's launch cannot run a syncer island pool (None: it can), from the launch
    argv resolved as ``yeto launch`` resolves it (sync route, then ``--rl-sync-mode``): a
    precondition of the pool (``yeto.launch.rl_args._spot_pool_off_reason``: asynchronous
    rounds, no critic, one syncer, no broadcast deltas, two islands or more), or rounds that
    wait for every island (``--rl-sync-mode strict``, or a count quorum of every island
    without a grace window or a base lag): the pool needs rounds that merge without some
    island, and lowering the quorum would change the merge the argv asks for."""
    import contextlib  # noqa: PLC0415
    import io  # noqa: PLC0415

    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args([str(a) for a in launch_argv])
    if args is None:
        return "the trainer argv does not parse as an RL yeto launch argv"
    from yeto.launch import rl_args  # noqa: PLC0415
    from yeto.rl.engine.sync_route import SyncRouteError, apply_sync_route  # noqa: PLC0415

    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            apply_sync_route(args)
            rl_args._apply_rl_sync_mode(args)  # noqa: SLF001
    except (SyncRouteError, ValueError, AttributeError, TypeError) as error:
        return f"its sync options do not resolve ({error})"
    reason = rl_args._spot_pool_off_reason(args, num_islands)  # noqa: SLF001
    if reason is not None:
        return reason
    quorum = getattr(args, "rl_sync_quorum", None)
    grace = int(getattr(args, "rl_sync_grace_ms", None) or 0)
    lag = int(getattr(args, "rl_sync_max_base_lag", 0) or 0)
    if getattr(args, "rl_sync_quorum_theta", None) is None and grace == 0 and lag == 0 \
            and (quorum is None or int(quorum) >= num_islands):
        if getattr(args, "rl_sync_mode_effective", "strict") == "strict" and quorum is None:
            return "--rl-sync-mode strict: every round waits for every island"
        return (f"rounds that wait for every island (--rl-sync-quorum {quorum if quorum is not None else num_islands}, "
                "no grace window, no base lag)")
    return None


@dataclass(frozen=True)
class Roster:
    """The fleet's roster (:func:`roster_of`): ``argv`` is what the compile adds to the launch."""

    fixed: bool
    min_islands: int
    lease_s: float | None
    reason: str | None
    argv: tuple[str, ...] = ()


def roster_of(cfg: dict[str, Any], trainer_argv: list[str], launch_argv: list[str], num_islands: int) -> Roster:
    """The fleet's roster. ``fleet.fixed_roster: true``: fixed, nothing added. A trainer argv
    that sets a membership option (:func:`membership_flags`): the island pool as given (plus
    ``fleet.island_lease_s`` when set and the argv names no lease). Else the island pool by
    default, ``--rl-sync-island-lease-s <fleet.island_lease_s, default the launcher's>``
    and nothing else (the resolved rounds already merge without some island), unless
    :func:`island_pool_problem` names a reason: the fixed roster is then kept, with a
    WARNING (a fleet that launched before keeps launching)."""
    from yeto.launch.syncer import DEFAULT_ISLAND_LEASE_S  # noqa: PLC0415

    fixed, minimum, lease = roster_options(cfg)
    if fixed:
        return Roster(True, minimum, None, "fleet.fixed_roster")
    flags = membership_flags()
    if any(_flag(a) in flags for a in trainer_argv):
        given = _argv_value(trainer_argv, LEASE_FLAG)
        if given is not None and lease is not None:
            raise _err(f"fleet.island_lease_s and fleet.trainer_argv {LEASE_FLAG} both set the syncer lease: "
                       "set it in one place")
        extra = (LEASE_FLAG, f"{lease:g}") if lease is not None else ()
        value = lease if lease is not None else (float(given) if given is not None else DEFAULT_ISLAND_LEASE_S)
        return Roster(False, minimum, value, None, extra)
    problem = island_pool_problem(launch_argv, num_islands)
    if problem is not None:
        print(f"WARNING: fleet: the islands keep a fixed roster ({problem}): one island given up ends the whole "
              "fleet (exit 4). Set fleet.fixed_roster: true to say so, or change the trainer argv for the "
              "syncer island pool", file=sys.stderr, flush=True)
        return Roster(True, minimum, None, problem)
    value = DEFAULT_ISLAND_LEASE_S if lease is None else lease
    return Roster(False, minimum, value, None, (LEASE_FLAG, f"{value:g}"))


def per_island_roster(cfg: dict[str, Any]) -> Roster:
    """The roster of a fleet launched island by island (``fleet.launch: per_island``): always
    the syncer island pool (every launch has ``--rl-sync-max-learners``), its membership as
    the launches resolve it themselves: none of the options :func:`roster_of` adds to a
    synced group's argv; ``argv`` holds only ``fleet.island_lease_s`` when set (the syncer
    launch and each island launch). The rounds' quorum is the launch's own default: with
    room for islands past the launched ones, the capacity quorum ``--rl-sync-quorum-theta
    1.0`` (every island in the pool answers each round). A fixed roster is refused: no
    launch holds every island."""
    from yeto.launch.syncer import DEFAULT_ISLAND_LEASE_S  # noqa: PLC0415

    fixed, minimum, lease = roster_options(cfg)
    if fixed:
        raise _err("fleet.fixed_roster applies to fleet.launch synced_group: a fleet launched island by island "
                   "(per_island) has its syncer island pool, no launch holds every island")
    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    trainer_argv = [str(a) for a in section.get("trainer_argv") or ()]
    given = _argv_value(trainer_argv, LEASE_FLAG)
    if given is not None and lease is not None:
        raise _err(f"fleet.island_lease_s and fleet.trainer_argv {LEASE_FLAG} both set the syncer lease: "
                   "set it in one place")
    value = lease if lease is not None else (float(given) if given is not None else DEFAULT_ISLAND_LEASE_S)
    return Roster(False, minimum, value, None, (LEASE_FLAG, f"{lease:g}") if lease is not None else ())


def launch_mode(cfg: dict[str, Any]) -> tuple[str, int | None]:
    """``(fleet.launch, fleet.max_islands)`` of a fleet run config. The limit defaults to
    the declared islands; it is the syncer's learner id limit (``--rl-sync-max-learners``),
    so no island past it can join while the fleet trains."""
    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    launch = str(section.get("launch", LAUNCH_SYNCED_GROUP))
    if launch != LAUNCH_PER_ISLAND:
        return launch, None
    declared = len(cfg.get("islands") or ())
    raw = section.get("max_islands", declared)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < max(1, declared):
        raise _err(f"fleet.max_islands must be a whole number >= the {declared} declared island(s) (got {raw!r})")
    return launch, int(raw)


def _argv_value(argv: list[str], flag: str) -> str | None:
    value = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
    return value


def _gpu(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip() or "," in value:
        raise _err(f"{where} must be one --gpu cluster, e.g. modal:2x8xh100@<region> (got {value!r})")
    return value.strip()


def _price(value: Any, where: str) -> float | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value < 0):
        raise _err(f"{where} must be a number >= 0 (got {value!r})")
    return None if value is None else float(value)


def _step(value: Any, where: str) -> float | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or value <= 0):
        raise _err(f"{where} must be a number of seconds > 0 (got {value!r})")
    return None if value is None else float(value)


def island_step_s(cfg: dict[str, Any]) -> dict[int, float]:
    """Island index -> its step time estimate in seconds (``islands[k].trainer.step_s``,
    else ``fleet.step_s``; module doc), for the islands that have one."""
    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    shared = section.get("step_s")
    out = {}
    for k, island in enumerate(cfg.get("islands") or ()):
        trainer = island.get("trainer") if isinstance(island, dict) else None
        own = trainer.get("step_s") if isinstance(trainer, dict) else None
        value = own if own is not None else shared
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0:
            out[k] = float(value)
    return out


def is_miles_entry(entry: Any) -> bool:
    """Whether an island's samplers entry is a Miles sampler island (``kind: miles``)."""
    return isinstance(entry, dict) and entry.get("kind") == MILES_KIND


def _miles_sampler(entry: dict[str, Any], k: int, j: int, island_where: str) -> tuple[str, str, float | None]:
    """``(source name, gpu, usd_per_h)`` of Miles sampler entry j of island k."""
    where = f"{island_where}.samplers[{j}]"
    unknown = set(entry) - MILES_KEYS
    if unknown:
        raise _err(f"unknown {where} keys {sorted(unknown)} (a Miles sampler takes {sorted(MILES_KEYS)}; its "
                   "sampler options are the launch's, in fleet.trainer_argv)")
    name = entry.get("name")
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise _err(f"{where}: a Miles sampler needs a name of letters, digits, '.', '_' or '-' (got {name!r})")
    return name, _gpu(entry.get("gpu"), f"{where}.gpu"), _price(entry.get("usd_per_h"), f"{where}.usd_per_h")


def _island_trainer(island: dict[str, Any], where: str) -> tuple[str, float | None, dict[str, int]]:
    if "trainer" not in island:
        raise _err(f"{where}: missing 'trainer' (every island runs one trainer: gpu, usd_per_h, argv)")
    tr = island["trainer"]
    if not isinstance(tr, dict):
        raise _err(f"{where}.trainer must be an object")
    unknown = set(tr) - ISLAND_TRAINER_KEYS
    if unknown:
        raise _err(f"unknown {where}.trainer keys {sorted(unknown)} (known: {sorted(ISLAND_TRAINER_KEYS)})")
    gpu = _gpu(tr.get("gpu"), f"{where}.trainer.gpu")
    price = _price(tr.get("usd_per_h"), f"{where}.trainer.usd_per_h")
    _step(tr.get("step_s"), f"{where}.trainer.step_s")
    flags = _flag_values(_string_list(tr.get("argv", []), f"{where}.trainer.argv"), f"{where}.trainer.argv")
    return gpu, price, flags


def _spec_facts(doc: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, Any]:
    """``(spec, engine, rollout_quant)`` of a pool config's rendered spec (None: unknown)."""
    from yeto.rl.up_consistency import effective_spec  # noqa: PLC0415

    spec = effective_spec(doc)
    if spec is None:
        return None, None, None
    adapter = (spec.get("engine") or {}).get("adapter") if isinstance(spec.get("engine"), dict) else None
    engine = str(adapter).removesuffix("-server") if adapter else None
    quant = spec.get("rollout_quant")
    if isinstance(quant, dict):
        quant = quant.get("tier")
    return spec, engine, quant


def port_share_key(spec: dict[str, Any] | None) -> tuple[str, ...] | None:
    """The port range a pool's samplers take one port each of, by shard index: a
    ``records_relay.listen_port`` range (with its relay host) or a ``tunnel.listen``
    range (with its tunnel host); None without one. Pools with the same key share it."""
    agentic = (spec or {}).get("agentic")
    setup = agentic.get("setup") if isinstance(agentic, dict) else None
    if not isinstance(setup, dict):
        return None
    relay = setup.get("records_relay")
    if isinstance(relay, dict) and "-" in str(relay.get("listen_port") or ""):
        return ("records_relay", str(relay.get("host") or ""), str(relay["listen_port"]))
    tunnel = setup.get("tunnel")
    listen = str(tunnel.get("listen") or "") if isinstance(tunnel, dict) else ""
    if "-" in listen.rpartition(":")[2]:
        sandbox = setup.get("sandbox") if isinstance(setup.get("sandbox"), dict) else {}
        return ("tunnel", str(tunnel.get("host") or sandbox.get("docker_host") or ""), listen)
    return None


def share_offsets(rows: list[tuple[str, tuple[str, ...] | None, int]]) -> dict[str, int]:
    """``{pool: shard_offset}`` of the pools that share a port range (``(pool, key,
    shard_limit)`` in fleet order, :func:`port_share_key`): each takes the next
    ``shard_limit`` ports of it. A pool alone on its range gets none."""
    groups: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    for name, key, limit in rows:
        if key is not None:
            groups.setdefault(key, []).append((name, int(limit)))
    out: dict[str, int] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        offset = 0
        for name, limit in members:
            out[name] = offset
            offset += limit
    return out


def has_sandbox_gate(doc: dict[str, Any]) -> bool:
    """Whether the pool's rendered spec has ``agentic.envs``: its samplers' dispenser
    reads the sandbox readiness manifest (``yeto.sandbox.gate``)."""
    spec, _engine, _quant = _spec_facts(doc)
    agentic = (spec or {}).get("agentic")
    return isinstance(agentic, dict) and bool(agentic.get("envs"))


def _pool(entry: Any, k: int, j: int, island_where: str, dispense_store: str | None,
          sandbox_store: str | None = None
          ) -> tuple[dict[str, Any], FleetPool, tuple[str, ...] | None, dict[str, Any]]:
    """The compiled pool entry of pool j of island k, its plan row, the port range it
    shares (:func:`port_share_key`) and its compiled pool config."""
    from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415
    from yeto.launch.nc.provider import provider_kind  # noqa: PLC0415
    from yeto.launch.rollout import pools  # noqa: PLC0415

    where = f"{island_where}.samplers[{j}]"
    if not isinstance(entry, dict):
        raise _err(f"{where} must be an object (a pool entry)")
    try:
        doc, _base, _path = pools.entry_doc(entry, where)
    except (OSError, ValueError) as error:
        raise _err(f"{where}: {error}") from None
    source = str(entry.get("name") or provider_kind(doc))
    data_path = str(entry.get("data_path", doc.get("data_path", "r2")))
    if data_path != "r2":
        raise _err(f"{where} ({source}): data_path {data_path!r}: a fleet's pools write to the run store "
                   "(data_path r2), each under its island's trainer<k>")
    for key in ("shard_base", "run_prefix"):
        if not str(doc.get(key) or "").strip("/"):
            raise _err(f"{where} ({source}): a fleet pool needs {key} (the compile moves it to island {k}'s "
                       f"{substore(k)})")
    if doc.get("shard_limit") is None:
        raise _err(f"{where} ({source}): a fleet pool needs shard_limit (island {k}'s trainer reads exactly "
                   "the shards below it)")
    sampler = doc.get("sampler") or {}
    if not isinstance(sampler, dict):
        raise _err(f"{where} ({source}): sampler must be an object")
    for key in ("island", "until_trainers_end", "dispense_store", "sandbox_store", "shard_offset"):
        if key in sampler:
            raise _err(f"{where} ({source}): sampler.{key} is written by the fleet compile; drop it")
    if isinstance(sampler.get("dispense"), dict) and sampler["dispense"].get("store"):
        raise _err(f"{where} ({source}): sampler.dispense.store: fleet.dispense decides where the task ledger "
                   "lives; drop it")
    base = str(doc["shard_base"])
    prefix = f"{str(doc['run_prefix']).strip('/')}/{substore(k)}"
    knobs = dict(sampler, island=k, until_trainers_end=1)
    if dispense_store:
        knobs["dispense_store"] = dispense_store
    if sandbox_store and has_sandbox_gate(doc):
        knobs["sandbox_store"] = sandbox_store  # the manifest yeto sandbox prepare writes in the run store
    out = dict(entry)  # its entry keys and inline pool keys, with the island's overrides
    out.update({"name": f"i{k}-{source}", "run_prefix": prefix, "shard_base": f"{base}-i{k}", "sampler": knobs})
    compiled = dict(doc, run_prefix=prefix, shard_base=f"{base}-i{k}", sampler=knobs)
    try:
        limit = nc_cli.pool_shard_limit(compiled)
        nc_cli._overrides(knobs)
    except (TypeError, ValueError) as error:
        raise _err(f"{where} ({source}): {error}") from None
    if limit is None or limit < 1:
        raise _err(f"{where} ({source}): shard_limit must be a whole number >= 1 (got {doc.get('shard_limit')!r})")
    spec, engine, quant = _spec_facts(compiled)
    target = doc.get("target")
    price = _price(entry.get("usd_per_h"), f"{where}.usd_per_h")
    if price is None:  # what run up bills a node hour at (up.UpRun.node_usd_per_h), USD providers only
        try:
            stated, currency = nc_cli.pool_price(doc)
            price = None if currency == "ucr" or stated is None else _price(stated, f"{where} price_usd_h")
        except (TypeError, ValueError, KeyError, IndexError):
            price = None
    row = FleetPool(name=out["name"], source_name=source, provider=provider_kind(doc), shard_base=f"{base}-i{k}",
                    shard_limit=int(limit), run_prefix=prefix, engine=engine, rollout_quant=quant,
                    target=target if isinstance(target, int) and not isinstance(target, bool) else None,
                    usd_per_h=price)
    return out, row, port_share_key(spec), compiled


def _file_sha256(path: str) -> str:
    import hashlib  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def miles_ledger(trainer_argv: list[str], pools: list[tuple[str, dict[str, Any]]]
                 ) -> dict[str, Any] | None:
    """The store dispenser settings the fleet's Miles samplers share with its pools
    (module doc): the pools' ``prompts.dispense`` (``nc_cli.pool_dispense``), None when
    no pool claims from a store ledger (every sampler keeps its data slot). ``pools``:
    ``(compiled name, compiled pool config)``. Raises ValueError (``run config: ...``)
    when the Miles samplers cannot join the ledger: pools with different settings, a
    hold-out file (a path on the pools' nodes), or a ``--data`` that is not the pools'
    prompt file (the ledger's plan pins the file's rows and SHA-256)."""
    from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415

    found = [(name, nc_cli.pool_dispense(doc), doc) for name, doc in pools]
    found = [(name, dispense, doc) for name, dispense, doc in found if dispense is not None]
    if not found:
        return None
    settings = {json.dumps({k: v for k, v in d.items() if k != "store"}, sort_keys=True) for _n, d, _doc in found}
    if len(settings) > 1:
        raise _err(f"pools {[n for n, _d, _doc in found]} claim from store ledgers with different settings "
                   f"({sorted(settings)}): the fleet's Miles samplers share one ledger setting with the pools; give "
                   "every pool the same prompts.dispense")
    config = {k: v for k, v in found[0][1].items() if k != "store"}
    config.setdefault("backend", "store")
    if config.get("holdout"):
        raise _err("a fleet with Miles samplers: the pools' prompts.dispense.holdout is a file on the pools' nodes, "
                   "which a Miles sampler cannot read; give the hold-out ids inline (holdout_ids)")
    data = _argv_value(trainer_argv, "--data")
    why = ("the fleet's Miles samplers claim their rows from the store ledger the pools claim from, and the "
           "ledger pins the rows and SHA-256 of the pools' prompt file: fleet.trainer_argv --data must be that "
           "file (a local JSONL with the same bytes as each pool's prompts_file)")
    from pathlib import Path  # noqa: PLC0415

    if not data or not Path(data).expanduser().is_file():
        raise _err(f"fleet.trainer_argv --data {data!r} is not a local prompt file: {why}. Without it a Miles sampler "
                   "and the pools would give out the same prompt in one epoch")
    digest = _file_sha256(data)
    for name, _dispense, doc in found:
        local = doc.get("prompts_file")
        if local and Path(str(local)).expanduser().is_file() and _file_sha256(str(local)) != digest:
            raise _err(f"pool {name}: prompts_file {local} is not the file fleet.trainer_argv --data {data} names "
                       f"(SHA-256 differs): {why}")
    return config


def training_form(trainer_argv: list[str]) -> dict[str, Any]:
    """The trainers' training form and sync route, read from ``fleet.trainer_argv``
    through the ``yeto launch`` parser and the launcher's own routing
    (``sync_route.apply_sync_route``, then ``syncer.rl_full_parameter_async``):
    ``backend``, ``tuning``, ``expert_selection``, ``requested`` / ``preset`` /
    ``rounds`` (the route), ``form`` and ``full_parameter_async``. Empty when the
    argv does not parse as an RL launch (``yeto launch --dry-run`` names why)."""
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args([str(a) for a in trainer_argv])
    if args is None:
        return {}
    from yeto.launch.syncer import rl_full_parameter_async  # noqa: PLC0415
    from yeto.rl.engine.sync_route import SyncRouteError, apply_sync_route  # noqa: PLC0415

    out: dict[str, Any] = {"backend": str(getattr(args, "rl_backend", None) or "miles"),
                           "tuning": str(getattr(args, "tuning", "lora")),
                           "expert_full_count": int(getattr(args, "expert_full_count", 0) or 0),
                           "expert_selection": bool(getattr(args, "rl_expert_selection", None))}
    try:
        route = apply_sync_route(args)
    except SyncRouteError as error:
        return {**out, "error": str(error)}
    if route is None:
        return out
    out.update({"requested": route.requested, "preset": route.preset, "transport": route.transport,
                "rounds": getattr(args, "rl_sync_full_parameter_rounds", None),
                "full_parameter_async": bool(rl_full_parameter_async(args))})
    out["form"] = ("partial experts" if out["tuning"] == "full" and out["expert_selection"]
                   else "full parameters" if out["tuning"] == "full" else "LoRA")
    return out


def fleet_backends(cfg: dict[str, Any]) -> dict[str, str]:
    """The training backend of every island of the fleet, by a label naming it: each
    trainer island (``fleet.trainer_argv`` ``--rl-backend``; an island sets no backend of
    its own) and each Miles sampler island (Miles). Empty when the trainer argv does not
    parse (:func:`form_problem` names that)."""
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    args = trainer_args([str(a) for a in section.get("trainer_argv") or ()])
    islands = cfg.get("islands") if isinstance(cfg.get("islands"), list) else []
    if args is None or not islands:
        return {}
    run_backend = str(getattr(args, "rl_backend", None) or "miles")
    out: dict[str, str] = {}
    for k, island in enumerate(islands):
        name = island.get("name") if isinstance(island, dict) else None
        out[f"island {k} ({name})"] = run_backend
        for entry in (island.get("samplers") if isinstance(island, dict) else None) or ():
            if is_miles_entry(entry):
                out[f"Miles sampler {entry.get('name')} of island {k} ({name})"] = MILES_KIND
    return out


def form_problem(form: dict[str, Any], *, backends: dict[str, str] | None = None) -> str | None:
    """Why a fleet with this :func:`training_form` cannot run (None: it can).

    A fleet's samplers are external engine pools that load each release whole and
    in place (vLLM ``reload_weights`` / SGLang ``update_weights_from_disk``); LoRA
    releases to external engines are out of scope. Every island runs one backend
    (``backends``: :func:`fleet_backends`; None: only the trainer argv's), and that
    backend declares the fleet's form as a capability (``fleet_full_parameter``,
    ``fleet_partial_experts``: ``yeto.rl.engine.backends.capabilities``)."""
    if not form:
        return ("fleet.trainer_argv does not parse as an RL yeto launch argv (--training-mode rl ...); "
                "`yeto launch --dry-run` names the argument")
    if form.get("error"):
        return f"fleet.trainer_argv: {form['error']}"
    if backends and len(set(backends.values())) > 1:
        by = ", ".join(f"{where}: {backend}" for where, backend in backends.items())
        return (f"the islands of a fleet run one training backend, these run {len(set(backends.values()))} "
                f"({by}): Miles and verl islands do not share a syncer session (a verl layout carries no shard "
                "identity, a Miles layout names every rank's shard; mixing them needs a shape-free layout, which "
                "a fleet does not support). Run every island on one backend")
    from yeto.rl.engine import backend_identity as bi  # noqa: PLC0415
    from yeto.rl.engine import backends as registry  # noqa: PLC0415

    backend = form["backend"]
    try:
        declared = registry.capabilities(backend)
    except registry.UnknownBackend as error:
        return f"fleet.trainer_argv: {error}"
    if bi.FLEET_FULL_PARAMETER not in declared:
        able = [name for name in registry.registered() if bi.FLEET_FULL_PARAMETER in registry.capabilities(name)]
        return (f"a fleet trainer runs on a backend that declares {bi.FLEET_FULL_PARAMETER} ({', '.join(able)}); "
                f"--rl-backend {backend} does not: its synced trainers cannot release to external sampler pools")
    if form["tuning"] == "full" and form.get("expert_selection") and bi.FLEET_PARTIAL_EXPERTS not in declared:
        able = [name for name in registry.registered() if bi.FLEET_PARTIAL_EXPERTS in registry.capabilities(name)]
        return (f"a partial-experts fleet (--rl-expert-selection) needs a backend that declares "
                f"{bi.FLEET_PARTIAL_EXPERTS} ({', '.join(able)}): --rl-backend {backend} does not train partial "
                "experts; train full parameters (drop --rl-expert-selection) or use "
                + " / ".join(f"--rl-backend {name}" for name in able))
    if form["tuning"] == "lora" or form.get("expert_full_count"):
        return ("a LoRA fleet (--tuning lora or --expert-full-count) is not supported: the islands' samplers are "
                "external vLLM / SGLang pools, which load full-parameter releases in place; LoRA releases to "
                "external engines wait for upstream support. Train full parameters or partial experts "
                "(--tuning full [--rl-expert-selection FILE]) with --rl-sync-preset decoupled")
    if not form.get("full_parameter_async"):
        return (f"a fleet merges its trainers with asynchronous full-parameter rounds: --rl-sync-preset "
                f"{form.get('requested')} runs {form.get('preset')}"
                + (f" with {form.get('rounds')} rounds" if form.get("rounds") else "")
                + "; use --rl-sync-preset decoupled (routed to full-parameter async rounds)")
    return None


# a trainer island's prompt row width (its pools' widest first prompt): --rl-island-buffer-max-prompt-len
# K:N, written by the compile for a backend whose buffer trainer sizes its prompt rows
PROMPT_LEN_FLAG = "--rl-island-buffer-max-prompt-len"


def sizes_prompt_rows(trainer_argv: list[str]) -> bool:
    """Whether the trainers' backend sizes its buffer trainer's prompt rows from its pools
    (its ``launch_flags.buffer_prompt_width``: verl); a Miles trainer packs rows by tokens."""
    backend = _argv_value([str(a) for a in trainer_argv], "--rl-backend") or MILES_KIND
    from yeto.rl.engine import backends  # noqa: PLC0415

    try:
        return getattr(backends.module("launch_flags", backend), "buffer_prompt_width", None) is not None
    except backends.UnknownBackend:
        return False


def pool_max_prompt_len(pool_doc: dict[str, Any]) -> int | None:
    """A compiled pool's ``agentic.max_prompt_len`` (its rendered sampler spec), None if unset."""
    spec, _engine, _quant = _spec_facts(pool_doc)
    agentic = (spec or {}).get("agentic")
    value = agentic.get("max_prompt_len") if isinstance(agentic, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def island_prompt_len(lens: list[int | None]) -> int | None:
    """An island's prompt row width from its pools: the widest bound, None when any pool
    bounds none (the trainer then takes its sequence budget)."""
    if not lens or any(n is None for n in lens):
        return None
    return max(n for n in lens if n is not None)


def prompt_width_problem(cfg: dict[str, Any], plan: FleetPlan) -> str | None:
    """Why an island's prompt rows do not fit its trainer (None: they do): the backend's
    ``launch_flags.buffer_prompt_width`` over the trainer's ``--seq-len`` and
    ``--rollout-max-response-len`` and the island's pools' bound (or the sequence
    budget), naming the island; the island launch refuses the same before it starts."""
    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    trainer_argv = [str(a) for a in section.get("trainer_argv") or ()]
    if not sizes_prompt_rows(trainer_argv):
        return None
    from yeto.rl.engine import backends  # noqa: PLC0415
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args(trainer_argv)
    if args is None:
        return None
    width = backends.module("launch_flags", args.rl_backend).buffer_prompt_width
    for i in plan.islands:
        try:
            width(seq_len=int(args.seq_len), response_len=int(args.rollout_max_response_len),
                  pool_max_prompt_len=i.buffer_max_prompt_len)
        except ValueError as error:
            pools = ", ".join(p.name for p in i.pools)
            return f"island {i.index} ({i.name}), pools {pools}: {error}"
    return None


def compaction_problem(cfg: dict[str, Any], plan: FleetPlan,
                       pools: Iterable[tuple[str, dict[str, Any]]] = ()) -> str | None:
    """Why a fleet whose backend cannot train compacted trials cannot run (None: it can):
    a pool's sampler spec with ``agentic.compaction`` (each segment of a trial is its own
    sample) or a trainer running CompactionRL (``advantage.gae_variant
    cross_segment_per_sample``) on a backend that does not declare ``compaction_segments``
    (verl: one sample is one token stream). ``pools``: ``(compiled pool name, its loaded
    pool config)``. The trainers' run-time stop on a compacted group
    (``trainer.compaction_round``) stays as the backstop."""
    form = plan.sync or {}
    backend = form.get("backend")
    if not backend:
        return None
    from yeto.rl.engine import backend_identity as bi  # noqa: PLC0415
    from yeto.rl.engine import backends as registry  # noqa: PLC0415

    try:
        if bi.COMPACTION_SEGMENTS in registry.capabilities(backend):
            return None
    except registry.UnknownBackend:
        return None  # form_problem names it
    able = " / ".join(f"--rl-backend {name}" for name in registry.registered()
                      if bi.COMPACTION_SEGMENTS in registry.capabilities(name))
    why = (f"a {backend} trainer cannot train the per-segment samples of a compacted trial as one trial (each "
           f"sample is its own token stream); use {able} for CompactionRL, or turn compaction off")
    section = cfg.get("fleet") if isinstance(cfg.get("fleet"), dict) else {}
    from yeto.rl.up_consistency import trainer_args  # noqa: PLC0415

    args = trainer_args([str(a) for a in section.get("trainer_argv") or ()])
    if args is not None:
        from yeto.launch.harness_env import _launch_algorithm  # noqa: PLC0415

        advantage = (_launch_algorithm(args) or {}).get("advantage") or {}
        if advantage.get("gae_variant") == "cross_segment_per_sample":
            return (f"fleet.trainer_argv runs CompactionRL (advantage.gae_variant cross_segment_per_sample) on "
                    f"--rl-backend {backend}: {why}")
    from yeto.rl.up_consistency import effective_spec  # noqa: PLC0415

    for name, pool_cfg in pools:
        spec = effective_spec(pool_cfg) or {}
        agentic = spec.get("agentic") if isinstance(spec.get("agentic"), dict) else {}
        if agentic.get("compaction") is not None:
            island = plan.island_of_pool(str(name))
            where = f"island {island.index} ({island.name}) " if island is not None else ""
            return (f"{where}pool {name}: its sampler spec sets agentic.compaction, and the fleet trains on "
                    f"--rl-backend {backend}: {why}")
    return None


# ---------------------------------------------------------------------- compile

AUTO_BACKLOG_KEY = "auto_backlog"  # yeto.launch.nc.cli.AUTO_BACKLOG_KEY


def island_auto_backlog(rows: list[FleetIsland], argv: list[str], samplers: list[dict[str, Any]],
                        docs: dict[str, dict[str, Any]]) -> None:
    """Give the pools of each island the ``max_backlog_groups: "auto"`` cap of that island
    (:data:`AUTO_BACKLOG_KEY` on the pool, ``{"groups", "why"}``): each island's trainer is
    the buffer trainer of its own pools, so the cap is the single trainer's formula with
    the island's release cadence and its pools' target samplers
    (:func:`yeto.rl.up_consistency.island_auto_backlog`). Without it a fleet's "auto"
    samplers took the fallback (256 groups each), wrote hundreds of base-version groups
    before the trainers started, and the trainers dropped them all as over age. A pool
    whose spec sets a number, or an island the formula does not cover, gets nothing."""
    from yeto.rl import up_consistency  # noqa: PLC0415

    by_name = {out["name"]: out for out in samplers}
    for island in rows:
        names = [p.name for p in island.pools]
        auto = up_consistency.island_auto_backlog(argv, [docs[n] for n in names],
                                                  release_every=island.release_every)
        if auto is None:
            continue
        why = f"island {island.index} ({island.name}): {auto['why']}"
        for name in names:
            if up_consistency.is_auto_backlog(up_consistency.effective_spec(docs[name])):
                by_name[name][AUTO_BACKLOG_KEY] = {"groups": int(auto["groups"]), "why": why}


def compile_fleet(cfg: dict[str, Any], *, strict_breakers: bool = True) -> tuple[dict[str, Any], FleetPlan]:
    """The single-trainer run config of an island fleet and its plan (module doc).
    Raises ValueError (``run config: ...``) on a bad ``islands`` / ``fleet`` section.
    The input is not changed; the output keeps ``islands`` and ``fleet`` as they were
    and adds ``trainer``, ``samplers`` and :data:`PLAN_KEY`."""
    check_exclusive(cfg)
    if "islands" not in cfg:
        raise _err("compile_fleet: no 'islands' section")
    store = str(cfg["store"])
    trainer_argv, dispense = _fleet_section(cfg)
    named = _argv_value(trainer_argv, "--rl-checkpoint-store")
    if named is None:
        raise _err("fleet.trainer_argv needs --rl-checkpoint-store <store>: each trainer releases to and reads "
                   "from <store>/trainer<k> of it")
    if _norm_store(named) != _norm_store(store):
        raise _err(f"fleet.trainer_argv: --rl-checkpoint-store {named} != the run config's store {store}: the "
                   "islands' pools write under the run store")
    islands = cfg["islands"]
    if not isinstance(islands, list) or not islands:
        raise _err("'islands' must be a non-empty list (one entry per island: name, trainer, samplers)")
    from yeto.rl.engine.store_dispense import fleet_dispense_store  # noqa: PLC0415

    dispense_store = fleet_dispense_store(store) if dispense == "fleet" else None
    rows: list[FleetIsland] = []
    samplers: list[dict[str, Any]] = []
    shared: list[tuple[str, tuple[str, ...] | None, int]] = []
    pool_docs: list[tuple[str, dict[str, Any]]] = []
    names: set[str] = set()
    # Miles samplers come after every trainer in the launch: count the trainers first
    launch_island = len(islands)
    sized = sizes_prompt_rows(trainer_argv)
    for k, island in enumerate(islands):
        prompt_lens: list[int | None] = []
        where = f"islands[{k}]"
        if not isinstance(island, dict):
            raise _err(f"{where} must be an object (name, trainer, samplers)")
        name = island.get("name")
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise _err(f"{where}: name must be a word of letters, digits, '.', '_' or '-' (got {name!r})")
        where = f"islands[{k}] ({name})"
        if name in names:
            raise _err(f"{where}: island names repeat ({name!r})")
        names.add(name)
        unknown = set(island) - ISLAND_KEYS
        if unknown:
            raise _err(f"unknown {where} keys {sorted(unknown)} (known: {sorted(ISLAND_KEYS)})")
        gpu, price, flags = _island_trainer(island, where)
        entries = island.get("samplers")
        if not isinstance(entries, list) or not entries:
            raise _err(f"{where}: samplers must be a non-empty list of pools or Miles samplers (every island has "
                       "its own)")
        pools = []
        miles = []
        for j, entry in enumerate(entries):
            if is_miles_entry(entry):
                source, sampler_gpu, sampler_price = _miles_sampler(entry, k, j, where)
                miles.append(FleetMilesSampler(name=f"i{k}-{source}", source_name=source, gpu=sampler_gpu,
                                               usd_per_h=sampler_price, launch_island=launch_island))
                launch_island += 1
                continue
            out, row, ports, doc = _pool(entry, k, j, where, dispense_store, store)
            samplers.append(out)
            pools.append(row)
            shared.append((row.name, ports, row.shard_limit))
            pool_docs.append((row.name, doc))
            if sized:
                prompt_lens.append(pool_max_prompt_len(doc))
        seen = [p.name for p in pools] + [m.name for m in miles]
        if len(set(seen)) != len(seen):
            raise _err(f"{where}: pool names repeat {seen} (Miles samplers included); set 'name' on each")
        parts = [price, *(m.usd_per_h for m in miles)]
        total = None if all(p is None for p in parts) else float(sum(p or 0.0 for p in parts))
        rows.append(FleetIsland(index=k, name=name, gpu=gpu, usd_per_h=total, store=substore_uri(store, substore(k)),
                                release_every=flags.get("release_every"),
                                max_policy_age=flags.get("max_policy_age"), pools=tuple(pools),
                                miles_samplers=tuple(miles), trainer_usd_per_h=price,
                                sample_wait_s=flags.get("sample_wait_s"),
                                buffer_max_prompt_len=island_prompt_len(prompt_lens)))
    offsets = share_offsets(shared)
    for out in samplers:
        if out["name"] in offsets:  # a port range several pools share: one block of it each
            out["sampler"] = dict(out["sampler"], shard_offset=offsets[out["name"]])
    if not samplers:
        raise _err("an island fleet needs at least one external pool: `yeto run up` drives the pools (a fleet of "
                   "Miles samplers only is a `yeto launch` synced group with --rl-sampler-follow trainer)")
    ledger = miles_ledger(trainer_argv, pool_docs) if any(i.miles_samplers for i in rows) else None
    if ledger is not None:  # every Miles sampler claims from its island's pools' ledger
        from yeto.rl.engine.store_dispense import DEFAULT_NAMESPACE  # noqa: PLC0415

        namespace = str(ledger.get("namespace") or DEFAULT_NAMESPACE)
        rows = [dataclasses.replace(i, miles_samplers=tuple(
            dataclasses.replace(m, dispense_store=dispense_store or i.store, dispense_namespace=namespace)
            for m in i.miles_samplers)) for i in rows]
    miles_rows = [(i, m) for i in rows for m in i.miles_samplers]
    argv = [*trainer_argv, "--gpu", ",".join([i.gpu for i in rows] + [m.gpu for _i, m in miles_rows]),
            "--rl-sample-role", "trainer", "--rl-sampler-follow", "trainer"]
    try:  # every trainer island's, unless the island sets its own
        argv += _trainer_section_argv(cfg, trainer_argv)
    except ValueError as error:
        if strict_breakers:
            raise
        # run down / status: never blocked by a breakers value (yeto run up refuses it)
        print(f"WARNING: {error}: not written into the trainer argv (yeto run up refuses this config)",
              file=sys.stderr, flush=True)
    launch, max_islands = launch_mode(cfg)
    # per_island: the launches resolve their membership themselves (per_island_roster), so no
    # option of a synced group's roster (its lease, a count quorum) reaches the syncer or the islands
    roster = (per_island_roster(cfg) if launch == LAUNCH_PER_ISLAND
              else roster_of(cfg, trainer_argv, argv, len(rows)))
    argv += list(roster.argv)
    for i in rows:
        if i.pools:  # an island with Miles samplers only names no pool shard
            argv += ["--rl-island-buffer-shards", f"{i.index}:{','.join(i.pool_shards())}"]
    for i, m in miles_rows:  # each Miles sampler island follows its own island's trainer
        argv += ["--rl-island-sample-role", f"{m.launch_island}:sampler",
                 "--rl-island-sampler-follows", f"{m.launch_island}:{i.learner}"]
        if m.dispense_store:
            argv += ["--rl-island-dispense-store", f"{m.launch_island}:{m.dispense_store}"]
    if ledger is not None:
        argv += ["--rl-store-dispense", json.dumps(ledger, sort_keys=True, separators=(",", ":"))]
    for i in rows:
        if i.release_every is not None:
            argv += ["--rl-island-release-every", f"{i.index}:{i.release_every}"]
        if i.max_policy_age is not None:
            argv += ["--rl-island-max-policy-age", f"{i.index}:{i.max_policy_age}"]
        if i.sample_wait_s is not None:
            argv += ["--rl-island-sample-wait-s", f"{i.index}:{i.sample_wait_s:g}"]
        if i.buffer_max_prompt_len is not None:
            argv += [PROMPT_LEN_FLAG, f"{i.index}:{i.buffer_max_prompt_len}"]
    island_auto_backlog(rows, argv, samplers, dict(pool_docs))
    from yeto.breakers.fleet import syncer_placement  # noqa: PLC0415

    prices = [i.usd_per_h for i in rows]
    syncer = cfg.get("syncer") if isinstance(cfg.get("syncer"), dict) else {}
    service = syncer_service(cfg["fleet"])
    plan = FleetPlan(run=str(cfg["run"]), store=store, dispense=dispense, islands=tuple(rows),
                     dispense_store=dispense_store, sync=training_form(trainer_argv),
                     syncer_usd_per_h=_price(syncer.get("usd_per_h"), "syncer.usd_per_h"),
                     syncer_placement=syncer_placement(argv, service=service).to_json(),
                     fixed_roster=roster.fixed, min_islands=roster.min_islands,
                     island_lease_s=roster.lease_s, roster_reason=roster.reason,
                     launch=launch, max_islands=max_islands, waiting_absent_after_s=waiting_absent_after_s(cfg),
                     controller_poll_s=controller_poll_s(argv), syncer_service=service)
    out = copy.deepcopy({k: v for k, v in cfg.items() if k != PLAN_KEY})
    out["trainer"] = {"argv": argv, "usd_per_h": None if all(p is None for p in prices)
                      else float(sum(p or 0.0 for p in prices))}
    if plan.per_island:
        from yeto.rl import island_launches  # noqa: PLC0415

        if miles_rows:
            raise _err("fleet.launch per_island: an island with Miles samplers (kind miles) is one launch with its "
                       "trainer; give its islands external pools only, or use fleet.launch synced_group")
        # what run up starts as its trainer is the fleet's syncer alone; each island is its own
        # launch, started by the trainer worker from the desired state (island_launches)
        out["trainer"] = {"argv": island_launches.syncer_argv(argv, plan), "usd_per_h": None}
    out["samplers"] = samplers
    out[PLAN_KEY] = plan
    return out, plan
