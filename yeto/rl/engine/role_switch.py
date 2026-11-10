"""Role switch of one island between engine sampler and full-parameter trainer.

The convertible pool (:mod:`.convertible`, :mod:`.convertible_remote`) converts
a *Miles colocated* island in place: the engines and the trainer are one Ray
job, the handoff moves GPU memory between them and the trained weights reach the
engines through Miles' own update path. This module is the other shape the
Kimi-K3.1 run needs: the island starts as an **engine sampler** (``python -m
yeto.rl.engine.engine_sampler``, an SGLang server the sampler drives through
the :class:`~.sampler_engine.SamplerEngine` port, writing groups to the object
store buffer) and later becomes a **trainer** (a Miles trainer island,
``--rl-sample-role trainer --rl-single-island-no-sync --tuning full``, reading
the same buffer and releasing whole checkpoints into the store). The two roles
are two processes of the same container, never alive at once, so the switch is:

1. the sampler loop takes no new round once :class:`SwitchTrigger` fires (a
   wall-clock budget, a number of groups written, the local marker file of
   :data:`SWITCH_MARKER_ENV` -- the shape of the spot reclaim marker -- or a
   marker object in the run store, ``convertible/<node>/switch.json``); the
   round under way is written whole and its index committed by the sampler
   entry; the engine process is stopped (``engine.stop``: SIGTERM to the
   server's process group, then kill) and the device memory is checked free
   (:class:`DeviceIdle`);
2. the role record ``convertible/<node>/role.json`` (:class:`RoleState`) says
   ``trainer`` with the incarnation, the groups this sampler wrote, the buffer
   end it saw and the consumed watermark the trainer starts from (the newest
   cut's, :class:`~.trainer_remote.StoreCutLog`; 0 for a first trainer);
3. the trainer phase runs (``trainer(state, plan)``: the adapter layer runs the
   island entry as a subprocess; a test runs a reader over the buffer and
   releases) and reports the version it released and the watermark it reached;
4. back to sampler (optional, ``then_sampler``): the sampler spec gets the
   newest release as its base weights (``base_dir`` / ``base_version``, gated by
   the spec's ``expert_selection_sha256`` against the release manifest) and the
   loop above repeats until ``max_cycles``.

Neutral core: no backend, no cloud SDK. The trainer process and the device
probe come from the adapter layer through the spec (``module:callable``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .convertible_remote import CONVERTIBLE_DIR, write_retrying
from .release_overlay import is_partial_release, overlay_release
from .trainer_remote import StoreCutLog, StoreFiles

SWITCH_MARKER_ENV = "YETO_ROLE_SWITCH_MARKER"
DEFAULT_SWITCH_MARKER = "/tmp/yeto-role-switch-requested"
ROLE_SWITCH = "role_switch"  # the spec section
SAMPLER, TRAINER = "sampler", "trainer"
ROLES = (SAMPLER, TRAINER)
SWITCH_EVENT = "rl_role_switch"
PHASE_EVENT = "rl_role_phase"
DEVICE_IDLE_TIMEOUT_S = 120.0
DEVICE_IDLE_POLL_S = 2.0
# device memory still held after the engine stopped, per device, that counts as free
DEVICE_IDLE_USED_MIB = 1024


class RoleSwitchError(RuntimeError):
    """The switch cannot proceed; the message names the step."""


def role_path(node: str) -> str:
    return f"{CONVERTIBLE_DIR}/{node}/role.json"


def switch_marker_path(node: str) -> str:
    return f"{CONVERTIBLE_DIR}/{node}/switch.json"


def switch_marker(environ: Mapping[str, str]) -> Path:
    return Path(environ.get(SWITCH_MARKER_ENV) or DEFAULT_SWITCH_MARKER)


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------
@dataclass
class SwitchTrigger:
    """``should_stop`` of the sampler loop during the sampler phase. Fires when any
    of its conditions holds (None: not watched): ``after_seconds`` since
    :meth:`start`; ``after_groups`` groups written by this phase (``groups()``);
    the local ``marker`` file exists; the store marker exists (``files`` and
    ``store_marker``). ``reason`` is the condition that fired first."""

    after_seconds: float | None = None
    after_groups: int | None = None
    marker: Path | None = None
    files: StoreFiles | None = None
    store_marker: str | None = None
    groups: Callable[[], int] = lambda: 0
    clock: Callable[[], float] = time.monotonic
    started_at: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.after_seconds is not None and float(self.after_seconds) <= 0:
            raise RoleSwitchError("role_switch.after_seconds must be > 0")
        if self.after_groups is not None and int(self.after_groups) < 1:
            raise RoleSwitchError("role_switch.after_groups must be >= 1")
        if self.store_marker is not None and self.files is None:
            raise RoleSwitchError("a store marker needs the store files")
        if all(v is None for v in (self.after_seconds, self.after_groups, self.marker, self.store_marker)):
            raise RoleSwitchError("role_switch needs at least one trigger: after_seconds, after_groups, "
                                  "marker or store_marker")

    def start(self) -> "SwitchTrigger":
        self.started_at = self.clock()
        return self

    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else self.clock() - self.started_at

    def __call__(self) -> bool:
        if self.reason is not None:
            return True
        if self.after_seconds is not None and self.elapsed() >= float(self.after_seconds):
            self.reason = f"after_seconds:{float(self.after_seconds):g}"
        elif self.after_groups is not None and int(self.groups()) >= int(self.after_groups):
            self.reason = f"after_groups:{int(self.after_groups)}"
        elif self.marker is not None and self.marker.exists():
            self.reason = f"marker:{self.marker}"
        elif self.store_marker is not None and self.files is not None and self.files.read(self.store_marker):
            self.reason = f"store_marker:{self.store_marker}"
        return self.reason is not None


def write_switch_marker(files: StoreFiles, node: str, *, by: str = "operator",
                        clock: Callable[[], float] = time.time) -> None:
    """Operator side: ask node ``node`` to switch at its next round boundary."""
    files.write(switch_marker_path(node), json.dumps({"to": TRAINER, "by": by, "time": round(clock(), 3)},
                                                     sort_keys=True).encode())


# ---------------------------------------------------------------------------
# Role record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoleState:
    """``convertible/<node>/role.json``: which role the island is in and what the
    last phase left. ``incarnation`` counts role phases (1 = the first sampler);
    ``buffer_end`` is the index length the sampler committed; ``consumed_watermark``
    the position the next trainer reads from; ``policy_version`` the version the
    engines served (sampler) or the trainer released (trainer)."""

    node: str
    role: str
    incarnation: int
    at: float
    reason: str | None = None
    sampler_rounds: int = 0
    sampler_groups: int = 0
    buffer_end: int = 0  # merged index length over every island's shard
    shard_ends: Mapping[str, int] = field(default_factory=dict)  # per island shard
    consumed_watermark: int = 0  # merged position the next trainer reads from
    shard_watermarks: Mapping[str, int] = field(default_factory=dict)  # the same, per shard
    policy_version: int = 0
    released_seq: int | None = None
    release_manifest: Mapping[str, Any] | None = None
    phase_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise RoleSwitchError(f"role {self.role!r} is not one of {ROLES}")

    def record(self) -> dict[str, Any]:
        out = asdict(self)
        out["shard_ends"], out["shard_watermarks"] = dict(self.shard_ends), dict(self.shard_watermarks)
        out["release_manifest"] = None if self.release_manifest is None else dict(self.release_manifest)
        return out


def write_role_state(files: StoreFiles, state: RoleState) -> None:
    write_retrying(files, role_path(state.node), json.dumps(state.record(), sort_keys=True).encode())


def read_role_state(files: StoreFiles, node: str) -> RoleState | None:
    raw = files.read(role_path(node))
    return None if not raw else RoleState(**json.loads(raw))


# ---------------------------------------------------------------------------
# Device idle check after the engine stopped
# ---------------------------------------------------------------------------
def nvidia_smi_used_mib(run: Callable[..., Any] = subprocess.run) -> list[int] | None:
    """Used memory per NVIDIA device in MiB; None when ``nvidia-smi`` is absent."""
    try:
        out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                  capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(out, "returncode", 1) != 0:
        return None
    return [int(float(line.strip())) for line in str(out.stdout).splitlines() if line.strip()]


@dataclass
class DeviceIdle:
    """Wait until every device holds at most ``used_mib`` after the engine process
    ended (a killed SGLang server frees its memory a moment after the kill). ``probe``
    returns the used MiB per device, or None when it cannot tell (CPU, no tool),
    which passes: the check never blocks a node without a device tool."""

    probe: Callable[[], list[int] | None] = nvidia_smi_used_mib
    used_mib: int = DEVICE_IDLE_USED_MIB
    timeout_s: float = DEVICE_IDLE_TIMEOUT_S
    poll_s: float = DEVICE_IDLE_POLL_S
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def wait(self) -> dict[str, Any]:
        deadline = self.clock() + self.timeout_s
        while True:
            used = self.probe()
            if used is None:
                return {"checked": False}
            if all(u <= self.used_mib for u in used):
                return {"checked": True, "used_mib": list(used)}
            if self.clock() >= deadline:
                raise RoleSwitchError(f"device memory still held after the engine stopped: {used} MiB "
                                      f"(limit {self.used_mib} MiB per device, waited {self.timeout_s:g} s)")
            self.sleep(self.poll_s)


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoleSwitchPlan:
    """The ``role_switch`` section of the spec. ``node``: the island's name (its
    ``convertible/<node>/`` channel); triggers as in :class:`SwitchTrigger`;
    ``trainer``: the trainer phase (``{"command": [argv...], "env": {...}}`` for the
    adapter's subprocess trainer, or ``{"run": "module:callable"}``); ``then_sampler``:
    after a trainer phase, run the sampler again on the released weights;
    ``max_cycles``: sampler->trainer cycles (default 1); ``device_idle``: the memory
    limit and timeout of :class:`DeviceIdle` (``false`` to skip)."""

    node: str
    after_seconds: float | None = None
    after_groups: int | None = None
    marker: bool = True
    store_marker: bool = True
    trainer: Mapping[str, Any] = field(default_factory=dict)
    then_sampler: bool = False
    max_cycles: int = 1
    device_idle: Mapping[str, Any] | None = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "RoleSwitchPlan":
        section = spec.get(ROLE_SWITCH)
        if not isinstance(section, Mapping):
            raise RoleSwitchError(f"the spec has no {ROLE_SWITCH} section")
        node = section.get("node") or spec.get("sampler_id")
        if not node:
            raise RoleSwitchError(f"{ROLE_SWITCH}.node (or sampler_id) is required")
        known = {"node", "after_seconds", "after_groups", "marker", "store_marker", "trainer", "then_sampler",
                 "max_cycles", "device_idle"}
        unknown = set(section) - known
        if unknown:
            raise RoleSwitchError(f"{ROLE_SWITCH}: unknown keys {sorted(unknown)}")
        trainer = section.get("trainer") or {}
        if not isinstance(trainer, Mapping) or not (trainer.get("command") or trainer.get("run")):
            raise RoleSwitchError(f"{ROLE_SWITCH}.trainer needs 'command' (argv) or 'run' (module:callable)")
        idle = section.get("device_idle", {})
        if idle is False:
            idle = None
        elif not isinstance(idle, Mapping):
            raise RoleSwitchError(f"{ROLE_SWITCH}.device_idle is a mapping or false")
        plan = cls(node=str(node),
                   after_seconds=None if section.get("after_seconds") is None else float(section["after_seconds"]),
                   after_groups=None if section.get("after_groups") is None else int(section["after_groups"]),
                   marker=bool(section.get("marker", True)), store_marker=bool(section.get("store_marker", True)),
                   trainer=dict(trainer), then_sampler=bool(section.get("then_sampler", False)),
                   max_cycles=int(section.get("max_cycles", 1)), device_idle=None if idle is None else dict(idle))
        if plan.max_cycles < 1:
            raise RoleSwitchError(f"{ROLE_SWITCH}.max_cycles must be >= 1")
        if plan.after_seconds is None and plan.after_groups is None and not plan.marker and not plan.store_marker:
            raise RoleSwitchError(f"{ROLE_SWITCH} needs a trigger")
        return plan


@dataclass(frozen=True)
class BufferView:
    """The trainer's view of the buffer from this node: the merged index length over
    every island's shard and the records of each shard below it (or below a cursor)."""

    end: int
    shards: Mapping[str, int] = field(default_factory=dict)
    shards_below: Callable[[int], Mapping[str, int]] = lambda cursor: {}


@dataclass(frozen=True)
class SamplerPhaseReport:
    rounds: int
    groups: int
    buffer_end: int
    shard_ends: Mapping[str, int]
    policy_version: int
    reason: str | None
    seconds: float
    device: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerPhaseReport:
    """What a trainer phase leaves: the version it released (0: none), the release
    record's seq and manifest, the buffer position it consumed up to."""

    policy_version: int
    released_seq: int | None = None
    release_manifest: Mapping[str, Any] | None = None
    consumed_watermark: int | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class TrainerContext:
    """What a trainer phase gets besides the role record: the run store files, the
    release store, a merged-shards buffer view and the process environment."""

    files: StoreFiles
    releases: Any
    buffer: Callable[[], BufferView]
    environ: Mapping[str, str]


TrainerPhase = Callable[[RoleState, RoleSwitchPlan, TrainerContext], TrainerPhaseReport]


# ---------------------------------------------------------------------------
# Sampler spec after a trainer phase
# ---------------------------------------------------------------------------
def sampler_spec_on_release(spec: Mapping[str, Any], releases: Any) -> dict[str, Any]:
    """The sampler spec whose engine starts on the newest release (its local
    directory, ``PolicyReleaseStore.resolve``) at that release's version. The spec's
    ``expert_selection_sha256`` must match the release manifest (the gate the
    follower applies to later releases). No release: the spec as given."""
    latest = releases.latest()
    if latest is None:
        return dict(spec)
    expected = spec.get("expert_selection_sha256")
    manifest = dict(getattr(latest, "manifest", None) or {})
    if expected is not None and manifest.get("expert_selection_sha256") != expected:
        raise RoleSwitchError(f"release r{latest.seq}-v{latest.version} carries selection "
                              f"{manifest.get('expert_selection_sha256')}, the spec expects {expected}")
    directory = Path(releases.resolve(latest))
    base = ((spec.get("engine") or {}).get("server") or {}).get("base_dir")
    if is_partial_release(manifest) and base:
        # the release holds the trained experts only: a server engine starts on the spec's
        # base checkpoint with them laid over it (an engine without a checkpoint directory,
        # e.g. the fake one, takes the release directory as is)
        directory = overlay_release(base, directory, directory.with_name(directory.name + ".overlay"))
    return {**spec, "base_dir": str(directory), "base_version": int(latest.version)}


# ---------------------------------------------------------------------------
# The island
# ---------------------------------------------------------------------------
class ConvertibleIsland:
    """Sampler phase, switch, trainer phase (module doc). ``run_sampler(spec, stop,
    on_event)`` runs the engine sampler island and returns its rounds (the entry's
    ``run``); ``trainer`` runs the trainer phase; ``files`` is the run store;
    ``releases`` the policy release store the sampler follows; ``buffer_view()``
    the merged index over every island's shard as the trainer will read it."""

    def __init__(self, spec: Mapping[str, Any], plan: RoleSwitchPlan, *, files: StoreFiles,
                 releases: Any, buffer_view: Callable[[], BufferView],
                 run_sampler: Callable[..., int], trainer: TrainerPhase,
                 environ: Mapping[str, str] | None = None,
                 device_idle: DeviceIdle | None = None,
                 emit: Callable[..., None] = lambda event, **fields: None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.spec = dict(spec)
        self.plan = plan
        self.files = files
        self.releases = releases
        self.buffer_view = buffer_view
        self.run_sampler = run_sampler
        self.trainer = trainer
        self.environ: Mapping[str, str] = {} if environ is None else environ
        self.device_idle = device_idle
        self.emit = emit
        self.clock = clock
        self.cuts = StoreCutLog(files)
        self.states: list[RoleState] = []
        previous = read_role_state(files, plan.node)
        self.incarnation = 0 if previous is None else int(previous.incarnation)

    # -- phases ----------------------------------------------------------------
    def trigger(self) -> SwitchTrigger:
        marker = switch_marker(self.environ) if self.plan.marker else None
        if marker is not None:
            marker.unlink(missing_ok=True)  # a restarted container keeps its /tmp
        return SwitchTrigger(after_seconds=self.plan.after_seconds, after_groups=self.plan.after_groups,
                             marker=marker, files=self.files if self.plan.store_marker else None,
                             store_marker=switch_marker_path(self.plan.node) if self.plan.store_marker else None,
                             clock=self.clock)

    def sampler_phase(self, spec: Mapping[str, Any]) -> SamplerPhaseReport:
        self.incarnation += 1
        counts = {"groups": 0, "rounds": 0, "version": int(spec.get("base_version", 0))}
        trigger = self.trigger()
        trigger.groups = lambda: counts["groups"]

        def on_event(event: str, **fields: Any) -> None:
            if event == "rl_sampler_engine_round":
                counts["groups"] += int(fields.get("groups", 0))
                counts["rounds"] += 1
                counts["version"] = int(fields.get("policy_version", counts["version"]))
            elif event == "rl_sampler_policy_loaded":
                counts["version"] = int(fields.get("policy_version", counts["version"]))

        self._record(SAMPLER, reason="start", policy_version=counts["version"])
        t0 = self.clock()
        trigger.start()
        rounds = self.run_sampler(spec, stop=trigger, on_event=on_event)
        # the entry stopped the engine (its finally); the device must be free before a trainer starts
        device = {} if self.device_idle is None else self.device_idle.wait()
        view = self.buffer_view()
        report = SamplerPhaseReport(rounds=int(rounds), groups=counts["groups"], buffer_end=int(view.end),
                                    shard_ends=dict(view.shards),
                                    policy_version=counts["version"], reason=trigger.reason,
                                    seconds=round(self.clock() - t0, 3), device=device)
        self.emit(PHASE_EVENT, role=SAMPLER, node=self.plan.node, incarnation=self.incarnation, **asdict(report))
        return report

    def switch_to_trainer(self, report: SamplerPhaseReport) -> RoleState:
        """The role record the trainer phase starts from: the watermark is the newest
        cut's (an earlier trainer phase or incarnation), never ahead of the buffer end."""
        self.incarnation += 1
        watermark = min(self.cuts.watermark(), report.buffer_end)
        view = self.buffer_view()
        state = self._record(TRAINER, reason=report.reason, sampler_rounds=report.rounds,
                             sampler_groups=report.groups, buffer_end=report.buffer_end,
                             shard_ends=dict(report.shard_ends), consumed_watermark=watermark,
                             shard_watermarks=dict(view.shards_below(watermark)),
                             policy_version=report.policy_version, phase_seconds=report.seconds)
        self.emit(SWITCH_EVENT, direction="to_trainer", **state.record())
        return state

    def trainer_phase(self, state: RoleState) -> TrainerPhaseReport:
        t0 = self.clock()
        context = TrainerContext(files=self.files, releases=self.releases, buffer=self.buffer_view,
                                 environ=self.environ)
        report = self.trainer(state, self.plan, context)
        report = replace(report, seconds=round(self.clock() - t0, 3))
        view = self.buffer_view()
        watermark = state.consumed_watermark if report.consumed_watermark is None else int(report.consumed_watermark)
        done = self._record(TRAINER, reason="released", sampler_rounds=state.sampler_rounds,
                            sampler_groups=state.sampler_groups, buffer_end=int(view.end),
                            shard_ends=dict(view.shards), consumed_watermark=watermark,
                            shard_watermarks=dict(view.shards_below(min(watermark, view.end))),
                            policy_version=int(report.policy_version), released_seq=report.released_seq,
                            release_manifest=report.release_manifest, phase_seconds=report.seconds)
        self.emit(PHASE_EVENT, **done.record())
        return report

    def run(self) -> list[RoleState]:
        """Sampler -> trainer, ``max_cycles`` times, then (``then_sampler``) one more
        sampler phase on the released weights. Returns the role records written."""
        # a node restarted on a run that already released (a resumed role switch) samples
        # with the newest release, as after a trainer phase; a fresh run: the spec as is
        spec = sampler_spec_on_release(self.spec, self.releases)
        for cycle in range(self.plan.max_cycles):
            report = self.sampler_phase(spec)
            state = self.switch_to_trainer(report)
            trained = self.trainer_phase(state)
            if not self.plan.then_sampler and cycle + 1 == self.plan.max_cycles:
                break
            spec = sampler_spec_on_release(self.spec, self.releases)
            self.emit(SWITCH_EVENT, direction="to_sampler", node=self.plan.node, incarnation=self.incarnation + 1,
                      policy_version=int(trained.policy_version), base_dir=spec.get("base_dir"))
        else:
            if self.plan.then_sampler:
                report = self.sampler_phase(spec)
                self._record(SAMPLER, reason="finished", sampler_rounds=report.rounds, sampler_groups=report.groups,
                             buffer_end=report.buffer_end, shard_ends=dict(report.shard_ends),
                             policy_version=report.policy_version, phase_seconds=report.seconds)
        return list(self.states)

    def _record(self, role: str, **fields: Any) -> RoleState:
        state = RoleState(node=self.plan.node, role=role, incarnation=self.incarnation, at=round(time.time(), 3),
                          **fields)
        write_role_state(self.files, state)
        self.states.append(state)
        return state


# ---------------------------------------------------------------------------
# Process entry
# ---------------------------------------------------------------------------
def _import(target: str) -> Any:
    import importlib

    module, _, attr = str(target).partition(":")
    if not module or not attr:
        raise RoleSwitchError(f"{target!r} is not 'module:attribute'")
    return getattr(importlib.import_module(module), attr)


def build_island(spec: Mapping[str, Any], *, environ: MutableMapping[str, str] | None = None,
                 trainer: TrainerPhase | None = None) -> ConvertibleIsland:
    """The island of a spec file: the sampler stores of the entry (release store and
    buffer), the store files over the buffer root, the trainer from
    ``role_switch.trainer`` (``run``: ``module:callable``; else the adapter layer's
    subprocess trainer, ``yeto.rl.adapters.miles.role_switch:subprocess_trainer``)."""
    import os

    from . import engine_sampler_entry as entry
    from .resume import store_for
    from .sample_buffer import open_trainer_buffer
    from .trainer_remote import LocalStoreFiles

    env = os.environ if environ is None else environ
    plan = RoleSwitchPlan.from_spec(spec)
    releases, _ = entry.build_stores(spec, env)
    buffer_store = store_for(str(spec.get("buffer_root") or spec["store"]), environ=env)
    files = LocalStoreFiles(buffer_store)
    # the trainer's view: every island's shard merged (the trainer island merges the same way)
    buffer = open_trainer_buffer(buffer_store, batch_size=1, shards=spec.get("buffer_shards"))
    if trainer is None:
        target = plan.trainer.get("run") or "yeto.rl.adapters.miles.role_switch:subprocess_trainer"
        trainer = _import(str(target))
    idle = None
    if plan.device_idle is not None:
        idle = DeviceIdle(used_mib=int(plan.device_idle.get("used_mib", DEVICE_IDLE_USED_MIB)),
                          timeout_s=float(plan.device_idle.get("timeout_s", DEVICE_IDLE_TIMEOUT_S)))

    def buffer_view() -> BufferView:
        buffer.refresh()
        end = len(buffer)
        return BufferView(end=end, shards=buffer.shard_positions(end), shards_below=buffer.shard_positions)

    def run_sampler(sampler_spec: Mapping[str, Any], *, stop: Callable[[], bool],
                    on_event: Callable[..., None]) -> int:
        return entry.run(sampler_spec, environ=env, stop=stop, on_event=on_event)

    tape = entry.tape_writer(Path(spec["event_tape"]).with_name("role-switch.jsonl"), str(spec["sampler_id"]))
    return ConvertibleIsland(spec, plan, files=files, releases=releases, buffer_view=buffer_view,
                             run_sampler=run_sampler, trainer=trainer, environ=env, device_idle=idle, emit=tape)


def main(argv: Sequence[str] | None = None) -> int:
    from . import engine_sampler_entry as entry

    parser = argparse.ArgumentParser(prog="python -m yeto.rl.engine.role_switch",
                                     description="run an island as engine sampler, then as trainer (module doc)")
    parser.add_argument("--spec", required=True, help="the engine sampler spec with a role_switch section")
    args = parser.parse_args(argv)
    try:
        spec = entry.load_spec(args.spec)
        states = build_island(spec).run()
    except (entry.SamplerSpecError, RoleSwitchError) as error:
        print(f"[role-switch] rejected: {error}", file=sys.stderr)
        return 2
    last = states[-1] if states else None
    print(f"[role-switch] done: {len(states)} role record(s); last role {None if last is None else last.role} "
          f"v{0 if last is None else last.policy_version}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
