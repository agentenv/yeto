"""E3 A8 / DEV-GATHER harness: arm sequence of plan-v3 §2 (rl-infra-spec 4.6 X4).

Runs INSIDE the GPU container (one arm per process, see ``learner_shim.py``)
or on CPU with a fake backend (``--dry-run`` path in tests). Engine specifics
live behind :class:`Backend`; this module only sequences arms, writes cuts
through :class:`MilesTrainerGroup` and records evidence:

    <work>/arms/<ARM>/events.jsonl      one JSON line per event (start/probe/train/cut/restore/dump/end)
    <work>/arms/<ARM>/state/<tag>_tp*_pp*_dp*.pt   per-rank dumps (e3_probe.dump_state)
    <work>/cuts/<cut_id>/...            ReconfigurationCut (manifest last)

Steps are 1-based: step k trains frozen rollout k-1 (8 frozen rollouts).
Cut context values the yeto driver would supply (data cursor, ledger, outer
state, policy hash) are harness constants: A8 is a trainer spike, no driver
runs; they only have to be complete and consistent (``cut.context_problems``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

FROZEN_ROLLOUTS = 8
POLICY_HASH = "e3-harness-frozen"


@dataclass(frozen=True)
class ArmSpec:
    name: str
    dp: int
    restore: str | None = None  # cut id to restore resharded at start
    restore_source_dp: int | None = None
    save_after_restore: str | None = None  # cut id saved right after the restore (B1 -> C1')
    cut_at_step2: str | None = None  # cut id saved after step 2 (A arms)
    last_step: int = 8


# plan-v3 §2.1, in order.
ARMS = (
    ArmSpec("A1", dp=1, cut_at_step2="C1"),
    ArmSpec("A2", dp=2, cut_at_step2="C2"),
    ArmSpec("B1", dp=2, restore="C1", restore_source_dp=1, save_after_restore="C1p"),
    ArmSpec("B1p", dp=2, restore="C1", restore_source_dp=1, last_step=3),
    ArmSpec("B2", dp=1, restore="C2", restore_source_dp=2),
    ArmSpec("RT", dp=1, restore="C1p", restore_source_dp=2, last_step=2),
)
ARM_BY_NAME = {a.name: a for a in ARMS}
DUMP_STEPS = (2, 3, 8)


class Backend(Protocol):
    """One trainer at a time. ``trainer`` is a MilesTrainerGroup over the running actor."""

    trainer: Any
    algorithm: Any  # AlgorithmSpec (default GRPO)
    global_batch_size: int
    micro_batch_size: int
    fingerprint: str

    def start_arm(self, dp: int) -> None: ...
    def plugin(self, fn_path: str, kwargs: dict | None = None) -> list: ...
    def train(self, rollout_id: int) -> dict: ...  # trains frozen rollout, returns metrics
    def stop_arm(self) -> None: ...


class Evidence:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / "events.jsonl"

    def event(self, kind: str, **fields: Any) -> None:
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": kind, **fields}, sort_keys=True, default=repr) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def read_events(directory: Path) -> list[dict[str, Any]]:
    path = Path(directory) / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _layout(dp: int) -> dict[str, int]:
    return {"world": dp, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": dp}


def cut_context(backend: Backend, root: Path, cut_id: str, local_step: int):
    from yeto.rl.engine.cut import AlgorithmIdentity, CutProgress
    from yeto.rl.engine.miles_adapter.trainer import CutContext

    gbs = backend.global_batch_size
    return CutContext(
        root=str(root), cut_id=cut_id, backend_fingerprint=backend.fingerprint,
        progress=CutProgress(local_step=local_step, scheduler_samples=local_step * gbs, global_batch_size=gbs,
                             next_rollout_id=local_step, policy_version=local_step, policy_hash=POLICY_HASH),
        algorithm=AlgorithmIdentity.from_spec(backend.algorithm),
        data={"sample_offset": local_step * gbs, "epoch_id": 0, "sample_group_index": local_step * gbs,
              "sample_index": local_step * gbs, "buffer_length": 0},
        ledger={"carried_over": 0, "ready_unconsumed": 0}, outer={"settled": True},
    )


def expectation(backend: Backend, source_dp: int, local_step: int):
    from yeto.rl.engine.cut import AlgorithmIdentity, RestoreExpectation

    return RestoreExpectation(algorithm=AlgorithmIdentity.from_spec(backend.algorithm), layout=_layout(source_dp),
                              backend_fingerprint=backend.fingerprint, local_step=local_step,
                              policy_version=local_step, epoch=0)


def _dump(backend: Backend, ev: Evidence, tag: str) -> None:
    from yeto.rl.engine.miles_adapter.e3_probe import DUMP_STATE

    ev.event("dump", tag=tag, ranks=backend.plugin(DUMP_STATE, {"directory": str(ev.dir / "state"), "tag": tag}))


def _probe(backend: Backend, ev: Evidence, when: str) -> None:
    from yeto.rl.engine.miles_adapter.e3_probe import RANK_INFO

    ev.event("rank_info", when=when, ranks=backend.plugin(RANK_INFO))


def run_arm(spec: ArmSpec, backend: Backend, work: Path) -> Path:
    """Run one arm of plan-v3 §2.1 and return its evidence directory."""
    from yeto.rl.engine.miles_adapter.e3_probe import DRAIN_PROBE, INSTALL_PROBE
    from yeto.rl.engine.miles_adapter.reshard import ReshardPlan

    work = Path(work)
    cuts = work / "cuts"
    ev = Evidence(work / "arms" / spec.name)
    ev.event("start", arm=spec.name, dp=spec.dp, spec=spec.__dict__)
    backend.start_arm(spec.dp)
    try:
        ev.event("probe_installed", ranks=backend.plugin(INSTALL_PROBE))
        _probe(backend, ev, "start")
        step = 0
        if spec.restore:
            plan = ReshardPlan(_layout(spec.restore_source_dp), _layout(spec.dp),
                               backend.global_batch_size, backend.micro_batch_size)
            result = backend.trainer.restore_cut_resharded(
                spec.restore, epoch=0, root=str(cuts), expect=expectation(backend, spec.restore_source_dp, 2),
                plan=plan, certified=None)  # the spike produces the certification
            step = 2
            ev.event("restore", cut_id=spec.restore, plan=result["plan"], rng_mapping=result["rng_mapping"],
                     full_state_digests=result["full_state_digests"], ranks=result["ranks"])
            _probe(backend, ev, "restored")
            _dump(backend, ev, "restored")
            if spec.save_after_restore:
                backend.trainer.save_cut(epoch=0, context=cut_context(backend, cuts, spec.save_after_restore, step))
                ev.event("cut", cut_id=spec.save_after_restore, step=step)
        while step < spec.last_step:
            rollout_id = step
            metrics = backend.train(rollout_id)
            step += 1
            ev.event("train", step=step, rollout_id=rollout_id, metrics=metrics,
                     probe=backend.plugin(DRAIN_PROBE))
            if step == 2 and spec.cut_at_step2:
                backend.trainer.save_cut(epoch=0, context=cut_context(backend, cuts, spec.cut_at_step2, step))
                ev.event("cut", cut_id=spec.cut_at_step2, step=step)
            if step in DUMP_STEPS:
                _dump(backend, ev, f"s{step}")
        _probe(backend, ev, "end")
        ev.event("end", arm=spec.name, step=step)
    except BaseException as exc:
        ev.event("error", arm=spec.name, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        backend.stop_arm()
    return ev.dir


def run_all(backend_for_arm, work: Path, arms: tuple[ArmSpec, ...] = ARMS) -> list[Path]:
    """Sequential arms (CPU dry-run); on GPU every arm is its own process (``learner_shim``)."""
    return [run_arm(a, backend_for_arm(a), work) for a in arms]
