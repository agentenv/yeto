"""Eval island runner (rl-eval-difficulty-buckets D11.1/D11.3/D11.4, tasks 5.1-5.6).

An inference-only island: it never joins the merge pool and never submits a
delta. It drains the store's queue of policy versions in order; per version it

1. loads the manifest and checks every file's sha256 (fail closed);
2. refuses a version whose sampling settings differ from the plan (D3: another
   ruler, the run stops);
3. loads base + policy through the injected ``loader`` and requires the served
   token to equal the manifest's ``rl/policy_token``;
4. runs each ``(task_id, trial)`` unit not yet done, writing a ``start`` record
   before and a ``result`` record after each attempt; an attempt lost to
   preemption leaves a bare ``start`` (counted, never scored, rerun);
5. once every unit has a result: per-bucket metrics (``stats``), one
   ``rl_eval`` event, ``done`` marker. An incomplete version emits nothing.

Ports (all injectable, CPU tests use fakes):

* ``loader.load(manifest, files_dir) -> served policy token``
* ``attempt(task, trial, *, policy_version, policy_token) -> mapping`` with
  ``reward``, ``success``, ``end_reason`` and optionally ``turns``, ``tokens``,
  ``trajectory_id``. Raise :class:`Preempted` when the inference side is gone;
  any other exception is recorded as ``end_reason=infra_error`` (D6.d).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol

from . import stats
from .store import EvalIntegrityError, EvalStore, iter_results

EVAL_EVENT = "rl_eval"
EVAL_ISLAND_ROLE = "eval"  # D11.1: island ledger role; never a training/merge member
DEFAULT_TRIALS_V0 = 4      # D2: TB2 version 0
DEFAULT_TRIALS = 2         # D2: TB2 later versions


class Preempted(RuntimeError):
    """The inference side was reclaimed mid-attempt; the unit is not scored."""


class EvalSettingError(RuntimeError):
    """The version was produced under different sampling settings (D3)."""


class PolicyLoader(Protocol):
    def load(self, manifest: Mapping[str, Any], files_dir: Path) -> str: ...


Attempt = Callable[..., Mapping[str, Any]]
Emit = Callable[..., None]


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    eval_bucket: str
    difficulty: str = "unknown"
    benchmark: str = "unknown"
    row: Mapping[str, Any] = field(default_factory=dict, compare=False)


def sampling_sha256(sampling: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(sampling), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class EvalPlan:
    tasks: tuple[EvalTask, ...]
    sampling: Mapping[str, Any]
    set_sha256: str
    trials_v0: int = DEFAULT_TRIALS_V0
    trials: int = DEFAULT_TRIALS

    def __post_init__(self) -> None:
        ids = [t.task_id for t in self.tasks]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("eval plan needs a non-empty list of distinct task ids")
        if self.trials_v0 < 1 or self.trials < 1:
            raise ValueError("trials must be >= 1")

    def trials_for(self, version: int) -> int:
        return self.trials_v0 if int(version) == 0 else self.trials

    def units(self, version: int) -> list[tuple[EvalTask, int]]:
        n = self.trials_for(version)
        return [(task, trial) for task in self.tasks for trial in range(n)]

    @classmethod
    def from_holdout(cls, holdout: Mapping[str, Any], *, sampling: Mapping[str, Any],
                     rows: Iterable[Mapping[str, Any]] = (), task_ids: Iterable[str] | None = None,
                     **kw: Any) -> "EvalPlan":
        from yeto.rl.harness.reward_env.benchmark import holdout_ids, holdout_sha256, task_id_of

        holdout_ids(dict(holdout))  # schema + duplicates
        by_id = {task_id_of(dict(r)): r for r in rows}
        keep = None if task_ids is None else set(task_ids)
        tasks = tuple(
            EvalTask(task_id=i["task_id"], eval_bucket=i["eval_bucket"], difficulty=i.get("difficulty", "unknown"),
                     benchmark=str(holdout.get("benchmark", "unknown")), row=by_id.get(i["task_id"], {}))
            for i in holdout["items"] if keep is None or i["task_id"] in keep)
        if keep is not None and len(tasks) != len(keep):
            raise ValueError(f"tasks not in the hold-out list: {sorted(keep - {t.task_id for t in tasks})}")
        return cls(tasks=tasks, sampling=dict(sampling), set_sha256=holdout_sha256(dict(holdout)), **kw)


def jsonl_emitter(path: str | Path, *, clock: Callable[[], float] = time.time, **labels: Any) -> Emit:
    """Append ``{"event": ..., "ts": ..., **fields}`` lines (the run tape format)."""
    target = Path(path)

    def emit(event: str, **fields: Any) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "ts": clock(), **labels, **fields},
                                sort_keys=True, default=str) + "\n")
    return emit


class EvalIsland:
    def __init__(self, store: EvalStore, plan: EvalPlan, *, loader: PolicyLoader, attempt: Attempt,
                 emit: Emit, island_id: str = "eval-0", clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.plan = plan
        self.loader = loader
        self.attempt = attempt
        self.emit = emit
        self.island_id = island_id
        self.clock = clock

    # -- one version ------------------------------------------------------
    def _check(self, version: int) -> dict[str, Any]:
        manifest = self.store.load_manifest(version, verify=True)
        want, got = sampling_sha256(self.plan.sampling), sampling_sha256(manifest.get("sampling") or {})
        if want != got:
            raise EvalSettingError(
                f"v{version}: sampling settings differ from the eval plan "
                f"({manifest.get('sampling')} != {dict(self.plan.sampling)})")
        return manifest

    def _record(self, task: EvalTask, trial: int, version: int, token: str,
                out: Mapping[str, Any], seconds: float) -> dict[str, Any]:
        end = str(out.get("end_reason") or "completed")
        if end not in stats.END_REASONS:
            raise ValueError(f"unknown end_reason {end!r} (allowed: {stats.END_REASONS})")
        return {
            "kind": "result", "policy_version": version, "task_id": task.task_id, "trial": trial,
            "eval_bucket": task.eval_bucket, "difficulty": task.difficulty, "rl/policy_token": token,
            "reward": float(out.get("reward") or 0.0),
            "success": bool(out.get("success")) if end != "infra_error" else False,
            "end_reason": end, "turns": out.get("turns"), "tokens": out.get("tokens"),
            "trajectory_id": out.get("trajectory_id"), "seconds": round(seconds, 3),
            "island": self.island_id,
            **({"error": str(out["error"])[:500]} if out.get("error") else {}),
            **({"timing": dict(out["timing"])} if out.get("timing") else {}),
        }

    def run_version(self, version: int) -> dict[str, Any] | None:
        """Evaluate ``version`` to completion; return the ``rl_eval`` payload.

        Raises :class:`Preempted` (resume later), :class:`EvalIntegrityError`
        or :class:`EvalSettingError` (fail closed)."""
        done = self.store.done_payload(version)
        if done is not None:
            return done
        started = self.clock()
        self.store.reload()
        manifest = self._check(version)
        token = str(manifest["rl/policy_token"])
        served = self.loader.load(manifest, self.store.files_dir(version))
        if str(served) != token:
            raise EvalIntegrityError(f"v{version}: served policy token {served!r} != manifest {token!r}")
        log = self.store.read_units(version)
        for task, trial in self.plan.units(version):
            if (version, task.task_id, trial) in log.results:
                continue
            self.store.append_unit({"kind": "start", "policy_version": version, "task_id": task.task_id,
                                    "trial": trial, "island": self.island_id})
            t0 = self.clock()
            try:
                out = self.attempt(task, trial, policy_version=version, policy_token=token)
            except Preempted:
                raise
            except Exception as exc:  # noqa: BLE001 - judge/sandbox infrastructure: not a failed attempt
                out = {"end_reason": "infra_error", "reward": 0.0, "success": False,
                       "error": f"{type(exc).__name__}: {exc}"}
            self.store.append_unit(self._record(task, trial, version, token, out, self.clock() - t0))
        return self._finish(version, token, manifest, wall_s=self.clock() - started)

    def _finish(self, version: int, token: str, manifest: Mapping[str, Any], *, wall_s: float) -> dict[str, Any]:
        log = self.store.read_units(version)
        wanted = {(version, t.task_id, n) for t, n in self.plan.units(version)}
        missing = wanted - set(log.results)
        if missing:  # cannot happen after a full pass; guard against a racing writer
            raise RuntimeError(f"v{version}: {len(missing)} units without a result")
        records = [r for r in iter_results(log) if (version, r["task_id"], r["trial"]) in wanted]
        baseline = None
        if version != 0:
            base_log = self.store.read_units(0)
            if base_log.results:
                baseline = list(iter_results(base_log))
        metrics = stats.bucket_metrics(records, baseline)
        pending = self.store.pending_versions()
        later = [v for v in pending if v > version]
        counted = [r for r in records if r["end_reason"] != "infra_error"]
        payload: dict[str, Any] = {
            "policy_version": version, "rl/policy_token": token, "source": "eval_island",
            "island": self.island_id, "role": EVAL_ISLAND_ROLE,
            "policy_tensor_hash": manifest.get("policy_tensor_hash"),
            "eval/units": len(records),
            "eval/pass_rate": (sum(bool(r["success"]) for r in counted) / len(counted)) if counted else None,
            "eval/infra_error_frac": (len(records) - len(counted)) / len(records) if records else None,
            "eval/preemptions": log.orphan_starts,
            "eval/duplicate_results": log.duplicate_results,
            "eval/trials": self.plan.trials_for(version),
            "eval/set_sha256": self.plan.set_sha256,
            "eval/sampling_sha256": sampling_sha256(self.plan.sampling),
            "eval/results_sha256": self.store.results_sha256(version),
            "eval/queue_len": len(later),
            "eval/lag_rounds": (max(later) - version) if later else 0,
            "eval/wall_s": round(wall_s, 3),
            **{f"eval/{k}": v for k, v in stats.flatten(metrics).items()},
        }
        self.emit(EVAL_EVENT, **payload)
        self.store.mark_done(version, payload)
        return payload

    # -- the queue ----------------------------------------------------------
    def run(self, *, idle: Callable[[], bool] | None = None, poll_s: float = 30.0,
            sleep: Callable[[float], None] = time.sleep) -> list[int]:
        """Drain the queue oldest first (no version dropped, D11.4).

        ``idle()`` is asked when the queue is empty: True = stop (training is
        finished and wrote its last version), False = wait ``poll_s`` and look
        again. Without ``idle`` the island stops at the first empty queue."""
        evaluated: list[int] = []
        while True:
            self.store.reload()
            pending = self.store.pending_versions()
            if not pending:
                if idle is None:
                    return evaluated
                if idle():
                    # training may have queued its last version just before finishing
                    self.store.reload()
                    if not self.store.pending_versions():
                        return evaluated
                    continue
                sleep(poll_s)
                continue
            self.run_version(pending[0])
            evaluated.append(pending[0])
