"""Benchmark-neutral reward environment interface (change rl-agentic-reward-env).

A *benchmark adapter* turns a task id into everything the sandbox side needs:
the base image and resources (``TaskSpec``), the dependencies to bake into the
image ahead of time (``PrebakePlan``), and the judge command plus its output
parser.  The sandbox backend and the judge runner below know nothing about any
particular benchmark; adding one = one adapter + ``register``.

No Modal / Miles imports here.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Protocol

PREBAKE_SCHEMA = "yeto-reward-env-prebake-v1"


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    task_id: str
    base_image: str
    cpus: int
    memory_mb: int
    workdir: str
    agent_timeout_s: float
    judge_timeout_s: float
    instruction: str | None = None
    # WP3 (rl-eval-difficulty-buckets D6a): official difficulty, verbatim, and its bucket
    benchmark_version: str = "unknown"
    difficulty: str = "unknown"
    difficulty_source: str = "unknown"
    eval_bucket: str | None = None

    def metadata(self) -> dict[str, Any]:
        """Dataset-row ``metadata`` in the WP3 D6a shape."""
        meta = {"task_id": self.task_id, "benchmark": self.benchmark,
                "benchmark_version": self.benchmark_version, "difficulty": self.difficulty,
                "difficulty_source": self.difficulty_source}
        if self.eval_bucket is not None:
            meta["eval_bucket"] = self.eval_bucket
        return meta


@dataclass(frozen=True)
class PrebakePlan:
    """Commands run once at image build time (as root, HOME=/root) on ``base_image``.

    ``complete`` is False when some setup step could not be recognised; the
    judge still works (the unchanged judge script installs what is missing),
    only slower.  ``skipped`` lists the unrecognised lines for the report.
    """

    base_image: str
    commands: tuple[str, ...]
    complete: bool
    skipped: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.commands

    def digest(self) -> str:
        """Content hash of (schema, base image, commands): the prebaked image identity."""
        payload = json.dumps(
            {"schema": PREBAKE_SCHEMA, "base_image": self.base_image, "commands": list(self.commands)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class JudgeResult:
    benchmark: str
    task_id: str
    reward: float
    passed: bool
    exit_code: int
    timed_out: bool
    seconds: float
    prebaked: bool
    log: str = ""
    # judge infrastructure failed (sandbox exec raised, unparseable environment
    # fault): WP3 D6d counts these apart, never as a failed attempt
    infra_error: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JudgeHandle(Protocol):
    """What the judge runner needs from a sandbox (``tb2_provider.SandboxHandle`` fits)."""

    def exec(self, command: str, *, timeout_s: float, workdir: str | None = None) -> Any: ...


class BenchmarkAdapter(Protocol):
    name: str

    def task_ids(self) -> list[str]: ...

    def task_spec(self, task_id: str) -> TaskSpec: ...

    def prebake_plan(self, task_id: str) -> PrebakePlan: ...

    def judge_command(self, task_id: str, submission: str | None = None) -> str:
        """Command run in the judge sandbox. ``submission``: the agent's answer when
        the benchmark judges it in a fresh sandbox (SWE-bench: a git diff);
        ``None`` = judge the sandbox as the agent left it (TB2) or, for patch
        benchmarks, the official no-patch (negative control) mode."""
        ...

    def parse_judge(self, task_id: str, output: str, exit_code: int, timed_out: bool) -> dict[str, Any]:
        """Return at least ``reward`` (float) and ``passed`` (bool); extra keys go to ``extra``."""
        ...


_REGISTRY: dict[str, Callable[..., BenchmarkAdapter]] = {}
# benchmark name -> module of this package that registers it (lazy, no import cost)
_BUILTIN_MODULES = {"tb2": "tb2", "swebench-verified": "swebench_verified"}


def register(name: str, factory: Callable[..., BenchmarkAdapter]) -> None:
    if name in _REGISTRY and _REGISTRY[name] is not factory:
        raise ValueError(f"benchmark {name!r} is already registered")
    _REGISTRY[name] = factory


def registered() -> tuple[str, ...]:
    return tuple(sorted(set(_REGISTRY) | set(_BUILTIN_MODULES)))


def get_adapter(name: str, **kwargs: Any) -> BenchmarkAdapter:
    module = _BUILTIN_MODULES.get(name)
    if name not in _REGISTRY and module is not None:
        import importlib

        importlib.import_module(f"{__package__}.{module}")  # built-ins register on import
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown benchmark {name!r}; registered: {', '.join(registered()) or 'none'}") from exc
    return factory(**kwargs)


LOG_CHARS = 2000


def run_judge(adapter: BenchmarkAdapter, task_id: str, handle: JudgeHandle, *, prebaked: bool,
              submission: str | None = None,
              clock: Callable[[], float] = time.monotonic) -> JudgeResult:
    """Run the adapter's judge command in ``handle`` and parse it (trusted side)."""
    spec = adapter.task_spec(task_id)
    command = adapter.judge_command(task_id, submission)
    started = clock()
    try:
        result = handle.exec(command, timeout_s=spec.judge_timeout_s)
    except Exception as exc:  # noqa: BLE001 - sandbox gone / API error: infrastructure, not a 0
        return JudgeResult(benchmark=adapter.name, task_id=task_id, reward=0.0, passed=False, exit_code=-1,
                           timed_out=False, seconds=clock() - started, prebaked=prebaked,
                           log=f"{type(exc).__name__}: {exc}"[:LOG_CHARS], infra_error=True)
    seconds = clock() - started
    output = getattr(result, "output", "") or ""
    exit_code = int(getattr(result, "exit_code", 0))
    timed_out = bool(getattr(result, "timed_out", False))
    parsed = dict(adapter.parse_judge(task_id, output, exit_code, timed_out))
    reward = float(parsed.pop("reward"))
    passed = bool(parsed.pop("passed"))
    infra_error = bool(parsed.pop("infra_error", False))
    return JudgeResult(
        benchmark=adapter.name, task_id=task_id, reward=reward, passed=passed, exit_code=exit_code,
        timed_out=timed_out, seconds=seconds, prebaked=prebaked, infra_error=infra_error,
        log=output if len(output) <= LOG_CHARS else "…" + output[-(LOG_CHARS - 1):], extra=parsed,
    )


def plans(adapter: BenchmarkAdapter, task_ids: Iterable[str] | None = None) -> dict[str, PrebakePlan]:
    return {tid: adapter.prebake_plan(tid) for tid in (task_ids if task_ids is not None else adapter.task_ids())}


# --- evaluation hold-out lists (WP3 rl-eval-difficulty-buckets D6b) -------------

HOLDOUT_SCHEMA = "yeto-eval-holdout/1"


def build_holdout(adapter: BenchmarkAdapter, quotas: dict[str, int | None], *, seed: int, rule: str,
                  task_ids: Iterable[str] | None = None, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Stratified, reproducible hold-out list in the WP3 D6b JSON shape.

    ``quotas`` maps ``eval_bucket`` -> number of tasks (``None`` = all of that
    bucket).  Within a bucket tasks are ranked by sha256(f"{seed}/{task_id}"),
    so the result does not depend on input order.  A bucket with fewer tasks
    than its quota fails closed.
    """
    excluded = set(exclude)
    specs = [adapter.task_spec(t) for t in (task_ids if task_ids is not None else adapter.task_ids())
             if t not in excluded]
    by_bucket: dict[str, list[TaskSpec]] = {}
    for spec in specs:
        if spec.eval_bucket in quotas:
            by_bucket.setdefault(spec.eval_bucket, []).append(spec)
    items = []
    versions = {spec.benchmark_version for spec in specs}
    for bucket, quota in quotas.items():
        pool = sorted(by_bucket.get(bucket, []),
                      key=lambda s: hashlib.sha256(f"{seed}/{s.task_id}".encode()).hexdigest())
        if quota is not None and len(pool) < quota:
            raise ValueError(f"bucket {bucket!r}: {len(pool)} tasks < quota {quota}")
        chosen = pool if quota is None else pool[:quota]
        items += [{"task_id": s.task_id, "difficulty": s.difficulty, "eval_bucket": bucket}
                  for s in sorted(chosen, key=lambda s: s.task_id)]
    return {"schema": HOLDOUT_SCHEMA, "benchmark": adapter.name,
            "benchmark_version": versions.pop() if len(versions) == 1 else sorted(versions),
            "seed": seed, "rule": rule, "items": items}


def holdout_sha256(holdout: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(holdout, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def holdout_ids(holdout: dict[str, Any]) -> list[str]:
    if holdout.get("schema") != HOLDOUT_SCHEMA:
        raise ValueError(f"not a {HOLDOUT_SCHEMA} file")
    ids = [item["task_id"] for item in holdout["items"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task ids in hold-out list")
    return ids


def task_id_of(row: dict[str, Any]) -> str | None:
    meta = row.get("metadata")
    if isinstance(meta, dict) and meta.get("task_id"):
        return str(meta["task_id"])
    return str(row["task_id"]) if row.get("task_id") else None


def split_rows(rows: Iterable[dict[str, Any]], heldout: Iterable[str]) -> tuple[list[Any], list[Any]]:
    """(train rows, held-out rows); rows without a task id fail closed."""
    held = set(heldout)
    train: list[Any] = []
    evaluation: list[Any] = []
    for row in rows:
        tid = task_id_of(row)
        if tid is None:
            raise ValueError("dataset row without metadata.task_id")
        (evaluation if tid in held else train).append(row)
    return train, evaluation


def assert_disjoint(train_ids: Iterable[str], heldout: Iterable[str]) -> None:
    overlap = sorted(set(train_ids) & set(heldout))
    if overlap:
        raise ValueError(f"held-out tasks leak into training data: {len(overlap)}, e.g. {overlap[:5]}")
