"""Task lists over Harbor task directories: TB2 hold-out, training split, curriculum.

Harbor reads its tasks from task directories (``<task>/task.toml``,
``instruction.md``, ``environment/``, ``tests/``, ``solution/``).
:class:`HarborTaskSet` reads only the ``task.toml`` metadata, so the lists
are cut from the same directories the trials run (``HARBOR_TASKS_DIR``).
Both Terminal-Bench layouts are accepted: TB2 (task directories at the
checkout root) and TB2.1 (under ``tasks/``).

Hold-out and training split (rewrite of the generator that left with the
self-written Codex harness): the rule is
the one recorded in ``data/eval/tb2-holdout.json``:
drop the smoke-6 and the unusable
table from the pool, then draw a stratified sample per official difficulty
with :func:`~yeto.rl.harness.reward_env.benchmark.build_holdout` (rank by
``sha256(f"{seed}/{task_id}")``, independent of input order).  The training
list is every other task minus the unusable table.  On the TB2 checkout
``tb2@2fd12b8`` the output is byte-identical to the files in ``data/eval``.

Curriculum (:func:`build_curriculum`): ordered stages of the training list,
from public facts only: the official ``metadata.difficulty``, the official
``expert_time_estimate_min`` / ``junior_time_estimate_min``, optionally the
result of running the reference solution in our judge environment and the
pass rates a baseline policy measured.  The stage cut points are yeto choices
(:data:`DEFAULT_STAGES`), not data; every item carries the facts it was
placed by, so a reader can re-check it.  A task with a measured pass rate of
0 or 1 gives GRPO no signal (every sample of a group gets the same reward);
``require_signal`` drops those.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yeto.rl.harness.reward_env import benchmark as bm

BENCHMARK = "tb2"
DIFFICULTIES = ("easy", "medium", "hard")

# --- TB2 hold-out rule (values of data/eval/tb2-holdout.json) -----------------
HOLDOUT_SEED = 20261008
HOLDOUT_QUOTAS: dict[str, int | None] = {"tb2-easy": 2, "tb2-medium": 18, "tb2-hard": 10}
HOLDOUT_RULE = "exclude smoke6 (trained in the smoke runs); stratified by difficulty: easy 2 / medium 18 / hard 10"
HOLDOUT_RULE_UNUSABLE = ("exclude smoke6 (trained in the smoke runs) and data/eval/tb2-unusable.json; "
                         "stratified by difficulty: easy 2 / medium 18 / hard 10")
# Trained on in both smoke GPU runs; they leave the eval pool
# (they may stay in training).
SMOKE6_TASK_IDS = ("fix-git", "regex-log", "sqlite-db-truncate", "log-summary-date-ranges",
                   "openssl-selfsigned-cert", "git-multibranch")
SMOKE6_REASON = "trained in the smoke runs"

UNUSABLE_SCHEMA = "yeto.tb2-unusable/v1"
UNUSABLE_DEFAULT_PATH = Path(__file__).resolve().parents[4] / "data" / "eval" / "tb2-unusable.json"

CURRICULUM_SCHEMA = "yeto-curriculum/1"


def smoke6_exclusions() -> dict[str, str]:
    return {t: SMOKE6_REASON for t in SMOKE6_TASK_IDS}


def load_unusable(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """``task_id -> "unusable/<category>: <reason>"`` from the unusable table (fail closed)."""
    source = Path(path or UNUSABLE_DEFAULT_PATH)
    data = json.loads(source.read_text())
    if data.get("schema") != UNUSABLE_SCHEMA or data.get("benchmark") != BENCHMARK:
        raise ValueError(f"{source}: not a {UNUSABLE_SCHEMA} {BENCHMARK} file")
    out: dict[str, str] = {}
    for item in data["items"]:
        tid, cat, why = item.get("task_id"), item.get("category"), item.get("reason")
        if not (tid and cat and why) or tid in out:
            raise ValueError(f"bad or duplicate unusable entry: {item!r}")
        out[tid] = f"unusable/{cat}: {why}"
    return out


# --- task directories ------------------------------------------------------------

def _git_rev(path: Path) -> str | None:
    try:
        rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return rev or None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


_WORKDIR = re.compile(r"^\s*WORKDIR\s+(\S+)\s*$", re.M)


@dataclass(frozen=True)
class TaskFacts:
    """The public facts of one task (``task.toml``)."""

    task_id: str
    difficulty: str
    category: str | None
    expert_time_min: float | None
    junior_time_min: float | None
    agent_timeout_s: float
    verifier_timeout_s: float
    docker_image: str
    cpus: int
    memory_mb: int
    workdir: str
    benchmark: str = BENCHMARK

    @property
    def eval_bucket(self) -> str | None:
        return f"{self.benchmark}-{self.difficulty}" if self.difficulty in DIFFICULTIES else None


class HarborTaskSet:
    """The Harbor task directories of one Terminal-Bench checkout (``BenchmarkAdapter`` for the lists).

    ``version``: ``tb2@<rev>`` for the TB2 layout, ``tb2.1@<rev>`` for TB2.1
    (``tasks/`` subdirectory); ``<rev>`` is the checkout's short git revision.

    ``benchmark``: the name of a task set that is not Terminal-Bench (any
    directory of Harbor tasks); it names the eval buckets (``<name>-<difficulty>``)
    and the default version (``<name>@<rev>``).
    """

    name = BENCHMARK

    def __init__(self, root: str | os.PathLike[str], *, version: str | None = None,
                 benchmark: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        nested = self.root / "tasks"
        self.tasks_dir = nested if nested.is_dir() and any(nested.glob("*/task.toml")) else self.root
        if not any(self.tasks_dir.glob("*/task.toml")):
            raise FileNotFoundError(f"{self.root}: no <task>/task.toml (not a Harbor task checkout)")
        if benchmark is not None:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", benchmark):
                raise ValueError(f"bad benchmark name {benchmark!r}")
            self.name = benchmark
        family = benchmark or ("tb2.1" if self.tasks_dir != self.root else "tb2")
        self.version = version or f"{family}@{_git_rev(self.root) or 'unknown'}"
        self._facts: dict[str, TaskFacts] = {}

    def task_ids(self) -> list[str]:
        return sorted(p.parent.name for p in self.tasks_dir.glob("*/task.toml"))

    def facts(self, task_id: str) -> TaskFacts:
        cached = self._facts.get(task_id)
        if cached is not None:
            return cached
        task_dir = self.tasks_dir / task_id
        config = tomllib.loads((task_dir / "task.toml").read_text())
        meta = config.get("metadata") or {}
        env = config.get("environment") or {}
        dockerfile = task_dir / "environment" / "Dockerfile"
        workdirs = _WORKDIR.findall(dockerfile.read_text()) if dockerfile.is_file() else []
        facts = TaskFacts(
            task_id=task_id,
            difficulty=str(meta.get("difficulty") or "unknown"),
            category=meta.get("category"),
            expert_time_min=_number(meta.get("expert_time_estimate_min")),
            junior_time_min=_number(meta.get("junior_time_estimate_min")),
            agent_timeout_s=float((config.get("agent") or {}).get("timeout_sec", 0.0)),
            verifier_timeout_s=float((config.get("verifier") or {}).get("timeout_sec", 0.0)),
            docker_image=str(env.get("docker_image") or ""),
            cpus=int(env.get("cpus", 1)),
            memory_mb=int(env.get("memory_mb", 0)),
            workdir=workdirs[-1] if workdirs else "",
            benchmark=self.name,
        )
        self._facts[task_id] = facts
        return facts

    def task_spec(self, task_id: str) -> bm.TaskSpec:
        f = self.facts(task_id)
        return bm.TaskSpec(benchmark=self.name, task_id=task_id, base_image=f.docker_image, cpus=f.cpus,
                           memory_mb=f.memory_mb, workdir=f.workdir, agent_timeout_s=f.agent_timeout_s,
                           judge_timeout_s=f.verifier_timeout_s, benchmark_version=self.version,
                           difficulty=f.difficulty, difficulty_source="tb2-task.toml", eval_bucket=f.eval_bucket)


# --- hold-out and training split ---------------------------------------------------

def build_tb2_lists(tasks: HarborTaskSet, *, unusable: Mapping[str, str] | None = None,
                    exclude: Mapping[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(holdout, train)`` in the ``data/eval/tb2-holdout.json`` / ``tb2-train.json`` shapes.

    ``unusable`` leaves both lists; ``exclude`` (default: smoke-6) leaves only
    the hold-out pool.  Unknown task ids and an id in both maps fail closed.
    """
    unusable = dict(unusable or {})
    return build_lists(tasks, quotas=HOLDOUT_QUOTAS, seed=HOLDOUT_SEED,
                       rule=HOLDOUT_RULE_UNUSABLE if unusable else HOLDOUT_RULE, unusable=unusable,
                       exclude=smoke6_exclusions() if exclude is None else exclude)


def build_lists(tasks: HarborTaskSet, *, quotas: Mapping[str, int | None], seed: int, rule: str,
                unusable: Mapping[str, str] | None = None,
                exclude: Mapping[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(holdout, train)`` of any Harbor task set (the TB2 rule is :func:`build_tb2_lists`).

    ``quotas``: hold-out tasks per eval bucket (``<benchmark>-<difficulty>``;
    None: all of the bucket); same meaning of ``unusable`` / ``exclude`` as there.
    """
    unusable = dict(unusable or {})
    exclude = dict(exclude or {})
    known = set(tasks.task_ids())
    for label, ids in (("unusable", unusable), ("excluded", exclude)):
        unknown = sorted(set(ids) - known)
        if unknown:
            raise ValueError(f"{label} tasks not in the checkout {tasks.version}: {unknown}")
    clash = sorted(set(unusable) & set(exclude))
    if clash:
        raise ValueError(f"tasks both in the exclusion list and the unusable table: {clash}")
    holdout = bm.build_holdout(tasks, dict(quotas), seed=seed, rule=rule, exclude={**exclude, **unusable})
    train = bm.build_train_split(tasks, holdout, exclude=unusable)
    return holdout, train


def dump_list(document: Mapping[str, Any]) -> str:
    """The on-disk form of the ``data/eval`` lists."""
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


# --- curriculum ----------------------------------------------------------------------

@dataclass(frozen=True)
class StageRule:
    """A task enters the first stage whose rule it meets (yeto choice, module doc).

    ``max_expert_min`` None: no bound; a task without an expert estimate meets
    only a stage without a bound.
    """

    name: str
    difficulties: tuple[str, ...]
    max_expert_min: float | None = None

    def admits(self, facts: TaskFacts) -> bool:
        if facts.difficulty not in self.difficulties:
            return False
        if self.max_expert_min is None:
            return True
        return facts.expert_time_min is not None and facts.expert_time_min <= self.max_expert_min

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "difficulties": list(self.difficulties), "max_expert_min": self.max_expert_min}


DEFAULT_STAGES: tuple[StageRule, ...] = (
    StageRule("s1-short", ("easy", "medium"), 30.0),
    StageRule("s2-medium", ("easy", "medium", "hard"), 120.0),
    StageRule("s3-long", DIFFICULTIES),
)
_DIFFICULTY_RANK = {d: i for i, d in enumerate(DIFFICULTIES)}


def load_reference_results(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """Reference-solution runs (jsonl rows with ``task``/``task_id`` and ``passed``) -> per task.

    Shape of a positive-control run (``pos-all.jsonl``).  A
    row with an ``error`` and no ``passed`` is a run that gave no verdict (an
    infrastructure failure): it is not a fact about the task and is skipped.
    A task with several verdicts must agree on ``passed`` or it fails closed.
    """
    out: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        tid = row.get("task_id") or row.get("task")
        if tid and "passed" not in row and row.get("error"):
            continue
        if not tid or not isinstance(row.get("passed"), bool):
            raise ValueError(f"{path}:{number}: row needs task/task_id and a boolean passed")
        entry = {"passed": row["passed"], **({"solve_s": float(row["solve_s"])}
                                             if _number(row.get("solve_s")) is not None else {})}
        if tid in out and out[tid]["passed"] != entry["passed"]:
            raise ValueError(f"{path}: reference results disagree for {tid}")
        out.setdefault(tid, entry)
    return out


def load_pass_rates(path: str | os.PathLike[str]) -> dict[str, tuple[int, int]]:
    """Measured baseline results: JSON ``{task_id: {"n": samples, "passed": k}}`` -> ``(k, n)``."""
    raw = json.loads(Path(path).read_text())
    out: dict[str, tuple[int, int]] = {}
    for tid, value in raw.items():
        n, k = value.get("n"), value.get("passed")
        if not (isinstance(n, int) and isinstance(k, int) and 0 <= k <= n and n > 0):
            raise ValueError(f"{path}: bad pass count for {tid}: {value!r}")
        out[tid] = (k, n)
    return out


def build_curriculum(tasks: HarborTaskSet, train: Mapping[str, Any], *,
                     stages: Sequence[StageRule] = DEFAULT_STAGES,
                     reference: Mapping[str, Mapping[str, Any]] | None = None,
                     pass_rates: Mapping[str, tuple[int, int]] | None = None,
                     require_signal: bool = False, source: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Ordered stages of the training list ``train`` (module doc).

    Within a stage: official difficulty, then expert estimate (missing last),
    then task id.  Dropped tasks are listed with the reason.
    """
    if train.get("schema") != bm.TRAIN_SPLIT_SCHEMA:
        raise ValueError(f"not a {bm.TRAIN_SPLIT_SCHEMA} list")
    if require_signal and not pass_rates:
        raise ValueError("require_signal needs measured pass rates")
    staged: dict[str, list[dict[str, Any]]] = {rule.name: [] for rule in stages}
    dropped: list[dict[str, str]] = []
    for item in train["items"]:
        tid = item["task_id"]
        facts = tasks.facts(tid)
        if reference is not None and tid in reference and not reference[tid]["passed"]:
            dropped.append({"task_id": tid, "reason": "reference solution fails in the judge environment"})
            continue
        rate = (pass_rates or {}).get(tid)
        if require_signal and (rate is None or rate[0] in (0, rate[1])):
            why = "no measured pass rate" if rate is None else f"no learning signal: {rate[0]}/{rate[1]} passed"
            dropped.append({"task_id": tid, "reason": why})
            continue
        rule = next((r for r in stages if r.admits(facts)), None)
        if rule is None:
            dropped.append({"task_id": tid, "reason": "no stage admits it"})
            continue
        entry: dict[str, Any] = {"task_id": tid, "difficulty": facts.difficulty, "eval_bucket": facts.eval_bucket,
                                 "expert_time_estimate_min": facts.expert_time_min,
                                 "junior_time_estimate_min": facts.junior_time_min, "category": facts.category}
        if reference is not None and tid in reference:
            entry["reference_solution"] = dict(reference[tid])
        if rate is not None:
            entry["baseline_passed"], entry["baseline_samples"] = rate
        staged[rule.name].append(entry)
    for entries in staged.values():
        entries.sort(key=lambda e: (_DIFFICULTY_RANK.get(e["difficulty"], len(DIFFICULTIES)),
                                    e["expert_time_estimate_min"] is None,
                                    e["expert_time_estimate_min"] or 0.0, e["task_id"]))
    return {
        "schema": CURRICULUM_SCHEMA, "benchmark": BENCHMARK, "benchmark_version": tasks.version,
        "train_split_sha256": bm.holdout_sha256(dict(train)),
        "holdout_sha256": train.get("holdout_sha256"),
        "rule": "first stage whose rule admits the task; within a stage by official difficulty, expert "
                "time estimate, task id. Stage bounds are yeto choices, not data.",
        "require_signal": require_signal,
        **({"source": dict(source)} if source else {}),
        "stages": [{**rule.to_dict(), "items": staged[rule.name]} for rule in stages],
        "dropped": sorted(dropped, key=lambda d: d["task_id"]),
    }


def curriculum_task_ids(curriculum: Mapping[str, Any], *, through: str | None = None) -> list[str]:
    """Task ids of the stages up to and including ``through`` (all when None), in curriculum order."""
    if curriculum.get("schema") != CURRICULUM_SCHEMA:
        raise ValueError(f"not a {CURRICULUM_SCHEMA} file")
    names = [s["name"] for s in curriculum["stages"]]
    if through is not None and through not in names:
        raise ValueError(f"unknown stage {through!r}; stages: {names}")
    out: list[str] = []
    for stage in curriculum["stages"]:
        out += [item["task_id"] for item in stage["items"]]
        if stage["name"] == through:
            break
    return out


def assert_no_holdout(task_ids: Iterable[str], holdout: Mapping[str, Any]) -> None:
    bm.assert_disjoint(task_ids, bm.holdout_ids(dict(holdout)))


# --- dataset rows for the two backends ----------------------------------------------
#
# Formats read from the adapter code and real runs (verl and Miles on TB2.1):
# * Miles (``--input-key prompt --metadata-key metadata``): ``metadata`` is the
#   sample metadata ``harbor_agent.task_from_metadata`` reads (``instance_id`` is
#   the Harbor task directory, ``agent_name`` the agent); Harbor reads the
#   instruction from the task directory, so ``prompt`` is only a label.
# * verl (``HarborAgentLoop``): ``agent_name: yeto_harbor`` selects the agent
#   loop; ``extra_info.agent_task`` is the ``AgentTask`` mapping.
# Model name, timeouts and the provider default are deployment facts and stay out
# of the rows (worker environment: AGENT_MODEL_NAME, AGENT_TIMEOUT, HARBOR_ENV_TYPE).

VERL_AGENT_LOOP = "yeto_harbor"


def _label(tasks: HarborTaskSet, task_id: str) -> str:
    return f"{tasks.version.split('@')[0].upper()} task {task_id}"


def task_images(tasks: HarborTaskSet, task_ids: Sequence[str]) -> list[str]:
    """The prebuilt images of ``task_ids`` (sorted, unique), to pre-pull on the sandbox hosts.

    A task without ``environment.docker_image`` is refused: its sandbox would
    be built on the host at trial time."""
    missing = [t for t in task_ids if not tasks.facts(t).docker_image]
    if missing:
        raise ValueError(f"tasks without a prebuilt docker_image: {missing}")
    return sorted({tasks.facts(t).docker_image for t in task_ids})


def miles_rows(tasks: HarborTaskSet, task_ids: Sequence[str], *, agent: str = "codex",
               provider: str | None = None, holdout: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Miles rows (also the agentic engine sampler's, ``task_key: metadata``).

    ``provider``: the row's sandbox provider (unset: the worker's ``HARBOR_ENV_TYPE``);
    ``holdout``: a hold-out list the ids must not touch (raises on a leak).
    """
    if holdout is not None:
        assert_no_holdout(task_ids, holdout)
    out = []
    for tid in task_ids:
        f = tasks.facts(tid)
        out.append({"prompt": _label(tasks, tid), "metadata": {
            "instance_id": tid, "agent_name": agent, "difficulty": f.difficulty, "eval_bucket": f.eval_bucket,
            "benchmark_version": tasks.version, **({"provider": provider} if provider else {})}})
    return out


def verl_rows(tasks: HarborTaskSet, task_ids: Sequence[str], *, agent: str = "codex",
              data_source: str = BENCHMARK, provider: str | None = None) -> list[dict[str, Any]]:
    out = []
    for index, tid in enumerate(task_ids):
        f = tasks.facts(tid)
        out.append({"data_source": data_source, "agent_name": VERL_AGENT_LOOP,
                    "prompt": [{"role": "user", "content": _label(tasks, tid)}],
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {"index": index, "difficulty": f.difficulty, "eval_bucket": f.eval_bucket,
                                   "benchmark_version": tasks.version,
                                   "agent_task": {"task_id": tid, "agent": agent,
                                                  **({"provider": provider} if provider else {})}}})
    return out


def dump_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)


# --- command line ---------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    """``python -m yeto.rl.harness.harbor.task_lists rows|images ...`` (any Harbor task directory)."""
    import argparse

    p = argparse.ArgumentParser(prog="python -m yeto.rl.harness.harbor.task_lists")
    p.add_argument("what", choices=("rows", "images"))
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--benchmark", help="task set name (default: tb2 / tb2.1 by layout)")
    p.add_argument("--ids", help="file with one task id per line (default: every task in --tasks-dir)")
    p.add_argument("--holdout", help="hold-out list JSON; its tasks are refused")
    p.add_argument("--agent", default="pi")
    p.add_argument("--provider")
    p.add_argument("--format", choices=("miles", "verl"), default="miles")
    p.add_argument("--out", help="output file (default: stdout)")
    args = p.parse_args(argv)
    tasks = HarborTaskSet(args.tasks_dir, benchmark=args.benchmark)
    ids = ([line.strip() for line in Path(args.ids).read_text().splitlines() if line.strip()
            and not line.startswith("#")] if args.ids else tasks.task_ids())
    unknown = sorted(set(ids) - set(tasks.task_ids()))
    if unknown:
        raise SystemExit(f"task ids not in {tasks.tasks_dir}: {unknown}")
    if args.holdout:
        assert_no_holdout(ids, json.loads(Path(args.holdout).read_text()))
    if args.what == "images":
        text = "".join(f"{image}\n" for image in task_images(tasks, ids))
    elif args.format == "miles":
        text = dump_jsonl(miles_rows(tasks, ids, agent=args.agent, provider=args.provider))
    else:
        text = dump_jsonl(verl_rows(tasks, ids, agent=args.agent, data_source=tasks.name, provider=args.provider))
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
