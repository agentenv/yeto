"""SWE-bench Verified adapter (change rl-agentic-reward-env, second benchmark).

Judging follows the official harness (``swebench`` 5.0.2,
``swebench/harness/run_evaluation.py::run_instance``) step by step:

1. fresh sandbox from the instance's prebuilt image (dataset field ``image``,
   ``swebench/sweb.eval.x86_64.<id>:latest`` on Docker Hub, pinned by digest in
   our image manifest);
2. the model patch is applied with the official fallback chain
   (``GIT_APPLY_CMDS``: ``git apply --verbose`` / ``--3way`` / ``--reject`` /
   ``patch --batch --forward --fuzz=5 -p1 -i``, resetting the tree between
   attempts, then ``git apply --check --reverse``); a patch that does not apply
   is "not resolved", exactly as the official run;
3. the dataset's ``eval_script`` (which resets the test files, applies the
   gold ``test_patch`` and runs the FAIL_TO_PASS / PASS_TO_PASS tests) runs as
   ``/eval.sh`` with the official exit-code recording;
4. the test output is graded on the trusted side by the official
   ``swebench.harness.grading.get_eval_report`` (resolved iff every
   FAIL_TO_PASS and PASS_TO_PASS test passes).

``submission=None`` is the official no-patch mode (negative control); the gold
``patch`` is the positive control.  The official package is imported lazily, so
this module (task list, metadata, judge script) works without it; grading
needs it.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from .benchmark import PrebakePlan, TaskSpec, register

NAME = "swebench-verified"
DATASET = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
DATASET_ENV = "YETO_REWARD_ENV_SWEV_DATA"  # local parquet/jsonl of the pinned revision
OFFICIAL_HARNESS = "swebench==5.0.2"

WORKDIR = "/testbed"
PATCH_PATH = "/tmp/patch.diff"
EVAL_PATH = "/eval.sh"
# swebench/harness/run_evaluation.py GIT_APPLY_CMDS (5.0.2)
GIT_APPLY_CMDS = (
    "git apply --verbose",
    "git apply --verbose --3way",
    "git apply --verbose --reject",
    "patch --batch --forward --fuzz=5 -p1 -i",
)
APPLY_MARK = "YETO_SWEV_APPLY="
BEGIN_MARK = "YETO_SWEV_TEST_OUTPUT_BEGIN"
END_MARK = "YETO_SWEV_TEST_OUTPUT_END"

# Official difficulty annotation (dataset column ``difficulty``) -> WP3 buckets.
DIFFICULTY_BUCKETS = {
    "<15 min fix": "swev-lt15m",
    "15 min - 1 hour": "swev-15m-1h",
    "1-4 hours": "swev-ge1h",
    ">4 hours": "swev-ge1h",
}
HOLDOUT_SEED = 20261008
HOLDOUT_QUOTAS: dict[str, int | None] = {"swev-lt15m": 30, "swev-15m-1h": 30, "swev-ge1h": None}
HOLDOUT_RULE = "evaluation only (never trained): <15 min 30 / 15 min-1 h 30 / >=1 h all 45"

DEFAULT_CPUS = 4  # swebench/harness/modal_eval/run_evaluation_modal.py uses cpu=4
DEFAULT_MEMORY_MB = 8192  # [estimate] to be measured (tasks 2.x)
DEFAULT_AGENT_TIMEOUT_S = 3600.0
JUDGE_TIMEOUT_S = 1800.0  # official run_evaluation default --timeout


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    return [str(v) for v in (value or [])]


def load_rows(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        rows = pq.read_table(path).to_pylist()
    elif path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text())
    required = {"instance_id", "repo", "version", "base_commit", "image", "eval_script", "log_parser", "eval_type",
                "FAIL_TO_PASS", "PASS_TO_PASS", "difficulty", "problem_statement"}
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"{row.get('instance_id')}: dataset row lacks {sorted(missing)} "
                             f"(use {DATASET}@{DATASET_REVISION[:8]}, not princeton-nlp)")
    ids = [r["instance_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate instance_id")
    return rows


def _b64_heredoc(path: str, text: str) -> str:
    return f"echo {base64.b64encode(text.encode()).decode()} | base64 -d > {path}"


def judge_script(row: dict[str, Any], submission: str | None) -> str:
    """Bash for a fresh sandbox of ``row['image']`` (official run_instance order)."""
    from_spec = eval_script_of(row)
    lines = ["cd " + WORKDIR]
    if submission is None:
        lines.append(f"echo {APPLY_MARK}skipped")
    else:
        lines.append(_b64_heredoc(PATCH_PATH, submission))
        attempts = []
        for index, cmd in enumerate(GIT_APPLY_CMDS):
            reset = "{ git checkout -- . ; git clean -fd ; } >/dev/null 2>&1; " if index else ""
            attempts.append(f"{reset}{cmd} {PATCH_PATH}")
        chain = " || ".join(f"( {a} )" for a in attempts)
        lines += [
            f"if {chain}; then echo {APPLY_MARK}pass; "
            f"elif git apply --check --reverse {PATCH_PATH}; then echo {APPLY_MARK}pass; "
            f"else echo {APPLY_MARK}fail; exit 0; fi",
        ]
    lines += [
        _b64_heredoc(EVAL_PATH, from_spec),
        f"echo {BEGIN_MARK}",
        f"/bin/bash {EVAL_PATH} 2>&1",
        f"echo; echo {END_MARK}",
    ]
    return "\n".join(lines) + "\n"


def eval_script_of(row: dict[str, Any]) -> str:
    """The official eval script with exit-code recording (``make_test_spec`` when available)."""
    try:
        from swebench.harness.utils import make_test_spec
    except ImportError:
        return row["eval_script"]
    return make_test_spec(dict(row)).eval_script


def submission_command(row: dict[str, Any]) -> str:
    """Extract the agent's answer from its own sandbox: everything changed since base_commit."""
    return (f"cd {WORKDIR} && git add -A >/dev/null && "
            f"git -c core.fileMode=false diff --cached {shlex.quote(row['base_commit'])}")


def _between(output: str, begin: str, end: str) -> str | None:
    if begin not in output:
        return None
    tail = output.split(begin, 1)[1]
    return tail.split(end, 1)[0] if end in tail else tail


def grade(row: dict[str, Any], output: str, submission: str | None) -> dict[str, Any]:
    """Official grading of one judge run's output (trusted side; needs ``swebench``)."""
    from swebench.harness.grading import get_eval_report
    from swebench.harness.utils import make_test_spec

    apply = output.split(APPLY_MARK, 1)[1].split()[0] if APPLY_MARK in output else "missing"
    base = {"applied": apply, "resolved": False, "infra_error": False}
    if apply == "fail":
        return base  # official: "Patch Apply Failed" -> not resolved
    test_output = _between(output, BEGIN_MARK, END_MARK)
    if apply == "missing" or test_output is None:
        return {**base, "infra_error": True, "infra_reason": "judge script did not run to the tests"}
    spec = make_test_spec(dict(row))
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
        handle.write(test_output)
        log_path = handle.name
    try:
        prediction = {"instance_id": row["instance_id"], "model_name_or_path": "yeto",
                      "model_patch": "" if submission is None else submission}
        report = get_eval_report(test_spec=spec, prediction=prediction, test_log_path=log_path,
                                 include_tests_status=True)[row["instance_id"]]
    finally:
        os.unlink(log_path)
    tests = report.get("tests_status") or {}
    counts = {f"{k}_{s}": len(v.get(s, [])) for k, v in tests.items() for s in ("success", "failure")}
    return {**base, "resolved": bool(report.get("resolved")), "infra_error": bool(report.get("infra_failure")),
            "infra_reason": report.get("infra_failure_reason"), "tests": counts}


class SwebenchVerified:
    name = NAME

    def __init__(self, data: str | os.PathLike[str] | None = None, *, images: dict[str, str] | None = None,
                 revision: str = DATASET_REVISION) -> None:
        raw = data or os.environ.get(DATASET_ENV)
        if not raw:
            raise RuntimeError(f"{DATASET_ENV} is not set (local copy of {DATASET}@{revision[:8]})")
        self.rows = {r["instance_id"]: r for r in load_rows(raw)}
        self.version = f"{DATASET}@{revision}"
        # instance_id -> pinned image reference (``repo@sha256:…``) from the image manifest
        self.images = dict(images or {})
        self._submissions: dict[str, str | None] = {}

    def task_ids(self) -> list[str]:
        return sorted(self.rows)

    def row(self, task_id: str) -> dict[str, Any]:
        try:
            return self.rows[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown SWE-bench Verified instance {task_id!r}") from exc

    def task_spec(self, task_id: str) -> TaskSpec:
        row = self.row(task_id)
        difficulty = str(row["difficulty"])
        return TaskSpec(benchmark=NAME, task_id=task_id, base_image=self.images.get(task_id, "docker.io/" + row["image"]),
                        cpus=DEFAULT_CPUS, memory_mb=DEFAULT_MEMORY_MB, workdir=WORKDIR,
                        agent_timeout_s=DEFAULT_AGENT_TIMEOUT_S, judge_timeout_s=JUDGE_TIMEOUT_S,
                        instruction=row["problem_statement"], benchmark_version=self.version,
                        difficulty=difficulty, difficulty_source="swebench-verified-hf-difficulty",
                        eval_bucket=DIFFICULTY_BUCKETS.get(difficulty))

    def prebake_plan(self, task_id: str) -> PrebakePlan:
        # The official instance images already carry the conda env and the repo at base_commit.
        return PrebakePlan(base_image=self.task_spec(task_id).base_image, commands=(), complete=True)

    def judge_command(self, task_id: str, submission: str | None = None) -> str:
        self._submissions[task_id] = submission
        return judge_script(self.row(task_id), submission)

    def parse_judge(self, task_id: str, output: str, exit_code: int, timed_out: bool) -> dict[str, Any]:
        if timed_out:
            # official: "Tests Timed Out" -> not resolved; recorded, not an infrastructure error
            return {"reward": 0.0, "passed": False, "applied": None, "timed_out_tests": True}
        result = grade(self.row(task_id), output, self._submissions.get(task_id))
        return {"reward": 1.0 if result["resolved"] else 0.0, "passed": result["resolved"], **result}

    def gold_patch(self, task_id: str) -> str:
        """Positive control: the official reference patch."""
        return self.row(task_id)["patch"]

    def contamination_keys(self, task_id: str) -> dict[str, str]:
        """WP3 D6c extra keys: (repo, base_commit) and normalised problem statement hash."""
        import hashlib

        row = self.row(task_id)
        statement = " ".join(str(row["problem_statement"]).lower().split())
        return {"repo_base_commit": f"{row['repo']}@{row['base_commit']}",
                "problem_statement_sha256": hashlib.sha256(statement.encode()).hexdigest()}


register(NAME, SwebenchVerified)


def build_holdout(adapter: SwebenchVerified, **kwargs: Any) -> dict[str, Any]:
    from .benchmark import build_holdout as _build

    return _build(adapter, HOLDOUT_QUOTAS, seed=HOLDOUT_SEED, rule=HOLDOUT_RULE, **kwargs)


def docker_hub_repo(image: str) -> tuple[str, str]:
    """``swebench/sweb.eval.x86_64.x:latest`` -> (``swebench/sweb.eval.x86_64.x``, ``latest``)."""
    ref = image.split("docker.io/", 1)[-1]
    repo, _, tag = ref.partition(":")
    return repo, tag or "latest"
