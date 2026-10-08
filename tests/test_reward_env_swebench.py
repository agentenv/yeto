"""CPU tests for the SWE-bench Verified reward-env adapter (rl-agentic-reward-env).

Grading tests use the official ``swebench`` package on a real dataset row of the
pinned revision; they skip when either is missing (set YETO_TEST_SWEV_DATA to the
parquet of SWE-bench/SWE-bench_Verified@78f471bf).
"""

from __future__ import annotations

import base64
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.harness.reward_env import benchmark as bm
from yeto.rl.harness.reward_env import swebench_verified as sv


def _row(iid, difficulty="15 min - 1 hour", repo="org/repo"):
    return {"instance_id": iid, "repo": repo, "version": "1.0", "base_commit": "abc123", "image": f"swebench/sweb.eval.x86_64.{iid}:latest",
            "eval_script": "#!/bin/bash\necho run tests\n", "log_parser": "parse_log_pytest", "eval_type": "pass_and_fail",
            "FAIL_TO_PASS": ["t::a"], "PASS_TO_PASS": ["t::b"], "difficulty": difficulty,
            "problem_statement": f"  Fix   {iid}\n", "patch": "diff --git a/x b/x\n"}


@pytest.fixture
def synthetic(tmp_path):
    rows = ([_row(f"lt{i:02d}", "<15 min fix") for i in range(32)] + [_row(f"mid{i:02d}") for i in range(31)]
            + [_row(f"hi{i:02d}", "1-4 hours") for i in range(4)] + [_row("huge", ">4 hours")])
    path = tmp_path / "swev.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return sv.SwebenchVerified(path)


def test_registry_and_metadata(synthetic, tmp_path):
    assert "swebench-verified" in bm.registered()
    adapter = bm.get_adapter("swebench-verified", data=tmp_path / "swev.jsonl")
    spec = adapter.task_spec("hi00")
    assert spec.metadata() == {"task_id": "hi00", "benchmark": "swebench-verified",
                               "benchmark_version": f"{sv.DATASET}@{sv.DATASET_REVISION}",
                               "difficulty": "1-4 hours", "difficulty_source": "swebench-verified-hf-difficulty",
                               "eval_bucket": "swev-ge1h"}
    assert synthetic.task_spec("huge").eval_bucket == "swev-ge1h"
    assert synthetic.task_spec("lt00").eval_bucket == "swev-lt15m"
    assert spec.workdir == "/testbed" and spec.base_image == "docker.io/swebench/sweb.eval.x86_64.hi00:latest"
    assert synthetic.prebake_plan("lt00").empty and synthetic.prebake_plan("lt00").complete


def test_pinned_images_override(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(_row("a")))
    adapter = sv.SwebenchVerified(path, images={"a": "docker.io/swebench/x@sha256:" + "0" * 64})
    assert adapter.task_spec("a").base_image.endswith("@sha256:" + "0" * 64)


def test_load_rows_rejects_old_format_and_duplicates(tmp_path):
    old = _row("a")
    del old["eval_script"], old["image"]
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="princeton-nlp"):
        sv.load_rows(path)
    path.write_text(json.dumps(_row("a")) + "\n" + json.dumps(_row("a")))
    with pytest.raises(ValueError, match="duplicate"):
        sv.load_rows(path)


def test_holdout_buckets(synthetic):
    h = sv.build_holdout(synthetic)
    assert Counter(i["eval_bucket"] for i in h["items"]) == {"swev-lt15m": 30, "swev-15m-1h": 30, "swev-ge1h": 5}
    assert "never trained" in h["rule"] and h == sv.build_holdout(synthetic)


def _decoded(script, path):
    for line in script.splitlines():
        if line.endswith(f"> {path}") and line.startswith("echo "):
            return base64.b64decode(line.split()[1]).decode()
    return None


def test_judge_script_official_apply_chain(synthetic):
    script = synthetic.judge_command("lt00", "diff --git a/f b/f\n")
    assert _decoded(script, sv.PATCH_PATH) == "diff --git a/f b/f\n"
    positions = [script.index(cmd + " " + sv.PATCH_PATH) for cmd in sv.GIT_APPLY_CMDS]
    assert positions == sorted(positions)  # official order
    assert script.count("git checkout -- . ; git clean -fd") == len(sv.GIT_APPLY_CMDS) - 1
    assert f"git apply --check --reverse {sv.PATCH_PATH}" in script
    assert f"{sv.APPLY_MARK}fail; exit 0" in script
    assert script.index(sv.BEGIN_MARK) < script.index(f"/bin/bash {sv.EVAL_PATH}") < script.index(sv.END_MARK)
    assert "echo run tests" in _decoded(script, sv.EVAL_PATH)


def test_no_patch_mode_is_negative_control(synthetic):
    script = synthetic.judge_command("lt00", None)
    assert f"{sv.APPLY_MARK}skipped" in script and sv.PATCH_PATH not in script
    assert synthetic.gold_patch("lt00").startswith("diff --git")


def test_submission_command_and_contamination(synthetic):
    assert "git diff --cached abc123" in sv.submission_command(synthetic.row("lt00")).replace("-c core.fileMode=false ", "")
    keys = synthetic.contamination_keys("lt00")
    assert keys["repo_base_commit"] == "org/repo@abc123"
    assert keys["problem_statement_sha256"] == synthetic.contamination_keys("lt00")["problem_statement_sha256"]


def test_timeout_is_not_resolved(synthetic):
    assert synthetic.parse_judge("lt00", "", 124, True) == {"reward": 0.0, "passed": False, "applied": None,
                                                            "timed_out_tests": True}


# --- official grading on a real row --------------------------------------------

def _real_adapter():
    pytest.importorskip("swebench")
    path = os.environ.get("YETO_TEST_SWEV_DATA", "/home/michael/work/swebench-data/swev-78f471bf.parquet")
    if not Path(path).is_file():
        pytest.skip("pinned SWE-bench Verified parquet not available")
    return sv.SwebenchVerified(path)


def _django_log(passed, failed=()):
    lines = [f"{t} ... ok" for t in passed] + [f"{t} ... FAIL" for t in failed]
    return "\n".join(lines) + "\n"


class FakeHandle:
    def __init__(self, output):
        self.output = output

    def exec(self, command, *, timeout_s, workdir=None):
        return SimpleNamespace(output=self.output, exit_code=0, timed_out=False)


def _output(apply, log):
    return f"{sv.APPLY_MARK}{apply}\n{sv.BEGIN_MARK}\n>>>>> Start Test Output\n{log}>>>>> End Test Output\n{sv.END_MARK}\n"


def test_official_grading_resolved_and_not(tmp_path):
    adapter = _real_adapter()
    iid = "django__django-11099"
    row = adapter.row(iid)
    everything = list(row["FAIL_TO_PASS"]) + list(row["PASS_TO_PASS"])
    ok = bm.run_judge(adapter, iid, FakeHandle(_output("pass", _django_log(everything))), prebaked=False,
                      submission=adapter.gold_patch(iid))
    assert ok.passed and ok.reward == 1.0 and not ok.infra_error
    assert ok.extra["tests"]["FAIL_TO_PASS_success"] == len(row["FAIL_TO_PASS"])
    one_f2p_fails = _django_log(everything[1:], failed=everything[:1])
    bad = bm.run_judge(adapter, iid, FakeHandle(_output("pass", one_f2p_fails)), prebaked=False, submission="x")
    assert not bad.passed and bad.extra["tests"]["FAIL_TO_PASS_failure"] == 1
    p2p_breaks = _django_log(row["FAIL_TO_PASS"], failed=row["PASS_TO_PASS"][:1])
    assert not bm.run_judge(adapter, iid, FakeHandle(_output("pass", p2p_breaks)), prebaked=False, submission="x").passed


def test_official_grading_apply_fail_and_infra():
    adapter = _real_adapter()
    iid = "django__django-11099"
    failed = bm.run_judge(adapter, iid, FakeHandle(f"{sv.APPLY_MARK}fail\n"), prebaked=False, submission="junk")
    assert not failed.passed and not failed.infra_error and failed.extra["applied"] == "fail"
    broken = bm.run_judge(adapter, iid, FakeHandle("bash: something died\n"), prebaked=False, submission="x")
    assert broken.infra_error and not broken.passed


def test_pinned_dataset_facts():
    adapter = _real_adapter()
    assert len(adapter.task_ids()) == 500
    buckets = Counter(adapter.task_spec(t).eval_bucket for t in adapter.task_ids())
    assert buckets == {"swev-lt15m": 194, "swev-15m-1h": 261, "swev-ge1h": 45}
    h = sv.build_holdout(adapter)
    assert len(h["items"]) == 105
