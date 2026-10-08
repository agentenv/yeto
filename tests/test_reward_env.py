"""CPU tests for rl-agentic-reward-env (benchmark-neutral reward sandbox, TB2 first)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.harness.reward_env import benchmark as bm
from yeto.rl.harness.reward_env import tb2 as rt

UVX_TEST_SH = """#!/bin/bash
# Install curl
apt-get update
apt-get install -y curl

curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env
if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    exit 1
fi
uvx \\
  -p 3.13 \\
  -w pytest==8.4.1 \\
  -w pytest-json-ctrf==0.3.5 \\
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
"""


def make_task(root: Path, task_id: str, test_sh: str = UVX_TEST_SH, image: str = "org/img:1",
              difficulty: str = "medium") -> Path:
    d = root / task_id
    (d / "tests").mkdir(parents=True)
    (d / "environment").mkdir()
    (d / "task.toml").write_text(
        f'[metadata]\ndifficulty = "{difficulty}"\n'
        f'[environment]\ndocker_image = "{image}"\ncpus = 2\nmemory_mb = 4096\n'
        "[agent]\ntimeout_sec = 600\n[verifier]\ntimeout_sec = 300\n"
    )
    (d / "tests" / "test.sh").write_text(test_sh)
    (d / "environment" / "Dockerfile").write_text("FROM x\nWORKDIR /app\n")
    (d / "instruction.md").write_text(f"do {task_id}\n")
    return d


# --- prebake plan extraction ---------------------------------------------------

def test_uvx_plan_is_setup_plus_warmup():
    plan = rt.prebake_from_test_sh(UVX_TEST_SH, "img")
    assert plan.complete
    assert plan.commands == (
        "apt-get update",
        "apt-get install -y curl",
        "curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh",
        "source $HOME/.local/bin/env",
        "uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version",
    )
    script = rt.prebake_script(plan)
    assert script.startswith("set -e\n") and script.rstrip().endswith("pytest --version")


def test_pip_plan():
    sh = 'pip install pytest==8.4.1 pytest-json-ctrf==0.3.5 --break-system-packages\npython -m pytest /tests -rA\n'
    plan = rt.prebake_from_test_sh(sh, "img")
    assert plan.complete and plan.commands == ("pip install pytest==8.4.1 pytest-json-ctrf==0.3.5 --break-system-packages",)


def test_uv_run_is_test_line_and_venv_setup_kept():
    sh = "uv venv -p 3.12 .tb\nsource .tb/bin/activate\nuv pip install pytest==8.4.1\nuv run pytest /tests -rA\n"
    plan = rt.prebake_from_test_sh(sh, "img")
    assert plan.complete and plan.commands[-1] == "uv pip install pytest==8.4.1"


def test_task_specific_lines_stay_at_judge_time_and_vars_block_warmup():
    sh = ("apt-get update\nX='abc'\ncurl -L -o w.pt https://h/w.pt\ncp /tests/a .\n"
          'uvx -p 3.11 -w "git+https://g@${X}" pytest /tests\n')
    plan = rt.prebake_from_test_sh(sh, "img")
    assert plan.commands == ("apt-get update",)
    assert not plan.complete
    assert "curl -L -o w.pt https://h/w.pt" in plan.skipped and "cp /tests/a ." in plan.skipped
    assert any(s.startswith("uvx ") for s in plan.skipped)


def test_uvx_value_flags():
    line = "uvx --index https://x/cpu --index-strategy unsafe-best-match -p 3.11 -w torch==2.5.1 pytest -q"
    assert rt._uvx_warmup(line).endswith("-w torch==2.5.1 pytest --version")


def test_no_test_line_means_no_guessing():
    plan = rt.prebake_from_test_sh("apt-get update\nmake test\n", "img")
    assert plan.commands == () and not plan.complete


def test_digest_is_content_hash():
    a = rt.prebake_from_test_sh(UVX_TEST_SH, "img")
    assert a.digest() == rt.prebake_from_test_sh(UVX_TEST_SH, "img").digest()
    assert a.digest() != rt.prebake_from_test_sh(UVX_TEST_SH, "img2").digest()
    assert a.digest() != rt.prebake_from_test_sh(UVX_TEST_SH.replace("8.4.1", "8.4.2"), "img").digest()


# --- registry + adapter ----------------------------------------------------------

def test_registry_lazy_import_and_errors(tmp_path):
    make_task(tmp_path, "t1")
    adapter = bm.get_adapter("tb2", tasks_dir=tmp_path)
    assert adapter.name == "tb2" and "tb2" in bm.registered()
    with pytest.raises(KeyError, match="unknown benchmark"):
        bm.get_adapter("no_such_bench")
    with pytest.raises(ValueError, match="already registered"):
        bm.register("tb2", lambda **_: None)


def test_adapter_spec_and_plans(tmp_path):
    make_task(tmp_path, "t1")
    make_task(tmp_path, "t2", image="org/other:2")
    adapter = rt.Tb2Benchmark(tmp_path)
    assert adapter.task_ids() == ["t1", "t2"]
    spec = adapter.task_spec("t2")
    assert (spec.base_image, spec.cpus, spec.memory_mb, spec.workdir, spec.judge_timeout_s) == ("org/other:2", 2, 4096, "/app", 300.0)
    assert spec.instruction.startswith("do t2")
    assert set(bm.plans(adapter)) == {"t1", "t2"}
    assert "test.sh" in adapter.judge_command("t1")


# --- judge runner ----------------------------------------------------------------

class FakeHandle:
    def __init__(self, output, exit_code=0, timed_out=False):
        self.result = SimpleNamespace(output=output, exit_code=exit_code, timed_out=timed_out)
        self.commands = []

    def exec(self, command, *, timeout_s, workdir=None):
        self.commands.append((command, timeout_s))
        return self.result


@pytest.mark.parametrize("reward,passed", [("1", True), ("0", False), ("", False)])
def test_run_judge(tmp_path, reward, passed):
    make_task(tmp_path, "t1")
    adapter = rt.Tb2Benchmark(tmp_path)
    handle = FakeHandle(f"pytest ok\nYETO_TB2_TESTSH_RC=0\nYETO_TB2_REWARD={reward}\n")
    ticks = iter([10.0, 12.5])
    result = bm.run_judge(adapter, "t1", handle, prebaked=True, clock=lambda: next(ticks))
    assert (result.passed, result.reward, result.seconds, result.prebaked) == (passed, float(passed), 2.5, True)
    assert result.extra["testsh_rc"] == 0 and handle.commands[0][1] == 300.0
    json.dumps(result.to_dict())


# --- Modal backend (fake modal module) -------------------------------------------

class _FakeImage:
    calls: list = []

    @classmethod
    def from_registry(cls, ref):
        img = cls()
        img.ops = [("from_registry", ref)]
        return img

    def run_commands(self, *cmds):
        self.ops.append(("run_commands", cmds))
        return self


@pytest.fixture
def fake_modal(monkeypatch):
    created = {}

    class Sandbox:
        @staticmethod
        def create(*args, **kwargs):
            created.update(kwargs, args=args)
            return SimpleNamespace(object_id="sb-1")

    class App:
        @staticmethod
        def lookup(name, create_if_missing=False):
            created["app_name"] = name
            return SimpleNamespace(name=name)

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Image=_FakeImage, Sandbox=Sandbox, App=App))
    return created


@pytest.mark.parametrize("prebake", [True, False])
def test_prebaked_backend(tmp_path, fake_modal, prebake):
    from yeto.rl.harness.codex import tb2_provider as tb2

    make_task(tmp_path, "t1")
    task = tb2.resolve_task("t1", tmp_path)
    from yeto.cloud import modal_reward_env

    backend = modal_reward_env.PrebakedModalSandboxBackend(prebake=prebake)
    handle = backend.create(task, "traj-1")
    assert fake_modal["app_name"] == "yeto-reward-env" and fake_modal["app"].name == "yeto-reward-env"
    assert handle.workdir == "/app"
    ops = fake_modal["image"].ops
    assert ops[0] == ("from_registry", "org/img:1")
    if prebake:
        assert ops[1][0] == "run_commands" and "pytest --version" in ops[1][1][0]
        assert fake_modal["tags"]["yeto-prebake"] == rt.prebake_from_test_sh(UVX_TEST_SH, "org/img:1").digest()[:16]
    else:
        assert len(ops) == 1 and fake_modal["tags"]["yeto-prebake"] == "none"
    assert fake_modal["cpu"] == 2.0 and fake_modal["memory"] == 4096


def test_build_never_targets_production_app():
    with pytest.raises(ValueError, match="production"):
        rt.check_build_app("yeto-tbench2")
    assert rt.check_build_app("yeto-reward-env-build") == "yeto-reward-env-build"


def test_build_tool_plan_only(tmp_path, capsys):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "reward_env"))
    import build_images

    make_task(tmp_path, "t1")
    out = tmp_path / "plan.json"
    assert build_images.main(["--tasks-dir", str(tmp_path), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["total"] == 1 and report["tasks"][0]["complete"]
    with pytest.raises(SystemExit):
        build_images.main(["--tasks-dir", str(tmp_path), "--build"])  # no --app


# --- hold-out list (WP3 D6b) + metadata (D6a) ------------------------------------

def _tb2_pool(root):
    for i in range(4):
        make_task(root, f"e{i}", difficulty="easy")
    for i in range(20):
        make_task(root, f"m{i:02d}", difficulty="medium")
    for i in range(12):
        make_task(root, f"h{i:02d}", difficulty="hard")
    return rt.Tb2Benchmark(root, version="tb2@test")


def test_tb2_metadata_shape(tmp_path):
    make_task(tmp_path, "t1", difficulty="hard")
    meta = rt.Tb2Benchmark(tmp_path, version="tb2@abc1234").task_spec("t1").metadata()
    assert meta == {"task_id": "t1", "benchmark": "tb2", "benchmark_version": "tb2@abc1234",
                    "difficulty": "hard", "difficulty_source": "tb2-task.toml", "eval_bucket": "tb2-hard"}


def test_tb2_holdout_stratified_and_reproducible(tmp_path):
    adapter = _tb2_pool(tmp_path)
    h = rt.build_holdout(adapter)
    assert h["schema"] == "yeto-eval-holdout/1" and h["seed"] == 20261008 and h["benchmark_version"] == "tb2@test"
    from collections import Counter

    assert Counter(i["eval_bucket"] for i in h["items"]) == {"tb2-easy": 2, "tb2-medium": 18, "tb2-hard": 10}
    assert rt.build_holdout(adapter, task_ids=list(reversed(adapter.task_ids()))) == h
    assert bm.holdout_sha256(h) == bm.holdout_sha256(rt.build_holdout(adapter))
    ids = bm.holdout_ids(h)
    excluded = rt.build_holdout(adapter, exclude=ids[:1] if ids[0].startswith("m") else ids[2:3])
    assert len(excluded["items"]) == 30
    with pytest.raises(ValueError, match="quota"):
        rt.build_holdout(adapter, exclude=["e0", "e1", "e2"])


def test_rows_split_and_leak_check():
    rows = [{"metadata": {"task_id": "a"}}, {"metadata": {"task_id": "c"}}, {"task_id": "b"}]
    train, ev = bm.split_rows(rows, ["a", "b"])
    assert [bm.task_id_of(r) for r in train] == ["c"] and len(ev) == 2
    with pytest.raises(ValueError):
        bm.split_rows([{"prompt": "x"}], ["a"])
    bm.assert_disjoint(["c"], ["a"])
    with pytest.raises(ValueError, match="leak"):
        bm.assert_disjoint(["a"], ["a"])
    with pytest.raises(ValueError, match="duplicate"):
        bm.holdout_ids({"schema": bm.HOLDOUT_SCHEMA, "items": [{"task_id": "a"}, {"task_id": "a"}]})


def test_run_judge_infra_error(tmp_path):
    make_task(tmp_path, "t1")

    class Gone:
        def exec(self, command, *, timeout_s, workdir=None):
            raise RuntimeError("sandbox terminated")

    result = bm.run_judge(rt.Tb2Benchmark(tmp_path), "t1", Gone(), prebaked=False)
    assert result.infra_error and not result.passed and "terminated" in result.log


# --- launcher recognises the new provider -----------------------------------------

def test_launcher_treats_prebaked_provider_as_modal():
    from yeto import launcher as L

    assert L.PREBAKED_MODAL_SANDBOX_PROVIDER in L.MODAL_SANDBOX_PROVIDERS
    assert L.MODAL_SANDBOX_PROVIDER in L.MODAL_SANDBOX_PROVIDERS
