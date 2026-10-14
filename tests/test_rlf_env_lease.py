"""RLF CTF environment lease (W4, breakpoints 3+5), CPU only.

Fake VM exec / fake verifier: orchestration order, flag hygiene (never in
stdout/logs/exceptions), fail-closed grade parsing, idempotent teardown, and
the finish_trusted dual path (RLF grade vs byte-identical TB2 fallback).
No Modal, no GPU, no nested Docker.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from yeto.rl.rlf import modal as rlf_modal
from yeto.rl.rlf.contracts import TrustedGrade
from yeto.rl.rlf.preflight import RolloutRejected
from yeto.rl.harness.codex import codex_openenv_agent_function as adapter
from yeto.rl import tbench_outcome

FLAG = "flag{trusted_vm_acceptance_7b3d}"
TASK = "fixture-ctf-ret2win"


# --- fakes -------------------------------------------------------------------

class FakeExec:
    """Scripted VmExec: records every command, replays canned results."""

    def __init__(self, responses: dict[str, tuple[int, str]] | None = None):
        self.commands: list[str] = []
        self.responses = responses or {}

    def __call__(self, command: str, *, timeout_s: float) -> rlf_modal.VmExecResult:
        self.commands.append(command)
        for needle, (rc, out) in self.responses.items():
            if needle in command:
                return rlf_modal.VmExecResult(rc, out)
        return rlf_modal.VmExecResult(0, "")


def healthy_exec() -> FakeExec:
    return FakeExec({
        "dockerd --host": (0, ""),
        " info ": (0, "Server: Docker"),
        " build --network host": (0, "Successfully built"),
        " run -d ": (0, "c0ffee123\n"),
        " inspect -f ": (0, "running healthy\n"),
        "rm -rf": (0, ""),
        "printf %s": (0, ""),
        "base64 -d": (0, ""),
    })


def make_package(root: Path) -> Path:
    pkg = root / TASK
    (pkg / "public").mkdir(parents=True)
    (pkg / "public" / "README.md").write_text("# ret2win\nConnect and obtain the flag.")
    (pkg / "public" / "challenge.c").write_text("int main(){}")
    (pkg / "Dockerfile").write_text("FROM scratch\n")
    (pkg / "entrypoint.sh").write_text("#!/bin/sh\n")
    (pkg / "task.yaml").write_text(f"task_id: {TASK}\ngrader_mode: flag\ntarget_port: 31337\n")
    (pkg / "private").mkdir()
    (pkg / "private" / "solution.py").write_text(f"print({FLAG!r})")
    (pkg / "MANIFEST.json").write_text(json.dumps({"task_id": TASK}))
    return pkg


def ctf_task(root: Path) -> rlf_modal.RlfCtfTask:
    return rlf_modal.resolve_ctf_task(TASK, root, {TASK: FLAG})


# --- orchestration (breakpoint 3) --------------------------------------------

def test_orchestration_order_and_flag_env_injection(tmp_path):
    make_package(tmp_path)
    fake = healthy_exec()
    orch = rlf_modal.CtfOrchestration(fake, ctf_task(tmp_path), sleep=lambda s: None,
                                      auxiliary_containers=("rlf-ctf-solver", "rlf-ctf-grader"))
    cid = orch.up()
    assert cid == "c0ffee123"
    cmds = fake.commands

    def idx(needle):
        return next(i for i, c in enumerate(cmds) if needle in c)

    assert idx("dockerd --host=unix:///run/docker/docker.sock --iptables=false --bridge=none") \
        < idx(" info ") < idx("build --network host") < idx("run -d") < idx("inspect -f")
    run_cmd = cmds[idx("run -d")]
    import shlex as _sh
    assert f"-e FLAG={_sh.quote(FLAG)}" in run_cmd and "--network host" in run_cmd
    # the flag crosses onto the VM exactly once, on the target run command
    assert sum(FLAG in c for c in cmds) == 1
    # dockerd was reset before launch
    assert "rm -rf /var/lib/docker /run/docker" in cmds[0]


def test_flag_never_in_logs_or_exceptions(tmp_path):
    make_package(tmp_path)
    fake = healthy_exec()
    fake.responses[" build --network host"] = (1, f"build exploded, FLAG={FLAG} leaked by docker")
    orch = rlf_modal.CtfOrchestration(fake, ctf_task(tmp_path), sleep=lambda s: None)
    with pytest.raises(rlf_modal.OrchestrationError) as excinfo:
        orch.up()
    assert FLAG not in str(excinfo.value) and FLAG not in repr(excinfo.value)
    assert "***" in str(excinfo.value)

    fake2 = healthy_exec()
    fake2.responses[" run -d "] = (1, f"docker: {FLAG} boom")
    orch2 = rlf_modal.CtfOrchestration(fake2, ctf_task(tmp_path), sleep=lambda s: None)
    with pytest.raises(rlf_modal.OrchestrationError) as excinfo2:
        orch2.up()
    assert FLAG not in str(excinfo2.value) and FLAG not in repr(excinfo2.value)


def test_stage_package_excludes_private_and_admission_metadata(tmp_path):
    pkg = make_package(tmp_path)
    blob = base64.b64decode(rlf_modal.package_tar_b64(pkg))
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = tar.getnames()
    assert "Dockerfile" in names and "public/challenge.c" in names
    assert not any("private" in n or "MANIFEST" in n or "task.yaml" in n for n in names)
    assert FLAG.encode() not in blob


def test_health_wait_fails_fast_on_exited_container(tmp_path):
    make_package(tmp_path)
    fake = healthy_exec()
    fake.responses[" inspect -f "] = (0, "exited\n")
    orch = rlf_modal.CtfOrchestration(fake, ctf_task(tmp_path), sleep=lambda s: None)
    with pytest.raises(rlf_modal.OrchestrationError, match="exited"):
        orch.up()


def test_teardown_is_idempotent_and_ordered(tmp_path):
    make_package(tmp_path)
    fake = healthy_exec()
    orch = rlf_modal.CtfOrchestration(fake, ctf_task(tmp_path), sleep=lambda s: None,
                                      auxiliary_containers=("rlf-ctf-solver", "rlf-ctf-grader"))
    orch.up()
    orch.down()
    orch.down()  # second call is a no-op
    rm = [c for c in fake.commands if "rm -f" in c and "docker -H" in c]
    assert len(rm) == 3  # solver, grader, target — exactly once each
    assert "rm -f rlf-ctf-solver " in rm[0] and "rm -f rlf-ctf-grader " in rm[1] \
        and "rm -f rlf-ctf-target " in rm[2]


def test_failed_up_cleans_containers(tmp_path):
    make_package(tmp_path)
    fake = healthy_exec()
    fake.responses[" inspect -f "] = (0, "running starting\n")
    orch = rlf_modal.CtfOrchestration(fake, ctf_task(tmp_path), sleep=lambda s: None,
                                      health_attempts=2)
    with pytest.raises(rlf_modal.OrchestrationError, match="healthy"):
        orch.up()
    assert any("rm -f" in c and "rlf-ctf-target" in c for c in fake.commands)


def test_resolve_ctf_task_fails_closed(tmp_path):
    make_package(tmp_path)
    with pytest.raises(rlf_modal.OrchestrationError, match="no trusted flag"):
        rlf_modal.resolve_ctf_task(TASK, tmp_path, {})
    with pytest.raises(rlf_modal.OrchestrationError, match="no task.yaml"):
        rlf_modal.resolve_ctf_task("nope", tmp_path, {"nope": FLAG})
    pkg = make_package(tmp_path / "hacked")
    (pkg / "task.yaml").write_text(f"task_id: {TASK}\ngrader_mode: hacked\n")
    with pytest.raises(rlf_modal.OrchestrationError, match="flag only"):
        rlf_modal.resolve_ctf_task(TASK, tmp_path / "hacked", {TASK: FLAG})


# --- relay + verifier ----------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def test_flag_verifier_grades_submission_and_never_leaks(tmp_path):
    relay = rlf_modal.CtfRelayEnvironment(FakeExec())
    verifier = rlf_modal.RlfFlagVerifier(relay, flag=FLAG, task_id=TASK,
                                         episode_id="ep-1", trajectory_id="traj-1")

    async def go():
        # no submission -> fail closed, error_class set
        empty = await verifier.evaluate("ep-1")
        assert empty["passed"] is False
        assert empty["rlf_grade"]["error_class"] == "missing_submission"
        # wrong flag -> 0 reward
        await relay.submit("ep-1", {"flag": "flag{wrong}"})
        wrong = await verifier.evaluate("ep-1")
        assert wrong["passed"] is False and wrong["rlf_grade"]["reward"] == 0.0
        # right flag -> 1 reward
        relay2 = rlf_modal.CtfRelayEnvironment(FakeExec())
        verifier2 = rlf_modal.RlfFlagVerifier(relay2, flag=FLAG, task_id=TASK,
                                              episode_id="ep-2", trajectory_id="traj-1")
        await relay2.submit("ep-2", {"flag": FLAG})
        right = await verifier2.evaluate("ep-2")
        assert right["passed"] is True and right["rlf_grade"]["reward"] == 1.0
        ev = right["rlf_grade"]["grader_evidence"]
        assert (ev["rlf_task_id"], ev["rlf_episode_id"], ev["rlf_trajectory_id"]) == (TASK, "ep-2", "traj-1")
        # the flag never appears in any payload/log/evidence
        blob = json.dumps(right)
        assert FLAG not in blob
        # verifier is bound to its episode
        with pytest.raises(ValueError, match="bound"):
            await verifier2.evaluate("ep-other")

    _run(go())


def test_relay_binds_one_episode_and_executes_on_vm():
    fake = FakeExec({"uname": (0, "Linux vm\n")})
    relay = rlf_modal.CtfRelayEnvironment(fake)

    async def go():
        out = await relay.execute("ep-1", "uname", timeout_seconds=5, output_bytes=100)
        assert out["exit_code"] == 0 and "Linux vm" in out["output"]
        assert any("uname" in c and "timeout -k 2 5" in c for c in fake.commands)
        with pytest.raises(Exception, match="unknown episode"):
            await relay.execute("ep-2", "uname", timeout_seconds=5, output_bytes=100)
        await relay.submit("ep-1", {"flag": "x"})
        with pytest.raises(Exception, match="terminal"):
            await relay.execute("ep-1", "uname", timeout_seconds=5, output_bytes=100)

    _run(go())


# --- full provider lease contract (fake VM) ------------------------------------

class _FakeProc:
    def __init__(self, result):
        self._result = result
        self.stdout = io.BytesIO(result.output.encode())
        self.stderr = io.BytesIO(b"")

    def wait(self):
        return self._result.exit_code


class _FakeSandbox:
    object_id = "sb-fake"

    def __init__(self, exec_fn):
        self._exec = exec_fn
        self.terminated = 0

    def exec(self, *args, **kwargs):
        return _FakeProc(self._exec(args[2], timeout_s=30))

    def poll(self):
        return 1 if self.terminated else None

    def terminate(self):
        self.terminated += 1


def _fake_vm_provider(monkeypatch, tmp_path):
    make_package(tmp_path)
    fake = healthy_exec()
    sandbox = _FakeSandbox(fake)
    monkeypatch.setattr(rlf_modal, "acquire_vm",
                        lambda task_id, episode_id, *, config=None, image=None, trajectory_id=None:
                        rlf_modal.RlfVmLease(sandbox, task_id=task_id, episode_id=episode_id,
                                             trajectory_id=trajectory_id))
    provider = rlf_modal.RlfModalProvider(
        ctf=rlf_modal.RlfCtfConfig(tasks_dir=str(tmp_path), flags={TASK: FLAG}))
    return provider, fake, sandbox


def test_provider_full_environment_lease_contract(monkeypatch, tmp_path):
    provider, fake, sandbox = _fake_vm_provider(monkeypatch, tmp_path)

    async def go():
        lease = await provider.acquire(TASK, "traj-9")
        # full EnvironmentLease surface
        assert lease.env_url.startswith("http://127.0.0.1:") and lease.env_token
        assert lease.deadline_seconds > 0 and callable(lease.destroy) and callable(lease.describe)
        assert lease.task_id == TASK and lease.episode_id and lease.trajectory_id == "traj-9"
        # target orchestrated before the lease is handed out
        assert any("run -d" in c and FLAG in c for c in fake.commands)
        # the worker wire reaches the VM (host namespace) and grading is trusted
        from yeto.rl.harness.codex.environment import HttpTerminalEnvironment
        async with HttpTerminalEnvironment(lease.env_url, lease.env_token) as env:
            out = await env.execute("ep-wire-1", "id", timeout_seconds=5, output_bytes=100)
            assert out["exit_code"] == 0
            await env.submit("ep-wire-1", {"flag": FLAG})
        evaluation = await lease.verifier.evaluate("ep-wire-1")
        assert evaluation["passed"] is True
        # teardown: containers -> runner -> VM, idempotent
        await lease.destroy()
        await lease.destroy()
        assert await lease.describe() == "gone"

    _run(go())
    assert sandbox.terminated == 1
    assert [c for c in fake.commands if "rm -f" in c and "docker -H" in c]


def test_provider_acquire_failure_cleans_vm(monkeypatch, tmp_path):
    provider, fake, sandbox = _fake_vm_provider(monkeypatch, tmp_path)
    fake.responses[" build --network host"] = (1, "nope")
    with pytest.raises(rlf_modal.OrchestrationError):
        _run(provider.acquire(TASK, "traj-10"))
    assert sandbox.terminated == 1  # failed acquire never leaks a VM
    assert any("rm -f" in c and "rlf-ctf-target" in c for c in fake.commands)


def test_provider_without_ctf_keeps_vm_only_lease(monkeypatch):
    sandbox = _FakeSandbox(FakeExec())
    monkeypatch.setattr(rlf_modal, "acquire_vm",
                        lambda task_id, episode_id, *, config=None, image=None, trajectory_id=None:
                        rlf_modal.RlfVmLease(sandbox, task_id=task_id, episode_id=episode_id,
                                             trajectory_id=trajectory_id))
    lease = _run(rlf_modal.RlfModalProvider().acquire(TASK, "traj-11"))
    assert isinstance(lease, rlf_modal.RlfVmLease) and not hasattr(lease, "env_url")
    lease.teardown()
    lease.teardown()
    assert sandbox.terminated == 1


def test_cloud_factory_env_switch(monkeypatch, tmp_path):
    from yeto.cloud import modal_reward_env as mre

    make_package(tmp_path)
    monkeypatch.delenv(mre.RLF_CTF_TASKS_DIR_ENV, raising=False)
    monkeypatch.delenv(mre.RLF_CTF_FLAGS_ENV, raising=False)
    assert mre.rlf_modal_provider(SimpleNamespace()).ctf is None
    monkeypatch.setenv(mre.RLF_CTF_TASKS_DIR_ENV, str(tmp_path))
    with pytest.raises(RuntimeError, match="FLAGS"):
        mre.rlf_modal_provider(SimpleNamespace())  # fail closed without flags
    monkeypatch.setenv(mre.RLF_CTF_FLAGS_ENV, json.dumps({TASK: FLAG}))
    provider = mre.rlf_modal_provider(SimpleNamespace())
    assert provider.ctf is not None and provider.ctf.flags[TASK] == FLAG


# --- finish_trusted dual path (breakpoint 5) -----------------------------------

class _GradeVerifier:
    def __init__(self, payload):
        self.payload = payload

    async def evaluate(self, episode_id):
        return self.payload


def _rlf_payload(passed=True):
    return {"passed": passed, "log": "flag submission matched",
            "rlf_grade": {"status": "completed", "reward": 1.0 if passed else 0.0,
                          "passed": passed,
                          "grader_evidence": {"grader": "rlf-flag", "rlf_task_id": TASK,
                                              "rlf_episode_id": "ep-rlf-1",
                                              "rlf_trajectory_id": "traj-rlf-1"}}}


def test_finish_trusted_rlf_grade_writes_back_episode():
    untrusted = {"status": "completed", "metrics": {"turns": 3}, "episode_id": "ep-wire"}
    md = _run(adapter.finish_trusted(untrusted, _GradeVerifier(_rlf_payload()),
                                     task_id=TASK, sample_id="s-1"))
    # RLF write-back: reward/success/grader evidence + identity quadruple
    assert md["reward"] == 1.0 and md["success"] is True
    assert md["grader_evidence"]["grader"] == "rlf-flag"
    assert (md["rlf_task_id"], md["rlf_episode_id"], md["rlf_trajectory_id"]) == \
        (TASK, "ep-rlf-1", "traj-rlf-1")
    assert md["rlf_status"] == "completed" and md["exit_status"] == "completed"
    # the outcome envelope still validates (tape consumers keep working)
    outcome = md[tbench_outcome.OUTCOME_KEY]
    assert outcome["reward"] == 1.0 and outcome["episode_id"] == "ep-wire"
    assert tbench_outcome.verified_outcome(md) == (outcome, 1.0)
    assert "tbench_trusted_outcome_hmac" not in md
    assert md["agent_metrics"]["turns"] == 3 and "evaluate_time" in md["agent_metrics"]


def test_finish_trusted_malformed_grade_fails_closed():
    untrusted = {"status": "completed", "metrics": {}, "episode_id": "ep-wire"}
    bad = {"rlf_grade": {"status": "completed"}}  # present but malformed
    with pytest.raises(RolloutRejected, match="malformed"):
        _run(adapter.finish_trusted(untrusted, _GradeVerifier(bad),
                                    task_id=TASK, sample_id="s-1"))
    missing_identity = _rlf_payload()
    del missing_identity["rlf_grade"]["grader_evidence"]["rlf_episode_id"]
    with pytest.raises(RolloutRejected, match="identity"):
        _run(adapter.finish_trusted(untrusted, _GradeVerifier(missing_identity),
                                    task_id=TASK, sample_id="s-1"))
    wrong_task = _rlf_payload()
    wrong_task["rlf_grade"]["grader_evidence"]["rlf_task_id"] = "other-task"
    with pytest.raises(RolloutRejected, match="does not match"):
        _run(adapter.finish_trusted(untrusted, _GradeVerifier(wrong_task),
                                    task_id=TASK, sample_id="s-1"))


def test_finish_trusted_tb2_fallback_byte_unchanged():
    untrusted = {"status": "completed", "metrics": {"turns": 2}, "episode_id": "ep-wire"}
    tb2_payload = {"passed": True, "testsh_rc": 0, "log": "ok"}
    md = _run(adapter.finish_trusted(untrusted, _GradeVerifier(tb2_payload),
                                     task_id="tb2-task", sample_id="s-1"))
    # no RLF keys at all on the fallback path
    assert not any(k.startswith("rlf_") or k == "grader_evidence" for k in md)
    assert "success" not in md
    outcome = md[tbench_outcome.OUTCOME_KEY]
    assert outcome["reward"] == 1.0 and outcome["verifier"] == tbench_outcome.TEST_SH_VERIFIER
    assert tbench_outcome.verified_outcome(md) == (outcome, 1.0)
    assert "tbench_trusted_outcome_hmac" not in md
    # timeout path never calls the verifier and never sees RLF keys
    md_timeout = _run(adapter.finish_trusted({"status": "timeout", "metrics": {}, "episode_id": "ep"},
                                             _GradeVerifier(tb2_payload),
                                             task_id="tb2-task", sample_id="s-1"))
    assert md_timeout[tbench_outcome.OUTCOME_KEY]["verifier"] == tbench_outcome.TIMEOUT_VERIFIER
    assert not any(k.startswith("rlf_") for k in md_timeout)


def test_acquire_vm_idle_timeout_follows_ttl(monkeypatch):
    """Sandbox idle kill must not reclaim a VM mid-generation (S21 gate-5)."""
    import sys
    created: dict[str, Any] = {}

    class _Sb:
        object_id = "sb-x"

        def poll(self):
            return None

        def terminate(self):
            pass

    fake_modal = SimpleNamespace(
        App=SimpleNamespace(lookup=lambda *a, **k: object()),
        Image=SimpleNamespace(from_registry=lambda *a, **k: object()),
        Sandbox=SimpleNamespace(create=lambda *a, **k: (created.update(k) or _Sb())),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    rlf_modal.acquire_vm("t", "ep")
    assert created["timeout"] == 1800
    assert created["idle_timeout"] == 1800  # not the old min(ttl, 300)

    created.clear()
    cfg = rlf_modal.RlfModalVmConfig(ttl_s=600, idle_timeout_s=900)
    rlf_modal.acquire_vm("t", "ep", config=cfg)
    assert created["timeout"] == 600 and created["idle_timeout"] == 900
