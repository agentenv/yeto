"""RLF Qwen3-30B-A3B baseline runner (W5, rlf-codex-rl 5.6), CPU only.

Fake provider / fake drive / fake session-server HTTP: serving argv, budget
env, episode orchestration, trace content, flag hygiene (the flag never lands
in any persisted record), classification (success / failed / timeout /
infrastructure / rejected) and the run summary.  No Modal, no GPU, no SGLang.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from yeto.rl.eval import rlf_baseline as rb
from yeto.rl.codex_backend import stock_codex_backend_profile
from yeto.rl.harness.codex import codex_harness_agent as harness
from yeto.rl.harness.codex.environment import HttpTerminalEnvironment, serve_environment
from yeto.rl.rlf import modal as rlf_modal

FLAG = "flag{rlf_baseline_cpu_test_9f2e}"
TASK = "fixture-ctf-ret2win"


def _run(coro):
    return asyncio.run(coro)


def _cfg(**over) -> rb.BaselineConfig:
    kwargs: dict[str, Any] = {"tasks_dir": "/nonexistent", "flags": {TASK: FLAG},
                              "model_path": "/models/qwen3-30b", "start_timeout_s": 5.0}
    kwargs.update(over)
    return rb.BaselineConfig(**kwargs)


# --- serving shape -------------------------------------------------------------

def test_sglang_argv_is_lora_free_32k_shape(tmp_path):
    local = tmp_path / "model"
    local.mkdir()
    cfg = _cfg(model_path=str(local))
    argv = rb.sglang_argv(cfg, python="python3",
                          serving={"reasoning_parser": "qwen3", "tool_call_parser": "qwen3",
                                   "chat_template_path": "/tpl.jinja"})
    assert argv[:4] == ["python3", "-m", "sglang.launch_server", "--model-path"]
    assert not any(a.startswith(("--enable-lora", "--max-lora", "--lora-")) for a in argv)
    assert argv[argv.index("--context-length") + 1] == "32768"
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.9"
    assert "--revision" not in argv  # local snapshot dir: no hub revision
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    missing = _cfg(model_path="Qwen/Qwen3-30B-A3B")  # not a local path
    argv2 = rb.sglang_argv(missing, serving={})
    assert argv2[argv2.index("--revision") + 1] == rb.QWEN3_30B_A3B_REVISION


def test_session_server_config_is_lora_free():
    cfg = _cfg()
    conf = rb.session_server_config(cfg, serving={"chat_template_path": "/tpl.jinja",
                                                  "chat_template_kwargs": {"enable_thinking": True}})
    # lora_rank is an inert int (pydantic rejects None; no adapter is ever loaded)
    assert isinstance(conf["lora_rank"], int) and conf["lora_adapter_path"] is None
    assert conf["lora_train_only"] is False
    assert conf["backend_url"] == cfg.sglang_url and conf["tito_model"] == "qwen3"
    assert conf["hf_checkpoint"] == cfg.model_path and conf["use_session_server"] is True
    assert conf["apply_chat_template_kwargs"] == {"enable_thinking": True}


def test_config_defaults_come_from_the_profile():
    cfg = rb.BaselineConfig.for_profile(tasks_dir="/t", flags={})
    profile = stock_codex_backend_profile("qwen3_30b_a3b")
    assert (cfg.context_length, cfg.max_tokens, cfg.max_turns) == \
        (profile["max_seq_len"], profile["max_tokens_per_turn"], profile["max_turns"]) == \
        (32768, 8192, 48)
    assert cfg.mem_fraction_static == 0.9 and cfg.episode_seconds == 1800.0
    assert cfg.start_timeout_s == 900.0  # 15 min serving budget
    assert cfg.model_path.endswith("/hf/Qwen3-30B-A3B/ad44e777")
    assert cfg.policy_id == f"{profile['model_identifier']}@{profile['model_revision']}"
    env = cfg.worker_budget_env()
    assert env == {"YETO_CODEX_CHAT_TEMPLATE": "qwen3_30b_a3b",
                   "SECRLENV_MAX_TURNS": "48", "YETO_CODEX_BACKEND_MAX_TOKENS": "8192"}


class _FakeProc:
    def __init__(self):
        self.killed = 0

    def poll(self):
        return None

    def terminate(self):
        self.killed += 1

    def wait(self, timeout=None):
        return 0


def test_serving_stack_ready_and_start_timeout():
    cfg = _cfg()
    spawned = []
    stack = rb.ServingStack(cfg, popen=lambda argv, **kw: spawned.append(argv) or _FakeProc(),
                            http=lambda *a, **kw: (200, {}), sleep=lambda s: None,
                            clock=iter([0, 1, 2, 3]).__next__, serving=lambda t: {})
    info = stack.start()
    assert info["sglang_start_s"] == 1 and "--context-length" in spawned[0]
    assert "session.server" in spawned[1][2]
    stack.close()

    # never ready within start_timeout_s -> InfrastructureError (the run aborts)
    clock_state = {"t": 0.0}

    def clock():
        clock_state["t"] += 400.0
        return clock_state["t"]

    stack2 = rb.ServingStack(_cfg(), popen=lambda argv, **kw: _FakeProc(),
                             http=lambda *a, **kw: (None, None), sleep=lambda s: None,
                             clock=clock, serving=lambda t: {})
    with pytest.raises(rb.InfrastructureError, match="infrastructure"):
        stack2.start()

    # process exits before ready -> InfrastructureError too
    class _Dead(_FakeProc):
        returncode = 3

        def poll(self):
            return 3

    stack3 = rb.ServingStack(_cfg(), popen=lambda argv, **kw: _Dead(),
                             http=lambda *a, **kw: (None, None), sleep=lambda s: None,
                             serving=lambda t: {})
    with pytest.raises(rb.InfrastructureError, match="exited"):
        stack3.start()


# --- fakes: provider / session server / drive ------------------------------------

class _FakeExec:
    def __init__(self, output=""):
        self.output = output
        self.commands: list[str] = []

    def __call__(self, command: str, *, timeout_s: float) -> rlf_modal.VmExecResult:
        self.commands.append(command)
        return rlf_modal.VmExecResult(0, self.output)


class _FakeLease:
    def __init__(self, relay, runner, url, token, *, flag=FLAG, trajectory_id=""):
        self.env_url = url
        self.env_token = token
        self.environment = relay
        self.verifier = rlf_modal.RlfFlagVerifier(relay, flag=flag, task_id=TASK,
                                                  episode_id="ep-lease-1",
                                                  trajectory_id=trajectory_id or "traj-lease-1")
        self.episode_id = "ep-lease-1"
        self.trajectory_id = trajectory_id or "traj-lease-1"
        self.deadline_seconds = 1800.0
        self.worker_env = None
        self._runner = runner
        self.destroyed = 0

    async def destroy(self):
        self.destroyed += 1
        await self._runner.cleanup()

    async def describe(self):
        return "gone"


class FakeProvider:
    """One freshly-served relay environment per acquire (the real wire, no VM)."""

    def __init__(self, *, exec_output="", flag=FLAG):
        self.exec_output = exec_output
        self.flag = flag
        self.leases: list[_FakeLease] = []
        self.acquired: list[tuple[str, str]] = []

    async def acquire(self, task_id: str, trajectory_id: str = "") -> _FakeLease:
        self.acquired.append((task_id, trajectory_id))
        relay = rlf_modal.CtfRelayEnvironment(_FakeExec(self.exec_output))
        runner, url = await serve_environment(relay, "tok-" + trajectory_id[:8], host="127.0.0.1")
        lease = _FakeLease(relay, runner, url, "tok-" + trajectory_id[:8],
                           flag=self.flag, trajectory_id=trajectory_id)
        self.leases.append(lease)
        return lease


class FakeSessions:
    """The session server's /sessions routes; the drive never reaches chat completions."""

    def __init__(self):
        self.created = 0
        self.deleted: list[str] = []

    def __call__(self, method: str, url: str, body: Any = None, *, timeout: float = 60.0):
        if method == "POST" and url.endswith("/sessions"):
            self.created += 1
            return 200, {"session_id": f"s-{self.created}"}
        if method == "DELETE" and "/sessions/" in url:
            self.deleted.append(url.rsplit("/", 1)[-1])
            return 200, {}
        raise AssertionError(f"unexpected {method} {url}")


GOOD_METRICS = {"turns": 3, "tool_calls": 3, "submit_calls": 1, "parse_failures": 0,
                "max_model_total_tokens": 12345, "max_seq_len_hit": 0,
                "turn_completion_tokens": [100, 200, 300],
                "turn_context_tokens": [1000, 2000, 3000],
                "turn_tool_output_bytes": [50, 60, 70],
                "end_reason": "submitted", "end_kind": "submit"}


def make_drive(*, submit=FLAG, commands=("cat /challenge/flag.txt",), metrics=None,
               recorded=None):
    """The untrusted worker: talks to the lease over the real HTTP wire, then reports."""

    async def drive(job, trajectory_id, board, worker_env):
        if recorded is not None:
            recorded.append({"job": dict(job), "worker_env": dict(worker_env or {})})
        async with HttpTerminalEnvironment(job["env_url"], job["env_token"]) as env:
            for command in commands:
                await env.execute(job["episode_id"], command, timeout_seconds=5,
                                  output_bytes=8192)
            if submit is not None:
                await env.submit(job["episode_id"], {"flag": submit})
        return {"status": "completed", "episode_id": job["episode_id"],
                "metrics": dict(GOOD_METRICS if metrics is None else metrics)}

    return drive


ROW = {"metadata": {"task_id": TASK, "rlf_task_id": TASK, "grader_mode": "flag",
                    "prompt": "Connect to the target and obtain the flag."},
       "prompt": "Connect to the target and obtain the flag."}


# --- episode orchestration --------------------------------------------------------

def test_episode_success_trace_and_flag_hygiene():
    provider = FakeProvider(exec_output=f"... {FLAG} ...")  # the flag appears in tool output
    sessions = FakeSessions()
    recorded: list[dict] = []
    trace = _run(rb.run_episode(_cfg(), provider, ROW, session_url="http://sess",
                                drive=make_drive(recorded=recorded), http=sessions,))
    assert trace["classification"] == "success" and trace["success"] is True
    assert trace["reward"] == 1.0 and trace["end_reason"] == "completed"
    # RLF identity quadruple + baseline policy identity
    assert trace["rlf_task_id"] == TASK and trace["rlf_episode_id"] == "ep-lease-1"
    assert trace["rlf_trajectory_id"] and trace["policy_id"].startswith("Qwen/Qwen3-30B-A3B@")
    assert trace["grader_evidence"]["grader"] == "rlf-flag"
    # per-turn token rows and per-tool trace events
    assert [t["completion_tokens"] for t in trace["turns"]] == [100, 200, 300]
    assert trace["turn_count"] == 3 and trace["tool_calls"] == 3 and trace["submit_calls"] == 1
    assert len(trace["tool_events"]) == 1
    event = trace["tool_events"][0]
    assert event["exit_code"] == 0 and event["flag_seen"] is True
    assert event["output_preview"] == "... *** ..."  # scrubbed, never the raw flag
    # the submission turn marker: 1 tool call happened before the submit
    assert trace["submissions"] == [{"after_tool_calls": 1, "has_flag": True, "flag_correct": True}]
    assert trace["max_model_total_tokens"] == 12345 and trace["max_seq_len_hit"] == 0
    assert trace["compactions"] == 0
    # the flag is nowhere in anything that would be persisted
    assert FLAG not in json.dumps(trace)
    # the job and the worker budget env reached the drive
    job, worker_env = recorded[0]["job"], recorded[0]["worker_env"]
    assert job["episode_id"] == "ep-lease-1" and job["max_seq_len"] == 32768
    assert job["request_kwargs"]["max_tokens"] == 8192 and job["driver"] == "stock"
    assert job["base_url"] == "http://sess/sessions/s-1"
    assert worker_env["SECRLENV_MAX_TURNS"] == "48"
    assert worker_env["YETO_CODEX_BACKEND_MAX_TOKENS"] == "8192"
    assert worker_env["YETO_CODEX_CHAT_TEMPLATE"] == "qwen3_30b_a3b"
    # teardown: session deleted, lease destroyed exactly once
    assert sessions.deleted == ["s-1"] and provider.leases[0].destroyed == 1


def test_episode_wrong_flag_is_failed_with_zero_reward():
    provider = FakeProvider()
    trace = _run(rb.run_episode(_cfg(), provider, ROW, session_url="http://sess",
                                drive=make_drive(submit="flag{nope}"), http=FakeSessions(),))
    assert trace["classification"] == "failed" and trace["success"] is False
    assert trace["reward"] == 0.0
    assert trace["submissions"][0]["flag_correct"] is False
    assert "flag{nope}" not in json.dumps(trace)  # submitted values are never recorded


def test_episode_hard_deadline_times_out_and_skips_grading():
    provider = FakeProvider()

    async def slow_drive(job, trajectory_id, board, worker_env):
        await asyncio.sleep(30)

    cfg = _cfg(episode_seconds=0.2)
    trace = _run(rb.run_episode(cfg, provider, ROW, session_url="http://sess",
                                drive=slow_drive, http=FakeSessions()))
    assert trace["status"] == "timeout" and trace["end_reason"] == "timed_out"
    assert trace["classification"] == "failed" and trace["reward"] == 0.0
    assert provider.leases[0].verifier.evaluations == 0  # a timeout never reaches the grader
    assert provider.leases[0].destroyed == 1


def test_episode_lease_deadline_tightens_the_hard_limit():
    provider = FakeProvider()
    seen = {}

    async def slow_drive(job, trajectory_id, board, worker_env):
        await asyncio.sleep(30)

    cfg = _cfg(episode_seconds=1800.0)
    orig_acquire = provider.acquire

    async def acquire(task_id, trajectory_id=""):
        lease = await orig_acquire(task_id, trajectory_id)
        lease.deadline_seconds = 0.2  # the lease's own deadline wins over 30 min
        return lease

    provider.acquire = acquire
    trace = _run(rb.run_episode(cfg, provider, ROW, session_url="http://sess",
                                drive=slow_drive, http=FakeSessions()))
    assert trace["status"] == "timeout"


def test_episode_harness_error_is_infrastructure():
    provider = FakeProvider()

    async def broken_drive(job, trajectory_id, board, worker_env):
        raise harness.CodexHarnessError("worker exited without a result")

    trace = _run(rb.run_episode(_cfg(), provider, ROW, session_url="http://sess",
                                drive=broken_drive, http=FakeSessions()))
    assert trace["classification"] == "infrastructure" and trace["reward"] is None
    assert "worker exited" in trace["error"]
    assert provider.leases[0].destroyed == 1


def test_episode_acquire_failure_is_infrastructure_and_never_graded():
    class DenyProvider:
        async def acquire(self, task_id, trajectory_id=""):
            raise rlf_modal.OrchestrationError(f"no trusted flag configured, FLAG={FLAG}")

    trace = _run(rb.run_episode(_cfg(), DenyProvider(), ROW, session_url="http://sess",
                                drive=make_drive(), http=FakeSessions()))
    assert trace["classification"] == "infrastructure"
    assert trace.get("rlf_episode_id") is None  # no lease, no identity
    assert "acquire" in trace["error"] and FLAG not in json.dumps(trace)  # scrubbed


def test_episode_malformed_grade_is_rejected_not_failed():
    provider = FakeProvider()

    async def drive(job, trajectory_id, board, worker_env):
        return {"status": "completed", "episode_id": job["episode_id"], "metrics": {}}

    orig_acquire = provider.acquire

    async def acquire(task_id, trajectory_id=""):
        lease = await orig_acquire(task_id, trajectory_id)
        lease.verifier = SimpleNamespace(evaluate=lambda episode_id: _bad_grade())
        return lease

    async def _bad_grade():
        return {"rlf_grade": {"status": "completed"}}  # malformed: no reward/identity

    provider.acquire = acquire
    trace = _run(rb.run_episode(_cfg(), provider, ROW, session_url="http://sess",
                                drive=drive, http=FakeSessions()))
    assert trace["classification"] == "rejected" and trace["reward"] is None
    assert "grade rejected" in trace["error"]


# --- run loop + summary -----------------------------------------------------------

def _catalog_package(root: Path) -> None:
    pkg = root / TASK
    (pkg / "public").mkdir(parents=True)
    (pkg / "public" / "README.md").write_text("# ret2win\nConnect and obtain the flag.")
    (pkg / "Dockerfile").write_text("FROM scratch\n")
    (pkg / "task.yaml").write_text(f"task_id: {TASK}\ngrader_mode: flag\ntarget_port: 31337\n")
    (pkg / "MANIFEST.json").write_text(json.dumps({"task_id": TASK, "grader_mode": "flag"}))


def test_run_from_config_end_to_end(tmp_path):
    _catalog_package(tmp_path)
    provider = FakeProvider()
    config = {"tasks_dir": str(tmp_path), "flags": {TASK: FLAG},
              "evidence": {"runtime_quality_gate": {"healthy": True}},
              "out_dir": str(tmp_path / "out"), "serve": "external",
              "session_url": "http://sess", "baseline": {"repeat": 2, "model_path": "/m"}}
    result = rb.run_from_config(config, provider=provider, drive=make_drive(),
                                http=FakeSessions(), progress=None)
    assert result["ok"] is True
    summary = result["summary"]
    assert summary["episodes"] == 2 and summary["evaluated"] == 2
    assert summary["success_count"] == 2 and summary["success_rate"] == 1.0
    assert summary["reward_distribution"] == {"mean": 1.0, "counts": {"1.0": 2}}
    assert summary["end_reasons"] == {"completed": 2}
    assert summary["max_seq_len_hit"] == {"count": 0, "rate": 0.0}
    assert summary["per_task"][TASK]["successes"] == 2
    assert summary["policy_id"].startswith("Qwen/Qwen3-30B-A3B@")
    traces = [json.loads(line) for line in
              Path(result["paths"]["traces"]).read_text().splitlines()]
    assert len(traces) == 2
    assert {t["rlf_trajectory_id"] for t in traces} and \
        len({t["rlf_episode_id"] for t in traces}) == 1  # fake provider reuses its lease id
    assert len({t["rlf_trajectory_id"] for t in traces}) == 2  # but trajectories are unique
    on_disk = Path(result["paths"]["traces"]).read_text() + \
        Path(result["paths"]["summary"]).read_text()
    assert FLAG not in on_disk


def test_run_from_config_serving_timeout_aborts_as_infrastructure(tmp_path, monkeypatch):
    _catalog_package(tmp_path)

    class NeverReady:
        def __init__(self, cfg):
            self.cfg = cfg

        def start(self):
            raise rb.InfrastructureError("sglang not ready after 900.0 s: infrastructure")

        def close(self):
            pass

    monkeypatch.setattr(rb, "ServingStack", NeverReady)
    config = {"tasks_dir": str(tmp_path), "flags": {TASK: FLAG},
              "evidence": {"runtime_quality_gate": {"healthy": True}},
              "out_dir": str(tmp_path / "out"), "serve": "start",
              "baseline": {"model_path": "/m"}}
    result = rb.run_from_config(config, provider=FakeProvider(), drive=make_drive(),
                                progress=None)
    assert result["ok"] is False
    aborted = result["summary"]["aborted"]
    assert aborted["classification"] == "infrastructure" and aborted["stage"] == "serving"
    assert result["summary"]["episodes"] == 0
    assert Path(result["paths"]["summary"]).is_file()


def test_summarize_mixed_classifications():
    traces = [
        {"task_id": "a", "classification": "success", "success": True, "reward": 1.0,
         "end_reason": "completed", "max_seq_len_hit": 0, "turn_count": 10, "tool_calls": 9,
         "max_model_total_tokens": 20000, "compactions": 0},
        {"task_id": "a", "classification": "failed", "success": False, "reward": 0.0,
         "end_reason": "max_seq_len", "max_seq_len_hit": 1, "turn_count": 48, "tool_calls": 47,
         "max_model_total_tokens": 32768, "compactions": 2},
        {"task_id": "b", "classification": "infrastructure", "success": False, "reward": None},
        {"task_id": "b", "classification": "rejected", "success": False, "reward": None},
    ]
    summary = rb.summarize(traces, policy_id="p@1")
    assert summary["episodes"] == 4 and summary["evaluated"] == 2
    assert summary["infrastructure"] == 1 and summary["rejected"] == 1
    assert summary["success_rate"] == 0.5 and summary["success_count"] == 1
    assert summary["reward_distribution"]["counts"] == {"0.0": 1, "1.0": 1}
    assert summary["end_reasons"] == {"completed": 1, "max_seq_len": 1}
    assert summary["max_seq_len_hit"] == {"count": 1, "rate": 0.5}
    assert summary["compactions_total"] == 2
    assert summary["turns"] == {"mean": 29.0, "max": 48.0}
    assert summary["per_task"]["a"]["success_rate"] == 0.5
    assert summary["per_task"]["b"]["infrastructure"] == 1
    # an all-infrastructure run has a null success rate, never a fake 0
    empty = rb.summarize([traces[2]])
    assert empty["success_rate"] is None and empty["reward_distribution"]["mean"] is None


def test_trace_recorder_bounds_previews_and_scrubs():
    relay = rlf_modal.CtfRelayEnvironment(_FakeExec(output=(FLAG + "x" * 100)))

    async def go():
        recorder = rb.TraceRecorder(relay, secrets=(FLAG,), preview_bytes=20).attach()
        out = await relay.execute("ep-1", f"echo {FLAG}", timeout_seconds=5, output_bytes=8192)
        await relay.submit("ep-1", {"flag": FLAG})
        return recorder, out

    recorder, _ = _run(go())
    event = recorder.tool_events[0]
    assert event["flag_seen"] is True and len(event["output_preview"]) <= 20
    assert event["command"] == "echo ***"
    assert recorder.submissions[0]["flag_correct"] is True
    assert FLAG not in json.dumps({"tool_events": recorder.tool_events,
                                   "submissions": recorder.submissions})
