"""CompactionRL switch wiring (progress.md "S13 Codex 压缩接线"; CPU only, no Ray).

- spec <-> switch: launcher env injection, preflight consistency, control arm;
- pre-created segment sessions are deleted when they cannot reach the
  generate wrapper (half-way creation, worker crash, finish_trusted error);
- R-D5a chain-count judge with/without compaction;
- module list / summary tool-table constants.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from aiohttp import web

from yeto import launcher as L
from yeto.rl import CODEX_HARNESS_AGENT, CODEX_OPENENV_AGENT_MODULES
from yeto.rl.algos import compactionrl as crl
from yeto.rl.harness.codex import chain_judge
from yeto.rl.harness.codex import codex_openenv_agent_function as adapter
from yeto.rl.harness.codex import codex_openenv_subprocess_agent_function as subprocess_agent
from yeto.rl.harness.codex import compaction_bridge as cb
from yeto.rl.harness.codex import preflight
from yeto.rl.harness.codex.environment import FakeTerminalEnvironment

from test_harness_codex_openenv import KEY, _Provider, _metadata
from test_rl_launcher_codex_bundle import PROVIDER, _codex_args, bundle  # noqa: F401 - fixture

SWITCH, TCOMP = crl.COMPACTION_SWITCH_ENV, crl.COMPACTION_T_COMP_ENV


def _run(coro):
    return asyncio.run(coro)


def _env(root, **extra):
    return {L.CODEX_BUNDLE_DIR_ENV: str(root), L.HARNESS_ENVIRONMENT_PROVIDER_ENV: PROVIDER, **extra}


def _with_spec(args, spec):
    args.rl_algorithm_spec_json = spec.canonical_json()
    return args


# --- constants shared by bridge / spec -------------------------------------

def test_switch_names_match_the_bridge_and_module_list_includes_it():
    assert crl.COMPACTION_SWITCH_ENV == cb.COMPACTIONRL_ENV and crl.COMPACTION_T_COMP_ENV == cb.T_COMP_ENV
    assert crl.compactionrl_spec().canonical_json() and crl.COMPACTED_GAE_VARIANT in crl.compactionrl_spec().canonical_json()
    assert "compaction_bridge.py" in CODEX_OPENENV_AGENT_MODULES
    assert preflight.required_pin_updates()["CODEX_OPENENV_AGENT_MODULES"] == CODEX_OPENENV_AGENT_MODULES
    assert cb.SUMMARY_REQUEST_TOOLS == () and cb.SUMMARY_REQUEST_TOOL_CHOICE == "none"


# --- launcher ---------------------------------------------------------------

def test_launcher_compactionrl_spec_switches_rollout_compaction_on(bundle):  # noqa: F811
    root, _ = bundle
    args = _with_spec(_codex_args(), crl.compactionrl_spec())
    # upstream agentic_tool_call.generate never collects the segment sessions
    # (s19-compaction-g1-20261010b: no tokens_after, trainer failed at round 0)
    with pytest.raises(ValueError, match="codex_openenv_generate"):
        L.codex_harness_launch(args, environ=_env(root))
    from yeto.rl.adapters.miles import config as miles_config

    args.custom_generate_function_path = miles_config.CODEX_OPENENV_GENERATE
    check = miles_config._requires_agentic_generate("--custom-agent-function-path")
    assert check("x.y", SimpleNamespace(agent=args)) is None  # the wrapper reads the agent flags
    assert check("x.y", SimpleNamespace(agent=SimpleNamespace(custom_generate_function_path="a.b")))
    _flags, envs, _ = L.codex_harness_launch(args, environ=_env(root))
    assert envs[SWITCH] == "1" and TCOMP not in envs
    # The island preflight forwards YETO_CODEX_* to the Ray rollout workers.
    assert preflight.worker_runtime_env(SimpleNamespace(), envs)[SWITCH] == "1"
    _flags, envs, _ = L.codex_harness_launch(args, environ=_env(root, **{TCOMP: "4096", SWITCH: "1"}))
    assert envs[SWITCH] == "1" and envs[TCOMP] == "4096"
    with pytest.raises(ValueError, match="contradicts"):
        L.codex_harness_launch(args, environ=_env(root, **{SWITCH: "0"}))
    with pytest.raises(ValueError, match="positive integer"):
        L.codex_harness_launch(args, environ=_env(root, **{TCOMP: "-3"}))


def test_launcher_default_specs_unchanged_and_switch_rejected(bundle):  # noqa: F811
    root, _ = bundle
    args = _codex_args()  # default spec
    _f, base, _ = L.codex_harness_launch(args, environ=_env(root))
    assert SWITCH not in base and TCOMP not in base
    _f, explicit_off, _ = L.codex_harness_launch(args, environ=_env(root, **{SWITCH: "0"}))
    assert explicit_off == base
    with pytest.raises(ValueError, match="only valid with cross_segment_per_sample"):
        L.codex_harness_launch(args, environ=_env(root, **{SWITCH: "1"}))
    with pytest.raises(ValueError, match="without"):
        L.codex_harness_launch(args, environ=_env(root, **{TCOMP: "4096"}))
    with pytest.raises(ValueError, match="boolean"):
        L.codex_harness_launch(args, environ=_env(root, **{SWITCH: "maybe"}))


def test_launcher_control_arm_rejected_on_codex(bundle):  # noqa: F811
    root, _ = bundle
    args = _with_spec(_codex_args(), crl.compactionrl_whole_rollout_control_spec())
    for extra in ({}, {SWITCH: "1"}):
        with pytest.raises(ValueError, match="cannot be concatenated"):
            L.codex_harness_launch(args, environ=_env(root, **extra))


def test_launcher_compactionrl_needs_the_compacting_harness(bundle):  # noqa: F811
    root, _ = bundle
    args = _with_spec(_codex_args(), crl.compactionrl_spec())
    args.custom_agent_function_path = CODEX_HARNESS_AGENT  # signed, but cannot compact
    with pytest.raises(ValueError, match="only the Codex OpenEnv harness"):
        L.codex_harness_launch(args, environ=_env(root))
    args.custom_agent_function_path = None  # not a Codex run at all
    with pytest.raises(ValueError, match="only the Codex OpenEnv harness"):
        L.codex_harness_launch(args, environ=_env(root))


# --- island preflight -------------------------------------------------------

@pytest.mark.parametrize(
    "variant,env,error",
    [
        ("cross_segment_per_sample", {SWITCH: "1"}, None),
        ("cross_segment_per_sample", {SWITCH: "1", TCOMP: "10240"}, None),
        (None, {}, None),
        ("vanilla", {SWITCH: "0"}, None),
        ("cross_segment_per_sample", {}, "requires YETO_CODEX_COMPACTIONRL=1"),
        (None, {SWITCH: "1"}, "only valid with"),
        ("vanilla", {TCOMP: "100"}, "without"),
        ("cross_segment_whole_rollout", {SWITCH: "1"}, "cannot be concatenated"),
        ("cross_segment_whole_rollout", {}, "cannot be concatenated"),
        ("cross_segment_per_sample", {SWITCH: "1", TCOMP: "x"}, "positive integer"),
    ],
)
def test_preflight_switch_matches_gae_variant(variant, env, error):
    args = SimpleNamespace(gae_variant=variant)
    if error is None:
        preflight.assert_compactionrl_consistent(args, env)
    else:
        with pytest.raises(preflight.PreflightError, match=error):
            preflight.assert_compactionrl_consistent(args, env)


# --- segment session cleanup ------------------------------------------------

class _SessionServer:
    """Fake router: POST /sessions + DELETE /sessions/{id} (pin: 204)."""

    def __init__(self, fail_after: int | None = None):
        self.fail_after = fail_after
        self.live: set[str] = set()
        self.deleted: list[str] = []
        self.n = 0

    async def __aenter__(self):
        async def create(_request):
            self.n += 1
            if self.fail_after is not None and self.n > self.fail_after:
                return web.Response(status=500)
            sid = f"seg{self.n}"
            self.live.add(sid)
            return web.json_response({"session_id": sid})

        async def delete(request):
            sid = request.match_info["sid"]
            if sid not in self.live:
                return web.Response(status=404)
            self.live.discard(sid)
            self.deleted.append(sid)
            return web.Response(status=204)

        app = web.Application()
        app.router.add_post("/sessions", create)
        app.router.add_delete("/sessions/{sid}", delete)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()


def _job(url):
    return {"base_url": f"{url}/sessions/S0", "prompt": "p", "request_kwargs": {}, "episode_id": "e",
            "max_seq_len": 8000}


def test_half_way_precreation_deletes_what_was_created(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv(TCOMP, "4000")

    async def scenario():
        async with _SessionServer(fail_after=2) as server:
            with pytest.raises(RuntimeError, match="HTTP 500"):
                await adapter.prepare_segment_sessions(_job(server.url))
            return server

    server = _run(scenario())
    assert server.deleted == ["seg1", "seg2"] and server.live == set()


def test_delete_segment_sessions_is_best_effort():
    calls = []

    async def delete(url):
        calls.append(url)
        if url.endswith("/b"):
            raise OSError("down")

    failed = _run(adapter.delete_segment_sessions("http://r:1/sessions/S0", ["a", "b", "c"], delete=delete))
    assert failed == ["b"] and calls == [f"http://r:1/sessions/{s}" for s in "abc"]
    assert _run(adapter.delete_segment_sessions("http://r:1/sessions/S0", None, delete=delete)) == []


def _in_process(monkeypatch, server, *, drive=None, finish=None):
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv(TCOMP, "4000")
    if drive is not None:
        monkeypatch.setattr(adapter, "drive_untrusted", drive)
    if finish is not None:
        monkeypatch.setattr(adapter, "finish_trusted", finish)
    env = FakeTerminalEnvironment(passed=True)
    return adapter.run(f"{server.url}/sessions/S0", "p", {}, {"task_id": "fix-git", "trajectory_id": "t",
                                                               "max_seq_len": 8000},
                       environment=env, verifier=env)


@pytest.mark.parametrize("where", ["drive", "finish"])
def test_in_process_run_deletes_sessions_on_unexpected_errors(monkeypatch, where):
    async def crash(*_a, **_k):
        raise ValueError("worker crashed")

    async def ok_drive(*_a, **_k):
        return {"status": "completed", "episode_id": "e", "metrics": {}}

    async def scenario():
        async with _SessionServer() as server:
            with pytest.raises(ValueError, match="worker crashed"):
                await _in_process(monkeypatch, server, drive=crash if where == "drive" else ok_drive,
                                  finish=crash if where == "finish" else None)
            return server

    server = _run(scenario())
    assert sorted(server.deleted) == ["seg1", "seg2", "seg3"] and server.live == set()


def test_in_process_run_hands_sessions_off_on_success(monkeypatch):
    async def ok_drive(*_a, **_k):
        return {"status": "completed", "episode_id": "e", "metrics": {}}

    async def finish(*_a, **_k):
        return {"signed": 1}

    async def scenario():
        async with _SessionServer() as server:
            result = await _in_process(monkeypatch, server, drive=ok_drive, finish=finish)
            return server, result

    server, result = _run(scenario())
    assert result[cb.SESSIONS_METADATA_KEY] == ["seg1", "seg2", "seg3"]
    assert server.deleted == [] and server.live == {"seg1", "seg2", "seg3"}  # the wrapper collects them


def _subprocess(monkeypatch, *, drive, finish=None):
    monkeypatch.setenv("TBENCH_REWARD_HMAC_KEY", KEY)
    monkeypatch.setenv(SWITCH, "1")
    monkeypatch.setenv(TCOMP, "4000")
    provider = _Provider(FakeTerminalEnvironment(passed=True))
    subprocess_agent.configure(provider=provider, tool_wait_board=None, harness_board=None, member="m0")
    monkeypatch.setattr(subprocess_agent, "_drive_worker", drive)
    if finish is not None:
        monkeypatch.setattr(adapter, "finish_trusted", finish)
    return provider


@pytest.mark.parametrize("where", ["worker", "finish", "harness_error"])
def test_subprocess_run_cleans_up_or_hands_off_segment_sessions(monkeypatch, where):
    async def crash(*_a, **_k):
        raise RuntimeError("worker crashed")

    async def harness_error(*_a, **_k):
        raise adapter.harness.CodexHarnessError("worker exited without a result")

    async def ok_drive(*_a, **_k):
        return {"status": "completed", "episode_id": "e", "metrics": {}}

    drive = {"worker": crash, "finish": ok_drive, "harness_error": harness_error}[where]
    provider = _subprocess(monkeypatch, drive=drive, finish=crash if where == "finish" else None)

    async def scenario():
        async with _SessionServer() as server:
            md = _metadata(max_seq_len=8000)
            if where == "harness_error":
                result = await subprocess_agent.run(f"{server.url}/sessions/S0", "p", {}, md)
            else:
                result = None
                with pytest.raises(RuntimeError, match="worker crashed"):
                    await subprocess_agent.run(f"{server.url}/sessions/S0", "p", {}, md)
            return server, result

    try:
        server, result = _run(scenario())
    finally:
        subprocess_agent.configure(provider=None)
    assert provider.destroyed == 1
    if where == "harness_error":  # metadata returned -> the generate wrapper drains them
        assert result[cb.SESSIONS_METADATA_KEY] == ["seg1", "seg2", "seg3"] and server.deleted == []
    else:
        assert sorted(server.deleted) == ["seg1", "seg2", "seg3"] and server.live == set()


# --- R-D5a chain-count judge ------------------------------------------------

def _seg(i, n, traj="t", **kw):
    return {"trajectory_id": traj, "chain_index": i, "segment_index": i, "chains_total": n,
            "num_segments": n, "compactions": n - 1,
            "chain_break_reason": "compaction_window" if i else None, **kw}


def test_chain_judge_off_keeps_chains_total_one():
    ok = [{"trajectory_id": "a", "chains_total": 1}, {"trajectory_id": "b", "chains_total": 1}]
    assert chain_judge.judge_chain_counts(ok, compaction_enabled=False) == []
    assert chain_judge.judge_chain_counts([_seg(0, 2), _seg(1, 2)], compaction_enabled=False)
    assert chain_judge.judge_chain_counts([{"trajectory_id": "a", "chains_total": 2}], compaction_enabled=False)


def test_chain_judge_on_counts_segments():
    good = [_seg(0, 3), _seg(1, 3), _seg(2, 3), _seg(0, 1, traj="u"),
            {"trajectory_id": "aborted", "chains_total": 1}]
    assert chain_judge.judge_chain_counts(good, compaction_enabled=True) == []
    assert chain_judge.judge_chain_counts([_seg(0, 3), _seg(1, 3)], compaction_enabled=True)  # missing segment
    bad_comp = [_seg(0, 2), {**_seg(1, 2), "compactions": 2}]
    assert any("compactions" in p for p in chain_judge.judge_chain_counts(bad_comp, compaction_enabled=True))
    bad_reason = [_seg(0, 2), {**_seg(1, 2), "chain_break_reason": "retry_fork"}]
    assert chain_judge.judge_chain_counts(bad_reason, compaction_enabled=True)
    bad_index = [_seg(0, 2), {**_seg(1, 2), "chain_index": 0}]
    assert chain_judge.judge_chain_counts(bad_index, compaction_enabled=True)
