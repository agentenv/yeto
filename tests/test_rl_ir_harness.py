"""codex-harness interface requests IR-1..IR-4 (INFRA-E1; design R-IR).

CPU only: config translation, harness preflight hook, drain blockers with
harness counts + admission, driver -> pool policy token, 1.7 schema.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine import run_config as rc
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import EventTape, IslandDriver, PolicyIdentityError, policy_token
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.miles_adapter import config as cfg
from yeto.rl.engine.miles_adapter import entry
from yeto.rl.engine.miles_adapter.rollout import HARNESS_NOT_AGENTIC, MilesRolloutPool
from yeto.rl.engine.timeline import (
    HARNESS_METRIC_KEYS,
    LOAD_SAMPLE_LABELS,
    LOAD_SAMPLE_SCHEMA,
    LoadSample,
    classify_load,
    validate_load_sample,
)
from yeto.rl.engine.tool_wait import (
    HARNESS_ZERO,
    HarnessAdmissionError,
    HarnessBoard,
    HarnessSnapshot,
    ToolWaitBoard,
    drain_blockers,
    read_harness,
)

from test_rl_miles_adapter_config import make_config, sub

AGENTIC = cfg.AGENTIC_TOOL_CALL_GENERATE
NAME = "base_model.model.layer.lora_A.weight"


# ---------------------------------------------------------------------------
# IR-1 config
# ---------------------------------------------------------------------------
def test_ir1_agent_flags_pass_through_with_agentic_generate():
    c = sub(make_config(), "agent", custom_generate_function_path=AGENTIC,
            custom_agent_function_path="yeto.rl.harness.codex.codex_openenv_agent_function",
            agent_max_seq_len=8192)
    argv = cfg.translate_run_config(c, AlgorithmSpec()).argv
    i = argv.index("--custom-agent-function-path")
    assert argv[i + 1] == "yeto.rl.harness.codex.codex_openenv_agent_function"
    assert argv[argv.index("--max-seq-len") + 1] == "8192"
    assert argv[argv.index("--custom-generate-function-path") + 1] == AGENTIC


@pytest.mark.parametrize("field, value", [
    ("custom_agent_function_path", "a.b.c"),
    ("agent_max_seq_len", 4096),
])
@pytest.mark.parametrize("generate", [None, "yeto.rl.tool_wait_workload.generate"])
def test_ir1_agent_flags_need_agentic_generate(field, value, generate):
    c = sub(make_config(), "agent", custom_generate_function_path=generate, **{field: value})
    with pytest.raises(cfg.UnmappedConfigError) as err:
        cfg.translate_run_config(c, AlgorithmSpec())
    assert f"agent.{field}" in str(err.value)
    assert f"requires custom_generate_function_path={AGENTIC}" in str(err.value)


def test_ir1_append_roles_rejected_with_template_reason():
    c = sub(make_config(), "agent", use_session_server=True, tito_model="qwen3",
            tito_allowed_append_roles=("tool",))
    with pytest.raises(cfg.UnmappedConfigError, match=r"decided by the --tito-model template \(allowed_append_roles\)"):
        cfg.translate_run_config(c, AlgorithmSpec())


def test_ir1_session_server_and_partial_rollout_are_mutually_exclusive():
    with pytest.raises(cfg.UnmappedConfigError) as err:
        cfg.check_session_server_partial_rollout(True, True)
    assert 'miles/utils/arguments.py:3240 "--use-session-server does not support --partial-rollout"' in str(err.value)
    cfg.check_session_server_partial_rollout(True, False)
    cfg.check_session_server_partial_rollout(False, True)
    # the parsed-namespace check (Miles normalization could set either)
    c = make_config()
    launch = cfg.translate_run_config(c, AlgorithmSpec())
    ns = SimpleNamespace(use_session_server=True, partial_rollout=True, indep_dp=False)
    with pytest.raises(cfg.UnmappedConfigError, match="arguments.py:3240"):
        cfg.validate_parsed_args(ns, launch, num_cells=lambda _a: 1)


def test_ir1_reward_scope_segment_fails_at_startup():
    launch = cfg.translate_run_config(make_config(), AlgorithmSpec())
    cfg.check_harness_reward_scope(None)
    cfg.check_harness_reward_scope("trajectory")
    with pytest.raises(cfg.MilesConfigError, match="reward_scope='segment'"):
        cfg.check_harness_reward_scope("segment")
    ns = SimpleNamespace(use_session_server=False, partial_rollout=False, indep_dp=False,
                         yeto_harness_reward_scope="segment")
    with pytest.raises(cfg.MilesConfigError, match="reward_scope"):
        cfg.validate_parsed_args(ns, launch, num_cells=lambda _a: 1)


def test_ir1_leaf_policy_still_covers_every_run_config_leaf():
    cfg.check_config_mapped(make_config())  # unchanged default config still maps


# ---------------------------------------------------------------------------
# IR-1 entry preflight hook
# ---------------------------------------------------------------------------
def _stage_args(monkeypatch):
    """Stub the contract preflight pieces so preflight_stage runs without Miles."""
    monkeypatch.setattr(entry, "ports_runtime_fingerprint", lambda launch: "fp")
    monkeypatch.setattr(entry, "miles_capabilities", lambda fp, unverified_mechanisms=(): "caps")
    monkeypatch.setattr(entry, "with_partitioned_serial", lambda caps: caps)
    monkeypatch.setattr(entry, "execution_profile_for", lambda *a, **k: "profile")
    monkeypatch.setattr(entry, "expected_algorithm_sha256", lambda a: None)
    monkeypatch.setattr(entry, "preflight", lambda profile, algorithm, caps: None)
    monkeypatch.setattr(entry, "elastic_wiring_for", lambda a, profile, fingerprint: None)
    allocations = []
    monkeypatch.setattr(entry, "connect_island_ray", lambda *a, **k: allocations.append("ray"))
    return allocations


def test_ir1_harness_preflight_runs_before_any_allocation(monkeypatch):
    allocations = _stage_args(monkeypatch)
    miles_args = SimpleNamespace(yeto_harness_preflight=None)
    seen = []

    def hook(args, launch):
        seen.append((args, launch))
        raise RuntimeError("codex binary sha mismatch")

    with pytest.raises(RuntimeError, match="codex binary sha mismatch"):
        entry.preflight_stage(miles_args, "launch", AlgorithmSpec(), yeto_policy_sync=False,
                              harness_preflight=hook)
    assert seen == [(miles_args, "launch")]
    assert allocations == []  # no Ray connect, no placement, no model allocation

    ok = entry.preflight_stage(miles_args, "launch", AlgorithmSpec(), yeto_policy_sync=False,
                               harness_preflight=lambda a, l: None)
    assert ok == ("fp", "caps", "profile", None)


def test_ir1_harness_preflight_resolves_from_args_or_env(monkeypatch):
    calls = []
    monkeypatch.setattr(entry, "_ir_test_hook", lambda a, l: calls.append("env"), raising=False)
    assert entry.resolve_harness_preflight(SimpleNamespace(), environ={}) is None
    hook = entry.resolve_harness_preflight(
        SimpleNamespace(), environ={entry.HARNESS_PREFLIGHT_ENV: "yeto.rl.engine.miles_adapter.entry:_ir_test_hook"})
    hook(None, None)
    assert calls == ["env"]
    direct = lambda a, l: calls.append("args")  # noqa: E731
    assert entry.resolve_harness_preflight(SimpleNamespace(yeto_harness_preflight=direct)) is direct
    with pytest.raises(cfg.MilesConfigError):
        entry.resolve_harness_preflight(SimpleNamespace(), environ={entry.HARNESS_PREFLIGHT_ENV: "nomodule"})


# ---------------------------------------------------------------------------
# IR-2 drain blockers, admission, bounded env_live
# ---------------------------------------------------------------------------
def test_ir2_drain_blockers_fail_closed_and_count_harness():
    tw = ToolWaitBoard()
    assert drain_blockers(0, tw.snapshot(), HARNESS_ZERO) == []
    assert drain_blockers(0, tw.snapshot(), None) == ["harness counts unknown"]
    # active=0 but tool_wait>0: not drained
    tw.enter("t")
    assert drain_blockers(0, tw.snapshot(), HARNESS_ZERO) == ["1 trajectories waiting on tools"]
    tw.exit("t")
    # active=0, tool_wait=0 but env_live>0 (an idle sandbox): not drained
    assert drain_blockers(0, tw.snapshot(), HarnessSnapshot(0, 2, 1)) == ["2 sandboxes live"]
    assert drain_blockers(0, tw.snapshot(), HarnessSnapshot(3, 2, 1)) == [
        "3 harness sessions in flight", "2 sandboxes live"]


def test_ir2_harness_board_admission_closes_on_drain_and_rejects_new_sessions():
    board = HarnessBoard()
    assert board.allow_new_session("rollout-0")
    board.enter_session("s1", "rollout-0")
    board.close_admission(["rollout-0"])
    assert board.allow_new_session("rollout-0") is False
    assert board.allow_new_session("rollout-1") is True  # re-target a non-drained member
    with pytest.raises(HarnessAdmissionError):
        board.enter_session("s2", "rollout-0")
    assert read_harness(board).in_flight == 1  # the refused session never counted
    board.exit_session("s1")
    assert drain_blockers(0, ToolWaitBoard().snapshot(), board.snapshot()) == []
    board.open_admission(["rollout-0"])
    board.enter_session("s3", "rollout-0")
    assert board.snapshot().in_flight == 1


def test_ir2_env_live_counts_idle_leases_and_is_bounded_by_hard_deadlines():
    now = [100.0]
    board = HarnessBoard(clock=lambda: now[0])
    board.lease_acquired("L1", deadline=130.0)  # idle sandbox, nobody in a session
    board.lease_acquired("L2", deadline=120.0)
    snap = board.snapshot()
    assert (snap.in_flight, snap.env_live, snap.latest_lease_deadline) == (0, 2, 130.0)
    assert drain_blockers(0, ToolWaitBoard().snapshot(), snap) == ["2 sandboxes live"]
    now[0] = 125.0  # L2 passed its hard deadline -> force-released, counted as infra error
    snap = board.snapshot()
    assert (snap.env_live, snap.leases_expired_total, snap.latest_lease_deadline) == (1, 1, 130.0)
    now[0] = 131.0
    assert board.snapshot().env_live == 0  # bounded: the drain cannot wait past 130
    assert drain_blockers(0, ToolWaitBoard().snapshot(), board.snapshot()) == []
    board.lease_acquired("L3", deadline=200.0)
    board.lease_released("L3")
    assert board.snapshot().env_live == 0


def _pool(**kw):
    class Fork:
        def __init__(self):
            self.calls = []

        async def drain_cells(self, cells, timeout_seconds):
            self.calls.append(("drain", tuple(sorted(cells))))
            return True

        async def uncordon_cells(self, cells):
            self.calls.append(("uncordon", tuple(sorted(cells))))

    fork = Fork()
    pool = MilesRolloutPool(inference_controller=fork, rollout_executor=None, metadata=None,
                            expected_policy=lambda: (0, "h"),
                            runner=SimpleNamespace(run=asyncio.run), declared_cells=("c0", "c1"),
                            **kw)
    pool.load_sample = lambda: {"active_requests": 0, "workers": 2, "cordoned": 0}
    return pool, fork


def test_ir2_pool_probe_reports_harness_and_fails_closed_without_it():
    tw = ToolWaitBoard()
    unknown, _ = _pool(tool_wait_board=tw)  # harness=None: unknown
    assert unknown.trajectory_load()["blockers"] == ["harness counts unknown"]
    zeros, _ = _pool(tool_wait_board=tw, harness=HARNESS_NOT_AGENTIC)
    assert zeros.trajectory_load()["blockers"] == []
    hb = HarnessBoard()
    hb.lease_acquired("L", deadline=time.monotonic() + 60)
    pool, fork = _pool(tool_wait_board=tw, harness=hb)
    load = pool.trajectory_load()
    assert load["env_live"] == 1 and load["harness_in_flight"] == 0
    assert load["blockers"] == ["1 sandboxes live"]  # active=0, tool_wait=0, env_live>0
    # drain closes admission to the members FIRST, undrain reopens it after the uncordon
    members = frozenset({"engine:c0"})
    assert pool.drain(members, time.time() + 5) is True
    assert hb.allow_new_session("engine:c0") is False and hb.allow_new_session("engine:c1")
    with pytest.raises(HarnessAdmissionError):
        hb.enter_session("new", "engine:c0")
    pool.undrain(members)
    assert hb.allow_new_session("engine:c0") is True
    assert [c[0] for c in fork.calls] == ["drain", "uncordon"]


def test_ir2_load_sample_carries_harness_fields_and_classify_ignores_them():
    hb = HarnessBoard()
    hb.enter_session("s", "m")
    hb.record_chain_break("retry_fork")
    hb.record_session_mismatch()
    hb.record_policy_age_violation(2)
    pool, _ = _pool(harness=hb)
    del pool.load_sample  # use the real one
    http = {
        "http://r:1/worker_inflight": {"inflight": {"http://e:1": 0}, "cordoned": []},
        "http://e:1/get_load": [{"num_reqs": 0, "num_waiting_reqs": 0}],
        "http://e:1/server_info": {"internal_states": [{"effective_max_running_requests_per_dp": 4}]},
    }
    pool._args = SimpleNamespace(sglang_router_ip="r", sglang_router_port=1)
    pool._load_tool_wait = ToolWaitBoard()
    sample = pool.load_sample(http_get=lambda url: http[url])
    assert sample["harness_in_flight"] == 1 and sample["env_live"] == 0
    assert sample["tito_chain_breaks"] == {"retry_fork": 1}
    assert sample["tito_session_mismatch"] == 1 and sample["policy_age_violation"] == 2
    assert sample["load_class"] == "rollout-idle"  # harness fields are not GPU gaps
    assert validate_load_sample(sample) == []
    assert classify_load(LoadSample(0, 0, 0, 0, 4, harness_in_flight=3, env_live=2)) == "rollout-idle"
    assert classify_load(LoadSample(0, 0, 1, 0, 4, env_live=2)) == "tool-wait"
    assert LoadSample(0, 0, 0, 0, 4) == LoadSample(0, 0, 0, 0, 4, 0, 0)  # old ctor unchanged


# ---------------------------------------------------------------------------
# IR-3 policy token driver -> pool
# ---------------------------------------------------------------------------
def _engine(**kw):
    return FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0, **kw)


def _driver(engine, tmp_path, rounds=2, **kw):
    kw.setdefault("capabilities", fake_capabilities())
    return IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                        policy_state=engine.policy_state, publisher=engine.publisher,
                        placement=engine.placement, algorithm=AlgorithmSpec(),
                        sync=LocalOnlySync(rounds), events=EventTape(tmp_path / "e.jsonl", 0), **kw)


def test_ir3_driver_passes_its_policy_token_to_every_generate(tmp_path):
    engine = _engine()
    driver = _driver(engine, tmp_path)
    driver.run()
    assert len(engine.expected_policy_versions) == 2
    for rollout_id, token in enumerate(engine.expected_policy_versions):
        assert token == policy_token(rollout_id, engine.rollout_digest(rollout_id)) if hasattr(engine, "rollout_digest") \
            else token.startswith(f"yeto:{rollout_id}:")
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    tokens = [e["rl/policy_token"] for e in events if e["event"] == "rl_publication"]
    assert tokens[:2] == engine.expected_policy_versions


def test_ir3_policy_version_drift_aborts_the_round_before_training(tmp_path):
    engine = _engine(policy_drift_rounds={1})
    driver = _driver(engine, tmp_path, rounds=3)
    with pytest.raises(PolicyIdentityError, match="policy_age_violation"):
        driver.run()
    assert ("train", 0) in engine.calls and ("train", 1) not in engine.calls


def test_ir3_miles_pool_refuses_a_foreign_target_token_before_sampling():
    class Controller:
        async def prepare_rollout(self, rollout_id):
            raise AssertionError("must not sample")

    tokens = []
    meta = SimpleNamespace(set_policy_token=tokens.append, take=lambda r: None)
    pool = MilesRolloutPool(inference_controller=Controller(), rollout_executor=None,
                            metadata=meta, expected_policy=lambda: (3, "abc"),
                            runner=SimpleNamespace(run=asyncio.run), harness=HARNESS_NOT_AGENTIC)
    with pytest.raises(PolicyIdentityError, match="driver expects yeto:3:zzz"):
        pool.generate(3, expected_policy_version="yeto:3:zzz")
    assert tokens == []  # nothing published to the rollout side


def test_ir3_rollout_side_reads_expected_version_and_sums_violations(tmp_path, monkeypatch):
    from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook

    sink = f"dir:{tmp_path}"
    monkeypatch.setenv(hook.META_SINK_ENV, sink)
    assert hook.expected_policy_version(SimpleNamespace(metadata={})) is None
    hook.put_policy_token("yeto:7:h")
    assert hook.expected_policy_version(SimpleNamespace(metadata={})) == "yeto:7:h"
    assert hook.expected_policy_version(SimpleNamespace(metadata={"expected_policy_version": "yeto:7:x"})) == "yeto:7:x"
    samples = [[SimpleNamespace(metadata={"policy_age_violation": 1, "tito_chain_breaks": {"retry_fork": 2}}),
                SimpleNamespace(metadata={"tito_session_mismatch": 1})],
               [SimpleNamespace(metadata=None)]]
    assert hook.harness_counters(samples) == {
        "policy_age_violation": 1, "tito_session_mismatch": 1, "tito_chain_breaks": {"retry_fork": 2}}
    assert hook.harness_counters([[SimpleNamespace(metadata={})]]) == {}  # old key set unchanged


def test_ir3_handle_from_metadata_carries_policy_age_violation():
    from yeto.rl.engine.miles_adapter.rollout import handle_from_metadata
    from yeto.rl.engine.miles_adapter.rollout_meta_hook import METADATA_SCHEMA

    payload = {"schema": METADATA_SCHEMA, "rollout_id": 1, "completed": 1, "aborted": 1,
               "groups": [{"group_id": "g0", "sample_ids": ["s0"], "policy_token": "yeto:1:h",
                           "reward_mean": 0.0, "reward_std": 0.0, "token_count": 1}]}
    h = handle_from_metadata(payload, rollout_id=1, policy_version=1, policy_hash="h", data_pack=None)
    assert h.policy_age_violation is None
    h = handle_from_metadata({**payload, "policy_age_violation": 2}, rollout_id=1,
                             policy_version=1, policy_hash="h", data_pack=None)
    assert h.policy_age_violation == 2


# ---------------------------------------------------------------------------
# IR-4 schema
# ---------------------------------------------------------------------------
def test_ir4_schema_registers_harness_metrics_with_labels():
    assert set(HARNESS_METRIC_KEYS) <= set(LOAD_SAMPLE_SCHEMA)
    assert LOAD_SAMPLE_SCHEMA["harness_in_flight"] == ("gauge", int)
    assert LOAD_SAMPLE_SCHEMA["env_live"] == ("gauge", int)
    assert LOAD_SAMPLE_SCHEMA["tito_session_mismatch"] == ("counter", int)
    assert LOAD_SAMPLE_SCHEMA["tito_chain_breaks"] == ("counter", dict)
    assert LOAD_SAMPLE_SCHEMA["policy_age_violation"] == ("counter", int)
    assert LOAD_SAMPLE_LABELS == ("profile_hash", "epoch")
    assert validate_load_sample({"tito_chain_breaks": {"bogus": 1}}) == ["tito_chain_breaks: unknown reason 'bogus'"]
    assert validate_load_sample({"env_live": "1"}) == ["env_live: expected int, got str"]
    assert validate_load_sample({"nope": 1}) == ["unknown load key 'nope'"]


def _observed_driver(engine, tmp_path, observe):
    profile = ExecutionProfile(name="p", execution_mode="partitioned-serial",
                               outer_protocol="none").bind_algorithm(AlgorithmSpec())
    d = IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                     policy_state=engine.policy_state, publisher=engine.publisher,
                     placement=engine.placement,
                     capabilities=fake_capabilities(execution_modes={"partitioned-serial"}),
                     algorithm=AlgorithmSpec(), sync=LocalOnlySync(1),
                     events=EventTape(tmp_path / "e.jsonl", 0), profile=profile, observe=observe)
    d.load_sample_interval_s = 0.02
    return d


@pytest.mark.parametrize("observe", [False, True])
def test_ir4_load_samples_carry_harness_fields_only_when_observing(tmp_path, observe):
    import time as _time

    engine = _engine(placement_kind="fixed-partition")
    original = engine.rollout.generate
    engine.rollout.generate = lambda r, **kw: (_time.sleep(0.12), original(r, **kw))[1]
    engine.rollout.load_sample = lambda: {
        "active_requests": 0, "workers": 1, "cordoned": 0, "harness_in_flight": 2, "env_live": 3,
        "tito_session_mismatch": 0, "tito_chain_breaks": {}, "policy_age_violation": 0}
    _observed_driver(engine, tmp_path, observe).run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    samples = [e for e in events if e["event"] == "rl_load_sample"]
    if not observe:
        assert samples == []  # old path: nothing emitted, nothing changed
        assert not any(k in e for e in events for k in HARNESS_METRIC_KEYS)
        return
    assert samples
    for s in samples:
        assert (s["harness_in_flight"], s["env_live"]) == (2, 3)
        assert isinstance(s["tito_chain_breaks"], dict) and s["policy_age_violation"] == 0
        assert s["profile_hash"] and "epoch" in s
        assert validate_load_sample({k: s[k] for k in LOAD_SAMPLE_SCHEMA if k in s}) == []
