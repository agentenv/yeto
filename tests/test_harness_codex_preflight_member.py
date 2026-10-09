"""T3 G4: the Codex harness preflight derives the INFRA rollout member key."""

from __future__ import annotations

from types import SimpleNamespace

from yeto.rl.adapters.miles.rollout import member_id
from yeto.rl.adapters.miles import entry
from yeto.rl.harness.codex import preflight


def test_resolve_member_accepts_member_key_cell_attr_or_env():
    # explicit full member key
    assert preflight.resolve_member(SimpleNamespace(yeto_rl_member_id="engine:cell-7"), {}) == "engine:cell-7"
    # bare cell id in yeto_rl_member_id -> INFRA key
    assert preflight.resolve_member(SimpleNamespace(yeto_rl_member_id="cell-7"), {}) == member_id("cell-7")
    # cell id attr set by entry/INFRA
    assert preflight.resolve_member(SimpleNamespace(yeto_rl_cell_id=3), {}) == "engine:3"
    # environment fallback
    assert preflight.resolve_member(SimpleNamespace(), {preflight.MEMBER_CELL_ENV: "c0"}) == "engine:c0"
    # nothing -> global admission key (single-island smoke)
    assert preflight.resolve_member(SimpleNamespace(), {}) is None
    assert preflight.resolve_member(SimpleNamespace(yeto_rl_member_id=""), {}) is None


def test_harness_preflight_configures_the_member(monkeypatch):
    seen = {}
    from yeto.rl.harness.codex import codex_openenv_subprocess_agent_function as agent

    monkeypatch.setattr(agent, "configure", lambda **kw: seen.update(kw))
    monkeypatch.setattr(preflight, "preflight_codex_openenv", lambda env: {})
    monkeypatch.setattr(preflight, "resolve_environment_provider", lambda a, e: object())
    monkeypatch.setattr(preflight, "island_boards", lambda a: (None, None))
    args = SimpleNamespace(custom_agent_function_path=preflight.EXPECTED_AGENT_FUNCTION,
                           yeto_harness_reward_scope="trajectory", yeto_rl_cell_id="cell-1")
    preflight.harness_preflight(args, None, env={})
    assert seen["member"] == "engine:cell-1"


# --- rl-fn-codex-rollout 0.6 (A16 / D5): the island entry publishes its own cell ---

def test_entry_publishes_cell_from_learner_id_single_island_is_engine_0():
    env = {}
    args = SimpleNamespace(yeto_rl_learner_id=0)
    assert entry.publish_member_cell(args, 0, env) == "0"
    assert args.yeto_rl_cell_id == "0" and env[preflight.MEMBER_CELL_ENV] == "0"
    assert preflight.resolve_member(args, {}) == "engine:0"  # field, no env needed
    assert preflight.resolve_member(SimpleNamespace(), env) == "engine:0"  # env fallback
    assert preflight.worker_runtime_env(args, env)[preflight.MEMBER_CELL_ENV] == "0"


def test_entry_publishes_distinct_cells_for_two_islands_and_admission_is_per_member():
    from yeto.rl.engine.tool_wait import HarnessBoard

    envs = [{}, {}]
    islands = [SimpleNamespace(yeto_rl_learner_id=i) for i in range(2)]
    members = [preflight.resolve_member(a, e) for a, e in zip(islands, envs)
               for _ in [entry.publish_member_cell(a, i, e) for i in [a.yeto_rl_learner_id]]]
    assert members == ["engine:0", "engine:1"]
    assert [e[preflight.MEMBER_CELL_ENV] for e in envs] == ["0", "1"]
    hb = HarnessBoard()  # the shared fake board: closing one island's member leaves the other open
    hb.close_admission([members[1]])
    assert hb.allow_new_session(members[1]) is False
    assert hb.allow_new_session(members[0]) is True


def test_entry_keeps_an_explicit_cell_and_falls_back_to_learner_field():
    env = {}
    args = SimpleNamespace(yeto_rl_cell_id="c7", yeto_rl_learner_id=3)
    assert entry.publish_member_cell(args, 3, env) == "c7" and env[preflight.MEMBER_CELL_ENV] == "c7"
    args = SimpleNamespace(yeto_rl_learner_id=2)
    assert entry.publish_member_cell(args, environ=env) == "2"
    assert preflight.resolve_member(args, {}) == "engine:2"


def test_configure_rollout_worker_reads_cell_from_env(monkeypatch):
    seen = {}
    from yeto.rl.harness.codex import codex_openenv_subprocess_agent_function as agent

    monkeypatch.setattr(agent, "configure", lambda **kw: seen.update(kw))
    monkeypatch.setattr(preflight, "resolve_environment_provider", lambda a, e: object())
    monkeypatch.setattr(preflight, "island_boards", lambda a: (None, None))
    assert preflight.configure_rollout_worker({preflight.ENVIRONMENT_PROVIDER_ENV: "x:y",
                                               preflight.MEMBER_CELL_ENV: "1"}) is True
    assert seen["member"] == "engine:1"
    assert preflight.configure_rollout_worker({}) is False


def test_preflight_stage_publishes_engine_0_on_a_single_island(monkeypatch):
    from yeto.rl.engine.algorithm import AlgorithmSpec

    for name, val in [("ports_runtime_fingerprint", lambda launch: "fp"),
                      ("miles_capabilities", lambda fp, unverified_mechanisms=(): "caps"),
                      ("with_partitioned_serial", lambda caps: caps),
                      ("execution_profile_for", lambda *a, **k: "profile"),
                      ("expected_algorithm_sha256", lambda a: None),
                      ("preflight", lambda profile, algorithm, caps: None),
                      ("elastic_wiring_for", lambda a, profile, fingerprint: None),
                      ("resolve_harness_preflight", lambda a, environ=None: None)]:
        monkeypatch.setattr(entry, name, val)
    monkeypatch.delenv(preflight.MEMBER_CELL_ENV, raising=False)
    args = SimpleNamespace(yeto_rl_learner_id=0)
    entry.preflight_stage(args, "launch", AlgorithmSpec(), yeto_policy_sync=False)
    import os
    assert args.yeto_rl_cell_id == "0" and os.environ[preflight.MEMBER_CELL_ENV] == "0"
    assert preflight.resolve_member(args) == "engine:0"
