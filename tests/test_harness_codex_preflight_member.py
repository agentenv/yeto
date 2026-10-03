"""T3 G4: the Codex harness preflight derives the INFRA rollout member key."""

from __future__ import annotations

from types import SimpleNamespace

from yeto.rl.engine.miles_adapter.rollout import member_id
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
