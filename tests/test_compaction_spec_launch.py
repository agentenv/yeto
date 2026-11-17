"""The sampler spec's ``agentic.compaction`` (async GRPO context compaction): parsing
and validation, the records proxy it reaches, the pool dry run, and the launch
checks (Miles + a GRPO-family estimator only; verl and PPO CompactionRL refused;
segment-wide rows; an agent whose own compaction is off; the drain warning).
Without the section everything is as before."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_agentic_drain_defer import _factory_world
from tests.test_nc_pool_run import SPEC, _cfg
from tests.test_nc_pool_run import _env  # noqa: F401 - autouse: the National Compute env of the dry runs
from tests.test_run_up_consistency import setup  # noqa: F401 - the fixture
from yeto.launch.harness_env import check_compaction_launch, check_grpo_compaction
from yeto.launch.nc import cli
from yeto.rl import compaction as C
from yeto.rl.agentic import HarnessConfigError
from yeto.rl.engine import agentic_round as ar
from yeto.rl.harness.harbor import pi_config

GRPO_SPEC = Path(__file__).resolve().parents[1] / "examples/k31/algorithm-spec.async-age2.json"
# the summary-turn keys of CompactionConfig (accepted once the config has them)
HAS_SUMMARY_KEYS = "summary_prompt" in C.SPEC_KEYS
AGENTIC = {"turn_max_tokens": 4096, "max_response_len": 1 << 20}


def _spec(section, *, max_model_len=None, **agentic):
    spec = {"agentic": {**AGENTIC, **agentic, "compaction": section}}
    if max_model_len is not None:
        spec["engine"] = {"server": {"max_model_len": max_model_len}}
    return spec


# -- parsing --------------------------------------------------------------------

@pytest.mark.parametrize("spec", [{}, {"agentic": {}}, _spec(None), {"agentic": None}, {"sampling": {}},
                                  {"agentic": {"task_defaults": {"agent": "codex"}}}])
def test_no_section_or_null_is_no_compaction(spec):
    assert C.config_from_spec(spec) is None
    assert ar.spec_compaction(spec) is None


def test_only_context_budget_is_required_and_the_rest_defaults_to_the_paper():
    cfg = C.config_from_spec(_spec({"context_budget": 65536}))
    assert cfg == C.CompactionConfig(context_budget=65536, t_comp=C.T_COMP,
                                     keep_recent_steps=C.KEEP_RECENT_STEPS, max_compactions=C.MAX_COMPACTIONS)


def test_every_key_reaches_the_config():
    cfg = C.config_from_spec(_spec({"context_budget": 262144, "t_comp": 10240, "keep_recent_steps": 0,
                                    "max_compactions": 0}, max_model_len=262144))
    assert (cfg.context_budget, cfg.t_comp, cfg.keep_recent_steps, cfg.max_compactions) == (262144, 10240, 0, 0)
    assert "context_budget 262144" in C.describe(cfg) and "at most 1 segments" in C.describe(cfg)
    assert "max_compactions 0: the proxy never compacts" in C.describe(cfg)


def test_max_turns_is_no_spec_key():
    """``max_turns`` is the episode driver's evaluation cap; the records proxy does not
    read it, so the spec cannot set it."""
    assert "max_turns" not in C.SPEC_KEYS
    assert {"context_budget", "t_comp", "keep_recent_steps", "max_compactions"} <= set(C.SPEC_KEYS)


@pytest.mark.parametrize(("section", "match"), [
    ({}, "context_budget is required"),
    ({"context_budget": 65536, "max_turns": 10}, r"unknown key\(s\) \['max_turns'\]"),
    ({"context_budget": 65536, "tcomp": 1}, "unknown key"),
    ({"context_budget": 0}, "context_budget must be >= 1"),
    ({"context_budget": -5}, "context_budget must be >= 1"),
    ({"context_budget": 65536.0}, "context_budget must be an integer"),
    ({"context_budget": "65536"}, "context_budget must be an integer"),
    ({"context_budget": True}, "context_budget must be an integer"),
    ({"context_budget": 8192}, "must exceed t_comp 10240"),
    ({"context_budget": 10240, "t_comp": 10240}, "must exceed t_comp"),
    ({"context_budget": 65536, "t_comp": 0}, "t_comp must be >= 1"),
    ({"context_budget": 65536, "keep_recent_steps": -1}, "keep_recent_steps must be >= 0"),
    ({"context_budget": 65536, "max_compactions": -1}, "max_compactions must be >= 0"),
    ({"context_budget": 65536, "max_compactions": None}, "max_compactions must be an integer"),
    ([65536], "must be an object or null"),
])
def test_a_bad_section_is_refused(section, match):
    with pytest.raises(ValueError, match=match):
        C.config_from_spec(_spec(section))
    with pytest.raises(HarnessConfigError, match=match):
        ar.spec_compaction(_spec(section))


def test_context_budget_must_fit_the_engine_context():
    with pytest.raises(ValueError, match="exceeds the engine context length max_model_len 32768"):
        C.config_from_spec(_spec({"context_budget": 65536}, max_model_len=32768))
    kwargs = {"agentic": {**AGENTIC, "compaction": {"context_budget": 65536}},
              "engine": {"kwargs": {"max_model_len": 4096}}}
    with pytest.raises(ValueError, match="max_model_len 4096"):
        C.config_from_spec(kwargs)
    assert C.config_from_spec(_spec({"context_budget": 32768}, max_model_len=32768)).context_budget == 32768
    # the running engine's own limit wins over the spec's
    with pytest.raises(HarnessConfigError, match="max_model_len 16384"):
        ar.spec_compaction(_spec({"context_budget": 32768}, max_model_len=32768), SimpleNamespace(max_model_len=16384))


def test_t_comp_must_hold_the_summary_turn():
    """Compaction starts once fewer than t_comp tokens remain: that room takes the
    summary request and the summary (CompactionConfig's rule where it has one)."""
    assert C.config_from_spec(_spec({"context_budget": 65536})) is not None  # t_comp 10240, turns of 4096
    with pytest.raises(ValueError, match="t_comp 2048 cannot hold the summary turn"):
        C.config_from_spec(_spec({"context_budget": 65536, "t_comp": 2048}))


@pytest.mark.skipif(hasattr(C.CompactionConfig, "check_summary_room"), reason="CompactionConfig owns the rule")
def test_the_summary_room_before_compaction_config_owns_it():
    """Before CompactionConfig has the rule, the summary turn is a plain turn
    (turn_max_tokens) with SUMMARY_PROMPT."""
    ok = 10240 - (len(C.SUMMARY_PROMPT.encode()) + 64)
    assert C.config_from_spec(_spec({"context_budget": 65536}, turn_max_tokens=ok)) is not None
    with pytest.raises(ValueError, match="t_comp 10240 cannot hold the summary turn"):
        C.config_from_spec(_spec({"context_budget": 65536}, turn_max_tokens=ok + 1))


@pytest.mark.skipif(not hasattr(C.CompactionConfig, "check_summary_headroom"),
                    reason="CompactionConfig has no summary headroom check yet")
def test_the_summary_turn_must_fit_the_engine_context():
    with pytest.raises(ValueError, match="agentic.compaction: the summary turn can overrun max_model_len 262144"):
        C.config_from_spec(_spec({"context_budget": 262144}, max_model_len=262144, turn_max_tokens=8192))
    assert C.config_from_spec(_spec({"context_budget": 262144 - 16384}, max_model_len=262144,
                                    turn_max_tokens=8192)) is not None


@pytest.mark.parametrize("missing", ["turn_max_tokens", "max_response_len"])
def test_compaction_needs_the_turn_and_row_caps(missing):
    spec = _spec({"context_budget": 65536})
    del spec["agentic"][missing]
    with pytest.raises(ValueError, match=f"needs agentic.{missing}"):
        C.config_from_spec(spec)


def test_the_sampler_row_must_hold_a_whole_segment():
    with pytest.raises(ValueError, match=r"agentic.max_response_len 16384 < agentic.compaction.context_budget "
                                         r"65536: .* the summary .* is the part cut"):
        C.config_from_spec(_spec({"context_budget": 65536}, max_response_len=16384))
    assert C.config_from_spec(_spec({"context_budget": 65536}, max_response_len=65536)) is not None


def test_a_default_agent_that_compacts_itself_is_refused():
    assert pi_config.PROXY_COMPACTION_AGENTS == {"pi"}
    assert pi_config.settings_json()["compaction"] == {"enabled": False}  # pi's own compaction is off
    for agent in ("codex", "terminus-2"):
        with pytest.raises(ValueError, match=f"task_defaults.agent '{agent}' may compact its own history"):
            C.config_from_spec(_spec({"context_budget": 65536}, task_defaults={"agent": agent}))
    assert C.config_from_spec(_spec({"context_budget": 65536}, task_defaults={"agent": "pi"})) is not None


@pytest.mark.skipif(not HAS_SUMMARY_KEYS, reason="CompactionConfig has no summary-turn fields yet")
def test_the_summary_turn_keys_reach_the_config():
    cfg = C.config_from_spec(_spec({"context_budget": 65536, "summary_max_tokens": 2048,
                                    "summary_prompt": "Summarize.", "resume_template": "Earlier: {summary}"},
                                   turn_max_tokens=16384))
    assert (cfg.summary_max_tokens, cfg.summary_prompt, cfg.resume_template) == (2048, "Summarize.",
                                                                                 "Earlier: {summary}")
    assert "summary_max_tokens 2048" in C.describe(cfg)
    with pytest.raises(ValueError, match="agentic.compaction: .*resume_template"):
        C.config_from_spec(_spec({"context_budget": 65536, "resume_template": "no field"}))
    with pytest.raises(ValueError, match="cannot hold the summary turn"):
        C.config_from_spec(_spec({"context_budget": 65536, "t_comp": 2048, "summary_max_tokens": 2048}))


# -- the records proxy and the trials get it -----------------------------------

def _round(monkeypatch, agentic, *, build=None):
    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.harness import harbor
    from yeto.rl.harness.gateway import proxy

    _factory_world(monkeypatch)
    seen = SimpleNamespace(proxy=[], harness=[])
    monkeypatch.setattr(proxy, "RecordsProxy", lambda *a, **k: seen.proxy.append((a, k)) or object())
    monkeypatch.setattr(harbor, "HarborHarness", lambda builder, **k: seen.harness.append(builder) or object())
    engine = FakeEngine(max_model_len=65536)
    engine.start(None, 0)
    spec = {"sampling": {}, "agentic": {"turn_max_tokens": 8, "template": "t", "max_in_flight": 2,
                                        "max_response_len": 16, **agentic}}
    ar.round_factory(spec, engine, {})
    (args, kwargs), = seen.proxy
    assert args[0] is engine
    return kwargs, seen.harness[0]


def test_round_factory_passes_the_compaction_to_the_records_proxy(monkeypatch):
    kwargs, _build = _round(monkeypatch, {"compaction": {"context_budget": 32768, "t_comp": 4096},
                                          "max_response_len": 32768})
    assert kwargs["compaction"] == C.CompactionConfig(context_budget=32768, t_comp=4096)


def test_round_factory_without_the_section_builds_as_before(monkeypatch):
    """No ``agentic.compaction``: the proxy's arguments are main's (``compaction=None``
    is RecordsProxy's own default) and the trial builder is the plain one."""
    from functools import partial

    from yeto.rl.harness.gateway.proxy import RecordsProxy
    from yeto.rl.harness.harbor import build_trial_config

    assert inspect.signature(RecordsProxy).parameters["compaction"].default is None
    for agentic in ({}, {"compaction": None}):
        kwargs, build = _round(monkeypatch, agentic)
        assert kwargs == {"model": "model", "signed_sampling": {"max_tokens": 8, "temperature": 1.0, "top_p": 1.0},
                          "sampler_id": "records-proxy", "compaction": None, "r3_response_len": 16}
        assert isinstance(build, partial) and build.func is build_trial_config


def test_round_factory_refuses_a_budget_above_the_engine_context(monkeypatch):
    with pytest.raises(HarnessConfigError, match="max_model_len 65536"):
        _round(monkeypatch, {"compaction": {"context_budget": 131072}, "max_response_len": 131072})


def test_a_compacting_round_refuses_a_task_whose_agent_compacts_itself():
    built = []
    config = SimpleNamespace(agent=SimpleNamespace(import_path=pi_config.PI_AGENT_IMPORT_PATH, kwargs={}))
    build = ar.proxy_compaction_trials(lambda task, endpoint: built.append((task.agent, endpoint)) or config)
    assert build(SimpleNamespace(agent="pi", task_id="t1"), "http://e") is config
    with pytest.raises(HarnessConfigError, match="task t2: agent 'codex' may compact its own history"):
        build(SimpleNamespace(agent="codex", task_id="t2"), "http://e")
    assert built == [("pi", "http://e")]


def test_round_factory_wraps_the_trial_builder_when_compacting(monkeypatch):
    _kwargs, build = _round(monkeypatch, {"compaction": {"context_budget": 32768, "t_comp": 4096},
                                          "max_response_len": 32768})
    with pytest.raises(HarnessConfigError, match="may compact its own history"):
        build(SimpleNamespace(agent="codex", task_id="t"), "http://e")


# -- weight switches, the trial wall clock, pi's call timeout -----------------------

def _args(**kw):
    return SimpleNamespace(**{"rl_max_policy_age": 8, "rl_release_every": 4, "rl_release_skip_busy": False, **kw})


def test_preflight_warns_on_a_compacting_drain_sampler():
    from yeto.rl.up_consistency import compaction_switch_warnings

    spec = _spec({"context_budget": 65536}, drain_timeout_s=600,
                 task_defaults={"wall_clock_s": 20000})
    spec["reload_max_defer_s"] = 3600
    (a, b, c) = compaction_switch_warnings(spec, _args(rl_max_policy_age=2, rl_release_every=1))
    assert "effective drain cut 600 s" in a and "below 2400 s" in a and "weight_switch live" in a
    assert "--rl-max-policy-age 2" in b and "N + 6" in b
    assert "release every version" in c and "--rl-release-every N" in c
    # a cut at the floor, an old enough age and a release every 4 versions: nothing on the switch
    spec["agentic"]["drain_timeout_s"] = 2400
    assert compaction_switch_warnings(spec, _args()) == []
    assert compaction_switch_warnings(spec, _args(rl_release_every=1, rl_release_skip_busy=True)) == []
    # reload_max_defer_s still bounds the cut
    spec["reload_max_defer_s"] = 600
    assert "effective drain cut 600 s" in compaction_switch_warnings(spec, _args())[0]


def test_preflight_switch_warnings_are_off_on_a_live_engine_and_without_compaction():
    from yeto.rl.up_consistency import compaction_switch_warnings

    live = _spec({"context_budget": 65536}, weight_switch="live", task_defaults={"wall_clock_s": 20000})
    assert compaction_switch_warnings(live, _args(rl_max_policy_age=0, rl_release_every=1)) == []
    auto_live = _spec({"context_budget": 65536}, task_defaults={"wall_clock_s": 20000})
    auto_live["engine"] = {"options": {"live_update": True}}
    assert compaction_switch_warnings(auto_live, _args(rl_max_policy_age=0)) == []
    assert compaction_switch_warnings(_spec(None), _args(rl_max_policy_age=0)) == []


def test_preflight_warns_on_the_default_trial_wall_clock():
    from yeto.rl.up_consistency import compaction_switch_warnings

    (w,) = compaction_switch_warnings(_spec({"context_budget": 65536}, weight_switch="live"), None)
    assert "trial wall clock at 7200 s (the default)" in w and "task_defaults.wall_clock_s" in w
    (w,) = compaction_switch_warnings(_spec({"context_budget": 65536}, weight_switch="live",
                                            task_defaults={"wall_clock_s": 3600}), None)
    assert "at 3600 s:" in w


def test_a_compacting_round_gives_pi_the_longer_call_timeout():
    """pi waits 3600 s on a call of a compacting round (settings.json httpIdleTimeoutMs and
    the provider timeout); without compaction it keeps 1800 s."""
    assert pi_config.DEFAULT_HTTP_IDLE_TIMEOUT_S == 1800 and pi_config.COMPACTION_HTTP_IDLE_TIMEOUT_S == 3600
    pi = SimpleNamespace(import_path=pi_config.PI_AGENT_IMPORT_PATH, kwargs={"max_tokens": 8})
    build = ar.proxy_compaction_trials(lambda task, endpoint: SimpleNamespace(agent=pi))
    assert build(SimpleNamespace(agent="pi", task_id="t"), "http://e").agent.kwargs == {
        "max_tokens": 8, "http_idle_timeout_s": 3600}
    settings = json.loads(pi_config.setup_command("m", 8, http_idle_timeout_s=3600).split("<<'JSON'\n")[1]
                          .split("\nJSON")[0])
    assert settings["httpIdleTimeoutMs"] == 3_600_000 and settings["retry"]["provider"]["timeoutMs"] == 3_600_000
    assert pi_config.settings_json()["httpIdleTimeoutMs"] == 1_800_000
    other = SimpleNamespace(import_path="elsewhere", kwargs={})
    pi_config.use_proxy_compaction_timeouts(other)
    assert other.kwargs == {}


# -- pool --dry-run --------------------------------------------------------------

def _dry(tmp_path, capsys, spec_doc):
    _doc, path = _cfg(tmp_path, spec_doc=spec_doc)
    rc = cli.main(["pool", "--config", path, "--dry-run"])
    return rc, capsys.readouterr().out


def test_the_pool_dry_run_shows_the_compaction(tmp_path, capsys):
    doc = {**SPEC, "agentic": {**AGENTIC, "compaction": {"context_budget": 32768, "keep_recent_steps": 1}}}
    rc, out = _dry(tmp_path, capsys, doc)
    assert rc == 0, out
    assert ("agentic.compaction: on: context_budget 32768, t_comp 10240 (compacts once fewer than 10240 tokens "
            "remain), keep_recent_steps 1, max_compactions 3 (at most 4 segments per trial)") in out


@pytest.mark.parametrize(("section", "agentic", "match"), [
    ({"context_budget": 8192}, {}, "must exceed t_comp"),
    ({"context_budget": 32768, "bogus": 1}, {}, "unknown key"),
    ({"context_budget": 65536}, {"max_model_len": 4096}, "exceeds the engine context length max_model_len 4096"),
    ({"context_budget": 32768}, {"max_response_len": 16384}, "agentic.max_response_len 16384 <"),
    ({"context_budget": 32768, "t_comp": 2048}, {}, "cannot hold the summary turn"),
    ({"context_budget": 32768}, {"task_defaults": {"agent": "codex"}}, "may compact its own history"),
])
def test_the_pool_dry_run_refuses_a_bad_compaction(tmp_path, capsys, section, agentic, match):
    agentic = dict(agentic)
    engine = SPEC["engine"]
    if "max_model_len" in agentic:
        engine = {**engine, "kwargs": {"max_model_len": agentic.pop("max_model_len")}}
    rc, out = _dry(tmp_path, capsys, {**SPEC, "engine": engine,
                                      "agentic": {**AGENTIC, **agentic, "compaction": section}})
    assert rc == 1
    assert any(line.startswith("problem: agentic.") and match in line for line in out.splitlines()), out


def test_the_pool_dry_run_output_is_unchanged_without_the_section(tmp_path, capsys):
    rc, plain = _dry(tmp_path, capsys, {**SPEC, "agentic": dict(AGENTIC)})
    rc_null, null = _dry(tmp_path, capsys, {**SPEC, "agentic": {**AGENTIC, "compaction": None}})
    assert rc == rc_null and plain == null
    assert "compaction" not in plain
    rc, no_agentic = _dry(tmp_path, capsys, SPEC)
    assert rc == 0 and "compaction" not in no_agentic


# -- launch checks ---------------------------------------------------------------

CFG = C.CompactionConfig(context_budget=65536)


def _trainer(**algorithm):
    return SimpleNamespace(rl_backend="miles", rl_algorithm_spec_json=json.dumps(algorithm) if algorithm else None)


@pytest.mark.parametrize("algorithm", [
    {"advantage": {"estimator": "grpo"}},
    {"advantage": {"estimator": "gspo"}},
    {"advantage": {"estimator": "grpo"}, "loss": {"variant": "score_centering"}},
    {"advantage_estimator": "grpo", "schema": "yeto-rl-algorithm-spec-v1"},
])
def test_miles_with_a_grpo_family_estimator_may_compact(algorithm):
    check_grpo_compaction(_trainer(**algorithm), CFG)


def test_verl_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="--rl-backend verl: a verl sample is one token stream"):
        check_grpo_compaction(SimpleNamespace(rl_backend="verl", rl_algorithm_spec_json=None), CFG)


@pytest.mark.parametrize(("algorithm", "match"), [
    ({"advantage": {"estimator": "gae", "gae_variant": "cross_segment_per_sample"}}, "PPO CompactionRL"),
    ({"advantage": {"estimator": "gae", "gae_variant": "cross_segment_whole_rollout"}}, "PPO CompactionRL"),
    ({"advantage": {"estimator": "gae"}}, "GRPO-family advantage estimator"),
    ({"advantage": {"estimator": "reinforce_plus_plus"}}, "GRPO-family advantage estimator"),
])
def test_ppo_compactionrl_and_other_estimators_are_refused(algorithm, match):
    with pytest.raises(ValueError, match=match):
        check_grpo_compaction(_trainer(**algorithm), CFG)


def test_no_compaction_checks_nothing():
    check_grpo_compaction(SimpleNamespace(rl_backend="verl"), None)
    check_grpo_compaction(_trainer(advantage={"estimator": "gae", "gae_variant": "cross_segment_per_sample"}),
                          None)


def test_the_trainer_launch_check_does_not_refuse_a_grpo_trainer_on_the_harbor_path():
    """The trainer launch (``check_compaction_launch``) only checks PPO CompactionRL:
    a GRPO trainer reading a compacting sampler's buffer launches; PPO CompactionRL on
    Harbor is still refused (``test_compactionrl_is_refused_on_the_harbor_path``)."""
    from yeto.rl.adapters.miles.harbor_agent import AGENT_FUNCTION

    grpo = SimpleNamespace(training_mode="rl", reward_function="yeto.rl.adapters.miles.harbor_agent.reward_func",
                           rl_algorithm_spec_json=GRPO_SPEC.read_text(), custom_agent_function_path=AGENT_FUNCTION)
    assert check_compaction_launch(grpo, environ={}) is None
    check_grpo_compaction(SimpleNamespace(rl_backend="miles", **vars(grpo)), CFG)
    with pytest.raises(ValueError, match="needs compacted rollouts"):
        check_compaction_launch(SimpleNamespace(**{**vars(grpo), "rl_algorithm_spec_json": json.dumps(
            {"advantage": {"gae_variant": "cross_segment_per_sample"}})}), environ={})


# -- yeto run up preflight -------------------------------------------------------

def _set(argv, flag, value):
    argv[argv.index(flag) + 1] = value


def _compacting(setup, budget=32768):  # noqa: F811
    setup.spec["agentic"] = {"turn_max_tokens": 4096, "max_response_len": budget,
                             "compaction": {"context_budget": budget}}
    _set(setup.argv, "--seq-len", str(budget))
    _set(setup.argv, "--rollout-max-response-len", str(budget))


def test_run_up_preflight_shows_the_compaction_and_passes_a_grpo_trainer(setup, capsys):  # noqa: F811
    _compacting(setup)
    setup.argv += ["--rl-algorithm-spec", str(GRPO_SPEC), "--rl-loss", "score_centering",
                   "--rl-max-policy-age", "2"]
    run = setup.make()
    assert run.consistency_problem() is None
    assert run.preflight() is True and run.calls
    out = capsys.readouterr().out
    assert "preflight: agentic.compaction on: context_budget 32768" in out
    # the default drain cut (600 s), --rl-max-policy-age 2, a release every version, the default wall clock
    assert "preflight: WARNING: agentic.compaction on a drain-mode sampler: the effective drain cut 600 s" in out
    assert "--rl-max-policy-age 2" in out and "release every version" in out
    assert "trial wall clock at 7200 s (the default)" in out


def test_run_up_preflight_refuses_compaction_on_verl(setup):  # noqa: F811
    _compacting(setup)
    setup.argv += ["--rl-backend", "verl"]
    run = setup.make()
    assert "--rl-backend verl" in run.consistency_problem()
    assert run.preflight() is False and run.calls == []


def test_run_up_preflight_refuses_the_k31_example_rows_with_compaction(setup):  # noqa: F811
    """The K3.1 example trainer (--seq-len 16384, --rollout-max-response-len 12288)
    cannot hold a 32k segment: refused before anything is ordered."""
    _compacting(setup)
    _set(setup.argv, "--seq-len", "16384")
    _set(setup.argv, "--rollout-max-response-len", "12288")
    run = setup.make()
    problem = run.consistency_problem()
    assert "--seq-len 16384 < the sampler spec's agentic.compaction.context_budget 32768" in problem
    assert run.preflight() is False and run.calls == []


def test_run_up_preflight_refuses_a_bad_section_and_says_nothing_without_one(setup, capsys):  # noqa: F811
    _compacting(setup, budget=65536)  # the spec's max_model_len is 32768
    assert "max_model_len 32768" in setup.make().consistency_problem()
    setup.spec["agentic"] = {}
    _set(setup.argv, "--seq-len", "16384")
    _set(setup.argv, "--rollout-max-response-len", "12288")
    run = setup.make()
    assert run.consistency_problem() is None and run.preflight() is True
    assert "compaction" not in capsys.readouterr().out


# -- buffer group size (one object PUT) -------------------------------------------

def _c12_size(trials, context_budget, *, max_compactions=0, codec="json"):
    """C12's bound (cmp1118/C12 ``buffer_payload.k31_group_payload_bytes``), for the
    wiring tests before it is merged."""
    return trials * (1 + max_compactions) * context_budget * {"json": 5324, "raw": 4001, "zstd": 2353}[codec]


@pytest.fixture
def sized(monkeypatch):
    from yeto.rl.engine import buffer_payload

    if not hasattr(buffer_payload, "k31_group_payload_bytes"):
        monkeypatch.setattr(buffer_payload, "k31_group_payload_bytes", _c12_size, raising=False)
        monkeypatch.setattr(buffer_payload, "SINGLE_PUT_LIMIT_BYTES", 5 * 2**30, raising=False)
    return buffer_payload


def _pool(tmp_path, spec):
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    return {"spec": str(tmp_path / "spec.json")}


K31_ROWS = {"rollout_fields": {"routed_experts": True, "top_logprobs_num": 128}, "sampling": {"n": 8},
            "engine": {"server": {"max_model_len": 262144}}}


def test_a_262k_group_without_compaction_is_over_one_put(sized, tmp_path):
    from yeto.rl.up_consistency import payload_size_warning

    spec = {**K31_ROWS, "agentic": {"max_prompt_len": 16384, "max_response_len": 262144 - 16384}}
    warning = payload_size_warning(_pool(tmp_path, spec), environ={})
    assert "one buffer group can reach 10.4 GiB (8 trials x 262144 tokens, codec json)" in warning
    assert "multipart write" in warning and "YETO_SAMPLE_BUFFER_PAYLOAD_CODEC=zstd" in warning
    assert "compaction" not in warning
    # the K3.1 example rows (16k + 16k): far below the limit
    small = {**K31_ROWS, "agentic": {"max_prompt_len": 16384, "max_response_len": 16384}}
    assert payload_size_warning(_pool(tmp_path, small), environ={}) is None
    # no routed experts: not a K3.1-shaped group, no estimate
    assert payload_size_warning(_pool(tmp_path, {**spec, "rollout_fields": {}}), environ={}) is None


def test_a_compacting_group_counts_every_segment(sized, tmp_path):
    from yeto.rl.up_consistency import payload_size_warning

    spec = {**K31_ROWS, "agentic": {**AGENTIC, "max_response_len": 262144,
                                    "compaction": {"context_budget": 262144, "max_compactions": 3}}}
    warning = payload_size_warning(_pool(tmp_path, spec), environ={})
    assert "41.6 GiB (8 trials x 4 segments x 262144 tokens (agentic.compaction), codec json)" in warning
    assert "context_budget / max_compactions" in warning
    zstd = payload_size_warning(_pool(tmp_path, spec), environ={"YETO_SAMPLE_BUFFER_PAYLOAD_CODEC": "zstd"})
    assert "codec zstd" in zstd


def test_no_size_bound_in_the_buffer_skips_the_check(monkeypatch, tmp_path):
    from yeto.rl.engine import buffer_payload
    from yeto.rl.up_consistency import payload_size_warning

    monkeypatch.delattr(buffer_payload, "k31_group_payload_bytes", raising=False)
    spec = {**K31_ROWS, "agentic": {"max_prompt_len": 16384, "max_response_len": 262144 - 16384}}
    assert payload_size_warning(_pool(tmp_path, spec), environ={}) is None


@pytest.mark.skipif(not hasattr(__import__("yeto.rl.engine.buffer_payload", fromlist=["x"]),
                                "k31_group_payload_bytes"), reason="buffer_payload has no size bound yet (C12)")
def test_the_real_size_bound_matches_the_wiring():
    from yeto.rl.engine import buffer_payload

    for codec in ("json", "raw", "zstd"):
        assert buffer_payload.k31_group_payload_bytes(8, 262144, max_compactions=3, codec=codec) == \
            _c12_size(8, 262144, max_compactions=3, codec=codec)


def test_run_up_preflight_warns_on_a_group_over_one_put(setup, sized, capsys):  # noqa: F811
    _compacting(setup)
    setup.spec["engine"]["server"]["max_model_len"] = 262144
    setup.spec["agentic"] = {"turn_max_tokens": 4096, "max_response_len": 262144,
                             "compaction": {"context_budget": 262144}}
    _set(setup.argv, "--seq-len", "262144")
    _set(setup.argv, "--rollout-max-response-len", "262144")
    run = setup.make()
    assert run.preflight() is True
    out = capsys.readouterr().out
    assert "preflight: WARNING: one buffer group can reach" in out and "multipart write" in out
