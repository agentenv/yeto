"""The harness layer of the breaker registry (``yeto.breakers.layers.harness``) and its readers
(``yeto.rl.harness.harbor.breaker_settings``): defaults equal today's constants, every env knob
still works as an alias, a ``breakers.harness`` value reaches the harness through the env
write-back, and trips become aggregated ``rl_breaker`` events (CPU, no Harbor needed except
where marked)."""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest

from yeto import breakers
from yeto.breakers.events import validate_event
from yeto.rl.agentic import HarnessConfigError
from yeto.rl.harness.harbor import agentenv_client, agentenv_network, boat_client, breaker_settings, capacity, trial
from yeto.rl.harness.harbor.trial import HarborHarness, HarborSettings, sandbox_ttl_s, setup_retries_from_env

ENV_KNOBS = ("HARBOR_SETUP_RETRIES", "HARBOR_SETUP_BACKOFF_S", "HARBOR_SETUP_BACKOFF_MAX_S",
             "YETO_AGENTENV_SKIP_REACH_CHECK", "YETO_AGENTENV_REACH_TRIES", "YETO_AGENTENV_REACH_TIMEOUT_S",
             "YETO_AGENTENV_REACH_INTERVAL_S", "HARBOR_MAX_SANDBOXES", "YETO_AGENTENV_MAX_ACTIVE",
             "YETO_AGENTENV_STARTS_PER_MINUTE", "YETO_AGENTENV_STARTS_PER_HOUR", "YETO_AGENTENV_REQUEST_TIMEOUT_S",
             "YETO_BOAT_MAX_ACTIVE", "YETO_BOAT_STARTS_PER_MINUTE", "YETO_BOAT_STARTS_PER_HOUR",
             "HARBOR_SANDBOX_TTL_S", "HARBOR_SANDBOX_TTL_MARGIN_S", "HARBOR_SANDBOX_TTL_DEFAULT_S")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    breaker_settings.reset()
    yield
    breaker_settings.reset()


def _harness_entries():
    return {b.name: b for b in breakers.default_registry().layer("harness")}


def test_the_layer_loads_and_reads_exactly_the_env_knobs():
    entries = _harness_entries()
    assert set(entries) == {"harness.setup_retries", "harness.reach_check", "harness.sandbox_caps",
                            "harness.sandbox_ttl", "harness.turn_max_tokens", "harness.agent_call"}
    assert set(breaker_settings.env_names()) == set(ENV_KNOBS)
    # the knobs that existed before the registry are deprecated aliases; the new ones are the write-back channel
    deprecated = {a.path for b in entries.values() for a in b.aliases if a.deprecated}
    assert deprecated == set(ENV_KNOBS) - {"HARBOR_SETUP_BACKOFF_S", "HARBOR_SETUP_BACKOFF_MAX_S",
                                           "YETO_AGENTENV_REACH_INTERVAL_S", "HARBOR_SANDBOX_TTL_S",
                                           "HARBOR_SANDBOX_TTL_MARGIN_S", "HARBOR_SANDBOX_TTL_DEFAULT_S"}
    assert not entries["harness.sandbox_caps"].can_disable and not entries["harness.sandbox_ttl"].can_disable


def test_alias_names_are_the_runtime_constants():
    assert trial.SETUP_RETRIES_ENV == "HARBOR_SETUP_RETRIES"
    assert capacity.MAX_SANDBOXES_ENV == "HARBOR_MAX_SANDBOXES"
    assert (agentenv_client.MAX_ACTIVE_ENV, agentenv_client.MAX_STARTS_ENV, agentenv_client.MAX_STARTS_HOUR_ENV,
            agentenv_client.REQUEST_TIMEOUT_ENV) == ("YETO_AGENTENV_MAX_ACTIVE", "YETO_AGENTENV_STARTS_PER_MINUTE",
                                                     "YETO_AGENTENV_STARTS_PER_HOUR", "YETO_AGENTENV_REQUEST_TIMEOUT_S")
    assert (boat_client.MAX_ACTIVE_ENV, boat_client.MAX_STARTS_ENV, boat_client.MAX_STARTS_HOUR_ENV) == (
        "YETO_BOAT_MAX_ACTIVE", "YETO_BOAT_STARTS_PER_MINUTE", "YETO_BOAT_STARTS_PER_HOUR")


def test_agentenv_env_alias_names_are_the_runtime_constants():
    pytest.importorskip("harbor")
    from yeto.rl.harness.harbor import agentenv_env

    assert (agentenv_env.SKIP_REACH_ENV, agentenv_env.REACH_TRIES_ENV, agentenv_env.REACH_TIMEOUT_ENV) == (
        "YETO_AGENTENV_SKIP_REACH_CHECK", "YETO_AGENTENV_REACH_TRIES", "YETO_AGENTENV_REACH_TIMEOUT_S")


def test_defaults_are_todays_constants():
    """The layer writes literals (the harness modules are not light); they must not drift."""
    s = breaker_settings.setting
    assert s("harness.setup_retries.max_n") == 0
    assert s("harness.setup_retries.backoff_initial_s") == 5.0 and s("harness.setup_retries.backoff_max_s") == 60.0
    assert s("harness.reach_check.attempts_n") == agentenv_network.DEFAULT_TRIES
    assert s("harness.reach_check.timeout_s") == agentenv_network.DEFAULT_TIMEOUT_S
    assert s("harness.reach_check.interval_s") == agentenv_network.DEFAULT_SLEEP_S
    assert breaker_settings.enabled("harness.reach_check")
    assert s("harness.sandbox_caps.max_sandboxes") is None
    assert s("harness.sandbox_caps.agentenv_max_active_n") == agentenv_client.MAX_ACTIVE_SANDBOXES
    assert s("harness.sandbox_caps.agentenv_starts_per_minute_n") == agentenv_client.MAX_STARTS_PER_MINUTE
    assert s("harness.sandbox_caps.agentenv_starts_per_hour_n") is None
    assert s("harness.sandbox_caps.agentenv_create_timeout_s") == agentenv_client.DEFAULT_REQUEST_TIMEOUT_S
    assert s("harness.sandbox_caps.boat_max_active_n") == boat_client.MAX_ACTIVE_SANDBOXES
    assert s("harness.sandbox_caps.boat_starts_per_minute_n") == boat_client.MAX_STARTS_PER_MINUTE
    assert s("harness.sandbox_caps.boat_starts_per_hour_n") is None
    assert s("harness.sandbox_ttl.limit_s") is None
    assert s("harness.sandbox_ttl.margin_s") == trial.AGENTENV_TTL_MARGIN_S
    assert s("harness.sandbox_ttl.default_s") == agentenv_client.DEFAULT_TTL_S == HarborSettings.sandbox_ttl_default_s
    assert HarborSettings.sandbox_ttl_margin_s == trial.AGENTENV_TTL_MARGIN_S
    assert breaker_settings.source("harness.setup_retries.max_n") == "default"
    # the harness's own defaults come from the registry when not given
    harness = HarborHarness(lambda t, u: None)
    assert (harness.setup_retries, harness._retry_backoff_s, harness._retry_backoff_max_s) == (0, 5.0, 60.0)


def test_boat_sandbox_ttl_default_is_the_layer_default():
    pytest.importorskip("harbor")
    from yeto.rl.harness.harbor import agentenv_sandbox, boat_sandbox

    assert boat_sandbox.DEFAULT_TTL_S == breaker_settings.setting("harness.sandbox_ttl.default_s")
    assert inspect.signature(agentenv_sandbox.AgentenvSandbox.__init__).parameters["ttl_s"].default == (
        breaker_settings.setting("harness.sandbox_ttl.default_s"))


def test_env_aliases_still_work(monkeypatch):
    monkeypatch.setenv("HARBOR_SETUP_RETRIES", "6")
    monkeypatch.setenv("YETO_AGENTENV_SKIP_REACH_CHECK", "1")
    monkeypatch.setenv("HARBOR_MAX_SANDBOXES", "16")
    monkeypatch.setenv("YETO_AGENTENV_MAX_ACTIVE", "32")
    monkeypatch.setenv("YETO_AGENTENV_STARTS_PER_HOUR", "600")
    monkeypatch.setenv("YETO_AGENTENV_REQUEST_TIMEOUT_S", "600")
    monkeypatch.setenv("YETO_BOAT_STARTS_PER_MINUTE", "4")
    assert setup_retries_from_env() == 6
    assert breaker_settings.source("harness.setup_retries.max_n") == "env HARBOR_SETUP_RETRIES"
    assert not breaker_settings.enabled("harness.reach_check")
    assert capacity.SandboxLimits.from_env().caps == {"*": 16}
    quota = agentenv_client.AgentenvQuota.from_env()
    assert (quota.max_active, quota.starts_per_minute, quota.starts_per_hour) == (32, 60, 600)
    assert agentenv_client.check_in_flight_cap(32) == 32
    with pytest.raises(ValueError, match="active-sandbox cap 32"):
        agentenv_client.check_in_flight_cap(33)
    assert breaker_settings.setting("harness.sandbox_caps.agentenv_create_timeout_s") == 600.0
    assert boat_client.BoatQuota.from_env().starts_per_minute == 4


@pytest.mark.parametrize("raw", ["", "  "])
def test_an_empty_variable_is_unset(monkeypatch, raw):
    monkeypatch.setenv("HARBOR_SETUP_RETRIES", raw)
    monkeypatch.setenv("HARBOR_MAX_SANDBOXES", raw)
    monkeypatch.setenv("YETO_AGENTENV_STARTS_PER_HOUR", raw)
    assert setup_retries_from_env() == 0
    assert capacity.SandboxLimits.from_env().caps == {}
    assert agentenv_client.AgentenvQuota.from_env().starts_per_hour is None


def test_skip_reach_check_other_than_1_keeps_it_on(monkeypatch):
    monkeypatch.setenv("YETO_AGENTENV_SKIP_REACH_CHECK", "0")
    assert breaker_settings.enabled("harness.reach_check")


@pytest.mark.parametrize("name,raw", [("HARBOR_SETUP_RETRIES", "-1"), ("HARBOR_SETUP_RETRIES", "x"),
                                      ("HARBOR_MAX_SANDBOXES", "0"), ("HARBOR_MAX_SANDBOXES", "{bad"),
                                      ("YETO_AGENTENV_MAX_ACTIVE", "0"), ("YETO_AGENTENV_REACH_TRIES", "0")])
def test_a_bad_value_is_a_harness_config_error(monkeypatch, name, raw):
    monkeypatch.setenv(name, raw)
    with pytest.raises(HarnessConfigError, match=name):
        breaker_settings.resolved()


def test_parse_caps_keeps_its_contract():
    assert capacity.parse_caps("") == {}
    assert capacity.parse_caps("8") == {"*": 8}
    assert capacity.parse_caps('{"remote-docker": 4, "*": 16}') == {"remote-docker": 4, "*": 16}
    with pytest.raises(HarnessConfigError, match="HARBOR_MAX_SANDBOXES"):
        capacity.parse_caps('{"x": 0}')


def test_a_breakers_section_reaches_the_harness_through_the_env_write_back():
    section = {"harness": {"setup_retries": {"max_n": 6, "backoff_max_s": 30},
                           "reach_check": {"enabled": False},
                           "sandbox_caps": {"max_sandboxes": {"agentenv": 32}, "agentenv_max_active_n": 32},
                           "sandbox_ttl": {"limit_s": 5400}}}
    sources = breakers.Sources(env={}, section=section)
    rb = breakers.resolve(sources, layers=("harness",))
    env = breakers.materialize(rb, sources, surfaces=["env"]).env
    assert env["HARBOR_SETUP_RETRIES"] == "6" and env["YETO_AGENTENV_SKIP_REACH_CHECK"] == "1"
    assert json.loads(env["HARBOR_MAX_SANDBOXES"]) == {"agentenv": 32}
    # what the sampler process then reads
    s = breaker_settings.setting
    assert s("harness.setup_retries.max_n", env) == 6 and s("harness.setup_retries.backoff_max_s", env) == 30.0
    assert not breaker_settings.enabled("harness.reach_check", env)
    assert capacity.SandboxLimits.from_env(env).caps == {"agentenv": 32}
    assert s("harness.sandbox_caps.agentenv_max_active_n", env) == 32
    assert s("harness.sandbox_ttl.limit_s", env) == 5400


def test_section_and_env_that_disagree_are_refused():
    sources = breakers.Sources(env={"HARBOR_SETUP_RETRIES": "2"},
                               section={"harness": {"setup_retries": {"max_n": 6}}})
    with pytest.raises(breakers.BreakerConfigError, match="HARBOR_SETUP_RETRIES"):
        breakers.resolve(sources, layers=("harness",))


def test_caps_and_ttl_cannot_be_turned_off():
    for entry in ("sandbox_caps", "sandbox_ttl"):
        with pytest.raises(breakers.BreakerConfigError):
            breakers.parse_section({"harness": {entry: {"enabled": False}}})


def test_without_a_section_nothing_is_written_back():
    sources = breakers.Sources(env={"HARBOR_SETUP_RETRIES": "2"})
    out = breakers.materialize(breakers.resolve(sources, layers=("harness",)), sources, surfaces=["env"])
    assert out.env == {"HARBOR_SETUP_RETRIES": "2"} and not out.changes


def test_sandbox_ttl_per_provider(tmp_path):
    settings = HarborSettings(tasks_dir=tmp_path)
    assert sandbox_ttl_s("agentenv", 3900, settings) == 3900 + trial.AGENTENV_TTL_MARGIN_S  # as since #619
    assert sandbox_ttl_s("boat", 3900, settings) == 3 * 3600  # the boat provider's own default
    assert sandbox_ttl_s("docker", 3900, settings) is None  # Harbor's providers take no ttl_s
    explicit = HarborSettings(tasks_dir=tmp_path, sandbox_ttl_s=5400)
    assert sandbox_ttl_s("agentenv", 3900, explicit) == sandbox_ttl_s("boat", 3900, explicit) == 5400
    wider = HarborSettings(tasks_dir=tmp_path, sandbox_ttl_margin_s=1200)
    assert sandbox_ttl_s("agentenv", 3900, wider) == 5100


def test_settings_read_the_ttl_entry(monkeypatch, tmp_path):
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.delenv("HARBOR_ENV_KWARGS", raising=False)
    settings = HarborSettings.from_env()
    assert (settings.sandbox_ttl_s, settings.sandbox_ttl_margin_s, settings.sandbox_ttl_default_s) == (
        None, 900.0, 10800)
    monkeypatch.setenv("HARBOR_SANDBOX_TTL_S", "5400")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"ttl_s": 5400}')  # agrees
    assert HarborSettings.from_env().sandbox_ttl_s == 5400
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"ttl_s": 600}')
    with pytest.raises(HarnessConfigError, match="disagrees with harness.sandbox_ttl.limit_s"):
        HarborSettings.from_env()


# ----------------------------------------------------------------------------- trips
class _Trial:
    """Harbor ``Trial`` stand-in: each ``create`` pops the next exception name (None = success)."""

    script: list[str | None] = []

    @classmethod
    async def create(cls, config):
        return cls()

    async def run(self):
        name = _Trial.script.pop(0)
        exc = None if name is None else SimpleNamespace(exception_type=name, exception_message="x")
        return SimpleNamespace(exception_info=exc, verifier_result=None, agent_execution=None,
                               agent_result=None)


def _run(harness, script):
    _Trial.script = list(script)
    return asyncio.run(harness.run(SimpleNamespace(task_id="t"), "http://e"))


def _no_sleep():
    delays = []

    async def sleep(seconds):
        delays.append(seconds)

    return delays, sleep


def test_backoff_comes_from_the_entry(monkeypatch):
    monkeypatch.setenv("HARBOR_SETUP_RETRIES", "3")
    monkeypatch.setenv("HARBOR_SETUP_BACKOFF_S", "8")
    monkeypatch.setenv("HARBOR_SETUP_BACKOFF_MAX_S", "10")
    delays, sleep = _no_sleep()
    verdict = _run(HarborHarness(lambda t, u: {}, trial_cls=_Trial, sleep=sleep), ["E", "E", "E", None])
    assert verdict.agent_metrics["setup_attempts"] == 4
    assert 4.0 <= delays[0] <= 8.0 and all(5.0 <= d <= 10.0 for d in delays[1:])  # capped at backoff_max_s


def test_spent_retries_and_failed_reach_checks_become_round_events():
    _, sleep = _no_sleep()
    harness = HarborHarness(lambda t, u: {}, trial_cls=_Trial, setup_retries=2, sleep=sleep)
    _run(harness, ["NetworkError", "E", "NetworkError"])  # spent: 2 retries, 2 failed reach checks
    _run(harness, ["E", None])  # recovered: no trip
    _run(harness, ["NetworkError", "E", "E"])  # spent again
    events = breaker_settings.trip_events(harness, scope={"rollout_id": 3})
    assert all(validate_event(e) == [] for e in events)
    by_name = {e["name"]: e for e in events}
    retries = by_name["harness.setup_retries"]
    assert (retries["count"], retries["value"], retries["limit"], retries["state"]) == (2, 2.0, 2.0, "trip")
    assert retries["layer"] == "harness" and retries["kind"] == "retry_budget" and "stop_reason" not in retries
    assert retries["scope"] == {"rollout_id": 3} and retries["config_source"] == "default"
    reach = by_name["harness.reach_check"]
    assert (reach["count"], reach["action"], reach["limit"]) == (3, "drop", None)
    assert breaker_settings.trip_events(harness) == []  # taken


def test_no_retries_means_no_retry_trip_but_reach_checks_count():
    harness = HarborHarness(lambda t, u: {}, trial_cls=_Trial, setup_retries=0)
    _run(harness, ["E"])
    _run(harness, ["NetworkError"])
    assert harness.take_trips() == {"harness.reach_check": 1}
    assert breaker_settings.trip_events(SimpleNamespace()) == []  # a harness that counts no trips


def test_the_agentic_round_emits_harness_trips_before_its_round_event(tmp_path):
    pytest.importorskip("aiohttp")
    from yeto.rl.adapters.sampler_engines.fake import FakeEngine
    from yeto.rl.agentic import TokenTurn, TrialVerdict
    from yeto.rl.engine.agentic_round import ROUND_EVENT, AgenticRound
    from yeto.rl.engine.buffer_payload import GroupWriter
    from yeto.rl.engine.engine_sampler import PromptItem
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.engine.sampler_engine import Provenance, SamplingParams

    class Recorder:
        async def open(self, trajectory_id, sampling_params):
            return "http://unused"

        async def close(self, trajectory_id):
            return [TokenTurn(prompt_ids=(1, 2), completion_ids=(3, 4), completion_logprobs=(-0.1, -0.2),
                              policy_version=None)]

    class Harness:
        setup_retries = 4

        def __init__(self):
            self.trips = {"harness.setup_retries": 2}

        async def run(self, task, endpoint):
            return TrialVerdict(rewards={"reward": 1.0})

        def take_trips(self):
            trips, self.trips = self.trips, {}
            return trips

    engine = FakeEngine()
    engine.start(None, 0)
    events = []
    sampling = SamplingParams(max_tokens=8, n=2)
    task = {"instance_id": "fix-git", "agent_name": "codex", "trajectory_id": "row-traj"}
    rnd = AgenticRound(engine, GroupWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1), "s"),
                       recorder=Recorder(), harness=Harness(), max_in_flight=4, max_response_len=64, min_samples=1,
                       task_defaults={"provider": "docker"}, weight_switch="drain",
                       prompts=lambda r: [PromptItem("p", "t", metadata={"agent_task": dict(task)})],
                       sampling=sampling, provenance=Provenance.of(engine.identity, sampler_id="s", sampling=sampling),
                       emit=lambda n, **f: events.append((n, f)))
    try:
        rnd(0)
        rnd.flush()
    finally:
        rnd.close()
    names = [n for n, _ in events]
    first_round = names.index(ROUND_EVENT)
    (trip_at,) = [i for i, n in enumerate(names) if n == "rl_breaker"]
    assert trip_at < first_round
    trip_event = events[trip_at][1]
    assert trip_event["name"] == "harness.setup_retries" and trip_event["count"] == 2 and trip_event["limit"] == 4
