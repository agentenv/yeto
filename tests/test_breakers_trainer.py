"""The trainer layer of the breaker registry (``yeto.breakers.layers.trainer``) and its wiring.

* the declarations load, their literal defaults equal the runtime constants, and the layer
  file stays import-light;
* without a ``breakers:`` section the resolved values are exactly what the trainer read from
  its env before (the NaN skip, the xhw guard, the buffer reader's limits, the release drain);
* invariants cannot be disabled; contradictory knobs are refused;
* every trip site writes one ``rl_breaker`` record and still raises (or drops) as before.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from yeto import breakers
from yeto.breakers.layers import trainer as layer
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine import nan_skip, trainer_breakers, xhw_guard
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.resume import CheckpointStore
from tests.buffer_rollout_procs import sample

ENTRIES = {b.name: b for b in layer.BREAKERS}


# ------------------------------------------------------------------ declarations
def test_the_trainer_layer_loads_and_its_invariants_cannot_be_disabled():
    reg = trainer_breakers.registry()
    assert {b.name for b in reg} == set(ENTRIES)
    invariants = {b.name for b in reg if b.kind == "invariant"}
    assert invariants == {"trainer.group_integrity", "trainer.sample_alignment", "trainer.compaction_round",
                          "trainer.nonfinite_reward", "trainer.gradient_flow", "trainer.policy_hash",
                          "trainer.strict_invariant"}
    assert all(not reg.get(name).can_disable and reg.get(name).native for name in invariants)
    assert "launch.stall" not in reg  # --rl-stall-timeout is the launch layer's
    assert set(trainer_breakers.STRICT_METRICS.values()) <= set(ENTRIES)


def test_literal_defaults_equal_the_runtime_constants():
    from yeto.rl.engine import trainer_remote

    assert (layer.SAMPLE_WAIT_ENV, layer.DEFAULT_SAMPLE_WAIT_S) == (br.WAIT_ENV, br.DEFAULT_WAIT_S)
    assert layer.POLICY_AGE_ENV == br.MAX_POLICY_AGE_ENV
    assert (layer.DAMAGED_RUN_ENV, layer.DEFAULT_DAMAGED_RUN) == (br.MAX_DAMAGED_RUN_ENV, br.DEFAULT_MAX_DAMAGED_RUN)
    assert layer.NAN_SKIP_ENV == nan_skip.ENV
    policy = nan_skip.NanSkipPolicy()
    assert layer.NAN_SKIP_DEFAULTS == {"max_consecutive": policy.max_consecutive,
                                       "max_fraction": policy.max_fraction, "min_rounds": policy.min_rounds}
    assert layer.XHW_ENV == xhw_guard.ENV
    assert set(layer.XHW_OFF) == set(xhw_guard.OFF_VALUES)
    assert layer.XHW_ACTIONS == xhw_guard.ACTIONS
    guard = xhw_guard.XhwGuard()
    assert layer.XHW_DEFAULTS == {
        **{f"{lvl}_{m}": getattr(getattr(guard, lvl), m) for lvl in ("warn", "act") for m in xhw_guard.METRICS},
        "window": guard.window, "min_rounds": guard.min_rounds, "quarantine_rounds": guard.quarantine_rounds,
        "weight": guard.weight, "max_cond": guard.max_cond, "confidence_z": guard.confidence_z}
    assert ENTRIES["trainer.xhw_guard"].default_action == guard.action
    assert (layer.RELEASE_DRAIN_ENV, layer.DEFAULT_RELEASE_DRAIN_S) == (
        trainer_remote.RELEASE_DRAIN_ENV, trainer_remote.DEFAULT_RELEASE_DRAIN_S)


def test_flag_aliases_exist_with_the_defaults_the_registry_assumes():
    """``--rl-sample-wait-s`` unset means the reader's default; ``--rl-max-policy-age``
    defaults to 0, the legacy spelling of ``trainer.policy_age`` off."""
    from yeto.cli import build_parser

    [launch] = [sub.choices["launch"] for sub in build_parser()._subparsers._group_actions]
    defaults = {a.dest: a.default for a in launch._actions}
    assert defaults["rl_sample_wait_s"] is None and defaults["rl_max_policy_age"] == 0
    flags = {a.path for b in layer.BREAKERS for a in b.aliases if a.surface == "trainer_flag"}
    assert flags == {"--rl-sample-wait-s", "--rl-max-policy-age"}


def test_the_layer_file_and_the_runtime_helper_stay_import_light():
    code = ("import sys; import yeto.rl.engine.trainer_breakers as t; t.resolved({}); "
            "heavy = [m for m in ('torch', 'numpy', 'ray', 'miles', 'modal', 'boto3') if m in sys.modules]; "
            "print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ------------------------------------------------------------------ resolution = today's readers
NAN_ENVS = [{}, {nan_skip.ENV: ""}, {nan_skip.ENV: "on"}, {nan_skip.ENV: "1"},
            {nan_skip.ENV: "max_consecutive=1,max_fraction=0.5,min_rounds=4"}, {nan_skip.ENV: "min_rounds=7"}]


@pytest.mark.parametrize("env", NAN_ENVS)
def test_nan_skip_resolves_as_the_env_reader(env):
    want = nan_skip.NanSkipPolicy.from_env(env)
    got = trainer_breakers.nan_skip_policy(trainer_breakers.resolved(env))
    assert (got is None) == (want is None)
    if want is not None:
        assert (got.max_consecutive, got.max_fraction, got.min_rounds) == (
            want.max_consecutive, want.max_fraction, want.min_rounds)


XHW_ENVS = [{}, {xhw_guard.ENV: "off"}, {xhw_guard.ENV: "0"}, {xhw_guard.ENV: ""}, {xhw_guard.ENV: "on"},
            {xhw_guard.ENV: "act_abs_diff=0.2,window=5,action=downweight,weight=0.25"},
            {xhw_guard.ENV: "warn_kl=0.02,act_clip=0.05,action=stop,quarantine_rounds=3,max_cond=10,confidence_z=1"}]


def _guard_fields(guard):
    return (guard.warn, guard.act, guard.window, guard.min_rounds, guard.action, guard.quarantine_rounds,
            guard.weight, guard.max_cond, guard.confidence_z)


@pytest.mark.parametrize("env", XHW_ENVS)
def test_xhw_guard_resolves_as_the_env_reader(env):
    want = xhw_guard.XhwGuard.from_env(env)
    got = trainer_breakers.xhw_guard(trainer_breakers.resolved(env))
    assert (got is None) == (want is None)
    if want is not None:
        assert _guard_fields(got) == _guard_fields(want)


@pytest.mark.parametrize("env, want", [
    ({br.MAX_POLICY_AGE_ENV: "2"}, (3600.0, 2, 8)),
    ({br.MAX_POLICY_AGE_ENV: "12", br.WAIT_ENV: "7200"}, (7200.0, 12, 8)),
    ({br.MAX_POLICY_AGE_ENV: "1", br.MAX_DAMAGED_RUN_ENV: "0"}, (3600.0, 1, 0)),
    ({br.MAX_POLICY_AGE_ENV: "1", br.MAX_DAMAGED_RUN_ENV: ""}, (3600.0, 1, 8)),
    ({br.MAX_POLICY_AGE_ENV: "1", br.MAX_DAMAGED_RUN_ENV: "3"}, (3600.0, 1, 3)),
    ({br.MAX_POLICY_AGE_ENV: "0"}, (3600.0, 0, 8)),  # refused by the reader, as before
])
def test_reader_limits_resolve_as_the_env_reader(env, want):
    limits = trainer_breakers.reader_limits(trainer_breakers.resolved(env))
    assert (limits.wait_s, limits.max_policy_age, limits.max_damaged_run) == want
    # what buffer_rollout_miles read before
    assert want == (float(env.get(br.WAIT_ENV, br.DEFAULT_WAIT_S)), int(env[br.MAX_POLICY_AGE_ENV]),
                    int(env.get(br.MAX_DAMAGED_RUN_ENV) or br.DEFAULT_MAX_DAMAGED_RUN))


@pytest.mark.parametrize("raw, want", [(None, 1800.0), ("", 1800.0), ("0.2", 0.2), ("600", 600.0)])
def test_release_drain_resolves_as_the_env_reader(raw, want):
    env = {} if raw is None else {layer.RELEASE_DRAIN_ENV: raw}
    assert trainer_breakers.release_drain_s(trainer_breakers.resolved(env)) == want


def test_the_flag_beats_the_env_the_launcher_overwrites():
    """``--rl-sample-wait-s`` replaces a user's ``YETO_SAMPLE_BUFFER_WAIT_S`` on the island,
    and ``--rl-max-policy-age`` (default 0) always writes ``YETO_SAMPLE_BUFFER_MAX_POLICY_AGE``."""
    env = {br.WAIT_ENV: "7200", br.MAX_POLICY_AGE_ENV: "5"}
    rb = breakers.resolve(breakers.Sources(argv=["--rl-sample-wait-s", "5400"], env=env),
                          registry=trainer_breakers.registry())
    assert rb.get("trainer.sample_wait.timeout_s") == 5400
    assert not rb.enabled("trainer.policy_age")  # no flag: 0, whatever the env said
    rb = breakers.resolve(breakers.Sources(argv=[], env={br.WAIT_ENV: "7200"}), registry=trainer_breakers.registry())
    assert rb.get("trainer.sample_wait.timeout_s") == 7200  # no flag: the env reaches the island


def test_trainer_env_pairs_in_the_argv_count():
    argv = ["--island-env", f"{nan_skip.ENV}=on", "--modal-env", f"{br.MAX_DAMAGED_RUN_ENV}=3"]
    rb = breakers.resolve(breakers.Sources(argv=argv, env={}), registry=trainer_breakers.registry())
    assert rb.get("trainer.nan.skip") is True and rb.get("trainer.buffer_damaged.max_run_n") == 3


def test_sources_name_the_legacy_knob():
    rb = trainer_breakers.resolved({br.WAIT_ENV: "7200", br.MAX_POLICY_AGE_ENV: "4"})
    assert rb.entry("trainer.sample_wait").sources["timeout_s"].location == f"env {br.WAIT_ENV}"
    assert rb.entry("trainer.policy_age").sources["max_versions"].kind == "legacy"
    assert rb.entry("trainer.release_drain").sources["timeout_s"].kind == "default"


# ------------------------------------------------------------------ refusals
@pytest.mark.parametrize("env", [{nan_skip.ENV: "bogus=1"}, {xhw_guard.ENV: "window=x"},
                                 {xhw_guard.ENV: "action=explode"}, {br.WAIT_ENV: "soon"}])
def test_a_malformed_trainer_knob_is_refused(env):
    with pytest.raises(breakers.BreakerConfigError):
        trainer_breakers.resolved(env)


def test_a_malformed_knob_refuses_only_the_caller_that_reads_it():
    """The driver reads the NaN skip and the xhw guard only: a bad buffer-reader knob on a
    trainer without a buffer stops nothing there, as before."""
    env = {br.WAIT_ENV: "soon", nan_skip.ENV: "on"}
    rb = trainer_breakers.resolved(env, names=(trainer_breakers.NAN, trainer_breakers.XHW_GUARD))
    assert trainer_breakers.nan_skip_policy(rb) is not None
    with pytest.raises(breakers.BreakerConfigError, match="sample_wait"):
        trainer_breakers.resolved(env, names=trainer_breakers.READER)


def test_section_and_legacy_knob_must_agree():
    section = {"trainer": {"sample_wait": {"timeout_s": 7200}}}
    agree = breakers.resolve(breakers.Sources(env={br.WAIT_ENV: "7200"}, section=section),
                             registry=trainer_breakers.registry())
    assert agree.get("trainer.sample_wait.timeout_s") == 7200
    with pytest.raises(breakers.BreakerConfigError, match="sample_wait"):
        breakers.resolve(breakers.Sources(env={br.WAIT_ENV: "3600"}, section=section),
                         registry=trainer_breakers.registry())


@pytest.mark.parametrize("name", ["group_integrity", "compaction_round", "strict_invariant"])
def test_an_invariant_cannot_be_turned_off_in_the_section(name):
    with pytest.raises(breakers.BreakerConfigError, match=name):
        breakers.parse_section({"trainer": {name: {"enabled": False}}}, registry=trainer_breakers.registry())


def test_the_section_turns_the_xhw_guard_on_through_its_env_knob():
    sources = breakers.Sources(env={}, section={"trainer": {"xhw_guard": {"enabled": True}}})
    rb = breakers.resolve(sources, registry=trainer_breakers.registry())
    assert rb.enabled("trainer.xhw_guard")
    out = breakers.materialize(rb, sources, surfaces=("trainer_env",))
    assert out.env[xhw_guard.ENV] == "on"
    assert xhw_guard.XhwGuard.from_env(out.env) is not None


# ------------------------------------------------------------------ trips
def _buffer(root):
    return FileSampleBuffer(CheckpointStore(root), batch_size=2)


def _group(g, version):
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version)]


def _breaker_events(events):
    return [fields for event, fields in events if event == "rl_breaker"]


def _echoed(capsys):
    """The ``rl_breaker`` records the reader's process echoed (``YETO_RL_EVENT`` lines)."""
    records = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()
               if line.startswith("YETO_RL_EVENT ")]
    return [r for r in records if r["event"] == "rl_breaker"]


def _reader(tmp_path, **kwargs):
    reader = br.BatchReader(_buffer(tmp_path), **kwargs)
    events = []
    reader.emit = lambda event, **fields: events.append((event, fields))
    return reader, events


def test_the_sample_wait_trip_is_reported_and_the_take_still_raises(tmp_path):
    breakers.clear_stop_reason()
    now = [0.0]

    def sleep(dt):
        now[0] += dt

    reader, events = _reader(tmp_path, max_policy_age=1, wait_s=5.0, poll_s=1.0,
                             clock=lambda: now[0], sleep=sleep)
    with pytest.raises(br.SampleBufferRolloutError, match="0/1 groups after 5 s"):
        reader.take(0, 1, current_version=1)
    [trip] = _breaker_events(events)
    assert (trip["name"], trip["state"], trip["action"], trip["limit"]) == ("trainer.sample_wait", "trip", "stop", 5.0)
    assert trip["value"] > 5.0 and trip["stop_reason"] == "breaker:trainer.sample_wait"
    assert trip["legacy_event"] == "SampleBufferRolloutError"
    assert breakers.current_stop_reason() == "breaker:trainer.sample_wait"
    breakers.clear_stop_reason()


def test_a_round_that_passes_groups_over_reports_one_record_per_guard(tmp_path):
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    for g, version in enumerate((1, 3, 2, 4)):
        writer.write(_group(g, version), group_key=f"g{g}")
    writer.write([sample(8, 4, 4)], group_key="g4-small")  # one trial: no in-group spread
    writer.write(_group(5, 5), group_key="g5")
    reader, events = _reader(tmp_path, max_policy_age=2, wait_s=0.0, poll_s=0.0, group_size=2)
    groups, report = reader.take(0, 3, current_version=5)  # floor 3: versions 1 and 2 are stale
    assert len(groups) == 3 and report.stale_groups == 2 and report.dropped_small_groups == 1
    by_name = {e["name"]: e for e in _breaker_events(events)}
    assert set(by_name) == {"trainer.policy_age", "trainer.group_integrity"}
    age = by_name["trainer.policy_age"]
    assert (age["action"], age["count"], age["limit"]) == ("drop", 2, 2) and "stop_reason" not in age
    assert by_name["trainer.group_integrity"]["count"] == 1
    assert by_name["trainer.group_integrity"]["kind"] == "invariant"


def test_a_clean_round_reports_nothing(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g in range(2):
        writer.write(_group(g, 1), group_key=f"g{g}")
    reader, events = _reader(tmp_path, max_policy_age=1, wait_s=0.0, poll_s=0.0)
    reader.take(0, 2, current_version=1)
    assert _breaker_events(events) == []


def test_unconvertible_takes_stop_as_a_sample_alignment_trip(capsys):
    error = br.unconvertible_round(7, 8, "buffer position 3: bad routing row")
    assert isinstance(error, br.SampleBufferRolloutError)
    assert str(error) == ("rollout 7: 8 takes in a row with no group that converts to Miles samples: "
                          "buffer position 3: bad routing row")
    [trip] = _echoed(capsys)
    assert (trip["name"], trip["action"], trip["stop_reason"]) == (
        "trainer.sample_alignment", "stop", "breaker:trainer.sample_alignment")
    breakers.clear_stop_reason()


def test_a_refused_compaction_round_is_reported(capsys):
    args = SimpleNamespace(seq_length=None)  # no --use-dynamic-batch-size: Miles would split the trials
    group = [{**sample(0, 0, 1), "metadata": {br.COMPACTION_KEY: {"segment": 0}}}]
    with pytest.raises(br.SampleBufferRolloutError, match="compacted trials"):
        br.check_segmented_round(args, [(0, group)])
    assert [e["name"] for e in _echoed(capsys)] == ["trainer.compaction_round"]
    breakers.clear_stop_reason()


def test_a_strict_failure_trips_its_entry_before_the_legacy_event(tmp_path, monkeypatch):
    import math

    from tests.test_rl_algorithm_gradient import _events, _run
    from yeto.rl.core import StrictRlInvariantError
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.driver import TrainStepMetrics

    monkeypatch.setenv(nan_skip.ENV, "max_consecutive=1,max_fraction=1,min_rounds=1")
    engine, driver = _run(tmp_path, AlgorithmSpec())
    calls = iter([math.inf, math.nan, 1.0])
    engine.trainer.step_metrics = lambda: TrainStepMetrics(grad_norm=next(calls), applied_lrs=(1e-5,))
    with pytest.raises(StrictRlInvariantError):
        driver.run()
    names = [e["event"] for e in _events(tmp_path)]
    at = names.index("rl_breaker")
    assert names[at + 1] == "rl_strict_failure"
    trip = _events(tmp_path)[at]
    assert (trip["name"], trip["stop_reason"]) == ("trainer.nan", "breaker:trainer.nan")
    assert trip["detail"].startswith("nan_skip_breaker:")
    breakers.clear_stop_reason()


@pytest.mark.parametrize("metric, name", [
    ("nonfinite_grad_norm", "trainer.nan"), ("xhw_guard_stop", "trainer.xhw_guard"),
    ("policy_hash_mismatch_after_apply", "trainer.policy_hash"),
    ("zero_grad_norm_with_nonzero_advantages", "trainer.gradient_flow"),
    ("layout_hash_mismatch", "trainer.strict_invariant"),
    ("session_contract_mismatch", "trainer.strict_invariant"),
    ("compat_group_mismatch", "trainer.strict_invariant")])
def test_strict_metrics_map_to_entries(metric, name):
    assert trainer_breakers.strict_breaker(metric) == name


def test_a_hung_release_upload_trips_the_release_drain():
    from yeto.rl.engine.trainer_remote import IslandTrainerAgent

    hang = threading.Event()
    driver = SimpleNamespace(publisher=SimpleNamespace(drain_releases=lambda: hang.wait(30)))
    events = []
    agent = IslandTrainerAgent(driver, commands=None, incarnation=0, resume_from=None,
                               emit=lambda event, **fields: events.append((event, fields)), release_drain_s=0.05)
    handle = SimpleNamespace(error=None)
    try:
        agent._drain_releases(handle)
    finally:
        hang.set()
    assert "still running" in str(handle.error)
    [trip] = _breaker_events(events)
    assert (trip["name"], trip["limit"], trip["stop_reason"]) == (
        "trainer.release_drain", 0.05, "breaker:trainer.release_drain")
    breakers.clear_stop_reason()


def test_the_commanded_island_reads_its_drain_bound_through_the_registry(monkeypatch):
    from yeto.rl.engine.trainer_remote import IslandTrainerAgent

    monkeypatch.setenv(layer.RELEASE_DRAIN_ENV, "0.2")
    agent = IslandTrainerAgent(SimpleNamespace(), commands=None, incarnation=0, resume_from=None,
                               emit=lambda *a, **k: None)
    assert agent.release_drain_s == 0.2
