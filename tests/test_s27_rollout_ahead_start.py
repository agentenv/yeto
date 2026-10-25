"""S27 rollout ahead with a serial warm-up (``--rl-rollout-ahead-start R``).

No Ray. Covered here (``yeto.rl.engine.rollout_ahead`` module doc):

* batches 0..R-1 are generated serially by their own round's policy (age 0);
  batch R is generated ahead during round R-1, and every later batch lags one
  version; the publication still waits for the batch in flight;
* unset (driver default 1) and ``R = 1`` give the plain rollout-ahead order;
* fail-closed refusals (launch and driver), the launcher forwards the flag to
  every island only when given (the island command is otherwise unchanged),
  and the island switches carry it to the driver.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_s27_rollout_ahead import _calls, _driver, _events, _launch

from yeto.rl.engine.driver import DriverError
from yeto.rl.engine.rollout_ahead import launch_problems

PLAIN_AHEAD_4 = [
    ("publish", 0), ("generate", 0), ("generate", 1), ("train", 0),
    ("generate_done", 1), ("publish", 1),
    ("generate", 2), ("train", 1), ("generate_done", 2), ("publish", 2),
    ("generate", 3), ("train", 2), ("generate_done", 3), ("publish", 3),
    ("train", 3), ("publish", 4),
]


def _trained(driver, engine):
    trained = []
    original = engine.trainer.train_step
    engine.trainer.train_step = lambda batch: trained.append(
        (batch.rollout_id, batch.policy_version)) or original(batch)
    driver.run()
    return trained


# -- driver --------------------------------------------------------------------------------


def test_warm_up_rounds_generate_their_own_batch_then_the_engines_run_ahead(tmp_path):
    driver, engine = _driver(tmp_path, rounds=5)
    driver.rollout_ahead_start = 3
    trained = _trained(driver, engine)

    assert trained == [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3)]
    assert _calls(engine) == [
        ("publish", 0), ("generate", 0), ("train", 0), ("publish", 1),
        ("generate", 1), ("train", 1), ("publish", 2),
        ("generate", 2), ("generate", 3), ("train", 2), ("generate_done", 3), ("publish", 3),
        ("generate", 4), ("train", 3), ("generate_done", 4), ("publish", 4),
        ("train", 4), ("publish", 5),  # known final round: nothing starts ahead
    ]
    events = _events(tmp_path)
    ahead = [(e["rollout_id"], e["policy_version"]) for e in events if e["event"] == "rl_rollout_ahead"]
    assert ahead == [(3, 2), (4, 3)]
    phases = [e["phase"] for e in events if e["event"] == "rl_driver_phase"]
    assert phases.count("generate") == 3 and phases.count("generate_wait") == 2
    labels = {e["policy_version"]: e["rl/policy_token"] for e in events if e["event"] == "rl_publication"}
    groups = {e["rollout_id"]: {g[1] for g in e["groups"]} for e in events if e["event"] == "rl_group_policies"}
    assert groups == {0: {labels[0]}, 1: {labels[1]}, 2: {labels[2]}, 3: {labels[2]}, 4: {labels[3]}}


@pytest.mark.parametrize("start", [None, 1])
def test_unset_or_one_is_the_plain_rollout_ahead(tmp_path, start):
    driver, engine = _driver(tmp_path, rounds=4)
    if start is not None:
        driver.rollout_ahead_start = start
    assert driver.rollout_ahead_start == 1
    trained = _trained(driver, engine)
    assert _calls(engine) == PLAIN_AHEAD_4
    assert trained == [(0, 0), (1, 0), (2, 1), (3, 2)]


def test_a_start_after_the_last_round_trains_every_round_serially(tmp_path):
    driver, engine = _driver(tmp_path, rounds=3)
    driver.rollout_ahead_start = 10
    trained = _trained(driver, engine)
    assert trained == [(0, 0), (1, 1), (2, 2)]
    assert _calls(engine) == [("publish", 0), ("generate", 0), ("train", 0), ("publish", 1),
                              ("generate", 1), ("train", 1), ("publish", 2),
                              ("generate", 2), ("train", 2), ("publish", 3)]
    assert '"rl_rollout_ahead"' not in (tmp_path / "e.jsonl").read_text()


def test_serial_rounds_ignore_the_start(tmp_path):
    driver, engine = _driver(tmp_path, rounds=2, limit=0, ahead=0)
    driver.rollout_ahead_start = 5
    driver.run()
    assert _calls(engine) == [("publish", 0), ("generate", 0), ("train", 0), ("publish", 1),
                              ("generate", 1), ("train", 1), ("publish", 2)]


@pytest.mark.parametrize("start", [0, -1, 1.5])
def test_the_driver_refuses_a_start_before_round_one(tmp_path, start):
    driver, engine = _driver(tmp_path)
    driver.rollout_ahead_start = start
    with pytest.raises(DriverError, match="rollout_ahead_start >= 1"):
        driver.run()
    assert not [c for c in engine.calls if c[0] in ("generate", "publish")]


# -- launch --------------------------------------------------------------------------------


def test_launch_problems_check_the_start():
    ok = dict(backend_max=1, max_policy_age=1, placements=["fixed-partition"], sync_supported=True)
    assert launch_problems(1, **ok, start=None) == []
    assert launch_problems(1, **ok, start=1) == []
    assert launch_problems(1, **ok, start=16) == []
    assert launch_problems(0, **ok, start=None) == []
    assert launch_problems(0, **ok, start=16) == ["--rl-rollout-ahead 1 (it sets the first batch generated ahead)"]
    assert launch_problems(1, **ok, start=0) == ["--rl-rollout-ahead-start >= 1 (round 0 always generates serially)"]


AHEAD = ("--rl-max-policy-age", "1", "--rl-rollout-ahead", "1")


def test_the_launcher_forwards_the_start_only_when_given(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    with_start = _launch(AHEAD + ("--rl-rollout-ahead-start", "16"))["island_requests"]
    without = _launch(AHEAD)["island_requests"]
    assert len(with_start) == len(without) == 2
    for a, b in zip(with_start, without, strict=True):
        assert " --rl-rollout-ahead 1 --rl-rollout-ahead-start 16" in a["learner_command"]
        assert "--rl-rollout-ahead-start" not in b["learner_command"]
        # the flag is the only change to the island command
        assert a["learner_command"].replace(" --rl-rollout-ahead-start 16", "") == b["learner_command"]
    assert "--rl-rollout-ahead-start" not in json.dumps(without)


@pytest.mark.parametrize(("extra", "message"), [
    (("--rl-rollout-ahead-start", "16"), r"--rl-rollout-ahead 1 \(it sets the first batch"),
    (AHEAD + ("--rl-rollout-ahead-start", "0"), r"--rl-rollout-ahead-start >= 1"),
])
def test_the_launcher_refuses_a_bad_start_before_any_machine(monkeypatch, extra, message):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    with pytest.raises(ValueError, match=message):
        _launch(extra)


# -- island ----------------------------------------------------------------------------------


def test_the_island_parser_and_switches_carry_the_start_to_the_driver(monkeypatch):
    from yeto.rl.adapters.miles.entry import wire_rollout_ahead
    from yeto.rl.adapters.miles.island_entry import apply_ports_infra_switches, build_parser

    parser = build_parser()
    assert parser.get_default("rl_rollout_ahead_start") is None
    monkeypatch.setattr("yeto.rl.adapters.miles.elastic_hook.apply_recommend_flags", lambda a, m: None)

    def run(**kw):
        base = dict(rl_observe_timeline=False, rl_heartbeat_interval=None, rl_resource_sample_interval=None)
        miles = SimpleNamespace()
        apply_ports_infra_switches(SimpleNamespace(**{**base, **kw}), miles, environ={})
        driver = SimpleNamespace(rollout_ahead=0, rollout_ahead_start=1)
        wire_rollout_ahead(driver, miles)
        return vars(miles), driver

    miles, driver = run(rl_rollout_ahead=1, rl_rollout_ahead_start=16)
    assert miles["yeto_rl_rollout_ahead_start"] == 16
    assert (driver.rollout_ahead, driver.rollout_ahead_start) == (1, 16)
    miles, driver = run(rl_rollout_ahead=1)
    assert "yeto_rl_rollout_ahead_start" not in miles
    assert (driver.rollout_ahead, driver.rollout_ahead_start) == (1, 1)
    miles, driver = run()
    assert miles == {} or "yeto_rl_rollout_ahead" not in miles
    assert (driver.rollout_ahead, driver.rollout_ahead_start) == (0, 1)
