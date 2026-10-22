"""S27 R-2: round cuts of synced islands and the timed cut interval (no Ray, no GPU)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from yeto.rl.engine import bridges
from yeto.rl.engine.cut_interval import (
    DEFAULT_MAX_INTERVAL_S,
    AutoCutInterval,
    FixedCutInterval,
    cut_interval_for,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------- interval
def test_fixed_interval_matches_should_cut():
    interval = FixedCutInterval(3)
    assert [r for r in range(0, 10) if interval.due(r)] == [3, 6, 9]
    assert interval.due(4, final=True) and not interval.due(0, final=True)
    with pytest.raises(ValueError, match="--rl-cut-every must be >= 1"):
        FixedCutInterval(0)


def test_auto_interval_cuts_first_then_waits_young_daly_seconds():
    clock = _Clock()
    interval = AutoCutInterval(mtbf_s=21600.0, clock=clock)
    assert interval.due(0) is False  # no published policy before rollout 1
    assert interval.due(1) is True  # nothing measured yet: cut and measure
    interval.record_cut(1, cost_s=3.0)
    expected = math.sqrt(2 * 3.0 * 21600.0)  # 360 s
    assert interval.interval_s() == pytest.approx(expected)
    clock.now += expected - 1
    assert interval.due(2) is False
    clock.now += 2
    assert interval.due(2) is False  # decided once per safe point
    assert interval.due(3) is True
    interval.record_cut(3, cost_s=5.0)
    assert interval.every == 2  # rounds between the last two cuts
    assert interval.cost_s == pytest.approx(4.0)  # mean of the measured cuts
    assert interval.due(4, final=True) is True  # the last cut is always taken


def test_auto_interval_is_capped_and_floored():
    interval = AutoCutInterval(mtbf_s=604800.0, max_interval_s=600.0, clock=_Clock())
    interval.record_cut(1, cost_s=270.0)  # 4B full parameter, inline (E6)
    assert interval.interval_s() == 600.0
    floored = AutoCutInterval(mtbf_s=10.0, min_interval_s=30.0, clock=_Clock())
    floored.record_cut(1, cost_s=0.01)
    assert floored.interval_s() == 30.0


def test_cut_interval_flags_exclude_each_other():
    assert isinstance(cut_interval_for(every=None, auto_mtbf_s=None, max_interval_s=None), FixedCutInterval)
    auto = cut_interval_for(every=None, auto_mtbf_s=100.0, max_interval_s=None)
    assert isinstance(auto, AutoCutInterval) and auto.max_interval_s == DEFAULT_MAX_INTERVAL_S
    with pytest.raises(ValueError, match="exclude each other"):
        cut_interval_for(every=2, auto_mtbf_s=100.0, max_interval_s=None)
    with pytest.raises(ValueError, match="needs --rl-cut-auto-mtbf-s"):
        cut_interval_for(every=None, auto_mtbf_s=None, max_interval_s=60.0)
    with pytest.raises(ValueError, match="must be positive"):
        AutoCutInterval(mtbf_s=0.0)


def test_round_cut_save_hands_the_critical_path_cost_to_the_interval():
    from yeto.rl.engine.round_cut import RoundCutCheckpoint

    interval = AutoCutInterval(mtbf_s=100.0, clock=_Clock())
    cp = RoundCutCheckpoint.__new__(RoundCutCheckpoint)
    cp.interval = interval
    infos = iter([{"total_s": 0.5, "save_s": 2.0, "prepared_before_offload": True},
                  {"total_s": 1.5, "save_s": 1.0, "prepared_before_offload": False}, None])
    cp._save = lambda driver, **kw: next(infos)
    cp.save(None, rollout_id=1)
    cp.save(None, rollout_id=2)
    cp.save(None, rollout_id=3)  # nothing cut: nothing recorded
    assert list(interval._costs) == [2.5, 1.5]
    cp.every = 4  # a fixed interval again
    assert isinstance(cp.interval, FixedCutInterval) and cp.every == 4


# ---------------------------------------------------------------- sync hooks
class _Cuts:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def due(self, rollout_id, *, final=False):
        self.calls.append(("due", rollout_id, final))
        return True

    def save(self, driver, *, rollout_id, final=False, **kw):
        self.calls.append(("save", rollout_id, final))
        return {"cut_id": f"r{rollout_id}"}

    def prepare(self, driver, *, rollout_id, state, final=False):
        self.calls.append(("prepare", rollout_id, final))


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, name, **fields):
        self.events.append((name, fields))


def test_synced_syncs_take_periodic_round_cuts_only():
    strict = bridges.StrictAvgSync.__new__(bridges.StrictAvgSync)
    decoupled = bridges.DecoupledSync.__new__(bridges.DecoupledSync)
    for sync in (strict, decoupled):
        assert isinstance(sync, bridges.RoundCutHooks)
        assert sync.round_cuts is None and sync.cut_due(4) is False  # off by default
        driver = _Emitter()
        sync.at_safe_point(driver, rollout_id=4)
        assert driver.events == []
        sync.round_cuts = _Cuts()
        assert sync.cut_due(4) is True
        sync.prepare_cut(driver, rollout_id=4, state=None)
        sync.at_safe_point(driver, rollout_id=4)
        assert sync.round_cuts.calls == [("due", 4, False), ("prepare", 4, False), ("save", 4, False)]
        assert [name for name, _ in driver.events] == ["rl_round_cut", "rl_cut_saved"]
    assert not isinstance(bridges.ElasticAvgSync.__new__(bridges.ElasticAvgSync), bridges.RoundCutHooks)


def test_local_only_sync_still_cuts_at_the_end():
    sync = bridges.LocalOnlySync(5)
    sync.round_cuts = _Cuts()
    sync.cut_due(4)
    sync.cut_due(5)
    sync.stop_after = 3
    sync.cut_due(3)
    assert sync.round_cuts.calls == [("due", 4, False), ("due", 5, True), ("due", 3, True)]


def test_resume_wiring_write_only_needs_a_synced_sync(tmp_path):
    from yeto.rl.adapters.miles.entry import resume_wiring_for

    config = {"store": str(tmp_path / "store" / "island1"), "state_dir": str(tmp_path / "state"),
              "every": None, "auto_mtbf_s": 600.0, "keep": 2, "allow_config_change": False,
              "stop_after": None, "write_only": True, "runtime_fingerprint": "miles@x"}
    miles_args = SimpleNamespace(yeto_rl_resume=config)
    strict = bridges.StrictAvgSync.__new__(bridges.StrictAvgSync)
    wiring = resume_wiring_for(miles_args, sync=strict, environ={})
    assert isinstance(wiring.interval, AutoCutInterval) and wiring.stop_after is None
    with pytest.raises(ValueError, match="takes no island round cuts"):
        resume_wiring_for(miles_args, sync=bridges.LocalOnlySync(2), environ={})
    with pytest.raises(ValueError, match="takes no island round cuts"):
        resume_wiring_for(miles_args, sync=bridges.ElasticAvgSync.__new__(bridges.ElasticAvgSync), environ={})
    config.pop("write_only")
    with pytest.raises(ValueError, match="needs --rl-single-island-no-sync"):
        resume_wiring_for(miles_args, sync=strict, environ={})


# ---------------------------------------------------------------- launcher
def _cli(*extra):
    from test_rl_engine_selection import _cli as cli

    return cli(extra)


def test_synced_launch_forwards_island_cut_store():
    from yeto.launch import rl_ports

    args = _cli("--rl-checkpoint-store", "/mnt/store", "--rl-cut-every", "4", "--rl-cut-keep", "3")
    assert rl_ports._check_resume_launch_flags(args) is False
    flags = rl_ports._ports_infra_flags(args)[1]
    assert " --rl-island-cut-store /mnt/store/island-cuts --rl-cut-keep 3" in flags
    assert " --rl-cut-every 4" in flags and "--rl-resume-store" not in flags
    auto = _cli("--rl-checkpoint-store", "/mnt/store", "--rl-cut-auto-mtbf-s", "21600",
                "--rl-cut-max-interval-s", "1800")
    flags = rl_ports._ports_infra_flags(auto)[1]
    assert "--rl-island-cut-store /mnt/store/island-cuts" in flags
    assert " --rl-cut-auto-mtbf-s 21600.0 --rl-cut-max-interval-s 1800.0" in flags


@pytest.mark.parametrize("extra, match", [
    (("--rl-checkpoint-store", "/s"), "needs --rl-single-island-no-sync, synced trainers, or a cut interval"),
    (("--rl-checkpoint-store", "/s", "--rl-cut-every", "2", "--rl-stop-after-rounds", "3"),
     "does not resume from round cuts"),
    (("--rl-checkpoint-store", "/s", "--rl-cut-keep", "2"), "or a cut interval"),
    (("--rl-checkpoint-store", "/s", "--rl-cut-every", "2", "--rl-cut-auto-mtbf-s", "60"),
     "exclude each other"),
    (("--rl-checkpoint-store", "/s", "--rl-cut-every", "2", "--rl-island-scheduling", "elastic"),
     "need --rl-island-scheduling legacy"),
    (("--rl-cut-auto-mtbf-s", "60"), "needs --rl-checkpoint-store"),
])
def test_synced_launch_refusals(extra, match):
    from yeto.launch import rl_ports

    with pytest.raises(ValueError, match=match):
        rl_ports._check_resume_launch_flags(_cli(*extra))


# ---------------------------------------------------------------- island side
def _learner(*extra, no_sync=False):
    from test_rl_engine_selection import _learner_argv

    from yeto.rl.adapters.miles import island_entry as learner

    argv = _learner_argv(extra + (("--rl-single-island-no-sync",) if no_sync else ()))
    if no_sync:
        i = argv.index("--syncer")
        del argv[i:i + 2]
    return learner, learner.parse_args(argv)


def test_island_cut_store_is_per_island_and_write_only():
    learner, parsed = _learner("--rl-island-cut-store", "/mnt/store/island-cuts/", "--rl-cut-auto-mtbf-s", "600")
    miles_args, env = SimpleNamespace(), {}
    learner.apply_ports_infra_switches(parsed, miles_args, env)
    assert miles_args.yeto_rl_resume == {
        "store": "/mnt/store/island-cuts/island0", "state_dir": "~/yeto-rl/resume-state", "every": None,
        "auto_mtbf_s": 600.0, "keep": 2, "allow_config_change": False, "stop_after": None,
        "write_only": True}
    assert miles_args.yeto_rl_cut_auto == {"auto_mtbf_s": 600.0}
    assert miles_args.yeto_rl_elastic_metadata is True and env


@pytest.mark.parametrize("extra, no_sync", [
    (("--rl-island-cut-store", "/s"), True),
    (("--rl-island-cut-store", "/s", "--rl-stop-after-rounds", "2"), False),
    (("--rl-cut-auto-mtbf-s", "60"), False),
])
def test_island_cut_store_refusals(extra, no_sync):
    with pytest.raises(SystemExit):
        _learner(*extra, no_sync=no_sync)


def test_verl_refuses_the_timed_interval():
    from yeto.rl.adapters.verl.round_cuts import resume_config

    with pytest.raises(ValueError, match="not wired for verl"):
        resume_config(SimpleNamespace(rl_cut_auto_mtbf_s=60.0))
