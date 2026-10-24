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


def test_the_default_interval_keeps_cuts_under_five_percent_of_the_wall_time():
    """S27 G-AT: no cut flag -> a timed interval; a 30B cut (about 400 s, s27-ao-a)
    is taken about every 8000 s, above the one-hour cap, instead of every round."""
    from yeto.rl.engine.cut_interval import DEFAULT_MAX_COST_SHARE, DEFAULT_MTBF_S

    clock = _Clock()
    default = cut_interval_for(every=None, auto_mtbf_s=None, max_interval_s=None)
    assert isinstance(default, AutoCutInterval)
    assert (default.mtbf_s, default.max_cost_share) == (DEFAULT_MTBF_S, DEFAULT_MAX_COST_SHARE) == (21600.0, 0.05)
    default.clock = clock
    assert default.due(1) is False  # S27 G-CD: the first cut waits DEFAULT_FIRST_CUT_S
    clock.now += 1800
    assert default.due(2) is True  # and measures the cost
    default.record_cut(2, cost_s=400.0)
    assert default.interval_s() == pytest.approx(8000.0)  # 400 / 0.05 > 3600 cap
    clock.now += 7999
    assert default.due(3) is False
    clock.now += 2
    assert default.due(4) is True
    small = AutoCutInterval(mtbf_s=DEFAULT_MTBF_S, max_cost_share=DEFAULT_MAX_COST_SHARE, clock=clock)
    small.record_cut(1, cost_s=5.0)  # a small model: the Young-Daly gap is the larger one
    assert small.interval_s() == pytest.approx(math.sqrt(2 * 5.0 * DEFAULT_MTBF_S))
    assert default.describe()["max_cost_share"] == 0.05
    assert "max_cost_share" not in AutoCutInterval(mtbf_s=100.0).describe()
    with pytest.raises(ValueError, match="cost share"):
        AutoCutInterval(mtbf_s=100.0, max_cost_share=0.0)


def test_cut_interval_flags_exclude_each_other():
    assert isinstance(cut_interval_for(every=3, auto_mtbf_s=None, max_interval_s=None), FixedCutInterval)
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
    assert [cost for _, cost in interval._costs] == [2.5, 1.5]
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
    pool = bridges.PoolAvgSync.__new__(bridges.PoolAvgSync)  # LoRA island pool (R-2)
    for sync in (strict, decoupled, pool):
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
    from yeto.rl.engine.full_parameter_sync import FullParameterSync

    full = FullParameterSync.__new__(FullParameterSync)
    full_args = SimpleNamespace(yeto_rl_resume={**config, "state_dir": str(tmp_path / "state-full")})
    assert resume_wiring_for(full_args, sync=full, environ={}) is not None
    assert full.cut_due(4) is False  # no checkpoint wired yet: no cut
    with pytest.raises(ValueError, match="takes no island round cuts"):
        resume_wiring_for(miles_args, sync=bridges.LocalOnlySync(2), environ={})
    pool = bridges.PoolAvgSync.__new__(bridges.PoolAvgSync)  # R-2: pool islands cut too
    pool_args = SimpleNamespace(yeto_rl_resume={**config, "state_dir": str(tmp_path / "pool-state")})
    assert resume_wiring_for(pool_args, sync=pool, environ={}).stop_after is None
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


def test_pool_launch_forwards_island_cut_store():
    """R-2: LoRA islands in an island pool write round cuts like strict islands."""
    from yeto.launch import rl_ports

    args = _cli("--rl-checkpoint-store", "/mnt/store", "--rl-cut-every", "4",
                "--rl-sync-quorum-theta", "0.75", "--rl-sync-max-base-lag", "2")
    assert rl_ports._check_resume_launch_flags(args) is False
    flags = rl_ports._ports_infra_flags(args)[1]
    assert " --rl-island-pool" in flags
    assert " --rl-island-cut-store /mnt/store/island-cuts" in flags and " --rl-cut-every 4" in flags


@pytest.mark.parametrize("extra, match", [
    (("--rl-checkpoint-store", "/s"), "needs --rl-single-island-no-sync, synced trainers, a cut interval"),
    (("--rl-checkpoint-store", "/s", "--rl-cut-every", "2", "--rl-stop-after-rounds", "3"),
     "does not resume from round cuts"),
    (("--rl-checkpoint-store", "/s", "--rl-cut-keep", "2"), "a cut interval"),
    (("--rl-checkpoint-store", "/s", "--rl-cut-every", "2", "--rl-cut-auto-mtbf-s", "60"),
     "exclude each other"),
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


def test_write_only_cuts_commit_inline_for_a_buffer_reader():
    from yeto.rl.adapters.miles.buffer_rollout import READER_FUNCTION_PATHS
    from yeto.rl.adapters.miles.entry import _commit_cuts_inline_for_buffer_reader

    reader = SimpleNamespace(background_commit=True)
    _commit_cuts_inline_for_buffer_reader(
        reader, miles_args=SimpleNamespace(rollout_function_path=READER_FUNCTION_PATHS[0]))
    assert reader.background_commit is False
    other = SimpleNamespace(background_commit=True)
    _commit_cuts_inline_for_buffer_reader(
        other, miles_args=SimpleNamespace(rollout_function_path="miles.rollout.sglang_rollout.generate_rollout"))
    assert other.background_commit is True
    _commit_cuts_inline_for_buffer_reader(None, miles_args=SimpleNamespace())


# ---------------------------------------------------------------- S27 G-CD: cut cost and commit time
def test_the_default_first_cut_waits_and_the_background_counts_in_the_cost():
    """s27-bb-b1 (Nano DP8, 473 GB cuts): the default interval cut at round 1 and
    then every ~52 rounds on the 144 s critical path alone, while each cut also
    took 529 s of store sync and a commit that timed out at 600 s."""
    from yeto.rl.engine.cut_interval import DEFAULT_FIRST_CUT_S

    clock = _Clock()
    default = cut_interval_for(every=None, auto_mtbf_s=None, max_interval_s=None)
    default.clock = clock
    assert default.describe()["first_cut_s"] == DEFAULT_FIRST_CUT_S == 1800.0
    assert [default.due(r) for r in (1, 2)] == [False, False]
    clock.now += DEFAULT_FIRST_CUT_S
    assert default.due(3) is True
    default.record_cut(3, cost_s=144.0)
    assert default.interval_s() == pytest.approx(144.0 / 0.05)
    default.add_cost(3, cost_s=1129.0)  # the background part, reported when committed
    default.add_cost(99, cost_s=1e6)  # not a measured cut: ignored
    assert default.cost_s == pytest.approx(1273.0)
    assert default.interval_s() == pytest.approx(1273.0 / 0.05)  # cuts stay under 5% of the wall
    clock.now += 2 * 144.0 / 0.05
    assert default.due(4) is False
    FixedCutInterval(2).add_cost(1, cost_s=5.0)  # rounds: nothing to size
    with pytest.raises(ValueError, match="first cut delay"):
        AutoCutInterval(mtbf_s=100.0, first_cut_s=-1.0)
    explicit = cut_interval_for(every=None, auto_mtbf_s=100.0, max_interval_s=None)
    assert explicit.due(1) is True and "first_cut_s" not in explicit.describe()


def test_a_joined_background_commit_reports_and_adds_its_seconds():
    import threading

    from yeto.rl.engine.round_cut import RoundCutCheckpoint

    interval = AutoCutInterval(mtbf_s=100.0, clock=_Clock())
    interval.record_cut(5, cost_s=3.0)
    cp = RoundCutCheckpoint.__new__(RoundCutCheckpoint)
    cp.interval = interval
    thread = threading.Thread(target=lambda: None)
    thread.start()
    cp._commit = (thread, {"cut_id": "r000005-x", "thread_s": 10.0,
                           "result": {"ok": True, "finish_s": 4.0, "commit_s": 6.0}})
    cp._commit_info = {"cut_id": "r000005-x", "rollout_id": 5, "collect_s": 2.0}
    events = []
    driver = SimpleNamespace(emit=lambda event, **f: events.append((event, f)))
    result = cp._join_running(driver)
    assert result["background_s"] == 12.0 and interval.cost_s == pytest.approx(15.0)
    assert events == [("rl_cut_committed", {
        "cut_id": "r000005-x", "rollout_id": 5, "ok": True, "finish_s": 4.0, "commit_s": 6.0,
        "pruned": None, "seq": None, "cut_bytes": None, "background_s": 12.0})]


def test_a_modal_commit_timeout_grows_with_the_bytes_of_the_cut(monkeypatch):
    import threading

    from yeto.cloud import modal_ckpt_store as mcs
    from yeto.rl.engine.resume import CheckpointStore, commit_budget

    store = mcs.ModalVolumeStore("/tmp/unused-store", "vol")
    timeouts = []
    monkeypatch.setenv("YETO_MODAL_PYTHON", "/bin/false")
    monkeypatch.setattr(store, "_runner_script", lambda code, *a, verb, timeout_s: timeouts.append(timeout_s))
    store.commit()
    with commit_budget(store, 473_000_000_000):
        store.commit()
        other = threading.Thread(target=store.commit)  # another thread: its own budget
        other.start()
        other.join()
        with commit_budget(store, 1000):
            store.commit()
        store.commit()
    store.commit()
    assert timeouts == [600.0, 4730.0, 600.0, 600.0, 4730.0, 600.0]
    with commit_budget(CheckpointStore("/tmp/unused-store"), 10**12):
        pass  # a store without a commit timeout: nothing to size
    with commit_budget(None, 10**12):
        pass
