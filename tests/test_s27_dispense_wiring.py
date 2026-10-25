"""S27 P2 task dispenser wiring: the launch plan, the syncer and island flags,
and the Miles data source hook (with a stand-in of Miles' data source; Miles
and Ray are not installed here)."""

from __future__ import annotations

import json
import shlex
import threading
import time
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_dispense import FP, ROOT, Syncer, crash, source, syncer_binary  # noqa: F401
from tools.eval.dispense_check import check
from yeto.rl.engine import task_dispense as td
from yeto.rl.engine.data_split import DATA_ROW_KEY, DISPENSE, STRIDE, plan_data_split

TWO_ISLANDS = "modal:1xh100,modal:1xh100"


def _flag(command: str, flag: str) -> str | None:
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


def _args(tmp_path, monkeypatch, gpu, *extra):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    args = _launcher_args("ports", tuple(extra), gpu=gpu)
    args.controller = "local"
    return args


# ------------------------------------------------------------------ the plan


def test_dispense_is_opt_in_and_refused_where_it_does_not_apply():
    assert plan_data_split(None, islands=[0, 1]).mode == STRIDE, "auto stays stride until the GPU check"
    plan = plan_data_split("dispense", islands=[0, 1])
    assert plan.mode == DISPENSE and plan.dispense and plan.slot(0) is None and plan.warning() is None
    assert "dispenser" in plan.describe()
    assert plan_data_split("dispense", islands=[0]).dispense, "one island may dispense (others join later)"
    # S27 P4: sample roles dispense to the samplers unless the launch shape cannot
    for kwargs in ({"tasks": True}, {"sample_roles": True, "sampler_blocker": "a pair has no syncer"},
                   {"external_learners": 1}):
        with pytest.raises(ValueError, match="dispense"):
            plan_data_split("dispense", islands=[0, 1], **kwargs)
    roles = plan_data_split("dispense", islands=[], sample_roles=True, samplers=[1, 2])
    assert roles.dispense and roles.samplers == (1, 2) and roles.islands == ()


def test_dry_run_gives_the_islands_the_dispenser_instead_of_slots(tmp_path, monkeypatch, capsys):
    from yeto import launcher
    from yeto.launch.syncer import syncer_command

    strict = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split", "dispense",
                   "--rl-data-passes", "2")
    with pytest.raises(ValueError, match="strict roster"):
        launcher._prepare_rl_args(strict)  # every island trains every round: no early end
    # an island pool: an island whose data ended may leave on its own
    args = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split", "dispense",
                 "--rl-data-passes", "2", "--rl-dispense-max-redo", "0",
                 "--rl-sync-island-lease-s", "60")
    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    commands = [i["learner_command"] for i in plan["island_requests"]]
    for command in commands:
        assert "--rl-data-dispense" in shlex.split(command)
        assert _flag(command, "--rl-data-slot") is None
        assert _flag(command, "--rl-data-position-store") is None
    assert plan["data_split"]["mode"] == DISPENSE
    assert [i["data_split"]["mode"] for i in plan["island_requests"]] == [DISPENSE, DISPENSE]
    assert "data split (dispense)" in capsys.readouterr().err
    syncer = syncer_command(args, 2)
    assert " --dispense " in syncer and "--dispense-passes 2" in syncer
    assert "--dispense-max-redo 0" in syncer and "--dispense-log ~/yeto-output/task-dispense.jsonl" in syncer


def test_dispense_options_need_dispense_and_island_flags_are_launcher_only(tmp_path, monkeypatch):
    from yeto import launcher

    with pytest.raises(ValueError, match="need --rl-data-split dispense"):
        launcher._prepare_rl_args(_args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split", "stride",
                                        "--rl-data-passes", "1"))
    with pytest.raises(ValueError, match="set by the launcher"):
        launcher._prepare_rl_args(_args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-dispense"))
    with pytest.raises(ValueError, match="round cuts"):
        launcher._prepare_rl_args(_args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split",
                                        "dispense", "--rl-cut-every", "2",
                                        "--rl-checkpoint-store", str(tmp_path / "store")))


def test_strict_and_stride_syncer_commands_carry_no_dispense_flag(tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.launch.syncer import syncer_command

    args = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split", "stride")
    launcher._prepare_rl_args(args)
    assert "--dispense" not in syncer_command(args, 2)


def test_auto_takes_the_dispenser_where_the_launch_can_use_it_else_the_row_slots(
        tmp_path, monkeypatch, capsys):
    """S27 G-DH: the default (auto) is the dispenser since its GPU checks; a launch
    the dispenser does not cover falls back to the row slots and says why."""
    from yeto import launcher
    from yeto.launch.syncer import syncer_command

    args = _args(tmp_path, monkeypatch, TWO_ISLANDS)
    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    assert plan["data_split"]["mode"] == DISPENSE and plan["data_split"]["requested"] == "auto"
    assert all("--rl-data-dispense" in i["learner_command"] for i in plan["island_requests"])
    assert " --dispense " in syncer_command(args, 2)
    assert "data split (dispense)" in capsys.readouterr().err
    cut = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-cut-every", "2",
                "--rl-checkpoint-store", str(tmp_path / "store"))
    launcher._prepare_rl_args(cut)
    plan = launcher.dry_run_plan(cut)
    assert plan["data_split"]["mode"] == STRIDE
    assert "no dispenser: --rl-cut-every" in plan["data_split"]["reason"]
    assert [_flag(i["learner_command"], "--rl-data-slot") for i in plan["island_requests"]] == ["0/2", "1/2"]
    assert "--dispense" not in syncer_command(cut, 2)
    assert "no dispenser: --rl-cut-every" in capsys.readouterr().err
    # one island: every row, no dispenser
    single = _args(tmp_path, monkeypatch, "modal:1xh100")
    launcher._prepare_rl_args(single)
    assert launcher.dry_run_plan(single)["data_split"]["mode"] != DISPENSE


# --------------------------------------------------------------- island side


def test_prompt_rows_are_all_numbered_for_the_dispenser(tmp_path):
    from yeto.rl.adapters.miles.island_entry import prepare_prompt_data

    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"q{i}", "label": i}) + "\n" for i in range(5)))
    out = prepare_prompt_data(str(data), None, tmp_path / "p.jsonl", number_rows=True)
    rows = [json.loads(line)["metadata"][DATA_ROW_KEY] for line in out.read_text().splitlines()]
    assert rows == [0, 1, 2, 3, 4]
    count, digest = td.file_fingerprint(out)
    assert count == 5 and len(digest) == 32


def test_miles_island_refuses_the_dispenser_where_it_does_not_belong():
    from yeto.rl.adapters.miles.island_entry import _island_dispense

    base = dict(rl_data_dispense=True, rl_data_slot=None, data_sha256=None, eval_only=False,
                rl_sample_role="both", rl_sampler_data_slot=None)
    assert _island_dispense(SimpleNamespace(**base))
    assert not _island_dispense(SimpleNamespace(**{**base, "rl_data_dispense": False}))
    for change in ({"rl_data_slot": "0/2"}, {"data_sha256": "ab"}, {"eval_only": True},
                   {"rl_sampler_data_slot": "0/2"}):
        with pytest.raises(ValueError):
            _island_dispense(SimpleNamespace(**{**base, **change}))


class FakeSample:
    """The parts of ``miles.utils.types.Sample`` the hook touches."""

    def __init__(self, row: int) -> None:
        self.metadata = {DATA_ROW_KEY: row}
        self.prompt = f"q{row}"
        self.group_index = None
        self.index = None


class FakeMilesSource:
    """Miles ``RolloutDataSourceWithBuffer`` as far as the hook needs it."""

    def __init__(self, rows: int, filtered: frozenset[int] = frozenset()) -> None:
        self.args = SimpleNamespace(n_samples_per_prompt=2, fully_async=False)
        samples = [FakeSample(i) for i in range(rows) if i not in filtered]
        self.dataset = SimpleNamespace(origin_samples=samples, samples=list(samples))
        self.sample_group_index = self.sample_index = self.sample_offset = self.epoch_id = 0
        self.buffer: list = []

    def get_samples(self, n):  # replaced by the hook
        raise AssertionError("the dispenser hook must replace get_samples")

    def _get_samples_from_buffer(self, n):
        out, self.buffer = self.buffer[:n], self.buffer[n:]
        return out


def test_miles_hook_draws_rows_by_position_and_reports_the_kept_groups(syncer_binary, tmp_path):
    from yeto.rl.adapters.miles import task_dispense as hook

    syncer = Syncer(syncer_binary, tmp_path)
    try:
        rows = 6
        fake = FakeMilesSource(rows, filtered=frozenset({2}))
        executor = SimpleNamespace(data_source=fake)
        config = {"address": f"127.0.0.1:{syncer.port}", "learner_id": 0, "rows": rows,
                  "fingerprint": FP.hex(), "seed": 5, "block": 3, "prefetch_blocks": 1}
        out = hook.install(executor, config, b"dispense-test-key")
        assert out["installed"] and out["filtered_rows"] == 1
        assert hook.install(executor, config, None)["installed"] is False, "idempotent"
        groups = fake.get_samples(3)
        # position 2 maps to row 2 (pass 0 is data order), which Miles filtered: skipped and dropped
        assert [g[0].metadata[DATA_ROW_KEY] for g in groups] == [0, 1, 3]
        assert [g[0].metadata[td.DISPENSE_POSITION_KEY] for g in groups] == [0, 1, 3]
        assert [s.index for g in groups for s in g] == list(range(6))
        assert [g[0].group_index for g in groups] == [0, 1, 2] and fake.sample_offset == 0
        # Miles keeps groups 0 and 3 (the hook runs first); then it puts the aborted
        # group 1 back into its buffer (sglang_rollout.generate_rollout: add_samples)
        counts = hook.on_trained([groups[0], groups[2]])
        assert counts == {"kept": 2, "dropped": 0, "rollout_id": 0}
        assert hook.current().pending_positions() == {0, 3}, "kept: wait for the round's commit"
        assert hook.commit_round(executor, 0) == {"rounds": [0], "trained": 2, "acked": True, "lost": 0}
        fake.buffer.append(groups[1])
        # next round: the buffered group first (not dropped), then new positions
        again = fake.get_samples(3)
        assert again[0] is groups[1]
        assert [g[0].metadata[td.DISPENSE_POSITION_KEY] for g in again[1:]] == [4, 5]
        assert hook.on_trained([again[0], again[2]]) == {"kept": 2, "dropped": 0, "rollout_id": 1}
        assert hook.commit_round(executor, 1)["trained"] == 2
        # group 4 was filtered for good: dropped at the next draw
        fake.get_samples(1)
        assert 4 not in hook.current().tasks.taken()
        hook.current().tasks.close()
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["trained"] == 4 and summary["dropped"] == 2  # rows 2 (filtered at load) and 4
    finally:
        hook._CURRENT = None
        syncer.stop()


def test_two_miles_islands_and_a_relaunch_share_the_rows(syncer_binary, tmp_path):
    """Two stand-in Miles islands draw rounds through the hook while island 1
    is relaunched (new process): no row trains twice, none is lost."""
    from yeto.rl.adapters.miles import task_dispense as hook

    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", lease_s=30)
    try:
        rows = 40

        def island(learner_id: int):
            fake = FakeMilesSource(rows)
            config = {"address": f"127.0.0.1:{syncer.port}", "learner_id": learner_id, "rows": rows,
                      "fingerprint": FP.hex(), "seed": 1, "block": 4, "prefetch_blocks": 1}
            hook._CURRENT = None
            hook.install(SimpleNamespace(data_source=fake), config, b"dispense-test-key")
            return fake, hook.current()

        a_src, a = island(0)
        b_src, b = island(1)
        trained: list[int] = []

        def round_of(fake, wrapped):
            groups = fake.get_samples(4)
            wrapped.on_trained(groups)
            wrapped.commit(wrapped._hooks)
            trained.extend(g[0].metadata[td.DISPENSE_POSITION_KEY] for g in groups)

        round_of(a_src, a)
        round_of(b_src, b)
        b_src.get_samples(4)  # island 1 draws a round, then its process dies before training
        crash(b.tasks)
        b_src, b = island(1)  # relaunched: supersedes, the drawn rows go back first
        lock = threading.Lock()

        def run(fake, wrapped):
            while True:
                try:
                    groups = fake.get_samples(4)
                except td.DispenseEnded:
                    return
                wrapped.on_trained(groups)
                wrapped.commit(wrapped._hooks)
                with lock:
                    trained.extend(g[0].metadata[td.DISPENSE_POSITION_KEY] for g in groups)
                time.sleep(0.01 if wrapped is a else 0.03)

        threads = [threading.Thread(target=run, args=pair) for pair in ((a_src, a), (b_src, b))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads)
        a.tasks.close()
        b.tasks.close()
        assert sorted(trained) == list(range(rows)), "every row trained exactly once"
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["islands"]["1"]["incarnations"] == 2
    finally:
        hook._CURRENT = None
        syncer.stop()


# ------------------------------------------- auto falls back to the row slots, byte for byte

ASYNC_FULL = ("--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
              "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4",
              "--rl-sync-full-parameter-rounds", "async")
UNSUPPORTED = {
    # path the dispenser does not cover -> extra launch args (2 islands each)
    # S27 G-EU: verl training islands take the dispenser now (tests/test_s27_verl_dispense.py)
    "miles_fully_async": ("--rl-max-policy-age", "1", "--rl-algorithm-spec",
                          str(ROOT / "tools/eval/efficiency/algo-eff-miles.json")),
    "island_round_cuts": ("--rl-cut-every", "2", "--rl-checkpoint-store", "/tmp/gdh-store"),
    "sharded_head": ASYNC_FULL + ("--syncer-shards", "2"),
    # G-DS catch-up: the store keeps only the data positions (no peer recovery)
    "catch_up_store": ASYNC_FULL + ("--rl-checkpoint-store", "/tmp/gdh-store", "--no-rl-peer-recovery"),
}


def _commands(tmp_path, monkeypatch, extra):
    from yeto import launcher
    from yeto.launch.syncer import syncer_command

    args = _args(tmp_path, monkeypatch, "modal:2xh100,modal:2xh100", *extra)
    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    islands = [i["learner_command"] for i in plan["island_requests"]]
    return plan["data_split"], islands, syncer_command(args, 2)


@pytest.mark.parametrize("path", sorted(UNSUPPORTED))
def test_auto_on_a_path_the_dispenser_does_not_cover_is_byte_identical_to_stride(
        tmp_path, monkeypatch, capsys, path):
    """S27 G-DH: before the default change, auto meant stride. On every path the
    dispenser does not cover, auto now plans stride again (no error) and names the
    reason, and the island and syncer commands are byte-identical to stride's."""
    split, islands, syncer = _commands(tmp_path, monkeypatch, UNSUPPORTED[path])
    err = capsys.readouterr().err
    stride_split, stride_islands, stride_syncer = _commands(
        tmp_path, monkeypatch, UNSUPPORTED[path] + ("--rl-data-split", "stride"))
    assert split["mode"] == STRIDE and split["requested"] == "auto"
    assert split["reason"].startswith("no dispenser: ")
    assert "data split (stride)" in err and "no dispenser: " in err
    assert islands == stride_islands
    assert syncer == stride_syncer and "--dispense" not in syncer
    assert all("--rl-data-dispense" not in c for c in islands)


def test_explicit_dispense_on_those_paths_is_still_refused(tmp_path, monkeypatch):
    from yeto import launcher

    # S27 G-EU: an explicit dispense covers verl training islands now
    # (tests/test_s27_verl_dispense.py); auto on verl still plans stride (above)
    for path in ("miles_fully_async", "island_round_cuts", "sharded_head"):
        args = _args(tmp_path, monkeypatch, "modal:2xh100,modal:2xh100",
                     *UNSUPPORTED[path], "--rl-data-split", "dispense")
        with pytest.raises(ValueError):
            launcher._prepare_rl_args(args)
