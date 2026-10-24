"""S27 task assignment P0/P1: which island reads which rows of --data. CPU, no Ray.

* P1: same-card training islands split --data by row number (``data_split``):
  two and three islands read disjoint rows that together cover the data, in
  the launcher plan, in the Miles and verl prompt files and on the island
  command line; ``--rl-island-tasks`` and ``--rl-data-split shared`` override.
* P0: a relaunched full-parameter island continues at its next row
  (``data_position``) instead of its first one; a LoRA pool island that joins
  late skips whole rounds of its own slot; several islands that read the same
  rows get a loud warning.
* Passes: pass 0 in data order, every later pass reshuffled (Miles adapter).
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.engine.bridges import LocalOnlySync, whole_round_restart_cursor
from yeto.rl.engine.data_position import DataPosition, DataPositionError, DataPositionLog, data_position_log
from yeto.rl.engine.data_split import (
    DATA_ROW_KEY,
    ROLES,
    SHARED,
    SINGLE,
    STRIDE,
    TASKS,
    DataSlot,
    pass_order,
    plan_data_split,
)
from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.algorithm import AlgorithmSpec

TWO_ISLANDS = "modal:1xh100,modal:1xh100"
THREE_ISLANDS = "modal:1xh100,modal:1xh100,modal:1xh100"
NAME = "base_model.model.layer.lora_A.weight"


def _write_jsonl(path: Path, count: int) -> Path:
    path.write_text("".join(json.dumps({"prompt": f"q{i}", "label": str(i),
                                        "metadata": {"task_id": f"t{i}"}}) + "\n"
                            for i in range(count)))
    return path


def _flag(command: str, flag: str) -> str | None:
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


# ------------------------------------------------------------------ the split rule
@pytest.mark.parametrize("islands", [2, 3])
def test_slots_are_disjoint_cover_the_data_and_keep_the_global_round_order(islands):
    rows, per_round = 23, 2
    slots = [DataSlot(k, islands).numbered(range(rows)) for k in range(islands)]
    numbers = [n for slot in slots for n, _row in slot]
    assert sorted(numbers) == list(range(rows)) and len(set(numbers)) == rows
    # round r of all islands reads what islands * per_round consecutive rows of one
    # island would read (no shuffle in pass 0)
    for r in range(3):
        drawn = sorted(n for slot in slots for n, _row in slot[r * per_round:(r + 1) * per_round])
        assert drawn == list(range(r * per_round * islands, (r + 1) * per_round * islands))


def test_plan_modes_and_refusals():
    assert plan_data_split(None, islands=[0, 1]).mode == STRIDE
    assert plan_data_split("stride", islands=[0, 1, 2]).slot(2) == DataSlot(2, 3)
    assert plan_data_split("auto", islands=[0]).mode == SINGLE
    assert plan_data_split("auto", islands=[0, 1]).slot(5) is None
    shared = plan_data_split("shared", islands=[0, 1])
    assert shared.mode == SHARED and shared.slot(0) is None and "same --data rows" in shared.warning()
    assert plan_data_split("auto", islands=[0, 1]).warning() is None
    assert plan_data_split("auto", islands=[0, 1], tasks=True).mode == TASKS
    assert plan_data_split("auto", islands=[], sample_roles=True).mode == ROLES
    external = plan_data_split("auto", islands=[0, 1], external_learners=1)
    assert external.mode == SHARED and "external-learners" in external.warning()
    for kwargs, message in (({"tasks": True}, "exclude each other"),
                            ({"sample_roles": True}, "sampler islands"),
                            ({"external_learners": 1}, "external-learners")):
        with pytest.raises(ValueError, match=message):
            plan_data_split("stride", islands=[0, 1], **kwargs)
    with pytest.raises(ValueError, match="expected one of"):
        plan_data_split("random", islands=[0, 1])


def test_pass_order_keeps_pass_zero_and_reshuffles_each_later_pass():
    assert pass_order(10, seed=7, data_pass=0) == list(range(10))
    first, second = (pass_order(10, seed=7, data_pass=p) for p in (1, 2))
    assert sorted(first) == sorted(second) == list(range(10))
    assert first != list(range(10)) and first != second
    assert pass_order(10, seed=7, data_pass=1) == first  # deterministic
    assert pass_order(10, seed=8, data_pass=1) != first


# ------------------------------------------------------------------ island prompt files
@pytest.mark.parametrize("islands", [2, 3])
def test_miles_prompt_files_of_the_slots_cover_the_data_once(tmp_path, islands):
    from yeto.rl.adapters.miles.island_entry import prepare_prompt_data

    source = _write_jsonl(tmp_path / "data.jsonl", 11)
    seen = []
    for k in range(islands):
        out = prepare_prompt_data(str(source), None, tmp_path / f"p{k}.jsonl",
                                  slot=DataSlot(k, islands))
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        numbers = [row["metadata"][DATA_ROW_KEY] for row in rows]
        assert numbers == list(range(k, 11, islands))  # data order
        assert [row["metadata"]["task_id"] for row in rows] == [f"t{n}" for n in numbers]
        seen += numbers
    assert sorted(seen) == list(range(11))
    whole = prepare_prompt_data(str(source), None, tmp_path / "all.jsonl", slot=DataSlot())
    assert all(DATA_ROW_KEY not in json.loads(line)["metadata"] for line in whole.read_text().splitlines())
    with pytest.raises(ValueError, match="slot 3/4 of the RL prompt dataset is empty"):
        prepare_prompt_data(str(_write_jsonl(tmp_path / "small.jsonl", 3)), None,
                            tmp_path / "x.jsonl", slot=DataSlot(3, 4))


def test_verl_train_file_keeps_the_slot_and_the_whole_validation_rows(tmp_path, monkeypatch):
    import pandas as pd

    from yeto.rl.adapters.verl import island_entry

    for split, count in (("train", 9), ("test", 4)):
        pd.DataFrame({"messages": [[{"role": "user", "content": f"{split}{i}"}] for i in range(count)],
                      "label": [str(i) for i in range(count)]}).to_parquet(tmp_path / f"src-{split}.parquet")
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda data, name, **_: str(tmp_path / f"src-{name}"))
    train, val = island_entry.prepare_data("org/ds", None, tmp_path / "out", slot=DataSlot(1, 3))
    rows = pd.read_parquet(train)
    assert [r[DATA_ROW_KEY] for r in rows["extra_info"]] == [1, 4, 7]
    assert [p[0]["content"] for p in rows["prompt"]] == ["train1", "train4", "train7"]
    assert len(pd.read_parquet(val)) == 4
    assert island_entry.island_data_slot(SimpleNamespace(rl_data_slot="1/3", rl_sample_role=None)) == DataSlot(1, 3)
    with pytest.raises(SystemExit, match="generates and trains"):
        island_entry.island_data_slot(SimpleNamespace(rl_data_slot="1/3", rl_sample_role="sampler"))


def test_miles_island_refuses_a_slot_where_it_does_not_belong():
    from yeto.rl.adapters.miles.island_entry import _island_data_slot

    base = dict(rl_data_slot="0/2", data_sha256=None, rl_sample_role=None, rl_sampler_data_slot=None,
                eval_only=False)
    assert _island_data_slot(SimpleNamespace(**base)) == DataSlot(0, 2)
    assert _island_data_slot(SimpleNamespace(**{**base, "rl_data_slot": None})) is None
    for change, message in (({"data_sha256": "a" * 64}, "exclude each other"),
                            ({"rl_sample_role": "sampler"}, "generates and trains"),
                            ({"rl_sampler_data_slot": "0/2"}, "generates and trains"),
                            ({"eval_only": True}, "eval-only")):
        with pytest.raises(ValueError, match=message):
            _island_data_slot(SimpleNamespace(**{**base, **change}))


# ------------------------------------------------------------------ launcher
def _args(tmp_path, monkeypatch, gpu, *extra):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    args = _launcher_args("ports", tuple(extra), gpu=gpu)
    args.controller = "local"
    return args


@pytest.mark.parametrize("gpu,count", [(TWO_ISLANDS, 2), (THREE_ISLANDS, 3)])
def test_dry_run_gives_each_island_its_row_slot(tmp_path, monkeypatch, capsys, gpu, count):
    from yeto import launcher

    args = _args(tmp_path, monkeypatch, gpu)
    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    commands = [i["learner_command"] for i in plan["island_requests"]]
    assert [_flag(c, "--rl-data-slot") for c in commands] == [f"{k}/{count}" for k in range(count)]
    assert {_flag(c, "--data") for c in commands} == {"org/data"}  # one shared file, split on the island
    assert [i["data_split"] for i in plan["island_requests"]] == [
        {"mode": STRIDE, "data_slot": f"{k}/{count}"} for k in range(count)]
    assert plan["data_split"]["mode"] == STRIDE and plan["data_split"]["warning"] is None
    assert f"i % {count} == k" in plan["data_split"]["rule"]
    assert "data split (stride)" in capsys.readouterr().err
    from yeto.launch.manifest import _data_split_manifest

    assert _data_split_manifest(args)["data_split"]["islands"] == list(range(count))


def test_shared_split_keeps_every_row_on_every_island_and_warns(tmp_path, monkeypatch, capsys):
    from yeto import launcher

    args = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split", "shared")
    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    assert all("--rl-data-slot" not in i["learner_command"] for i in plan["island_requests"])
    assert "same --data rows" in plan["data_split"]["warning"]
    assert "WARNING 2 training islands read the same --data rows" in capsys.readouterr().err


def test_one_island_and_sampler_islands_get_no_training_slot(tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.launch.island_tasks import island_task_data
    from yeto.rl.engine import sample_roles

    one = _args(tmp_path, monkeypatch, "modal:1xh100")
    launcher._prepare_rl_args(one)
    assert "--rl-data-slot" not in launcher.dry_run_plan(one)["island_requests"][0]["learner_command"]
    from yeto.launch.manifest import _data_split_manifest

    assert _data_split_manifest(one) == {}
    # P0-2: a trainer island with three sampler islands (the disaggregated shape): every
    # sampler reads its own DataSlot (--rl-sampler-data-slot), nobody a training slot
    pair = SimpleNamespace(rl_island_sample_role=["0:sampler", "1:sampler", "2:sampler", "3:trainer"],
                           rl_sampler_max_backlog_rounds=None, rl_sample_wait_s=None, total_steps=9,
                           rl_sampler_spot=False, gpu=",".join(["modal:1xh100"] * 4), rl_island_tasks=None,
                           rl_data_split="auto", data="org/data", external_learners=0)
    slots = []
    for m in range(4):
        iargs = sample_roles.island_args(pair, m, 4)
        _data, flags, _mounts = island_task_data(iargs, m)
        assert "--rl-data-slot" not in flags
        slots.append(iargs.rl_sampler_data_slot)
    assert slots == ["0/3", "1/3", "2/3", None]
    with pytest.raises(ValueError, match="sampler islands always read"):
        island_task_data(SimpleNamespace(**{**vars(pair), "rl_data_split": "stride"}), 3)


def test_island_tasks_override_and_overlapping_lists_warn(tmp_path, monkeypatch, capsys):
    from yeto import launcher

    catalog = _write_jsonl(tmp_path / "tasks.jsonl", 6)
    args = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-island-tasks", "l0=t0,t1,t2;l1=t2,t3")
    args.data = str(catalog)
    launcher._prepare_rl_args(args)
    err = capsys.readouterr().err
    assert "islands 0 and 1: 1 shared tasks" in err and "data split (tasks)" in err
    commands = [i["learner_command"] for i in launcher.dry_run_plan(args)["island_requests"]]
    assert all("--rl-data-slot" not in c and "--data-sha256" in c for c in commands)
    both = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-island-tasks", "l0=t0;l1=t1",
                 "--rl-data-split", "stride")
    both.data = str(catalog)
    with pytest.raises(ValueError, match="exclude each other"):
        launcher._prepare_rl_args(both)


def test_the_island_flags_are_not_launch_options(tmp_path, monkeypatch):
    from yeto import launcher

    for flag in (("--rl-data-slot", "0/2"), ("--rl-data-position-store", "/s")):
        args = _args(tmp_path, monkeypatch, TWO_ISLANDS, *flag)
        with pytest.raises(ValueError, match="set by the launcher"):
            launcher._prepare_rl_args(args)


def test_data_position_store_reaches_synced_full_parameter_islands_only():
    from yeto.launch.rl_ports import data_positions_kept

    full = dict(rl_checkpoint_store="modal-volume://v/run", rl_sync_preset="full-parameter",
                training_mode="rl", rl_elastic=False, rl_single_island_no_sync=False, rl_sample_role=None)
    assert data_positions_kept(SimpleNamespace(**full))
    for change in ({"rl_checkpoint_store": None}, {"rl_sync_preset": "decoupled-avg"},
                   {"rl_single_island_no_sync": True}, {"rl_sample_role": "trainer"},
                   {"rl_elastic": True}):
        assert not data_positions_kept(SimpleNamespace(**{**full, **change}))


# ------------------------------------------------------------------ data position log
def test_data_position_log_reads_the_last_record_and_refuses_another_slot(tmp_path):
    commits = []
    log = DataPositionLog(learner_id=1, slot="1/2", root=tmp_path, commit=lambda: commits.append(1))
    assert log.last() is None
    log.start({"sample_offset": 0})
    log.record(0, {"sample_offset": 4})
    log.record(1, {"sample_offset": 8})
    records = tmp_path / "data-positions" / "island1"
    assert sorted(p.name for p in records.iterdir()) == [f"{n:010d}.json" for n in range(3)]
    (records / "0000000003.json").write_text('{"round": 2, "cur')  # cut short by a stop
    assert log.close() and commits
    assert DataPositionLog(learner_id=1, slot="1/2", root=tmp_path).last() == \
        DataPosition(1, {"sample_offset": 8}, "1/2")
    other = DataPositionLog(learner_id=1, slot="1/3", root=tmp_path)
    with pytest.raises(DataPositionError, match="data slot 1/2"):
        other.last()
    assert DataPositionLog(learner_id=0, slot="0/2", root=tmp_path).last() is None  # one file per island


def test_data_position_log_in_an_object_store():
    class Objects:
        def __init__(self):
            self.data = {}

        def list(self, prefix):
            return [k for k in self.data if k.startswith(prefix)]

        def read(self, key):
            return self.data[key]

        def write(self, key, value):
            self.data[key] = value

    objects = Objects()
    log = DataPositionLog(learner_id=0, slot=None, objects=objects)
    log.start(None)
    for r in range(11):
        log.record(r, {"sample_offset": 2 * (r + 1)})
    assert sorted(objects.data)[0] == "data-positions/island0/0000000000.json"
    again = DataPositionLog(learner_id=0, slot=None, objects=objects)
    assert again.last() == DataPosition(10, {"sample_offset": 22}, None)


def test_data_position_log_for_a_directory_store(tmp_path):
    log = data_position_log(str(tmp_path), learner_id=2, slot="2/3", environ={})
    log.record(0, {"sample_offset": 1})
    assert log.close() and log.last().cursor == {"sample_offset": 1}


# ------------------------------------------------------------------ relaunch: driver
class SlotRowPool:
    """The fake rollout over one island's rows: each round draws the next
    ``groups`` rows of the slot; the cursor is Miles-shaped (``sample_offset``)."""

    def __init__(self, inner, rows: list[int], *, die_at: int | None = None) -> None:
        self._inner, self._rows, self._die_at = inner, rows, die_at
        self.offset = 0
        self.drawn: list[list[int]] = []
        self.seeks: list[dict[str, int]] = []

    def generate(self, rollout_id, *, expected_policy_version=None):
        if self._die_at is not None and rollout_id == self._die_at:
            raise RuntimeError("container lost (test)")
        batch = self._inner.generate(rollout_id, expected_policy_version=expected_policy_version)
        take = len(batch.groups)
        self.drawn.append([self._rows[(self.offset + j) % len(self._rows)] for j in range(take)])
        self.offset += take
        return batch

    def data_cursor(self):
        return {"sample_offset": self.offset}

    def seek_data_cursor(self, cursor):
        self.seeks.append(dict(cursor))
        self.offset = int(cursor["sample_offset"])
        return self.data_cursor()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class LocalRoundsSync(LocalOnlySync):
    """What the driver sees of a full-parameter session: every process starts its
    rounds at 0, and the session says whether it joined a running pool."""

    def __init__(self, rounds: int, *, joined: bool) -> None:
        super().__init__(rounds)
        self.joined_running_pool = joined


def _process(tmp_path, *, name, rows, rounds, joined, die_at=None, log=True):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0)
    pool = SlotRowPool(engine.rollout, rows, die_at=die_at)
    driver = IslandDriver(
        learner_id=1, rollout=pool, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=engine.publisher, placement=engine.placement, algorithm=AlgorithmSpec(),
        sync=LocalRoundsSync(rounds, joined=joined), events=EventTape(tmp_path / name, 1),
        capabilities=fake_capabilities())
    driver.data_slot = "1/2"
    if log:
        driver.data_positions = data_position_log(str(tmp_path / "store"), learner_id=1, slot="1/2",
                                                  environ={})
    return driver, pool


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_relaunched_full_parameter_island_continues_at_its_next_row(tmp_path):
    rows = DataSlot(1, 2).rows(list(range(40)))  # island 1 of 2: rows 1, 3, 5, ...
    first, pool = _process(tmp_path, name="a.jsonl", rows=rows, rounds=6, joined=False, die_at=3)
    with pytest.raises(RuntimeError, match="container lost"):
        first.run()
    assert pool.drawn == [[1, 3], [5, 7], [9, 11]]
    start = next(e for e in _events(tmp_path / "a.jsonl") if e["event"] == "rl_driver_start")
    assert start["data_slot"] == "1/2"

    second, pool = _process(tmp_path, name="b.jsonl", rows=rows, rounds=3, joined=True)
    second.run()
    assert pool.seeks == [{"sample_offset": 6}]
    assert pool.drawn == [[13, 15], [17, 19], [21, 23]]  # no row of the first process again
    restored = [e for e in _events(tmp_path / "b.jsonl") if e["event"] == "rl_data_position_restored"]
    assert [(e["after_round"], e["data_cursor"], e["slot"]) for e in restored] == [(2, {"sample_offset": 6}, "1/2")]


def test_without_the_log_a_relaunched_island_reads_its_first_rows_again(tmp_path):
    """Q-4 as it was before the data position log: a full-parameter session starts at
    rollout 0, so the driver's restart path never moves the data source."""
    rows = DataSlot(1, 2).rows(list(range(40)))
    first, _ = _process(tmp_path, name="a.jsonl", rows=rows, rounds=6, joined=False, die_at=2, log=False)
    with pytest.raises(RuntimeError):
        first.run()
    second, pool = _process(tmp_path, name="b.jsonl", rows=rows, rounds=2, joined=True, log=False)
    second.run()
    assert pool.seeks == [] and pool.drawn == [[1, 3], [5, 7]]


def test_a_process_that_starts_the_run_ignores_older_records(tmp_path):
    rows = list(range(20))
    stale = data_position_log(str(tmp_path / "store"), learner_id=1, slot="1/2", environ={})
    stale.record(7, {"sample_offset": 16})  # left by an earlier run in a reused store
    stale.close()
    fresh, pool = _process(tmp_path, name="a.jsonl", rows=rows, rounds=1, joined=False)
    fresh.run()
    assert pool.seeks == [] and pool.drawn == [[0, 1]]
    # the start record superseded the stale one: a relaunch now continues after round 0
    again, pool = _process(tmp_path, name="b.jsonl", rows=rows, rounds=1, joined=True)
    again.run()
    assert pool.seeks == [{"sample_offset": 2}]


def test_a_store_that_refuses_a_record_never_stops_the_round(tmp_path):
    rows = list(range(20))
    driver, pool = _process(tmp_path, name="a.jsonl", rows=rows, rounds=2, joined=False)

    def refuse(*_args):
        raise OSError("read-only file system")

    driver.data_positions.record = refuse
    driver.run()
    assert pool.drawn == [[0, 1], [2, 3]]
    failed = [e for e in _events(tmp_path / "a.jsonl") if e["event"] == "rl_data_position_write_failed"]
    assert len(failed) == 2 and "read-only" in failed[0]["error"]


def test_a_joined_process_without_a_record_says_so_and_a_bad_seek_fails_closed(tmp_path):
    rows = list(range(20))
    lonely, pool = _process(tmp_path, name="a.jsonl", rows=rows, rounds=1, joined=True)
    lonely.run()
    assert pool.seeks == []
    assert [e["reason"] for e in _events(tmp_path / "a.jsonl")
            if e["event"] == "rl_data_position_unknown"] == ["no recorded data position"]
    broken, pool = _process(tmp_path, name="b.jsonl", rows=rows, rounds=1, joined=True)
    pool.seek_data_cursor = lambda cursor: {"sample_offset": 0}
    with pytest.raises(DriverError, match="landed on"):
        broken.run()


def test_full_parameter_session_reports_a_join_into_a_running_pool():
    from yeto.rl.engine.full_parameter_sync import FullParameterSync

    def session(versions, outcome=None):
        stream = SimpleNamespace(recovery_outcome=outcome, resident_state=lambda version: ("state", version),
                                 client=None)
        pairing = SimpleNamespace(streams={"actor": stream}, profile=SimpleNamespace(partial_quorum=False),
                                  gather_roles=lambda fn: None, fragment_versions=lambda: tuple(versions))
        sync = FullParameterSync(pairing, groups={"actor": object()}, run=lambda coro: None,
                                 trajectories=lambda rollout_id: ())
        start = sync.start(SimpleNamespace(emit=lambda *a, **k: None))
        return sync.joined_running_pool, start.rollout_id

    assert session((0, 0, 0)) == (False, 0)
    assert session((7, 6, 6)) == (True, 0)
    recovered = SimpleNamespace(path="peer", donor=0, t_s=1, done_step=1, steps=1, replayed=0,
                                refused_sources=0, seconds=0.1, versions=(0, 0))
    assert session((0, 0), outcome=recovered) == (True, 0)


# ------------------------------------------------------------------ LoRA pool rejoin
def test_lora_pool_rejoin_skips_whole_rounds_of_its_own_slot():
    """A LoRA pool island joining at base version v with no ledger skips v rounds of
    ITS slot (the prompt file holds the slot only), so it never reads the other
    island's rows; the rows of its absent rounds are skipped (known limit until P2)."""
    per_round, base = 2, 3
    cursor = whole_round_restart_cursor(None, base, per_round)
    mine = DataSlot(1, 2).rows(list(range(40)))
    other = set(DataSlot(0, 2).rows(list(range(40))))
    next_rows = mine[cursor["sample_offset"]:cursor["sample_offset"] + per_round]
    assert next_rows == [13, 15] and not other & set(next_rows)


# ------------------------------------------------------------------ Miles passes
class _Dataset:
    def __init__(self, rows):
        self.origin_samples = list(rows)
        self.samples = list(rows)
        self.epoch_id = -1

    def __len__(self):
        return len(self.samples)


class MilesSource:
    """Miles' RolloutDataSource.get_samples without --rollout-shuffle (data_source.py)."""

    def __init__(self, rows):
        self.args = SimpleNamespace(rollout_shuffle=False, n_samples_per_prompt=1)
        self.dataset = _Dataset(rows)
        self.sample_offset = self.epoch_id = self.sample_group_index = self.sample_index = 0

    def get_samples(self, n):
        if self.sample_offset + n <= len(self.dataset):
            prompts = self.dataset.samples[self.sample_offset:self.sample_offset + n]
            self.sample_offset += n
        else:
            prompts = self.dataset.samples[self.sample_offset:]
            n -= len(prompts)
            self.epoch_id += 1
            prompts += self.dataset.samples[:n]
            self.sample_offset = n
        groups = []
        for prompt in prompts:
            groups.append([(prompt, self.sample_index)])
            self.sample_index += 1
            self.sample_group_index += 1
        return groups

    def add_samples(self, groups):
        raise RuntimeError


def test_miles_passes_reshuffle_after_the_first_and_seek_reproduces_them():
    from yeto.rl.adapters.miles import data_passes
    from yeto.rl.adapters.miles.rollout import seek_executor_cursor

    rows = list(range(5))
    source = MilesSource(rows)
    assert data_passes.install(SimpleNamespace(data_source=source), seed=3)
    assert data_passes.install(SimpleNamespace(data_source=source), seed=3)  # once
    drawn = [p for _ in range(5) for p, _i in (g[0] for g in source.get_samples(3))]
    second = [rows[i] for i in pass_order(5, seed=3, data_pass=1)]
    third = [rows[i] for i in pass_order(5, seed=3, data_pass=2)]
    assert drawn == rows + second + third[:5]
    assert (source.epoch_id, source.sample_offset, source.sample_group_index) == (2, 5, 15)
    # a replacement process seeks to the cursor after the first 7 draws: same rows next
    replacement = MilesSource(rows)
    data_passes.install(SimpleNamespace(data_source=replacement), seed=3)
    landed = seek_executor_cursor(SimpleNamespace(data_source=replacement),
                                  {"sample_offset": 2, "epoch_id": 1, "sample_group_index": 7,
                                   "sample_index": 7})
    assert landed["epoch_id"] == 1
    assert [g[0][0] for g in replacement.get_samples(4)] == drawn[7:11]
    # Miles shuffles itself with --rollout-shuffle: the pass order stays out
    shuffled = MilesSource(rows)
    shuffled.args.rollout_shuffle = True
    assert not data_passes.install(SimpleNamespace(data_source=shuffled), seed=3)


def test_miles_pool_installs_the_passes_once_before_its_first_draw_or_seek():
    from yeto.rl.adapters.miles.rollout import MilesRolloutPool

    source = MilesSource(list(range(4)))
    pool = SimpleNamespace(_data_passes_pending=True, _args=SimpleNamespace(seed=5),
                           _in_executor=lambda fn, **kw: fn(SimpleNamespace(data_source=source), **kw))
    MilesRolloutPool._install_data_passes(pool)
    assert not pool._data_passes_pending and hasattr(source, "_yeto_data_passes")
    MilesRolloutPool._install_data_passes(pool)  # no second install


def test_only_a_full_parameter_island_keeps_a_data_position_log(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import entry

    args = SimpleNamespace(yeto_rl_data_position_store=str(tmp_path), yeto_rl_data_slot="1/2")
    monkeypatch.setattr(entry, "_is_full_parameter_sync", lambda sync: sync == "full")
    log = entry.data_position_log_for(args, "full", learner_id=1)
    assert isinstance(log, DataPositionLog) and (log.learner_id, log.slot) == (1, "1/2")
    assert entry.data_position_log_for(args, "lora-pool", learner_id=1) is None
    args.yeto_rl_data_position_store = None
    assert entry.data_position_log_for(args, "full", learner_id=1) is None


def test_trajectory_reward_records_carry_the_data_row():
    from yeto.rl.adapters.miles.rollout_meta_hook import trajectory_reward_records
    from yeto.rl.engine.timeline import TRAJECTORY_REWARD_OPTIONAL

    def sample(meta):
        return SimpleNamespace(index=0, group_index=0, metadata=meta, reward=1.0, status="completed",
                               rollout_id=None)

    args = SimpleNamespace(yeto_rl_max_policy_age=0, rl_max_policy_age=0)
    split, whole = trajectory_reward_records(
        args, [[sample({DATA_ROW_KEY: 7, "task_id": "t7"})], [sample({"task_id": "t8"})]], limit=8)
    assert split["data_row"] == 7 and split["task_id"] == "t7" and "data_row" not in whole
    assert TRAJECTORY_REWARD_OPTIONAL["data_row"] == (int,)


def test_data_split_check_reads_the_island_tapes(tmp_path):
    from tools.eval.data_split_check import check, read_records

    def tape(name, island, records):
        path = tmp_path / name
        path.write_text("".join(json.dumps({"island_id": island, "time_unix": t, **r}) + "\n"
                                for t, r in records))
        return path

    def rewards(t, rollout_id, rows):
        return [(t, {"event": "rl_trajectory_reward", "rollout_id": rollout_id, "data_row": row,
                     "task_id": f"t{row}"}) for row in rows]

    start = {"event": "rl_driver_start", "data_slot": "0/2"}
    island0 = tape("i0.jsonl", 0, [(1, start), *rewards(2, 0, [0, 2]),
                                   (3, {"event": "rl_full_parameter_round", "policy_version": 1}),
                                   *rewards(4, 1, [4, 6]),  # killed before this round's boundary
                                   (10, start), (11, {"event": "rl_data_position_restored"}),
                                   *rewards(12, 0, [4, 6, 8, 10])])
    island1 = tape("i1.jsonl", 1, [(1, {"event": "rl_driver_start", "data_slot": "1/2"}),
                                   *rewards(2, 0, [1, 3, 5, 7])])
    report = check(read_records([island0, island1]))
    assert report["ok"], report["problems"]
    assert report["relaunches"] == [{"island": 0, "process": 1, "restored": True, "unknown": False,
                                     "first_row": 4}]
    bad = tape("bad.jsonl", 1, [(1, {"event": "rl_driver_start", "data_slot": "1/2"}),
                                *rewards(2, 0, [1, 2])])
    report = check(read_records([island0, bad]))
    assert not report["ok"]
    assert any(p.startswith("disjoint") for p in report["problems"])
    assert any(p.startswith("own_slot: island 1") for p in report["problems"])
