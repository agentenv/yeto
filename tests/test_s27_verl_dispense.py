"""S27 G-EU (VERL-PARITY #31): verl training islands take their rows from the task
dispenser. The verl trainer is a stand-in (verl and Ray are not installed here);
the dispenser is the real syncer (``test_s27_dispense.Syncer``)."""

from __future__ import annotations

import json
import shlex
import threading
import time
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_dispense import Syncer, crash, syncer_binary  # noqa: F401
from tools.eval.dispense_check import check
from yeto.rl.adapters.verl import task_dispense as vd
from yeto.rl.engine import task_dispense as td
from yeto.rl.engine.data_split import DATA_ROW_KEY, DISPENSE, STRIDE

KEY = b"dispense-test-key"
TWO_ISLANDS = "modal:1xh100,modal:1xh100"


# ------------------------------------------------------------- the stand-in trainer


class FakeDataset:
    """verl's RLHFDataset as far as the hook reads it: ``dataframe["extra_info"]``
    (rows verl filtered at load are missing) and ``__getitem__``."""

    def __init__(self, rows: int, filtered: frozenset[int] = frozenset()) -> None:
        self.kept = [r for r in range(rows) if r not in filtered]
        self.dataframe = {"extra_info": [{"index": r, DATA_ROW_KEY: r} for r in self.kept]}

    def __len__(self) -> int:
        return len(self.kept)

    def __getitem__(self, index: int) -> dict:
        return {"row": self.kept[index], "raw_prompt": [{"role": "user", "content": f"q{self.kept[index]}"}]}


class FakeTrainer:
    def __init__(self, rows: int, *, filtered: frozenset[int] = frozenset(), gen_batch_size: int = 1,
                 train_batch_size: int = 3) -> None:
        self.train_dataset = FakeDataset(rows, filtered)
        self.config = SimpleNamespace(data={"gen_batch_size": gen_batch_size,
                                            "train_batch_size": train_batch_size})

    def _fetch_one_gen_batch(self):  # replaced by the hook
        raise AssertionError("the dispenser hook must replace _fetch_one_gen_batch")

    def next_train_batch(self) -> list[dict]:
        """verl's ``_next_train_batch``: whole gen_batch_size fetches for one round."""
        data = self.config.data
        chunks = [self._fetch_one_gen_batch()
                  for _ in range(data["train_batch_size"] // data["gen_batch_size"])]
        return [item for chunk in chunks for item in chunk]


def _collate(items):
    return {"items": list(items)}


def _finish(batch_dict, uids):
    return [{**item, "uid": uid} for item, uid in zip(batch_dict["items"], uids)]


def _rows_file(tmp_path, rows: int, labels=None):
    data = [{"messages": [{"role": "user", "content": f"q{r}"}], "label": r,
             **({"task_label": labels[r]} if labels else {})} for r in range(rows)]
    path = tmp_path / vd.ROWS_FILE
    vd.write_rows_file(path, data)
    return path


def _section(syncer, path, learner_id: int, *, block: int = 3, seed: int = 5, prefetch: int = 1):
    count, digest = td.file_fingerprint(path)
    return {"address": f"127.0.0.1:{syncer.port}", "learner_id": learner_id, "rows": count,
            "fingerprint": digest.hex(), "seed": seed, "block": block, "prefetch_blocks": prefetch,
            "rows_file": str(path), "mix": None, "by_label": False, "labels": None}


def _install(trainer, section, events=None):
    return vd.install(trainer, section, key=KEY, collate=_collate, finish=_finish,
                      emit=(lambda kind, **f: events.append((kind, f))) if events is not None else None)


def _round(trainer, hook, rollout_id: int, *, samples: int = 4):
    """One verl round through the hook: draw, hand the groups to the trainer, commit."""
    batch = trainer.next_train_batch()
    rows = hook.kept_round(rollout_id, {item["uid"]: samples for item in batch})
    commit = hook.commit(rollout_id)
    return batch, rows, commit


# ------------------------------------------------------------------- island entry


def test_the_verl_island_refuses_the_dispenser_where_it_does_not_belong():
    base = dict(rl_data_dispense=True, syncer="10.0.0.1:7000", rl_single_island_no_sync=False,
                rl_data_slot=None, rl_data_position_store=None)
    assert vd.island_dispense(SimpleNamespace(**base)) is True
    assert vd.island_dispense(SimpleNamespace(**{**base, "rl_data_dispense": False})) is False
    refused = {
        "--rl-data-slot": (dict(rl_data_slot="0/2"), {}),
        "no syncer": (dict(rl_single_island_no_sync=True, syncer=None), {}),
        "megatron island": ({}, dict(engine="megatron")),
        "fully_async": (dict(rl_max_policy_age=1), {}),
        "buffer trainer": ({}, dict(buffer_trainer=True)),
        "--rl-data-position-store": (dict(rl_data_position_store="/store"), {}),
    }
    for needle, (flags, kwargs) in refused.items():
        with pytest.raises(vd.DispenseIslandError, match=needle):
            vd.island_dispense(SimpleNamespace(**{**base, **flags}), **kwargs)


def test_the_rows_file_is_canonical_so_two_islands_send_one_fingerprint(tmp_path):
    rows = [{"messages": [{"content": "q", "role": "user"}], "label": 34, "task_label": "web",
             "metadata": {"yeto_data_source": 1, "other": "x"}},
            {"messages": [{"role": "user", "content": "r"}], "label": "7", "task_label": float("nan")}]
    a = vd.write_rows_file(tmp_path / "a.jsonl", rows)
    b = vd.write_rows_file(tmp_path / "b.jsonl", list(reversed(list(reversed(rows)))))
    assert a == b and a[0] == 2
    lines = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert lines[0] == {"label": "34", "messages": [{"content": "q", "role": "user"}],
                        "metadata": {"yeto_data_source": 1}, "task_label": "web"}
    assert lines[1] == {"label": "7", "messages": [{"content": "r", "role": "user"}]}, "NaN: no label"


def test_prepare_data_numbers_every_train_row_for_the_dispenser(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    from yeto.rl.adapters.verl.island_entry import prepare_data

    data = tmp_path / "tasks.jsonl"
    data.write_text("".join(json.dumps({"messages": [{"role": "user", "content": f"q{i}"}], "label": i}) + "\n"
                            for i in range(5)))
    train, _ = prepare_data(str(data), None, tmp_path / "plain")
    assert "yeto_data_row" not in json.dumps(pd.read_parquet(train)["extra_info"].tolist(), default=str)
    assert not (tmp_path / "plain" / vd.ROWS_FILE).exists(), "without the dispenser: as before"
    train, _ = prepare_data(str(data), None, tmp_path / "d", number_rows=True)
    infos = pd.read_parquet(train)["extra_info"].tolist()
    assert [int(i[DATA_ROW_KEY]) for i in infos] == list(range(5))
    assert td.file_fingerprint(tmp_path / "d" / vd.ROWS_FILE)[0] == 5
    again, _ = prepare_data(str(data), None, tmp_path / "e", number_rows=True)
    assert (td.file_fingerprint(tmp_path / "d" / vd.ROWS_FILE)
            == td.file_fingerprint(tmp_path / "e" / vd.ROWS_FILE)), "same data, same fingerprint"
    from yeto.rl.engine.data_split import DataSlot

    with pytest.raises(ValueError, match="exclude"):
        prepare_data(str(data), None, tmp_path / "f", number_rows=True, slot=DataSlot.parse("0/2"))


def test_the_plan_section_names_the_actor_syncer_the_block_and_the_streams(tmp_path):
    path = _rows_file(tmp_path, 6, labels=["web", "web", "pwn", "pwn", "pwn", "web"])
    args = SimpleNamespace(syncer="10.0.0.1:7000,10.0.0.1:7001", learner_id=1, seed=17,
                           rl_dispense_prefetch_blocks=2, rl_dispense_mix=None,
                           rl_dispense_by_label=False, rl_dispense_labels=None)
    plain = vd.dispense_section(args, path, block=32)
    count, digest = td.file_fingerprint(path)
    assert plain["address"] == "10.0.0.1:7000", "the actor syncer (shard 0) dispenses"
    assert (plain["learner_id"], plain["rows"], plain["fingerprint"], plain["seed"], plain["block"]) == (
        1, count, digest.hex(), 17, 32)
    assert "data_streams" not in plain
    args.rl_dispense_labels = "pwn"
    labelled = vd.dispense_section(args, path, block=32)
    assert labelled["by_label"] and labelled["labels"] == ["pwn"]
    assert labelled["data_streams"]["accept"] == [0]
    assert [s["label"] for s in labelled["data_streams"]["streams"]] == ["pwn", "web"]
    args.rl_dispense_labels = "crypto"
    with pytest.raises(vd.DispenseIslandError, match="crypto"):
        vd.dispense_section(args, path, block=32)


# ------------------------------------------------------------ the trainer hook


def test_the_verl_hook_draws_rows_by_position_and_commits_the_kept_groups(syncer_binary, tmp_path):
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        rows = 6
        path = _rows_file(tmp_path, rows)
        trainer = FakeTrainer(rows, filtered=frozenset({2}))
        events: list = []
        hook = _install(trainer, _section(syncer, path, 0), events)
        assert hook.filtered_rows == 1 and events[0][0] == "rl_dispense_install"
        assert events[0][1]["rows"] == rows and events[0][1]["filtered_rows"] == 1
        with pytest.raises(vd.DispenseIslandError, match="installed already"):
            _install(trainer, _section(syncer, path, 0))
        batch, data_rows, commit = _round(trainer, hook, 0)
        # position 2 maps to row 2 (pass 0 is data order), which verl filtered: skipped and dropped
        assert [item["row"] for item in batch] == [0, 1, 3]
        assert len({item["uid"] for item in batch}) == 3
        assert data_rows == {"samples": 12, "rows": [[0, 1, 1], [3, 3, 1]]}
        assert commit == {"rounds": [0], "trained": 3, "acked": True, "lost": 0}
        assert hook.cursor() == {"dispensed": 3}
        # a group the trainer did not get (none on the sync path) is dropped at the next draw
        batch = trainer.next_train_batch()
        assert [item["row"] for item in batch][:2] == [4, 5]
        assert batch[2]["row"] != 2, "pass 1 (reshuffled): a filtered row is never handed out"
        hook.kept_round(1, {batch[0]["uid"]: 4, batch[1]["uid"]: 4})
        assert hook.commit(1)["trained"] == 2
        hook.tasks.flush()
        dropped_before = hook.tasks.stats["dropped"]
        trainer.next_train_batch()
        hook.tasks.flush()
        assert hook.tasks.stats["dropped"] >= dropped_before + 1, "the unkept group: dropped at the draw"
        hook.kept_round(2, {})
        assert batch[2]["uid"] not in hook._uid_position, "the uid map forgets settled positions"
        closed = hook.close()
        assert closed["trained"] == 5 and closed["released"] >= 3  # the last draw and the prefetch
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["trained"] == 5
    finally:
        syncer.stop()


def test_two_verl_islands_and_a_relaunch_train_every_row_once(syncer_binary, tmp_path):
    """Two stand-in verl islands draw rounds through the hook while island 1 is
    relaunched (new process, same learner id): the relaunch supersedes the dead
    process at once, its drawn rows go back first, no row trains twice."""
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", lease_s=30)
    try:
        rows = 40
        path = _rows_file(tmp_path, rows)

        def island(learner_id: int):
            trainer = FakeTrainer(rows, gen_batch_size=1, train_batch_size=4)
            return trainer, _install(trainer, _section(syncer, path, learner_id, block=4, seed=1))

        a_trainer, a = island(0)
        b_trainer, b = island(1)
        trained: list[int] = []
        lock = threading.Lock()

        def one(trainer, hook, rollout_id):
            batch, _, commit = _round(trainer, hook, rollout_id)
            assert commit["acked"] and commit["lost"] == 0
            with lock:
                trained.extend(item["row"] for item in batch)

        one(a_trainer, a, 0)
        one(b_trainer, b, 0)
        b_trainer.next_train_batch()  # island 1 draws a round, then its process dies before training
        crash(b.tasks)
        b_trainer, b = island(1)  # relaunched: supersedes, the drawn rows go back first

        def run(trainer, hook):
            rollout_id = 1 if hook is a else 0  # island 0 trained round 0 already
            while True:
                try:
                    one(trainer, hook, rollout_id)
                except td.DispenseEnded:
                    return
                rollout_id += 1
                time.sleep(0.01 if hook is a else 0.03)

        threads = [threading.Thread(target=run, args=pair) for pair in ((a_trainer, a), (b_trainer, b))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads)
        a.close()
        b.close()
        assert sorted(trained) == list(range(rows)), "every row trained exactly once"
        records = syncer.records()
        summary = check(records)
        assert summary["ok"], summary["problems"]
        assert summary["islands"]["1"]["incarnations"] == 2
        assert any(r.get("kind") == "dispense_reclaim" and r.get("reason") == "superseded" for r in records)
    finally:
        syncer.stop()


# ------------------------------------------------------------------- the rollout pool


def test_the_pool_reports_the_dispensed_cursor_offers_no_seek_and_commits_once():
    from yeto.rl.adapters.verl.ports_impl import VerlRolloutPool

    calls = []
    dispense = SimpleNamespace(cursor=lambda: {"dispensed": 7},
                               commit=lambda r: calls.append(("commit", r)) or {"trained": 1},
                               close=lambda: calls.append(("close",)) or {"released": 2})
    island = SimpleNamespace(trainer=FakeTrainer(4))
    pool = VerlRolloutPool(island, dispense=dispense)
    assert pool.data_cursor() == {"dispensed": 7}
    assert pool.seek_data_cursor is None, "the driver's restart paths skip a pool without a seek"
    assert pool.commit_dispensed(3) == {"trained": 1}
    assert pool.close_dispensed() == {"released": 2}
    assert pool.close_dispensed() is None and pool.commit_dispensed(4) is None
    assert calls == [("commit", 3), ("close",)]
    plain = VerlRolloutPool(SimpleNamespace(trainer=SimpleNamespace(
        _fetch_one_gen_batch=lambda: [], config=SimpleNamespace(data={"train_batch_size": 2}))))
    assert plain.commit_dispensed(0) is None and plain.close_dispensed() is None
    assert callable(plain.seek_data_cursor) and plain.data_cursor() == {"sample_offset": 0}


# ----------------------------------------------------------------------- the launcher


def _args(tmp_path, monkeypatch, gpu, *extra):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    args = _launcher_args("ports", tuple(extra), gpu=gpu)
    args.controller = "local"
    return args


def _plan(tmp_path, monkeypatch, *extra):
    from yeto import launcher
    from yeto.launch.syncer import syncer_command

    args = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-backend", "verl", *extra)
    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    return plan, [i["learner_command"] for i in plan["island_requests"]], syncer_command(args, 2)


def test_an_explicit_dispense_gives_verl_training_islands_the_dispenser(tmp_path, monkeypatch, capsys):
    plan, islands, syncer = _plan(tmp_path, monkeypatch, "--rl-data-split", "dispense",
                                  "--rl-dispense-lease-s", "900")
    assert plan["data_split"]["mode"] == DISPENSE
    for command in islands:
        tokens = shlex.split(command)
        assert "--rl-data-dispense" in tokens and "--rl-data-slot" not in tokens
    assert " --dispense " in syncer and "--dispense-lease-s 900.0" in syncer
    assert "data split (dispense)" in capsys.readouterr().err


def test_auto_takes_the_dispenser_on_verl_and_keeps_the_row_slots_on_fully_async(
        tmp_path, monkeypatch):
    """Main agent 17:1x: the auto split's verl fallback becomes the dispenser; only
    fully_async (verl's rollouter draws the rows) keeps the row slots."""
    plan, islands, syncer = _plan(tmp_path, monkeypatch)
    assert plan["data_split"]["mode"] == DISPENSE and plan["data_split"]["requested"] == "auto"
    assert all("--rl-data-dispense" in c for c in islands) and " --dispense " in syncer
    from yeto.launch.data_split_plan import dispense_blockers

    verl_async = SimpleNamespace(rl_backend="verl", rl_max_policy_age=1)
    assert any("fully_async" in b for b in dispense_blockers(verl_async))
    assert not dispense_blockers(SimpleNamespace(rl_backend="verl"))
    # with rollout ahead (G-EV, VERL-PARITY #10) the synchronous trainer draws the rows
    assert not dispense_blockers(SimpleNamespace(rl_backend="verl", rl_max_policy_age=1, rl_rollout_ahead=1))
    ahead = SimpleNamespace(rl_data_dispense=True, syncer="h:1", rl_single_island_no_sync=False,
                            rl_data_slot=None, rl_data_position_store=None, rl_max_policy_age=1,
                            rl_rollout_ahead=1)
    assert vd.island_dispense(ahead)


def test_islands_of_two_backends_cannot_share_the_dispenser(tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.launch import data_split_plan

    monkeypatch.setattr(data_split_plan, "island_backends", lambda args: {"miles", "verl"})
    args = _args(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split", "dispense")
    with pytest.raises(ValueError, match="different backends"):
        launcher._prepare_rl_args(args)
