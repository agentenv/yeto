"""A Miles sampler island on an island fleet's task ledger (``adapters.miles.task_dispense``
store branch, ``store_dispense.StoreTaskSource``): it claims the slots of the store ledger
the external pools' engine samplers claim from (``<store>/fleet/dispense``) under its shard
name, so a fleet island of one Miles sampler and one external pool gives out every prompt
of an epoch once. Both sides run their real code on one ledger: a local directory store,
and the s3 store over a fake S3 client."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_dispense_wiring import FakeMilesSource
from tests.test_rl_object_store import FakeS3
from yeto.cloud import s3_store
from yeto.rl.adapters.miles import task_dispense as hook
from yeto.rl.adapters.miles.sampler_island import store_written_slots
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import store_dispense as sd
from yeto.rl.engine.data_split import DATA_ROW_KEY
from yeto.rl.engine.resume import STORE_IMPL_ENV
from yeto.rl.engine.task_dispense import (
    DISPENSE_POSITION_KEY,
    SAMPLER_INCARNATION_ENV,
    DispenseEnded,
    DispenseFencedEnd,
)

ROWS = 13  # not a multiple of the Miles block: the last short draw is given back to the pool
NS = "task-dispense"
DISPENSE = {"backend": "store", "namespace": NS, "scan_s": 0.0, "max_scan_s": 0.0}
S3_URI = "s3://bkt/runs/r"
MOUNT = str(Path(s3_store.ISLAND_MOUNT).expanduser() / "runs" / "r")


class LedgerS3(FakeS3):
    """The fake S3 client with what the ledger's scans read: each object's
    ``LastModified`` and the listing's ``Date`` header (the server's clock)."""

    def __init__(self) -> None:
        super().__init__()
        self.modified: dict = {}

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None, ChecksumSHA256=None):
        super().put_object(Bucket=Bucket, Key=Key, Body=Body, IfNoneMatch=IfNoneMatch, ChecksumSHA256=ChecksumSHA256)
        self.modified[(Bucket, Key)] = datetime.now(timezone.utc)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix, StartAfter=""):
                with fake.lock:
                    keys = sorted(k for b, k in fake.objects if b == Bucket and k.startswith(Prefix) and k > StartAfter)
                    stamps = {k: fake.modified[(Bucket, k)] for k in keys}
                date = {"ResponseMetadata": {"HTTPHeaders": {"date": format_datetime(datetime.now(timezone.utc),
                                                                                     usegmt=True)}}}
                for i in range(0, max(len(keys), 1), 2):
                    yield {**date, "Contents": [{"Key": k, "LastModified": stamps[k]} for k in keys[i:i + 2]]}

        return Paginator()


@pytest.fixture(autouse=True)
def _no_current_source(monkeypatch):
    monkeypatch.setenv(SAMPLER_INCARNATION_ENV, "miles-inc-1")
    hook._CURRENT = None
    yield
    if hook._CURRENT is not None:
        hook._CURRENT.tasks.close()
    hook._CURRENT = None


def _prompts(tmp_path: Path, n: int = ROWS) -> Path:
    """The pools' prompt file, which is also the Miles sampler's --data."""
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps({"q": f"q{i}", "a": str(i), "metadata": {"task_id": f"t{i}"}}) + "\n"
                            for i in range(n)))
    return path


def _pool(own: str, ledger: str, prompts: Path, environ=None, groups: int = 2) -> sd.StoreDispensedPrompts:
    """An external pool's engine sampler (``engine_sampler_entry.build_prompts``)."""
    spec = {"sampler_id": "sg-i0-0", "store": own, "dispense_store": ledger,
            "prompts": {"path": str(prompts), "input_key": "q", "label_key": "a", "groups_per_round": groups,
                        "dispense": dict(DISPENSE)}}
    prompts_ = entry.build_prompts(spec, {SAMPLER_INCARNATION_ENV: "pool-inc-1", **(environ or {})})
    assert isinstance(prompts_, sd.StoreDispensedPrompts)
    return prompts_


def _miles(ledger: str, prompts: Path, *, holder: str = "island1", block: int = 2, recover=()):
    """The Miles sampler island's data source on the ledger (what ``island_entry``
    configures and ``sampler_island`` installs in the rollout executor)."""
    fake = FakeMilesSource(ROWS)
    executor = SimpleNamespace(data_source=fake)
    config = {"backend": "store", "store": ledger, "path": str(prompts), "holder": holder,
              "dispense": dict(DISPENSE), "block": block, "learner_id": 1, "recover": [list(r) for r in recover]}
    installed = hook.install(executor, config, None)
    assert installed["installed"] and installed["backend"] == "store" and installed["holder"] == holder
    return fake, installed


def _miles_round(fake, rollout_id: int, block: int = 2) -> list[int]:
    """One Miles round: draw a block, gate the writes, report the written groups. The rows written."""
    groups = fake.get_samples(block)
    positions = [g[0].metadata[DISPENSE_POSITION_KEY] for g in groups]
    fields = hook.write_fields(positions)
    kept = [(g, p) for g, p, f in zip(groups, positions, fields) if f is not None]
    for (_g, p), f in zip(kept, [f for f in fields if f is not None]):
        assert f["dispense_attempt"] is not None and f["dispense_plan"].startswith(f"{NS}:")
    hook.written(rollout_id, [p for _g, p in kept])
    return [g[0].metadata[DATA_ROW_KEY] for g, _p in kept]


def _bounded(obj) -> None:
    """A sampler that would wait forever fails the test instead (one thread runs both)."""
    calls = [0]

    def sleep(_s: float) -> None:
        calls[0] += 1
        if calls[0] > 200:
            raise AssertionError(f"{type(obj).__name__} waits for a slot that never comes")

    obj._sleep = sleep


def _epoch(fake, pool) -> tuple[Counter, Counter, dict | None]:
    """Both samplers in turns until each sees the data end; every claimed group is written."""
    _bounded(pool)
    _bounded(hook.current().tasks)
    miles_rows: Counter = Counter()
    pool_rows: Counter = Counter()
    miles_end = pool_end = None
    for rollout_id in range(4 * ROWS):
        if miles_end is None:
            try:
                miles_rows.update(_miles_round(fake, rollout_id))
            except DispenseEnded as error:
                miles_end = error
        if pool_end is None:
            try:
                items = pool(rollout_id)
            except DispenseEnded as error:
                pool_end = error
            else:
                pool_rows.update(item.metadata["row"] for item in items)
                pool.written(rollout_id, [item.metadata[DISPENSE_POSITION_KEY] for item in items])
        if miles_end is not None and pool_end is not None:
            break
    assert miles_end is not None and pool_end is not None, "both samplers see the end of the epoch"
    return miles_rows, pool_rows, getattr(pool_end, "dispense", None)


def _settled(objects, limit: int) -> Counter:
    kinds: Counter = Counter()
    for key, _t in objects.list_meta(f"{NS}/done/")[0]:
        kinds[key.rsplit(".", 1)[1]] += 1
    assert sum(kinds.values()) == limit
    return kinds


def _check_once(miles_rows: Counter, pool_rows: Counter) -> None:
    every = miles_rows + pool_rows
    assert sorted(every) == list(range(ROWS)) and set(every.values()) == {1}, "every prompt exactly once"
    assert miles_rows and pool_rows, "both samplers took part"
    assert not set(miles_rows) & set(pool_rows)


def test_a_miles_sampler_and_a_pool_give_out_every_prompt_once_on_a_local_ledger(tmp_path):
    prompts = _prompts(tmp_path)
    run = tmp_path / "run"
    ledger = sd.fleet_dispense_store(str(run))
    fake, installed = _miles(ledger, prompts)
    pool = _pool(str(run / "trainer0"), ledger, prompts)
    miles_rows, pool_rows, end = _epoch(fake, pool)
    _check_once(miles_rows, pool_rows)
    objects = sd.DirObjects(ledger)
    assert _settled(objects, ROWS) == Counter({"trained": ROWS})
    # one ledger, one plan: both name it (the trainer's reader trains one group per (plan, slot))
    assert end["plan"] == installed["plan"] == pool.plan_id and end["slots"] == ROWS
    # the claims carry the holders' shard names, unique in the fleet
    holders = {json.loads(objects.read(k))["holder"] for k, _t in objects.list_meta(f"{NS}/claims/")[0]}
    assert holders == {"island1", "sg-i0-0"}
    # the Miles sampler's own store holds no ledger: only the shared one moves
    assert not (run / "trainer1").exists() and not (run / "trainer0" / NS).exists()


def _island_store_env(monkeypatch, tmp_path, own: str, client) -> dict:
    """The store environment of a Modal island whose own store is ``own`` (a sampler
    island of fleet island k: ``<run store>/trainer<k>``), staged on local disk."""
    monkeypatch.setattr(s3_store, "_client", lambda *a, **k: client)
    env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: own,
           s3_store.STAGE_ENV: str(tmp_path / f"stage-{own.rsplit('/', 1)[1]}")}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


def test_the_same_epoch_on_an_s3_ledger_from_islands_that_own_trainer_substores(tmp_path, monkeypatch):
    """As on Modal: the Miles sampler's island store is ``<run>/trainer1``, the pool's
    ``<run>/trainer0``; both open ``<run>/fleet/dispense`` by its URI through the s3
    store hook and claim from that one ledger (one fake S3 bucket)."""
    client = LedgerS3()
    prompts = _prompts(tmp_path)
    ledger = f"{S3_URI}/fleet/dispense"
    _island_store_env(monkeypatch, tmp_path, f"{S3_URI}/trainer1", client)
    fake, installed = _miles(ledger, prompts)
    assert hook.current().tasks.prompts.ledger.objects.prefix == "runs/r/fleet/dispense"
    pool_env = {STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: f"{S3_URI}/trainer0"}
    pool = _pool(f"{MOUNT}/trainer0", ledger, prompts, environ=pool_env)
    miles_rows, pool_rows, end = _epoch(fake, pool)
    _check_once(miles_rows, pool_rows)
    assert end["plan"] == installed["plan"]
    keys = [k for (bucket, k) in client.objects if bucket == "bkt" and f"/{NS}/" in k]
    assert keys and all(k.startswith(f"runs/r/fleet/dispense/{NS}/") for k in keys), "one ledger, nowhere else"
    done = [k for k in keys if "/done/" in k]
    assert len(done) == ROWS
    holders = {json.loads(client.objects[("bkt", k)])["holder"] for k in keys if f"/{NS}/claims/" in k}
    assert holders == {"island1", "sg-i0-0"}


def test_an_island_path_outside_the_islands_own_store_is_why_the_ledger_goes_by_uri(tmp_path, monkeypatch):
    """The island's store hook maps a path only below its own store; the mount path of
    ``<run>/fleet/dispense`` maps back to ``<run>/trainer1`` (a second ledger, every slot
    given out twice). The launcher passes the URI, and a store that opens elsewhere than
    the URI asked for is refused."""
    from yeto.rl.engine.resume import store_for

    _island_store_env(monkeypatch, tmp_path, f"{S3_URI}/trainer1", LedgerS3())
    assert store_for(f"{MOUNT}/fleet/dispense").uri == f"{S3_URI}/trainer1"
    assert hook.ledger_store(f"{S3_URI}/fleet/dispense").uri == f"{S3_URI}/fleet/dispense"
    assert hook.ledger_store("r2://bkt/runs/r/fleet/dispense").uri == f"{S3_URI}/fleet/dispense"
    monkeypatch.setattr("yeto.rl.engine.resume.store_at", lambda location: SimpleNamespace(uri=f"{S3_URI}/trainer1"))
    with pytest.raises(hook.DispenseInstallError, match="opened as .*trainer1"):
        hook.ledger_store(f"{S3_URI}/fleet/dispense")


def test_a_slot_the_ledger_gave_to_another_sampler_is_not_written(tmp_path):
    prompts = _prompts(tmp_path)
    ledger = sd.fleet_dispense_store(str(tmp_path / "run"))
    fake, _installed = _miles(ledger, prompts)
    groups = fake.get_samples(2)
    positions = [g[0].metadata[DISPENSE_POSITION_KEY] for g in groups]
    # the pool's scan took the first slot over (attempt 1): the Miles sampler lost it
    ledger_view = hook.current().tasks.prompts.ledger
    assert ledger_view.objects.create(ledger_view.key("claims", sd._pkey(positions[0]), sd._akey(1)), b"{}")
    fields = hook.write_fields(positions)
    assert fields[0] is None and fields[1]["dispense_attempt"] == 0
    hook.written(0, [positions[1]])
    # the lost slot is not settled by this sampler: its new holder settles it
    assert ledger_view.objects.read(ledger_view.key("done", f"{sd._pkey(positions[1])}.trained")) is not None
    assert not [k for k, _t in ledger_view.objects.list_meta(f"{NS}/done/")[0] if sd._pkey(positions[0]) in k]


def test_a_restarted_miles_sampler_settles_what_its_buffer_holds_and_frees_its_old_claims(tmp_path, monkeypatch):
    prompts = _prompts(tmp_path)
    ledger = sd.fleet_dispense_store(str(tmp_path / "run"))
    fake, installed = _miles(ledger, prompts)
    groups = fake.get_samples(2)
    positions = [g[0].metadata[DISPENSE_POSITION_KEY] for g in groups]
    fields = hook.write_fields(positions)
    # the process wrote the first group to its buffer and died before the ledger heard of it
    entries = [SimpleNamespace(sampler_round={"sampler": "island1", "dispense_position": positions[0],
                                              "dispense_attempt": fields[0]["dispense_attempt"],
                                              "dispense_plan": fields[0]["dispense_plan"]}),
               SimpleNamespace(sampler_round={"sampler": "sg-i0-0", "dispense_position": 7, "dispense_attempt": 0}),
               SimpleNamespace(sampler_round={"sampler": "island1", "dispense_position": 9})]  # no attempt: not a store slot
    recover = store_written_slots(entries, "island1")
    assert recover == [[positions[0], installed["plan"]]]
    hook._CURRENT.tasks.prompts._stop.set()  # the old process is gone (it released nothing)
    hook._CURRENT = None
    monkeypatch.setenv(SAMPLER_INCARNATION_ENV, "miles-inc-2")
    _fake2, again = _miles(ledger, prompts, recover=recover)
    assert (again["recovered"], again["released"]) == (1, 1)
    objects = sd.DirObjects(ledger)
    assert objects.read(f"{NS}/done/{sd._pkey(positions[0])}.trained") is not None
    assert objects.read(f"{NS}/fail/{sd._pkey(positions[1])}/{sd._akey(0)}.release") is not None


def test_a_miles_sampler_that_settles_the_last_slot_names_the_ledgers_plan(tmp_path):
    prompts = _prompts(tmp_path)
    ledger = sd.fleet_dispense_store(str(tmp_path / "run"))
    fake, installed = _miles(ledger, prompts, block=1)
    _bounded(hook.current().tasks)
    rows = [row for rollout_id in range(ROWS) for row in _miles_round(fake, rollout_id, block=1)]
    assert hook.ended_detail(None) is None
    with pytest.raises(DispenseEnded) as end:
        _miles_round(fake, ROWS, block=1)
    # every slot settled: the end names the plan (the shard's end record, sampler_loop)
    assert end.value.dispense == hook.ended_detail(None)
    assert end.value.dispense["plan"] == installed["plan"] and end.value.dispense["slots"] == ROWS
    assert sorted(rows) == list(range(ROWS))


def test_a_short_last_draw_gives_its_rows_back_and_ends_without_the_plan(tmp_path):
    """13 rows in blocks of 4: the fourth draw finds one row and nothing fresh after it. It
    gives the row back (another sampler fills a batch with it) and ends short of the ledger's
    end: no plan in its end (the ledger still has an open slot, a plain end record)."""
    prompts = _prompts(tmp_path)
    run = tmp_path / "run"
    ledger = sd.fleet_dispense_store(str(run))
    fake, installed = _miles(ledger, prompts, block=4)
    _bounded(hook.current().tasks)
    rows = [row for rollout_id in range(3) for row in _miles_round(fake, rollout_id, block=4)]
    with pytest.raises(DispenseEnded) as short:
        _miles_round(fake, 3, block=4)
    assert getattr(short.value, "dispense", None) is None and hook.ended_detail(None) is None
    pool = _pool(str(run / "trainer0"), ledger, prompts)
    _bounded(pool)
    items = pool(0)
    assert [i.metadata["row"] for i in items] == [12] and items[0].metadata["dispense_attempt"] == 1
    pool.written(0, [i.metadata[DISPENSE_POSITION_KEY] for i in items])
    with pytest.raises(DispenseEnded) as end:
        pool(1)
    assert end.value.dispense["plan"] == installed["plan"]
    assert sorted(rows + [12]) == list(range(ROWS))


def test_a_retired_miles_sampler_ends_fenced_and_writes_nothing_it_held(tmp_path):
    prompts = _prompts(tmp_path)
    ledger = sd.fleet_dispense_store(str(tmp_path / "run"))
    fake, _installed = _miles(ledger, prompts)
    groups = fake.get_samples(2)
    positions = [g[0].metadata[DISPENSE_POSITION_KEY] for g in groups]
    out = sd.retire(sd.DirObjects(ledger), NS, "island1", reason="evict")
    assert out["released"] == 2
    with pytest.raises(DispenseFencedEnd):
        fake.get_samples(2)
    assert hook.write_fields(positions) == [None, None]


# -- the launch side: flags, island args, island entry -----------------------------------------

def _group(**kw):
    """A synced group of one trainer (island 0) and one sampler island (island 1)."""
    from yeto.rl.engine import sample_roles

    base = dict(rl_sample_role="trainer", rl_single_island_no_sync=False, rl_island_sample_role=["1:sampler"],
                rl_sampler_follow=sample_roles.FOLLOW_TRAINER, rl_island_sampler_follows=["1:0"],
                rl_island_buffer_shards=["0:sg-i0-0"], rl_island_dispense_store=[f"1:{S3_URI}/fleet/dispense"],
                rl_store_dispense=json.dumps({"backend": "store", "namespace": NS}), rl_data_split=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_store_ledger_flags_are_checked_together():
    from yeto.rl.engine import sample_roles

    assert sample_roles.sampler_dispense_stores(_group(), 2) == {1: f"{S3_URI}/fleet/dispense"}
    sample_roles.check_store_dispense(_group(), 2)
    for args, match in (
            (_group(rl_store_dispense=None), "go together"),
            (_group(rl_island_dispense_store=None), "go together"),
            (_group(rl_island_dispense_store=["0:s3://bkt/x"]), "island 0 is not a sampler island"),
            (_group(rl_island_dispense_store=["1:"]), "name the ledger's store"),
            (_group(rl_store_dispense="{nope"), "--rl-store-dispense"),
            (_group(rl_store_dispense=json.dumps({"backend": "store", "nmespace": NS})), "unknown keys"),
            (_group(rl_store_dispense=json.dumps({"backend": "store", "store": "s3://bkt/y"})), "without 'store'"),
            (_group(rl_store_dispense=json.dumps({"backend": "store", "holdout": "/x"})), "holdout_ids"),
            (_group(rl_data_split="dispense"), "drop --rl-data-split dispense")):
        with pytest.raises(ValueError, match=match):
            sample_roles.check_store_dispense(args, 2)


def test_the_store_ledger_flags_belong_to_a_mixed_group():
    from yeto.rl.engine import sample_roles

    with pytest.raises(ValueError, match="--rl-island-dispense-store belongs to the sampler islands"):
        sample_roles.check_external_pool_flags(_group(rl_island_sampler_follows=None, rl_island_buffer_shards=None))


def test_the_sampler_island_takes_the_ledger_instead_of_its_slot():
    import shlex

    from yeto.launch.island_tasks import store_dispense_flags
    from yeto.rl.engine import sample_roles

    args = _group(rl_checkpoint_store=S3_URI, total_steps=4, rl_sync_preset="decoupled")
    roles = {0: "trainer", 1: "sampler"}
    sampler = sample_roles._synced_group_island_args(args, 1, 2, roles)
    trainer = sample_roles._synced_group_island_args(args, 0, 2, roles)
    assert sampler.rl_dispense_store == f"{S3_URI}/fleet/dispense" and sampler.rl_sampler_data_slot is None
    assert trainer.rl_dispense_store is None and store_dispense_flags(trainer) == ""
    words = shlex.split(store_dispense_flags(sampler))
    # an object store ledger goes by its URI: the island's store hook maps paths only below
    # its own store (trainer<k>), the island opens the URI itself (resume.store_at)
    assert words[:2] == ["--rl-dispense-store", f"{S3_URI}/fleet/dispense"]
    assert words[2] == "--rl-store-dispense" and json.loads(words[3]) == {"backend": "store", "namespace": NS}
    # a ledger on another bucket than the run store is refused at launch
    other = sample_roles._synced_group_island_args(
        _group(rl_checkpoint_store=S3_URI, total_steps=4, rl_sync_preset="decoupled",
               rl_island_dispense_store=["1:s3://other/runs/r/fleet/dispense"]), 1, 2, roles)
    with pytest.raises(ValueError, match="--rl-island-dispense-store"):
        store_dispense_flags(other)
    # a Volume store has no object API on the island: the ledger goes as the island path
    volume = sample_roles._synced_group_island_args(
        _group(rl_checkpoint_store="modal-volume://vol/runs/r", total_steps=4, rl_sync_preset="decoupled",
               rl_island_dispense_store=["1:modal-volume://vol/runs/r/fleet/dispense"]), 1, 2, roles)
    words = shlex.split(store_dispense_flags(volume))
    assert "://" not in words[1] and words[1].endswith("/runs/r/fleet/dispense")


def _island(tmp_path, **kw):
    base = dict(rl_sample_role="sampler", rl_dispense_store="/mnt/store/fleet/dispense",
                rl_store_dispense=json.dumps({"backend": "store", "namespace": NS, "seed": 3}),
                data=str(_prompts(tmp_path)), rl_sampler_node="island2", learner_id=0, rl_data_dispense=False,
                rl_data_slot=None, rl_sampler_data_slot=None, data_sha256=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_island_entry_builds_the_store_ledger_config(tmp_path):
    from yeto.rl.adapters.miles import island_entry

    args = _island(tmp_path)
    assert island_entry._island_dispense(args)
    miles_args = SimpleNamespace(rollout_batch_size=8, over_sampling_batch_size=None, fully_async=False)
    assert island_entry._dispense_config(args, miles_args, None) == {
        "backend": "store", "store": "/mnt/store/fleet/dispense", "path": args.data, "holder": "island2",
        "dispense": {"backend": "store", "namespace": NS, "seed": 3}, "block": 8, "learner_id": 0}


@pytest.mark.parametrize("change, match", [
    ({"data": "openai/gsm8k"}, "is not a local prompt file"),
    ({"rl_sample_role": "trainer"}, "belongs to a sampler island"),
    ({"rl_sampler_data_slot": "0/2"}, "exclude each other"),
    ({"rl_store_dispense": None}, "needs --rl-store-dispense"),
    ({"rl_dispense_store": None}, "needs --rl-dispense-store"),
])
def test_the_island_entry_refuses_a_ledger_it_cannot_join(tmp_path, change, match):
    from yeto.rl.adapters.miles import island_entry

    with pytest.raises(ValueError, match=match):
        island_entry._island_dispense(_island(tmp_path, **change))
