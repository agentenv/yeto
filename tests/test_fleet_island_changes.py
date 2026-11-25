"""``yeto fleet island add|remove|move`` (``yeto.rl.fleet_islands``) on a fleet launched
island by island, and what the trainer worker does with each change (local store, fake
launches; no cloud).

* 5.4: an added island gets the lowest learner id never used (a claim in the store) and
  its pools compiled for that island (shards ``-i<k>``, store ``trainer<k>``, the fleet's
  task ledger); a removed island takes its pools with it; a dry run writes nothing.
* 5.5: a move (Modal -> an AWS cluster) keeps the learner id and the store sub-directory;
  the worker stops the island on the old machines (its island STOP, no RETIRE) and starts
  its next incarnation on the new ones. An island whose trainer shape differs from the
  fleet's is refused by its launch's dry run, before anything is ordered.
"""

from __future__ import annotations

import json
import types

import pytest

from test_fleet_island_launches import _prepared, example
from test_fleet_trainer_worker import FakeLaunches
from yeto.launch import island_lifecycle as lifecycle
from yeto.launch.learner_ids import claim_key
from yeto.launch.pool import desired as dz
from yeto.rl.engine.member_table import lane_matches
from yeto.rl import fleet, fleet_islands, island_launches, trainer_worker
from yeto.rl.engine.object_store import DirectoryObjectStore


def prepare(args) -> None:
    """``prepare_launch_args`` without the network (the revisions pinned)."""
    from yeto.launcher import _prepare_rl_args

    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)


@pytest.fixture
def fleet_store(tmp_path, monkeypatch):
    """A seeded per-island fleet (islands mixed = 0, sglang = 1; max 4) on a local store, and
    ``yeto fleet island`` wired to it. Returns (cfg, objects, store, run_cli)."""
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-t5b-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    cfg = example(tmp_path)
    plan = fleet.plan_of(cfg)
    objects = DirectoryObjectStore(tmp_path / "store")
    store = dz.DesiredStore(objects)
    pools = [{"kind": "sampler_pool", "name": p.name, "island": i.index, "pool": {"provider": "modal", "target": 1}}
             for i in plan.islands for p in i.pools]
    store.apply({"members": [*island_launches.trainer_members(cfg, plan), *pools]}, expect=0)
    fake_run = types.SimpleNamespace(cfg=cfg, _objects=lambda: objects)
    monkeypatch.setattr("yeto.launch.pool.fleet_cli._store", lambda config: (fake_run, store))
    samplers = tmp_path / "north.samplers.json"
    samplers.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-north")]))

    def run_cli(*argv: str) -> int:
        from yeto.cli import build_parser

        args = build_parser().parse_args(["fleet", "island", *argv, "--config", "run.json"])
        return fleet_islands.main(args, prepare=prepare)

    return cfg, objects, store, run_cli, samplers


def test_an_added_island_is_compiled_for_its_learner_id(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "k")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "s")
    cfg = example(tmp_path)
    plan = fleet.plan_of(cfg)
    entry = dict(cfg["islands"][1]["samplers"][0], name="sglang-north")
    member, pools = fleet_islands.compile_island(cfg, plan, k=2, name="north", gpu="modal:1x2xh100", usd_per_h=8.0,
                                                 samplers=[entry], island_argv=["--rl-release-every", "3"])
    assert member["learner_id"] == 2 and member["prefix"] == "r-i2" and member["pools"] == ["i2-sglang-north"]
    argv = member["argv"]
    assert argv[argv.index("--rl-fleet-learner-id") + 1] == "2"
    assert argv[argv.index("--rl-island-buffer-shards") + 1].startswith("0:r-sglang-i2-0,")
    assert argv[argv.index("--rl-island-release-every") + 1] == "0:3"
    assert member["fleet_shape"]
    assert argv[-4:] == ["--island-env", "YETO_RL_MEMBER_TABLE=1", "--island-env", "YETO_RL_MEMBER_LANES=i2"]
    (pool,) = pools
    assert pool["name"] == "i2-sglang-north" and pool["island"] == 2
    assert pool["pool"]["shard_base"] == "r-sglang-i2" and pool["pool"]["run_prefix"].endswith("/trainer2")
    assert pool["pool"]["sampler"]["island"] == 2
    assert pool["pool"]["sampler"]["dispense_store"] == plan.dispense_store  # the fleet's one task ledger
    # its lane takes its own pool's member table, not another island's (i1, i12)
    assert lane_matches(pool["pool"]["shard_base"], "i2")
    assert not lane_matches("r-sglang-i1", "i2") and not lane_matches("r-sglang-i12", "i2")


def test_add_dry_run_writes_nothing_and_claims_no_id(fleet_store, capsys):
    cfg, objects, store, run_cli, samplers = fleet_store
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(samplers),
                   "--dry-run") == 0
    out = capsys.readouterr().out
    assert "add pool i2-sglang-north (target 2)" in out and "trainer north: added" in out
    assert "launch of north (learner 2, --gpu modal:1x2xh100) passes its dry run" in out
    assert store.read()["generation"] == 1 and objects.read(claim_key(2)) is None


def test_add_then_remove_an_island_moves_its_pools_with_it(fleet_store, capsys):
    cfg, objects, store, run_cli, samplers = fleet_store
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(samplers)) == 0
    doc = store.read()
    assert doc["generation"] == 2 and objects.read(claim_key(2)) is not None
    assert dz.trainers(doc)["north"]["learner_id"] == 2 and "i2-sglang-north" in dz.sampler_pools(doc)
    # a second island takes id 3; the limit (fleet.max_islands 4) then refuses a third
    other = samplers.parent / "south.samplers.json"
    other.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    assert run_cli("add", "--name", "south", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(other)) == 0
    assert dz.trainers(store.read())["south"]["learner_id"] == 3
    assert run_cli("add", "--name", "east", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(other)) == 2
    assert "no learner id left" in capsys.readouterr().err
    assert run_cli("remove", "--name", "sglang") == 0
    doc = store.read()
    assert "sglang" not in dz.trainers(doc)
    assert sorted(dz.sampler_pools(doc)) == ["i0-sglang-l40s", "i0-vllm-h100", "i2-sglang-north", "i3-sglang-south"]
    # a removed island's id is never handed out again
    assert run_cli("remove", "--name", "north") == 0
    assert run_cli("add", "--name", "west", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(samplers)) == 2


def test_a_move_keeps_the_learner_id_and_sub_store(fleet_store):
    cfg, objects, store, run_cli, _ = fleet_store
    assert run_cli("move", "--name", "sglang", "--gpu", "aws:2xh100@us-east-2", "--usd-per-h", "6") == 0
    member = dz.trainers(store.read())["sglang"]
    assert member["learner_id"] == 1 and member["gpu"] == "aws:2xh100@us-east-2" and member["usd_per_h"] == 6
    assert member["fleet_shape"] == island_launches.fleet_shape(cfg["fleet"]["trainer_argv"])
    from yeto.launcher import dry_run_plan

    args = _prepared(island_launches.launch_argv(member, incarnation=1, syncer={"addr": "h:1"}))
    args.dry_run = True
    (request,) = dry_run_plan(args)["island_requests"]
    assert request["cloud"] == "aws" and request["sample_substore"] == "trainer1"


def test_an_island_of_another_trainer_shape_is_refused_at_dry_run(fleet_store, capsys):
    cfg, objects, store, run_cli, _ = fleet_store
    plan = fleet.plan_of(cfg)
    member = dict(dz.trainers(store.read())["sglang"], fleet_shape=island_launches.fleet_shape(
        cfg["fleet"]["trainer_argv"]))
    fleet_islands.launch_dry_run(member, prepare=prepare)  # same shape: passes
    wide = {**member, "argv": [*member["argv"], "--tensor-parallel", "2"]}
    with pytest.raises(ValueError, match="differs from the fleet"):
        fleet_islands.launch_dry_run(wide, prepare=prepare)
    # an operator's edit to TP2 (yeto fleet apply), then a move: refused, nothing written
    store.update(lambda doc: dz.trainers(doc)["sglang"].update(argv=wide["argv"]))
    generation = store.read()["generation"]
    assert run_cli("move", "--name", "sglang", "--gpu", "aws:2xh100@us-east-2", "--usd-per-h", "6") == 2
    assert "differs from the fleet" in capsys.readouterr().err
    assert store.read()["generation"] == generation
    assert plan.islands[1].learner == 1


def test_the_worker_moves_an_island_from_modal_to_aws_under_its_learner_id(fleet_store):
    cfg, objects, store, run_cli, _ = fleet_store
    island_launches.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=1, learners=2, max_learners=4,
                                        run="r")
    states = lifecycle.IslandStateStore(objects, log=lambda m: None)
    launches = FakeLaunches()
    worker = trainer_worker.TrainerWorker(store, objects, launches, states, log=lambda m: None)
    worker.step()
    assert sorted(launches.live) == ["r-i0", "r-i1"]
    assert run_cli("move", "--name", "sglang", "--gpu", "aws:2xh100@us-east-2", "--usd-per-h", "6") == 0
    worker.step()
    assert states.stop_requested(1)["reason"].startswith("moving to aws:2xh100") and states.stop_requested(0) is None
    assert worker.load(1)["state"] == trainer_worker.MOVING
    worker.step()
    assert launches.started("r-i1") == 1 and "r-i1-m1" not in launches.live  # the old launch still runs
    launches.end("r-i1", 0)  # its launch drained on its island STOP
    worker.step()
    argv = launches.live["r-i1-m1"]
    assert argv[argv.index("--gpu") + 1] == "aws:2xh100@us-east-2"
    assert argv[argv.index("--rl-fleet-learner-id") + 1] == "1"
    assert argv[argv.index("--rl-fleet-trainer-shape") + 1] == island_launches.fleet_shape(cfg["fleet"]["trainer_argv"])
    record = worker.load(1)
    assert record["incarnation"] == 1 and record["prefix_now"] == "r-i1-m1"
    assert states.stop_requested(1) is None  # the new incarnation is not stopped by the old STOP
    assert launches.started("r-i0") == 1 and launches.alive("r-i0") and launches.stops == []
    worker.step()
    assert worker.load(1)["state"] == trainer_worker.RUNNING and len(launches.starts) == 3


def test_an_add_whose_write_failed_takes_its_claimed_id_again(fleet_store, monkeypatch, capsys):
    cfg, objects, store, run_cli, samplers = fleet_store
    real = store.update

    def conflict(*a, **k):
        raise dz.DesiredConflict(1, 2)

    monkeypatch.setattr(store, "update", conflict)
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(samplers)) == 3
    assert objects.read(claim_key(2)) is not None and "north" not in dz.trainers(store.read())
    monkeypatch.setattr(store, "update", real)
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(samplers),
                   "--dry-run") == 0
    assert "launch of north (learner 2," in capsys.readouterr().out
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(samplers)) == 0
    assert dz.trainers(store.read())["north"]["learner_id"] == 2 and objects.read(claim_key(3)) is None
    # another island's add never takes that claim: id 3
    other = samplers.parent / "south.samplers.json"
    other.write_text(json.dumps([dict(cfg["islands"][1]["samplers"][0], name="sglang-south")]))
    assert run_cli("add", "--name", "south", "--gpu", "modal:1x2xh100", "--usd-per-h", "8", "--samplers", str(other)) == 0
    assert dz.trainers(store.read())["south"]["learner_id"] == 3


def test_an_add_without_a_price_is_refused_when_the_run_has_a_budget(fleet_store, capsys):
    """Under a run budget an island needs a price: --usd-per-h, or a billing row of the
    unified price table for its --gpu (a Modal H100 has one; an AWS H100 here has none)."""
    cfg, objects, store, run_cli, samplers = fleet_store
    assert cfg["budget"]["usd"] is not None
    assert run_cli("add", "--name", "north", "--gpu", "aws:2xh100@us-east-2", "--samplers", str(samplers)) == 2
    assert "--usd-per-h" in capsys.readouterr().err and objects.read(claim_key(2)) is None
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--samplers", str(samplers)) == 0
    assert dz.trainers(store.read())["north"]["usd_per_h"] is None  # billed at the table's price


def test_a_move_without_a_price_takes_the_table_s_never_the_old_one(fleet_store, capsys):
    cfg, objects, store, run_cli, samplers = fleet_store
    from yeto.rl import trainer_worker

    before = dz.trainers(store.read())["sglang"]["usd_per_h"]
    assert before is not None
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:2x8xb200") == 0
    member = dz.trainers(store.read())["sglang"]
    assert member["usd_per_h"] is None
    record = {"usd_per_h": member["usd_per_h"], "gpu": member["gpu"]}
    assert trainer_worker.incarnation_price(record) == trainer_worker.table_usd_per_h("modal:2x8xb200") != before
    # machines the table does not price: refused under the budget, nothing written
    generation = store.read()["generation"]
    assert run_cli("move", "--name", "sglang", "--gpu", "aws:2xh100@us-east-2") == 2
    assert "--usd-per-h" in capsys.readouterr().err and store.read()["generation"] == generation



def test_a_budget_set_in_the_breakers_section_also_needs_a_price():
    cfg = {"breakers": {"run": {"budget": {"usd": 50}}}}
    assert fleet_islands.run_budget_usd(cfg) == 50.0
    assert fleet_islands.run_budget_usd({"budget": {"usd": 20}}) == 20.0
    assert fleet_islands.run_budget_usd({}) is None


def test_a_claim_whose_sub_directory_has_data_is_not_taken_again(fleet_store, monkeypatch):
    cfg, objects, store, run_cli, samplers = fleet_store
    real = store.update
    monkeypatch.setattr(store, "update", lambda *a, **k: (_ for _ in ()).throw(dz.DesiredConflict(1, 2)))
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8",
                   "--samplers", str(samplers)) == 3
    monkeypatch.setattr(store, "update", real)
    objects.write("trainer2/groups/r-sglang-north-i2-0/000.json", b"{}")  # its pool sampled meanwhile
    assert run_cli("add", "--name", "north", "--gpu", "modal:1x2xh100", "--usd-per-h", "8",
                   "--samplers", str(samplers)) == 0
    assert dz.trainers(store.read())["north"]["learner_id"] == 3  # a fresh id: trainer2/ is not reused
