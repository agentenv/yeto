"""Trainer islands of one fleet with one shape but different placements (own-cluster G-M7).

A Miles full-parameter island (the routed ``--sync-preset full-parameter``: also
``decoupled`` with ``--tuning full``) binds its placement into its training contract
(``island_entry._DENSE_PLACEMENT_MILES_FLAGS``: ``--actor-num-nodes``,
``--actor-num-gpus-per-node``, ``--num-gpus-per-node``, ``--rollout-num-gpus``, ...). Two
islands of one trainer shape (same TP/PP/EP/CP/DP: same layout) on ``2x4`` and ``1x8``
GPUs offer two session contracts, and the real syncer refuses the later island at HELLO
(``session mismatch``, both layout fingerprints equal). The fleet preflight
(``fleet_session``: ``yeto run up`` and its dry run, ``yeto fleet island add/move``) refuses
that before anything is ordered. Islands of two shapes are the shape check's
(``shape_problems``), which keeps its own message. A launch that gives its islands their own
placement (``--rl-island-placement``) forwards ``--rl-per-island-placement``: the placement
flags leave the contract and the syncer admits the islands, so the preflight compares nothing.
"""

from __future__ import annotations

import json
import re
import subprocess
import time

import numpy as np
import pytest

from process_groups import group_popen, kill_group
from syncer_port import bound_port, on_port_zero
from test_fleet_compat_group_preflight import _island_env, _members  # noqa: F401 - autouse fixture
from test_fleet_island_changes import fleet_store  # noqa: F401 - fixture
from test_rl_full_parameter_async import FULL, _hmac_key, _plan, _wait, needs_cargo  # noqa: F401 - fixture
from test_rl_full_parameter_async import syncer_binary  # noqa: F401 - fixture
from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
from yeto.launch.dispense_evidence import DISPENSE_LOG
from yeto.protocol import DTYPE_F32, SyncerClient, layout_fingerprint
from yeto.rl import fleet_session
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine.backend_identity import session_contract_hash
from yeto.rl.up_consistency import trainer_args

BASE = ["--training-mode", "rl", "--model", "Qwen/Qwen3-4B", "--data", "d.jsonl", "--tuning", "full"]
DECOUPLED = ["--rl-sync-preset", "decoupled"]  # with --tuning full: routed to full-parameter
FLEET = [*DECOUPLED, "--rl-sample-role", "trainer"]  # a colocated island as a fleet compiles it
SHAPE_FREE = ["--rl-sync-layout", "shape-free"]
FIXED = ["--rl-placement", "fixed-partition", "--rl-rollout-gpus", "4"]
TWO_BY_FOUR, ONE_BY_EIGHT = "nebius:2x4xh100", "nebius:1x8xh100"  # TP1: DP 8 on both
PLACEMENT = "trainer placement"
SHAPE = "trainer shape"


@pytest.mark.parametrize("extra, found", [
    # one shape (TP1 PP1 EP1 CP1 DP8), two placements: refused, naming both
    ([*FLEET, "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"],
     "island b (learner 1): trainer placement 1x8 GPUs (colocated, trainer 1x8), island a (learner 0) trains on "
     "2x4 GPUs (colocated, trainer 2x4)"),
    (["--rl-sync-preset", "full-parameter", "--rl-sample-role", "trainer", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"],
     PLACEMENT),
    # fixed partition: 4 trainer GPUs on both (DP4), on 2x4 and 1x8 machines
    ([*DECOUPLED, *FIXED, "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"],
     "trainer placement 1x8 GPUs (fixed-partition, trainer 1x4, rollout 4, standby 0)"),
    # a shape-free layout drops the parallel shape from the contract, not the placement
    ([*FLEET, *SHAPE_FREE, "--gpu", f"nebius:1x4xh100,{ONE_BY_EIGHT}"],
     "island b (learner 1): trainer placement 1x8 GPUs (colocated, trainer 1x8), island a (learner 0) trains on 1x4"),
    # same machines: one placement
    ([*FLEET, "--gpu", f"{ONE_BY_EIGHT},{ONE_BY_EIGHT}"], None),
    ([*FLEET, "--gpu", f"{ONE_BY_EIGHT},aws:1x8xh100@us-east-1"], None),
    ([*DECOUPLED, *FIXED, "--gpu", f"{ONE_BY_EIGHT},{ONE_BY_EIGHT}"], None),
    # LoRA (decoupled stays decoupled) and the default strict preset: no placement in the contract
    ([*FLEET, "--tuning", "lora", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    (["--rl-sample-role", "trainer", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    ([*FLEET, "--rl-backend", "verl", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    # per-island placement (--rl-island-placement): the islands get --rl-per-island-placement and
    # their contracts leave the placement out; the syncer admits them (REV G1, G2, G3, G6)
    ([*FLEET, *SHAPE_FREE, "--rl-island-placement", "1:use-gpus-per-node=4", "--gpu",
      f"{ONE_BY_EIGHT},{ONE_BY_EIGHT}"], None),
    ([*FLEET, *SHAPE_FREE, "--rl-island-placement", "1:tensor-parallel=2", "--gpu",
      f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    ([*FLEET, "--rl-island-placement", "1:standby-gpus=0", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    ([*DECOUPLED, *FIXED, *SHAPE_FREE, "--rl-island-placement", "1:tensor-parallel=2", "--gpu",
      f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    # the island's own view: a verl island of a mixed-engine launch binds no placement
    ([*FLEET, "--rl-island-engine", "1:backend=verl", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
])
def test_the_fleet_preflight_compares_the_placement_of_islands_of_one_shape(extra, found):
    problems = fleet_session.synced_shape_problems([*BASE, *extra], ["a", "b"])
    if found is None:
        assert problems == []
        return
    assert len(problems) == 1 and found in problems[0], problems
    assert "session mismatch" in problems[0] and "same nodes x GPUs per node" in problems[0]
    assert SHAPE not in problems[0]


def test_islands_of_two_shapes_keep_the_shape_message_alone():
    """DP 8 and DP 4: the shape check names the shapes; the placement is not reported on top."""
    problems = fleet_session.synced_shape_problems([*BASE, *FLEET, "--gpu", f"{ONE_BY_EIGHT},nebius:1x4xh100"],
                                                   ["a", "b"])
    assert len(problems) == 1 and SHAPE in problems[0] and PLACEMENT not in problems[0]
    assert "DP4, island a (learner 0) trains on TP1 PP1 EP1 CP1 DP8" in problems[0]
    # their placements differ too: a shape-free layout would not help, so it is not offered
    assert "or set --rl-sync-layout shape-free" not in problems[0]
    assert "the same nodes x GPUs per node and parallel flags" in problems[0] and "does not help here" in problems[0]


@pytest.mark.parametrize("extra", [
    # one placement (per-island placement: TP2 on the same machines): shape-free takes the shape out
    [*FLEET, "--rl-island-placement", "1:tensor-parallel=2", "--gpu", f"{ONE_BY_EIGHT},{ONE_BY_EIGHT}"],
    # an island that binds no placement (verl): shape-free is the way out it was
    [*FLEET, "--rl-backend", "verl", "--gpu", f"{ONE_BY_EIGHT},nebius:1x4xh100"],
], ids=["per-island-placement", "verl"])
def test_shape_free_is_offered_where_it_helps(extra):
    from yeto.rl.fleet_session import LaunchSession, shape_problems
    from yeto.rl.up_consistency import trainer_args as parsed

    found = fleet_session.synced_shape_problems([*BASE, *extra], ["a", "b"])
    if not found:  # a launch whose layout is not shape-bound: the shape check itself, with shapes given
        args = parsed([*BASE, *extra])
        placements = fleet_session.launch_placements_of(args) or [None, None]
        found = shape_problems([LaunchSession(name=n, learner_id=i, profile={}, profile_sha256="", learners=0,
                                              hello="", syncer_epoch=0, max_learners=None, shape=shape,
                                              placement=placements[i])
                                for i, (n, shape) in enumerate(zip("ab", [(1, 1, 1, 1, 8), (1, 1, 1, 1, 4)]))])
    assert len(found) == 1 and "or set --rl-sync-layout shape-free for the whole fleet" in found[0]


def test_each_island_is_compared_with_the_first_of_its_shape():
    """DP8 on 1x8, DP4 on 1x4, DP4 on 2x2, DP8 on 2x4: b and c are of another shape than a
    (the shape check); c's placement is compared with b's, d's with a's."""
    problems = fleet_session.synced_shape_problems(
        [*BASE, *FLEET, "--gpu", f"{ONE_BY_EIGHT},nebius:1x4xh100,nebius:2x2xh100,{TWO_BY_FOUR}"],
        ["a", "b", "c", "d"])
    assert [p.split(" trains on ")[0] for p in problems] == [
        "island b (learner 1): trainer shape TP1 PP1 EP1 CP1 DP4, island a (learner 0)",
        "island c (learner 2): trainer shape TP1 PP1 EP1 CP1 DP4, island a (learner 0)",
        "island c (learner 2): trainer placement 2x2 GPUs (colocated, trainer 2x2), island b (learner 1)",
        "island d (learner 3): trainer placement 2x4 GPUs (colocated, trainer 2x4), island a (learner 0)"]


def test_run_up_preflight_names_the_placement_of_a_synced_group_fleet():
    """``yeto run up`` (and its dry run): ``Run.fleet_shape_problem`` of a synced-group fleet."""
    from types import SimpleNamespace

    from yeto.rl import up

    def problem(gpu, per_island=False):
        run = SimpleNamespace(trainer_argv=lambda: [*BASE, *FLEET, "--gpu", gpu],
                              fleet_plan=lambda: SimpleNamespace(per_island=per_island, islands=[
                                  SimpleNamespace(name="own"), SimpleNamespace(name="cloud")]))
        return up.Run.fleet_shape_problem(run)

    found = problem(f"{TWO_BY_FOUR},{ONE_BY_EIGHT}")
    assert found.startswith("fleet: the trainer islands differ in shape or placement (the syncer would refuse")
    assert "island cloud (learner 1): trainer placement 1x8 GPUs" in found
    assert problem(f"{ONE_BY_EIGHT},{ONE_BY_EIGHT}") is None
    assert problem(f"{TWO_BY_FOUR},{ONE_BY_EIGHT}", per_island=True) is None  # fleet_session_problem's


def _one(gpu: str, *extra: str):
    return trainer_args([*BASE, *FLEET, *extra, "--gpu", gpu])


def test_placement_problem_compares_two_island_launches():
    """The form ``yeto fleet island add/move`` reuses: two one-cluster launches' args."""
    found = fleet_session.placement_problem(("island new", _one(TWO_BY_FOUR)), ("island 0", _one(ONE_BY_EIGHT)))
    assert found.startswith("island new: trainer placement 2x4 GPUs (colocated, trainer 2x4), island 0 trains on "
                            "1x8 GPUs (colocated, trainer 1x8)")
    assert fleet_session.placement_problem(("a", _one(ONE_BY_EIGHT)), ("b", _one("aws:1x8xh100@us-east-1"))) is None
    assert fleet_session.placement_problem(("a", _one("nebius:1x4xh100")), ("b", _one(ONE_BY_EIGHT))) is None  # shape
    assert fleet_session.placement_problem(("a", _one(TWO_BY_FOUR, "--tuning", "lora")),
                                           ("b", _one(ONE_BY_EIGHT, "--tuning", "lora"))) is None
    assert fleet_session.launch_placement_of(_one(f"{TWO_BY_FOUR},{ONE_BY_EIGHT}")) is None  # not one island's
    p = fleet_session.launch_placement_of(_one(TWO_BY_FOUR))
    assert (p.nodes, p.gpus_per_node, p.trainer_nodes, p.trainer_gpus_per_node) == (2, 4, 2, 4)
    assert fleet_session.routed_preset(_one(TWO_BY_FOUR)) == "full-parameter"
    assert fleet_session.routed_preset(_one(TWO_BY_FOUR, "--tuning", "lora")) == "decoupled"
    assert fleet_session.launch_placement_of(_one(TWO_BY_FOUR, "--rl-island-placement", "0:standby-gpus=0")) is None
    mixed = _one(f"{TWO_BY_FOUR},{ONE_BY_EIGHT}", "--rl-island-engine", "1:backend=verl")
    assert [p is None for p in fleet_session.launch_placements_of(mixed)] == [False, True]
    verl = _one(f"{TWO_BY_FOUR},{ONE_BY_EIGHT}", "--rl-island-engine", "0:backend=verl", "--rl-island-engine",
                "1:backend=verl")
    assert fleet_session.launch_placements_of(verl) is None  # no island binds one


def test_an_own_bundle_map_is_another_placement():
    """A fixed partition whose ``--rl-elastic-initial-config`` places the trainer on the last
    GPUs (``--rl-island-bundle-map``, ``--yeto-placement-map`` in the contract) next to the
    leading-bundle layout on the same machines."""
    import copy

    lead = trainer_args([*BASE, *DECOUPLED, *FIXED, "--gpu", ONE_BY_EIGHT])
    own = copy.copy(lead)
    own.rl_elastic_initial_placement_slots = {"trainer": [(0, g) for g in range(4, 8)],
                                              "rollout": [[(0, g)] for g in range(4)], "standby": []}
    found = fleet_session.placement_problem(("own", own), ("lead", lead))
    assert found.startswith("own: trainer placement 1x8 GPUs (fixed-partition, trainer 1x4, rollout 4, standby 0, "
                            "own bundle map), lead trains on 1x8 GPUs (fixed-partition, trainer 1x4, rollout 4, "
                            "standby 0):")
    assert fleet_session.placement_problem(("own", own), ("again", copy.copy(own))) is None


def test_a_per_island_fleet_refuses_an_island_on_other_machines(tmp_path):
    """fleet.launch per_island: each island launch is compiled; one on 2x4 next to one on 1x8
    (same DP) is refused, the same machines agree."""
    cfg, members = _members(tmp_path)
    cfg["trainer"]["argv"] += ["--tuning", "full"]
    full = [{**m, "argv": [*m["argv"], "--tuning", "full"]} for m in members]
    with pytest.raises(ValueError) as refused:
        fleet_session.check(cfg["trainer"]["argv"], [{**full[0], "gpu": "modal:1x8xh100"},
                                                     {**full[1], "gpu": "modal:2x4xh100"}])
    assert re.search(r"island \S+ \(learner 1\): trainer placement 2x4 GPUs .* trains on 1x8 GPUs", str(refused.value))
    fleet_session.check(cfg["trainer"]["argv"], [{**m, "gpu": "modal:1x8xh100"} for m in full])
    # an island added next to the running ones: compared with them (yeto fleet island add)
    with pytest.raises(ValueError, match="trainer placement 2x4 GPUs"):
        fleet_session.check(cfg["trainer"]["argv"], [{**full[1], "gpu": "modal:2x4xh100"}],
                            running=[{**full[0], "gpu": "modal:1x8xh100"}])


# -- the real syncer ----------------------------------------------------------------------------

NUMEL, P = 3, 2


def _layout() -> FragmentLayout:
    """The one layout of both islands (same model, shard and DP)."""
    return FragmentLayout([Fragment(MERGE_AVG, [(f"actor.w{i}", NUMEL)]) for i in range(P)])


def _miles_argv(nodes: int, gpus_per_node: int) -> list[str]:
    """The parallel and placement part of a dense island's Miles argv (``miles.config``)."""
    return ["--tensor-model-parallel-size", "1", "--pipeline-model-parallel-size", "1",
            "--actor-num-nodes", str(nodes), "--actor-num-gpus-per-node", str(gpus_per_node),
            "--num-gpus-per-node", str(gpus_per_node), "--rollout-num-gpus-per-engine", "1",
            "--global-batch-size", "8", "--advantage-estimator", "grpo"]


def _contract(nodes: int, gpus_per_node: int, *, per_island_placement: bool = False) -> bytes:
    """The island's HELLO session contract: the layout bound to its training contract
    (``per_island_placement``: the launch's ``--rl-per-island-placement``)."""
    args = type("A", (), dict(rl_engine="ports", seed=1, model="m", model_revision="a" * 40,
                              source_sha256="b" * 64, reward_sha256="d" * 64,
                              rl_per_island_placement=per_island_placement))()
    training = island_entry._dense_full_training_contract_sha256(  # noqa: SLF001
        args, _miles_argv(nodes, gpus_per_node), model_config_sha256="c" * 64)
    return session_contract_hash(layout_fingerprint(_layout()), training)


def _start_syncer(binary, tmp_path, total_steps: int):
    """The launcher's async full-parameter syncer for two learners, on its own port (port 0,
    read back from its log: ``tests/syncer_port.py``)."""
    from yeto import launcher

    args, _ = _plan(FULL[:-1] + (str(P), "--rl-sync-full-parameter-rounds", "async", "--total-steps",
                                 str(total_steps), "--rl-sync-final-cut", "download"))
    command = (on_port_zero(launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"])
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl"))
               .replace(DISPENSE_LOG, str(tmp_path / "dispense.jsonl")))
    log = tmp_path / "syncer.log"
    with log.open("w") as out:
        process = group_popen(["bash", "-c", command], stdout=out, stderr=subprocess.STDOUT, text=True)
    return process, bound_port(process, log_file=log), launcher.rl_syncer_profile(args, 2).sha256


def _client(learner: int, port: int, profile: str, contract: bytes) -> SyncerClient:
    return SyncerClient(("127.0.0.1", port), learner, _layout(), dtype=DTYPE_F32, num_streams=1, connect_timeout=10,
                        max_reconnects=0, finalization_timeout=30, session_contract_hash=contract,
                        syncer_profile_hash=bytes.fromhex(profile))


def _f32(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_the_syncer_refuses_an_island_of_one_shape_on_other_machines(syncer_binary, tmp_path):  # noqa: F811
    """2x4 and 1x8 GPUs (DP 8 both): one layout, two contracts; the later HELLO is refused
    (session mismatch, the layout fingerprints equal) and the session keeps running."""
    assert _contract(2, 4) != _contract(1, 8) and _contract(1, 8) == _contract(1, 8)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, total_steps=2)
    a = _client(0, port, profile, _contract(2, 4))
    b = _client(1, port, profile, _contract(1, 8))
    try:
        a.start()
        a.send_init(0, _f32(np.zeros(NUMEL)))
        time.sleep(0.2)  # A's HELLO establishes the session first
        b.start()

        def refused():
            try:
                b.check_health()
            except Exception as error:  # noqa: BLE001 - the refusal is the expected outcome
                return error
            return None

        text = str(_wait(refused, "the syncer did not refuse island B", timeout=30))
        assert "session mismatch" in text, text
        assert text.count(f"layout_fingerprint={layout_fingerprint(_layout()).hex()}") == 2, text
        a.check_health()
        assert process.poll() is None
    finally:
        a.close()
        b.close()
        kill_group(process)
        process.wait(timeout=10)


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
@pytest.mark.parametrize("placements, per_island", [(((1, 8), (1, 8)), False), (((2, 4), (1, 8)), True)],
                         ids=["same-machines", "per-island-placement"])
def test_islands_of_one_contract_are_admitted_and_merged(syncer_binary, tmp_path, placements, per_island):  # noqa: F811
    """The same machines, or per-island placement (the placement flags out of the contract:
    2x4 and 1x8 offer one contract): both islands are admitted and every round merges both."""
    contracts = [_contract(*p, per_island_placement=per_island) for p in placements]
    assert contracts[0] == contracts[1]
    process, port, profile = _start_syncer(syncer_binary, tmp_path, total_steps=1)
    clients = [_client(k, port, profile, contracts[k]) for k in (0, 1)]
    seen: list[set] = [set(), set()]

    def poll():
        for k, client in enumerate(clients):
            client.check_health()
            client.drain_updates()
            for permit in client.drain_pulls():
                if (permit.global_step, permit.round_attempt) in seen[k]:
                    continue
                seen[k].add((permit.global_step, permit.round_attempt))
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     max(0, permit.global_step - P), permit.global_step, 1, 10,
                                     _f32(np.full(NUMEL, 0.1 * (k + 1))))
        return all(client.finalizing.is_set() for client in clients)

    try:
        for client in clients:
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(np.zeros(NUMEL)))
        _wait(poll, "the syncer never finalized", timeout=90)
        for client in clients:
            manifest, _fragments = client.wait_for_final_fragments(timeout=30)
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients:
            client.close()
        kill_group(process)
        process.wait(timeout=10)
    merges = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines() if '"responders"' in line]
    assert len(merges) == P and all(sorted(r["id"] for r in m["responders"]) == [0, 1] for m in merges)


# -- yeto fleet island add/move: the running islands as they were started ----------------------

def _started(objects, store, name: str, gpu: str, state: str = "running") -> None:
    """Island ``name``'s trainer worker record: started with ``--gpu gpu`` (not its desired one)."""
    from yeto.launch.pool import desired as dz
    from yeto.rl import island_launches as il
    from yeto.rl import trainer_worker

    member = dz.trainers(store.read())[name]
    trainer_worker.save_record(objects, int(member["learner_id"]), {
        "name": name, "prefix": member["prefix"], "gpu": gpu, "state": state,
        "argv": il.launch_argv({**member, "gpu": gpu})}, 0.0)


def test_a_move_is_compared_with_the_running_island_as_it_was_started(fleet_store, capsys):  # noqa: F811
    """mixed runs on H200 (its record), its desired member says H100: a move of sglang to H100
    is compared with the H200 launch, and refused."""
    _cfg, objects, store, run_cli, _ = fleet_store
    move = ("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "4", "--dry-run")
    assert run_cli(*move) == 0
    _started(objects, store, "mixed", "modal:1x2xh200")
    capsys.readouterr()
    assert run_cli(*move) == 2
    err = capsys.readouterr().err
    assert "compat_group nvidia-h100" in err and "nvidia-h200" in err and "island(s) mixed" in err


@pytest.mark.parametrize("state", ["down", "moving", "stopping"])
def test_a_record_that_is_not_a_running_launch_is_not_compared(fleet_store, state):  # noqa: F811
    """DOWN, MOVING, STOPPING: the desired state's member is compared (for MOVING its new target)."""
    _cfg, objects, store, run_cli, _ = fleet_store
    _started(objects, store, "mixed", "modal:1x2xh200", state=state)
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "4", "--dry-run") == 0


def _unparsable(objects, store, name="mixed"):
    from yeto.launch.pool import desired as dz
    from yeto.rl import trainer_worker

    member = dz.trainers(store.read())[name]
    trainer_worker.save_record(objects, int(member["learner_id"]), {
        "name": name, "prefix": member["prefix"], "state": "running",
        "argv": ["--training-mode", "rl", "--no-such-flag"]}, 0.0)


@pytest.mark.parametrize("moved", ["sglang", "mixed"], ids=["another-island", "the-island-itself"])
def test_an_unparsable_running_record_is_refused_unless_allowed(fleet_store, capsys, moved):  # noqa: F811
    """mixed's recorded launch does not compile: a move of another island, or of mixed itself
    (its old machines), is refused by default with the reason and both ways out; the narrow
    --allow-unparsable-running makes it a WARN and the change goes on."""
    from yeto.rl import island_launches as il

    _cfg, objects, store, run_cli, _ = fleet_store
    _unparsable(objects, store)
    # a syncer holds the session: a moved island's old machines are compared too
    il.write_syncer_record(objects, addr="SYNCER:29400", syncer_epoch=None, learners=2, max_learners=4, run="r")
    before = store.read()["generation"]
    move = ("move", "--name", moved, "--gpu", "modal:1x2xh100", "--usd-per-h", "4")
    capsys.readouterr()
    assert run_cli(*move) == 2
    err = capsys.readouterr().err
    assert "island mixed" in err and "its launch as started does not compile" in err
    assert "written when the incarnation started: a changed config does not change it" in err
    assert "Pass --allow-unparsable-running" in err and "yeto fleet island remove --name mixed)" in err
    assert store.read()["generation"] == before
    assert run_cli(*move, "--allow-unparsable-running", "--dry-run") == 0
    err = capsys.readouterr().err
    assert "fleet session: WARN:" in err and "the island is left out of the comparison" in err


def test_the_narrow_way_out_does_not_cover_an_island_whose_desired_launch_does_not_compile(tmp_path):
    """--allow-unparsable-running is about a recorded launch only: a running member rebuilt
    from the desired state that does not compile stays a refusal (its config is to be fixed)."""
    cfg, members = _members(tmp_path)
    with pytest.raises(ValueError, match="fix that island's configuration"):
        fleet_session.check(cfg["trainer"]["argv"], [members[0]],
                            running=[{**members[0], "name": "broken", "gpu": "nowhere:9xq"}],
                            allow_unparsable_running=True)


def test_a_running_island_whose_launch_does_not_compile_refuses_the_change(fleet_store, capsys):  # noqa: F811
    _cfg, objects, store, run_cli, _ = fleet_store
    from yeto.launch.pool import desired as dz
    from yeto.rl import trainer_worker

    member = dz.trainers(store.read())["mixed"]
    trainer_worker.save_record(objects, int(member["learner_id"]), {
        "name": "mixed", "prefix": member["prefix"], "state": "running",
        "argv": ["--training-mode", "rl", "--no-such-flag"]}, 0.0)
    before = store.read()["generation"]
    assert run_cli("move", "--name", "sglang", "--gpu", "modal:1x2xh100", "--usd-per-h", "4") == 2
    err = capsys.readouterr().err
    assert "island mixed (learner 0), which holds the running session: its launch as started does not compile" in err
    assert store.read()["generation"] == before
