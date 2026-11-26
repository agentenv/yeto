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
(``shape_problems``), which keeps its own message.
"""

from __future__ import annotations

import json
import re
import subprocess
import time

import numpy as np
import pytest

from process_groups import group_popen, kill_group
from syncer_port import bound_port
from test_fleet_compat_group_preflight import _island_env, _members  # noqa: F401 - autouse fixture
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
TWO_BY_FOUR, ONE_BY_EIGHT = "nebius:2x4xh100", "nebius:1x8xh100"  # TP1: DP 8 on both
PLACEMENT = "trainer placement"
SHAPE = "trainer shape"


@pytest.mark.parametrize("extra, found", [
    # one shape (TP1 PP1 EP1 CP1 DP8), two placements: refused, naming both
    ([*DECOUPLED, "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"],
     "island b (learner 1): trainer placement 1x8 GPUs (colocated, trainer 1x8), island a (learner 0) trains on "
     "2x4 GPUs (colocated, trainer 2x4)"),
    (["--rl-sync-preset", "full-parameter", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], PLACEMENT),
    # fixed partition: 4 trainer GPUs on both (DP4), on 2x4 and 1x8 machines
    ([*DECOUPLED, "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}", "--rl-placement", "fixed-partition",
      "--rl-rollout-gpus", "4"], "trainer placement 1x8 GPUs (fixed-partition, trainer 1x4, rollout 4, standby 0)"),
    # a shape-free layout drops the parallel shape from the contract, not the placement
    ([*DECOUPLED, "--rl-sync-layout", "shape-free", "--gpu", f"nebius:1x4xh100,{ONE_BY_EIGHT}"],
     "island b (learner 1): trainer placement 1x8 GPUs (colocated, trainer 1x8), island a (learner 0) trains on 1x4"),
    # same machines: one placement
    ([*DECOUPLED, "--gpu", f"{ONE_BY_EIGHT},{ONE_BY_EIGHT}"], None),
    ([*DECOUPLED, "--gpu", f"{ONE_BY_EIGHT},aws:1x8xh100@us-east-1"], None),
    # LoRA (decoupled stays decoupled) and the default strict preset: no placement in the contract
    ([*DECOUPLED, "--tuning", "lora", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    (["--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
    ([*DECOUPLED, "--rl-backend", "verl", "--gpu", f"{TWO_BY_FOUR},{ONE_BY_EIGHT}"], None),
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
    problems = fleet_session.synced_shape_problems([*BASE, *DECOUPLED, "--gpu", f"{ONE_BY_EIGHT},nebius:1x4xh100"],
                                                   ["a", "b"])
    assert len(problems) == 1 and SHAPE in problems[0] and PLACEMENT not in problems[0]
    assert "DP4, island a (learner 0) trains on TP1 PP1 EP1 CP1 DP8" in problems[0]


def test_each_island_is_compared_with_the_first_of_its_shape():
    """DP8 on 1x8, DP4 on 1x4, DP4 on 2x2, DP8 on 2x4: b and c are of another shape than a
    (the shape check); c's placement is compared with b's, d's with a's."""
    problems = fleet_session.synced_shape_problems(
        [*BASE, *DECOUPLED, "--gpu", f"{ONE_BY_EIGHT},nebius:1x4xh100,nebius:2x2xh100,{TWO_BY_FOUR}"],
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
        run = SimpleNamespace(trainer_argv=lambda: [*BASE, *DECOUPLED, "--gpu", gpu],
                              fleet_plan=lambda: SimpleNamespace(per_island=per_island, islands=[
                                  SimpleNamespace(name="own"), SimpleNamespace(name="cloud")]))
        return up.Run.fleet_shape_problem(run)

    found = problem(f"{TWO_BY_FOUR},{ONE_BY_EIGHT}")
    assert found.startswith("fleet: the trainer islands differ in shape or placement (the syncer would refuse")
    assert "island cloud (learner 1): trainer placement 1x8 GPUs" in found
    assert problem(f"{ONE_BY_EIGHT},{ONE_BY_EIGHT}") is None
    assert problem(f"{TWO_BY_FOUR},{ONE_BY_EIGHT}", per_island=True) is None  # fleet_session_problem's


def _one(gpu: str, *extra: str):
    return trainer_args([*BASE, *DECOUPLED, *extra, "--gpu", gpu])


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

PORTS = range(8880, 8900)  # fixed local ports (never port 0); the first one free is taken
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


def _contract(nodes: int, gpus_per_node: int) -> bytes:
    """The island's HELLO session contract: the layout bound to its training contract."""
    args = type("A", (), dict(rl_engine="ports", seed=1, model="m", model_revision="a" * 40,
                              source_sha256="b" * 64, reward_sha256="d" * 64))()
    training = island_entry._dense_full_training_contract_sha256(  # noqa: SLF001
        args, _miles_argv(nodes, gpus_per_node), model_config_sha256="c" * 64)
    return session_contract_hash(layout_fingerprint(_layout()), training)


def _start_syncer(binary, tmp_path, total_steps: int):
    """The launcher's async full-parameter syncer for two learners on a fixed port of ``PORTS``."""
    from yeto import launcher

    args, _ = _plan(FULL[:-1] + (str(P), "--rl-sync-full-parameter-rounds", "async", "--total-steps",
                                 str(total_steps), "--rl-sync-final-cut", "download"))
    planned = launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"]
    planned = (planned.replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl"))
               .replace(DISPENSE_LOG, str(tmp_path / "dispense.jsonl")))
    for port in PORTS:
        command, count = re.subn(r"(--port[ =])\d+", rf"\g<1>{port}", planned)
        assert count == 1
        log = tmp_path / f"syncer-{port}.log"
        with log.open("w") as out:
            process = group_popen(["bash", "-c", command], stdout=out, stderr=subprocess.STDOUT, text=True)
        try:
            assert bound_port(process, log_file=log) == port
        except RuntimeError:
            process.wait(timeout=10)
            if "Address already in use" in log.read_text():
                continue  # another process holds this port
            raise
        return process, port, launcher.rl_syncer_profile(args, 2).sha256
    pytest.skip(f"no free port in {PORTS.start}-{PORTS.stop - 1}")


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
def test_islands_on_the_same_machines_are_admitted_and_merged(syncer_binary, tmp_path):  # noqa: F811
    process, port, profile = _start_syncer(syncer_binary, tmp_path, total_steps=1)
    clients = [_client(k, port, profile, _contract(1, 8)) for k in (0, 1)]
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
