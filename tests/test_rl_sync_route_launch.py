"""``--rl-sync-preset decoupled`` routed by trainable form at launch. No Ray, no GPU.

A ``decoupled`` launch with full parameters (dense, partial routed experts) or
the DeepSeek V4 expert-full hybrid must produce exactly the syncer and island
commands of the explicit ``--rl-sync-preset full-parameter
--rl-sync-full-parameter-rounds async`` launch; every combination the
launcher accepted before routing keeps its preset.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pytest

from free_ports import free_port
from process_groups import group_popen, kill_group
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_async import (
    CARGO,
    NUMEL,
    P,
    _assert_matches,
    _base,
    _client,
    _delta,
    _f32,
    _initial,
    _nesterov_reference,
    _wait,
    syncer_binary,  # noqa: F401
)
from test_rl_k31_synced_cpu import STORE, _island_hmac_key, _selection, _tis_spec  # noqa: F401
from yeto.launch.dispense_evidence import DISPENSE_LOG

LAYOUT = "e" * 64
TWO = "modal:2xh100,modal:2xh100"
FULL_ASYNC = ("--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async")
DECOUPLED = ("--rl-sync-preset", "decoupled")
DENSE = ("--tuning", "full", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
         "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "4", "--local-rl-rounds-per-sync", "4")
EXPERT_FULL = (
    "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "8", "--rl-parameter-layout-sha256", LAYOUT,
    "--fragments", "4", "--local-rl-rounds-per-sync", "4",
    "--rl-model-recipe", "deepseek-v4-flash", "--lora-targets", "attention",
    "--expert-full-count", "4", "--expert-full-lr", "1e-06",
    "--expert-selection-sha256", "1" * 64, "--expert-selection-contract-sha256", "2" * 64,
    "--tensor-parallel", "8", "--expert-parallel", "8", "--rollout-num-gpus-per-engine", "8",
    "--sglang-tp-size", "8", "--sglang-ep-size", "8", "--sglang-attention-backend", "dsv4",
    "--sglang-page-size", "256", "--no-sglang-deterministic-inference",
)


def _k31(tmp_path) -> tuple[str, ...]:
    selection = tmp_path / "expert_selection.json"
    _selection().write(selection)
    return ("--tuning", "full", "--rl-model-recipe", "kimi_k31", "--rl-expert-selection", str(selection),
            "--expert-parallel", "8", "--tensor-parallel", "1", "--rl-sample-role", "trainer",
            "--rl-checkpoint-store", STORE, "--rl-max-policy-age", "2", "--rl-rollout-top-logprobs-num", "128",
            "--rl-algorithm-spec", str(_tis_spec(tmp_path)), "--cluster-prefix", "k31-synced",
            "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "2", "--local-rl-rounds-per-sync", "2")


def _prepare(flags, gpu):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", tuple(flags), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _plan(flags, gpu):
    from yeto.launcher import dry_run_plan

    args = _prepare(flags, gpu)
    return args, json.loads(json.dumps(dry_run_plan(args), sort_keys=True, default=str))


def _flag(command: str, flag: str) -> str:
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1]


def _island(plan: dict, k: int):
    from yeto.rl.adapters.miles import island_entry

    tail = plan["island_requests"][k]["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    tokens = [{"$LEARNER_ID": str(k), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
    return island_entry.parse_args(tokens)


def _assert_same_launch(decoupled_flags, explicit_flags, gpu, capsys):
    routed_args, routed = _plan(decoupled_flags, gpu)
    line = capsys.readouterr().out
    explicit_args, explicit = _plan(explicit_flags, gpu)
    assert "[sync-route] decoupled -> full-parameter (async rounds)" in line
    assert routed_args.rl_sync_preset_requested == "decoupled"
    assert routed_args.rl_sync_route.rerouted and not explicit_args.rl_sync_route.rerouted
    assert routed_args.rl_sync_preset == explicit_args.rl_sync_preset == "full-parameter"
    assert routed_args.rl_sync_full_parameter_rounds == explicit_args.rl_sync_full_parameter_rounds == "async"
    # byte for byte: every syncer and island command and the rest of the plan
    assert routed == explicit
    return routed


def test_dense_full_parameters_through_decoupled_equal_the_explicit_async_launch(capsys):
    plan = _assert_same_launch((*DECOUPLED, *DENSE), (*FULL_ASYNC, *DENSE), TWO, capsys)
    [actor] = plan["role_syncers"]
    assert "--policy-sweep-fragments" not in actor["command"]  # asynchronous rounds
    assert _flag(actor["command"], "--pipeline") == "2"
    for k in range(2):
        island = _island(plan, k)
        assert (island.sync_preset, island.full_parameter_rounds, island.parameter_mode) == (
            "full-parameter", "async", "full")


def test_layout_auto_through_decoupled_equals_the_explicit_async_launch(capsys):
    """``--rl-parameter-layout-sha256 auto`` (the syncer binds the layout of the first HELLO)
    reaches the routed islands as it reaches an explicit full-parameter async launch."""
    from yeto.rl.engine.selection import PARAMETER_LAYOUT_AUTO

    dense_auto = tuple(PARAMETER_LAYOUT_AUTO if f == LAYOUT else f for f in DENSE)
    plan = _assert_same_launch((*DECOUPLED, *dense_auto), (*FULL_ASYNC, *dense_auto), TWO, capsys)
    for k in range(2):
        island = _island(plan, k)
        assert (island.sync_preset, island.full_parameter_rounds) == ("full-parameter", "async")
        assert island.parameter_layout_sha256 == PARAMETER_LAYOUT_AUTO


def test_partial_experts_through_decoupled_equal_the_explicit_async_launch(tmp_path, capsys):
    """Kimi-K3.1 synced trainers. Before routing, ``decoupled`` with this form
    passed the launcher but every island refused it at start-up
    (``--parameter-mode full requires a full-parameter --sync-preset``)."""
    flags = _k31(tmp_path)
    gpu = "modal:1x8xb300,modal:1x8xb300"
    plan = _assert_same_launch((*DECOUPLED, *flags), (*FULL_ASYNC, *flags), gpu, capsys)
    from yeto.rl.adapters.miles import island_entry

    for k in range(2):
        island = _island(plan, k)
        assert island.sync_preset == "full-parameter" and island.full_parameter_rounds == "async"
        island_entry._require_ports_supported(island)
        assert island_entry._full_parameter_mode(island, island.sync_preset)


def test_expert_full_hybrid_through_decoupled_equals_the_explicit_async_launch(capsys):
    plan = _assert_same_launch((*DECOUPLED, *EXPERT_FULL), (*FULL_ASYNC, *EXPERT_FULL),
                               "modal:16xh200,modal:16xh200", capsys)
    from yeto.rl.adapters.miles import island_entry

    island = _island(plan, 1)
    assert (island.parameter_mode, island.sync_preset, island.expert_full_count) == ("lora", "full-parameter", 4)
    island_entry._check_dense_full_sweep(island, yeto_policy_sync=True)


def test_decoupled_with_an_explicit_sweep_is_refused():
    with pytest.raises(ValueError, match="asynchronous fragment rounds"):
        _prepare((*DECOUPLED, *DENSE, "--rl-sync-full-parameter-rounds", "sweep"), TWO)


def test_decoupled_pipeline_maps_to_the_fragment_pipeline_and_an_explicit_one_wins():
    args = _prepare((*DECOUPLED, *DENSE, "--pipeline", "3"), TWO)
    assert args.rl_sync_fragment_pipeline == 3
    args = _prepare((*DECOUPLED, *DENSE, "--pipeline", "3", "--rl-sync-fragment-pipeline", "1"), TWO)
    assert args.rl_sync_fragment_pipeline == 1


@pytest.mark.parametrize("flags", [
    ("--rl-sync-preset", "strict-avg"),
    ("--rl-sync-preset", "decoupled", "--fragments", "2", "--local-rl-rounds-per-sync", "2"),
    ("--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
     "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "2", "--rl-sync-full-parameter-rounds", "sweep"),
    ("--rl-sync-preset", "dense-full", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
     "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "2"),
    (*FULL_ASYNC, *DENSE),
])
def test_combinations_accepted_before_routing_keep_their_preset(flags, capsys):
    before = dict(zip(flags[::2], flags[1::2])).get("--rl-sync-preset")
    args = _prepare(flags, TWO)
    assert "[sync-route]" not in capsys.readouterr().out
    assert not args.rl_sync_route.rerouted
    assert args.rl_sync_preset == {"dense-full": "full-parameter"}.get(before, before)


def test_lora_decoupled_island_command_is_unchanged_by_routing():
    """The LoRA fragment stack: the island still receives the decoupled preset."""
    _, plan = _plan(("--rl-sync-preset", "decoupled", "--fragments", "2", "--local-rl-rounds-per-sync", "2"), TWO)
    command = plan["island_requests"][0]["learner_command"]
    assert _flag(command, "--sync-preset") == "decoupled"
    assert "--full-parameter-rounds" not in command


# -- the routed launch against the real Rust syncer ----------------------------------------------


def _routed_syncer(binary, tmp_path, *, total_steps: int):
    """The syncer command of a ``decoupled --tuning full`` launch (the strict sync-mode values: full
    roster, equal weights, f32, so the merge has an exact numpy reference)."""
    from yeto import launcher

    flags = (*DECOUPLED, "--rl-sync-mode", "strict", "--tuning", "full", "--rl-placement", "fixed-partition",
             "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", LAYOUT, "--fragments", str(P),
             "--local-rl-rounds-per-sync", "4", "--total-steps", str(total_steps),
             "--rl-sync-final-cut", "download")
    args = _prepare(flags, TWO)
    assert args.rl_sync_route.rerouted and args.rl_sync_full_parameter_rounds == "async"
    command = launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"]
    assert "--policy-sweep-fragments" not in command
    port = free_port()
    command = (command.replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl"))
               .replace(DISPENSE_LOG, str(tmp_path / "task-dispense.jsonl")))
    log = tmp_path / "syncer.log"
    with log.open("w") as out:
        process = group_popen(["bash", "-c", command], stdout=out, stderr=subprocess.STDOUT, text=True)
    return process, port, launcher.rl_syncer_profile(args, 2).sha256, log


@pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")
def test_routed_decoupled_full_parameters_run_to_the_final_cut(syncer_binary, tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "sync-route-test-key")
    total_steps = 2
    process, port, profile_hash, log = _routed_syncer(syncer_binary, tmp_path, total_steps=total_steps)
    clients = [_client(learner, port, profile_hash) for learner in (0, 1)]
    rounds: dict[int, dict] = {}
    seen: list[set] = [set(), set()]

    def poll():
        for learner, client in enumerate(clients):
            client.check_health()
            client.drain_updates()
            for permit in client.drain_pulls():
                if (permit.global_step, permit.round_attempt) in seen[learner]:
                    continue
                seen[learner].add((permit.global_step, permit.round_attempt))
                delta = _delta(learner, permit.global_step)
                rounds.setdefault(permit.global_step, {})[learner] = (1.0, delta)
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     _base(permit.global_step), permit.global_step, 4, 40, _f32(delta))
        return all(client.finalizing.is_set() for client in clients)

    try:
        for client in clients:
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=60)
        finals = []
        for client in clients:
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals.append({f.fragment_id: np.frombuffer(bytes(f.data), dtype=np.float32)[-NUMEL:]
                           for f in fragments})
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients:
            client.close()
        kill_group(process)
        process.wait(timeout=10)
    assert sorted(rounds) == list(range(1, total_steps * P + 1))
    assert all(set(contributions) == {0, 1} for contributions in rounds.values())
    # H = 4: the decoupled outer step (Nesterov 0.7 / 0.9) over equal-weight means
    expected = _nesterov_reference(rounds, lr=0.7, mu=0.9)
    for final in finals:
        _assert_matches(final, expected)
