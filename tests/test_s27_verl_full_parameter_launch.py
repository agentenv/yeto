"""S27 G-DN: ``yeto launch`` starts verl full-parameter islands.

The full-parameter preset takes its shape from ``--rl-backend``: Miles keeps
its fixed-partition islands (command unchanged), verl islands are colocated
and refuse what the verl stream does not offer.  Every island flag the
launcher emits for the preset reaches the verl island's stream config (none
is ignored).  The last test runs two launcher-built verl islands (fake
workers, the real worker functions) through the launcher-built Rust syncer in
the async mode's defaults (quorum 1, int4 pushes, RDA, local init, ack final
cut).  CPU only; the last test needs cargo.
"""
from __future__ import annotations

import hashlib
import shlex
import socket
import subprocess
import threading

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    MAX_FRAGMENT_BYTES, SHAPES, _run_dict, _run_island, needs_cargo, syncer_binary,
)
from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl import launch_flags as verl_flags
from yeto.rl.local_learner import ComponentIdentity

ENTRY = "-m yeto.rl.adapters.verl.island_entry"
GPU = "modal:1xh200,modal:1xh200"
# Island argv the verl island leaves to the Miles island (parsed and not used):
# provenance, the Miles trainer shape and engine knobs.  Nothing of the sync.
MILES_ONLY = {"--rl-model-recipe", "--reward-sha256", "--source-sha256", "--sync-preset",
              "--over-sampling-batch-size", "--rl-distributed-timeout-minutes", "--optimizer-steps",
              "--completed-groups-path", "--actor-num-nodes", "--actor-num-gpus-per-node",
              "--tensor-parallel", "--pipeline-parallel", "--rollout-num-gpus-per-engine",
              "--sglang-mem-fraction-static", "--miles-root", "--rl-expected-algorithm-sha256",
              "--trust-remote-code"}


def _verl_args(*extra, layout_hash: str = "e" * 64, fragments: int = 4):
    return _launcher_args("ports", (
        "--rl-backend", "verl", "--rl-sync-preset", "full-parameter",
        "--rl-parameter-layout-sha256", layout_hash, "--fragments", str(fragments),
        "--reward-function", "gsm8k_reward:score", *extra), gpu=GPU)


def _prepared(args):
    from yeto.launcher import _prepare_rl_args

    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _island_argv(args, learner_id: int, syncer: str = "127.0.0.1:1") -> list[str]:
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task

    spec = parse_gpu_spec(args.gpu)[learner_id]
    task = launcher._island_task(make_miles_island_task, args, spec, learner_id, 2, syncer, {})
    line = next(line for line in task.run.splitlines() if ENTRY in line)
    return shlex.split(line.split(ENTRY, 1)[1].replace("$LEARNER_ID", str(learner_id))
                       .replace("$SYNCER_ADDR", syncer))


def _flags(argv: list[str]) -> set[str]:
    return {token for token in argv if token.startswith("--")}


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    return tmp_path


# ----------------------------------------------------------------------------- the shape by backend
def test_miles_keeps_its_shape_and_verl_gives_its_own():
    from yeto.launch.rl_args import _full_parameter_backend, _full_parameter_backend_problems

    miles = _launcher_args("ports", ("--rl-sync-preset", "full-parameter"))
    assert _full_parameter_backend(miles) is None and _full_parameter_backend_problems(miles) is None
    verl = _verl_args()
    assert _full_parameter_backend(verl) is verl_flags
    assert _full_parameter_backend_problems(verl) == []


def test_a_miles_full_parameter_launch_still_needs_fixed_partition():
    with pytest.raises(ValueError, match="--rl-placement fixed-partition"):
        _prepared(_launcher_args("ports", ("--rl-sync-preset", "full-parameter",
                                           "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4"),
                                 gpu=GPU))


@pytest.mark.parametrize("extra, match", [
    # refused earlier by the verl placement role; full_parameter_problems names it too
    (("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1"), "fixed-partition"),
    (("--rl-sync-full-parameter-rounds", "sweep", "--rl-sync-overlap"), "no --rl-sync-overlap"),
    (("--rl-sync-expert-token-weights",), "no --rl-sync-expert-token-weights"),
])
def test_verl_refuses_what_its_stream_does_not_offer(extra, match):
    with pytest.raises(ValueError, match=match):
        _prepared(_verl_args(*extra))


def test_a_missing_layout_hash_names_the_verl_planner():
    with pytest.raises(ValueError, match="full_parameter_plan"):
        _prepared(_launcher_args("ports", ("--rl-backend", "verl", "--rl-sync-preset", "full-parameter",
                                           "--fragments", "4"), gpu=GPU))


def test_verl_problems_cover_the_island_refusals():
    from types import SimpleNamespace

    args = SimpleNamespace(rl_placement="fixed-partition", rl_sample_role="trainer", rl_max_policy_age=2,
                           rl_rollout_quant_tier=1, expert_full_count=2, rl_peer_recovery=True,
                           rl_parameter_layout_sha256="e" * 64)
    problems = " | ".join(verl_flags.full_parameter_problems(args))
    for needle in ("--rl-placement colocated", "--rl-sample-role trainer", "--rl-max-policy-age 0", "--rl-rollout-quant-tier",
                   "--expert-full-count"):
        assert needle in problems, needle
    assert "--rl-peer-recovery" not in problems  # S27 G-ED: the verl stream recovers from a peer


def test_the_async_mode_turns_peer_recovery_on_for_verl_as_for_miles():
    """S27 G-ED: without a snapshot store it stays off; a store turns it on (as on Miles)."""
    from yeto.launch.rl_args import _peer_recovery_applies

    args = _prepared(_verl_args())
    assert args.rl_sync_mode_effective == "async" and args.rl_peer_recovery is False
    args.rl_checkpoint_store = "modal-volume://v/run"
    assert _peer_recovery_applies(args) is True


# ----------------------------------------------------------------------------- the island argv
def test_the_async_mode_island_argv_reaches_the_verl_stream(model_dir):
    from yeto import launcher

    args = _prepared(_verl_args())
    assert (args.rl_sync_full_parameter_rounds, args.rl_sync_push_dtype, args.rl_sync_matrix_merge) == (
        "async", "int4", "rda")
    profile = launcher.rl_syncer_profile(args, 2).sha256
    for learner_id in range(2):
        argv = _island_argv(args, learner_id)
        parsed, ignored = island_entry.parse_args(argv)
        assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
        assert parsed.parameter_mode == "full" and parsed.syncer_profile_sha256 == profile
        assert parsed.rl_data_slot == f"{learner_id}/2"
        section = vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                                   algorithm_spec_sha256="a" * 64)["full_parameter"]
        assert section["async_rounds"] and section["local_init"] and section["final_cut"] == "ack"
        assert (section["push_dtype"], section["push_int4_block"], section["matrix_merge"]) == ("int4", 128, "rda")
        assert section["pipeline"] == 2 and section["total_fragment_steps"] == 3 * 4
        config = vfs.role_stream_config(section, syncer="127.0.0.1:1", learner_id=learner_id,
                                        model_revision="a" * 40)
        assert config.partial_quorum and config.push_dtype == "int4" and config.push_tensor_blocks


def test_every_push_and_transport_option_reaches_the_stream_config(model_dir):
    args = _prepared(_verl_args(
        "--rl-sync-push-dtype", "int4", "--rl-sync-push-int4-block", "64", "--rl-sync-push-int4-error-feedback",
        "bf16", "--rl-sync-push-no-tensor-blocks", "--rl-sync-max-local-rounds-ahead", "2",
        "--wan-stream-schedule", "adaptive", "--wan-stall-timeout", "90", "--rl-sync-wait-timeout-s", "120",
        "--rl-sync-fragment-max-bytes", "1048576"))
    parsed, ignored = island_entry.parse_args(_island_argv(args, 1))
    assert _flags(ignored) <= MILES_ONLY
    section = vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                               algorithm_spec_sha256="a" * 64)["full_parameter"]
    config = vfs.role_stream_config(section, syncer="h:1", learner_id=1, model_revision="a" * 40)
    assert (config.push_int4_block, config.push_int4_error_feedback, config.push_tensor_blocks) == (64, "bf16", False)
    assert (config.wan_stream_schedule, config.wan_stall_timeout) == ("adaptive", 90.0)
    assert (config.wait_timeout, config.max_fragment_bytes) == (120.0, 1048576)
    assert section["max_local_rounds_ahead"] == 2


@pytest.mark.parametrize("flag", ["--sweep-overlap", "--sweep-expert-token-weights"])
def test_the_island_refuses_the_flags_its_stream_does_not_offer(model_dir, flag):
    args = _prepared(_verl_args())
    parsed, _ignored = island_entry.parse_args(_island_argv(args, 0) + [flag])
    with pytest.raises(ValueError, match="verl 全参数岛不支持"):
        vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                         algorithm_spec_sha256="a" * 64)


def test_a_strict_sweep_through_the_launcher_keeps_the_p4_flags(model_dir):
    args = _prepared(_verl_args("--rl-sync-full-parameter-rounds", "sweep", "--rl-sync-final-cut", "ack"))
    parsed, _ignored = island_entry.parse_args(_island_argv(args, 0))
    section = vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                               algorithm_spec_sha256="a" * 64)["full_parameter"]
    assert not section["async_rounds"] and section["push_dtype"] == "f32" and section["matrix_merge"] == "avg"
    assert (section["pipeline"], section["local_horizon"], section["final_cut"]) == (1, 1, "ack")


# ----------------------------------------------------------------------------- two islands, real syncer
@needs_cargo
def test_two_launcher_built_verl_islands_run_async_rounds_through_the_real_syncer(
        syncer_binary, tmp_path, monkeypatch, model_dir):
    """The async mode's defaults end to end: the launcher's syncer command and
    island argv, parsed by the verl island entry, through the real syncer."""
    from yeto import launcher

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-full-parameter-launch-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    rounds = 3
    # The island binds its model config (model_dir) and revision into the layout.
    component = ComponentIdentity(role="actor", model_revision="a" * 40,
                                  config_hash=hashlib.sha256((model_dir / "config.json").read_bytes()).hexdigest())
    layout = fp.plan_layout(fp.island_specs(SHAPES), component=component, minimum_fragments=1,
                            max_fragment_bytes=MAX_FRAGMENT_BYTES, matrix_merge="rda")
    fragments = layout.fragments.num_fragments
    args = _prepared(_verl_args("--total-steps", str(rounds), "--rl-sync-fragment-max-bytes", str(MAX_FRAGMENT_BYTES),
                                "--rl-sync-wait-timeout-s", "60", layout_hash=layout.layout_hash,
                                fragments=fragments))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = (launcher.role_syncer_commands(args, 2, binary=str(syncer_binary))["actor"]
               .replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    assert "--final-cut-payload manifest" in command and "--quorum 1" in command
    plans = []
    for learner_id in range(2):
        parsed, _ignored = island_entry.parse_args(_island_argv(args, learner_id, f"127.0.0.1:{port}"))
        section = vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                                   algorithm_spec_sha256="a1" * 32)
        assert section["full_parameter"]["push_dtype"] == "int4"
        plans.append({"learner_id": learner_id, "syncer": parsed.syncer, "model_revision": parsed.model_revision,
                      vfs.PLAN_KEY: section["full_parameter"]})
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    results: dict = {}
    try:
        threads = [threading.Thread(target=_run_island, args=(i, plans[i], 20 * rounds, results))
                   for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        result = results.get(learner_id)
        assert result is not None and not isinstance(result, BaseException), (result, log[-3000:])
        events = result["events"]
        acks = [fields for event, fields in events if event == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True, "sync/final_cut": "ack"}], (learner_id, log[-2000:])
        assert [fields for event, fields in events if event == "rl_full_parameter_round"], learner_id
    assert "all learners acknowledged final cut" in log, log[-2000:]
