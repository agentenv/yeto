"""``--rl-sync-preset decoupled`` on verl islands, routed by trainable form. CPU only.

Full parameters through ``decoupled`` must give exactly the launch of the explicit
``--rl-sync-preset full-parameter --rl-sync-full-parameter-rounds async`` (the verl
full-parameter stream, ``launch_flags.full_parameter_problems``); LoRA keeps
``decoupled`` and runs the verl LoRA fragment session (``decoupled_session``).
"""
from __future__ import annotations

import json
import shlex

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.verl import island_entry

LAYOUT = "e" * 64
GPU = "modal:1xh200,modal:1xh200"
ENTRY = "-m yeto.rl.adapters.verl.island_entry"
VERL = ("--rl-backend", "verl", "--reward-function", "gsm8k_reward:score")
FULL = ("--tuning", "full", "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "4",
        "--local-rl-rounds-per-sync", "4")


def _plan(*flags):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", (*VERL, *flags), gpu=GPU)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, json.loads(json.dumps(dry_run_plan(args), sort_keys=True, default=str))


def _island(plan: dict, k: int):
    tail = plan["island_requests"][k]["learner_command"].split(ENTRY, 1)[1]
    tokens = [{"$LEARNER_ID": str(k), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
    return island_entry.parse_args(tokens)[0]


def test_verl_full_parameters_through_decoupled_equal_the_explicit_async_launch(capsys):
    routed_args, routed = _plan("--rl-sync-preset", "decoupled", *FULL)
    assert "[sync-route] decoupled -> full-parameter (async rounds)" in capsys.readouterr().out
    explicit_args, explicit = _plan("--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async",
                                    *FULL)
    assert routed_args.rl_sync_preset == explicit_args.rl_sync_preset == "full-parameter"
    assert routed_args.rl_sync_full_parameter_rounds == "async"
    assert routed == explicit  # every syncer and island command, byte for byte
    for k in range(2):
        island = _island(routed, k)
        assert (island.sync_preset, island.full_parameter_rounds, island.parameter_mode) == (
            "full-parameter", "async", "full")


def test_verl_lora_through_decoupled_keeps_the_lora_fragment_session():
    args, plan = _plan("--rl-sync-preset", "decoupled", "--fragments", "2", "--local-rl-rounds-per-sync", "2")
    assert args.rl_sync_preset == "decoupled"
    for k in range(2):
        island = _island(plan, k)
        assert island.sync_preset == "decoupled" and island.parameter_mode == "lora"
