"""A4/A4b 8-card local end-to-end dry-run (T4R2S2<->T4R4S0): real CLI -> island run (prelude
executed under a temp HOME) -> learner.parse_args -> Miles argv. Every case's switches must reach
the island. No cloud, no GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "tests"))
from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

RES = Path(__file__).with_name("resources-8.json")
BASE = ("--gpu", "modal:8xh100", "--modal-gpu-exact", "--total-steps", "12", "--seed", "17",
        "--rl-single-island-no-sync", "--controller", "local",
        "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "2", "--rl-standby-gpus", "2",
        "--rl-elastic", "--rl-elastic-declare-cells", "--rl-elastic-cells", "c0,c1,c2,c3",
        "--rl-elastic-resources", str(RES), "--rl-elastic-initial-config", "T4R2S2",
        "--rl-observe-timeline", "--modal-retries", "0", "--modal-timeout-s", "3000")
CASES = {
    "base": (),
    "x2": (),
    "e1b": ("--rl-test-inject-weight-override", "/root/.cache/base-ckpt"),
    "e1c": ("--custom-generate-function-path", "yeto.rl.tool_wait_workload.generate",
            "--rl-test-tool-delay-s", "30", "--rl-elastic-tool-wait-board"),
    "d34": ("--rl-test-inject-stop-failures", "1"),
    "d5": ("--rl-elastic-restart-attempts", "1", "--rl-test-kill-learner-at", "COMMITTED"),
    "d6": ("--rl-elastic-restart-attempts", "1", "--rl-test-kill-learner-at", "QUIESCING"),
    "wd": ("--rl-test-inject-update-weights-block-s", "600"),
    "fp": ("--rl-print-attestation-fingerprint",),
}


def _provider():
    return SimpleNamespace(hidden_size=1024, num_attention_heads=16, num_layers=28,
                           ffn_hidden_size=3072, num_query_groups=8, kv_channels=128,
                           multi_latent_attention=False, num_moe_experts=None,
                           seq_length=40960, layernorm_epsilon=1e-6, rotary_base=1000000,
                           vocab_size=151936, max_position_embeddings=40960, qk_layernorm=True,
                           gated_linear_unit=True, share_embeddings_and_output_weights=True,
                           add_bias_linear=False, add_qkv_bias=False, activation_func=None)


@pytest.mark.parametrize("case", sorted(CASES))
def test_case(case, tmp_path, monkeypatch):
    from yeto.rl import learner
    from yeto.rl.engine.run_config import resolve_rl_run_config

    run = island_run(BASE + CASES[case], monkeypatch)
    args, env = learner_from_run(run, tmp_path / "home")
    rc = resolve_rl_run_config(args, model_path="/m", rollout_model_path=None, prompt_path="/p",
                               eval_prompt_path=None, provider=_provider(),
                               target_modules=["q_proj"], yeto_policy_sync=False)
    argv = list(learner.build_ports_launch(args, rc, ()).argv)
    pm = json.loads(argv[argv.index("--yeto-placement-map") + 1])
    assert pm["trainer"] == [0, 1, 2, 3] and pm["rollout"] == [4, 5] and pm["standby"] == [6, 7]
    assert [(c["name"], c["bundles"], c["start"]) for c in pm["rollout_cells"]] == [
        ("c0", [4], True), ("c1", [5], True), ("c2", [6], False), ("c3", [7], False)]
    assert "--use-miles-router" in argv and args.rl_elastic_declare_cells
    got = {"argv_has_miles_router": True, "placement_map": pm,
           "env": {k: v for k, v in env.items() if k.startswith("YETO_RL_TEST")}}
    if case == "e1b":
        assert env.get("YETO_RL_TEST_INJECT_WEIGHT_OVERRIDE_PATH") == "/root/.cache/base-ckpt"
    if case == "d34":
        assert env.get("YETO_RL_TEST_INJECT_STOP_FAILURES") == "1"
    if case in ("d5", "d6"):
        assert env.get("YETO_RL_TEST_KILL_LEARNER_AT") == CASES[case][-1]
        assert env.get("YETO_RL_RESTART_ATTEMPTS") == "1" and "yeto_rl_restart_loop" in run
    if case == "wd":
        assert float(env.get("YETO_RL_TEST_INJECT_UPDATE_WEIGHTS_BLOCK_S")) == 600.0
    if case == "e1c":
        assert args.rl_elastic_tool_wait_board and float(env["YETO_RL_TEST_TOOL_DELAY_S"]) == 30.0
        assert "--custom-generate-function-path" in argv
    if case == "fp":
        assert args.rl_print_attestation_fingerprint
    (Path(__file__).parent / f"dryrun-{case}.json").write_text(json.dumps(got, indent=1))
