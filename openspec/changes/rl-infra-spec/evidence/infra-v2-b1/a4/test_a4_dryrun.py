"""A4/A4b 8-card local end-to-end dry-run (T4R2S2<->T4R4S0): real CLI -> island run (prelude
executed under a temp HOME) -> learner.parse_args -> Miles argv. Every case's switches must reach
the island. No cloud, no GPU."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# YETO_REPO: run the same dry-run against another worktree (the r* cases need the 3.7 restart recovery,
# infra-e1-recovery >= 0e68962: `--rl-elastic-max-recovery-attempts`; on a repo without it they are skipped).
REPO = Path(os.environ.get("YETO_REPO") or next(
    p for p in Path(__file__).resolve().parents if (p / "yeto" / "cli.py").is_file())).resolve()
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "tests"))
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
    "e1b": ("--rl-test-inject-lora-perturb", "0.01"),
    "e1c": ("--custom-generate-function-path", "yeto.rl.tool_wait_workload.generate",
            "--rl-test-tool-delay-s", "30", "--rl-elastic-tool-wait-board",
            "--rl-elastic-drain-timeout-s", "5"),
    "d34": ("--rl-test-inject-stop-failures", "1"),
    "d4": ("--rl-test-inject-stop-failures", "100000", "--rl-elastic-recovery-timeout-s", "120"),
    "d5": ("--rl-elastic-restart-attempts", "1", "--rl-test-kill-learner-at", "COMMITTED"),
    "d6": ("--rl-elastic-restart-attempts", "1", "--rl-test-kill-learner-at", "QUIESCING"),
    "wd": ("--rl-test-inject-update-weights-block-s", "600"),
    "fp": ("--rl-print-attestation-fingerprint",),
}
# E1-D ⑤⑥⑦ with the 3.7 restart recovery, "方案 A" (recovery-design.md §10 / a8go_strict.sh): strict-avg single island, NO
# --rl-single-island-no-sync (the island command carries `--syncer $SYNCER_ADDR`; the head syncer is the restart point).
# + the syncer quorum timeout: the strict pause budget is 0.5 x it (pause_audit); the default 900 -> 450 s rejected the 600 s
# deadlines in the first s0 run ("pause not allowed: expected pause 600s exceeds budget 450s", a4s7-20261001-1-s0).
STRICT_BASE = tuple(a for a in BASE if a != "--rl-single-island-no-sync") + ("--rl-elastic-quorum-timeout-s", "1800")
EXR = ("--rl-elastic-restart-attempts", "2", "--rl-elastic-max-recovery-attempts", "3")
STRICT_CASES = {
    "s0": (),
    "r6": EXR + ("--rl-test-kill-learner-at", "QUIESCING"),
    "r7": EXR,
    "r5": EXR + ("--rl-test-kill-learner-at", "COMMITTED"),
    "r5c": EXR + ("--rl-test-kill-learner-at", "COMMITTED"),
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
        assert float(env.get("YETO_RL_TEST_INJECT_LORA_PERTURB")) == 0.01
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
    if case == "e1c":
        assert args.rl_elastic_drain_timeout_s == 5
    if case == "d4":
        assert args.rl_elastic_recovery_timeout_s == 120
    if case == "fp":
        assert args.rl_print_attestation_fingerprint
    (Path(__file__).parent / f"dryrun-{case}.json").write_text(json.dumps(got, indent=1))


def _has_recovery_flag():
    # the flag lives on the `launch` subparser; the repo text is the simplest exact probe
    return "--rl-elastic-max-recovery-attempts" in (REPO / "yeto" / "cli.py").read_text()


def _strict_argv(case, tmp_path, monkeypatch):
    from yeto.rl import learner
    from yeto.rl.engine.run_config import resolve_rl_run_config

    run = island_run(STRICT_BASE + STRICT_CASES[case], monkeypatch)
    args, env = learner_from_run(run, tmp_path / "home")
    rc = resolve_rl_run_config(args, model_path="/m", rollout_model_path=None, prompt_path="/p",
                               eval_prompt_path=None, provider=_provider(),
                               target_modules=["q_proj"], yeto_policy_sync=True)
    return run, args, env, list(learner.build_ports_launch(args, rc, ()).argv)


@pytest.mark.parametrize("case", sorted(STRICT_CASES))
def test_strict_case(case, tmp_path, monkeypatch):
    """Strict variant: the island gets `--syncer $SYNCER_ADDR` (no no-sync), the same placement, and
    the restart-loop / kill / recovery-budget switches reach the learner."""
    if case != "s0":
        if not _has_recovery_flag():
            pytest.skip("needs the 3.7 restart recovery (infra-e1-recovery >= 0e68962) in this repo")
    run, args, env, argv = _strict_argv(case, tmp_path, monkeypatch)
    assert "--rl-single-island-no-sync" not in run and "--syncer $SYNCER_ADDR" in run
    assert args.syncer == "127.0.0.1:1" and not getattr(args, "rl_single_island_no_sync", False)
    assert args.rl_elastic_quorum_timeout_s == 1800
    pm = json.loads(argv[argv.index("--yeto-placement-map") + 1])
    assert pm["trainer"] == [0, 1, 2, 3] and pm["rollout"] == [4, 5] and pm["standby"] == [6, 7]
    assert [(c["name"], c["start"]) for c in pm["rollout_cells"]] == [("c0", True), ("c1", True), ("c2", False), ("c3", False)]
    assert "--use-miles-router" in argv and args.rl_elastic_declare_cells
    got = {"strict": True, "syncer_arg": args.syncer, "placement_map": pm,
           "env": {k: v for k, v in env.items() if k.startswith("YETO_RL")}}
    if case == "s0":
        assert "YETO_RL_TEST_KILL_LEARNER_AT" not in env and "yeto_rl_restart_loop" not in run
    else:
        assert env.get("YETO_RL_RESTART_ATTEMPTS") == "2" and "yeto_rl_restart_loop" in run
        assert args.rl_elastic_max_recovery_attempts == 3
        got["rl_elastic_max_recovery_attempts"] = args.rl_elastic_max_recovery_attempts
    if case in ("r5", "r5c", "r6"):
        assert env.get("YETO_RL_TEST_KILL_LEARNER_AT") == STRICT_CASES[case][-1]
    if case == "r7":
        assert "YETO_RL_TEST_KILL_LEARNER_AT" not in env   # r7 kills from dctl.py kill_after_tx, not by the learner's own hook
    (Path(__file__).parent / f"dryrun-{case}.json").write_text(json.dumps(got, indent=1))


def test_strict_fingerprint_equals_no_sync(tmp_path, monkeypatch):
    """Dropping --rl-single-island-no-sync (and the restart/recovery switches) does not change the Miles argv
    runtime fingerprint: cfg/attestation-8-6.json (172652ea...) stays valid for the strict chain."""
    from yeto.rl import learner
    from yeto.rl.engine.miles_adapter.entry import ports_runtime_fingerprint
    from yeto.rl.engine.run_config import resolve_rl_run_config

    def fp(extra, home):
        run = island_run(extra, monkeypatch)
        args, _ = learner_from_run(run, tmp_path / home)
        rc = resolve_rl_run_config(args, model_path="/m", rollout_model_path=None, prompt_path="/p",
                                   eval_prompt_path=None, provider=_provider(),
                                   target_modules=["q_proj"], yeto_policy_sync="--rl-single-island-no-sync" not in extra)
        return ports_runtime_fingerprint(learner.build_ports_launch(args, rc, ()))

    base = fp(BASE, "a")
    assert fp(STRICT_BASE, "b") == base
    if _has_recovery_flag():
        assert fp(STRICT_BASE + STRICT_CASES["r5"], "c") == base


def test_strict_pause_budget_admits_600s_deadline():
    """s0 root cause, reproduced on CPU: strict-avg pauses are audited against 0.5 x quorum timeout; the default (900 s)
    rejects the 600 s request deadline, --rl-elastic-quorum-timeout-s 1800 admits it (budget 900 s)."""
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.execution_profile import ExecutionProfile
    from yeto.rl.engine.pause_audit import DEFAULT_QUORUM_TIMEOUT_S, PAUSABLE_PHASE, pause_decision

    profile = ExecutionProfile(name="t", execution_mode="partitioned-serial",
                               outer_protocol="strict-avg").bind_algorithm(AlgorithmSpec())
    old = pause_decision(profile, outer_phase=PAUSABLE_PHASE, expected_pause_s=600,
                         quorum_timeout_s=DEFAULT_QUORUM_TIMEOUT_S)
    assert not old.allowed and "exceeds budget 450s" in old.reason
    new = pause_decision(profile, outer_phase=PAUSABLE_PHASE, expected_pause_s=600, quorum_timeout_s=1800)
    assert new.allowed and new.budget_s == 900
