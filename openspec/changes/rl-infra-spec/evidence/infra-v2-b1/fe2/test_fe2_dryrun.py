"""F-E2 (4.4 rebuild-trainer smoke) local end-to-end dry-run. No cloud, no GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "tests"))
from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

RES = Path(__file__).with_name("resources-T2R1S0.json")


def _provider():
    return SimpleNamespace(hidden_size=1024, num_attention_heads=16, num_layers=28,
                           ffn_hidden_size=3072, num_query_groups=8, kv_channels=128,
                           multi_latent_attention=False, num_moe_experts=None,
                           seq_length=40960, layernorm_epsilon=1e-6, rotary_base=1000000,
                           vocab_size=151936, max_position_embeddings=40960, qk_layernorm=True,
                           gated_linear_unit=True, share_embeddings_and_output_weights=True,
                           add_bias_linear=False, add_qkv_bias=False, activation_func=None)


def test_fe2(tmp_path, monkeypatch):
    from yeto.rl import learner
    from yeto.rl.engine import controller as ctl
    from yeto.rl.engine.algorithm import resolve_ports_algorithm
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.miles_adapter.trainer_rebuild import rebuild_preconditions
    from yeto.rl.engine.run_config import resolve_rl_run_config

    cli = ("--gpu", "modal:3xl40s", "--total-steps", "4", "--seed", "17",
           "--rl-single-island-no-sync", "--controller", "local",
           "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
           "--rl-elastic", "--rl-elastic-resources", str(RES),
           "--rl-elastic-initial-config", "T2R1S0", "--rl-elastic-cells", "c0")
    args, _ = learner_from_run(island_run(cli, monkeypatch), tmp_path / "home")
    assert args.rl_elastic and args.rl_elastic_state_dir and args.rl_single_island_no_sync
    rc = resolve_rl_run_config(args, model_path="/m", rollout_model_path=None, prompt_path="/p",
                               eval_prompt_path=None, provider=_provider(),
                               target_modules=["q_proj"], yeto_policy_sync=False)
    ml = learner.build_ports_launch(args, rc, ())
    argv = list(ml.argv)
    val = lambda f: argv[argv.index(f) + 1] if f in argv else None  # noqa: E731
    assert val("--actor-num-gpus-per-node") == "2" and val("--rollout-num-gpus") == "1"
    ns = SimpleNamespace(requested_load=None if "--load" not in argv else val("--load"),
                         use_fault_tolerance="--use-fault-tolerance" in argv,
                         indep_dp="--indep-dp" in argv, trainer_controller_addrs=None)
    assert rebuild_preconditions(ns) == []
    miles_args = SimpleNamespace(yeto_rl_learner_id=0, rollout_batch_size=4,
                                 n_samples_per_prompt=8, num_steps_per_rollout=1)
    learner.apply_ports_infra_switches(args, miles_args, {})
    launch = SimpleNamespace(placement=SimpleNamespace(kind="fixed-partition"), argv=())
    profile = entry.execution_profile_for(miles_args, launch,
                                          resolve_ports_algorithm(args, rl_engine="ports"),
                                          yeto_policy_sync=False,
                                          expected_sha256=args.rl_expected_algorithm_sha256)
    assert profile.execution_mode == "partitioned-serial"
    # real controller on the real manifest: a wired rebuilder accepts the request (no attestation)
    state = tmp_path / "state"
    c = ctl.IslandController(state_dir=state, configs=__import__(
        "yeto.rl.elastic_benchmark.capabilities", fromlist=["x"]).parse_configs(json.loads(RES.read_text())),
        attestation=None, profile=profile, initial_config="T2R1S0", runtime_fingerprint="sha256:" + "0" * 64,
        inbox=ctl.CommandInbox(state / "inbox"))
    c.trainer_rebuilder = object()
    rc_cli = ctl.main(["--state-dir", str(state), "rebuild-trainer", "rb1", "--expected-epoch", "0",
                       "--deadline-s", "900"])
    assert rc_cli == 0
    answers = c.poll_commands()
    assert answers and "rejected" not in answers[0], answers
    (Path(__file__).parent / "dryrun.json").write_text(json.dumps(
        {"argv": {f: val(f) for f in ("--actor-num-gpus-per-node", "--rollout-num-gpus", "--load")},
         "profile": profile.execution_mode, "answer": answers[0]}, indent=1, default=str))
