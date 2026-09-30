"""(INFRA-A a2-fix copy of gpu-b1 evidence/infra-v2-b1/a2/test_a2_dryrun.py: miles_args now
carries eval_temperature from the ev config, as the island's Miles namespace does; the
overlap profile refuses a non-greedy eval since the A2 fix.)

A2 (L-2.3) local end-to-end dry-run: real launch argv -> island run -> learner -> profile.

Run: PYTHONPATH=<repo>:<repo>/tests pytest -q <this file>. No cloud, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO / "tests"))
from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

HELDOUT = Path(__file__).with_name("heldout-gsm8k-test-first32.jsonl")
# identical to the Modal launch args (a2 runs), minus model/data ids the helper fixes
COMMON = ("--gpu", "modal:2xh100", "--modal-gpu-exact", "--total-steps", "3", "--seed", "17",
          "--rl-sync-preset", "strict-avg",  # eval needs the external policy boundary (no-sync refuses eval)
          "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
          "--rl-observe-timeline",
          "--rl-eval-interval", "1", "--rl-eval-data", str(HELDOUT),
          "--rl-eval-dataset-name", "gsm8k-test32", "--rl-eval-samples-per-prompt", "1",
          "--rl-eval-temperature", "0", "--rl-eval-max-response-len", "384")
ARMS = {"S": (), "O": ("--rl-overlap-eval",), "OD": ("--rl-overlap-eval",)}


def _provider():
    # Qwen3-0.6B shape (config.json): only what resolve_rl_run_config reads
    return SimpleNamespace(hidden_size=1024, num_attention_heads=16, num_layers=28,
                           ffn_hidden_size=3072, num_query_groups=8, kv_channels=128,
                           multi_latent_attention=False, num_moe_experts=None,
                           seq_length=40960, layernorm_epsilon=1e-6, rotary_base=1000000,
                           vocab_size=151936, max_position_embeddings=40960,
                           qk_layernorm=True, gated_linear_unit=True,
                           share_embeddings_and_output_weights=True, add_bias_linear=False,
                           add_qkv_bias=False, activation_func=None)


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm(arm, tmp_path, monkeypatch):
    from yeto.rl import learner
    from yeto.rl.engine.algorithm import resolve_ports_algorithm
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.run_config import _resolve_eval

    run = island_run(COMMON + ARMS[arm], monkeypatch)
    args, env = learner_from_run(run, tmp_path / "home")
    out = {"arm": arm}
    # launcher -> learner flags
    assert args.eval_interval == 1 and args.eval_temperature == 0.0
    assert args.rl_placement == "fixed-partition" and args.rl_observe_timeline
    assert bool(args.rl_overlap_eval) == (arm != "S")
    assert not args.rl_single_island_no_sync
    args.parameter_mode = "lora"
    assert learner._verify_eval_dataset_identity(args).read_text() == HELDOUT.read_text()
    ev = _resolve_eval(args, parameter_mode="lora", prompt_path="/p", eval_prompt_path="/e",
                       yeto_policy_sync=True)
    assert ev.temperature == 0.0 and ev.interval == 1
    out["eval"] = {"temperature": ev.temperature, "interval": ev.interval}
    # learner -> miles_args switches
    miles_args = SimpleNamespace(yeto_rl_learner_id=0, rollout_batch_size=4,
                                 n_samples_per_prompt=8, num_steps_per_rollout=1,
                                 eval_interval=1, eval_uses_snapshots=False,
                                 eval_temperature=ev.temperature, rollout_temperature=1.0)
    learner.apply_ports_infra_switches(args, miles_args, {})
    assert getattr(miles_args, "yeto_rl_observe_timeline", False) is True
    # profile: fixed-partition -> partitioned-serial / partitioned-overlap
    launch = SimpleNamespace(placement=SimpleNamespace(kind="fixed-partition"), argv=())
    algorithm = resolve_ports_algorithm(args, rl_engine="ports")
    profile = entry.execution_profile_for(miles_args, launch, algorithm, yeto_policy_sync=True,
                                          expected_sha256=args.rl_expected_algorithm_sha256)
    want = "partitioned-serial" if arm == "S" else "partitioned-overlap"
    assert profile.execution_mode == want
    # run config -> Miles argv (translate_run_config), as the learner builds it
    from yeto.rl.engine.run_config import resolve_rl_run_config
    rc = resolve_rl_run_config(args, model_path="/m", rollout_model_path=None, prompt_path="/p",
                               eval_prompt_path="/e", provider=_provider(),
                               target_modules=["q_proj"], yeto_policy_sync=True)
    ml = learner.build_ports_launch(args, rc, ())
    argv = list(ml.argv)
    def val(flag):
        return argv[argv.index(flag) + 1] if flag in argv else None
    assert val("--eval-interval") == "1" and float(val("--eval-temperature")) == 0.0
    assert val("--rollout-num-gpus") == "1" and ml.placement.kind == "fixed-partition"
    out["miles_argv_eval"] = {f: val(f) for f in ("--eval-interval", "--eval-temperature",
                              "--eval-max-response-len", "--n-samples-per-eval-prompt",
                              "--rollout-num-gpus", "--actor-num-gpus-per-node")}
    out["profile"] = {"mode": profile.execution_mode, "overlap": sorted(profile.allowed_overlap)}
    (Path(__file__).parent / f"dryrun-{arm}.json").write_text(json.dumps(out, indent=1, default=str))
