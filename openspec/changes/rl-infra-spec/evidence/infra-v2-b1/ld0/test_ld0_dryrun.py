"""L-D0 local end-to-end dry-run: --rl-deterministic-trainer from the real launch argv to the
learner, the Miles argv (--deterministic-mode) and the env that connect_island_ray forwards."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "tests"))
from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

CLI = ("--gpu", "modal:2xh100", "--modal-gpu-exact", "--total-steps", "3", "--seed", "17",
       "--rl-sync-preset", "strict-avg", "--rl-placement", "fixed-partition",
       "--rl-rollout-gpus", "1", "--rl-deterministic-trainer", "--rl-observe-timeline",
       "--modal-retries", "0", "--modal-timeout-s", "2100")


def _provider():
    return SimpleNamespace(hidden_size=1024, num_attention_heads=16, num_layers=28,
                           ffn_hidden_size=3072, num_query_groups=8, kv_channels=128,
                           multi_latent_attention=False, num_moe_experts=None,
                           seq_length=40960, layernorm_epsilon=1e-6, rotary_base=1000000,
                           vocab_size=151936, max_position_embeddings=40960, qk_layernorm=True,
                           gated_linear_unit=True, share_embeddings_and_output_weights=True,
                           add_bias_linear=False, add_qkv_bias=False, activation_func=None)


def test_ld0(tmp_path, monkeypatch):
    from yeto.rl import learner
    from yeto.rl.engine.miles_adapter.entry import DETERMINISM_ENV
    from yeto.rl.engine.run_config import resolve_rl_run_config

    args, _ = learner_from_run(island_run(CLI, monkeypatch), tmp_path / "home")
    assert args.rl_deterministic_trainer and args.eval_interval is None
    rc = resolve_rl_run_config(args, model_path="/m", rollout_model_path=None, prompt_path="/p",
                               eval_prompt_path=None, provider=_provider(),
                               target_modules=["q_proj"], yeto_policy_sync=True)
    argv = list(learner.build_ports_launch(args, rc, ()).argv)
    assert "--deterministic-mode" in argv
    env = {}
    learner.apply_ports_infra_switches(args, SimpleNamespace(yeto_rl_learner_id=0), env)
    assert env == {**env, **DETERMINISM_ENV} and env["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    (Path(__file__).parent / "dryrun.json").write_text(json.dumps(
        {"env": env, "deterministic_mode_in_argv": True,
         "sglang_deterministic": "--sglang-enable-deterministic-inference" in argv}, indent=1))
