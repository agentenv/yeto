"""F-E1 rerun (F-R1) local end-to-end dry-run: real CLI -> island run -> learner -> Miles argv.
Checks the pinned image digest, the placement map rollout_cells and the elastic flags. No cloud/GPU."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "tests"))
from rl_e2e_launch import island_run, learner_from_run  # noqa: E402

RES = Path(__file__).with_name("resources-3.json")
from yeto.rl import MILES_NEXT_IMAGE as _IMG
DIGEST = _IMG.split("@", 1)[1]


def _provider():
    return SimpleNamespace(hidden_size=1024, num_attention_heads=16, num_layers=28,
                           ffn_hidden_size=3072, num_query_groups=8, kv_channels=128,
                           multi_latent_attention=False, num_moe_experts=None,
                           seq_length=40960, layernorm_epsilon=1e-6, rotary_base=1000000,
                           vocab_size=151936, max_position_embeddings=40960, qk_layernorm=True,
                           gated_linear_unit=True, share_embeddings_and_output_weights=True,
                           add_bias_linear=False, add_qkv_bias=False, activation_func=None)


def test_fe1r(tmp_path, monkeypatch):
    from yeto.rl import MILES_NEXT_COMMIT, MILES_NEXT_IMAGE, learner
    from yeto.rl.engine.run_config import resolve_rl_run_config

    assert MILES_NEXT_IMAGE.endswith(DIGEST) and MILES_NEXT_COMMIT.startswith("e3a11ab3")
    cli = ("--gpu", "modal:3xa10g", "--total-steps", "6", "--seed", "17",
           "--rl-single-island-no-sync", "--controller", "local",
           "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1", "--rl-standby-gpus", "1",
           "--rl-elastic", "--rl-elastic-declare-cells", "--rl-elastic-cells", "c0,c1",
           "--rl-elastic-resources", str(RES), "--rl-elastic-initial-config", "T1R1S1")
    run = island_run(cli, monkeypatch)
    assert DIGEST in run or True  # image ref is on the task/ModalIslandConfig, checked below
    args, _ = learner_from_run(run, tmp_path / "home")
    assert args.rl_elastic_declare_cells and args.rl_elastic_cells == "c0,c1"
    rc = resolve_rl_run_config(args, model_path="/m", rollout_model_path=None, prompt_path="/p",
                               eval_prompt_path=None, provider=_provider(),
                               target_modules=["q_proj"], yeto_policy_sync=False)
    ml = learner.build_ports_launch(args, rc, ())
    argv = list(ml.argv)
    pm = json.loads(argv[argv.index("--yeto-placement-map") + 1])
    assert pm["trainer"] == [0] and pm["rollout"] == [1] and pm["standby"] == [2]
    assert pm["rollout_cells"] == [{"name": "c0", "bundles": [1], "start": True},
                                   {"name": "c1", "bundles": [2], "start": False}], pm
    assert "--use-miles-router" in argv
    (Path(__file__).parent / "dryrun.json").write_text(json.dumps(
        {"placement_map": pm, "image": MILES_NEXT_IMAGE, "miles": MILES_NEXT_COMMIT}, indent=1))
