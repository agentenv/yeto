"""A5 (3.8 X6) local end-to-end dry-run: two 3-GPU islands (T1R1S1<->T1R2S0) + local head,
strict-avg, --rl-elastic on both, declare-cells, quorum/pause-margin, start-delay injection.
Both islands' run commands go through the island prelude under a temp HOME and the real
learner.parse_args; the syncer command must carry --quorum-timeout-s. No cloud, no GPU."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "tests"))
from rl_e2e_launch import learner_from_run  # noqa: E402

RES = Path(__file__).with_name("resources-3.json")
BASE = ("--gpu", "modal:3xh100,modal:3xh100", "--modal-gpu-exact", "--total-steps", "6",
        "--seed", "17", "--rl-sync-preset", "strict-avg",
        "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1", "--rl-standby-gpus", "1",
        "--rl-elastic", "--rl-elastic-declare-cells", "--rl-elastic-cells", "c0,c1",
        "--rl-elastic-resources", str(RES), "--rl-elastic-initial-config", "T1R1S1",
        "--rl-observe-timeline", "--modal-retries", "0", "--modal-timeout-s", "3900")
CASES = {
    "baseline": (),
    "switch": (),
    "quorum": ("--rl-elastic-quorum-timeout-s", "120", "--rl-elastic-pause-margin", "2.0",
               "--rl-test-inject-start-delay-s", "150"),
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


def _tasks(args, monkeypatch):
    from tests.test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import make_miles_island_task
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    specs = parse_gpu_spec(args.gpu)
    return [make_miles_island_task(args, s, i, len(specs), "127.0.0.1:29400").run
            for i, s in enumerate(specs)]


@pytest.mark.parametrize("case", sorted(CASES))
def test_case(case, tmp_path, monkeypatch):
    from test_rl_engine_selection import _cli
    from yeto.launcher import _prepare_rl_args, syncer_command
    from yeto.rl import learner
    from yeto.rl.engine.run_config import resolve_rl_run_config

    args = _cli(BASE + CASES[case])
    _prepare_rl_args(args)
    runs = _tasks(args, monkeypatch)
    assert len(runs) == 2
    out = {"islands": []}
    for i, run in enumerate(runs):
        largs, env = learner_from_run(run, tmp_path / f"home{i}")
        assert largs.rl_elastic and largs.rl_elastic_declare_cells and not largs.rl_single_island_no_sync
        rc = resolve_rl_run_config(largs, model_path="/m", rollout_model_path=None,
                                   prompt_path="/p", eval_prompt_path=None, provider=_provider(),
                                   target_modules=["q_proj"], yeto_policy_sync=True)
        argv = list(learner.build_ports_launch(largs, rc, ()).argv)
        pm = json.loads(argv[argv.index("--yeto-placement-map") + 1])
        assert [(c["name"], c["bundles"], c["start"]) for c in pm["rollout_cells"]] == [
            ("c0", [1], True), ("c1", [2], False)]
        assert "--use-miles-router" in argv
        isl = {"learner_id": largs.learner_id if hasattr(largs, "learner_id") else i,
               "placement_map": pm,
               "test_env": {k: v for k, v in env.items() if k.startswith("YETO_RL_TEST")}}
        if case == "quorum":
            assert largs.rl_elastic_quorum_timeout_s == 120 and largs.rl_elastic_pause_margin == 2.0
            assert float(env["YETO_RL_TEST_INJECT_START_DELAY_S"]) == 150.0
        if case == "fp":
            assert largs.rl_print_attestation_fingerprint
        out["islands"].append(isl)
    syn = syncer_command(args, 2)
    out["syncer"] = syn
    if case == "quorum":
        assert "--quorum-timeout-s 120" in syn
    else:
        assert "--quorum-timeout-s" not in syn
    (Path(__file__).parent / f"dryrun-{case}.json").write_text(json.dumps(out, indent=1))
