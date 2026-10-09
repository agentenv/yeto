"""Flash-Next (fnrun.sh fn32s / fn32b / fn8s / fn8r) variant of fp_local22.py: the ports-path Miles
argv + attestation ``runtime_fingerprint`` for the 4x8 H200 cases, on CPU.

usage: python fp_fn.py <repo> <fn32s|fn32b|fn8s|fn8r> [--seed N] [--total-steps N] [extra cli args]

The fingerprint is the pinned Miles commit + the FULL Miles argv, so it changes
with seed / total-steps / any flag: an E1 attestation is valid only for the exact
parameters it was generated with (re-run this script with the run's params).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
# node-side --hf-checkpoint: the snapshot_download() path.  The full model is on the
# model FS (HF_HUB_CACHE=/mnt/yeto-models/hub when its yeto-complete marker exists);
# the 4-layer one only if it was populated there too (FN_4L_HUB overrides: set it to
# /root/.cache/huggingface/hub when the run falls back to the Hub).
FS_HUB = "/mnt/yeto-models/hub"
FN_SNAPSHOT = (f"{FS_HUB}/models--Qwen--Qwen3.8-Flash-Next/snapshots/"
               "de4b8e4d43b917e7706784d8bb445c9af86a3540")
FN_4L_SNAPSHOT = (f"{os.environ.get('FN_4L_HUB', FS_HUB)}/models--CharyZeng--Qwen3.8-Flash-Next-4layer/"
                  "snapshots/d19a6b60c0df8f90faf92c7c592b37df2e15b060")


def fnrun_cli(case: str, *, seed: int | None = None, total_steps: int | None = None,
              extra: tuple[str, ...] = ()) -> list[str]:
    """fnrun.sh's launch tokens (no cloud call), minus ``launch``, with overrides.
    BOOT_ONLY is dropped: the fingerprint is always the training (fna) argv."""
    env = {k: v for k, v in os.environ.items() if k not in ("ATTEST", "COSTS", "IMAGE", "BOOT_ONLY")}
    if total_steps is not None:
        env["STEPS"] = str(total_steps)
    out = subprocess.run(["bash", os.path.join(HERE, "fnrun.sh"), case], env=env, check=True,
                         capture_output=True, text=True).stdout
    toks = shlex.split(out)
    assert toks[0] == "launch", toks[:1]
    toks = toks[1:]
    if seed is not None:
        toks[toks.index("--seed") + 1] = str(seed)
    return toks + list(extra)


def fn_provider(num_layers: int = 48):
    class Qwen4ExpModelProvider(SimpleNamespace):
        pass

    # text_config of Qwen/Qwen3.8-Flash-Next (profiles/qwen3_8_next.model_args("full"));
    # the 4-layer slice differs only in num_layers (model_args("4layer"))
    return Qwen4ExpModelProvider(
        hidden_size=2560, num_attention_heads=24, num_layers=num_layers, ffn_hidden_size=640,
        num_query_groups=2, kv_channels=256, multi_latent_attention=False, num_moe_experts=512,
        moe_ffn_hidden_size=640, moe_router_topk=10, moe_layer_freq=1,
        moe_shared_expert_intermediate_size=640, experimental_attention_variant="gated_delta_net",
        seq_length=262144, layernorm_epsilon=1e-6, rotary_base=10000000, rotary_percent=0.25,
        vocab_size=248320, max_position_embeddings=262144, qk_layernorm=True,
        gated_linear_unit=True, share_embeddings_and_output_weights=False, add_bias_linear=False,
        add_qkv_bias=False, activation_func=None)


def fn_fingerprint(repo: str, case: str, *, seed: int | None = None,
                   total_steps: int | None = None, extra: tuple[str, ...] = ()) -> dict:
    for p in (repo, repo + "/tests"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import _pytest.monkeypatch as _m

    from rl_e2e_launch import island_run, learner_from_run
    from yeto.rl import learner
    from yeto.rl.adapters.miles.entry import ports_runtime_fingerprint
    from yeto.rl.engine.run_config import resolve_rl_run_config
    from yeto.rl.engine import run_config
    from yeto.rl.profiles import qwen3_8_next as q

    small = case in ("fn8s", "fn8r")
    mp = _m.MonkeyPatch()
    try:
        run = island_run(tuple(fnrun_cli(case, seed=seed, total_steps=total_steps, extra=extra)), mp)
        tmp = Path(tempfile.mkdtemp())
        args, _env = learner_from_run(run, tmp / "home")
        # the torch_dist dir exists only on the node's model FS: keep the path verbatim
        # (the node resolves the same absolute non-symlink path)
        mp.setattr(run_config, "_resolve_ref_load",
                   lambda a, model_path: a.megatron_ref_load or str(model_path))
        rc = resolve_rl_run_config(
            args, model_path=FN_4L_SNAPSHOT if small else FN_SNAPSHOT, rollout_model_path=None,
            prompt_path="/root/yeto-rl/prompts.jsonl", eval_prompt_path=None,
            provider=fn_provider(4 if small else 48),
            target_modules=sorted({m.rsplit(".", 1)[-1] for m in q.LORA_TARGET_MODULES}),
            yeto_policy_sync=False)
    finally:
        mp.undo()
    pl = learner.build_ports_launch(args, rc, ())
    return {"case": case, "recipe": rc.model_recipe.name, "fp": ports_runtime_fingerprint(pl),
            "argv": list(pl.argv)}


def main(argv: list[str]) -> int:
    repo, case, rest = argv[0], argv[1], list(argv[2:])
    kw = {}
    for flag, key in (("--seed", "seed"), ("--total-steps", "total_steps")):
        if flag in rest:
            i = rest.index(flag)
            kw[key] = int(rest[i + 1])
            del rest[i:i + 2]
    print(json.dumps(fn_fingerprint(repo, case, extra=tuple(rest), **kw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
