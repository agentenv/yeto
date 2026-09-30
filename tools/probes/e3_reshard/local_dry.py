"""Local pre-GPU dry-run: the SAME argument construction and check the container runs first.

    python local_dry.py {dev-gather|a8} <learner_flags.txt> [<out.json>]

Builds the Miles argv exactly like the container does -- ``learner.parse_args``
on the dry-run learner flags, ``resolve_rl_run_config``, ``build_ports_launch``
-- except that the Megatron-Bridge model provider is a stub with the public
Qwen3-0.6B dimensions (only model-shape flags depend on it; none of the
checked flags do), then runs ``learner_shim.argv_check`` with the profile
overrides of ``modal_run.OVERRIDES``. ``modal_run.main`` refuses to start a
Sandbox unless this passes. The container repeats the identical check on the
real argv and additionally checks the parsed Miles args and their agreement
with the argv (``learner_shim.phase_summary``).

The Miles argv is the production trainer-edge translation
(``RLRunConfig.trainer_dp_edges=True``, set by ``--rl-elastic-trainer-edges``),
so A8's configuration equals the production trainer-edge configuration.

Why the first DEV-GATHER needed this: the old local step only ran
``yeto launch --dry-run`` (the learner command line); the Miles argv and the
reshard checks were first built inside the container, where the hard-coded
``--balance-data`` of the ports translation (config.py) was refused.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Public Qwen3-0.6B config (HF c1899de2): model-shape flags only.
QWEN3_0_6B_PROVIDER = dict(
    hidden_size=1024, num_attention_heads=16, num_query_groups=8, num_layers=28, ffn_hidden_size=3072,
    kv_channels=128, vocab_size=151936, seq_length=40960, gated_linear_unit=True, layernorm_epsilon=1e-6,
    rotary_base=1000000, share_embeddings_and_output_weights=True, qk_layernorm=True,
    position_embedding_type="rope", normalization="RMSNorm", add_bias_linear=False, add_qkv_bias=False,
    make_vocab_size_divisible_by=128, max_position_embeddings=40960, rotary_percent=1.0,
    apply_rope_fusion=False, bias_activation_fusion=True, untie_embeddings_and_output_weights=False,
)
TARGETS = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]


def learner_argv(flags_text: str) -> list[str]:
    text = flags_text.replace("$LEARNER_ID", "0")
    return [os.path.expanduser(t) for t in shlex.split(text)]


def build_launch(flags_text: str):
    from yeto.rl import learner
    from yeto.rl.engine.run_config import resolve_rl_run_config

    from learner_shim import with_trainer_edges

    args = learner.parse_args(learner_argv(flags_text))
    config = with_trainer_edges(resolve_rl_run_config)(
        args, model_path="/local-dry/model", prompt_path="/local-dry/prompts.jsonl",
        provider=SimpleNamespace(**QWEN3_0_6B_PROVIDER), target_modules=list(TARGETS),
        yeto_policy_sync=not getattr(args, "rl_single_island_no_sync", False))
    return learner.build_ports_launch(args, config)


def local_dry(profile: str, flags_text: str) -> dict:
    from learner_shim import argv_check, parse_overrides, summary_problems
    from modal_run import OVERRIDES, REQUIRED_ARGV

    launch = build_launch(flags_text)
    summary = argv_check(list(launch.argv), launch.algorithm, parse_overrides(OVERRIDES[profile]))
    summary["problems"] = summary_problems(summary)
    summary["problems"] += [f"argv lacks {flag}" for flag in REQUIRED_ARGV[profile] if flag not in launch.argv]
    if "--balance-data" in launch.argv:
        summary["problems"].append("--balance-data present: not the production trainer-edge translation")
    summary["argv"] = list(launch.argv)
    return summary


def main(argv: list[str]) -> int:
    summary = local_dry(argv[0], Path(argv[1]).read_text())
    if len(argv) > 2:
        Path(argv[2]).write_text(json.dumps(summary, indent=1, sort_keys=True, default=repr))
    print(json.dumps({"problems": summary["problems"], "argv_profile": summary["argv_profile"]}, default=repr))
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
