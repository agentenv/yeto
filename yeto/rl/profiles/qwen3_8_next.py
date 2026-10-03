"""Qwen3.8-Flash-Next (``qwen4_exp``) native per-expert LoRA GRPO profile.

M4 of NEXT-WEEK-PLAN: the 4-layer debug variant on one 8xH100 node, driven by
upstream Miles' own launcher (``scripts/run_qwen3_8_next.py``) with the LoRA
layout implemented by the Miles branch ``m3-qwen4exp-lora``
(``miles_plugins/models/qwen3_8_next/lora.py``).

Three things this module pins so the shell launchers and the CPU tests see
one source of truth:

* ``MODEL_ARGS`` -- a verbatim copy of ``scripts/models/qwen3.8-flash-next.py``
  rendered at 4 / 48 layers (``tests/test_rl_qwen3_8_next_profile.py``
  cross-checks it against a Miles checkout when one is present);
* ``LORA_TARGET_MODULES`` -- the 14 HF leaf names the native plugin requires
  (``_REQUIRED_TARGET_SUFFIXES``; any other set is rejected at argument
  parsing), under the ``model.language_model.layers.*`` prefix;
* the convert / launch command renderers.

Nothing here imports Miles.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
from collections.abc import Sequence

PROFILE_NAME = "qwen3_8_next_4layer_lora"

HF_REPO_FULL = "Qwen/Qwen3.8-Flash-Next"
HF_REPO_4LAYER = "CharyZeng/Qwen3.8-Flash-Next-4layer"
# huggingface.co/api/models/CharyZeng/Qwen3.8-Flash-Next-4layer (2026-10-01):
# 22 files, 30.52 GB, text_config.num_hidden_layers=4,
# layer_types=[linear_attention x3, full_attention].
HF_REVISION_4LAYER = "d19a6b60c0df8f90faf92c7c592b37df2e15b060"
HF_SIZE_GB_4LAYER = 30.5
# huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next (2026-10-01): 144 files, 360 GB.
HF_REVISION_FULL = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
HF_SIZE_GB_FULL = 360.0

# scripts/run_qwen3_8_next.py::_MODEL_REGISTRY -- ``megatron_model_type`` picks
# the model-args file and the ``<ckpt_dir>/<type>_torch_dist`` directory.
MODEL_NAMES = {
    "full": "Qwen3.8-Flash-Next",
    "4layer": "Qwen3.8-Flash-Next-4layer",
}
MEGATRON_MODEL_TYPES = {
    "full": "qwen3.8-flash-next",
    "4layer": "qwen3.8-flash-next-4layer",
}
NUM_LAYERS = {"full": 48, "4layer": 4}

HF_LAYER_PREFIX = "model.language_model.layers.*."
_GDN_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
# miles_plugins/models/qwen3_8_next/lora.py::_REQUIRED_TARGET_SUFFIXES (M3).
LORA_TARGET_SUFFIXES: tuple[str, ...] = (
    *(f"linear_attn.{name}" for name in _GDN_PROJECTIONS),
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)
LORA_TARGET_MODULES: tuple[str, ...] = tuple(HF_LAYER_PREFIX + s for s in LORA_TARGET_SUFFIXES)
# Deferred on both sides (M3 user ruling); the trainer rejects it explicitly.
LORA_DEFERRED_TARGET = HF_LAYER_PREFIX + "self_attn.indexer.index_qk_proj"

# sglang ``lora_target_modules`` leaf names after Miles' normalisation (M3
# design contract); read-back ``lora:*`` keys per engine for the 4-layer
# variant: 3 GDN layers x 10 + 1 QSA layer x 8 + 4 MoE layers x 4 = 54.
SGLANG_LORA_LEAVES = frozenset(
    {"in_proj_qkvz", "in_proj_ba", "out_proj", "qkv_proj", "o_proj", "gate_up_proj", "down_proj"}
)
EXPECTED_LORA_KEYS_4LAYER = 54

# M3 acceptance #1: trainable LoRA parameters for the 4-layer variant,
# rank r (attention / shared / GDN) and r_e (routed experts).
_GDN_PARAMS_PER_RANK = 35424  # per GDN layer, x3 layers
_QSA_PARAMS_PER_RANK = 29696  # per QSA layer, x1 layer
_SHARED_PARAMS_PER_RANK = 7040  # per MoE layer, x4 layers
_EXPERT_PARAMS_PER_RANK = 512 * 7040  # per MoE layer, x4 layers


def expected_trainable_params_4layer(lora_rank: int, lora_expert_rank: int) -> int:
    return (
        _GDN_PARAMS_PER_RANK * lora_rank * 3
        + _QSA_PARAMS_PER_RANK * lora_rank
        + _SHARED_PARAMS_PER_RANK * lora_rank * 4
        + _EXPERT_PARAMS_PER_RANK * lora_expert_rank * 4
    )


def expected_rank_trainable_4layer(lora_rank: int, lora_expert_rank: int, num_gpus: int) -> tuple[int, int]:
    """Per-rank ``trainable=`` values logged by ``apply_qwen3_8_next_lora`` (T2-S7 F2).

    The trainer log sums the *local* ``model.parameters()`` of one rank under
    the 4-layer layout TP2 / PP2 / EP=num_gpus/2 / ETP1: PP stage 0 holds
    layers 0-1 (GDN, GDN), stage 1 holds layers 2-3 (GDN, QSA).  QSA keeps a
    single ``qkv_lora_A`` (the HF export fans it out to q/k/v), TP2 halves
    ``qkv_lora_B`` / ``o_lora_A`` / ``fc1_lora_B`` / ``fc2_lora_A``, GDN LoRA is
    replicated, routed experts are sharded by EP only.  Returns (stage0, stage1).
    """
    if num_gpus not in (4, 8):
        raise ValueError(f"the 4-layer layout is validated on 4 or 8 GPUs, got {num_gpus}")
    r, r_e, ep = lora_rank, lora_expert_rank, num_gpus // 2
    gdn = _GDN_PARAMS_PER_RANK * r
    qsa = (2560 + 13312 // 2 + 6144 // 2 + 2560) * r  # qkv A | qkv B/2 | o A/2 | o B
    shared = (2560 + 2 * 640 // 2 + 640 // 2 + 2560) * r  # fc1 A | fc1 B/2 | fc2 A/2 | fc2 B
    experts = (512 // ep) * 7040 * r_e
    stage0 = 2 * (gdn + shared + experts)
    stage1 = gdn + qsa + 2 * (shared + experts)
    return stage0, stage1


def _moe_layer_freq(nlayers: int) -> str:
    # model_args_utils.moe_layer_freq(nlayers=nlayers, first_k_dense_replace=0)
    return "[" + ",".join(["1"] * nlayers) + "]"


def model_args(variant: str = "4layer") -> tuple[str, ...]:
    """Verbatim ``scripts/models/qwen3.8-flash-next.py::model_args(nlayers)`` tokens."""
    nlayers = NUM_LAYERS[variant]
    text = (
        "--spec miles_plugins.models.qwen3_8_next.qwen3_8_next get_qwen3_8_next_spec "
        "--disable-bias-linear "
        "--qk-layernorm "
        "--group-query-attention "
        "--num-attention-heads 24 "
        "--num-query-groups 2 "
        "--kv-channels 256 "
        f"--num-layers {nlayers} "
        "--hidden-size 2560 "
        "--ffn-hidden-size 640 "
        "--normalization RMSNorm "
        "--apply-layernorm-1p "
        "--position-embedding-type rope "
        "--norm-epsilon 1e-6 "
        "--rotary-percent 0.25 "
        "--swiglu "
        "--untie-embeddings-and-output-weights "
        "--vocab-size 248320 "
        "--rotary-base 10000000 "
        "--moe-ffn-hidden-size 640 "
        "--moe-shared-expert-intermediate-size 640 "
        "--moe-router-score-function softmax "
        "--moe-token-dispatcher-type alltoall "
        "--moe-router-topk 10 "
        f"--moe-layer-freq {_moe_layer_freq(nlayers)} "
        "--num-experts 512 "
        "--moe-grouped-gemm "
        "--moe-token-drop-policy probs "
        "--moe-router-dtype fp32 "
        "--moe-permute-fusion "
        "--moe-aux-loss-coeff 0 "
        "--attention-output-gate "
        "--moe-shared-expert-gate "
    )
    return tuple(text.split())


@dataclasses.dataclass(frozen=True)
class Qwen38NextLoraProfile:
    """Everything the one-command launcher needs, with the Miles CI 4-layer defaults."""

    name: str = PROFILE_NAME
    variant: str = "4layer"
    hf_repo: str = HF_REPO_4LAYER
    hf_revision: str = HF_REVISION_4LAYER
    # container-side layout, identical to Miles' run_qwen3_8_next.py defaults
    model_dir: str = "/root/models"
    ckpt_dir: str = "/root/ckpt"
    data_dir: str = "/root/datasets"
    save_dir: str = "/root/shared_data"
    miles_root: str = "/root/miles"
    megatron_path: str = "/root/Megatron-LM"
    dataset: str = "zhuzilin/dapo-math-17k"
    # held-out eval set (T2-S7 F5; Miles gemma recipe source, prompt/label keys)
    eval_dataset: str = "zhuzilin/aime-2024"
    # Miles' recipe defaults to disk offload; CI and this profile use cpu so the
    # per-rank train state never hits the (small, slow) container disk.
    offload_train_target: str = "cpu"
    # node shape (run_qwen3_8_next.py asserts 4 or 8 GPUs for the 4-layer layout)
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    # conversion layout (tests/e2e/.../test_qwen3_8_next_4layer_ci.py::prepare);
    # the torch_dist output re-shards at load so it need not match training.
    convert_tp: int = 2
    convert_pp: int = 1
    # LoRA layout (M3): single serving rank r, routed experts at r_e <= r
    # (zero-padded on export; r_e < r exercises M3/M4 acceptance #4).
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_expert_rank: int = 8
    lora_dropout: float = 0.0
    # run shape; run_id (Miles --run-id) fixes <save_dir>/<run_id>/checkpoints so a
    # restart can point --lora-adapter-path at a known iter (T2-S7 F4); "" = Miles default
    run_id: str = ""
    num_rollout: int = 5
    rollout_max_response_len: int = 512
    save_interval: int = 10
    skip_saving: bool = False
    enable_r3: bool = True

    def __post_init__(self) -> None:
        if self.variant not in MODEL_NAMES:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {sorted(MODEL_NAMES)}")
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
        # miles/utils/lora/arguments.py::validate_lora_args: 0 (follow) <= r_e <= r
        if not 0 <= self.lora_expert_rank <= self.lora_rank:
            raise ValueError(
                f"lora_expert_rank must satisfy 0 <= r_e <= lora_rank "
                f"({self.lora_expert_rank} vs {self.lora_rank})"
            )
        total = self.num_nodes * self.num_gpus_per_node
        if self.variant == "4layer" and total not in (4, 8):
            raise ValueError(f"the 4-layer layout is validated on 4 or 8 GPUs, got {total}")
        if self.variant == "full" and total != 32:
            raise ValueError(f"the full-model layout is validated on 32 GPUs, got {total}")

    # ---- derived paths -------------------------------------------------
    @property
    def model_name(self) -> str:
        return MODEL_NAMES[self.variant]

    @property
    def megatron_model_type(self) -> str:
        return MEGATRON_MODEL_TYPES[self.variant]

    @property
    def hf_checkpoint(self) -> str:
        return f"{self.model_dir}/{self.model_name}"

    @property
    def torch_dist(self) -> str:
        return f"{self.ckpt_dir}/{self.megatron_model_type}_torch_dist"

    @property
    def effective_expert_rank(self) -> int:
        return self.lora_expert_rank or self.lora_rank

    # ---- parallel layout (run_qwen3_8_next.py::_train) --------------------
    @property
    def parallel(self) -> dict[str, int]:
        num_gpus = self.num_nodes * self.num_gpus_per_node
        pp, engine = (8, 8) if self.variant == "full" else (2, 4)
        return {
            "tp": 2,
            "pp": pp,
            "cp": 1,
            "ep": num_gpus // pp,
            "etp": 1,
            "rollout_num_gpus_per_engine": engine,
            "sglang_tp": engine,
            "sglang_ep": engine,
        }

    # ---- commands ---------------------------------------------------------
    def download_commands(self) -> list[list[str]]:
        return [
            ["mkdir", "-p", self.model_dir, self.ckpt_dir, self.data_dir],
            ["hf", "download", self.hf_repo, "--revision", self.hf_revision, "--local-dir", self.hf_checkpoint],
            [
                "hf", "download", "--repo-type", "dataset", self.dataset,
                "--local-dir", f"{self.data_dir}/{self.dataset.split('/')[1]}",
            ],
            [
                "hf", "download", "--repo-type", "dataset", self.eval_dataset,
                "--local-dir", f"{self.data_dir}/{self.eval_dataset.split('/')[1]}",
            ],
        ]

    def convert_command(
        self,
        *,
        hf_checkpoint: str | None = None,
        output: str | None = None,
        nproc: int | None = None,
    ) -> list[str]:
        """``torchrun tools/convert_hf_to_torch_dist.py`` exactly as Miles CI invokes it.

        ``CONVERT_KEEP_PP1=1`` (set by the shell launcher) keeps the explicit
        PP=1 instead of the tool's auto pipeline split.
        """
        return [
            "torchrun",
            "--nproc-per-node", str(nproc or self.num_gpus_per_node),
            f"{self.miles_root}/tools/convert_hf_to_torch_dist.py",
            *model_args(self.variant),
            "--hf-checkpoint", hf_checkpoint or self.hf_checkpoint,
            "--save", output or self.torch_dist,
            "--tensor-model-parallel-size", str(self.convert_tp),
            "--pipeline-model-parallel-size", str(self.convert_pp),
        ]

    def convert_env(self) -> dict[str, str]:
        return {
            "CONVERT_KEEP_PP1": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONPATH": f"{self.miles_root}:{self.megatron_path}",
        }

    def lora_extra_args(self) -> list[str]:
        """The Miles ``train.py`` flags that turn the stock recipe into native LoRA."""
        return [
            "--megatron-to-hf-mode", "raw",
            "--lora-rank", str(self.lora_rank),
            "--lora-alpha", str(self.lora_alpha),
            "--lora-dropout", format(self.lora_dropout, "g"),
            "--lora-expert-rank", str(self.lora_expert_rank),
            "--target-modules", ",".join(LORA_TARGET_MODULES),
            # grouped-GEMM LoRA has no fused grad-accumulation path (same as Kimi-K3)
            "--no-gradient-accumulation-fusion",
            # colocated: host mirror of the rollout base so the per-round release
            # does not re-ship the base weights (run_kimi_k3.py)
            "--lora-base-cpu-backup",
            "--check-lora-weight-equal",
            "--sglang-lora-backend", "triton",
            "--sglang-lora-strict-loading",
            "--sglang-max-lora-rank", str(self.lora_rank),
            "--offload-train-target", self.offload_train_target,
        ]

    def launcher_command(self, *, extra_args: Sequence[str] = ()) -> list[str]:
        """``python scripts/run_qwen3_8_next.py train ...`` (needs MILES_SCRIPT_EXTERNAL_RAY=1)."""
        extra = [*self.lora_extra_args(), *extra_args]
        # run_qwen3_8_next.py registers a single typer command, which typer
        # flattens: passing the documented `train` word is rejected by the image
        # ("Got unexpected extra argument(s) (train)", T2-S7 G3 attempt 1).
        cmd = [
            "python3", f"{self.miles_root}/scripts/run_qwen3_8_next.py",
            "--model-name", self.model_name,
            "--num-nodes", str(self.num_nodes),
            "--num-gpus-per-node", str(self.num_gpus_per_node),
            "--model-dir", self.model_dir,
            "--ckpt-dir", self.ckpt_dir,
            "--data-dir", self.data_dir,
            "--save-dir", self.save_dir,
            "--megatron-path", self.megatron_path,
            "--num-rollout", str(self.num_rollout),
            "--rollout-max-response-len", str(self.rollout_max_response_len),
            *(["--run-id", self.run_id] if self.run_id else []),
            # full-weight equality is meaningless under LoRA; the LoRA check is in extra args
            "--no-check-weight-update-equal",
            "--enable-r3" if self.enable_r3 else "--no-enable-r3",
            "--skip-saving" if self.skip_saving else "--no-skip-saving",
            "--extra-args", " ".join(shlex.quote(t) for t in extra),
        ]
        return cmd

    def launcher_env(self) -> dict[str, str]:
        return {
            "MILES_SCRIPT_EXTERNAL_RAY": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTHONPATH": f"{self.miles_root}:{self.megatron_path}",
            "TRITON_CACHE_DIR": "/tmp/triton_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
        }

    def expected_lora_keys(self) -> int:
        if self.variant != "4layer":
            raise ValueError("lora key count is only pinned for the 4-layer variant")
        return EXPECTED_LORA_KEYS_4LAYER

    def manifest(self) -> dict[str, object]:
        return {
            "profile": self.name,
            "variant": self.variant,
            "hf_repo": self.hf_repo,
            "hf_revision": self.hf_revision,
            "hf_checkpoint": self.hf_checkpoint,
            "torch_dist": self.torch_dist,
            "megatron_model_type": self.megatron_model_type,
            "parallel": self.parallel,
            "expected_rank_trainable": list(
                expected_rank_trainable_4layer(
                    self.lora_rank, self.effective_expert_rank, self.num_nodes * self.num_gpus_per_node
                )
            ) if self.variant == "4layer" else None,
            "lora": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "expert_rank": self.effective_expert_rank,
                "target_modules": list(LORA_TARGET_MODULES),
            },
        }


def profile_from_env(env: dict[str, str]) -> Qwen38NextLoraProfile:
    """Override profile fields with ``YETO_Q38N_<FIELD>`` environment variables."""
    kwargs: dict[str, object] = {}
    for field in dataclasses.fields(Qwen38NextLoraProfile):
        raw = env.get(f"YETO_Q38N_{field.name.upper()}")
        if raw is None:
            continue
        if field.type == "int":
            kwargs[field.name] = int(raw)
        elif field.type == "float":
            kwargs[field.name] = float(raw)
        elif field.type == "bool":
            kwargs[field.name] = raw.lower() in ("1", "true", "yes")
        else:
            kwargs[field.name] = raw
    return Qwen38NextLoraProfile(**kwargs)


def _quote(tokens: Sequence[str]) -> str:
    return " ".join(shlex.quote(t) for t in tokens)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m yeto.rl.profiles.qwen3_8_next")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("model-args", help="Megatron model args (shell-quoted, one line)")
    conv = sub.add_parser("convert-command", help="HF -> torch_dist conversion command")
    conv.add_argument("--hf-checkpoint")
    conv.add_argument("--output")
    conv.add_argument("--nproc", type=int)
    sub.add_parser("convert-env", help="KEY=VALUE lines for the conversion")
    launch = sub.add_parser("launch-command", help="Miles launcher command")
    launch.add_argument("extra", nargs="*")
    sub.add_parser("launch-env", help="KEY=VALUE lines for the launch")
    sub.add_parser("download-commands", help="one shell line per download step")
    sub.add_parser("manifest", help="profile identity as JSON")
    args = parser.parse_args(argv)
    import os

    profile = profile_from_env(dict(os.environ))
    out = sys.stdout
    if args.command == "model-args":
        out.write(_quote(model_args(profile.variant)) + "\n")
    elif args.command == "convert-command":
        out.write(
            _quote(profile.convert_command(hf_checkpoint=args.hf_checkpoint, output=args.output, nproc=args.nproc))
            + "\n"
        )
    elif args.command == "convert-env":
        out.writelines(f"{k}={v}\n" for k, v in profile.convert_env().items())
    elif args.command == "launch-command":
        out.write(_quote(profile.launcher_command(extra_args=args.extra)) + "\n")
    elif args.command == "launch-env":
        out.writelines(f"{k}={v}\n" for k, v in profile.launcher_env().items())
    elif args.command == "download-commands":
        out.writelines(_quote(c) + "\n" for c in profile.download_commands())
    elif args.command == "manifest":
        out.write(json.dumps(profile.manifest(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
