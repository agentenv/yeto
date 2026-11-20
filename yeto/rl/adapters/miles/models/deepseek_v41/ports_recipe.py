"""Ports Miles argv of the DeepSeek-V4.1-Flash LoRA recipe.

Model arguments: Miles ``scripts/models/deepseek-v4.1.py`` (trainer image Miles
f61eb35d); trainer flags: ``scripts/run_deepseek_v4_1.py`` (``--train-mode lora``).
The trainer runs Miles' own V4.1 implementation (``--dsv4-impl miles``, the
``get_dsv41_spec`` plugin spec, bshd), reads the BF16 HF checkpoint through the
``deepseekv41`` mbridge converter (``--megatron-to-hf-mode raw``) and trains a
native LoRA adapter (``canonical_lora``; Miles
``lora/native/registry.py`` ``deepseek_v41``) on the attention projections
(``wq_a/wq_b/wkv/wo_a/wo_b``) and optionally the shared expert (``w1/w2/w3``).
Routed experts (``--exclude-modules "*.ffn.experts.*"``), the router
(``--moe-router-freeze-gate``, ``--freeze-e-score-correction-bias``), the
compressor, the indexer and engram stay frozen. Full uniform recompute is always
on (the recipe's validated memory layout).

Only DeepSeek-V4.1-Flash geometry is accepted (5120 hidden, 64 heads, 384 routed
experts top-6, 129280 vocab); the layer count and the CSA compress ratios follow
the checkpoint (a cut-down smoke model passes fewer layers).
"""

from __future__ import annotations

from collections.abc import Sequence

from yeto.rl.adapters.miles.models.glm5_next import _flag_groups

MODEL_TYPE = "deepseek_v41"
MODEL_NAME = "deepseekv41"  # Miles --model-name (mbridge plugin)
MEGATRON_TO_HF_MODE = "raw"
SPEC = ("miles_plugins.models.deepseek_v4_1.deepseek_v4_1", "get_dsv41_spec")

NUM_LAYERS = 40
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEADS = 64
NUM_EXPERTS = 384
TOPK = 6
VOCAB_SIZE = 129280
MOE_FFN_HIDDEN_SIZE = 2304
# Miles scripts/models/deepseek-v4.1.py COMPRESS_RATIOS (40 layers)
COMPRESS_RATIOS: tuple[int, ...] = (0, 0) + (2,) * 18 + (1,) * 20

# yeto --lora-targets -> (Miles --target-modules groups)
TARGET_MODULES: dict[str, tuple[str, ...]] = {
    "attention": ("attn",),
    "auto": ("attn",),
    # attn + mlp: the mlp group names the routed experts too; --exclude-modules drops them,
    # leaving the shared expert (Miles run_deepseek_v4_1.py default target set)
    "all-linear": ("attn", "mlp"),
}
EXCLUDE_MODULES = "*.ffn.experts.*"


class DeepseekV41RecipeError(ValueError):
    """The requested run violates a DeepSeek-V4.1 recipe constraint."""


def target_modules(lora_targets: str | None) -> tuple[str, ...]:
    try:
        return TARGET_MODULES[str(lora_targets)]
    except KeyError:
        raise DeepseekV41RecipeError(
            f"--lora-targets {lora_targets}: the deepseek_v41 recipe adapts attention "
            "(attention/auto) or attention + the shared expert (all-linear); routed experts "
            "stay frozen") from None


def check_geometry(*, hidden_size: int, num_attention_heads: int, num_experts: int,
                   router_topk: int, vocab_size: int) -> None:
    got = {"hidden_size": hidden_size, "num_attention_heads": num_attention_heads,
           "num_experts": num_experts, "router_topk": router_topk, "vocab_size": vocab_size}
    want = {"hidden_size": HIDDEN_SIZE, "num_attention_heads": NUM_ATTENTION_HEADS,
            "num_experts": NUM_EXPERTS, "router_topk": TOPK, "vocab_size": VOCAB_SIZE}
    bad = {k: (got[k], want[k]) for k in want if int(got[k]) != want[k]}
    if bad:
        raise DeepseekV41RecipeError(
            "the deepseek_v41 recipe carries DeepSeek-V4.1-Flash model arguments; the checkpoint "
            "differs in " + ", ".join(f"{k}={g} (want {w})" for k, (g, w) in bad.items()))


def compress_ratios_for(num_layers: int, ratios: Sequence[int] | None = None) -> tuple[int, ...]:
    if ratios is not None:
        out = tuple(int(r) for r in ratios)
    elif int(num_layers) == NUM_LAYERS:
        out = COMPRESS_RATIOS
    else:
        out = COMPRESS_RATIOS[: int(num_layers)]
    if len(out) != int(num_layers):
        raise DeepseekV41RecipeError(f"{len(out)} compress ratios for {num_layers} layers")
    return out


def make_vocab_size_divisible_by(tensor_parallel: int) -> int:
    """Miles run_deepseek_v4_1.py: 64, or 256 // TP when 129280 is not a multiple of 64 * TP
    (TP8 -> 32)."""
    tp = int(tensor_parallel)
    return 256 // tp if VOCAB_SIZE % (64 * tp) else 64


def model_argv(*, num_layers: int = NUM_LAYERS, compress_ratios: Sequence[int] | None = None,
               tensor_parallel: int = 1) -> list[str]:
    """Miles ``scripts/models/deepseek-v4.1.py`` ``model_args`` for ``num_layers``."""
    layers = int(num_layers)
    if layers < 1:
        raise DeepseekV41RecipeError(f"num_layers {layers} must be positive")
    ratios = compress_ratios_for(layers, compress_ratios)
    return [
        "--disable-bias-linear",
        "--num-layers", str(layers),
        "--hidden-size", str(HIDDEN_SIZE),
        "--ffn-hidden-size", str(MOE_FFN_HIDDEN_SIZE),
        "--num-attention-heads", str(NUM_ATTENTION_HEADS),
        "--normalization", "RMSNorm",
        "--position-embedding-type", "rope",
        "--norm-epsilon", "1e-20",
        "--swiglu",
        "--untie-embeddings-and-output-weights",
        "--vocab-size", str(VOCAB_SIZE),
        "--make-vocab-size-divisible-by", str(make_vocab_size_divisible_by(tensor_parallel)),
        "--hidden-dropout", "0.0",
        "--attention-dropout", "0.0",
        "--multi-latent-attention",
        "--q-lora-rank", "1280",
        "--kv-lora-rank", "512",
        "--qk-head-dim", "512",
        "--qk-pos-emb-head-dim", "64",
        "--v-head-dim", "512",
        "--qk-layernorm",
        "--rotary-scaling-factor", "16",
        "--rotary-base", "10000",
        "--original-max-position-embeddings", "65536",
        "--beta-fast", "32",
        "--beta-slow", "1",
        "--attention-softmax-in-fp32",
        "--no-rope-fusion",
        "--num-experts", str(NUM_EXPERTS),
        "--moe-layer-freq", "[" + ",".join(["1"] * layers) + "]",
        "--moe-ffn-hidden-size", str(MOE_FFN_HIDDEN_SIZE),
        "--moe-router-topk", str(TOPK),
        "--moe-shared-expert-intermediate-size", str(MOE_FFN_HIDDEN_SIZE),
        "--moe-router-pre-softmax",
        "--moe-router-score-function", "sqrtsoftplus",
        "--moe-router-enable-expert-bias",
        "--moe-router-load-balancing-type", "seq_aux_loss",
        "--moe-token-dispatcher-type", "alltoall",
        "--moe-aux-loss-coeff", "0",
        "--moe-grouped-gemm",
        "--moe-router-topk-scaling-factor", "1.5",
        "--num-residual-streams", "4",
        "--mhc-sinkhorn-iterations", "20",
        "--csa-compress-ratios", "[" + ",".join(str(r) for r in ratios) + "]",
        "--csa-compress-rotary-base", "160000",
        "--o-groups", "8",
        "--o-lora-rank", "1024",
        "--moe-n-hash-layers", "0",
        "--csa-window-size", "128",
        "--dsa-indexer-n-heads", "32",
        "--dsa-indexer-head-dim", "128",
        "--dsa-indexer-topk", "512",
        "--spec", *SPEC,
        "--activation-func-clamp-value", "10",
        "--no-bias-swiglu-fusion",
    ]


# scripts/run_deepseek_v4_1.py misc/lora args of the trainer (no engine flags)
RECIPE_ARGV: tuple[str, ...] = (
    "--megatron-to-hf-mode", MEGATRON_TO_HF_MODE,
    "--model-name", MODEL_NAME,
    "--dsv4-impl", "miles",
    "--qkv-format", "bshd",
    "--moe-router-freeze-gate",
    "--freeze-e-score-correction-bias",
    "--transformer-impl", "transformer_engine",
    "--exclude-modules", EXCLUDE_MODULES,
    "--no-gradient-accumulation-fusion",
    "--recompute-granularity", "full",
    "--recompute-method", "uniform",
    "--recompute-num-layers", "1",
)

# generic-argv flags the Miles model arguments do not carry (Megatron derives them)
DROPPED_FLAGS = frozenset({"--num-query-groups", "--kv-channels", "--max-position-embeddings",
                           "--rotary-percent", "--add-qkv-bias", "--group-query-attention",
                           "--rope-type", "--lora-base-cpu-backup"})
# recompute granularities the recipe replaces (it always runs full uniform recompute)
_RECOMPUTE = frozenset({"--recompute-granularity", "--recompute-method", "--recompute-num-layers",
                        "--recompute-modules"})


def recipe_argv(*, num_layers: int = NUM_LAYERS, compress_ratios: Sequence[int] | None = None,
                tensor_parallel: int = 1) -> list[str]:
    return [*model_argv(num_layers=num_layers, compress_ratios=compress_ratios,
                        tensor_parallel=tensor_parallel), *RECIPE_ARGV]


def apply_recipe(argv: Sequence[str], recipe: Sequence[str]) -> list[str]:
    """``argv`` (``argv[0]`` the script) without every flag the recipe sets, the
    flags it drops and any other recompute setting, then the recipe."""
    groups = list(_flag_groups(argv[1:]))
    for group in groups:
        if group[0] == "--recompute-granularity" and len(group) > 1 and group[1] != "full":
            raise DeepseekV41RecipeError(
                f"--recompute-granularity {group[1]}: the deepseek_v41 recipe runs full recompute")
    override = {group[0] for group in _flag_groups(recipe)} | DROPPED_FLAGS | _RECOMPUTE
    kept = [token for group in groups if group[0] not in override for token in group]
    return [*argv[:1], *kept, *recipe]


__all__ = ["COMPRESS_RATIOS", "DROPPED_FLAGS", "EXCLUDE_MODULES", "MEGATRON_TO_HF_MODE", "MODEL_NAME",
           "MODEL_TYPE", "RECIPE_ARGV", "SPEC", "TARGET_MODULES", "DeepseekV41RecipeError",
           "apply_recipe", "check_geometry", "compress_ratios_for", "make_vocab_size_divisible_by",
           "model_argv", "recipe_argv", "target_modules"]
