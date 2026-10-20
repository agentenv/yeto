"""Miles argv of the GLM5.3-Flash (``glm5_next``) recipe (S25 stage 1, gap 4).

Model arguments: Miles ``scripts/models/glm5.3-flash.py`` ``model_args()`` at
the pinned Miles commit (cc76e2391, with the s25-no-te fallback a38decedf),
copied as one tuple so the launch layer does not import a Miles script. The
model trains through the Miles plugin spec
``miles_plugins.models.glm5_next.glm5_next:get_glm5_next_spec`` (KDA + DSA MLA
layers, 288 routed experts), not through a Bridge-derived geometry: the
recipe replaces every geometry flag the generic translation derives.

``transformer_impl``:

* ``transformer_engine`` (default, main-agent ruling 10-19 (3)): TE layers,
  grouped GEMM experts and fused permutation;
* ``local``: the TE-free fallback of N7.5 (Miles ``glm5_next/local_layers.py``);
  the TE-only flags are dropped (Megatron raises on ``--moe-permute-fusion``
  without TE) and gradient-accumulation fusion is off.

Not run on a PPU; the argv was compared with the Miles script by reading only.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

MODEL_NAME = "glm5_next"
TRANSFORMER_ENGINE = "transformer_engine"
LOCAL = "local"
TRANSFORMER_IMPLS = (TRANSFORMER_ENGINE, LOCAL)
SPEC = ("miles_plugins.models.glm5_next.glm5_next", "get_glm5_next_spec")
NUM_LAYERS = 45
FIRST_K_DENSE_REPLACE = 3
HIDDEN_SIZE = 4096
VOCAB_SIZE = 154880
NUM_EXPERTS = 288

# N7.5 E6/E7: flags that exist only with TransformerEngine.
TE_ONLY_ARGV: tuple[str, ...] = ("--moe-grouped-gemm", "--moe-permute-fusion")
LOCAL_ARGV: tuple[str, ...] = ("--transformer-impl", LOCAL, "--no-gradient-accumulation-fusion")

_MODEL_ARGV: tuple[str, ...] = (
    "--hidden-size", str(HIDDEN_SIZE),
    "--ffn-hidden-size", "12288",
    "--num-attention-heads", "64",
    "--multi-latent-attention",
    "--q-lora-rank", "1536",
    "--kv-lora-rank", "512",
    "--qk-head-dim", "256",
    "--qk-pos-emb-head-dim", "0",
    "--v-head-dim", "256",
    "--kv-channels", "256",
    "--qk-layernorm",
    "--moe-router-topk", "8",
    "--moe-router-score-function", "sigmoid",
    "--moe-router-pre-softmax",
    "--moe-router-enable-expert-bias",
    "--moe-router-bias-update-rate", "0",
    "--moe-router-topk-scaling-factor", "2.5",
    "--moe-ffn-hidden-size", "2048",
    "--moe-shared-expert-intermediate-size", "2048",
    "--moe-router-dtype", "fp32",
    "--vocab-size", str(VOCAB_SIZE),
    "--rotary-base", "10000",
    "--rope-type", "rope",
    "--no-rope-fusion",
    "--position-embedding-type", "rope",
    "--normalization", "RMSNorm",
    "--norm-epsilon", "1e-5",
    "--swiglu",
    # the checkpoint sets swiglu_limit=10, which SGLang applies to dense, shared and routed MLPs
    "--activation-func-clamp-value", "10",
    "--disable-bias-linear",
    "--untie-embeddings-and-output-weights",
    "--make-vocab-size-divisible-by", "16",
    "--enable-experimental",
)

# scripts/run_glm5_3_flash.py engine arguments the model needs (packed sequences).
RUN_ARGV: tuple[str, ...] = ("--qkv-format", "thd")

# Flags of the generic translation that the plugin spec does not take (the
# geometry the Bridge derives for other models).
DROPPED_FLAGS = frozenset({"--num-query-groups", "--max-position-embeddings", "--rotary-percent",
                           "--add-qkv-bias"})


def moe_layer_freq(num_layers: int = NUM_LAYERS, first_k_dense_replace: int = FIRST_K_DENSE_REPLACE) -> str:
    """Megatron's ``--moe-layer-freq`` list: dense for the first layers, MoE after
    (Miles ``model_args_utils.moe_layer_freq``)."""
    if not 0 <= first_k_dense_replace <= num_layers:
        raise ValueError(f"first_k_dense_replace {first_k_dense_replace} is outside 0..{num_layers}")
    return "[" + ",".join(["0"] * first_k_dense_replace + ["1"] * (num_layers - first_k_dense_replace)) + "]"


def checkpoint_geometry(config: dict) -> dict[str, int]:
    """``num_layers``, ``first_k_dense_replace`` and ``num_experts`` of an HF
    ``config.json`` (the text config of ``Glm5NextForConditionalGeneration``),
    as :func:`model_argv` keywords; a key the config lacks keeps the default."""
    text = config.get("text_config") or config
    out = {}
    for key, hf_key in (("num_layers", "num_hidden_layers"),
                        ("first_k_dense_replace", "first_k_dense_replace"),
                        ("num_experts", "n_routed_experts")):
        if text.get(hf_key) is not None:
            out[key] = int(text[hf_key])
    return out


def check_transformer_impl(transformer_impl: str) -> str:
    if transformer_impl not in TRANSFORMER_IMPLS:
        raise ValueError(f"transformer_impl must be one of {TRANSFORMER_IMPLS}, got {transformer_impl!r}")
    return transformer_impl


def model_argv(*, num_layers: int = NUM_LAYERS, first_k_dense_replace: int = FIRST_K_DENSE_REPLACE,
               num_experts: int = NUM_EXPERTS, transformer_impl: str = TRANSFORMER_ENGINE) -> list[str]:
    """The model arguments of the recipe (module doc) and the run arguments the
    model needs (:data:`RUN_ARGV`). ``num_layers``, ``first_k_dense_replace`` and
    ``num_experts`` default to GLM5.3-Flash; a cut-down checkpoint (e.g. Miles'
    4-layer CI model) passes its own (:func:`checkpoint_geometry`)."""
    impl = check_transformer_impl(transformer_impl)
    if int(num_experts) < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    return ["--spec", *SPEC, "--num-layers", str(int(num_layers)), *_MODEL_ARGV,
            "--num-experts", str(int(num_experts)),
            "--moe-layer-freq", moe_layer_freq(num_layers, first_k_dense_replace),
            *(TE_ONLY_ARGV if impl == TRANSFORMER_ENGINE else LOCAL_ARGV), *RUN_ARGV]


def _flag_groups(argv: Sequence[str]) -> Iterator[list[str]]:
    group: list[str] = []
    for token in argv:
        if token.startswith("--") and group:
            yield group
            group = []
        group.append(token)
    if group:
        yield group


def apply_recipe(argv: Sequence[str], recipe: Sequence[str]) -> list[str]:
    """``argv`` (``argv[0]`` the script) without every flag the recipe sets or
    drops, then the recipe. ``--transformer-impl transformer_engine`` stays
    implicit (Megatron's default), so a generic argv that also leaves it out
    is not changed by the default recipe."""
    override = {group[0] for group in _flag_groups(recipe)} | DROPPED_FLAGS
    if "--transformer-impl" in recipe:  # local: gradient-accumulation fusion goes off as well
        override.add("--no-gradient-accumulation-fusion")
    kept = [token for group in _flag_groups(argv[1:]) if group[0] not in override for token in group]
    return [*argv[:1], *kept, *recipe]


# ---------------------------------------------------------------- provider view
# Megatron-Bridge's AutoBridge cannot read a glm5_next checkpoint: transformers
# 5.12.1 has no such model type (Modal run-3), and Miles trains it through its
# plugin spec, not through the Bridge. The island only needs the geometry that
# ``run_config.resolve_rl_run_config`` reads, so it takes this read-only view of
# the checkpoint's config.json (the same approach as the Flash-Next view).
MODEL_TYPES = frozenset({"glm5_next", "glm5_next_text"})


@dataclasses.dataclass(frozen=True)
class Glm5NextModelProvider:
    """Read-only stand-in for a Megatron-Bridge provider (the fields run_config reads)."""

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    kv_channels: int
    ffn_hidden_size: int
    num_moe_experts: int
    moe_ffn_hidden_size: int
    moe_router_topk: int
    moe_layer_freq: tuple[int, ...]
    moe_shared_expert_intermediate_size: int
    vocab_size: int
    seq_length: int
    max_position_embeddings: int
    layernorm_epsilon: float
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_pos_emb_head_dim: int
    v_head_dim: int
    rotary_base: int = 10000  # the recipe's --rotary-base (the checkpoint has no RoPE dims: qk_rope_head_dim 0)
    rotary_percent: float = 1.0
    share_embeddings_and_output_weights: bool = False
    multi_latent_attention: bool = True
    qk_layernorm: bool = True
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    normalization: str = "RMSNorm"
    position_embedding_type: str = "rope"

    def finalize(self) -> None:  # provider API parity; nothing to derive
        return None


def is_glm5_next(model_path: str | Path) -> bool:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("model_type") in MODEL_TYPES
    except (OSError, ValueError):
        return False


def provider_view(config: dict) -> Glm5NextModelProvider:
    """The provider view of a glm5_next ``config.json`` (module section above)."""
    if config.get("model_type") not in MODEL_TYPES:
        raise ValueError(f"not a glm5_next config: model_type={config.get('model_type')!r}")
    text = config.get("text_config") or config
    layers = int(text["num_hidden_layers"])
    dense = int(text.get("first_k_dense_replace", FIRST_K_DENSE_REPLACE))
    if text.get("hidden_act", "silu") != "silu":
        raise ValueError(f"glm5_next expects SwiGLU (silu), got {text.get('hidden_act')!r}")
    if bool(text.get("attention_bias", False)):
        raise ValueError("glm5_next expects bias-free linears")
    max_positions = int(text["max_position_embeddings"])
    return Glm5NextModelProvider(
        num_layers=layers,
        hidden_size=int(text["hidden_size"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_query_groups=int(text.get("num_key_value_heads", text["num_attention_heads"])),
        kv_channels=int(text["v_head_dim"]),
        ffn_hidden_size=int(text["intermediate_size"]),
        num_moe_experts=int(text["n_routed_experts"]),
        moe_ffn_hidden_size=int(text["moe_intermediate_size"]),
        moe_router_topk=int(text["num_experts_per_tok"]),
        moe_layer_freq=tuple([0] * dense + [1] * (layers - dense)),
        moe_shared_expert_intermediate_size=int(text["moe_intermediate_size"]) * int(text.get("n_shared_experts", 1)),
        vocab_size=int(text["vocab_size"]),
        seq_length=max_positions,
        max_position_embeddings=max_positions,
        layernorm_epsilon=float(text["rms_norm_eps"]),
        q_lora_rank=int(text["q_lora_rank"]),
        kv_lora_rank=int(text["kv_lora_rank"]),
        qk_head_dim=int(text["qk_head_dim"]),
        qk_pos_emb_head_dim=int(text.get("qk_rope_head_dim", 0)),
        v_head_dim=int(text["v_head_dim"]),
        share_embeddings_and_output_weights=bool(config.get("tie_word_embeddings", False)
                                                 or text.get("tie_word_embeddings", False)),
    )


def glm5_next_provider(model_path: str | Path) -> Glm5NextModelProvider:
    return provider_view(json.loads((Path(model_path) / "config.json").read_text(encoding="utf-8")))
