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

from collections.abc import Iterator, Sequence

MODEL_NAME = "glm5_next"
TRANSFORMER_ENGINE = "transformer_engine"
LOCAL = "local"
TRANSFORMER_IMPLS = (TRANSFORMER_ENGINE, LOCAL)
SPEC = ("miles_plugins.models.glm5_next.glm5_next", "get_glm5_next_spec")
NUM_LAYERS = 45
FIRST_K_DENSE_REPLACE = 3

# N7.5 E6/E7: flags that exist only with TransformerEngine.
TE_ONLY_ARGV: tuple[str, ...] = ("--moe-grouped-gemm", "--moe-permute-fusion")
LOCAL_ARGV: tuple[str, ...] = ("--transformer-impl", LOCAL, "--no-gradient-accumulation-fusion")

_MODEL_ARGV: tuple[str, ...] = (
    "--hidden-size", "4096",
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
    "--num-experts", "288",
    "--moe-router-topk", "8",
    "--moe-router-score-function", "sigmoid",
    "--moe-router-pre-softmax",
    "--moe-router-enable-expert-bias",
    "--moe-router-bias-update-rate", "0",
    "--moe-router-topk-scaling-factor", "2.5",
    "--moe-ffn-hidden-size", "2048",
    "--moe-shared-expert-intermediate-size", "2048",
    "--moe-router-dtype", "fp32",
    "--vocab-size", "154880",
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


def check_transformer_impl(transformer_impl: str) -> str:
    if transformer_impl not in TRANSFORMER_IMPLS:
        raise ValueError(f"transformer_impl must be one of {TRANSFORMER_IMPLS}, got {transformer_impl!r}")
    return transformer_impl


def model_argv(*, num_layers: int = NUM_LAYERS, first_k_dense_replace: int = FIRST_K_DENSE_REPLACE,
               transformer_impl: str = TRANSFORMER_ENGINE) -> list[str]:
    """The model arguments of the recipe (module doc) and the run arguments the
    model needs (:data:`RUN_ARGV`)."""
    impl = check_transformer_impl(transformer_impl)
    return ["--spec", *SPEC, "--num-layers", str(int(num_layers)), *_MODEL_ARGV,
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
