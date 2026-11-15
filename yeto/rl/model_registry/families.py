"""Seed entries of the model registry (A21 sections 3.3 and 4).

Status: verified (a run or smoke on that path), smoke-only, inferred (upstream support
lists / same-architecture reasoning, no yeto run), unsupported (refused with a reason and
an alternative). Images are pin constant names (``module:CONST``); the tests resolve each.
"""

from __future__ import annotations

from .schema import BackendRecipe, FamilyEntry, QuantPlan

MILES_PINS = "yeto.rl.adapters.miles.pins"
VERL_PINS = "yeto.rl.adapters.verl.pins"

from yeto.launch.engine_images import default_image_entry


def _generic(family: str, role: str, engine: str) -> str:
    """The generic image ref of (family, role, engine): the pinned row of
    ``yeto.launch.engine_images.DEFAULT_IMAGES``, the one source of the
    device-family defaults (the registry keeps no copy)."""
    ref = default_image_entry(family, role, engine).ref
    if ref is None:
        raise ValueError(f"engine_images has no pinned {role} image for {family}/{engine}")
    return ref


# Generic images: read from engine_images, not repeated here.
MILES_NEXT = _generic("cuda", "trainer", "miles")
VERL = _generic("cuda", "trainer", "verl")
SGLANG_CUDA = _generic("cuda", "sampler", "sglang")
VLLM_CUDA = _generic("cuda", "sampler", "vllm")
VLLM_ROCM = _generic("rocm", "sampler", "vllm")
# Registry-only images (engine_images has no row for them): the K3.1 forms, the verl
# Megatron engine variant, and the unverified CANN base (engine_images' ascend rows
# carry no pinned ref; the registry names it with an "unverified" pitfall).
MILES_K31_FULL = f"{MILES_PINS}:MILES_K31_FULL_IMAGE"
MILES_K31 = f"{MILES_PINS}:MILES_K31_IMAGE"
VERL_MEGATRON = f"{VERL_PINS}:VERL_MEGATRON_IMAGE"
VERL_NPU = f"{VERL_PINS}:VERL_NPU_BASE_IMAGE"

INF = float("inf")
Q_CUDA = QuantPlan("bf16", "bf16 (FP8 optional on Hopper/Blackwell)", "bf16")
Q_ASCEND = QuantPlan("bf16", "bf16 (W8A8 optional)", "bf16",
                     "optional sampler-side W8A8 requant; 910B has no FP8 / MXFP4")
Q_K31 = QuantPlan("bf16 trainable experts over a frozen MXFP4 base", "mxfp4", "mxfp4",
                  "trained experts requantized to MXFP4 per release (gfx950 only)")

P_DROPOUT = "Miles generic recipe forces dropout 0 (miles/config.py:976-980) so train/rollout logprobs match"
P_VLLM_ASCEND_SLEEP = ("vllm-ascend 0.23 sleeps at level 1 only (verl/pins.py:50-53): the NPU sampler keeps "
                       "more HBM resident while training")
P_VERL_NPU_UNVERIFIED = "the verl NPU image pin is 'NOT VERIFIED on hardware' (verl/pins.py:42)"
P_HOST_LOAD = ("verl FSDP: every rank reads a host copy unless patch_verl #8/#9 apply; a 120B model "
               "read in FP32 on rank0 took 864 GiB host RAM")
P_AUTOBRIDGE = "Megatron-Bridge AutoBridge cannot read a model_type transformers does not know; needs a provider view"
P_NO_MILES_NPU = "Miles (and its SGLang rollout) does not run on Ascend"


def _r(name, hardware, trainer, sampler, recipe, tunings, band, timg, simg, quant, status, evidence, **kw):
    return BackendRecipe(name=name, hardware=tuple(hardware), trainer=trainer, sampler=sampler,
                         rl_model_recipe=recipe, tunings=tuple(tunings), size_band_b=band,
                         trainer_image=timg, sampler_image=simg, quant=quant, status=status,
                         evidence=evidence, **kw)


def _unsupported(name, hardware, band, reason, alternative):
    return BackendRecipe(name=name, hardware=tuple(hardware), trainer="-", sampler="-",
                         rl_model_recipe=None, tunings=("lora", "full"), size_band_b=band,
                         trainer_image="", sampler_image="", quant=Q_CUDA, status="unsupported",
                         evidence="A21 section 2", reason=reason, alternative=alternative)


def _dense_recipes(fam: str, *, cuda_status: str, ascend_status: str, ascend_evidence: str) -> tuple:
    return (
        _r(f"{fam}/cuda/miles", ["cuda"], "miles", "sglang", "generic", ["lora", "full"], (0, INF),
           MILES_NEXT, SGLANG_CUDA, Q_CUDA, cuda_status, "historical GPU runs (generic recipe)",
           pitfalls=(P_DROPOUT,)),
        _r(f"{fam}/cuda/verl", ["cuda"], "verl", "vllm", "generic", ["lora", "full"], (0, INF),
           VERL, VLLM_CUDA, Q_CUDA, cuda_status, "verl FSDP2 + vLLM island (S16)",
           pitfalls=(P_HOST_LOAD,)),
        _r(f"{fam}/ascend/verl", ["ascend"], "verl", "vllm-ascend", "generic", ["lora", "full"], (0, 300),
           VERL_NPU, VERL_NPU, Q_ASCEND, ascend_status, ascend_evidence,
           pitfalls=(P_VLLM_ASCEND_SLEEP, P_VERL_NPU_UNVERIFIED, P_NO_MILES_NPU, P_HOST_LOAD)),
        _unsupported(f"{fam}/ascend/>300B", ["ascend"], (300, INF),
                     "a >300B model on 910B (64 GiB) needs a multi-node RoCE sampler that vllm-ascend "
                     "does not support for this family; the trainer state (~16 B/param) does not fit "
                     "the nodes that are available",
                     "train and sample on NVIDIA (miles + sglang), or use LoRA on a smaller size"),
    )


QWEN3_DENSE = FamilyEntry(
    "qwen3-dense", ("Qwen3ForCausalLM", "Qwen2ForCausalLM"), ("qwen3", "qwen2"),
    _dense_recipes("qwen3-dense", cuda_status="verified", ascend_status="smoke-only",
                   ascend_evidence="hetero-1114 A02/A04/A07: Qwen3-0.6B on 910B (vllm-ascend, verl FSDP2)"),
)

QWEN3_MOE = FamilyEntry(
    "qwen3-moe", ("Qwen3MoeForCausalLM", "Qwen2MoeForCausalLM"), ("qwen3_moe", "qwen2_moe"),
    (
        _r("qwen3-moe/cuda/miles", ["cuda"], "miles", "sglang", "generic", ["lora", "full"], (0, INF),
           MILES_NEXT, SGLANG_CUDA, Q_CUDA, "inferred", "A21 section 2 (Megatron-Bridge + SGLang list Qwen3-MoE)",
           pitfalls=(P_DROPOUT, "pick EP so that num_experts % EP == 0; routing replay (R2/R3) keeps "
                     "train/rollout expert choice aligned")),
        _r("qwen3-moe/cuda/verl", ["cuda"], "verl", "vllm", "generic", ["lora", "full"], (0, INF),
           VERL_MEGATRON, VLLM_CUDA, Q_CUDA, "inferred", "A21 section 2 (verl Megatron engine, EP)",
           pitfalls=(P_HOST_LOAD,)),
        _r("qwen3-moe/ascend/verl", ["ascend"], "verl", "vllm-ascend", "generic", ["lora", "full"], (0, 100),
           VERL_NPU, VERL_NPU, Q_ASCEND, "inferred", "vllm-ascend support matrix: Qwen3-30B-A3B supported",
           pitfalls=(P_VLLM_ASCEND_SLEEP, P_VERL_NPU_UNVERIFIED, P_NO_MILES_NPU)),
        _unsupported("qwen3-moe/ascend/235B", ["ascend"], (100, INF),
                     "vllm-ascend marks Qwen3-235B-A22B unmaintained (red) and Miles has no NPU path",
                     "NVIDIA: miles + sglang with EP8 (qwen3-moe/cuda/miles)"),
    ),
)

LLAMA3 = FamilyEntry(
    "llama3", ("LlamaForCausalLM",), ("llama",),
    _dense_recipes("llama3", cuda_status="inferred", ascend_status="inferred",
                   ascend_evidence="vllm-ascend support list does not list Llama 3 (not a refusal)"),
)

MISTRAL_GEMMA = FamilyEntry(
    "mistral-gemma", ("MistralForCausalLM", "MixtralForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM"),
    ("mistral", "mixtral", "gemma2", "gemma3", "gemma3_text"),
    (
        _r("mistral-gemma/cuda/verl", ["cuda"], "verl", "vllm", "generic", ["lora", "full"], (0, INF),
           VERL, VLLM_CUDA, Q_CUDA, "inferred", "A21 section 2 (verl FSDP2 partial; ms-swift has no support)",
           pitfalls=("Megatron-Bridge lacks Gemma 2 / Mistral: prefer verl FSDP2 over Miles",
                     "ms-swift Megatron does not support Mistral/Mixtral/Gemma")),
        _r("mistral-gemma/ascend/verl", ["ascend"], "verl", "vllm-ascend", "generic", ["lora", "full"], (0, 100),
           VERL_NPU, VERL_NPU, Q_ASCEND, "inferred", "vllm-ascend support list does not list these families",
           pitfalls=(P_VLLM_ASCEND_SLEEP, P_VERL_NPU_UNVERIFIED)),
    ),
)

DEEPSEEK_V3 = FamilyEntry(
    "deepseek-v3", ("DeepseekV3ForCausalLM", "KimiK2ForCausalLM"), ("deepseek_v3", "kimi_k2"),
    (
        _r("deepseek-v3/cuda/miles", ["cuda"], "miles", "sglang", "generic", ["lora", "full"], (0, INF),
           MILES_NEXT, SGLANG_CUDA, Q_CUDA, "inferred", "A21 section 2 (Megatron-Bridge + SGLang support DeepSeek-V3)",
           pitfalls=(P_DROPOUT, "MLA: TP must divide num_attention_heads; KV heads are the latent (no GQA split)",
                     "the FP8 HF checkpoint (block-wise e4m3) needs a bf16 conversion before Megatron-Bridge loads it",
                     "MTP layers (num_nextn_predict_layers) are not trained by the generic recipe")),
        _unsupported("deepseek-v3/ascend", ["ascend"], (0, INF),
                     "vllm-ascend marks DeepSeek-V3/R1 unmaintained (red); Miles has no NPU path; 671B bf16 "
                     "weights (~1.3 TB) need >=3 RoCE-linked 910B nodes just to sample",
                     "NVIDIA: miles + sglang (deepseek-v3/cuda/miles), LoRA, H200/B200/B300"),
    ),
)

DEEPSEEK_V4_FLASH = FamilyEntry(
    "deepseek-v4-flash", ("DeepseekV4ForCausalLM",), ("deepseek_v4",),
    (
        _r("deepseek-v4-flash/cuda/miles", ["cuda"], "miles", "sglang", "deepseek-v4-flash", ["lora"], (0, INF),
           MILES_NEXT, SGLANG_CUDA, Q_CUDA, "verified", "tests/test_rl_deepseek_v4*",
           hard_constraints=("tp==8", "ep==8", "rollout_gpus_per_engine==8"),
           pitfalls=("expanded DeepSeek V4 requires TP8/EP8 and 8-GPU rollout replicas (run_config.py:937-956)",)),
        _unsupported("deepseek-v4-flash/ascend", ["ascend"], (0, INF),
                     "the deepseek-v4-flash recipe is a Miles recipe and Miles has no NPU path",
                     "NVIDIA: miles + sglang"),
    ),
)

GLM5_NEXT = FamilyEntry(
    "glm5_next", (), ("glm5_next", "glm5_next_text"),
    (
        _r("glm5_next/cuda/miles", ["cuda"], "miles", "sglang", "glm5_next", ["full"], (0, INF),
           MILES_NEXT, SGLANG_CUDA, Q_CUDA, "smoke-only", "Modal run-7 (stage1.py:492)",
           hard_constraints=("needs:megatron_ref_load",),
           pitfalls=("full-parameter only; convert to torch_dist first and pass --megatron-ref-load "
                     "(run_config.py:960-966)", P_AUTOBRIDGE)),
        _unsupported("glm5_next/ascend", ["ascend"], (0, INF),
                     "the glm5_next recipe is a Miles recipe and Miles has no NPU path",
                     "NVIDIA: miles + sglang (glm5_next/cuda/miles)"),
    ),
)

KIMI_K31 = FamilyEntry(
    "kimi_k31", ("KimiK31ForConditionalGeneration", "KimiK3ForConditionalGeneration"), ("kimi_k31", "kimi_k3"),
    (
        _r("kimi_k31/cuda-trainer+rocm-sampler", ["cuda"], "miles", "vllm", "kimi_k31", ["full", "lora"],
           (0, INF), MILES_K31_FULL, VLLM_ROCM, Q_K31, "verified",
           "yeto main 1d682d51 (K3.1 async GRPO delivery, #418-#445)",
           sampler_hardware="rocm:gfx950",
           hard_constraints=("card:B300", "full:needs:rl_expert_selection", "ep==8"),
           pitfalls=("full = partial experts (128 of 896 per layer) and needs --rl-expert-selection",
                     "LoRA: --lora-r with the MILES_K31_IMAGE; vLLM samplers take attention-only LoRA targets",
                     "K3.1 does not start on the vLLM ROCm nightly: keep the day-0 VLLM_SAMPLER_IMAGE_ROCM",
                     "MXFP4 is gfx950-only: samplers on MI355X (NC A-cards), one sampler per node",
                     "AutoBridge cannot read kimi_k31: the provider view is models/kimi_k31/provider.py")),
        _unsupported("kimi_k31/ascend", ["ascend"], (0, INF),
                     "2.8T MoE: no MXFP4 on 910B (W8A8 is ~2.8 TB, >=7 RoCE nodes), vllm-ascend K3.1 is "
                     "only planned (blue), Miles has no NPU path, full-parameter state is ~45 TB",
                     "the verified path: Miles trainer on 6x8 B300 + vLLM MXFP4 samplers on MI355X "
                     "(--hardware nvidia)"),
    ),
)

NEMOTRON_H = FamilyEntry(
    "nemotron-h", ("NemotronHForCausalLM",), ("nemotron_h",),
    (
        _r("nemotron-h/cuda/miles", ["cuda"], "miles", "sglang", "generic", ["lora", "full"], (0, INF),
           MILES_NEXT, SGLANG_CUDA, Q_CUDA, "verified", "run_config.py:1002 (Megatron-Bridge Nemotron-H recipe)",
           pitfalls=("the recipe is chosen from the provider (hybrid_layer_pattern), not --rl-model-recipe",)),
        _r("nemotron-h/cuda/verl", ["cuda"], "verl", "vllm", "generic", ["lora", "full"], (0, INF),
           VERL_MEGATRON, VLLM_CUDA, Q_CUDA, "verified", "verl/nemotron_h.py, patch_verl #7-#9",
           pitfalls=("verl picks AutoModel from auto_map (patch_verl #7)", P_HOST_LOAD,
                     "vLLM Mamba cache flags change between releases")),
        _unsupported("nemotron-h/ascend/rl", ["ascend"], (0, INF),
                     "on 910B only the ms-swift Megatron SFT path is verified (120B DiLoCo SFT); there is "
                     "no RL trainer/sampler path for Nemotron-H on Ascend (vllm-ascend does not list it)",
                     "SFT on Ascend via ms-swift Megatron; RL on NVIDIA (nemotron-h/cuda/miles)"),
    ),
)

GENERIC_DENSE = FamilyEntry(
    "generic-dense", (), (),
    _dense_recipes("generic-dense", cuda_status="inferred", ascend_status="inferred",
                   ascend_evidence="no family entry: same path as qwen3-dense, unverified"),
    pitfalls=("no registry entry matched: the recommendation is a guess; run a 4-layer / small smoke first",
              P_AUTOBRIDGE),
)

GENERIC_MOE = FamilyEntry(
    "generic-moe", (), (),
    QWEN3_MOE.recipes,
    pitfalls=("no registry entry matched: the recommendation is a guess; run a 4-layer / small smoke first",
              P_AUTOBRIDGE),
)

FAMILIES: tuple[FamilyEntry, ...] = (
    QWEN3_DENSE, QWEN3_MOE, LLAMA3, MISTRAL_GEMMA, DEEPSEEK_V3, DEEPSEEK_V4_FLASH, GLM5_NEXT,
    KIMI_K31, NEMOTRON_H,
)
ALL_ENTRIES: tuple[FamilyEntry, ...] = FAMILIES + (GENERIC_DENSE, GENERIC_MOE)
