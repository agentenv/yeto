"""``vllm-server``: a vLLM OpenAI-compatible server behind the ``SamplerEngine``
port, on two lines of vLLM (``VllmOptions.platform``):

* ``None`` / ``npu``: the legacy line, vllm-ascend 0.30.0rc1 on 910B (stage 2:
  TP4 per instance, W4A8) and the vLLM of the verl image. Unchanged argv
  (``python -m vllm.entrypoints.openai.api_server --model ...``) and rules;
* ``cuda`` / ``rocm``: upstream vLLM main as pinned for the Kimi-K3.1 sampler
  (``sampler_pins.VLLM_SAMPLER_*``: the dedicated ``kimi-k31`` images, interface of main
  335dd77b; main 083fbac02 checked too).
  Started as ``python -m vllm.entrypoints.cli.main serve <model>`` (the
  ``vllm serve`` entry; ``openai.api_server`` is deprecated there).

Vendor facts of the legacy line (A-INF §3, report items 15-18; none verified on a
910B by this code):

* reload: restart only. In-place reload of W4A8_DYNAMIC on vllm-ascend is not
  tested (GOAL §8); ``in_place_reload`` is refused on this line;
* item 17: 910B reads ``int-quantized`` only, not ``pack-quantized``; the
  default ``expected_format`` for a quantized scheme is ``int-quantized``.
  W4A8 stages int8 weights in host memory (``VLLM_ASCEND_W4A8_CPU_STAGE=1``),
  so set ``min_host_free_gib`` from the measured peak (report: about 610 GiB);
* item 18: eager mode serves about 2.75 tok/s against about 38 tok/s with
  graphs; ``--enforce-eager`` is refused unless ``allow_eager`` (debug only);
* item 15: a missing CANN 9.1 libprofapi stub hangs HCCL without an error; the
  start timeout names it;
* items 16 and the PT_INTERP fix belong to the image or rootfs start script,
  not to this adapter.

The upstream line (checked against the vLLM source at the pin; on-card
behaviour NOT verified by this code):

* R3 (``return_routed_experts``): ``--enable-return-routed-experts`` (server-wide;
  ``AuxOutputConfig``). Every ``/v1/completions`` choice then carries
  ``routed_experts``: base64 of ``.npy`` bytes (``numpy2base64``: the array with
  its own dtype and shape header, uint8 up to 256 experts else uint16) of shape
  ``[prompt + output - 1, num_hidden_layers, top_k]``. The layer axis is the
  model's every decoder layer (``get_total_num_hidden_layers``; the capture
  buffer is zero-initialised and only MoE routers write their ``layer_id`` row),
  so the ``first_k_dense_replace`` leading dense layers come as all-zero rows;
  :func:`decode_routed_experts` drops them (the buffer field holds ``L_moe``
  layers, as from SGLang). The request asks ``return_token_ids`` so the prompt
  length of a text prompt is known. vLLM refuses R3 together with context
  parallelism (``VllmConfig._verify_aux_output_compatibility``: "incompatible with
  context parallelism (DCP/PCP > 1)", ``config/vllm.py`` of both kimi-k31 images
  and of main), so an R3 server drops every DCP / PCP size above 1 (a preset's DCP
  or one in ``extra_args``; :attr:`VllmServerEngine.cp_overrides`, logged before a
  start and recorded in the load detail ``r3_cp_overrides``): the MLA KV cache is
  then whole on every TP rank, not split over them. The other R3 refusals of that
  check (PP > 1, prefix caching off, the V1 model runner) are refused at build;
* top-K log-probs (score_centering): ``logprobs: K`` with
  ``return_tokens_as_token_ids``; the server's ``--max-logprobs`` (default 20)
  is raised to K. Values are ``logprobs_mode`` ``raw_logprobs`` (vLLM's default:
  before temperature / top-p / top-k), equal to the processed ones at the
  score_centering sampling (temperature 1, top_p 1, top_k -1);
* fp32 lm-head (``fp32_lm_head``): ``--hf-overrides '{"head_dtype": "float32"}'``
  (``ModelConfig.head_dtype``; ``LogitsProcessor._apply_head`` accumulates the
  bf16 x bf16 projection into fp32 with ``torch.mm(out_dtype=float32)`` on CUDA
  and ROCm). Only for an unquantized lm_head (K3.1 keeps it BF16);
* in-place reload (``ServerConfig.in_place_reload``, off unless an on-card test
  passed): ``POST /pause?mode=wait`` (drains, clears the prefix cache), ``POST
  /collective_rpc {"method": "reload_weights", "kwargs": {"weights_path": DIR}}``
  (``model_loader.reload.inplace.reload_weights``: every rank reads the release
  from disk in checkpoint format and re-runs the post-load processing), ``POST
  /resume`` (also after a failure). Both routers exist only with
  ``VLLM_SERVER_DEV_MODE=1``, which the adapter sets when in-place reload is on.
  verl #8574 rebuilds the frozen MXFP4 Kimi-K3.1 base this way on MI355X; this
  adapter's path is not measured;
* ROCm: :data:`ROCM_ENV_PRESET` (``extra_env`` wins) and the AMD card count
  checked against TP before a start; the Kimi-K3.1-Pro serving layouts of the
  published recipes are :data:`K31_PRESETS` (``VllmOptions.k31_preset``). Kimi-K3.1
  on ROCm needs the dedicated ``vllm/vllm-openai-rocm:kimi-k31`` image: vLLM main
  (083fbac02) builds the AMD model without the DSA indexer (``amd/linear.py``
  ``indexer=None, is_sparse=False``; the checkpoint's ``indexer.k_norm.bias`` then
  has no parameter) and its ROCm sparse MLA backend cannot read ``fp8_ds_mla``,
  which the K3.1 model config forces for every fp8 / auto KV dtype off SM10x
  (open upstream: #66532 split 4/6, #65685 split 5/6);
* the build the server runs: ``GET /version`` after ready, compared with
  ``VllmOptions.pin_version`` (``server_info.vllm_pin_match``; recorded, not refused).

API calls (both lines): ``GET /health``; ``POST /v1/completions`` with
``logprobs`` and ``return_tokens_as_token_ids`` (tokens come back as
``token_id:<id>``); teacher-forced scores with ``echo: true`` and
``max_tokens: 1``. Field support in vllm-ascend 0.30.0rc1 is NOT verified.
"""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from yeto.rl.engine import vram_guard
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    Completion,
    SamplerEngineError,
    SamplingParams,
)

from . import k31_release, vllm_expert_reload
from .server import ServerConfig, ServerEngine
from .sglang_server import dense_layer_count, moe_layer_count, rocm_card_count

INT_QUANTIZED = "int-quantized"
PLATFORMS = ("cuda", "rocm", "npu")  # VllmOptions.platform; None is the legacy line
UPSTREAM_PLATFORMS = ("cuda", "rocm")  # upstream vLLM main (sampler_pins.VLLM_SAMPLER_*)
LEGACY_REFUSED_FORMATS = frozenset({"pack-quantized", "dense"})  # 910B item 17; item 5
UPSTREAM_REFUSED_FORMATS = frozenset({"dense"})  # an MXFP4 export read wrongly (item 5)
LEGACY_HANG_HINT = ("Suspected HCCL hang: check that the CANN 9.1 libprofapi stub is installed "
                    "(report item 15).")
UPSTREAM_HANG_HINT = ("A Kimi-K3.1 load (about 1.56 TB of MXFP4 safetensors) can take over 20 min from "
                      "network storage; raise start_timeout_s and VLLM_ENGINE_READY_TIMEOUT_S together. "
                      "On MI355X a hang without an error is the MEC firmware scratch reclaim "
                      "(HSA_NO_SCRATCH_RECLAIM=1).")
DEFAULT_MAX_LOGPROBS = 20  # vLLM ModelConfig.max_logprobs
RESUME_ATTEMPTS = 3  # /resume tries after an in-place reload before the paused server is killed
RESUME_RETRY_S = 2.0
DEV_MODE_ENV = "VLLM_SERVER_DEV_MODE"  # mounts /collective_rpc and /pause + /resume
# Every ROCm server (extra_env wins): AITER kernels, the V2 model runner (the K3.1 day-0
# image predates V1's removal), no HSA scratch reclaim (MEC firmware hang without an
# error) and the engine-ready timeout of a 1.56 TB load with first AITER JIT builds.
ROCM_ENV_PRESET = {"VLLM_ROCM_USE_AITER": "1", "VLLM_USE_V2_MODEL_RUNNER": "1", "HSA_NO_SCRATCH_RECLAIM": "1",
                   "VLLM_ENGINE_READY_TIMEOUT_S": "7200"}
CUDA_ENV_PRESET = {"VLLM_ENGINE_READY_TIMEOUT_S": "3600"}  # the vLLM K3.1 recipe's Blackwell profile
FP32_HEAD_OVERRIDE = {"head_dtype": "float32"}
VERSION_KEYS = ("version",)

# R3 vs vLLM's AuxOutput check (VllmConfig._verify_aux_output_compatibility; module doc).
# The context-parallel sizes are dropped on an R3 server; the rest is refused at build.
CP_SIZE_FLAGS = {"--decode-context-parallel-size": "DCP", "-dcp": "DCP",
                 "--prefill-context-parallel-size": "PCP", "-pcp": "PCP"}
PP_SIZE_FLAGS = ("--pipeline-parallel-size", "-pp")
R3_CP_RULE = ("vLLM refuses --enable-return-routed-experts with context parallelism (DCP/PCP > 1; "
              "VllmConfig._verify_aux_output_compatibility)")

# Kimi-K3.1-Pro presets (VllmOptions.k31_preset), the serving layout of the published
# recipes without the parts a sampler does not use (DSpark speculative decoding, CPU KV
# offload, the capture-size list tuned for 1 + 5 draft rows, host / port):
#
# * ``amd-article`` (rocm): AMD's "Kimi K3.1 Pro on AMD Instinct GPUs" vLLM recipe
#   (image vllm/vllm-openai-rocm:kimi-k31): TP8 + DCP8 (a2a; DCP 1 on an R3 server, whose
#   DCP-only flags then have no effect), FP8 MLA KV cache (the
#   model config makes it fp8_ds_mla), MXFP4 index keys, FlyDSL sparse MLA, fused KDA
#   prefill, prefix caching, FP8 per-channel KDA input projections, INT4 quick-reduce
#   all-reduce and FP8 MoE stage 2;
# * ``rl-precision`` (rocm): the same layout with the precision policy of vLLM's
#   K3.1 recipe for rollouts that a trainer scores (BF16 all-reduce, BF16 MoE combine,
#   BF16 KDA input projections, FP32 KDA recurrent state). The recipe's
#   ``AITER_BF16_FP8_MOE_BOUND=0`` is left out: on MI355X (image kimi-k31
#   @5f47cc10) it crashed the engine within ~3 minutes of 64 concurrent top-128
#   log-prob requests (``hipErrorLaunchOutOfResources: too many resources requested
#   for launch`` in ``rocm_aiter_grouped_topk``, then EngineDeadError); without it the
#   same load ran clean and the teacher-forced log-prob noise floor stayed the same
#   (run-to-run mean |d| 0.049 vs 0.042-0.047, p99 0.65-0.73 vs 0.64-0.69);
# * ``vllm-recipe`` (cuda): vLLM's Kimi-K3.1-Pro recipe baseline on B300 / GB300
#   (image vllm/vllm-openai:kimi-k31): plain TP8, text only, fastsafetensors load;
# * ``rl-precision-cuda`` (cuda): that baseline with the FP32 KDA recurrent state.
#
# --gpu-memory-utilization 0.84 (was 0.88, before that 0.92): at 0.92 an MI355X TP8 sampler ran
# its GPUs at 99% and an RCCL kernel launch then failed (HSA_STATUS_ERROR_OUT_OF_RESOURCES,
# "Available Free mem : 0 MB"), hanging the engine with its container still up. At 0.88
# (253 of 288 GiB) a K3.1 agentic sampler (TP8, R3, top-128 log-probs, max_model_len 262144
# and 1048576) started at 256 GiB per GPU and grew in ~3.5 GiB steps to 275 GiB (95.5%) within
# 30 minutes of serving, outside vLLM's budget (allocator growth of the serving path, the
# collectives' scratch); an in-place expert reload then stages each layer's released experts
# and their kernel-format copy on the same GPUs, the margin a 0.88 run lost (99.8% and a
# killed sampler). 0.84 keeps ~11 GiB more free and still leaves a 2.4M+ token KV cache.
# A --gpu-memory-utilization above K31_ROCM_MAX_GPU_MEMORY_UTILIZATION is refused on the
# ROCm presets (VllmOptions.allow_high_gpu_memory overrides it).
K31_ROCM_GPU_MEMORY_UTILIZATION = "0.84"
K31_ROCM_MAX_GPU_MEMORY_UTILIZATION = 0.86
#
# VLLM_USE_BREAKABLE_CUDAGRAPH (prefill graphs) is left out with the capture list: its
# graph memory comes out of the KV cache (the recipe's 2..576 list did not start).
K31_INDEXER_FP8_OVERRIDE = {"text_config": {"dsa_config": {"indexer_use_fp8": True}}}
_K31_ROCM_LAYOUT = ("--language-model-only", "--dcp-comm-backend", "a2a", "--cp-kv-cache-interleave-size", "1",
                    "--kv-cache-dtype", "fp8", "--enable-prefix-caching", "--prefix-match-unit", "64",
                    "--attention-config", '{"indexer_kv_dtype":"mxfp4","mla_prefill_backend":"ROCM_AITER_FA"}',
                    "--additional-config", '{"kda_prefill_backend":"fused"}',
                    "--max-num-batched-tokens", "10240", "--gpu-memory-utilization", K31_ROCM_GPU_MEMORY_UTILIZATION)
_K31_ROCM_ENV = {"VLLM_ROCM_DCP_AITER_ALLGATHER": "1", "VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA": "1",
                 "AITER_FLYDSL_SPARSE_MLA": "1"}
K31_PRESETS: dict[str, dict[str, Any]] = {
    "amd-article": {
        "platform": "rocm", "dcp": True,
        "args": (*_K31_ROCM_LAYOUT, "--quantization-config", '{"targets":{"*in_proj_qkvgfab":"fp8_per_channel"}}'),
        "env": {**_K31_ROCM_ENV, "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "INT4",
                "VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_M8": "16", "AITER_FLYDSL_STAGE2_FP8": "1"},
        "hf_overrides": K31_INDEXER_FP8_OVERRIDE},
    "rl-precision": {
        "platform": "rocm", "dcp": True,
        "args": (*_K31_ROCM_LAYOUT, "--mamba-ssm-cache-dtype", "float32"),
        "env": {**_K31_ROCM_ENV, "VLLM_ROCM_QUICK_REDUCE_QUANTIZATION": "NONE", "AITER_FLYDSL_STAGE2_FP8": "0"},
        "hf_overrides": K31_INDEXER_FP8_OVERRIDE},
    "vllm-recipe": {
        "platform": "cuda", "dcp": False, "args": ("--language-model-only", "--load-format", "fastsafetensors"),
        "env": {}, "hf_overrides": {}},
    "rl-precision-cuda": {
        "platform": "cuda", "dcp": False,
        "args": ("--language-model-only", "--load-format", "fastsafetensors", "--mamba-ssm-cache-dtype", "float32"),
        "env": {}, "hf_overrides": {}},
}


def flag_values(args: Sequence[str], flags: Sequence[str]) -> list[tuple[int, int, str, str]]:
    """``(start, length, flag, value)`` of every ``flag value`` / ``flag=value`` of
    ``flags`` in ``args`` (a flag without a value has value ``""``)."""
    out = []
    i = 0
    while i < len(args):
        arg = str(args[i])
        name, eq, value = arg.partition("=")
        if eq and name in flags:
            out.append((i, 1, name, value))
        elif arg in flags:
            has_value = i + 1 < len(args) and not str(args[i + 1]).startswith("-")
            out.append((i, 2 if has_value else 1, arg, str(args[i + 1]) if has_value else ""))
            i += 1 if has_value else 0
        i += 1
    return out


def gpu_memory_utilization(args: Sequence[str]) -> float | None:
    """The ``--gpu-memory-utilization`` vLLM's argparse takes from ``args`` (the last one)."""
    found = flag_values(args, ("--gpu-memory-utilization",))
    if not found:
        return None
    value = found[-1][3]
    try:
        return float(value)
    except ValueError:
        raise SamplerEngineError(f"vllm-server: --gpu-memory-utilization {value!r} is not a number") from None


def _size(value: str, flag: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise SamplerEngineError(f"vllm-server: {flag} {value!r} is not a size") from None


def drop_cp_sizes(args: Sequence[str]) -> tuple[list[str], list[dict[str, Any]]]:
    """``args`` without the DCP / PCP sizes above 1, and what was dropped
    (``{"flag", "size", "axis"}``); sizes of 1 stay."""
    drop: set[int] = set()
    dropped = []
    for start, length, flag, value in flag_values(args, tuple(CP_SIZE_FLAGS)):
        size = _size(value, flag)
        if size > 1:
            drop.update(range(start, start + length))
            dropped.append({"flag": flag, "size": size, "axis": CP_SIZE_FLAGS[flag]})
    return [a for i, a in enumerate(args) if i not in drop], dropped


def merge_overrides(*parts: dict[str, Any]) -> dict[str, Any]:
    """``--hf-overrides`` mappings merged key by key (nested mappings recursively);
    a leaf given twice with different values is an error."""
    out: dict[str, Any] = {}
    for part in parts:
        for key, value in (part or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = merge_overrides(out[key], value)
            elif key in out and out[key] != value:
                raise SamplerEngineError(f"vllm hf_overrides: {key!r} is {out[key]!r} and {value!r}")
            else:
                out[key] = value
    return out


@dataclass
class VllmOptions:
    tp: int = 4
    max_num_seqs: int | None = None  # measured with 8 / 32 / 64
    served_model_name: str = "yeto-policy"
    allow_eager: bool = False
    # refuse nothing for a --gpu-memory-utilization above the ROCm presets' ceiling
    allow_high_gpu_memory: bool = False
    # score_centering: K top candidates per response token (``logprobs: K``; 0: the
    # sampled token only). On the upstream line the server's --max-logprobs follows K.
    top_logprobs_num: int = 0
    # the vLLM line (module doc): None / "npu" legacy, "cuda" / "rocm" upstream main
    platform: str | None = None
    # R3, upstream line only: --enable-return-routed-experts; each choice's
    # ``routed_experts`` becomes Completion.routed_experts (uint16[T, L_moe, top_k])
    return_routed_experts: bool = False
    # --hf-overrides head_dtype float32, upstream line only (recorded in every
    # sample's engine.version_fields for the trainer's check)
    fp32_lm_head: bool = False
    # MoE layer count of the served model (the L_moe of the R3 array); None: read
    # ``num_hidden_layers - first_k_dense_replace`` from the release's config.json
    moe_layers: int | None = None
    # tool and reasoning parsers (the /v1/chat/completions and /v1/responses
    # front ends; K3.1: "kimi_k3" for both). A tool parser turns auto tool choice on
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    # extra --hf-overrides entries (merged with the preset's and the fp32 head; one flag)
    hf_overrides: dict[str, Any] = field(default_factory=dict)
    # a Kimi-K3.1-Pro serving preset (K31_PRESETS; its platform must match): flags and
    # environment ahead of ServerConfig.extra_args / extra_env, which come later and win
    k31_preset: str | None = None
    # the ``GET /version`` the pinned image reports (sampler_pins.sampler_vllm_version of the
    # platform; the spec carries it, adapters do not import each other). A server that
    # reports another version is recorded (server_info.vllm_pin_match), not refused
    pin_version: str | None = None
    # in_place_reload of a partial-experts overlay through the yeto worker extension
    # (vllm_expert_reload: only the released experts, the cold start's transform);
    # the yeto package must be importable in the server (PYTHONPATH)
    expert_reload: bool = False
    # the in-place expert reload's memory guard (yeto.rl.engine.vram_guard.reload_plan):
    # GiB left free on the fullest GPU while it stages; the reload goes in smaller
    # chunks, or by restart, when the staging would not fit
    reload_reserve_gib: float = 2.0


def decode_top_logprobs(logprobs: dict[str, Any], *, k: int) -> tuple[Any, Any] | None:
    """``(ids int32[T, K], logprobs float32[T, K])`` from a choice's ``logprobs``
    (``top_logprobs``: one ``{"token_id:<id>": logprob}`` mapping per position,
    sorted by log-prob; fewer than K entries are padded with -1 / -inf)."""
    from yeto.rl.engine.buffer_payload import TOP_LOGPROBS_DTYPE, TOP_LOGPROBS_IDS_DTYPE

    if k <= 0:
        return None
    rows = logprobs.get("top_logprobs")
    if rows is None:
        raise SamplerEngineError(f"vllm returned no top_logprobs; {k} per token were asked")
    ids = np.full((len(rows), k), -1, dtype=TOP_LOGPROBS_IDS_DTYPE)
    vals = np.full((len(rows), k), -np.inf, dtype=TOP_LOGPROBS_DTYPE)
    for t, row in enumerate(rows):
        ranked = sorted(((float(v), _token_id(tok)) for tok, v in (row or {}).items()), reverse=True)[:k]
        for c, (value, token) in enumerate(ranked):
            vals[t, c], ids[t, c] = value, token
    return ids, vals


def decode_routed_experts(encoded: str | None, *, prompt_tokens: int, completion_tokens: int,
                          moe_layers: int, dense_layers: int = 0) -> Any:
    """A choice's ``routed_experts`` (base64 ``.npy``, module doc) as
    ``uint16[T, L_moe, top_k]`` with ``T = prompt_tokens + completion_tokens - 1``
    (the Miles decoding). The server's layer axis is every model layer: when it is
    ``moe_layers + dense_layers`` long the leading ``dense_layers`` rows must be
    all zero and are dropped; an axis of ``moe_layers`` is taken as is. None when
    the choice has no field (the server runs without R3 or the request was aborted
    before a forward pass)."""
    from yeto.rl.engine.buffer_payload import routed_experts_uint16

    if encoded is None:
        return None
    try:
        array = np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)
    except (ValueError, OSError) as error:
        raise SamplerEngineError(f"vllm routed_experts is not base64 .npy: {error}") from None
    rows = int(prompt_tokens) + int(completion_tokens) - 1
    if array.ndim != 3 or array.dtype.kind not in "iu":
        raise SamplerEngineError(f"vllm routed_experts has shape {array.shape} dtype {array.dtype}; "
                                 "expected an integer [T, layers, top_k] array")
    if array.shape[0] != rows:
        raise SamplerEngineError(f"vllm routed_experts has {array.shape[0]} rows for {rows} positions "
                                 f"({prompt_tokens} prompt + {completion_tokens} output tokens - 1); a "
                                 "routed_experts_prompt_start or a prefix-cache hit without stored routing?")
    layers, dense = int(moe_layers), max(int(dense_layers or 0), 0)
    if dense and array.shape[1] == layers + dense:
        head = array[:, :dense]
        if head.size and int(head.max()) != 0:
            raise SamplerEngineError(f"vllm routed_experts: the {dense} leading dense layer(s) carry expert "
                                     f"ids (max {int(head.max())}); the model layout is not the one expected")
        array = array[:, dense:]
    elif array.shape[1] != layers:
        raise SamplerEngineError(f"vllm routed_experts has {array.shape[1]} layers; the model has {layers} MoE "
                                 f"layers" + (f" after {dense} dense layer(s)" if dense else ""))
    if array.shape[2] > 1 and array.size and int(array.max()) == 0:
        raise SamplerEngineError("vllm routed_experts are all 0: no MoE router wrote the capture buffer")
    return routed_experts_uint16(np.ascontiguousarray(array), where="vllm routed_experts")


def _token_id(token: str) -> int:
    if not isinstance(token, str) or not token.startswith("token_id:"):
        raise SamplerEngineError(f"vllm token {token!r} is not 'token_id:<id>'; "
                                 "the server ignored return_tokens_as_token_ids")
    return int(token.split(":", 1)[1])


def summarize_version(info: Any) -> dict[str, Any]:
    """The facts of ``GET /version`` the adapter records (``{"version": ...}``)."""
    if not isinstance(info, dict):
        return {}
    return {key: info[key] for key in VERSION_KEYS if key in info}


class VllmServerEngine(ServerEngine):
    ENGINE = "vllm"
    HANG_HINT = LEGACY_HANG_HINT
    REFUSED_FORMATS = LEGACY_REFUSED_FORMATS

    def __init__(self, config: ServerConfig, options: VllmOptions | None = None, *, run: Any = None,
                 card_memory: Any = None, **kw: Any) -> None:
        """``run(argv) -> str``: runs the card tool of the ROCm check (default:
        ``yeto.hw.card_telemetry.run_command``). ``card_memory() -> [card sample]``:
        the per-GPU memory the in-place reload guard reads (default: the card
        tool of ``config.device_family``)."""
        options = options or VllmOptions()
        if options.platform is not None and options.platform not in PLATFORMS:
            raise SamplerEngineError(f"vllm platform must be one of {PLATFORMS} or None, got {options.platform!r}")
        upstream = options.platform in UPSTREAM_PLATFORMS
        if not upstream:
            if config.quant_scheme != "bf16" and config.expected_format is None:
                config.expected_format = INT_QUANTIZED  # item 17
            if config.in_place_reload:
                raise SamplerEngineError("vllm-server: in-place reload is not built for vllm-ascend / the legacy "
                                         "line; restart is the only reload mode until an on-card test covers it")
            for name in ("return_routed_experts", "fp32_lm_head"):
                if getattr(options, name):
                    raise SamplerEngineError(f"vllm-server {name} needs upstream vLLM (platform cuda or rocm), "
                                             f"not platform {options.platform!r}")
        super().__init__(config, **kw)
        self.last_resume_failure: dict[str, Any] | None = None
        self.options = options
        self.upstream = upstream
        self.HANG_HINT = UPSTREAM_HANG_HINT if upstream else LEGACY_HANG_HINT
        self.REFUSED_FORMATS = UPSTREAM_REFUSED_FORMATS if upstream else LEGACY_REFUSED_FORMATS
        if options.k31_preset is not None:
            preset = K31_PRESETS.get(options.k31_preset)
            if preset is None:
                raise SamplerEngineError(f"vllm k31_preset must be one of {sorted(K31_PRESETS)}, "
                                         f"got {options.k31_preset!r}")
            if preset["platform"] != options.platform:
                raise SamplerEngineError(f"vllm k31_preset {options.k31_preset!r} is a {preset['platform']} preset; "
                                         f"the server runs on {options.platform!r}")
        self._run = run
        self._card_memory = card_memory
        self._served_dir: Path | None = None  # the directory the running server loaded
        self._moe_layers: int | None = options.moe_layers
        self._dense_layers = 0
        self.server_info: dict[str, Any] = {}
        if "--enforce-eager" in config.extra_args and not options.allow_eager and not upstream:
            raise SamplerEngineError("vllm-server: --enforce-eager serves about 14x slower on 910B "
                                     "(report item 18); set allow_eager only for debugging")
        if upstream and "--hf-overrides" in config.extra_args:
            raise SamplerEngineError("vllm-server: give --hf-overrides as VllmOptions.hf_overrides (merged with "
                                     "the fp32 head into one flag), not in extra_args")
        if self.preset.get("platform") == "rocm" and not options.allow_high_gpu_memory:
            gmu = gpu_memory_utilization([*(self.preset.get("args") or ()), *config.extra_args])
            if gmu is not None and gmu > K31_ROCM_MAX_GPU_MEMORY_UTILIZATION:
                raise SamplerEngineError(
                    f"vllm-server: --gpu-memory-utilization {gmu} on the ROCm K3.1 preset "
                    f"{options.k31_preset!r} is above {K31_ROCM_MAX_GPU_MEMORY_UTILIZATION}: a TP8 MI355X sampler "
                    f"grows ~20 GiB per GPU past vLLM's budget while it serves and an in-place reload stages "
                    f"experts on top of that (0.88 reached 95.5% before any reload); use the preset's "
                    f"{K31_ROCM_GPU_MEMORY_UTILIZATION} or set engine.options.allow_high_gpu_memory")
        self.cp_overrides: list[dict[str, Any]] = self._r3_cp_overrides() if upstream else []
        if upstream and options.return_routed_experts:
            self._check_r3_layout()

    @property
    def top_logprobs_num(self) -> int:
        """The port capability (``sampler_engine.top_logprobs_num``)."""
        return max(0, int(self.options.top_logprobs_num))

    @property
    def returns_routed_experts(self) -> bool:
        """The port capability (``sampler_engine.returns_routed_experts``): R3 on."""
        return bool(self.options.return_routed_experts)

    @property
    def fp32_lm_head(self) -> bool:
        """``engine.version_fields.fp32_lm_head`` of every sample (``sampler_engine.version_fields``)."""
        return bool(self.options.fp32_lm_head)

    @property
    def preset(self) -> dict[str, Any]:
        """The K3.1 preset's flags / environment / overrides (empty without one)."""
        return K31_PRESETS[self.options.k31_preset] if self.options.k31_preset else {}

    def _r3_cp_overrides(self) -> list[dict[str, Any]]:
        """The DCP / PCP sizes an R3 server drops (module doc): the preset's DCP
        (``tp``) and every size above 1 in the preset's flags or ``extra_args``.
        Empty without R3 (nothing is changed then)."""
        if not self.options.return_routed_experts:
            return []
        out = []
        if self.preset.get("dcp") and int(self.options.tp) > 1:
            out.append({"flag": "--decode-context-parallel-size", "size": int(self.options.tp), "axis": "DCP",
                        "source": f"k31_preset {self.options.k31_preset}"})
        for source, args in ((f"k31_preset {self.options.k31_preset}", self.preset.get("args") or ()),
                             ("extra_args", self.config.extra_args)):
            out += [{**d, "source": source} for d in drop_cp_sizes(list(args))[1]]
        return [{**d, "now": 1} for d in out]

    def _check_r3_layout(self) -> None:
        """Refuse, before any start, the other layouts vLLM's AuxOutput check
        rejects for R3 (module doc)."""
        args = [*(self.preset.get("args") or ()), *self.config.extra_args]
        for _, _, flag, value in flag_values(args, PP_SIZE_FLAGS):
            if _size(value, flag) > 1:
                raise SamplerEngineError(f"vllm-server: R3 (routed_experts) with {flag} {value}: vLLM refuses "
                                         "--enable-return-routed-experts with pipeline parallelism (PP > 1)")
        if "--no-enable-prefix-caching" in args:
            raise SamplerEngineError("vllm-server: R3 (routed_experts) with --no-enable-prefix-caching: vLLM's "
                                     "AuxOutput connector requires prefix caching")
        runner = str(self.env().get("VLLM_USE_V2_MODEL_RUNNER", "")).strip().lower()
        if runner in ("0", "false"):
            raise SamplerEngineError("vllm-server: R3 (routed_experts) with VLLM_USE_V2_MODEL_RUNNER=0: vLLM's "
                                     "AuxOutput connector requires Model Runner V2")

    def r3_layout_note(self) -> str | None:
        """The log line of the R3 DCP / PCP override (None without one)."""
        if not self.cp_overrides:
            return None
        what = ", ".join(f"{d['flag']} {d['size']} ({d['source']})" for d in self.cp_overrides)
        return (f"R3 on: {R3_CP_RULE}; dropped {what}, so DCP / PCP are 1 and the MLA KV cache is no longer "
                f"split over the {self.options.tp} TP ranks (about 1/{self.options.tp} of the KV token capacity)")

    def hf_overrides(self) -> dict[str, Any]:
        """The ``--hf-overrides`` mapping of the upstream line (empty: no flag): the
        preset's, the options', and the fp32 head."""
        return merge_overrides(self.preset.get("hf_overrides") or {}, dict(self.options.hf_overrides or {}),
                               FP32_HEAD_OVERRIDE if self.options.fp32_lm_head else {})

    def argv(self, model_dir: Path) -> list[str]:
        if self.upstream:
            return self._upstream_argv(model_dir)
        c, o = self.config, self.options
        argv = [c.python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model_dir),
                "--host", c.host, "--port", str(c.port), "--tensor-parallel-size", str(o.tp),
                "--max-model-len", str(c.max_model_len), "--served-model-name", o.served_model_name,
                "--trust-remote-code"]
        if o.max_num_seqs is not None:
            argv += ["--max-num-seqs", str(int(o.max_num_seqs))]
        return argv + self.base_tokenizer_argv(model_dir) + list(c.extra_args)

    def base_tokenizer_argv(self, model_dir: Path) -> list[str]:
        """A release is served with the tokenizer of the base checkpoint: RL never
        changes the tokenizer, and the trainer's HF export rewrites its files in the
        trainer's transformers format (transformers 5 writes ``extra_special_tokens``
        as a list, which the 4.x tokenizer of a sampler image cannot read: the
        restart of the engine on that release then fails at start). ``--tokenizer``
        in ``extra_args`` wins."""
        if "--tokenizer" in self.config.extra_args or Path(model_dir) == Path(self.config.base_dir):
            return []
        return ["--tokenizer", str(self.config.base_dir)]

    def _upstream_argv(self, model_dir: Path) -> list[str]:
        c, o = self.config, self.options
        argv = [c.python, "-m", "vllm.entrypoints.cli.main", "serve", str(model_dir),
                "--host", c.host, "--port", str(c.port), "--tensor-parallel-size", str(o.tp),
                "--max-model-len", str(c.max_model_len), "--served-model-name", o.served_model_name,
                "--trust-remote-code"]
        r3 = bool(o.return_routed_experts)  # R3: no context parallelism (module doc)
        if self.preset.get("dcp") and not r3:
            argv += ["--decode-context-parallel-size", str(o.tp)]
        if o.max_num_seqs is not None:
            argv += ["--max-num-seqs", str(int(o.max_num_seqs))]
        argv += self.base_tokenizer_argv(model_dir)
        preset_args = list(self.preset.get("args") or ())
        argv += drop_cp_sizes(preset_args)[0] if r3 else preset_args
        if self.top_logprobs_num > DEFAULT_MAX_LOGPROBS and "--max-logprobs" not in c.extra_args:
            argv += ["--max-logprobs", str(self.top_logprobs_num)]
        if o.return_routed_experts:
            argv += ["--enable-return-routed-experts"]
        if o.expert_reload and c.in_place_reload:
            argv += ["--worker-extension-cls", vllm_expert_reload.EXTENSION]
        overrides = self.hf_overrides()
        if overrides:
            argv += ["--hf-overrides", json.dumps(overrides, sort_keys=True, separators=(",", ":"))]
        if o.tool_call_parser:
            argv += ["--enable-auto-tool-choice", "--tool-call-parser", o.tool_call_parser]
        if o.reasoning_parser:
            argv += ["--reasoning-parser", o.reasoning_parser]
        return argv + (drop_cp_sizes(list(c.extra_args))[0] if r3 else list(c.extra_args))

    def env(self) -> dict[str, str]:
        env = super().env()
        platform = self.options.platform
        if platform == "rocm":  # the presets first: extra_env (super) wins
            env = {**ROCM_ENV_PRESET, **(self.preset.get("env") or {}), **env}
        elif platform == "cuda":
            env = {**CUDA_ENV_PRESET, **(self.preset.get("env") or {}), **env}
        if self.upstream and self.config.in_place_reload:
            env.setdefault(DEV_MODE_ENV, "1")
        if not self.upstream and self.config.precision == "w4a8":
            env["VLLM_ASCEND_W4A8_CPU_STAGE"] = "1"  # int8 staging in host memory
        return env

    # ------------------------------------------------------------------ start
    def preflight(self, model_dir: Path) -> None:
        super().preflight(model_dir)
        if self.options.platform == "rocm":
            self._check_rocm_cards()
        note = self.r3_layout_note()
        if note and self._process is None:  # once per start, not per in-place reload
            print(f"[vllm-server] {note}", flush=True)
        if self.options.return_routed_experts:
            self._dense_layers = dense_layer_count(model_dir)
            if self.options.moe_layers is None:
                self._moe_layers = moe_layer_count(model_dir)
                if self._moe_layers is None:
                    raise SamplerEngineError(f"{model_dir}: config.json gives no MoE layer count; R3 needs "
                                             "VllmOptions.moe_layers to shape the routed experts")

    def _check_rocm_cards(self) -> None:
        """At least ``tp`` AMD GPUs must be visible before the server starts (a
        missing card hangs the RCCL setup without an error)."""
        if self._run is None:
            from yeto.hw.card_telemetry import run_command  # noqa: PLC0415

            self._run = run_command
        count = rocm_card_count(self._run)
        if count < int(self.options.tp):
            raise SamplerEngineError(f"vllm on ROCm: the card tool lists {count} GPU(s), TP {self.options.tp} "
                                     "needs that many")

    def after_ready(self) -> dict[str, Any]:
        if not self.upstream:
            return {}
        try:
            info = summarize_version(self.call("GET", "/version"))
        except SamplerEngineError:
            info = {}
        pin = self.options.pin_version
        if info.get("version") and pin is not None:
            info["vllm_pin_version"] = pin
            info["vllm_pin_match"] = str(info["version"]) == pin  # recorded, not refused
        self.server_info = info
        detail: dict[str, Any] = {"server_info": dict(info)}
        if self.cp_overrides:
            detail["r3_cp_overrides"] = [dict(d) for d in self.cp_overrides]
        return detail

    def start(self, release_dir: Path | None, version: int):
        report = super().start(release_dir, version)
        self._served_dir = Path(release_dir) if release_dir is not None else Path(self.config.base_dir)
        return report

    def reload_in_place(self, model_dir: Path, version: int) -> dict[str, Any]:
        """``in_place_reload`` on the upstream line (module doc): pause with drain,
        ``reload_weights`` from ``model_dir`` on every rank, resume (also after a
        failure, which is then raised)."""
        if not self.upstream:
            raise SamplerEngineError("vllm-server: in-place reload needs the upstream line (platform cuda / rocm)")
        if self.options.expert_reload:
            return self._reload_experts(model_dir)
        result = self._paused_rpc({"method": "reload_weights", "kwargs": {"weights_path": str(model_dir)}})
        return {"reload": "collective_rpc reload_weights", "rpc_results": len((result or {}).get("results") or [])}

    def _reload_experts(self, model_dir: Path) -> dict[str, Any]:
        """The overlay's released experts through the worker extension
        (``vllm_expert_reload``); a release that is not an overlay of the served
        base and expert selection (``k31_release.overlay_update_plan``) restarts."""
        served = self._served_dir or Path(self.config.base_dir)
        try:
            plan = k31_release.overlay_update_plan(model_dir, served, self.config.base_dir)
        except k31_release.FileSparseUnavailable as err:
            plan, why = None, err.reason
        else:
            why = "not_an_overlay"
        if plan is None:
            report = self.start(model_dir, self._version or 0)
            self._served_dir = Path(model_dir)
            return {"reload": "restart", "expert_reload_off": why, "restart_s": report.pause_s}
        guard = self._reload_memory_guard(model_dir, plan)
        if guard["mode"] == vram_guard.SAFE_RESTART:
            # not even one expert per chunk fits next to the served weights: a restart
            # frees the GPUs first instead of staging into memory that is not free
            report = self.start(model_dir, self._version or 0)
            self._served_dir = Path(model_dir)
            return {"reload": "restart", "expert_reload_off": "gpu_memory", "restart_s": report.pause_s,
                    "memory_guard": guard}
        kwargs = {"weights_path": str(model_dir), "files": json.dumps(list(plan.files))}
        if guard["mode"] == vram_guard.SAFE_CHUNKED:
            kwargs["max_experts"] = str(int(guard["max_experts"]))
        began = self._clock()
        result = self._paused_rpc({"method": vllm_expert_reload.RELOAD_METHOD, "kwargs": kwargs})
        self._served_dir = Path(model_dir)
        ranks = (result or {}).get("results") or []
        return {"reload": "collective_rpc " + vllm_expert_reload.RELOAD_METHOD, "rpc_results": len(ranks),
                "experts": plan.expert_count, "pause_s": round(self._clock() - began, 3),
                "rank_seconds": max((r.get("seconds", 0.0) for r in ranks if isinstance(r, dict)), default=None),
                "memory_guard": guard}

    def _paused_rpc(self, body: dict[str, Any]) -> Any:
        """``/pause?mode=wait`` -> ``/collective_rpc`` -> ``/resume``, resuming also
        after a failed RPC. The RPC's own error is what propagates; a ``/resume``
        that keeps failing is attached to it (or raised when the RPC succeeded)."""
        self.call("POST", "/pause?mode=wait")
        try:
            result = self.call("POST", "/collective_rpc", body)
        except BaseException as err:
            self._resume(cause=err)
            raise
        self._resume(cause=None)
        return result

    def _resume(self, *, cause: BaseException | None) -> None:
        """``/resume``, retried. A server that cannot be resumed would stay paused
        (alive, healthy, never generating) without anybody noticing, so it is
        killed instead: the failure is recorded in ``last_resume_failure``, the
        next ``call`` fails loudly and the next load restarts it."""
        errors: list[str] = []
        for attempt in range(RESUME_ATTEMPTS):
            try:
                self.call("POST", "/resume")
                return
            except Exception as err:  # noqa: BLE001 - every failure is recorded below
                errors.append(f"{type(err).__name__}: {err}"[:300])
                if attempt + 1 < RESUME_ATTEMPTS:
                    self._sleep(RESUME_RETRY_S)
        self.last_resume_failure = {"attempts": len(errors), "errors": errors,
                                    "after": "failed rpc" if cause is not None else "rpc"}
        note = (f"vllm-server: /resume failed {len(errors)} times after the reload "
                f"(last: {errors[-1]}); the paused server was killed so the next load restarts it")
        print(f"[vllm-server] {note}", flush=True)
        self._kill()
        if cause is not None:
            cause.add_note(note)
            return
        raise SamplerEngineError(note)

    def _free_bytes(self) -> int | None:
        """Free memory of the fullest GPU (None: no reading; the guard then lets the
        reload run as before)."""
        reader = self._card_memory
        if reader is None:
            from yeto.hw.card_telemetry import collector_for, run_command  # noqa: PLC0415
            from yeto.rl.engine.telemetry import load_nvml  # noqa: PLC0415

            family = "rocm" if self.options.platform == "rocm" else "cuda"
            try:
                collector = collector_for(family, nvml_loader=load_nvml, run=self._run or run_command)
                collector.open()
            except Exception:  # noqa: BLE001 - no card tool on this image
                return None
            reader = self._card_memory = collector.sample
        try:
            return vram_guard.min_free_bytes(reader())
        except Exception:  # noqa: BLE001
            return None

    def _reload_memory_guard(self, model_dir: Path, plan: Any) -> dict[str, Any]:
        """Free GPU memory against the release's staging (``vram_guard.reload_plan``);
        short: the engine's caching allocator is emptied on every rank and the cards
        are read again before the reload is cut into chunks or turned into a restart."""
        try:
            layers = vllm_expert_reload.group_by_layer(vllm_expert_reload.released_tensors(model_dir, plan.files))
            estimate = vram_guard.staging_bytes(vram_guard.layer_bytes(model_dir, layers), layers,
                                                tp=int(self.options.tp))
        except Exception as error:  # noqa: BLE001 - the extension refuses a bad release itself
            return {"mode": vram_guard.SAFE_IN_PLACE, "estimate_error": f"{type(error).__name__}: {error}"[:200]}
        reserve = int(float(self.options.reload_reserve_gib) * vram_guard.GIB)
        guard = vram_guard.reload_plan(self._free_bytes(), estimate, reserve_bytes=reserve)
        if guard["mode"] == vram_guard.SAFE_IN_PLACE:
            return guard
        try:
            self.call("POST", "/collective_rpc", {"method": vllm_expert_reload.RELEASE_CACHE_METHOD, "kwargs": {}})
        except SamplerEngineError as error:
            guard["empty_cache_error"] = str(error)[:200]
        first = guard
        guard = vram_guard.reload_plan(self._free_bytes(), estimate, reserve_bytes=reserve)
        guard["first_free_bytes"] = first["free_bytes"]
        guard["emptied_cache"] = "empty_cache_error" not in first
        print(f"[vllm-server] in-place reload memory guard: {guard}", flush=True)
        return guard

    # ------------------------------------------------------------------ requests
    def _body(self, prompt: Any, sampling: SamplingParams, **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.options.served_model_name, "prompt": prompt, "n": int(sampling.n),
            "max_tokens": int(sampling.max_tokens), "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p), "top_k": int(sampling.top_k),
            "repetition_penalty": float(sampling.repetition_penalty),
            "frequency_penalty": float(sampling.frequency_penalty),
            "presence_penalty": float(sampling.presence_penalty),
            "logprobs": max(1, self.top_logprobs_num), "return_tokens_as_token_ids": True, **extra}
        if self.options.return_routed_experts:
            body["return_token_ids"] = True  # the prompt length of a text prompt (R3 rows)
        if sampling.stop:
            body["stop"] = list(sampling.stop)
        if sampling.seed is not None:
            body["seed"] = int(sampling.seed)
        return body

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        return self._complete(list(prompts), sampling)

    def generate_tokens(self, prompts: Sequence[Sequence[int]], sampling: SamplingParams, *,
                        stop_token_ids: Sequence[int] = ()) -> list[list[Completion]]:
        """Token-id prompts (a list of id lists as ``prompt``), for the records
        proxy. vLLM does not re-tokenize an id prompt, so the
        template's own BOS tokens are not doubled."""
        ids = [[int(t) for t in p] for p in prompts]
        extra = {"stop_token_ids": [int(t) for t in stop_token_ids]} if stop_token_ids else {}
        out = self._complete(ids, sampling, prompt_ids=ids, **extra)
        return [[replace(c, prompt_token_ids=tuple(ids[i])) for c in group] for i, group in enumerate(out)]

    def _complete(self, prompts: list[Any], sampling: SamplingParams, *, prompt_ids: list[list[int]] | None = None,
                  **extra: Any) -> list[list[Completion]]:
        n = int(sampling.n)
        body = self.call("POST", "/v1/completions", self._body(prompts, sampling, **extra))
        choices = sorted(body.get("choices") or [], key=lambda c: int(c["index"]))
        if len(choices) != len(prompts) * n:
            raise SamplerEngineError(f"vllm returned {len(choices)} choices for {len(prompts)} x {n}")
        out: list[list[Completion]] = []
        for i in range(len(prompts)):
            group = []
            for choice in choices[i * n:(i + 1) * n]:
                logprobs = choice.get("logprobs") or {}
                reason = choice.get("finish_reason")
                if reason not in ("stop", "length"):
                    raise SamplerEngineError(f"vllm finish_reason {reason!r}")
                top = decode_top_logprobs(logprobs, k=self.top_logprobs_num)
                tokens = tuple(_token_id(t) for t in logprobs.get("tokens") or [])
                prompt = tuple(int(t) for t in (choice.get("prompt_token_ids")
                                                or (prompt_ids[i] if prompt_ids is not None else ())))
                group.append(Completion(
                    token_ids=tokens,
                    logprobs=tuple(float(x) for x in logprobs.get("token_logprobs") or []),
                    text=str(choice.get("text", "")),
                    finish_reason=FINISH_LENGTH if reason == "length" else FINISH_STOP,
                    prompt_token_ids=prompt,
                    routed_experts=self._routed_experts(choice, prompt_tokens=len(prompt),
                                                        completion_tokens=len(tokens)),
                    top_logprobs_ids=None if top is None else top[0],
                    top_logprobs=None if top is None else top[1]))
            out.append(group)
        return out

    def _routed_experts(self, choice: dict[str, Any], *, prompt_tokens: int, completion_tokens: int) -> Any:
        if not self.options.return_routed_experts:
            return None
        encoded = choice.get("routed_experts")
        if encoded is None:
            raise SamplerEngineError("vllm returned no routed_experts although the server was started with "
                                     "--enable-return-routed-experts")
        if not prompt_tokens:
            raise SamplerEngineError("vllm returned no prompt_token_ids (return_token_ids); the R3 rows "
                                     "cannot be checked")
        if self._moe_layers is None:
            raise SamplerEngineError("vllm R3: the MoE layer count is unknown (the server was not started)")
        return decode_routed_experts(encoded, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                     moe_layers=self._moe_layers, dense_layers=self._dense_layers)

    def _echo_logprobs(self, text: str) -> list[float | None]:
        greedy = SamplingParams(max_tokens=1, temperature=0.0)
        body = self.call("POST", "/v1/completions", self._body(text, greedy, echo=True))
        choice = (body.get("choices") or [{}])[0]
        values = (choice.get("logprobs") or {}).get("token_logprobs") or []
        return list(values[:-1])  # the last value is the one generated token

    def score(self, prompt: str, continuation: str) -> list[float]:
        prompt_len = len(self._echo_logprobs(prompt))
        tail = self._echo_logprobs(prompt + continuation)[prompt_len:]
        if not tail or any(x is None for x in tail):
            raise SamplerEngineError(f"vllm gave no prompt log-probs for continuation {continuation!r}")
        return [float(x) for x in tail]
