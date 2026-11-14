"""``sglang-server``: an SGLang HTTP server behind the ``SamplerEngine`` port
(PPU, stage 1: TP8 per machine, MXFP4 experts RTN; user decision 2026-10-19).

Vendor facts handled here (A-INF §3; none verified on a PPU by this code):

* item 1: SGLang's ``load_weights`` does not map ``weight_packed`` to
  ``weight``; expert weights stored as ``weight_packed`` are skipped without an
  error. A release whose index names ``*.weight_packed`` is refused unless the
  image carries the rename patch (``renames_weight_packed=True``). The yeto
  patch (:mod:`.sglang_load_report`, applied to the image's model file) renames
  when ``YETO_SGLANG_RENAME_WEIGHT_PACKED=1`` and, with ``load_report_dir``,
  reports the expert matrices each rank loaded (post-load check 1);
* item 5: an MXFP4 export with ``quantization_config.format == "dense"``
  (llmcompressor 0.14) is read wrongly; ``dense`` is refused for every
  quantized scheme;
* item 7: the MXFP4 W4A16 path needs ``MX_PATCH=1``; set when the precision is
  ``mxfp4`` and recorded in the load report;
* item 8: dp8 + DeepEP cannot run NEXTN; the pair is refused before start;
* item 9: dp8 + DeepGEMM compiles per new batch size M (75-113 s each); with
  ``warmup_batch_sizes`` the adapter sends one batch per size before the
  engine is checked and records the seconds;
* item 13: TP8 exits 134 from ``pcclCommAbort`` after a stop request; normal
  then, a failure while serving.

API calls (SGLang native HTTP API; the exact fields of SGLang 0.5.17 on PPU are
NOT verified): ``GET /health``; ``POST /generate`` with ``return_logprob``
(``meta_info.output_token_logprobs`` = ``[logprob, token_id, ...]`` per
token, ``meta_info.finish_reason.type``); teacher-forced scores from
``meta_info.input_token_logprobs`` with ``max_new_tokens: 0``;
``POST /update_weights_from_disk`` for an in-place reload (off unless
``in_place_reload``: refit of quantized experts is not tested, A-INF 2.1).

Platform rule of the in-place reload (SMP, 2026-11-09): with MXFP4 experts the
in-place paths crash the server on ROCm (aiter shuffle replaces the parameters,
MI355X) and on pre-Blackwell CUDA (``upcast_from_mxfp`` to bf16, H100 / H200):
``'Parameter' object has no attribute 'weight_loader'``. There ``load`` restarts
the server on the release whatever the spec says (``IN_PLACE_RELOAD_OFF_*``,
``reload_policy()`` in every load report with the restart pause estimate,
:func:`restart_reload_estimate_s`); Blackwell keeps in-place and file-level updates.

Updates during a generation (``in_place_reload`` and ``live_update``; the
``updates_during_generation`` capability). Without a pause, SGLang's
``update_weights_from_disk`` takes the model-update writer lock and waits for
every running request, so a release waits behind the whole round. With
``live_update`` the adapter instead:

1. ``POST /pause_generation`` with ``mode: retract``: the scheduler stops and
   moves the running requests back to its waiting queue (their sampled tokens
   are kept, their KV cache is freed);
2. ``POST /update_weights_from_disk`` with ``flush_cache: false`` (a cache
   flush asserts an idle scheduler, and the retracted requests are queued);
   the paused server takes the update at once;
3. ``POST /continue_generation`` (also after a failed update).

The retracted requests then go on from the KV cache SGLang still holds: the
radix (prefix) cache is kept across the update (``live_prefix_cache``, default
on), so prefix KV computed with older weights may be reused, as PipelineRL,
prime-rl and SkyRL do (PipelineRL measured the extra divergence as small). The
generation log-probabilities are recorded as sampled, so TIS corrects against
the real behaviour. ``live_prefix_cache=False`` starts the server with
``--disable-radix-cache`` instead (no stale prefix, but every agentic turn
re-prefills its whole history). Each completion's ``weight_version``
is the version of its first output token and ``final_weight_version`` the
version of its last, from ``meta_info.weight_versions`` (per-token version
spans of the pinned SGLang fork 2fa88018). An SGLang without spans gives
only ``meta_info.weight_version`` (the version when it ended); the adapter
then uses the version served when the request was sent as
``weight_version`` (never newer than the real one: a request still queued
at the update counts at the older version).
Checked against SGLang 0.5.20 (miles-next venv; the miles-ports image pins
0.5.21.dev70) source, and again against sgl-project/sglang main 842b4738 (the
sampler pin ``sampler_pins.SGLANG_SAMPLER_UPSTREAM_COMMIT``, 2026-11-09) and branch
``kimi-k3.1-day-0`` 0e43ada79f: every endpoint and field above, the
``--enable-return-routed-experts`` / ``--enable-return-indexer-topk`` /
``--enable-fp32-lm-head`` / ``--custom-weight-loader`` / ``--attn-dp-size``
flags, the int32 base64 ``routed_experts`` wire, the flat top-logprob fields
and ``UpdateWeightsFromTensorReqInput.load_format`` are the same as on the
sglang-miles-kimi-k31 pin (35708f76). Only ``/get_server_info`` is deprecated
there in favour of ``/server_info`` (``SERVER_INFO_PATHS``). The build the
server runs is recorded from its environment (``BUILD_ENV_KEYS``) and compared
with the sampler pin of the platform (``server_info.sglang_pin_match``).

Sparse updates (``sparse_update`` ``auto``; off by default). A full in-place
reload reads the whole checkpoint on every rank. When the new release was
rebuilt from the served one with change tracking (``release_delta``
``DeltaMirror(track_changes=True)``, ``YETO_RL_MIRROR_TRACK_CHANGES=1``), the
adapter instead sends only the changed elements (:mod:`.sparse_update`):

1. at start, the custom weight loader (:mod:`.sglang_sparse_loader`, passed as
   ``--custom-weight-loader``) probes how ``load_weights`` places every
   checkpoint element on every rank and writes an element map; the served
   release is then reloaded from disk (the probe may leave side effects) and a
   sample of elements is compared with the release (a layout the probe did not
   see, e.g. a post-load reshuffle, disables sparse updates);
2. per update, before any pause: the changed words of the manifest become
   changed tensor elements, and those become one patch per rank (flat
   parameter positions and new bits) plus check samples of changed and
   unchanged elements;
3. ``POST /update_weights_from_tensor`` with ``load_format`` = the loader (one
   command per rank, no weight data over HTTP; under ``live_update`` between
   the same pause and continue as a full reload); each rank writes its patch
   and compares the samples.

It falls back to the full reload, before touching the weights, when the
release has no change manifest, was not rebuilt from the served directory,
changes more than ``sparse_max_ratio`` of the elements, or changes a tensor
the probe could not map; after a failed write or check it reloads in full at
once (``sparse_repair`` in the load record). Supported: bf16 weights without
NEXTN (a draft model would not be updated). Not supported: quantized weights
(FP8/MXFP4 parameters are not the checkpoint's bf16 words) and models whose
``load_weights`` derives other tensors from the loaded ones.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from yeto.rl.engine.sampler_checks import INDEX_FILE
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_STOP,
    Completion,
    SamplerEngineError,
    SamplingParams,
)

from . import k31_release, sglang_load_report, sglang_mxfp4_reload, sparse_update
from .server import ServerConfig, ServerEngine
from .sglang_sparse_loader import LOADER as SPARSE_LOADER
from .sglang_sparse_loader import command_name

TOPOLOGY_TP8 = "tp8"
TOPOLOGY_DP8 = "dp8"
_TOPOLOGY = re.compile(r"(tp|dp)([1-9][0-9]*)")


def sglang_topology(tp: int, *, dp_attention: bool = False) -> str:
    """The topology label of one SGLang instance over ``tp`` cards: ``tp<N>``, or
    ``dp<N>`` with DP attention (``--attn-dp-size N`` on the K3.1 pin and later;
    ``--dp N --enable-dp-attention`` before, see :func:`dp_attention_argv`)."""
    return f"{'dp' if dp_attention else 'tp'}{int(tp)}"


# How DP attention is asked for on the command line (SGLangOptions.dp_attention_flag):
DP_FLAG_AUTO = "auto"  # probe ``launch_server --help`` once; the new form when the probe fails
DP_FLAG_ATTN_DP_SIZE = "attn-dp-size"  # sglang-miles-kimi-k31@35708f76 and later: --attn-dp-size N
DP_FLAG_LEGACY = "enable-dp-attention"  # the 0.5.x pin: --dp N --enable-dp-attention (deprecated upstream)
DP_FLAGS = (DP_FLAG_AUTO, DP_FLAG_ATTN_DP_SIZE, DP_FLAG_LEGACY)
HELP_PROBE_TIMEOUT_S = 600.0  # importing sglang for --help takes a while on a cold image


def dp_attention_argv(flag: str, dp: int) -> list[str]:
    """The DP-attention arguments of one form (``--moe-a2a-backend deepep`` goes
    with both: DP attention without DeepEP duplicates the experts per rank)."""
    if flag == DP_FLAG_LEGACY:
        return ["--dp", str(int(dp)), "--enable-dp-attention", "--moe-a2a-backend", "deepep"]
    if flag == DP_FLAG_ATTN_DP_SIZE:
        return ["--attn-dp-size", str(int(dp)), "--moe-a2a-backend", "deepep"]
    raise SamplerEngineError(f"dp_attention_flag {flag!r} is not resolved (one of {DP_FLAGS[1:]})")


def dp_flag_from_help(help_text: str) -> str:
    """Which form ``launch_server --help`` advertises: ``--attn-dp-size`` wins;
    ``--enable-dp-attention`` without it is the legacy form."""
    if "--attn-dp-size" in help_text:
        return DP_FLAG_ATTN_DP_SIZE
    if "--enable-dp-attention" in help_text:
        return DP_FLAG_LEGACY
    raise SamplerEngineError("sglang launch_server --help names neither --attn-dp-size nor --enable-dp-attention")


def run_help_probe(argv: Sequence[str], timeout_s: float = HELP_PROBE_TIMEOUT_S) -> str:
    """``launch_server --help`` output (stdout + stderr; argparse prints help to stdout)."""
    import subprocess

    result = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s, check=False)  # noqa: S603
    return (result.stdout or "") + (result.stderr or "")


SERVER_INFO_KEYS = ("version", "attn_dp_size", "dp_size", "enable_dp_attention", "enable_return_routed_experts",
                    "enable_return_indexer_topk", "enable_fp32_lm_head", "moe_runner_backend", "quantization",
                    "load_format", "tp_size", "model_path",
                    # the K3.1 inference knobs of upstream main (resolved values; absent on older servers)
                    "attention_backend", "dsa_prefill_backend", "dsa_decode_backend", "dsa_topk_backend",
                    "dcp_size", "attn_cp_size", "enable_prefill_cp", "moe_a2a_backend", "speculative_algorithm")
# The server's SGLang build, as the upstream images stamp it into the environment
# (lmsysorg/sglang nightly/dev: SGLANG_BUILD_COMMIT, SGLANG_IMAGE_TAG); the sampler
# records them next to the server facts and compares the commit with the pin.
BUILD_ENV_KEYS = ("SGLANG_BUILD_COMMIT", "SGLANG_IMAGE_TAG")
# ``/server_info`` first: upstream main logs a deprecation warning for every
# ``/get_server_info`` call; the 0.5.x pin serves both.
SERVER_INFO_PATHS = ("/server_info", "/get_server_info")


def summarize_server_info(info: Any) -> dict[str, Any]:
    """The facts of ``GET /server_info`` (or ``/get_server_info``) the adapter
    records and checks: the SGLang version and the resolved launch options
    (:data:`SERVER_INFO_KEYS`); the 0.5.x pin nests them under ``server_args``."""
    if not isinstance(info, dict):
        return {}
    flat = dict(info)
    nested = info.get("server_args")
    if isinstance(nested, dict):
        flat = {**nested, **{k: v for k, v in info.items() if k != "server_args"}}
    return {key: flat[key] for key in SERVER_INFO_KEYS if key in flat}


def dp_flag_from_server_info(summary: dict[str, Any]) -> str | None:
    """The DP form a running server understands: ``attn_dp_size`` among its
    resolved options -> the new form; ``enable_dp_attention`` alone -> legacy."""
    if "attn_dp_size" in summary:
        return DP_FLAG_ATTN_DP_SIZE
    if "enable_dp_attention" in summary:
        return DP_FLAG_LEGACY
    return None


def is_dp_attention(topology: str) -> bool:
    return topology.startswith("dp")


@dataclass
class SGLangOptions:
    tp: int = 8
    nextn: bool = False
    renames_weight_packed: bool = False  # the image patches the weight_packed -> weight map
    warmup_batch_sizes: tuple[int, ...] = ()
    served_model_name: str = "yeto-policy"
    # the yeto load-report patch writes each rank's loaded expert count here (None: no report)
    load_report_dir: str | None = None
    # update the weights while a generation runs (module doc); needs ServerConfig.in_place_reload
    live_update: bool = False
    # keep the radix prefix cache under live_update (stale prefix KV is reused; module doc)
    live_prefix_cache: bool = True
    # update only the changed weight elements (module doc): "off" or "auto" (a full
    # reload whenever a sparse update is not possible)
    sparse_update: str = "off"
    sparse_max_ratio: float = 0.05  # changed elements / all elements above which the update is full
    sparse_dir: str | None = None  # element maps and patches (default: /dev/shm or the temp directory)
    sparse_probe_device: str = "cpu"  # where the probe's float64 parameters live
    sparse_check_tensors: int = 8  # check samples: changed tensors and as many other tensors
    sparse_check_elements: int = 256  # elements per sampled tensor
    sparse_prepare_timeout_s: float = 3600.0
    # R3: start with ``--enable-return-routed-experts`` and ask every request for
    # ``return_routed_experts``; ``meta_info.routed_experts`` (int32 base64 of
    # ``[prompt + response - 1, L_moe, top_k]``) becomes ``Completion.routed_experts``
    # as uint16 (buffer_payload.ROUTED_EXPERTS_DTYPE)
    return_routed_experts: bool = False
    # indexer top-k replay: the server flag only (``--enable-return-indexer-topk``);
    # the payload is not decoded in this version (about 196 KB per token on K3.1)
    return_indexer_topk: bool = False
    # score_centering: the K most likely tokens and their log-probs at every response
    # position (``top_logprobs_num=K`` on each request; 0: none). The flat base64 form
    # (``return_flat_raw_output_top_logprobs`` + ``_b64``) is asked for; a server
    # without it answers with the nested ``output_top_logprobs`` list, read too
    top_logprobs_num: int = 0
    # ``--enable-fp32-lm-head``: the log-probs come from an fp32 head (recorded in
    # every sample's engine.version_fields for the trainer's check)
    fp32_lm_head: bool = False
    # file-level sparse expert update of a K3.1 partial-experts release
    # (:mod:`.k31_release`): "off", or "auto" (the changed expert files go through
    # /update_weights_from_tensor; anything else is a full reload). Needs in_place_reload
    expert_file_update: str = "off"
    # before the first expert update, write one served expert back with its own bytes
    # and compare teacher-forced log-probs: a change (Marlin repack / aiter shuffle,
    # module doc of k31_release) turns the path off for this server
    expert_update_probe: bool = True
    expert_update_probe_prompt: str = "The capital of France is"
    expert_update_probe_tolerance: float = 1e-3  # max |delta log-prob| the probe accepts
    expert_update_batch_bytes: int = k31_release.DEFAULT_BATCH_BYTES
    # the accelerator platform the server runs on (PLATFORMS). ``rocm`` sets
    # ROCM_ENV_PRESET (extra_env overrides), checks the AMD card count against tp
    # at start (amd-smi / rocm-smi) and is the only platform that may set rocm_aiter
    platform: str = "cuda"
    # ROCm: SGLANG_USE_AITER=1 (the aiter MXFP4 MoE kernels; the preset's
    # AITER_SITUV2_A8W4 only matters with it). None: leave the image's default
    rocm_aiter: bool | None = None
    # CUDA: SGLANG_USE_CUDA_MXFP4_DECODE_MOE=1, upstream #48638's opt-in persistent
    # MXFP4 decode MoE kernel for Kimi-K3.1 (SM100/SM103 only; the server prints
    # "not used" and keeps the flashinfer path elsewhere, so it is safe to ask
    # for on Hopper). None: leave the image's default (off at the pin)
    cuda_mxfp4_decode_moe: bool | None = None
    # how DP attention is asked for (DP_FLAGS): "auto" probes ``launch_server --help``
    # once before the first dp<N> start and remembers what GET /get_server_info says
    dp_attention_flag: str = DP_FLAG_AUTO
    # MoE layer count of the served model, the L_moe of the R3 array (the wire
    # carries ids only); None: read ``num_hidden_layers - first_k_dense_replace``
    # from the release's config.json at start (DeepSeek / Kimi convention)
    moe_layers: int | None = None


SPARSE_MODES = ("off", "auto")
PLATFORMS = ("cuda", "rocm", "ppu", "npu")
# The ROCm environment of the K3.1 upstream recipe (INTERFACES 2.8), set on
# ``platform: rocm`` under ``extra_env`` (which wins): no quick-reduce quantization
# of the all-reduce, 10 GiB of runtime reserve, aiter SITU v2 A8W4 experts, and no
# HSA scratch reclaim (the reclaim stalls the decode kernels).
ROCM_ENV_PRESET = {"ROCM_QUICK_REDUCE_QUANTIZATION": "NONE", "SGLANG_ROCM_RUNTIME_RESERVE_GB": "10",
                   "AITER_SITUV2_A8W4": "1", "HSA_NO_SCRATCH_RECLAIM": "1"}
# upstream main environ.py: EnvBoolWithAlias(False, deprecated_name="SGLANG_K31_DECODE_MOE")
CUDA_MXFP4_DECODE_MOE_ENV = "SGLANG_USE_CUDA_MXFP4_DECODE_MOE"

# In-place weight reload of MXFP4 experts (``update_weights_from_disk`` /
# ``update_weights_from_tensor``) crashes the server wherever the quantization method
# replaces the expert parameters after loading: ROCm aiter shuffles them (MI355X,
# GPU-B probe) and pre-Blackwell CUDA upcasts them to bf16 (``upcast_from_mxfp``, H100 /
# H200, SGL-PERF probe) -- ``'Parameter' object has no attribute 'weight_loader'`` in both.
# Those platforms reload by restart (stop -> start on the release directory) whatever the
# spec's ``in_place_reload`` says; Blackwell (SM100+) keeps the in-place and file-level paths.
IN_PLACE_RELOAD_OFF_ROCM = ("rocm: aiter replaces the MXFP4 expert parameters after loading; "
                            "update_weights_from_disk crashes the server (reload by restart)")
IN_PLACE_RELOAD_OFF_PRE_BLACKWELL = ("cuda sm{major}{minor}: MXFP4 experts are upcast to bf16 before Blackwell; "
                                     "update_weights_from_disk crashes the server (reload by restart)")
IN_PLACE_RELOAD_MIN_CAPABILITY = (10, 0)  # Blackwell: SM100 / SM103
IN_PLACE_RELOAD_PRECISIONS = ("mxfp4",)  # the precisions the rule is about
# Restart reload estimate: a fixed server start (process, distributed init, CUDA graphs:
# 240 s measured on the 4-layer K3 at TP2, H200) plus reading the release from local disk
# (10 GB/s: 1.5 TB in about 2.5 min on NVMe). The release cadence must leave this much
# sampling pause per release on a restart platform.
RESTART_RELOAD_START_OVERHEAD_S = 240.0
RESTART_RELOAD_READ_BYTES_PER_S = 10e9
RELOAD_POLICY_IN_PLACE = "in_place"
RELOAD_POLICY_RESTART = "restart"


def cuda_compute_capability(run: Any) -> tuple[int, int] | None:
    """The lowest ``compute_cap`` ``nvidia-smi`` lists (``(9, 0)`` Hopper, ``(10, 0)``
    Blackwell); None when the tool or its output is unusable (the policy then keeps
    the spec's choice and records the gap)."""
    try:
        text = run(("nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"))
    except Exception:  # noqa: BLE001 - advisory probe
        return None
    caps = []
    for line in str(text).splitlines():
        major, _, minor = line.strip().partition(".")
        if major.isdigit():
            caps.append((int(major), int(minor or 0)))
    return min(caps) if caps else None


def restart_reload_estimate_s(release_bytes: int) -> float:
    """Seconds a restart reload of ``release_bytes`` is expected to pause sampling."""
    return RESTART_RELOAD_START_OVERHEAD_S + max(int(release_bytes), 0) / RESTART_RELOAD_READ_BYTES_PER_S


def release_bytes(model_dir: Path) -> int:
    """Bytes of the safetensors files under ``model_dir`` (0 when none)."""
    try:
        return sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    except OSError:
        return 0


def rocm_card_count(run: Any) -> int:
    """How many AMD GPUs the node's card tool lists (``amd-smi list --json``,
    else ``rocm-smi --showid --json``); ``run(argv) -> stdout`` raises when the
    tool is missing. Raises SamplerEngineError when neither tool answers."""
    errors = []
    for argv, count in ((("amd-smi", "list", "--json"), _amd_smi_list_count),
                        (("rocm-smi", "--showid", "--json"), _rocm_smi_count)):
        try:
            return count(run(argv))
        except Exception as error:  # noqa: BLE001 - try the other tool, report both
            errors.append(f"{argv[0]}: {type(error).__name__}: {str(error)[:120]}")
    raise SamplerEngineError("no AMD card tool answered (" + "; ".join(errors) + ")")


def _amd_smi_list_count(text: str) -> int:
    doc = json.loads(text)
    if isinstance(doc, dict):
        doc = doc.get("gpus") or doc.get("gpu_data") or [doc]
    cards = [c for c in doc if isinstance(c, dict) and "gpu" in c]
    if not cards:
        raise ValueError("no gpu entry")
    return len(cards)


def _rocm_smi_count(text: str) -> int:
    doc = json.loads(text)
    cards = [k for k in doc if re.fullmatch(r"card\d+", str(k))] if isinstance(doc, dict) else []
    if not cards:
        raise ValueError("no card entry")
    return len(cards)


def decode_top_logprobs(meta: dict[str, Any], *, completion_tokens: int, k: int) -> tuple[Any, Any] | None:
    """``(ids int32[T_resp, K], logprobs float32[T_resp, K])`` from a ``/generate``
    ``meta_info``: the flat base64 fields (``output_top_logprobs_idx_flat_b64`` /
    ``_val_flat_b64`` with ``output_top_logprobs_shape`` and ``_null_prefix``),
    the flat list fields, or the nested ``output_top_logprobs`` (one list of
    ``[logprob, token_id, text]`` per position; a position with fewer than K
    candidates is padded with id -1 / -inf). None when ``k`` is 0 or the response
    carries none; a response that covers fewer positions than the output
    tokens is an error (score_centering needs every position)."""
    import base64

    from yeto.rl.engine.buffer_payload import TOP_LOGPROBS_DTYPE, TOP_LOGPROBS_IDS_DTYPE

    if k <= 0:
        return None
    shape = meta.get("output_top_logprobs_shape")
    null_prefix = int(meta.get("output_top_logprobs_null_prefix") or 0)
    ids = vals = None
    if meta.get("output_top_logprobs_idx_flat_b64") is not None:
        ids = np.frombuffer(base64.b64decode(meta["output_top_logprobs_idx_flat_b64"]),
                            dtype=np.dtype(meta.get("output_top_logprobs_idx_flat_b64_dtype") or "int32").newbyteorder("<"))
        vals = np.frombuffer(base64.b64decode(meta["output_top_logprobs_val_flat_b64"]),
                             dtype=np.dtype(meta.get("output_top_logprobs_val_flat_b64_dtype") or "float32").newbyteorder("<"))
    elif meta.get("output_top_logprobs_idx_flat") is not None:
        ids = np.asarray(meta["output_top_logprobs_idx_flat"], dtype=np.int64)
        vals = np.asarray(meta["output_top_logprobs_val_flat"], dtype=np.float64)
    if ids is not None:
        if not shape or len(shape) != 2:
            raise SamplerEngineError("sglang flat top log-probs without output_top_logprobs_shape")
        rows, width = int(shape[0]), int(shape[1])
        if ids.size != rows * width or vals.size != rows * width:
            raise SamplerEngineError(f"sglang flat top log-probs: {ids.size} ids / {vals.size} values for shape {shape}")
        ids, vals = ids.reshape(rows, width), vals.reshape(rows, width)
        if null_prefix:  # positions before the first covered one: no candidates
            ids = np.concatenate([np.full((null_prefix, width), -1, dtype=ids.dtype), ids])
            vals = np.concatenate([np.full((null_prefix, width), -np.inf, dtype=vals.dtype), vals])
    else:
        nested = meta.get("output_top_logprobs")
        if nested is None:
            return None
        width = max([k] + [len(row or ()) for row in nested])
        ids = np.full((len(nested), width), -1, dtype=np.int64)
        vals = np.full((len(nested), width), -np.inf, dtype=np.float64)
        for t, row in enumerate(nested):
            for c, entry in enumerate(row or ()):
                vals[t, c], ids[t, c] = float(entry[0]), int(entry[1])
    if ids.shape[0] != int(completion_tokens):
        raise SamplerEngineError(f"sglang top log-probs cover {ids.shape[0]} positions for {completion_tokens} "
                                 "output tokens")
    if ids.shape[1] < k:
        raise SamplerEngineError(f"sglang returned {ids.shape[1]} top candidates per token, {k} asked")
    return (np.ascontiguousarray(ids[:, :k]).astype(TOP_LOGPROBS_IDS_DTYPE),
            np.ascontiguousarray(vals[:, :k]).astype(TOP_LOGPROBS_DTYPE))


def moe_layer_count(model_dir: Path) -> int | None:
    """``num_hidden_layers - first_k_dense_replace`` of ``model_dir/config.json``
    (the layers with routed experts in a DeepSeek-style MoE; ``text_config`` is
    read when the config nests it), None when the config does not say."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    layers = text.get("num_hidden_layers")
    if layers is None or not (text.get("n_routed_experts") or text.get("num_experts")):
        return None
    dense = text.get("first_k_dense_replace") or 0
    return max(int(layers) - int(dense), 0) or None


def dense_layer_count(model_dir: Path) -> int:
    """``first_k_dense_replace`` of ``model_dir/config.json`` (``text_config``
    when nested): the leading dense layers the server's R3 array carries as
    all-zero rows (:func:`decode_routed_experts`). 0 when the config does not say."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    return max(int(text.get("first_k_dense_replace") or 0), 0)


def routed_top_k(model_dir: Path) -> int | None:
    """``num_experts_per_tok`` of ``model_dir/config.json`` (``text_config`` when
    nested): the ids per layer and position of the R3 array, which fixes its
    layout (:func:`decode_routed_experts`). None when the config does not say."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    top_k = text.get("num_experts_per_tok") or text.get("num_experts_per_token")  # Kimi-K3.1 spells it out
    return int(top_k) if top_k else None


ROUTED_EXPERTS_WIRE_DTYPE = "<i4"  # meta_info.routed_experts: int32 little-endian, no shape


def decode_routed_experts(meta: dict[str, Any], *, prompt_tokens: int, completion_tokens: int,
                          moe_layers: int | None = None, dense_layers: int = 0,
                          top_k: int | None = None) -> Any:
    """``meta_info.routed_experts`` as ``uint16[T, L_moe, top_k]`` with
    ``T = prompt_tokens + completion_tokens - 1`` (the Miles decoding,
    ``get_routed_experts_from_response``) and ``L_moe`` = ``moe_layers`` (else
    ``meta_info.routed_experts_num_layers`` when the server sends it). None
    when the response has no field.

    The wire carries one row per *model* layer (Miles reshapes with
    ``args.num_layers``): on sglang-miles-kimi-k31@35708f76 the Kimi-K3-4layer
    server returns ``[T, 4, 16]`` with the dense layer 0 all zeros (measured on
    NVIDIA, SMP). ``dense_layers`` (``first_k_dense_replace``) leading rows are
    dropped when the payload splits into ``L_moe + dense_layers`` layers; a
    payload that already has ``L_moe`` layers is taken as is.

    A payload whose size does not divide into ``T`` rows and either layer count,
    or an id above uint16, is an error: the MoE runner then bypassed the top-k
    capture (e.g. ``flashinfer_trtllm``) or the model is not the one the island
    expects. ``top_k`` (``num_experts_per_tok``) settles the layout: a row must
    then hold ``L_moe`` or ``L_moe + dense_layers`` layers of exactly ``top_k``
    ids, and the dense head of the latter must carry no routing (0 or -1)."""
    import base64

    from yeto.rl.engine.buffer_payload import routed_experts_uint16

    encoded = meta.get("routed_experts")
    if encoded is None:
        return None
    rows = int(prompt_tokens) + int(completion_tokens) - 1
    flat = np.frombuffer(base64.b64decode(encoded), dtype=ROUTED_EXPERTS_WIRE_DTYPE)
    if rows <= 0 or flat.size == 0 or flat.size % rows:
        raise SamplerEngineError(f"sglang routed_experts has {flat.size} ids for {rows} positions "
                                 f"({prompt_tokens} prompt + {completion_tokens} output tokens - 1)")
    if int(flat.min()) < -1 or int(flat.max()) > np.iinfo(np.uint16).max - 1:
        raise SamplerEngineError(f"sglang routed_experts holds id {int(flat.min())}..{int(flat.max())}, "
                                 "outside uint16 (-1 is the pad)")
    if int(flat.max()) == 0 and flat.size > 1:
        raise SamplerEngineError("sglang routed_experts are all 0: the MoE runner backend bypasses the "
                                 "top-k capture (e.g. flashinfer_trtllm); use a backend that routes "
                                 "through TopK (Marlin MXFP4 on K3.1)")
    layers = moe_layers or meta.get("routed_experts_num_layers")
    per_row = flat.size // rows
    if not layers:
        raise SamplerEngineError("sglang routed_experts: the MoE layer count is unknown (set "
                                 "SGLangOptions.moe_layers or serve a release whose config.json has "
                                 "num_hidden_layers and first_k_dense_replace)")
    layers = int(layers)
    dense = max(int(dense_layers or 0), 0)
    if top_k:
        k = int(top_k)
        if per_row == layers * k:
            return routed_experts_uint16(flat.reshape(rows, layers, k), where="sglang routed_experts")
        if not (dense and per_row == (layers + dense) * k):
            raise SamplerEngineError(f"sglang routed_experts: {per_row} ids per position are neither {layers} MoE "
                                     f"layers nor {layers + dense} model layers of top-{k}")
        full = flat.reshape(rows, layers + dense, k)
        head = full[:, :dense]
        if ((head != 0) & (head != -1)).any():
            raise SamplerEngineError(f"sglang routed_experts: the {dense} leading dense layer(s) carry expert ids "
                                     f"(max {int(head.max())}); the model layout is not the one expected")
        return routed_experts_uint16(np.ascontiguousarray(full[:, dense:]), where="sglang routed_experts")
    # Both layouts may divide per_row (3 x 8 = 4 x 6): the model layout is taken only when
    # its dense head rows are all zero, which an L_moe layout of real routings never is.
    if dense and per_row % (layers + dense) == 0:
        full = flat.reshape(rows, layers + dense, per_row // (layers + dense))
        head = full[:, :dense]
        if int(head.max()) == 0:
            return routed_experts_uint16(np.ascontiguousarray(full[:, dense:]), where="sglang routed_experts")
        if per_row % layers:
            raise SamplerEngineError(f"sglang routed_experts: the {dense} leading dense layer(s) carry expert ids "
                                     f"(max {int(head.max())}); the model layout is not the one expected")
    if per_row % layers == 0:
        return routed_experts_uint16(flat.reshape(rows, layers, per_row // layers), where="sglang routed_experts")
    raise SamplerEngineError(f"sglang routed_experts: {per_row} ids per position do not split into "
                             f"{layers} MoE layers" + (f" (nor {layers + dense} model layers)" if dense else ""))


EXPERT_FILES_PER_CALL = 8  # changed safetensors files per rank-local expert update call


def _placeholder() -> Any:
    """The one-element tensor a command-only ``serialized_named_tensors`` entry carries."""
    try:
        import torch
    except ImportError:  # CPU tests with an injected serializer
        return 0
    return torch.zeros(1)


def _sparse_payload(command: dict[str, Any]) -> str:
    """One rank's ``serialized_named_tensors`` entry: the command in the tensor
    name and a one-element placeholder tensor (plain pickle, which SGLang's
    restricted unpickler accepts, base64 for JSON)."""
    import base64
    import pickle

    import torch

    return base64.b64encode(pickle.dumps([(command_name(command), torch.zeros(1))])).decode("ascii")


class _ProbeReloadFailed(SamplerEngineError):
    """The reload of the served release after the sparse probe failed: the
    weights may still hold probe side effects, so the start fails."""


def _finish(meta: dict[str, Any]) -> str:
    reason = meta.get("finish_reason")
    kind = reason.get("type") if isinstance(reason, dict) else reason
    if kind == "length":
        return FINISH_LENGTH
    if kind == "stop":
        return FINISH_STOP
    raise SamplerEngineError(f"sglang finish_reason {reason!r}: request aborted or unknown")


def _completion_versions(meta: dict[str, Any], sent: int, started: int) -> dict[str, int]:
    """``weight_version`` and ``final_weight_version`` of one completion (module doc).

    With ``meta_info.weight_versions`` (spans ``{version, start, end}`` of the
    output tokens; the pinned SGLang fork) they are the first and last span's
    version. Without spans: the version the request was sent at, and
    ``meta_info.weight_version``. ``default`` (no in-place update yet) is the
    version the server was started on. The final version may be below the
    first after a new lineage (a lower version with a higher seq)."""

    def number(value: Any, fallback: int) -> int:
        if value == "default":
            return started
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    spans = meta.get("weight_versions")
    if isinstance(spans, list) and spans:
        first = number(spans[0].get("version"), sent)
        return {"weight_version": first,
                "final_weight_version": number(spans[-1].get("version"), first)}
    return {"weight_version": sent, "final_weight_version": number(meta.get("weight_version"), sent)}


class SGLangServerEngine(ServerEngine):
    ENGINE = "sglang"
    HANG_HINT = "Check the PCCL/ACCL-P collective setup of the PPU image."
    EXIT_CODES_AFTER_STOP = frozenset({134})  # pcclCommAbort at TP8 shutdown (item 13)
    REFUSED_FORMATS = frozenset({"dense"})  # item 5

    def __init__(self, config: ServerConfig, options: SGLangOptions | None = None, *,
                 help_probe: Any = run_help_probe, run: Any = None,
                 read_tensors: k31_release.ReadTensors = k31_release.read_tensors,
                 serialize: k31_release.Serialize = k31_release.serialize_named_tensors, **kw: Any) -> None:
        """``help_probe(argv) -> str``: runs ``launch_server --help`` for the DP-flag
        probe; ``run(argv) -> str``: runs the card tool of the platform check
        (default: ``yeto.hw.card_telemetry.run_command``); ``read_tensors`` /
        ``serialize``: the expert-file update's tensor IO (:mod:`.k31_release`)."""
        super().__init__(config, **kw)
        self.options = options or SGLangOptions()
        self._help_probe = help_probe
        self._run = run
        self._read_tensors = read_tensors
        self._serialize = serialize
        self._started_version = 0
        if self.options.expert_file_update not in SPARSE_MODES:
            raise SamplerEngineError(f"sglang expert_file_update must be one of {SPARSE_MODES}, "
                                     f"got {self.options.expert_file_update!r}")
        if self.options.expert_file_update != "off" and not config.in_place_reload:
            raise SamplerEngineError("sglang expert_file_update sends tensors into the running server; "
                                     "set in_place_reload")
        # why file-level expert updates are off for this server (None: they are tried)
        self.expert_update_off: str | None = None if self.options.expert_file_update == "auto" else "off"
        # why in-place reload is off on this platform (None: the spec's in_place_reload holds);
        # ROCm is known at construction, pre-Blackwell CUDA at preflight (nvidia-smi)
        self.in_place_reload_off: str | None = None
        self.compute_capability: tuple[int, int] | None = None
        self._reload_policy_probed = False
        if config.in_place_reload and self.options.platform == "rocm" and config.precision in IN_PLACE_RELOAD_PRECISIONS:
            self._turn_in_place_reload_off(IN_PLACE_RELOAD_OFF_ROCM)
        self.expert_update_prepare: dict[str, Any] | None = None  # the probe's record
        if self.options.platform not in PLATFORMS:
            raise SamplerEngineError(f"sglang platform must be one of {PLATFORMS}, got {self.options.platform!r}")
        if self.options.rocm_aiter is not None and self.options.platform != "rocm":
            raise SamplerEngineError("sglang rocm_aiter is a ROCm option; set platform: rocm")
        if self.options.dp_attention_flag not in DP_FLAGS:
            raise SamplerEngineError(f"sglang dp_attention_flag must be one of {DP_FLAGS}, "
                                     f"got {self.options.dp_attention_flag!r}")
        # the resolved DP form (None: not probed yet) and the last server's facts
        self.dp_flag: str | None = None if self.options.dp_attention_flag == DP_FLAG_AUTO else self.options.dp_attention_flag
        self.server_info: dict[str, Any] = {}
        # The topology is the source label of every sample: it must name the real
        # layout, tp<N> or dp<N> with N = the instance's card count (options.tp).
        match = _TOPOLOGY.fullmatch(config.topology)
        if match is None or int(match.group(2)) != int(self.options.tp):
            raise SamplerEngineError(
                f"sglang topology {config.topology!r} does not match TP {self.options.tp}: expected "
                f"{sglang_topology(self.options.tp)!r} or {sglang_topology(self.options.tp, dp_attention=True)!r} "
                "(PD disaggregation is not used)")
        if self.options.live_update and not config.in_place_reload:
            raise SamplerEngineError("sglang live_update loads weights in place; set in_place_reload")
        if is_dp_attention(config.topology) and self.options.nextn:
            raise SamplerEngineError("sglang DP attention + DeepEP cannot run NEXTN (A-INF item 8); "
                                     "turn one of them off")
        if self.options.sparse_update not in SPARSE_MODES:
            raise SamplerEngineError(f"sglang sparse_update must be one of {SPARSE_MODES}, "
                                     f"got {self.options.sparse_update!r}")
        # why sparse updates are off for this engine (None: they are tried)
        self.sparse_off: str | None = self._sparse_unsupported()
        self._served_dir: Path | None = None
        self._sparse_maps: list[dict[str, Any]] | None = None
        self._sparse_shapes: dict[str, list[int]] | None = None
        self.sparse_prepare: dict[str, Any] | None = None  # the last probe's record
        self._weight_sessions: bool | None = None  # /begin_weight_update exists (None: not asked yet)
        self._moe_layers: int | None = self.options.moe_layers  # L_moe of the R3 array (preflight)
        self._dense_layers: int = 0  # leading dense layers the wire carries as zero rows (preflight)
        self._routed_top_k: int | None = None  # num_experts_per_tok: the R3 row layout (preflight)

    def _sparse_unsupported(self) -> str | None:
        c, o = self.config, self.options
        if o.sparse_update == "off":
            return "off"
        if not c.in_place_reload:
            return "needs in_place_reload"
        if c.precision != "bf16" or c.quant_scheme != "bf16":
            return f"weights are {c.precision}/{c.quant_scheme}, not bf16"
        if o.nextn:
            return "NEXTN draft model"
        return None

    @property
    def updates_during_generation(self) -> bool:
        """The port capability (``sampler_engine.updates_during_generation``)."""
        return bool(self.options.live_update)

    @property
    def returns_routed_experts(self) -> bool:
        """The port capability (``sampler_engine.returns_routed_experts``): R3 on."""
        return bool(self.options.return_routed_experts)

    @property
    def top_logprobs_num(self) -> int:
        """The port capability (``sampler_engine.top_logprobs_num``)."""
        return max(0, int(self.options.top_logprobs_num))

    @property
    def fp32_lm_head(self) -> bool:
        """``engine.version_fields.fp32_lm_head`` of every sample (``sampler_engine.version_fields``)."""
        return bool(self.options.fp32_lm_head)

    def argv(self, model_dir: Path) -> list[str]:
        c, o = self.config, self.options
        argv = [c.python, "-m", "sglang.launch_server", "--model-path", str(model_dir),
                "--host", c.host, "--port", str(c.port), "--tp", str(o.tp),
                "--context-length", str(c.max_model_len),
                "--served-model-name", o.served_model_name, "--trust-remote-code"]
        if "--tokenizer-path" not in c.extra_args and Path(model_dir) != Path(c.base_dir):
            # a release is served with the base checkpoint's tokenizer (VllmServerEngine.base_tokenizer_argv)
            argv += ["--tokenizer-path", str(c.base_dir)]
        if is_dp_attention(c.topology):
            argv += dp_attention_argv(self._resolve_dp_flag(), o.tp)
        if o.nextn:
            argv += ["--speculative-algorithm", "NEXTN"]
        if o.live_update and not o.live_prefix_cache and "--disable-radix-cache" not in c.extra_args:
            argv += ["--disable-radix-cache"]  # no prefix KV of older weights after an update
        loaders = [SPARSE_LOADER] if self.sparse_off is None else []
        if self._expert_files_rank_local:
            loaders.append(sglang_mxfp4_reload.FILE_LOADER)
        if loaders:
            argv += ["--custom-weight-loader", *loaders]
        if o.return_routed_experts:
            argv += ["--enable-return-routed-experts"]
        if o.return_indexer_topk:
            argv += ["--enable-return-indexer-topk"]
        if o.fp32_lm_head:
            argv += ["--enable-fp32-lm-head"]
        return argv + list(c.extra_args)

    def _resolve_dp_flag(self) -> str:
        """The DP form to launch with (module constants). ``auto``: what a
        running server reported last, else ``launch_server --help``; a failed
        probe takes the new form (the K3.1 pin) and says so in ``server_info``."""
        if self.dp_flag is None:
            try:
                self.dp_flag = dp_flag_from_help(self._help_probe([self.config.python, "-m", "sglang.launch_server",
                                                                   "--help"]))
            except Exception as error:  # noqa: BLE001 - the probe is advisory
                self.dp_flag = DP_FLAG_ATTN_DP_SIZE
                self.server_info["dp_flag_probe_error"] = f"{type(error).__name__}: {str(error)[:200]}"
        return self.dp_flag

    def _read_server_info(self) -> dict[str, Any]:
        """``GET /server_info`` (``/get_server_info`` on a server without it;
        :data:`SERVER_INFO_PATHS`) summarized; an unreachable endpoint gives an
        empty summary."""
        for path in SERVER_INFO_PATHS:
            try:
                summary = summarize_server_info(self.call("GET", path))
            except SamplerEngineError:
                continue
            if summary:
                return summary
        return {}

    def _build_facts(self) -> dict[str, Any]:
        """The SGLang build the server runs (:data:`BUILD_ENV_KEYS` of the server's
        environment, which the adapter's own environment seeds) and whether it is
        the sampler pin of this platform (``sglang_pin_match``: None when the
        environment names no build or the platform has no pin). A mismatch is
        recorded, not refused: a node may run a newer image on purpose."""
        env = {**os.environ, **self.env()}  # the server inherits the process environment plus env()
        facts: dict[str, Any] = {k.lower(): env[k] for k in BUILD_ENV_KEYS if env.get(k)}
        commit = facts.get("sglang_build_commit")
        if commit:
            from yeto.rl.sampler_pins import sampler_sglang_commit  # noqa: PLC0415

            pin = sampler_sglang_commit(self.options.platform)
            facts["sglang_pin_match"] = None if pin is None else (str(commit).lower() == pin.lower())
            if pin is not None:
                facts["sglang_pin_commit"] = pin
        return facts

    def _check_server_features(self, summary: dict[str, Any]) -> None:
        """A feature the island relies on that the server resolved off is a start
        failure: with it silently off the samples would lack the field."""
        wants = {"enable_return_routed_experts": self.options.return_routed_experts,
                 "enable_return_indexer_topk": self.options.return_indexer_topk,
                 "enable_fp32_lm_head": self.options.fp32_lm_head}
        off = [key for key, want in wants.items() if want and key in summary and not summary[key]]
        if off:
            raise SamplerEngineError(f"sglang resolved {off} off although the island asked for them "
                                     f"(server_info: {summary})")
        if is_dp_attention(self.config.topology) and "attn_dp_size" in summary \
                and int(summary["attn_dp_size"] or 1) != int(self.options.tp):
            raise SamplerEngineError(f"sglang attn_dp_size is {summary['attn_dp_size']}, the topology "
                                     f"{self.config.topology} needs {self.options.tp}")

    def env(self) -> dict[str, str]:
        env = super().env()
        if self.options.platform == "rocm":  # the preset first: extra_env (super) wins
            env = {**ROCM_ENV_PRESET, **({} if self.options.rocm_aiter is None else
                                         {"SGLANG_USE_AITER": "1" if self.options.rocm_aiter else "0"}), **env}
        if self.options.platform == "cuda" and self.options.cuda_mxfp4_decode_moe is not None \
                and CUDA_MXFP4_DECODE_MOE_ENV not in self.config.extra_env:  # extra_env wins
            env[CUDA_MXFP4_DECODE_MOE_ENV] = "1" if self.options.cuda_mxfp4_decode_moe else "0"
        if self.config.precision == "mxfp4" and self.options.platform == "cuda" and self.config.in_place_reload \
                and sglang_mxfp4_reload.ENV not in self.config.extra_env:  # extra_env wins
            env[sglang_mxfp4_reload.ENV] = "1"  # reloadable MXFP4 experts on SM100 (sglang_mxfp4_reload)
        if self.config.precision == "mxfp4" and self.options.platform != "rocm":
            env["MX_PATCH"] = "1"  # item 7 (the PPU W4A16 path; ROCm serves MXFP4 through aiter / Marlin)
        if self.options.renames_weight_packed:
            env[sglang_load_report.RENAME_ENV] = "1"
        if self.options.load_report_dir:
            env[sglang_load_report.REPORT_DIR_ENV] = str(self.options.load_report_dir)
        return env

    def start(self, release_dir: Path | None, version: int):
        self._started_version = int(version)  # what ``default`` means in this server's versions
        if self.options.load_report_dir:  # reports of an earlier server must not count for this one
            sglang_load_report.clear_reports(self.options.load_report_dir)
        self._served_dir = Path(release_dir) if release_dir is not None else Path(self.config.base_dir)
        self._sparse_maps = None  # a new server: probe again
        if self.options.expert_file_update == "auto":
            self.expert_update_off = None  # a new server: probe again
        self._weight_sessions = None  # whether it has weight-update sessions: asked at the first call
        return super().start(release_dir, version)

    def _check_rocm_cards(self) -> None:
        """The node must show at least ``tp`` AMD GPUs before the server starts
        (a missing or half-visible card hangs the RCCL setup without an error)."""
        if self._run is None:
            from yeto.hw.card_telemetry import run_command  # noqa: PLC0415

            self._run = run_command
        count = rocm_card_count(self._run)
        if count < int(self.options.tp):
            raise SamplerEngineError(f"sglang on ROCm: the card tool lists {count} GPU(s), TP {self.options.tp} "
                                     "needs that many")

    def _turn_in_place_reload_off(self, reason: str) -> None:
        """Reload by restart from now on (module constants): file-level expert updates and
        live updates need the in-place path, so they go off / are refused with it."""
        if self.options.live_update:
            raise SamplerEngineError(f"sglang live_update needs in-place reload, which this platform cannot do: {reason}")
        self.in_place_reload_off = reason
        if self.options.expert_file_update != "off":
            self.expert_update_off = reason

    def _probe_reload_policy(self) -> None:
        """Once per engine, before the first start: on CUDA with an MXFP4 release and
        ``in_place_reload`` asked for, read the compute capability; below Blackwell the
        reload is by restart (module constants). An unreadable capability keeps the spec's
        choice and is recorded in ``reload_policy()``."""
        if self._reload_policy_probed:
            return
        self._reload_policy_probed = True
        if self.in_place_reload_off is not None or not self.config.in_place_reload:
            return
        if self.options.platform != "cuda" or self.config.precision not in IN_PLACE_RELOAD_PRECISIONS:
            return
        if self._run is None:
            from yeto.hw.card_telemetry import run_command  # noqa: PLC0415

            self._run = run_command
        self.compute_capability = cuda_compute_capability(self._run)
        if self.compute_capability is not None and self.compute_capability < IN_PLACE_RELOAD_MIN_CAPABILITY:
            major, minor = self.compute_capability
            self._turn_in_place_reload_off(IN_PLACE_RELOAD_OFF_PRE_BLACKWELL.format(major=major, minor=minor))

    def reload_in_place_effective(self) -> bool:
        """Whether ``load`` updates the running server (the spec's ``in_place_reload``
        and the platform rule agree)."""
        return bool(self.config.in_place_reload) and self.in_place_reload_off is None

    def reload_policy(self) -> dict[str, Any]:
        """The reload mode of this engine and, for a restart, the pause one release
        costs (:func:`restart_reload_estimate_s` of the served directory's size)."""
        served = self._served_dir if self._served_dir is not None else Path(self.config.base_dir)
        size = release_bytes(served)
        record: dict[str, Any] = {
            "mode": RELOAD_POLICY_IN_PLACE if self.reload_in_place_effective() else RELOAD_POLICY_RESTART,
            "configured_in_place": bool(self.config.in_place_reload), "off_reason": self.in_place_reload_off,
            "platform": self.options.platform, "precision": self.config.precision,
            "compute_capability": None if self.compute_capability is None else list(self.compute_capability),
            "release_bytes": size}
        if record["mode"] == RELOAD_POLICY_RESTART:
            record["restart_estimate_s"] = round(restart_reload_estimate_s(size), 1)
        return record

    def load(self, release_dir: Path, version: int):
        if self.config.in_place_reload and self.in_place_reload_off is not None:
            return self.start(release_dir, version)  # the platform rule: restart on the release
        return super().load(release_dir, version)

    def loaded_expert_matrices(self) -> int | None:
        if not self.options.load_report_dir:
            return None
        report = sglang_load_report.read_reports(self.options.load_report_dir)
        if report is None:
            return None
        if report["ranks"] < int(self.options.tp):
            return 0  # a rank that did not report loaded nothing we can vouch for
        return int(report["expert_matrices"])

    def preflight(self, model_dir: Path) -> None:
        super().preflight(model_dir)
        if self.options.platform == "rocm":
            self._check_rocm_cards()
        self._probe_reload_policy()
        if self.options.return_routed_experts:
            self._dense_layers = dense_layer_count(model_dir)
            self._routed_top_k = routed_top_k(model_dir)
        if self.options.return_routed_experts and self.options.moe_layers is None:
            self._moe_layers = moe_layer_count(model_dir)
            if self._moe_layers is None:
                raise SamplerEngineError(f"{model_dir}: config.json gives no MoE layer count; R3 needs "
                                         "SGLangOptions.moe_layers to shape the routed experts")
        index = Path(model_dir) / INDEX_FILE
        if index.is_file() and not self.options.renames_weight_packed and not k31_release.native_packed_names(model_dir):
            names = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
            packed = [n for n in names if n.endswith(".weight_packed")]
            if packed:
                raise SamplerEngineError(
                    f"{model_dir}: {len(packed)} tensors are named *.weight_packed (e.g. "
                    f"{packed[0]}); SGLang skips them without an error (A-INF item 1). Use an "
                    "image with the rename patch and set renames_weight_packed")

    def after_ready(self) -> dict[str, Any]:
        shown = ("MX_PATCH", "SGLANG_USE_AITER", CUDA_MXFP4_DECODE_MOE_ENV, sglang_mxfp4_reload.ENV, *ROCM_ENV_PRESET)
        detail: dict[str, Any] = {"env": {k: v for k, v in self.env().items() if k in shown}}
        detail["reload_policy"] = self.reload_policy()
        summary = self._read_server_info()
        build = self._build_facts()
        if summary or build:
            learned = dp_flag_from_server_info(summary)
            if learned is not None and self.options.dp_attention_flag == DP_FLAG_AUTO:
                self.dp_flag = learned  # the next start launches with what this server understands
            self.server_info = {**{k: v for k, v in self.server_info.items() if k.endswith("_error")},
                                **summary, **build}
            if summary:
                self._check_server_features(summary)
            detail["server_info"] = dict(self.server_info)
        if self.options.warmup_batch_sizes:
            began = self._clock()
            for size in self.options.warmup_batch_sizes:
                self._generate_raw(["warm-up"] * int(size),
                                   SamplingParams(max_tokens=8, temperature=0.0))
            detail["warmup_batch_sizes"] = list(self.options.warmup_batch_sizes)
            detail["warmup_s"] = round(self._clock() - began, 3)
        if self.sparse_off is None:
            detail["sparse_prepare"] = self._sparse_prepare()
        if self.expert_update_off is None:
            detail["expert_update_prepare"] = self._expert_update_prepare()
        return detail

    def reload_in_place(self, model_dir: Path, version: int) -> dict[str, Any] | None:
        if self.in_place_reload_off is not None:
            raise SamplerEngineError(f"sglang in-place reload is off on this platform: {self.in_place_reload_off}")
        sparse: dict[str, Any] = {}
        if self.expert_update_off is None:
            try:
                result = self._expert_file_update(Path(model_dir), int(version))
            except k31_release.FileSparseUnavailable as error:
                sparse = {"expert_update_fallback": error.reason, "expert_update_fallback_detail": str(error)[:300]}
            else:
                self._served_dir = Path(model_dir)
                return result
        elif self.options.expert_file_update != "off":
            sparse = {"expert_update_fallback": self.expert_update_off}
        if self.sparse_off is None and self._sparse_maps is not None:
            try:
                result = self._sparse_update(Path(model_dir), int(version))
            except sparse_update.SparseUnavailable as error:
                sparse = {"sparse_fallback": error.reason, "sparse_fallback_detail": str(error)[:300]}
            else:  # written sparsely, or reloaded in full after a failed write or check
                self._served_dir = Path(model_dir)
                return result
        elif self.options.sparse_update != "off":
            sparse = {"sparse_fallback": self.sparse_off or "not_prepared"}
        detail = self._reload_full(model_dir, version)
        self._served_dir = Path(model_dir)
        merged = {**(detail or {}), **sparse}
        return merged or None

    def _reload_full(self, model_dir: Path, version: int) -> dict[str, Any] | None:
        request = {"model_path": str(model_dir), "weight_version": str(int(version))}
        if not self.options.live_update:
            self._update_from_disk(request)
            return None
        began = self._clock()
        self.call("POST", "/pause_generation", {"mode": "retract"})
        paused = self._clock()
        try:
            body = self._update_from_disk({**request, "flush_cache": False})
        finally:
            updated = self._clock()
            self.call("POST", "/continue_generation", {})
        return {"live": True, "pause_call_s": round(paused - began, 3),
                "update_s": round(updated - paused, 3), "paused_s": round(self._clock() - paused, 3),
                "num_paused_requests": body.get("num_paused_requests") if isinstance(body, dict) else None}

    # ------------------------------------------------------------- expert file updates
    @property
    def _expert_files_rank_local(self) -> bool:
        """Expert updates read the changed files on every rank (``sglang_mxfp4_reload.load_files``)
        instead of pickling their tensors into the request: on with the MXFP4 in-place hook."""
        return self.options.expert_file_update != "off" and self.env().get(sglang_mxfp4_reload.ENV) == "1"

    def _send_files(self, plan: k31_release.ExpertUpdatePlan, *, version: int | None) -> dict[str, Any]:
        """The plan's files by name through the rank-local file loader (one call per
        ``EXPERT_FILES_PER_CALL`` files; ``weight_version`` on the last call)."""
        tp, step = int(self.options.tp), EXPERT_FILES_PER_CALL
        groups = [list(plan.files[i:i + step]) for i in range(0, len(plan.files), step)]
        session = self._begin_weight_update()
        try:
            for i, files in enumerate(groups):
                name = sglang_mxfp4_reload.files_command_name(str(plan.release_dir), files)
                body: dict[str, Any] = {"serialized_named_tensors": [self._serialize([(name, _placeholder())])] * tp,
                                        "load_format": sglang_mxfp4_reload.FILE_LOADER, "flush_cache": False}
                if version is not None and i == len(groups) - 1:
                    body["weight_version"] = str(int(version))
                reply = self.call("POST", "/update_weights_from_tensor", body)
                if isinstance(reply, dict) and reply.get("success") is False:
                    raise SamplerEngineError(f"sglang update_weights_from_tensor: {reply.get('message')}")
        finally:
            if session:
                self.call("POST", "/end_weight_update", {})
        return {"batches": len(groups), "rank_local_files": True, "tensors": plan.tensor_count,
                "experts": plan.expert_count, "files": len(plan.files)}

    def _send_tensors(self, plan: k31_release.ExpertUpdatePlan, *, version: int | None) -> dict[str, Any]:
        """The plan's tensors through ``/update_weights_from_tensor`` in batches
        (the model's own ``load_weights``; ``weight_version`` on the last batch)."""
        if self._expert_files_rank_local:
            return self._send_files(plan, version=version)
        tp = int(self.options.tp)
        sent_batches = sent_bytes = 0
        chunks = list(k31_release.batches(plan, self._read_tensors, max_bytes=self.options.expert_update_batch_bytes))
        session = self._begin_weight_update()
        try:
            for i, batch in enumerate(chunks):
                payload = self._serialize(batch)
                body: dict[str, Any] = {"serialized_named_tensors": [payload] * tp, "flush_cache": False}
                if version is not None and i == len(chunks) - 1:
                    body["weight_version"] = str(int(version))
                reply = self.call("POST", "/update_weights_from_tensor", body)
                if isinstance(reply, dict) and reply.get("success") is False:
                    raise SamplerEngineError(f"sglang update_weights_from_tensor: {reply.get('message')}")
                sent_batches += 1
                sent_bytes += sum(k31_release.tensor_bytes(t) for _, t in batch)
        finally:
            if session:
                self.call("POST", "/end_weight_update", {})
        return {"batches": sent_batches, "tensor_bytes": sent_bytes, "tensors": plan.tensor_count,
                "experts": plan.expert_count, "files": len(plan.files)}

    def _expert_update_prepare(self) -> dict[str, Any]:
        """The shuffle probe (:mod:`.k31_release` module doc): one served expert
        written back from disk must leave the teacher-forced log-probs of a fixed
        prompt unchanged. On any failure the path stays off for this server and
        the served release is reloaded from disk."""
        assert self._served_dir is not None
        began = self._clock()
        record: dict[str, Any] = {"probe": bool(self.options.expert_update_probe)}
        if not self.options.expert_update_probe:
            record.update(enabled=True, prepare_s=0.0)
            self.expert_update_prepare = record
            return record
        try:
            plan = k31_release.shuffle_probe_plan(self._served_dir)
            prompt = self.options.expert_update_probe_prompt
            before = self.score(prompt, " Paris.")
            try:
                record.update(self._send_tensors(plan, version=None))
                after = self.score(prompt, " Paris.")
            except Exception as error:  # the weights may hold a half-written expert: reload them
                self._repair_from_disk()
                raise SamplerEngineError(f"probe write failed: {error}") from error
            delta = max((abs(a - b) for a, b in zip(after, before)), default=0.0)
            record["max_delta_logprob"] = round(float(delta), 6)
            if len(after) != len(before) or delta > float(self.options.expert_update_probe_tolerance):
                self._repair_from_disk()
                raise SamplerEngineError(
                    f"writing a served expert back changed the log-probs by {delta:.3g}: the engine transforms "
                    "expert weights after loading (Marlin repack / aiter shuffle), so a tensor update would "
                    "corrupt them")
        except (SamplerEngineError, k31_release.FileSparseUnavailable, OSError, ValueError, KeyError) as error:
            self.expert_update_off = getattr(error, "reason", None) or "probe_failed"
            record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        record["prepare_s"] = round(self._clock() - began, 3)
        record["enabled"] = self.expert_update_off is None
        self.expert_update_prepare = record
        return record

    def _repair_from_disk(self) -> None:
        assert self._served_dir is not None
        try:
            self._update_from_disk({"model_path": str(self._served_dir), "weight_version": str(self._started_version)})
        except Exception as error:  # the served weights are unknown: the start fails (Ctrl-C passes)
            raise _ProbeReloadFailed(f"reload after the expert probe: {error}") from error

    def _expert_file_update(self, model_dir: Path, version: int) -> dict[str, Any]:
        """Apply a release through its changed expert files only (module doc of
        :mod:`.k31_release`); :class:`~.k31_release.FileSparseUnavailable` before
        anything is sent when the release does not qualify."""
        from yeto.rl.engine.release_delta import read_changes

        began = self._clock()
        manifest = read_changes(model_dir)
        served = self._served_dir
        if manifest is None:  # a partial-experts release laid over the base: its release files
            plan = k31_release.overlay_update_plan(model_dir, served, self.config.base_dir)
            if plan is None:
                raise k31_release.FileSparseUnavailable("no_change_manifest", str(model_dir))
        elif served is None or Path(manifest["base_dir"]).resolve() != served.resolve():
            raise k31_release.FileSparseUnavailable("base_mismatch", f"rebuilt from {manifest['base_dir']}, serving {served}")
        else:
            plan = k31_release.expert_update_plan(model_dir, (manifest.get("files") or {}).keys())
        live = bool(self.options.live_update)
        if live:
            self.call("POST", "/pause_generation", {"mode": "retract"})
        paused = self._clock()
        failure: BaseException | None = None
        try:
            detail = self._send_tensors(plan, version=version)
        except SamplerEngineError as error:
            failure = error
            detail = {}
        finally:
            updated = self._clock()
            if live:
                self.call("POST", "/continue_generation", {})
        detail.update({"expert_update": failure is None, "plan_s": round(paused - began, 3),
                       "update_s": round(updated - paused, 3), **({"live": True} if live else {})})
        if failure is not None:  # a batch may be half applied: the whole release from disk
            repaired = self._reload_full(model_dir, version) or {}
            return {**detail, "expert_update_repair": str(failure)[:300], **repaired}
        return detail

    # ------------------------------------------------------------------ sparse updates
    def _sparse_root(self) -> Path:
        if self.options.sparse_dir:
            return Path(self.options.sparse_dir)
        import tempfile

        shm = Path("/dev/shm")
        return (shm if shm.is_dir() else Path(tempfile.gettempdir())) / f"yeto-sparse-{self.config.port}"

    def _sparse_call(self, commands: list[dict[str, Any]], *, version: int | None = None,
                     timeout_s: float | None = None) -> list[dict[str, Any]]:
        """Send one command per rank through the custom loader; the ranks' results."""
        for command in commands:
            Path(command["result"]).unlink(missing_ok=True)
        body: dict[str, Any] = {"serialized_named_tensors": [_sparse_payload(c) for c in commands],
                                "load_format": SPARSE_LOADER, "flush_cache": False}
        if version is not None:
            body["weight_version"] = str(int(version))
        session = self._begin_weight_update()
        try:
            reply = self.call("POST", "/update_weights_from_tensor", body, timeout_s=timeout_s)
        finally:
            if session:
                self.call("POST", "/end_weight_update", {})
        if isinstance(reply, dict) and reply.get("success") is False:
            raise SamplerEngineError(f"sglang update_weights_from_tensor: {reply.get('message')}")
        results = []
        for command in commands:
            try:
                results.append(json.loads(Path(command["result"]).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                results.append({"ok": False, "error": f"no result from rank {command['rank']} "
                                f"(is yeto importable by the server's python?)"})
        return results

    def _begin_weight_update(self) -> bool:
        """Open a weight-update session where the server has them (the pinned
        fork: ``/update_weights_from_tensor`` outside a ``/begin_weight_update``
        session fails an assertion that ends the scheduler). False: the server has
        no sessions (upstream SGLang; remembered until the next start)."""
        if self._weight_sessions is False:
            return False
        try:
            self.call("POST", "/begin_weight_update", {})
        except SamplerEngineError as error:
            if self._weight_sessions is None and "HTTP 404" in str(error):
                self._weight_sessions = False
                return False
            raise
        self._weight_sessions = True
        return True

    def _sparse_prepare(self) -> dict[str, Any]:
        """Probe the element maps of the started server (module doc). On any
        failure sparse updates stay off for this server (``sparse_off``)."""
        assert self._served_dir is not None
        root, tp = self._sparse_root(), int(self.options.tp)
        began = self._clock()
        record: dict[str, Any] = {}
        try:
            slots = sparse_update.release_slots(self._served_dir)
            if not slots:
                raise SamplerEngineError(f"{self._served_dir}: no *.safetensors files")
            shapes = {name: list(slot.shape) for name, slot in slots.items()}
            maps_dir = root / "maps"
            maps_dir.mkdir(parents=True, exist_ok=True)
            commands = [{"op": "prepare", "rank": r, "tensors": shapes, "out": str(maps_dir / f"rank{r}.pkl"),
                         "result": str(maps_dir / f"prepare-rank{r}.json"),
                         "probe_device": self.options.sparse_probe_device,
                         "threads": max(1, (os.cpu_count() or 1) // tp)} for r in range(tp)]
            try:
                results = self._sparse_call(commands, timeout_s=self.options.sparse_prepare_timeout_s)
            finally:  # the probe may have touched real tensors: serve the release as loaded from disk
                try:
                    self._update_from_disk({"model_path": str(self._served_dir),
                                            "weight_version": str(self._started_version)})
                except Exception as error:  # not a sparse failure: the served weights are unknown
                    raise _ProbeReloadFailed(f"reload after the sparse probe: {error}") from error
            failed = [r for r in results if not r.get("ok")]
            if failed:
                raise SamplerEngineError(f"sparse probe failed: {failed[0].get('error')}")
            maps = [sparse_update.load_map(Path(c["out"])) for c in commands]
            record = {"mapped": min(r["mapped"] for r in results),
                      "unmapped": max(r["unmapped"] for r in results), "probe_s": max(r["seconds"] for r in results),
                      "exact_groups": max(len(r.get("exact_groups") or []) for r in results)}
            mapped = set.intersection(*(set(m["tensors"]) for m in maps))
            rng = np.random.default_rng(int(self._started_version))
            checks = sparse_update.pick_checks(rng, slots, mapped, {}, tensors=4 * self.options.sparse_check_tensors,
                                               per_tensor=self.options.sparse_check_elements)
            summary = sparse_update.rank_patches(root / "check", maps, slots, {}, checks,
                                                 {"version": self._started_version})
            results = self._sparse_call([{"op": "check", "rank": r, "patch": summary[r]["dir"],
                                          "result": str(root / f"check-rank{r}.json")} for r in range(tp)])
            bad = [r for r in results if not r.get("ok") or r.get("mismatches")]
            record["checked"] = sum(int(r.get("checked") or 0) for r in results)
            if bad:
                raise SamplerEngineError(f"sparse layout check failed: {bad[0].get('error') or bad[0]}")
            self._sparse_maps = maps
            self._sparse_shapes = shapes
        except (SamplerEngineError, sparse_update.SparseUnavailable, OSError, ValueError, KeyError) as error:
            self._sparse_maps = None
            import shutil

            shutil.rmtree(root, ignore_errors=True)  # sparse stays off: free the maps (shared memory)
            if isinstance(error, _ProbeReloadFailed):
                raise
            record["error"] = f"{type(error).__name__}: {str(error)[:300]}"
        record["prepare_s"] = round(self._clock() - began, 3)
        record["enabled"] = self._sparse_maps is not None
        self.sparse_prepare = record
        return record

    def _sparse_plan(self, model_dir: Path, version: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """The rank patches of the update to ``model_dir`` (written before any pause),
        or :class:`~.sparse_update.SparseUnavailable`."""
        from yeto.rl.engine.release_delta import MIRROR_STAMP, read_changes

        manifest = read_changes(model_dir)
        if manifest is None:
            raise sparse_update.SparseUnavailable("no_change_manifest", str(model_dir))
        served = self._served_dir
        if served is None or Path(manifest["base_dir"]).resolve() != served.resolve():
            raise sparse_update.SparseUnavailable("base_mismatch", f"rebuilt from {manifest['base_dir']}, "
                                                  f"serving {served}")
        if manifest.get("base_stamp") is not None:
            try:
                stamp = json.loads((served / MIRROR_STAMP).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stamp = None
            if stamp != manifest["base_stamp"]:
                raise sparse_update.SparseUnavailable("base_mismatch", "the served copy is another release")
        slots = sparse_update.release_slots(model_dir)
        if {n: list(s.shape) for n, s in slots.items()} != self._sparse_shapes:
            raise sparse_update.SparseUnavailable("layout_changed", "tensor names or shapes differ")
        changes = sparse_update.changed_elements(model_dir, manifest["files"])
        changed = sum(int(i.size) for i in changes.values())
        total = sum(s.numel for s in slots.values())
        ratio = changed / total if total else 1.0
        if ratio > float(self.options.sparse_max_ratio):
            raise sparse_update.SparseUnavailable("too_many_changes", f"{ratio:.4f} > {self.options.sparse_max_ratio}")
        maps = self._sparse_maps or []
        rng = np.random.default_rng(int(version))
        mapped = set.intersection(*(set(m["tensors"]) for m in maps))
        checks = sparse_update.pick_checks(rng, slots, mapped, changes, tensors=self.options.sparse_check_tensors,
                                           per_tensor=self.options.sparse_check_elements)
        summary = sparse_update.rank_patches(self._sparse_root() / "patch", maps, slots, changes, checks,
                                             {"version": int(version)})
        return summary, {"changed_elements": changed, "changed_tensors": len(changes),
                         "changed_ratio": round(ratio, 6)}

    def _sparse_update(self, model_dir: Path, version: int) -> dict[str, Any]:
        began = self._clock()
        try:  # may raise SparseUnavailable: nothing touched
            summary, detail = self._sparse_plan(model_dir, version)
        except (sparse_update.SparseUnavailable, OSError, ValueError, KeyError) as error:
            import shutil

            # e.g. /dev/shm full: still nothing touched, but a half-written patch must not hold the RAM
            shutil.rmtree(self._sparse_root() / "patch", ignore_errors=True)
            if isinstance(error, sparse_update.SparseUnavailable):
                raise
            raise sparse_update.SparseUnavailable("plan_error", f"{type(error).__name__}: {error}") from error
        planned = self._clock()
        root = self._sparse_root()
        commands = [{"op": "apply", "rank": r, "patch": s["dir"], "result": str(root / f"apply-rank{r}.json")}
                    for r, s in enumerate(summary)]
        live = bool(self.options.live_update)
        if live:
            self.call("POST", "/pause_generation", {"mode": "retract"})
        paused = self._clock()
        try:
            try:
                results = self._sparse_call(commands, version=version)
            except SamplerEngineError as error:
                results = [{"ok": False, "error": str(error)}]
        finally:
            updated = self._clock()
            resumed_error = None
            if live:
                try:
                    self.call("POST", "/continue_generation", {})
                except SamplerEngineError as error:  # the patch may be partly written: repair below
                    resumed_error = error
        if resumed_error is not None:
            results = [{"ok": False, "error": f"continue_generation after the sparse write: {resumed_error}"}]
        detail.update({"plan_s": round(planned - began, 3), "update_s": round(updated - paused, 3),
                       "patch_elements": sum(s["elements"] for s in summary),
                       "checked": sum(int(r.get("checked") or 0) for r in results),
                       "write_s": max(float(r.get("write_s") or 0.0) for r in results)})
        if live:
            detail.update({"live": True, "paused_s": round(self._clock() - paused, 3)})
        bad = [r for r in results if not r.get("ok") or r.get("mismatches")]
        if not bad:
            return {"sparse": True, **detail}
        # the weights may be partly written: load the whole release now
        reason = bad[0].get("error") or f"{bad[0].get('mismatches')} check mismatches {bad[0].get('examples')}"
        repaired = self._reload_full(model_dir, version) or {}
        return {"sparse": False, "sparse_repair": str(reason)[:300], **{f"sparse_{k}": v for k, v in detail.items()},
                **repaired}

    def _remove_sparse_dir(self) -> None:
        if self.sparse_off is None:  # element maps and patches (in /dev/shm by default)
            import shutil

            shutil.rmtree(self._sparse_root(), ignore_errors=True)

    def stop(self) -> None:
        super().stop()
        self._remove_sparse_dir()

    def _kill(self) -> None:
        super()._kill()
        self._remove_sparse_dir()

    def _update_from_disk(self, request: dict[str, Any]) -> Any:
        body = self.call("POST", "/update_weights_from_disk", request)
        if isinstance(body, dict) and body.get("success") is False:
            raise SamplerEngineError(f"sglang update_weights_from_disk: {body.get('message')}")
        return body

    # ------------------------------------------------------------------ generation
    @staticmethod
    def _params(sampling: SamplingParams, max_new_tokens: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {
            "max_new_tokens": int(sampling.max_tokens if max_new_tokens is None else max_new_tokens),
            "temperature": float(sampling.temperature), "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k),
            "repetition_penalty": float(sampling.repetition_penalty),
            "frequency_penalty": float(sampling.frequency_penalty),
            "presence_penalty": float(sampling.presence_penalty),
        }
        if sampling.stop:
            params["stop"] = list(sampling.stop)
        if sampling.seed is not None:
            params["sampling_seed"] = int(sampling.seed)
        return params

    def _request_fields(self) -> dict[str, Any]:
        """The per-request flags of every ``/generate`` that samples (not the
        teacher-forced scores): log-probs always; R3 routed experts when on."""
        fields: dict[str, Any] = {"return_logprob": True}
        if self.options.return_routed_experts:
            fields["return_routed_experts"] = True
        if self.top_logprobs_num:
            fields.update(top_logprobs_num=self.top_logprobs_num, return_flat_raw_output_top_logprobs=True,
                          return_flat_raw_top_logprobs_b64=True)
        return fields

    def _generate_raw(self, texts: Sequence[str], sampling: SamplingParams) -> list[dict[str, Any]]:
        body = self.call("POST", "/generate", {"text": list(texts), "sampling_params": self._params(sampling),
                                               **self._request_fields()})
        items = body if isinstance(body, list) else [body]
        if len(items) != len(texts):
            raise SamplerEngineError(f"sglang returned {len(items)} outputs for {len(texts)} inputs")
        return items

    def _sent_version(self) -> int | None:
        """The version a request sent now starts on (only with live updates)."""
        return self._version if self.options.live_update else None

    def generate(self, prompts: Sequence[str], sampling: SamplingParams) -> list[list[Completion]]:
        n = int(sampling.n)
        texts = [p for p in prompts for _ in range(n)]  # one request per sample: order is explicit
        sent = self._sent_version()
        return self._completions(self._generate_raw(texts, sampling.replace(n=1)), len(prompts), n,
                                 sent_version=sent, started_version=self._started_version,
                                 moe_layers=self._moe_layers, dense_layers=self._dense_layers,
                                 routed_top_k=self._routed_top_k, top_k=self.top_logprobs_num)

    def generate_tokens(self, prompts: Sequence[Sequence[int]], sampling: SamplingParams, *,
                        stop_token_ids: Sequence[int] = ()) -> list[list[Completion]]:
        """Token-id prompts (``input_ids``), for the records proxy."""
        n = int(sampling.n)
        inputs = [[int(t) for t in p] for p in prompts for _ in range(n)]
        params = self._params(sampling.replace(n=1))
        sent = self._sent_version()
        if stop_token_ids:
            params["stop_token_ids"] = [int(t) for t in stop_token_ids]
        body = self.call("POST", "/generate", {"input_ids": inputs, "sampling_params": params,
                                               **self._request_fields()})
        items = body if isinstance(body, list) else [body]
        if len(items) != len(inputs):
            raise SamplerEngineError(f"sglang returned {len(items)} outputs for {len(inputs)} inputs")
        completions = self._completions(items, len(prompts), n, sent_version=sent,
                                        started_version=self._started_version,
                                        prompt_lengths=[len(p) for p in inputs], moe_layers=self._moe_layers,
                                        dense_layers=self._dense_layers, routed_top_k=self._routed_top_k,
                                        top_k=self.top_logprobs_num)
        return [[c if c.prompt_token_ids else replace(c, prompt_token_ids=tuple(inputs[i * n]))
                 for c in group] for i, group in enumerate(completions)]

    @staticmethod
    def _completions(items: Sequence[dict[str, Any]], prompts: int, n: int, *,
                     sent_version: int | None = None, started_version: int = 0,
                     prompt_lengths: Sequence[int] | None = None,
                     moe_layers: int | None = None, dense_layers: int = 0,
                     routed_top_k: int | None = None, top_k: int = 0) -> list[list[Completion]]:
        """``sent_version`` (live updates only): set both completion versions
        (:func:`_completion_versions`). ``prompt_lengths``: the prompt token
        count of each item (id prompts); else ``meta_info.prompt_tokens``."""
        out: list[list[Completion]] = []
        for i in range(prompts):
            group = []
            for j, item in enumerate(items[i * n:(i + 1) * n]):
                meta = item.get("meta_info") or {}
                tokens = meta.get("output_token_logprobs") or []
                versions = ({} if sent_version is None else
                            _completion_versions(meta, sent_version, started_version))
                prompt_tokens = (prompt_lengths[i * n + j] if prompt_lengths is not None
                                 else int(meta.get("prompt_tokens") or 0))
                experts = decode_routed_experts(meta, prompt_tokens=prompt_tokens, completion_tokens=len(tokens),
                                                moe_layers=moe_layers, dense_layers=dense_layers,
                                                top_k=routed_top_k)
                top = decode_top_logprobs(meta, completion_tokens=len(tokens), k=top_k)
                if top_k and top is None:
                    raise SamplerEngineError(f"sglang returned no top log-probs; {top_k} per token were asked")
                group.append(Completion(token_ids=tuple(int(t[1]) for t in tokens),
                                        logprobs=tuple(float(t[0]) for t in tokens),
                                        text=str(item.get("text", "")), finish_reason=_finish(meta),
                                        routed_experts=experts,
                                        top_logprobs_ids=None if top is None else top[0],
                                        top_logprobs=None if top is None else top[1], **versions))
            out.append(group)
        return out

    def _input_token_logprobs(self, prompt: dict[str, Any]) -> list[tuple[float | None, int]]:
        """``(logprob, token id)`` of every input token; ``prompt`` is ``{"text": ...}`` or
        ``{"input_ids": [...]}``."""
        body = self.call("POST", "/generate", {
            **prompt, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "return_logprob": True, "logprob_start_len": 0})
        body = body[0] if isinstance(body, list) else body
        return [(None if t[0] is None else float(t[0]), int(t[1]))
                for t in (body.get("meta_info") or {}).get("input_token_logprobs") or []]

    def _input_logprobs(self, text: str) -> list[float | None]:
        return [lp for lp, _ in self._input_token_logprobs({"text": text})]

    def score(self, prompt: str, continuation: str) -> list[float]:
        prompt_len = len(self._input_logprobs(prompt))
        full = self._input_logprobs(prompt + continuation)
        tail = full[prompt_len:]
        if not tail or any(x is None for x in tail):
            raise SamplerEngineError(f"sglang gave no input log-probs for continuation {continuation!r}")
        return [float(x) for x in tail]

    def score_tokens(self, prompt: str, token_ids: Sequence[int]) -> list[float]:
        """Teacher-forced log-probs of exactly ``token_ids`` after ``prompt`` (no re-tokenizing
        of the continuation, so a trailing EOS that has no text is scored too)."""
        prompt_ids = [tid for _, tid in self._input_token_logprobs({"text": prompt})]
        full = self._input_token_logprobs({"input_ids": [*prompt_ids, *map(int, token_ids)]})
        tail = [lp for lp, _ in full[len(prompt_ids):]]
        if len(tail) != len(token_ids) or any(x is None for x in tail):
            raise SamplerEngineError(f"sglang scored {len(tail)} of {len(token_ids)} continuation tokens")
        return [float(x) for x in tail]
