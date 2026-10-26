"""Model FLOPs of a Nemotron-H hybrid (Mamba-2 + MoE + attention) for MFU.

verl's ``flops_counter`` has no ``nemotron_h`` entry (fork acad9875), so its
``perf/mfu`` is not usable for this model; this module is the formula yeto
reports. Forward FLOPs per sequence of length ``s`` (multiply-add = 2 FLOPs):

* attention layer: ``2*h*(q + 2*kv)`` + ``2*q*h`` per token for the
  projections (``q = heads*head_dim``, ``kv = kv_heads*head_dim``), plus the
  causal score and value products ``2 * 2*q * s^2/2`` per sequence;
* Mamba-2 layer: ``in_proj`` ``2*h*(2*d_in + 2*g*N + H)``, ``out_proj``
  ``2*d_in*h``, causal conv ``2*k*(d_in + 2*g*N)``, and the SSD scan
  ``6*d_in*N`` per token (state update, state readout and the intra-chunk
  term counted as one more ``2*d_in*N``; an approximation, small next to the
  projections);
* MoE layer: router ``2*h*E``; latent projections ``2*2*h*L`` when
  ``moe_latent_size`` is set (experts then run at width ``L``); each of the
  ``top_k`` routed experts ``2*2*w*I`` (relu2, not gated: up and down only);
  the shared expert ``2*2*h*I_s`` on the full hidden size;
* LM head ``2*h*V`` per token.

Training FLOPs = 3 x forward (forward + backward). Activation recompute is
not counted: this is model FLOPs (MFU), not hardware FLOPs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .hybrid_params import hybrid_pattern, language_model_config

# Dense BF16 peak, no sparsity (NVIDIA H100/H200 SXM datasheets: 989.4 TFLOPS).
PEAK_BF16_TFLOPS = {"H100": 989.4, "H200": 989.4}


def _get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


def forward_flops(cfg: Mapping[str, Any], seqlens: Iterable[int]) -> float:
    """Forward FLOPs of a batch of sequences for an HF ``nemotron_h`` config dict
    (or a VL config: its ``llm_config``; the vision part and MTP are not counted)."""
    seqlens = [int(s) for s in seqlens]
    tokens = sum(seqlens)
    sq = sum(s * s for s in seqlens)
    pattern = hybrid_pattern(cfg)
    cfg = language_model_config(cfg)
    h = int(cfg["hidden_size"])
    v = int(cfg["vocab_size"])
    head_dim = int(_get(cfg, "head_dim", h // int(cfg["num_attention_heads"])))
    q = int(cfg["num_attention_heads"]) * head_dim
    kv = int(cfg["num_key_value_heads"]) * head_dim
    m_heads = int(cfg["mamba_num_heads"])
    d_in = m_heads * int(cfg["mamba_head_dim"])
    g, n = int(cfg["n_groups"]), int(cfg["ssm_state_size"])
    k = int(_get(cfg, "conv_kernel", 4))
    e = int(_get(cfg, "n_routed_experts", 0))
    top_k = int(_get(cfg, "num_experts_per_tok", 0))
    inter = int(_get(cfg, "moe_intermediate_size", 0))
    shared = int(_get(cfg, "moe_shared_expert_intermediate_size", 0)) * int(_get(cfg, "n_shared_experts", 1))
    latent = _get(cfg, "moe_latent_size")
    width = int(latent) if latent else h
    dense_inter = int(_get(cfg, "intermediate_size", 0))

    per_token = {
        "*": 2 * h * (q + 2 * kv) + 2 * q * h,
        "M": 2 * h * (2 * d_in + 2 * g * n + m_heads) + 2 * d_in * h + 2 * k * (d_in + 2 * g * n) + 6 * d_in * n,
        "E": 2 * h * e + (4 * h * width if latent else 0) + top_k * 4 * width * inter + 4 * h * shared,
        "-": 4 * h * dense_inter,
    }
    total = 0.0
    for symbol in pattern:
        if symbol not in per_token:
            raise ValueError(f"unknown layer symbol {symbol!r}")
        total += per_token[symbol] * tokens
        if symbol == "*":
            total += 2 * q * sq  # 2 products x 2 FLOPs x q x s^2/2 (causal)
    return total + 2 * h * v * tokens


def train_flops(cfg: Mapping[str, Any], seqlens: Iterable[int]) -> float:
    return 3.0 * forward_flops(cfg, seqlens)


def mfu(cfg: Mapping[str, Any], seqlens: Iterable[int], seconds: float, n_gpus: int, gpu: str = "H200") -> float:
    """Model FLOPs utilization of a training phase that processed ``seqlens`` in ``seconds``."""
    return train_flops(cfg, seqlens) / (seconds * n_gpus * PEAK_BF16_TFLOPS[gpu] * 1e12)
