"""Megatron (Miles) expert weights -> re-quantization source slices (S25 N8.2 / N8.5b adapter).

Megatron holds the routed experts of a rank as one weight per local expert:

* grouped GEMM (TE ``TEGroupedMLP``): ``decoder.layers.<l>.mlp.experts.linear_fc1.weight<e>``
  ``[2I, H]`` and ``...linear_fc2.weight<e>`` ``[H, I]``;
* sequential MLP (the ``--transformer-impl local`` fallback, N7.5):
  ``decoder.layers.<l>.mlp.experts.local_experts.<e>.linear_fc{1,2}.weight``.

``<l>`` is the layer index inside the pipeline stage and ``<e>`` the local
expert index; the global expert is ``ep_rank * num_local_experts + e``.
``linear_fc1`` holds the gate rows, then the up rows (SwiGLU ``chunk(2)``).
This adapter renames them to the per-expert HF source names of the
quantization layout; the neutral code (:mod:`yeto.rl.quant.shards`) does the rest.

Non-expert weights are not mapped here: Miles already converts them to HF names
for the sampler weight update (``convert_to_hf``, SGLang names ``model.layers.*``);
exactly one rank per replica group passes those with :func:`hf_slices`, which
renames them to the checkpoint names (``model.language_model.layers.*``; module doc of
:mod:`yeto.rl.quant.shards`, "Ownership").

Limits (explicit errors): expert tensor parallelism must be 1 (a column split of
``linear_fc2`` would cut the W4A8 per-row scale). Status: CPU tests with
synthetic Megatron names; not run inside Miles.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

import torch

from yeto.rl.quant.shards import SourceSlice
from yeto.rl.quant.stacked import DOWN, expert_weight_name, split_gate_up

HF_LAYER_PREFIX = "model.language_model.layers."
_EXPERT = re.compile(
    r"^(?:module\.)*decoder\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?:linear_fc(?P<fc>[12])\.weight(?P<e>\d+)|local_experts\.(?P<e2>\d+)\.linear_fc(?P<fc2>[12])\.weight)$"
)


def megatron_expert_slices(named_params: Iterable[tuple[str, torch.Tensor]], *, ep_rank: int,
                           num_local_experts: int, layer_offset: int = 0, expert_tp_size: int = 1,
                           hf_layer_prefix: str = HF_LAYER_PREFIX) -> Iterator[SourceSlice]:
    """Source slices of this rank's routed experts; other parameters are skipped.

    ``layer_offset``: the global index of the stage's first layer (pipeline
    parallelism)."""
    if int(expert_tp_size) != 1:
        raise ValueError(f"expert tensor parallel size {expert_tp_size} is not supported; use 1")
    if int(num_local_experts) < 1 or int(ep_rank) < 0:
        raise ValueError(f"bad expert parallel position: ep_rank={ep_rank}, num_local_experts={num_local_experts}")
    for name, tensor in named_params:
        match = _EXPERT.match(name)
        if match is None:
            continue
        local = int(match["e"] if match["e"] is not None else match["e2"])
        if local >= num_local_experts:
            raise ValueError(f"{name}: local expert {local} >= num_local_experts {num_local_experts}")
        expert = int(ep_rank) * int(num_local_experts) + local
        prefix = f"{hf_layer_prefix}{int(layer_offset) + int(match['layer'])}.mlp.experts"
        weight = tensor.detach()
        if (match["fc"] or match["fc2"]) == "1":
            for hf_name, part in split_gate_up(prefix, expert, weight):
                yield SourceSlice(hf_name, part)
        else:
            yield SourceSlice(expert_weight_name(prefix, expert, DOWN), weight)


def checkpoint_name(name: str) -> str:
    """Miles ``convert_to_hf`` name -> GLM-5.3 checkpoint name.

    ``convert_to_hf`` emits the SGLang weight-update names (``model.layers.<l>...``,
    ``model.embed_tokens``, ``model.norm``); the GLM-5.3 checkpoint, and so the
    quantization layout, keeps the text model under ``model.language_model.``
    (the same rule as Miles ``Glm5NextBridge._nest_language_model``). Names that
    already carry it, ``model.visual.*`` and top-level names (``lm_head.weight``)
    are unchanged."""
    if name.startswith("model.") and not name.startswith(("model.language_model.", "model.visual.")):
        return "model.language_model." + name[len("model."):]
    return name


def hf_slices(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> Iterator[SourceSlice]:
    """Whole tensors from Miles ``convert_to_hf``, renamed to checkpoint names (:func:`checkpoint_name`)."""
    for name, tensor in named_tensors:
        yield SourceSlice(checkpoint_name(name), tensor.detach())
