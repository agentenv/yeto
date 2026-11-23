"""Launch-time check of a generic partial-experts run (any MoE model).

``--tuning full --rl-expert-selection FILE`` with the generic recipe trains
the routed experts an ``in_place`` selection names and freezes every other
parameter (``models/partial_experts/in_place.py`` on the island). Before any
cloud work the launcher checks the file (schema, digest, layout), its EP size
against ``--expert-parallel`` and, when the checkpoint's ``config.json`` is
readable here, its expert count and MoE layers (:mod:`yeto.rl.moe_geometry`).
The island checks the same against the Megatron-Bridge provider it builds.
"""

from __future__ import annotations

import os
from typing import Any

IN_PLACE = "in_place"
HINT = ("generate an in_place selection from the checkpoint's config.json with "
        "python -m yeto.rl.adapters.miles.models.partial_experts.selection (docs/MILES_RL.md, "
        "'Partial routed experts') and check it with --validate")


def check_generic_selection(args) -> dict[str, Any]:
    """Validate ``--rl-expert-selection`` of a generic ``--tuning full`` launch; returns a summary."""

    from ..rl.engine import backends
    from ..rl.moe_geometry import MoeGeometryError, from_hf_config
    from .k31_checks import _model_config

    sel = backends.module("expert_selection", "miles")
    path = os.path.expanduser(str(args.rl_expert_selection))
    try:
        selection = sel.ExpertSelection.load_file(path)
    except FileNotFoundError:
        raise ValueError(f"--rl-expert-selection {path}: no such file; {HINT}") from None
    except (sel.SelectionError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"--rl-expert-selection {path} is not a valid selection: {exc}; {HINT}") from None
    if selection.layout != IN_PLACE:
        raise ValueError(f"--rl-expert-selection {path} has layout {selection.layout!r}; the generic recipe "
                         f"trains an in_place selection (slot_order is the kimi_k31 recipe's); {HINT}")
    ep = int(getattr(args, "expert_parallel", 0) or 0)
    if ep and ep != selection.ep_size:
        raise ValueError(f"--rl-expert-selection was built for EP {selection.ep_size} but the launch uses "
                         f"--expert-parallel {ep}; regenerate it (--input {path} --ep-size {ep})")
    warnings: list[str] = []
    config = _model_config(getattr(args, "model", ""))
    if config is None:
        warnings.append("the model's config.json is not readable here; the island checks the selection "
                        "against the checkpoint's MoE layers")
    else:
        try:
            geometry = from_hf_config(config)
        except MoeGeometryError as exc:
            raise ValueError(f"--rl-expert-selection: cannot read the model's MoE layout: {exc}") from None
        if geometry is None:
            raise ValueError("--rl-expert-selection needs a mixture-of-experts model; the checkpoint has no "
                             "routed experts")
        if geometry.num_experts != selection.num_experts:
            raise ValueError(f"--rl-expert-selection covers {selection.num_experts} experts per layer; the "
                             f"checkpoint has {geometry.num_experts}")
        try:
            selection.check_layers(geometry.moe_layers)
        except ValueError as exc:
            raise ValueError(f"--rl-expert-selection does not match the checkpoint: {exc}") from None
    return {"sha256": selection.sha256, "layout": selection.layout, "num_experts": selection.num_experts,
            "num_trainable": selection.num_trainable, "ep_size": selection.ep_size,
            "moe_layers": len(selection.trainable), "warnings": warnings}
