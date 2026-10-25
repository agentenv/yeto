"""The verl full-parameter layout of a model, before any GPU work (S27 G-DN).

``yeto launch --rl-backend verl --rl-sync-preset full-parameter`` needs the
island's fragment layout hash (``--rl-parameter-layout-sha256``) and fragment
count (``--fragments``).  A verl island plans its layout from the FP32
parameter shapes of its FSDP2 actor (:func:`.full_parameter.plan_layout`), and
those are the HF model's trainable parameters (tied weights once), whatever the
card count.  This module builds the HF model on the meta device (no weights are
downloaded, only ``config.json``) and runs the same planner, so the launcher
gets the hash the island will observe.  An island whose layout differs refuses
at start and prints its observed layout (``[yeto-verl-layout-probe]``).

Usage::

    python -m yeto.rl.adapters.verl.full_parameter_plan --model Qwen/Qwen3-0.6B \\
        --model-revision <40 hex> --fragments 4 --matrix-merge rda

``--matrix-merge`` must be the run's merge mode (the launcher's async mode
uses ``rda``, the strict sweep ``avg``): it is bound into the layout hash.
Needs ``torch``, ``transformers`` and ``huggingface_hub``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .full_parameter import DEFAULT_MAX_FRAGMENT_BYTES


def model_shapes(model_dir: str | Path, *, trust_remote_code: bool = False) -> dict[str, tuple[int, ...]]:
    """Trainable parameter shapes of the HF model in ``model_dir`` (its
    ``config.json``), built on the meta device; tied weights appear once, as in
    the FSDP2 actor's ``named_parameters``."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    with torch.device("meta"):
        net = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    return {name: tuple(parameter.shape) for name, parameter in net.named_parameters()
            if parameter.requires_grad}


def plan(model: str, model_revision: str, *, fragments: int, matrix_merge: str,
         max_fragment_bytes: int = DEFAULT_MAX_FRAGMENT_BYTES, trust_remote_code: bool = False) -> dict[str, Any]:
    """The layout a verl full-parameter island of ``model`` plans, and the
    launcher flags that name it."""
    from huggingface_hub import hf_hub_download

    from yeto.rl.local_learner import ComponentIdentity

    from . import full_parameter as fp

    if matrix_merge not in ("avg", "rda"):
        raise ValueError("--matrix-merge must be avg or rda")
    config_path = Path(hf_hub_download(model, "config.json", revision=model_revision))
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    shapes = model_shapes(config_path.parent, trust_remote_code=trust_remote_code)
    component = ComponentIdentity(role="actor", model_revision=model_revision.lower(), config_hash=config_sha256)
    layout = fp.plan_layout(fp.island_specs(shapes), component=component, minimum_fragments=int(fragments),
                            max_fragment_bytes=int(max_fragment_bytes), matrix_merge=matrix_merge)
    count = layout.fragments.num_fragments
    return {
        "model": model, "model_revision": model_revision.lower(), "model_config_sha256": config_sha256,
        "matrix_merge": matrix_merge, "max_fragment_bytes": int(max_fragment_bytes),
        "parameter_layout_sha256": layout.layout_hash, "fragments": count,
        "tensors": len(shapes), "scalars": sum(spec.numel for spec in layout.specs),
        "fragment_bytes": [fragment.numel * 4 for fragment in layout.fragments.fragments],
        "launcher_flags": f"--rl-parameter-layout-sha256 {layout.layout_hash} --fragments {count}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m yeto.rl.adapters.verl.full_parameter_plan",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--fragments", type=int, required=True, help="minimum fragment count")
    parser.add_argument("--matrix-merge", choices=("avg", "rda"), required=True)
    parser.add_argument("--max-fragment-bytes", type=int, default=DEFAULT_MAX_FRAGMENT_BYTES,
                        help="the launcher's --rl-sync-fragment-max-bytes")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(plan(args.model, args.model_revision, fragments=args.fragments,
                          matrix_merge=args.matrix_merge, max_fragment_bytes=args.max_fragment_bytes,
                          trust_remote_code=args.trust_remote_code), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
