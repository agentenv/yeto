"""Offline full-parameter layout of a Miles Nemotron-H island (CPU only, no weights).

Prints the fragment count and layout hash that an island with this
parallelism reports at its layout check, from the checkpoint's config.json:

  python tools/nemotron_h_offline_layout.py CONFIG.json --model-revision SHA \\
      --tp 8 --pp 4 [--matrix-merge rda] [--fragment-max-bytes 2147483648]

The values are what the launcher takes as ``--fragments`` and
``--rl-parameter-layout-sha256``. ``config_sha256`` is the sha256 of the file
given (the island hashes ``<model>/config.json``), so pass the exact file the
island loads. Scope: :mod:`yeto.rl.adapters.miles.models.nemotron_h_layout`
(DP=1, CP=1, EP=TP, MTP dropped, vision not built).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    from yeto.rl.adapters.miles.models import nemotron_h_layout as L

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path, help="the checkpoint's config.json")
    parser.add_argument("--model-revision", required=True, help="HF commit of the checkpoint (40 hex)")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--ep", type=int, default=None, help="default: TP (required at DP=1)")
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--matrix-merge", default="avg", choices=("avg", "rda"), help="the run's --rl-sync-matrix-merge")
    parser.add_argument("--fragment-max-bytes", type=int, default=2 << 30)
    parser.add_argument("--minimum-fragments", type=int, default=1)
    args = parser.parse_args(argv)

    raw = args.config.read_bytes()
    geometry = L.NemotronHGeometry.from_hf_config(json.loads(raw))
    manifests = L.rank_manifests(geometry, tp=args.tp, ep=args.ep or args.tp, pp=args.pp)
    layout = L.offline_layout(
        manifests,
        model_revision=args.model_revision,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        matrix_merge=args.matrix_merge,
        max_fragment_bytes=args.fragment_max_bytes,
        minimum_fragments=args.minimum_fragments,
    )
    fragment_bytes = [fragment.numel * 4 for fragment in layout.fragments.fragments]
    rank_numels = [manifest.numel for manifest in manifests]
    print(json.dumps({
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "model_revision": args.model_revision.lower(),
        "parallel": {"tp": args.tp, "ep": args.ep or args.tp, "pp": args.pp, "cp": 1, "dp": 1,
                     "training_gpus": len(manifests)},
        "layer_counts": geometry.layer_counts(),
        "mtp_layers_dropped": geometry.mtp_layers,
        "parameter_tensor_count": sum(len(manifest.specs) for manifest in manifests),
        "parameter_scalar_count": sum(rank_numels),
        "rank_scalars": {"max": max(rank_numels), "min": min(rank_numels)},
        "matrix_merge": args.matrix_merge,
        "fragments": layout.fragments.num_fragments,
        "fragment_bytes": {"max": max(fragment_bytes), "min": min(fragment_bytes)},
        "parameter_layout_sha256": layout.layout_hash,
        "manifests": [
            {"shard_id": manifest.shard_id, "parameter_count": len(manifest.specs),
             "scalar_count": manifest.numel, "source_layout_sha256": L.miles_source_layout_sha256(manifest)}
            for manifest in manifests
        ],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
