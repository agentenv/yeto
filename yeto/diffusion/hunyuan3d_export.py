"""Export Hunyuan3D training batches as Yeto diffusion rows.

This command is meant to run inside a Tencent-Hunyuan/Hunyuan3D-2.1 checkout or
an environment where ``hy3dshape`` is importable. It reuses Hunyuan3D's own
Lightning datamodule configured by the training YAML, stores complete batches as
``torch.save`` artifacts, and writes a JSONL manifest consumed by
``yeto.diffusion.adapters.hunyuan3d``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _expand(value: str | os.PathLike | None) -> Path | None:
    if not value:
        return None
    return Path(os.path.expanduser(str(value))).resolve()


def _add_python_path(path: Path | None) -> None:
    if path is not None and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _setup_hunyuan_paths(root: Path | None) -> None:
    if root is None:
        root = _expand(os.environ.get("YETO_HUNYUAN3D_ROOT") or os.environ.get("HUNYUAN3D_ROOT"))
    if root is None:
        return
    _add_python_path(root)
    _add_python_path(root / "hy3dshape")
    _add_python_path(root / "hy3dpaint")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export pre-collated Hunyuan3D batches for Yeto training"
    )
    p.add_argument("--config", required=True, type=Path, help="Hunyuan3D hy3dshape training YAML")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--rows-file", default="yeto_hunyuan3d_rows.jsonl")
    p.add_argument("--batch-count", type=int, default=1)
    p.add_argument("--hunyuan3d-root", default=None, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="datamodule loader to export from",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist override, e.g. dataset.params.batch_size=1",
    )
    return p.parse_args(argv)


def _import_hunyuan_training_api(root: Path | None):
    _setup_hunyuan_paths(root)
    try:
        import torch
        from omegaconf import OmegaConf
        from pytorch_lightning import seed_everything
        from hy3dshape.utils import get_config_from_file, instantiate_from_config
    except ImportError as exc:
        raise RuntimeError(
            "yeto-hunyuan3d-export-batch must run where Hunyuan3D-2.1 training "
            "dependencies are importable. Set YETO_HUNYUAN3D_ROOT or pass "
            "--hunyuan3d-root to the checkout."
        ) from exc
    return torch, OmegaConf, seed_everything, get_config_from_file, instantiate_from_config


def _load_config(args):
    _torch, OmegaConf, _seed_everything, get_config_from_file, _instantiate = _import_hunyuan_training_api(
        _expand(args.hunyuan3d_root)
    )
    config = get_config_from_file(str(args.config.expanduser().resolve()))
    if args.set:
        overrides = OmegaConf.from_dotlist(args.set)
        config = OmegaConf.merge(config, overrides)
    return config


def _build_datamodule(args):
    _torch, _OmegaConf, seed_everything, _get_config, instantiate_from_config = _import_hunyuan_training_api(
        _expand(args.hunyuan3d_root)
    )
    seed_everything(args.seed, workers=True)
    config = _load_config(args)
    if "dataset" not in config:
        raise RuntimeError(f"{args.config}: expected a top-level 'dataset' config")
    data = instantiate_from_config(config.dataset)
    if hasattr(data, "prepare_data"):
        data.prepare_data()
    if hasattr(data, "setup"):
        data.setup("fit")
    return data


def _loader(data, split: str):
    name = "train_dataloader" if split == "train" else "val_dataloader"
    fn = getattr(data, name, None)
    if fn is None:
        raise RuntimeError(f"Hunyuan3D datamodule has no {name}()")
    loader = fn()
    if isinstance(loader, (list, tuple)):
        if not loader:
            raise RuntimeError(f"{name}() returned no loaders")
        return loader[0]
    return loader


def export_batches(args) -> Path:
    torch, *_rest = _import_hunyuan_training_api(_expand(args.hunyuan3d_root))
    data = _build_datamodule(args)
    loader = _loader(data, args.split)

    out = args.output_dir.expanduser().resolve()
    batch_dir = out / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out / args.rows_file

    with rows_path.open("w", encoding="utf-8") as rows_f:
        for idx, batch in zip(range(args.batch_count), loader):
            batch_path = batch_dir / f"batch-{idx:06d}.pt"
            torch.save(batch, batch_path)
            rows_f.write(
                json.dumps(
                    {
                        "hunyuan3d_batch_path": str(batch_path),
                        "source": "hunyuan3d",
                        "config": str(args.config),
                        "split": args.split,
                        "batch_index": idx,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return rows_path


def main(argv=None) -> None:
    args = parse_args(argv)
    rows_path = export_batches(args)
    print(f"wrote Yeto Hunyuan3D rows to {rows_path}")


if __name__ == "__main__":
    main()
