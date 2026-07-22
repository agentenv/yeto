"""Export Protenix dataloader batches as Yeto diffusion rows.

This command is meant to run inside a Protenix checkout/environment. It asks
Protenix to build real training batches, stores each batch as a ``torch.save``
artifact, and writes a JSONL manifest that Yeto's Protenix diffusion adapter can
consume via ``protenix_batch_path``.
"""

from __future__ import annotations

import argparse
import json
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Mapping


def _deep_update(base: dict, update: Mapping) -> dict:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export pre-collated Protenix batches for Yeto training"
    )
    p.add_argument("--model-name", default="protenix_base_default_v1.0.0")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--rows-file", default="yeto_protenix_rows.jsonl")
    p.add_argument("--batch-count", type=int, default=1)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--error-dir", default=None)
    p.add_argument(
        "--arg-str",
        default="",
        help="additional Protenix parse_configs arg string; use this for data.train_sets, "
        "crop size, dtype, kernel choices, etc.",
    )
    return p.parse_args(argv)


def _normalize_arg_str(arg_str: str) -> str:
    tokens = shlex.split(arg_str or "")
    normalized = []
    for token in tokens:
        if token.startswith("--") or "=" not in token:
            normalized.append(token)
            continue
        key, value = token.split("=", 1)
        if key == "data.eval_sets":
            key = "data.test_sets"
        if key:
            normalized.extend([f"--{key}", value if value else '""'])
        else:
            normalized.append(token)
    return " ".join(normalized)


def _import_protenix_training_api():
    try:
        import torch
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        from protenix.data.pipeline.dataloader import get_dataloaders
    except ImportError as exc:
        raise RuntimeError(
            "yeto-protenix-export-batch must run in an environment where Protenix "
            "and its training dependencies are importable."
        ) from exc
    return torch, configs_base, data_configs, model_configs, parse_configs, get_dataloaders


def build_configs(args):
    (
        _torch,
        configs_base,
        data_configs,
        model_configs,
        parse_configs,
        _get_dataloaders,
    ) = _import_protenix_training_api()
    if args.model_name not in model_configs:
        raise RuntimeError(
            f"unknown Protenix model {args.model_name!r}; "
            f"available models include {sorted(model_configs)[:8]}"
        )
    configs = deepcopy(dict(configs_base))
    configs["data"] = data_configs
    _deep_update(configs, model_configs[args.model_name])
    base_arg_str = (
        f"--model_name {args.model_name} "
        f"--seed {args.seed} "
        "--run_name yeto_protenix_export "
        "--base_dir ./yeto-protenix-export-run "
        "--use_wandb false "
    )
    return parse_configs(
        configs=configs,
        arg_str=base_arg_str + _normalize_arg_str(args.arg_str),
        fill_required_with_null=True,
    )


def _batch_payload(batch):
    required = ("input_feature_dict", "label_dict", "label_full_dict")
    missing = [key for key in required if key not in batch]
    if missing:
        raise RuntimeError(f"Protenix batch is missing required keys: {missing}")
    return {key: batch[key] for key in required}


def export_batches(args) -> Path:
    torch, *_rest, get_dataloaders = _import_protenix_training_api()
    configs = build_configs(args)
    out = args.output_dir.expanduser().resolve()
    batch_dir = out / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    error_dir = args.error_dir or str(out / "errors")
    Path(error_dir).mkdir(parents=True, exist_ok=True)

    train_dl, _test_dls = get_dataloaders(
        configs,
        args.world_size,
        seed=configs.seed,
        error_dir=error_dir,
    )

    rows_path = out / args.rows_file
    with rows_path.open("w", encoding="utf-8") as rows_f:
        for idx, batch in zip(range(args.batch_count), train_dl):
            batch_path = batch_dir / f"batch-{idx:06d}.pt"
            torch.save(_batch_payload(batch), batch_path)
            rows_f.write(
                json.dumps(
                    {
                        "protenix_batch_path": str(batch_path),
                        "source": "protenix",
                        "model_name": args.model_name,
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
    print(f"wrote Yeto Protenix rows to {rows_path}")


if __name__ == "__main__":
    main()
