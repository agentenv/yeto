#!/usr/bin/env python3
"""Materialize the frozen UltraChat shard for tonight-8.5 v13.

The SmolLM2 studies use trl-lib/Capybara.  V13 instead takes deterministic,
disjoint rows from UltraChat's public train_sft/test_sft splits and keeps only
the ``messages`` field consumed by Yeto.  The source revisions, row indices,
bytes, and Pythia model files are all recorded before any v13 training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path


DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
MODEL_ID = "EleutherAI/pythia-160m"
MODEL_REVISION = "50f5173d932e8e61f858120bcb800b97af589f46"
TRAIN_ROWS = 15_000
EVAL_ROWS = 1_024
TRAIN_INDEX_SEED = 20_260_727
EVAL_INDEX_SEED = 20_260_728


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def selected_indices(size: int, count: int, seed: int) -> list[int]:
    if size < count:
        raise RuntimeError(f"source split has {size} rows, needs {count}")
    population = list(range(size))
    random.Random(seed).shuffle(population)
    return population[:count]


def canonical_row(row: dict, split: str, index: int) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RuntimeError(f"{split}[{index}] has no nonempty messages list")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RuntimeError(
                f"{split}[{index}].messages[{message_index}] is not an object"
            )
        if message.get("role") not in {"system", "user", "assistant"}:
            raise RuntimeError(
                f"{split}[{index}].messages[{message_index}] has invalid role"
            )
        if not isinstance(message.get("content"), str):
            raise RuntimeError(
                f"{split}[{index}].messages[{message_index}] has non-text content"
            )
    return {"messages": messages}


def dump_rows(path: Path, dataset, indices: list[int], split: str) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for index in indices:
            destination.write(
                json.dumps(
                    canonical_row(dataset[index], split, index),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )


def model_inventory(model: Path) -> dict:
    files = {}
    for path in sorted(model.iterdir()):
        if path.name.startswith(".") or path.name in {"README.md"}:
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            continue
        files[path.name] = {
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }
    required = {"config.json", "model.safetensors", "tokenizer.json"}
    missing = required - set(files)
    if missing:
        raise RuntimeError(f"Pythia snapshot lacks required files: {sorted(missing)}")
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "path": str(model),
        "files": files,
        "canonical_inventory_sha256": canonical_sha256(files),
        "total_bytes": sum(record["bytes"] for record in files.values()),
    }


def tokenizer_smoke(model: Path, train_path: Path) -> dict:
    from transformers import AutoTokenizer

    from yeto.data import _row_tokens

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    nonempty = 0
    token_count = 0
    assistant_weight = 0.0
    with train_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line_number > 64:
                break
            ids, weights = _row_tokens(tokenizer, json.loads(line), "assistant")
            if ids:
                nonempty += 1
                token_count += len(ids)
                assistant_weight += sum(weights)
    if nonempty != 64 or token_count < 128 or assistant_weight <= 0:
        raise RuntimeError("Pythia tokenizer smoke did not produce weighted tokens")
    return {
        "rows": nonempty,
        "tokens": token_count,
        "assistant_weight_sum": assistant_weight,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("/root/yeto-data/tonight85-v13")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/root/yeto-hf-cache/hub/models--EleutherAI--pythia-160m/"
            f"snapshots/{MODEL_REVISION}"
        ),
    )
    args = parser.parse_args()
    out = args.out.resolve()
    model = args.model.resolve(strict=True)

    if out.exists():
        manifest_path = out / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"refusing incomplete existing output: {out}")
        manifest = json.loads(manifest_path.read_text())
        for label in ("train", "eval"):
            record = manifest["files"][label]
            path = Path(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or sha256_file(path) != record["sha256"]
            ):
                raise SystemExit(f"existing {label} file fails verification: {path}")
        print(
            json.dumps(
                {
                    "status": "PASS_EXISTING",
                    "manifest": str(manifest_path),
                    "sha256": sha256_file(manifest_path),
                },
                sort_keys=True,
            )
        )
        return 0

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    train_source = dataset["train_sft"]
    eval_source = dataset["test_sft"]
    train_indices = selected_indices(len(train_source), TRAIN_ROWS, TRAIN_INDEX_SEED)
    eval_indices = selected_indices(len(eval_source), EVAL_ROWS, EVAL_INDEX_SEED)

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tonight85-v13-", dir=out.parent))
    try:
        train_path = temporary / "train.jsonl"
        eval_path = temporary / "eval.jsonl"
        dump_rows(train_path, train_source, train_indices, "train_sft")
        dump_rows(eval_path, eval_source, eval_indices, "test_sft")
        final_train = out / "train.jsonl"
        final_eval = out / "eval.jsonl"
        manifest = {
            "schema": "yeto_tonight85_v13_ultrachat_inputs_v1",
            "status": "FROZEN",
            "source": {
                "dataset": DATASET_ID,
                "revision": DATASET_REVISION,
                "train_split": "train_sft",
                "eval_split": "test_sft",
                "smollm2_pipeline_corpus": "trl-lib/Capybara",
                "corpus_is_different": True,
            },
            "selection": {
                "algorithm": "python_random_mt19937_full_index_shuffle_then_prefix",
                "train_index_seed": TRAIN_INDEX_SEED,
                "eval_index_seed": EVAL_INDEX_SEED,
                "train_indices_sha256": canonical_sha256(train_indices),
                "eval_indices_sha256": canonical_sha256(eval_indices),
                "train_rows": len(train_indices),
                "eval_rows": len(eval_indices),
                "split_disjoint_by_source": True,
            },
            "files": {
                "train": {
                    "path": str(final_train),
                    "bytes": train_path.stat().st_size,
                    "sha256": sha256_file(train_path),
                    "rows": TRAIN_ROWS,
                },
                "eval": {
                    "path": str(final_eval),
                    "bytes": eval_path.stat().st_size,
                    "sha256": sha256_file(eval_path),
                    "rows": EVAL_ROWS,
                },
            },
            "model": model_inventory(model),
            "tokenizer_smoke": tokenizer_smoke(model, train_path),
            "required_environment": {
                "HF_DATASETS_CACHE": "/data/hf-datasets-cache"
            },
        }
        write_json_atomic(temporary / "manifest.json", manifest)
        temporary.replace(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    manifest_path = out / "manifest.json"
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "train_sha256": manifest["files"]["train"]["sha256"],
                "eval_sha256": manifest["files"]["eval"]["sha256"],
                "model_inventory_sha256": manifest["model"][
                    "canonical_inventory_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
