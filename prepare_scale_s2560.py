#!/usr/bin/env python3
"""Materialize tokenizer-specific packed inputs for S=2560 scale runs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


SEQUENCE_LENGTH = 128
LEARNERS = 4
STEPS = 2_560
REQUIRED_TOKENS_PER_LEARNER = SEQUENCE_LENGTH * STEPS
SMOL_360_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
SMOL_17_REVISION = "effd688a12921b4cc83e3312b6feb579f70f9c71"
QWEN_REVISION = "d149729398750b98c0af14eb82c78cfe92750796"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def dump_rows(path: Path, dataset, indices: list[int]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for index in indices:
            row = dataset[index]
            payload = {key: row[key] for key in ("messages", "tools") if key in row}
            destination.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )


def tokenizer_hashes(snapshot: Path) -> dict[str, str]:
    result = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        path = snapshot / name
        if path.is_file():
            result[name] = sha256_file(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/yeto"))
    parser.add_argument(
        "--source", type=Path, default=Path("/root/yeto-data/train.parquet")
    )
    parser.add_argument(
        "--v3-root", type=Path, default=Path("/root/yeto-data/outer-mup-v3")
    )
    parser.add_argument(
        "--hf-cache", type=Path, default=Path("/root/yeto-hf-cache/hub")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/root/yeto-data/outer-mup-v3/scale-s2560"),
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from yeto.data import _row_tokens, load_rows

    source = args.source.resolve()
    v3_root = args.v3_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing existing output: {out}")

    provenance = json.loads(
        (v3_root / "seed-301" / "split_provenance.json").read_text()
    )
    train_pool = provenance["train_pool_source_indices"]
    eval_indices = provenance["eval_source_indices"]
    audit_indices = provenance["audit_eval_source_indices"]
    if len(train_pool) != 13_758 or len(set(train_pool)) != 13_758:
        raise SystemExit("v3 training pool is invalid")
    if set(train_pool) & (set(eval_indices) | set(audit_indices)):
        raise SystemExit("training pool overlaps frozen evaluation")
    dataset = load_rows(str(source))

    hf_cache = args.hf_cache.resolve()
    smol_360 = (
        hf_cache
        / "models--HuggingFaceTB--SmolLM2-360M"
        / "snapshots"
        / SMOL_360_REVISION
    )
    smol_17 = (
        hf_cache
        / "models--HuggingFaceTB--SmolLM2-1.7B"
        / "snapshots"
        / SMOL_17_REVISION
    )
    qwen = (
        hf_cache
        / "models--Qwen--Qwen2.5-7B"
        / "snapshots"
        / QWEN_REVISION
    )
    smol_hashes = tokenizer_hashes(smol_360)
    if smol_hashes != tokenizer_hashes(smol_17):
        raise SystemExit("SmolLM2-360M and SmolLM2-1.7B tokenizer files differ")
    variants = (
        {
            "name": "smollm2",
            "model_ids": [
                "HuggingFaceTB/SmolLM2-360M",
                "HuggingFaceTB/SmolLM2-1.7B",
            ],
            "revisions": [SMOL_360_REVISION, SMOL_17_REVISION],
            "snapshot": smol_360,
            "tokenizer_files": smol_hashes,
        },
        {
            "name": "qwen2.5-7b",
            "model_ids": ["Qwen/Qwen2.5-7B"],
            "revisions": [QWEN_REVISION],
            "snapshot": qwen,
            "tokenizer_files": tokenizer_hashes(qwen),
        },
    )

    temporary = Path(tempfile.mkdtemp(prefix=".scale-s2560-", dir=v3_root))
    try:
        raw_root = temporary / "raw"
        raw_root.mkdir(parents=True)
        raw_train = raw_root / "train.jsonl"
        dump_rows(raw_train, dataset, train_pool)
        shutil.copy2(
            v3_root / "seed-301" / "eval.jsonl", raw_root / "eval.jsonl"
        )
        shutil.copy2(
            v3_root / "seed-301" / "confirmation-audit.jsonl",
            raw_root / "confirmation-audit.jsonl",
        )
        raw_files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(raw_root.iterdir())
        }

        variant_records = {}
        for variant in variants:
            tokenizer = AutoTokenizer.from_pretrained(
                variant["snapshot"],
                trust_remote_code=True,
                local_files_only=True,
            )
            selected_ids: list[list[int]] = [[] for _ in range(LEARNERS)]
            selected_weights: list[list[float]] = [[] for _ in range(LEARNERS)]
            available_tokens = [0] * LEARNERS
            for position, source_index in enumerate(train_pool):
                learner = position % LEARNERS
                ids, weights = _row_tokens(tokenizer, dataset[source_index], "assistant")
                available_tokens[learner] += len(ids)
                needed = REQUIRED_TOKENS_PER_LEARNER - len(selected_ids[learner])
                if needed > 0:
                    selected_ids[learner].extend(ids[:needed])
                    selected_weights[learner].extend(weights[:needed])
            if any(len(values) != REQUIRED_TOKENS_PER_LEARNER for values in selected_ids):
                raise SystemExit(f"{variant['name']}: insufficient packed input ids")
            if any(
                len(values) != REQUIRED_TOKENS_PER_LEARNER
                for values in selected_weights
            ):
                raise SystemExit(f"{variant['name']}: insufficient packed weights")

            variant_root = temporary / variant["name"] / "m4"
            variant_root.mkdir(parents=True)
            learners = {}
            for learner in range(LEARNERS):
                input_ids = torch.tensor(selected_ids[learner], dtype=torch.int64).view(
                    STEPS, SEQUENCE_LENGTH
                )
                weights = torch.tensor(
                    selected_weights[learner], dtype=torch.float32
                ).view(STEPS, SEQUENCE_LENGTH)
                destination = variant_root / f"learner-{learner:02d}.safetensors"
                save_file(
                    {"input_ids": input_ids, "weights": weights},
                    destination,
                    metadata={
                        "schema": "yeto_packed_scale_input_v1",
                        "variant": variant["name"],
                        "learner_id": str(learner),
                        "learners": str(LEARNERS),
                        "sequence_length": str(SEQUENCE_LENGTH),
                        "steps": str(STEPS),
                        "train_on": "assistant",
                    },
                )
                with safe_open(destination, framework="pt", device="cpu") as handle:
                    loaded_ids = handle.get_tensor("input_ids")
                    loaded_weights = handle.get_tensor("weights")
                    metadata = handle.metadata()
                if not torch.equal(input_ids, loaded_ids) or not torch.equal(
                    weights, loaded_weights
                ):
                    raise SystemExit(f"{destination}: safetensors round-trip mismatch")
                if metadata.get("schema") != "yeto_packed_scale_input_v1":
                    raise SystemExit(f"{destination}: metadata round-trip mismatch")
                learners[str(learner)] = {
                    "path": str(out / variant["name"] / "m4" / destination.name),
                    "file_bytes": destination.stat().st_size,
                    "file_sha256": sha256_file(destination),
                    "input_ids": {
                        "dtype": str(input_ids.dtype),
                        "shape": list(input_ids.shape),
                        "sha256": tensor_sha256(input_ids),
                    },
                    "weights": {
                        "dtype": str(weights.dtype),
                        "shape": list(weights.shape),
                        "sha256": tensor_sha256(weights),
                        "trained_token_weight_sum": float(weights.sum().item()),
                    },
                    "available_raw_tokens": available_tokens[learner],
                    "available_complete_blocks": (
                        available_tokens[learner] // SEQUENCE_LENGTH
                    ),
                    "selected_raw_tokens": input_ids.numel(),
                    "selected_complete_blocks": input_ids.shape[0],
                }
            variant_records[variant["name"]] = {
                "model_ids": variant["model_ids"],
                "revisions": variant["revisions"],
                "tokenizer_snapshot": str(variant["snapshot"]),
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_files": variant["tokenizer_files"],
                "learners": learners,
            }
            del tokenizer, selected_ids, selected_weights
            gc.collect()

        manifest = {
            "schema": "yeto_scale_s2560_inputs_v1",
            "status": "PASS",
            "pipeline": {
                "implementation": "yeto.data._row_tokens/build_packed_dataset",
                "train_on": "assistant",
                "learners": LEARNERS,
                "micro_batch_size": 1,
                "sequence_length": SEQUENCE_LENGTH,
                "steps_per_learner": STEPS,
                "tokens_per_learner": REQUIRED_TOKENS_PER_LEARNER,
                "tokens_all_learners": REQUIRED_TOKENS_PER_LEARNER * LEARNERS,
            },
            "raw": {
                "train_pool_rule": "v3 seed-independent expanded train pool order",
                "train_rows": len(train_pool),
                "files": raw_files,
            },
            "variants": variant_records,
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
                "manifest": str(manifest_path),
                "sha256": sha256_file(manifest_path),
                "variants": [variant["name"] for variant in variants],
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
