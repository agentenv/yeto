#!/usr/bin/env python3
"""Verify v8 shard integrity and exact SmolLM2 packed-token capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


V8_SEEDS = (801, 809, 811, 821, 823)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/root/yeto"))
    parser.add_argument(
        "--source", type=Path, default=Path("/root/yeto-data/train.parquet")
    )
    parser.add_argument(
        "--root", type=Path, default=Path("/root/yeto-data/outer-mup-v8")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/root/yeto-data/model")
    )
    parser.add_argument("--learners", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20_480)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(args.repo.resolve()))
    from transformers import AutoTokenizer
    from yeto.data import _row_tokens, load_rows

    root = args.root.resolve()
    combined_path = root / "input-manifest.json"
    combined = json.loads(combined_path.read_text())
    if combined.get("schema") != "yeto_outer_mup_v8_inputs_v1":
        raise SystemExit("v8 input manifest schema mismatch")
    observed_seeds = tuple(sorted(int(seed) for seed in combined["seeds"]))
    if observed_seeds != V8_SEEDS:
        raise SystemExit(f"unexpected seed set: {observed_seeds}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.resolve(), trust_remote_code=True, local_files_only=True
    )
    dataset = load_rows(str(args.source.resolve()))
    row_tokens = [len(_row_tokens(tokenizer, row, "assistant")[0]) for row in dataset]
    required_blocks = args.steps * args.micro_batch_size
    required_tokens = required_blocks * args.seq_len
    seed_results: dict[str, object] = {}
    minimum: dict[str, int | None] = {
        "blocks": None,
        "tokens": None,
        "seed": None,
        "learner": None,
    }

    for seed in V8_SEEDS:
        seed_root = root / f"seed-{seed}"
        manifest = json.loads((seed_root / "input-manifest.json").read_text())
        provenance = json.loads((seed_root / "split_provenance.json").read_text())
        train_indices = provenance["train_source_indices"]
        if len(train_indices) != 13_758 or len(set(train_indices)) != 13_758:
            raise SystemExit(f"seed {seed}: invalid expanded train index set")
        for label, item in manifest["files"].items():
            path = Path(item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise SystemExit(f"seed {seed}: missing or wrong-size {label}")
            if sha256_file(path) != item["sha256"]:
                raise SystemExit(f"seed {seed}: hash mismatch for {label}")

        learner_tokens = [
            sum(row_tokens[index] for index in train_indices[learner :: args.learners])
            for learner in range(args.learners)
        ]
        learner_blocks = [tokens // args.seq_len for tokens in learner_tokens]
        if min(learner_blocks) < required_blocks:
            raise SystemExit(
                f"seed {seed}: only {min(learner_blocks)} blocks; "
                f"need {required_blocks}"
            )
        for learner, (tokens, blocks) in enumerate(zip(learner_tokens, learner_blocks)):
            if minimum["blocks"] is None or blocks < minimum["blocks"]:
                minimum = {
                    "blocks": blocks,
                    "tokens": tokens,
                    "seed": seed,
                    "learner": learner,
                }
        seed_results[str(seed)] = {
            "learner_raw_tokens": learner_tokens,
            "learner_complete_blocks": learner_blocks,
            "learner_usable_packed_tokens": [
                blocks * args.seq_len for blocks in learner_blocks
            ],
            "minimum_block_slack": min(learner_blocks) - required_blocks,
            "train_sha256": manifest["files"]["train"]["sha256"],
            "train_bytes": manifest["files"]["train"]["bytes"],
        }

    tokenizer_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = args.tokenizer.resolve() / name
        if path.is_file():
            tokenizer_files[name] = sha256_file(path)
    result = {
        "schema": "yeto_outer_mup_v8_token_verification_v1",
        "status": "PASS",
        "pipeline": {
            "implementation": "yeto.data._row_tokens+build_packed_dataset semantics",
            "train_on": "assistant",
            "learners": args.learners,
            "micro_batch_size": args.micro_batch_size,
            "sequence_length": args.seq_len,
            "steps_per_learner": args.steps,
            "required_blocks_per_learner": required_blocks,
            "required_tokens_per_learner": required_tokens,
            "required_tokens_all_learners": required_tokens * args.learners,
        },
        "tokenizer": {
            "path": str(args.tokenizer.resolve()),
            "model_id": (args.tokenizer.resolve() / "model-id.txt").read_text().strip(),
            "revision": (
                args.tokenizer.resolve() / "model-revision.txt"
            ).read_text().strip(),
            "files": tokenizer_files,
        },
        "source_rows": len(dataset),
        "source_row_token_total": sum(row_tokens),
        "input_manifest_sha256": sha256_file(combined_path),
        "minimum_across_all_seeds_and_learners": minimum,
        "seeds": seed_results,
    }
    output = root / "token-counts-smollm2.json"
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "minimum": minimum,
                "required_blocks": required_blocks,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
