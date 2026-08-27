#!/usr/bin/env python3
"""Split immutable Miles offline value-replay buckets into five islands.

The source format is the current Miles debug-rollout payload::

    {"rollout_id": 0, "samples": [sample_dict, ...], ...metadata}

Samples are assigned by their position in each source bucket: position ``i``
goes to island ``i % 5``.  No trace is decoded, rendered, or tokenized.  The
sample dictionaries themselves are placed in the output payload unchanged.

The complete output is built in a sibling staging directory and published by
one directory rename.  Existing output is never replaced unless
``--overwrite`` is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import numbers
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


NUM_ISLANDS = 5
NOMINAL_GLOBAL_BATCH_SIZE = 5
TRAIN_ROLLOUT_IDS = range(0, 364)
VALIDATION_ROLLOUT_IDS = range(364, 395)
SOURCE_FILE_TEMPLATE = "data_{rollout_id}.pt"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1

_RESPONSE_ALIGNED_SAMPLE_FIELDS = (
    "rollout_log_probs",
    "teacher_log_probs",
    "opd_reverse_kl",
)
_TOKEN_MINUS_ONE_ALIGNED_SAMPLE_FIELDS = (
    "rollout_routed_experts",
    "rollout_indexer_topk",
)


class SplitValidationError(ValueError):
    """The source cannot be split without risking a semantic change."""


@dataclass(frozen=True)
class Counts:
    trajectories: int = 0
    supervised_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            trajectories=self.trajectories + other.trajectories,
            supervised_tokens=self.supervised_tokens + other.supervised_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "trajectories": self.trajectories,
            "supervised_tokens": self.supervised_tokens,
            "total_tokens": self.total_tokens,
        }


def _path_exists(path: Path) -> bool:
    """Like Path.exists(), but also sees a broken symlink."""
    return os.path.lexists(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rollout_ids(values: Iterable[int], *, split: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise SplitValidationError(f"{split} rollout IDs must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in result):
        raise SplitValidationError(f"{split} rollout IDs must all be integers")
    normalized = tuple(int(value) for value in result)
    if any(value < 0 for value in normalized):
        raise SplitValidationError(f"{split} rollout IDs must be non-negative")
    if len(set(normalized)) != len(normalized):
        raise SplitValidationError(f"{split} rollout IDs contain duplicates")
    if tuple(sorted(normalized)) != normalized:
        raise SplitValidationError(f"{split} rollout IDs must be ordered")
    return normalized


def _sequence_length(value: Any, *, context: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise SplitValidationError(f"{context} must be one-dimensional, got shape {tuple(value.shape)}")
        return value.numel()
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise SplitValidationError(f"{context} must be a one-dimensional sequence")
    return len(value)


def _loss_mask_counts(mask: Any, *, context: str) -> tuple[int, int]:
    length = _sequence_length(mask, context=context)
    if isinstance(mask, torch.Tensor):
        if mask.is_floating_point() and not bool(torch.isfinite(mask).all().item()):
            raise SplitValidationError(f"{context} contains NaN or infinity")
        if not bool(((mask == 0) | (mask == 1)).all().item()):
            raise SplitValidationError(f"{context} must contain only binary 0/1 values")
        return length, int(mask.to(dtype=torch.int64).sum().item())

    supervised = 0
    for offset, value in enumerate(mask):
        if isinstance(value, bool):
            supervised += int(value)
            continue
        if not isinstance(value, numbers.Real) or value not in (0, 1):
            raise SplitValidationError(
                f"{context}[{offset}] must be a binary 0/1 value, got {value!r}"
            )
        supervised += int(value)
    return length, supervised


def _nonnegative_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise SplitValidationError(f"{context} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise SplitValidationError(f"{context} must be a non-negative integer")
    return result


def _sample_counts(sample: Any, *, rollout_id: int, position: int) -> Counts:
    prefix = f"rollout {rollout_id} sample position {position}"
    if not isinstance(sample, Mapping):
        raise SplitValidationError(f"{prefix} must be a dictionary")
    if "tokens" not in sample:
        raise SplitValidationError(f"{prefix} is missing tokens")
    if "loss_mask" not in sample or sample["loss_mask"] is None:
        raise SplitValidationError(f"{prefix} is missing loss_mask")
    if "response_length" not in sample:
        raise SplitValidationError(f"{prefix} is missing response_length")
    if "remove_sample" in sample:
        remove_sample = sample["remove_sample"]
        if not isinstance(remove_sample, bool):
            raise SplitValidationError(f"{prefix}.remove_sample must be boolean")
        if remove_sample:
            raise SplitValidationError(
                f"{prefix}.remove_sample is true; Miles would replace its loss mask with zeros"
            )

    token_count = _sequence_length(sample["tokens"], context=f"{prefix}.tokens")
    response_length = _nonnegative_integer(
        sample["response_length"], context=f"{prefix}.response_length"
    )
    mask_length, supervised_tokens = _loss_mask_counts(
        sample["loss_mask"], context=f"{prefix}.loss_mask"
    )
    if token_count < response_length:
        raise SplitValidationError(
            f"{prefix} has {token_count} tokens but response_length={response_length}"
        )
    if mask_length != response_length:
        raise SplitValidationError(
            f"{prefix}.loss_mask length {mask_length} != response_length {response_length}"
        )

    metadata = sample.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise SplitValidationError(f"{prefix}.metadata must be a dictionary or null")
    if isinstance(metadata, Mapping):
        if "token_length" in metadata:
            recorded_token_count = _nonnegative_integer(
                metadata["token_length"], context=f"{prefix}.metadata.token_length"
            )
            if recorded_token_count != token_count:
                raise SplitValidationError(
                    f"{prefix}.metadata.token_length={recorded_token_count} "
                    f"!= len(tokens) ({token_count})"
                )
        if "prompt_token_length" in metadata:
            prompt_token_count = _nonnegative_integer(
                metadata["prompt_token_length"],
                context=f"{prefix}.metadata.prompt_token_length",
            )
            if prompt_token_count + response_length != token_count:
                raise SplitValidationError(
                    f"{prefix}.metadata.prompt_token_length ({prompt_token_count}) + "
                    f"response_length ({response_length}) != len(tokens) ({token_count})"
                )

    for field in _RESPONSE_ALIGNED_SAMPLE_FIELDS:
        value = sample.get(field)
        if value is None:
            continue
        actual = _sequence_length(value, context=f"{prefix}.{field}")
        if actual != response_length:
            raise SplitValidationError(
                f"{prefix}.{field} length {actual} != response_length {response_length}"
            )

    expected_token_aligned = max(token_count - 1, 0)
    for field in _TOKEN_MINUS_ONE_ALIGNED_SAMPLE_FIELDS:
        value = sample.get(field)
        if value is None:
            continue
        try:
            actual = len(value)
        except TypeError as exc:
            raise SplitValidationError(f"{prefix}.{field} must be a sequence") from exc
        if actual != expected_token_aligned:
            raise SplitValidationError(
                f"{prefix}.{field} length {actual} != len(tokens) - 1 "
                f"({expected_token_aligned})"
            )

    return Counts(1, supervised_tokens, token_count)


def _load_and_validate_payload(path: Path, *, rollout_id: int) -> tuple[dict[str, Any], list[Any], list[Counts]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise SplitValidationError(f"failed to load {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SplitValidationError(f"{path} must contain a dictionary payload")
    stored_rollout_id = payload.get("rollout_id")
    if (
        isinstance(stored_rollout_id, bool)
        or not isinstance(stored_rollout_id, numbers.Integral)
        or int(stored_rollout_id) != rollout_id
    ):
        raise SplitValidationError(
            f"{path} rollout_id={stored_rollout_id!r}, expected integer {rollout_id}"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise SplitValidationError(f"{path} must contain a top-level samples list")
    if not samples:
        raise SplitValidationError(f"{path} contains an empty samples list")
    if len(samples) % NUM_ISLANDS:
        raise SplitValidationError(
            f"{path} has {len(samples)} samples; an original DP5 bucket must be divisible by {NUM_ISLANDS}"
        )
    counts = [
        _sample_counts(sample, rollout_id=rollout_id, position=position)
        for position, sample in enumerate(samples)
    ]
    return payload, samples, counts


def _payload_for_island(payload: dict[str, Any], samples: list[Any], island_id: int) -> dict[str, Any]:
    # A shallow copy preserves every top-level non-sample value.  The sample
    # objects are selected, not copied or normalized, so all nested fields are
    # handed directly to torch.save without semantic transformation.
    result = payload.copy()
    result["samples"] = samples[island_id::NUM_ISLANDS]
    return result


def _empty_split_counts() -> dict[str, Counts]:
    return {"train": Counts(), "validation": Counts(), "total": Counts()}


def _add_counts(target: dict[str, Counts], split: str, counts: Counts) -> None:
    target[split] = target[split] + counts
    target["total"] = target["total"] + counts


def _counts_dict(counts: dict[str, Counts]) -> dict[str, dict[str, int]]:
    return {
        "trajectory_counts": {name: value.trajectories for name, value in counts.items()},
        "supervised_token_counts": {
            name: value.supervised_tokens for name, value in counts.items()
        },
        "total_token_counts": {name: value.total_tokens for name, value in counts.items()},
    }


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _publish(staging: Path, output: Path, *, overwrite: bool) -> None:
    if not _path_exists(output):
        os.replace(staging, output)
        return
    if not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output {output}")

    backup = output.with_name(f".{output.name}.replaced-{os.getpid()}")
    if _path_exists(backup):
        raise FileExistsError(f"refusing to replace output while backup path exists: {backup}")
    os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        os.replace(backup, output)
        raise
    try:
        _remove_path(backup)
    except OSError as exc:
        warnings.warn(f"published {output}, but could not remove old output backup {backup}: {exc}")


def _validate_roots(source: Path, output: Path, *, overwrite: bool) -> tuple[Path, Path]:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_dir():
        raise SplitValidationError(f"source is not a directory: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise SplitValidationError("source and output trees must be disjoint")
    if _path_exists(output) and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    return source, output


def split_dataset(
    source: Path,
    output: Path,
    *,
    train_rollout_ids: Iterable[int] = TRAIN_ROLLOUT_IDS,
    validation_rollout_ids: Iterable[int] = VALIDATION_ROLLOUT_IDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split the requested rollout IDs and atomically publish a new tree."""
    train_ids = _rollout_ids(train_rollout_ids, split="train")
    validation_ids = _rollout_ids(validation_rollout_ids, split="validation")
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        raise SplitValidationError(f"train and validation rollout IDs overlap: {overlap}")
    source, output = _validate_roots(Path(source), Path(output), overwrite=overwrite)

    requested = [(rollout_id, "train") for rollout_id in train_ids]
    requested += [(rollout_id, "validation") for rollout_id in validation_ids]
    missing = [
        source / SOURCE_FILE_TEMPLATE.format(rollout_id=rollout_id)
        for rollout_id, _ in requested
        if not (source / SOURCE_FILE_TEMPLATE.format(rollout_id=rollout_id)).is_file()
    ]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise SplitValidationError(f"missing source rollout files: {preview}{suffix}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    source_hashes: dict[str, str] = {}
    rollout_mapping: list[dict[str, Any]] = []
    island_counts = [_empty_split_counts() for _ in range(NUM_ISLANDS)]
    source_counts = _empty_split_counts()

    try:
        for island_id in range(NUM_ISLANDS):
            (staging / f"island_{island_id}").mkdir()

        for rollout_id, split in requested:
            source_name = SOURCE_FILE_TEMPLATE.format(rollout_id=rollout_id)
            source_path = source / source_name
            source_hash = _sha256(source_path)
            source_hashes[source_name] = source_hash
            payload, samples, sample_counts = _load_and_validate_payload(
                source_path, rollout_id=rollout_id
            )

            combined_counts = Counts()
            islands_for_rollout: list[dict[str, Any]] = []
            for island_id in range(NUM_ISLANDS):
                positions = list(range(island_id, len(samples), NUM_ISLANDS))
                counts = Counts()
                for position in positions:
                    counts = counts + sample_counts[position]
                combined_counts = combined_counts + counts
                _add_counts(island_counts[island_id], split, counts)

                relative_output = Path(f"island_{island_id}") / source_name
                torch.save(
                    _payload_for_island(payload, samples, island_id),
                    staging / relative_output,
                )
                islands_for_rollout.append(
                    {
                        "island_id": island_id,
                        "output_file": relative_output.as_posix(),
                        "source_sample_positions": positions,
                        **counts.as_dict(),
                    }
                )

            source_rollout_counts = Counts()
            for counts in sample_counts:
                source_rollout_counts = source_rollout_counts + counts
            if combined_counts != source_rollout_counts:
                raise AssertionError(f"rollout {rollout_id} split counts do not reconstruct the source")
            _add_counts(source_counts, split, source_rollout_counts)
            rollout_mapping.append(
                {
                    "rollout_id": rollout_id,
                    "split": split,
                    "source_file": source_name,
                    "source_sha256": source_hash,
                    "source_file_size_bytes": source_path.stat().st_size,
                    **source_rollout_counts.as_dict(),
                    "islands": islands_for_rollout,
                }
            )

        combined_islands = _empty_split_counts()
        for split in ("train", "validation", "total"):
            value = Counts()
            for counts in island_counts:
                value = value + counts[split]
            combined_islands[split] = value
        if combined_islands != source_counts:
            raise AssertionError("combined island counts do not reconstruct source counts")

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "strategy": "source_sample_position_modulo_5",
            "source": {
                "root": str(source),
                "file_template": SOURCE_FILE_TEMPLATE,
                "hash_algorithm": "sha256",
            },
            "num_islands": NUM_ISLANDS,
            "nominal_global_batch_size_per_island": NOMINAL_GLOBAL_BATCH_SIZE,
            "dynamic_global_batch_size_required": True,
            "train_rollout_ids": list(train_ids),
            "validation_rollout_ids": list(validation_ids),
            "source_hashes": source_hashes,
            "source_counts": _counts_dict(source_counts),
            "islands": [
                {
                    "island_id": island_id,
                    "directory": f"island_{island_id}",
                    "nominal_global_batch_size": NOMINAL_GLOBAL_BATCH_SIZE,
                    **_counts_dict(island_counts[island_id]),
                }
                for island_id in range(NUM_ISLANDS)
            ],
            "rollout_mapping": rollout_mapping,
        }
        with (staging / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        _publish(staging, output, overwrite=overwrite)
        return manifest
    except BaseException:
        if _path_exists(staging):
            _remove_path(staging)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="directory containing data_<id>.pt")
    parser.add_argument("--output", type=Path, required=True, help="new five-island output tree")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace an existing output tree after the new tree is complete",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = split_dataset(args.source, args.output, overwrite=args.overwrite)
    except (FileExistsError, SplitValidationError, OSError) as exc:
        print(f"split_miles_value_islands: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
