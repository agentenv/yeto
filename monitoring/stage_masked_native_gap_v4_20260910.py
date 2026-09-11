"""Safely publish an exact, immutable v4 runtime archive on one host."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import uuid

SCHEMA = "qwen38-native-gap-v4-runtime-code/v1"
REQUIRED = {
    "monitoring/attest_masked_native_gap_v4_dataset_runtime_compatibility_20260911.py",
    "monitoring/build_cot_masked_native_gap_v4_n3_20260911.py",
    "monitoring/freeze_masked_native_gap_v4_runtime_20260910.py",
    "monitoring/launch_cot_masked_native_gap_v4_n3_20260910.py",
    "monitoring/prepare_cot_masked_native_gap_v4_n3_20260910.py",
    "monitoring/promote_masked_native_gap_v4_exclusions_20260911.py",
    "monitoring/stage_masked_native_gap_v4_20260910.py",
    "monitoring/transfer_cot_masked_native_gap_v4_20260910.py",
    "monitoring/stage_wandb_auth_v4_20260911.py",
    "monitoring/verify_cot_masked_native_gap_v4_wandb_20260911.py",
    "training/__init__.py",
    "training/qwen38_native_gap_v3/__init__.py",
    "training/qwen38_native_gap_v3/masked.py",
    "training/qwen38_native_gap_v3/masked_data.py",
    "training/qwen38_native_gap_v3/masked_recipe.py",
    "training/qwen38_native_gap_v3/masked_replay.py",
    "training/qwen38_native_gap_v3/masked_train.py",
    "training/qwen38_native_gap_v3/normalize.py",
    "training/qwen38_native_gap_v3/prepare_masked_full.py",
    "training/qwen38_native_gap_v3/source_adapters.py",
    "training/qwen38_turn_boundary_v2/masked.py",
    "training/qwen38_turn_boundary_v2/normalize.py",
    "training/qwen38_turn_boundary_v2/source_adapters.py",
    "training/qwen38_turn_boundary_v2/__init__.py",
    "training/qwen38_cot_experimental/__init__.py",
    "training/qwen38_cot_experimental/train.py",
    "training/qwen38_cot_experimental/recipe.py",
    "training/qwen38_cot_experimental/data.py",
    "training/qwen38_cot_experimental/prepare_full.py",
    "training/qwen38_cot_experimental/replay.py",
    "training/qwen38_cot_experimental/generation_contract.py",
    "training/qwen38_cot_experimental/render.py",
    "training/qwen38_cot_masked/__init__.py",
    "training/qwen38_cot_masked/render.py",
    "training/qwen38_no_cot/__init__.py",
    "training/qwen38_no_cot/nemo_train.py",
    "training/qwen38_no_cot/nemo_data.py",
    "training/qwen38_no_cot/nemo_recipe.py",
    "training/qwen38_no_cot/nemo_checkpoint.py",
    "training/qwen38_no_cot/nemo_checkpoint_probe.py",
    "training/qwen38_no_cot/nemo_model_probe.py",
    "training/qwen38_no_cot/nemo_cp_memory.py",
    "training/qwen38_no_cot/nemo_block_checkpoint.py",
    "training/qwen38_no_cot/prepare_data.py",
    "training/qwen38_no_cot/render.py",
    "training/qwen38_no_cot/assets/chat_template.jinja",
    "training/qwen38_no_cot/assets/tokenizer.json",
    "training/qwen38_no_cot/assets/tokenizer_config.json",
    "training/qwen38_no_cot/upstream_manager/__init__.py",
    "training/qwen38_no_cot/upstream_manager/render.py",
    "cot_filler/__init__.py",
    "cot_filler/core.py",
    "cot_filler/corpus_worker.py",
    "cot_filler/corpus_source.py",
    "cot_filler/provider.py",
    "cot_filler/regeneration.py",
    "cot_filler/grounding_review.py",
    "cot_filler/grounding_review_ids.py",
    "cot_filler/grounding_review_v3.py",
    "cot_filler/grounding_review_v4.py",
    "cot_filler/grounding_review_fast.py",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(name):
    value = PurePosixPath(name)
    return (isinstance(name, str) and name and "\\" not in name
            and not value.is_absolute() and value.name not in {"", ".", ".."}
            and all(part not in {"", ".", ".."} for part in value.parts)
            and str(value) == name)


def stage(archive, archive_sha256, manifest_sha256, output):
    archive, output = Path(archive), Path(output)
    if sha(archive) != archive_sha256 or output.exists() or output.is_symlink():
        raise ValueError("Archive identity or fresh output requirement failed")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if (not names or any(not member.isfile() or not _safe(member.name) for member in members)
                or len(names) != len(set(names))
                or len({name.casefold() for name in names}) != len(names)
                or names.count("code-manifest.json") != 1):
            raise ValueError("Archive contains unsafe, colliding or non-regular members")
        raw = bundle.extractfile("code-manifest.json").read()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError("Code manifest changed")
        manifest = json.loads(raw)
        records = manifest.get("files")
        if (manifest.get("schema") != SCHEMA or not isinstance(records, list) or not records):
            raise ValueError("Wrong v4 code manifest")
        expected, folded = {}, set()
        for record in records:
            if (not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}
                    or not _safe(record.get("path")) or record["path"].casefold() in folded
                    or type(record.get("bytes")) is not int or record["bytes"] < 1
                    or not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64):
                raise ValueError("Invalid code manifest member")
            expected[record["path"]] = record
            folded.add(record["path"].casefold())
        if set(expected) != REQUIRED or set(names) != REQUIRED | {"code-manifest.json"}:
            raise ValueError("Archive and required runtime closure differ from its manifest")
        payload = {}
        for name, record in expected.items():
            contents = bundle.extractfile(name).read()
            if len(contents) != record["bytes"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
                raise ValueError("Archived code member changed: " + name)
            if name.endswith(".py"):
                compile(contents, name, "exec")
            payload[name] = contents
    incoming = output.with_name("." + output.name + ".incoming-" + uuid.uuid4().hex)
    incoming.mkdir(mode=0o700, parents=True)
    try:
        for name, contents in {**payload, "code-manifest.json": raw}.items():
            target = incoming / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(contents); stream.flush(); os.fsync(stream.fileno())
            target.chmod(0o444)
        for directory in sorted((path for path in incoming.rglob("*") if path.is_dir()),
                                key=lambda path: len(path.parts), reverse=True):
            directory.chmod(0o555)
        incoming.chmod(0o555)
        descriptor = os.open(incoming.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
            incoming.rename(output)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        # Leave a non-admitted incoming tree for explicit inspection; never
        # publish a partial runtime at the requested output path.
        raise
    receipt = {
        "schema": "qwen38-native-gap-v4-runtime-stage/v1",
        "status": "passed",
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "archive_sha256": archive_sha256,
        "code_manifest_sha256": manifest_sha256,
        "output": str(output.resolve(strict=True)),
        "files": len(expected),
        "byte_exact": True,
        "regular_files_only": True,
        "read_only": True,
        "atomic_publish": True,
    }
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(stage(**vars(parser.parse_args())), sort_keys=True))
