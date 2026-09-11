"""Attest that an immutable v4 dataset remains valid with a runtime-only patch.

This is a deliberately narrow compatibility bridge.  It permits reuse only when
the complete published dataset and every conversion dependency are byte-identical
to the code that built it.  The only accepted code delta is the reviewed training
guard/orchestration update listed below.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid

SCHEMA = "qwen38-native-gap-v4-dataset-runtime-compatibility/v1"
CODE_SCHEMA = "qwen38-native-gap-v4-runtime-code/v1"
BUILD_SCHEMA = "qwen38-native-gap-v4-build-publication/v1"
IMAGE = "sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee"
SELF = "monitoring/attest_masked_native_gap_v4_dataset_runtime_compatibility_20260911.py"
CP_PROBE_PATH = Path("/data/sft_baseline_20260908/plans/"
    "cot-masked-native-gap-v4-cp-guard-regression-20260911T032451Z/host-receipt.json")
CP_PROBE_SHA256 = "7350499e8991043c9dc48d033118eeb7c3b4dcab811fe675272edfb30e58780b"
CP_PROBE_SCHEMA = "qwen38-native-gap-v4-aux-cp-probe-host-receipt/v1"
CP_NESTED_SHA256 = "6a21497a7842d515d2292e26c97c2679c0481b4e0864ec90d365c6517536b465"
CP_NESTED_SCHEMA = "qwen38-native-gap-v4-aux-cp-synthetic-regression/v1"

# This one-off bridge is intentionally exact.  Any other added, removed, or
# modified production member requires a new dataset build or a newly reviewed
# attestation contract.
EXPECTED_ADDED = frozenset({SELF})
EXPECTED_REMOVED = frozenset()
EXPECTED_MODIFIED = frozenset({
    "monitoring/launch_cot_masked_native_gap_v4_n3_20260910.py",
    "monitoring/prepare_cot_masked_native_gap_v4_n3_20260910.py",
    "monitoring/stage_masked_native_gap_v4_20260910.py",
    "training/qwen38_native_gap_v3/masked_recipe.py",
    "training/qwen38_native_gap_v3/masked_train.py",
})


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _safe(name):
    value = PurePosixPath(name) if isinstance(name, str) else None
    return (value is not None and bool(name) and "\\" not in name
            and not value.is_absolute() and str(value) == name
            and all(part not in {"", ".", ".."} for part in value.parts))


def _strict_json(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " must be a regular file")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(label + " contains a duplicate key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(label + " contains a non-finite number: " + value)

    try:
        return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                          parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(label + " is not strict JSON") from error


def _canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _declared_conversion_dependencies(code_root):
    """Read the converter's literal dependency tuple without importing ML packages."""
    path = Path(code_root) / "training/qwen38_native_gap_v3/prepare_masked_full.py"
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ValueError("Cannot read the converter dependency declaration") from error
    values = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name)
                   and target.id == "CONVERSION_DEPENDENCIES" for target in targets):
                values.append(ast.literal_eval(node.value))
    if (len(values) != 1 or not isinstance(values[0], tuple) or not values[0]
            or any(not _safe(name) for name in values[0])
            or len(values[0]) != len(set(values[0]))):
        raise ValueError("Converter dependency declaration is not one safe literal tuple")
    return values[0]


def _manifest_records(root, expected_sha256, label):
    root = Path(root)
    manifest = root / "code-manifest.json"
    if (not root.is_absolute() or root.is_symlink() or not root.is_dir()
            or root.resolve(strict=True) != root or root.stat().st_mode & 0o222
            or manifest.is_symlink() or not manifest.is_file()
            or manifest.stat().st_mode & 0o222 or digest(manifest) != expected_sha256):
        raise ValueError(label + " code tree or manifest changed")
    raw = _strict_json(manifest, label + " code manifest")
    rows = raw.get("files")
    if raw.get("schema") != CODE_SCHEMA or not isinstance(rows, list) or not rows:
        raise ValueError(label + " has the wrong code-manifest contract")
    records, folded = {}, set()
    for row in rows:
        name = row.get("path") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}
                or not _safe(name) or name.casefold() in folded
                or type(row.get("bytes")) is not int or row["bytes"] < 1
                or not _hash(row.get("sha256"))):
            raise ValueError(label + " code manifest has an unsafe member")
        records[name] = row
        folded.add(name.casefold())
    files = []
    for member in sorted(root.rglob("*")):
        mode = member.lstat().st_mode
        if (member.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                or mode & 0o222):
            raise ValueError(label + " code tree has a mutable or indirect member")
        if stat.S_ISREG(mode):
            files.append(member)
    actual = {str(member.relative_to(root)) for member in files}
    if actual != set(records) | {"code-manifest.json"}:
        raise ValueError(label + " code-tree membership differs from its manifest")
    for name, row in records.items():
        member = root / name
        if member.stat().st_size != row["bytes"] or digest(member) != row["sha256"]:
            raise ValueError(label + " code member changed: " + name)
    return records


def _dataset_publication(dataset, build_receipt, build_receipt_sha256,
                         old_code_manifest_sha256):
    dataset, build_receipt = Path(dataset), Path(build_receipt)
    expected_receipt = dataset.with_name(dataset.name + ".build-receipt.json")
    if (not dataset.is_absolute() or dataset.is_symlink() or not dataset.is_dir()
            or dataset.resolve(strict=True) != dataset or dataset.stat().st_mode & 0o222
            or build_receipt != expected_receipt or build_receipt.is_symlink()
            or not build_receipt.is_file() or build_receipt.stat().st_mode & 0o222
            or digest(build_receipt) != build_receipt_sha256):
        raise ValueError("Immutable dataset publication or receipt changed")
    receipt = _strict_json(build_receipt, "dataset build receipt")
    rows = receipt.get("tree_inventory")
    if (receipt.get("schema") != BUILD_SCHEMA or receipt.get("status") != "complete"
            or receipt.get("mode") != "final" or receipt.get("output") != str(dataset)
            or receipt.get("runtime_image") != IMAGE
            or receipt.get("code_manifest_sha256") != old_code_manifest_sha256
            or receipt.get("immutable_mode") != "files0444_dirs0555"
            or receipt.get("atomic_direct_dataset_publication") is not True
            or not isinstance(rows, list) or not rows):
        raise ValueError("Dataset build receipt has the wrong contract")
    expected, folded = {}, set()
    for row in rows:
        name = row.get("path") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}
                or not _safe(name) or name.casefold() in folded
                or type(row.get("bytes")) is not int or row["bytes"] < 1
                or not _hash(row.get("sha256"))):
            raise ValueError("Dataset receipt has an unsafe inventory member")
        expected[name] = row
        folded.add(name.casefold())
    files = []
    for member in sorted(dataset.rglob("*")):
        mode = member.lstat().st_mode
        if (member.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                or mode & 0o222):
            raise ValueError("Dataset tree contains a mutable or indirect member")
        if stat.S_ISREG(mode):
            files.append(member)
    if {str(member.relative_to(dataset)) for member in files} != set(expected):
        raise ValueError("Dataset tree membership differs from its build receipt")
    for name, row in expected.items():
        member = dataset / name
        if member.stat().st_size != row["bytes"] or digest(member) != row["sha256"]:
            raise ValueError("Dataset member differs from build receipt: " + name)
    if (receipt.get("files") != len(rows)
            or receipt.get("bytes") != sum(row["bytes"] for row in rows)
            or receipt.get("tree_sha256") != _canonical_sha(rows)):
        raise ValueError("Dataset build-receipt aggregate changed")
    core = {}
    for name in ("manifest.json", "index.json", "COMPLETE.json"):
        if name not in expected:
            raise ValueError("Dataset build receipt omits required core member")
        core[name] = expected[name]["sha256"]
    return receipt, core


def _cp_guard_probe(path, expected_sha256, new_code):
    path = Path(path)
    if (path != CP_PROBE_PATH or expected_sha256 != CP_PROBE_SHA256
            or path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222
            or path.parent.is_symlink() or path.parent.stat().st_mode & 0o222
            or digest(path) != expected_sha256):
        raise ValueError("Exact immutable eight-rank CP guard probe is required")
    host = _strict_json(path, "CP guard host receipt")
    nested_path = path.parent / "probe-output/receipt.json"
    nested = _strict_json(nested_path, "CP guard nested receipt")
    assertions = nested.get("assertions")
    ranks = nested.get("rank_records")
    if (host.get("schema") != CP_PROBE_SCHEMA or host.get("status") != "passed"
            or host.get("runtime_image_id") != IMAGE or host.get("world_size") != 8
            or host.get("backend") != "nccl" or host.get("docker_exit_code") != 0
            or host.get("dataset_accessed") is not False
            or host.get("model_loaded") is not False
            or host.get("gpu_compute_processes_after") != []
            or host.get("runner_inactive_after") is not True
            or host.get("transient_container_present_after_run") is not False
            or host.get("patched_masked_train_sha256") !=
               new_code.get("training/qwen38_native_gap_v3/masked_train.py", {}).get("sha256")
            or host.get("probe_receipt_path") != str(nested_path)
            or host.get("probe_receipt_sha256") != CP_NESTED_SHA256
            or host.get("probe_source_sha256") != nested.get("probe_sha256")
            or host.get("probe_receipt") != nested
            or nested_path.is_symlink() or not nested_path.is_file()
            or nested_path.stat().st_mode & 0o222 or nested_path.parent.stat().st_mode & 0o222
            or digest(nested_path) != CP_NESTED_SHA256
            or nested.get("schema") != CP_NESTED_SCHEMA or nested.get("status") != "passed"
            or nested.get("runtime_image") != IMAGE or nested.get("world_size") != 8
            or nested.get("cp_size") != 8 or nested.get("backend") != "nccl"
            or nested.get("patched_guard_sha256") != host.get("patched_masked_train_sha256")
            or nested.get("primary_contract") != "full-input-then-in-forward-round-robin"
            or nested.get("aux_contract") != "deferred-context-entry-round-robin"
            or not isinstance(assertions, dict) or not assertions
            or any(value is not True for value in assertions.values())
            or not isinstance(ranks, list) or len(ranks) != 8
            or {row.get("rank") for row in ranks if isinstance(row, dict)} != set(range(8))):
        raise ValueError("Eight-rank CP guard probe has the wrong contract")
    return host, nested


def _evaluate(*, dataset, build_receipt, build_receipt_sha256, old_code_root,
              old_code_manifest_sha256, new_code_root,
              new_code_manifest_sha256, cp_guard_probe,
              cp_guard_probe_sha256):
    if (not _hash(build_receipt_sha256) or not _hash(old_code_manifest_sha256)
            or not _hash(new_code_manifest_sha256)
            or old_code_manifest_sha256 == new_code_manifest_sha256):
        raise ValueError("Compatibility bridge requires distinct exact code identities")
    old = _manifest_records(old_code_root, old_code_manifest_sha256, "old")
    new = _manifest_records(new_code_root, new_code_manifest_sha256, "new")
    cp_host, cp_nested = _cp_guard_probe(cp_guard_probe, cp_guard_probe_sha256, new)
    receipt, core = _dataset_publication(dataset, build_receipt,
        build_receipt_sha256, old_code_manifest_sha256)
    manifest = _strict_json(Path(dataset) / "manifest.json", "dataset manifest")
    dependencies = manifest.get("conversion_dependency_sha256")
    declared_dependencies = _declared_conversion_dependencies(new_code_root)
    if (not isinstance(dependencies, dict) or not dependencies
            or set(dependencies) != set(declared_dependencies)):
        raise ValueError("Dataset manifest lacks conversion dependency identities")
    folded = set()
    for name, expected in dependencies.items():
        if (not _safe(name) or name.casefold() in folded or not _hash(expected)
                or name not in old or name not in new
                or old[name]["sha256"] != expected or new[name]["sha256"] != expected):
            raise ValueError("A dataset conversion dependency changed: " + str(name))
        folded.add(name.casefold())
    preparation = "training/qwen38_native_gap_v3/prepare_masked_full.py"
    if manifest.get("preparation_sha256") != dependencies.get(preparation):
        raise ValueError("Dataset preparation identity is not conversion-bound")
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    modified = sorted(name for name in set(old) & set(new)
                      if old[name] != new[name])
    if (set(added) != EXPECTED_ADDED or set(removed) != EXPECTED_REMOVED
            or set(modified) != EXPECTED_MODIFIED):
        raise ValueError("Code delta is not the exact reviewed runtime-only patch")
    changed = set(added) | set(removed) | set(modified)
    if changed & set(dependencies):
        raise ValueError("A reviewed runtime-only change overlaps conversion code")
    if new.get(SELF, {}).get("sha256") != digest(__file__):
        raise ValueError("Attestation helper is not bound into the new runtime")
    return {
        "schema": SCHEMA,
        "status": "passed",
        "runtime_image": IMAGE,
        "dataset": str(Path(dataset)),
        "dataset_build_receipt": str(Path(build_receipt)),
        "dataset_build_receipt_sha256": build_receipt_sha256,
        "dataset_tree_sha256": receipt["tree_sha256"],
        "dataset_files": receipt["files"],
        "dataset_bytes": receipt["bytes"],
        "dataset_manifest_sha256": core["manifest.json"],
        "dataset_index_sha256": core["index.json"],
        "dataset_complete_sha256": core["COMPLETE.json"],
        "old_code_root": str(Path(old_code_root)),
        "old_code_manifest_sha256": old_code_manifest_sha256,
        "new_code_root": str(Path(new_code_root)),
        "new_code_manifest_sha256": new_code_manifest_sha256,
        "conversion_dependency_count": len(dependencies),
        "conversion_dependency_sha256": dependencies,
        "conversion_dependency_map_sha256": _canonical_sha(dependencies),
        "added_runtime_files": added,
        "removed_runtime_files": removed,
        "modified_runtime_files": modified,
        "exact_runtime_only_delta": True,
        "full_dataset_tree_rehashed": True,
        "old_code_tree_rehashed": True,
        "new_code_tree_rehashed": True,
        "attestation_helper_sha256": digest(__file__),
        "cp_guard_probe_path": str(Path(cp_guard_probe)),
        "cp_guard_probe_sha256": cp_guard_probe_sha256,
        "cp_guard_probe_schema": cp_host["schema"],
        "cp_guard_nested_receipt_path": cp_host["probe_receipt_path"],
        "cp_guard_nested_receipt_sha256": cp_host["probe_receipt_sha256"],
        "cp_guard_nested_receipt_schema": cp_nested["schema"],
        "cp_guard_probe_source_sha256": cp_host["probe_source_sha256"],
    }


def verify_receipt(*, receipt_path, receipt_sha256, dataset, build_receipt,
                   build_receipt_sha256, new_code_root,
                   new_code_manifest_sha256, cp_guard_probe,
                   cp_guard_probe_sha256):
    path = Path(receipt_path)
    if (path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222
            or digest(path) != receipt_sha256):
        raise ValueError("Runtime compatibility receipt changed or is mutable")
    record = _strict_json(path, "runtime compatibility receipt")
    old_root = Path(record.get("old_code_root", ""))
    expected = _evaluate(dataset=dataset, build_receipt=build_receipt,
        build_receipt_sha256=build_receipt_sha256, old_code_root=old_root,
        old_code_manifest_sha256=record.get("old_code_manifest_sha256"),
        new_code_root=new_code_root,
        new_code_manifest_sha256=new_code_manifest_sha256,
        cp_guard_probe=cp_guard_probe,
        cp_guard_probe_sha256=cp_guard_probe_sha256)
    if record != expected:
        raise ValueError("Runtime compatibility receipt claims differ from re-attestation")
    return record


def attest(*, dataset, build_receipt, build_receipt_sha256, old_code_root,
           old_code_manifest_sha256, new_code_root,
           new_code_manifest_sha256, cp_guard_probe,
           cp_guard_probe_sha256, output):
    if os.environ.get("YETA_TRAINING_IMAGE") != IMAGE:
        raise ValueError("Compatibility attestation requires the exact pinned image")
    output = Path(output)
    if (not output.is_absolute() or output.exists() or output.is_symlink()
            or not output.name.endswith(".json")):
        raise ValueError("Use a fresh absolute compatibility-receipt path")
    record = _evaluate(dataset=dataset, build_receipt=build_receipt,
        build_receipt_sha256=build_receipt_sha256, old_code_root=old_code_root,
        old_code_manifest_sha256=old_code_manifest_sha256,
        new_code_root=new_code_root,
        new_code_manifest_sha256=new_code_manifest_sha256,
        cp_guard_probe=cp_guard_probe,
        cp_guard_probe_sha256=cp_guard_probe_sha256)
    raw = (json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    incoming = output.with_name("." + output.name + ".incoming-" + uuid.uuid4().hex)
    with incoming.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    incoming.chmod(0o400)
    descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.link(incoming, output)
        incoming.unlink()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    verify_receipt(receipt_path=output, receipt_sha256=digest(output),
        dataset=dataset, build_receipt=build_receipt,
        build_receipt_sha256=build_receipt_sha256,
        new_code_root=new_code_root,
        new_code_manifest_sha256=new_code_manifest_sha256,
        cp_guard_probe=cp_guard_probe,
        cp_guard_probe_sha256=cp_guard_probe_sha256)
    return {**record, "receipt": str(output), "receipt_sha256": digest(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--build-receipt", required=True)
    parser.add_argument("--build-receipt-sha256", required=True)
    parser.add_argument("--old-code-root", required=True)
    parser.add_argument("--old-code-manifest-sha256", required=True)
    parser.add_argument("--new-code-root", required=True)
    parser.add_argument("--new-code-manifest-sha256", required=True)
    parser.add_argument("--cp-guard-probe", required=True)
    parser.add_argument("--cp-guard-probe-sha256", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(attest(**vars(parser.parse_args())), sort_keys=True))
