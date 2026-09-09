"""Full-corpus mask audit and immutable exact-token loader for corrected no-CoT."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from training.qwen38_no_cot.nemo_data import shift_next_token_row

SCHEMA = "qualified-turn-normalized-token-index/v2"
CONTRACT = "qwen38-no-cot-source-turns/v2"
ASSISTANT_PREFIX = [248045, 74455, 198, 248068, 271, 248069, 271]
IM_START, IM_END = 248045, 248046
ROLE_IDS = {74455, 846, 8678, 13766}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_identity(identity):
    from .render import normalizer_identity
    from .normalize import VERSION
    from training.qwen38_no_cot.upstream_manager.render import TOKENIZER_SHA256, OFFICIAL_TEMPLATE_SHA256
    from training.qwen38_no_cot import render as prefix
    from training.qwen38_no_cot.upstream_manager import render as native
    expected = {"turn_boundary_contract": CONTRACT, "tokenizer_sha256": TOKENIZER_SHA256,
                "official_template_sha256": OFFICIAL_TEMPLATE_SHA256,
                "prefix_adapter_sha256": digest_file(prefix.__file__), "renderer_sha256": digest_file(native.__file__),
                "mask_policy": prefix.MASK_POLICY, "sequence_policy": prefix.SEQUENCE_POLICY,
                "max_sequence_length": 262144, "enable_thinking": False, **normalizer_identity()}
    if not isinstance(identity, dict) or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Corrected renderer/tokenizer/mask identity differs from this implementation")
    return VERSION


def validate_row(row, identity):
    ids, labels = row["input_ids"], row["labels"]
    shifted = shift_next_token_row(row)
    metadata = row.get("metadata", {})
    audit = metadata.get("turn_boundary_audit", {})
    sequence = metadata.get("sequence_audit", {})
    if (metadata.get("renderer_identity") != identity or not isinstance(row.get("group_id"), str)
            or not row["group_id"] or row.get("source") not in ("trace", "replay")
            or audit.get("excluded") is not False or audit.get("reasoning_policy") != "drop"
            or audit.get("normalizer_source_sha256") != identity["turn_normalizer_sha256"]
            or sequence.get("causal_shift_applied") is not False or sequence.get("eot_appended") is not False
            or sequence.get("retained_input_tokens") != len(ids)
            or metadata.get("behavioral_recovery_validated") is not False):
        raise ValueError("Corrected row is missing consistent source-turn and loss provenance")
    starts = [i for i, token in enumerate(ids) if token == IM_START]
    if not starts or starts[0] != 0:
        raise ValueError("Expected native message headers")
    expected = [-100] * len(ids)
    for ordinal, start in enumerate(starts):
        stop = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(ids)
        if start + 2 >= stop:
            if ordinal != len(starts)-1 or not sequence.get("truncated"):
                raise ValueError("Incomplete native header before sequence cutoff")
            continue
        if ids[start+1] not in ROLE_IDS or ids[start+2] != 198:
            raise ValueError("Unrecognized native role header")
        try:
            end = ids.index(IM_END, start+3, stop)
        except ValueError:
            end = None
            if ordinal != len(starts)-1 or not sequence.get("truncated"):
                raise ValueError("An incomplete native message precedes another message")
        if ids[start+1] != 74455:
            continue
        available = min(len(ASSISTANT_PREFIX), stop-start)
        if ids[start:start+available] != ASSISTANT_PREFIX[:available]:
            raise ValueError("Assistant header/empty-think prefix differs from no-CoT contract")
        body = start + len(ASSISTANT_PREFIX)
        if end is not None and end <= body:
            raise ValueError("Empty assistant would train only an end token")
        bound = end+1 if end is not None else stop
        expected[body:bound] = ids[body:bound]
    if labels != expected:
        raise ValueError("Saved labels do not match assistant body/tool/EOT-only supervision")
    return shifted


def _qualify_shard(task):
    """Run the unchanged complete byte/row/mask checks for one closed shard."""
    directory, split, entry, shard, identity = task
    path = (Path(directory) / entry["path"]).resolve()
    if (shard["path"] != entry["path"] or shard["sha256"] != entry["sha256"]
            or digest_file(path) != entry["sha256"] or path.stat().st_size != shard["size_bytes"]
            or len(shard["rows"]) != entry["rows"]):
        raise ValueError("Corrected shard/index bytes changed")
    groups, normalized, source_ids, retained = set(), set(), set(), set()
    position, rows, targets = 0, 0, 0
    with path.open("rb") as stream:
        for offset, length, checksum, tokens in shard["rows"]:
            if offset != position or type(length) is not int or length <= 0:
                raise ValueError("Index does not cover the shard in exact byte order")
            raw = stream.read(length); position += length
            if hashlib.sha256(raw).hexdigest() != checksum:
                raise ValueError("Actual row differs from its recorded hash")
            row = json.loads(raw)
            if len(row["input_ids"]) != tokens:
                raise ValueError("Index token length differs from actual row")
            validate_row(row, identity)
            meta = row["metadata"]
            provenance = meta["provenance"]
            if ("original_baseline_group_id" in provenance
                    and provenance["original_baseline_group_id"] != row["group_id"]):
                raise ValueError("Original baseline source session group changed")
            if "original_baseline_split" in provenance:
                from training.qwen38_no_cot.prepare_data import SEED
                expected_split = ("validation" if int(hashlib.sha256(
                    (SEED + ":split:" + row["group_id"]).encode()).hexdigest()[:8], 16) % 100 == 0 else "train")
                if provenance["original_baseline_split"] != split or expected_split != split:
                    raise ValueError("Original baseline source split changed")
            normalized_hash = provenance["messages_sha256"]
            source_id = (row["source"], provenance["sha256"])
            retained_hash = hashlib.sha256(json.dumps([row["input_ids"], row["labels"]], separators=(",", ":")).encode()).hexdigest()
            if meta.get("retained_training_arrays_sha256") != retained_hash:
                raise ValueError("Retained training-array identity is missing or incorrect")
            if normalized_hash in normalized or source_id in source_ids or retained_hash in retained:
                raise ValueError("Duplicate normalized sequence, source or retained training arrays in corrected corpus")
            normalized.add(normalized_hash); source_ids.add(source_id); retained.add(retained_hash)
            groups.add(row["group_id"])
            rows += 1; targets += sum(t != -100 for t in row["labels"])
        if position != shard["size_bytes"] or stream.read(1):
            raise ValueError("Index omitted source bytes")
    return {"split": split, "groups": groups, "normalized": normalized, "source_ids": source_ids,
            "retained": retained, "rows": rows, "targets": targets}


def qualify(manifest_path, input_index, output, *, workers=1):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError("Qualification workers must be an integer from 1 through 64")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    identity = manifest["renderer_identity"]
    validate_identity(identity)
    index = json.loads(Path(input_index).read_bytes())
    if (index.get("schema") != "turn-normalized-token-index/v2"
            or index.get("manifest_sha256") != digest_file(manifest_path)
            or index.get("renderer_identity") != identity):
        raise ValueError("Prepared index differs from its corrected source manifest")
    groups = {"train": set(), "validation": set()}
    normalized, source_ids, retained = set(), set(), set()
    rows, targets = 0, 0
    tasks = []
    for split in ("train", "validation"):
        entries, cached = manifest["splits"][split], index["splits"][split]
        if len(entries) != len(cached):
            raise ValueError("Index shard count mismatch")
        for entry, shard in zip(entries, cached):
            tasks.append((str(manifest_path.parent), split, entry, shard, identity))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(_qualify_shard, tasks) if pool else map(_qualify_shard, tasks)
        for result in results:
            if (normalized & result["normalized"] or source_ids & result["source_ids"]
                    or retained & result["retained"]):
                raise ValueError("Duplicate normalized sequence, source or retained training arrays in corrected corpus")
            normalized.update(result["normalized"]); source_ids.update(result["source_ids"])
            retained.update(result["retained"]); groups[result["split"]].update(result["groups"])
            rows += result["rows"]; targets += result["targets"]
    finally:
        if pool: pool.shutdown(wait=True, cancel_futures=True)
    if groups["train"] & groups["validation"]:
        raise ValueError("Source session groups overlap training and validation")
    if rows != sum(manifest["counts"].values()) or rows + sum(manifest["exclusions"].values()) != manifest["source_jobs"]:
        raise ValueError("Full source accounting differs from actual rows")
    index.update(schema=SCHEMA, validator_sha256=digest_file(__file__), split_group_overlap=0,
                 rows_validated=rows, supervised_tokens_validated=targets, full_mask_audit=True)
    with Path(output).open("x") as stream:
        stream.write(json.dumps(index, separators=(",", ":"), sort_keys=True) + "\n")
    return {"schema": SCHEMA, "index_sha256": digest_file(output), "rows_validated": rows,
            "supervised_tokens_validated": targets, "split_group_overlap": 0, "full_mask_audit": True}


class TurnTokenDataset:
    def __init__(self, path_or_dataset, *, index_path, index_sha256, split="train", seq_len=262144,
                 require_provenance=True, tokenizer=None, max_input_tokens=None, order="source", max_samples=None):
        del tokenizer
        if seq_len != 262144 or require_provenance is not True or order != "source":
            raise ValueError("Corrected dataset requires its full-context source-order provenance contract")
        path = Path(path_or_dataset)
        manifest = json.loads(path.read_bytes())
        self.identity = manifest["renderer_identity"]
        validate_identity(self.identity)
        if digest_file(index_path) != index_sha256:
            raise ValueError("Qualified corrected index identity changed")
        index = json.loads(Path(index_path).read_bytes())
        expected = {"schema": SCHEMA, "manifest_sha256": digest_file(path), "renderer_identity": self.identity,
                    "validator_sha256": digest_file(__file__), "split_group_overlap": 0, "full_mask_audit": True}
        if any(index.get(k) != v for k, v in expected.items()):
            raise ValueError("Corrected dataset has not passed its current full-corpus validator")
        self.refs = []
        entries, cached = manifest["splits"][split], index["splits"][split]
        if len(entries) != len(cached):
            raise ValueError("Qualified index shard coverage mismatch")
        seen = set()
        for entry, shard in zip(entries, cached):
            shard_path = (path.parent / entry["path"]).resolve()
            if (shard_path in seen or entry["path"] != shard["path"] or entry["sha256"] != shard["sha256"]
                    or digest_file(shard_path) != entry["sha256"] or shard_path.stat().st_size != shard["size_bytes"]):
                raise ValueError("Qualified training shard identity changed")
            seen.add(shard_path)
            position = 0
            for offset, length, checksum, tokens in shard["rows"]:
                if offset != position or length <= 0 or not 2 <= tokens <= seq_len:
                    raise ValueError("Invalid qualified row offsets")
                position += length
                if max_input_tokens is None or tokens <= max_input_tokens:
                    self.refs.append((shard_path, offset, length, checksum, tokens))
            if position != shard["size_bytes"]:
                raise ValueError("Qualified index omitted shard bytes")
        if max_samples is not None:
            if type(max_samples) is not int or max_samples < 1:
                raise ValueError("max_samples must be positive")
            self.refs = self.refs[:max_samples]
        if not self.refs:
            raise ValueError("Empty corrected dataset")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, item):
        path, offset, size, checksum, tokens = self.refs[item]
        with path.open("rb") as stream:
            stream.seek(offset); raw = stream.read(size)
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError("Corrected row changed after dataset audit")
        row = json.loads(raw)
        if len(row["input_ids"]) != tokens:
            raise ValueError("Corrected row length differs from audited index")
        return validate_row(row, self.identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--input-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    print(json.dumps(qualify(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
