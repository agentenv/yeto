"""Distinct, immutable pretokenized dataset contract for masked filled-CoT SFT.

This module does CPU I/O only. It never calls a teacher, renders a trace, packs
examples, or changes the existing no-CoT dataset. Labels in JSONL are unshifted;
the qualified baseline shift/collator are reused only after the new audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.qwen38_no_cot.nemo_data import collate_exact, shift_next_token_row

MAX_TOKENS = 262144
MANIFEST_SCHEMA = "qwen38-filled-cot-masked-manifest/v1"
INDEX_SCHEMA = "qwen38-filled-cot-masked-index/v1"
TRAINING_CONTRACT = "qwen38-xhigh-filled-cot-loss-zero/v1"
MASK_POLICY = "assistant_content_and_eos_only_reviewed_cot_masked_xhigh_v1"
SEQUENCE_POLICY = "first_262144_tokens_drop_remainder/v1"
RENDERER_VERSION = "qwen3.8-xhigh-reviewed-cot-input-only/v1"
NATIVE_ASSET_IDENTITY = {
    "repository": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "tokenizer_sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config_sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "official_template_sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
}


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def validate_renderer_identity(identity):
    """Require the new renderer's policy, never a baseline-labelled identity."""
    folder = Path(__file__).parent
    baseline = folder.parent / "qwen38_no_cot"
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in {
            **NATIVE_ASSET_IDENTITY,
            "version": RENDERER_VERSION,
            "mask_policy": MASK_POLICY,
            "sequence_policy": SEQUENCE_POLICY,
            "max_sequence_length": MAX_TOKENS,
            "secret_redaction": False,
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
            "renderer_sha256": digest_file(folder / "render.py"),
            "baseline_normalizer_sha256": digest_file(baseline / "upstream_manager/render.py"),
            "source_adapter_sha256": digest_file(baseline / "prepare_data.py"),
            "prefix_adapter_sha256": digest_file(baseline / "render.py"),
    }.items()):
        raise ValueError("Dataset renderer is not the filled-CoT masked training contract")
    if not all(_hash(identity.get(key)) for key in ("adapted_template_sha256", "loss_template_sha256")):
        raise ValueError("Dataset lacks exact native rendering/masking template identities")
    # The complete renderer identity is subsequently compared, byte-for-byte in
    # canonical JSON terms, across the manifest, rows, index and recipe.
    return identity


def validate_row(row, renderer_identity):
    validate_renderer_identity(renderer_identity)
    if not isinstance(row, dict):
        raise ValueError("Expected a pretokenized training row")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict) or metadata.get("renderer_identity") != renderer_identity:
        raise ValueError("Row renderer identity differs from the filled-CoT manifest")
    if not isinstance(row.get("group_id"), str) or not row["group_id"] or row.get("source") not in {"trace", "replay"}:
        raise ValueError("A stable trace/replay source group is required")
    shifted = shift_next_token_row(row, seq_len=MAX_TOKENS)
    audit = metadata.get("sequence_audit", {})
    if (not isinstance(audit, dict) or audit.get("policy") != renderer_identity["sequence_policy"]
            or audit.get("max_sequence_length") != MAX_TOKENS
            or audit.get("retained_input_tokens") != len(row["input_ids"])
            or audit.get("eot_appended") is not False
            or audit.get("causal_shift_applied") is not False):
        raise ValueError("Row lacks exact native-prefix truncation evidence")
    mask_audit = metadata.get("cot_mask_audit")
    if (not isinstance(mask_audit, dict)
            or mask_audit.get("policy") != "filled_native_think_wrapper_and_body_loss0/v1"):
        raise ValueError("Row lacks filled-CoT masking evidence")
    ranges = mask_audit.get("retained_token_ranges")
    if not isinstance(ranges, list):
        raise ValueError("CoT token ranges must be an explicit list")
    previous_end = 0
    for span in ranges:
        if (not isinstance(span, list) or len(span) != 2
                or any(type(n) is not int for n in span)
                or not previous_end <= span[0] < span[1] <= len(row["labels"])):
            raise ValueError("CoT token ranges must be ordered, disjoint retained offsets")
        if any(label != -100 for label in row["labels"][span[0]:span[1]]):
            raise ValueError("Filled CoT or its thinking wrapper has nonzero loss")
        previous_end = span[1]
    for key in ("filled_gap_count", "approved_review_count"):
        if type(mask_audit.get(key)) is not int or mask_audit[key] < 0:
            raise ValueError("Invalid CoT approval count")
    if mask_audit["filled_gap_count"] != mask_audit["approved_review_count"]:
        raise ValueError("Every filled gap requires a bound approved review")
    retained = mask_audit.get("retained_filled_gap_count")
    if (type(retained) is not int or retained != len(ranges)
            or not 0 <= retained <= mask_audit["filled_gap_count"]):
        raise ValueError("Retained filled gaps and CoT ranges disagree")
    if row["source"] == "trace" and mask_audit["filled_gap_count"] < 1:
        raise ValueError("Filled-CoT trace has no approved filled gap")
    provenance = metadata.get("provenance", {}).get("masked_cot", {})
    if (not isinstance(provenance, dict) or provenance.get("model_review_revalidated") is not True
            or provenance.get("reasoning_loss") != 0 or provenance.get("reasoning_effort") != "xhigh"
            or provenance.get("filled_gaps") != mask_audit["filled_gap_count"]
            or not _hash(provenance.get("source_digest"))
            or not _hash(provenance.get("reviewer_config_sha256"))
            or not isinstance(provenance.get("trace_id"), str) or not provenance["trace_id"]):
        raise ValueError("Row lacks source-bound, revalidated prefix-review provenance")
    reviews = provenance.get("approved_gaps")
    if not isinstance(reviews, list) or len(reviews) != mask_audit["approved_review_count"]:
        raise ValueError("CoT masking and approved review receipt coverage differ")
    seen = set()
    for ref in reviews:
        if (not isinstance(ref, dict) or not isinstance(ref.get("gap_id"), str) or not ref["gap_id"]
                or ref["gap_id"] in seen or ref.get("source_digest") != provenance["source_digest"]
                or not all(_hash(ref.get(key)) for key in (
                    "candidate_sha256", "prefix_hash", "review_record_sha256", "review_prompt_hash"))):
            raise ValueError("Invalid or duplicate source-bound CoT review reference")
        seen.add(ref["gap_id"])
    return shifted


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_bytes())
    if (manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("training_contract") != TRAINING_CONTRACT):
        raise ValueError("Expected a distinct filled-CoT manifest; no-CoT input is forbidden")
    validate_renderer_identity(manifest.get("renderer_identity"))
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or "train" not in splits or set(splits) - {"train", "validation"}:
        raise ValueError("Manifest requires explicit train and optional validation splits")
    seen = set()
    for split, shards in splits.items():
        if not isinstance(shards, list) or not shards:
            raise ValueError("Every declared split must contain shards")
        for shard in shards:
            if (not isinstance(shard, dict) or not isinstance(shard.get("path"), str)
                    or not shard["path"] or not _hash(shard.get("sha256"))):
                raise ValueError("Every shard needs a path and SHA256 identity")
            resolved = (path.parent / shard["path"]).resolve()
            if resolved in seen:
                raise ValueError("Duplicate shard would repeat data or cross splits")
            seen.add(resolved)
    return manifest


def _scan_shard(path, expected, identity):
    refs, groups, h = [], set(), hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            h.update(line)
            if not line.strip():
                raise ValueError("Blank lines are not training records")
            row = json.loads(line)
            validate_row(row, identity)
            refs.append((path, offset, len(line), hashlib.sha256(line).hexdigest(), len(row["input_ids"])))
            groups.add(row["group_id"])
    if h.hexdigest() != expected:
        raise ValueError("Training shard identity changed")
    if not refs:
        raise ValueError("Empty training shard")
    return refs, groups


def validator_identity():
    from training.qwen38_no_cot import nemo_data
    return {"validator_sha256": digest_file(__file__),
            "shared_shift_collator_sha256": digest_file(nemo_data.__file__)}


def build_index(manifest_path, output):
    """Audit all rows/splits and exclusively publish a new CPU offset index."""
    manifest_path = Path(manifest_path).resolve()
    raw_digest = digest_file(manifest_path)
    manifest = read_manifest(manifest_path)
    splits, groups_by_split = {}, {}
    for split, entries in manifest["splits"].items():
        shards, groups = [], set()
        for entry in entries:
            path = (manifest_path.parent / entry["path"]).resolve()
            refs, source_groups = _scan_shard(path, entry["sha256"], manifest["renderer_identity"])
            groups.update(source_groups)
            shards.append({"path": str(path), "sha256": entry["sha256"],
                           "size_bytes": path.stat().st_size,
                           "rows": [list(ref[1:]) for ref in refs]})
        splits[split], groups_by_split[split] = shards, groups
    if groups_by_split.get("train", set()) & groups_by_split.get("validation", set()):
        raise ValueError("A source group crosses train/validation")
    if digest_file(manifest_path) != raw_digest:
        raise ValueError("Manifest changed during audit")
    result = {"schema": INDEX_SCHEMA, "training_contract": TRAINING_CONTRACT,
              "manifest_sha256": raw_digest, "renderer_identity": manifest["renderer_identity"],
              **validator_identity(), "seq_len": MAX_TOKENS, "splits": splits,
              "split_group_overlap": 0}
    with Path(output).open("x") as stream:
        stream.write(json.dumps(result, separators=(",", ":")) + "\n")
    return {"index": str(Path(output).resolve()), "sha256": digest_file(output),
            "rows": {split: sum(len(shard["rows"]) for shard in shards) for split, shards in splits.items()}}


class FilledMaskedTokenDataset:
    """Immutable map dataset with a separate, audited filled-CoT index.

    Passing only a JSONL filename or disabling provenance is unsupported. The
    index is required so split-overlap auditing and full-corpus row validation
    happen before a training process starts. Each fetched row is checked again.
    """

    def __init__(self, path_or_dataset, *, index_path, index_sha256, split="train",
                 seq_len=MAX_TOKENS, require_provenance=True, tokenizer=None,
                 max_input_tokens=None, order="source", max_samples=None):
        del tokenizer
        if seq_len != MAX_TOKENS or require_provenance is not True:
            raise ValueError("Filled-CoT requires exact 262144 context and provenance")
        if not _hash(index_sha256):
            raise ValueError("An explicit filled-CoT index digest is required")
        manifest_path = Path(path_or_dataset).resolve()
        manifest = read_manifest(manifest_path)
        raw = Path(index_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != index_sha256:
            raise ValueError("Filled-CoT index identity changed")
        index = json.loads(raw)
        expected = {"schema": INDEX_SCHEMA, "training_contract": TRAINING_CONTRACT,
                    "manifest_sha256": digest_file(manifest_path), "seq_len": MAX_TOKENS,
                    "renderer_identity": manifest["renderer_identity"], "split_group_overlap": 0,
                    **validator_identity()}
        if any(index.get(key) != value for key, value in expected.items()):
            raise ValueError("Index does not match the filled-CoT validator/manifest contract")
        self.renderer_identity = manifest["renderer_identity"]
        self.refs = []
        if split not in manifest["splits"]:
            raise ValueError("Requested split is absent")
        entries = manifest["splits"][split]
        indexed = index.get("splits", {}).get(split)
        if not isinstance(indexed, list) or len(indexed) != len(entries):
            raise ValueError("Index shard coverage differs")
        for entry, cached in zip(entries, indexed):
            path = (manifest_path.parent / entry["path"]).resolve()
            if (cached.get("path") != str(path) or cached.get("sha256") != entry["sha256"]
                    or digest_file(path) != entry["sha256"]
                    or cached.get("size_bytes") != path.stat().st_size):
                raise ValueError("Training shard identity or size changed")
            offset = 0
            rows = cached.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ValueError("Index is missing rows")
            for ref in rows:
                if (not isinstance(ref, list) or len(ref) != 4
                        or type(ref[0]) is not int or ref[0] != offset
                        or type(ref[1]) is not int or ref[1] < 1
                        or not _hash(ref[2]) or type(ref[3]) is not int
                        or not 2 <= ref[3] <= MAX_TOKENS):
                    raise ValueError("Index must cover each shard exactly once in source order")
                offset += ref[1]
                self.refs.append((path, *ref))
            if offset != cached["size_bytes"]:
                raise ValueError("Index omitted or duplicated source bytes")
        if max_input_tokens is not None:
            if type(max_input_tokens) is not int or not 2 <= max_input_tokens <= MAX_TOKENS:
                raise ValueError("Invalid input-length filter")
            self.refs = [ref for ref in self.refs if ref[4] <= max_input_tokens]
        if order == "longest_first":
            self.refs.sort(key=lambda ref: -ref[4])
        elif order != "source":
            raise ValueError("Unsupported dataset ordering")
        if max_samples is not None:
            if type(max_samples) is not int or max_samples < 1:
                raise ValueError("max_samples must be positive")
            self.refs = self.refs[:max_samples]
        if not self.refs:
            raise ValueError("Empty selected dataset")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        path, offset, size, expected, tokens = self.refs[index]
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(size)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Training row changed after audit")
        row = json.loads(raw)
        if len(row.get("input_ids", [])) != tokens:
            raise ValueError("Indexed token length differs from row")
        return validate_row(row, self.renderer_identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index-output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_index(args.manifest, args.index_output)))


if __name__ == "__main__":
    main()
