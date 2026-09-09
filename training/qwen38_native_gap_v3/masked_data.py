"""Immutable token dataset for explicitly authorized, unreviewed generated-CoT SFT.

This module does CPU I/O only. It never calls a teacher, renders a trace, packs
examples, or changes the existing no-CoT dataset. Labels in JSONL are unshifted;
the qualified baseline shift/collator are reused only after the new audit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from training.qwen38_no_cot.nemo_data import collate_exact, shift_next_token_row

MAX_TOKENS = 262144
MANIFEST_SCHEMA = "qwen38-generated-cot-native-gap-manifest/v3"
INDEX_SCHEMA = "qwen38-generated-cot-native-gap-index/v3"
TRAINING_CONTRACT = "qwen38-xhigh-native-gap-cot-loss-zero/v3"
MASK_POLICY = "assistant_content_and_eos_only_native_gap_cot_masked_v3"
ACCEPTANCE_POLICY = "all-structurally-valid-frozen-generations-experimental/v1"
DATASET_TARGET = "training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset"
COLLATOR_TARGET = "training.qwen38_native_gap_v3.masked_data.collate_exact"
SEQUENCE_POLICY = "first_262144_tokens_drop_remainder/v1"
RENDERER_VERSION = "qwen3.8-xhigh-native-leading-gap-cot-input-only/v3"
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
            "acceptance_policy": ACCEPTANCE_POLICY,
            "semantic_quality_qualified": False,
            "future_information_checked": False,
            "renderer_sha256": digest_file(folder / "masked.py"),
            "native_parent_renderer_sha256": digest_file(folder.parent / "qwen38_turn_boundary_v2/masked.py"),
            "omitted_gap_policy": "keep-native-leading-gap-omit-only-conflicting-cot/v3",
            "inline_reasoning_inserted": False,
            "ordinary_native_assistant_template": True,
            "generated_renderer_sha256": digest_file(folder.parent / "qwen38_cot_experimental/render.py"),
            "turn_boundary_normalizer_sha256": digest_file(folder / "normalize.py"),
            "turn_boundary_source_adapter_sha256": digest_file(folder / "source_adapters.py"),
            "turn_boundary_policy": "qwen38-assistant-turn-coalescing/v2",
            "reasoning_placement_policy": "keep-native-leading-gap-omit-only-conflicting-cot/v3",
            "generation_contract_sha256": digest_file(folder.parent / "qwen38_cot_experimental/generation_contract.py"),
            "reviewed_renderer_sha256": digest_file(folder.parent / "qwen38_cot_masked/render.py"),
            "baseline_normalizer_sha256": digest_file(baseline / "upstream_manager/render.py"),
            "source_adapter_sha256": digest_file(baseline / "prepare_data.py"),
            "prefix_adapter_sha256": digest_file(baseline / "render.py"),
    }.items()):
        raise ValueError("Dataset renderer is not the experimental generated-CoT masked training contract")
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
        raise ValueError("Row renderer identity differs from the experimental generated-CoT manifest")
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
            or mask_audit.get("policy") != "filled_native_think_wrapper_and_body_loss0/v1"
            or "approved_review_count" in mask_audit):
        raise ValueError("Row lacks experimental generated-CoT masking evidence")
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
    for key in ("filled_gap_count", "generated_receipt_count"):
        if type(mask_audit.get(key)) is not int or mask_audit[key] < 0:
            raise ValueError("Invalid generated-CoT coverage count")
    if mask_audit["filled_gap_count"] != mask_audit["generated_receipt_count"]:
        raise ValueError("Every filled gap requires a bound original generation receipt")
    retained = mask_audit.get("retained_filled_gap_count")
    if (type(retained) is not int or retained != len(ranges)
            or not 0 <= retained <= mask_audit["filled_gap_count"]):
        raise ValueError("Retained generated gaps and CoT ranges disagree")
    outer = metadata.get("provenance", {})
    if not isinstance(outer, dict) or "masked_cot" in outer:
        raise ValueError("Experimental generated rows cannot masquerade as reviewed masked-CoT exports")
    provenance = outer.get("generated_cot", {})
    if (not isinstance(provenance, dict)
            or provenance.get("generation_receipts_verified") is not True
            or provenance.get("reviewed") is not False
            or provenance.get("semantic_review_required") is not False
            or provenance.get("semantic_quality_qualified") is not False
            or provenance.get("future_information_checked") is not False
            or ("semantic_certainty_claimed" in provenance and provenance["semantic_certainty_claimed"] is not False)
            or provenance.get("acceptance_policy") != ACCEPTANCE_POLICY
            or type(provenance.get("reasoning_loss")) is not int or provenance["reasoning_loss"] != 0
            or provenance.get("reasoning_effort") != "xhigh"
            or provenance.get("source_kind") != row["source"]
            or type(provenance.get("filled_gaps")) is not int
            or provenance["filled_gaps"] != mask_audit["filled_gap_count"]
            or type(provenance.get("unreviewed_gap_count")) is not int
            or provenance["unreviewed_gap_count"] != mask_audit["filled_gap_count"]
            or not _hash(provenance.get("source_digest"))
            or not isinstance(provenance.get("trace_id"), str) or not provenance["trace_id"]
            or any(k in provenance for k in ("approved_gaps", "approved_review_count", "model_review_revalidated"))):
        raise ValueError("Row requires explicit source-bound generation provenance and unreviewed experimental status")
    refs = provenance.get("generated_gaps")
    if not isinstance(refs, list) or len(refs) != mask_audit["generated_receipt_count"]:
        raise ValueError("CoT masks and original generation receipt coverage differ")
    if row["source"] == "replay":
        if refs or provenance.get("generation_config_sha256") is not None:
            raise ValueError("Replay rows contain no injected generated gaps or generation config")
    elif not _hash(provenance.get("generation_config_sha256")):
        raise ValueError("Selected source traces require the frozen generation-config identity")
    seen_gaps, seen_events = set(), set()
    for ref in refs:
        if (not isinstance(ref, dict) or not isinstance(ref.get("event_id"), str) or not ref["event_id"]
                or not _hash(ref.get("gap_id"))
                or ref["gap_id"] in seen_gaps or ref["event_id"] in seen_events
                or ref.get("source_digest") != provenance["source_digest"]
                or ref.get("generation_config_sha256") != provenance["generation_config_sha256"]
                or not all(_hash(ref.get(key)) for key in (
                    "candidate_sha256", "prefix_hash", "lookahead_hash", "prompt_hash",
                    "generation_receipt_sha256", "candidate_record_sha256", "generation_config_sha256"))):
            raise ValueError("Invalid or duplicate source-bound original generation reference")
        seen_gaps.add(ref["gap_id"])
        seen_events.add(ref["event_id"])
    validate_omission_ledger(row)
    validate_turn_boundary_row(row, renderer_identity)
    validate_native_mask(row)
    return shifted


def validate_omission_ledger(row):
    provenance=row['metadata']['provenance']['generated_cot']
    ledger=provenance.get('native_gap_omission',{})
    retained=provenance['generated_gaps'];inputs=ledger.get('input_generated_gaps');omitted=ledger.get('omitted_generated_gaps')
    if (ledger.get('policy')!='keep-native-leading-gap-omit-only-conflicting-cot/v3'
            or any(ledger.get(key)is not False for key in ('original_generation_receipts_changed','original_source_changed','candidate_semantic_quality_assessed'))
            or ledger.get('trace_retained_when_all_cot_omitted') is not True
            or not isinstance(inputs,list) or not isinstance(omitted,list)
            or ledger.get('bound_input_generation_count')!=len(inputs)
            or ledger.get('retained_generation_count')!=len(retained)
            or ledger.get('omitted_generation_count')!=len(omitted)):
        raise ValueError('Missing exact original-input/native-gap omission ledger')
    input_map={ref.get('gap_id'):ref for ref in inputs}
    if len(input_map)!=len(inputs):raise ValueError('Duplicate bound input generation')
    seen=set()
    for ref in retained+omitted:
        key=ref.get('gap_id');original=input_map.get(key)
        if not isinstance(original,dict) or key in seen or any(ref.get(k)!=v for k,v in original.items()):
            raise ValueError('Retained/omitted gap changed the original generation receipt')
        seen.add(key)
        if (original.get('source_digest')!=provenance['source_digest']
                or original.get('generation_config_sha256')!=provenance.get('generation_config_sha256')
                or not all(_hash(original.get(k)) for k in ('gap_id','candidate_sha256','prefix_hash','lookahead_hash','prompt_hash','generation_receipt_sha256','candidate_record_sha256','generation_config_sha256'))):
            raise ValueError('Omitted gap lacks exact original provenance')
    if seen!=set(input_map):raise ValueError('An original generated gap was silently omitted')
    mapping=provenance.get('original_event_message_mapping')
    for ref in omitted:
        if (ref.get('reason') not in {'reasoning_after_visible_content','multiple_reasoning_blocks'}
                or type(ref.get('original_message_index')) is not int
                or not isinstance(mapping,list)
                or not 0 <= ref['original_message_index'] < len(mapping)
                or ref.get('original_event_ids') != mapping[ref['original_message_index']]
                or ref.get('event_id') not in ref.get('original_event_ids',[])):
            raise ValueError('Gap omission lacks an original native placement conflict')
    mask=row['metadata']['cot_mask_audit']
    if (mask.get('bound_input_generation_count')!=len(inputs) or mask.get('omitted_generation_count')!=len(omitted)
            or len(inputs)!=len(retained)+len(omitted)):
        raise ValueError('Native gap mask ledger counts disagree')


def validate_native_mask(row):
    """Reconstruct body/action/EOT supervision from actual pinned native IDs."""
    ids, labels = row['input_ids'], row['labels']
    truncated = row['metadata']['sequence_audit'].get('truncated') is True
    starts = [i for i, token in enumerate(ids) if token == 248045]
    if not starts or starts[0] != 0:
        raise ValueError('Expected pinned native message headers')
    expected = [-100] * len(ids)
    for ordinal, start in enumerate(starts):
        stop = starts[ordinal+1] if ordinal+1 < len(starts) else len(ids)
        final_cutoff = truncated and ordinal == len(starts)-1
        if start+2 >= stop:
            if not final_cutoff: raise ValueError('Incomplete native header')
            continue
        if ids[start+1] not in {74455, 846, 8678, 13766} or ids[start+2] != 198:
            raise ValueError('Unknown pinned native role header')
        try: end = ids.index(248046, start+3, stop)
        except ValueError:
            end = None
            if not final_cutoff: raise ValueError('Incomplete native message before end of trace')
        if ids[start+1] != 74455: continue
        if start+3 >= stop:
            if not final_cutoff: raise ValueError('Missing native think opening')
            continue
        if ids[start+3] != 248068: raise ValueError('Missing native think opening')
        try: close = ids.index(248069, start+4, stop)
        except ValueError:
            if not final_cutoff or end is not None: raise ValueError('Unclosed native reasoning')
            continue
        if end is not None and close >= end: raise ValueError('Reasoning crossed the native end token')
        if close+1 >= stop:
            if not final_cutoff: raise ValueError('Missing native post-think whitespace')
            continue
        if ids[close+1] != 271: raise ValueError('Native think/body separator differs')
        body = close+2
        if end is not None and end <= body:
            raise ValueError('Empty assistant would train only an end token')
        bound = end+1 if end is not None else stop
        expected[body:bound] = ids[body:bound]
    if expected != labels:
        raise ValueError('Saved labels differ from native body/action/EOT-only mask')


def validate_turn_boundary_row(row, renderer_identity):
    metadata = row['metadata']
    audit = metadata.get('turn_boundary_audit', {})
    if (not isinstance(audit, dict) or audit.get('schema') != 'qwen38-assistant-turn-coalescing/v2'
            or audit.get('normalizer_source_sha256') != renderer_identity['turn_boundary_normalizer_sha256']
            or audit.get('excluded') is not False or audit.get('exclusions') != []
            or metadata.get('normalization', {}).get('turn_boundary') != audit
            or not all(_hash(audit.get(k)) for k in ('input_messages_sha256', 'output_messages_sha256', 'boundaries_sha256'))):
        raise ValueError('Row lacks successful exact versioned turn-boundary normalization evidence')
    mapping, groups = audit.get('input_to_output'), audit.get('output_to_input')
    if (not isinstance(mapping, list) or not isinstance(groups, list)
            or len(mapping) != audit.get('input_messages') or len(groups) != audit.get('output_messages')):
        raise ValueError('Turn-boundary mapping coverage is invalid')
    seen = set()
    for output, group in enumerate(groups):
        if not isinstance(group, list) or not group or group != sorted(group):
            raise ValueError('Output message has no ordered original inputs')
        for index in group:
            if type(index) is not int or not 0 <= index < len(mapping) or index in seen or mapping[index] != output:
                raise ValueError('Turn-boundary input/output mapping disagrees')
            seen.add(index)
    if any(value is not None and index not in seen for index, value in enumerate(mapping)):
        raise ValueError('Turn-boundary input is neither retained nor explicitly dropped')
    provenance = metadata['provenance']['generated_cot']
    if (provenance.get('original_gap_positions_preserved') is not True
            or provenance.get('reasoning_relocated') is not False
            or provenance.get('turn_boundary_policy') != renderer_identity['turn_boundary_policy']):
        raise ValueError('Original generated reasoning positions are not preserved')
    placements = provenance.get('filled_gap_message_placements')
    refs = provenance['generated_gaps']
    if not isinstance(placements, list) or len(placements) != len(refs):
        raise ValueError('Every generated gap needs an original-to-merged placement')
    if refs:
        original = provenance.get('original_event_message_mapping')
        output = provenance.get('event_message_mapping')
        if (not isinstance(original, list) or len(original) != len(mapping)
                or output != [[event for index in group for event in original[index]] for group in groups]):
            raise ValueError('Original event identities changed across turn mapping')
        seen_output = set()
        for ref in refs:
            matches = [p for p in placements if ref['event_id'] in p.get('original_event_ids', [])]
            if len(matches) != 1:
                raise ValueError('Generated event placement is absent or duplicated')
            placement = matches[0]
            index, merged = placement.get('original_message_index'), placement.get('merged_message_index')
            if (type(index) is not int or not 0 <= index < len(mapping) or mapping[index] != merged
                    or merged in seen_output or placement.get('original_event_ids') != original[index]
                    or placement.get('reasoning_sha256') != ref['candidate_sha256']):
                raise ValueError('Generated reasoning or its source position changed')
            seen_output.add(merged)


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_bytes())
    if (manifest.get("schema") != MANIFEST_SCHEMA
            or manifest.get("training_contract") != TRAINING_CONTRACT):
        raise ValueError("Expected a distinct experimental generated-CoT manifest; no-CoT input is forbidden")
    if (manifest.get("experimental") is not True
            or manifest.get("acceptance_policy") != ACCEPTANCE_POLICY
            or any(manifest.get(key) is not False for key in (
                "reviewed", "semantic_review_required", "semantic_quality_qualified", "future_information_checked"))
            or type(manifest.get("unreviewed_gap_count")) is not int
            or manifest["unreviewed_gap_count"] < 0):
        raise ValueError("Manifest must explicitly identify the unreviewed experiment and its gap count")
    validate_renderer_identity(manifest.get("renderer_identity"))
    if manifest.get("full_export"):
        if (manifest.get("all_frozen_generations_accounted_for") is not True
                or manifest.get("unexpected_trace_failures") != 0
                or manifest.get("gap_omission_policy") != "keep-native-leading-gap-omit-only-conflicting-cot/v3"
                or type(manifest.get("omitted_valid_candidate_count")) is not int
                or manifest["omitted_valid_candidate_count"] < 0
                or manifest.get("turn_boundary_exclusion_policy") != "whole-source-exclusion-no-cot-relocation/v2"
                or manifest.get("native_render_exclusion_policy") != "unchanged-native-shape-guards-whole-source/v2"
                or manifest.get("included_valid_candidate_count") != manifest["unreviewed_gap_count"]
                or manifest.get("included_valid_candidate_count", -1) + manifest.get("omitted_valid_candidate_count", -1) + manifest.get("excluded_valid_candidate_count", -1)
                    != manifest.get("frozen_valid_candidate_count")):
            raise ValueError("Corrected full export does not account for every frozen generated gap")
        for key in ("turn_boundary_excluded_sources", "turn_boundary_excluded_generations", "native_render_excluded_sources", "native_render_excluded_generations", "duplicate_excluded_sources", "duplicate_excluded_generations"):
            if (not isinstance(manifest.get(key), dict)
                    or any(type(value) is not int or value < 0 for value in manifest[key].values())):
                raise ValueError("Turn-boundary whole-source exclusions need explicit nonnegative counts")
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


def _scan_shard(path, expected, identity, *, split=None, all_train=True):
    refs, groups, h, gaps = [], set(), hashlib.sha256(), 0
    seen = set()
    omitted_gaps = 0
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
            if split is not None:
                from .prepare_masked_full import split_for_group
                provenance = row['metadata']['provenance']
                unknown = provenance.get('session_identity_verified') is False
                expected_split = 'train' if all_train or unknown else split_for_group(row['group_id'])
                if split != expected_split:
                    raise ValueError('Actual row violates frozen known-session/unknown-capture split')
            normalized = 'normalized:' + row['metadata']['turn_boundary_audit']['output_messages_sha256']
            retained = 'retained:' + hashlib.sha256(json.dumps([row['input_ids'],row['labels']],separators=(',',':')).encode()).hexdigest()
            if normalized in seen or retained in seen:
                raise ValueError('Duplicate normalized or retained sequence in masked shard')
            seen.update((normalized,retained)); groups.update((normalized,retained))
            gaps += row["metadata"]["cot_mask_audit"]["filled_gap_count"]
            omitted_gaps += row["metadata"]["cot_mask_audit"]["omitted_generation_count"]
            refs.append((path, offset, len(line), hashlib.sha256(line).hexdigest(), len(row["input_ids"])))
            groups.add("group:" + row["group_id"])
            groups.add("source:" + row["metadata"]["provenance"]["generated_cot"]["source_digest"])
    if h.hexdigest() != expected:
        raise ValueError("Training shard identity changed")
    if not refs:
        raise ValueError("Empty training shard")
    return refs, groups, gaps, omitted_gaps


def validator_identity():
    from training.qwen38_no_cot import nemo_data
    return {"validator_sha256": digest_file(__file__),
            "shared_shift_collator_sha256": digest_file(nemo_data.__file__)}


def _scan_index_job(job):
    split, path, expected, identity, all_train = job
    return split, _scan_shard(path, expected, identity, split=split, all_train=all_train)


def build_index(manifest_path, output, *, workers=1):
    """Audit all rows/splits and exclusively publish a new CPU offset index."""
    manifest_path = Path(manifest_path).resolve()
    raw_digest = digest_file(manifest_path)
    manifest = read_manifest(manifest_path)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('Bounded full-mask index concurrency required')
    jobs = [(split, (manifest_path.parent/entry['path']).resolve(), entry['sha256'],
             manifest['renderer_identity'], manifest.get('all_train') is True)
            for split,entries in manifest['splits'].items() for entry in entries]
    splits = {split:[] for split in manifest['splits']}
    groups_by_split = {split:set() for split in splits}
    total_gaps = 0
    total_omitted = 0
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs))) as pool:
        for job, (split, (refs, source_groups, gaps, omitted)) in zip(jobs,pool.map(_scan_index_job,jobs)):
            _,path,checksum,_,_ = job
            total_gaps += gaps
            total_omitted += omitted
            groups = groups_by_split[split]
            duplicates = {value for value in source_groups if value.startswith(('normalized:','retained:'))}
            if groups & duplicates:
                raise ValueError('Duplicate normalized or retained sequence across masked shards')
            groups.update(source_groups)
            splits[split].append({'path':str(path),'sha256':checksum,'size_bytes':path.stat().st_size,
                'rows':[list(ref[1:]) for ref in refs]})
    if groups_by_split.get("train", set()) & groups_by_split.get("validation", set()):
        raise ValueError("A source group crosses train/validation")
    if total_omitted != manifest["omitted_valid_candidate_count"]:
        raise ValueError("Manifest omitted-gap count differs from actual row ledgers")
    if total_gaps != manifest["unreviewed_gap_count"]:
        raise ValueError("Manifest unreviewed gap count differs from the actual training rows")
    if digest_file(manifest_path) != raw_digest:
        raise ValueError("Manifest changed during audit")
    result = {"schema": INDEX_SCHEMA, "training_contract": TRAINING_CONTRACT,
              "manifest_sha256": raw_digest, "renderer_identity": manifest["renderer_identity"],
              **validator_identity(), "seq_len": MAX_TOKENS, "splits": splits,
              "split_group_overlap": 0, "unreviewed_gap_count": total_gaps, "omitted_valid_candidate_count":total_omitted}
    with Path(output).open("x") as stream:
        stream.write(json.dumps(result, separators=(",", ":")) + "\n")
    return {"index": str(Path(output).resolve()), "sha256": digest_file(output),
            "rows": {split: sum(len(shard["rows"]) for shard in shards) for split, shards in splits.items()},
            "unreviewed_gap_count": total_gaps, "omitted_valid_candidate_count":total_omitted}


class GeneratedMaskedTokenDataset:
    """Immutable map dataset with a separate, audited experimental generated-CoT index.

    Passing only a JSONL filename or disabling provenance is unsupported. The
    index is required so split-overlap auditing and full-corpus row validation
    happen before a training process starts. Each fetched row is checked again.
    """

    def __init__(self, path_or_dataset, *, index_path, index_sha256, split="train",
                 seq_len=MAX_TOKENS, require_provenance=True, tokenizer=None,
                 max_input_tokens=None, order="source", max_samples=None):
        del tokenizer
        if seq_len != MAX_TOKENS or require_provenance is not True:
            raise ValueError("Experimental generated-CoT requires exact 262144 context and provenance")
        if not _hash(index_sha256):
            raise ValueError("An explicit experimental generated-CoT index digest is required")
        manifest_path = Path(path_or_dataset).resolve()
        manifest = read_manifest(manifest_path)
        raw = Path(index_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != index_sha256:
            raise ValueError("Experimental generated-CoT index identity changed")
        index = json.loads(raw)
        expected = {"schema": INDEX_SCHEMA, "training_contract": TRAINING_CONTRACT,
                    "manifest_sha256": digest_file(manifest_path), "seq_len": MAX_TOKENS,
                    "renderer_identity": manifest["renderer_identity"], "split_group_overlap": 0,
                    "unreviewed_gap_count": manifest["unreviewed_gap_count"],
                    "omitted_valid_candidate_count": manifest["omitted_valid_candidate_count"],
                    **validator_identity()}
        if any(index.get(key) != value for key, value in expected.items()):
            raise ValueError("Index does not match the experimental generated-CoT validator/manifest contract")
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
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(build_index(args.manifest, args.index_output, workers=args.workers)))


if __name__ == "__main__":
    main()
