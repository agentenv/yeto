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
import math
import os
from pathlib import Path

from training.qwen38_no_cot.nemo_data import collate_exact, shift_next_token_row

MAX_TOKENS = 262144
MANIFEST_SCHEMA = "qwen38-generated-cot-native-gap-manifest/v4"
INDEX_SCHEMA = "qwen38-generated-cot-native-gap-index/v4"
TRAINING_CONTRACT = "qwen38-xhigh-native-gap-cot-loss-zero/v4"
MASK_POLICY = "assistant_content_and_eos_only_native_gap_cot_masked_v4"
ACCEPTANCE_POLICY = "all-structurally-valid-frozen-generations-experimental/v1"
DATASET_TARGET = "training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset"
COLLATOR_TARGET = "training.qwen38_native_gap_v3.masked_data.collate_exact"
SEQUENCE_POLICY = "first_262144_tokens_drop_remainder/v1"
RENDERER_VERSION = "qwen3.8-xhigh-native-leading-gap-cot-input-only/v4"
CROSS_ARM_PARITY_SCHEMA = "qwen38-cross-arm-no-cot-parity/v2"
COMPLETE_SCHEMA = "qwen38-generated-cot-native-gap-complete/v4"
TOKENIZER_ID_COUNT = 248077
MAX_TOKEN_ID = 248076
PINNED_CONTROL_TOKEN_IDS = {
    "<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046,
    "<|object_ref_start|>": 248047, "<|object_ref_end|>": 248048,
    "<|box_start|>": 248049, "<|box_end|>": 248050,
    "<|quad_start|>": 248051, "<|quad_end|>": 248052,
    "<|vision_start|>": 248053, "<|vision_end|>": 248054,
    "<|vision_pad|>": 248055, "<|image_pad|>": 248056,
    "<|video_pad|>": 248057, "<tool_call>": 248058,
    "</tool_call>": 248059, "<|fim_prefix|>": 248060,
    "<|fim_middle|>": 248061, "<|fim_suffix|>": 248062,
    "<|fim_pad|>": 248063, "<|repo_name|>": 248064,
    "<|file_sep|>": 248065, "<tool_response>": 248066,
    "</tool_response>": 248067, "<think>": 248068,
    "</think>": 248069, "<|audio_start|>": 248070,
    "<|audio_end|>": 248071, "<tts_pad>": 248072,
    "<tts_text_bos>": 248073, "<tts_text_eod>": 248074,
    "<tts_text_bos_single>": 248075, "<|audio_pad|>": 248076,
}
ROW_TOKEN_TOTAL_KEYS = (
    "input_tokens", "target_tokens", "cross_arm_baseline_input_tokens",
    "cross_arm_zero_cot_input_tokens", "cross_arm_baseline_target_tokens",
    "cross_arm_zero_cot_target_tokens", "cross_arm_target_delta_vs_baseline",
    "filled_gaps", "retained_filled_gaps", "dropped_input_tokens",
    "dropped_target_tokens", "omitted_gaps", "bound_input_gaps",
)
BASE_ASSET_RECEIPT = {
    "path": "/data/sft_baseline_20260908/model-receipts/"
            "qwen38-base-r1d4bf0f-full-hash-68dd1925b826-20260910/verification.json",
    "sha256": "245478d0cad19b1f1970182dc6a5f6b5a924185d18698bdb5777ffe8d9d9ec45",
    "schema": "qwen38-base-full-byte-verification/v1",
    "source_manifest_sha256": "68dd1925b82632af27368b81383ed7d267e8861d3b5edc1c5a49d8a0a733eafe",
    "file_count": 29,
    "weight_shards": 18,
    "total_bytes": 55586036737,
}
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


def _atomic_json(path, value):
    """Publish a JSON admission marker atomically after durable file data."""
    path = Path(path).resolve()
    pending = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError("Refusing to replace an existing admission marker")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with pending.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    os.chmod(path, 0o400)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest_file(path)


def publish_complete(manifest_path, index_result, output):
    """Atomically admit one co-located immutable manifest and audited index."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    output = Path(output).resolve()
    index_path = Path(index_result["index"]).resolve(strict=True)
    if output.parent != manifest_path.parent or index_path.parent != manifest_path.parent:
        raise ValueError("Completion marker, manifest and index must be co-located")
    for durable in (manifest_path, index_path):
        with durable.open("rb") as stream:
            os.fsync(stream.fileno())
        os.chmod(durable, 0o400)
    directory = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    manifest = read_manifest(manifest_path)
    if (index_result.get("sha256") != digest_file(index_path)
            or index_result.get("rows") != {
                split: sum(value for key, value in manifest["counts"].items()
                           if key.startswith(split + "/"))
                for split in manifest["splits"]}
            or index_result.get("unreviewed_gap_count") != manifest["unreviewed_gap_count"]
            or index_result.get("omitted_valid_candidate_count") != manifest["omitted_valid_candidate_count"]):
        raise ValueError("Index result does not cover the complete manifest")
    payload = {
        "schema": COMPLETE_SCHEMA,
        "status": "complete",
        "manifest": {"path": manifest_path.name, "sha256": digest_file(manifest_path)},
        "index": {"path": index_path.name, "sha256": index_result["sha256"]},
        "counts": manifest["counts"],
        "token_counts": manifest["token_counts"],
        "rows": index_result["rows"],
        "unreviewed_gap_count": manifest["unreviewed_gap_count"],
        "omitted_valid_candidate_count": manifest["omitted_valid_candidate_count"],
    }
    complete_sha256 = _atomic_json(output, payload)
    return {"path": str(output), "sha256": complete_sha256, **payload}


def read_complete(path, expected_sha256, *, manifest_path, index_path, index_sha256):
    supplied = [Path(value) for value in (path, manifest_path, index_path)]
    if any(value.is_symlink() for value in supplied):
        raise ValueError("Completion, manifest and index must be regular non-symlink files")
    path, manifest_path, index_path = [value.resolve(strict=True) for value in supplied]
    if any(not value.is_file() for value in (path, manifest_path, index_path)):
        raise ValueError("Completion, manifest and index must be regular files")
    raw = path.read_bytes()
    if (not _hash(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256
            or path.parent != manifest_path.parent or path.parent != index_path.parent):
        raise ValueError("Atomic completion marker identity or location changed")
    complete = _strict_json(raw)
    manifest = read_manifest(manifest_path)
    expected_rows = {
        split: sum(value for key, value in manifest["counts"].items()
                   if key.startswith(split + "/"))
        for split in manifest["splits"]}
    if (complete.get("schema") != COMPLETE_SCHEMA or complete.get("status") != "complete"
            or complete.get("manifest") != {"path": manifest_path.name,
                                              "sha256": digest_file(manifest_path)}
            or complete.get("index") != {"path": index_path.name, "sha256": index_sha256}
            or digest_file(index_path) != index_sha256
            or complete.get("counts") != manifest["counts"]
            or complete.get("token_counts") != manifest["token_counts"]
            or complete.get("rows") != expected_rows
            or complete.get("unreviewed_gap_count") != manifest["unreviewed_gap_count"]
            or complete.get("omitted_valid_candidate_count") != manifest["omitted_valid_candidate_count"]):
        raise ValueError("Completion marker does not bind the complete manifest/index export")
    return complete


def _hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _strict_json(raw):
    """Parse JSON while refusing the non-finite extensions accepted by Python."""
    def invalid_constant(value):
        raise ValueError("Non-finite JSON value: " + value)

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite JSON float")
        return parsed

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key: " + key)
            result[key] = value
        return result

    return json.loads(raw, parse_constant=invalid_constant, parse_float=finite_float,
                      object_pairs_hook=unique_object)


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
            "omitted_gap_policy": "keep-native-leading-gap-omit-only-conflicting-cot/v4",
            "inline_reasoning_inserted": False,
            "ordinary_native_assistant_template": True,
            "generated_renderer_sha256": digest_file(folder.parent / "qwen38_cot_experimental/render.py"),
            "turn_boundary_normalizer_sha256": digest_file(folder / "normalize.py"),
            "turn_boundary_source_adapter_sha256": digest_file(folder / "source_adapters.py"),
            "cross_arm_baseline_source_adapter_sha256": digest_file(folder.parent / "qwen38_turn_boundary_v2/source_adapters.py"),
            "cross_arm_parity_schema": CROSS_ARM_PARITY_SCHEMA,
            "turn_boundary_policy": "qwen38-assistant-turn-coalescing/v2",
            "reasoning_placement_policy": "keep-native-leading-gap-omit-only-conflicting-cot/v4",
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
    ids, labels = row.get("input_ids"), row.get("labels")
    if (not isinstance(ids, list) or not isinstance(labels, list)
            or any(type(token) is not int or not 0 <= token <= MAX_TOKEN_ID for token in ids)):
        raise ValueError("Token IDs must fit the pinned Qwen vocabulary and torch.long input contract")
    shifted = shift_next_token_row(row, seq_len=MAX_TOKENS)
    audit = metadata.get("sequence_audit", {})
    active = sum(label != -100 for label in labels)
    integer_fields = ("original_input_tokens", "retained_input_tokens", "dropped_input_tokens",
                      "original_supervised_tokens", "retained_supervised_tokens",
                      "dropped_supervised_tokens")
    if (not isinstance(audit, dict) or audit.get("policy") != renderer_identity["sequence_policy"]
            or audit.get("max_sequence_length") != MAX_TOKENS
            or any(type(audit.get(key)) is not int or audit[key] < 0 for key in integer_fields)
            or audit.get("retained_input_tokens") != len(ids)
            or audit.get("retained_input_tokens") > MAX_TOKENS
            or audit.get("original_input_tokens") != audit.get("retained_input_tokens") + audit.get("dropped_input_tokens")
            or audit.get("retained_supervised_tokens") != active
            or audit.get("original_supervised_tokens") != audit.get("retained_supervised_tokens") + audit.get("dropped_supervised_tokens")
            or audit.get("retained_supervised_tokens") > audit.get("retained_input_tokens")
            or audit.get("dropped_supervised_tokens") > audit.get("dropped_input_tokens")
            or audit.get("truncated") is not (audit.get("dropped_input_tokens") > 0)
            or audit.get("eot_appended") is not False
            or audit.get("causal_shift_applied") is not False
            or row.get("attention_mask") != [1] * len(ids)
            or metadata.get("retained_token_ids_sha256") != hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()).hexdigest()):
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
    validate_cross_arm_parity(row)
    validate_omission_ledger(row)
    validate_turn_boundary_row(row, renderer_identity)
    validate_native_mask(row)
    return shifted


def validate_cross_arm_parity(row):
    """Require exact baseline-visible parity before any row reaches the loader."""
    audit=row['metadata']['provenance']['generated_cot'].get('cross_arm_no_cot_parity',{})
    actual=sum(label!=-100 for label in row['labels'])
    integer_keys=('baseline_input_tokens','masked_zero_cot_input_tokens','actual_masked_input_tokens',
        'baseline_target_tokens','masked_zero_cot_target_tokens','actual_masked_target_tokens',
        'zero_cot_target_delta_vs_baseline','actual_target_delta_vs_baseline','target_inflation_limit_tokens')
    rendered=audit.get('rendered_input_difference',{})
    ranges=rendered.get('retained_reasoning_token_ranges')
    expected_ranges=row['metadata']['cot_mask_audit'].get('retained_token_ranges')
    retained_ids=[]
    if isinstance(ranges,list):
        for item in ranges:
            if (not isinstance(item,list) or len(item)!=2
                    or any(type(value)is not int for value in item)
                    or not 0<=item[0]<item[1]<=len(row['input_ids'])):
                raise ValueError('Rendered input difference has invalid retained token ranges')
            retained_ids.extend(row['input_ids'][item[0]:item[1]])
    if (audit.get('schema')!=CROSS_ARM_PARITY_SCHEMA or audit.get('verified') is not True
            or audit.get('adapter_divergences')!=[]
            or audit.get('baseline_messages_sha256')!=audit.get('masked_messages_sha256')
            or not _hash(audit.get('baseline_messages_sha256'))
            or any(type(audit.get(key)) is not int for key in integer_keys)
            or audit.get('target_inflation_limit_tokens')!=0
            or audit.get('target_sequence_policy')!='exact-prefix-after-262144-cutoff/v1'
            or audit.get('input_difference_policy')!='official-xhigh-system-and-filled-think-input-only/v1'
            or audit.get('actual_masked_input_tokens')!=len(row['input_ids'])
            or audit.get('actual_masked_target_tokens')!=actual
            or audit.get('masked_zero_cot_target_tokens',-1)>audit.get('baseline_target_tokens',-1)
            or audit.get('actual_masked_target_tokens',-1)>audit.get('masked_zero_cot_target_tokens',-1)
            or audit.get('zero_cot_target_delta_vs_baseline') !=
                audit.get('masked_zero_cot_target_tokens',0)-audit.get('baseline_target_tokens',0)
            or audit.get('actual_target_delta_vs_baseline') !=
                audit.get('actual_masked_target_tokens',0)-audit.get('baseline_target_tokens',0)
            or audit.get('actual_target_delta_vs_baseline',1)>0):
        raise ValueError('Row lacks strict baseline-visible cross-arm no-CoT parity')
    if (rendered.get('schema')!='qwen38-filled-input-difference-audit/v1'
            or rendered.get('verified') is not True
            or rendered.get('difference_policy')!='replace-exact-receipt-bound-filled-think-with-empty-think/v1'
            or any(not _hash(rendered.get(key)) for key in (
                'zero_cot_rendered_sha256','actual_rendered_sha256',
                'stripped_actual_rendered_sha256','retained_reasoning_token_ids_sha256'))
            or rendered['zero_cot_rendered_sha256']!=rendered['stripped_actual_rendered_sha256']
            or type(rendered.get('expected_reasoning_count'))is not int
            or rendered.get('expected_reasoning_count')!=rendered.get('located_reasoning_count')
            or rendered.get('expected_reasoning_count')!=len(
                row['metadata']['provenance']['generated_cot'].get('generated_gaps',[]))
            or ranges!=expected_ranges
            or rendered.get('retained_reasoning_token_count')!=len(retained_ids)
            or rendered.get('retained_reasoning_token_ids_sha256')!=hashlib.sha256(
                json.dumps(retained_ids,separators=(',',':'),sort_keys=True).encode()).hexdigest()
            or not isinstance(rendered.get('filled_rendered_segment_sha256'),list)
            or len(rendered['filled_rendered_segment_sha256'])!=rendered['expected_reasoning_count']
            or any(not _hash(value) for value in rendered['filled_rendered_segment_sha256'])):
        raise ValueError('Row lacks exact receipt-bound rendered-input difference proof')


def validate_omission_ledger(row):
    provenance=row['metadata']['provenance']['generated_cot']
    ledger=provenance.get('native_gap_omission',{})
    retained=provenance['generated_gaps'];inputs=ledger.get('input_generated_gaps');omitted=ledger.get('omitted_generated_gaps')
    if (ledger.get('policy')!='keep-native-leading-gap-omit-only-conflicting-cot/v4'
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
    semantic = audit.get('semantic_action_boundary', {}) if isinstance(audit, dict) else {}
    tools = audit.get('raw_tool_cardinality', {}) if isinstance(audit, dict) else {}
    if (not isinstance(audit, dict) or audit.get('schema') != 'qwen38-assistant-turn-coalescing/v2'
            or audit.get('normalizer_source_sha256') != renderer_identity['turn_boundary_normalizer_sha256']
            or audit.get('excluded') is not False or audit.get('exclusions') != []
            or metadata.get('normalization', {}).get('turn_boundary') != audit
            or not all(_hash(audit.get(k)) for k in ('input_messages_sha256', 'output_messages_sha256', 'boundaries_sha256'))):
        raise ValueError('Row lacks successful exact versioned turn-boundary normalization evidence')
    if (semantic.get('schema') != 'qwen38-assistant-action-boundary-audit/v1'
            or semantic.get('verified') is not True
            or semantic.get('output_messages_checked') != audit.get('output_messages')
            or any(type(semantic.get(key)) is not int or semantic[key] < 0 for key in (
                'combined_visible_action_messages', 'explicitly_split_announcement_call_pairs',
                'unbarriered_announcement_call_splits', 'explicit_boundary_merges'))
            or semantic.get('unbarriered_announcement_call_splits') != 0
            or semantic.get('explicit_boundary_merges') != 0):
        raise ValueError('Row lacks an independent assistant action-boundary audit')
    if (tools.get('schema') != 'qwen38-raw-tool-cardinality-audit/v1'
            or tools.get('verified') is not True
            or any(type(tools.get(key)) is not int or tools[key] < 0 for key in (
                'source_tool_calls', 'native_structured_tool_calls',
                'raw_tool_call_events',
                'intentionally_omitted_incomplete_terminal_tool_calls',
                'source_tool_results', 'native_tool_results',
                'raw_tool_call_event_content_bytes',
                'raw_tool_call_event_content_projected_bytes'))
            or tools['source_tool_calls'] != tools['native_structured_tool_calls']
            or tools['raw_tool_call_events'] != (tools['source_tool_calls']
                + tools['intentionally_omitted_incomplete_terminal_tool_calls'])
            or tools['source_tool_results'] != tools['native_tool_results']
            or tools['raw_tool_call_event_content_projected_bytes'] != 0
            or any(not _hash(tools.get(key)) for key in (
                'source_tool_call_ids_sha256', 'native_tool_call_ids_sha256',
                'source_tool_result_ids_sha256', 'native_tool_result_ids_sha256',
                'source_tool_call_payloads_sha256', 'native_tool_call_payloads_sha256',
                'source_tool_result_payloads_sha256', 'native_tool_result_payloads_sha256'))
            or tools['source_tool_call_ids_sha256'] != tools['native_tool_call_ids_sha256']
            or tools['source_tool_result_ids_sha256'] != tools['native_tool_result_ids_sha256']
            or tools['source_tool_call_payloads_sha256'] != tools['native_tool_call_payloads_sha256']
            or tools['source_tool_result_payloads_sha256'] != tools['native_tool_result_payloads_sha256']):
        raise ValueError('Row lacks independent raw-event tool cardinality validation')
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
    supplied = Path(path)
    if supplied.is_symlink():
        raise ValueError("Manifest must be a regular non-symlink file")
    path = supplied.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Manifest must be a regular file")
    children = list(path.parent.iterdir())
    if (any(child.is_symlink() or not child.is_file() for child in children)
            or len({child.name.casefold() for child in children}) != len(children)):
        raise ValueError("Every portable export member must be a regular non-symlink file without case collisions")
    manifest = _strict_json(path.read_bytes())
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
    parity=manifest.get('cross_arm_no_cot_parity',{})
    semantic_boundary = manifest.get('semantic_action_boundary_audit', {})
    raw_tools = manifest.get('raw_tool_cardinality_audit', {})
    token_counts=manifest.get('token_counts',{})
    counts=manifest.get('counts',{})
    if (not isinstance(counts, dict) or not counts
            or any(key not in {split + "/" + source for split in ("train", "validation")
                               for source in ("trace", "replay")}
                   or type(value) is not int or value < 0 for key, value in counts.items())
            or not isinstance(token_counts, dict) or set(token_counts) != set(ROW_TOKEN_TOTAL_KEYS)
            or any(type(value) is not int for value in token_counts.values())
            or any(token_counts[key] < 0 for key in set(ROW_TOKEN_TOTAL_KEYS) - {"cross_arm_target_delta_vs_baseline"})):
        raise ValueError("Manifest row/token aggregates have an invalid shape")
    accepted=sum(counts.values())
    if (semantic_boundary.get('schema') != 'qwen38-assistant-action-boundary-corpus-audit/v1'
            or semantic_boundary.get('verified') is not True
            or semantic_boundary.get('rows_verified') != accepted
            or any(type(semantic_boundary.get(key)) is not int or semantic_boundary[key] < 0
                   for key in ('combined_visible_action_messages',
                               'explicitly_split_announcement_call_pairs',
                               'unbarriered_announcement_call_splits',
                               'explicit_boundary_merges'))
            or semantic_boundary.get('unbarriered_announcement_call_splits') != 0
            or semantic_boundary.get('explicit_boundary_merges') != 0):
        raise ValueError('Manifest lacks full-corpus assistant action-boundary validation')
    raw_tool_integer_keys = (
        'rows_verified', 'source_tool_calls', 'native_structured_tool_calls',
        'raw_tool_call_events', 'intentionally_omitted_incomplete_terminal_tool_calls',
        'source_tool_results', 'native_tool_results',
        'raw_tool_call_event_content_bytes',
        'raw_tool_call_event_content_projected_bytes',
    )
    if (raw_tools.get('schema') != 'qwen38-raw-tool-cardinality-corpus-audit/v1'
            or raw_tools.get('verified') is not True
            or any(type(raw_tools.get(key)) is not int or raw_tools[key] < 0
                   for key in raw_tool_integer_keys)
            or raw_tools.get('rows_verified') != accepted
            or raw_tools.get('source_tool_calls') !=
                raw_tools.get('native_structured_tool_calls')
            or raw_tools.get('raw_tool_call_events') != (
                raw_tools.get('source_tool_calls', -1)
                + raw_tools.get('intentionally_omitted_incomplete_terminal_tool_calls', -1))
            or raw_tools.get('source_tool_results') != raw_tools.get('native_tool_results')
            or raw_tools.get('raw_tool_call_event_content_projected_bytes') != 0):
        raise ValueError('Manifest lacks full-corpus raw-event tool cardinality validation')
    parity_excluded_sources=manifest.get('cross_arm_parity_excluded_sources')
    parity_excluded_generations=manifest.get('cross_arm_parity_excluded_generations')
    if (parity.get('schema')!='qwen38-cross-arm-corpus-parity/v1' or parity.get('verified') is not True
            or accepted<=0 or parity.get('rows_verified')!=accepted
            or not isinstance(parity_excluded_sources,dict)
            or not isinstance(parity_excluded_generations,dict)
            or any(type(value) is not int or value<0 for value in parity_excluded_sources.values())
            or any(type(value) is not int or value<0 for value in parity_excluded_generations.values())
            or parity.get('adapter_divergence_rows')!=sum(parity_excluded_sources.values())
            or sum(parity_excluded_generations.values())!=0
            or parity.get('adapter_divergence_rows')!=0
            or parity.get('target_inflation_limit_tokens')!=0
            or parity.get('actual_target_tokens')!=token_counts.get('target_tokens')
            or parity.get('zero_cot_target_tokens')!=token_counts.get('cross_arm_zero_cot_target_tokens')
            or parity.get('baseline_target_tokens')!=token_counts.get('cross_arm_baseline_target_tokens')
            or token_counts.get('cross_arm_target_delta_vs_baseline')!=parity.get('actual_target_delta_vs_baseline')
            or parity.get('actual_target_delta_vs_baseline') !=
                parity.get('actual_target_tokens',0)-parity.get('baseline_target_tokens',0)
            or parity.get('zero_cot_target_delta_vs_baseline') !=
                parity.get('zero_cot_target_tokens',0)-parity.get('baseline_target_tokens',0)
            or parity.get('actual_target_tokens',1)>parity.get('zero_cot_target_tokens',0)
            or parity.get('zero_cot_target_tokens',1)>parity.get('baseline_target_tokens',0)):
        raise ValueError('Manifest lacks full-corpus baseline-visible target parity')
    trace_membership = manifest.get("trace_source_membership", {})
    journal = manifest.get("journal_cardinality_audit", {})
    build_runtime = manifest.get("build_runtime", {})
    unexpected_trace_failures = manifest.get("unexpected_trace_failures")
    if (build_runtime.get("schema") != "qwen38-native-gap-v4-read-only-build-inputs/v1"
            or build_runtime.get("verified") is not True
            or build_runtime.get("runtime_image") !=
                "sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee"
            or not isinstance(build_runtime.get("paths"), dict)
            or not build_runtime["paths"]):
        raise ValueError("Manifest lacks the exact read-only CPU-build runtime identity")
    from .prepare_masked_full import conversion_dependency_sha256
    if manifest.get("conversion_dependency_sha256") != conversion_dependency_sha256():
        raise ValueError("Transitive conversion dependency identity changed")
    if (journal.get("schema") != "qwen38-frozen-generation-cardinality/v1"
            or journal.get("verified") is not True
            or journal.get("review_pending_gaps") != manifest.get("frozen_valid_candidate_count")
            or journal.get("selected_candidates") != manifest.get("frozen_valid_candidate_count")
            or type(journal.get("generation_invalid_candidates_excluded")) is not int
            or journal["generation_invalid_candidates_excluded"] < 0
            or journal.get("all_candidates") != (journal.get("selected_candidates", -1)
                + journal["generation_invalid_candidates_excluded"])
            or journal.get("unexpected_state_candidates") != 0
            or any(journal.get(key) != 0 for key in (
                "orphan_candidates", "orphan_candidate_sources",
                "ambiguous_source_event_targets",
                "selected_candidates_missing_exact_generate_response",
                "all_candidates_missing_exact_generate_response",
                "candidate_gap_cardinality_failures",
                "review_pending_gaps_missing_candidate"))
            or not isinstance(journal.get("frozen_gap_states"), dict)
            or journal["frozen_gap_states"].get("review_pending") !=
                journal.get("selected_candidates")
            or journal["frozen_gap_states"].get("generation_invalid", 0) !=
                journal.get("generation_invalid_candidates_excluded")):
        raise ValueError("Manifest lacks exact frozen journal candidate cardinality")
    if (trace_membership.get("schema") != "qwen38-exact-trace-source-membership/v1"
            or trace_membership.get("verified") is not True
            or type(unexpected_trace_failures) is not int or unexpected_trace_failures < 0
            or any(type(trace_membership.get(key)) is not int or trace_membership[key] < 0
                   for key in ("frozen_trace_sources", "preaudited_excluded_trace_sources",
                               "selected_trace_sources", "processed_trace_sources",
                               "accepted_trace_sources", "failed_trace_sources"))
            or trace_membership["processed_trace_sources"] != trace_membership["selected_trace_sources"]
            or trace_membership["accepted_trace_sources"] + trace_membership["failed_trace_sources"]
                != trace_membership["processed_trace_sources"]
            or trace_membership["accepted_trace_sources"] != sum(
                counts.get(split + "/trace", 0) for split in ("train", "validation"))):
        raise ValueError("Manifest does not account for every selected frozen trace source")
    if manifest.get("full_export"):
        if (manifest.get("all_frozen_generations_accounted_for") is not True
                or manifest.get("all_train") is not True
                or manifest.get("internal_validation") is not False
                or manifest.get("group_split") !=
                    "all-source-train-only-capture-identity-for-missing-session/v1"
                or manifest.get("unexpected_trace_failures") != 0
                or trace_membership.get("full_frozen_coverage_required") is not True
                or trace_membership.get("preaudited_excluded_trace_sources") != 0
                or trace_membership["frozen_trace_sources"] != (
                    trace_membership["preaudited_excluded_trace_sources"]
                    + trace_membership["selected_trace_sources"])
                or manifest.get("gap_omission_policy") != "keep-native-leading-gap-omit-only-conflicting-cot/v4"
                or type(manifest.get("omitted_valid_candidate_count")) is not int
                or manifest["omitted_valid_candidate_count"] < 0
                or manifest.get("turn_boundary_exclusion_policy") != "whole-source-exclusion-no-cot-relocation/v2"
                or manifest.get("native_render_exclusion_policy") != "unchanged-native-shape-guards-whole-source/v2"
                or manifest.get("included_valid_candidate_count") != manifest["unreviewed_gap_count"]
                or manifest.get("all_frozen_generations_bound_before_truncation") is not
                    (manifest.get("included_valid_candidate_count") == manifest.get("frozen_valid_candidate_count"))
                or type(manifest.get("bound_generated_gaps_wholly_after_cutoff")) is not int
                or manifest["bound_generated_gaps_wholly_after_cutoff"] != (
                    token_counts.get("filled_gaps", -1) - token_counts.get("retained_filled_gaps", -1))
                or manifest.get("all_bound_generated_gaps_retain_at_least_one_token") is not
                    (token_counts.get("filled_gaps") == token_counts.get("retained_filled_gaps"))
                or manifest.get("all_frozen_generations_included_before_cutoff") is not (
                    manifest.get("all_frozen_generations_bound_before_truncation") is True
                    and manifest.get("all_bound_generated_gaps_retain_at_least_one_token") is True)
                or manifest.get("included_valid_candidate_count", -1) + manifest.get("omitted_valid_candidate_count", -1) + manifest.get("excluded_valid_candidate_count", -1)
                    != manifest.get("frozen_valid_candidate_count")):
            raise ValueError("Corrected full export does not account for every frozen generated gap")
        for key in ("turn_boundary_excluded_sources", "turn_boundary_excluded_generations", "native_render_excluded_sources", "native_render_excluded_generations", "duplicate_excluded_sources", "duplicate_excluded_generations"):
            if (not isinstance(manifest.get(key), dict)
                    or any(type(value) is not int or value < 0 for value in manifest[key].values())):
                raise ValueError("Turn-boundary whole-source exclusions need explicit nonnegative counts")
    dynamic = manifest.get("dynamic_trace_exclusion_audit", {})
    dynamic_policy = manifest.get("dynamic_trace_exclusion_policy")
    dynamic_sources = sum(sum(manifest.get(key, {}).values()) for key in (
        "turn_boundary_excluded_sources", "native_render_excluded_sources",
        "duplicate_excluded_sources"))
    dynamic_gaps = sum(sum(manifest.get(key, {}).values()) for key in (
        "turn_boundary_excluded_generations", "native_render_excluded_generations",
        "duplicate_excluded_generations"))
    if trace_membership["failed_trace_sources"] != (
            dynamic_sources + parity_excluded_sources.get("trace", 0)
            + unexpected_trace_failures):
        raise ValueError("Trace failure categories differ from exact source membership")
    if (dynamic.get("schema") != "qwen38-exact-dynamic-trace-exclusions/v2"
            or dynamic.get("verified") is not True
            or dynamic.get("expected_record_count") != dynamic_sources
            or dynamic.get("observed_record_count") != dynamic_sources
            or dynamic.get("expected_generated_gap_count") != dynamic_gaps
            or dynamic.get("observed_generated_gap_count") != dynamic_gaps
            or dynamic.get("candidate_path") != "dynamic-trace-exclusions.candidate.json"
            or not _hash(dynamic.get("candidate_sha256"))):
        raise ValueError("Manifest lacks exact allowlisted dynamic trace exclusions")
    if dynamic_sources:
        if (not isinstance(dynamic_policy, dict)
                or dynamic_policy.get("schema") != dynamic["schema"]
                or dynamic_policy.get("sha256") != dynamic.get("allowlist_sha256")
                or not _hash(dynamic_policy.get("sha256"))
                or type(dynamic_policy.get("record_count")) is not int
                or dynamic_policy["record_count"] < dynamic_sources
                or type(dynamic_policy.get("generated_gap_count")) is not int
                or dynamic_policy["generated_gap_count"] < dynamic_gaps):
            raise ValueError("Dynamic trace exclusions lack a hash-bound external allowlist")
    elif dynamic_policy is not None or dynamic.get("allowlist_sha256") is not None:
        # A bounded/preflight selection may project a full allowlist to zero;
        # in that case the policy identity is still explicit and hash-bound.
        if (not isinstance(dynamic_policy, dict)
                or dynamic_policy.get("sha256") != dynamic.get("allowlist_sha256")
                or not _hash(dynamic_policy.get("sha256"))):
            raise ValueError("Empty dynamic exclusion projection has invalid policy provenance")
    replay=manifest.get('replay_source_membership',{})
    accepted_replay=sum(manifest.get('counts',{}).get(split+'/replay',0)
                        for split in ('train','validation'))
    if (replay.get('schema')!='qwen38-exact-replay-membership/v1'
            or replay.get('verified') is not True
            or type(replay.get('expected_selected_sources')) is not int
            or replay.get('expected_selected_sources')<0
            or replay.get('accepted_sources')!=accepted_replay
            or replay.get('accepted_sources')!=replay.get('expected_selected_sources')
            or replay.get('excluded_sources')!={}
            or replay.get('excluded_source_count')!=0):
        raise ValueError('Manifest lacks exact hash-bound replay source membership')
    stability = manifest.get("end_of_build_input_stability", {})
    if (stability.get("schema") != "qwen38-end-of-build-input-stability/v1"
            or stability.get("verified") is not True
            or stability.get("snapshot_rehashed") is not True
            or stability.get("source_rehashed") is not True
            or stability.get("renderer_and_policy_inputs_rehashed") is not True
            or type(stability.get("selected_replay_files_rehashed")) is not int
            or stability["selected_replay_files_rehashed"] != replay["accepted_sources"]
            or type(stability.get("replay_inventories_rehashed")) is not int
            or stability["replay_inventories_rehashed"] < 1):
        raise ValueError("Manifest lacks end-of-build frozen-input revalidation")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or "train" not in splits or set(splits) - {"train", "validation"}:
        raise ValueError("Manifest requires explicit train and optional validation splits")
    if manifest.get("full_export") and set(splits) != {"train"}:
        raise ValueError("The full manager-authorized mix must contain only the train split")
    seen = set()
    seen_casefold = set()
    for split, shards in splits.items():
        if not isinstance(shards, list) or not shards:
            raise ValueError("Every declared split must contain shards")
        for shard in shards:
            if (not isinstance(shard, dict) or not isinstance(shard.get("path"), str)
                    or not shard["path"] or Path(shard["path"]).name != shard["path"]
                    or "/" in shard["path"] or "\\" in shard["path"]
                    or Path(shard["path"]).is_absolute() or shard["path"].casefold() in seen_casefold
                    or not _hash(shard.get("sha256"))
                    or type(shard.get("rows")) is not int or shard["rows"] < 1
                    or type(shard.get("bytes")) is not int or shard["bytes"] < 1):
                raise ValueError("Every shard needs a path, SHA256, row count and byte count")
            supplied = path.parent / shard["path"]
            if supplied.is_symlink() or not supplied.is_file():
                raise ValueError("Declared training shard must be a regular non-symlink file")
            resolved = supplied.resolve(strict=True)
            if resolved.parent != path.parent:
                raise ValueError("Training shard escapes its portable export directory")
            if resolved in seen:
                raise ValueError("Duplicate shard would repeat data or cross splits")
            seen.add(resolved)
            seen_casefold.add(shard["path"].casefold())
    return manifest


def _row_token_totals(row):
    sequence = row["metadata"]["sequence_audit"]
    cot = row["metadata"]["cot_mask_audit"]
    parity = row["metadata"]["provenance"]["generated_cot"]["cross_arm_no_cot_parity"]
    return {
        "input_tokens": len(row["input_ids"]),
        "target_tokens": sum(label != -100 for label in row["labels"]),
        "cross_arm_baseline_input_tokens": parity["baseline_input_tokens"],
        "cross_arm_zero_cot_input_tokens": parity["masked_zero_cot_input_tokens"],
        "cross_arm_baseline_target_tokens": parity["baseline_target_tokens"],
        "cross_arm_zero_cot_target_tokens": parity["masked_zero_cot_target_tokens"],
        "cross_arm_target_delta_vs_baseline": parity["actual_target_delta_vs_baseline"],
        "filled_gaps": cot["filled_gap_count"],
        "retained_filled_gaps": cot["retained_filled_gap_count"],
        "dropped_input_tokens": sequence["dropped_input_tokens"],
        "dropped_target_tokens": sequence["dropped_supervised_tokens"],
        "omitted_gaps": cot["omitted_generation_count"],
        "bound_input_gaps": cot["bound_input_generation_count"],
    }


def _scan_shard(path, expected, identity, *, split=None, all_train=True,
                expected_rows=None, expected_bytes=None):
    refs, groups, h, gaps = [], set(), hashlib.sha256(), 0
    seen = set()
    omitted_gaps = 0
    counts, token_totals = {}, {key: 0 for key in ROW_TOKEN_TOTAL_KEYS}
    boundary_totals = {
        "rows_verified": 0, "combined_visible_action_messages": 0,
        "explicitly_split_announcement_call_pairs": 0,
        "unbarriered_announcement_call_splits": 0, "explicit_boundary_merges": 0,
    }
    tool_totals = {
        "rows_verified": 0, "source_tool_calls": 0,
        "native_structured_tool_calls": 0, "source_tool_results": 0,
        "raw_tool_call_events": 0,
        "intentionally_omitted_incomplete_terminal_tool_calls": 0,
        "native_tool_results": 0, "raw_tool_call_event_content_bytes": 0,
        "raw_tool_call_event_content_projected_bytes": 0,
    }
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            h.update(line)
            if not line.strip():
                raise ValueError("Blank lines are not training records")
            row = _strict_json(line)
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
            count_key = split + "/" + row["source"] if split is not None else row["source"]
            counts[count_key] = counts.get(count_key, 0) + 1
            for key, value in _row_token_totals(row).items():
                token_totals[key] += value
            semantic = row["metadata"]["turn_boundary_audit"]["semantic_action_boundary"]
            boundary_totals["rows_verified"] += int(semantic["verified"])
            for key in set(boundary_totals) - {"rows_verified"}:
                boundary_totals[key] += semantic[key]
            tools = row["metadata"]["turn_boundary_audit"]["raw_tool_cardinality"]
            tool_totals["rows_verified"] += int(tools["verified"])
            for key in set(tool_totals) - {"rows_verified"}:
                tool_totals[key] += tools[key]
            refs.append((path, offset, len(line), hashlib.sha256(line).hexdigest(), len(row["input_ids"])))
            groups.add("group:" + row["group_id"])
            groups.add("source:" + row["metadata"]["provenance"]["generated_cot"]["source_digest"])
    if h.hexdigest() != expected:
        raise ValueError("Training shard identity changed")
    if not refs:
        raise ValueError("Empty training shard")
    if expected_rows is not None and len(refs) != expected_rows:
        raise ValueError("Manifest shard row count differs from actual immutable rows")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError("Manifest shard byte count differs from the immutable shard")
    return refs, groups, gaps, omitted_gaps, counts, token_totals, boundary_totals, tool_totals


def validator_identity():
    from training.qwen38_no_cot import nemo_data
    return {"validator_sha256": digest_file(__file__),
            "shared_shift_collator_sha256": digest_file(nemo_data.__file__)}


def _scan_index_job(job):
    split, relative, path, expected, identity, all_train, expected_rows, expected_bytes = job
    return split, _scan_shard(path, expected, identity, split=split, all_train=all_train,
                              expected_rows=expected_rows, expected_bytes=expected_bytes)


def build_index(manifest_path, output, *, workers=1):
    """Audit all rows/splits and exclusively publish a new CPU offset index."""
    supplied_manifest = Path(manifest_path)
    if supplied_manifest.is_symlink():
        raise ValueError("Manifest cannot be a symlink")
    manifest_path = supplied_manifest.resolve(strict=True)
    raw_digest = digest_file(manifest_path)
    manifest = read_manifest(manifest_path)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('Bounded full-mask index concurrency required')
    jobs = [(split, entry['path'], (manifest_path.parent/entry['path']).resolve(), entry['sha256'],
             manifest['renderer_identity'], manifest.get('all_train') is True,
             entry.get('rows'), entry.get('bytes'))
            for split,entries in manifest['splits'].items() for entry in entries]
    splits = {split:[] for split in manifest['splits']}
    groups_by_split = {split:set() for split in splits}
    total_gaps = 0
    total_omitted = 0
    actual_counts = {}
    actual_tokens = {key: 0 for key in ROW_TOKEN_TOTAL_KEYS}
    actual_boundaries = {
        "rows_verified": 0, "combined_visible_action_messages": 0,
        "explicitly_split_announcement_call_pairs": 0,
        "unbarriered_announcement_call_splits": 0, "explicit_boundary_merges": 0,
    }
    actual_tools = {
        "rows_verified": 0, "source_tool_calls": 0,
        "native_structured_tool_calls": 0, "source_tool_results": 0,
        "raw_tool_call_events": 0,
        "intentionally_omitted_incomplete_terminal_tool_calls": 0,
        "native_tool_results": 0, "raw_tool_call_event_content_bytes": 0,
        "raw_tool_call_event_content_projected_bytes": 0,
    }
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs))) as pool:
        for job, (split, (refs, source_groups, gaps, omitted, counts, token_totals,
                          boundary_totals, tool_totals)) in zip(
                              jobs, pool.map(_scan_index_job, jobs)):
            _,relative,path,checksum,_,_,_,_ = job
            total_gaps += gaps
            total_omitted += omitted
            for key, value in counts.items():
                actual_counts[key] = actual_counts.get(key, 0) + value
            for key, value in token_totals.items():
                actual_tokens[key] += value
            for key, value in boundary_totals.items():
                actual_boundaries[key] += value
            for key, value in tool_totals.items():
                actual_tools[key] += value
            groups = groups_by_split[split]
            duplicates = {value for value in source_groups if value.startswith(('normalized:','retained:'))}
            if groups & duplicates:
                raise ValueError('Duplicate normalized or retained sequence across masked shards')
            groups.update(source_groups)
            splits[split].append({'path':relative,'sha256':checksum,'size_bytes':path.stat().st_size,
                'rows':[list(ref[1:]) for ref in refs]})
    if groups_by_split.get("train", set()) & groups_by_split.get("validation", set()):
        raise ValueError("A source group crosses train/validation")
    if actual_counts != manifest.get("counts"):
        raise ValueError("Manifest source/split row counts differ from actual shard contents")
    if actual_tokens != manifest.get("token_counts"):
        raise ValueError("Manifest aggregate token counts differ from actual shard contents")
    expected_boundaries = {key: manifest["semantic_action_boundary_audit"][key]
                           for key in actual_boundaries}
    if actual_boundaries != expected_boundaries:
        raise ValueError("Manifest semantic action-boundary totals differ from actual rows")
    expected_tools = {key: manifest["raw_tool_cardinality_audit"][key]
                      for key in actual_tools}
    if actual_tools != expected_tools:
        raise ValueError("Manifest raw-event tool-cardinality totals differ from actual rows")
    if total_omitted != manifest["omitted_valid_candidate_count"]:
        raise ValueError("Manifest omitted-gap count differs from actual row ledgers")
    if total_gaps != manifest["unreviewed_gap_count"]:
        raise ValueError("Manifest unreviewed gap count differs from the actual training rows")
    if digest_file(manifest_path) != raw_digest:
        raise ValueError("Manifest changed during audit")
    result = {"schema": INDEX_SCHEMA, "training_contract": TRAINING_CONTRACT,
              "manifest_sha256": raw_digest, "renderer_identity": manifest["renderer_identity"],
              **validator_identity(), "seq_len": MAX_TOKENS, "splits": splits,
              "split_group_overlap": 0, "counts": actual_counts, "token_counts": actual_tokens,
              "semantic_action_boundary_totals": actual_boundaries,
              "raw_tool_cardinality_totals": actual_tools,
              "unreviewed_gap_count": total_gaps, "omitted_valid_candidate_count":total_omitted}
    with Path(output).open("x") as stream:
        stream.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"index": str(Path(output).resolve()), "sha256": digest_file(output),
            "rows": {split: sum(len(shard["rows"]) for shard in shards) for split, shards in splits.items()},
            "unreviewed_gap_count": total_gaps, "omitted_valid_candidate_count":total_omitted}


class GeneratedMaskedTokenDataset:
    """Immutable map dataset with a separate, audited experimental generated-CoT index.

    Passing only a JSONL filename or disabling provenance is unsupported. The
    index is required so split-overlap auditing and full-corpus row validation
    happen before a training process starts. Each fetched row is checked again.
    """

    def __init__(self, path_or_dataset, *, index_path, index_sha256,
                 complete_path=None, complete_sha256=None, split="train",
                 seq_len=MAX_TOKENS, require_provenance=True, tokenizer=None,
                 max_input_tokens=None, order="source", max_samples=None):
        del tokenizer
        if seq_len != MAX_TOKENS or require_provenance is not True:
            raise ValueError("Experimental generated-CoT requires exact 262144 context and provenance")
        if not _hash(index_sha256):
            raise ValueError("An explicit experimental generated-CoT index digest is required")
        supplied_manifest = Path(path_or_dataset)
        if supplied_manifest.is_symlink():
            raise ValueError("Manifest cannot be a symlink")
        manifest = read_manifest(supplied_manifest)
        manifest_path = supplied_manifest.resolve(strict=True)
        if complete_path is None or complete_sha256 is None:
            raise ValueError("Atomic COMPLETE identity is required before any training row can load")
        read_complete(complete_path, complete_sha256, manifest_path=manifest_path,
                      index_path=index_path, index_sha256=index_sha256)
        raw = Path(index_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != index_sha256:
            raise ValueError("Experimental generated-CoT index identity changed")
        index = _strict_json(raw)
        expected = {"schema": INDEX_SCHEMA, "training_contract": TRAINING_CONTRACT,
                    "manifest_sha256": digest_file(manifest_path), "seq_len": MAX_TOKENS,
                    "renderer_identity": manifest["renderer_identity"], "split_group_overlap": 0,
                    "counts": manifest["counts"], "token_counts": manifest["token_counts"],
                    "semantic_action_boundary_totals": {
                        key: manifest["semantic_action_boundary_audit"][key]
                        for key in ("rows_verified", "combined_visible_action_messages",
                                    "explicitly_split_announcement_call_pairs",
                                    "unbarriered_announcement_call_splits",
                                    "explicit_boundary_merges")},
                    "raw_tool_cardinality_totals": {
                        key: manifest["raw_tool_cardinality_audit"][key]
                        for key in ("rows_verified", "source_tool_calls",
                                    "native_structured_tool_calls", "raw_tool_call_events",
                                    "intentionally_omitted_incomplete_terminal_tool_calls",
                                    "source_tool_results",
                                    "native_tool_results", "raw_tool_call_event_content_bytes",
                                    "raw_tool_call_event_content_projected_bytes")},
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
            if (cached.get("path") != entry["path"] or cached.get("sha256") != entry["sha256"]
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
        expected_split_rows = sum(value for key, value in manifest["counts"].items()
                                  if key.startswith(split + "/"))
        if len(self.refs) != expected_split_rows:
            raise ValueError("Index split row count differs from the manifest and shard audit")
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
        row = _strict_json(raw)
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
