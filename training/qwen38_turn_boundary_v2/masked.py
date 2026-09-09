"""Versioned masked-CoT turn repair; incompatible reasoning placement excludes a source.

Receipt-bound generations stay at their original causal gap. Existing unreviewed
records remain explicitly unreviewed and unchecked for future information.
"""
from copy import deepcopy
from collections import Counter
import hashlib
from pathlib import Path

from cot_filler.core import digest
from training.qwen38_cot_experimental import render as generated_render
from training.qwen38_cot_experimental.generation_contract import bind_generations, ACCEPTANCE_POLICY
from training.qwen38_cot_masked import render as original_masked_render
from training.qwen38_no_cot.render import MAX_SEQUENCE_LENGTH, training_row as base_training_row
from . import normalize, source_adapters

VERSION = 'qwen3.8-xhigh-generated-cot-input-only-turn-boundary/v2'
MASK_POLICY = 'assistant_content_and_eos_only_generated_cot_masked_turn_boundary_v2'
ADAPTER_VERSION = 'original-events-to-native-generated-reasoning-preserving-turns/v2'
EXCLUSION_SCHEMA = 'yeta.masked-turn-boundary-exclusion/v1'
TEMPLATE_OPTIONS = deepcopy(generated_render.TEMPLATE_OPTIONS)
_METADATA_KEYS = {'channel', 'source_turn_id', 'barrier_before', 'barrier_after'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MaskedBoundaryExclusion(ValueError):
    """An entire source is excluded; callers must count its lost generated gaps."""
    def __init__(self, code, *, provenance, filled_gaps, audit=None):
        self.code = code
        self.audit = {
            'schema': EXCLUSION_SCHEMA, 'reason': code, 'excluded_source_count': 1,
            'excluded_generated_gap_count': filled_gaps,
            'source_digest': provenance['source_digest'], 'trace_id': provenance['trace_id'],
            'source_kind': provenance['source_kind'], 'reasoning_relocated': False,
            'turn_boundary_audit': deepcopy(audit or {}),
        }
        super().__init__(code)


def _event_metadata(event):
    data = event.get('data', {})
    block = data.get('block', data) if isinstance(data, dict) else data
    return source_adapters.boundary_metadata(event, block)


def canonical_boundary_metadata(messages, event_mapping, source_events):
    """Retain explicit barriers even when their original event has no visible text."""
    if len(messages) != len(event_mapping):
        raise ValueError('Canonical event mapping length mismatch')
    events = {event['event_id']: event for event in source_events}
    positions = {event['event_id']: index for index, event in enumerate(source_events)}
    covered = {event_id for ids in event_mapping for event_id in ids}
    omitted = {event['event_id']: _event_metadata(event) for event in source_events
               if event['event_id'] not in covered}
    omitted = {event_id: metadata for event_id, metadata in omitted.items()
               if metadata.get('barrier_before') or metadata.get('barrier_after')
               or metadata.get('source_turn_id')}
    entries = []
    for message, ids in zip(messages, event_mapping):
        item = deepcopy(message)
        indices = [positions[event_id] for event_id in ids]
        if indices != sorted(indices) or not indices:
            raise ValueError('Canonical event mapping is not ordered')
        if any(indices[0] < positions[event_id] < indices[-1] for event_id in omitted):
            raise ValueError('Previously combined source fragments cross an omitted explicit boundary')
        combined = {}
        for event_id in ids:
            for key, value in _event_metadata(events[event_id]).items():
                if key in combined and combined[key] != value:
                    raise ValueError('Combined source fragments cross an explicit turn boundary')
                combined[key] = value
        item.update(combined)
        entries.append((indices[0], item, list(ids)))
    for event_id, metadata in omitted.items():
        entries.append((positions[event_id], {'role': 'assistant', 'content': '', **metadata}, [event_id]))
    entries.sort(key=lambda entry: entry[0])
    return [entry[1] for entry in entries], [entry[2] for entry in entries]


class Qwen38TurnBoundaryMaskedRenderer(generated_render.Qwen38ExperimentalCotRenderer):
    def __init__(self, tokenizer_dir, *, max_tokens=MAX_SEQUENCE_LENGTH):
        super().__init__(tokenizer_dir, max_tokens=max_tokens)
        self.identity.update({
            'version': VERSION, 'mask_policy': MASK_POLICY, 'adapter_version': ADAPTER_VERSION,
            'renderer_sha256': _sha(__file__),
            'generated_renderer_sha256': _sha(generated_render.__file__),
            'turn_boundary_normalizer_sha256': _sha(normalize.__file__),
            'turn_boundary_source_adapter_sha256': _sha(source_adapters.__file__),
            'turn_boundary_policy': normalize.VERSION,
            'reasoning_placement_policy': 'original-gap-no-relocation-whole-source-exclusion/v1',
        })
        self.identity['adaptations'] = self.identity['adaptations'] + [
            'coalesce_assistant_continuations_before_native_render',
            'exclude_whole_source_if_filled_reasoning_would_move',
        ]

    def render_generated_trace(self, original_trace, generated_candidates, generation_config, *, boundaries=None):
        source, reasoning, refs = bind_generations(original_trace, generated_candidates, generation_config)
        provenance = {'source_digest': digest(source), 'trace_id': source['trace_id'], 'source_kind': 'trace',
            'generated_gaps': refs, 'generation_config_sha256': digest(generation_config),
            'event_message_mapping': [], 'original_gap_count': len(source['gap_targets']),
            'unfilled_gap_count': len(source['gap_targets']) - len(refs),
            'original_trace_complete': True, 'generation_lookahead_conditioned': bool(refs)}
        visible_events, removed_private = [], 0
        for event in source['events']:
            data = event.get('data', {})
            block = data.get('block', data) if isinstance(data, dict) else {}
            private = (event.get('kind') in source_adapters.original.PRIVATE
                or event.get('kind') == 'message' and (
                    event.get('channel') in source_adapters.original.PRIVATE
                    or isinstance(block, dict) and block.get('channel') in source_adapters.original.PRIVATE))
            if private:
                if event['event_id'] in reasoning:
                    raise MaskedBoundaryExclusion('filled_gap_targets_original_private_event',
                        provenance=provenance, filled_gaps=len(refs))
                visible_events.append({**deepcopy(event), 'kind': 'empty_message', 'role': 'assistant', 'content': ''})
                removed_private += 1
            else:
                visible_events.append(event)
        messages, mapping, counts = original_masked_render._canonical_messages(visible_events, reasoning)
        counts['original_private_events_removed_before_turn_mapping'] = removed_private
        provenance['event_message_mapping'] = mapping
        try:
            original_mapping = deepcopy(mapping)
            messages, mapping = canonical_boundary_metadata(messages, mapping, source['events'])
            if boundaries is not None:
                if not isinstance(boundaries, (list, tuple)) or len(boundaries) != len(original_mapping):
                    raise ValueError('Supplied boundaries must align with original canonical messages')
                supplied = {tuple(ids): metadata for ids, metadata in zip(original_mapping, boundaries)}
                boundaries = [deepcopy(supplied.get(tuple(ids), {})) for ids in mapping]
            provenance['event_message_mapping'] = mapping
        except ValueError as error:
            raise MaskedBoundaryExclusion('contradictory_source_boundary_metadata',
                provenance=provenance, filled_gaps=len(refs)) from error
        return self._render_coalesced(messages, refs, provenance, counts,
            reinsert_generated=True, boundaries=boundaries)

    def render_replay_messages(self, messages, *, source_digest, trace_id, boundaries=None):
        if not isinstance(source_digest, str) or len(source_digest) != 64 or not isinstance(trace_id, str) or not trace_id:
            raise ValueError('Replay requires original source digest and stable trace identity')
        provenance = {'source_digest': source_digest, 'trace_id': trace_id, 'source_kind': 'replay',
            'generated_gaps': [], 'generation_config_sha256': None,
            'generation_lookahead_conditioned': False, 'replay_original_reasoning_policy': 'removed-like-baseline'}
        return self._render_coalesced(deepcopy(messages), [], provenance, {},
            reinsert_generated=False, boundaries=boundaries)

    def _render_coalesced(self, messages, refs, provenance, counts, *, reinsert_generated, boundaries):
        # Apply the existing visible-content policy before deciding whether an
        # assistant is empty. Hidden original reasoning must not leave an EOT-only row.
        messages = deepcopy(messages)
        content_counts = Counter()
        for message in messages:
            message['content'] = self._baseline._content(
                message.get('content'), message.get('role'), content_counts)
        counts = {**counts, **{key: counts.get(key, 0) + value for key, value in content_counts.items()}}
        try:
            if boundaries is not None and (not isinstance(boundaries, (list, tuple)) or len(boundaries) != len(messages)):
                raise ValueError('Supplied boundaries must align with input messages')
            boundary_rows = []
            payloads = []
            for index, message in enumerate(messages):
                metadata = {key: deepcopy(value) for key, value in message.items() if key in _METADATA_KEYS}
                extra = boundaries[index] if boundaries is not None else {}
                if extra is not None and not isinstance(extra, dict):
                    raise ValueError('Supplied boundary must be an object')
                for key, value in (extra or {}).items():
                    if key in metadata and metadata[key] != value:
                        raise ValueError('Supplied boundary contradicts original source metadata')
                    metadata[key] = deepcopy(value)
                boundary_rows.append(metadata)
                payloads.append({key: deepcopy(value) for key, value in message.items() if key not in _METADATA_KEYS})
            merged, audit = normalize.coalesce_assistant_continuations(payloads, boundaries=boundary_rows,
                reasoning_policy='preserve' if reinsert_generated else 'drop')
        except normalize.ContinuationMappingError as error:
            raise MaskedBoundaryExclusion(error.code, provenance=provenance,
                filled_gaps=len(refs), audit=error.audit) from error
        except ValueError as error:
            raise MaskedBoundaryExclusion('conflicting_boundary_metadata',
                provenance=provenance, filled_gaps=len(refs)) from error
        mapping = audit['input_to_output']
        groups = audit['output_to_input']
        placements, occupied = [], set()
        for index, message in enumerate(messages):
            if not reinsert_generated or not message.get('reasoning_content'):
                continue
            output_index = mapping[index]
            if output_index is None or merged[output_index].get('reasoning_content') != message['reasoning_content']:
                raise MaskedBoundaryExclusion('filled_reasoning_removed_or_changed', provenance=provenance,
                    filled_gaps=len(refs), audit=audit)
            if output_index in occupied:
                raise MaskedBoundaryExclusion('multiple_filled_gaps_in_one_native_reasoning_block',
                    provenance=provenance, filled_gaps=len(refs), audit=audit)
            occupied.add(output_index)
            earlier = [i for i in groups[output_index] if i < index]
            if any(messages[i].get('content') or messages[i].get('tool_calls') for i in earlier):
                raise MaskedBoundaryExclusion('reasoning_after_visible_content', provenance=provenance,
                    filled_gaps=len(refs), audit=audit)
            placements.append({'original_message_index': index, 'merged_message_index': output_index,
                'reasoning_sha256': digest(message['reasoning_content'])})
        clean = [{k: deepcopy(v) for k, v in message.items() if k not in _METADATA_KEYS} for message in merged]
        provenance = deepcopy(provenance)
        original_mapping = provenance.get('event_message_mapping')
        if original_mapping is not None:
            provenance['original_event_message_mapping'] = original_mapping
            provenance['event_message_mapping'] = [
                [event for index in group for event in original_mapping[index]] for group in groups]
            for placement in placements:
                placement['original_event_ids'] = original_mapping[placement['original_message_index']]
        provenance.update({'turn_boundary_policy': normalize.VERSION, 'original_gap_positions_preserved': True,
            'reasoning_relocated': False, 'filled_gap_message_placements': placements})
        result = super()._render(clean, refs, provenance, counts, reinsert_generated=reinsert_generated)
        result['turn_boundary_audit'] = deepcopy(audit)
        result['normalization']['turn_boundary'] = deepcopy(audit)
        return result


def training_row(result, *, group_id, source='trace', provenance=None):
    if (result.get('identity', {}).get('mask_policy') != MASK_POLICY
            or result.get('cot_provenance', {}).get('acceptance_policy') != ACCEPTANCE_POLICY
            or result['cot_provenance'].get('source_kind') != source
            or result['cot_provenance'].get('reasoning_relocated') is not False):
        raise ValueError('Expected the versioned original-gap-preserving masked turn renderer result')
    row = base_training_row(result, group_id=group_id, source=source,
        provenance={**(provenance or {}), 'generated_cot': result['cot_provenance']})
    row['metadata']['cot_mask_audit'] = deepcopy(result['cot_mask_audit'])
    row['metadata']['turn_boundary_audit'] = deepcopy(result['turn_boundary_audit'])
    return row
