"""Preserve assistant continuation order without adding intermediate EOT targets."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = 'qwen38-assistant-turn-coalescing/v2'
FALLBACK_POLICY = 'contiguous-assistant-missing-turn-provenance/v2'
BOUNDARY_KEYS = {'channel', 'source_turn_id', 'barrier_before', 'barrier_after'}


def message_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class ContinuationMappingError(ValueError):
    """The entire input is excluded; the audit never releases partial output."""
    def __init__(self, code, audit):
        self.code, self.audit = code, deepcopy(audit)
        super().__init__(code)


def coalesce_assistant_continuations(messages, *, boundaries=None, reasoning_policy='drop'):
    """Return detached canonical messages and a content-bound, text-free audit.

    Missing explicit turn provenance uses the named contiguous-assistant
    fallback. A role, final channel, supplied barrier or differing known turn
    ID starts a new segment. Empty dropped assistants still affect segments.
    ``preserve`` refuses later reasoning after visible content/calls, and
    multiple reasoning blocks, instead of moving their original gaps.
    """
    if reasoning_policy not in {'drop', 'preserve'}:
        raise ValueError('reasoning_policy must be drop or preserve')
    if not isinstance(messages, (list, tuple)):
        raise TypeError('messages must be a sequence')
    original = deepcopy(list(messages))
    if boundaries is None:
        metadata = [{} for _ in original]
    else:
        if not isinstance(boundaries, (list, tuple)) or len(boundaries) != len(original):
            raise ValueError('boundaries must have one entry per message')
        if any(x is not None and not isinstance(x, dict) for x in boundaries):
            raise ValueError('Boundary entries must be mappings or null')
        metadata = [deepcopy(x) if x is not None else {} for x in boundaries]
    for meta in metadata:
        if not isinstance(meta, dict) or set(meta) - BOUNDARY_KEYS:
            raise ValueError('Unsupported boundary metadata')
        if any(key in meta and type(meta[key]) is not bool for key in ('barrier_before', 'barrier_after')):
            raise ValueError('Boundary flags must be booleans')
        if meta.get('channel') is not None and not isinstance(meta['channel'], str):
            raise ValueError('channel must be text or null')
        turn = meta.get('source_turn_id')
        if turn is not None and (type(turn) not in (str, int) or turn == ''):
            raise ValueError('source_turn_id must be a nonempty string, integer or null')
    out, mapping, reverse = [], [None] * len(original), []
    counts, barriers = Counter(), Counter()
    active = None
    segment_turn = None
    after_barrier = False
    audit = {'schema': VERSION, 'version': VERSION, 'reasoning_policy': reasoning_policy,
        'fallback_policy': FALLBACK_POLICY, 'input_messages': len(original),
        'original_messages': len(original), 'input_messages_sha256': message_digest(original),
        'boundaries_sha256': message_digest(metadata),
        'normalizer_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'exclusions': []}

    def finish(excluded=False):
        audit.update(output_messages=len(out), output_messages_sha256=None if excluded else message_digest(out),
            input_to_output=list(mapping), output_to_input=deepcopy(reverse),
            merged_message_count=counts['merged_message_count'], merges=counts['merged_message_count'],
            dropped_empty_assistant_messages=counts['dropped_empty_assistant_messages'],
            dropped_empty_boundary_effects=counts['dropped_empty_boundary_effects'],
            reasoning_fields_removed=counts['reasoning_fields_removed'],
            nonempty_reasoning_fields_removed=counts['nonempty_reasoning_fields_removed'],
            fallback_merges=counts['fallback_merges'], barrier_counts=dict(barriers),
            merge_groups=[{'output_index':i, 'input_indices':list(indices)}
                          for i, indices in enumerate(reverse) if len(indices) > 1],
            excluded=excluded)
        return deepcopy(audit)

    def fail(code, index):
        audit['exclusions'].append({'code': code, 'input_index': index,
                                   'previous_input_indices': list(reverse[active]) if active is not None else []})
        raise ContinuationMappingError(code, finish(True))

    for index, (raw, meta) in enumerate(zip(original, metadata)):
        if not isinstance(raw, dict) or raw.get('role') not in {'assistant', 'user', 'system', 'developer', 'tool'}:
            fail('unsupported_message_role', index)
        message = deepcopy(raw)
        role = message['role']
        if role != 'assistant':
            active = None; segment_turn = None; after_barrier = False
            barriers['role'] += 1
            mapping[index] = len(out); out.append(message); reverse.append([index])
            continue
        content = message.get('content')
        if content is None:
            content = ''
        if not isinstance(content, str):
            fail('unsupported_assistant_content', index)
        calls = message.get('tool_calls')
        if calls is None:
            calls = []
        if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
            fail('unsupported_tool_calls', index)
        reasoning = message.get('reasoning_content')
        if reasoning is not None and not isinstance(reasoning, str):
            fail('unsupported_reasoning_content', index)
        if reasoning_policy == 'drop':
            if 'reasoning_content' in message:
                counts['reasoning_fields_removed'] += 1
                counts['nonempty_reasoning_fields_removed'] += int(bool(reasoning and reasoning.strip()))
                del message['reasoning_content']
            reasoning = None
        if 'reasoning' in message or 'analysis' in message:
            fail('noncanonical_reasoning_field', index)
        final = meta.get('channel') == 'final'
        turn = meta.get('source_turn_id')
        reasons = []
        if after_barrier: reasons.append('explicit_or_final_after')
        if meta.get('barrier_before'): reasons.append('explicit_before')
        if final: reasons.append('final_channel')
        if turn is not None and segment_turn is not None and turn != segment_turn:
            reasons.append('different_source_turn')
        if reasons:
            active = None; segment_turn = None
            barriers.update(reasons)
        known_before = segment_turn
        if turn is not None:
            segment_turn = turn
        after_barrier = bool(meta.get('barrier_after') or final)
        nonempty_reasoning = bool(reasoning and reasoning.strip())
        if not content.strip() and not calls and not nonempty_reasoning:
            extra = set(message) - {'role', 'content', 'tool_calls', 'reasoning_content'}
            if any(message[key] not in (None, '', [], {}) for key in extra):
                fail('empty_assistant_has_unrepresented_metadata', index)
            counts['dropped_empty_assistant_messages'] += 1
            counts['dropped_empty_boundary_effects'] += int(bool(reasons or after_barrier or turn is not None))
            if after_barrier:
                active = None
            continue
        if active is None:
            mapping[index] = len(out); out.append(message); reverse.append([index]); active = len(out) - 1
            continue
        previous = out[active]
        previous_content = previous.get('content') or ''
        previous_calls = previous.get('tool_calls') or []
        previous_reasoning = previous.get('reasoning_content') or ''
        if reasoning_policy == 'preserve' and nonempty_reasoning:
            if previous_content.strip() or previous_calls:
                fail('reasoning_after_visible_content', index)
            if previous_reasoning.strip():
                fail('multiple_reasoning_blocks', index)
        if previous_calls and content.strip():
            fail('unrepresentable_text_after_tool', index)
        extra = (set(previous) | set(message)) - {'role', 'content', 'tool_calls', 'reasoning_content'}
        if any(previous.get(key) != message.get(key) for key in extra):
            fail('incompatible_assistant_metadata', index)
        if previous_content and content:
            previous['content'] = previous_content + '\n\n' + content
        elif content:
            previous['content'] = content
        if calls:
            previous['tool_calls'] = deepcopy(previous_calls) + deepcopy(calls)
        if reasoning_policy == 'preserve' and nonempty_reasoning:
            previous['reasoning_content'] = reasoning
        reverse[active].append(index); mapping[index] = active
        counts['merged_message_count'] += 1
        if known_before is None or turn is None:
            counts['fallback_merges'] += 1
    result_audit = finish()
    return deepcopy(out), result_audit
