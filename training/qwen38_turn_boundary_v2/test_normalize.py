"""Pure continuation semantics; no model, data export or remote calls."""
from copy import deepcopy

import pytest

from .normalize import (ContinuationMappingError, FALLBACK_POLICY, VERSION,
                        coalesce_assistant_continuations, message_digest)


def assistant(text='', **extra):
    return {'role': 'assistant', 'content': text, **extra}


def call(identity):
    return {'id': identity, 'type': 'function',
            'function': {'name': 'read', 'arguments': {'path': identity}}}


def test_text_then_calls_coalesces_exact_order_and_binds_audit():
    source = [{'role': 'user', 'content': 'inspect'}, assistant('First'), assistant('Then'),
              assistant(tool_calls=[call('a')]), assistant(tool_calls=[call('b')]),
              {'role': 'tool', 'tool_call_id': 'b', 'content': 'B'}, assistant('Done')]
    result, audit = coalesce_assistant_continuations(source)
    assert len(result) == 4
    assert result[1]['content'] == 'First\n\nThen'
    assert [x['id'] for x in result[1]['tool_calls']] == ['a', 'b']
    assert audit['input_to_output'] == [0, 1, 1, 1, 1, 2, 3]
    assert audit['output_to_input'] == [[0], [1, 2, 3, 4], [5], [6]]
    assert audit['merged_message_count'] == audit['merges'] == audit['fallback_merges'] == 3
    assert audit['input_messages_sha256'] == message_digest(source)
    assert audit['output_messages_sha256'] == message_digest(result)
    assert audit['schema'] == VERSION and audit['fallback_policy'] == FALLBACK_POLICY
    assert audit['excluded'] is False and audit['exclusions'] == []


def test_all_inputs_outputs_detached_including_nested_calls_and_audit():
    source = [assistant('Plan'), assistant(tool_calls=[call('a')])]
    boundaries = [{'source_turn_id': 't'}, {'source_turn_id': 't'}]
    before, before_boundaries = deepcopy(source), deepcopy(boundaries)
    result, audit = coalesce_assistant_continuations(source, boundaries=boundaries)
    result[0]['tool_calls'][0]['function']['arguments']['path'] = 'changed'
    audit['output_to_input'][0].append(99)
    assert source == before and boundaries == before_boundaries
    assert audit['input_to_output'] == [0, 0]


@pytest.mark.parametrize('role', ['user', 'system', 'developer', 'tool'])
def test_every_nonassistant_role_is_a_real_barrier(role):
    middle = {'role': role, 'content': 'observation'}
    result, audit = coalesce_assistant_continuations([assistant('A'), middle, assistant('B')])
    assert len(result) == 3 and audit['merges'] == 0


@pytest.mark.parametrize('boundary', [
    {'channel': 'final'}, {'barrier_before': True}, {'source_turn_id': 'next'}])
def test_explicit_before_boundary_preserves_separate_assistant(boundary):
    result, audit = coalesce_assistant_continuations(
        [assistant('A'), assistant('B')],
        boundaries=[{'source_turn_id': 'first'}, boundary])
    assert len(result) == 2 and audit['merges'] == 0


def test_final_is_a_barrier_on_both_sides():
    result, audit = coalesce_assistant_continuations(
        [assistant('A'), assistant('final'), assistant('B')],
        boundaries=[{}, {'channel': 'final'}, {}])
    assert len(result) == 3 and audit['merges'] == 0


def test_explicit_after_boundary_and_same_turn_merge_evidence():
    result, audit = coalesce_assistant_continuations(
        [assistant('A'), assistant('B'), assistant('C')],
        boundaries=[{'source_turn_id': 't'}, {'source_turn_id': 't', 'barrier_after': True},
                    {'source_turn_id': 't'}])
    assert [x['content'] for x in result] == ['A\n\nB', 'C']
    assert audit['merges'] == 1 and audit['fallback_merges'] == 0


def test_drop_empty_reasoning_retains_explicit_changed_turn_for_missing_next_turn():
    source = [assistant('A'), assistant(reasoning_content='private'), assistant(tool_calls=[call('a')])]
    result, audit = coalesce_assistant_continuations(source,
        boundaries=[{'source_turn_id': 'T1'}, {'source_turn_id': 'T2'}, {}])
    assert len(result) == 2 and 'tool_calls' not in result[0]
    assert audit['input_to_output'] == [0, None, 1]
    assert audit['dropped_empty_assistant_messages'] == 1
    assert audit['barrier_counts']['different_source_turn'] == 1


@pytest.mark.parametrize('boundary', [{'barrier_before': True}, {'barrier_after': True}, {'channel': 'final'}])
def test_dropped_empty_keeps_other_boundary_effects(boundary):
    result, audit = coalesce_assistant_continuations(
        [assistant('A'), assistant(reasoning_content='private'), assistant('B')],
        boundaries=[{}, boundary, {}])
    assert [x['content'] for x in result] == ['A', 'B']
    assert audit['dropped_empty_boundary_effects'] == 1


def test_reasoning_only_replay_creates_no_empty_eot_messages():
    source = [assistant(reasoning_content='one'), assistant('   '), assistant(reasoning_content='two')]
    result, audit = coalesce_assistant_continuations(source)
    assert result == [] and audit['input_to_output'] == [None, None, None]
    assert audit['dropped_empty_assistant_messages'] == 3
    assert audit['reasoning_fields_removed'] == audit['nonempty_reasoning_fields_removed'] == 2


def test_preserve_first_reasoning_without_moving_or_dropping_it():
    source = [assistant(reasoning_content='reason at original gap'), assistant('Action'),
              assistant(tool_calls=[call('a')])]
    result, audit = coalesce_assistant_continuations(source, reasoning_policy='preserve')
    assert len(result) == 1
    assert result[0]['reasoning_content'] == source[0]['reasoning_content']
    assert result[0]['content'] == 'Action'
    assert audit['reasoning_fields_removed'] == 0


@pytest.mark.parametrize('first', [assistant('visible'), assistant(tool_calls=[call('a')])])
def test_later_reasoning_cannot_move_across_visible_content(first):
    source = [first, assistant(reasoning_content='later original gap')]
    before = deepcopy(source)
    with pytest.raises(ContinuationMappingError) as caught:
        coalesce_assistant_continuations(source, reasoning_policy='preserve')
    assert caught.value.code == 'reasoning_after_visible_content'
    assert caught.value.audit['excluded'] is True
    assert caught.value.audit['output_messages_sha256'] is None
    assert caught.value.audit['exclusions'][0]['input_index'] == 1
    assert source == before


def test_two_nonempty_reasoning_blocks_cannot_be_relocated_or_concatenated():
    with pytest.raises(ContinuationMappingError, match='multiple_reasoning_blocks'):
        coalesce_assistant_continuations([assistant(reasoning_content='a'), assistant(reasoning_content='b')],
                                        reasoning_policy='preserve')


def test_explicit_turn_boundary_keeps_later_reasoning_at_original_message():
    source = [assistant('Earlier'), assistant('Later', reasoning_content='original later gap')]
    result, audit = coalesce_assistant_continuations(source,
        boundaries=[{'source_turn_id': 'one'}, {'source_turn_id': 'two'}], reasoning_policy='preserve')
    assert result == source and audit['merges'] == 0


def test_same_continuation_text_after_tool_fails_instead_of_reordering():
    with pytest.raises(ContinuationMappingError, match='unrepresentable_text_after_tool'):
        coalesce_assistant_continuations([assistant(tool_calls=[call('a')]), assistant('Later text')])


def test_real_boundary_allows_text_after_tool_in_separate_turn():
    source = [assistant(tool_calls=[call('a')]), assistant('Later text')]
    result, audit = coalesce_assistant_continuations(source, boundaries=[{'barrier_after': True}, {}])
    assert result == source and audit['merges'] == 0


@pytest.mark.parametrize('entry', [False, 0, '', [], 'turn'])
def test_invalid_boundary_entries_fail(entry):
    with pytest.raises(ValueError):
        coalesce_assistant_continuations([assistant('x')], boundaries=[entry])


@pytest.mark.parametrize('calls', [False, 0, '', {}, ['call']])
def test_invalid_tool_call_container_is_not_silently_emptied(calls):
    with pytest.raises(ContinuationMappingError, match='unsupported_tool_calls'):
        coalesce_assistant_continuations([assistant('x', tool_calls=calls)])


def test_unknown_reasoning_alias_fails_instead_of_discarding_content():
    with pytest.raises(ContinuationMappingError, match='noncanonical_reasoning_field'):
        coalesce_assistant_continuations([assistant('x', reasoning='private')], reasoning_policy='preserve')
