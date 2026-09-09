"""Synthetic receipt-bound CoT tests; no teacher or GPU requests."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
from cot_filler.core import digest
from training.qwen38_cot_experimental.test_render import fixture, ASSETS, selected
from training.qwen38_cot_experimental.render import Qwen38ExperimentalCotRenderer
from .masked import Qwen38TurnBoundaryMaskedRenderer, MaskedBoundaryExclusion, training_row, MASK_POLICY


def events():
    return [
        {'event_id':'u','kind':'message','role':'user','content':'Inspect config.json.'},
        {'event_id':'a','kind':'message','role':'assistant','content':'I will inspect the configuration.'},
        {'event_id':'c','kind':'tool_call','role':'assistant','name':'exec_command','call_id':'c1',
         'data':{'block':{'type':'function_call','arguments':'{"cmd":"cat config.json"}'}}},
        {'event_id':'o','kind':'tool_result','role':'tool','call_id':'c1','content':'{"version":3}','data':{'block':{}}},
        {'event_id':'f','kind':'message','role':'assistant','content':'The version is 3.'},
    ]


@pytest.fixture(scope='module')
def renderer():
    return Qwen38TurnBoundaryMaskedRenderer(ASSETS)


def test_first_cot_stays_at_original_gap_and_one_eot_before_observation(renderer):
    args=fixture(events(),targets=['a'])
    before=deepcopy(args)
    result=renderer.render_generated_trace(*args)
    visible=selected(renderer,result)
    assert visible.count('<|im_end|>')==2
    assert 'I will inspect the configuration.' in visible and '<tool_call>' in visible
    assert args[1][0]['candidate']['text'] not in visible
    assert args[1][0]['candidate']['text'] in result['full_rendered_text']
    assert result['full_rendered_text'].count(args[1][0]['candidate']['text'])==1
    start,end=result['cot_mask_audit']['retained_token_ranges'][0]
    assert all(x==-100 for x in result['labels'][start:end])
    p=result['cot_provenance']
    assert p['event_message_mapping']==[['u'],['a','c'],['o'],['f']]
    assert p['original_event_message_mapping']==[['u'],['a'],['c'],['o'],['f']]
    assert p['generated_gaps'][0]['prefix_hash']==args[1][0]['gap']['prefix_hash']
    assert p['reasoning_relocated'] is False and p['original_gap_positions_preserved'] is True
    assert p['reviewed'] is False and p['future_information_checked'] is False
    assert args==before


def test_later_cot_excludes_entire_source_without_rewriting_receipt(renderer):
    args=fixture(events(),targets=['c']);before=deepcopy(args)
    with pytest.raises(MaskedBoundaryExclusion) as caught:
        renderer.render_generated_trace(*args)
    assert caught.value.code=='reasoning_after_visible_content'
    assert caught.value.audit['excluded_source_count']==1
    assert caught.value.audit['excluded_generated_gap_count']==1
    assert caught.value.audit['source_digest']==digest(args[0])
    assert caught.value.audit['reasoning_relocated'] is False
    assert args==before


def test_two_filled_gaps_are_not_concatenated_or_dropped(renderer):
    args=fixture(events(),targets=['a','c'])
    with pytest.raises(MaskedBoundaryExclusion) as caught:
        renderer.render_generated_trace(*args)
    assert caught.value.audit['excluded_generated_gap_count']==2


def test_explicit_source_turn_boundary_preserves_later_gap(renderer):
    src=events();src[1]['source_turn_id']='turn-a';src[2]['source_turn_id']='turn-b'
    args=fixture(src,targets=['c']);result=renderer.render_generated_trace(*args)
    assert result['cot_provenance']['event_message_mapping']==[['u'],['a'],['c'],['o'],['f']]
    assert selected(renderer,result).count('<|im_end|>')==3
    assert result['cot_mask_audit']['filled_gap_count']==1


def test_final_channel_barrier_is_derived_from_original_block(renderer):
    src=events();src[1]['data']={'block':{'type':'message','channel':'final'}}
    args=fixture(src,targets=['c']);result=renderer.render_generated_trace(*args)
    assert result['cot_provenance']['event_message_mapping'][1:3]==[['a'],['c']]


def test_conflicting_source_turn_metadata_excludes_source(renderer):
    src=events();src[1]['source_turn_id']='a';src[1]['data']={'turn_id':'b'}
    with pytest.raises(MaskedBoundaryExclusion) as caught:
        renderer.render_generated_trace(*fixture(src,targets=['a']))
    assert caught.value.code=='contradictory_source_boundary_metadata'


def test_reasoning_only_replay_is_removed_before_empty_eot_decision(renderer):
    messages=[{'role':'user','content':'Go.'},
        {'role':'assistant','content':'<think>Original hidden reasoning</think>','reasoning_content':'Other original'},
        {'role':'assistant','content':'Visible action.'}]
    original=deepcopy(messages)
    result=renderer.render_replay_messages(messages,source_digest=digest(messages),trace_id='replay')
    assert selected(renderer,result)=='Visible action.<|im_end|>'
    assert 'Original hidden reasoning' not in result['full_rendered_text']
    assert 'Other original' not in result['full_rendered_text']
    assert messages==original
    assert result['cot_mask_audit']['filled_gap_count']==0


def test_filled_reasoning_only_at_head_is_kept_before_call(renderer):
    src=events();src[1]['content']=''
    result=renderer.render_generated_trace(*fixture(src,targets=['a']))
    assert result['cot_mask_audit']['filled_gap_count']==1
    assert selected(renderer,result).count('<|im_end|>')==2
    assert result['cot_provenance']['event_message_mapping'][1]==['a','c']


def test_true_tool_observation_barrier_keeps_reasoning_after_observation(renderer):
    src=events();src[1]['channel']='final'
    result=renderer.render_generated_trace(*fixture(src,targets=['f']))
    assert result['cot_provenance']['filled_gap_message_placements'][0]['original_event_ids']==['f']
    assert result['full_rendered_text'].index('{"version":3}') < result['full_rendered_text'].index('I will inspect the function before choosing a repair.')


def test_cutoff_inside_reasoning_is_exact_and_does_not_append_eot(renderer):
    src=events();src[1]['channel']='final'
    args=fixture(src,targets=['c'])
    full=renderer.render_generated_trace(*args)
    lo,hi=full['cot_mask_audit']['retained_token_ranges'][0]
    limit=lo+4
    part=Qwen38TurnBoundaryMaskedRenderer(ASSETS,max_tokens=limit).render_generated_trace(*args)
    assert part['input_ids']==full['input_ids'][:limit]
    assert part['labels']==full['labels'][:limit]
    assert part['cot_mask_audit']['retained_token_ranges']==[[lo,limit]]
    assert part['sequence_audit']['eot_appended'] is False


def test_new_identity_and_row_remain_explicitly_unreviewed(renderer):
    result=renderer.render_generated_trace(*fixture(events(),targets=['a']))
    row=training_row(result,group_id='synthetic-session')
    assert row['metadata']['renderer_identity']['mask_policy']==MASK_POLICY
    assert row['metadata']['renderer_identity']['renderer_sha256']==hashlib.sha256(Path(__file__).with_name('masked.py').read_bytes()).hexdigest()
    assert row['metadata']['renderer_identity']['turn_boundary_normalizer_sha256']==hashlib.sha256(Path(__file__).with_name('normalize.py').read_bytes()).hexdigest()
    assert row['metadata']['provenance']['generated_cot']['reviewed'] is False
    assert row['metadata']['provenance']['generated_cot']['unreviewed_gap_count']==1
    assert row['metadata']['provenance']['generated_cot']['future_information_checked'] is False
    legacy=Qwen38ExperimentalCotRenderer(ASSETS).render_generated_trace(*fixture(events(),targets=['a']))
    with pytest.raises(ValueError):training_row(legacy,group_id='synthetic-session')


def test_original_private_block_channel_is_removed_separately_from_generated_cot(renderer):
    src=events();src.insert(1,{'event_id':'private','kind':'message','role':'assistant',
        'content':'ORIGINAL_PRIVATE_REASONING','data':{'block':{'type':'message','channel':'analysis'}}})
    result=renderer.render_generated_trace(*fixture(src,targets=['a']))
    assert 'ORIGINAL_PRIVATE_REASONING' not in result['full_rendered_text']
    assert result['cot_mask_audit']['filled_gap_count']==1
    assert selected(renderer,result).count('<|im_end|>')==2


def test_original_private_target_cannot_be_mislabelled_visible_generated_action(renderer):
    src=events();src[1]['data']={'block':{'type':'message','channel':'analysis'}}
    with pytest.raises(MaskedBoundaryExclusion) as caught:
        renderer.render_generated_trace(*fixture(src,targets=['a']))
    assert caught.value.code=='filled_gap_targets_original_private_event'


def test_omitted_tool_definition_retains_its_explicit_final_barrier(renderer):
    src=events();src.insert(2,{'event_id':'cap','kind':'tool_definition','role':'assistant','channel':'final',
        'content':'','data':{'block':{'type':'tool_definition'}}})
    result=renderer.render_generated_trace(*fixture(src,targets=['c']))
    assert selected(renderer,result).count('<|im_end|>')==3
    assert result['turn_boundary_audit']['dropped_empty_assistant_messages']==1
    assert result['cot_provenance']['original_event_message_mapping'][2]==['cap']


def test_typed_tool_call_on_analysis_channel_is_retained(renderer):
    src=events();src[2]['data']['block']['channel']='analysis'
    result=renderer.render_generated_trace(*fixture(src,targets=['a']))
    assert '<tool_call>' in selected(renderer,result)
    assert result['cot_provenance']['event_message_mapping'][1]==['a','c']


def test_original_top_level_private_projection_gate_is_not_bypassed(renderer):
    src=events();src[2]['channel']='analysis'
    with pytest.raises(ValueError, match='Visible projection'):
        renderer.render_generated_trace(*fixture(src,targets=['a']))


def test_replay_typed_action_on_analysis_channel_retains_action(renderer):
    messages=[{'role':'user','content':'Inspect config.'},
        {'role':'assistant','content':'I will inspect it.'},
        {'role':'assistant','channel':'analysis','content':'','tool_calls':[
            {'id':'c1','type':'function','function':{'name':'exec_command','arguments':{'cmd':'cat config.json'}}}]},
        {'role':'tool','tool_call_id':'c1','content':'done'}]
    result=renderer.render_replay_messages(messages,source_digest=digest(messages),trace_id='replay-analysis-action')
    assert '<tool_call>' in selected(renderer,result)
    assert selected(renderer,result).count('<|im_end|>')==1
