from copy import deepcopy
import json
from pathlib import Path
import pytest
from training.qwen38_turn_boundary_v2.test_masked import events
from training.qwen38_cot_experimental.test_render import fixture,ASSETS,selected
from .masked import Qwen38TurnBoundaryMaskedRenderer,training_row,MaskedBoundaryExclusion
from . import masked_data as data

@pytest.fixture(scope='module')
def renderer():return Qwen38TurnBoundaryMaskedRenderer(ASSETS)

def test_later_gap_omitted_but_original_trace_and_native_action_retained(renderer):
    args=fixture(events(),targets=['c']);before=deepcopy(args)
    result=renderer.render_generated_trace(*args);row=training_row(result,group_id='g')
    assert args==before and result['cot_mask_audit']['filled_gap_count']==0
    p=result['cot_provenance'];ledger=p['native_gap_omission']
    assert ledger['bound_input_generation_count']==ledger['omitted_generation_count']==1
    assert ledger['omitted_generated_gaps'][0]['event_id']=='c'
    assert ledger['omitted_generated_gaps'][0]['prefix_hash']==args[1][0]['gap']['prefix_hash']
    assert 'I will inspect the configuration.' in selected(renderer,result) and '<tool_call>' in selected(renderer,result)
    assert selected(renderer,result).count('<|im_end|>')==2
    data.validate_row(row,result['identity'])

def test_only_first_native_eligible_cot_is_kept_without_changing_receipts(renderer):
    args=fixture(events(),targets=['a','c']);before=deepcopy(args)
    result=renderer.render_generated_trace(*args);row=training_row(result,group_id='g')
    p=result['cot_provenance'];ledger=p['native_gap_omission']
    assert p['filled_gaps']==1 and ledger['omitted_generation_count']==1
    assert p['generated_gaps'][0]['event_id']=='a' and ledger['omitted_generated_gaps'][0]['event_id']=='c'
    assert args==before and p['reviewed'] is False and p['future_information_checked'] is False
    data.validate_row(row,result['identity'])
    bad=deepcopy(row);bad['metadata']['provenance']['generated_cot']['native_gap_omission']['input_generated_gaps'].pop()
    with pytest.raises(ValueError):data.validate_row(bad,result['identity'])

def test_true_original_turn_boundary_preserves_both_native_cots(renderer):
    source=events();source[1]['source_turn_id']='a';source[2]['source_turn_id']='b'
    result=renderer.render_generated_trace(*fixture(source,targets=['a','c']))
    assert result['cot_mask_audit']['filled_gap_count']==2 and result['cot_mask_audit']['omitted_generation_count']==0
    assert selected(renderer,result).count('<|im_end|>')==3
    data.validate_row(training_row(result,group_id='g'),result['identity'])

def test_text_after_tool_is_still_explicit_native_source_exclusion(renderer):
    source=events();source.insert(3,{'event_id':'after-call','kind':'message','role':'assistant','content':'More visible commentary.'})
    with pytest.raises(MaskedBoundaryExclusion) as caught:
        renderer.render_generated_trace(*fixture(source,targets=['a']))
    assert caught.value.code=='unrepresentable_text_after_tool'


def test_omission_result_exactly_matches_official_native_merged_replay(renderer):
    from training.qwen38_cot_masked.render import _canonical_messages
    from training.qwen38_cot_experimental.render import Qwen38ExperimentalCotRenderer
    from .normalize import coalesce_assistant_continuations
    from cot_filler.core import digest
    args=fixture(events(),targets=['c']);result=renderer.render_generated_trace(*args)
    messages,_,_=_canonical_messages(args[0]['events'],{})
    merged,_=coalesce_assistant_continuations(messages,reasoning_policy='drop')
    expected=Qwen38ExperimentalCotRenderer(ASSETS).render_replay_messages(merged,source_digest=digest(merged),trace_id='native-reference')
    assert result['full_rendered_text']==expected['full_rendered_text']
    assert result['input_ids']==expected['input_ids'] and result['labels']==expected['labels']


def test_omitted_reference_cannot_change_or_disappear_in_loader(renderer):
    result=renderer.render_generated_trace(*fixture(events(),targets=['c']))
    row=training_row(result,group_id='g')
    for key,value in [('prefix_hash','0'*64),('reason','unapproved_reason'),('original_message_index',-1)]:
        bad=deepcopy(row);bad['metadata']['provenance']['generated_cot']['native_gap_omission']['omitted_generated_gaps'][0][key]=value
        with pytest.raises(ValueError):data.validate_row(bad,result['identity'])


def test_multiple_leading_reasoning_fragments_keep_first_without_moving_second(renderer):
    source=events();source[1]['content']=''
    result=renderer.render_generated_trace(*fixture(source,targets=['a','c']))
    ledger=result['cot_provenance']['native_gap_omission']
    assert ledger['omitted_generated_gaps'][0]['reason']=='multiple_reasoning_blocks'
    assert result['cot_provenance']['generated_gaps'][0]['event_id']=='a'
    assert '<tool_call>' in selected(renderer,result)
    data.validate_row(training_row(result,group_id='g'),result['identity'])
