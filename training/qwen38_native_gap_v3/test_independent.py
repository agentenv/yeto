"""Independent per-gap omission, exact native bytes and provenance checks."""
from copy import deepcopy
import pytest
from training.qwen38_cot_experimental.test_render import fixture,ASSETS
from training.qwen38_turn_boundary_v2.test_masked import events
from training.qwen38_cot_masked.render import _canonical_messages
from training.qwen38_turn_boundary_v2.masked import Qwen38TurnBoundaryMaskedRenderer as ParentRenderer
from . import masked_data as data
from .masked import Qwen38TurnBoundaryMaskedRenderer,training_row,MaskedBoundaryExclusion
from .normalize import coalesce_assistant_continuations


@pytest.fixture(scope='module')
def renderer():return Qwen38TurnBoundaryMaskedRenderer(ASSETS)


def test_genuine_bound_later_gap_omission_keeps_exact_complete_source_and_zero_cot(renderer):
    args=fixture(events(),targets=['c']);before=deepcopy(args)
    result=renderer.render_generated_trace(*args)
    reference=ParentRenderer(ASSETS).render_generated_trace(args[0],[],args[2])
    assert args==before
    assert result['full_rendered_text']==reference['full_rendered_text']
    assert result['input_ids']==reference['input_ids'] and result['labels']==reference['labels']
    assert result['cot_mask_audit']['filled_gap_count']==0 and result['cot_mask_audit']['retained_token_ranges']==[]
    assert result['cot_provenance']['native_gap_omission']['omitted_generation_count']==1
    assert sum(label!=-100 for label in result['labels'])>1
    data.validate_row(training_row(result,group_id='same-original-session'),result['identity'])


def test_kept_native_reasoning_and_original_actions_match_official_template(renderer):
    args=fixture(events(),targets=['a','c']);before=deepcopy(args)
    result=renderer.render_generated_trace(*args)
    retained=[entry for entry in args[1] if entry['gap']['event_id']=='a']
    reference=ParentRenderer(ASSETS).render_generated_trace(args[0],retained,args[2])
    assert result['input_ids']==reference['input_ids'] and result['labels']==reference['labels']
    reasoning={'a':retained[0]['candidate']['text']}
    messages,_,_=_canonical_messages(args[0]['events'],reasoning)
    merged,_=coalesce_assistant_continuations(messages,reasoning_policy='preserve')
    official=renderer.tokenizer.apply_chat_template(merged,tokenize=False,add_generation_prompt=False,
        enable_thinking=True,preserve_thinking=True,reasoning_effort='xhigh')
    assert result['full_rendered_text']==official
    assert all(label==-100 for lo,hi in result['cot_mask_audit']['retained_token_ranges'] for label in result['labels'][lo:hi])
    ledger=result['cot_provenance']['native_gap_omission']
    assert ledger['bound_input_generation_count']==2 and ledger['retained_generation_count']==ledger['omitted_generation_count']==1
    assert result['identity']['inline_reasoning_inserted'] is False
    assert args==before


@pytest.mark.parametrize('mutation',['outside-index','wrong-event-list','wrong-event-id','changed-receipt','lost-gap'])
def test_omitted_reference_must_match_exact_original_mapping_and_receipt(renderer,mutation):
    result=renderer.render_generated_trace(*fixture(events(),targets=['c']))
    row=training_row(result,group_id='g');bad=deepcopy(row)
    p=bad['metadata']['provenance']['generated_cot'];ledger=p['native_gap_omission'];omitted=ledger['omitted_generated_gaps'][0]
    if mutation=='outside-index':omitted['original_message_index']=100000
    elif mutation=='wrong-event-list':omitted['original_event_ids']=['c','invented-event']
    elif mutation=='wrong-event-id':omitted['event_id']='invented-event'
    elif mutation=='changed-receipt':omitted['generation_receipt_sha256']='f'*64
    elif mutation=='lost-gap':ledger['omitted_generated_gaps']=[];ledger['omitted_generation_count']=0
    with pytest.raises(ValueError):data.validate_row(bad,result['identity'])


def test_unrepresentable_source_order_excludes_source_with_all_input_gap_count(renderer):
    source=events();source.insert(3,{'event_id':'late-comment','kind':'message','role':'assistant','content':'A source comment after the call.'})
    args=fixture(source,targets=['a','c']);before=deepcopy(args)
    with pytest.raises(MaskedBoundaryExclusion) as caught:renderer.render_generated_trace(*args)
    assert caught.value.code=='unrepresentable_text_after_tool'
    assert caught.value.audit['excluded_generated_gap_count']==2
    assert args==before
