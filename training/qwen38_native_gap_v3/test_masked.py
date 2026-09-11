from copy import deepcopy
import json
from pathlib import Path
import pytest
from training.qwen38_turn_boundary_v2.test_masked import events
from training.qwen38_turn_boundary_v2.render import Qwen38Renderer as BaselineRenderer
from training.qwen38_turn_boundary_v2.source_adapters import canonical_messages as baseline_messages
from training.qwen38_cot_experimental.test_render import fixture,ASSETS,selected
from .masked import Qwen38TurnBoundaryMaskedRenderer,training_row,MaskedBoundaryExclusion
from . import masked_data as data
from . import source_adapters
from .source_adapters import canonical_messages, rollout_messages, atif_messages

@pytest.fixture(scope='module')
def renderer():return Qwen38TurnBoundaryMaskedRenderer(ASSETS)

def active_ids(result):
    return [token for token,label in zip(result['input_ids'],result['labels']) if label!=-100]

def duplicated_tool_source():
    source=events()
    source[2]['content']='{"cmd":"cat config.json"}'
    source[2]['tool_calls']=[{'id':'non-authoritative','type':'function',
        'function':{'name':'exec_command','arguments':{'cmd':'echo wrong projection'}}}]
    return source

def test_zero_cot_tool_call_matches_baseline_except_xhigh_system_wrapper(renderer):
    source=duplicated_tool_source()
    masked=renderer.render_generated_trace(*fixture(source,targets=[]))
    messages,audit=baseline_messages(source)
    baseline=BaselineRenderer(ASSETS).render(messages,turn_boundary_audit=audit['turn_boundary_audit'])
    xhigh=('\u003c|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, '
        'validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and '
        'clarity in the final answer.\u003c|im_end|>\n')
    assert masked['full_rendered_text']==xhigh+baseline['full_rendered_text']
    assert masked['input_ids']==renderer.tokenizer.encode(xhigh+baseline['full_rendered_text'],add_special_tokens=False)
    assert active_ids(masked)==active_ids(baseline)
    assert selected(renderer,masked)==selected(renderer,baseline)
    assert '{"cmd":"cat config.json"}' not in masked['full_rendered_text']
    assert masked['full_rendered_text'].count('<parameter=cmd>\ncat config.json\n</parameter>')==1
    assert 'wrong projection' not in masked['full_rendered_text']


def test_direct_adapter_tool_cardinality_and_incomplete_terminal_accounting():
    source = duplicated_tool_source()
    messages, audit = canonical_messages(source)
    tools = audit['turn_boundary_audit']['raw_tool_cardinality']
    assert tools['verified'] is True
    assert tools['raw_tool_call_events'] == tools['source_tool_calls'] == 1
    assert tools['native_structured_tool_calls'] == 1
    assert tools['source_tool_results'] == tools['native_tool_results'] == 1
    assert tools['raw_tool_call_event_content_bytes'] > 0
    assert tools['raw_tool_call_event_content_projected_bytes'] == 0
    assert messages[1]['content'] == 'I will inspect the configuration.'

    incomplete = [
        {'event_id': 'u', 'kind': 'message', 'role': 'user', 'content': 'Inspect.'},
        {'event_id': 'a', 'kind': 'message', 'role': 'assistant', 'content': 'Starting.'},
        {'event_id': 'c', 'kind': 'tool_call', 'role': 'assistant', 'content': 'RAW',
         'source': {'pointer': '/response/output/2'},
         'data': {'block': {'type': 'function_call', 'name': 'exec_command', 'call_id': 'c1'}}},
    ]
    messages, audit = canonical_messages(incomplete)
    tools = audit['turn_boundary_audit']['raw_tool_cardinality']
    assert len(messages) == 2
    assert tools['raw_tool_call_events'] == 1
    assert tools['source_tool_calls'] == tools['native_structured_tool_calls'] == 0
    assert tools['intentionally_omitted_incomplete_terminal_tool_calls'] == 1


def test_replay_adapters_account_each_tool_call_and_result_once():
    rollout = [
        {'type': 'session_meta', 'payload': {'session_id': 's'}},
        {'type': 'response_item', 'payload': {'type': 'message', 'role': 'user', 'content': 'Inspect.'}},
        {'type': 'response_item', 'payload': {'type': 'function_call', 'name': 'exec_command',
                                              'call_id': 'c1', 'arguments': '{"cmd":"pwd"}'}},
        {'type': 'response_item', 'payload': {'type': 'function_call_output',
                                              'call_id': 'c1', 'output': 'ok'}},
    ]
    _, audit, _ = rollout_messages(rollout)
    tools = audit['turn_boundary_audit']['raw_tool_cardinality']
    assert tools['verified'] is True
    assert tools['raw_tool_call_events'] == tools['source_tool_calls'] == 1
    assert tools['source_tool_results'] == tools['native_tool_results'] == 1

    doc = {'session_id': 'a', 'steps': [
        {'source': 'user', 'message': 'Inspect.'},
        {'source': 'agent', 'message': '', 'tool_calls': [{
            'tool_call_id': 'c1', 'function_name': 'exec_command',
            'arguments': {'cmd': 'pwd'}}],
         'observation': {'results': [{'source_call_id': 'c1', 'content': 'ok'}]}},
    ]}
    _, audit, _ = atif_messages(doc)
    tools = audit['turn_boundary_audit']['raw_tool_cardinality']
    assert tools['verified'] is True
    assert tools['raw_tool_call_events'] == tools['source_tool_calls'] == 1
    assert tools['source_tool_results'] == tools['native_tool_results'] == 1


def test_adapter_rejects_tool_payload_mutation_during_coalescing(monkeypatch):
    real=source_adapters.coalesce_assistant_continuations
    def mutate(messages,**kwargs):
        output,audit=real(messages,**kwargs)
        for message in output:
            if message.get('tool_calls'):
                message['tool_calls'][0]['function']['arguments']={'cmd':'mutated'}
                break
        return output,audit
    monkeypatch.setattr(source_adapters,'coalesce_assistant_continuations',mutate)
    with pytest.raises(Exception,match='tool_cardinality_or_result_identity_changed'):
        source_adapters.canonical_messages(duplicated_tool_source())

def test_filled_cot_keeps_duplicate_projection_out_and_masks_every_cot_token(renderer):
    args=fixture(duplicated_tool_source(),targets=['a'])
    result=renderer.render_generated_trace(*args)
    candidate=args[1][0]['candidate']['text']
    assert '{"cmd":"cat config.json"}' not in result['full_rendered_text']
    assert result['full_rendered_text'].count('<parameter=cmd>\ncat config.json\n</parameter>')==1
    assert candidate in result['full_rendered_text'] and candidate not in selected(renderer,result)
    assert result['cot_mask_audit']['retained_token_ranges']
    for lo,hi in result['cot_mask_audit']['retained_token_ranges']:
        assert all(label==-100 for label in result['labels'][lo:hi])
    proof=result['cot_provenance']['cross_arm_no_cot_parity']['rendered_input_difference']
    assert proof['verified'] is True
    assert proof['zero_cot_rendered_sha256']==proof['stripped_actual_rendered_sha256']
    assert proof['expected_reasoning_count']==proof['located_reasoning_count']==1


def test_cross_arm_rejects_unsupervised_input_duplication(renderer):
    source=duplicated_tool_source()
    zero=renderer.render_generated_trace(*fixture(source,targets=[]))
    actual=deepcopy(zero)
    marker='<|im_start|>user\n'
    actual['full_rendered_text']=actual['full_rendered_text'].replace(
        marker,marker+'UNAUTHORIZED_DUPLICATED_INPUT\n',1)
    with pytest.raises(Exception,match='cross_arm_no_cot_parity_failed'):
        renderer._cross_arm_audit(
            baseline_messages(source)[0],
            baseline_messages(source)[1]['turn_boundary_audit'],
            zero,actual,require_exact_zero=False,retained_texts=())


def test_rendered_input_proof_matches_official_jinja_trim(renderer):
    source=duplicated_tool_source()
    args=fixture(source,targets=['a'])
    args[1][0]['candidate']['text']='  '+args[1][0]['candidate']['text']+'\n\t'
    # Keep the frozen candidate/response binding exact after adding whitespace.
    from cot_filler.core import canonical,digest
    from cot_filler.corpus_worker import gap_at
    gap=gap_at(args[0],1)
    text=args[1][0]['candidate']['text']
    args[1][0]['generation_receipt']['text']=text
    raw=json.loads(args[1][0]['generation_receipt']['data'])
    raw['candidate_hash']=digest(text)
    args[1][0]['generation_receipt']['data']=canonical(raw)
    args[1][0]['candidate']['text']=text
    args[1][0]['candidate']['data']=canonical(raw)
    result=renderer.render_generated_trace(*args)
    proof=result['cot_provenance']['cross_arm_no_cot_parity']['rendered_input_difference']
    assert proof['verified'] is True
    assert proof['zero_cot_rendered_sha256']==proof['stripped_actual_rendered_sha256']

def adapter_edge_sources():
    embedded=events()
    embedded[1]['tool_calls']=[{'id':'ignored-embedded','type':'function',
        'function':{'name':'exec_command','arguments':{'cmd':'echo ignored'}}}]
    boundary=[
        {'event_id':'u','kind':'message','role':'user','content':'Inspect config.json.'},
        {'event_id':'a1','kind':'message','role':'assistant','content':'First.','source':{'pointer':'/output/0/content/0'},'source_turn_id':'one'},
        {'event_id':'a2','kind':'message','role':'assistant','content':'Second.','source':{'pointer':'/output/0/content/1'},'source_turn_id':'two'},
    ]
    nonmapping_source=events();nonmapping_source[1]['source']='legacy-pointer'
    empty_and_definition=[
        {'event_id':'u0','kind':'message','role':'user','content':''},
        {'event_id':'d','kind':'message','role':'developer','content':'Inspect safely.'},
        {'event_id':'cap','kind':'tool_definition','role':'assistant','content':'','channel':'final',
            'data':{'block':{'type':'tool_definition'}}},
        {'event_id':'u','kind':'message','role':'user','content':'Inspect config.json.'},
        {'event_id':'a','kind':'message','role':'assistant','content':'Done.'},
    ]
    return [embedded,boundary,nonmapping_source,empty_and_definition]

@pytest.mark.parametrize('source',adapter_edge_sources())
def test_cross_arm_gate_accepts_every_supported_adapter_edge(renderer,source):
    result=renderer.render_generated_trace(*fixture(source,targets=[]))
    parity=result['cot_provenance']['cross_arm_no_cot_parity']
    assert parity['verified'] is True and parity['adapter_divergences']==[]
    assert parity['baseline_messages_sha256']==parity['masked_messages_sha256']
    assert parity['actual_masked_target_tokens']<=parity['masked_zero_cot_target_tokens']<=parity['baseline_target_tokens']

def test_training_row_and_loader_reject_changed_cross_arm_totals(renderer):
    result=renderer.render_generated_trace(*fixture(duplicated_tool_source(),targets=['a']))
    broken=deepcopy(result)
    broken['cot_provenance']['cross_arm_no_cot_parity']['actual_target_delta_vs_baseline']=1
    with pytest.raises(ValueError,match='cross-arm parity'):
        training_row(broken,group_id='g')
    row=training_row(result,group_id='g')
    row['metadata']['provenance']['generated_cot']['cross_arm_no_cot_parity']['actual_masked_target_tokens']+=1
    with pytest.raises(ValueError,match='cross-arm no-CoT parity'):
        data.validate_row(row,result['identity'])


@pytest.mark.parametrize('mutation',[
    lambda row: row['metadata']['sequence_audit'].__setitem__('original_input_tokens',
        row['metadata']['sequence_audit']['original_input_tokens']+1),
    lambda row: row['metadata']['sequence_audit'].__setitem__('retained_input_tokens',
        row['metadata']['sequence_audit']['retained_input_tokens']+1),
    lambda row: row['metadata']['sequence_audit'].__setitem__('dropped_input_tokens',
        row['metadata']['sequence_audit']['dropped_input_tokens']+1),
    lambda row: row['metadata']['sequence_audit'].__setitem__('original_supervised_tokens',
        row['metadata']['sequence_audit']['original_supervised_tokens']+1),
    lambda row: row['metadata']['sequence_audit'].__setitem__('retained_supervised_tokens',
        row['metadata']['sequence_audit']['retained_supervised_tokens']+1),
    lambda row: row['metadata']['sequence_audit'].__setitem__('dropped_supervised_tokens',
        row['metadata']['sequence_audit']['dropped_supervised_tokens']+1),
    lambda row: row['metadata']['sequence_audit'].__setitem__('truncated',
        not row['metadata']['sequence_audit']['truncated']),
    lambda row: row['metadata']['sequence_audit'].__setitem__('eot_appended',True),
    lambda row: row['metadata']['sequence_audit'].__setitem__('causal_shift_applied',True),
    lambda row: row['attention_mask'].__setitem__(0,0),
    lambda row: row['metadata'].__setitem__('retained_token_ids_sha256','0'*64),
])
def test_loader_recomputes_complete_sequence_audit(renderer,mutation):
    result=renderer.render_generated_trace(*fixture(duplicated_tool_source(),targets=['a']))
    row=training_row(result,group_id='g')
    mutation(row)
    with pytest.raises(ValueError,match='native-prefix truncation evidence|already been shifted|unpadded'):
        data.validate_row(row,result['identity'])

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


def test_omitted_same_pointer_fragment_rebuilds_exact_zero_cot_join(renderer):
    source=[
        {'event_id':'u','kind':'message','role':'user','content':'Continue the answer.'},
        {'event_id':'a1','kind':'message','role':'assistant','content':'First',
            'source':{'pointer':'/response/output/0/content/0'}},
        {'event_id':'a2','kind':'message','role':'assistant','content':'Second',
            'source':{'pointer':'/response/output/0/content/1'}},
    ]
    args=fixture(source,targets=['a2'])
    result=renderer.render_generated_trace(*args)
    zero=renderer.render_generated_trace(args[0],[],args[2])
    assert result['input_ids']==zero['input_ids']
    assert result['labels']==zero['labels']
    assert 'FirstSecond' in result['full_rendered_text']
    assert 'First\n\nSecond' not in result['full_rendered_text']
    ledger=result['cot_provenance']['native_gap_omission']
    assert ledger['omitted_generation_count']==1
    assert ledger['omitted_generated_gaps'][0]['original_event_ids']==['a1','a2']
    data.validate_row(training_row(result,group_id='g'),result['identity'])

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
