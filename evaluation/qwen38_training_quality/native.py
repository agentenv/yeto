"""Actual native-input parity checks independent of checkpoint task success."""
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid
from evaluation.tb2_checkpoint_xhigh import verify_native_proxy as shared
from . import protocol


def live(model,model_name,api_key,checkpoint,*,base_url='http://127.0.0.1:35810'):
    cpu=shared.cpu(model);_,tito=shared.runtime(model)
    session='quality-native-'+uuid.uuid4().hex
    tools=[{'type':'function','function':{'name':'quality_read','description':'Read a synthetic qualification value.',
        'parameters':{'type':'object','properties':{},'additionalProperties':False}}}]
    prefix=[{'role':'system','content':'This is a synthetic input-encoding check.'},
            {'role':'user','content':'Acknowledge the input briefly.'}]
    supplied={'role':'assistant','content':'','reasoning_content':'I will read the synthetic value.',
        'tool_calls':[{'id':'quality-fixed-call','type':'function','function':{'name':'quality_read','arguments':{}}}]}
    observation={'role':'tool','tool_call_id':'quality-fixed-call','content':'synthetic-input-value'}
    def post(route,body):
        request=urllib.request.Request(base_url+route,data=json.dumps(body).encode(),headers={
            'Content-Type':'application/json','Authorization':'Bearer '+api_key,
            'X-Session-Id':session,'X-Turn-Id':session+':0'})
        with urllib.request.urlopen(request,timeout=300) as response:return json.load(response)
    def generate(messages):
        return post('/v1/chat/completions',{'model':model_name,'messages':messages,'tools':tools,
            'max_tokens':256,'temperature':0.0,'stream':False})
    first=generate(prefix)
    assistant=first['choices'][0]['message']
    # The tool history below is an explicitly supplied synthetic fixture. No successful
    # model-chosen tool call or answer is required to admit weak checkpoints to evaluation.
    second_messages=prefix+[assistant,supplied,observation]
    generate(second_messages)
    finalized=post('/session/finalize',{'session_id':session})
    records=post('/trajectory/read',{'trajectory_id':session,'drain':False,'segment_view':'lineage'})['data']
    if len(records)!=1:raise ValueError('Expected one saved qualification lineage')
    record=records[0];tokens=record['tokens'];mask=record['full_loss_mask']
    if len(tokens)!=len(mask):raise ValueError('Malformed qualification token mask')
    starts=[i for i,value in enumerate(mask) if value and (i==0 or not mask[i-1])]
    if len(starts)!=2:raise ValueError('Expected two actual generated spans for parity')
    initial=tito._encode_text(tito._render_messages(prefix,add_generation_prompt=True,tools=tools))
    if tokens[:starts[0]]!=initial:raise ValueError('Actual initial input is not native xhigh')
    first_end=next(i for i in range(starts[0],len(mask)) if not mask[i])
    incremental=tito.merge_tokens(old_messages=prefix+[assistant],new_messages=second_messages,
        pretokenized_token_ids=tokens[:first_end],tools=tools)
    if tokens[:starts[1]]!=incremental:raise ValueError('Actual appended synthetic tool history differs from native input')
    if finalized.get('token_build_model')!='qwen3_6' or finalized.get('num_steps')!=2:
        raise ValueError('Unexpected actual proxy mode or step count')
    receipt={'schema':'yeta.qwen38-quality-live/v1','checkpoint':checkpoint,'model_name':model_name,
        'native_template_sha256':protocol.TEMPLATE_SHA,'cpu':cpu,
        'actual_initial_native_xhigh_parity':True,'actual_incremental_tool_result_parity':True,
        'supplied_synthetic_tool_history':True,'model_action_success_required':False,
        'excluded_from_quality_score':True,'qualification_requests':2,'session_id':session,
        'initial_token_sha256':shared.digest(tokens[:starts[0]]),
        'incremental_token_sha256':shared.digest(tokens[:starts[1]]),
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_verifier_sha256':hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest()}
    return receipt,record


def validate(receipt,run):
    expected={'schema':'yeta.qwen38-quality-live/v1','checkpoint':run['checkpoint'],
        'model_name':run['model_name'],'native_template_sha256':protocol.TEMPLATE_SHA,
        'actual_initial_native_xhigh_parity':True,'actual_incremental_tool_result_parity':True,
        'supplied_synthetic_tool_history':True,'model_action_success_required':False,
        'excluded_from_quality_score':True,'qualification_requests':2,
        'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'shared_verifier_sha256':hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest()}
    if any(receipt.get(key)!=value for key,value in expected.items()):raise ValueError('Native proof identity or actual parity mismatch')
    cpu=receipt.get('cpu',{})
    if (cpu.get('native_template_sha256')!=protocol.TEMPLATE_SHA or cpu.get('qualified_mode')!='qwen3_6'
        or any(cpu.get(key) is not True for key in ('initial_explicit_xhigh_parity','incremental_tool_result_parity','thinking_prefix_correct'))):
        raise ValueError('CPU native template proof is missing')
    return receipt
