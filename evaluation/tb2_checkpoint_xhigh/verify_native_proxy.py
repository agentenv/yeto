"""Native-Qwen CPU and two-call live proxy qualification, outside task scores."""
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid

TEMPLATE_SHA='c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041'
XHIGH='Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.'
def digest(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode()).hexdigest()

def runtime(model):
    from transformers import AutoTokenizer
    from dressage.proxy.tito import configure_tito_chat_template,create_tito_tokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    if hashlib.sha256(tok.chat_template.encode()).hexdigest()!=TEMPLATE_SHA:raise ValueError('Wrong native template')
    configure_tito_chat_template(tok,model_type='qwen3_6')
    if hashlib.sha256(tok.chat_template.encode()).hexdigest()!=TEMPLATE_SHA:raise ValueError('Native template was replaced')
    return tok,create_tito_tokenizer(tok,model_type='qwen3_6')

def cpu(model):
    tok,tito=runtime(model)
    tools=[{'type':'function','function':{'name':'proof_read','description':'Read the qualification nonce.','parameters':{'type':'object','properties':{},'additionalProperties':False}}}]
    prefix=[{'role':'system','content':'Use the available tool when asked.'},{'role':'user','content':'Call proof_read and report its output.'}]
    text=tito._render_messages(prefix,add_generation_prompt=True,tools=tools)
    explicit=tok.apply_chat_template(prefix,tools=tools,add_generation_prompt=True,tokenize=False,enable_thinking=True,preserve_thinking=True,reasoning_effort='xhigh')
    if text!=explicit or XHIGH not in text or not text.endswith('<|im_start|>assistant\n<think>\n'):raise ValueError('Initial native xhigh prompt mismatch')
    assistant={'role':'assistant','content':'','reasoning_content':'I should read the requested nonce.','tool_calls':[{'id':'proof-call','type':'function','function':{'name':'proof_read','arguments':{}}}]}
    old=prefix+[assistant];new=old+[{'role':'tool','tool_call_id':'proof-call','content':'native-proof-fixture'}]
    pre=tito._encode_text(tito._render_messages(old,add_generation_prompt=False,tools=tools))
    if pre[-1]==tok.encode('\n',add_special_tokens=False)[0]:pre=pre[:-1]
    merged=tito.merge_tokens(old_messages=old,new_messages=new,pretokenized_token_ids=pre,tools=tools)
    expected=tito._encode_text(tito._render_messages(new,add_generation_prompt=True,tools=tools))
    if merged!=expected:raise ValueError('Native incremental tool-result tokens differ')
    return {'schema':'yeta.native-qwen38-proxy-cpu/v1','native_template_sha256':TEMPLATE_SHA,'initial_explicit_xhigh_parity':True,'incremental_tool_result_parity':True,'thinking_prefix_correct':True,'tool_function_schema_count':1,'qualified_mode':'qwen3_6'}

def live(model,model_name,api_key,base_url='http://127.0.0.1:35810'):
    cpu_receipt=cpu(model);tok,tito=runtime(model);session='native38-proof-'+uuid.uuid4().hex
    nonce='native-proof-'+uuid.uuid4().hex
    tools=[{'type':'function','function':{'name':'proof_read','description':'Read the qualification nonce; call once with no arguments.','parameters':{'type':'object','properties':{},'additionalProperties':False}}}]
    prefix=[{'role':'system','content':'Follow the user request and use the provided function. This is a tool interface qualification.'},{'role':'user','content':'Call proof_read exactly once with no arguments, then return exactly its output. Do not guess the output.'}]
    def post(path,body):
        req=urllib.request.Request(base_url+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+api_key,'X-Session-Id':session,'X-Turn-Id':session+':0'})
        with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)
    def generate(messages):return post('/v1/chat/completions',{'model':model_name,'messages':messages,'tools':tools,'max_tokens':2048,'temperature':0.0,'stream':False})
    first=generate(prefix);assistant=first['choices'][0]['message'];calls=assistant.get('tool_calls') or []
    if len(calls)!=1 or calls[0]['function']['name']!='proof_read':raise ValueError('Live proxy did not return the requested tool call')
    args=calls[0]['function']['arguments'];args=json.loads(args) if isinstance(args,str) else args
    if args!={}:raise ValueError('Unexpected live proof tool arguments')
    tool={'role':'tool','tool_call_id':calls[0]['id'],'content':nonce}
    second=generate(prefix+[assistant,tool]);reply=second['choices'][0]['message']
    if (reply.get('content') or '').strip()!=nonce or reply.get('tool_calls'):raise ValueError('Tool-result round trip failed')
    finalized=post('/session/finalize',{'session_id':session})
    records=post('/trajectory/read',{'trajectory_id':session,'drain':False,'segment_view':'lineage'})['data']
    if len(records)!=1:raise ValueError('Expected one complete native qualification lineage')
    record=records[0];tokens=record['tokens'];mask=record['full_loss_mask']
    starts=[i for i,v in enumerate(mask) if v and (i==0 or not mask[i-1])]
    if len(starts)!=2:raise ValueError('Expected two actual generated token spans')
    expected=tito._encode_text(tito._render_messages(prefix,add_generation_prompt=True,tools=tools))
    if tokens[:starts[0]]!=expected:raise ValueError('Actual live first input differs from native xhigh')
    first_end=next(i for i in range(starts[0],len(mask)) if not mask[i])
    expected_second=tito.merge_tokens(old_messages=prefix+[assistant],new_messages=prefix+[assistant,tool],pretokenized_token_ids=tokens[:first_end],tools=tools)
    if tokens[:starts[1]]!=expected_second:raise ValueError('Actual live appended tool result differs from native tokens')
    if finalized.get('token_build_model')!='qwen3_6' or finalized.get('num_steps')!=2:raise ValueError('Unexpected live proxy mode or step count')
    return {'schema':'yeta.native-qwen38-proxy-live/v1','cpu':cpu_receipt,'checkpoint_step_zero_based':299,'model_name':model_name,
            'native_template_sha256':TEMPLATE_SHA,'actual_initial_native_xhigh_parity':True,'actual_incremental_tool_result_parity':True,
            'actual_tool_roundtrip':True,'qualification_requests':2,'session_id':session,'initial_token_count':starts[0],
            'initial_token_sha256':digest(tokens[:starts[0]]),'second_token_count':starts[1],
            'second_token_sha256':digest(tokens[:starts[1]]),'excluded_from_benchmark':True,
            'verifier_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
