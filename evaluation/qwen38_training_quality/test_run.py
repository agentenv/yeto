from copy import deepcopy
import hashlib
import json
from pathlib import Path
import types
import sys
import pytest
from . import native,protocol,run,suite
from .prepare import prepare,verify_run


def export_fixture(model,identity):
    model.mkdir()
    weights=model/'model-00001-of-00001.safetensors';weights.write_bytes(b'CPU synthetic weight fixture')
    asset_names=['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']
    for name in asset_names:(model/name).write_text('{}')
    index=model/'model.safetensors.index.json';index.write_text(json.dumps({'weight_map':{'model.weight':weights.name}}))
    receipt={'status':'complete','checkpoint':identity,'checkpoint_manifest_sha256':identity['checkpoint_manifest_sha256'],
        'checkpoint_step_zero_based':identity['zero_based_checkpoint_step'],'completed_optimizer_updates':identity['completed_updates'],
        'source_snapshot_manifest_sha256':'d'*64,'exact_tensor_readback':True,'speculative_decoding_allowed':False,
        'files':[{'path':weights.name,'size':weights.stat().st_size,'sha256':suite.digest(weights)}],
        'assets':[{'path':name,'sha256':suite.digest(model/name)} for name in asset_names],
        'index_sha256':suite.digest(index)}
    path=model/'export-receipt.json';path.write_text(json.dumps(receipt));return suite.digest(path),receipt


def staged(tmp_path):
    frozen=tmp_path/'frozen';suite.build(frozen)
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'models':[{'slug':'old','display_name':'old'}]}))
    identity=protocol.checkpoint_identity(arm='masked-cot-native-gap-v3',completed_updates=25,
        checkpoint_sha256='a'*64,cumulative_supervised_tokens=250000)
    root=tmp_path/'ready'
    prepare(suite_dir=frozen,output=root,remote_root=str(root),checkpoint=identity,model_catalog=catalog)
    sha,receipt=export_fixture(root/'hf',identity)
    return root,sha,receipt


def test_actual_export_identity_and_every_file_checked(tmp_path):
    root,sha,receipt=staged(tmp_path);manifest=verify_run(root,allow_runtime_outputs=True)
    assert run.verify_export(root/'hf',manifest,sha)==receipt
    (root/'hf/model-00001-of-00001.safetensors').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='file differs'):run.verify_export(root/'hf',manifest,sha)


@pytest.mark.parametrize('key,value',[('checkpoint_step_zero_based',299),('exact_tensor_readback',False),
    ('speculative_decoding_allowed',True),('source_snapshot_manifest_sha256',None)])
def test_export_cannot_relabel_old_or_unverified_checkpoint(tmp_path,key,value):
    root,sha,receipt=staged(tmp_path);manifest=verify_run(root,allow_runtime_outputs=True)
    receipt[key]=value;path=root/'hf/export-receipt.json';path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):run.verify_export(root/'hf',manifest,suite.digest(path))


def test_native_service_alias_partition_and_context_guard(tmp_path):
    root,_,_=staged(tmp_path);manifest=verify_run(root,allow_runtime_outputs=True)
    services=json.loads((root/'runtime/services.json').read_bytes())
    assert run.require_service_plan(services,manifest,root)==services
    changed=deepcopy(services);changed['services_in_dependency_order'][0]['environment']['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    with pytest.raises(ValueError,match='partition'):run.require_service_plan(changed,manifest,root)
    changed=deepcopy(services);proxy=changed['services_in_dependency_order'][-1]['argv'];proxy[proxy.index('--tito-model')+1]='qwen3_5'
    with pytest.raises(ValueError,match='proxy'):run.require_service_plan(changed,manifest,root)


def test_existing_gpu_work_rejected_without_stopping(monkeypatch):
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**k:'123\n')
    with pytest.raises(RuntimeError,match='existing GPU work'):run.assert_idle()


def test_actual_native_gate_does_not_require_model_action(monkeypatch,tmp_path):
    checkpoint=protocol.checkpoint_identity(arm='no-cot-turn-v2',completed_updates=25,
        checkpoint_sha256='a'*64,cumulative_supervised_tokens=123)
    cpu={'native_template_sha256':protocol.TEMPLATE_SHA,'qualified_mode':'qwen3_6',
         'initial_explicit_xhigh_parity':True,'incremental_tool_result_parity':True,'thinking_prefix_correct':True}
    class Tito:
        def _render_messages(self,*a,**kw):return 'fixed-initial'
        def _encode_text(self,text):return [1,2]
        def merge_tokens(self,**kwargs):
            assert kwargs['new_messages'][-2]['tool_calls'][0]['id']=='quality-fixed-call'
            assert kwargs['new_messages'][-1]['role']=='tool'
            return [1,2,9,248046,3,4]
    monkeypatch.setattr(native.shared,'cpu',lambda model:cpu)
    monkeypatch.setattr(native.shared,'runtime',lambda model:(None,Tito()))
    records={'tokens':[1,2,9,248046,3,4,9,248046],'full_loss_mask':[0,0,1,1,0,0,1,1]}
    calls=[]
    class Response:
        def __init__(self,value):self.value=value
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,*args):return json.dumps(self.value).encode()
    def request(req,**kwargs):
        body=json.loads(req.data);calls.append((req.full_url,body))
        if req.full_url.endswith('/v1/chat/completions'):
            # Both checkpoint responses stop in prose with no tool call; admission still succeeds.
            return Response({'choices':[{'message':{'role':'assistant','content':'I will do that.'}}]})
        if req.full_url.endswith('/session/finalize'):return Response({'token_build_model':'qwen3_6','num_steps':2})
        return Response({'data':[records]})
    monkeypatch.setattr(native.urllib.request,'urlopen',request)
    proof,record=native.live(tmp_path,'actual-checkpoint-alias','synthetic-test-key',checkpoint)
    assert proof['model_action_success_required'] is False and proof['qualification_requests']==2
    assert native.validate(proof,{'checkpoint':checkpoint,'model_name':'actual-checkpoint-alias'})==proof
    assert record==records and len(calls)==4
    assert 'synthetic-test-key' not in json.dumps(proof)
    corrupted=deepcopy(proof);corrupted['checkpoint']['completed_updates']=300
    with pytest.raises(ValueError):native.validate(corrupted,{'checkpoint':checkpoint,'model_name':'actual-checkpoint-alias'})


def test_native_failure_stops_only_owned_services_and_never_starts_harbor(monkeypatch,tmp_path):
    root,sha,_=staged(tmp_path)
    module=types.ModuleType('harbor.models.job.config')
    class JobConfig:
        @classmethod
        def model_validate_json(cls,value):assert len(json.loads(value)['tasks'])==16
    module.JobConfig=JobConfig;monkeypatch.setitem(sys.modules,'harbor.models.job.config',module)
    monkeypatch.setattr(run,'assert_idle',lambda:None)
    def verify(*args,**kwargs):
        run.os.environ['CUDA_VISIBLE_DEVICES']=''
        return {'status':'passed'}
    monkeypatch.setattr(run,'verify_tokenizer',verify)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','original-visible')
    launched=[];procs={}
    class Process:
        def __init__(self,argv,**kwargs):
            self.pid=11000+len(launched);self.returncode=None;launched.append((argv,kwargs));procs[self.pid]=self
        def poll(self):return self.returncode
        def wait(self,**kwargs):return self.returncode
    monkeypatch.setattr(run.subprocess,'Popen',Process)
    signaled=[]
    def kill(pid,sig):signaled.append(pid);procs[pid].returncode=-15
    monkeypatch.setattr(run.os,'killpg',kill)
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(run.urllib.request,'urlopen',lambda *a,**k:Response())
    monkeypatch.setattr(run.native,'live',lambda *a,**k:(_ for _ in ()).throw(ValueError('actual native parity mismatch')))
    with pytest.raises(ValueError,match='parity mismatch'):run.run_suite(root,export_receipt_sha256=sha)
    assert len(launched)==6 and len(signaled)==6 and set(signaled)==set(procs)
    assert all('harbor_explicit_task_metrics.py' not in str(argv) for argv,env in launched)
    assert run.os.environ['CUDA_VISIBLE_DEVICES']=='original-visible'
    assert json.loads((root/'launch/failure.json').read_bytes())['type']=='ValueError'
    assert not (root/'jobs').exists()


def test_preserved_cpu_converter_receipt_is_hash_bound(tmp_path):
    root,sha,receipt=staged(tmp_path);manifest=verify_run(root,allow_runtime_outputs=True)
    original=root/'hf/cpu-converter-receipt.json';original.write_text(json.dumps(receipt))
    receipt['cpu_converter_receipt_sha256']=suite.digest(original)
    path=root/'hf/export-receipt.json';path.write_text(json.dumps(receipt))
    assert run.verify_export(root/'hf',manifest,suite.digest(path))==receipt
    original.write_text('{}')
    with pytest.raises(ValueError,match='converter evidence'):run.verify_export(root/'hf',manifest,suite.digest(path))
