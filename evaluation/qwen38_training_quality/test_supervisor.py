"""Independent CPU supervisor identity, interruption and handoff tests."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from . import checkpoint,protocol,supervisor as s


@pytest.fixture(autouse=True)
def no_real_subprocess(monkeypatch):
    monkeypatch.setattr(s.subprocess,'run',lambda *_a,**_k:pytest.fail('CPU test must not execute a real subprocess'))


def plan():
    return {'trainer_container':'a'*64,'trainer_started_at':'2026-09-09T23:00:00Z',
        'trainer_argv':['torchrun','-m','training.qwen38_turn_boundary_v2.train'],
        'config_path':'/fake/train.json','config_sha256':'b'*64,'quality_code_dir':'/code',
        'quality_code_manifest_sha256':'e'*64,'arm':'no-cot-turn-v2'}


def trainer(p):
    return {'id':p['trainer_container'],'image':s.IMAGE,'started_at':p['trainer_started_at'],
        'path':p['trainer_argv'][0],'args':p['trainer_argv'][1:],'running':True,'exit_code':0,'oom_killed':False}


@pytest.mark.parametrize('key,value',[('id','c'*64),('started_at','restarted'),('image','wrong'),('args',['wrong'])])
def test_container_replacement_or_restart_rejected(monkeypatch,key,value):
    p=plan();info=trainer(p);info[key]=value
    monkeypatch.setattr(s,'inspect_container',lambda _:info)
    monkeypatch.setattr(s,'bound_file',lambda *_:None)
    with pytest.raises(ValueError,match='Trainer identity changed'):s.verify_trainer(p)


def test_cpu_export_has_no_gpu_network_and_finite_resources():
    argv=s.export_command(plan(),25,Path('/source'),Path('/exports/u25'),'cpu-fixture')
    for key,value in [('--runtime','runc'),('--network','none'),('--cpus','16'),('--memory','192g'),('--memory-swap','192g')]:
        assert argv[argv.index(key)+1]==value
    assert '--gpus' not in argv and '--privileged' not in argv
    assert 'NVIDIA_VISIBLE_DEVICES=void' in argv and 'CUDA_VISIBLE_DEVICES=' in argv
    assert '--read-only' in argv and '/source:/source:ro' in argv
    assert argv[-5:]==['evaluation.qwen38_training_quality.export','--source','/source','--destination','/exports/u25']


def instance(tmp_path):
    obj=s.Supervisor.__new__(s.Supervisor);obj.root=tmp_path;obj.plan=plan();obj.plan_sha=protocol.sha(obj.plan)
    obj.state={'status':'running','checkpoints':{str(n):{'state':'waiting'} for n in s.SCHEDULE}}
    obj.paths={'checkpoints':tmp_path/'checkpoints','config':tmp_path/'train.json'}
    obj.paths['checkpoints'].mkdir();obj.paths['config'].write_text('{}');obj.stop_requested=False
    obj.save=lambda:None
    return obj


def test_resource_queue_does_not_skip_later_checkpoint_capture(tmp_path,monkeypatch):
    obj=instance(tmp_path);captured=[]
    for n in s.SCHEDULE:
        source=tmp_path/f'source-{n}';source.mkdir();(source/'snapshot-manifest.json').write_text(json.dumps({'files':[{'bytes':64}]}))
    def capture(n,*_):
        captured.append(n);obj.state['checkpoints'][str(n)]={'state':'snapshot_complete','snapshot':str(tmp_path/f'source-{n}')}
    monkeypatch.setattr(obj,'capture',capture)
    monkeypatch.setattr(s,'verify_trainer',lambda _:trainer(obj.plan))
    monkeypatch.setattr(s,'optimizer_history',lambda _:('',100))
    monkeypatch.setattr(s,'resource_status',lambda *_:{'ready':False})
    monkeypatch.setattr(s,'run',lambda *_:pytest.fail('resource blocked must not launch export'))
    assert obj.tick() is False
    assert captured==[25,50,100]
    assert all(item['state']=='snapshot_complete' for item in obj.state['checkpoints'].values())


def completed_export(obj,tmp_path):
    dest=tmp_path/'export';dest.mkdir()
    ident=protocol.checkpoint_identity(arm='no-cot-turn-v2',completed_updates=25,checkpoint_sha256='d'*64,cumulative_supervised_tokens=100)
    item={'state':'export_running','export_container':'c'*64,'export_argv':['python3','-m','export'],
        'export_destination':str(dest),'checkpoint':ident,'snapshot_manifest_sha256':'d'*64}
    receipt={'schema':'yeta.qwen38-training-quality-hf-export/v1','status':'complete','exact_tensor_readback':True,
        'gpu_used':False,'checkpoint':ident,'source_snapshot_manifest_sha256':'d'*64}
    path=dest/'export-receipt.json';path.write_text(json.dumps(receipt))
    info={'id':item['export_container'],'image':s.IMAGE,'path':'python3','args':['-m','export'],
        'running':False,'exit_code':0,'oom_killed':False}
    return item,receipt,path,info


def test_actual_receipt_binds_checkpoint_and_duplicate_ready_is_idempotent(tmp_path,monkeypatch):
    obj=instance(tmp_path);item,receipt,path,info=completed_export(obj,tmp_path)
    monkeypatch.setattr(s,'inspect_container',lambda _:info)
    monkeypatch.setattr(s,'verify_export_isolation',lambda *_:None)
    assert obj.finish_export(25,item) is False
    ready=tmp_path/'ready-updates-25.json';before=ready.read_bytes()
    assert obj.finish_export(25,item) is False and ready.read_bytes()==before
    assert json.loads(before)['evaluation_launched'] is False
    receipt['checkpoint']={**receipt['checkpoint'],'completed_updates':50};path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='captured checkpoint'):obj.finish_export(25,item)


@pytest.mark.parametrize('state',['snapshotting','export_launch_pending','export_created'])
def test_interrupted_mutations_do_not_blindly_retry(tmp_path,monkeypatch,state):
    obj=instance(tmp_path);obj.state['checkpoints']['25']['state']=state
    monkeypatch.setattr(obj,'tick',lambda:pytest.fail('no retry after interrupted mutation'))
    with pytest.raises(ValueError,match='reconciliation'):obj.run(once=True)


def test_cpu_isolation_inspection_rejects_device_request(monkeypatch):
    p=plan();info={'runtime':'runc','network':'none','cpus':16*10**9,'memory':192*s.GIB,
        'memory_swap':192*s.GIB,'readonly':True,'plan':protocol.sha(p),'updates':'25',
        'devices':[],'device_requests':[{'Count':-1}]}
    monkeypatch.setattr(s,'run',lambda *_:json.dumps(info))
    with pytest.raises(ValueError,match='isolation'):s.verify_export_isolation('c'*64,p,25)


def test_optimizer_history_reads_both_streams_and_keeps_only_budget(monkeypatch):
    rows=[json.dumps({'event':'verified_optimizer_update','step':n,'measurements':{'num_label_tokens':n+1,
        'num_tokens_per_step':n+10,'private_extra':'must not be retained'}}) for n in range(2)]
    monkeypatch.setattr(s.subprocess,'run',lambda *_a,**_k:SimpleNamespace(returncode=0,stdout=rows[0],stderr=rows[1]))
    text,completed=s.optimizer_history('a'*64)
    assert completed==2 and 'private_extra' not in text
    assert checkpoint.budget_from_events(text,2)=={'cumulative_supervised_tokens':3,'cumulative_input_tokens':21}
