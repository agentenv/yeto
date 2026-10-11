import json
from pathlib import Path
import pytest
from yeto.rl.rlf import *

def package(tmp_path, mode='flag'):
    (tmp_path/'public').mkdir(); (tmp_path/'public'/'README').write_text('x')
    (tmp_path/'task.yaml').write_text(f'task_id: demo\ngrader_mode: {mode}\n')
    (tmp_path/'MANIFEST.json').write_text(json.dumps({'task_id':'demo','grader_mode':mode}))

def test_admission_requires_runtime_evidence(tmp_path):
    package(tmp_path)
    with pytest.raises(AdmissionError): admit_task_package(tmp_path)
    a=admit_task_package(tmp_path,evidence={'runtime_quality_gate':{'healthy':True}},image_digest='sha256:x')
    assert a.task_id=='demo' and a.grader_mode=='flag'

def test_agentflow_trace_cannot_admit(tmp_path):
    package(tmp_path)
    with pytest.raises(AdmissionError, match='AgentFlow'): admit_task_package(tmp_path,evidence={'runtime_quality_gate':1,'agentflow_node_output':'x'})

def test_finish_preserves_identity_and_cleanup():
    l=EpisodeLease('demo','traj-1',secret='private')
    r=finish_episode(l,TrustedGrade('completed',1,True,{'flag':'matched'}))
    assert r.value==1 and r.metadata['success'] and l.closed and l.secret is None

def test_gate_rejects_constant_reward_and_accepts_update():
    with pytest.raises(AdmissionError): rl_gate([{'reward':0,'advantage':0,'policy_update':False}])
    out=rl_gate([{'reward':0,'advantage':-.1,'metadata':{'success':False},'policy_update':True},{'reward':1,'advantage':.2,'metadata':{'success':True}}])
    assert out['success_count']==1 and out['advantage_abs_sum']>.0

def test_gate_requires_success_count():
    with pytest.raises(AdmissionError, match='success_count_positive'):
        rl_gate([{'reward':0,'advantage':-.5,'metadata':{'success':False},'policy_update':True},{'reward':1,'advantage':.5,'metadata':{'success':False},'policy_update':True}])
