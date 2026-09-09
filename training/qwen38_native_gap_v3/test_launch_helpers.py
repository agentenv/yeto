"""CPU-only launch contract tests; no Docker or GPU commands are executed."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from monitoring import launch_cot_masked_native_gap_v3_n3_20260909 as helper


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(helper,'ROOT',tmp_path)
    run=tmp_path/'runs/cot-masked-native-gap-v3-test';run.mkdir(parents=True)
    code=tmp_path/'code/runtime';code.mkdir(parents=True)
    dataset=tmp_path/'datasets/cot-masked-native-gap-v3-test';dataset.mkdir(parents=True)
    (code/'runtime.py').write_text('bound runtime\n')
    (code/'code-manifest.json').write_text(json.dumps({'files':[{'path':'runtime.py','bytes':14,'sha256':helper.digest(code/'runtime.py')}]}))
    for name in ('manifest.json','index.json','source-manifest.json','source-COMPLETE.json'):(dataset/name).write_text('{}')
    config={'training_contract':'qwen38-xhigh-native-gap-cot-loss-zero/v3','experiment_phase':'train',
        'dataset':{'_target_':'training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset',
            'path_or_dataset':str(dataset/'manifest.json'),'index_path':str(dataset/'index.json'),'index_sha256':helper.digest(dataset/'index.json')},
        'validation_dataset':{'_target_':'training.qwen38_native_gap_v3.masked_data.GeneratedMaskedTokenDataset','max_samples':32,'max_input_tokens':32768},
        'checkpoint':{'checkpoint_dir':str(run/'checkpoints')},'step_scheduler':{'num_epochs':1}}
    (run/'train.json').write_text(json.dumps(config))
    cpu={'status':'passed','runtime_image':helper.IMAGE,'training_contract':config['training_contract'],
        'config_sha256':helper.digest(run/'train.json'),'code_manifest_sha256':helper.digest(code/'code-manifest.json'),
        'cuda_initialized':False,'loss_internal_shift':False,'causal_shift':'dataset-once-before-CP',
        'loss_source_sha256':'fd4754cb1eb4f28373a75fff9efef0524768bf3a779659f8aa2c3df3b392cbad',
        'semantic_quality_qualified':False,'dataset_rows':{'train':5,'validation':1},'runtime_dataset_rows':{'validation':1},
        'frozen_valid_candidate_count':31504,'included_valid_candidate_count':24000,'omitted_valid_candidate_count':4000,
        'excluded_valid_candidate_count':3504,'unreviewed_gap_count':24000,
        'gap_omission_policy':'keep-native-leading-gap-omit-only-conflicting-cot/v3',
        'full_native_mask_validation':True,'original_gap_positions_preserved':True,'reasoning_relocated':False,
        'known_session_split_verified':True,'unknown_sessions_train_only':True,
        'manifest_sha256':helper.digest(dataset/'manifest.json'),'index_sha256':helper.digest(dataset/'index.json'),
        'source_manifest_sha256':helper.digest(dataset/'source-manifest.json'),'source_complete_sha256':helper.digest(dataset/'source-COMPLETE.json')}
    path=dataset/'cpu-preflight.json';path.write_text(json.dumps(cpu))
    plan={'schema':'qwen38-native-gap-masked-launch-plan/v3','run':str(run),'code':str(code),
        'config_sha256':cpu['config_sha256'],'code_manifest_sha256':cpu['code_manifest_sha256'],
        'cpu_receipt':str(path),'cpu_receipt_sha256':helper.digest(path)}
    calls=[]
    def output(argv,**kwargs):
        calls.append(argv)
        return '\n'.join(f'{i}, 0, 143771' for i in range(8)) if argv[0]=='nvidia-smi' and '--query-gpu=index,memory.used,memory.total' in argv else ''
    monkeypatch.setattr(helper.subprocess,'check_output',output)
    monkeypatch.setattr(helper.subprocess,'run',lambda argv,**kw:(calls.append(argv) or SimpleNamespace(returncode=0,stdout='a'*64,stderr='')))
    return plan,cpu,calls


def test_native_gap_coverage_launch_uses_fresh_full_exact_arm(tmp_path,monkeypatch):
    plan,_,calls=fixture(tmp_path,monkeypatch)
    assert helper.launch(plan)==0
    command=calls[-1]
    assert command[command.index('-m')+1]=='training.qwen38_native_gap_v3.masked_train'
    assert '--nproc-per-node=8' in command and '--validate-memory-repair-in-production' in command
    assert 'PYTORCH_ALLOC_CONF=expandable_segments:True' in command
    assert command[command.index('--mode')+1]=='train'
    with pytest.raises(AssertionError):helper.launch(plan)


@pytest.mark.parametrize('key,value',[('omitted_valid_candidate_count',0),('code_manifest_sha256','a'*64),
    ('cuda_initialized',True),('reasoning_relocated',True),('training_contract','qwen38-xhigh-generated-cot-loss-zero-turn-boundary/v2')])
def test_bad_coverage_identity_or_template_rejected_before_gpu_check(tmp_path,monkeypatch,key,value):
    plan,cpu,calls=fixture(tmp_path,monkeypatch);cpu[key]=value
    path=Path(plan['cpu_receipt']);path.write_text(json.dumps(cpu));plan['cpu_receipt_sha256']=helper.digest(path)
    with pytest.raises(AssertionError):helper.launch(plan)
    assert calls==[]
