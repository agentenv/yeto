"""CPU-only v4 archive/orchestration tests; no Docker or GPU work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import hashlib
import importlib.util
from types import SimpleNamespace

import pytest

from monitoring import launch_cot_masked_native_gap_v3_n3_20260909 as old_launch
from monitoring import prepare_cot_masked_native_gap_v3_n3_20260909 as old_prepare
from monitoring import stage_masked_native_gap_v3_20260909 as old_stage
from monitoring import transfer_cot_masked_native_gap_v3_20260909 as old_transfer
from monitoring import stage_masked_native_gap_v4_20260910 as stage_v4
from monitoring import freeze_masked_native_gap_v4_runtime_20260910 as freeze_v4
from monitoring import launch_cot_masked_native_gap_v4_n3_20260910 as launch_v4
from monitoring import prepare_cot_masked_native_gap_v4_n3_20260910 as prepare_v4
from monitoring import transfer_cot_masked_native_gap_v4_20260910 as transfer_v4
from monitoring import build_cot_masked_native_gap_v4_n3_20260911 as build_v4
from monitoring import stage_wandb_auth_v4_20260911 as wandb_auth_v4
from monitoring import verify_cot_masked_native_gap_v4_wandb_20260911 as external_wandb_v4


ROOT=Path(__file__).resolve().parents[2]


def test_invalid_v3_orchestration_is_hard_retired():
    with pytest.raises(RuntimeError,match='Retired invalid v3 launcher'):
        old_launch.launch({})
    with pytest.raises(RuntimeError,match='Retired invalid v3 preflight'):
        old_prepare.prepare(None,None,None)
    with pytest.raises(RuntimeError,match='Retired v3 code stager'):
        old_stage.stage(None,None,None,None,None)
    with pytest.raises(RuntimeError,match='Retired manifest-rewriting v3 transfer'):
        old_transfer.main()


def _copy_closure(destination, *, omit=None):
    for name in stage_v4.REQUIRED:
        if name == omit:
            continue
        target=destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)


def test_v4_freezer_and_stager_package_exact_executable_closure(tmp_path):
    source=tmp_path/'source';source.mkdir();_copy_closure(source)
    archive=tmp_path/'runtime.tar.gz'
    frozen=freeze_v4.freeze(source.resolve(),archive)
    destination=tmp_path/'staged'
    staged=stage_v4.stage(archive,frozen['archive_sha256'],
                          frozen['code_manifest_sha256'],destination)
    assert frozen['file_count']==staged['files']==len(stage_v4.REQUIRED)
    manifest=json.loads((destination/'code-manifest.json').read_text())
    assert {item['path'] for item in manifest['files']}==stage_v4.REQUIRED
    assert all((destination/name).is_file() for name in stage_v4.REQUIRED)
    assert all((destination/name).stat().st_mode&0o222==0 for name in stage_v4.REQUIRED)
    launch_v4._verify_code_tree(destination,frozen['code_manifest_sha256'])
    if importlib.util.find_spec('transformers') is None:
        pytest.skip('isolated import runs in the pinned production image')
    env={**os.environ,'PYTHONPATH':str(destination),'PYTHONDONTWRITEBYTECODE':'1'}
    probe=('from training.qwen38_native_gap_v3 import masked_train,prepare_masked_full;'
           'assert masked_train.VERSION.endswith("/v4");'
           'assert prepare_masked_full.VERSION.endswith("/v4")')
    result=subprocess.run([sys.executable,'-c',probe],cwd=destination,env=env,
                          text=True,capture_output=True)
    assert result.returncode==0,result.stderr


def test_v4_runtime_archive_is_byte_deterministic_across_output_names(tmp_path):
    source=tmp_path/'source';source.mkdir();_copy_closure(source)
    first=tmp_path/'first.tar.gz';second=tmp_path/'different-name.tar.gz'
    one=freeze_v4.freeze(source.resolve(),first)
    two=freeze_v4.freeze(source.resolve(),second)
    assert one['code_manifest_sha256']==two['code_manifest_sha256']
    assert one['archive_sha256']==two['archive_sha256']
    assert first.read_bytes()==second.read_bytes()


def test_v4_freezer_never_publishes_when_postwrite_source_check_fails(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();_copy_closure(source)
    output=tmp_path/'runtime.tar.gz'
    def changed(*_):
        raise ValueError('source changed fixture')
    monkeypatch.setattr(freeze_v4,'_assert_source_stable',changed)
    with pytest.raises(ValueError,match='source changed fixture'):
        freeze_v4.freeze(source.resolve(),output)
    assert not output.exists()
    assert len(list(tmp_path.glob('.runtime.tar.gz.failed-*')))==1


def test_v4_freezer_never_overwrites_a_concurrently_created_archive(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();_copy_closure(source)
    output=tmp_path/'runtime.tar.gz';real_link=freeze_v4.os.link
    def collide(incoming,destination):
        destination.write_bytes(b'concurrent owner')
        raise FileExistsError(destination)
    monkeypatch.setattr(freeze_v4.os,'link',collide)
    with pytest.raises(FileExistsError):
        freeze_v4.freeze(source.resolve(),output)
    assert output.read_bytes()==b'concurrent owner'
    assert len(list(tmp_path.glob('.runtime.tar.gz.failed-*')))==1
    monkeypatch.setattr(freeze_v4.os,'link',real_link)


def test_v4_freezer_rejects_missing_transitive_or_baseline_runtime_member(tmp_path):
    source=tmp_path/'source';source.mkdir()
    missing='training/qwen38_no_cot/nemo_checkpoint.py'
    _copy_closure(source,omit=missing)
    with pytest.raises(ValueError,match='missing or indirect'):
        freeze_v4.freeze(source.resolve(),tmp_path/'runtime.tar.gz')


def test_v4_stage_required_set_includes_dynamic_import_closure():
    assert 'cot_filler/provider.py' in stage_v4.REQUIRED
    assert 'cot_filler/grounding_review_fast.py' in stage_v4.REQUIRED
    assert 'training/qwen38_cot_experimental/train.py' in stage_v4.REQUIRED
    assert 'training/qwen38_no_cot/nemo_block_checkpoint.py' in stage_v4.REQUIRED
    assert 'training/qwen38_no_cot/assets/tokenizer.json' in stage_v4.REQUIRED
    assert 'training/qwen38_no_cot/assets/tokenizer_config.json' in stage_v4.REQUIRED
    assert 'training/qwen38_no_cot/assets/chat_template.jinja' in stage_v4.REQUIRED
    assert 'monitoring/launch_cot_masked_native_gap_v4_n3_20260910.py' in stage_v4.REQUIRED
    assert 'monitoring/promote_masked_native_gap_v4_exclusions_20260911.py' in stage_v4.REQUIRED
    assert 'monitoring/verify_cot_masked_native_gap_v4_wandb_20260911.py' in stage_v4.REQUIRED


def test_v4_stager_rejects_any_member_beyond_exact_reviewed_closure(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();_copy_closure(source)
    archive=tmp_path/'runtime.tar.gz';frozen=freeze_v4.freeze(source.resolve(),archive)
    removed=next(iter(stage_v4.REQUIRED))
    monkeypatch.setattr(stage_v4,'REQUIRED',stage_v4.REQUIRED-{removed})
    with pytest.raises(ValueError,match='required runtime closure'):
        stage_v4.stage(archive,frozen['archive_sha256'],frozen['code_manifest_sha256'],
                       tmp_path/'staged')


def test_v4_launcher_uses_only_explicit_ro_inputs_and_fresh_rw_outputs(tmp_path,monkeypatch):
    root=tmp_path
    (root/'runs').mkdir();(root/'configs').mkdir();(root/'triton-cache').mkdir()
    run=root/'runs/cot-masked-native-gap-v4-run'
    config_dir=root/'configs'/run.name;config_dir.mkdir()
    code=root/'code/cot-masked-native-gap-v4-code';code.mkdir(parents=True)
    dataset=root/'datasets/cot-masked-native-gap-v4-data';dataset.mkdir(parents=True)
    base=root/'models/Qwen3.8-27B';base.mkdir(parents=True)
    base_receipt=root/'model-receipts/base/verification.json';base_receipt.parent.mkdir(parents=True)
    base_receipt.write_text('{}')
    runtime=code/'training/runtime.py';runtime.parent.mkdir(parents=True)
    runtime.write_text('VALUE = 1\n')
    code_manifest={'schema':'qwen38-native-gap-v4-runtime-code/v1','files':[{
        'path':'training/runtime.py','bytes':runtime.stat().st_size,
        'sha256':hashlib.sha256(runtime.read_bytes()).hexdigest()}]}
    (code/'code-manifest.json').write_text(json.dumps(code_manifest))
    for name in ('manifest.json','index.json','COMPLETE.json'):
        (dataset/name).write_text('{}')
    manifest_sha=launch_v4.digest(dataset/'manifest.json')
    tree_inventory=[{'path':name,'bytes':(dataset/name).stat().st_size,
                     'sha256':launch_v4.digest(dataset/name)}
                    for name in ('COMPLETE.json','index.json','manifest.json')]
    build_receipt=dataset.with_name(dataset.name+'.build-receipt.json')
    build_receipt.write_text(json.dumps({
        'schema':'qwen38-native-gap-v4-build-publication/v1','status':'complete','mode':'final',
        'output':str(dataset),'runtime_image':launch_v4.IMAGE,
        'code_manifest_sha256':launch_v4.digest(code/'code-manifest.json'),
        'immutable_mode':'files0444_dirs0555','atomic_direct_dataset_publication':True,
        'files':len(tree_inventory),'bytes':sum(item['bytes'] for item in tree_inventory),
        'tree_inventory':tree_inventory,'tree_sha256':hashlib.sha256(json.dumps(
            tree_inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()}))
    build_receipt.chmod(0o444)
    config={'training_contract':'qwen38-xhigh-native-gap-cot-loss-zero/v4',
        'experiment_phase':'train','distributed':{'cp_size':8,'strategy':'fsdp2'},
        'optimizer':{'lr':1e-5},'step_scheduler':{'num_epochs':1,'max_steps':None},
        'checkpoint':{'checkpoint_dir':str(run/'checkpoints')},
        'wandb':{'project':'yeto-h200','entity':'yeta','name':
            'qwen38-27b-generated-cot-masked-native-gap-v4-'+manifest_sha[:12]+'-train'}}
    (config_dir/'train.json').write_text(json.dumps(config))
    cpu={'schema':'qwen38-native-gap-v4-cpu-preflight/v1','status':'passed',
        'runtime_image':launch_v4.IMAGE,'config_sha256':launch_v4.digest(config_dir/'train.json'),
        'code_manifest_sha256':launch_v4.digest(code/'code-manifest.json'),
        'build_code_manifest_sha256':launch_v4.digest(code/'code-manifest.json'),
        'manifest_sha256':manifest_sha,
        'index_sha256':launch_v4.digest(dataset/'index.json'),
        'complete_sha256':launch_v4.digest(dataset/'COMPLETE.json'),
        'build_receipt_path':str(build_receipt),
        'build_receipt_sha256':launch_v4.digest(build_receipt),
        'build_tree_sha256':json.loads(build_receipt.read_text())['tree_sha256'],
        'runtime_compatibility':None,
        'full_export':True,'all_train':True,'internal_validation':False,
        'validation_dataset_absent':True,'resolved_validation_dataloader_absent':True,
        'cuda_initialized':False,'gpu_runtime_qualified':False,'training_started':False}
    cpu_path=config_dir/'cpu-preflight.json';cpu_path.write_text(json.dumps(cpu))
    for path in config_dir.iterdir():path.chmod(0o444)
    config_dir.chmod(0o555)
    secret_root=tmp_path/'dev-shm';secret_root.mkdir()
    netrc=secret_root/'wandb-netrc-fixture';netrc.write_text('machine api.wandb.ai login fixture password redacted\n');netrc.chmod(0o600)
    plan={'schema':'qwen38-native-gap-v4-launch-plan/v1','run':str(run),
        'config_dir':str(config_dir),
        'code':str(code),'dataset':str(dataset),'cpu_preflight':str(cpu_path),
        'cpu_preflight_sha256':launch_v4.digest(cpu_path),
        'launcher_sha256':launch_v4.digest(launch_v4.__file__),
        'wandb_netrc':str(netrc),'wandb_run_id':'abc123xy',
        'first_metric_timeout_seconds':3600}
    monkeypatch.setattr(launch_v4,'ROOT',root)
    monkeypatch.setattr(launch_v4,'BASE',base)
    monkeypatch.setattr(launch_v4,'BASE_RECEIPT',base_receipt)
    monkeypatch.setattr(launch_v4,'WANDB_SECRET_ROOT',secret_root)
    launch_v4.validate(plan)
    _,command=launch_v4.command_for(plan)
    mounts=[command[index+1] for index,value in enumerate(command) if value=='--mount']
    assert 'type=bind,src='+str(code)+',dst=/workspace/yeto,readonly' in mounts
    assert 'type=bind,src='+str(dataset)+',dst='+str(dataset)+',readonly' in mounts
    assert 'type=bind,src='+str(base)+',dst='+str(base)+',readonly' in mounts
    assert 'type=bind,src='+str(base_receipt)+',dst='+str(base_receipt)+',readonly' in mounts
    assert 'type=bind,src='+str(config_dir)+',dst='+str(config_dir)+',readonly' in mounts
    assert 'type=bind,src='+str(netrc)+',dst=/root/.netrc,readonly' in mounts
    assert 'type=bind,src='+str(run)+',dst='+str(run) in mounts
    assert 'type=bind,src='+str(root/'triton-cache'/run.name)+',dst=/root/.triton/cache' in mounts
    assert not any(mount.startswith('type=bind,src='+str(root)+',dst='+str(root)) for mount in mounts)
    assert launch_v4.IMAGE_REF in command and 'WANDB_MODE=online' in command
    assert 'WANDB_RUN_ID=abc123xy' in command and 'WANDB_RESUME=never' in command
    runtime.write_text('VALUE = 2\n')
    with pytest.raises(ValueError,match='Runtime code member changed'):
        launch_v4.validate(plan)


def test_v4_cpu_preflight_build_receipt_rehashes_exact_tree(tmp_path):
    dataset=tmp_path/'cot-masked-native-gap-v4-fixture';dataset.mkdir()
    for name,value in (('manifest.json','manifest'),('index.json','index'),
                       ('COMPLETE.json','complete')):
        (dataset/name).write_text(value)
    records=[{'path':name,'bytes':(dataset/name).stat().st_size,
              'sha256':launch_v4.digest(dataset/name)}
             for name in ('COMPLETE.json','index.json','manifest.json')]
    receipt=dataset.with_name(dataset.name+'.build-receipt.json')
    payload={'schema':'qwen38-native-gap-v4-build-publication/v1','status':'complete',
        'mode':'final','output':str(dataset),'runtime_image':launch_v4.IMAGE,
        'code_manifest_sha256':'c'*64,'immutable_mode':'files0444_dirs0555',
        'atomic_direct_dataset_publication':True,'files':len(records),
        'bytes':sum(item['bytes'] for item in records),'tree_inventory':records,
        'tree_sha256':hashlib.sha256(json.dumps(records,sort_keys=True,
            separators=(',',':')).encode()).hexdigest()}
    receipt.write_text(json.dumps(payload));receipt.chmod(0o444)
    for member in dataset.iterdir():member.chmod(0o444)
    dataset.chmod(0o555)
    assert prepare_v4._verify_build_publication(
        dataset,receipt,launch_v4.digest(receipt),'c'*64)['tree_sha256']==payload['tree_sha256']
    (dataset/'index.json').chmod(0o644);(dataset/'index.json').write_text('tampered')
    (dataset/'index.json').chmod(0o444)
    with pytest.raises(ValueError,match='member differs'):
        prepare_v4._verify_build_publication(
            dataset,receipt,launch_v4.digest(receipt),'c'*64)


def test_v4_transfer_publishes_immutable_uid1000_readable_modes():
    assert transfer_v4.PUBLISHED_FILE_MODE==0o444
    assert transfer_v4.PUBLISHED_DIRECTORY_MODE==0o555
    assert transfer_v4.modes_allow_unprivileged_read()
    assert not transfer_v4.modes_allow_unprivileged_read(0o400,0o500)


def test_v4_cpu_builder_binds_minimal_read_capability_and_direct_output(tmp_path,monkeypatch):
    code=tmp_path/'code';source=tmp_path/'source';output=tmp_path/'dataset';scratch=tmp_path/'scratch'
    assets=code/'training/qwen38_no_cot/assets';assets.mkdir(parents=True);source.mkdir();scratch.mkdir()
    plan={'mode':'candidate','freeze_receipt':'/mnt/lvm_data/sft_analysis/sft_baseline_20260908/freeze.json',
          'baseline_selection':'/mnt/lvm_data/sft_analysis/sft_baseline_20260908/selection.json',
          'baseline_selection_sha256':'a'*64}
    monkeypatch.setattr(build_v4,'validate',lambda _:(code,source,output,assets))
    command=build_v4.command_for(plan,scratch)
    assert command[command.index('--cap-drop')+1]=='ALL'
    assert command[command.index('--cap-add')+1]=='DAC_READ_SEARCH'
    assert command[command.index('--security-opt')+1]=='no-new-privileges'
    assert command[command.index('--user')+1]=='0:0'
    assert '--gpus' not in command and command[command.index('--network')+1]=='none'
    image_index=command.index(build_v4.IMAGE_REF)
    assert command[image_index+1:image_index+4]==['python','-B','-m']
    assert command[command.index('--output')+1]=='/output/export'
    assert '--all-train' in command and '--dynamic-trace-exclusions' not in command
    assert 'type=bind,src='+str(scratch)+',dst=/output' in command


def test_v4_builder_requires_exact_frozen_inner_root(tmp_path,monkeypatch):
    root=tmp_path/'stage';source=root/'tree/mnt/lvm_data/sft_analysis/sft_baseline_20260908'
    source.mkdir(parents=True);code=tmp_path/'code/cot-masked-native-gap-v4-code';code.mkdir(parents=True)
    builder=code/build_v4.BUILDER_RELATIVE_PATH;builder.parent.mkdir(parents=True)
    shutil.copyfile(Path(build_v4.__file__),builder)
    output=tmp_path/'datasets/cot-masked-native-gap-v4-data';output.parent.mkdir()
    receipt=tmp_path/'stage.json';receipt.write_text('{}')
    plan={'schema':'qwen38-native-gap-v4-build-plan/v1','mode':'candidate','code':str(code),
          'staged_source_root':str(source),'output':str(output),
          'raw_stage_receipt':str(receipt),'raw_stage_receipt_sha256':build_v4.digest(receipt),
          'raw_source_inventory_sha256':'a'*64,'code_manifest_sha256':'b'*64,
          'builder_sha256':build_v4.digest(builder)}
    monkeypatch.setattr(build_v4,'ROOT',tmp_path)
    monkeypatch.setattr(build_v4,'__file__',str(builder))
    monkeypatch.setattr(build_v4,'_verify_code_tree',lambda *_:None)
    staged={'schema':'qwen38-masked-v4-frozen-raw-n3-stage/v1','status':'complete',
            'all_member_hashes_verified_after_transfer':True,'tree_files_mode':'0400',
            'tree_directories_mode':'0500','final_path':str(root/'unexpected-stage'),
            'source_inventory_sha256':'a'*64}
    monkeypatch.setattr(build_v4.json,'loads',lambda _:staged)
    with pytest.raises(ValueError,match='does not bind'):
        build_v4.validate(plan)


def test_v4_builder_plan_binds_exact_staged_builder(tmp_path,monkeypatch):
    root=tmp_path/'runtime';stage=tmp_path/'raw-stage'
    source=stage/'tree/mnt/lvm_data/sft_analysis/sft_baseline_20260908'
    source.mkdir(parents=True)
    code=root/'code/cot-masked-native-gap-v4-code';code.mkdir(parents=True)
    builder=code/build_v4.BUILDER_RELATIVE_PATH;builder.parent.mkdir(parents=True)
    shutil.copyfile(Path(build_v4.__file__),builder)
    assets=code/'training/qwen38_no_cot/assets';assets.mkdir(parents=True)
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        (assets/name).write_text('fixture')
    freeze=source/'freeze.json';freeze.write_text('freeze')
    selection=source/'selection.json';selection.write_text('selection')
    output=root/'datasets/cot-masked-native-gap-v4-data';output.parent.mkdir(parents=True)
    stage_receipt=tmp_path/'raw-stage.json'
    stage_receipt.write_text(json.dumps({
        'schema':'qwen38-masked-v4-frozen-raw-n3-stage/v1','status':'complete',
        'all_member_hashes_verified_after_transfer':True,'tree_files_mode':'0400',
        'tree_directories_mode':'0500','final_path':str(stage),
        'source_inventory_sha256':'a'*64}))
    plan={'schema':'qwen38-native-gap-v4-build-plan/v1','mode':'candidate',
        'code':str(code),'staged_source_root':str(source),'output':str(output),
        'code_manifest_sha256':'c'*64,'builder_sha256':build_v4.digest(builder),
        'raw_stage_receipt':str(stage_receipt),
        'raw_stage_receipt_sha256':build_v4.digest(stage_receipt),
        'raw_source_inventory_sha256':'a'*64,
        'freeze_receipt':str(build_v4.ORIGINAL_ROOT/'freeze.json'),
        'freeze_receipt_sha256':build_v4.digest(freeze),
        'baseline_selection':str(build_v4.ORIGINAL_ROOT/'selection.json'),
        'baseline_selection_sha256':build_v4.digest(selection)}
    monkeypatch.setattr(build_v4,'ROOT',root)
    monkeypatch.setattr(build_v4,'__file__',str(builder))
    monkeypatch.setattr(build_v4,'_verify_code_tree',lambda *_:None)
    assert build_v4.validate(plan)==(code,source,output,assets)
    plan['builder_sha256']='0'*64
    with pytest.raises(ValueError,match='exact staged builder'):
        build_v4.validate(plan)


def test_v4_final_builder_requires_exact_independently_promoted_policy(tmp_path,monkeypatch):
    root=tmp_path/'runtime';monkeypatch.setattr(build_v4,'ROOT',root)
    code=root/'code/cot-masked-native-gap-v4-code';code.mkdir(parents=True)
    (code/'code-manifest.json').write_text('manifest')
    helper=code/build_v4.PROMOTION_RELATIVE_PATH;helper.parent.mkdir(parents=True)
    helper.write_text('reviewed promotion helper')
    policy_dir=root/'policies/cot-masked-native-gap-v4-policy';policy_dir.mkdir(parents=True)
    policy=policy_dir/'dynamic-trace-exclusions.json';policy.write_text('policy')
    candidate=root/'datasets/cot-masked-native-gap-v4-candidate';candidate.mkdir(parents=True)
    for name in ('manifest.json','BLOCKED.json','dynamic-trace-exclusions.candidate.json',
                 'jobs.private.jsonl','row-receipts.jsonl','exclusions.jsonl'):
        value=('policy' if name=='dynamic-trace-exclusions.candidate.json' else
               '{}\n' if name.endswith('.jsonl') else name)
        (candidate/name).write_text(value)
        (candidate/name).chmod(0o444)
    core={name:build_v4.digest(candidate/name) for name in (
        'manifest.json','BLOCKED.json','dynamic-trace-exclusions.candidate.json',
        'row-receipts.jsonl','exclusions.jsonl')}
    inventory=[{'path':path.name,'bytes':path.stat().st_size,
                'sha256':build_v4.digest(path)} for path in sorted(candidate.iterdir())]
    tree_sha=hashlib.sha256(json.dumps(
        inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    candidate.chmod(0o555)
    candidate_receipt=candidate.with_name(candidate.name+'.build-receipt.json')
    candidate_receipt.write_text(json.dumps({
        'schema':'qwen38-native-gap-v4-build-publication/v1',
        'status':'candidate_blocked_for_review','mode':'candidate','output':str(candidate),
        'runtime_image':build_v4.IMAGE,'code_manifest_sha256':'c'*64,
        'builder_sha256':'b'*64,'tree_sha256':tree_sha,'tree_inventory':inventory,
        'freeze_receipt_sha256':'f'*64,'baseline_selection_sha256':'s'*64,
        'raw_stage_receipt_sha256':'q'*64,'raw_source_inventory_sha256':'r'*64,
        'dynamic_trace_exclusions_sha256':None,
        'files':len(inventory),'bytes':sum(item['bytes'] for item in inventory),
        'core_sha256':core}))
    candidate_receipt.chmod(0o444)
    record={key:None for key in build_v4.PROMOTION_RECEIPT_KEYS}
    record.update({
        'schema':build_v4.PROMOTION_SCHEMA,'status':'complete','runtime_image':build_v4.IMAGE,
        'output_dir':str(policy_dir),'policy_path':str(policy),
        'policy_sha256':build_v4.digest(policy),'policy_bytes':policy.stat().st_size,
        'policy_schema':'qwen38-exact-dynamic-trace-exclusions/v2',
        'code_manifest_path':str(code/'code-manifest.json'),'code_manifest_sha256':'c'*64,
        'promotion_helper_sha256':build_v4.digest(helper),
        'freeze_receipt_sha256':'f'*64,'baseline_selection_sha256':'s'*64,
        'all_non_dynamic_gates_passed':True,'candidate_non_dynamic_gates':{'coverage':True},
        'candidate_bytes_promoted_unchanged':True,'independent_record_derivation':True,
        'reason_counts':{'UnsupportedTrace':1},'record_count':1,'generated_gap_count':2,
        'independent_rerender':{'excluded_traces_rerendered':1,
            'duplicate_exclusions_verified':0,'exact_earlier_owner_proof':True,
            'result_set_sha256':'a'*64},
        'independent_journal_cardinality_audit':{
            'schema':'qwen38-frozen-generation-cardinality/v1','verified':True,
            'review_pending_gaps':4,'selected_candidates':4,'all_candidates':5,
            'generation_invalid_candidates_excluded':1,'unexpected_state_candidates':0,
            'orphan_candidates':0,'orphan_candidate_sources':0,
            'ambiguous_source_event_targets':0,
            'selected_candidates_missing_exact_generate_response':0,
            'all_candidates_missing_exact_generate_response':0,
            'candidate_gap_cardinality_failures':0,
            'review_pending_gaps_missing_candidate':0},
        'candidate_dir':str(candidate),'candidate_build_receipt_path':str(candidate_receipt),
        'candidate_build_receipt_sha256':build_v4.digest(candidate_receipt),
        'candidate_tree_sha256':tree_sha,'candidate_build_core_sha256':core,
        'candidate_manifest_sha256':core['manifest.json'],
        'candidate_blocked_sha256':core['BLOCKED.json'],
        'candidate_policy_path':str(candidate/'dynamic-trace-exclusions.candidate.json'),
        'candidate_policy_sha256':core['dynamic-trace-exclusions.candidate.json'],
        'candidate_jobs_sha256':build_v4.digest(candidate/'jobs.private.jsonl'),
        'candidate_jobs_count':1,
        'candidate_exclusions_sha256':build_v4.digest(candidate/'exclusions.jsonl'),
        'candidate_exclusions_count':1,
        'candidate_row_receipts_sha256':build_v4.digest(candidate/'row-receipts.jsonl'),
        'candidate_row_receipts_count':1})
    promotion=policy_dir/'promotion-receipt.json';promotion.write_text(json.dumps(record))
    promotion.chmod(0o400);policy.chmod(0o400);policy_dir.chmod(0o500)
    plan={'dynamic_trace_exclusions_sha256':build_v4.digest(policy),
        'dynamic_trace_exclusions_promotion_receipt':str(promotion),
        'dynamic_trace_exclusions_promotion_receipt_sha256':build_v4.digest(promotion),
        'code_manifest_sha256':'c'*64,'builder_sha256':'b'*64,
        'freeze_receipt_sha256':'f'*64,'baseline_selection_sha256':'s'*64,
        'raw_stage_receipt_sha256':'q'*64,'raw_source_inventory_sha256':'r'*64}
    assert build_v4._verify_promotion(plan,code,policy)==record
    plan['builder_sha256']='x'*64
    with pytest.raises(ValueError,match='exact candidate build'):
        build_v4._verify_promotion(plan,code,policy)
    plan['builder_sha256']='b'*64
    (candidate/'manifest.json').chmod(0o644)
    (candidate/'manifest.json').write_text('changed after promotion')
    (candidate/'manifest.json').chmod(0o444)
    with pytest.raises(ValueError,match='exact candidate build'):
        build_v4._verify_promotion(plan,code,policy)


def test_v4_builder_persists_sanitized_failure_receipt(tmp_path):
    result=SimpleNamespace(returncode=9,stdout='possible trace text',stderr='more trace text')
    path=build_v4._blocked_receipt({'mode':'final','code_manifest_sha256':'a'*64},
                                   tmp_path/'scratch','converter_outcome',RuntimeError('secret'),result)
    payload=json.loads(path.read_text())
    assert payload['status']=='blocked' and payload['error_type']=='RuntimeError'
    assert payload['stdout_bytes'] and payload['stderr_bytes']
    raw=path.read_text()
    assert 'possible trace text' not in raw and 'more trace text' not in raw and 'secret' not in raw


def test_v4_builder_quarantines_canonical_output_on_postrename_failure(tmp_path,monkeypatch):
    root=tmp_path;datasets=root/'datasets';datasets.mkdir()
    code=root/'code';code.mkdir();source=root/'source';source.mkdir()
    output=datasets/'cot-masked-native-gap-v4-fixture'
    freeze=root/'freeze';freeze.write_text('freeze')
    selection=root/'selection';selection.write_text('selection')
    raw_stage=root/'stage';raw_stage.write_text('stage')
    plan={'mode':'candidate','code_manifest_sha256':'c'*64,
          'freeze_receipt':'freeze_receipt','freeze_receipt_sha256':build_v4.digest(freeze),
          'baseline_selection':'baseline_selection','baseline_selection_sha256':build_v4.digest(selection),
          'raw_stage_receipt':str(raw_stage),'raw_stage_receipt_sha256':build_v4.digest(raw_stage),
          'raw_source_inventory_sha256':'r'*64}
    monkeypatch.setattr(build_v4,'ROOT',root)
    monkeypatch.setattr(build_v4,'validate',lambda _:(code,source,output,code/'assets'))
    inputs={'freeze_receipt':freeze,'baseline_selection':selection}
    monkeypatch.setattr(build_v4,'_host_source',lambda _plan,value:inputs[value])
    def command_for(_plan,scratch,_validated=None):
        (scratch/'export').mkdir()
        return ['fixture']
    monkeypatch.setattr(build_v4,'command_for',command_for)
    monkeypatch.setattr(build_v4.subprocess,'run',lambda *a,**k:
                        SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(build_v4,'_validate_outcome',lambda *a:None)
    monkeypatch.setattr(build_v4,'_freeze_tree',lambda tree:([],[]))
    monkeypatch.setattr(build_v4,'_verify_code_tree',lambda *a:None)
    monkeypatch.setattr(build_v4,'_verify_manifest_shards',lambda *a:{})
    calls=iter(([],RuntimeError('post rename fixture')))
    def inventory(_):
        value=next(calls)
        if isinstance(value,Exception):raise value
        return value
    monkeypatch.setattr(build_v4,'_inventory',inventory)
    with pytest.raises(RuntimeError,match='post rename fixture'):
        build_v4.build(plan)
    assert not output.exists()
    failed=list(datasets.glob('.'+output.name+'.failed-*'))
    assert len(failed)==1 and (failed[0]/'ORCHESTRATION-BLOCKED.json').is_file()


def test_v4_candidate_publisher_rejects_coverage_failure_hidden_by_dynamic_reason(tmp_path):
    tree=tmp_path/'export';tree.mkdir();candidate=tree/'dynamic-trace-exclusions.candidate.json'
    candidate.write_text('{}')
    manifest={
        'full_export':True,'all_train':True,'internal_validation':False,
        'splits':{'train':[]},'unexpected_trace_failures':0,
        'all_frozen_generations_accounted_for':True,
        'cross_arm_no_cot_parity':{'verified':True,'adapter_divergence_rows':0},
        'semantic_action_boundary_audit':{'verified':True},
        'raw_tool_cardinality_audit':{'verified':True},
        'trace_source_membership':{'verified':False,'processed_trace_sources':1,
                                   'accepted_trace_sources':1,'failed_trace_sources':0},
        'replay_source_membership':{'verified':True,'expected_selected_sources':2,
                                    'accepted_sources':2,'excluded_source_count':0},
        'journal_cardinality_audit':{'verified':True},
        'end_of_build_input_stability':{'verified':True},'build_runtime':{'verified':True},
        'dynamic_trace_exclusion_policy':None,
        'dynamic_trace_exclusion_audit':{'verified':False,'expected_record_count':0,
            'observed_record_count':1,'candidate_path':candidate.name,
            'candidate_sha256':build_v4.digest(candidate)},
        'frozen_valid_candidate_count':3,'included_valid_candidate_count':1,
        'excluded_valid_candidate_count':1,'omitted_valid_candidate_count':1,
    }
    manifest_path=tree/'manifest.json';manifest_path.write_text(json.dumps(manifest))
    (tree/'BLOCKED.json').write_text(json.dumps({
        'reason':'dynamic_trace_exclusion_allowlist_mismatch',
        'manifest_sha256':build_v4.digest(manifest_path)}))
    with pytest.raises(RuntimeError,match='expected exclusion-review gate'):
        build_v4._validate_outcome({'mode':'candidate'},tree,SimpleNamespace(returncode=1))


def test_v4_launcher_rejects_netrc_with_any_non_wandb_machine(tmp_path):
    path=tmp_path/'netrc'
    path.write_text('machine api.wandb.ai login user password fixture\n'
                    'machine unrelated.example login user password unrelated\n')
    path.chmod(0o600)
    with pytest.raises(ValueError,match='exactly one W&B machine stanza'):
        launch_v4._verify_minimal_wandb_netrc(path)
    path.write_text('machine api.wandb.ai login user password first\n'
                    'machine api.wandb.ai login user password second\n')
    with pytest.raises(ValueError,match='exactly one W&B machine stanza'):
        launch_v4._verify_minimal_wandb_netrc(path)


def test_v4_wandb_probe_requires_exact_first_metric_values_and_url(monkeypatch):
    expected={'entity':'yeta','project':'yeto-h200','run_id':'abc123xy',
              'name':'fixture','training_step':0,'num_label_tokens':123}
    good={'viewer':'walden-lee','entity':'yeta','project':'yeto-h200','run_id':'abc123xy',
          'name':'fixture','state':'running',
          'url':'https://wandb.ai/yeta/yeto-h200/runs/abc123xy',
          'first_optimizer_metric':{'history_step':0,'loss':0.5,'lr':1e-5,
                                    'num_label_tokens':123}}
    monkeypatch.setattr(launch_v4.subprocess,'run',lambda *args,**kwargs:
                        SimpleNamespace(returncode=0,stdout=json.dumps(good),stderr=''))
    assert launch_v4._wandb_probe('container',expected)==good
    for key,value in (('url',''),('state','finished')):
        bad={**good,key:value}
        monkeypatch.setattr(launch_v4.subprocess,'run',lambda *args,_bad=bad,**kwargs:
                            SimpleNamespace(returncode=0,stdout=json.dumps(_bad),stderr=''))
        assert launch_v4._wandb_probe('container',expected) is None
    bad={**good,'first_optimizer_metric':{**good['first_optimizer_metric'],
                                          'num_label_tokens':122}}
    monkeypatch.setattr(launch_v4.subprocess,'run',lambda *args,**kwargs:
                        SimpleNamespace(returncode=0,stdout=json.dumps(bad),stderr=''))
    assert launch_v4._wandb_probe('container',expected) is None


def test_v4_external_wandb_receipt_binds_exact_marker_metric_and_checkpoint():
    helper_sha='h'*64;marker={'schema':'qwen38-native-gap-v4-first-update-data-semantics/v1',
        'status':'passed','manifest_sha256':'m'*64,'config_canonical_sha256':'c'*64,
        'expected_once_shifted_label_tokens':123,'reported_num_label_tokens':123,
        'training_step':0}
    expected={'entity':'yeta','project':'yeto-h200','run_id':'abc123xy','name':'fixture',
              'manifest_sha256':'m'*64,'config_canonical_sha256':'c'*64}
    query={'viewer':'walden-lee','entity':'yeta','project':'yeto-h200','run_id':'abc123xy',
        'name':'fixture','state':'running','url':'https://wandb.ai/yeta/yeto-h200/runs/abc123xy',
        'first_optimizer_metric':{'history_step':0,'loss':0.4,'lr':1e-5,'num_label_tokens':123}}
    state={'launch_sha256':'l'*64,'marker_sha256':'u'*64,'marker':marker,
        'helper':{'path':external_wandb_v4.HELPER_PATH,'bytes':123,'sha256':helper_sha},
        'launch':{'schema':'qwen38-native-gap-v4-gpu-launch/v1',
            'status':'node_first_update_observed_pending_external_wandb',
            'first_update_sha256':'u'*64,'wandb_expected':expected,
            'update1_checkpoint':{'status':'passed','latest_target':'epoch_0_step_0',
                                  'latest_resolver_exact':True}}}
    assert external_wandb_v4.validate(state,query,helper_sha)==(
        expected,query['first_optimizer_metric'])
    query['first_optimizer_metric']['num_label_tokens']=124
    with pytest.raises(ValueError,match='does not match update 1'):
        external_wandb_v4.validate(state,query,helper_sha)


def test_v4_external_wandb_refuses_unverified_secret_cleanup(monkeypatch):
    monkeypatch.setattr(external_wandb_v4.subprocess,'run',lambda *args,**kwargs:
                        SimpleNamespace(returncode=1,stdout=b'',stderr=b''))
    assert external_wandb_v4._remove_secret('fixture','/dev/shm/yeta-wandb-netrc-'+'a'*32) is False


def test_v4_external_wandb_reads_root_owned_state_through_sudo(monkeypatch):
    payload={'launch':{},'launch_sha256':'a'*64,'marker':{},'marker_sha256':'b'*64,
             'helper':{'path':external_wandb_v4.HELPER_PATH,'bytes':1,'sha256':'c'*64}}
    seen=[]
    def remote(node,argv,**kwargs):
        seen.append((node,argv,kwargs))
        return json.dumps(payload).encode()
    monkeypatch.setattr(external_wandb_v4,'_remote',remote)
    assert external_wandb_v4._state('fixture-node',
        '/data/sft_baseline_20260908/runs/cot-masked-native-gap-v4-fixture')==payload
    assert len(seen)==1
    assert seen[0][0]=='fixture-node'
    assert seen[0][1][:3]==['sudo','-n','python3']


def test_v4_external_wandb_query_uses_writable_tmpfs_cache():
    secret='/dev/shm/yeta-wandb-netrc-'+'a'*32
    expected={'entity':'yeta','project':'yeto-h200','run_id':'abc123xy'}
    command=external_wandb_v4._query_command(secret,expected)
    assert '--read-only' in command
    assert command[command.index('--tmpfs')+1]=='/tmp:rw,noexec,nosuid,size=512m'
    environment=[command[index+1] for index,value in enumerate(command) if value=='-e']
    assert environment==['WANDB_CACHE_DIR=/tmp/wandb-cache',
                         'XDG_CACHE_HOME=/tmp/wandb-cache']
    assert ('type=bind,src='+secret+',dst=/root/.netrc,readonly') in command


def test_v4_external_wandb_query_accepts_info_before_one_final_json_object():
    raw=(b'wandb: Network retry succeeded\n'
         b'wandb: Querying run history\n'
         b'{"run_id":"abc123xy","state":"running"}\n')
    assert external_wandb_v4._parse_final_json_object(raw)=={
        'run_id':'abc123xy','state':'running'}


@pytest.mark.parametrize('raw',[b'{"first":1}\n{"second":2}\n',
                                b'{"run_id":"abc123xy"}\nwandb: trailing output\n',
                                b'wandb: only informational output\n',
                                b'["not","an","object"]\n'])
def test_v4_external_wandb_query_rejects_ambiguous_or_nonfinal_json(raw):
    with pytest.raises(ValueError,match='W&B query'):
        external_wandb_v4._parse_final_json_object(raw)


def test_wandb_auth_stager_extracts_one_machine_and_shell_quotes_receiver(tmp_path,monkeypatch):
    source=tmp_path/'netrc'
    source.write_text('machine api.wandb.ai login user password wandbfixture\n'
                      'machine unrelated.example login other password unrelatedfixture\n')
    seen={}
    def run(argv,**kwargs):
        seen.update(argv=argv,kwargs=kwargs)
        return SimpleNamespace(returncode=0,stdout=json.dumps({
            'path':'/dev/shm/yeta-wandb-netrc-fixture','bytes':64,'mode':'0600',
            'owner':'root','machines':['api.wandb.ai']}).encode(),stderr=b'')
    monkeypatch.setattr(wandb_auth_v4.subprocess,'run',run)
    result=wandb_auth_v4.stage(source,node='fixture-node')
    assert result['machines']==['api.wandb.ai']
    assert b'wandbfixture' in seen['kwargs']['input']
    assert b'unrelatedfixture' not in seen['kwargs']['input']
    remote=seen['argv'][-1]
    assert 'wandbfixture' not in remote and 'unrelatedfixture' not in remote
    assert "sudo -n python3 -c '" in remote


def test_v4_launcher_reuses_one_allowed_receipt_and_unlinks_ephemeral_auth(tmp_path,monkeypatch):
    root=tmp_path;(root/'runs').mkdir();run=root/'runs/cot-masked-native-gap-v4-run'
    code=root/'code';dataset=root/'dataset';code.mkdir();dataset.mkdir()
    netrc=root/'wandb-netrc';netrc.write_text('secret fixture\n');netrc.chmod(0o600)
    plan={'schema':'qwen38-native-gap-v4-launch-plan/v1','wandb_netrc':str(netrc),
          'wandb_run_id':'abc123xy','first_metric_timeout_seconds':3600,
          'config_dir':str(root/'configs/cot-masked-native-gap-v4-run')}
    monkeypatch.setattr(launch_v4,'ROOT',root)
    config={'wandb':{'name':'qwen38-27b-generated-cot-masked-native-gap-v4-fixture-train'}}
    cpu={'manifest_sha256':'c'*64}
    monkeypatch.setattr(launch_v4,'validate',lambda _:(run,code,dataset,cpu,config,netrc))
    monkeypatch.setattr(launch_v4,'command_for',lambda _:('v4-fixture',['docker','run']))
    def output(argv,**kwargs):
        if argv[:2]==['nvidia-smi','--query-compute-apps=pid,used_memory']:
            return ''
        if argv[:2]==['nvidia-smi','--query-gpu=index,memory.used,memory.total']:
            return '\n'.join(f'{index}, 0, 143000' for index in range(8))
        if argv[:3]==['docker','ps','-a']:
            return ''
        raise AssertionError(argv)
    monkeypatch.setattr(launch_v4.subprocess,'check_output',output)
    monkeypatch.setattr(launch_v4.subprocess,'run',lambda *args,**kwargs:
                        SimpleNamespace(returncode=0,stdout='a'*64+'\n',stderr=''))
    monkeypatch.setattr(launch_v4,'verify_first_update',lambda **kwargs:{
        'first_update_sha256':'b'*64,'wandb_node_probe':{
            'url':'https://wandb.ai/yeta/yeto-h200/runs/abc123xy'},
        'update1_checkpoint':{'status':'passed'}})
    assert launch_v4.launch(plan)==0
    receipt=json.loads((run/'launch.json').read_text())
    assert receipt['status']=='node_first_update_observed_pending_external_wandb'
    assert receipt['container']=='a'*64
    assert 'wandb_netrc' not in receipt and not netrc.exists()
    assert str(netrc) not in (run/'launch.json').read_text()
    assert not (run/'launch.final.json').exists()
