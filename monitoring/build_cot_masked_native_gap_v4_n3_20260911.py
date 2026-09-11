"""Build and atomically publish a direct-path v4 dataset in the pinned CPU image."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import uuid

from monitoring.launch_cot_masked_native_gap_v4_n3_20260910 import (
    IMAGE, IMAGE_REF, ROOT, _verify_code_tree, digest,
)
from monitoring.promote_masked_native_gap_v4_exclusions_20260911 import (
    PROMOTION_RECEIPT_KEYS, PROMOTION_SCHEMA,
)

ORIGINAL_ROOT=Path('/mnt/lvm_data/sft_analysis/sft_baseline_20260908')
BUILDER_RELATIVE_PATH='monitoring/build_cot_masked_native_gap_v4_n3_20260911.py'
PROMOTION_RELATIVE_PATH='monitoring/promote_masked_native_gap_v4_exclusions_20260911.py'


def _hash(value):
    return (isinstance(value,str) and len(value)==64
            and all(character in '0123456789abcdef' for character in value))


def _strict_json_file(path,label):
    path=Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(label+' must be a regular file')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError(label+' contains a duplicate JSON key')
            result[key]=value
        return result
    def constant(value):
        raise ValueError(label+' contains a non-finite number: '+value)
    try:
        return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_constant=constant)
    except (json.JSONDecodeError,UnicodeDecodeError) as error:
        raise ValueError(label+' is not strict JSON') from error


def _strict_jsonl_count(path,label):
    raw=Path(path).read_bytes()
    if not raw or not raw.endswith(b'\n') or any(not line for line in raw.splitlines()):
        raise ValueError(label+' must be nonempty newline-terminated JSONL')
    for index,line in enumerate(raw.splitlines(),1):
        try:
            value=json.loads(line,parse_constant=lambda item:(_ for _ in ()).throw(
                ValueError(label+' contains a non-finite value')))
        except (json.JSONDecodeError,UnicodeDecodeError) as error:
            raise ValueError(label+f' line {index} is not JSON') from error
        if not isinstance(value,dict):
            raise ValueError(label+f' line {index} is not an object')
    return len(raw.splitlines())


def _verify_promotion(plan,code,policy):
    receipt=Path(plan.get('dynamic_trace_exclusions_promotion_receipt',''))
    expected_directory=ROOT/'policies'
    if (policy.name!='dynamic-trace-exclusions.json'
            or policy.parent.parent!=expected_directory
            or not policy.parent.name.startswith('cot-masked-native-gap-v4-')
            or policy.parent.is_symlink() or policy.parent.resolve(strict=True)!=policy.parent
            or policy.parent.stat().st_mode&0o222
            or receipt!=policy.parent/'promotion-receipt.json'
            or not receipt.is_absolute() or receipt.is_symlink() or not receipt.is_file()
            or receipt.stat().st_mode&0o222
            or digest(receipt)!=plan.get('dynamic_trace_exclusions_promotion_receipt_sha256')):
        raise ValueError('Final policy is not bound to one immutable v4 promotion receipt')
    record=_strict_json_file(receipt,'promotion receipt')
    helper=code/PROMOTION_RELATIVE_PATH
    gates=record.get('candidate_non_dynamic_gates')
    rerender=record.get('independent_rerender')
    journal=record.get('independent_journal_cardinality_audit')
    reason_counts=record.get('reason_counts')
    if (set(record)!=PROMOTION_RECEIPT_KEYS
            or record.get('schema')!=PROMOTION_SCHEMA or record.get('status')!='complete'
            or record.get('runtime_image')!=IMAGE
            or record.get('output_dir')!=str(policy.parent)
            or record.get('policy_path')!=str(policy)
            or record.get('policy_sha256')!=plan.get('dynamic_trace_exclusions_sha256')
            or record.get('policy_bytes')!=policy.stat().st_size
            or record.get('policy_schema')!='qwen38-exact-dynamic-trace-exclusions/v2'
            or record.get('code_manifest_path')!=str(code/'code-manifest.json')
            or record.get('code_manifest_sha256')!=plan.get('code_manifest_sha256')
            or helper.is_symlink() or not helper.is_file()
            or record.get('promotion_helper_sha256')!=digest(helper)
            or record.get('freeze_receipt_sha256')!=plan.get('freeze_receipt_sha256')
            or record.get('baseline_selection_sha256')!=plan.get('baseline_selection_sha256')
            or record.get('all_non_dynamic_gates_passed') is not True
            or not isinstance(gates,dict) or not gates
            or any(value is not True for value in gates.values())
            or record.get('candidate_bytes_promoted_unchanged') is not True
            or record.get('independent_record_derivation') is not True
            or not isinstance(reason_counts,dict)
            or any(not isinstance(key,str) or type(value) is not int or value<0
                   for key,value in reason_counts.items())
            or sum(reason_counts.values())!=record.get('record_count')
            or type(record.get('generated_gap_count')) is not int
            or record['generated_gap_count']<0
            or not isinstance(rerender,dict)
            or rerender.get('excluded_traces_rerendered')!=record.get('record_count')
            or rerender.get('duplicate_exclusions_verified')!=sum(
                value for key,value in reason_counts.items() if key.startswith('duplicate_'))
            or rerender.get('exact_earlier_owner_proof') is not True
            or not _hash(rerender.get('result_set_sha256'))
            or not isinstance(journal,dict) or journal.get('verified') is not True
            or journal.get('selected_candidates')!=journal.get('review_pending_gaps')
            or journal.get('all_candidates')!=journal.get('selected_candidates',-1)+
                journal.get('generation_invalid_candidates_excluded',-1)
            or any(journal.get(key)!=0 for key in (
                'unexpected_state_candidates','orphan_candidates','orphan_candidate_sources',
                'ambiguous_source_event_targets','selected_candidates_missing_exact_generate_response',
                'all_candidates_missing_exact_generate_response','candidate_gap_cardinality_failures',
                'review_pending_gaps_missing_candidate'))):
        raise ValueError('Promotion receipt does not prove the reviewed v4 policy')
    candidate=Path(record.get('candidate_dir',''))
    build_receipt=Path(record.get('candidate_build_receipt_path',''))
    if (candidate.parent!=ROOT/'datasets' or candidate.is_symlink() or not candidate.is_dir()
            or candidate.stat().st_mode&0o222
            or build_receipt!=candidate.with_name(candidate.name+'.build-receipt.json')
            or build_receipt.is_symlink() or not build_receipt.is_file()
            or build_receipt.stat().st_mode&0o222
            or digest(build_receipt)!=record.get('candidate_build_receipt_sha256')):
        raise ValueError('Promotion receipt candidate provenance changed')
    candidate_receipt=_strict_json_file(build_receipt,'candidate build receipt')
    candidate_inventory=_inventory(candidate)
    inventory_by_path={item['path']:item for item in candidate_inventory}
    candidate_tree_sha=hashlib.sha256(json.dumps(
        candidate_inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if (candidate_receipt.get('schema')!='qwen38-native-gap-v4-build-publication/v1'
            or candidate_receipt.get('status')!='candidate_blocked_for_review'
            or candidate_receipt.get('mode')!='candidate'
            or candidate_receipt.get('output')!=str(candidate)
            or candidate_receipt.get('runtime_image')!=IMAGE
            or candidate_receipt.get('code_manifest_sha256')!=plan.get('code_manifest_sha256')
            or candidate_receipt.get('builder_sha256')!=plan.get('builder_sha256')
            or candidate_receipt.get('freeze_receipt_sha256')!=plan.get('freeze_receipt_sha256')
            or candidate_receipt.get('baseline_selection_sha256')!=plan.get('baseline_selection_sha256')
            or candidate_receipt.get('raw_stage_receipt_sha256')!=plan.get('raw_stage_receipt_sha256')
            or candidate_receipt.get('raw_source_inventory_sha256')!=
                plan.get('raw_source_inventory_sha256')
            or candidate_receipt.get('dynamic_trace_exclusions_sha256') is not None
            or candidate_receipt.get('tree_inventory')!=candidate_inventory
            or candidate_receipt.get('files')!=len(candidate_inventory)
            or candidate_receipt.get('bytes')!=sum(item['bytes'] for item in candidate_inventory)
            or candidate_tree_sha!=candidate_receipt.get('tree_sha256')
            or candidate_receipt.get('tree_sha256')!=record.get('candidate_tree_sha256')
            or candidate_receipt.get('core_sha256')!=record.get('candidate_build_core_sha256')
            or candidate_receipt.get('core_sha256',{}).get('manifest.json')!=
                record.get('candidate_manifest_sha256')
            or candidate_receipt.get('core_sha256',{}).get('BLOCKED.json')!=
                record.get('candidate_blocked_sha256')
            or candidate_receipt.get('core_sha256',{}).get(
                'dynamic-trace-exclusions.candidate.json')!=record.get('candidate_policy_sha256')):
        raise ValueError('Promoted policy does not bind the exact candidate build')
    if (set(candidate_receipt.get('core_sha256',{}))!={
            'manifest.json','row-receipts.jsonl','exclusions.jsonl',
            'dynamic-trace-exclusions.candidate.json','BLOCKED.json'}
            or any(digest(candidate/name)!=value for name,value in
                   candidate_receipt['core_sha256'].items())):
        raise ValueError('Promoted policy candidate core bytes changed')
    named=(('candidate_jobs','jobs.private.jsonl'),
           ('candidate_exclusions','exclusions.jsonl'),
           ('candidate_row_receipts','row-receipts.jsonl'))
    if (record.get('candidate_policy_path')!=str(candidate/'dynamic-trace-exclusions.candidate.json')
            or record.get('candidate_policy_sha256')!=record.get('policy_sha256')
            or (candidate/'dynamic-trace-exclusions.candidate.json').read_bytes()!=policy.read_bytes()
            or any(record.get(prefix+'_sha256')!=inventory_by_path.get(name,{}).get('sha256')
                   or record.get(prefix+'_count')!=_strict_jsonl_count(candidate/name,name)
                   for prefix,name in named)):
        raise ValueError('Promotion receipt named candidate evidence changed')
    return record


def _inside(path,parent):
    try:Path(path).relative_to(parent);return True
    except ValueError:return False


def _host_source(plan,path):
    supplied=Path(path)
    if not supplied.is_absolute() or not _inside(supplied,ORIGINAL_ROOT):
        raise ValueError('Frozen inputs must retain their pinned absolute namespace')
    return Path(plan['staged_source_root'])/supplied.relative_to(ORIGINAL_ROOT)


def validate(plan):
    if plan.get('schema')!='qwen38-native-gap-v4-build-plan/v1' or plan.get('mode') not in {'candidate','final'}:
        raise ValueError('Wrong v4 build plan')
    code=Path(plan['code']);source=Path(plan['staged_source_root']);output=Path(plan['output'])
    stale=(list(output.parent.glob('.'+output.name+'.building-*'))
           +list(output.parent.glob('.'+output.name+'.failed-*'))
           +list(output.parent.glob('.'+output.name+'.build-receipt.json.tmp-*')))
    if (code.parent!=ROOT/'code' or not code.name.startswith('cot-masked-native-gap-v4-')
            or code.is_symlink() or code.resolve(strict=True)!=code
            or not source.is_absolute() or source.name!='sft_baseline_20260908'
            or source.is_symlink() or source.resolve(strict=True)!=source
            or output.parent!=ROOT/'datasets' or not output.name.startswith('cot-masked-native-gap-v4-')
            or output.exists() or output.is_symlink()
            or output.with_name(output.name+'.build-receipt.json').exists()
            or output.with_name(output.name+'.build-receipt.json').is_symlink()
            or stale):
        raise ValueError('Build paths must use fresh canonical v4 namespaces')
    _verify_code_tree(code,plan['code_manifest_sha256'])
    builder=code/BUILDER_RELATIVE_PATH
    if (Path(__file__).resolve(strict=True)!=builder.resolve(strict=True)
            or digest(builder)!=plan.get('builder_sha256')):
        raise ValueError('Build plan is not bound to this exact staged builder')
    stage_receipt=Path(plan.get('raw_stage_receipt',''))
    if (not stage_receipt.is_absolute() or stage_receipt.is_symlink()
            or not stage_receipt.is_file()
            or digest(stage_receipt)!=plan.get('raw_stage_receipt_sha256')):
        raise ValueError('Frozen raw-stage receipt changed')
    staged=json.loads(stage_receipt.read_bytes())
    if (staged.get('schema')!='qwen38-masked-v4-frozen-raw-n3-stage/v1'
            or staged.get('status')!='complete'
            or staged.get('all_member_hashes_verified_after_transfer') is not True
            or staged.get('tree_files_mode')!='0400' or staged.get('tree_directories_mode')!='0500'
            or source!=Path(staged.get('final_path',''))/'tree/mnt/lvm_data/sft_analysis/sft_baseline_20260908'
            or staged.get('source_inventory_sha256')!=plan.get('raw_source_inventory_sha256')):
        raise ValueError('Frozen raw-stage receipt does not bind this immutable source tree')
    assets=code/'training/qwen38_no_cot/assets'
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
        if (assets/name).is_symlink() or not (assets/name).is_file():
            raise ValueError('Pinned tokenizer/template asset is absent from runtime closure')
    for key,sha_key in (('freeze_receipt','freeze_receipt_sha256'),
                        ('baseline_selection','baseline_selection_sha256')):
        host=_host_source(plan,plan[key])
        if host.is_symlink() or not host.is_file() or digest(host)!=plan[sha_key]:
            raise ValueError('Frozen build input changed: '+key)
    dynamic=plan.get('dynamic_trace_exclusions')
    if plan['mode']=='candidate' and (dynamic is not None or plan.get('dynamic_trace_exclusions_sha256') is not None):
        raise ValueError('Candidate discovery must not receive an exclusion allowlist')
    if plan['mode']=='final':
        policy=Path(dynamic or '')
        if (not policy.is_absolute() or not _inside(policy,ROOT/'policies') or policy.is_symlink()
                or not policy.is_file() or digest(policy)!=plan.get('dynamic_trace_exclusions_sha256')
                or policy.stat().st_mode&0o222):
            raise ValueError('Final build requires one immutable reviewed v4 exclusion policy')
        _verify_promotion(plan,code,policy)
    return code,source,output,assets


def command_for(plan,scratch,_validated=None):
    code,source,_,assets=_validated or validate(plan)
    command=['docker','run','--rm','--network','none','--read-only','--cap-drop','ALL',
        '--cap-add','DAC_READ_SEARCH','--security-opt','no-new-privileges','--user','0:0',
        '--tmpfs','/tmp:rw,noexec,nosuid,size=32g',
        '--mount','type=bind,src='+str(code)+',dst=/workspace/yeto,readonly',
        '--mount','type=bind,src='+str(source)+',dst='+str(ORIGINAL_ROOT)+',readonly',
        '--mount','type=bind,src='+str(scratch)+',dst=/output',
        '-e','PYTHONPATH=/workspace/yeto','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','TOKENIZERS_PARALLELISM=false','-e','YETA_TRAINING_IMAGE='+IMAGE,
        '-w','/workspace/yeto',IMAGE_REF,'python','-B','-m',
        'training.qwen38_native_gap_v3.prepare_masked_full',
        '--freeze-receipt',plan['freeze_receipt'],'--tokenizer-dir',str(assets).replace(str(code),'/workspace/yeto'),
        '--output','/output/export','--workers',str(plan.get('workers',32)),
        '--shard-rows',str(plan.get('shard_rows',256)),'--all-train',
        '--baseline-selection',plan['baseline_selection'],
        '--baseline-selection-sha256',plan['baseline_selection_sha256']]
    if plan['mode']=='final':
        policy=Path(plan['dynamic_trace_exclusions'])
        command[command.index('-e'):command.index('-e')]=[
            '--mount','type=bind,src='+str(policy)+',dst=/policies/dynamic.json,readonly']
        command += ['--dynamic-trace-exclusions','/policies/dynamic.json',
                    '--dynamic-trace-exclusions-sha256',plan['dynamic_trace_exclusions_sha256']]
    return command


def _freeze_tree(tree):
    members=sorted(tree.rglob('*'))
    files=[];directories=[]
    for path in members:
        mode=path.lstat().st_mode
        if path.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError('Build output contains a non-regular member')
        (files if stat.S_ISREG(mode) else directories).append(path)
    for path in files:
        descriptor=os.open(path,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        path.chmod(0o444)
    for path in sorted(directories,key=lambda value:len(value.parts),reverse=True):
        descriptor=os.open(path,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        path.chmod(0o555)
    descriptor=os.open(tree,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    tree.chmod(0o555)
    return files,directories


def _inventory(tree):
    members=sorted(tree.rglob('*'))
    paths=[]
    for path in members:
        mode=path.lstat().st_mode
        if (path.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                or path.stat().st_mode&0o222):
            raise RuntimeError('Published dataset contains a mutable or non-regular member')
        if stat.S_ISREG(mode):paths.append(path)
    def one(path):
        before=path.stat();value=digest(path);after=path.stat()
        signature=lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns)
        if path.is_symlink() or signature(before)!=signature(after):
            raise RuntimeError('Published dataset member changed while hashing')
        return {'path':str(path.relative_to(tree)),'bytes':after.st_size,'sha256':value}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(one,paths))
    if len({item['path'].casefold() for item in records})!=len(records):
        raise RuntimeError('Published dataset has colliding member names')
    return records


def _verify_manifest_shards(tree,records):
    available={item['path']:item for item in records}
    manifest=json.loads((tree/'manifest.json').read_bytes())
    for entries in manifest.get('splits',{}).values():
        for item in entries:
            if available.get(item.get('path'))!={key:item[key] for key in ('path','bytes','sha256')}:
                raise RuntimeError('Published shard differs from the portable manifest')
    return manifest


def _blocked_receipt(plan,scratch,stage,error,result=None):
    """Persist bounded failure evidence without copying trace/error text."""
    payload={'schema':'qwen38-native-gap-v4-build-blocked/v1','status':'blocked',
        'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'mode':plan.get('mode'),'stage':stage,'error_type':type(error).__name__,
        'scratch':str(scratch),'runtime_image':IMAGE,
        'code_manifest_sha256':plan.get('code_manifest_sha256'),
        'builder_sha256':plan.get('builder_sha256'),
        'freeze_receipt_sha256':plan.get('freeze_receipt_sha256'),
        'raw_stage_receipt_sha256':plan.get('raw_stage_receipt_sha256'),
        'raw_source_inventory_sha256':plan.get('raw_source_inventory_sha256'),
        'baseline_selection_sha256':plan.get('baseline_selection_sha256'),
        'dynamic_trace_exclusions_sha256':plan.get('dynamic_trace_exclusions_sha256')}
    if plan.get('dynamic_trace_exclusions_promotion_receipt_sha256') is not None:
        payload['dynamic_trace_exclusions_promotion_receipt_sha256']=plan[
            'dynamic_trace_exclusions_promotion_receipt_sha256']
    if result is not None:
        stdout=(result.stdout or '').encode();stderr=(result.stderr or '').encode()
        payload.update(container_exit_code=result.returncode,
            stdout_bytes=len(stdout),stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            stderr_bytes=len(stderr),stderr_sha256=hashlib.sha256(stderr).hexdigest())
    scratch.mkdir(mode=0o700,parents=True,exist_ok=True)
    destination=scratch/'ORCHESTRATION-BLOCKED.json'
    raw=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with destination.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    destination.chmod(0o400)
    return destination


def _validate_outcome(plan,tree,result):
    blocked=tree/'BLOCKED.json';complete=tree/'COMPLETE.json'
    if plan['mode']=='candidate':
        manifest=tree/'manifest.json'
        if not manifest.is_file() or manifest.is_symlink():
            raise RuntimeError('Candidate build did not publish its review manifest')
        contents=json.loads(manifest.read_bytes());block=json.loads(blocked.read_bytes()) if blocked.is_file() else {}
        dynamic=contents.get('dynamic_trace_exclusion_audit',{})
        candidate_path=tree/dynamic.get('candidate_path','missing')
        candidate_ok=(candidate_path.is_file() and not candidate_path.is_symlink()
                      and digest(candidate_path)==dynamic.get('candidate_sha256'))
        trace=contents.get('trace_source_membership',{})
        replay=contents.get('replay_source_membership',{})
        journal=contents.get('journal_cardinality_audit',{})
        gates=(
            contents.get('full_export') is True,
            contents.get('all_train') is True,
            contents.get('internal_validation') is False,
            set(contents.get('splits',{}))=={'train'},
            contents.get('unexpected_trace_failures')==0,
            contents.get('all_frozen_generations_accounted_for') is True,
            contents.get('cross_arm_no_cot_parity',{}).get('verified') is True,
            contents.get('cross_arm_no_cot_parity',{}).get('adapter_divergence_rows')==0,
            contents.get('semantic_action_boundary_audit',{}).get('verified') is True,
            contents.get('raw_tool_cardinality_audit',{}).get('verified') is True,
            trace.get('verified') is True,
            trace.get('processed_trace_sources')==trace.get('accepted_trace_sources',0)+trace.get('failed_trace_sources',0),
            replay.get('verified') is True,
            replay.get('expected_selected_sources')==replay.get('accepted_sources'),
            replay.get('excluded_source_count')==0,
            journal.get('verified') is True,
            contents.get('end_of_build_input_stability',{}).get('verified') is True,
            contents.get('build_runtime',{}).get('verified') is True,
            contents.get('dynamic_trace_exclusion_policy') is None,
            dynamic.get('verified') is False,
            dynamic.get('expected_record_count')==0,
            type(dynamic.get('observed_record_count')) is int and dynamic.get('observed_record_count')>0,
            candidate_ok,
            contents.get('frozen_valid_candidate_count')==
                contents.get('included_valid_candidate_count',-1)+contents.get('excluded_valid_candidate_count',-1)+
                contents.get('omitted_valid_candidate_count',-1),
        )
        if (result.returncode==0 or complete.exists() or not blocked.is_file()
                or block.get('reason')!='dynamic_trace_exclusion_allowlist_mismatch'
                or block.get('manifest_sha256')!=digest(manifest) or not all(gates)):
            raise RuntimeError('Candidate build did not stop at the expected exclusion-review gate')
    elif result.returncode or blocked.exists() or not complete.is_file():
        raise RuntimeError('Final build did not publish atomic COMPLETE')
    if complete.exists():
        record=json.loads(complete.read_bytes())
        for key,name in (('manifest','manifest.json'),('index','index.json')):
            if record.get(key)!={'path':name,'sha256':digest(tree/name)}:
                raise RuntimeError('Final COMPLETE does not bind the exact portable dataset tree')


def build(plan):
    started=datetime.datetime.now(datetime.timezone.utc)
    code,source,output,_=validate(plan)
    (ROOT/'datasets').mkdir(mode=0o700,exist_ok=True)
    scratch=output.with_name('.'+output.name+'.building-'+uuid.uuid4().hex)
    scratch.mkdir(mode=0o700)
    command=command_for(plan,scratch,(code,source,output,
                                      code/'training/qwen38_no_cot/assets'))
    result=None;stage='container_conversion'
    destination=output.with_name(output.name+'.build-receipt.json')
    pending_receipt=destination.with_name('.'+destination.name+'.tmp-'+uuid.uuid4().hex)
    try:
        result=subprocess.run(command,text=True,capture_output=True)
        export=scratch/'export'
        if not export.is_dir() or export.is_symlink():
            raise RuntimeError('Builder produced no inspectable export tree')
        stage='converter_outcome';_validate_outcome(plan,export,result)
        stage='publication_freeze';files,directories=_freeze_tree(export)
        stage='end_of_build_identity';_verify_code_tree(code,plan['code_manifest_sha256'])
        for key,sha_key in (('freeze_receipt','freeze_receipt_sha256'),
                            ('baseline_selection','baseline_selection_sha256')):
            if digest(_host_source(plan,plan[key]))!=plan[sha_key]:
                raise RuntimeError('Frozen build input changed during conversion: '+key)
        if (plan['mode']=='final'
                and digest(plan['dynamic_trace_exclusions'])!=plan['dynamic_trace_exclusions_sha256']):
            raise RuntimeError('Reviewed dynamic exclusion policy changed during conversion')
        if digest(plan['raw_stage_receipt'])!=plan['raw_stage_receipt_sha256']:
            raise RuntimeError('Frozen raw-stage receipt changed during conversion')
        stage='prepublication_inventory';records=_inventory(export)
        manifest=_verify_manifest_shards(export,records)
        stage='atomic_publication';export.rename(output);scratch.rmdir()
        parent=os.open(output.parent,os.O_RDONLY)
        try:os.fsync(parent)
        finally:os.close(parent)
        if _inventory(output)!=records:
            raise RuntimeError('Published tree changed across its atomic rename')
        stage='publication_receipt'
        tree_sha=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        core={name:digest(output/name) for name in (
            ['manifest.json','row-receipts.jsonl','exclusions.jsonl','dynamic-trace-exclusions.candidate.json']+
            (['BLOCKED.json'] if plan['mode']=='candidate' else ['index.json','COMPLETE.json']))}
        finished=datetime.datetime.now(datetime.timezone.utc)
        receipt={'schema':'qwen38-native-gap-v4-build-publication/v1','status':'candidate_blocked_for_review'
                 if plan['mode']=='candidate' else 'complete','at':finished.isoformat(),
                 'mode':plan['mode'],'output':str(output),'runtime_image':IMAGE,
                 'code_manifest_sha256':plan['code_manifest_sha256'],
                 'builder_sha256':plan['builder_sha256'],
                 'freeze_receipt_sha256':plan['freeze_receipt_sha256'],
                 'raw_stage_receipt_sha256':plan['raw_stage_receipt_sha256'],
                 'raw_source_inventory_sha256':plan['raw_source_inventory_sha256'],
                 'baseline_selection_sha256':plan['baseline_selection_sha256'],
                 'dynamic_trace_exclusions_sha256':plan.get('dynamic_trace_exclusions_sha256'),
                 'dynamic_trace_exclusions_promotion_receipt_sha256':
                     plan.get('dynamic_trace_exclusions_promotion_receipt_sha256'),
                 'files':len(records),'bytes':sum(item['bytes'] for item in records),
                 'tree_sha256':tree_sha,'tree_inventory':records,'core_sha256':core,
                 'manifest_counts':manifest.get('counts'),'manifest_token_counts':manifest.get('token_counts'),
                 'manifest_source_jobs':manifest.get('source_jobs'),
                 'manifest_frozen_valid_candidate_count':manifest.get('frozen_valid_candidate_count'),
                 'directories':len(directories)+1,'immutable_mode':'files0444_dirs0555',
                 'atomic_direct_dataset_publication':True,'container_exit_code':result.returncode,
                 'cpu_only':True,'network':'none','capabilities':['DAC_READ_SEARCH'],
                 'elapsed_seconds':(finished-started).total_seconds()}
        raw=(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        with pending_receipt.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        pending_receipt.chmod(0o444);pending_receipt.replace(destination)
        parent=os.open(output.parent,os.O_RDONLY)
        try:os.fsync(parent)
        finally:os.close(parent)
        return receipt
    except Exception as error:
        pending_receipt.unlink(missing_ok=True)
        if output.exists():
            evidence=output.with_name('.'+output.name+'.failed-'+uuid.uuid4().hex)
            output.rename(evidence)
        elif scratch.exists():
            evidence=scratch
        else:
            evidence=output.with_name('.'+output.name+'.failed-'+uuid.uuid4().hex)
        if evidence.exists():
            evidence.chmod(0o700)
        if destination.exists():
            evidence.mkdir(mode=0o700,parents=True,exist_ok=True)
            destination.rename(evidence/'UNQUALIFIED-BUILD-RECEIPT.json')
        _blocked_receipt(plan,evidence,stage,error,result)
        parent=os.open(output.parent,os.O_RDONLY)
        try:os.fsync(parent)
        finally:os.close(parent)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    print(json.dumps(build(json.loads(Path(parser.parse_args().plan).read_bytes())),sort_keys=True))
