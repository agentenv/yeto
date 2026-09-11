"""Create and qualify a fresh all-train v4 recipe inside the pinned CPU image."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import hashlib
import stat

from monitoring import attest_masked_native_gap_v4_dataset_runtime_compatibility_20260911 as compatibility
from training.qwen38_native_gap_v3 import masked_data as data
from training.qwen38_native_gap_v3 import masked_recipe as recipe
from training.qwen38_native_gap_v3 import masked_train as train


def _verify_build_publication(dataset,path,expected_sha256,code_manifest_sha256=None):
    path=Path(path);expected_path=dataset.with_name(dataset.name+'.build-receipt.json')
    if (path!=expected_path or path.is_symlink() or not path.is_file()
            or path.stat().st_mode&0o222 or data.digest_file(path)!=expected_sha256):
        raise ValueError('Final dataset build-publication receipt changed or is not immutable')
    receipt=data._strict_json(path.read_bytes())
    if (receipt.get('schema')!='qwen38-native-gap-v4-build-publication/v1'
            or receipt.get('status')!='complete' or receipt.get('mode')!='final'
            or receipt.get('output')!=str(dataset)
            or receipt.get('runtime_image')!=train.RUNTIME_IMAGE
            or not data._hash(receipt.get('code_manifest_sha256'))
            or (code_manifest_sha256 is not None
                and receipt.get('code_manifest_sha256')!=code_manifest_sha256)
            or receipt.get('immutable_mode')!='files0444_dirs0555'
            or receipt.get('atomic_direct_dataset_publication') is not True):
        raise ValueError('Final dataset build-publication receipt has the wrong contract')
    records=receipt.get('tree_inventory')
    if (not isinstance(records,list) or receipt.get('files')!=len(records)
            or receipt.get('bytes')!=sum(item.get('bytes',-1) for item in records)):
        raise ValueError('Final dataset build inventory has invalid aggregates')
    expected={};folded=set()
    for item in records:
        name=item.get('path') if isinstance(item,dict) else None
        relative=PurePosixPath(name) if isinstance(name,str) else None
        if (not isinstance(item,dict) or set(item)!={'path','bytes','sha256'}
                or relative is None or relative.is_absolute() or '\\' in name
                or any(part in {'','.','..'} for part in relative.parts)
                or name.casefold() in folded or type(item.get('bytes')) is not int
                or item['bytes']<1 or not data._hash(item.get('sha256'))):
            raise ValueError('Final dataset build inventory has an unsafe member')
        expected[name]=item;folded.add(name.casefold())
    members=sorted(dataset.rglob('*'))
    files=[]
    for member in members:
        mode=member.lstat().st_mode
        if (member.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                or mode&0o222):
            raise ValueError('Final dataset tree contains an indirect or special member')
        if stat.S_ISREG(mode):files.append(member)
    if dataset.stat().st_mode&0o222:
        raise ValueError('Final dataset root must be immutable before qualification')
    if {str(member.relative_to(dataset)) for member in files}!=set(expected):
        raise ValueError('Final dataset tree membership differs from build receipt')
    for name,item in expected.items():
        member=dataset/name
        if member.stat().st_size!=item['bytes'] or data.digest_file(member)!=item['sha256']:
            raise ValueError('Final dataset member differs from build receipt: '+name)
    canonical=json.dumps(records,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(canonical).hexdigest()!=receipt.get('tree_sha256'):
        raise ValueError('Final dataset tree aggregate differs from build receipt')
    return receipt


def _verify_code(expected_sha256):
    root=Path(train.__file__).resolve().parents[2]
    manifest=root/'code-manifest.json'
    if (manifest.is_symlink() or not manifest.is_file()
            or data.digest_file(manifest)!=expected_sha256):
        raise ValueError('Exact runtime code manifest changed')
    raw=data._strict_json(manifest.read_bytes())
    if raw.get('schema')!='qwen38-native-gap-v4-runtime-code/v1':
        raise ValueError('Wrong runtime code contract')
    records=raw.get('files')
    if not isinstance(records,list) or not records:
        raise ValueError('Runtime manifest is empty')
    seen=set()
    for item in records:
        name=item.get('path');relative=PurePosixPath(name) if isinstance(name,str) else None
        if (not isinstance(item,dict) or set(item)!={'path','bytes','sha256'}
                or relative is None or relative.is_absolute() or '\\' in name
                or any(part in {'','.','..'} for part in relative.parts)
                or name in seen or name.casefold() in {value.casefold() for value in seen}):
            raise ValueError('Unsafe or colliding runtime member')
        path=root/name
        if (path.is_symlink() or not path.is_file() or path.resolve(strict=True)!=path
                or path.stat().st_size!=item['bytes'] or data.digest_file(path)!=item['sha256']):
            raise ValueError('Runtime member changed: '+name)
        seen.add(name)
    actual={str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    if actual!=seen|{'code-manifest.json'}:
        raise ValueError('Staged runtime tree differs from its exact manifest')
    for module in (data,recipe,train):
        if str(Path(module.__file__).resolve().relative_to(root)) not in seen:
            raise ValueError('Core v4 runtime module is not hash-bound')
    return root


def _runtime_compatibility(*,dataset,build_receipt,build_receipt_sha256,
                           publication,code,code_manifest_sha256,receipt,
                           receipt_sha256,cp_guard_probe,cp_guard_probe_sha256):
    build_code=publication['code_manifest_sha256']
    if build_code==code_manifest_sha256:
        if (receipt is not None or receipt_sha256 is not None
                or cp_guard_probe is not None or cp_guard_probe_sha256 is not None):
            raise ValueError('Do not supply a compatibility bridge for identical code')
        return None
    if (receipt is None or receipt_sha256 is None or cp_guard_probe is None
            or cp_guard_probe_sha256 is None):
        raise ValueError('Dataset build/runtime mismatch requires a compatibility receipt')
    record=compatibility.verify_receipt(receipt_path=receipt,
        receipt_sha256=receipt_sha256,dataset=dataset,build_receipt=build_receipt,
        build_receipt_sha256=build_receipt_sha256,new_code_root=code,
        new_code_manifest_sha256=code_manifest_sha256,
        cp_guard_probe=cp_guard_probe,
        cp_guard_probe_sha256=cp_guard_probe_sha256)
    if (record.get('old_code_manifest_sha256')!=build_code
            or record.get('dataset_tree_sha256')!=publication.get('tree_sha256')):
        raise ValueError('Compatibility receipt does not bind this dataset publication')
    return record


def prepare(*,dataset,run,config_dir,code_manifest_sha256,build_receipt,
            build_receipt_sha256,wandb_project,wandb_entity=None,
            runtime_compatibility=None,runtime_compatibility_sha256=None,
            cp_guard_probe=None,cp_guard_probe_sha256=None):
    code=_verify_code(code_manifest_sha256)
    dataset=Path(dataset);run=Path(run);config_dir=Path(config_dir)
    root=Path('/data/sft_baseline_20260908')
    if (not dataset.is_absolute() or dataset.parent!=root/'datasets'
            or not dataset.name.startswith('cot-masked-native-gap-v4-')
            or dataset.is_symlink() or dataset.resolve(strict=True)!=dataset
            or not run.is_absolute() or run.parent!=root/'runs'
            or not run.name.startswith('cot-masked-native-gap-v4-') or run.exists() or run.is_symlink()
            or not config_dir.is_absolute() or config_dir.parent!=root/'configs'
            or config_dir.name!=run.name or config_dir.exists() or config_dir.is_symlink()):
        raise ValueError('Use fresh canonical v4 dataset/run paths')
    manifest=dataset/'manifest.json';index=dataset/'index.json';complete=dataset/'COMPLETE.json'
    if wandb_project!='yeto-h200' or wandb_entity!='yeta':
        raise ValueError('V4 production metrics require the user-owned yeta/yeto-h200 target')
    publication=_verify_build_publication(dataset,build_receipt,build_receipt_sha256)
    compatibility_record=_runtime_compatibility(dataset=dataset,
        build_receipt=build_receipt,build_receipt_sha256=build_receipt_sha256,
        publication=publication,code=code,code_manifest_sha256=code_manifest_sha256,
        receipt=runtime_compatibility,receipt_sha256=runtime_compatibility_sha256,
        cp_guard_probe=cp_guard_probe,cp_guard_probe_sha256=cp_guard_probe_sha256)
    complete_sha=data.digest_file(complete)
    completion=data.read_complete(complete,complete_sha,manifest_path=manifest,
                                  index_path=index,index_sha256=data.digest_file(index))
    contents=data.read_manifest(manifest)
    if (publication.get('manifest_counts')!=contents.get('counts')
            or publication.get('manifest_token_counts')!=contents.get('token_counts')
            or publication.get('manifest_source_jobs')!=contents.get('source_jobs')
            or publication.get('manifest_frozen_valid_candidate_count')!=
                contents.get('frozen_valid_candidate_count')):
        raise ValueError('Build-publication receipt aggregates differ from final manifest')
    if (contents.get('full_export') is not True or contents.get('all_train') is not True
            or contents.get('internal_validation') is not False
            or set(contents.get('splits',{}))!={'train'}
            or contents.get('unexpected_trace_failures')!=0):
        raise ValueError('CPU qualification requires the full all-train v4 export')
    config=recipe.make_recipe(model_dir=str(train.BASE_MODEL_DIR),manifest=manifest,
        index_path=index,index_sha256=data.digest_file(index),output_dir=run,
        wandb_project=wandb_project,wandb_entity=wandb_entity)
    if 'validation_dataset' in config or 'validation_dataloader' in config:
        raise ValueError('All-train v4 must not construct a validation loader')
    config_dir.mkdir(mode=0o700)
    compatibility_contract=None
    if compatibility_record is not None:
        source=Path(runtime_compatibility)
        compatibility_path=config_dir/'dataset-runtime-compatibility.json'
        raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=runtime_compatibility_sha256:
            raise ValueError('Compatibility receipt changed while copying it')
        with compatibility_path.open('xb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        compatibility_path.chmod(0o444)
        compatibility_contract={
            'schema':compatibility.SCHEMA,
            'receipt_path':str(compatibility_path),
            'receipt_sha256':runtime_compatibility_sha256,
            'dataset_tree_sha256':compatibility_record['dataset_tree_sha256'],
            'dataset_build_receipt_sha256':build_receipt_sha256,
            'build_code_manifest_sha256':publication['code_manifest_sha256'],
            'runtime_code_manifest_sha256':code_manifest_sha256,
            'conversion_dependency_map_sha256':
                compatibility_record['conversion_dependency_map_sha256'],
            'cp_guard_probe_path':compatibility_record['cp_guard_probe_path'],
            'cp_guard_probe_sha256':compatibility_record['cp_guard_probe_sha256'],
            'cp_guard_nested_receipt_sha256':
                compatibility_record['cp_guard_nested_receipt_sha256'],
        }
        config['dataset_contract']['runtime_compatibility']=compatibility_contract
        recipe.require_recipe(config)
    config_path=config_dir/'train.json'
    with config_path.open('x') as stream:
        stream.write(json.dumps(config,sort_keys=True,indent=2,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    output=io.StringIO()
    with contextlib.redirect_stdout(output):
        train.main(['--config',str(config_path),'--mode','preflight'])
    records=[json.loads(line) for line in output.getvalue().splitlines()
             if line.startswith('{')]
    if (not records or records[0].get('schema')!=train.VERSION
            or records[-1].get('status')!='configuration_parsed_gpu_unqualified'
            or records[0].get('dataset_rows',{}).get('train',0)<1
            or set(records[0].get('dataset_rows',{}))!={'train'}):
        raise ValueError('Pinned v4 CPU preflight did not complete exactly')
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes._typed_config import RecipeConfig
    typed=RecipeConfig(ConfigNode(config))
    if typed.vlm_validation_dataloader is not None:
        raise ValueError('All-train recipe resolved an unintended validation loader')
    receipt={**records[0],
        'schema':'qwen38-native-gap-v4-cpu-preflight/v1','status':'passed',
        'runtime_image':os.environ.get('YETA_TRAINING_IMAGE'),
        'code_root':str(code),'code_manifest_sha256':code_manifest_sha256,
        'build_code_manifest_sha256':publication['code_manifest_sha256'],
        'build_receipt_path':str(Path(build_receipt)),
        'build_receipt_sha256':build_receipt_sha256,
        'build_tree_sha256':publication['tree_sha256'],
        'runtime_compatibility':compatibility_contract,
        'config_path':str(config_path),'config_sha256':data.digest_file(config_path),
        'manifest_sha256':data.digest_file(manifest),
        'index_sha256':data.digest_file(index),'complete_sha256':complete_sha,
        'complete_contract':completion,'full_export':True,'all_train':True,
        'internal_validation':False,'validation_dataset_absent':True,
        'resolved_validation_dataloader_absent':True,'cuda_initialized':False,
        'gpu_runtime_qualified':False,'training_started':False}
    receipt_path=config_dir/'cpu-preflight.json'
    with receipt_path.open('x') as stream:
        stream.write(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    config_path.chmod(0o444);receipt_path.chmod(0o444);config_dir.chmod(0o555)
    return {**receipt,'cpu_preflight_path':str(receipt_path),
            'cpu_preflight_sha256':data.digest_file(receipt_path)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True);parser.add_argument('--run',required=True)
    parser.add_argument('--config-dir',required=True)
    parser.add_argument('--code-manifest-sha256',required=True)
    parser.add_argument('--build-receipt',required=True)
    parser.add_argument('--build-receipt-sha256',required=True)
    parser.add_argument('--runtime-compatibility')
    parser.add_argument('--runtime-compatibility-sha256')
    parser.add_argument('--cp-guard-probe')
    parser.add_argument('--cp-guard-probe-sha256')
    parser.add_argument('--wandb-project',default='yeto-h200');parser.add_argument('--wandb-entity',default='yeta')
    print(json.dumps(prepare(**vars(parser.parse_args())),sort_keys=True))
