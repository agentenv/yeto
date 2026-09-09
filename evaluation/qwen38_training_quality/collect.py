"""Collect all original finite quality outcomes and compare saved native inputs."""
import json
from pathlib import Path
from . import native,protocol
from .aggregate import aggregate,INVENTORY_SCHEMA
from .prepare import verify_run,write_json
from .suite import digest,SOURCE


def reference(path):
    path=Path(path)
    return {'path':str(path),'sha256':digest(path)}


def collect(root):
    root=Path(root).resolve(strict=True);run=verify_run(root,allow_runtime_outputs=True)
    if (root/'outcomes.json').exists() or (root/'quality-summary.json').exists():
        raise FileExistsError('Preserve existing outcome inventory and summary')
    proof_path=root/'launch/native-live-proof.json';proof=json.loads(proof_path.read_bytes());native.validate(proof,run)
    if (proof.get('export_receipt_sha256')!=digest(root/'hf/export-receipt.json')
        or proof.get('token_record_sha256')!=digest(root/'launch/native-qualification-tokens.json')):
        raise ValueError('Checkpoint-native evidence changed')
    _,tito=native.shared.runtime(root/'hf')
    bundles={}
    for manifest in sorted((root/'gateway-artifacts').glob('*/*/manifest.json')):
        contents=json.loads(manifest.read_bytes())
        for item in contents.get('final_trials',[])+contents.get('attempts',[]):
            path=Path(item['bundle'])
            if not path.is_absolute():path=manifest.parent/path
            path=path.resolve(strict=True)
            if not path.is_relative_to(root/'gateway-artifacts'):raise ValueError('Bundle escapes this owned job')
            bundles.setdefault(item['trial_name'],set()).add(path)
    trials={}
    for result in sorted((root/'jobs').glob('*/*/result.json')):
        value=json.loads(result.read_bytes());task=value.get('config',{}).get('task',{})
        if not task:continue
        task_id=Path(task.get('path','')).name
        if task.get('source')!=SOURCE or task_id not in run['task_ids']:raise ValueError('Unexpected task in this finite job')
        if task_id in trials:raise ValueError('Multiple attempts for one task; never choose a best outcome')
        trials[task_id]=(result,value)
    analysis=root/'analysis';analysis.mkdir(exist_ok=True)
    rows=[]
    for task_id in run['task_ids']:
        if task_id not in trials:continue
        result,value=trials[task_id];row={'task_id':task_id,'result':reference(result)}
        trajectory=result.parent/'agent/trajectory.json'
        if trajectory.exists():row['trajectory']=reference(trajectory)
        found=bundles.get(result.parent.name,set())
        if len(found)>1:raise ValueError('Ambiguous bundle identity for original trial')
        if found:
            path=next(iter(found));row['bundle']=reference(path)
            bundle=json.loads(path.read_bytes());segments=bundle.get('segments') or []
            parity={'schema':'yeta.qwen38-quality-native-parity/v1',
                'checkpoint_manifest_sha256':run['checkpoint']['checkpoint_manifest_sha256'],
                'result_sha256':row['result']['sha256'],'bundle_sha256':row['bundle']['sha256'],
                'native_template_sha256':protocol.TEMPLATE_SHA,
                'checkpoint_native_proof_sha256':digest(proof_path),
                'actual_initial_native_xhigh_parity':False,
                'actual_incremental_tool_result_parity':proof['actual_incremental_tool_result_parity'],
                'incremental_evidence_scope':'checkpoint qualification with explicitly supplied synthetic tool history',
                'collector_sha256':digest(__file__)}
            if segments:
                segment=segments[0];messages=segment.get('messages',[])
                # Every task launches a fresh Codex session. Its first assistant in
                # this saved lineage is the output following the original task prompt.
                first=next((i for i,message in enumerate(messages) if message.get('role')=='assistant'),None)
                tokens=segment.get('tokens',[]);mask=segment.get('full_loss_mask',[])
                output=next((i for i,value in enumerate(mask) if value),None)
                if first is not None and output is not None and len(tokens)==len(mask):
                    expected=tito._encode_text(tito._render_messages(messages[:first],add_generation_prompt=True,tools=segment.get('tools')))
                    parity['actual_initial_native_xhigh_parity']=tokens[:output]==expected
                    parity['actual_initial_tokens']=output
                    parity['expected_initial_tokens']=len(expected)
                    parity['actual_initial_token_sha256']=native.shared.digest(tokens[:output])
                    parity['expected_initial_token_sha256']=native.shared.digest(expected)
            destination=analysis/(task_id+'-native-parity.json')
            with destination.open('x') as stream:json.dump(parity,stream,indent=2,sort_keys=True);stream.write('\n')
            row['native_parity']=reference(destination)
        rows.append(row)
    inventory={'schema':INVENTORY_SCHEMA,'run_sha256':digest(root/'run.json'),
        'first_attempts_only':True,'best_of_selection':False,'outcomes':rows,
        'collector_sha256':digest(__file__),'missing_tasks':[task for task in run['task_ids'] if task not in trials]}
    with (root/'outcomes.json').open('x') as stream:json.dump(inventory,stream,indent=2,sort_keys=True);stream.write('\n')
    return aggregate(root,root/'outcomes.json',root/'quality-summary.json')
