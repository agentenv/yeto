"""Qualify the complete corrected masked corpus in the pinned CPU-only image."""
import argparse
from collections import Counter
import contextlib
import io
import json
import os
from pathlib import Path

from training.qwen38_native_gap_v3 import masked_data as data, masked_recipe as recipe, masked_train as train


def verify_code(expected):
    code=Path(train.__file__).resolve().parents[2]
    manifest=code/'code-manifest.json'
    assert data.digest_file(manifest)==expected
    entries=json.loads(manifest.read_bytes())['files'];seen=set()
    assert entries
    for item in entries:
        relative=Path(item['path']);path=code/relative
        assert not relative.is_absolute() and '..' not in relative.parts and item['path'] not in seen
        seen.add(item['path'])
        assert path.is_file() and path.resolve(strict=True)==path
        assert path.stat().st_size==item['bytes'] and data.digest_file(path)==item['sha256']
    for module in (data,recipe,train):
        assert str(Path(module.__file__).resolve().relative_to(code)) in seen
    return code


def prepare(dataset,run,code_manifest_sha256):
    code=verify_code(code_manifest_sha256)
    dataset,run=Path(dataset).resolve(strict=True),Path(run)
    assert run.resolve()==run
    assert dataset.parent==Path('/data/sft_baseline_20260908/datasets') and dataset.name.startswith('cot-masked-native-gap-v3-')
    assert run.parent==Path('/data/sft_baseline_20260908/runs') and run.name.startswith('cot-masked-native-gap-v3-')
    transfer=json.loads((dataset/'transfer-complete.json').read_bytes())
    source=json.loads((dataset/'source-manifest.json').read_bytes())
    complete=json.loads((dataset/'source-COMPLETE.json').read_bytes())
    contents=data.read_manifest(dataset/'manifest.json')
    assert transfer['complete'] is True
    assert data.digest_file(dataset/'manifest.json')==transfer['manifest_sha256']
    assert data.digest_file(dataset/'source-manifest.json')==transfer['source_manifest_sha256']==complete['manifest_sha256']
    assert data.digest_file(dataset/'source-COMPLETE.json')==transfer['source_complete_sha256']
    assert data.digest_file(dataset/'exclusions.jsonl')==transfer['exclusions_sha256']
    relocated=json.loads(json.dumps(source))
    for entries in relocated['splits'].values():
        for entry in entries:entry['path']=Path(entry['path']).name
    assert relocated==contents
    assert contents['full_export'] and contents['all_frozen_generations_accounted_for']
    assert contents['frozen_valid_candidate_count']==31504 and contents['unexpected_trace_failures']==0
    assert contents['gap_omission_policy']=='keep-native-leading-gap-omit-only-conflicting-cot/v3'
    assert contents['included_valid_candidate_count']+contents['omitted_valid_candidate_count']+contents['excluded_valid_candidate_count']==31504
    assert contents['all_train'] is False and contents['internal_validation'] is True
    assert contents['group_split']=='identical-baseline-sha256-seed-group-mod100'
    assert contents['unknown_session_policy']=='capture-identity-train-only-no-holdout-claim/v2'
    selection=contents['baseline_replay_selection']
    assert selection['sha256']=='cb1c4adca210e7f7fe2249cb96af6ee8df08920f9b2c856b308750c31b32a686'
    assert selection['selected_replay_sources']==2346 and selection['input_replay_sources']==2457
    excluded=[json.loads(line) for line in (dataset/'exclusions.jsonl').read_text().splitlines() if line.strip()]
    assert len({(x['source'],x['identity']) for x in excluded})==len(excluded)
    trace_excluded=[x for x in excluded if x['source']=='trace']
    assert all(x.get('turn_boundary_exclusion') or x.get('native_source_exclusion') or x.get('duplicate_source_exclusion') for x in trace_excluded)
    assert sum(x['generated_gaps'] for x in trace_excluded)==contents['excluded_valid_candidate_count']
    assert sum(value for key,value in contents['counts'].items() if key.endswith('/trace'))+len(trace_excluded)==14146
    assert sum(contents['counts'].values())+len(excluded)==contents['source_jobs']==14146+2346
    assert dict(Counter(x['source']+'/'+x['reason'] for x in excluded))==contents['exclusions']
    index=data.build_index(dataset/'manifest.json',dataset/'index.json',workers=32)
    assert index['unreviewed_gap_count']==contents['unreviewed_gap_count']
    assert index['omitted_valid_candidate_count']==contents['omitted_valid_candidate_count']
    assert index['rows'].get('validation',0)>0
    config=recipe.make_recipe(model_dir=train.BASE_MODEL_DIR,manifest=dataset/'manifest.json',
        index_path=dataset/'index.json',index_sha256=index['sha256'],output_dir=run,wandb_entity='yeta',wandb_project='yeto-h200')
    run.mkdir(mode=0o700,exist_ok=False)
    config_path=run/'train.json'
    with config_path.open('x') as stream:stream.write(json.dumps(config,sort_keys=True,indent=2)+'\n')
    output=io.StringIO()
    with contextlib.redirect_stdout(output):train.main(['--config',str(config_path),'--mode','preflight'])
    lines=[json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
    assert lines[0]['schema']==train.VERSION and lines[-1]['status']=='configuration_parsed_gpu_unqualified'
    import ast,hashlib,inspect,textwrap
    import torch
    from nemo_automodel.components.datasets.vlm.loader import VlmCollatorConfig
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes._typed_config import RecipeConfig
    typed=RecipeConfig(ConfigNode(config))
    for name in ('checkpoint','step_scheduler','lr_scheduler','optimizer','loss_fn','wandb','dataloader'):
        getattr(typed,name)
    loss_path=Path(inspect.getfile(FusedLinearCrossEntropy))
    loss_sha=hashlib.sha256(loss_path.read_bytes()).hexdigest()
    assert loss_sha=='fd4754cb1eb4f28373a75fff9efef0524768bf3a779659f8aa2c3df3b392cbad'
    forward=ast.parse(textwrap.dedent(inspect.getsource(FusedLinearCrossEntropy.forward)))
    calls=[n for n in ast.walk(forward) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='linear_cross_entropy']
    assert len(calls)==1
    shifts=[k.value for k in calls[0].keywords if k.arg=='shift']
    assert len(shifts)==1 and isinstance(shifts[0],ast.Constant) and shifts[0].value is False
    factory=VlmCollatorConfig(factory=data.collate_exact,kwargs={'pad_token_id':248044,'pad_to_multiple_of':16}).build(processor=object())
    checked=data.GeneratedMaskedTokenDataset(dataset/'manifest.json',index_path=dataset/'index.json',index_sha256=index['sha256'])
    probes=[]
    for position in sorted({0,max(range(len(checked.refs)),key=lambda i:checked.refs[i][4])}):
        row=checked[position];batch=factory([row]);n=len(row['input_ids'])
        assert batch['input_ids'][0,:n].tolist()==row['input_ids'] and batch['labels'][0,:n].tolist()==row['labels']
        assert batch['labels'][0,n:].eq(-100).all().item() and batch['attention_mask'][0,n:].eq(0).all().item()
        assert batch['input_ids'].shape[-1]<=262144
        assert int(batch['input_ids'].shape[0])==1 and row['labels'][-1]==-100
        probes.append({'row':position,'input_tokens':len(row['input_ids']),'padded_tokens':int(batch['input_ids'].shape[-1])})
    assert not torch.cuda.is_initialized()
    record={**lines[0],'schema':'qwen38-native-gap-masked-cpu-preflight/v3','status':'passed',
        'code_manifest_sha256':code_manifest_sha256,'cuda_initialized':False,
        'loss_internal_shift':False,'loss_source_sha256':loss_sha,'causal_shift':'dataset-once-before-CP',
        'config_sha256':data.digest_file(config_path),'manifest_sha256':data.digest_file(dataset/'manifest.json'),
        'index_sha256':index['sha256'],'source_manifest_sha256':transfer['source_manifest_sha256'],
        'source_complete_sha256':transfer['source_complete_sha256'],'runtime_image':os.environ['YETA_TRAINING_IMAGE'],
        'frozen_valid_candidate_count':31504,'included_valid_candidate_count':contents['included_valid_candidate_count'],
        'omitted_valid_candidate_count':contents['omitted_valid_candidate_count'],
        'gap_omission_policy':contents['gap_omission_policy'],
        'excluded_valid_candidate_count':contents['excluded_valid_candidate_count'],'unreviewed_gap_count':contents['unreviewed_gap_count'],
        'full_native_mask_validation':True,'original_gap_positions_preserved':True,'reasoning_relocated':False,
        'known_session_split_verified':True,'unknown_sessions_train_only':True,'session_grouping_complete':contents['session_grouping_complete'],
        'validation_scope':'first32-known-session-heldout-rows-with-at-most32768-tokens',
        'loader_collator_probes':probes,'gpu_runtime_qualified':False,'training_started':False}
    with (dataset/'cpu-preflight.json').open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--run',required=True)
    p.add_argument('--code-manifest-sha256',required=True)
    prepare(**vars(p.parse_args()))
