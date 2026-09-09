"""Prepare immutable checkpoint-specific CPU artifacts. GPU dispatch is separate."""
import argparse
import json
from pathlib import Path
import shutil
from . import protocol
from .suite import digest, verify_suite, SOURCE

REFERENCE = Path(__file__).with_name('harness.json')
SCHEMA = 'yeta.qwen38-training-quality-run/v1'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def replace_flag(argv, flag, value):
    if argv.count(flag) != 1 or argv.index(flag)+1 >= len(argv):
        raise ValueError('Missing or ambiguous serving flag: '+flag)
    argv[argv.index(flag)+1] = str(value)


def prepare(*, suite_dir, output, remote_root, checkpoint, model_catalog, concurrency=4):
    suite_dir=Path(suite_dir).resolve(); suite=verify_suite(suite_dir)
    if checkpoint != protocol.checkpoint_identity(arm=checkpoint['arm'], completed_updates=checkpoint['completed_updates'],
            checkpoint_sha256=checkpoint['checkpoint_manifest_sha256'],
            cumulative_supervised_tokens=checkpoint['cumulative_supervised_tokens'],
            cumulative_input_tokens=checkpoint['cumulative_input_tokens'], training_gpu_hours=checkpoint['training_gpu_hours']):
        raise ValueError('Checkpoint identity changed')
    remote=Path(remote_root)
    if not remote.is_absolute() or '..' in remote.parts or len(remote.parts) < 4:
        raise ValueError('Use a fresh explicit absolute remote evaluation directory')
    if type(concurrency) is not int or not 1 <= concurrency <= 16:
        raise ValueError('Concurrency must fit this finite 16-task suite')
    catalog_bytes=Path(model_catalog).read_bytes();catalog=json.loads(catalog_bytes)
    models=catalog.get('models')
    if not isinstance(models, list) or len(models)!=1:
        raise ValueError('Use the preserved single-model Codex catalog')
    alias=f'Qwen3.8-27B-{checkpoint["arm"]}-u{checkpoint["completed_updates"]:06d}'
    models[0]['slug']=alias;models[0]['display_name']=alias
    out=Path(output).resolve();out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(suite_dir, out/'suite');(out/'runtime').mkdir()
    reference=json.loads(REFERENCE.read_bytes())
    if reference.get('schema')!='yeta.qwen38-training-quality-harness-reference/v1':raise ValueError('Wrong harness reference')
    config=reference['job']
    config.update(job_name='quality-'+checkpoint['arm']+'-u'+str(checkpoint['completed_updates']),
                  jobs_dir=str(remote/'jobs'), n_attempts=1, n_concurrent_trials=concurrency,
                  tasks=[{'path':str(remote/'suite/tasks'/task), 'source':SOURCE} for task in suite['task_ids']])
    config['retry']={'max_retries':0}
    config['agents'][0].update(model_name=alias,n_concurrent=concurrency)
    agent=config['agents'][0]['kwargs']
    assert agent['model_context_window']==protocol.CONTEXT and agent['model_auto_compact_token_limit']==protocol.COMPACTION
    assert agent['reasoning_effort']=='xhigh' and agent['version']=='0.142.5'
    config['environment']['force_build']=True
    config['environment']['extra_docker_compose']=[str(remote/'runtime/host-gateway.compose.yaml')]
    integration=reference['integration']
    integration['artifacts']['root']=str(remote/'gateway-artifacts')
    integration['training']['model_override']=alias
    integration['gateway']['limits'].update(max_inflight_global=concurrency,max_active_routes=2*concurrency)
    services=reference['services']
    services.update(schema='yeta.qwen38-training-quality-services/v1',model_name=alias,model_path=str(remote/'hf'))
    services['harbor_commands']={'quality':['harbor','run','--config',str(remote/'job.json'),'--plugin',
        'dressage.integrations.harbor.plugin:DressageHarborPlugin','--pk','config_path='+str(remote/'runtime/integration.json')]}
    for service in services['services_in_dependency_order']:
        if service['name'].startswith('engine-'):
            replace_flag(service['argv'],'--model-path',remote/'hf')
            replace_flag(service['argv'],'--served-model-name',alias)
        elif service['name']=='proxy':
            replace_flag(service['argv'],'--tokenizer-path',remote/'hf')
            for flag in ('--token-build-model','--tito-model'):
                if service['argv'][service['argv'].index(flag)+1]!='qwen3_6':raise ValueError('Native template override regressed')
    services['prerequisites']=['Hash-verify the actual completed checkpoint export and native tokenizer',
        'Run actual native initial and incremental tool-request parity on this served checkpoint',
        'Verify task base image, node ownership and all service/gateway ports',
        'Use a checkpoint-aware dispatcher; historical step299-only runners are incompatible',
        'Publish all 16 first-attempt outcomes, including failures and infrastructure exceptions']
    write_json(out/'job.json',config);write_json(out/'runtime/integration.json',integration)
    write_json(out/'runtime/services.json',services);write_json(out/'runtime/qwen38-codex-models.json',catalog)
    compose=reference['compose'].replace('$REMOTE_ROOT',str(remote))
    (out/'runtime/host-gateway.compose.yaml').write_text(compose)
    sources=reference['reference_files_sha256']
    run={'schema':SCHEMA,'checkpoint':checkpoint,'model_name':alias,'remote_root':str(remote),
         'suite_sha256':digest(suite_dir/'suite.json'),'suite_source':SOURCE,'task_ids':suite['task_ids'],
         'policy':protocol.policy(),'expected_native_template_sha256':protocol.TEMPLATE_SHA,
         'reference_files_sha256':sources,'harness_reference_sha256':digest(REFERENCE),'original_model_catalog_sha256':digest(model_catalog),
         'dispatch_ready':False,'checkpoint_aware_runner_available':True,
         'dispatch_blockers':['complete hash-bound checkpoint export and actual native parity proof'],
         'runtime_code_sha256':runtime_code_identity(),
         'model_calls':False,'gpu_allocated':False,'training_quality_measured':False,
         'preparer_sha256':digest(__file__),'protocol_sha256':digest(protocol.__file__),
         'files':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/'run.json',run)
    return run


def runtime_code_identity():
    from . import native,run,collect,aggregate,checkpoint
    from evaluation.tb2_checkpoint_xhigh import verify_native_proxy,verify_tokenizer,harbor_explicit_task_metrics
    modules=[protocol,native,run,collect,aggregate,checkpoint,verify_native_proxy,verify_tokenizer,harbor_explicit_task_metrics]
    files={str(Path(module.__file__).resolve()):digest(module.__file__) for module in modules}
    project=Path(__file__).resolve().parents[2]
    files[str(Path(__file__).resolve())]=digest(__file__)
    return {str(Path(path).relative_to(project)):sha for path,sha in files.items()}


def verify_run(path, *, allow_runtime_outputs=False):
    root=Path(path).resolve();run=json.loads((root/'run.json').read_bytes())
    if run.get('schema')!=SCHEMA:raise ValueError('Wrong run schema')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'run.json'}
    extras=actual-set(run['files'])
    allowed={'hf','jobs','gateway-artifacts','launch','analysis'}
    runtime_ok=allow_runtime_outputs and all(Path(name).parts[0] in allowed or name in {'outcomes.json','quality-summary.json'} for name in extras)
    if set(run['files'])-actual or (extras and not runtime_ok):raise ValueError('Run file coverage changed')
    if run.get('runtime_code_sha256')!=runtime_code_identity():raise ValueError('Runtime source identity changed')
    for name,sha in run['files'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or digest(p)!=sha:raise ValueError('Run artifact changed')
    suite=verify_suite(root/'suite')
    if digest(root/'suite/suite.json')!=run['suite_sha256'] or suite['task_ids']!=run['task_ids']:
        raise ValueError('Suite identity changed')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('suite-dir','output','remote-root','checkpoint-json','model-catalog'):parser.add_argument('--'+field,required=True)
    parser.add_argument('--concurrency',type=int,default=4)
    args=parser.parse_args()
    run=prepare(suite_dir=args.suite_dir,output=args.output,remote_root=args.remote_root,
                checkpoint=json.loads(Path(args.checkpoint_json).read_bytes()),model_catalog=args.model_catalog,concurrency=args.concurrency)
    print(json.dumps({'run':str(Path(args.output)/'run.json'),'tasks':len(run['task_ids']),'dispatch_ready':False}))
