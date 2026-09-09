"""Run one finite checkpoint quality suite; only this invocation's services stop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import time
import urllib.request
from evaluation.tb2_checkpoint_xhigh.verify_tokenizer import verify as verify_tokenizer
from evaluation.tb2_checkpoint_xhigh import harbor_explicit_task_metrics
from . import native,protocol
from .prepare import verify_run,write_json
from .checkpoint import digest


def verify_export(model,run,expected_receipt_sha256):
    model=Path(model).resolve(strict=True);receipt_path=model/'export-receipt.json'
    if not protocol.hash_string(expected_receipt_sha256) or digest(receipt_path)!=expected_receipt_sha256:
        raise ValueError('Checkpoint export receipt differs from dispatched identity')
    receipt=json.loads(receipt_path.read_bytes())
    expected={'status':'complete','exact_tensor_readback':True,'speculative_decoding_allowed':False,
        'checkpoint':run['checkpoint'],'checkpoint_manifest_sha256':run['checkpoint']['checkpoint_manifest_sha256'],
        'checkpoint_step_zero_based':run['checkpoint']['zero_based_checkpoint_step'],
        'completed_optimizer_updates':run['checkpoint']['completed_updates']}
    if any(receipt.get(key)!=value for key,value in expected.items()):raise ValueError('Wrong arm, checkpoint, budget or incomplete export')
    if not protocol.hash_string(receipt.get('source_snapshot_manifest_sha256')):
        raise ValueError('Export must bind the actual immutable checkpoint snapshot')
    files=receipt.get('files');assets=receipt.get('assets')
    if not isinstance(files,list) or not files or not isinstance(assets,list) or not assets:
        raise ValueError('Export file inventory missing')
    seen=set()
    for item in files+assets:
        path=Path(item['path'])
        if path.is_absolute() or '..' in path.parts or str(path) in seen:raise ValueError('Duplicate or escaping export file')
        seen.add(str(path));actual=model/path
        if actual.is_symlink() or not actual.resolve(strict=True).is_relative_to(model) or digest(actual)!=item['sha256']:
            raise ValueError('Actual served checkpoint file differs')
        if 'size' in item and actual.stat().st_size!=item['size']:raise ValueError('Actual served checkpoint size differs')
    index=model/'model.safetensors.index.json'
    if digest(index)!=receipt.get('index_sha256'):raise ValueError('Served tensor index changed')
    weight_files=set(json.loads(index.read_bytes())['weight_map'].values())
    if weight_files!={item['path'] for item in files}:
        raise ValueError('Tensor shard inventory differs from actual model index')
    required={'config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'}
    if not required.issubset(seen):raise ValueError('Pinned native tokenizer/config assets missing')
    evidence={'model.safetensors.index.json','export-receipt.json'}
    if 'cpu_converter_receipt_sha256' in receipt:
        original=model/'cpu-converter-receipt.json'
        if original.is_symlink() or digest(original)!=receipt['cpu_converter_receipt_sha256']:
            raise ValueError('Original CPU converter evidence changed')
        evidence.add(original.name)
    actual_files={str(path.relative_to(model)) for path in model.rglob('*') if path.is_file()}
    if actual_files!=seen|evidence:
        raise ValueError('Unexpected or unverified file in actual served model directory')
    return receipt


def require_service_plan(plan,run,root):
    if plan.get('model_name')!=run['model_name'] or Path(plan['model_path'])!=root/'hf' or plan.get('speculative_decoding') is not False:
        raise ValueError('Service plan refers to a different model or speculative mode')
    services=plan['services_in_dependency_order']
    if [item['name'] for item in services]!=['engine-0','engine-1','engine-2','engine-3','router','proxy']:
        raise ValueError('Expected the qualified four TP2 engine stack')
    for index,item in enumerate(services[:4]):
        values={'--model-path':str(root/'hf'),'--served-model-name':run['model_name'],'--tp-size':'2','--context-length':'262144'}
        for flag,value in values.items():
            argv=item['argv']
            if argv.count(flag)!=1 or argv[argv.index(flag)+1]!=value:raise ValueError('Changed engine setting '+flag)
        if item['environment'].get('CUDA_VISIBLE_DEVICES')!=f'{index*2},{index*2+1}':raise ValueError('GPU partition changed')
    proxy=services[-1]['argv']
    for flag,value in {'--token-build-mode':'tito','--token-build-model':'qwen3_6','--tito-model':'qwen3_6',
        '--model-mask-type':'qwen3_5','--model-tool-call-type':'qwen3_5','--context-window':'262144',
        '--max-output-tokens':'32768','--tokenizer-path':str(root/'hf')}.items():
        if proxy.count(flag)!=1 or proxy[proxy.index(flag)+1]!=value:raise ValueError('Changed native proxy '+flag)
    return plan


def assert_idle():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if value.strip():raise RuntimeError('Node has existing GPU work; no automatic stop is allowed')
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(gpus)!=8:raise RuntimeError('Expected one isolated eight-GPU host')
    for port in (35000,35001,35002,35003,35004,35080,35810,35101):
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))


def run_suite(root,*,export_receipt_sha256,startup_timeout=1800):
    root=Path(root).resolve(strict=True);run=verify_run(root,allow_runtime_outputs=True)
    if root!=Path(run['remote_root']):raise ValueError('Prepared paths differ from actual evaluation directory')
    if not 1<=startup_timeout<=3600:raise ValueError('Bounded startup timeout required')
    plan=require_service_plan(json.loads((root/'runtime/services.json').read_bytes()),run,root)
    verify_export(root/'hf',run,export_receipt_sha256)
    # Parse the actual installed Harbor schema before starting GPU services.
    from harbor.models.job.config import JobConfig
    JobConfig.model_validate_json((root/'job.json').read_text())
    assert_idle();output=root/'launch';output.mkdir(exist_ok=False)
    # verify_tokenizer is CPU-only and sets CUDA_VISIBLE_DEVICES internally.
    # Restore the parent environment before starting services with their explicit GPU pairs.
    old_cuda=os.environ.get('CUDA_VISIBLE_DEVICES')
    try:tokenizer=verify_tokenizer(root/'hf',expected_template_sha256=protocol.TEMPLATE_SHA)
    finally:
        if old_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=old_cuda
    write_json(output/'tokenizer-preflight.json',tokenizer)
    environ=os.environ.copy();environ['TB2_PROXY_API_KEY']=secrets.token_urlsafe(32)
    processes=[];logs=[]
    original_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def interrupted(signum,frame):raise InterruptedError('Quality runner interrupted')
    for sig in original_handlers:signal.signal(sig,interrupted)
    def start(name,argv,extra=None):
        log=(output/(name+'.log')).open('xb');logs.append(log)
        process=subprocess.Popen(argv,env={**environ,**(extra or {})},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        processes.append((name,process))
        write_json(output/'owned-processes.json',[{'name':n,'pid':p.pid} for n,p in processes])
        return process
    def ready(service,process):
        until=time.monotonic()+startup_timeout
        while time.monotonic()<until:
            if process.poll() is not None:raise RuntimeError(service['name']+' exited during startup')
            try:
                with urllib.request.urlopen(service['ready_url'],timeout=3) as response:
                    if response.status==200:return
            except Exception:pass
            time.sleep(2)
        raise TimeoutError(service['name']+' readiness timed out')
    try:
        pending=[(service,start(service['name'],service['argv'],service['environment'])) for service in plan['services_in_dependency_order'][:4]]
        for service,process in pending:ready(service,process)
        for service in plan['services_in_dependency_order'][4:]:
            argv=list(service['argv'])
            if 'secret_argument' in service:
                secret=service['secret_argument'];argv.append(secret['flag']+'='+environ[secret['environment']])
            process=start(service['name'],argv,service['environment']);ready(service,process)
        proof,record=native.live(root/'hf',run['model_name'],environ['TB2_PROXY_API_KEY'],run['checkpoint'])
        native.validate(proof,run)
        # Only local synthetic qualification data, retained separately from the scored tasks.
        write_json(output/'native-qualification-tokens.json',record)
        proof['token_record_sha256']=digest(output/'native-qualification-tokens.json')
        proof['export_receipt_sha256']=export_receipt_sha256
        write_json(output/'native-live-proof.json',proof)
        write_json(output/'status.json',{'status':'native_parity_passed_starting_quality','checkpoint':run['checkpoint'],'task_count':16})
        argv=list(plan['harbor_commands']['quality'])
        argv[:1]=['python3',str(Path(harbor_explicit_task_metrics.__file__).resolve())]
        child=start('harbor',argv)
        while child.poll() is None:
            for name,service in processes[:-1]:
                if service.poll() is not None:raise RuntimeError(name+' exited during quality evaluation')
            time.sleep(2)
        write_json(output/'status.json',{'status':'completed' if child.returncode==0 else 'failed',
            'harbor_exit_code':child.returncode,'checkpoint':run['checkpoint'],'task_count':16,
            'runner_sha256':digest(__file__),'native_live_proof_sha256':digest(output/'native-live-proof.json')})
        from .collect import collect
        summary=collect(root)
        write_json(output/'summary-reference.json',{'summary_sha256':digest(root/'quality-summary.json'),
            'outcome_count':summary['outcome_count'],'counts':summary['counts'],'complete_coverage':summary['complete_coverage']})
        if child.returncode:raise RuntimeError('Harbor failed; original outcomes and partial aggregate are preserved')
    except BaseException as error:
        write_json(output/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        for name,process in reversed(processes):
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for name,process in reversed(processes):
            if process.poll() is None:
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
        for log in logs:log.close()
        for sig,handler in original_handlers.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--export-receipt-sha256',required=True)
    parser.add_argument('--startup-timeout',type=float,default=1800)
    args=parser.parse_args();run_suite(args.root,export_receipt_sha256=args.export_receipt_sha256,startup_timeout=args.startup_timeout)
