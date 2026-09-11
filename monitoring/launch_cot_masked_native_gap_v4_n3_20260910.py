"""Launch one fresh v4 CP8/FSDP2 run from an exact CPU preflight receipt."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import netrc as netrc_lib
import os
from pathlib import Path
import shlex
import stat
import subprocess
import time

from monitoring import attest_masked_native_gap_v4_dataset_runtime_compatibility_20260911 as compatibility

IMAGE='sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee'
IMAGE_REF='nvcr.io/nvidia/nemo-automodel@'+IMAGE
ROOT=Path('/data/sft_baseline_20260908')
BASE=ROOT/'models/Qwen3.8-27B'
BASE_RECEIPT=ROOT/'model-receipts/qwen38-base-r1d4bf0f-full-hash-68dd1925b826-20260910/verification.json'
WANDB_SECRET_ROOT=Path('/dev/shm')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):value.update(block)
    return value.hexdigest()


def _atomic_json(path,payload):
    pending=path.with_name('.'+path.name+'.tmp-'+str(os.getpid()))
    with pending.open('x') as stream:
        stream.write(json.dumps(payload,sort_keys=True,indent=2)+'\n')
        stream.flush();os.fsync(stream.fileno())
    pending.replace(path)


def _verify_code_tree(code,expected_manifest_sha256):
    manifest=code/'code-manifest.json'
    if manifest.is_symlink() or not manifest.is_file() or digest(manifest)!=expected_manifest_sha256:
        raise ValueError('Runtime code manifest changed after CPU preflight')
    raw=json.loads(manifest.read_bytes());records=raw.get('files')
    if raw.get('schema')!='qwen38-native-gap-v4-runtime-code/v1' or not isinstance(records,list) or not records:
        raise ValueError('Wrong runtime code manifest')
    expected={}
    for item in records:
        name=item.get('path') if isinstance(item,dict) else None
        if (not isinstance(name,str) or not name or '/' not in name or '\\' in name
                or name.startswith('/') or any(part in {'','.','..'} for part in name.split('/'))
                or name.casefold() in {value.casefold() for value in expected}
                or set(item)!={'path','bytes','sha256'}):
            raise ValueError('Runtime manifest contains an unsafe or colliding path')
        expected[name]=item
    members=sorted(code.rglob('*'))
    files=[]
    for path in members:
        mode=path.lstat().st_mode
        if path.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError('Runtime tree contains a non-regular member')
        if stat.S_ISREG(mode):
            files.append(path)
    actual={str(path.relative_to(code)) for path in files}
    if actual!=set(expected)|{'code-manifest.json'}:
        raise ValueError('Runtime tree membership changed after CPU preflight')
    for name,item in expected.items():
        path=code/name
        if path.stat().st_size!=item.get('bytes') or digest(path)!=item.get('sha256'):
            raise ValueError('Runtime code member changed after CPU preflight: '+name)


def _verify_minimal_wandb_netrc(path):
    try:
        tokens=shlex.split(Path(path).read_text(),comments=True,posix=True)
    except (OSError,ValueError,UnicodeError) as error:
        raise ValueError('W&B credential is not a valid minimal netrc') from error
    if (len(tokens)!=6 or tokens[:2]!=['machine','api.wandb.ai']
            or tokens[2]!='login' or tokens[4]!='password'
            or not tokens[3] or not tokens[5]):
        raise ValueError('Ephemeral netrc must contain exactly one W&B machine stanza')
    try:
        parsed=netrc_lib.netrc(str(path))
    except (OSError,netrc_lib.NetrcParseError) as error:
        raise ValueError('W&B credential is not a valid minimal netrc') from error
    if set(parsed.hosts)!={'api.wandb.ai'} or parsed.macros:
        raise ValueError('Ephemeral netrc may contain only the W&B API machine')
    login,account,password=parsed.authenticators('api.wandb.ai')
    if not isinstance(login,str) or not login or account not in (None,'') or not isinstance(password,str) or not password:
        raise ValueError('Ephemeral W&B credentials are incomplete')


def _wandb_probe(container,expected):
    """Query W&B from the already-mounted ephemeral credential, returning no secret."""
    script=r'''import json,math,sys,wandb
expected=json.loads(sys.argv[1]);api=wandb.Api(timeout=30);viewer=api.viewer
run=api.run(expected['entity']+'/'+expected['project']+'/'+expected['run_id'])
rows=[]
for row in run.scan_history(page_size=50):
    rows.append(row)
    if len(rows)>=50:break
def number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
metric=None
for row in rows:
    present=[key for key in ('loss','lr','num_label_tokens') if key in row]
    if present:
        if present!=['loss','lr','num_label_tokens'] or not all(number(row[key]) for key in present):break
        step=row.get('_step',row.get('step'))
        if not number(step):break
        metric={'history_step':int(step),'loss':float(row['loss']),'lr':float(row['lr']),
                'num_label_tokens':int(row['num_label_tokens'])};break
print(json.dumps({'viewer':getattr(viewer,'username',None),'entity':run.entity,
 'project':run.project,'run_id':run.id,'name':run.name,'state':run.state,
 'url':run.url,'first_optimizer_metric':metric},sort_keys=True))'''
    result=subprocess.run(['docker','exec',container,'python3','-c',script,
                           json.dumps(expected,sort_keys=True,separators=(',',':'))],
                          text=True,capture_output=True)
    if result.returncode:
        return None
    try:
        record=json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if (record.get('viewer')!='walden-lee' or record.get('entity')!=expected['entity']
            or record.get('project')!=expected['project'] or record.get('run_id')!=expected['run_id']
            or record.get('name')!=expected['name'] or record.get('state')!='running'
            or record.get('url')!='https://wandb.ai/'+expected['entity']+'/'+expected['project']+
                                  '/runs/'+expected['run_id']
            or not record.get('first_optimizer_metric')
            or record['first_optimizer_metric'].get('history_step')!=expected['training_step']
            or record['first_optimizer_metric'].get('num_label_tokens')!=expected['num_label_tokens']):
        return None
    return record


def _checkpoint_probe(container,config_path):
    script=r'''import json,os,sys
from pathlib import Path
from training.qwen38_native_gap_v3 import masked_train
from training.qwen38_no_cot.nemo_checkpoint import resume_identity,validate_checkpoint_directory
from nemo_automodel.components.checkpoint.utils import resolve_restore_from_to_checkpoint_dir
config=json.loads(Path(sys.argv[1]).read_bytes())
checkpoint_dir=Path(config['checkpoint']['checkpoint_dir']);checkpoint=checkpoint_dir/'epoch_0_step_0'
if not checkpoint.is_dir() or (checkpoint/'.incomplete').exists():raise SystemExit(3)
latest=checkpoint_dir/'LATEST'
if not latest.is_symlink() or os.readlink(latest)!='epoch_0_step_0':raise SystemExit(5)
resolved=Path(resolve_restore_from_to_checkpoint_dir(checkpoint_dir,'LATEST'))
if resolved.resolve(strict=True)!=checkpoint.resolve(strict=True):raise SystemExit(6)
identity=resume_identity(config,masked_train.contract_identity(config))
result=validate_checkpoint_directory(checkpoint,identity,world_size=8)
members=list(checkpoint.rglob('*'))
if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in members):raise SystemExit(4)
result.update(schema='qwen38-native-gap-v4-update1-checkpoint-metadata/v1',status='passed',
 latest_target='epoch_0_step_0',latest_resolver_exact=True,
 file_count=sum(path.is_file() for path in members),bytes=sum(path.stat().st_size for path in members if path.is_file()),
 all_checkpoint_bytes_hashed=False)
print(json.dumps(result,sort_keys=True))'''
    result=subprocess.run(['docker','exec',container,'python3','-c',script,str(config_path)],
                          text=True,capture_output=True)
    if result.returncode:
        return None
    try:
        record=json.loads(result.stdout.splitlines()[-1])
    except (IndexError,json.JSONDecodeError):
        return None
    if (record.get('schema')!='qwen38-native-gap-v4-update1-checkpoint-metadata/v1'
            or record.get('status')!='passed' or record.get('next_step')!=1
            or record.get('epoch')!=0 or record.get('loader_batches_yielded')!=8
            or record.get('latest_target')!='epoch_0_step_0'
            or record.get('latest_resolver_exact') is not True
            or record.get('file_count',0)<20 or record.get('bytes',0)<1):
        return None
    return record


def verify_first_update(*,run,container,expected,config_path,timeout_seconds):
    deadline=time.monotonic()+timeout_seconds
    while time.monotonic()<deadline:
        marker=run/'first-update-data-semantics.json'
        if marker.is_file() and not marker.is_symlink():
            payload=json.loads(marker.read_bytes())
            if (payload.get('schema')!='qwen38-native-gap-v4-first-update-data-semantics/v1'
                    or payload.get('status')!='passed'
                    or payload.get('world_size')!=8 or payload.get('cp_size')!=8
                    or payload.get('dp_size')!=1 or payload.get('pre_cp_batches_identical') is not True
                    or payload.get('reported_num_label_tokens')!=payload.get('expected_once_shifted_label_tokens')
                    or payload.get('manifest_sha256')!=expected['manifest_sha256']
                    or payload.get('config_canonical_sha256')!=expected['config_canonical_sha256']):
                raise RuntimeError('First optimizer update failed its live CP8/data semantics receipt')
            online=_wandb_probe(container,{**expected,
                'training_step':payload.get('training_step'),
                'num_label_tokens':payload.get('reported_num_label_tokens')})
            checkpoint=(_checkpoint_probe(container,config_path)
                        if (run/'checkpoints/epoch_0_step_0').is_dir() else None)
            if online and checkpoint:
                state=subprocess.run(['docker','inspect','--format','{{.State.Running}}',container],
                                     text=True,capture_output=True)
                if state.returncode or state.stdout.strip()!='true':
                    raise RuntimeError('Training container stopped during first-update qualification')
                return {'first_update_sha256':digest(marker),'wandb_node_probe':online,
                        'update1_checkpoint':checkpoint,
                        'container_running_at_qualification':True}
        state=subprocess.run(['docker','inspect','--format','{{.State.Running}}',container],
                             text=True,capture_output=True)
        if state.returncode or state.stdout.strip()!='true':
            raise RuntimeError('Training container exited before the first qualified optimizer metric')
        time.sleep(20)
    raise TimeoutError('First optimizer update and online W&B metric were not both observed in time')


def command_for(plan):
    run,code,dataset,config_dir=(Path(plan[key]) for key in ('run','code','dataset','config_dir'))
    triton=ROOT/'triton-cache'/run.name
    name='qwen38-'+run.name.lower()
    command=['docker','run','-d','--name',name,'--gpus','all','--ipc','host',
        '--ulimit','memlock=-1','--ulimit','stack=67108864','--cpus','64','--memory','1400g',
        '--mount','type=bind,src='+str(code)+',dst=/workspace/yeto,readonly',
        '--mount','type=bind,src='+str(dataset)+',dst='+str(dataset)+',readonly',
        '--mount','type=bind,src='+str(BASE)+',dst='+str(BASE)+',readonly',
        '--mount','type=bind,src='+str(BASE_RECEIPT)+',dst='+str(BASE_RECEIPT)+',readonly',
        '--mount','type=bind,src='+str(config_dir)+',dst='+str(config_dir)+',readonly',
        '--mount','type=bind,src='+str(run)+',dst='+str(run),
        '--mount','type=bind,src='+str(triton)+',dst=/root/.triton/cache',
        '-w','/workspace/yeto','-e','PYTHONPATH=/workspace/yeto',
        '--mount','type=bind,src='+str(plan['wandb_netrc'])+',dst=/root/.netrc,readonly',
        '-e','YETA_TRAINING_IMAGE='+IMAGE,'-e','WANDB_MODE=online',
        '-e','WANDB_RUN_ID='+plan['wandb_run_id'],'-e','WANDB_RESUME=never',
        '-e','WANDB_DISABLE_CODE=true','-e','WANDB_LOG_MODEL=false',
        '-e','TOKENIZERS_PARALLELISM=false','-e','OMP_NUM_THREADS=4',
        '-e','PYTORCH_ALLOC_CONF=expandable_segments:True','--entrypoint','torchrun',IMAGE_REF,
        '--standalone','--nnodes=1','--nproc-per-node=8','-m',
        'training.qwen38_native_gap_v3.masked_train','--config',str(config_dir/'train.json'),
        '--mode','train','--validate-memory-repair-in-production']
    return name,command


def _verify_runtime_compatibility(*,plan,cpu,config,publication,dataset,
                                  build_receipt,code,config_dir):
    build_code=publication.get('code_manifest_sha256')
    runtime_code=cpu.get('code_manifest_sha256')
    contract=cpu.get('runtime_compatibility')
    if build_code==runtime_code:
        if contract is not None or config.get('dataset_contract',{}).get(
                'runtime_compatibility') is not None or plan.get(
                'runtime_compatibility_sha256') is not None or plan.get(
                'cp_guard_probe_path') is not None or plan.get(
                'cp_guard_probe_sha256') is not None:
            raise ValueError('Compatibility bridge supplied for identical code')
        return None
    path=config_dir/'dataset-runtime-compatibility.json'
    if (not isinstance(contract,dict)
            or config.get('dataset_contract',{}).get('runtime_compatibility')!=contract
            or contract.get('receipt_path')!=str(path)
            or contract.get('receipt_sha256')!=plan.get('runtime_compatibility_sha256')
            or contract.get('cp_guard_probe_path')!=plan.get('cp_guard_probe_path')
            or contract.get('cp_guard_probe_sha256')!=plan.get('cp_guard_probe_sha256')
            or contract.get('build_code_manifest_sha256')!=build_code
            or contract.get('runtime_code_manifest_sha256')!=runtime_code
            or contract.get('dataset_tree_sha256')!=publication.get('tree_sha256')
            or contract.get('dataset_build_receipt_sha256')!=cpu.get('build_receipt_sha256')):
        raise ValueError('CPU/config compatibility binding changed')
    record=compatibility.verify_receipt(receipt_path=path,
        receipt_sha256=contract['receipt_sha256'],dataset=dataset,
        build_receipt=build_receipt,
        build_receipt_sha256=cpu['build_receipt_sha256'],new_code_root=code,
        new_code_manifest_sha256=runtime_code,
        cp_guard_probe=plan.get('cp_guard_probe_path'),
        cp_guard_probe_sha256=plan.get('cp_guard_probe_sha256'))
    if record.get('conversion_dependency_map_sha256')!=contract.get(
            'conversion_dependency_map_sha256'):
        raise ValueError('Compatibility conversion-dependency identity changed')
    return record


def validate(plan):
    if plan.get('schema')!='qwen38-native-gap-v4-launch-plan/v1':
        raise ValueError('Wrong v4 launch plan')
    if plan.get('launcher_sha256')!=digest(__file__):
        raise ValueError('V4 launcher identity changed')
    run,code,dataset,config_dir=(Path(plan[key]) for key in ('run','code','dataset','config_dir'))
    netrc=Path(plan.get('wandb_netrc',''))
    if (run.parent!=ROOT/'runs' or not run.name.startswith('cot-masked-native-gap-v4-')
            or run.exists() or run.is_symlink() or run.parent.resolve(strict=True)!=ROOT/'runs'
            or code.parent!=ROOT/'code' or not code.name.startswith('cot-masked-native-gap-v4-')
            or dataset.parent!=ROOT/'datasets' or not dataset.name.startswith('cot-masked-native-gap-v4-')
            or config_dir.parent!=ROOT/'configs' or config_dir.name!=run.name
            or any(path.is_symlink() or path.resolve(strict=True)!=path
                   for path in (code,dataset,config_dir))):
        raise ValueError('Launch paths are outside fresh v4 namespaces')
    netrc_stat=netrc.lstat()
    if (not netrc.is_absolute() or netrc.parent!=WANDB_SECRET_ROOT or netrc.is_symlink()
            or not stat.S_ISREG(netrc_stat.st_mode) or netrc_stat.st_uid!=os.geteuid()
            or netrc_stat.st_mode&0o077 or netrc_stat.st_size<1):
        raise ValueError('W&B credential must be an ephemeral owner-only /dev/shm file')
    _verify_minimal_wandb_netrc(netrc)
    run_id=plan.get('wandb_run_id')
    if (not isinstance(run_id,str) or len(run_id)!=8
            or any(value not in '0123456789abcdefghijklmnopqrstuvwxyz' for value in run_id)):
        raise ValueError('W&B run ID must be a fresh eight-character lowercase identity')
    timeout=plan.get('first_metric_timeout_seconds')
    if type(timeout) is not int or not 600<=timeout<=7200:
        raise ValueError('First-metric timeout must be 600..7200 seconds')
    cpu_path=Path(plan['cpu_preflight'])
    if (cpu_path!=config_dir/'cpu-preflight.json' or cpu_path.is_symlink()
            or digest(cpu_path)!=plan.get('cpu_preflight_sha256')):
        raise ValueError('CPU preflight identity changed')
    cpu=json.loads(cpu_path.read_bytes())
    config_path=config_dir/'train.json';config=json.loads(config_path.read_bytes())
    if (cpu.get('schema')!='qwen38-native-gap-v4-cpu-preflight/v1'
            or cpu.get('status')!='passed' or cpu.get('runtime_image')!=IMAGE
            or cpu.get('config_sha256')!=digest(config_path)
            or cpu.get('code_manifest_sha256')!=digest(code/'code-manifest.json')
            or cpu.get('full_export') is not True or cpu.get('all_train') is not True
            or cpu.get('internal_validation') is not False
            or cpu.get('validation_dataset_absent') is not True
            or cpu.get('resolved_validation_dataloader_absent') is not True
            or cpu.get('cuda_initialized') is not False
            or cpu.get('gpu_runtime_qualified') is not False
            or cpu.get('training_started') is not False):
        raise ValueError('CPU preflight does not qualify this exact full train')
    _verify_code_tree(code,cpu['code_manifest_sha256'])
    build_receipt=dataset.with_name(dataset.name+'.build-receipt.json')
    if (cpu.get('build_receipt_path')!=str(build_receipt)
            or digest(build_receipt)!=cpu.get('build_receipt_sha256')):
        raise ValueError('Dataset build-publication receipt changed after CPU preflight')
    publication=json.loads(build_receipt.read_bytes())
    if (publication.get('schema')!='qwen38-native-gap-v4-build-publication/v1'
            or publication.get('status')!='complete' or publication.get('mode')!='final'
            or publication.get('tree_sha256')!=cpu.get('build_tree_sha256')
            or publication.get('output')!=str(dataset)
            or publication.get('code_manifest_sha256')!=cpu.get('build_code_manifest_sha256')):
        raise ValueError('Dataset publication receipt no longer binds this preflight')
    compatibility_record=_verify_runtime_compatibility(plan=plan,cpu=cpu,config=config,
        publication=publication,dataset=dataset,build_receipt=build_receipt,code=code,
        config_dir=config_dir)
    for name,key in (('manifest.json','manifest_sha256'),('index.json','index_sha256'),
                     ('COMPLETE.json','complete_sha256')):
        if digest(dataset/name)!=cpu.get(key):
            raise ValueError('Portable dataset identity changed after preflight')
    if (config.get('training_contract')!='qwen38-xhigh-native-gap-cot-loss-zero/v4'
            or 'validation_dataset' in config or 'validation_dataloader' in config
            or config.get('experiment_phase')!='train'
            or config.get('distributed',{}).get('cp_size')!=8
            or config.get('distributed',{}).get('strategy')!='fsdp2'
            or config.get('optimizer',{}).get('lr')!=1e-5
            or config.get('step_scheduler',{}).get('num_epochs')!=1
            or config.get('step_scheduler',{}).get('max_steps') is not None
            or config.get('checkpoint',{}).get('checkpoint_dir')!=str(run/'checkpoints')
            or config.get('checkpoint',{}).get('restore_from')
            or config.get('wandb',{}).get('project')!='yeto-h200'
            or config.get('wandb',{}).get('entity')!='yeta'):
        raise ValueError('Training recipe differs from the full v4 CP8 contract')
    allowed={'train.json','cpu-preflight.json'}
    if compatibility_record is not None:
        allowed.add('dataset-runtime-compatibility.json')
    if ({path.name for path in config_dir.iterdir()}!=allowed
            or any(path.is_symlink() or not path.is_file() or path.stat().st_mode&0o222
                   for path in config_dir.iterdir())
            or config_dir.stat().st_mode&0o222):
        raise ValueError('Qualified config directory must be exact and read-only')
    triton=ROOT/'triton-cache'/run.name
    if triton.exists() or triton.is_symlink():
        raise ValueError('Fresh launch requires a new separate Triton cache')
    if not BASE.is_dir() or not BASE_RECEIPT.is_file():
        raise ValueError('Pinned base model or full-byte receipt is absent')
    expected_name=config.get('wandb',{}).get('name')
    manifest_sha=cpu.get('manifest_sha256')
    if (not isinstance(manifest_sha,str) or len(manifest_sha)!=64
            or expected_name!='qwen38-27b-generated-cot-masked-native-gap-v4-'+manifest_sha[:12]+'-train'):
        raise ValueError('W&B name must bind the preflight manifest')
    return run,code,dataset,cpu,config,netrc


def launch(plan):
    run,code,dataset,cpu,config,netrc=validate(plan)
    name,command=command_for(plan)
    lock_path=ROOT/'runs/cot-masked-native-gap-v4-launch.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'],text=True).strip()
        if compute:
            raise RuntimeError('GPU compute processes remain')
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total',
            '--format=csv,noheader,nounits'],text=True).strip().splitlines()
        if len(gpus)!=8 or any(int(row.split(',')[1])>=1024 for row in gpus):
            raise RuntimeError('All eight GPUs must be idle')
        if name in subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines():
            raise RuntimeError('Container name already exists')
        if run.exists():
            raise RuntimeError('Fresh launch cannot reuse a run directory')
        run.mkdir(mode=0o700)
        (ROOT/'triton-cache').mkdir(mode=0o700,exist_ok=True)
        triton=ROOT/'triton-cache'/run.name
        triton.mkdir(mode=0o700)
        receipt={**plan,'schema':'qwen38-native-gap-v4-gpu-launch/v1','status':'launching',
            'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'runtime_image':IMAGE,'container_name':name,'argv':command,
            'mount_policy':{'code':'ro','dataset':'ro','base_model':'ro',
                            'base_receipt':'ro','config':'ro','run':'rw','triton_cache':'separate-rw'},
            'prelaunch_gpus':gpus,'first_update_semantics_required':True,
            'wandb_transport':'online-ephemeral-read-only-netrc',
            'wandb_expected':{'viewer':'walden-lee','entity':'yeta','project':'yeto-h200',
                              'run_id':plan['wandb_run_id'],'name':config['wandb']['name'],
                              'manifest_sha256':cpu['manifest_sha256'],
                              'config_canonical_sha256':hashlib.sha256(json.dumps(
                                  config,sort_keys=True,separators=(',',':'),
                                  allow_nan=False).encode()).hexdigest()},
            'base_full_byte_revalidation_before_model_load':True}
        # Never persist the credential pathname in a launch receipt.
        receipt.pop('wandb_netrc',None)
        receipt['argv']=[part.replace(str(netrc),'<ephemeral-wandb-netrc>') for part in command]
        _atomic_json(run/'launch.json',receipt)
        try:
            result=subprocess.run(command,text=True,capture_output=True)
        except Exception as error:
            receipt['status']='failed'
            receipt['launch_error_type']=type(error).__name__
            _atomic_json(run/'launch.json',receipt)
            return 2
        finally:
            # Docker has either established the bind mount or failed.  In both
            # cases remove the only host pathname immediately.
            netrc.unlink(missing_ok=True)
        receipt['docker_exit_code']=result.returncode
        receipt['status']='started_pending_first_metric' if result.returncode==0 else 'failed'
        if result.returncode==0:
            receipt['container']=result.stdout.strip()
            if len(receipt['container'])!=64:
                raise RuntimeError('Docker returned an invalid container identity')
        else:
            receipt['launch_error_tail']=result.stderr[-1500:].replace(str(netrc),'<redacted>')
        _atomic_json(run/'launch.json',receipt)
        if result.returncode:
            return result.returncode
        try:
            qualified=verify_first_update(run=run,container=receipt['container'],
                expected=receipt['wandb_expected'],
                config_path=Path(plan['config_dir'])/'train.json',
                timeout_seconds=plan['first_metric_timeout_seconds'])
        except Exception as error:
            receipt['status']='started_unqualified'
            receipt['qualification_error']=type(error).__name__
            _atomic_json(run/'launch.json',receipt)
            return 2
        receipt.update(status='node_first_update_observed_pending_external_wandb',**qualified)
        _atomic_json(run/'launch.json',receipt)
        return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    raise SystemExit(launch(json.loads(Path(parser.parse_args().plan).read_bytes())))
