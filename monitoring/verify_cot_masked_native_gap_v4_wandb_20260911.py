"""Independently qualify update 1 through Walden's W&B account.

This controller runs outside the training container.  It stages a one-machine
credential in n3 tmpfs, queries W&B from a separate no-GPU pinned-image
container, removes the tmpfs pathname, and publishes only sanitized evidence.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess

from monitoring import stage_wandb_auth_v4_20260911 as auth
from monitoring.launch_cot_masked_native_gap_v4_n3_20260910 import IMAGE_REF

ROOT='/data/sft_baseline_20260908/runs/'
RUN_NAME=re.compile(r'cot-masked-native-gap-v4-[A-Za-z0-9._-]+')
HELPER_PATH='monitoring/verify_cot_masked_native_gap_v4_wandb_20260911.py'


def digest_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(path):
    return digest_bytes(Path(path).read_bytes())


def _remote(node,argv,*,input=None):
    result=subprocess.run(auth.SSH+[node,shlex.join(argv)],input=input,capture_output=True)
    if result.returncode:
        raise RuntimeError('Independent n3 verification command failed')
    return result.stdout


def _remove_secret(node,secret):
    script=('import os,sys;os.unlink(sys.argv[1]) if os.path.lexists(sys.argv[1]) else None;'
            'raise SystemExit(1 if os.path.lexists(sys.argv[1]) else 0)')
    result=subprocess.run(auth.SSH+[node,shlex.join(['sudo','-n','python3','-c',script,secret])],
                          capture_output=True)
    return result.returncode==0


def _state(node,run):
    script=r'''import hashlib,json,os,sys
from pathlib import Path
run=Path(sys.argv[1]);launch=run/'launch.json';marker=run/'first-update-data-semantics.json'
def sha(raw):return hashlib.sha256(raw).hexdigest()
lr=launch.read_bytes();mr=marker.read_bytes();lp=json.loads(lr);mp=json.loads(mr)
code=Path(lp['code']);cm=json.loads((code/'code-manifest.json').read_bytes())
helpers=[x for x in cm['files'] if x['path']==sys.argv[2]]
if len(helpers)!=1:raise ValueError('external verifier is not in exact runtime closure')
print(json.dumps({'launch':lp,'launch_sha256':sha(lr),'marker':mp,'marker_sha256':sha(mr),
 'helper':helpers[0]},sort_keys=True))'''
    return json.loads(_remote(node,['sudo','-n','python3','-c',script,run,HELPER_PATH]))


QUERY=r'''import json,math,sys,wandb
expected=json.loads(sys.argv[1]);api=wandb.Api(timeout=30);viewer=api.viewer
run=api.run(expected['entity']+'/'+expected['project']+'/'+expected['run_id'])
metric=None
for row in run.scan_history(page_size=50):
 if all(key in row for key in ('loss','lr','num_label_tokens')):
  values=(row['loss'],row['lr'],row['num_label_tokens']);step=row.get('_step',row.get('step'))
  number=lambda value:isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
  if all(number(value) for value in values) and number(step):
   metric={'history_step':int(step),'loss':float(values[0]),'lr':float(values[1]),
           'num_label_tokens':int(values[2])}
  break
print(json.dumps({'viewer':getattr(viewer,'username',None),'entity':run.entity,
 'project':run.project,'run_id':run.id,'name':run.name,'state':run.state,
 'url':run.url,'first_optimizer_metric':metric},sort_keys=True))'''


def _parse_final_json_object(raw):
    """Accept informational lines but require one JSON object as the final line."""
    if isinstance(raw,bytes):
        try:
            value=raw.decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError('W&B query output is not UTF-8') from error
    elif isinstance(raw,str):
        value=raw
    else:
        raise TypeError('W&B query output must be bytes or text')
    lines=[line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        raise ValueError('W&B query emitted no JSON object')
    try:
        record=json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise ValueError('W&B query final line is not a JSON object') from error
    if not isinstance(record,dict):
        raise ValueError('W&B query final line is not a JSON object')
    for line in lines[:-1]:
        try:
            earlier=json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(earlier,dict):
            raise ValueError('W&B query emitted more than one JSON object')
    return record


def _query_command(secret,expected):
    return ['sudo','-n','docker','run','--rm','--network','bridge','--read-only',
        '--cap-drop','ALL','--security-opt','no-new-privileges',
        '--tmpfs','/tmp:rw,noexec,nosuid,size=512m',
        '-e','WANDB_CACHE_DIR=/tmp/wandb-cache',
        '-e','XDG_CACHE_HOME=/tmp/wandb-cache',
        '--mount','type=bind,src='+secret+',dst=/root/.netrc,readonly',IMAGE_REF,
        'python3','-c',QUERY,json.dumps(expected,sort_keys=True,separators=(',',':'))]


def validate(state,query,helper_sha256):
    launch=state.get('launch',{});marker=state.get('marker',{});expected=launch.get('wandb_expected',{})
    metric=query.get('first_optimizer_metric') or {}
    url='https://wandb.ai/yeta/yeto-h200/runs/'+str(expected.get('run_id',''))
    finite=lambda value:isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    checkpoint=launch.get('update1_checkpoint',{})
    if (launch.get('schema')!='qwen38-native-gap-v4-gpu-launch/v1'
            or launch.get('status')!='node_first_update_observed_pending_external_wandb'
            or launch.get('first_update_sha256')!=state.get('marker_sha256')
            or state.get('helper')!={'path':HELPER_PATH,'bytes':state.get('helper',{}).get('bytes'),
                                     'sha256':helper_sha256}
            or marker.get('schema')!='qwen38-native-gap-v4-first-update-data-semantics/v1'
            or marker.get('status')!='passed'
            or marker.get('manifest_sha256')!=expected.get('manifest_sha256')
            or marker.get('config_canonical_sha256')!=expected.get('config_canonical_sha256')
            or marker.get('reported_num_label_tokens')!=marker.get('expected_once_shifted_label_tokens')
            or query.get('viewer')!='walden-lee' or query.get('entity')!='yeta'
            or query.get('project')!='yeto-h200' or query.get('run_id')!=expected.get('run_id')
            or query.get('name')!=expected.get('name') or query.get('state')!='running'
            or query.get('url')!=url or not all(finite(metric.get(key)) for key in ('loss','lr','num_label_tokens'))
            or metric.get('history_step')!=marker.get('training_step')
            or metric.get('num_label_tokens')!=marker.get('reported_num_label_tokens')
            or checkpoint.get('status')!='passed' or checkpoint.get('latest_target')!='epoch_0_step_0'
            or checkpoint.get('latest_resolver_exact') is not True):
        raise ValueError('Independent W&B/checkpoint evidence does not match update 1')
    return expected,metric


def _publish_remote(node,run,raw,pending_launch_sha256,marker_sha256):
    script=r'''import hashlib,json,os,subprocess,sys
from pathlib import Path
run=Path(sys.argv[1]);launch=run/'launch.json';marker=run/'first-update-data-semantics.json'
raw=sys.stdin.buffer.read(1<<20)
if not raw or len(raw)>=1<<20:raise ValueError('bad external receipt size')
sha=lambda value:hashlib.sha256(value).hexdigest()
lr=launch.read_bytes();mr=marker.read_bytes();payload=json.loads(raw);state=json.loads(lr)
if (sha(lr)!=sys.argv[2] or sha(mr)!=sys.argv[3]
 or payload.get('pending_launch_sha256')!=sys.argv[2]
 or payload.get('first_update_sha256')!=sys.argv[3]
 or payload.get('status')!='passed' or payload.get('run')!=str(run)
 or state.get('status')!='node_first_update_observed_pending_external_wandb'):
 raise ValueError('remote qualification inputs changed')
live=subprocess.run(['docker','inspect','--format','{{.State.Running}}',state['container']],
 text=True,capture_output=True)
if live.returncode or live.stdout.strip()!='true':raise ValueError('training container is not running')
external=run/'external-wandb-qualification.json'
if external.exists():raise FileExistsError(external)
tmp=run/('.external-wandb-qualification.tmp-'+str(os.getpid()))
with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
tmp.chmod(0o444);tmp.replace(external)
state['status']='qualified_first_update_external_wandb'
state['external_wandb_qualification_sha256']=sha(raw)
state['external_wandb_qualification_path']=str(external)
encoded=(json.dumps(state,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
tmp=run/('.launch.json.tmp-'+str(os.getpid()))
with tmp.open('xb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
tmp.replace(launch)
fd=os.open(run,os.O_RDONLY)
try:os.fsync(fd)
finally:os.close(fd)
print(json.dumps({'status':state['status'],'receipt_sha256':sha(raw),
 'receipt_path':str(external),'container_running':True},sort_keys=True))'''
    return json.loads(_remote(node,['sudo','-n','python3','-c',script,run,
                                     pending_launch_sha256,marker_sha256],input=raw))


def qualify(*,node,run,local_netrc,receipt):
    if (not run.startswith(ROOT) or not RUN_NAME.fullmatch(Path(run).name)
            or str(Path(run).parent) != ROOT.rstrip('/')):
        raise ValueError('External verification requires the exact v4 n3 run path')
    destination=Path(receipt)
    if destination.exists() or destination.is_symlink():
        raise ValueError('External verification receipt must be fresh')
    helper_sha=digest(__file__);before=_state(node,run)
    staged=auth.stage(local_netrc,node=node);secret=staged.get('path','')
    if not re.fullmatch(r'/dev/shm/yeta-wandb-netrc-[0-9a-f]{32}',secret):
        raise ValueError('W&B stager returned an unsafe tmpfs path')
    expected=before['launch']['wandb_expected']
    try:
        query=_parse_final_json_object(_remote(node,_query_command(secret,expected)))
    finally:
        # Cleanup is repeated even if the one-shot API container fails.
        if not _remove_secret(node,secret):
            raise RuntimeError('Ephemeral W&B credential removal could not be verified')
    after=_state(node,run)
    if before!=after:
        raise RuntimeError('Launch or first-update evidence changed during external verification')
    expected,metric=validate(after,query,helper_sha)
    container=after['launch'].get('container')
    running=_remote(node,['sudo','-n','docker','inspect','--format','{{.State.Running}}',container])
    if running.strip()!=b'true':
        raise RuntimeError('Training stopped before independent qualification')
    payload={'schema':'qwen38-native-gap-v4-external-wandb-qualification/v1','status':'passed',
        'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':node,'run':run,
        'helper_sha256':helper_sha,'runtime_image':IMAGE_REF.removeprefix(
            'nvcr.io/nvidia/nemo-automodel@'),'pending_launch_sha256':after['launch_sha256'],
        'first_update_sha256':after['marker_sha256'],'wandb':query,
        'first_optimizer_metric':metric,'container_running':True,
        'credential_transport':'local-minimal-wandb-machine-to-node-tmpfs-one-shot-no-gpu',
        'credential_path_removed':True,'credential_persisted':False,
        'remote_qualification_path':run+'/external-wandb-qualification.json'}
    raw=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    published=_publish_remote(node,run,raw,after['launch_sha256'],after['marker_sha256'])
    if (published.get('status')!='qualified_first_update_external_wandb'
            or published.get('receipt_sha256')!=digest_bytes(raw)
            or published.get('receipt_path')!=payload['remote_qualification_path']
            or published.get('container_running') is not True):
        raise RuntimeError('n3 did not atomically publish the external qualification')
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_name('.'+destination.name+'.tmp-'+str(os.getpid()))
    with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    pending.chmod(0o444);pending.replace(destination)
    descriptor=os.open(destination.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node',default=auth.NODE);parser.add_argument('--run',required=True)
    parser.add_argument('--local-netrc',default='~/.netrc');parser.add_argument('--receipt',required=True)
    print(json.dumps(qualify(**vars(parser.parse_args())),sort_keys=True))
