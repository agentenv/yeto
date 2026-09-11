"""Overlap closed-shard transfer with CPU preparation using existing encrypted SSH.

Only builder-published immutable shards are transferred. The final manifest is
published after complete exact-byte coverage; n3 builds its own absolute index.
"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','StrictHostKeyChecking=yes',
     '-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3',
     '-o','UserKnownHostsFile=/private/tmp/yeta-labeling-known-hosts']
MANAGER='c@65.19.161.135'
NODE='ubuntu@100.66.92.111'


def remote_json(host, script, *, sudo=False):
    result=subprocess.run(SSH+[host,('sudo -n ' if sudo else '')+'python3 -'],
                          input=script.encode(),capture_output=True)
    if result.returncode:
        raise RuntimeError('Remote operation failed: '+result.stderr.decode(errors='replace')[-1500:])
    return json.loads(result.stdout)


def inventory(source, qualified=False):
    return remote_json(MANAGER, '''import json
from pathlib import Path
p=Path(%r);qualified=%r
feed=p/'finalized-shards.jsonl'
raw=feed.read_bytes() if feed.exists() else b''
rows=[json.loads(line) for line in raw.splitlines(keepends=True) if line.endswith(b'\\n') and line.strip()]
complete=p/('COMPLETE-qualified.json' if qualified else 'COMPLETE.json'); blocked=p/'BLOCKED.json'
out={'shards':rows,'blocked':blocked.exists(),'complete':complete.exists()}
if complete.exists():
 import hashlib,base64
 m=p/('manifest-qualified.json' if qualified else 'manifest.json');out['manifest']=json.loads(m.read_bytes());out['manifest_sha256']=hashlib.sha256(m.read_bytes()).hexdigest()
 e=p/'exclusions.jsonl';out['exclusions_raw']=e.read_text()
 if not qualified:
  c=json.loads(complete.read_bytes())
  if c['manifest_sha256']!=out['manifest_sha256']:raise ValueError('Complete manifest changed')
  out['source_files']={'source-COMPLETE.json':base64.b64encode(complete.read_bytes()).decode(),'source-manifest.json':base64.b64encode(m.read_bytes()).decode()}
 if qualified:
  c=json.loads(complete.read_bytes())
  if c['manifest_sha256']!=out['manifest_sha256'] or c.get('every_row_validated_with_existing_validator') is not True:raise ValueError('Unqualified manifest')
  q=Path(c['qualification_receipt'])
  if q.parent!=p.parent or q.name!='incomplete-terminal-source-exclusion-policy.json':raise ValueError('Unexpected qualification path')
  if hashlib.sha256(q.read_bytes()).hexdigest()!=c['qualification_receipt_sha256']:raise ValueError('Qualification changed')
  out['source_files']={'source-COMPLETE-qualified.json':base64.b64encode(complete.read_bytes()).decode(),'source-qualification.json':base64.b64encode(q.read_bytes()).decode()}
  out['qualified']=True
print(json.dumps(out))
''' % (source,qualified))


def recover(shards, destination):
    """Rehash final destination files, including completed but unjournaled receivers."""
    return remote_json(NODE, '''import hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
p=Path(%r);items=%r
def check(item):
 target=p/Path(item['path']).name
 if not target.exists():return None
 if target.is_symlink() or not target.is_file():raise ValueError('Unsafe existing shard')
 h=hashlib.sha256();size=0
 with target.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block);size+=len(block)
 if h.hexdigest()!=item['sha256'] or size!=item['bytes']:raise ValueError('Existing shard differs from source')
 return item['path'],{'path':str(target),'source':item['path'],'sha256':h.hexdigest(),'bytes':size,'recovered':True}
with ThreadPoolExecutor(max_workers=4) as pool:out=dict(x for x in pool.map(check,items) if x)
print(json.dumps(out))
''' % (destination,shards),sudo=True)


def transfer(item, source, destination):
    src=Path(item['path'])
    if src.parent != Path(source) or src.name not in {item['split']+'-'+str(i).zfill(5)+'.jsonl' for i in range(1000)}:
        raise ValueError('Unexpected immutable shard path')
    target=destination+'/'+src.name
    script='''import gzip,hashlib,json,sys,uuid,os
from pathlib import Path
target=Path(%r);expected=%r;size_expected=%r
target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
if target.exists():raise ValueError('Destination already exists; inspect transfer receipt before retry')
temp=target.with_name(target.name+'.incoming-'+expected[:12]+'-'+uuid.uuid4().hex)
h=hashlib.sha256();size=0
with gzip.GzipFile(fileobj=sys.stdin.buffer,mode='rb') as src,temp.open('xb') as dst:
 while True:
  block=src.read(8*1024**2)
  if not block:break
  dst.write(block);h.update(block);size+=len(block)
 dst.flush();os.fsync(dst.fileno())
if size!=size_expected or h.hexdigest()!=expected:raise ValueError('Transfer size/hash mismatch')
temp.chmod(0o400);temp.rename(target)
print(json.dumps({'path':str(target),'sha256':h.hexdigest(),'bytes':size}))
''' % (target,item['sha256'],item['bytes'])
    started=time.monotonic()
    producer=subprocess.Popen(SSH+[MANAGER,'gzip -1 -c -- '+shlex.quote(str(src))],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    consumer=subprocess.Popen(SSH+[NODE,'sudo -n python3 -c '+shlex.quote(script)],stdin=producer.stdout,
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    producer.stdout.close()
    output,error=consumer.communicate();_,source_error=producer.communicate()
    if producer.returncode or consumer.returncode:
        raise RuntimeError('Transfer failed: '+(error+source_error).decode(errors='replace')[-1000:])
    result=json.loads(output);result.update(seconds=time.monotonic()-started,source=str(src))
    return result


def publish(state, destination, completed):
    manifest=copy.deepcopy(state['manifest'])
    wanted={entry['path']:entry for entries in manifest['splits'].values() for entry in entries}
    if set(wanted)!=set(completed):raise ValueError('Complete manifest and delivered shards differ')
    for path,entry in wanted.items():
        if entry['sha256']!=completed[path]['sha256']:raise ValueError('Final shard digest changed')
        entry['path']=Path(path).name
    text=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    excluded=state['exclusions_raw']
    receipt={'schema':'qwen38-full-generated-cot-transfer/v1','complete':True,
             'source_manifest_sha256':state['manifest_sha256'],
             'manifest_sha256':hashlib.sha256(text.encode()).hexdigest(),
             'exclusions_sha256':hashlib.sha256(excluded.encode()).hexdigest(),
             'shards':len(completed),'bytes':sum(x['bytes'] for x in completed.values()),
             'transport':'two-existing-encrypted-ssh-streams','absolute_index_rebuild_required':True}
    source_files=state.get('source_files',{})
    if 'source-COMPLETE.json' in source_files:
        receipt['source_complete_sha256']=hashlib.sha256(base64.b64decode(source_files['source-COMPLETE.json'])).hexdigest()
    if 'source-COMPLETE-qualified.json' in source_files:
        receipt['source_complete_sha256']=hashlib.sha256(base64.b64decode(source_files['source-COMPLETE-qualified.json'])).hexdigest()
        receipt['source_qualification_sha256']=hashlib.sha256(base64.b64decode(source_files['source-qualification.json'])).hexdigest()
    return remote_json(NODE, '''import json,base64
from pathlib import Path
p=Path(%r)
def exact(name,raw):
 target=p/name
 if target.exists():
  if target.is_symlink() or target.read_bytes()!=raw:raise ValueError('Existing metadata differs: '+name)
 else:
  with target.open('xb') as f:f.write(raw)
exact('manifest.json',%r.encode())
exact('exclusions.jsonl',%r.encode())
for name,data in %r.items():
 exact(name,base64.b64decode(data))
receipt=%r
exact('transfer-complete.json',(json.dumps(receipt,indent=2)+'\\n').encode())
print(json.dumps(receipt))
''' % (destination,text,excluded,source_files,receipt),sudo=True)


def main():
    raise RuntimeError('Retired manifest-rewriting v3 transfer; use transfer_cot_masked_native_gap_v4_20260910.py')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--destination',required=True)
    parser.add_argument('--state',required=True,type=Path);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--qualified',action='store_true')
    args=parser.parse_args()
    if not args.source.startswith('/mnt/lvm_data/sft_analysis/sft_baseline_20260908/cot-native-gap-v3-'):
        raise ValueError('Only the new frozen experiment may be read')
    if not args.destination.startswith('/data/sft_baseline_20260908/datasets/cot-masked-native-gap-v3-'):
        raise ValueError('Only a new n3 experiment dataset may be written')
    if not 1<=args.workers<=4:raise ValueError('Bounded transfer concurrency required')
    if args.state.exists() and not args.resume:raise ValueError('A transfer state exists; inspect before restarting')
    completed={};pending={};started=time.monotonic()
    if args.resume:
        state=inventory(args.source,args.qualified)
        completed=recover(state['shards'],args.destination)
        print(json.dumps({'recovered_shards':len(completed),'bytes':sum(x['bytes'] for x in completed.values())}),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            state=inventory(args.source,args.qualified)
            if state['blocked'] and not state.get('qualified'):raise ValueError('Source preparation is blocked; nothing is training-ready')
            pending_paths={item['path'] for item in pending.values()}
            for item in state['shards']:
                if len(pending)>=args.workers:break
                if item['path'] not in completed and item['path'] not in pending_paths:
                    pending[pool.submit(transfer,item,args.source,args.destination)]=item
                    pending_paths.add(item['path'])
            finished,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED) if pending else (set(),set())
            for future in finished:
                item=pending.pop(future);completed[item['path']]=future.result()
            progress={'complete':False,'seconds':time.monotonic()-started,'completed':completed,
                      'active_shards':len(pending),'published_shards':len(state['shards']),
                      'bytes':sum(x['bytes'] for x in completed.values())}
            tmp=args.state.with_suffix('.tmp');tmp.write_text(json.dumps(progress,indent=2)+'\n');tmp.replace(args.state)
            if finished:print(json.dumps({k:v for k,v in progress.items() if k!='completed'}),flush=True)
            if state['complete'] and not pending and len(completed)==len(state['shards']):
                progress.update(publish(state,args.destination,completed))
                args.state.write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True);return
            if not pending:time.sleep(10)


if __name__=='__main__':main()
