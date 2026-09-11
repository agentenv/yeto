"""Copy a completed v4 export manager->n3 byte-for-byte through existing SSH, then atomically publish."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/private/tmp/yeta-labeling-known-hosts','-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3']
MANAGER='c@65.19.161.135';NODE='ubuntu@100.66.92.111'
PUBLISHED_FILE_MODE=0o444
PUBLISHED_DIRECTORY_MODE=0o555


def modes_allow_unprivileged_read(file_mode=PUBLISHED_FILE_MODE,
                                  directory_mode=PUBLISHED_DIRECTORY_MODE):
 return (file_mode&0o222)==0 and bool(file_mode&0o004) and (directory_mode&0o222)==0 and bool(directory_mode&0o001)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def remote_json(host,script,sudo=False):
 r=subprocess.run(SSH+[host,('sudo -n ' if sudo else '')+'python3 -'],input=script.encode(),capture_output=True)
 if r.returncode:raise RuntimeError((r.stderr+r.stdout).decode(errors='replace')[-3000:])
 return json.loads(r.stdout)
def inventory(root):
 return remote_json(MANAGER,'''from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json
root=Path(%r).resolve(strict=True)
if not root.name=='export' or not root.parent.name.startswith('cot-native-gap-v4-'):raise ValueError('wrong source')
if (root/'BLOCKED.json').exists() or not (root/'COMPLETE.json').is_file():raise ValueError('source is not atomically complete')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
members=sorted(root.rglob('*'))
if any(p.is_symlink() or not p.is_file() or p.resolve(strict=True)!=p for p in members):raise ValueError('source export must be a flat regular-file tree')
paths=members
if any(len(p.relative_to(root).parts)!=1 for p in paths) or len({p.name.casefold() for p in paths})!=len(paths):raise ValueError('source export paths are not portable')
def one(p):return {'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p),'mode':p.stat().st_mode&0o777}
with ThreadPoolExecutor(max_workers=8) as pool:files=list(pool.map(one,paths))
complete=json.loads((root/'COMPLETE.json').read_text());manifest=json.loads((root/'manifest.json').read_text());index=json.loads((root/'index.json').read_text())
if complete.get('schema')!='qwen38-generated-cot-native-gap-complete/v4' or complete.get('status')!='complete' or complete.get('manifest')!={'path':'manifest.json','sha256':sha(root/'manifest.json')} or complete.get('index')!={'path':'index.json','sha256':sha(root/'index.json')}:raise ValueError('COMPLETE mismatch')
for entries in manifest['splits'].values():
 for x in entries:
  if Path(x['path']).is_absolute() or len(Path(x['path']).parts)!=1:raise ValueError('nonportable manifest path')
for entries in index['splits'].values():
 for x in entries:
  if Path(x['path']).is_absolute() or len(Path(x['path']).parts)!=1:raise ValueError('nonportable index path')
result={'source':str(root),'files':files,'file_count':len(files),'bytes':sum(x['bytes'] for x in files),'manifest_sha256':sha(root/'manifest.json'),'index_sha256':sha(root/'index.json'),'complete_sha256':sha(root/'COMPLETE.json')}
result['tree_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
print(json.dumps(result))
'''%root)
def recover(entries,incoming):
 return remote_json(NODE,'''from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json
root=Path(%r);entries=%r
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def one(x):
 p=root/x['path']
 if not p.exists():return None
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('bad existing incoming file: '+x['path'])
 return x['path']
with ThreadPoolExecutor(max_workers=8) as pool:ok=[x for x in pool.map(one,entries) if x]
print(json.dumps({'recovered':ok}))
'''%(incoming,entries),sudo=True)
def transfer_one(item,source,incoming):
 rel=item['path'];src=str(Path(source)/rel);target=str(Path(incoming)/rel)
 receiver='''import gzip,hashlib,json,os,sys,uuid
from pathlib import Path
p=Path(%r);want=%r;size_want=%d
p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
if p.exists():raise FileExistsError(p)
tmp=p.with_name('.'+p.name+'.incoming-'+want[:12]+'-'+uuid.uuid4().hex);h=hashlib.sha256();n=0
with gzip.GzipFile(fileobj=sys.stdin.buffer,mode='rb') as src,tmp.open('xb') as dst:
 for b in iter(lambda:src.read(8<<20),b''):dst.write(b);h.update(b);n+=len(b)
 dst.flush();os.fsync(dst.fileno())
if n!=size_want or h.hexdigest()!=want:raise ValueError('stream mismatch')
tmp.chmod(0o400);os.rename(tmp,p)
print(json.dumps({'path':%r,'bytes':n,'sha256':h.hexdigest()}))
'''%(target,item['sha256'],item['bytes'],rel)
 started=time.monotonic();producer=subprocess.Popen(SSH+[MANAGER,'gzip -1 -c -- '+shlex.quote(src)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 consumer=subprocess.Popen(SSH+[NODE,'sudo -n python3 -c '+shlex.quote(receiver)],stdin=producer.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);producer.stdout.close()
 out,err=consumer.communicate();_,perr=producer.communicate()
 if consumer.returncode or producer.returncode:raise RuntimeError((err+perr).decode(errors='replace')[-2000:])
 result=json.loads(out);result['seconds']=time.monotonic()-started;return result
def publish(state,incoming,destination):
 return remote_json(NODE,'''from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,datetime
incoming=Path(%r);dest=Path(%r);state=%r
file_mode=%r;directory_mode=%r
if dest.exists():raise FileExistsError(dest)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def one(x):
 p=incoming/x['path']
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise ValueError('destination verify failed: '+x['path'])
 return x['path']
members=sorted(incoming.rglob('*'))
if any(p.is_symlink() or not p.is_file() for p in members):raise ValueError('destination contains a non-regular member')
actual=sorted(str(p.relative_to(incoming)) for p in members)
if actual!=sorted(x['path'] for x in state['files']):raise ValueError('destination tree membership differs')
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(one,state['files']))
# Recheck exact portable evidence before atomic directory publication.
complete=json.loads((incoming/'COMPLETE.json').read_text())
if complete.get('manifest')!={'path':'manifest.json','sha256':state['manifest_sha256']} or complete.get('index')!={'path':'index.json','sha256':state['index_sha256']}:raise ValueError('destination gates differ')
for p in members:p.chmod(file_mode)
incoming.chmod(directory_mode)
# The transfer receiver runs under sudo. World-readable regular files and a
# world-traversable immutable directory let the UID-1000 pinned container read
# the exact tree without changing any dataset bytes or ownership metadata.
if any((p.stat().st_mode&0o777)!=file_mode for p in members) or (incoming.stat().st_mode&0o777)!=directory_mode:raise ValueError('destination immutable/readable mode publication failed')
os.rename(incoming,dest)
fd=os.open(dest.parent,os.O_RDONLY)
try:os.fsync(fd)
finally:os.close(fd)
receipt={'schema':'qwen38-native-gap-v4-exact-tree-transfer/v1','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','source':state['source'],'destination':str(dest),'file_count':state['file_count'],'bytes':state['bytes'],'tree_sha256':state['tree_sha256'],'manifest_sha256':state['manifest_sha256'],'index_sha256':state['index_sha256'],'complete_sha256':state['complete_sha256'],'transport':'manager-to-mac-to-n3-existing-encrypted-ssh-gzip1','byte_preserving':True,'metadata_rewritten':False,'index_rebuilt':False,'published_file_mode':'0444','published_directory_mode':'0555','uid1000_readable_by_mode':True,'atomic_publish':True}
raw=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode();rp=dest.with_name(dest.name+'.transfer-receipt.json')
with rp.open('xb') as f:f.write(raw)
rp.chmod(0o400)
print(json.dumps({**receipt,'receipt':str(rp),'receipt_sha256':hashlib.sha256(raw).hexdigest()}))
'''%(incoming,destination,state,PUBLISHED_FILE_MODE,PUBLISHED_DIRECTORY_MODE),sudo=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--state',required=True,type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if not a.source.startswith('/mnt/lvm_data/sft_analysis/sft_baseline_20260908/cot-native-gap-v4-') or not a.source.endswith('/export'):raise ValueError('wrong source scope')
 if not a.destination.startswith('/data/sft_baseline_20260908/datasets/cot-masked-native-gap-v4-'):raise ValueError('wrong destination scope')
 if not 1<=a.workers<=4:raise ValueError('bad concurrency')
 state=inventory(a.source);incoming=str(Path(a.destination).with_name('.'+Path(a.destination).name+'.incoming-'+state['tree_sha256'][:12]))
 recovered=set(recover(state['files'],incoming)['recovered']);pending=[x for x in state['files'] if x['path'] not in recovered];started=time.monotonic();done=len(recovered)
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futs={pool.submit(transfer_one,x,a.source,incoming):x for x in pending}
  for f in as_completed(futs):
   f.result();done+=1
   progress={'status':'transferring','done':done,'files':state['file_count'],'bytes_total':state['bytes'],'elapsed_seconds':time.monotonic()-started,'tree_sha256':state['tree_sha256']}
   tmp=a.state.with_suffix('.tmp');tmp.write_text(json.dumps(progress,indent=2)+'\n');tmp.replace(a.state)
   if done%8==0 or done==state['file_count']:print(json.dumps(progress),flush=True)
 if inventory(a.source)!=state:raise ValueError('source tree changed during transfer')
 result=publish(state,incoming,a.destination);a.state.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
