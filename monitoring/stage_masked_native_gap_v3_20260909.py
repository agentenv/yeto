"""Stage the corrected masked runtime over verified existing same-host code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

MODULES = {'training/qwen38_native_gap_v3/'+name+'.py' for name in (
    '__init__','normalize','source_adapters','masked','masked_data','masked_recipe',
    'masked_train','masked_replay','prepare_masked_full')}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def stage(archive,archive_sha256,manifest_sha256,original_root,output):
    raise RuntimeError('Retired v3 code stager; use stage_masked_native_gap_v4_20260910.py')
    archive,original_root,output=map(Path,(archive,original_root,output))
    if sha(archive)!=archive_sha256:raise ValueError('Archive changed')
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        if ({m.name for m in members}!=MODULES|{'delta-manifest.json'} or len(members)!=len(MODULES)+1
                or any(not m.isfile() for m in members)):raise ValueError('Unexpected runtime delta scope')
        raw=bundle.extractfile('delta-manifest.json').read()
        if hashlib.sha256(raw).hexdigest()!=manifest_sha256:raise ValueError('Manifest changed')
        manifest=json.loads(raw)
        if manifest['schema']!='masked-native-gap-runtime-delta/v3' or set(manifest['new_files'])!=MODULES:
            raise ValueError('Wrong masked runtime contract')
        payload={name:bundle.extractfile(name).read() for name in MODULES}
    sources={}
    for item in manifest['existing_dependencies']:
        name=item['path'];relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe dependency path')
        source=original_root/relative
        if source.is_symlink() or not source.is_file() or source.stat().st_size!=item['bytes'] or sha(source)!=item['sha256']:
            raise ValueError('Existing same-host dependency changed: '+name)
        sources[name]=source
    for name,contents in payload.items():
        if hashlib.sha256(contents).hexdigest()!=manifest['new_files'][name]:raise ValueError('New module changed')
        compile(contents,name,'exec')
    output.mkdir(parents=True,exist_ok=False)
    files=[]
    for name,source in sources.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for name,contents in payload.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(contents)
    for name in sorted(set(sources)|set(payload)):
        target=output/name;target.chmod(0o444)
        files.append({'path':name,'sha256':sha(target),'bytes':target.stat().st_size})
    code={'schema':'masked-native-gap-runtime-code/v3','files':files}
    (output/'code-manifest.json').write_text(json.dumps(code,sort_keys=True,indent=2)+'\n')
    receipt={'schema':'masked-native-gap-runtime-stage/v3','code':str(output),'archive_sha256':archive_sha256,
        'delta_manifest_sha256':manifest_sha256,'code_manifest_sha256':sha(output/'code-manifest.json'),
        'new_modules':len(payload),'existing_dependencies_copied_same_host':len(sources),'old_code_changed':False}
    (output/'stage-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--archive-sha256',required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--original-root',required=True);p.add_argument('--output',required=True)
    print(json.dumps(stage(**vars(p.parse_args()))))
