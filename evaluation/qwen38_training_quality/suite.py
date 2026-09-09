"""Freeze 16 original synthetic filesystem-action tasks without running a model."""
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'yeta.qwen38-action-quality-suite/v1'
SOURCE = 'yeta-synthetic-action-quality@1'
BASE_IMAGE = 'sha256:332c9f06adba1ad58a8829205e6f9b65023a162ee0714ef99c29763de5700584'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task(name, request, inputs, output, expected, kind='exact'):
    return {'id':name,'request':request,'inputs':inputs,'output':output,'expected':expected,'verifier_kind':kind}


def specifications():
    """Public fixtures, written independently of the scored benchmark tasks."""
    text='A small synthetic checksum fixture.\n'
    return [
      task('write-marker','Write ACTION-CHECK-OK followed by a newline to result.txt.',{},'result.txt','ACTION-CHECK-OK\n'),
      task('copy-observation','Read inputs/message.txt and copy its exact bytes into result.txt.',{'message.txt':'observed value: cedar-47\n'},'result.txt','observed value: cedar-47\n'),
      task('sum-json','Sum the integers in inputs/numbers.json and write the decimal result plus a newline to result.txt.',{'numbers.json':'[7, -3, 12, 9, -5]\n'},'result.txt','20\n'),
      task('sort-unique-lines','Read inputs/items.txt; write its distinct nonempty lines in alphabetical order, one per line, to result.txt.',{'items.txt':'pear\napple\npear\n\nfig\napple\n'},'result.txt','apple\nfig\npear\n'),
      task('csv-total','Sum the amount column in inputs/amounts.csv. Write the integer total and a newline to result.txt.',{'amounts.csv':'item,amount\nalpha,13\nbeta,-4\ngamma,8\n'},'result.txt','17\n'),
      task('edit-config','Copy inputs/config.json to result.json, changing enabled to true and preserving every other field.',{'config.json':'{"enabled":false,"port":8123,"labels":["red","blue"]}\n'},'result.json',{'enabled':True,'port':8123,'labels':['red','blue']},'json'),
      task('patch-text','Copy inputs/settings.txt to result.txt while replacing only the line mode=old with mode=new. Preserve all other bytes.',{'settings.txt':'# test settings\nmode=old\ntimeout=17\n'},'result.txt','# test settings\nmode=new\ntimeout=17\n'),
      task('append-log','Copy inputs/log.txt to result.txt and append exactly third event followed by a newline.',{'log.txt':'first event\nsecond event\n'},'result.txt','first event\nsecond event\nthird event\n'),
      task('extract-json','Read inputs/object.json. Write the value of service.region followed by a newline to result.txt.',{'object.json':'{"service":{"region":"west-3","port":9134},"owner":"test"}\n'},'result.txt','west-3\n'),
      task('inventory-files','List every regular file below inputs/tree, recursively, using relative paths from inputs/tree. Sort the paths and write one per line to result.txt.',{'tree/beta.txt':'b\n','tree/nested/zeta.txt':'z\n','tree/alpha.txt':'a\n'},'result.txt','alpha.txt\nbeta.txt\nnested/zeta.txt\n'),
      task('checksum-file','Compute the SHA256 of the exact bytes in inputs/payload.txt; write its lowercase hex digest plus a newline to result.txt.',{'payload.txt':text},'result.txt',hashlib.sha256(text.encode()).hexdigest()+'\n'),
      task('count-matches','Count lines exactly equal to red in inputs/colors.txt. Write the count plus a newline to result.txt.',{'colors.txt':'red\nblue\nredwood\nred\ngreen\nred\n'},'result.txt','3\n'),
      task('filter-json','Read inputs/users.json. Write a JSON array containing the name of each active user, preserving input order, to result.json.',{'users.json':'[{"name":"Ada","active":true},{"name":"Ben","active":false},{"name":"Cy","active":true}]\n'},'result.json',['Ada','Cy'],'json'),
      task('merge-config','Merge inputs/base.json and inputs/override.json into result.json. Override keys replace base values; preserve other base keys.',{'base.json':'{"color":"blue","width":8,"debug":false}\n','override.json':'{"width":11,"debug":true}\n'},'result.json',{'color':'blue','width':11,'debug':True},'json'),
      task('evaluate-expression','Read the JSON object in inputs/values.json. Compute (left + right) * scale and write the decimal value plus a newline to result.txt.',{'values.json':'{"left":6,"right":-2,"scale":7}\n'},'result.txt','28\n'),
      task('write-function','Create solution.py with a function total_even(values) that returns the sum of even integers in values. It must handle an empty list and negative integers.',{},'solution.py',None,'python-function'),
    ]


VERIFY = r'''import importlib.util,json
from pathlib import Path
spec=json.loads(Path('/tests/expected.json').read_text())
root=Path('/work/quality')/spec['id'];p=root/spec['output']
ok=False
try:
 if spec['verifier_kind']=='exact':ok=p.is_file() and p.read_bytes()==spec['expected'].encode()
 elif spec['verifier_kind']=='json':ok=p.is_file() and json.loads(p.read_text())==spec['expected']
 elif spec['verifier_kind']=='python-function':
  module_spec=importlib.util.spec_from_file_location('candidate_solution',p)
  module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
  cases=[([],0),([1,2,3,4],6),([-6,-3,0,8],2),([7,9],0)]
  ok=all(module.total_even(values)==expected for values,expected in cases)
except Exception:ok=False
Path('/logs/verifier').mkdir(parents=True,exist_ok=True)
Path('/logs/verifier/reward.txt').write_text('1' if ok else '0')
'''


def build(output, *, base_image=BASE_IMAGE):
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',base_image):raise ValueError('Pin the actual prevalidated task base image by digest')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    specs=specifications();assert len(specs)==len({s['id'] for s in specs})==16
    for spec in specs:
        root=out/'tasks'/spec['id'];env=root/'environment';tests=root/'tests'
        (env/'inputs').mkdir(parents=True);tests.mkdir()
        (env/'inputs/.quality-inputs').write_text('synthetic inputs only\n')
        for name,content in spec['inputs'].items():
            p=env/'inputs'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
        (env/'Dockerfile').write_text(f'FROM {base_image}\nENTRYPOINT []\nCMD ["sleep", "infinity"]\nWORKDIR /work/quality/{spec["id"]}\nCOPY inputs/ ./inputs/\n')
        (root/'instruction.md').write_text(spec['request']+'\n\nThe working directory is /work/quality/'+spec['id']+'. Complete the requested file change, check the result, and report completion.\n')
        (root/'task.toml').write_text('''version = "1.0"
[metadata]
author_name = "Yeta Labs"
difficulty = "easy"
category = "synthetic-action-quality"
tags = ["synthetic", "fixed-monitor", "not-terminal-bench"]
[agent]
timeout_sec = 300.0
[verifier]
timeout_sec = 60.0
[environment]
build_timeout_sec = 300.0
cpus = 2
memory_mb = 4096
storage_mb = 8192
''')
        (tests/'expected.json').write_text(json.dumps({k:v for k,v in spec.items() if k not in ('inputs','request')},sort_keys=True)+'\n')
        (tests/'verify.py').write_text(VERIFY)
        (tests/'test.sh').write_text('#!/usr/bin/env bash\nset -eu\nmkdir -p /logs/verifier\npython3 /tests/verify.py\n')
    files={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    manifest={'schema':SCHEMA,'source':SOURCE,'task_count':16,'task_ids':[s['id'] for s in specs],
        'base_image':base_image,'files':files,'synthetic_only':True,'terminal_bench_score':False,
        'training_data':False,'single_attempt_per_task':True,'best_of_selection':False,
        'suite_source_sha256':digest(__file__)}
    (out/'suite.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def verify_suite(path):
    root=Path(path);m=json.loads((root/'suite.json').read_bytes())
    if m.get('schema')!=SCHEMA or m.get('task_count')!=16 or len(set(m['task_ids']))!=16:raise ValueError('Wrong frozen suite')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='suite.json'}
    if actual!=set(m['files']):raise ValueError('Frozen task coverage changed')
    for name,expected in m['files'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or digest(p)!=expected:raise ValueError('Frozen task bytes changed')
    return m


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.output),sort_keys=True))
