import importlib.util,json
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
