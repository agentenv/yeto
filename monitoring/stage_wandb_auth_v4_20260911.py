"""Stage only Walden's W&B credential in n3 tmpfs; never persist or print it."""
from __future__ import annotations

import argparse
import json
import netrc
from pathlib import Path
import re
import shlex
import subprocess
import uuid

SSH = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=yes',
       '-o','UserKnownHostsFile=/private/tmp/yeta-labeling-known-hosts']
NODE = 'ubuntu@100.66.92.111'
TOKEN = re.compile(r'[^\s]+')


def minimal_netrc_bytes(source):
    parsed=netrc.netrc(str(Path(source).expanduser()))
    if 'api.wandb.ai' not in parsed.hosts:
        raise ValueError('Local login has no api.wandb.ai credential')
    login,account,password=parsed.authenticators('api.wandb.ai')
    if account not in (None,'') or not all(isinstance(value,str) and TOKEN.fullmatch(value)
                                          for value in (login,password)):
        raise ValueError('W&B credential cannot be serialized as one minimal machine stanza')
    return ('machine api.wandb.ai\n  login '+login+'\n  password '+password+'\n').encode()


def stage(source, node=NODE):
    payload=minimal_netrc_bytes(source)
    name='/dev/shm/yeta-wandb-netrc-'+uuid.uuid4().hex
    receiver=r'''import json,netrc,os,sys
path=sys.argv[1];raw=sys.stdin.buffer.read(1024)
if not raw or len(raw)>=1024:raise ValueError('bad credential length')
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
try:
 os.write(fd,raw);os.fsync(fd)
finally:os.close(fd)
parsed=netrc.netrc(path)
if set(parsed.hosts)!={'api.wandb.ai'} or parsed.macros:raise ValueError('nonminimal credential')
login,account,password=parsed.authenticators('api.wandb.ai')
if not login or account not in (None,'') or not password:raise ValueError('incomplete credential')
st=os.lstat(path)
if st.st_uid!=0 or st.st_mode&0o777!=0o600:raise ValueError('wrong tmpfs credential mode')
print(json.dumps({'path':path,'bytes':st.st_size,'mode':'0600','owner':'root','machines':['api.wandb.ai']}))'''
    result=subprocess.run(SSH+[node,'sudo -n python3 -c '+shlex.quote(receiver)+' '+shlex.quote(name)],
                          input=payload,capture_output=True)
    if result.returncode:
        raise RuntimeError('Ephemeral W&B credential staging failed')
    return json.loads(result.stdout)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='~/.netrc');parser.add_argument('--node',default=NODE)
    print(json.dumps(stage(**vars(parser.parse_args())),sort_keys=True))
