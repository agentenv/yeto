#!/bin/bash
# runcmd.sh <name> <remote command writing into /work/out/<name>>
cd /tmp/yeto-gpu; S=$(cat sid); P=/tmp/modal-venv/bin/python; N=$1
T=5000 $P sbx.py exec $S "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto && export PYTHONPATH=/opt/miles-next:/work/yeto:/work/harness:\$PYTHONPATH && rm -rf /work/out/$N && mkdir -p /work/out/$N && ( $2 ) > /work/out/$N/run.log 2>&1; echo rc=\$?; tail -5 /work/out/$N/run.log; tar czf /tmp/$N.tgz --exclude='*.pt' --exclude='*.safetensors' --exclude='state.ckpt*' -C /work/out $N" 2>&1 | grep -v -i deprec
E=/home/michael/yeto/openspec/changes/rl-engine-ports/evidence/2026-09-29-$N; mkdir -p $E
$P sbx.py get $S /tmp/$N.tgz /tmp/yeto-gpu/$N.tgz 2>&1 | grep -v -i deprec; tar xzf /tmp/yeto-gpu/$N.tgz -C $E --strip-components=1; echo copied $E
