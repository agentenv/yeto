#!/bin/bash
# run.sh <mode> <name> [envprefix]; copies evidence back after the run
cd /tmp/yeto-gpu; S=$(cat sid); P=/tmp/modal-venv/bin/python
T=4000 $P sbx.py exec $S "cd /work/yeto && export PYTHONPATH=/opt/miles-next:/work/yeto:/work/harness:\$PYTHONPATH && rm -rf /work/out/$2 && $3 python /work/harness/smoke.py $1 /work/out/$2 2>&1 | tail -5; tar czf /tmp/$2.tgz --exclude='*.pt' -C /work/out $2" 2>&1 | grep -v -i deprec
E=/home/michael/yeto/openspec/changes/rl-engine-ports/evidence/2026-09-29-$2; mkdir -p $E
$P sbx.py get $S /tmp/$2.tgz /tmp/yeto-gpu/$2.tgz 2>&1 | grep -v -i deprec; tar xzf /tmp/yeto-gpu/$2.tgz -C $E --strip-components=1 && ls $E
