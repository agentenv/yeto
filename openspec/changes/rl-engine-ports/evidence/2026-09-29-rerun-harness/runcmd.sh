#!/bin/bash
cd /tmp/yeto-rerun; S=$(cat $1); P=/tmp/modal-venv/bin/python; N=$2
T=6000 $P sbx.py exec $S "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto && export PYTHONPATH=/opt/miles-next:/work/yeto:/work/harness:\$PYTHONPATH && rm -rf /work/out/$N && mkdir -p /work/out/$N && cp /work/harness/YETO_SHA /work/out/$N/ && echo '$3' > /work/out/$N/cmd.txt && ( $3 ) > /work/out/$N/run.log 2>&1; echo rc=\$? | tee /work/out/$N/rc; tail -5 /work/out/$N/run.log; tar czf /tmp/$N.tgz --exclude='*.pt' --exclude='*.f32' --exclude='*.safetensors' --exclude='state.ckpt*' -C /work/out $N" 2>&1 | grep -v -i deprec
E=/home/michael/work/gpu-rerun/evidence/2026-09-29-rerun-$N; mkdir -p $E
$P sbx.py get $S /tmp/$N.tgz /tmp/yeto-rerun/$N.tgz 2>&1 | grep -v -i deprec; tar xzf /tmp/yeto-rerun/$N.tgz -C $E --strip-components=1; echo copied $E
