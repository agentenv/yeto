#!/bin/bash
# drive.sh <legacy|ports>: one H100 sandbox, kill44x, copy evidence, guaranteed teardown
ENG=$1; C=/home/michael/work/gpu-kill/ctl; cd $C; P=/tmp/modal-venv/bin/python
EV=/home/michael/work/r0-integ/openspec/changes/rl-engine-ports/evidence/2026-09-29-kill44-legacy-vs-ports; mkdir -p $EV/env-$ENG
teardown() { S=$(cat $ENG.sid 2>/dev/null); [ -n "$S" ] && { T=300 $P sbx.py exec $S "nvidia-smi; cd /opt/miles-legacy 2>/dev/null || cd /opt/miles-next; git rev-parse HEAD; cd /opt/sglang-legacy 2>/dev/null || cd /opt/sglang-next; git rev-parse HEAD; pip freeze 2>/dev/null" > $EV/env-$ENG/sandbox-env.txt 2>&1; $P sbx.py kill $S; }; date -u +%s > $ENG.t_down; echo DOWN; }
trap 'echo TRAP; teardown' EXIT
date -u +%s > $ENG.t_create
timeout 3600 $P sbx.py create $ENG 'H100!:1' 4200 > $ENG.create.log 2>&1 || { echo CREATE_FAIL; tail $ENG.create.log; exit 1; }
tail -1 $ENG.create.log > $ENG.sid; S=$(cat $ENG.sid); date -u +%s > $ENG.t_sbx; echo SBX $S
X() { T=${T:-600} $P sbx.py exec $S "$1" 2>&1 | grep -v -i deprec; }
G=$(T=120 $P sbx.py exec $S "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null); echo "GPU: $G"; echo "$G" > $EV/env-$ENG/gpu.txt
if ! echo "$G" | grep -q H100 || echo "$G" | grep -qv H100; then echo "GPU_NOT_H100 abort"; exit 3; fi
X "mkdir -p /work/out"
$P sbx.py put $S yeto.tgz /work >/dev/null; $P sbx.py put $S harness.tgz /work >/dev/null; $P sbx.py put $S data.tgz /work >/dev/null
if [ $ENG = legacy ]; then $P sbx.py put $S shim.tgz /usr/local/lib/python3.12/dist-packages >/dev/null
  X "echo 'import yeto_legacy_router_timeout' > /usr/local/lib/python3.12/dist-packages/zz_yeto_legacy_router_timeout.pth"; MR=/opt/miles-legacy; else MR=/opt/miles-next; fi
T=1500 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto/syncer && cargo build --release --locked --quiet 2>&1 | grep -i '^error'; ls target/release/yeto-syncer; wc -l /work/data/gsm8k.jsonl; cat /work/yeto/YETO_SHA"
N=kill44-$ENG
T=3000 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto && export PYTHONPATH=$MR:/work/yeto:/work/harness:\$PYTHONPATH && rm -rf /work/out/$N && mkdir -p /work/out/$N && nvidia-smi > /work/out/$N/nvidia-smi.txt && cp /work/yeto/YETO_SHA /work/out/$N/ && echo 'python /work/harness/kill44x.py $ENG /work/out/$N' > /work/out/$N/cmd.txt && ( timeout 2700 python /work/harness/kill44x.py $ENG /work/out/$N ) > /work/out/$N/run.log 2>&1; echo rc=\$? | tee /work/out/$N/rc; tail -4 /work/out/$N/run.log | cut -c1-400; cd /work/out && tar czf /tmp/$N.tgz --exclude='*.pt' --exclude='*.f32' --exclude='*.safetensors' --exclude='state.ckpt*' $N"
$P sbx.py get $S /tmp/$N.tgz $C/$N.tgz && tar xzf $C/$N.tgz -C $EV && echo "copied $EV/$N"
echo ALLDONE
