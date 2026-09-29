#!/bin/bash
# diag.sh <tag> <legacy|ports> <run names...>  -- tf-diagnosis 2026-09-29: TF replays with passive tfdiag shim
TAG=$1; ENG=$2; shift 2; RUNS="$@"; C=/home/michael/work/gpu-eq/ctl; cd $C; P=/tmp/modal-venv/bin/python
EV=/home/michael/work/gpu-eq/evidence/2026-09-29-tf-diagnosis; mkdir -p $EV/env-$ENG
teardown() { S=$(cat $TAG.sid 2>/dev/null); [ -n "$S" ] && { T=300 $P sbx.py exec $S "nvidia-smi; pip freeze 2>/dev/null" > $EV/env-$ENG/sandbox-env-$TAG.txt 2>&1; $P sbx.py kill $S; }; date -u +%s > $TAG.t_down; echo DOWN; }
trap 'echo TRAP; teardown' EXIT
date -u +%s > $TAG.t_create
timeout 3600 $P sbx.py create $ENG H100!:2 7200 > $TAG.create.log 2>&1 || { echo CREATE_FAIL; tail $TAG.create.log; exit 1; }
tail -1 $TAG.create.log > $TAG.sid; S=$(cat $TAG.sid); date -u +%s > $TAG.t_sbx; echo SBX $S
X() { T=${T:-600} $P sbx.py exec $S "$1" 2>&1 | grep -v -i deprec; }
G=$(T=120 $P sbx.py exec $S "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null); echo "GPU: $G"
if ! echo "$G" | grep -q H100 || echo "$G" | grep -qv H100; then echo "GPU_NOT_H100 abort"; exit 3; fi
X "mkdir -p /work/out"
$P sbx.py put $S yeto.tgz /work >/dev/null; $P sbx.py put $S harness.tgz /work >/dev/null; $P sbx.py put $S data.tgz /work >/dev/null
SP=$(X "python -c 'import site;print(site.getsitepackages()[0])'" | tail -1); echo SITE $SP
$P sbx.py put $S tfdiag-shim.tgz $SP >/dev/null
if [ $ENG = legacy ]; then MR=/opt/miles-legacy; $P sbx.py put $S shim.tgz $SP >/dev/null
  X "echo 'import yeto_legacy_router_timeout' > $SP/zz_yeto_legacy_router_timeout.pth"; else MR=/opt/miles-next; fi
X "touch /work/tfdiag.on; cd /work/yeto && git rev-parse HEAD 2>/dev/null; python -c 'import yeto_tfdiag' && echo SHIM_OK"
$P sbx.py put $S pt2-strict.tgz /work/out >/dev/null
T=1500 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto/syncer && cargo build --release --locked --quiet 2>&1 | grep -i '^error'; ls target/release/yeto-syncer; ls /work/out/legacy-s17/work/seed-17/*/rollouts/island-*/0.pt | wc -l; sha256sum /work/out/legacy-s17/work/seed-17/*/rollouts/island-*/0.pt"
LA="$(cat /home/michael/work/gpu-eq/evidence/2026-09-29-eq62-v2/launch-args-$ENG.txt)"
TFX="--global-rounds 1 --custom-generate-function-path yeto.rl.teacher_forcing.replay_generate"
REPLAY="YETO_RL_AUDIT_GRADS=1 YETO_RL_REPLAY_ROLLOUTS='/work/out/legacy-s17/work/seed-17/*/rollouts/island-*/0.pt'"
for N in $RUNS; do
  T=3000 X "export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto && export PYTHONPATH=$MR:/work/yeto:/work/harness:\$PYTHONPATH && rm -rf /work/out/$N && mkdir -p /work/out/$N && nvidia-smi > /work/out/$N/nvidia-smi.txt && nvidia-smi --query-gpu=name --format=csv,noheader | grep -q H100 && date -u +%s > /work/out/$N/t_start; $REPLAY python scripts/benchmark_rl.py $LA $TFX --seeds 17 --rl-engine $ENG --work-dir /work/out/$N/work --report-dir /work/out/$N/report > /work/out/$N/launch.log 2>&1; echo \$? > /work/out/$N/rc; date -u +%s > /work/out/$N/t_end; echo $N rc=\$(cat /work/out/$N/rc); tail -2 /work/out/$N/launch.log; cd /work/out && tar czf /tmp/$N.tgz --exclude='*.pt' --exclude='*.safetensors' --exclude='*.f32' --exclude='state.ckpt*' --exclude='*.tfd' $N; find $N -path '*/audit/round-00000001.grad.*' -o -name '*.tfd' | tar czf /tmp/$N.x.tgz -T -"
  $P sbx.py get $S /tmp/$N.tgz $C/$TAG-$N.tgz; $P sbx.py get $S /tmp/$N.x.tgz $C/$TAG-$N.x.tgz
  tar xzf $C/$TAG-$N.tgz -C $EV && tar xzf $C/$TAG-$N.x.tgz -C $EV; echo "copied $N rc=$(cat $EV/$N/rc) tfd=$(find $EV/$N -name '*.tfd' | wc -l)"
done
echo ALLDONE
