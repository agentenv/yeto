#!/bin/bash
# full legacy-v2 baseline with guaranteed teardown
cd /home/michael/work/gpu-legacy/ctl2; P=/tmp/modal-venv/bin/python
E=/home/michael/yeto/openspec/changes/rl-engine-ports/evidence/2026-09-29-legacy-baseline-v2; mkdir -p $E/env
copyenv() { S=$(cat sid 2>/dev/null) && T=200 $P lsbx.py exec $S "nvidia-smi; cd /opt/miles-legacy && git rev-parse HEAD && git remote get-url origin && git status --porcelain --untracked-files=all | wc -l; cd /opt/sglang-legacy && git rev-parse HEAD; pip freeze 2>/dev/null" > $E/env/sandbox-env.txt 2>&1; cp lsbx.py runleg.sh teardown.sh driver.sh yeto_legacy_router_timeout.py create.log $E/env/ 2>/dev/null; }
trap 'echo TRAP; copyenv; ./teardown.sh; date -u +%s > t_down; echo DOWN' EXIT
date -u +%s > t_create
timeout 3000 $P lsbx.py create H100:2 14400 > create.log 2>&1 || { echo CREATE_FAIL; exit 1; }
tail -1 create.log > sid; S=$(cat sid); date -u +%s > t_sbx; echo SBX $S
cd /home/michael/work/gpu-legacy; tar czf ctl2/yeto.tgz --exclude=.git --exclude=__pycache__ --exclude=syncer/target yeto2; cd ctl2
T=900 $P lsbx.py exec $S "mkdir -p /work" >/dev/null 2>&1
$P lsbx.py put $S yeto.tgz /work 2>&1 | grep -v -i deprec
T=100 $P lsbx.py exec $S "mv /work/yeto2 /work/yeto" 2>&1 | grep -v -i deprec
$P lsbx.py put $S ../ctl/harness.tgz /work 2>&1 | grep -v -i deprec
$P lsbx.py put $S ../ctl/data.tgz /work 2>&1 | grep -v -i deprec
$P lsbx.py put $S shim.tgz /usr/local/lib/python3.12/dist-packages 2>&1 | grep -v -i deprec
T=1500 $P lsbx.py exec $S "echo 'import yeto_legacy_router_timeout' > /usr/local/lib/python3.12/dist-packages/zz_yeto_legacy_router_timeout.pth; cd /opt/miles-legacy && git rev-parse HEAD; export PATH=\$HOME/.cargo/bin:\$PATH; cd /work/yeto/syncer && cargo build --release --locked --quiet 2>&1 | grep -i '^error' ; ls target/release/yeto-syncer" 2>&1 | grep -v -i deprec
./runleg.sh legacy-0
G=$(grep -a -h -o "'train/grad_norm': [0-9.e+-]*" $E/legacy-0/work/seed-17/*/island-*/miles.log | sort -u); echo "GRAD $G"
if ! echo "$G" | grep -q "'train/grad_norm': [0-9.e+-]*[1-9]"; then echo ZERO_GRAD_STOP; exit 2; fi
./runleg.sh legacy-1
./runleg.sh legacy-2
echo ALLDONE
