#!/bin/bash
# usage: n2arm.sh <run dir> <script.py in gpu-b1-runs> [args...]  -- upload and start an in-container helper (nohup) once the island is up
R=$1; S=$2; shift 2; CL=$(cat $R/cluster.txt); B=/home/michael/work/gpu-b1-runs
until [ -s $R/pulled/gpu.txt ]; do [ -f $R/rc.txt ] && exit 1; sleep 10; done
b64=$(base64 -w0 $B/$S); ab=$(printf '%s\0' "$@" | base64 -w0)
HOME=/home/michael timeout 120 ssh -o StrictHostKeyChecking=no $CL "mkdir -p ~/yeto-rl && echo $b64 | base64 -d > ~/yeto-rl/$S && python3 - <<PY
import base64, subprocess, os
args = [a.decode() for a in base64.b64decode('$ab').split(b'\0') if a]
subprocess.Popen(['python3', os.path.expanduser('~/yeto-rl/$S')] + args, stdout=open(os.path.expanduser('~/yeto-rl/$S.out'), 'w'), stderr=subprocess.STDOUT, start_new_session=True)
PY
sleep 2; ps -eo pid,args | grep [$(echo ${S:0:1})]${S:1} | cut -c1-200" > $R/arm-$S.txt 2>&1
echo "armed $S rc=$? $(date -u +%FT%TZ)" >> $R/arm-$S.txt
