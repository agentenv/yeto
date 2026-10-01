#!/bin/bash
# usage: n2inwatch.sh <run dir> '<json triggers>'
R=$1; T=$2; CL=$(cat $R/cluster.txt)
until [ -s $R/pulled/gpu.txt ]; do [ -f $R/rc.txt ] && exit 1; sleep 10; done
b64=$(base64 -w0 /home/michael/work/gpu-b1-runs/inwatch.py); sb=$(base64 -w0 /home/michael/work/gpu-b1-runs/router_sampler.py); tb=$(printf '%s' "$T" | base64 -w0)
HOME=/home/michael timeout 120 ssh -o StrictHostKeyChecking=no $CL "mkdir -p ~/yeto-rl && echo $sb | base64 -d > ~/yeto-rl/router_sampler.py && (nohup python3 ~/yeto-rl/router_sampler.py > ~/yeto-rl/sampler.out 2>&1 &) && echo $b64 | base64 -d > ~/yeto-rl/inwatch.py && (nohup python3 ~/yeto-rl/inwatch.py \"\$(echo $tb | base64 -d)\" > ~/yeto-rl/inwatch.out 2>&1 &) ; sleep 2; ps -eo pid,args | grep [i]nwatch" > $R/inwatch-arm.txt 2>&1
echo "armed rc=$? $(date -u +%FT%TZ)" >> $R/inwatch-arm.txt
