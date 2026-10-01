#!/bin/bash
R=/home/michael/work/infra-a-gpu/b12
export HOME=$R/home
M=/tmp/modal-venv/bin/modal
while [ ! -f $R/rc.txt ]; do
  for c in $($M container list --json 2>/dev/null | /usr/bin/python3 -c 'import json,sys
try:
  d=json.load(sys.stdin)
except Exception: d=[]
for x in d:
  if "infra-a-b12" in json.dumps(x): print(x.get("Container ID") or x.get("container_id") or "")'); do
    for i in 0 1; do
      timeout 60 $M container exec $c -- sh -c "cat /root/yeto-output/rl-island-$i.jsonl 2>/dev/null" > $R/pulled/.tmp.$c.$i 2>/dev/null
      [ -s $R/pulled/.tmp.$c.$i ] && mv $R/pulled/.tmp.$c.$i $R/pulled/rl-island-$i.jsonl || rm -f $R/pulled/.tmp.$c.$i
    done
    timeout 60 $M container exec $c -- sh -c "nvidia-smi --query-gpu=name,uuid --format=csv,noheader" > $R/pulled/gpu-$c.txt 2>/dev/null
  done
  sleep 45
done
