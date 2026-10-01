#!/bin/bash
# usage: after_term.sh <run dir> <prefix> <n_terminal> <extra_wait_s> : wait until the journal shows n terminal phases,
# wait extra_wait_s, run the fork status probe, pull samplers, then stop the run (nstop)
source /home/michael/work/gpu-b1-runs/probelib.sh
R=$1; P=$2; N=$3; W=$4; CL=$(cat $R/cluster.txt)
until [ -f $R/rc.txt ]; do
  if [ -s $R/pulled/elastic-state.tgz.b64 ]; then
    rm -rf $R/.st; mkdir -p $R/.st; base64 -d $R/pulled/elastic-state.tgz.b64 2>/dev/null | tar xz -C $R/.st 2>/dev/null
    n=$(grep -c '"phase":"\(SUCCEEDED\|REBUILT_OLD\|CANCELLED\|RECOVERY_REQUIRED\)"' $R/.st/elastic-state/reconfig/journal.jsonl 2>/dev/null)
    if [ "${n:-0}" -ge "$N" ]; then
      date -u +%FT%TZ > $R/terminal_seen.txt; sleep $W
      probe_ok $R $R/probe_after.txt status
      HOME=/home/michael timeout 120 ssh $CL 'cat ~/yeto-rl/gpu_samples.jsonl' > $R/gpu_samples.jsonl 2>/dev/null
      /home/michael/work/gpu-b1-runs/nstop.sh $P > $R/after_stop.txt 2>&1
      exit 0
    fi
  fi
  sleep 10
done
