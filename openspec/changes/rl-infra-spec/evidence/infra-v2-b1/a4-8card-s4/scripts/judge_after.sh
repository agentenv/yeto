#!/bin/bash
# usage: judge_after.sh <run dir> <case: e1a_c|e1b|wd|a4b|a4bu|d123|d4|r5|r6|r7|r5c> [extra judge_inject args...]
# (after the run's final pull; writes judgment.json + markers into <run dir>; feeds samplers/probes/dkill log to the judge)
R=$1; C=$2; shift 2; B=/home/michael/work/gpu-b1-runs
rm -rf $R/.j; mkdir -p $R/.j
b=$R/pulled/elastic-state-final.b64; [ -s $b ] || b=$R/pulled/elastic-state.tgz.b64
base64 -d $b 2>/dev/null | tar xz -C $R/.j 2>/dev/null
T=$R/pulled/rl-island-0.final.jsonl; [ -s $T ] || T=$R/pulled/rl-island-0.jsonl
G=$R/pulled/gpu_samples.jsonl; [ -s $G ] || G=$R/gpu_samples.after.jsonl
# probes: the in-container term_probe.py output (pulled/) first, else the host-side hook's files
for p in after stale oldepoch; do f=$R/pulled/probe_$p.txt; grep -qs '"probe_attested": true' $f || f=$R/probe_$p.txt; eval P_$p=$f; done
python3 $B/judge_inject.py $C $R/.j/elastic-state/reconfig/journal.jsonl $T --router-samples $R/pulled/router_samples.jsonl \
  --gpu-samples $G --probe-after $P_after --probe-stale $P_stale --probe-oldepoch $P_oldepoch \
  --dkill-log $R/pulled/dkill.log --launch-log $R/launch.log --ledger $R/.j/elastic-state/ledger/journal.jsonl \
  --out $R/judgment.json --marker-dir $R "$@"
