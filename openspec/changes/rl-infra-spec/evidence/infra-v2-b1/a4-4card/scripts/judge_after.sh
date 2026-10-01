#!/bin/bash
# usage: judge_after.sh <run dir> <case: e1a_c|e1b|wd|a4b> [extra judge_inject args...]  (after the run's final pull; writes judgment.json + markers into <run dir>)
R=$1; C=$2; shift 2; B=/home/michael/work/gpu-b1-runs
rm -rf $R/.j; mkdir -p $R/.j
b=$R/pulled/elastic-state-final.b64; [ -s $b ] || b=$R/pulled/elastic-state.tgz.b64
base64 -d $b 2>/dev/null | tar xz -C $R/.j 2>/dev/null
T=$R/pulled/rl-island-0.final.jsonl; [ -s $T ] || T=$R/pulled/rl-island-0.jsonl
python3 $B/judge_inject.py $C $R/.j/elastic-state/reconfig/journal.jsonl $T --router-samples $R/pulled/router_samples.jsonl --out $R/judgment.json --marker-dir $R "$@"
