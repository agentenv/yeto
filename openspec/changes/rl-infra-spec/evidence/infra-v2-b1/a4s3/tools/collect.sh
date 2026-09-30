#!/bin/bash
# usage: collect.sh <prefix> <evidence-subdir-name> : copy a run's evidence into gpu-b1/.../evidence/infra-v2-b1/a4s3/<name>
P=$1; N=$2; R=/home/michael/work/gpu-b1-runs/$P; D=/home/michael/work/gpu-b1/openspec/changes/rl-infra-spec/evidence/infra-v2-b1/a4s3/$N
mkdir -p $D
for f in args.txt yeto_sha.txt start_utc.txt end_utc.txt rc.txt cluster.txt selfcheck.txt selfcheck_probe.txt terminal_seen.txt probe_after.txt probe_stale.txt probe_oldepoch.txt progress_stop.txt inwatch-arm.txt nstop.out watchdog.out; do [ -f $R/$f ] && cp $R/$f $D/; done
cp $R/probe_after.txt.attempts $D/ 2>/dev/null
for f in launch.log; do gzip -c $R/$f > $D/$f.gz; done
[ -f $R/pulled/rl-island-0.final.jsonl ] && [ -s $R/pulled/rl-island-0.final.jsonl ] && cp $R/pulled/rl-island-0.final.jsonl $D/rl-island-0.jsonl || cp $R/pulled/rl-island-0.jsonl $D/
for f in gpu.txt compute-apps.txt inwatch.final.log dkill.log dctl.log; do [ -s $R/pulled/$f ] && cp $R/pulled/$f $D/; done
[ -s $R/pulled/inwatch.final.log ] || cp $R/pulled/inwatch.log $D/ 2>/dev/null
for f in ps.txt router_samples.jsonl gpu_samples.jsonl; do [ -s $R/pulled/$f ] && gzip -c $R/pulled/$f > $D/$f.gz; done
[ -s $R/gpu_samples.jsonl ] && gzip -c $R/gpu_samples.jsonl > $D/gpu_samples.jsonl.gz
b=$R/pulled/elastic-state-final.b64; [ -s $b ] || b=""; 
mkdir -p $D/elastic-state-x; 
if [ -n "$b" ]; then base64 -d $b | tar xz -C $D/elastic-state-x 2>/dev/null; else base64 -d $R/pulled/elastic-state.tgz.b64 | tar xz -C $D/elastic-state-x 2>/dev/null; fi
rm -rf $D/elastic-state; mv $D/elastic-state-x/elastic-state $D/elastic-state 2>/dev/null; rmdir $D/elastic-state-x 2>/dev/null
ls $D | tr '\n' ' '; du -sh $D | cut -f1
