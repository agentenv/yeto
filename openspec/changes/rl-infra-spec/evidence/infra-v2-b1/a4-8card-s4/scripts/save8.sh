#!/bin/bash
# usage: save8.sh <prefix> <dest dir>   copy the evidence of one run (compact; big logs gzipped; no credentials: the run dir's home/ and yeto/ are NOT copied)
B=/home/michael/work/gpu-b1-runs; R=$B/$1; D=$2; mkdir -p $D
cp $(ls $R/runs/*/events/*.jsonl 2>/dev/null | head -1) $D/rl-island-0.jsonl 2>/dev/null
for f in args.txt rc.txt start_utc.txt end_utc.txt yeto_sha.txt cluster.txt selfcheck.txt selfcheck_probe.txt cleanup.out cleanup_rc.txt final_stop.txt judge.out judgment.json injection_not_reached recovery_failed startup_failed progress_stop.txt scan.md scan.json probe_after.txt probe_after.txt.attempts inwatch-arm.txt terminal_seen.txt autostop.out watchdog.out; do cp $R/$f $D/ 2>/dev/null; done
for f in gpu.txt compute-apps.txt inwatch.final.log inwatch.log router_samples.jsonl gpu_samples.jsonl dkill.log dctl.log sampler.out; do [ -s $R/pulled/$f ] && cp $R/pulled/$f $D/; done
for f in ps.txt launch.log launch.ts.log; do src=$R/$f; [ -f $src ] || src=$R/pulled/$f; [ -s $src ] && gzip -c $src > $D/$f.gz; done
[ -s $R.n2run.out ] && gzip -c $R.n2run.out > $D/n2run.out.gz
for f in $R/arm-*.txt; do [ -f $f ] && cp $f $D/; done
mkdir -p $D/diag; cp $R/pulled/diag/* $D/diag/ 2>/dev/null; [ -n "$(ls $D/diag)" ] || rmdir $D/diag
# elastic-state (final preferred; extracted so analyzers can read elastic-state/reconfig/journal.jsonl)
b=$R/pulled/elastic-state-final.b64; [ -s $b ] || b=$R/pulled/elastic-state.tgz.b64
[ -s $b ] && base64 -d $b 2>/dev/null | tar xz -C $D/ 2>/dev/null
find $D -name '*.lock' -delete 2>/dev/null
# sky provision logs of this cluster (best effort)
[ -d $R/home/sky_logs ] && (cd $R/home/sky_logs && tar czf $D/sky_logs.tgz . 2>/dev/null)
[ -d $R/runs/$1 ] && cp $R/runs/$1/launcher.log $R/runs/$1/meta.json $D/ 2>/dev/null
du -sh $D
