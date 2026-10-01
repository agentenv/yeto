#!/bin/bash
# usage: diag_pull.sh <run dir>   -- extra error-collection loop (runs until rc.txt/startup_failed): every ~45 s one ssh snapshot into <run>/pulled/diag/
#   snap.txt   : date, full nvidia-smi, process table (pid ppid etime stat args), memory/disk, threads
#   dmesg.txt  : dmesg grep OOM/Xid/killed/segfault (best effort; "unavailable" if the container cannot read it)
#   pyspy-*.txt: py-spy dump of learner / sglang / scheduler pids when the tape has not grown for >= STUCK_S (default 240 s); py-spy installed on demand (once) if absent
#   raylogs.tgz.b64 : /tmp/ray/session_latest/logs (every ~3 min, capped 25 MB) -- learner/driver/engine stderr that Ray keeps on the island
# Everything is best effort; failures are recorded in diag/diag.err, never fatal.
R=$1; B=/home/michael/work/gpu-b1-runs; D=$R/pulled/diag; mkdir -p $D; CL=$(cat $R/cluster.txt)
export HOME=/home/michael; S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $CL"; STUCK_S=${STUCK_S:-240}
until [ -s $R/pulled/gpu.txt ]; do [ -f $R/rc.txt ] || [ -f $R/startup_failed ] && exit 0; sleep 10; done
n=0; lastsz=x; lastt=$(date +%s); spied=0
while [ ! -f $R/rc.txt ] && [ ! -f $R/startup_failed ]; do
  { echo "=== $(date -u +%FT%TZ)"; timeout 60 $S 'nvidia-smi; echo ---ps; ps -eo pid,ppid,etime,stat,args --no-headers | cut -c1-220; echo ---mem; free -m; df -h / /tmp 2>/dev/null | head -4; echo ---threads; ps -eLf | wc -l' ; } >> $D/snap.txt 2>> $D/diag.err
  { echo "=== $(date -u +%FT%TZ)"; timeout 60 $S '(dmesg -T 2>&1 || dmesg 2>&1) | grep -iE "oom|out of memory|xid|killed process|segfault|nvrm" | tail -50; echo "dmesg-first-line: $(dmesg 2>&1 | head -1 | cut -c1-120)"'; } > $D/.dm 2>> $D/diag.err; [ "$(wc -l < $D/.dm)" -gt 1 ] && mv $D/.dm $D/dmesg.txt   # keep the last non-failed answer
  if [ $((n % 4)) = 0 ]; then
    timeout 120 $S 'cd /tmp/ray/session_latest/logs 2>/dev/null && tar czf - . 2>/dev/null | head -c 26214400 | base64 -w0' > $D/.rl 2>> $D/diag.err && [ -s $D/.rl ] && mv $D/.rl $D/raylogs.tgz.b64
  fi
  sz=$(stat -c %s $R/pulled/rl-island-0.jsonl 2>/dev/null || echo 0); now=$(date +%s)
  if [ "$sz" != "$lastsz" ]; then lastsz=$sz; lastt=$now; spied=0; fi
  if [ $((now - lastt)) -ge $STUCK_S ] && [ $spied -lt 2 ]; then
    spied=$((spied+1)); echo "stuck: no new tape events for $((now - lastt)) s at $(date -u +%FT%TZ)" >> $D/stuck.txt
    timeout 200 $S 'command -v py-spy >/dev/null || (python3 -m pip install -q py-spy 2>&1 | tail -2); command -v py-spy || ls ~/.local/bin/py-spy 2>&1; PS=$(command -v py-spy || echo ~/.local/bin/py-spy); for p in $(pgrep -f "yeto.rl.learner|sglang|scheduler|ray::" | head -12); do echo "##### pid $p $(tr "\0" " " < /proc/$p/cmdline | cut -c1-160)"; timeout 20 $PS dump --pid $p 2>&1 | head -80; done' > $D/pyspy-$spied.txt 2>&1
  fi
  n=$((n+1)); sleep 45
done
echo "diag loop ended $(date -u +%FT%TZ) cycles=$n" >> $D/diag.err
