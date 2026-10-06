#!/bin/bash
# usage: syncer_host_clean.sh <run dir> [port]   -- kill the host-side yeto-syncer tree that belongs to THIS run dir (command line contains "<run dir>/home/yeto-syncer"),
# then report whether the syncer port is free.  Idempotent; a run without a host syncer (no <run dir>/syncer_host.txt) is a no-op (exit 0).
# Exit 0 = nothing of this run left and (if the run had a syncer) the port is not held by any gpu-b1-runs syncer; 1 = a syncer of this run survived SIGKILL or the port is still held.
R=${1:?run dir}; PORT=${2:-$(cut -d: -f2 $R/syncer_host.txt 2>/dev/null)}; PORT=${PORT:-29400}; PKILL=${PKILL:-pkill}; PGREP=${PGREP:-pgrep}
[ -f $R/syncer_host.txt ] || { echo "no host syncer for $R"; exit 0; }
pat="$R/home/yeto-syncer"
echo "syncer clean $(date -u +%FT%TZ) run=$R port=$PORT before: $($PGREP -af -- "$pat" | cut -c1-160 | tr '\n' ';')"
for sig in TERM KILL; do
  $PGREP -f -- "$pat" >/dev/null 2>&1 || break
  $PKILL -$sig -f -- "$pat" 2>/dev/null
  for i in 1 2 3 4 5 6 7 8 9 10; do $PGREP -f -- "$pat" >/dev/null 2>&1 || break; sleep 1; done
done
left=$($PGREP -af -- "$pat" | cut -c1-160 | tr '\n' ';')
held=$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" 'index($4, p) == length($4) - length(p) + 1' | grep -o 'yeto-syncer' | head -1)
echo "after: left=[${left}] port_${PORT}_held_by=[${held:-none}]"
[ -z "$left" ] && [ -z "$held" ] && exit 0
exit 1
