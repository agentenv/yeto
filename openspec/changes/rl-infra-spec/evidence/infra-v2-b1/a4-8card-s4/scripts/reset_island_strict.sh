#!/bin/bash
# usage: reset_island_strict.sh <cluster name> <log file>   (chain8.sh RESET= hook for strict-avg chains)
# Host side first: the previous item's head syncer must be gone and the syncer port free (an orphan would make the next LocalSyncer fail to bind, or worse,
# the next island would PUSH into the old syncer's state).  Kills any gpu-b1-runs syncer tree still holding the port (scoped: command line under $B), waits for
# the port, logs, then runs the unchanged reset_island.sh on the island (which also removes ~/yeto-rl/elastic-state, ~/yeto-output: the learner keeps no other
# syncer-client state on disk -- learner.py writes only prompts.jsonl/eval-prompts.jsonl under ~/yeto-rl).  Exit = reset_island.sh's, or 1 if the port stays held.
CL=$1; LOG=$2; B=${BDIR:-/home/michael/work/gpu-b1-runs}; PORT=${SYNCER_PORT:-29400}; PKILL=${PKILL:-pkill}; PGREP=${PGREP:-pgrep}; RESET_INNER=${RESET_INNER:-$B/reset_island.sh}
{
echo "host syncer check $(date -u +%FT%TZ) port=$PORT"
pat="$B/.*/home/yeto-syncer"
$PGREP -af -- "$pat" | cut -c1-200
if $PGREP -f -- "$pat" >/dev/null 2>&1; then echo "orphan syncer from a previous item: killing"; $PKILL -TERM -f -- "$pat"; sleep 3; $PKILL -KILL -f -- "$pat" 2>/dev/null; fi
ok=0; for i in $(seq 1 12); do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then sleep 5; else ok=1; break; fi
done
[ $ok = 1 ] && echo "SYNCER_PORT_FREE $PORT" || { echo "SYNCER_PORT_HELD $PORT: $(ss -ltnp 2>/dev/null | grep ":$PORT " | cut -c1-200)"; }
} > $LOG 2>&1
[ $ok = 1 ] || { echo "reset rc=1 (host syncer port held)" >> $LOG; exit 1; }
$RESET_INNER $CL $LOG.island; rc=$?; cat $LOG.island >> $LOG 2>/dev/null; rm -f $LOG.island
exit $rc
