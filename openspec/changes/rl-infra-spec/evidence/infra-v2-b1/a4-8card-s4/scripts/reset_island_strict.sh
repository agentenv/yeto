#!/bin/bash
# usage: reset_island_strict.sh <cluster name> <log file>   (chain8.sh RESET= hook for strict-avg chains)
# Host side first: the previous item's head syncer must be gone and the syncer port free (an orphan would make the next LocalSyncer fail to bind, or worse,
# the next island would PUSH into the old syncer's state).  Kills any gpu-b1-runs syncer tree still holding the port (scoped: command line under $B), waits for
# the port, logs, then runs the unchanged reset_island.sh on the island (which also removes ~/yeto-rl/elastic-state, ~/yeto-output: the learner keeps no other
# syncer-client state on disk -- learner.py writes only prompts.jsonl/eval-prompts.jsonl under ~/yeto-rl).  Exit = reset_island.sh's, or 1 if the port stays held.
# GATE (user ruling 2026-10-02, "r 用例逐个检查结果，失败后停止后续用例"): chain8 calls this between items, after the previous item's item_done (its judge has run by
# then: both a8go.sh and a8go_strict.sh touch item_done after judge_after).  For a gated previous item (GATED_CASES) whose judgment.json verdict is not PASS
# (or is missing) we write <chain>/ABORT (reason gate_<case>_<verdict>) and exit 1 ("island reset NOT CLEAN" -> chain8 stops and releases).  Race-free, and it
# covers the a8go.sh cases without touching a8go.sh.  The chain dir is the log file's directory; the previous item = last "ran" line of <chain>/items.jsonl.
CL=$1; LOG=$2; B=${BDIR:-/home/michael/work/gpu-b1-runs}; PORT=${SYNCER_PORT:-29400}; PKILL=${PKILL:-pkill}; PGREP=${PGREP:-pgrep}; RESET_INNER=${RESET_INNER:-$B/reset_island.sh}
GATED_CASES=${GATED_CASES:-"s0 d2 a4bc r6 r7 r5 r5c"}; CHAIN=$(cd "$(dirname "$LOG")" 2>/dev/null && pwd); CPX=${CL%-l0-*}
gate() {
  [ -s "$CHAIN/items.jsonl" ] || { echo "gate: no items.jsonl yet (first item)"; return 0; }
  local prev; prev=$(python3 -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ran=[r for r in rows if r.get('status')=='ran']
print(ran[-1]['item'] if ran else '')" "$CHAIN/items.jsonl" 2>/dev/null)
  [ -n "$prev" ] || { echo "gate: no previous ran item"; return 0; }
  case " $GATED_CASES " in *" $prev "*) ;; *) echo "gate: previous item $prev is not gated"; return 0;; esac
  local j="$CHAIN/items/$CPX-$prev/judgment.json" v
  v=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('verdict','NO_VERDICT'))" "$j" 2>/dev/null || echo NO_JUDGMENT)
  echo "gate: previous item $prev verdict=$v ($j)"
  [ "$v" = PASS ] && return 0
  echo "{\"item\":\"$CPX-$prev\",\"reason\":\"gate_${prev}_${v}\",\"ts\":\"$(date -u +%FT%TZ)\"}" > "$CHAIN/ABORT"
  echo "GATE_STOP: $prev verdict $v -> ABORT written, no further item"; return 1
}
if ! gate > $LOG 2>&1; then echo "reset rc=1 (gate: previous judged item not PASS)" >> $LOG; exit 1; fi
{
echo "host syncer check $(date -u +%FT%TZ) port=$PORT"
pat="$B/.*/home/yeto-syncer"
$PGREP -af -- "$pat" | cut -c1-200
if $PGREP -f -- "$pat" >/dev/null 2>&1; then echo "orphan syncer from a previous item: killing"; $PKILL -TERM -f -- "$pat"; sleep 3; $PKILL -KILL -f -- "$pat" 2>/dev/null; fi
ok=0; for i in $(seq 1 12); do
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}\$"; then sleep 5; else ok=1; break; fi
done
[ $ok = 1 ] && echo "SYNCER_PORT_FREE $PORT" || { echo "SYNCER_PORT_HELD $PORT: $(ss -ltnp 2>/dev/null | grep ":$PORT " | cut -c1-200)"; }
} >> $LOG 2>&1
[ $ok = 1 ] || { echo "reset rc=1 (host syncer port held)" >> $LOG; exit 1; }
$RESET_INNER $CL $LOG.island; rc=$?; cat $LOG.island >> $LOG 2>/dev/null; rm -f $LOG.island
exit $rc
