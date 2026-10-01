#!/bin/bash
# usage: cleanup_run.sh <prefix>      (prefix = this round's unique prefix, e.g. infra-v2-b1-a4x-20261001-1)
# Four phases; touches ONLY resources whose name starts with "<prefix>-" (or Modal app yeto-<prefix>[-...]):
#   1. stop everything that could create resources: STOP flag, kill process groups (worker/launch/watchdog/puller/selfcheck/after-hook), wait until gone
#   2. release: sky down by cluster name, modal app stop
#   3. verify twice, CHECK_GAP_S (60) apart, at the authoritative places: Nebius API (project/region filtered by prefix), modal app list --json, sky status
#   4. any residue (or any unverifiable query) -> exit 2 (unverifiable: also 2, reason printed as UNVERIFIED); 0 = clean twice
# Test hooks (env): NEBIUS SKY MODAL (binaries), RUNS_BASE, CHECK_GAP_S, SETTLE_MAX_S, POLL_S, KILL_WAIT_S, CLEANUP_SKIP_PS=1 (no process phase)
set -u
P=${1:-}
[[ "$P" =~ ^[A-Za-z0-9][A-Za-z0-9-]{5,80}$ ]] || { echo "usage: cleanup_run.sh <prefix> (bad prefix '$P')" >&2; exit 64; }
# ---- single place for the cloud coordinates ----
NEBIUS_PROJECT_ID=project-e00eqrj3pr00622zrgdeyc     # eu-north1 project (e00 = eu-north1)
NEBIUS_REGION=eu-north1
NEBIUS=${NEBIUS:-/home/michael/.nebius/bin/nebius --profile michael}
SKY=${SKY:-/home/michael/work/gpu-head/venv/bin/sky}
MODAL=${MODAL:-/tmp/modal-venv/bin/modal}
RUNS_BASE=${RUNS_BASE:-/home/michael/work/gpu-b1-runs}
CHECK_GAP_S=${CHECK_GAP_S:-60}; SETTLE_MAX_S=${SETTLE_MAX_S:-300}; POLL_S=${POLL_S:-10}; KILL_WAIT_S=${KILL_WAIT_S:-30}
export HOME=${HOME:-/home/michael}
R=$RUNS_BASE/$P
log() { echo "[cleanup $(date -u +%T)] $*"; }

# ---------- process scan: only this prefix, anchored (never a prefix of a longer id) ----------
SELF_ANC=" $$ "; q=$$; while q=$(ps -o ppid= -p "$q" 2>/dev/null | tr -d ' '); [ -n "$q" ] && [ "$q" != 0 ] && [ "$q" != 1 ]; do SELF_ANC="$SELF_ANC$q "; done
procs() {  # prints "pid pgid" per matching process
  [ "${CLEANUP_SKIP_PS:-0}" = 1 ] && return 0
  ps -eo pid=,pgid=,args= | grep -E -- "(--cluster-prefix ${P}( |\$)|cli _worker ${P}( |\$)|/${P}(/| |\$)|[ /]${P}-l[0-9]+)" | grep -v -E -- 'grep -E|cleanup_run\.sh' | awk '{print $1, $2}' | while read -r pid pg; do
    case "$SELF_ANC" in *" $pid "*) continue;; esac; echo "$pid $pg"; done
}
phase1() {
  log "phase 1: STOP flag + kill process groups"
  mkdir -p "$R" 2>/dev/null
  mkdir -p "$R/runs/$P" 2>/dev/null   # <run_dir>/STOP, run_dir = $YETO_RUNS_DIR/<name>
  [ -d "$R/yeto" ] && (cd "$R/yeto" && HOME=$R/home YETO_RUNS_DIR=$R/runs PYTHONPATH=$R/yeto timeout 30 ${YETO_PY:-/home/michael/work/gpu-head/venv/bin/python} -m yeto.cli stop-run "$P" >/dev/null 2>&1)
  for d in "$R" "$R/runs" "$R"/runs/*/; do [ -d "$d" ] && touch "${d%/}/STOP" 2>/dev/null; done
  [ -f "$R/stop_cmd.sh" ] && bash "$R/stop_cmd.sh" >/dev/null 2>&1   # optional hook (e.g. yeto stop-run)
  local myg; myg=$(ps -o pgid= -p $$ | tr -d ' ')
  for sig in TERM KILL; do
    local list; list=$(procs)
    [ -z "$list" ] && break
    while read -r pid pg; do
      [ -z "$pid" ] && continue
      if [ "$pg" != "$myg" ] && ! [[ "$SELF_ANC" == *" $pg "* ]] ; then kill -$sig -- -"$pg" 2>/dev/null; else kill -$sig "$pid" 2>/dev/null; fi
    done <<< "$list"
    local t=0; while [ -n "$(procs)" ] && [ $t -lt $KILL_WAIT_S ]; do sleep 1; t=$((t+1)); done
  done
  local left; left=$(procs)
  if [ -n "$left" ]; then log "PROCESS_RESIDUE: $(echo "$left" | tr '\n' ';')"; return 1; fi
  log "phase 1 done: no process for $P left"; return 0
}

# ---------- queries (each prints residue lines; rc 0 = answered, 1 = could not answer) ----------
q_nebius() {
  local out; out=$(timeout 90 $NEBIUS compute instance list --parent-id "$NEBIUS_PROJECT_ID" --format json 2>&1) || { echo "UNVERIFIED nebius: $(echo "$out" | tail -c 200)"; return 1; }
  echo "$out" | python3 -c '
import json,sys
p,reg=sys.argv[1],sys.argv[2]
try: d=json.load(sys.stdin)
except Exception as e: print("UNVERIFIED nebius: bad json",e); sys.exit(1)
for i in d.get("items",[]):
    m=i.get("metadata",{}); n=m.get("name","")
    if n.startswith(p+"-") and (m.get("region") or reg)==reg:
        print("RESIDUE nebius instance",m.get("id"),n,(i.get("status") or {}).get("state"))
' "$P" "$NEBIUS_REGION"
}
q_modal() {
  local out; out=$(timeout 90 $MODAL app list --json 2>&1) || { echo "UNVERIFIED modal: $(echo "$out" | tail -c 200)"; return 1; }
  echo "$out" | python3 -c '
import json,sys
p=sys.argv[1]
try: d=json.load(sys.stdin)
except Exception as e: print("UNVERIFIED modal: bad json",e); sys.exit(1)
g=lambda a,*ks: next((str(a[k]) for k in ks if k in a), "")
for a in d:
    n=g(a,"Description","Name","description","name"); st=g(a,"State","state").lower(); tk=g(a,"Tasks","tasks") or "0"
    if (n in ("yeto-"+p,p) or n.startswith("yeto-"+p+"-") or n.startswith(p+"-")) and (st!="stopped" or tk not in ("0","")):
        print("RESIDUE modal app",g(a,"App ID","app_id"),n,st,"tasks="+tk)
' "$P"
}
q_sky() {
  local out; out=$(timeout 120 $SKY status 2>&1) || { echo "UNVERIFIED sky status: $(echo "$out" | tail -c 200)"; return 1; }
  echo "$out" | awk -v p="$P-" 'index($1,p)==1 {print "RESIDUE sky cluster", $1, $0}' | cut -c1-200
}
residue_all() {  # prints all residue lines; rc 1 if any query unverifiable
  local rc=0; q_nebius || rc=1; q_modal || rc=1; q_sky || rc=1; procs | sed 's/^/RESIDUE process /'; return $rc
}

phase2() {
  log "phase 2: release (sky down by cluster name, modal app stop)"
  local names; names=$( { [ -f "$R/cluster.txt" ] && cat "$R/cluster.txt"; q_sky | awk '{print $4}'; q_nebius | awk '/RESIDUE nebius/{n=$5; sub(/-[0-9a-f]+-head$/,"",n); print n}'; } 2>/dev/null | grep -E "^${P}-" | sort -u)
  for c in $names; do log "sky down $c"; timeout 600 $SKY down -y "$c" 2>&1 | tail -3; done
  q_modal | awk '/RESIDUE modal/{print $4}' | while read -r id; do [ -n "$id" ] && { log "modal app stop $id"; timeout 120 $MODAL app stop -y "$id" 2>&1 | tail -2; }; done
  local t=0
  while [ $t -lt $SETTLE_MAX_S ]; do [ -z "$(residue_all 2>/dev/null)" ] && break; sleep "$POLL_S"; t=$((t+POLL_S)); done
  log "phase 2 settle waited ${t}s"
}

phase3() {  # two checks CHECK_GAP_S apart; sets BAD=1 on any residue/unverified
  for n in 1 2; do
    local o; o=$(residue_all); local rc=$?
    if [ -n "$o" ] || [ $rc != 0 ]; then BAD=1; log "check $n: NOT CLEAN"; echo "$o" | sed 's/^/   /'; else log "check $n: clean (nebius $NEBIUS_PROJECT_ID/$NEBIUS_REGION, modal, sky, procs)"; fi
    [ $n = 1 ] && sleep "$CHECK_GAP_S"
  done
}

BAD=0
phase1 || BAD=1
phase2
phase3
if [ $BAD = 1 ]; then
  log "RESULT: RESIDUE or UNVERIFIED -> exit 2 (manual: sky down -y <cluster>; modal app stop -y <id>; nebius compute instance delete --id <id>; only ids printed above)"; exit 2
fi
log "RESULT: clean twice -> exit 0"; exit 0
