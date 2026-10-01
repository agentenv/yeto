#!/bin/bash
# CPU-only test of cleanup_run.sh with stub nebius/sky/modal (no cloud calls). usage: tests/test_cleanup.sh
set -u
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/cleanup_run.sh ] || B=$B/scripts; T=$(mktemp -d); trap 'rm -rf $T; pkill -f "STUBWORKER-$$" 2>/dev/null' EXIT
P=infra-v2-test-20261001-1; fail=0
mkdir -p $T/bin $T/st $T/runs/$P
cat > $T/bin/nebius <<'S'
#!/bin/bash
# stub: instances from $ST/inst ("id name"), plus unrelated ones; appear file adds a relaunch
echo "nebius $*" >> $ST/calls.log
[ -f $ST/appear ] && grep -q . $ST/appear && cat $ST/appear >> $ST/inst && : > $ST/appear
python3 - $ST/inst <<'PY'
import json,sys
items=[{"metadata":{"id":"computeinstance-unrelated","name":"rlf-h200-pilot-03"},"status":{"state":"RUNNING"}},
       {"metadata":{"id":"computeinstance-longer","name":"%s0-l0-eu-north1-aa-head"%__import__("os").environ["PFX"]},"status":{"state":"RUNNING"}}]
for l in open(sys.argv[1]):
    if l.strip():
        i,n=l.split(); items.append({"metadata":{"id":i,"name":n},"status":{"state":"RUNNING"}})
print(json.dumps({"items":items}))
PY
S
cat > $T/bin/sky <<'S'
#!/bin/bash
echo "sky $*" >> $ST/calls.log
case "$1" in
 status) echo "Clusters"; echo "NAME STATUS"; cat $ST/sky 2>/dev/null; echo "unrelated-cluster UP";;
 down) c=${3:-$2}; [ -f $ST/stuck ] && exit 0; grep -v "^$c " $ST/sky > $ST/sky.n; mv $ST/sky.n $ST/sky; grep -v " $c-" $ST/inst > $ST/inst.n; mv $ST/inst.n $ST/inst; [ -f $ST/relaunch ] && echo "computeinstance-reborn $c-zz-head" >> $ST/appear; echo done;;
esac
S
cat > $T/bin/modal <<'S'
#!/bin/bash
echo "modal $*" >> $ST/calls.log
case "$1 $2" in
 "app list") cat $ST/modal.json;;
 "app stop") python3 - $ST/modal.json "${@: -1}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for a in d:
    if a["App ID"]==sys.argv[2]: a["State"]="stopped"; a["Tasks"]="0"
json.dump(d,open(sys.argv[1],"w"))
PY
 echo stopped;;
esac
S
chmod +x $T/bin/*
run() { # name expected_rc setup_fn
  rm -f $T/st/*; : > $T/st/inst; : > $T/st/sky; echo '[{"App ID":"ap-other","Description":"yeto-other","State":"deployed","Tasks":"1"}]' > $T/st/modal.json
  $3
  out=$(ST=$T/st PFX=$P NEBIUS=$T/bin/nebius SKY=$T/bin/sky MODAL=$T/bin/modal RUNS_BASE=$T/runs CHECK_GAP_S=6 SETTLE_MAX_S=8 POLL_S=1 KILL_WAIT_S=5 PATH=$T/bin:$PATH bash $B/cleanup_run.sh $P 2>&1); rc=$?
  if [ $rc = $2 ]; then echo "PASS $1 (rc=$rc)"; else echo "FAIL $1 rc=$rc want $2"; echo "$out"; fail=1; fi
  # never touch others
  if grep -E "rlf-h200|unrelated|other|${P}0" $T/st/calls.log | grep -E "down|stop"; then echo "FAIL $1: touched foreign resource"; fail=1; fi
  echo "$out" > $T/out.$1
}
s_clean() { setsid bash -c "exec -a 'python -m yeto.cli launch --cluster-prefix $P' sleep 300" >/dev/null 2>&1 & setsid bash -c "exec -a 'python -m yeto.cli launch --cluster-prefix ${P}0 STUBWORKER-$$' sleep 300" >/dev/null 2>&1 &
  echo "computeinstance-a $P-l0-eu-north1-ab-head" >> $T/st/inst; echo "$P-l0-eu-north1 UP" >> $T/st/sky
  python3 - $T/st/modal.json $P <<'PY'
import json,sys; d=json.load(open(sys.argv[1])); d.append({"App ID":"ap-mine","Description":"yeto-"+sys.argv[2],"State":"deployed","Tasks":"2"}); json.dump(d,open(sys.argv[1],"w"))
PY
  echo $P-l0-eu-north1 > $T/runs/$P/cluster.txt; sleep 1; }
run teardown_ok 0 s_clean
pgrep -f "cluster-prefix $P sleep|cluster-prefix $P\$|cluster-prefix $P 300" >/dev/null && { echo "FAIL worker survived"; fail=1; } || echo "PASS worker killed"
pgrep -f "cluster-prefix ${P}0 " >/dev/null && echo "PASS longer-prefix process untouched" || { echo "FAIL longer-prefix process was killed"; fail=1; }
pkill -f "cluster-prefix ${P}0 " ; [ -f $T/runs/$P/STOP ] && echo "PASS STOP flag written" || { echo "FAIL no STOP"; fail=1; }
s_none() { :; }; run no_residue 0 s_none
s_resid() { touch $T/st/stuck; echo "computeinstance-a $P-l0-eu-north1-ab-head" >> $T/st/inst; echo "$P-l0-eu-north1 UP" >> $T/st/sky; echo $P-l0-eu-north1 > $T/runs/$P/cluster.txt; }
run residue_stuck 2 s_resid
s_relaunch() { touch $T/st/relaunch; echo "computeinstance-a $P-l0-eu-north1-ab-head" >> $T/st/inst; echo "$P-l0-eu-north1 UP" >> $T/st/sky; echo $P-l0-eu-north1 > $T/runs/$P/cluster.txt
  ( sleep 3; echo "computeinstance-late $P-l0-eu-north1-zz-head" > $T/st/appear ) & }
run relaunched_after_down 2 s_relaunch
grep -q "check 2: NOT CLEAN" $T/out.relaunched_after_down && echo "PASS relaunch caught by the second check" || { echo "FAIL relaunch not caught at check 2"; fail=1; }
[ $fail = 0 ] && echo "ALL PASS" || { echo "SOME FAILED"; exit 1; }
