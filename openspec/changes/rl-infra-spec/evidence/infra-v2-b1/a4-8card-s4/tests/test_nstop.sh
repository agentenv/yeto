#!/bin/bash
# CPU-only regression test: nstop.sh -> cleanup_run.sh must complete (rc 0, "RESULT: clean twice") when the run dir path contains the prefix.
# (8xH100 smoke a8sm-20261001-1: `| tee <run dir>/cleanup.out` was killed by cleanup phase 1's prefix scan -> SIGPIPE, rc 141.)  usage: tests/test_nstop.sh [nstop.sh to test]
set -u
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/cleanup_run.sh ] || B=$B/scripts; NS=${1:-$B/nstop.sh}; T=$(mktemp -d); trap 'rm -rf $T' EXIT
P=infra-v2-test-20261001-9; mkdir -p $T/bin $T/runs/$P; cp $B/cleanup_run.sh $T/runs/; cp $NS $T/runs/nstop.sh
printf '#!/bin/bash\necho "{\\"items\\":[]}"\n' > $T/bin/nebius; printf '#!/bin/bash\necho Clusters\n' > $T/bin/sky; printf '#!/bin/bash\necho "[]"\n' > $T/bin/modal; chmod +x $T/bin/*
out=$(BDIR=$T/runs RUNS_BASE=$T/runs NEBIUS=$T/bin/nebius SKY=$T/bin/sky MODAL=$T/bin/modal CHECK_GAP_S=2 SETTLE_MAX_S=3 POLL_S=1 KILL_WAIT_S=3 bash $T/runs/nstop.sh $P 2>&1); rc=$?
if [ $rc = 0 ] && grep -q "RESULT: clean twice" $T/runs/$P/cleanup.out; then echo "PASS nstop -> cleanup_run completes (rc=0)"; else echo "FAIL nstop rc=$rc"; echo "$out" | tail -5; exit 1; fi
