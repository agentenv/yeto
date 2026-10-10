#!/bin/bash
B=/home/michael/work/s1-runs; P=$1; R=$B/$P
for i in $(seq 1 120); do [ $(ps -eLo user= | grep -c '^michael') -lt 2750 ] && break; sleep 15; done
exec 9>/home/michael/work/infra-drafts/GPU-LAUNCH.lock; flock 9
( exec 9>&-; SCHED=legacy PAUSE=0 SPEC=$B/s19-p1-specs/g3-knobs.json SPEC_SHA=bf55a45f7005b8fd83a6e82d2b1ae04c8ca2bcd1054fcc65640cc917aa3c60a0 STEPS=3 HARD=3600 bash $B/s19-p1-g3.sh $P > $B/$P.driver.log 2>&1 ) 9>&- &
for i in $(seq 1 180); do grep -q "modal-island" $R/launch.ts.log 2>/dev/null && break; grep -q "abort\|done $P" $B/$P.driver.log 2>/dev/null && break; sleep 5; done
exec 9>&-
wait
