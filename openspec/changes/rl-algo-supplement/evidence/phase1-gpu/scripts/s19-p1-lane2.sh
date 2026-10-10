#!/bin/bash
# sequential, after pid $1 exits; each item "case|letter|ENV assignments"; retries preflight refusals with next letter
B=/home/michael/work/s1-runs; W=$1; shift
while kill -0 $W 2>/dev/null; do sleep 20; done
for item in "$@"; do IFS='|' read -r c letters envs <<< "$item"
  for L in $(echo $letters | fold -w1); do P=s19-p1-$c-20261010$L
    for i in $(seq 1 120); do [ $(ps -eLo user= | grep -c '^michael') -lt 2700 ] && break; sleep 15; done
    echo "$(date -u +%FT%TZ) start $P [$envs]" >> $B/s19-p1-lane2b.log
    bash -c "export CASE=$c $envs; exec bash $B/s19-p1-run.sh $P" > $B/$P.driver.log 2>&1
    echo "$(date -u +%FT%TZ) end $P $(tail -1 $B/$P.driver.log)" >> $B/s19-p1-lane2b.log
    grep -q "modal-island 0\]" $B/$P/launch.ts.log 2>/dev/null && break
  done
done
