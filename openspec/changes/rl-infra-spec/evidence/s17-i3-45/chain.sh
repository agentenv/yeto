#!/bin/bash
# S17 N7 I3: run the selected G-4.5 rows one at a time; stop the chain if the rb control fails.
B=/home/michael/work/s17-i3-runs; M=/tmp/modal-venv/bin/modal
for r in c3-rbold c3-f1 c3-f3 c3-f5 c3-f6a; do
  R=$B/s17-i3-$r
  for i in $(seq 1 60); do t=$(ps -L -u michael | wc -l); [ $t -lt 2800 ] && break; sleep 30; done
  [ $t -lt 2800 ] || { echo "$r: threads $t >= 2800 for 30 min, chain stopped" >> $B/chain.log; exit 3; }
  echo "| $(date -u +%FT%TZ) | S17 I3（N7）4.5 | **启动** \`s17-i3-$r\`（Modal 3×H100!，C3 Qwen3-1.7B，main 80e944b6，复核 S17-I3-45-PRELAUNCH-REVIEW.md）：最坏 30 min ≈\$5.9。开机线程 $t。 |" >> /home/michael/work/infra-drafts/gpu-spend.md
  echo "$r start $(date -u +%FT%TZ) threads $t" >> $B/chain.log
  (cd $R && YETO_E2_GPU_APPROVED=1 bash ./run.sh > ./run.out 2>&1 < /dev/null)
  kill $(cat $R/watchdog.pid 2>/dev/null) $(cat $R/puller.pid 2>/dev/null) 2>/dev/null
  HOME=$R/home timeout 120 $M app list --json > $R/applist-final.json 2>&1
  echo "$r end $(date -u +%FT%TZ) $(cat $R/rc.txt 2>/dev/null) abort=$(cat $R/DELIVER_ABORT 2>/dev/null)" >> $B/chain.log
  last=$(ls $R/es-snapshots/*.tgz 2>/dev/null | sort | tail -1); [ -n "$last" ] && mkdir -p $R/es-last && tar xzf $last -C $R/es-last
  if [ "$r" = c3-rb ] && ! grep -rqs '"phase": *"SUCCEEDED"' $R/pulled $R/es-last; then
    echo "rb control did not reach SUCCEEDED: chain stopped" >> $B/chain.log; exit 4
  fi
done
echo "chain done $(date -u +%FT%TZ)" >> $B/chain.log
