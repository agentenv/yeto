#!/bin/bash
# in-container terminal-state probe (head node): every 5 s -> ~/yeto-rl/s1probe.log : utc, epoch, active ray nodes (island Ray on :6379), journal lines, last journal record
IP=$(hostname -I | awk '{print $1}'); J=~/yeto-rl/elastic-state/reconfig/journal.jsonl
while :; do
  act=$(timeout 20 ray status --address="$IP:6379" 2>/dev/null | awk '/^Active:/{a=1;next} /^(Idle|Pending|Recent failures):/{a=0} a && /node_/{n++} END{print n+0}')
  jl=$(wc -l < $J 2>/dev/null || echo 0); last=$(tail -1 $J 2>/dev/null | cut -c1-300)
  echo "$(date -u +%FT%TZ) $(date +%s) active_nodes=$act journal_lines=$jl last=$last" >> ~/yeto-rl/s1probe.log
  sleep 5
done
