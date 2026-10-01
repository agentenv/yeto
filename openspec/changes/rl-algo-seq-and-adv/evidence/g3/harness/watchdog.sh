#!/bin/bash
# algo2a-g3 watchdog (independent of the launcher and of this agent's shell).
# usage: watchdog.sh <deadline-epoch> <app> <syncer-path> <logfile>
# Heartbeat every 60 s into <logfile> so a disappearance is timestamped.
trap '' HUP INT TERM   # ignore stray pattern kills' default signals; only SIGKILL stops it
dl=$1; app=$2; syn=$3; log=$4
echo "start pid $$ pgid $(ps -o pgid= $$) deadline $(date -u -d @$dl +%FT%TZ)" >> "$log"
while [ "$(date +%s)" -lt "$dl" ]; do echo "alive $(date -u +%FT%TZ)" >> "$log"; sleep 60; done
echo "deadline reached $(date -u +%FT%TZ): stopping $app" >> "$log"
/tmp/modal-venv/bin/modal app stop -y "$app" >> "$log" 2>&1; pkill -f "$syn" 2>/dev/null
echo "done $(date -u +%FT%TZ)" >> "$log"
