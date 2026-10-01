#!/bin/bash
# algo1a_watchdog.sh <seconds> <modal-app-name> [pidfile-of-local-process-to-stop]
# Independent reclaim for ALGO-1a runs. The wait uses a bash builtin (read -t), so no bare
# `sleep N` process exists and nobody else's pattern can match ours. It stops only the named app
# and the pid recorded in the given file (after checking its command line holds /tmp/algo1a/).
set -u
case "$2" in yeto-algo1a-*|algo1a-*) ;; *) echo "refusing: app $2 has no algo1a prefix"; exit 2;; esac
read -r -t "$1" <> <(:) || true
/tmp/modal-venv/bin/modal app stop -y "$2" >/dev/null 2>&1
if [ $# -ge 3 ] && [ -f "$3" ]; then
  p=$(cat "$3"); tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "/tmp/algo1a/" && kill "$p"
fi
echo "algo1a watchdog fired $(date -u +%FT%TZ) app=$2"
