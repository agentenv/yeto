#!/usr/bin/env bash
# Retry s11d1chain.sh on provision/capacity failure (exit 5) up to N times, 30 min apart.
# Usage: s11d1retry.sh <run-prefix-base> [max_tries=3] [sleep_s=1800]
set -u
BASE=${1:?run prefix base, e.g. s11-d1-20261005}; MAX=${2:-3}; SLEEP=${3:-1800}
D=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 1 "$MAX"); do
  suffix=$(printf '%c' $(( 97 + i )))   # b, c, d ...
  P="${BASE}${suffix}"
  echo "[retry $(date -u +%FT%TZ)] try $i/$MAX -> $P"
  bash "$D/s11d1chain.sh" "$P"; rc=$?
  echo "[retry $(date -u +%FT%TZ)] $P rc=$rc"
  [ "$rc" -eq 5 ] || exit "$rc"          # only capacity/provision failures are retried
  [ "$i" -lt "$MAX" ] && { echo "[retry] capacity blocked, sleeping ${SLEEP}s"; sleep "$SLEEP"; }
done
echo "[retry $(date -u +%FT%TZ)] capacity blocked ${MAX} times, giving up"; exit 5
