#!/usr/bin/env bash
# Poll a National Compute Public Research ticket until it is active, then write
# the node record and pin the node's SSH host key.
#
#   scripts/sampler_node/nc_lease.sh --account N [--out node.json] [--interval 60] [--max-polls 240]
#                                    [--known-hosts ~/.ssh/known_hosts] [--nc-query CMD]
#
# `--nc-query` is the status command of the National Compute tool (default
# `nc-query`, i.e. `nc-query status --account N`), which prints JSON with a
# `ticket` object (state, position, estimated_grant_at, node.public_ip,
# node.ssh_user, node.host_key, granted_at, lease_until, rate_ucr_per_gpu_hour,
# gpus). No account, token or site value lives here: the tool reads its own
# account file; this script only passes the account index through.
#
# Exit: 0 active (node.json written, host key pinned), 1 the ticket ended or
# there is none, 2 still waiting after --max-polls. Requires jq.
set -uo pipefail
ACCOUNT=""; OUT="node.json"; INTERVAL=60; MAX_POLLS=240; KNOWN_HOSTS="${HOME}/.ssh/known_hosts"; NC_QUERY="${NC_QUERY:-nc-query}"
while [ $# -gt 0 ]; do
  case "$1" in
    --account) ACCOUNT=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --interval) INTERVAL=$2; shift 2;;
    --max-polls) MAX_POLLS=$2; shift 2;;
    --known-hosts) KNOWN_HOSTS=$2; shift 2;;
    --nc-query) NC_QUERY=$2; shift 2;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 64;;
  esac
done
[ -n "$ACCOUNT" ] || { echo "usage: $0 --account N [--out node.json]" >&2; exit 64; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 69; }

state=none
for _ in $(seq 1 "$MAX_POLLS"); do
  st=$($NC_QUERY status --account "$ACCOUNT" 2>&1) || true
  state=$(jq -r '.ticket.state // "none"' <<<"$st" 2>/dev/null || echo parse-error)
  printf '%s state=%s pos=%s eta=%s\n' "$(date -u +%FT%TZ)" "$state" \
    "$(jq -r '.ticket.position // "-"' <<<"$st" 2>/dev/null)" \
    "$(jq -r '.ticket.estimated_grant_at // "-"' <<<"$st" 2>/dev/null)"
  if [ "$state" = active ]; then
    jq '{id: .ticket.id, state: .ticket.state, site: .ticket.site_name, ip: .ticket.node.public_ip,
         user: .ticket.node.ssh_user, host_key: .ticket.node.host_key, granted_at: .ticket.granted_at,
         lease_until: .ticket.lease_until, rate_ucr_per_gpu_hour: .ticket.rate_ucr_per_gpu_hour,
         gpus: .ticket.gpus}' <<<"$st" > "$OUT"
    ip=$(jq -r .ip "$OUT")
    key=$(jq -r '.host_key // empty' "$OUT")
    if [ -n "$ip" ] && [ -n "$key" ]; then
      mkdir -p "$(dirname "$KNOWN_HOSTS")"
      ssh-keygen -R "$ip" -f "$KNOWN_HOSTS" >/dev/null 2>&1 || true
      echo "$ip $key" >> "$KNOWN_HOSTS"
    fi
    cat "$OUT"; echo ACTIVE; exit 0
  fi
  case "$state" in
    ended|releasing|none|parse-error) echo "$st" | head -40; exit 1;;
  esac
  sleep "$INTERVAL"
done
echo "TIMEOUT still $state"; exit 2
