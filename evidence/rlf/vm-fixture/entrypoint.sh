#!/usr/bin/env bash
set -euo pipefail
: "${FLAG:?factory runtime must supply FLAG}"
umask 077
printf '%s' "$FLAG" > /tmp/rlf-target-flag
unset FLAG
exec socat TCP-LISTEN:31337,reuseaddr,fork EXEC:/opt/challenge,stderr

