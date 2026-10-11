#!/usr/bin/env bash
set -euo pipefail
: "${FLAG:?factory runtime must supply FLAG}"
umask 077
printf '%s' "$FLAG" > /flag.txt
unset FLAG
exec socat TCP-LISTEN:31337,reuseaddr,fork EXEC:/opt/service.sh,stderr
