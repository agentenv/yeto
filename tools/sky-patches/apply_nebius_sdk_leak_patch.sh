#!/usr/bin/env bash
# Apply / check / revert the SkyPilot 0.13.0 Nebius SDK leak backport.
#
# Problem: SkyPilot 0.13.0 (and 0.14.0, 0.14.1) caches nebius.sdk.SDK objects
# per request and never closes them. With nebius SDK >= 0.5 each SDK owns an
# event-loop thread, a gRPC channel pool and a token-renewal task, so the API
# server leaks about one thread and 3-4 MB per Nebius cluster per status-refresh
# cycle (60 s), plus one per request that touches Nebius.
# Upstream fix: skypilot-org/skypilot#11036 (commit d9d5bbcf05, master only,
# not in any release as of 2026-10-25). This patch backports it to 0.13.0.
#
# Usage:
#   apply_nebius_sdk_leak_patch.sh <venv-python> check    # report state, change nothing
#   apply_nebius_sdk_leak_patch.sh <venv-python> apply    # back up original, patch
#   apply_nebius_sdk_leak_patch.sh <venv-python> revert   # restore the backup
# The running API server keeps the old code until restart
# (`sky api stop && sky api start`, or `systemctl restart <unit>`).
set -euo pipefail

PY=${1:?venv python path}
ACTION=${2:-check}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH="$HERE/skypilot-0.13.0-nebius-sdk-leak.patch"
ORIG_SHA=0f60d0614526e4061e91695c95f1698c681b08866ce80c4536e49ab7cb2eac6b
NEW_SHA=2bf5c948417bda344c1d5ae43a661faa39c7fa4b853dc8b4188ee67afdbb2386

SITE=$("$PY" -I -c 'import os, sky; print(os.path.dirname(os.path.dirname(sky.__file__)))')
VER=$("$PY" -I -c 'import sky; print(sky.__version__)')
TARGET="$SITE/sky/adaptors/nebius.py"
BACKUP="$TARGET.orig-0.13.0"
CUR_SHA=$(sha256sum "$TARGET" | cut -d' ' -f1)

state=unknown
[ "$CUR_SHA" = "$ORIG_SHA" ] && state=unpatched
[ "$CUR_SHA" = "$NEW_SHA" ] && state=patched
echo "skypilot=$VER target=$TARGET state=$state"

case "$ACTION" in
  check)
    exit 0 ;;
  apply)
    if [ "$state" = patched ]; then echo "already patched"; exit 0; fi
    if [ "$VER" != 0.13.0 ] || [ "$state" != unpatched ]; then
      echo "refusing: expected skypilot 0.13.0 with the stock adaptor" >&2; exit 2
    fi
    cp -p "$TARGET" "$BACKUP"
    patch -p1 -d "$SITE" < "$PATCH"
    "$PY" -I -m py_compile "$TARGET"
    [ "$(sha256sum "$TARGET" | cut -d' ' -f1)" = "$NEW_SHA" ]
    echo "patched; backup at $BACKUP; restart the API server to load it" ;;
  revert)
    [ -f "$BACKUP" ] || { echo "no backup at $BACKUP" >&2; exit 2; }
    cp -p "$BACKUP" "$TARGET"
    echo "reverted; restart the API server to load it" ;;
  *)
    echo "unknown action: $ACTION" >&2; exit 2 ;;
esac
