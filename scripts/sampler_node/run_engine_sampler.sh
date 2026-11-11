#!/usr/bin/env bash
# Start the yeto engine sampler (one rollout island) on a sampler node, driving
# the SGLang server run_sglang.sh started.
#
#   scripts/sampler_node/run_engine_sampler.sh --spec SPEC.json [--yeto-dir DIR] [--python PY]
#       [--r2-env-file FILE] [--log FILE] [--foreground] [-- EXTRA ARGS...]
#
#   --spec         the sampler spec JSON (`python -m yeto.rl.engine.engine_sampler
#                  --spec`): engine endpoint, store, buffer, release follower,
#                  sampler id. Everything the island does comes from the spec.
#   --yeto-dir     the yeto checkout (default: this script's repository)
#   --python       interpreter with yeto's dependencies (default: $YETO_DIR/.venv/bin/python
#                  when it exists, else python3)
#   --r2-env-file  object store credentials exported into the sampler's environment
#                  (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL, AWS_REGION,
#                  R2_BUCKET); mode 0600 required; the values are never printed
#   --pip-install  install the Python packages the sampler needs and the interpreter
#                  lacks (safetensors, boto3) before starting; default: refuse to start
#
# Detached by default (nohup, pid in $LOG.pid); exit code 75 from the sampler is a
# reclaim exit (run it again on the next lease).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SPEC=""; YETO_DIR=$(cd "$HERE/../.." && pwd); PY=""; R2_ENV_FILE=""; LOG=""; DETACH=1; PIP_INSTALL=0
REQUIRED_MODULES="safetensors boto3"  # the release follower reads safetensors, the store client is boto3
while [ $# -gt 0 ]; do
  case "$1" in
    --spec) SPEC=$2; shift 2;;
    --yeto-dir) YETO_DIR=$2; shift 2;;
    --python) PY=$2; shift 2;;
    --r2-env-file) R2_ENV_FILE=$2; shift 2;;
    --log) LOG=$2; shift 2;;
    --foreground) DETACH=0; shift;;
    --pip-install) PIP_INSTALL=1; shift;;
    --) shift; break;;
    -h|--help) sed -n '2,22p' "$0"; exit 0;;
    *) echo "unknown option $1" >&2; exit 64;;
  esac
done
[ -n "$SPEC" ] || { echo "usage: $0 --spec SPEC.json [...]" >&2; exit 64; }
[ -f "$SPEC" ] || { echo "spec $SPEC does not exist" >&2; exit 66; }
[ -d "$YETO_DIR/yeto" ] || { echo "$YETO_DIR is not a yeto checkout" >&2; exit 66; }
if [ -z "$PY" ]; then
  if [ -x "$YETO_DIR/.venv/bin/python" ]; then PY="$YETO_DIR/.venv/bin/python"; else PY=python3; fi
fi
if [ -n "$R2_ENV_FILE" ]; then
  [ -f "$R2_ENV_FILE" ] || { echo "$R2_ENV_FILE does not exist" >&2; exit 66; }
  mode=$(stat -c %a "$R2_ENV_FILE" 2>/dev/null || stat -f %Lp "$R2_ENV_FILE")
  [ "$mode" = 600 ] || { echo "$R2_ENV_FILE must be mode 600 (is $mode)" >&2; exit 77; }
  set -a
  # shellcheck disable=SC1090
  . "$R2_ENV_FILE"
  set +a
fi
export PYTHONPATH="$YETO_DIR${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
missing=$("$PY" - $REQUIRED_MODULES <<'PY'
import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$missing" ]; then
  if [ "$PIP_INSTALL" = 1 ]; then
    echo "installing into $PY: $missing"
    # shellcheck disable=SC2086
    "$PY" -m pip install -q $missing
  else
    echo "$PY lacks: $missing (pass --pip-install, or install them into the interpreter)" >&2; exit 69
  fi
fi
"$PY" -c 'import yeto.rl.engine.engine_sampler' || { echo "$PY cannot import yeto from $YETO_DIR" >&2; exit 69; }
cmd=("$PY" -m yeto.rl.engine.engine_sampler --spec "$SPEC" "$@")
if [ "$DETACH" = 1 ]; then
  LOG=${LOG:-$(dirname "$SPEC")/engine-sampler.log}
  nohup "${cmd[@]}" >> "$LOG" 2>&1 &
  echo $! > "$LOG.pid"
  echo "started engine sampler (pid $(cat "$LOG.pid")), spec $SPEC, log $LOG"
else
  exec "${cmd[@]}"
fi
