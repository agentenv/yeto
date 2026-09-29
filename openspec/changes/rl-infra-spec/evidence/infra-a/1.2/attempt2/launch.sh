#!/bin/bash
R=/home/michael/work/infra-a-gpu/b12
export HOME=$R/home YETO_RUNS_DIR=$R/runs SYNCER_PUBLIC_IP=185.189.44.160 PYTHONPATH=/home/michael/work/infra-a-gpu/b12/yeto
eval "$(/usr/bin/python3 - <<'PY'
import base64, json, shlex
a = json.load(open("/home/michael/.docker/config.json"))["auths"]["ghcr.io"]["auth"]
u, t = base64.b64decode(a).decode().split(":", 1)
print(f"export SKYPILOT_DOCKER_USERNAME={shlex.quote(u)} SKYPILOT_DOCKER_PASSWORD={shlex.quote(t)} SKYPILOT_DOCKER_SERVER=ghcr.io")
PY
)"
cd /home/michael/work/infra-a-gpu/b12/yeto
date -u +%FT%TZ > $R/start_utc.txt
timeout 5400 /home/michael/work/gpu-head/venv/bin/python $R/run_local_head.py $R/args.txt
echo "rc=$?" > $R/rc.txt
date -u +%FT%TZ > $R/end_utc.txt
