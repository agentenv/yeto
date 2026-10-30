#!/bin/bash
# S29 spot: run one spot rollout island inside the sampler image on a Nebius preemptible VM
# (sky task tools/probes/s29_spot/sky_spot.yaml). ISLAND (0/1) and CARD (H100/B200) come from the task envs;
# BOAT_API_KEY and YETO_ISLAND_HMAC_KEY from sky secrets. The syncer is 127.0.0.1:29400 through the
# reverse tunnel npu1 opens to this VM (tools/probes/s29_spot/npu1_tunnel.sh); the run waits for it.
set -euo pipefail
OUT=/root/spot
DATA=/root/s29data
BASE=/root/models/Qwen3-0.6B
YETO=/root/yeto-root
[ -d "$OUT" ] && mv "$OUT" "$OUT-$(date -u +%H%M%S)"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/vm_run.log") 2>&1
echo "[s29-spot] island $ISLAND on $CARD $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"

PY=$(command -v python3)
if [ ! -f /opt/yeto/image/versions.lock.json ]; then  # the Harbor layer (image_layer.layer_commands)
  "$PY" - <<EOF > /tmp/harbor_layer.sh
import json, sys
sys.path.insert(0, "$YETO")
from yeto.rl.harness.harbor.image_layer import layer_commands, LAYER_DIR
spec = json.loads((LAYER_DIR / "versions.json").read_text())
print("set -e")
for c in layer_commands("miles", "$PY", uv_version=spec["uv"], harbor_requirement=spec["harbor"]["requirement"],
                        boat_sdk=spec["boat_sdk"]):
    print(c)
EOF
  mkdir -p /opt/yeto/harbor-layer /opt/yeto/image
  cp -r "$YETO/docker/harbor-layer/." /opt/yeto/harbor-layer/
  cp "$YETO/docker/image-assert/assert_image_versions.py" /opt/yeto/image/
  bash /tmp/harbor_layer.sh
fi

if [ ! -f "$BASE/config.json" ]; then
  "$PY" -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='$BASE')"
fi

PUBLIC_IP=$(curl -s --max-time 10 https://api.ipify.org)
SGL=$("$PY" -c "import sglang; print(sglang.__version__)")
"$PY" "$YETO/tools/probes/s29_spot/spot_plan.py" --island "$ISLAND" --public-ip "$PUBLIC_IP" \
  --engine-version "$SGL" --python "$PY" --card-model "$CARD" --base "$BASE" --data "$DATA" --out "$OUT" \
  --mode "${MODE:-tb21}" --groups-per-round "${GROUPS_PER_ROUND:-1}" > "$OUT/plan.json"
"$PY" -c "import json; d=json.load(open('$OUT/plan.json')); json.dump(d['spec'], open('$OUT/spec.json','w'), indent=1)"
eval "$("$PY" -c "import json,shlex; d=json.load(open('$OUT/plan.json')); print('\n'.join(f'export {k}={shlex.quote(v)}' for k,v in d['env'].items()))")"

umask 077; "$PY" -c "import secrets; print(secrets.token_urlsafe(24))" > /root/.yeto-records-proxy-secret; umask 022
export YETO_RECORDS_PROXY_SECRET_FILE=/root/.yeto-records-proxy-secret
export YETO_SPOT_RECLAIM_MARKER=/tmp/yeto-reclaim-requested
export PYTHONPATH=$YETO PYTHONUNBUFFERED=1 HOME=/root

echo "[s29-spot] waiting for the syncer tunnel on 127.0.0.1:29400"
for i in $(seq 1 720); do (exec 3<>/dev/tcp/127.0.0.1/29400) 2>/dev/null && break; sleep 5; done
(exec 3<>/dev/tcp/127.0.0.1/29400) 2>/dev/null || { echo "[s29-spot] no tunnel after 1 h"; exit 3; }
echo "[s29-spot] tunnel up; sampler start $(date -u +%FT%TZ)"
cd "$YETO"
"$PY" -m yeto.rl.spot_rollout.sampler --spec "$OUT/spec.json" 2>&1 | tee -a "$OUT/sampler.log"
echo "[s29-spot] sampler exited ${PIPESTATUS[0]} $(date -u +%FT%TZ)"
