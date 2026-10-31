#!/bin/bash
# S29 120B spot island on one Nebius 8 x H100 preemptible VM (sky task sky_spot120.yaml).
# The base release r1-v0 (225 GB, byte-identical to the npu1 trainer's) comes from the Modal
# volume near the VM, never over the npu1 link. Later releases arrive as cut manifests
# (yeto.rl.spot_rollout.cut_manifest): the delta files on the Modal volume, the manifest digest
# over the control plane. The engine sampler reads the local store /root/store and writes its
# groups to sample-buffer/shards/spot-0.jsonl, which the mirror copies back to npu1.
# Re-runnable: a restarted VM keeps /root/store and only restarts the sampler.
set -uo pipefail
STORE=/root/store
OUT=/root/spot120
YETO=/root/yeto-root
VOL=yeto-s27-xc
mkdir -p "$OUT" "$STORE/policy-releases" "$STORE/sample-buffer"
exec > >(tee -a "$OUT/vm_run.log") 2>&1
echo "[s29-120] start $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"
PY=/opt/sglang/bin/python3; export PATH=/opt/sglang/bin:$PATH
"$PY" -c "import modal" 2>/dev/null || "$PY" -m pip install -q modal==1.2.6
if [ ! -f /opt/yeto/image/versions.lock.json ]; then
  "$PY" - <<PYEOF > /tmp/harbor_layer.sh
import json, sys
sys.path.insert(0, "$YETO")
from yeto.rl.harness.harbor.image_layer import layer_commands, LAYER_DIR
spec = json.loads((LAYER_DIR / "versions.json").read_text())
print("set -e")
for c in layer_commands("miles", "$PY", uv_version=spec["uv"], harbor_requirement=spec["harbor"]["requirement"], boat_sdk=spec["boat_sdk"]):
    print(c)
PYEOF
  mkdir -p /opt/yeto/harbor-layer /opt/yeto/image
  cp -r "$YETO/docker/harbor-layer/." /opt/yeto/harbor-layer/; cp "$YETO/docker/image-assert/assert_image_versions.py" /opt/yeto/image/
  bash /tmp/harbor_layer.sh
fi
R1=$STORE/policy-releases/r1-v0
if [ ! -f "$R1/.complete" ]; then
  t0=$(date +%s)
  modal volume get --force $VOL /xc120/policy-releases/r1-v0 "$STORE/policy-releases/" && touch "$R1/.complete"
  echo "[s29-120] r1-v0 from the Modal volume in $(( $(date +%s) - t0 )) s ($(du -sh "$R1" | cut -f1))"
fi
[ -f "$STORE/policy-releases/LATEST.json" ] || cp "$R1/RELEASE.json" "$STORE/policy-releases/LATEST.json"
# PROMPTS_VOL: a prompt file on the volume (default the TB2.1 rows of xc120); S29: the 30 tasks
# whose earlier groups had mixed rewards, so GRPO advantages are not all zero
if [ ! -s "$STORE/prompts/agentic.jsonl" ]; then
  mkdir -p "$STORE/prompts"; modal volume get --force $VOL "${PROMPTS_VOL:-/xc120/prompts/agentic.jsonl}" "$STORE/prompts/agentic.jsonl"
fi
[ -d "$STORE/harbor-tasks" ] || modal volume get --force $VOL /xc120/harbor-tasks "$STORE/"
PUBLIC_IP=$(curl -s --max-time 10 https://api.ipify.org)
SGL=$("$PY" -c "import sglang; print(sglang.__version__)")
"$PY" - <<PYEOF
import json, sys
sys.path.insert(0, "$YETO/tools/probes/s28_g120")
import os; os.environ["YETO_G120_GPU"] = os.environ.get("CARD", "H100"); os.environ["PLAN_ONLY"] = "0"
src = open("$YETO/tools/probes/s28_g120/agentic_sampler_app.py").read().split("\nif os.environ.get(\"PLAN_ONLY\")")[0]
ns = {"__name__": "s29plan", "__file__": "$YETO/tools/probes/s28_g120/agentic_sampler_app.py"}
exec(compile(src, "agentic_sampler_app", "exec"), ns)
cfg = {**ns["DEFAULTS"], "store": "$STORE", "base_dir": "$R1", "groups_per_round": int(os.environ.get("GROUPS", "4")), "samples_per_group": 4,
       "max_in_flight": int(os.environ.get("MAX_IN_FLIGHT", "16")), "max_backlog_groups": 8, "num_rollouts": 64, "mem_fraction": 0.85}
sp = ns["spec"]("$STORE", "$OUT", engine_version="$SGL", cfg=cfg)
sp["sampler_id"] = "spot-0"; sp["buffer_shards"] = ["spot-0"]
json.dump(sp, open("$OUT/spec.json", "w"), indent=1)
env = ns["harbor_env"](cfg)
env.update({"YETO_BOAT_RUN_ID": os.environ.get("RUN_ID", ""), "YETO_BOAT_ISLAND": "spot-0",
            "YETO_RECORDS_PROXY_PUBLIC_BASE": "http://$PUBLIC_IP:31600", "YETO_BOAT_STARTS_PER_HOUR": "30"})
open("$OUT/env.sh", "w").write("".join(f"export {k}='{v}'\n" for k, v in env.items()))
PYEOF
. "$OUT/env.sh"
umask 077; "$PY" -c "import secrets; print(secrets.token_urlsafe(24))" > /root/.yeto-records-proxy-secret; umask 022
export YETO_RECORDS_PROXY_SECRET_FILE=/root/.yeto-records-proxy-secret YETO_SPOT_RECLAIM_MARKER=/tmp/yeto-reclaim-requested
export YETO_RL_RELEASE_MIRROR=/root/release-mirror PYTHONPATH=$YETO PYTHONUNBUFFERED=1 HOME=/root
echo "[s29-120] sampler start $(date -u +%FT%TZ)"
cd "$YETO"; "$PY" -m yeto.rl.engine.engine_sampler --spec "$OUT/spec.json" >> "$OUT/sampler.log" 2>&1
echo "[s29-120] sampler exited $? $(date -u +%FT%TZ)"
