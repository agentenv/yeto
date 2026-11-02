#!/bin/bash
# S29 120B mirror on the dev machine (npu1 <-> dev machine <-> Nebius VM / Modal volume, no proxy).
#
#   cut120_mirror.sh samples          # once: VM sample-buffer -> npu1 trainer store (+ read heartbeat back)
#   cut120_mirror.sh read-position    # only the npu1 read heartbeat -> VM (S30 D29)
#   cut120_mirror.sh release r2-v2    # npu1 release -> Modal volume; manifest digest -> VM; VM places it
#   cut120_mirror.sh place r2-v2      # only the VM half (a VM that came back after the upload)
#
# S30 P0-b: DELIVERY=s3 (default) keeps the dev machine out of the data path: npu1 uploads the
# release to S3 through its proxy (tools/probes/s29_spot/s3_delivery.py publish) and the VM pulls it
# (s3_delivery.py pull, manifest + digest from S3, cut_manifest place). The steady state is the two
# resident loops (npu1 `publish-loop`, VM `pull-loop`, started by vm120_run.sh with S3_DELIVERY=1);
# this script is the manual / catch-up path. DELIVERY=mac: the old path below (scp + Modal volume).
#
# Bytes: the release delta (~13 GB) goes npu1 -> dev machine (scp) -> Modal volume (modal volume put);
# the VM pulls it from the volume in its own region. Only the manifest and its sha256 travel to the
# VM directly (cut_manifest.py checks every file before LATEST.json names the release).
set -uo pipefail
VM_IP=${VM_IP:?VM_IP=<public ip of the 8 x H100 VM>}
SKY_KEY=${SKY_KEY:-$(ls ~/.sky/clients/*/ssh/sky-key | head -1)}
# a command in the VM's sky container (its sshd does not come back after a Nebius stop/start)
vmsh() { ssh -i "$SKY_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
  "ubuntu@$VM_IP" "sudo docker exec -i sky_container bash -c $(printf %q "$1")"; }
NPU1=${NPU1:-npu1}
STORE=${STORE:-/data/work/s28-g120/s29r120}
VOL=${VOL:-yeto-s27-xc}
VOLDIR=${VOLDIR:-/s29r120}
LOCAL=${LOCAL:-/tmp/s29r/cut120}
NPU1_PY="cd ${NPU1_YETO:-/data/work/s29r/yetosrc} && PYTHONPATH=. python3"
DELIVERY=${DELIVERY:-s3}
# a checkout with tools/probes/s29_spot/s3_delivery.py (deploy: git archive <sha> | ssh npu1 tar x -C ...)
NPU1_SRC=${NPU1_SRC:?set NPU1_SRC}
S3_PREFIX=${S3_PREFIX:-s29r120}
S3_TRANSFERS=${S3_TRANSFERS:-4}
NPU1_S3_ENV="S3_PREFIX=$S3_PREFIX S3_TRANSFERS=$S3_TRANSFERS S3_BUCKET=${S3_BUCKET:?set S3_BUCKET} S3_PROXY=${S3_PROXY:-} RCLONE=${NPU1_RCLONE:-rclone}"
ISLAND=${ISLAND:-spot-0}  # S30 AB: one mirror per island (its shard and temp names)
mkdir -p "$LOCAL"

samples() {
  # the VM writes sample-buffer/shards/spot-0.jsonl + payloads/. A restarted or replaced VM starts its
  # shard over, so the trainer copy is append-only: payloads are copied, shard lines whose sample_id
  # the trainer copy lacks are appended (a rewritten shard never hides new groups behind old offsets)
  vmsh "cd /root/store && tar cf - sample-buffer/shards sample-buffer/payloads" > "$LOCAL/sb-$ISLAND.tar" || return 1
  scp -q "$LOCAL/sb-$ISLAND.tar" "$NPU1:/tmp/s29r120-sb-$ISLAND.tar" &&
  ssh "$NPU1" "set -e; rm -rf /tmp/s29r120-sb-$ISLAND && mkdir -p /tmp/s29r120-sb-$ISLAND $STORE/sample-buffer/shards $STORE/sample-buffer/payloads && \
    tar xf /tmp/s29r120-sb-$ISLAND.tar -C /tmp/s29r120-sb-$ISLAND && cp -n /tmp/s29r120-sb-$ISLAND/sample-buffer/payloads/* $STORE/sample-buffer/payloads/ 2>/dev/null; \
    python3 -c \"
import json, sys
src, dst = '/tmp/s29r120-sb-$ISLAND/sample-buffer/shards/$ISLAND.jsonl', '$STORE/sample-buffer/shards/$ISLAND.jsonl'
try: have = {json.loads(l)['sample_id'] for l in open(dst)}
except FileNotFoundError: have = set()
new = [l for l in open(src) if l.strip() and json.loads(l)['sample_id'] not in have]
open(dst, 'a').writelines(new)
print(len(have) + len(new), 'lines', len(new), 'new')
\"" || return 1
  read_position
}

read_position() {
  # S30 D29: the trainer's read heartbeat back to the VM, so the sampler's backlog follows what the
  # trainer read, not only the read position of its last (sparse / slow) release. Missing on npu1
  # (a trainer before D29, nothing read yet): skipped, the sampler keeps the release position.
  scp -q "$NPU1:$STORE/sample-buffer/read-position.json" "$LOCAL/read-position.json" 2>/dev/null ||
    { echo "[mirror] no read-position.json on npu1 (skipped)"; return 0; }
  vmsh "mkdir -p /root/store/sample-buffer && cat > /root/store/sample-buffer/read-position.json.tmp && \
    mv /root/store/sample-buffer/read-position.json.tmp /root/store/sample-buffer/read-position.json" \
    < "$LOCAL/read-position.json" && echo "[mirror] read-position -> VM: $(cat "$LOCAL/read-position.json")"
}

release_s3() {
  local rel=$1 t0
  t0=$(date +%s)
  ssh "$NPU1" "cd $NPU1_SRC && $NPU1_S3_ENV python3 tools/probes/s29_spot/s3_delivery.py publish $rel --store $STORE" || return 1
  echo "[mirror] $rel npu1 -> S3 $(( $(date +%s) - t0 )) s"
  place_s3 "$rel"
}

place_s3() {
  local rel=$1 t0
  t0=$(date +%s)
  vmsh "cd /root/yeto-root && S3_PREFIX=$S3_PREFIX S3_TRANSFERS=${S3_PULL_TRANSFERS:-16} RCLONE=/root/bin/rclone \
    /opt/sglang/bin/python3 tools/probes/s29_spot/s3_delivery.py pull $rel --store /root/store" || return 1
  echo "[mirror] $rel S3 -> VM placed $(( $(date +%s) - t0 )) s"
}

release() {
  [ "$DELIVERY" = s3 ] && { release_s3 "$@"; return; }
  local rel=$1 t0
  t0=$(date +%s)
  ssh "$NPU1" "$NPU1_PY -m yeto.rl.spot_rollout.cut_manifest make --store $STORE --release $rel --out /tmp/s29r120-$rel.manifest.json" \
    > "$LOCAL/$rel.make.txt" || return 1
  local digest; digest=$(awk '{print $1}' "$LOCAL/$rel.make.txt" | tail -1)
  scp -q "$NPU1:/tmp/s29r120-$rel.manifest.json" "$LOCAL/$rel.manifest.json"
  rm -rf "${LOCAL:?}/$rel"; mkdir -p "$LOCAL/$rel"
  # one scp per file, 8 at a time: the npu1 uplink is per-stream limited (~3.6 MB/s per stream)
  ssh "$NPU1" "cd $STORE/policy-releases/$rel && find . -type f | sed 's#^./##'" > "$LOCAL/$rel.files" || return 1
  (cd "$LOCAL/$rel" && awk -F/ 'NF>1{NF--; print}' OFS=/ "$LOCAL/$rel.files" | sort -u | xargs -r mkdir -p)
  xargs -P 8 -I{} scp -q "$NPU1:$STORE/policy-releases/$rel/{}" "$LOCAL/$rel/{}" < "$LOCAL/$rel.files" || return 1
  echo "[mirror] $rel npu1 -> dev machine $(( $(date +%s) - t0 )) s ($(du -sh "$LOCAL/$rel" | cut -f1))"
  modal volume put -f "$VOL" "$LOCAL/$rel" "$VOLDIR/$rel" >/dev/null || return 1
  echo "[mirror] $rel dev machine -> Modal volume $(( $(date +%s) - t0 )) s"
  place "$rel" "$digest"
}

place() {
  # the VM half alone (the files are already on the volume): manifest to the VM, pull, verify, place.
  # D24: the download dir must exist first; modal 1.2.6 `volume get <dir> <missing-path>` writes every
  # file onto that one path (the last file wins), and the manifest check then sees only one blob
  [ "$DELIVERY" = s3 ] && { place_s3 "$1"; return; }
  local rel=$1 digest=${2:-$(awk '{print $1}' "$LOCAL/$1.make.txt" | tail -1)} t0
  t0=$(date +%s)
  vmsh "cat > /root/$rel.manifest.json" < "$LOCAL/$rel.manifest.json"
  vmsh "set -e; rm -rf /root/staged-$rel /root/staged-$rel-dl; mkdir -p /root/staged-$rel-dl; export PATH=/opt/sglang/bin:\$PATH; modal volume get --force $VOL $VOLDIR/$rel /root/staged-$rel-dl >/dev/null; \
    d=/root/staged-$rel-dl; [ -d \$d/$rel ] && d=\$d/$rel; mv \$d /root/staged-$rel; \
    cd /root/yeto-root && PYTHONPATH=. python3 -m yeto.rl.spot_rollout.cut_manifest place --manifest /root/$rel.manifest.json \
      --digest $digest --staged /root/staged-$rel --store /root/store" || return 1
  echo "[mirror] $rel placed on the VM (digest $digest) $(( $(date +%s) - t0 )) s"
}

case "${1:-}" in
  samples) samples ;;
  release) release "${2:?release name}" ;;
  place) place "${2:?release name}" ;;
  read-position) read_position ;;
  *) echo "usage: $0 samples | release <rN-vM>"; exit 2 ;;
esac
