#!/usr/bin/env bash
# Bake a Nebius VM image whose /var/lib/docker already holds the pinned
# MILES_NEXT_IMAGE layers, so sky's `docker pull` of that digest on a fresh
# GPU node is a no-op (COLDSTART-PLAN.md #3).  CPU VM only; ~15 min, < $0.2.
#
#   scripts/bake_nebius_image.sh            # bakes yeto.rl.MILES_NEXT_IMAGE
#   IMAGE_REF=ghcr.io/...@sha256:... scripts/bake_nebius_image.sh
#
# Re-run whenever MILES_NEXT_IMAGE changes, then add the printed line to
# NEBIUS_BAKED_IMAGES in yeto/rl/__init__.py (key = the docker digest; the
# launcher refuses a stale mapping and falls back to the stock image).
# Registry credentials: ghcr.io entry of ~/.docker/config.json (as s1run.sh);
# they go over ssh stdin and are removed from the VM before imaging.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/nebius_cpu_vm_lib.sh"
: "${IMAGE_REF:=$(cd "$HERE/.." && python3 -c 'from yeto.rl import MILES_NEXT_IMAGE as m; print(m.removeprefix("docker:"))')}"
: "${DISK_GIB:=100}"     # image min disk size; sky grows it to --disk-size
DIGEST=${IMAGE_REF##*@sha256:}
[ ${#DIGEST} = 64 ] || { echo "IMAGE_REF must be pinned by @sha256: digest" >&2; exit 2; }
TAG=$(date -u +%Y%m%d%H%M)
NAME="yeto-cs2-bake-${DIGEST:0:12}-$TAG"
AUTH=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.docker/config.json")))["auths"]["ghcr.io"]["auth"])')

VM=; DISK=
cleanup() { [ -n "$VM" ] && vm_delete "$VM"; [ -n "$DISK" ] && disk_delete "$DISK"; }
trap cleanup EXIT

echo "[bake] $(date -u +%T) disk from $BASE_IMAGE_FAMILY (${DISK_GIB} GiB)"
DISK=$(vm_disk_create "$NAME" "$DISK_GIB")
echo "[bake] $(date -u +%T) VM $CPU_PLATFORM/$CPU_PRESET"
VM=$(vm_create "$NAME" "$DISK")
IP=$(vm_ip "$VM"); vm_wait_ssh "$IP"
echo "[bake] $(date -u +%T) ssh up $IP; pulling $IMAGE_REF"
echo "$AUTH" | vm_ssh "$IP" 'set -e; a=$(cat); u=$(echo "$a"|base64 -d|cut -d: -f1); echo "$a"|base64 -d|cut -d: -f2-|sudo docker login ghcr.io -u "$u" --password-stdin >/dev/null'
vm_ssh "$IP" "set -e; t0=\$(date +%s); sudo docker pull -q '$IMAGE_REF'; echo \"[bake] pull \$((\$(date +%s)-t0)) s\";
  sudo docker logout ghcr.io >/dev/null; sudo rm -rf /root/.docker;
  sudo docker image inspect '$IMAGE_REF' --format '{{.Id}} {{.Size}}'; nvidia-smi -L 2>&1 | head -1 || true; df -h / | tail -1"
# Generalize: next boot is a new instance (cloud-init re-runs, new host keys).
vm_ssh "$IP" 'sudo cloud-init clean --logs >/dev/null; sudo rm -f /etc/ssh/ssh_host_*; sudo truncate -s0 /etc/machine-id; sudo rm -rf /home/yeto/.ssh; sync; (sleep 2; sudo poweroff) >/dev/null 2>&1 &' || true
echo "[bake] $(date -u +%T) stopping"
nebius compute instance stop --id "$VM" >/dev/null 2>&1 || true
vm_delete "$VM"; VM=
echo "[bake] $(date -u +%T) creating image"
IMG=$(nebius compute image create --parent-id "$NEBIUS_PROJECT" --name "$NAME" --source-disk-id "$DISK" \
  --cpu-architecture AMD64 --labels "docker-digest=${DIGEST:0:63}" --format json 2>/dev/null | _jq 'd["metadata"]["id"]')
disk_delete "$DISK"; DISK=
echo "[bake] $(date -u +%T) done"
echo "NEBIUS_BAKED_IMAGES entry:  \"sha256:$DIGEST\": {\"eu-north1\": \"$IMG\"},"
