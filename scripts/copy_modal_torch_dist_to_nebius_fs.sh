#!/usr/bin/env bash
# Copy the Flash-Next full torch_dist checkpoint from the Modal Volume
# `yeto-fn-models` (/torch_dist/qwen3.8-flash-next_torch_dist, TP1 PP4,
# ~234 GiB; FN-TRAIN-PLAN.md B0) into the Nebius shared filesystem that the
# training islands mount at /mnt/yeto-models, so that
#   --rl-megatron-ref-load /mnt/yeto-models/torch_dist/qwen3.8-flash-next_torch_dist
# (tests/multinode_gpu/fntrain.sh) resolves.  CPU-only Nebius VM; no GPU.
#
#   scripts/copy_modal_torch_dist_to_nebius_fs.sh [FILESYSTEM_ID]
#
# Flow: create disk+VM (FS attached as virtiofs tag yeto-models) -> mount ->
# scp ~/.modal.toml (0600, never echoed) -> docker python:3.12-slim with
# `pip install modal` -> `modal volume get --force` -> compare file count and
# bytes against the Modal side (listed from this machine via the modal SDK)
# -> write /mnt/yeto-models/yeto-complete/torch_dist-qwen3.8-flash-next.json
# -> delete VM + disk (also on failure, via trap).
# Env: CPU_PRESET (default 16vcpu-64gb for network bandwidth), KEEP_VM=1,
#      MODAL_PY (python that has the modal SDK, for the source-side listing),
#      VERIFY_ONLY=1 (skip the download; just compare what is on the FS with
#      the Modal listing and write the yeto-complete record).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${CPU_PRESET:=16vcpu-64gb}"
. "$HERE/nebius_cpu_vm_lib.sh"
FS=${1:-computefilesystem-e00nm64w4cqpkqd0ch}
VOL=yeto-fn-models
REMOTE=/torch_dist/qwen3.8-flash-next_torch_dist
DEST_PARENT=/mnt/yeto-models/torch_dist
DEST=$DEST_PARENT/qwen3.8-flash-next_torch_dist
: "${MODAL_PY:=/home/michael/.local/share/uv/tools/modal/bin/python}"
: "${KEEP_VM:=0}"
: "${VERIFY_ONLY:=0}"
[ -f "$HOME/.modal.toml" ] || { echo "[copy] ~/.modal.toml missing" >&2; exit 2; }

log() { echo "[copy] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

# Source-side truth: exact file count and bytes from the Modal volume listing.
read -r SRC_FILES SRC_BYTES < <("$MODAL_PY" - "$VOL" "$REMOTE" <<'PY'
import sys, modal
v = modal.Volume.from_name(sys.argv[1])
fs = [e for e in v.listdir(sys.argv[2], recursive=True) if e.type == modal.volume.FileEntryType.FILE]
print(len(fs), sum(e.size for e in fs))
PY
)
log "modal side: $SRC_FILES files, $SRC_BYTES bytes ($VOL:$REMOTE)"

NAME="yeto-fn-tdcopy-$(date -u +%Y%m%d%H%M)"
VM=; DISK=; T_VM0=
cleanup() {
  local rc=$?
  if [ "$KEEP_VM" = 1 ]; then echo "[copy] KEEP_VM=1: $VM $DISK left running" >&2; return; fi
  [ -n "$VM" ] && { vm_delete "$VM"; log "deleted VM $VM"; }
  [ -n "$DISK" ] && { sleep 5; disk_delete "$DISK"; log "deleted disk $DISK"; }
  [ -n "$T_VM0" ] && log "VM wall time: $(( $(date +%s) - T_VM0 )) s (preset $CPU_PLATFORM/$CPU_PRESET)"
  log "exit rc=$rc"
}
trap cleanup EXIT
T_VM0=$(date +%s)
DISK=$(vm_disk_create "$NAME" 50)
VM=$(vm_create "$NAME" "$DISK" "$FS")
IP=$(vm_ip "$VM"); vm_wait_ssh "$IP"
log "ssh up $IP ($VM, disk $DISK)"
vm_ssh "$IP" "sudo mkdir -p /mnt/yeto-models && sudo mount -t virtiofs yeto-models /mnt/yeto-models && sudo chmod a+w /mnt/yeto-models && sudo mkdir -p $DEST_PARENT && df -h /mnt/yeto-models | tail -1"
T0=$(date +%s)
if [ "$VERIFY_ONLY" != 1 ]; then
  # Credentials: file copy only, mode 0600, never printed.
  scp "${SSH_OPTS[@]}" -q "$HOME/.modal.toml" "yeto@$IP:/home/yeto/.modal.toml"
  vm_ssh "$IP" "chmod 600 /home/yeto/.modal.toml"
  vm_ssh "$IP" "sudo docker run --rm --net=host -v /mnt/yeto-models:/mnt/yeto-models -v /home/yeto/.modal.toml:/root/.modal.toml:ro -e MODAL_LOG_LEVEL=WARNING python:3.12-slim bash -c '
    set -e; pip install -q modal >/dev/null 2>&1; modal --version
    modal volume get --force $VOL $REMOTE $DEST_PARENT 2>&1 | tail -3'"
fi
T1=$(date +%s)
log "download finished in $((T1-T0)) s (VERIFY_ONLY=$VERIFY_ONLY)"
# Note: `read < <(... | tr)` would return 1 on the missing trailing newline
# under set -e, so capture first.
DST_STAT=$(vm_ssh "$IP" "sudo find $DEST -type f | wc -l; sudo du -sb $DEST | cut -f1" | tr '\n' ' ')
read -r DST_FILES DST_BYTES <<<"$DST_STAT"
log "fs side: $DST_FILES files, $DST_BYTES bytes ($DEST)"
vm_ssh "$IP" "sudo find $DEST -type f -printf '%s %p\n' | sort -k2; sudo cat $DEST/latest_checkpointed_iteration.txt; echo"
if [ "$DST_FILES" != "$SRC_FILES" ] || [ "$DST_BYTES" != "$SRC_BYTES" ]; then
  log "MISMATCH: modal $SRC_FILES/$SRC_BYTES vs fs $DST_FILES/$DST_BYTES"; exit 3
fi
vm_ssh "$IP" "sudo mkdir -p /mnt/yeto-models/yeto-complete && sudo tee /mnt/yeto-models/yeto-complete/torch_dist-qwen3.8-flash-next.json >/dev/null <<J
{\"path\": \"$DEST\", \"files\": $DST_FILES, \"bytes\": $DST_BYTES, \"seconds\": $((T1-T0)), \"source\": \"modal volume $VOL:$REMOTE\", \"completed_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"layout\": \"megatron torch_dist TP1 PP4\"}
J
sudo cat /mnt/yeto-models/yeto-complete/torch_dist-qwen3.8-flash-next.json; echo; df -h /mnt/yeto-models | tail -1"
python3 -c "print('[copy] throughput %.0f MB/s' % ($DST_BYTES/max($T1-$T0,1)/1e6))"
log "done: $DEST verified ($DST_FILES files, $DST_BYTES bytes)"
