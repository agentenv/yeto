#!/usr/bin/env bash
# Write pinned HF snapshots into the Nebius shared filesystem that
# `yeto ... --model-store nebius-fs://<filesystem-id>` mounts on every island
# node (COLDSTART-PLAN.md #4).  CPU VM only.  Layout (HF cache, so the learner
# just points HF_HUB_CACHE at it):
#   <fs>/hub/models--ORG--NAME/snapshots/<rev>/...
#   <fs>/yeto-complete/ORG--NAME@<rev>.json   # written only after a full,
#                                             # size-checked download
#   scripts/populate_nebius_model_store.sh computefilesystem-... Qwen/Qwen3-0.6B@c1899de...
# BOOT_IMAGE=computeimage-... boots a baked image (also checks the pre-pull).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/nebius_cpu_vm_lib.sh"
FS=$1; shift
[ $# -ge 1 ] || { echo "usage: $0 FILESYSTEM_ID ORG/NAME@REV..." >&2; exit 2; }
: "${IMAGE_REF:=$(cd "$HERE/.." && python3 -c 'from yeto.rl import MILES_NEXT_IMAGE as m; print(m.removeprefix("docker:"))')}"
: "${BOOT_IMAGE:=}"
: "${KEEP_VM:=0}"
NAME="yeto-cs2-populate-$(date -u +%Y%m%d%H%M)"
VM=; DISK=
cleanup() { [ "$KEEP_VM" = 1 ] && { echo "[populate] KEEP_VM=1: $VM $DISK left running" >&2; return; }; [ -n "$VM" ] && vm_delete "$VM"; [ -n "$DISK" ] && disk_delete "$DISK"; }
trap cleanup EXIT
DISK=$(vm_disk_create "$NAME" 100 "$BOOT_IMAGE")
VM=$(vm_create "$NAME" "$DISK" "$FS")
IP=$(vm_ip "$VM"); vm_wait_ssh "$IP"
echo "[populate] $(date -u +%T) ssh up $IP ($VM)"
vm_ssh "$IP" "sudo mkdir -p /mnt/yeto-models && sudo mount -t virtiofs yeto-models /mnt/yeto-models && sudo chmod a+w /mnt/yeto-models && df -h /mnt/yeto-models | tail -1"
if [ -n "$BOOT_IMAGE" ]; then
  # By-digest pulls are untagged, so `docker images` does not list them.
  vm_ssh "$IP" "sudo docker image inspect '$IMAGE_REF' --format '{{.Id}} {{.Size}}'" \
    && echo "[populate] baked image has ${IMAGE_REF##*@} pre-pulled" || echo "[populate] WARNING: digest not in baked image" >&2
  RUN_IMAGE=$IMAGE_REF
else
  RUN_IMAGE=python:3.12-slim
fi
for spec in "$@"; do
  repo=${spec%@*}; rev=${spec#*@}
  echo "[populate] $(date -u +%T) $repo@$rev"
  vm_ssh "$IP" "sudo docker run --rm --net=host -v /mnt/yeto-models:/store -e HF_HUB_ENABLE_HF_TRANSFER=1 -e HF_HUB_DISABLE_PROGRESS_BARS=1 --entrypoint bash '$RUN_IMAGE' -c '
    set -e; python3 -c \"import hf_transfer\" 2>/dev/null || pip install -q huggingface_hub hf_transfer
    python3 - <<PY
import json, os, time
from huggingface_hub import HfApi, snapshot_download
repo, rev = \"$repo\", \"$rev\"
t0 = time.time()
path = snapshot_download(repo, revision=rev, cache_dir=\"/store/hub\", max_workers=16)
dt = time.time() - t0
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
want = {s.rfilename: s.size for s in info.siblings}
got = {f: os.path.getsize(os.path.join(path, f)) for f in want}
bad = [f for f in want if got[f] != want[f]]
assert info.sha == rev and not bad, (info.sha, bad[:5])
n = sum(want.values())
os.makedirs(\"/store/yeto-complete\", exist_ok=True)
rec = dict(repo=repo, revision=rev, files=len(want), bytes=n, seconds=round(dt, 1), snapshot=os.path.relpath(path, \"/store\"))
json.dump(rec, open(\"/store/yeto-complete/\" + repo.replace(\"/\", \"--\") + \"@\" + rev + \".json\", \"w\"))
print(\"[populate]\", json.dumps(rec), \"%.0f MB/s\" % (n / max(dt, 1e-3) / 1e6))
PY'"
done
# Cold read of the whole store from this node (page cache dropped).
vm_ssh "$IP" "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; t0=\$(date +%s.%N); b=\$(sudo find /mnt/yeto-models/hub -type f -name '*.safetensors' -print0 | xargs -0 -P 16 -n 1 sh -c 'cat \"\$0\" | wc -c' | awk '{s+=\$1} END {print s}'); t1=\$(date +%s.%N); python3 -c \"b=\$b;d=\$t1-\$t0;print('[populate] cold read %.1f GB in %.0f s = %.0f MB/s' % (b/1e9, d, b/d/1e6))\""
vm_ssh "$IP" "ls /mnt/yeto-models/yeto-complete; df -h /mnt/yeto-models | tail -1"
echo "[populate] $(date -u +%T) done"
