# Sourced helpers for scripts/bake_nebius_image.sh and
# scripts/populate_nebius_model_store.sh: one CPU-only Nebius VM with a
# public IPv4 (quota: 3 per tenant), created from a disk we own so that
# deleting the instance keeps the disk.  No GPU is ever requested.
set -euo pipefail
export PATH="$HOME/.nebius/bin:$PATH"
: "${NEBIUS_PROJECT:=project-e00eqrj3pr00622zrgdeyc}"   # eu-north1 (~/.sky/config.yaml)
: "${NEBIUS_SUBNET:=$(nebius vpc subnet list --parent-id "$NEBIUS_PROJECT" --format json | python3 -c 'import json,sys;print(json.load(sys.stdin)["items"][0]["metadata"]["id"])')}"
: "${CPU_PLATFORM:=cpu-d3}"
: "${CPU_PRESET:=8vcpu-32gb}"
# Base of every GPU VM sky creates on Nebius (sky/clouds/nebius.py: platform
# gpu-* -> image family ubuntu24.04-cuda13.0); it carries the NVIDIA driver,
# docker and the container toolkit.  A baked image must start from it.
: "${BASE_IMAGE_FAMILY:=ubuntu24.04-cuda13.0}"
: "${WORK:=$HOME/.cache/yeto-nebius-cpu-vm}"
mkdir -p "$WORK"; chmod 700 "$WORK"
KEY="$WORK/id_ed25519"
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N '' -f "$KEY" -C yeto-cpu-vm
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10)

_jq() { python3 -c "import json,sys;d=json.load(sys.stdin);print($1)"; }

# vm_disk_create NAME SIZE_GIB [image_id]  -> echoes disk id
vm_disk_create() {
  local src=(--source-image-family-image-family "$BASE_IMAGE_FAMILY" --source-image-family-parent-id project-e00public-images)
  [ -n "${3:-}" ] && src=(--source-image-id "$3")
  nebius compute disk create --parent-id "$NEBIUS_PROJECT" --name "$1" --type network_ssd \
    --size-gibibytes "$2" --block-size-bytes 4096 "${src[@]}" --format json 2>/dev/null | _jq 'd["metadata"]["id"]'
}

# vm_create NAME DISK_ID [FILESYSTEM_ID] -> echoes instance id
vm_create() {
  local name=$1 disk=$2 fs=${3:-} pub; pub=$(cat "$KEY.pub")
  python3 - "$name" "$disk" "$fs" "$pub" "$NEBIUS_PROJECT" "$NEBIUS_SUBNET" "$CPU_PLATFORM" "$CPU_PRESET" > "$WORK/$name.json" <<'PY'
import json, sys
name, disk, fs, pub, proj, subnet, plat, preset = sys.argv[1:]
ci = "#cloud-config\nusers:\n  - name: yeto\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys: [%s]\n" % json.dumps(pub)
spec = {"resources": {"platform": plat, "preset": preset},
        "boot_disk": {"attach_mode": "READ_WRITE", "existing_disk": {"id": disk}},
        "cloud_init_user_data": ci,
        "network_interfaces": [{"name": "eth0", "subnet_id": subnet, "ip_address": {}, "public_ip_address": {}}]}
if fs:
    spec["filesystems"] = [{"attach_mode": "READ_WRITE", "mount_tag": "yeto-models", "existing_filesystem": {"id": fs}}]
print(json.dumps({"metadata": {"parent_id": proj, "name": name}, "spec": spec}))
PY
  nebius compute instance create --file "$WORK/$name.json" --format json 2>/dev/null | _jq 'd["metadata"]["id"]'
}

vm_ip() {
  nebius compute instance get --id "$1" --format json | _jq 'd["status"]["network_interfaces"][0]["public_ip_address"]["address"].split("/")[0]'
}

vm_wait_ssh() {
  local ip=$1 i
  for i in $(seq 60); do ssh "${SSH_OPTS[@]}" "yeto@$ip" true 2>/dev/null && return 0; sleep 10; done
  echo "ssh to $ip never came up" >&2; return 1
}

vm_ssh() { local ip=$1; shift; ssh "${SSH_OPTS[@]}" "yeto@$ip" "$@"; }

vm_delete() { nebius compute instance delete --id "$1" >/dev/null 2>&1 || true; }
disk_delete() { nebius compute disk delete --id "$1" >/dev/null 2>&1 || true; }
