"""A leased National Compute node as a SkyPilot SSH node pool.

Per node (one pool per lease, so a node that reaches its lease end takes only
its own pool with it):

1. an OpenSSH ``Host`` alias in ``<state_dir>/ssh_config`` (included from
   ``~/.ssh/config``): the node IP, the login user, the lease key, the host key
   pinned from the ticket, and the workspace relay as ``ProxyCommand``
   (:mod:`.tunnel`); sky's SSH node pool code uses the alias as it is
   (``check_host_in_ssh_config``), so every sky step - the k3s install, the
   kubectl tunnel - rides the relay;
2. the host bootstrap over that alias, before k3s (it refuses a node that
   already runs a k3s it did not install: National Compute MI355X nodes ship the
   provider's single-node "health k3s", and sky's k3s install reuses its
   ``/etc/rancher/k3s/config.yaml`` and data and crash-loops it on a token
   mismatch - seen on ticket 8b5b5d88596a, 2026-11-09): ``/mnt/nvme`` must be
   mounted; ``/var/lib/rancher``, ``/var/lib/kubelet`` and ``/var/lib/containerd``
   are bind-mounted onto NVMe (the 123 GB root disk does not hold a 15 GB
   serving image plus its writable layer - GPU-B report section 6.0);
3. an entry in ``~/.sky/ssh_node_pools.yaml`` and ``sky ssh up --infra <pool>``;
4. AMD GPUs for sky: sky's pool deploy installs only the NVIDIA operator, so the
   ROCm device plugin DaemonSet is applied and the node is labelled
   ``skypilot.co/accelerator=mi355x`` (sky reads it as ``MI355X``);
   tasks then ask for ``MI355X:8`` on ``infra: ssh/<pool>``.

Teardown is the reverse: ``sky ssh down --infra <pool>``, drop the pool entry and
the alias.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .client import NodeLease
from .tunnel import RELAY_PORT, state_dir

SKY_POOLS_FILE = Path(os.path.expanduser("~/.sky/ssh_node_pools.yaml"))
SSH_CONFIG = Path(os.path.expanduser("~/.ssh/config"))
ACCELERATOR_LABEL = "skypilot.co/accelerator"
SKY_ACCELERATOR = "MI355X"
AMD_DP_IMAGE = os.environ.get("YETO_NC_AMD_DP_IMAGE", "rocm/k8s-device-plugin:latest")
BIND_DIRS = ("/var/lib/rancher", "/var/lib/kubelet", "/var/lib/containerd")


def pool_name(ticket: str) -> str:
    return f"nc-{ticket}"


def host_alias(ticket: str) -> str:
    return f"yeto-nc-{ticket}"


def key_path() -> Path:
    return state_dir() / "id_ed25519"


def ensure_key(run: Callable[..., Any] = subprocess.run) -> str:
    """The lease key (generated once, never leaves the launcher host); returns the public key."""
    key = key_path()
    if not key.exists():
        run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "yeto-nc", "-f", str(key)], check=True)
    return Path(f"{key}.pub").read_text().strip()


def proxy_command(account: int, relay_port: int = RELAY_PORT) -> str:
    env = []
    root = str(Path(__file__).resolve().parents[3])
    env.append(f"PYTHONPATH={shlex.quote(root)}")
    for name in ("NC_TOOL_DIR", "NC_NODE_BIN", "YETO_NC_STATE_DIR"):
        if os.environ.get(name):
            env.append(f"{name}={shlex.quote(os.environ[name])}")
    return (f"env {' '.join(env)} {shlex.quote(sys.executable)} -P -m yeto.launch.nc proxy "
            f"--account {account} --relay-port {relay_port} %h %p")


def ssh_host_block(lease: NodeLease, *, relay_port: int = RELAY_PORT) -> str:
    known = state_dir() / "known_hosts"
    return "\n".join([
        f"Host {host_alias(lease.ticket)}",
        f"  HostName {lease.ip}",
        f"  User {lease.user}",
        f"  IdentityFile {key_path()}",
        "  IdentitiesOnly yes",
        f"  UserKnownHostsFile {known}",
        "  StrictHostKeyChecking yes",
        "  ServerAliveInterval 15",
        "  ServerAliveCountMax 4",
        f"  ProxyCommand {proxy_command(lease.account, relay_port)}",
        "",
    ])


def _blocks(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines(keepends=True):
        if line.startswith("Host "):
            current = line.split()[1]
            out[current] = ""
        if current:
            out[current] += line
    return out


def write_ssh_alias(lease: NodeLease, *, ssh_config: Path = SSH_CONFIG, relay_port: int = RELAY_PORT) -> str:
    """Add the node's alias and pin its host key; make ``~/.ssh/config`` include our file."""
    ours = state_dir() / "ssh_config"
    blocks = _blocks(ours.read_text()) if ours.exists() else {}
    blocks[host_alias(lease.ticket)] = ssh_host_block(lease, relay_port=relay_port)
    ours.write_text("".join(blocks.values()))
    os.chmod(ours, 0o600)
    if lease.host_key:
        known = state_dir() / "known_hosts"
        lines = [ln for ln in (known.read_text().splitlines() if known.exists() else [])
                 if not ln.startswith(f"{lease.ip} ")]
        lines.append(f"{lease.ip} {lease.host_key.strip()}")
        known.write_text("\n".join(lines) + "\n")
        os.chmod(known, 0o600)
    include = f"Include {ours}"
    text = ssh_config.read_text() if ssh_config.exists() else ""
    if include not in text.splitlines():
        ssh_config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        ssh_config.write_text(f"{include}\n{text}")  # Include must precede any Host block
    return host_alias(lease.ticket)


def remove_ssh_alias(ticket: str) -> None:
    ours = state_dir() / "ssh_config"
    if ours.exists():
        blocks = _blocks(ours.read_text())
        blocks.pop(host_alias(ticket), None)
        ours.write_text("".join(blocks.values()))


def write_pool(ticket: str, *, path: Path = SKY_POOLS_FILE) -> str:
    """Add ``nc-<ticket>`` (one host: the alias) to sky's pool file; other pools are kept."""
    doc = yaml.safe_load(path.read_text()) if path.exists() else None
    doc = doc or {}
    doc[pool_name(ticket)] = {"hosts": [host_alias(ticket)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=True))
    return pool_name(ticket)


def remove_pool(ticket: str, *, path: Path = SKY_POOLS_FILE) -> None:
    if not path.exists():
        return
    doc = yaml.safe_load(path.read_text()) or {}
    if doc.pop(pool_name(ticket), None) is not None:
        path.write_text(yaml.safe_dump(doc, sort_keys=True))


def host_bootstrap_script(nvme_root: str = "/mnt/nvme") -> str:
    """Runs on the node before k3s: NVMe checks and the state directories on NVMe."""
    binds = " ".join(BIND_DIRS)
    return f"""set -eu
N={shlex.quote(nvme_root)}
mountpoint -q "$N" || {{ echo "YETO_NC_BOOTSTRAP_FAIL $N is not a mount point"; lsblk -o NAME,SIZE,MOUNTPOINT; exit 3; }}
sudo -n true || {{ echo "YETO_NC_BOOTSTRAP_FAIL passwordless sudo is required"; exit 3; }}
if [ -e /etc/rancher/k3s/config.yaml ] || systemctl list-unit-files k3s.service >/dev/null 2>&1; then
  echo "YETO_NC_BOOTSTRAP_FAIL a k3s not installed by sky is on the node (provider health k3s?): not touching it"
  sudo head -3 /etc/rancher/k3s/config.yaml 2>/dev/null; exit 5
fi
sudo mkdir -p "$N/models" "$N/cache" "$N/logs" "$N/work" "$N/state"
sudo chown "$(id -u):$(id -g)" "$N/models" "$N/cache" "$N/logs" "$N/work"
for d in {binds}; do
  if mountpoint -q "$d"; then continue; fi
  src="$N/state/$(basename "$d")"; sudo mkdir -p "$src" "$d"
  for s in docker.socket docker containerd k3s k3s-agent; do sudo systemctl stop "$s" >/dev/null 2>&1 || true; done
  sudo rsync -aHAX "$d/" "$src/" 2>/dev/null || true
  sudo mount --bind "$src" "$d"
  grep -q " $d none bind" /etc/fstab || echo "$src $d none bind 0 0" | sudo tee -a /etc/fstab >/dev/null
done
systemctl list-unit-files containerd.service >/dev/null 2>&1 && sudo systemctl start containerd >/dev/null 2>&1 || true
systemctl list-unit-files docker.service >/dev/null 2>&1 && sudo systemctl start docker >/dev/null 2>&1 || true
echo "YETO_NC_BOOTSTRAP_OK root=$(df -h / | awk 'NR==2{{print $4}}') nvme=$(df -h "$N" | awk 'NR==2{{print $4}}') binds=$(for d in {binds}; do mountpoint -q $d && printf y || printf n; done) gpus=$(ls /dev/dri 2>/dev/null | grep -c renderD) kfd=$(test -e /dev/kfd && echo y || echo n)"
"""


AMD_DEVICE_PLUGIN = """apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: amdgpu-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: amdgpu-dp-ds
  template:
    metadata:
      labels:
        name: amdgpu-dp-ds
    spec:
      priorityClassName: system-node-critical
      tolerations:
      - key: CriticalAddonsOnly
        operator: Exists
      containers:
      - image: {image}
        name: amdgpu-dp-cntr
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: dp
          mountPath: /var/lib/kubelet/device-plugins
        - name: sys
          mountPath: /sys
      volumes:
      - name: dp
        hostPath:
          path: /var/lib/kubelet/device-plugins
      - name: sys
        hostPath:
          path: /sys
"""


def amd_gpu_script(gpus: int = 8, *, image: str = AMD_DP_IMAGE, timeout_s: int = 300) -> str:
    """Runs on the node after ``sky ssh up``: ROCm device plugin + the sky accelerator label."""
    manifest = AMD_DEVICE_PLUGIN.format(image=image)
    return f"""set -eu
K="sudo k3s kubectl"
cat <<'__YETO_DP__' | $K apply -f -
{manifest}__YETO_DP__
$K label node --all {ACCELERATOR_LABEL}={SKY_ACCELERATOR.lower()} --overwrite
for i in $(seq 1 {timeout_s // 5}); do
  n=$($K get nodes -o jsonpath='{{.items[0].status.allocatable.amd\\.com/gpu}}' 2>/dev/null || true)
  [ "${{n:-0}}" -ge {gpus} ] && {{ echo "YETO_NC_AMD_GPU_OK amd.com/gpu=$n"; exit 0; }}
  sleep 5
done
echo "YETO_NC_AMD_GPU_FAIL amd.com/gpu=${{n:-0}}"; $K -n kube-system logs -l name=amdgpu-dp-ds --tail 20 || true; exit 4
"""


def ssh_run(alias: str, script: str, *, run: Callable[..., Any] = subprocess.run,
            timeout: float = 1800) -> subprocess.CompletedProcess:
    """``ssh -T alias bash -s`` with the script on stdin (no tty, BatchMode)."""
    return run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=60", alias, "bash", "-s"],
               input=script, capture_output=True, text=True, timeout=timeout, check=False)


def sky_cmd() -> list[str]:
    return shlex.split(os.environ.get("YETO_SKY_BIN", "sky"))


def sky_ssh_up(pool: str, *, run: Callable[..., Any] = subprocess.run, timeout: float = 1800):
    return run([*sky_cmd(), "ssh", "up", "--infra", pool], capture_output=True, text=True,
               timeout=timeout, check=False)


def sky_ssh_down(pool: str, *, run: Callable[..., Any] = subprocess.run, timeout: float = 900):
    return run([*sky_cmd(), "ssh", "down", "--infra", pool], capture_output=True, text=True,
               timeout=timeout, check=False)
