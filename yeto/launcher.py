"""SkyPilot orchestration: one syncer VM + one cluster per learner.

Flow:
  1. build the Rust syncer locally (release) — the binary is file-mounted to
     a cheap CPU VM whose TCP port is opened to the learners; a submitter
     that is not x86_64 Linux (e.g. a Mac) instead has the VM build the
     syncer from the synced repo (SYNCER_REMOTE_BUILD);
  2. launch the syncer cluster, read its head public IP;
  3. launch all learner clusters in parallel (each pinned to its cloud/region
     from the --gpu spec), with the repo synced as the workdir and the syncer
     address passed via env;
  4. stream all job logs with per-cluster prefixes while a fleet controller
     polls job/cluster health, re-provisions failed or preempted clusters
     with their original spec, and abandons (tears down) any learner that
     cannot be restored within --recover-timeout — the run continues with
     the shrunken fleet;
  5. tear everything down (unless --keep; abandoned learners are always
     torn down).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import contextlib
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import delivery
from .gpu_spec import ClusterSpec, parse_gpu_spec
from .models import MODEL_WEIGHT_GB

SYNCER_PORT = 29400
# rl-algo-critic-family 4.2.1 (design D4 plan a): a critic algorithm runs a
# second syncer beside the actor's, on its own port, checkpoint and tape; its
# layout is the critic layout (critic_layout_hash), never the actor LoRA one.
CRITIC_SYNCER_PORT = 29401
REPO_ROOT = Path(__file__).resolve().parent.parent

# WAN transport tuning applied to every node at setup: BBR keeps throughput
# up on lossy long-RTT paths, and raised buffer ceilings let kernel
# auto-tuning grow windows to high-BDP sizes (we deliberately do NOT set
# per-socket SO_SNDBUF/SO_RCVBUF — static values disable auto-tuning).
# Every line is best-effort: restricted kernels just keep their defaults.
WAN_TUNING = (
    "sudo modprobe tcp_bbr 2>/dev/null || true; "
    "sudo sysctl -qw net.core.default_qdisc=fq "
    "net.ipv4.tcp_congestion_control=bbr 2>/dev/null || true; "
    "sudo sysctl -qw net.core.rmem_max=67108864 net.core.wmem_max=67108864 "
    "2>/dev/null || true; "
    "sudo sysctl -qw 'net.ipv4.tcp_rmem=4096 131072 67108864' "
    "'net.ipv4.tcp_wmem=4096 131072 67108864' 2>/dev/null || true"
)

# GPU nodes (p4/p5/p6, g5/g6) ship terabytes of local instance-store NVMe
# that sky leaves untouched, while the EBS root's default throughput
# (~125 MB/s) turns a 300 GB weight download into a ~40-minute disk stall.
# Stripe every instance-store device into RAID0 at /opt/yeto-nvme and point
# the HF caches there (NVME_ENV below, applied in setup AND run). Instance
# store is ephemeral by design — exactly right for a re-downloadable cache;
# durable state (checkpoints, outputs) stays on EBS. Idempotent: a
# recovery relaunch on a node that already has the mount skips everything.
# Ephemeral local-disk model strings per cloud: AWS instance store, GCP
# local SSD, Azure temp/local NVMe. Allowlist by model (never "anything
# unmounted") so a persistent data disk is never wiped. RunPod pods need
# nothing here — their container volume is already local-NVMe-backed.
NVME_SETUP = """
if mountpoint -q /opt/yeto-nvme; then
  echo "[yeto-setup] NVMe scratch already mounted"
else
  DEVS=$(lsblk -dno NAME,MODEL | grep -Ei \
    'Instance Storage|nvme_card|EphemeralDisk|NVMe Direct Disk' \
    | awk '{print "/dev/"$1}')
  N=$(printf '%s\\n' "$DEVS" | grep -c /dev || true)
  # Some images (e.g. AWS DLAMIs) already RAID and mount the instance
  # store themselves; reuse that filesystem via bind-mount rather than
  # fighting busy devices with mkfs.
  EXISTING=""
  for d in $DEVS; do
    mp=$(lsblk -rno MOUNTPOINT "$d" 2>/dev/null | grep -m1 '^/' || true)
    if [ -n "$mp" ]; then EXISTING="$mp"; break; fi
  done
  if [ -n "$EXISTING" ]; then
    sudo mkdir -p /opt/yeto-nvme
    if sudo mount --bind "$EXISTING" /opt/yeto-nvme; then
      sudo chown "$(whoami)" /opt/yeto-nvme
      echo "[yeto-setup] reusing image-mounted NVMe at $EXISTING via /opt/yeto-nvme"
    else
      echo "[yeto-setup] NVMe setup failed; staying on the boot disk" >&2
    fi
  elif [ "$N" -eq 0 ]; then
    echo "[yeto-setup] no local ephemeral NVMe; HF cache stays on the boot disk"
  else
    sudo mkdir -p /opt/yeto-nvme
    if [ "$N" -ge 2 ]; then
      command -v mdadm >/dev/null || sudo apt-get -qq install -y mdadm
      sudo mdadm --create /dev/md0 --level=0 --force --run --raid-devices="$N" $DEVS
      DEV=/dev/md0
    else
      DEV=$DEVS
    fi
    if sudo mkfs.ext4 -q -F "$DEV" && sudo mount -o noatime "$DEV" /opt/yeto-nvme; then
      sudo chown "$(whoami)" /opt/yeto-nvme
      echo "[yeto-setup] striped $N NVMe device(s) at /opt/yeto-nvme"
    else
      echo "[yeto-setup] NVMe setup failed; staying on the boot disk" >&2
    fi
  fi
fi
"""

# Route HF model/dataset caches to the NVMe scratch when it is REALLY
# mounted (a failed stripe must never divert the cache into a plain dir on
# the small boot disk); `|| true` keeps NVMe-less nodes working. Used by
# both the setup shell (so the background prefetch lands on NVMe) and the
# run command (so from_pretrained reads from it).
NVME_ENV = (
    "mountpoint -q /opt/yeto-nvme && export HF_HOME=/opt/yeto-nvme/hf "
    "HF_HUB_CACHE=/opt/yeto-nvme/hf/hub "
    "HF_DATASETS_CACHE=/opt/yeto-nvme/hf/datasets || true"
)

# huggingface_hub reads the token from $HF_HOME/token, so a token mounted
# to the default location must follow HF_HOME when NVME_ENV moves it.
# Authenticated Hub requests get a 1000-req/5-min per-IP quota vs 500
# anonymous — 8 ranks each revalidating configs/tokenizers burn through the
# anonymous one in a few crash-loop cycles.
HF_TOKEN_PATH = "~/.cache/huggingface/token"
HF_TOKEN_ENV = (
    '[ -f ~/.cache/huggingface/token ] && [ -n "$HF_HOME" ] && '
    "mkdir -p $HF_HOME && cp -n ~/.cache/huggingface/token $HF_HOME/token || true"
)

# Pick the torch wheel on the node from what the HARDWARE requires (GPU
# compute capability) and what the HOST supports (driver version):
# Blackwell (SM100+) only has kernels in cu128 wheels (torch >= 2.7),
# which need driver >= 570; Ampere/Hopper AMIs run driver 535, whose
# ceiling is cu121-era wheels — a mismatched wheel does not error, it
# silently drops the GPUs (torch.cuda.is_available() -> False).
#
# Robustness properties, each one paid for by a prior incident:
#  * decision by compute cap first — a driver-only heuristic mis-selects
#    when the parse fails on a Blackwell node;
#  * hard, loud failures at SETUP time (impossible combos, missing
#    nvidia-smi, and a post-install is_available() verification) so a bad
#    node dies in provisioning logs instead of crash-looping the fleet
#    controller through job restarts;
#  * idempotent: a recovery relaunch whose torch already sees CUDA skips
#    the reinstall entirely;
#  * runs BEFORE `pip install -r requirements.txt` so resolution treats
#    torch as satisfied instead of dragging in a default wheel.
TORCH_SETUP = """
torch_cuda_ok() { python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; }
if torch_cuda_ok; then
  echo "[yeto-setup] existing torch already sees CUDA; keeping it"
else
  if ! nvidia-smi -L >/dev/null 2>&1; then
    if lspci 2>/dev/null | grep -qi nvidia; then
      # An NVIDIA device exists but the resident driver cannot drive it —
      # e.g. sky's default AMI ships 535, which predates Blackwell and
      # never loads against B200 silicon. Install the newest -open driver
      # (SM100+ REQUIRES the open kernel modules) and give DKMS time.
      echo "[yeto-setup] NVIDIA GPU present but driver not working; installing an open-module driver"
      sudo apt-get -qq update
      sudo apt-get -yqq install "linux-headers-$(uname -r)" >/dev/null 2>&1 || true
      for pkg in nvidia-driver-580-open nvidia-driver-575-open nvidia-driver-570-open; do
        if sudo apt-get -yqq install "$pkg" >/dev/null 2>&1; then
          echo "[yeto-setup] installed $pkg"
          break
        fi
      done
      sudo modprobe nvidia 2>/dev/null || true
    fi
    tries=0
    until nvidia-smi -L >/dev/null 2>&1; do
      tries=$((tries+1))
      if [ "$tries" -ge 24 ]; then
        echo "[yeto-setup] ERROR: NVIDIA driver not ready after 120s (install failed or no GPU)" >&2
        exit 1
      fi
      sleep 5
    done
  fi
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
  echo "[yeto-setup] GPU compute cap ${CAP:-unknown}.x, driver ${DRV:-unknown}"
  case "$CAP" in ''|*[!0-9]*)
    echo "[yeto-setup] ERROR: unparseable GPU compute capability: '$CAP'" >&2
    exit 1
  ;; esac
  case "$DRV" in ''|*[!0-9]*)
    echo "[yeto-setup] ERROR: unparseable NVIDIA driver version: '$DRV'" >&2
    exit 1
  ;; esac
  if [ "$CAP" -ge 10 ] && [ "$DRV" -lt 570 ]; then
    echo "[yeto-setup] ERROR: SM${CAP}x GPU needs cu128 (driver >= 570) but driver is $DRV" >&2
    exit 1
  fi
  if [ "$CAP" -ge 10 ] || [ "$DRV" -ge 570 ]; then
    pip install -q "torch==2.8.*" --index-url https://download.pytorch.org/whl/cu128
  elif [ "$DRV" -ge 525 ]; then
    pip install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
  else
    echo "[yeto-setup] ERROR: driver $DRV predates CUDA 12 (need >= 525)" >&2
    exit 1
  fi
  if ! torch_cuda_ok; then
    echo "[yeto-setup] ERROR: freshly installed torch cannot see the GPUs" >&2
    exit 1
  fi
fi
"""

# Rough per-GPU training capacity sanity check (bf16 LoRA, GB).
GPU_MEM_GB = {"A100": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 180, "L4": 24, "A10G": 24, "T4": 16, "V100": 16, "L40S": 48, "RTX-PRO-6000": 96, "RTX-6000-Ada": 48}


# rl-multinode-island D6: per-cloud NCCL/GLOO socket settings for a multi-node
# island (values are the CPU-side best guess; GPU task G1 confirms them).
# "auto": the first non-virtual interface that is up, resolved on the node
# (G1 2026-10-03: a Nebius L40S node's NIC is "network-interface-0", shown
# truncated as "network-interfa"; "eth0" made gloo fail with "Unable to find
# address for: eth0" in the SGLang scheduler).  "" leaves NCCL/gloo to their
# own detection.  NCCL_SOCKET_IFNAME in the environment always wins.
MULTINODE_IB_CLOUDS = frozenset({"modal"})
MULTINODE_SOCKET_IFNAME = {"nebius": "auto", "aws": "", "gcp": "", "ssh": ""}
_DETECT_IFACE = (
    "YETO_IFACE=${NCCL_SOCKET_IFNAME:-$(for d in /sys/class/net/*; do n=$(basename \"$d\"); "
    "case \"$n\" in lo|docker*|veth*|br-*|virbr*) continue;; esac; "
    "[ \"$(cat \"$d/operstate\" 2>/dev/null)\" = up ] && { echo \"$n\"; break; }; done)}"
)


# Megatron-LM checkout inside the radixark/miles-based ports images.
PORTS_MEGATRON_PATH = "/root/Megatron-LM"
# --rl-island-pre-run: the snippet runs in the island run script right after
# `cd ~/sky_workdir` (image environment, HOME=/root, GPUs idle, before Ray).
ISLAND_PRE_RUN_START = "[yeto-island] pre-run start"
ISLAND_PRE_RUN_DONE = "[yeto-island] pre-run done"
ISLAND_PRE_RUN_FAILED = "[yeto-island] pre-run FAILED"


def read_island_pre_run(path) -> str:
    """The --rl-island-pre-run snippet: a readable, non-empty local file."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"--rl-island-pre-run: {path} is not a file")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"--rl-island-pre-run: {path} is empty")
    return text if text.endswith("\n") else text + "\n"


def island_pre_run_block(script: str | None) -> str:
    """Run-script lines for the pre-run snippet ("" when none): a subshell so
    the snippet's own `set -e` failures surface as one FAILED line + exit 1."""
    if not script:
        return ""
    return (
        f'echo "{ISLAND_PRE_RUN_START} $(date -u +%FT%TZ)"\n'
        "(\nset -e\n" + script + ")\n"
        '_pre_rc=$?\n'
        '[ "$_pre_rc" = 0 ] || { echo "' + ISLAND_PRE_RUN_FAILED + ' rc=$_pre_rc $(date -u +%FT%TZ)" >&2; exit 1; }\n'
        f'echo "{ISLAND_PRE_RUN_DONE} $(date -u +%FT%TZ)"\n'
    )



def multinode_env_prelude(cloud: str, num_nodes: int) -> str:
    """Shell exports every island node runs before Ray starts; "" on one node."""
    if num_nodes <= 1:
        return ""
    # Modal clustered islands request RDMA (ModalIslandConfig.rdma): leave
    # NCCL's IB transport on there; other clouds run NCCL over TCP.
    ib_default = 0 if cloud in MULTINODE_IB_CLOUDS else 1
    lines = [f"export NCCL_IB_DISABLE=${{NCCL_IB_DISABLE:-{ib_default}}}",
             "export NCCL_DEBUG=${NCCL_DEBUG:-WARN}"]
    iface = MULTINODE_SOCKET_IFNAME.get(cloud, "")
    if iface == "auto":
        lines += [_DETECT_IFACE,
                  'echo "[yeto-island] socket interface: ${YETO_IFACE:-<none>}"',
                  'export NCCL_SOCKET_IFNAME="$YETO_IFACE" GLOO_SOCKET_IFNAME="$YETO_IFACE"']
    elif iface:
        lines += [f"export NCCL_SOCKET_IFNAME={iface}", f"export GLOO_SOCKET_IFNAME={iface}"]
    return "\n".join(lines) + "\n"


# Clouds that accept sky's ``network_tier="best"`` only for some shapes: Nebius
# offers the InfiniBand tier for H100:8 / H200:8 nodes only and REJECTS the
# request for anything else ("Catalog does not contain any instances"), so a
# 2x1xL40S island must not ask for it (G1 of rl-multinode-island: the D6
# prelude already runs NCCL over TCP with NCCL_IB_DISABLE=1 there).
NETWORK_TIER_BEST_SHAPES = {"nebius": {("H100", 8), ("H200", 8)}}


def multinode_network_tier(cloud: str, gpu: str, gpus_per_node: int) -> str | None:
    """``"best"`` for a multi-node island when the cloud can honor it, else None."""
    shapes = NETWORK_TIER_BEST_SHAPES.get(cloud)
    if shapes is not None and (gpu.upper(), int(gpus_per_node)) not in shapes:
        return None
    return "best"


def rl_island_spec(args, spec):
    """The island view of ``spec`` under ``--rl-island-use-gpus-per-node M`` (m5 on
    8-GPU machines: provision N physical GPUs per node, give the island only local
    GPUs ``0..M-1``). Every island-side rule (trainer shape, layout, colocated
    engine check, gpu_pool, learner ``--rl-island-gpus-per-node``) sees M; the sky
    resources keep the physical ``spec``. Unset -> ``spec`` itself (no change)."""
    m = getattr(args, "rl_island_use_gpus_per_node", None)
    if m is None:
        return spec
    m = int(m)
    if m < 1 or m > spec.gpus_per_node:
        raise ValueError(f"--rl-island-use-gpus-per-node {m} must be in 1..{spec.gpus_per_node} "
                         f"(physical GPUs per node of {spec})")
    if spec.cloud == "modal":
        raise ValueError("--rl-island-use-gpus-per-node is not supported on Modal islands")
    if m == spec.gpus_per_node:
        return spec
    return dataclasses.replace(spec, gpus_per_node=m)


def rl_island_gpu_env(args, spec) -> str:
    """Shell exports pinning the island to local GPUs ``0..M-1`` (exported before
    ``ray start`` so Ray, the learner, Miles actors and the sglang subprocesses all
    inherit them); "" without ``--rl-island-use-gpus-per-node`` or when M equals the
    physical count."""
    island = rl_island_spec(args, spec)
    if island is spec:
        return ""
    m = island.gpus_per_node
    return (f"export CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(m))} "
            f"YETO_ISLAND_USE_GPUS_PER_NODE={m}\n"
            f'echo "[yeto-island] GPUs: physical {spec.gpus_per_node}/node, allocated {m}/node '
            f'(CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"\n')


def rl_island_ray_gpus(args, spec) -> str:
    island = rl_island_spec(args, spec)
    return "" if island is spec else f" --num-gpus={island.gpus_per_node}"


def rl_actor_gpus_per_node(args, spec) -> int:
    """Trainer GPUs per node: all of them when colocated; under
    ``--rl-placement fixed-partition`` the rest after rollout and standby."""
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return spec.gpus_per_node
    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    standby = int(getattr(args, "rl_standby_gpus", 0) or 0)
    if rollout < 1:
        raise ValueError("--rl-placement fixed-partition needs --rollout-num-gpus >= 1")
    if spec.num_nodes == 1:
        return spec.gpus_per_node - rollout - standby
    return rl_trainer_shape(args, spec)[1]


def rl_trainer_shape(args, spec) -> tuple[int, int]:
    """``(actor_num_nodes, actor_num_gpus_per_node)`` of the island's trainer
    (rl-multinode-island D5). Colocated: every GPU. Fixed partition: the island
    total minus ``--rl-rollout-gpus`` and ``--rl-standby-gpus`` (island totals),
    laid out as the leading logical bundles; it must occupy the same number of
    GPUs on every node it uses (Miles' rectangle) and keep every TP*PP group
    on one node."""
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return spec.num_nodes, spec.gpus_per_node
    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    standby = int(getattr(args, "rl_standby_gpus", 0) or 0)
    if rollout < 1:
        raise ValueError("--rl-placement fixed-partition needs --rollout-num-gpus >= 1")
    trainer = spec.total_gpus - rollout - standby
    if trainer < 1:
        raise ValueError(
            "--rl-placement fixed-partition needs --rl-rollout-gpus + --rl-standby-gpus "
            f"< island GPUs ({spec.total_gpus})"
        )
    layout = rl_island_layout(args, spec)
    if layout is not None:
        return layout[0], layout[1]
    from .rl.engine.multinode import Topology, TopologyError, rectangular_trainer

    topology = Topology(spec.num_nodes, spec.gpus_per_node)
    try:
        return rectangular_trainer([topology.slot_of(b) for b in range(trainer)])
    except TopologyError as exc:
        raise ValueError(f"--rl-placement fixed-partition on {spec}: {exc}") from None


def rl_cross_node_switches(args) -> tuple[bool, bool]:
    """``(allow_cross_node_tp, allow_cross_node_engine_tp)`` (ruling 2026-10-04 v2):
    the CLI flags ``--rl-allow-cross-node-tp`` / ``--rl-allow-cross-node-engine-tp`` OR
    the initial elastic cfg's ``parallel.allow_cross_node_*`` (set by
    ``_check_ports_infra_switches``); either spelling lifts the default preference and
    both are forwarded to the learner."""
    cfg = getattr(args, "rl_elastic_initial_cfg_switches", None) or {}
    return (bool(getattr(args, "rl_allow_cross_node_tp", False) or cfg.get("allow_cross_node_tp")),
            bool(getattr(args, "rl_allow_cross_node_engine_tp", False) or cfg.get("allow_cross_node_engine_tp")))


def rl_colocated_engine_check(args, spec) -> list[dict[str, int]]:
    """m5: node rules of a colocated multi-node island's rollout engines (ruling
    2026-10-04 v2): engine i = logical bundles ``i*g .. i*g+g-1`` of the island; one
    node by default, a cross-node engine only with ``--rl-allow-cross-node-engine-tp``
    and as whole nodes. Returns the expected sglang TP rank map of engine 0 (the
    m5 judge's rank -> GPU reconciliation reference); raises ValueError otherwise."""
    from .rl.engine.multinode import (Topology, TopologyError, colocated_engine_slots,
                                      node_placement_rejection, sglang_tp_rank_map)

    g = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    topology = Topology(spec.num_nodes, spec.gpus_per_node)
    _tp, cross_engine = rl_cross_node_switches(args)
    try:
        engines = colocated_engine_slots(topology.total, g, topology)
    except TopologyError as exc:
        raise ValueError(f"colocated rollout engines on {spec}: {exc}") from None
    why = node_placement_rejection({"trainer": [], "rollout": engines, "standby": []},
                                   gpus_per_engine=g, allow_cross_node_engine=cross_engine,
                                   gpus_per_node=spec.gpus_per_node)
    if why:
        raise ValueError(f"colocated rollout engines on {spec}: {why}")
    if cross_engine and g > spec.gpus_per_node:
        print("[launcher] WARNING: --rl-allow-cross-node-engine-tp: a colocated rollout engine of "
              f"{g} GPUs spans {g // spec.gpus_per_node} whole nodes (sglang nnodes>1; scales "
              "as one replica)")
    return sglang_tp_rank_map(engines[0])


def rl_cross_node_flags(args) -> str:
    tp, engine = rl_cross_node_switches(args)
    return _rl_backend_module(args, "launch_flags").cross_node_flags(tp, engine)


def rl_island_layout(args, spec) -> tuple[int, int, dict[str, tuple[int, ...]]] | None:
    """Q2 (2026-10-04 ruling, mixed rollout/trainer nodes): the trainer shape and
    role -> logical bundle map derived from the ``--rl-elastic-initial-config``'s
    explicit ``placement`` (``args.rl_elastic_initial_placement_slots``, set by
    ``_check_ports_infra_switches``), or None when the manifest declares no
    placement (the leading-bundle layout of :func:`rl_trainer_shape` stays).
    The placement must match ``--gpu`` / ``--rl-rollout-gpus`` / ``--rl-standby-gpus``
    (island totals), be a rectangle of contiguous per-node runs
    (``multinode.trainer_layout``) and agree with ``PlacementRequest.trainer_shape``
    (the learner-side rule set); any disagreement fails here, before any cloud work."""
    slots = getattr(args, "rl_elastic_initial_placement_slots", None)
    if slots is None or getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return None
    PlacementRequest = _rl_backend_module(args, "placement").PlacementRequest
    from .rl.engine.multinode import Topology, TopologyError, trainer_layout

    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    standby = int(getattr(args, "rl_standby_gpus", 0) or 0)
    name = getattr(args, "rl_elastic_initial_config", None)
    counts = (len(slots["trainer"]), sum(len(e) for e in slots["rollout"]), len(slots["standby"]))
    if counts[1] != rollout or counts[2] != standby or sum(counts) != spec.total_gpus:
        raise ValueError(
            f"--rl-elastic-initial-config {name!r} placement is T{counts[0]} R{counts[1]} "
            f"S{counts[2]} but the launch asks for --rl-rollout-gpus {rollout} "
            f"--rl-standby-gpus {standby} on {spec} ({spec.total_gpus} GPUs)")
    topology = Topology(spec.num_nodes, spec.gpus_per_node)
    try:
        nodes, per_node, bundle_map = trainer_layout(slots, topology)
    except TopologyError as exc:
        raise ValueError(f"--rl-elastic-initial-config {name!r} placement on {spec}: {exc}") from None
    engine = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    tp = int(getattr(args, "tensor_parallel", 1) or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    cp = int(getattr(args, "context_parallel", 1) or 1)
    ep = int(getattr(args, "expert_parallel", 1) or 1)
    try:
        # Q1/Q3: only the in-node group tp*cp must stay on one node; PP and EP
        # groups may span nodes (M1: T2 = n0:0 + n1:0 with --pipeline-parallel 2).
        request = PlacementRequest(
            kind="fixed-partition", trainer_gpus=counts[0], rollout_gpus=rollout,
            gpus_per_engine=engine, standby_gpus=standby,
            gpus_per_node=spec.gpus_per_node if spec.num_nodes > 1 else None,
            model_parallel=tp * pp * cp, node_parallel=tp * cp, expert_parallel=ep,
            bundle_map=bundle_map, allow_cross_node_tp=rl_cross_node_switches(args)[0],
            allow_cross_node_engine_tp=rl_cross_node_switches(args)[1])
        shape = request.trainer_shape()
    except ValueError as exc:
        raise ValueError(f"--rl-elastic-initial-config {name!r} placement on {spec}: {exc}") from None
    if shape != (nodes, per_node):
        raise ValueError(
            f"--rl-elastic-initial-config {name!r}: launcher trainer shape {(nodes, per_node)} "
            f"disagrees with PlacementRequest.trainer_shape {shape}")
    return nodes, per_node, bundle_map


def rl_island_bundle_map_flag(args, spec) -> str:
    """`` --rl-island-bundle-map JSON`` for the learner when the cfg placement differs
    from the leading-bundle layout; "" otherwise (single node and whole-node
    trainer layouts are byte-identical to the pre-Q2 launcher)."""
    layout = rl_island_layout(args, spec)
    if layout is None:
        return ""
    from .rl.engine.multinode import leading_bundle_map

    bundle_map = layout[2]
    counts = tuple(len(bundle_map[r]) for r in ("trainer", "rollout", "standby"))
    if bundle_map == leading_bundle_map(*counts):
        return ""
    return _rl_backend_module(args, "launch_flags").bundle_map_flag(bundle_map)


def rl_min_nodes(args, spec) -> int:
    """Minimum learner nodes (rl-multinode-island D8): one trainer model-parallel
    replica + one rollout engine + standby, or ``--rl-min-nodes-per-learner``
    when that is larger."""
    from .rl.engine.multinode import TopologyError, min_nodes, trainer_replica_gpus

    explicit = int(getattr(args, "rl_min_nodes_per_learner", 0) or 0)
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return max(1, explicit)
    tp = int(getattr(args, "tensor_parallel", 1) or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    cp = int(getattr(args, "context_parallel", 1) or 1)
    ep = int(getattr(args, "expert_parallel", 1) or 1)
    etp = int(getattr(args, "expert_tensor_parallel", 1) or 1)  # the learner pins --expert-tensor-parallel-size 1
    engine = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    try:
        # Q1/Q3 ruling 2026-10-04: the smallest replica may span nodes (EP/PP
        # cross nodes); only tp*cp must stay inside a node. EP shares ranks with
        # TP x DP (trainer_replica_gpus), it is not a further multiplier.
        replica = trainer_replica_gpus({"tp": tp, "cp": cp, "pp": pp, "ep": ep, "etp": etp})
        cross_tp, cross_engine = rl_cross_node_switches(args)
        derived = min_nodes(trainer_min_gpus=replica, rollout_min_gpus=engine,
                            standby_gpus=int(getattr(args, "rl_standby_gpus", 0) or 0),
                            gpus_per_node=spec.gpus_per_node, node_parallel=tp * cp,
                            allow_cross_node_tp=cross_tp, allow_cross_node_engine=cross_engine)
    except TopologyError as exc:
        raise ValueError(f"recipe parallelism does not fit {spec}: {exc}") from None
    return max(derived, explicit)


def build_syncer_binary() -> Path:
    binary = REPO_ROOT / "syncer/target/release/yeto-syncer"
    print("[launcher] building syncer (cargo build --release)...")
    subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT / "syncer", check=True)
    return binary


# The syncer's JSONL merge log. Head-controller mode tails it into W&B
# (yeto.wandb_tape), so the path has to be the same on both sides — and RL
# runs put it under the output dir they collect, so it varies by mode.
SYNCER_EVENT_TAPE = "~/yeto-tape.jsonl"
RL_SYNCER_EVENT_TAPE = "~/yeto-output/yeto-tape.jsonl"
RL_CRITIC_SYNCER_EVENT_TAPE = "~/yeto-output/yeto-critic-tape.jsonl"
RL_CRITIC_SYNCER_CHECKPOINT = "~/yeto-output/yeto-critic-state.ckpt"


def rl_needs_critic(args) -> bool:
    """True iff the prepared RL algorithm spec trains a critic (execution.needs_critic)."""
    if getattr(args, "training_mode", "sft") != "rl":
        return False
    raw = getattr(args, "rl_algorithm_spec_json", None)
    if not raw:
        return False
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return False
    return bool(((payload or {}).get("execution") or {}).get("needs_critic"))


def critic_syncer_address(syncer_addr: str) -> str:
    """The critic syncer runs on the actor syncer's host, on CRITIC_SYNCER_PORT."""
    if syncer_addr == "$SYNCER_ADDR":
        return "$CRITIC_SYNCER_ADDR"
    host, _, _port = syncer_addr.rpartition(":")
    if not host:
        raise ValueError(f"syncer address {syncer_addr!r} has no host:port")
    return f"{host}:{CRITIC_SYNCER_PORT}"


def critic_syncer_command(args, num_learners: int, binary: str = "~/yeto-syncer") -> str:
    """The critic channel: the same strict RL syncer invocation as the actor's
    (rounds, quorum, outer optimizer), with its own port, checkpoint and tape."""
    return (
        syncer_command(args, num_learners, binary=binary, critic=False)
        .replace(f" --port {SYNCER_PORT}", f" --port {CRITIC_SYNCER_PORT}", 1)
        .replace("~/yeto-output/yeto-state.ckpt", RL_CRITIC_SYNCER_CHECKPOINT)
        .replace(f" --event-tape {RL_SYNCER_EVENT_TAPE}", f" --event-tape {RL_CRITIC_SYNCER_EVENT_TAPE}", 1)
        .replace("mkdir -p ~/yeto-output && ", "", 1)
    )


def syncer_event_tape(args) -> str:
    """Where the syncer writes its merge record for this run's mode.

    syncer_command, LocalSyncer's strict-failure reader, and the W&B tape
    forwarder all resolve the path through here, so none of them can drift
    onto a file the syncer is not writing.
    """
    if getattr(args, "training_mode", "sft") == "rl":
        return RL_SYNCER_EVENT_TAPE
    return SYNCER_EVENT_TAPE


def _resume_if_exists(checkpoint: str) -> str:
    """Shell fragment adding --resume only when the checkpoint file exists.

    The syncer refuses --resume without a checkpoint, so an unconditional
    flag kills every fresh run; evaluated by the shell at each (re)start,
    a restart after the first checkpoint still resumes from it."""
    return f"$(test -f {checkpoint} && echo --resume)"


def _syncer_quorum_timeout(args) -> str:
    """``--rl-elastic-quorum-timeout-s`` (3.8): the syncer's --quorum-timeout-s,
    the same value the island's pause budget uses; "" by default (syncer 900 s)."""
    value = getattr(args, "rl_elastic_quorum_timeout_s", None)
    if value is None or not getattr(args, "rl_elastic", False):
        return ""
    if float(value) != int(value):
        raise ValueError("--rl-elastic-quorum-timeout-s must be whole seconds (syncer u64)")
    return f" --quorum-timeout-s {int(value)}"


_ISLAND_SCHEDULING_PARAMS = (
    ("rl_quorum_theta", "--rl-quorum-theta"),
    ("rl_carry_gamma", "--rl-carry-gamma"),
    ("rl_soft_deadline_s", "--rl-soft-deadline-s"),
    ("rl_q_min", "--rl-q-min"),
    ("rl_max_carry_lag", "--rl-max-carry-lag"),
    ("rl_syncer_epoch", "--rl-syncer-epoch"),
    ("rl_island_lease_s", "--rl-island-lease-s"),
    ("rl_final_grace_s", "--rl-final-grace-s"),
)

ELASTIC_DEBUG_DELAY_ENV = "YETO_RL_ELASTIC_DEBUG_DELAY"


def elastic_debug_delay_env(args) -> dict[str, str]:
    """0.25 test switch: islands delay each DELTA_TENSOR (soft deadline / carried_over
    on real hardware). Elastic only; {} by default."""
    spec = getattr(args, "rl_elastic_debug_delay_s", None)
    if not spec:
        return {}
    if island_scheduling_mode(args) != "elastic":
        raise ValueError("--rl-elastic-debug-delay-s needs --rl-island-scheduling elastic")
    from .rl.elastic_client import parse_elastic_debug_delay

    parse_elastic_debug_delay(spec)
    return {ELASTIC_DEBUG_DELAY_ENV: str(spec)}


ELASTIC_DEBUG_PAUSE_ENV = "YETO_RL_ELASTIC_DEBUG_PAUSE"


def elastic_debug_pause_env(args) -> dict[str, str]:
    """Test switch: the given island's link to the syncer goes silent (no heartbeat,
    sends/receives held) for S seconds after base AFTER_V; the process keeps running.
    Used to exercise lease expiry -> "rejoin required" -> automatic re-JOIN on real
    hardware. Elastic only; {} by default."""
    spec = getattr(args, "rl_elastic_debug_pause", None)
    if not spec:
        return {}
    if island_scheduling_mode(args) != "elastic":
        raise ValueError("--rl-elastic-debug-pause needs --rl-island-scheduling elastic")
    from .rl.elastic_client import parse_elastic_debug_pause

    parse_elastic_debug_pause(spec)
    return {ELASTIC_DEBUG_PAUSE_ENV: str(spec)}


DEFAULT_ISLAND_LEASE_S = 300.0  # 0.22: 1b expired a 30 s lease during a 60-90 s round


def island_lease_s(args) -> float:
    value = getattr(args, "rl_island_lease_s", None)
    value = DEFAULT_ISLAND_LEASE_S if value is None else float(value)
    if not value > 0:
        raise ValueError("--rl-island-lease-s must be > 0")
    return value


def island_syncer_epoch(args) -> int:
    value = getattr(args, "rl_syncer_epoch", None)
    value = 0 if value is None else int(value)
    if value < 0:
        raise ValueError("--rl-syncer-epoch must be >= 0")
    return value


def syncer_status_reader(*, local_tape: str | None = None, ssh_host: str | None = None):
    """0.17: a zero-argument reader of the elastic syncer's status.json (next to its
    --event-tape): a local file on a head node, else ``ssh <syncer cluster> cat``.
    Returns the parsed snapshot, or None when absent / not the elastic schema."""
    from .rl.engine.island_status import read_syncer_status, STATUS_SCHEMA

    if local_tape is not None:
        return lambda: read_syncer_status(os.path.expanduser(local_tape))

    remote = RL_SYNCER_EVENT_TAPE.rsplit("/", 1)[0] + "/status.json"

    def read() -> dict | None:
        try:
            out = subprocess.run(["ssh", "-o", "BatchMode=yes", ssh_host, f"cat {remote}"],
                                 capture_output=True, text=True, timeout=30)
            raw = json.loads(out.stdout) if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        return raw if isinstance(raw, dict) and raw.get("schema") == STATUS_SCHEMA else None
    return read


def resolve_island_syncer_epoch(args, read_status, *, timeout_s: float = 120.0,
                                poll_s: float = 2.0, sleep=time.sleep, clock=time.monotonic) -> int:
    """0.17: elastic without an explicit --rl-syncer-epoch takes the term the running
    syncer reports in status.json (and records it on ``args`` so every island
    command carries it). Unreadable within ``timeout_s``: keep 0 with a warning.
    Legacy, or an explicit value: unchanged, nothing is read."""
    if island_scheduling_mode(args) != "elastic" or getattr(args, "rl_syncer_epoch", None) is not None:
        return island_syncer_epoch(args)
    deadline = clock() + timeout_s
    while True:
        status = read_status()
        epoch = status.get("syncer_epoch") if status else None
        if isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 0:
            args.rl_syncer_epoch = epoch
            print(f"[launcher] syncer_epoch {epoch} read from the syncer's status.json")
            return epoch
        if clock() >= deadline:
            print("[launcher] WARN: could not read syncer_epoch from the syncer's status.json "
                  f"within {timeout_s:.0f}s; islands JOIN with syncer_epoch 0")
            return 0
        sleep(poll_s)


def island_scheduling_mode(args) -> str:
    """``--rl-island-scheduling`` (rl-inter-island-scheduling 0.13); legacy unless
    elastic is given explicitly. Elastic-only parameters with legacy are refused."""
    mode = getattr(args, "rl_island_scheduling", None) or "legacy"
    if mode not in ("legacy", "elastic"):
        raise ValueError(f"--rl-island-scheduling must be legacy|elastic, got {mode!r}")
    if mode == "legacy":
        given = [f for n, f in _ISLAND_SCHEDULING_PARAMS if getattr(args, n, None) is not None]
        if given:
            raise ValueError(", ".join(given) + " need --rl-island-scheduling elastic")
    return mode


ISLAND_HMAC_KEY_ENV = "YETO_ISLAND_HMAC_KEY"


def island_hmac_secret(args) -> dict[str, str]:
    """elastic only (0.20): ``{YETO_ISLAND_HMAC_KEY: key}`` read from the launching
    environment, to be shipped as a secret (sky ``secrets=``, Modal ``cfg.envs`` which
    becomes a ``modal.Secret``) -- never spliced into a command line, log or
    process list. ``{}`` in legacy."""
    if island_scheduling_mode(args) == "legacy":
        return {}
    key = os.environ.get(ISLAND_HMAC_KEY_ENV, "")
    if not key:
        raise ValueError(f"--rl-island-scheduling elastic needs {ISLAND_HMAC_KEY_ENV} "
                         "in the launching environment (island message HMAC key)")
    return {ISLAND_HMAC_KEY_ENV: key}


def _syncer_island_hmac_export(args) -> str:
    """elastic only: fail fast in the syncer's shell when the secret did not reach
    it. The value itself arrives as a secret env (island_hmac_secret) and the
    syncer reads the env var. "" in legacy, so the legacy line is unchanged."""
    if not island_hmac_secret(args):
        return ""
    return f': "${{{ISLAND_HMAC_KEY_ENV}:?island HMAC key secret missing}}" && '


def _syncer_island_scheduling(args) -> str:
    """Syncer flags for the island-scheduling mode: "" in legacy, so the legacy
    command line is byte-for-byte the previous one. The syncer folds these into
    its session contract hash (the Python side does not compute that hash)."""
    if island_scheduling_mode(args) == "legacy":
        return ""
    from .rl.engine.island_ledger import (DEFAULT_GAMMA, DEFAULT_MAX_CARRY_LAG,
                                          DEFAULT_QUORUM_MIN, DEFAULT_THETA)

    theta = getattr(args, "rl_quorum_theta", None)
    gamma = getattr(args, "rl_carry_gamma", None)
    soft = getattr(args, "rl_soft_deadline_s", None)
    if soft is None:
        soft = getattr(args, "rl_elastic_quorum_timeout_s", None) or 900
    q_min = getattr(args, "rl_q_min", None)
    lag = getattr(args, "rl_max_carry_lag", None)
    theta = DEFAULT_THETA if theta is None else float(theta)
    gamma = DEFAULT_GAMMA if gamma is None else float(gamma)
    q_min = DEFAULT_QUORUM_MIN if q_min is None else int(q_min)
    lag = DEFAULT_MAX_CARRY_LAG if lag is None else int(lag)
    if not 0.0 < theta <= 1.0 or not 0.0 <= gamma <= 1.0 or q_min < 1 or lag < 0:
        raise ValueError("--rl-quorum-theta in (0,1], --rl-carry-gamma in [0,1], "
                         "--rl-q-min >= 1, --rl-max-carry-lag >= 0")
    if float(soft) != int(soft) or int(soft) <= 0:
        raise ValueError("--rl-soft-deadline-s must be positive whole seconds")
    return (" --island-scheduling-mode elastic"
            f" --quorum-theta {theta!r}"
            f" --carry-gamma {gamma!r}"
            f" --soft-deadline-s {int(soft)}"
            f" --q-min {q_min}"
            f" --max-carry-lag {lag}"
            f" --syncer-epoch {island_syncer_epoch(args)}"
            f" --island-lease-s {island_lease_s(args)!r}"
            f" --final-grace-s {_final_grace_s(args, int(soft))}")


def _final_grace_s(args, soft_deadline_s: int) -> int:
    """0.26: the syncer's FINISHED window (whole seconds); default the soft deadline."""
    value = getattr(args, "rl_final_grace_s", None)
    value = soft_deadline_s if value is None else int(value)
    if value < 0:
        raise ValueError("--rl-final-grace-s must be >= 0")
    return value


def syncer_command(args, num_learners: int, binary: str = "~/yeto-syncer",
                   critic: bool | None = None) -> str:
    """The syncer invocation shared by the syncer-cluster task (local
    controller mode) and the head-node subprocess (head controller mode).
    --resume makes any restart pick up from the on-disk checkpoint.

    A critic algorithm (4.2.1) backgrounds the critic syncer first; the actor
    syncer stays the foreground process (the job's health)."""
    if getattr(args, "training_mode", "sft") == "rl":
        total_steps = getattr(args, "rl_total_fragment_steps", args.total_steps)
        if critic is None:
            critic = rl_needs_critic(args)
        critic_prefix = (
            f"{{ {critic_syncer_command(args, num_learners, binary=binary)} & }} && "
            if critic else ""
        )
        return (
            "mkdir -p ~/yeto-output && "
            + _syncer_island_hmac_export(args)
            + critic_prefix
            + f"{binary}"
            f" --port {SYNCER_PORT}"
            f" --learners {num_learners}"
            f" --quorum {args.quorum}"
            f" --grace-ms {args.grace_ms}"
            f" --grace-gamma {args.grace_gamma}"
            f" --grace-tau {args.grace_tau}"
            f" --pipeline {args.pipeline}"
            f" --sync-interval-steps {args.sync_interval_steps}"
            f" --delta-correction {args.delta_correction}"
            f" --total-steps {total_steps}"
            f"{_syncer_quorum_timeout(args)}"
            f"{_syncer_island_scheduling(args)}"
            f" --outer-lr {args.outer_lr}"
            f" --outer-momentum {args.outer_momentum}"
            " --max-base-lag 0 --learner-weight equal"
            " --checkpoint-path ~/yeto-output/yeto-state.ckpt"
            " --checkpoint-every 1"
            f" {_resume_if_exists('~/yeto-output/yeto-state.ckpt')}"
            f" --event-tape {RL_SYNCER_EVENT_TAPE}"
        )
    if island_scheduling_mode(args) != "legacy":
        raise ValueError("--rl-island-scheduling elastic only applies to --training-mode rl")
    return (
        f"{binary}"
        f" --port {SYNCER_PORT}"
        f" --learners {num_learners}"
        f" --quorum {args.quorum}"
        f" --grace-ms {args.grace_ms}"
        f" --grace-gamma {args.grace_gamma}"
        f" --grace-tau {args.grace_tau}"
        f" --pipeline {getattr(args, 'pipeline', 2)}"
        f" --sync-interval-steps {getattr(args, 'sync_interval_steps', 24.0)}"
        f" --delta-correction {args.delta_correction}"
        f" --total-steps {args.total_steps}"
        f" --outer-lr {args.outer_lr}"
        f" --outer-momentum {args.outer_momentum}"
        " --checkpoint-path ~/yeto-state.ckpt"
        f" {_resume_if_exists('~/yeto-state.ckpt')}"
        f" --mark-final-checkpoint"
        f" --event-tape {SYNCER_EVENT_TAPE}"
    )


# The syncer VM is x86_64 Linux; a binary built on any other submitting
# machine (macOS arm64 in particular) is an Exec-format-error away from a
# silent dead fleet, so cross builds happen ON the VM from the synced repo.
# rustup + a release build of the syncer adds ~2-4 min to syncer provision.
SYNCER_REMOTE_BUILD = (
    'if ! command -v cc >/dev/null; then sudo apt-get update -qq && '
    "sudo apt-get install -y -qq build-essential; fi\n"
    "command -v ~/.cargo/bin/cargo >/dev/null || "
    "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -q\n"
    "~/.cargo/bin/cargo build --release --quiet "
    "--manifest-path ~/sky_workdir/syncer/Cargo.toml\n"
    "cp ~/sky_workdir/syncer/target/release/yeto-syncer ~/yeto-syncer"
)


def syncer_tape_sidecar(args, num_learners: int, binary: str = "~/yeto-syncer") -> tuple[str, str, dict]:
    """Setup step, run prefix, and envs that put a W&B tape forwarder on the
    syncer VM. Returns empty strings when telemetry is off.

    Local controller mode runs the syncer on its own cluster, so the event
    tape is not reachable from the controller on the submitting machine.
    Rather than shipping the tape back, the reader goes to the tape:
    `yeto.wandb_tape --follow` runs beside the syncer.

    It is backgrounded deliberately. The syncer must stay the job's
    foreground process, because FleetController reads that job's exit code
    as the syncer's health; and as a separate process the forwarder cannot
    take the syncer down with it. The forwarder resumes from its stored
    byte offset, so a restarted job continues the tape instead of logging
    every past merge a second time.

    Only yeto.wandb_tape and yeto.wandb_logger are imported over there, and
    both are stdlib-only — the syncer VM never needs the training stack.
    """
    if not getattr(args, "wandb", False):
        return "", "", {}
    setup = (
        "\npip install -q wandb || echo '[yeto-setup] wandb install failed; "
        "the syncer will run without telemetry' >&2"
    )
    forwarder = (
        f"python3 -m yeto.wandb_tape {syncer_event_tape(args)} --follow"
        f" --wandb-group {shlex.quote(args.cluster_prefix)}"
        f" --wandb-project {shlex.quote(args.wandb_project)}"
        f" --wandb-mode {args.wandb_mode}"
    )
    if getattr(args, "wandb_entity", None):
        forwarder += f" --wandb-entity {shlex.quote(args.wandb_entity)}"
    # Head mode's forwarder is handed the launch namespace directly; the
    # sidecar is a separate process, so the same config rides across as JSON
    # (the head job itself is launched the same way, see cli.cmd_launch_head).
    # build_config drops anything credential-shaped before it is serialized.
    from .wandb_logger import build_config

    config = build_config(
        args, {"syncer_command": syncer_command(args, num_learners, binary=binary)}
    )
    forwarder += f" --config-json {shlex.quote(json.dumps(config, sort_keys=True))}"
    run_prefix = (
        "(cd ~/sky_workdir && PYTHONPATH=~/sky_workdir nohup "
        f"{forwarder} >/tmp/yeto-tape-wandb.log 2>&1 &) || true\n"
    )
    envs = {"YETO_RUN_GROUP": args.cluster_prefix}
    if os.environ.get("WANDB_API_KEY"):
        envs["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    return setup, run_prefix, envs


# Clouds where sky cannot open ports (sky/clouds/verda.py: no OPEN_PORTS).
# There the syncer's port is reachable because the platform has no
# firewall; yeto closes everything but SSH and the syncer with ufw inside
# the VM and proves reachability with an external TCP probe (design D7).
NO_OPEN_PORTS_CLOUDS = frozenset({"verda"})
PROBE_BANNER = b"yeto-probe"


def syncer_cloud(args) -> str:
    """Cloud of --syncer-region ('region' means AWS, else 'cloud/region')."""
    region = getattr(args, "syncer_region", "") or ""
    return region.split("/", 1)[0] if "/" in region else "aws"


def syncer_ports(args) -> list[int] | None:
    """`ports=` for the syncer/head resources; None where sky can't open them."""
    if syncer_cloud(args) in NO_OPEN_PORTS_CLOUDS:
        return None
    return [SYNCER_PORT, CRITIC_SYNCER_PORT] if rl_needs_critic(args) else [SYNCER_PORT]


def ufw_setup(port: int = SYNCER_PORT) -> str:
    """In-VM firewall for a public syncer on a cloud without security
    groups: deny inbound except SSH and the syncer port."""
    return (
        'SUDO=""; [ "$(id -u)" = 0 ] || SUDO=sudo\n'
        "command -v ufw >/dev/null || { $SUDO apt-get update -qq && $SUDO apt-get install -y -qq ufw; }\n"
        "$SUDO ufw default deny incoming\n"
        "$SUDO ufw default allow outgoing\n"
        "$SUDO ufw allow 22/tcp\n"
        f"$SUDO ufw allow {int(port)}/tcp\n"
        "$SUDO ufw --force enable\n"
        "$SUDO ufw status verbose"
    )


def probe_listener_command(port: int = SYNCER_PORT, timeout_s: int = 300) -> str:
    """A one-shot listener on the syncer port that greets the first
    connection with PROBE_BANNER (used before the real syncer starts)."""
    code = (
        "import socket;s=socket.socket();"
        "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
        f"s.bind(('0.0.0.0',{int(port)}));s.listen(1);s.settimeout({int(timeout_s)});"
        f"c,_=s.accept();c.sendall({PROBE_BANNER!r});c.close();s.close()"
    )
    return f'python3 -c "{code}"'


def tcp_probe(host: str, port: int = SYNCER_PORT, *, expect: bytes | None = None, attempts: int = 20,
              delay: float = 3.0, timeout: float = 5.0, connect=None, sleep=time.sleep) -> tuple[bool, str]:
    """Connect from THIS machine (outside the cloud) to host:port.
    With `expect`, the peer must send those bytes first. (ok, detail)."""
    import socket

    connect = connect or socket.create_connection
    last = "not tried"
    for i in range(max(1, attempts)):
        try:
            with connect((host, int(port)), timeout=timeout) as conn:
                if expect is not None:
                    conn.settimeout(timeout)
                    got = conn.recv(len(expect))
                    if got != expect:
                        raise OSError(f"unexpected greeting {got!r}")
            return True, f"{host}:{port} reachable (attempt {i + 1})"
        except OSError as e:
            last = f"{type(e).__name__}: {e}"
        if i + 1 < attempts:
            sleep(delay)
    return False, f"{host}:{port} unreachable after {attempts} attempt(s): {last}"


def island_syncer_addrs(envs) -> str:
    """The syncer address(es) an island dials, for launch.log (no credentials)."""
    envs = envs or {}
    return " ".join(f"{k}={envs[k]}" for k in ("SYNCER_ADDR", "CRITIC_SYNCER_ADDR") if k in envs) or "no syncer"


def probe_syncer_ports(args, host: str, *, probe=None) -> list[str]:
    """Fail fast, before any island is started, unless every syncer port
    (actor 29400 and, for a critic algorithm, critic 29401) accepts a TCP
    connection from this machine. On clouds where sky opens the ports
    (security groups, e.g. Nebius) this is the only check that the rule
    really took effect: s13-g3-modal-20261007b spent the 900 s stall timeout
    with both islands silently redialing a Nebius syncer that never saw a
    connection. A pass here does not prove the islands' egress (Modal), only
    that the head side is open."""
    probe = probe or tcp_probe
    ports = [SYNCER_PORT, CRITIC_SYNCER_PORT] if rl_needs_critic(args) else [SYNCER_PORT]
    details = []
    for port in ports:
        ok, detail = probe(host, port)
        if not ok:
            raise RuntimeError(
                f"syncer port {port} not reachable from outside {syncer_cloud(args)} ({detail}); "
                "no island was started"
            )
        print(f"[launcher] syncer probe: {detail}", flush=True)
        details.append(detail)
    return details


def make_syncer_task(args, num_learners: int):
    import platform

    import sky

    cross = (platform.system(), platform.machine()) != ("Linux", "x86_64")
    if cross:
        tape_setup, tape_run, tape_envs = syncer_tape_sidecar(args, num_learners)
        print("[launcher] non-x86-Linux submitter: building the syncer on the syncer VM")
        task = sky.Task(
            name="yeto-syncer",
            setup=WAN_TUNING + "\n" + SYNCER_REMOTE_BUILD + tape_setup,
            run=tape_run + syncer_command(args, num_learners),
            workdir=str(REPO_ROOT),
            envs=tape_envs,
            secrets=island_hmac_secret(args) or None,
        )
        infra = args.syncer_region if "/" in args.syncer_region else f"aws/{args.syncer_region}"
        task.set_resources(
            sky.Resources(
                infra=infra,
                cpus="8+",
                memory=f"{args.syncer_memory}+",
                ports=syncer_ports(args),
                use_spot=False,
            )
        )
        if syncer_ports(args) is None:
            task.setup = ufw_setup() + "\n" + (task.setup or "")
        return task

    binary = build_syncer_binary()
    tape_setup, tape_run, tape_envs = syncer_tape_sidecar(args, num_learners)
    cmd = tape_run + "chmod +x ~/yeto-syncer && " + syncer_command(args, num_learners)
    task = sky.Task(
        name="yeto-syncer",
        setup=WAN_TUNING + tape_setup,
        run=cmd,
        file_mounts={"~/yeto-syncer": str(binary)},
        # The fast path normally ships only the binary; the forwarder needs
        # the yeto package, so telemetry pulls in the same workdir the
        # cross-build path already uses.
        workdir=str(REPO_ROOT) if getattr(args, "wandb", False) else None,
        envs=tape_envs,
        secrets=island_hmac_secret(args) or None,
    )
    # --syncer-region accepts "region" (AWS assumed) or "cloud/region".
    infra = args.syncer_region if "/" in args.syncer_region else f"aws/{args.syncer_region}"
    task.set_resources(
        sky.Resources(
            infra=infra,
            cpus="8+",
            memory=f"{args.syncer_memory}+",
            ports=syncer_ports(args),
            use_spot=False,
        )
    )
    if syncer_ports(args) is None:
        task.setup = ufw_setup() + "\n" + (task.setup or "")
    return task


PICKLED_LOSS_FILE = ".yeto_loss.pkl"
PICKLED_LOSS_PREFIX = ".yeto_loss."


def pickled_loss_path(spec: str) -> Path:
    """Resolve a pickle spec, including a staged workdir-relative payload."""

    if not spec.startswith("pickle:"):
        raise ValueError(f"not a pickle loss spec: {spec!r}")
    source_text = spec.split(":", 1)[1]
    source_name = Path(source_text).name
    staged = (
        source_text == source_name
        and (
            source_name == PICKLED_LOSS_FILE
            or (
                source_name.startswith(PICKLED_LOSS_PREFIX)
                and source_name.endswith(".pkl")
            )
        )
    )
    return REPO_ROOT / source_name if staged else Path(source_text).expanduser()


def _stage_pickled_loss(payload: bytes) -> str:
    """Content-address a legacy executable payload to avoid cross-run races."""

    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    filename = f"{PICKLED_LOSS_PREFIX}{digest}.pkl"
    destination = REPO_ROOT / filename
    if destination.is_symlink():
        raise RuntimeError(
            f"refusing symlink at content-addressed pickle path {destination}"
        )
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(
                f"content-addressed pickle collision at {destination}"
            )
    else:
        temporary = destination.with_name(
            f"{PICKLED_LOSS_PREFIX}{digest}.tmp-{os.getpid()}.pkl"
        )
        temporary.write_bytes(payload)
        os.chmod(temporary, 0o600)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return f"pickle:{filename}"


def resolve_loss_function(
    loss_function,
    *,
    allow_unsafe_pickled_loss: bool = False,
) -> str:
    """Return the --loss-function string to pass to learners.

    A callable or a ``custom:<file.py>`` spec can be shipped through the
    legacy pickle lane only after the explicit unsafe opt-in.  Named losses
    pass through. All pickle inputs are copied to a content-addressed,
    shell-neutral workdir path and attested before learner execution.
    """
    from .losses import load_custom_loss

    if callable(loss_function):
        fn = loss_function
    elif isinstance(loss_function, str) and loss_function.startswith("custom:"):
        if not allow_unsafe_pickled_loss:
            raise PermissionError(
                "custom loss transport uses legacy pickle and requires "
                "--allow-unsafe-pickled-loss"
            )
        fn = load_custom_loss(loss_function)
    elif isinstance(loss_function, str) and loss_function.startswith("pickle:"):
        if not allow_unsafe_pickled_loss:
            raise PermissionError(
                "legacy pickle loss transport requires "
                "--allow-unsafe-pickled-loss"
            )
        source = pickled_loss_path(loss_function)
        if not source.is_file():
            raise FileNotFoundError(f"pickled loss {str(source)!r} does not exist")
        return _stage_pickled_loss(source.read_bytes())
    else:
        return loss_function
    if not allow_unsafe_pickled_loss:
        raise PermissionError(
            "callable/custom loss transport uses legacy pickle and requires "
            "--allow-unsafe-pickled-loss"
        )
    import cloudpickle

    return _stage_pickled_loss(cloudpickle.dumps(fn))


def resolve_default_rl_image(args) -> None:
    """Fill an omitted ``--rl-image`` with the engine's pinned default."""

    if getattr(args, "training_mode", "sft") != "rl":
        return
    from .rl.engine import backends

    try:  # decoupling 5.4: a backend without a registered adapter is refused here
        backends.get(getattr(args, "rl_backend", None) or backends.DEFAULT_BACKEND)
    except backends.UnknownBackend as exc:
        raise ValueError(str(exc)) from None
    if getattr(args, "rl_image", None) is None:
        flags = _rl_backend_module(args, "launch_flags")
        if getattr(flags, "DEFAULT_RL_IMAGE", None):  # rl-verl-backend: Modal-built engine image
            args.rl_image = flags.DEFAULT_RL_IMAGE
            return
        from .rl import default_rl_image

        args.rl_image = default_rl_image(getattr(args, "rl_engine", "ports"))


def prepare_launch_args(
    args,
    *,
    allow_local_rl_data: bool = False,
    allow_remote_rl_model: bool = False,
) -> None:
    """Resolve immutable inputs and executable artifacts before cloud spend."""
    # --rl-allow-local-data (ports) is honoured by every caller (CLI, worker,
    # head second hop), not only the one that parsed the flag.
    allow_local_rl_data = allow_local_rl_data or bool(
        getattr(args, "rl_allow_local_data", False)
    )

    from .provenance import (
        file_sha256,
        pin_runtime_provenance,
        python_spec_path,
        python_spec_sha256,
        verify_source_tree_sha256,
    )

    args.source_sha256 = verify_source_tree_sha256(
        getattr(args, "source_sha256", None)
    )
    payload = pin_runtime_provenance(
        args, allow_remote_model=allow_remote_rl_model
    )
    args.model_requested_identifier = payload["model"]["requested_identifier"]
    args.model_requested_revision = payload["model"]["requested_revision"]
    if "dataset" in payload:
        args.data_requested_identifier = payload["dataset"]["requested_identifier"]
        args.data_requested_revision = payload["dataset"]["requested_revision"]
    from .adapter_lifecycle import (
        inspect_parent_adapter,
        prepare_parent_source,
        selected_parent,
    )

    prepare_parent_source(args)
    expected_loss_sha256 = getattr(args, "loss_sha256", None)
    args.loss_function = resolve_loss_function(
        args.loss_function,
        allow_unsafe_pickled_loss=bool(
            getattr(args, "allow_unsafe_pickled_loss", False)
        ),
    )
    if args.loss_function.startswith("pickle:"):
        actual_loss_sha256 = file_sha256(pickled_loss_path(args.loss_function))
        if (
            expected_loss_sha256 is not None
            and actual_loss_sha256 != expected_loss_sha256.lower()
        ):
            raise ValueError(
                "pickled loss SHA256 mismatch: expected "
                f"{expected_loss_sha256.lower()}, got {actual_loss_sha256}"
            )
        args.loss_sha256 = actual_loss_sha256
    elif args.loss_function.startswith("custom:"):
        args.loss_sha256 = file_sha256(args.loss_function.split(":", 2)[1])
    else:
        args.loss_sha256 = None
    _parent_mode, parent_source = selected_parent(args)
    if parent_source is not None:
        from .datasource import kind

        if kind(parent_source) == "local":
            # Fail on malformed artifacts and recipe drift on the submitter
            # (and again on every learner), before provisioning GPU capacity.
            inspect_parent_adapter(
                args,
                expected_sha256=args.adapter_sha256,
            )
    adapter_spec = getattr(args, "diffusion_adapter", None)
    if adapter_spec:
        adapter_path = python_spec_path(adapter_spec, base_dir=REPO_ROOT)
        try:
            relative_adapter_path = adapter_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"diffusion adapter source {adapter_path} is outside the synced "
                "Yeto workdir; copy it into the repository before launch"
            ) from exc
        target, separator, factory_name = adapter_spec.partition(":")
        if target.endswith(".py") or os.path.sep in target:
            args.diffusion_adapter = (
                f"{relative_adapter_path.as_posix()}{separator}{factory_name}"
            )
            adapter_sha256 = file_sha256(adapter_path)
        else:
            adapter_sha256 = python_spec_sha256(adapter_spec)
        expected_adapter_sha256 = getattr(args, "diffusion_adapter_sha256", None)
        if (
            expected_adapter_sha256 is not None
            and adapter_sha256 != expected_adapter_sha256.lower()
        ):
            raise ValueError(
                "diffusion adapter SHA256 mismatch: expected "
                f"{expected_adapter_sha256.lower()}, got {adapter_sha256}"
            )
        args.diffusion_adapter_sha256 = adapter_sha256
    resolve_default_rl_image(args)
    if getattr(args, "gpu", None):
        check_cloud_prerequisites(parse_gpu_spec(args.gpu), args=args)
    _prepare_rl_args(
        args,
        allow_local_data=allow_local_rl_data,
        allow_remote_model=allow_remote_rl_model,
    )


# Where each cloud's own tooling stores credentials on the submitting
# machine, and the env vars that stand in for the file. The head VM gets
# the files mounted (or the vars forwarded) for every cloud its fleet
# touches — and nothing for clouds it does not.
CLOUD_CREDENTIAL_PATHS: dict[str, tuple[str, ...]] = {
    "aws": ("~/.aws",),
    "gcp": ("~/.config/gcloud",),
    "runpod": ("~/.runpod",),
    "nebius": ("~/.nebius",),
    "verda": ("~/.verda",),
    "modal": ("~/.modal.toml",),
}
CLOUD_CREDENTIAL_ENV: dict[str, tuple[str, ...]] = {
    "aws": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    "runpod": ("RUNPOD_API_KEY",),
    "nebius": ("NEBIUS_IAM_TOKEN", "NEBIUS_TENANT_ID"),
    "verda": ("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET"),
    "modal": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
}
# Everything a learner island may legitimately receive; anything in
# CLOUD_CREDENTIAL_ENV or CLOUD_CREDENTIAL_PATHS must never reach one.
CLOUD_CREDENTIAL_ENV_NAMES = frozenset(v for vals in CLOUD_CREDENTIAL_ENV.values() for v in vals)


def head_cloud(args) -> str:
    """The cloud the head/syncer VM is placed on (`--syncer-region` is
    'region' for AWS or 'cloud/region')."""
    region = getattr(args, "syncer_region", None) or ""
    return region.split("/", 1)[0].lower() if "/" in region else "aws"


def fleet_clouds(args) -> list[str]:
    """Every cloud this launch touches: the learner islands' plus the head's."""
    clouds = [head_cloud(args)] if getattr(args, "controller", "local") == "head" else []
    if getattr(args, "gpu", None):
        clouds += [s.cloud for s in parse_gpu_spec(args.gpu)]
    return list(dict.fromkeys(clouds))


def head_cloud_credentials(clouds: list[str], environ=None) -> tuple[dict[str, str], dict[str, str]]:
    """(file_mounts, envs) that carry each cloud's credentials onto the
    head. A cloud with neither its file(s) nor its env vars present is a
    ValueError naming the expected path — before the head is submitted.
    GCP is only ever optional (gs:// output uploads)."""
    environ = os.environ if environ is None else environ
    mounts: dict[str, str] = {}
    envs: dict[str, str] = {}
    for cloud in clouds:
        paths = CLOUD_CREDENTIAL_PATHS.get(cloud)
        if paths is None:
            continue  # a cloud sky handles with no local files we know of
        present = [p for p in paths if os.path.exists(os.path.expanduser(p))]
        for p in present:
            mounts[p] = os.path.expanduser(p)
        if present:
            continue
        env_names = CLOUD_CREDENTIAL_ENV.get(cloud, ())
        if env_names and all(environ.get(v) for v in env_names):
            for v in env_names:
                envs[v] = environ[v]
            continue
        if cloud == "gcp":
            continue
        hint = f" or env {' + '.join(env_names)}" if env_names else ""
        raise ValueError(
            f"{cloud} credentials not found at {', '.join(paths)}{hint}; the head VM "
            f"needs them to launch and tear down {cloud} islands (see docs/CLOUDS.md)"
        )
    gcloud = os.path.expanduser("~/.config/gcloud")
    if os.path.isdir(gcloud):
        mounts.setdefault("~/.config/gcloud", gcloud)  # enables gs:// --output
    if "nebius" in clouds:
        # Nebius binds one project to one region and sky reads the project
        # per region from its config; the head re-runs the project check and
        # provisions Nebius islands itself, so it needs the same file.
        # (Presence is checked before submission by check_cloud_prerequisites.)
        from .shape.providers import SKY_CONFIG_PATH

        sky_config = os.path.expanduser(SKY_CONFIG_PATH)
        if os.path.isfile(sky_config):
            mounts[SKY_CONFIG_PATH] = sky_config
    return mounts, envs


def check_cloud_prerequisites(
    specs: list[ClusterSpec],
    project_ids: dict[str, str] | None = None,
    args=None,
    modal_ok: bool | None = None,
) -> None:
    """Per-cloud facts that must hold before any cloud spend.

    Nebius binds one project to one region, and sky reads the project for
    a region from `nebius.region_configs.<region>.project_id` in its
    config; a fleet naming a Nebius region without one fails at provision
    time, after the head VM is up. Fail here instead, naming every region
    that lacks a project. `project_ids` is injectable for tests.

    Modal islands have scheduling rules Modal only enforces at launch
    (whole nodes for multi-container groups), need a token on this
    machine, cannot mount object-store data, and (for RL) need the Miles
    image pinned by digest. `modal_ok` overrides the token check in tests.
    """
    nebius_regions = sorted({s.region for s in specs if s.cloud == "nebius" and s.region})
    if nebius_regions:
        from .shape.providers import SKY_CONFIG_PATH, nebius_project_ids

        have = project_ids if project_ids is not None else nebius_project_ids()
        missing = [r for r in nebius_regions if r not in have]
        if missing:
            raise ValueError(
                "Nebius needs a project per region: no project_id for "
                f"{', '.join(missing)} under nebius.region_configs in "
                f"{SKY_CONFIG_PATH} (one Nebius project is bound to one region)"
            )
    modal_specs = [s for s in specs if s.cloud == "modal"]
    store = str(getattr(args, "rl_checkpoint_store", None) or "") if args is not None else ""
    if store:
        # rl-resume-from-checkpoint 1.2: no silent drop; each cloud gets a store it can mount
        if modal_specs:
            modal_checkpoint_store(args)  # raises on a non-modal-volume store
        if store.startswith("modal-volume://") and len(modal_specs) != len(specs):
            raise ValueError("--rl-checkpoint-store modal-volume://... only mounts on Modal islands; "
                             "use a bucket URI or a shared path for the other clouds")
    if modal_specs:
        from .modal_runner import (
            image_ref_from_rl_image,
            modal_available,
            modal_credential_hint,
            validate_modal_shape,
        )

        for spec in modal_specs:
            validate_modal_shape(spec.gpu, spec.gpus_per_node, spec.num_nodes)
        if not (modal_ok if modal_ok is not None else modal_available()):
            raise ValueError(f"modal islands need a Modal token: {modal_credential_hint()}")
        if args is not None:
            if getattr(args, "training_mode", "sft") == "rl":
                resolve_default_rl_image(args)
                image_ref_from_rl_image(getattr(args, "rl_image", "") or "")
            data = getattr(args, "data", None)
            if data:
                from .datasource import kind as data_kind

                if data_kind(data) == "cloud":
                    raise ValueError(
                        f"modal islands cannot mount object-store data ({data}); "
                        "pass an HF dataset id or a local path"
                    )
            override = getattr(args, "syncer_public_addr", None)
            if override and ":" not in override:
                raise ValueError("--syncer-public-addr must be HOST:PORT")


def _rl_callable(value: str | None, flag: str, *, required: bool) -> None:
    if value is None and not required:
        return
    module, separator, function = (value or "").partition(":")
    if (
        not separator
        or not module
        or module.endswith(".py")
        or not function.isidentifier()
    ):
        raise ValueError(f"{flag} must be package.module:function")


def _rl_miles_function(
    value: str | None, flag: str = "--custom-generate-function-path"
) -> None:
    if value is None:
        return
    parts = value.split(".")
    if len(parts) < 2 or any(not part.isidentifier() for part in parts):
        raise ValueError(f"{flag} must be package.module.function")


def _rl_backend_image_ok(args) -> bool:
    """A backend may accept its own --rl-image form (verl: ``verl-build:<commit>``)."""
    check = getattr(_rl_backend_module(args, "launch_flags"), "rl_image_ok", None)
    return bool(check and check(getattr(args, "rl_image", None) or ""))


def _rl_backend_module(args, role: str):
    """Adapter module for ``role`` of the selected training backend (decoupling 5.4).

    The launcher never imports an adapter directly; ``--rl-backend`` (default
    miles) picks the row of ``yeto.rl.engine.backends``.
    """
    from .rl.engine import backends

    return backends.module(role, getattr(args, "rl_backend", None) or backends.DEFAULT_BACKEND)


def _prepare_ports_algorithm(args, rl_engine: str) -> None:
    """rl-algorithm-capabilities D8/D9/D11, before any cloud or GPU work.

    ports: build the run's AlgorithmSpec once (``--rl-algorithm-spec`` or the
    legacy CLI; the launcher has no extra Miles argv, so nothing is absorbed
    here), refuse the rejection matrix, the registered launch checks, anything
    the Miles adapter does not declare, and every unverified-mechanism
    allowance (a launched run always has outer sync, design D11); keep the
    canonical JSON + expected hash every island receives. An island whose
    learner absorbs extra argv into a different hash is refused there
    (``rl_algorithm_mismatch``). legacy: the ports-only options are refused.
    """

    from .rl.engine.algorithm import (
        AlgorithmSpecError,
        check_unverified_allowance,
        launch_problems,
        resolve_ports_algorithm,
    )
    from .rl.engine.capabilities import CapabilityMismatch

    if rl_engine != "ports" and (
        getattr(args, "rl_placement", "colocated") != "colocated"
        or (getattr(args, "rl_standby_gpus", 0) or 0) != 0
    ):
        raise ValueError("--rl-placement/--rl-standby-gpus only apply to --rl-engine ports")
    _check_ports_infra_switches(args, rl_engine)
    steps = getattr(args, "rl_optimizer_steps", 1)
    if steps is None:
        steps = 1
    if type(steps) is not int or steps < 1:
        raise ValueError(f"--rl-optimizer-steps must be a positive int (got {steps!r})")
    if steps != 1:
        if rl_engine != "ports":
            raise ValueError("--rl-optimizer-steps > 1 only applies to --rl-engine ports")
        samples = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
        if samples % steps:
            # run_config: rollout_batch_size * n_samples == global_batch * optimizer_steps
            raise ValueError(
                f"--rl-optimizer-steps {steps} must divide --rollout-batch-size * "
                f"--n-samples-per-prompt ({samples}) into equal optimizer batches"
            )
    if rl_engine != "ports" and getattr(args, "rl_single_island_no_sync", False):
        raise ValueError("--rl-single-island-no-sync only applies to --rl-engine ports")
    try:
        spec = resolve_ports_algorithm(args, rl_engine=rl_engine)
        if spec is None:
            return
        problems = spec.rejections()
        # Run-configuration checks of the follow-up changes and the Miles
        # adapter's declaration, before any cloud resource exists (the learner
        # repeats both on the island).
        problems += launch_problems(spec, {
            "rollout_batch_size": getattr(args, "rollout_batch_size", None),
            "rollout_max_response_len": getattr(args, "rollout_max_response_len", None),
            "context_parallel_size": 1,  # ports emits --context-parallel-size 1
            "multi_lora": False,
            # rl-algo-critic-family 2.3 (critic run-level rejections)
            "sync_preset": getattr(args, "rl_sync_preset", "strict-avg"),
            "elastic": bool(getattr(args, "rl_elastic", False)),
        })
        if problems:
            raise AlgorithmSpecError("algorithm spec rejected: " + "; ".join(problems))
        islands = (
            len(parse_gpu_spec(args.gpu)) if getattr(args, "gpu", None) else 1
        ) + max(0, getattr(args, "external_learners", 0) or 0)
        no_sync = bool(getattr(args, "rl_single_island_no_sync", False))
        if no_sync and (islands != 1 or getattr(args, "rl_sync_preset", "strict-avg")
                        != "strict-avg" or getattr(args, "rl_initial_adapter", None)):
            raise AlgorithmSpecError(
                "--rl-single-island-no-sync needs exactly one island (one --gpu entry, no "
                "external learners), the default sync preset and no initial adapter"
            )
        args.rl_allow_unverified_mechanism = list(
            check_unverified_allowance(
                getattr(args, "rl_allow_unverified_mechanism", None) or (),
                islands=islands,
                outer_sync=not no_sync,  # a launched run has a syncer unless no-sync
                sync_preset=getattr(args, "rl_sync_preset", "strict-avg"),
            )
        )
        _entry = _rl_backend_module(args, "entry")
        miles_capabilities, with_partitioned_serial = (
            _entry.miles_capabilities, _entry.with_partitioned_serial)

        partitioned = getattr(args, "rl_placement", "colocated") == "fixed-partition"
        capabilities = miles_capabilities(
            "sha256:" + "0" * 64, unverified_mechanisms=args.rl_allow_unverified_mechanism
        )
        if partitioned:
            capabilities = with_partitioned_serial(capabilities)
        capabilities.check(
            layout="lora",
            placement="fixed-partition" if partitioned else "colocated",
            execution_mode="partitioned-serial" if partitioned else "colocated-serial",
            algorithm=spec,
        )
    except (AlgorithmSpecError, CapabilityMismatch) as error:
        raise ValueError(str(error)) from error
    _check_critic_warmup_options(args, spec)
    args.rl_algorithm_spec_json = spec.canonical_json()
    args.rl_expected_algorithm_sha256 = spec.sha256()


def _check_critic_warmup_options(args, spec) -> None:
    """--rl-critic-load/--rl-critic-init-sha256/--rl-critic-baseline-rounds (5.3)."""

    load = getattr(args, "rl_critic_load", None)
    digest = getattr(args, "rl_critic_init_sha256", None)
    rounds = getattr(args, "rl_critic_baseline_rounds", 0) or 0
    if load is None and digest is None and not rounds:
        return
    if (load is None) != (digest is None):
        raise ValueError("--rl-critic-load and --rl-critic-init-sha256 go together")
    if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("--rl-critic-init-sha256 must be a lowercase hex SHA256")
    critic = spec.critic
    warm = (spec.execution.needs_critic and critic.init == "copy_actor_backbone"
            and bool(critic.warmup_steps))
    if not warm:
        raise ValueError(
            "--rl-critic-load/--rl-critic-baseline-rounds need a critic algorithm with "
            "critic.init='copy_actor_backbone' and critic.warmup_steps > 0"
        )
    if type(rounds) is not int or rounds < 0:
        raise ValueError(f"--rl-critic-baseline-rounds must be >= 0 (got {rounds!r})")
    if rounds and (load is not None or not getattr(args, "rl_single_island_no_sync", False)):
        raise ValueError(
            "--rl-critic-baseline-rounds needs --rl-single-island-no-sync and no --rl-critic-load"
        )


_ELASTIC_LAUNCH_FLAGS = (
    ("rl_elastic_resources", "--rl-elastic-resources"),
    ("rl_elastic_attestation", "--rl-elastic-attestation"),
    ("rl_elastic_initial_config", "--rl-elastic-initial-config"),
    ("rl_elastic_cells", "--rl-elastic-cells"),
)
# 3.8 strict pause budget: forwarded to the learner; the quorum timeout also
# goes to the syncer so both sides use the same value.
_ELASTIC_PAUSE_FLAGS = (
    ("rl_elastic_quorum_timeout_s", "--rl-elastic-quorum-timeout-s"),
    ("rl_elastic_idle_flow_timeout_s", "--rl-elastic-idle-flow-timeout-s"),
    ("rl_elastic_pause_margin", "--rl-elastic-pause-margin"),
    # controller D4 timeouts T_drain / T_recovery (plan.md E1-C, E1-D 4)
    ("rl_elastic_drain_timeout_s", "--rl-elastic-drain-timeout-s"),
    ("rl_elastic_recovery_timeout_s", "--rl-elastic-recovery-timeout-s"),
    # 3.7 restart recovery budget (recovery-design.md §3)
    ("rl_elastic_max_recovery_attempts", "--rl-elastic-max-recovery-attempts"),
)
# Test-only fault injection for GPU acceptance runs (gpu-plan-v2 A5 quorum case):
# exported into the island run command; off unless given.
_ELASTIC_TEST_FLAGS = (
    ("rl_test_inject_start_delay_s", "--rl-test-inject-start-delay-s"),
    ("rl_test_inject_update_weights_block_s", "--rl-test-inject-update-weights-block-s"),
    ("rl_test_inject_stop_failures", "--rl-test-inject-stop-failures"),
    ("rl_test_hold_before_check_s", "--rl-test-hold-before-check-s"),
    ("rl_test_inject_tool_wait_s", "--rl-test-inject-tool-wait-s"),
    ("rl_test_inject_undrain_fail", "--rl-test-inject-undrain-fail"),
    ("rl_elastic_restart_attempts", "--rl-elastic-restart-attempts"),
)
# (attr, flag, env) of the test-only switches exported into the island run
# command; each needs --rl-elastic, all are off by default.
_ELASTIC_TEST_EXPORTS = (
    ("rl_test_hold_before_check_s", "--rl-test-hold-before-check-s",
     "YETO_RL_TEST_HOLD_BEFORE_CHECK_S"),
    ("rl_test_inject_tool_wait_s", "--rl-test-inject-tool-wait-s",
     "YETO_RL_TEST_INJECT_TOOL_WAIT_S"),
    ("rl_test_inject_undrain_fail", "--rl-test-inject-undrain-fail",
     "YETO_RL_TEST_INJECT_UNDRAIN_FAIL"),
    # A4bc / 3.3 X5: side-effect journal of the injected tool (store_true -> "1")
    ("rl_test_tool_side_effect_log", "--rl-test-tool-side-effect-log",
     "YETO_RL_TEST_TOOL_SIDE_EFFECT_LOG"),
    ("rl_test_inject_lora_perturb", "--rl-test-inject-lora-perturb",
     "YETO_RL_TEST_INJECT_LORA_PERTURB"),
    ("rl_test_inject_stop_failures", "--rl-test-inject-stop-failures",
     "YETO_RL_TEST_INJECT_STOP_FAILURES"),
    ("rl_test_kill_learner_at", "--rl-test-kill-learner-at", "YETO_RL_TEST_KILL_LEARNER_AT"),
    ("rl_test_inject_rebuild_fail", "--rl-test-inject-rebuild-fail",
     "YETO_RL_TEST_INJECT_REBUILD_FAIL"),
    # E2 G-4.5 (plan-v3), read by miles_adapter.cut_injection
    ("rl_test_inject_cut_save_kill_rank", "--rl-test-inject-cut-save-kill-rank",
     "YETO_RL_TEST_INJECT_CUT_SAVE_KILL_RANK"),
    ("rl_test_inject_cut_restore_kill_rank", "--rl-test-inject-cut-restore-kill-rank",
     "YETO_RL_TEST_INJECT_CUT_RESTORE_KILL_RANK"),
    ("rl_test_inject_cut_restore_sleep", "--rl-test-inject-cut-restore-sleep",
     "YETO_RL_TEST_INJECT_CUT_RESTORE_SLEEP"),
    ("rl_test_inject_rebuild_cursor_shift", "--rl-test-inject-rebuild-cursor-shift",
     "YETO_RL_TEST_INJECT_REBUILD_CURSOR_SHIFT"),
)
KILL_PHASES = ("QUIESCING", "TRANSFERRING", "INITIALIZING", "VERIFYING", "COMMITTED",
               "RESUMING", "REBUILDING_TRAINER")
# In-place learner restarts (E1-D ⑤⑥⑦): the learner command runs in a loop that
# re-executes it with the same arguments (same --rl-elastic-state-dir) after a
# non-zero exit, at most N times. The Ray head stays up; the old driver's job
# (fork controller, engines, trainer) dies with it, so the fork restarts at
# membership epoch 0 and the journal reconciles it.
RESTART_LOOP_FN = (
    "yeto_rl_restart_loop() {\n"
    "  local attempt=0 rc=0\n"
    "  while :; do\n"
    "    \"$@\" && return 0\n"
    "    rc=$?\n"
    "    attempt=$((attempt + 1))\n"
    "    if [ \"$attempt\" -gt \"$YETO_RL_RESTART_ATTEMPTS\" ]; then return $rc; fi\n"
    "    echo \"[yeto] learner exited $rc; in-place restart $attempt/$YETO_RL_RESTART_ATTEMPTS\" >&2\n"
    "  done\n"
    "}\n"
)
ELASTIC_ISLAND_STATE_DIR = "~/yeto-rl/elastic-state"
# rl-multinode-island Q4 (C5): where a --rl-checkpoint-store bucket is mounted on the island
ELASTIC_CHECKPOINT_STORE_MOUNT = "~/yeto-checkpoint-store"  # not under the spot ~/yeto-rl mount
NODE0_LOCAL_CHECKPOINT_WARNING = (
    "[launcher] warning: --rl-checkpoint-store not set: the island state dir (journal, cuts, "
    "ledger) stays on node0's local disk; losing a node loses the consistent checkpoint and "
    "the island cannot be rebuilt on other machines (rl-multinode-island Q4)"
)


def _check_resume_launch_flags(args, rl_engine: str) -> bool:
    """rl-resume-from-checkpoint: True when ``--rl-checkpoint-store`` is the non-elastic
    resume store (``--rl-single-island-no-sync``, ports). Validates the cadence flags."""
    store = getattr(args, "rl_checkpoint_store", None)
    elastic = bool(getattr(args, "rl_elastic", False))
    for name, flag in (("rl_cut_every", "--rl-cut-every"), ("rl_cut_keep", "--rl-cut-keep"),
                       ("rl_stop_after_rounds", "--rl-stop-after-rounds")):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise ValueError(f"{flag} must be >= 1")
        if value is not None and not store:
            raise ValueError(f"{flag} needs --rl-checkpoint-store")
    if getattr(args, "rl_resume_allow_config_change", False) and not store:
        raise ValueError("--rl-resume-allow-config-change needs --rl-checkpoint-store")
    if elastic:
        for name, flag in (("rl_cut_keep", "--rl-cut-keep"), ("rl_stop_after_rounds", "--rl-stop-after-rounds")):
            if getattr(args, name, None) is not None:
                raise ValueError(f"{flag} applies to a non-elastic --rl-checkpoint-store run")
        return False
    if not store:
        return False
    if not getattr(args, "rl_single_island_no_sync", False) or rl_engine != "ports":
        raise ValueError("--rl-checkpoint-store without --rl-elastic needs --rl-engine ports and "
                         "--rl-single-island-no-sync (multi-island resume: rl-resume-from-checkpoint 3.3)")
    rl_checkpoint_store_plan(args)  # validates the form
    return True


def modal_checkpoint_store(args) -> tuple[str, str] | None:
    """``(volume name, path in the container)`` for a ``modal-volume://`` store, None
    without a store. Any other store form on a Modal island is an error: Modal mounts no
    bucket and a container path does not outlive the container (it used to be dropped
    silently, rl-resume-from-checkpoint 1.2)."""
    value = getattr(args, "rl_checkpoint_store", None)
    if not value:
        return None
    from .cloud.modal_ckpt_store import MODAL_VOLUME_SCHEME, parse_modal_volume_uri

    if not str(value).startswith(MODAL_VOLUME_SCHEME + "://"):
        raise ValueError(f"--rl-checkpoint-store {value!r} on a Modal island: Modal mounts neither "
                         "buckets nor host paths here; use modal-volume://NAME[/PREFIX] (a v1 Volume)")
    name, prefix = parse_modal_volume_uri(str(value))
    return name, MODAL_CHECKPOINT_STORE_MOUNT + (f"/{prefix}" if prefix else "")


MODAL_CHECKPOINT_STORE_MOUNT = "/root/yeto-checkpoint-store"


def resumes_in_new_container(args) -> bool:
    """A Modal re-run of the island (preemption) is a resume, not a failure, when the run
    has a non-elastic modal-volume:// store (rl-resume-from-checkpoint 2.5)."""
    store = str(getattr(args, "rl_checkpoint_store", None) or "")
    return (store.startswith("modal-volume://") and not getattr(args, "rl_elastic", False)
            and bool(getattr(args, "rl_single_island_no_sync", False)))


def rl_checkpoint_store_plan(args) -> tuple[str, str | None] | None:
    """``(path on the island, bucket URI or None)`` for ``--rl-checkpoint-store``:
    a ``scheme://`` URI is mounted at :data:`ELASTIC_CHECKPOINT_STORE_MOUNT` (sky
    Storage MOUNT, persistent); an absolute / ``~/`` path is used as is (a share
    the replacement machine also mounts). None when the flag is not given."""
    value = getattr(args, "rl_checkpoint_store", None)
    if not value:
        return None
    value = str(value)
    if value.startswith("modal-volume://"):
        from .cloud.modal_ckpt_store import parse_modal_volume_uri

        _name, prefix = parse_modal_volume_uri(value)
        return MODAL_CHECKPOINT_STORE_MOUNT + (f"/{prefix}" if prefix else ""), None
    if "://" in value:
        scheme, _, rest = value.partition("://")
        if not scheme.isalnum() or not rest.strip("/") or rest.startswith("/"):
            raise ValueError(f"--rl-checkpoint-store {value!r} is not a bucket URI (scheme://bucket[/prefix])")
        # sky Storage MOUNT only mounts a bucket root ("MOUNT mode does not support
        # mounting specific files"): mount the bucket, use the prefix as a subdir.
        bucket, _, prefix = rest.strip("/").partition("/")
        path = ELASTIC_CHECKPOINT_STORE_MOUNT + (f"/{prefix.strip('/')}" if prefix.strip("/") else "")
        return path, f"{scheme}://{bucket}"
    if not (value.startswith("/") or value.startswith("~/")) or ".." in PurePosixPath(value).parts:
        raise ValueError("--rl-checkpoint-store must be a bucket URI or an absolute / ~/ path on the island")
    return value.rstrip("/") or "/", None
_EVAL_LAUNCH_FLAGS = (
    ("rl_eval_data", "--rl-eval-data"),
    ("rl_eval_dataset_name", "--rl-eval-dataset-name"),
    ("rl_eval_samples_per_prompt", "--rl-eval-samples-per-prompt"),
)
# optional eval knobs: launcher flag attr -> learner flag (forwarded when given)
_EVAL_OPTIONAL_FLAGS = (
    ("rl_eval_temperature", "--eval-temperature"),
    ("rl_eval_top_p", "--eval-top-p"),
    ("rl_eval_max_prompt_len", "--eval-max-prompt-len"),
    ("rl_eval_max_response_len", "--eval-max-response-len"),
    ("rl_eval_max_context_len", "--eval-max-context-len"),
)
EVAL_ISLAND_DATA_PATH = "~/yeto-rl/eval-heldout.jsonl"
EVAL_INLINE_MAX_BYTES = 96 * 1024  # shipped inline in the run command (ARG_MAX headroom)


def _check_ports_eval(args, rl_engine: str) -> int | None:
    """``--rl-eval-*``: the launcher's ONE eval source (ports LoRA heldout eval).

    Returns the eval interval (None when eval is off) — the same value the
    learner receives as ``--eval-interval`` — and stages the heldout bytes and
    their SHA256 on ``args`` for :func:`_ports_infra_flags`.
    """

    interval = getattr(args, "rl_eval_interval", None)
    given = [flag for name, flag in _EVAL_LAUNCH_FLAGS if getattr(args, name, None) is not None]
    given += ["--rl-" + flag[2:] for name, flag in _EVAL_OPTIONAL_FLAGS
              if getattr(args, name, None) is not None]
    args.rl_eval_data_text = None
    if interval is None:
        if given:
            raise ValueError(", ".join(given) + " need --rl-eval-interval")
        return None
    if rl_engine != "ports":
        raise ValueError("--rl-eval-interval only applies to --rl-engine ports")
    if type(interval) is not int or interval <= 0:
        raise ValueError(f"--rl-eval-interval must be a positive int (got {interval!r})")
    missing = [flag for name, flag in _EVAL_LAUNCH_FLAGS if getattr(args, name, None) is None]
    if missing:
        raise ValueError("--rl-eval-interval needs " + ", ".join(missing))
    samples = args.rl_eval_samples_per_prompt
    if type(samples) is not int or samples <= 0:
        raise ValueError("--rl-eval-samples-per-prompt must be a positive int")
    source = Path(args.rl_eval_data).expanduser()
    if source.is_symlink() or not source.is_file():
        raise ValueError("--rl-eval-data must be one regular local file")
    data = getattr(args, "data", None)
    if isinstance(data, str) and Path(data).expanduser().exists() and (
        Path(data).expanduser().resolve() == source.resolve()
    ):
        raise ValueError("--rl-eval-data must be a heldout file distinct from --data")
    raw = source.read_bytes()
    if len(raw) > EVAL_INLINE_MAX_BYTES:
        raise ValueError(f"--rl-eval-data is larger than {EVAL_INLINE_MAX_BYTES} bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("--rl-eval-data is not UTF-8") from error
    rows = [line for line in text.splitlines() if line.strip()]
    if not rows or len(rows) != len(text.splitlines()):
        raise ValueError("--rl-eval-data must be non-empty JSONL without blank rows")
    args.rl_eval_data_text = text
    args.rl_eval_data_sha256 = hashlib.sha256(raw).hexdigest()
    return interval


def _check_ports_infra_switches(args, rl_engine: str) -> None:
    """``--rl-overlap-eval`` / ``--rl-elastic`` (rl-infra-spec 2.3 / 3.x): opt-in,
    ports-only; the elastic manifest is read here so a bad file fails locally."""

    from .rl.engine.execution_profile import check_elastic_placement, check_overlap_eval

    from .rl.engine.execution_profile import UNKNOWN

    placement = getattr(args, "rl_placement", "colocated") or "colocated"
    _check_test_tool_delay(args, rl_engine)
    eval_interval = _check_ports_eval(args, rl_engine)
    if getattr(args, "rl_overlap_eval", False):
        if rl_engine != "ports":
            raise ValueError("--rl-overlap-eval only applies to --rl-engine ports")
        # Same check the island's execution profile applies, run before any
        # resource is provisioned, on the values the launcher really forwards:
        # --rl-placement is the island's launch.placement.kind and
        # --rl-eval-interval is the learner's --eval-interval (-> Miles
        # eval_interval). --eval-uses-snapshots has no launcher source (the
        # learner never sets it), so that one stays with the island's check.
        check_overlap_eval(placement_kind=placement, eval_interval=eval_interval,
                           eval_uses_snapshots=UNKNOWN)
    if getattr(args, "rl_observe_timeline", False) and rl_engine != "ports":
        raise ValueError("--rl-observe-timeline only applies to --rl-engine ports")
    sample_s = getattr(args, "rl_resource_sample_interval", None)
    if sample_s is not None and (rl_engine != "ports" or sample_s < 0):
        raise ValueError("--rl-resource-sample-interval needs --rl-engine ports and a value >= 0")
    _rl_backend_module(args, "elastic_hook").check_recommend_flags(args)
    dropout = getattr(args, "rl_lora_dropout", None)
    if dropout is not None and (rl_engine != "ports" or not 0.0 <= dropout < 1.0):
        raise ValueError("--rl-lora-dropout needs --rl-engine ports and a value in [0, 1)")
    expert_rank = getattr(args, "rl_lora_expert_rank", None)
    if expert_rank is not None and (rl_engine != "ports" or expert_rank < 0):
        raise ValueError("--rl-lora-expert-rank needs --rl-engine ports and a value >= 0")
    ref_load = getattr(args, "rl_megatron_ref_load", None)
    if ref_load is not None and (rl_engine != "ports" or not ref_load.startswith("/")
                                 or any(c.isspace() for c in ref_load)):
        raise ValueError("--rl-megatron-ref-load needs --rl-engine ports and an absolute path "
                         "without whitespace")
    if getattr(args, "rl_boot_only", False) and (
            rl_engine != "ports" or not getattr(args, "rl_single_island_no_sync", False)):
        raise ValueError("--rl-boot-only needs --rl-engine ports and --rl-single-island-no-sync")
    pre_run = getattr(args, "rl_island_pre_run", None)
    if pre_run is not None:
        if rl_engine != "ports" or not getattr(args, "rl_single_island_no_sync", False):
            raise ValueError("--rl-island-pre-run needs --rl-engine ports and --rl-single-island-no-sync")
        args.rl_island_pre_run_script = read_island_pre_run(pre_run)
    if getattr(args, "rl_deterministic_trainer", False) and rl_engine != "ports":
        raise ValueError("--rl-deterministic-trainer only applies to --rl-engine ports")
    given = [flag for name, flag in _ELASTIC_LAUNCH_FLAGS + _ELASTIC_PAUSE_FLAGS
             if getattr(args, name, None) is not None]
    if getattr(args, "rl_elastic_tool_wait_board", False):
        given.append("--rl-elastic-tool-wait-board")
    if getattr(args, "rl_elastic_trainer_edges", False):
        given.append("--rl-elastic-trainer-edges")
    if getattr(args, "rl_elastic_accept_rebind", False):
        given.append("--rl-elastic-accept-rebind")
    resume_store = _check_resume_launch_flags(args, rl_engine)
    if getattr(args, "rl_checkpoint_store", None) and not resume_store:
        given.append("--rl-checkpoint-store")
    if getattr(args, "rl_elastic_declare_cells", False):
        given.append("--rl-elastic-declare-cells")
    for name, flag in _ELASTIC_PAUSE_FLAGS + _ELASTIC_TEST_FLAGS:
        value = getattr(args, name, None)
        if value is not None and not value > 0:
            raise ValueError(f"{flag} must be positive")
    given += [flag for name, flag in _ELASTIC_TEST_FLAGS if getattr(args, name, None) is not None]
    given += [flag for name, flag, _ in _ELASTIC_TEST_EXPORTS
              if getattr(args, name, None) is not None and getattr(args, name) is not False
              and flag not in given]  # rank 0 is a valid value (E2 cut injections)
    if getattr(args, "rl_elastic_state_dir", None) is not None:
        given.append("--rl-elastic-state-dir")
    if getattr(args, "rl_test_hold_before_check_s", None) is not None and not any(
            getattr(args, n, None) not in (None, False)
            for n, _f, e in _ELASTIC_TEST_EXPORTS if n != "rl_test_hold_before_check_s"):
        raise ValueError("--rl-test-hold-before-check-s only acts together with another "
                         "--rl-test-* injection (test-injection mode)")
    if (getattr(args, "rl_test_inject_tool_wait_s", None) is not None
            and not getattr(args, "rl_elastic_tool_wait_board", False)):
        raise ValueError("--rl-test-inject-tool-wait-s needs --rl-elastic-tool-wait-board")
    if (getattr(args, "rl_test_tool_side_effect_log", False)
            and getattr(args, "rl_test_inject_tool_wait_s", None) is None):
        raise ValueError("--rl-test-tool-side-effect-log needs --rl-test-inject-tool-wait-s "
                         "(the injected tool is the journaled tool)")
    kill_at = getattr(args, "rl_test_kill_learner_at", None)
    if kill_at is not None and kill_at not in KILL_PHASES:
        raise ValueError(f"--rl-test-kill-learner-at must be one of {list(KILL_PHASES)}")
    if kill_at is not None and not getattr(args, "rl_elastic_restart_attempts", None):
        raise ValueError("--rl-test-kill-learner-at needs --rl-elastic-restart-attempts")
    if not getattr(args, "rl_elastic", False):
        if given:
            raise ValueError(", ".join(given) + " need --rl-elastic")
        return
    if rl_engine != "ports":
        raise ValueError("--rl-elastic only applies to --rl-engine ports")
    missing = [flag for name, flag in _ELASTIC_LAUNCH_FLAGS
               if name not in ("rl_elastic_attestation", "rl_elastic_cells")
               and not getattr(args, name, None)]
    if missing:
        raise ValueError("--rl-elastic needs " + ", ".join(missing))
    check_elastic_placement(placement)
    if args.rl_elastic_cells is not None:
        cells = [c.strip() for c in args.rl_elastic_cells.split(",") if c.strip()]
        if not cells or any(not re.fullmatch(r"[A-Za-z0-9_.:@+-]+", c) for c in cells):
            raise ValueError(f"--rl-elastic-cells: bad cell ids {args.rl_elastic_cells!r}")
    if getattr(args, "rl_elastic_declare_cells", False) and args.rl_elastic_cells is None:
        raise ValueError("--rl-elastic-declare-cells needs --rl-elastic-cells (the names)")
    from .rl.elastic_benchmark.capabilities import load_attestation, parse_configs

    resources = json.loads(Path(args.rl_elastic_resources).expanduser().read_text(encoding="utf-8"))
    configs = parse_configs(resources)  # the island's build_elastic parses the same way
    if args.rl_elastic_initial_config not in configs:
        raise ValueError(f"--rl-elastic-initial-config {args.rl_elastic_initial_config!r} "
                         f"is not a manifest config ({sorted(configs)})")
    args.rl_elastic_resources_json = json.dumps(resources, sort_keys=True, separators=(",", ":"))
    # Q2: the initial config's explicit (node, gpu) placement, when the manifest
    # declares a topology; rl_island_layout derives the trainer shape from it.
    args.rl_elastic_initial_placement_slots = getattr(
        configs[args.rl_elastic_initial_config], "placement_slots", None)
    # ruling 2026-10-04 v2: cfg parallel.allow_cross_node_* count like the CLI flags
    args.rl_elastic_initial_cfg_switches = {
        k: bool(getattr(configs[args.rl_elastic_initial_config], k, False))
        for k in ("allow_cross_node_tp", "allow_cross_node_engine_tp")}
    attestation = getattr(args, "rl_elastic_attestation", None)
    args.rl_elastic_attestation_json = None
    if attestation:
        path = Path(attestation).expanduser()
        load_attestation(path)  # malformed attestation fails locally
        args.rl_elastic_attestation_json = json.dumps(
            json.loads(path.read_text(encoding="utf-8")), sort_keys=True, separators=(",", ":"),
        )


def _check_test_tool_delay(args, rl_engine: str) -> None:
    """``--rl-test-tool-delay-s`` (TEST ONLY): the tool-wait workload's delay."""
    from .rl.tool_wait_workload import GENERATE_PATH

    delay = getattr(args, "rl_test_tool_delay_s", None)
    if delay is None:
        return
    if rl_engine != "ports":
        raise ValueError("--rl-test-tool-delay-s only applies to --rl-engine ports")
    if not delay > 0:
        raise ValueError("--rl-test-tool-delay-s must be positive")
    if getattr(args, "custom_generate_function_path", None) != GENERATE_PATH:
        raise ValueError(f"--rl-test-tool-delay-s needs --custom-generate-function-path {GENERATE_PATH}")


def _ports_infra_flags(args) -> tuple[str, str]:
    """(prelude, learner flags) for the opt-in 2.3/3.x switches; ("", "") by default."""

    prelude, flags = "", ""
    if getattr(args, "rl_test_tool_delay_s", None) is not None:
        from .rl.tool_wait_workload import TOOL_DELAY_ENV

        prelude += f"export {TOOL_DELAY_ENV}={float(args.rl_test_tool_delay_s)!r}\n"
    if getattr(args, "rl_eval_interval", None) is not None:
        prelude += (
            "mkdir -p ~/yeto-rl && printf '%s' "
            f"{shlex.quote(args.rl_eval_data_text)} > {EVAL_ISLAND_DATA_PATH}\n"
        )
        flags += (
            f" --eval-interval {int(args.rl_eval_interval)}"
            f" --eval-data {EVAL_ISLAND_DATA_PATH}"
            f" --eval-data-sha256 {args.rl_eval_data_sha256}"
            f" --eval-dataset-name {shlex.quote(args.rl_eval_dataset_name)}"
            f" --eval-samples-per-prompt {int(args.rl_eval_samples_per_prompt)}"
        )
        for name, flag in _EVAL_OPTIONAL_FLAGS:
            value = getattr(args, name, None)
            if value is not None:
                flags += f" {flag} {value!r}"
    if getattr(args, "rl_overlap_eval", False):
        flags += " --rl-overlap-eval"
    if getattr(args, "rl_observe_timeline", False):
        flags += " --rl-observe-timeline"
    if getattr(args, "rl_resource_sample_interval", None) is not None:
        flags += f" --rl-resource-sample-interval {float(args.rl_resource_sample_interval)!r}"
    flags += _rl_backend_module(args, "elastic_hook").recommend_flags(args)
    if getattr(args, "rl_deterministic_trainer", False):
        flags += " --rl-deterministic-trainer"
    if getattr(args, "rl_lora_dropout", None) is not None:
        flags += f" --rl-lora-dropout {float(args.rl_lora_dropout)!r}"
    if getattr(args, "rl_lora_expert_rank", None) is not None:
        flags += f" --rl-lora-expert-rank {int(args.rl_lora_expert_rank)}"
    if getattr(args, "rl_megatron_ref_load", None) is not None:
        flags += f" --megatron-ref-load {shlex.quote(args.rl_megatron_ref_load)}"
    if getattr(args, "rl_print_attestation_fingerprint", False):
        flags += " --rl-print-attestation-fingerprint"
    if getattr(args, "rl_boot_only", False):
        flags += " --rl-boot-only"
    if island_scheduling_mode(args) == "elastic":  # legacy adds nothing
        flags += (" --rl-island-scheduling elastic"
                  f" --rl-syncer-epoch {island_syncer_epoch(args)}")
    resume_store = (rl_checkpoint_store_plan(args)
                    if getattr(args, "rl_checkpoint_store", None) and not getattr(args, "rl_elastic", False)
                    else None)
    if resume_store is not None:  # rl-resume-from-checkpoint: non-elastic resume store
        flags += f" --rl-resume-store {shlex.quote(resume_store[0])}"
        for name, flag in (("rl_cut_keep", "--rl-cut-keep"), ("rl_stop_after_rounds", "--rl-stop-after-rounds")):
            if getattr(args, name, None) is not None:
                flags += f" {flag} {int(getattr(args, name))}"
    if getattr(args, "rl_checkpoint_store", None) and getattr(args, "rl_cut_every", None) is not None:
        flags += f" --rl-cut-every {int(args.rl_cut_every)}"
    if getattr(args, "rl_checkpoint_store", None) and getattr(args, "rl_resume_allow_config_change", False):
        flags += " --rl-resume-allow-config-change"
    if getattr(args, "rl_elastic", False):
        prelude += (
            "mkdir -p ~/yeto-rl && printf '%s' "
            f"{shlex.quote(args.rl_elastic_resources_json)} > ~/yeto-rl/elastic_resources.json\n"
        )
        state_dir = (shlex.quote(args.rl_elastic_state_dir)
                     if getattr(args, "rl_elastic_state_dir", None) else ELASTIC_ISLAND_STATE_DIR)
        flags += (
            " --rl-elastic --rl-elastic-resources ~/yeto-rl/elastic_resources.json"
            f" --rl-elastic-state-dir {state_dir}"
            f" --rl-elastic-initial-config {shlex.quote(args.rl_elastic_initial_config)}"
        )
        if args.rl_elastic_cells is not None:
            flags += f" --rl-elastic-cells {shlex.quote(args.rl_elastic_cells)}"
        if getattr(args, "rl_elastic_declare_cells", False):
            flags += " --rl-elastic-declare-cells"
        if getattr(args, "rl_elastic_tool_wait_board", False):
            flags += " --rl-elastic-tool-wait-board"
        if getattr(args, "rl_elastic_trainer_edges", False):
            flags += " --rl-elastic-trainer-edges"
        if getattr(args, "rl_elastic_accept_rebind", False):
            flags += " --rl-elastic-accept-rebind"
        store = rl_checkpoint_store_plan(args)
        if store is not None:
            flags += f" --rl-elastic-checkpoint-store {shlex.quote(store[0])}"
        for name, flag in _ELASTIC_PAUSE_FLAGS:
            value = getattr(args, name, None)
            if value is not None:
                flags += f" {flag} {value!r}"
        delay = getattr(args, "rl_test_inject_start_delay_s", None)
        if delay is not None:
            INJECT_START_DELAY_ENV = _rl_backend_module(args, "rollout").INJECT_START_DELAY_ENV

            prelude += f"export {INJECT_START_DELAY_ENV}={float(delay)!r}\n"
        for name, _flag, env in _ELASTIC_TEST_EXPORTS:
            value = getattr(args, name, None)
            if value is True:
                value = 1
            if value is not None and value is not False:  # rank 0 is valid
                prelude += f"export {env}={shlex.quote(str(value))}\n"
        attempts = getattr(args, "rl_elastic_restart_attempts", None)
        if attempts:
            prelude += f"export YETO_RL_RESTART_ATTEMPTS={int(attempts)}\n" + RESTART_LOOP_FN
            args.rl_learner_launch_prefix = "yeto_rl_restart_loop "
        block = getattr(args, "rl_test_inject_update_weights_block_s", None)
        if block is not None:
            INJECT_UPDATE_BLOCK_ENV = _rl_backend_module(args, "publish").INJECT_UPDATE_BLOCK_ENV

            prelude += f"export {INJECT_UPDATE_BLOCK_ENV}={float(block)!r}\n"
        if getattr(args, "rl_elastic_attestation_json", None):
            prelude += (
                "printf '%s' "
                f"{shlex.quote(args.rl_elastic_attestation_json)} > ~/yeto-rl/elastic_attestation.json\n"
            )
            flags += " --rl-elastic-attestation ~/yeto-rl/elastic_attestation.json"
    return prelude, flags


def _ports_algorithm_flags(args) -> tuple[str, str]:
    """(run prelude, learner flags) carrying the algorithm to a ports island."""

    if getattr(args, "rl_engine", "ports") != "ports":
        return "", ""
    expected = getattr(args, "rl_expected_algorithm_sha256", None)
    if expected is None:
        return "", ""
    prelude = ""
    flags = f" --rl-expected-algorithm-sha256 {expected}"
    if getattr(args, "rl_algorithm_spec", None):
        prelude = (
            "mkdir -p ~/yeto-rl && printf '%s' "
            f"{shlex.quote(args.rl_algorithm_spec_json)} > ~/yeto-rl/algorithm_spec.json\n"
        )
        flags += " --rl-algorithm-spec ~/yeto-rl/algorithm_spec.json"
    # rl-infra-spec 2.1 placement options, forwarded only when non-default
    # (the default argv is unchanged).
    if getattr(args, "rl_placement", "colocated") != "colocated":
        flags += f" --rl-placement {shlex.quote(args.rl_placement)}"
    if getattr(args, "rl_standby_gpus", 0):
        flags += f" --rl-standby-gpus {int(args.rl_standby_gpus)}"
    if getattr(args, "rl_placement", "colocated") == "fixed-partition":
        flags += f" --rollout-num-gpus {int(args.rollout_num_gpus)}"
    allowed = list(getattr(args, "rl_allow_unverified_mechanism", None) or ())
    for name in allowed:
        flags += f" --rl-allow-unverified-mechanism {shlex.quote(name)}"
    # rl-algo-critic-family 5.3: critic warm-up options, forwarded only when set
    # (every other argv is unchanged).
    if getattr(args, "rl_critic_load", None):
        flags += (f" --rl-critic-load {shlex.quote(args.rl_critic_load)}"
                  f" --rl-critic-init-sha256 {shlex.quote(args.rl_critic_init_sha256)}")
    if getattr(args, "rl_critic_baseline_rounds", 0):
        flags += f" --rl-critic-baseline-rounds {int(args.rl_critic_baseline_rounds)}"
    infra_prelude, infra_flags = _ports_infra_flags(args)
    return prelude + infra_prelude, flags + infra_flags


def _prepare_rl_args(
    args,
    *,
    allow_local_data: bool = False,
    allow_remote_model: bool = False,
) -> None:
    resolve_default_rl_image(args)
    if getattr(args, "training_mode", "sft") != "rl":
        if getattr(args, "rl_initial_adapter", None) is not None or getattr(
            args, "rl_initial_adapter_sha256", None
        ) is not None:
            raise ValueError("--rl-initial-adapter requires --training-mode rl")
        if getattr(args, "codex_reasoning_effort", None) is not None:
            raise ValueError("--codex-reasoning-effort requires --training-mode rl")
        return

    from .adapter_lifecycle import directory_sha256, selected_parent

    if selected_parent(args)[1] is not None:
        raise ValueError(
            "--resume-from/--branch-from are not supported with --training-mode rl"
        )
    initial_adapter = getattr(args, "rl_initial_adapter", None)
    expected_initial_sha256 = getattr(args, "rl_initial_adapter_sha256", None)
    if initial_adapter is None and expected_initial_sha256 is not None:
        raise ValueError(
            "--rl-initial-adapter-sha256 requires --rl-initial-adapter"
        )
    if initial_adapter is not None:
        if args.rl_sync_preset != "decoupled":
            raise ValueError(
                "--rl-initial-adapter is only supported with "
                "--rl-sync-preset decoupled"
            )
        root = Path(initial_adapter).expanduser()
        if not root.is_dir():
            raise ValueError("--rl-initial-adapter must be a local directory")
        actual_initial_sha256 = directory_sha256(root)
        if (
            expected_initial_sha256 is not None
            and actual_initial_sha256 != expected_initial_sha256.lower()
        ):
            raise ValueError(
                "RL initial adapter SHA256 mismatch: expected "
                f"{expected_initial_sha256.lower()}, got {actual_initial_sha256}"
            )
        args.rl_initial_adapter = str(root.resolve())
        args.rl_initial_adapter_sha256 = actual_initial_sha256

    from .models import resolve_model_kind

    rl_engine = getattr(args, "rl_engine", "ports")
    if rl_engine not in ("legacy", "ports"):
        raise ValueError(f"--rl-engine must be legacy or ports, got {rl_engine!r}")
    if rl_engine == "ports":
        from .rl.engine.selection import require_ports_supported

        # Rejected before any cloud or GPU work (spec rl-engine-selection).
        require_ports_supported(
            sync_preset=args.rl_sync_preset,
            model_kind=resolve_model_kind(args.model, args.model_kind),
            tuning=args.tuning,
            model_recipe=getattr(args, "rl_model_recipe", "generic"),
            lora_targets=getattr(args, "lora_targets", None),
            expert_full_count=getattr(args, "expert_full_count", 0) or 0,
            rollout_num_gpus=getattr(args, "rollout_num_gpus", None),
            placement=getattr(args, "rl_placement", "colocated"),
        )
    _prepare_ports_algorithm(args, rl_engine)
    if resolve_model_kind(args.model, args.model_kind) != "causal-lm":
        raise ValueError("RL v0 supports only causal language models")
    if args.tuning != "lora":
        raise ValueError("RL v0 requires --tuning lora")
    if args.lora_r <= 0:
        raise ValueError("RL v0 requires a positive LoRA rank")
    if args.total_steps <= 0:
        raise ValueError("RL v0 requires --total-steps > 0")
    rollout_model = getattr(args, "rollout_model", None)
    rollout_revision = getattr(args, "rollout_model_revision", None)
    if rollout_model is None and rollout_revision is not None:
        raise ValueError("--rollout-model-revision requires --rollout-model")
    if rollout_model is not None and rollout_model != args.model:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", str(rollout_revision or "")):
            raise ValueError(
                "a distinct --rollout-model requires an immutable "
                "--rollout-model-revision"
            )
        args.rollout_model_revision = rollout_revision.lower()
    if args.rollout_batch_size <= 0 or args.n_samples_per_prompt <= 0:
        raise ValueError("RL v0 requires positive rollout batch and sample counts")
    if args.rollout_max_response_len <= 0:
        raise ValueError("RL v0 requires --rollout-max-response-len > 0")
    if args.cybergym_timeout <= 0:
        raise ValueError("RL v0 requires a positive CyberGym timeout")
    if args.rl_distributed_timeout_minutes <= 0:
        raise ValueError(
            "RL v0 requires --rl-distributed-timeout-minutes > 0"
        )
    if args.dynamic_sampling_max_replacements is not None and (
        args.dynamic_sampling_max_replacements < 0
    ):
        raise ValueError(
            "RL --dynamic-sampling-max-replacements must be non-negative"
        )
    if not args.inner_lr > 0:
        raise ValueError(f"RL requires --inner-lr > 0 (got {args.inner_lr})")
    if args.rl_sync_preset == "strict-avg":
        if args.local_rl_rounds_per_sync != 1:
            raise ValueError("RL v0 requires --local-rl-rounds-per-sync 1")
    elif args.local_rl_rounds_per_sync < 2:
        raise ValueError("decoupled RL requires at least 2 local RL rounds per sync")
    if args.seq_len < 2:
        raise ValueError("RL v0 requires --seq-len >= 2")
    args.seq_len = max(args.seq_len, args.rollout_max_response_len)
    if args.over_sampling_batch_size is None:
        args.over_sampling_batch_size = args.rollout_batch_size
    elif args.over_sampling_batch_size < args.rollout_batch_size:
        raise ValueError(
            "RL --over-sampling-batch-size must be at least --rollout-batch-size"
        )
    _rl_miles_function(args.custom_generate_function_path)
    custom_agent = getattr(args, "custom_agent_function_path", None)
    _rl_miles_function(custom_agent, "--custom-agent-function-path")
    from .rl import (
        SECRLENV_AGENTS,
        SECRLENV_GENERATE,
        SECRLENV_GROUP_FILTER,
        SECRLENV_INFRASTRUCTURE_REPLACEMENTS,
        SECRLENV_REWARD,
        SECRLENV_ZERO_VARIANCE_REPLACEMENTS,
        SIGNED_CODEX_AGENTS,
    )

    if custom_agent is not None:
        if args.custom_generate_function_path is None:
            raise ValueError(
                "--custom-agent-function-path requires "
                "--custom-generate-function-path"
            )
        if not args.use_session_server:
            raise ValueError(
                "--custom-agent-function-path requires --use-session-server"
            )
        if args.tito_model is None:
            raise ValueError(
                "--custom-agent-function-path requires --tito-model"
            )
        if (
            args.apply_chat_template_kwargs is not None
            and custom_agent not in SIGNED_CODEX_AGENTS
        ):
            raise ValueError(
                "agentic session rollouts preserve raw messages and do not accept "
                "--apply-chat-template-kwargs"
            )

    codex_reasoning_effort = getattr(args, "codex_reasoning_effort", None)
    codex_backend_profile = getattr(args, "codex_backend_profile", None)
    if custom_agent in SIGNED_CODEX_AGENTS:
        from .rl.codex_backend import (
            stock_codex_backend_profile,
            validate_stock_codex_fields,
        )

        codex_backend_profile = codex_backend_profile or str(args.tito_model)
        args.codex_backend_profile = codex_backend_profile
        codex_profile = stock_codex_backend_profile(codex_backend_profile)
        codex_chat_template_kwargs = codex_profile["chat_template_kwargs"]
        if args.apply_chat_template_kwargs is None:
            args.apply_chat_template_kwargs = codex_chat_template_kwargs
        validate_stock_codex_fields(
            tito_model=str(args.tito_model),
            codex_backend_profile=codex_backend_profile,
            rl_model_recipe=getattr(args, "rl_model_recipe", "generic"),
            model=str(args.model),
            model_revision=str(args.model_revision),
            rollout_model=getattr(args, "rollout_model", None),
            rollout_model_revision=getattr(args, "rollout_model_revision", None),
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            tito_allowed_append_roles=list(
                getattr(args, "tito_allowed_append_roles", None) or ()
            ),
            codex_reasoning_effort=codex_reasoning_effort,
            lora_targets=str(args.lora_targets),
            expert_full_count=int(getattr(args, "expert_full_count", 0)),
            lora_expert_rank=int(getattr(args, "rl_lora_expert_rank", None) or 0),
        )
    elif codex_reasoning_effort is not None or codex_backend_profile is not None:
        raise ValueError(
            "--codex-reasoning-effort/--codex-backend-profile requires a signed Codex agent"
        )
    if getattr(args, "codex_exclude_reasoning_from_loss", False):
        if custom_agent not in SIGNED_CODEX_AGENTS:
            raise ValueError(
                "--codex-exclude-reasoning-from-loss requires a signed Codex agent"
            )
        from .rl.codex_backend import stock_codex_reasoning_markers

        if stock_codex_reasoning_markers(codex_backend_profile) is None:
            raise ValueError(
                f"Codex profile {codex_backend_profile!r} declares no reasoning "
                "markers; --codex-exclude-reasoning-from-loss is unavailable"
            )
    secrlenv_infrastructure_replacements = getattr(
        args, "secrlenv_max_infrastructure_replacements", None
    )
    if custom_agent in SECRLENV_AGENTS:
        if args.custom_generate_function_path != SECRLENV_GENERATE:
            raise ValueError(
                "the SecRLEnv agents require the signed SecRLEnv generate wrapper"
            )
        if args.reward_function != SECRLENV_REWARD:
            raise ValueError(
                "the SecRLEnv agents require the signed SecRLEnv reward function"
            )
        configured_filter = getattr(args, "dynamic_sampling_filter_path", None)
        if configured_filter not in {None, SECRLENV_GROUP_FILTER}:
            raise ValueError(
                "the SecRLEnv agents require the exact signed group filter"
            )
        zero_variance_replacements = getattr(
            args, "dynamic_sampling_max_replacements", None
        )
        if zero_variance_replacements not in {
            None,
            SECRLENV_ZERO_VARIANCE_REPLACEMENTS,
        } or isinstance(zero_variance_replacements, bool):
            raise ValueError(
                "the SecRLEnv agents require zero variance replacements"
            )
        if secrlenv_infrastructure_replacements not in {
            None,
            SECRLENV_INFRASTRUCTURE_REPLACEMENTS,
        } or isinstance(secrlenv_infrastructure_replacements, bool):
            raise ValueError(
                "the SecRLEnv agents require exactly one infrastructure replacement"
            )
        expected_oversampling = args.rollout_batch_size + 1
        if args.over_sampling_batch_size == args.rollout_batch_size:
            args.over_sampling_batch_size = expected_oversampling
        elif args.over_sampling_batch_size != expected_oversampling:
            raise ValueError(
                "the SecRLEnv agents require oversampling equal to the training "
                "batch plus one"
            )
        args.dynamic_sampling_filter_path = SECRLENV_GROUP_FILTER
        args.dynamic_sampling_max_replacements = (
            SECRLENV_ZERO_VARIANCE_REPLACEMENTS
        )
        args.secrlenv_max_infrastructure_replacements = (
            SECRLENV_INFRASTRUCTURE_REPLACEMENTS
        )
    elif secrlenv_infrastructure_replacements is not None:
        raise ValueError(
            "--secrlenv-max-infrastructure-replacements requires a signed "
            "SecRLEnv agent"
        )
    agent_max_seq_len = getattr(args, "agent_max_seq_len", None)
    if agent_max_seq_len is not None:
        if custom_agent is None:
            raise ValueError(
                "--agent-max-seq-len requires --custom-agent-function-path"
            )
        if agent_max_seq_len <= 0 or agent_max_seq_len > args.seq_len:
            raise ValueError(
                "--agent-max-seq-len must be positive and no greater than --seq-len"
            )
    dynamic_filter = getattr(args, "dynamic_sampling_filter_path", None)
    _rl_miles_function(
        dynamic_filter,
        "--dynamic-sampling-filter-path",
    )
    if (
        dynamic_filter is not None
        and args.over_sampling_batch_size <= args.rollout_batch_size
    ):
        raise ValueError(
            "variance-aware RL filtering requires --over-sampling-batch-size "
            "greater than --rollout-batch-size"
        )
    if args.dynamic_sampling_max_replacements is not None:
        if dynamic_filter is None:
            raise ValueError(
                "--dynamic-sampling-max-replacements requires "
                "--dynamic-sampling-filter-path"
            )
        stock_filter = (
            "miles.rollout.filter_hub.dynamic_sampling_filters."
            "check_reward_nonzero_std"
        )
        bounded_filter = "yeto.rl.filters.bounded_nonzero_reward_std"
        if dynamic_filter == stock_filter:
            args.dynamic_sampling_filter_path = bounded_filter
        elif dynamic_filter not in {
            bounded_filter,
            SECRLENV_GROUP_FILTER,
        }:
            raise ValueError(
                "--dynamic-sampling-max-replacements is only supported with "
                "an approved bounded nonzero-variance filter"
            )
    if not args.use_session_server and (
        args.session_server_ip is not None
        or args.session_server_port is not None
        or args.tito_model is not None
        or getattr(args, "codex_backend_profile", None) is not None
        or getattr(args, "tito_allowed_append_roles", None) is not None
    ):
        raise ValueError(
            "--session-server-ip/--session-server-port/--tito-model/"
            "--codex-backend-profile/"
            "--tito-allowed-append-roles requires --use-session-server"
        )
    roles = getattr(args, "tito_allowed_append_roles", None)
    if roles is not None:
        args.tito_allowed_append_roles = list(dict.fromkeys(roles))
    if args.session_server_port is not None and (
        len(args.session_server_port) not in {1, 2}
        or any(port <= 0 or port > 65535 for port in args.session_server_port)
        or (
            len(args.session_server_port) == 2
            and args.session_server_port[1] <= args.session_server_port[0]
        )
    ):
        raise ValueError(
            "--session-server-port requires one positive port or an increasing range"
        )

    specs = [rl_island_spec(args, spec) for spec in parse_gpu_spec(args.gpu)]
    if getattr(args, "external_learners", 0):
        raise ValueError("RL v0 does not support external learner slots")
    if args.tensor_parallel <= 0 or args.pipeline_parallel <= 0:
        raise ValueError("RL tensor and pipeline parallelism must be positive")
    model_parallel = args.tensor_parallel * args.pipeline_parallel
    if args.expert_parallel is not None:
        if args.expert_parallel <= 0:
            raise ValueError("RL expert parallelism must be positive")
        if any(spec.total_gpus % args.expert_parallel for spec in specs):
            raise ValueError("RL expert parallelism must divide every island")
    for spec in specs:
        if getattr(args, "rl_placement", "colocated") == "fixed-partition":
            # rl-infra-spec 2.1: the node's GPUs are trainer + rollout + standby;
            # parallel-size checks below apply to the trainer part.
            if spec.num_nodes == 1 and rl_actor_gpus_per_node(args, spec) < 1:
                raise ValueError(
                    "--rl-placement fixed-partition needs one node with "
                    "--rollout-num-gpus + --rl-standby-gpus < GPUs per node"
                )
            if int(args.rollout_num_gpus) % args.rollout_num_gpus_per_engine:
                raise ValueError("--rollout-num-gpus must be a multiple of --rollout-num-gpus-per-engine")
            from .gpu_spec import require_min_nodes

            require_min_nodes(spec, rl_min_nodes(args, spec))
            actor_nodes, actor = rl_trainer_shape(args, spec)
            node_parallel = args.tensor_parallel * int(getattr(args, "context_parallel", 1) or 1)
            cross_tp, cross_engine = rl_cross_node_switches(args)
            if spec.num_nodes > 1 and not cross_tp and (node_parallel > spec.gpus_per_node
                                                        or spec.gpus_per_node % node_parallel):
                raise ValueError("RL TP*CP must fit and divide one node (rl-multinode-island Q1/Q3: "
                                 "TP stays inside a node by default; EP/PP may span nodes; "
                                 "--rl-allow-cross-node-tp or cfg parallel.allow_cross_node_tp lifts it)")
            if spec.num_nodes > 1 and cross_tp:
                print("[launcher] WARNING: --rl-allow-cross-node-tp: the trainer TP*CP group "
                      f"({node_parallel}) may span nodes (NCCL over the inter-node fabric); "
                      "journaled as topology.layout.cross_node_tp=1")
            if spec.num_nodes > 1 and cross_engine:
                print("[launcher] WARNING: --rl-allow-cross-node-engine-tp: a rollout engine of "
                      f"{args.rollout_num_gpus_per_engine} GPUs may span whole nodes (sglang "
                      "nnodes>1; scales as one replica)")
            spec = dataclasses.replace(spec, num_nodes=actor_nodes, gpus_per_node=actor)
        elif spec.num_nodes > 1:
            # m5 (ruling 2026-10-04 v2): a colocated multi-node island carves its engines out
            # of the whole island; an engine larger than a node (sglang nnodes>1) needs the
            # explicit opt-in and must take whole nodes, before any cloud work.
            rl_colocated_engine_check(args, spec)
        if spec.total_gpus % model_parallel:
            raise ValueError(
                "RL TP*PP must divide every island GPU world size"
            )
        dp = spec.total_gpus // model_parallel
        if args.rollout_batch_size * args.n_samples_per_prompt % dp:
            raise ValueError(
                "RL rollout_batch_size*n_samples_per_prompt must be divisible "
                "by every island data-parallel size"
            )
        if (
            args.rollout_num_gpus_per_engine <= 0
            or spec.total_gpus % args.rollout_num_gpus_per_engine
        ):
            raise ValueError(
                "RL --rollout-num-gpus-per-engine must be positive and divide "
                "every island GPU world size"
            )
    for name in ("sglang_tp_size", "sglang_dp_size", "sglang_ep_size"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"RL --{name.replace('_', '-')} must be positive")
    if not 0.0 < args.sglang_mem_fraction_static < 1.0:
        raise ValueError("RL --sglang-mem-fraction-static must be between 0 and 1")
    if (
        getattr(args, "sglang_deterministic_inference", True)
        and getattr(args, "sglang_attention_backend", None) in {"dsv4", "compressed"}
    ):
        raise ValueError(
            "DeepSeek V4 dsv4/compressed attention requires "
            "--no-sglang-deterministic-inference"
        )
    expert_full_count = getattr(args, "expert_full_count", 0)
    selection_sha256 = getattr(args, "expert_selection_sha256", None)
    selection_contract_sha256 = getattr(
        args,
        "expert_selection_contract_sha256",
        None,
    )
    if expert_full_count < 0 or expert_full_count > 32:
        raise ValueError("--expert-full-count must be between 1 and 32 when enabled")
    if expert_full_count == 0:
        if selection_sha256 is not None or selection_contract_sha256 is not None:
            raise ValueError("expert selection hash requires --expert-full-count")
    else:
        if getattr(args, "rl_model_recipe", "generic") != "deepseek-v4-flash":
            raise ValueError(
                "--expert-full-count is only supported by "
                "--rl-model-recipe deepseek-v4-flash"
            )
        if args.lora_targets != "attention":
            raise ValueError("expert-full RL requires --lora-targets attention")
        if not 0 < args.expert_full_lr < float("inf"):
            raise ValueError("--expert-full-lr must be finite and positive")
        for flag, value in (
            ("--expert-selection-sha256", selection_sha256),
            (
                "--expert-selection-contract-sha256",
                selection_contract_sha256,
            ),
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(value or "")):
                raise ValueError(f"{flag} must be 64 lowercase hex characters")
    if getattr(args, "rl_model_recipe", "generic") == "deepseek-v4-flash":
        expected_lora_targets = (
            "attention" if expert_full_count else "attention-routed-experts"
        )
        if args.lora_targets != expected_lora_targets:
            raise ValueError(
                "expanded DeepSeek V4 Flash recipe requires "
                f"--lora-targets {expected_lora_targets}"
            )
        if (
            args.tensor_parallel != 8
            or args.expert_parallel != 8
            or args.rollout_num_gpus_per_engine != 8
            or args.sglang_tp_size != 8
            or args.sglang_ep_size != 8
            or args.sglang_dp_size not in (None, 1)
        ):
            raise ValueError(
                "expanded DeepSeek V4 Flash requires TP8/EP8 pipeline stages "
                "and per-node eight-GPU rollout replicas"
            )
        if args.sglang_attention_backend not in {"dsv4", "compressed"}:
            raise ValueError(
                "DeepSeek V4 Flash recipe requires --sglang-attention-backend dsv4"
            )
        if args.sglang_page_size != 256:
            raise ValueError(
                "DeepSeek V4 Flash recipe requires --sglang-page-size 256"
            )
    for name in (
        "sglang_page_size",
        "sglang_max_running_requests",
        "sglang_chunked_prefill_size",
    ):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"RL --{name.replace('_', '-')} must be positive")

    _rl_callable(args.reward_function, "--reward-function", required=True)
    if not re.fullmatch(
        r"docker:[^\s@]+@sha256:[0-9a-fA-F]{64}", args.rl_image or ""
    ) and not _rl_backend_image_ok(args):
        raise ValueError(
            "--rl-image must be docker:<repository>@sha256:<64 hex digest>"
        )
    if getattr(args, "learner_image", None) is not None:
        raise ValueError("RL uses digest-pinned --rl-image, not --learner-image")
    args.rl_total_fragment_steps = args.total_steps
    if args.rl_sync_preset == "decoupled":
        if args.experimental_rl_sync:
            raise ValueError("decoupled RL does not accept --experimental-rl-sync")
        if args.fragments < 2:
            raise ValueError("decoupled RL requires at least 2 fragments")
        if not 1 <= args.pipeline <= args.fragments:
            raise ValueError("decoupled RL pipeline must be between 1 and fragments")
        if args.fragment_pattern != "binpack":
            raise ValueError("decoupled RL requires --fragment-pattern binpack")
        args.quorum = len(specs)
        args.grace_ms = 0
        args.sync_interval_steps = float(args.local_rl_rounds_per_sync)
        args.delta_correction = "none"
        args.outer_lr = 0.7
        args.outer_momentum = 0.9
        args.merge_alpha = 0.0
        args.wire_dtype = "f32"
        args.rl_total_fragment_steps = args.total_steps * args.fragments
    elif not args.experimental_rl_sync:
        args.fragments = 1
        args.quorum = len(specs)
        args.grace_ms = 0
        args.pipeline = 1
        args.sync_interval_steps = 0.0
        args.delta_correction = "none"
        args.outer_lr = 1.0
        args.outer_momentum = 0.0
        args.merge_alpha = 0.0
        args.wire_dtype = "f32"
    else:
        for name, expected, flag in (
            ("fragments", 1, "--fragments 1"),
            ("pipeline", 1, "--pipeline 1"),
            ("merge_alpha", 0.0, "--merge-alpha 0"),
            ("wire_dtype", "f32", "--wire-dtype f32"),
        ):
            if getattr(args, name) != expected:
                raise ValueError(
                    f"RL's one-fragment f32 bridge still requires {flag}"
                )
    if args.spot:
        _rl_checkpoint_mount(args.rl_completed_groups_path)

    provenance = getattr(args, "_provenance", None)
    if provenance and rl_engine == "ports":
        # Legacy provenance stays byte-identical; ports is recorded.
        provenance["rl_engine"] = "ports"
    if provenance:
        for name in ("model", "dataset"):
            source = provenance.get(name) or {}
            if (
                name == "dataset"
                and allow_local_data
                and source.get("source") == "local"
                and source.get("resolved_revision") is None
            ):
                continue
            if (
                name == "model"
                and allow_remote_model
                and source.get("source") == "remote-local"
                and source.get("resolved_revision")
            ):
                continue
            if source.get("source") != "huggingface" or not source.get(
                "resolved_revision"
            ):
                raise ValueError(
                    f"RL v0 requires a revision-pinned Hugging Face {name}"
                )
        from .provenance import python_spec_path, python_spec_sha256

        reward_path = python_spec_path(args.reward_function, base_dir=REPO_ROOT)
        try:
            reward_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                "RL reward source must be inside the synced Yeto workdir"
            ) from exc
        args.reward_sha256 = python_spec_sha256(
            args.reward_function, base_dir=REPO_ROOT
        )
    # The fixed Miles commit enables remote model code in its internal
    # Megatron and SGLang loaders. Keep that trust decision explicit through
    # Yeto's existing flag.
    if not args.trust_remote_code:
        raise ValueError("pinned Miles requires explicit --trust-remote-code")


# AWS keeps this SSM parameter pointing at the CURRENT Deep Learning Base
# OSS-NVIDIA-driver AMI per region (open kernel modules, Blackwell-ready),
# so resolving at launch time needs no region x AMI table of our own.
_DLAMI_SSM_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
)


def resolve_blackwell_image(region: str) -> str | None:
    """Current Blackwell-capable DLAMI for a region, or None (callers then
    rely on the setup-time driver remediation instead)."""
    try:
        import boto3

        ssm = boto3.client("ssm", region_name=region)
        return ssm.get_parameter(Name=_DLAMI_SSM_PARAM)["Parameter"]["Value"]
    except Exception as e:  # noqa: BLE001 - degrade to driver remediation
        print(
            f"[launcher] could not resolve a Blackwell DLAMI for {region} ({e}); "
            "relying on setup-time driver install",
            file=sys.stderr,
        )
        return None


# Internal image-override table: (cloud, GPU) pairs whose provider-default
# image is known stale/broken, mapped to a `region -> image id` resolver.
# First entry earned in production: sky's pinned AMI ships driver 535,
# which never binds to SM100 silicon. Second class earned the same way:
# NVSwitch instances (p4d/p4de A100, p5 H100, p5e H200) need
# nvidia-fabricmanager running before CUDA will initialize at all
# (cudaGetDeviceCount -> Error 802), and sky's pinned AMI does not ship it;
# the DL Base GPU AMI does, preinstalled and enabled. Extend here as new
# GPU generations outpace provider image pins; an explicit --learner-image
# always wins, and a resolver returning None degrades to the setup-time
# driver install.
GPU_IMAGE_OVERRIDES: dict[tuple[str, str], object] = {
    ("aws", "B200"): resolve_blackwell_image,
    # NVSwitch (fabric manager required); same current DL Base GPU AMI.
    ("aws", "A100"): resolve_blackwell_image,
    ("aws", "A100-80GB"): resolve_blackwell_image,
    ("aws", "H100"): resolve_blackwell_image,
    ("aws", "H200"): resolve_blackwell_image,
}


# The Megatron backend runs INSIDE an NGC container that ships the whole
# stack prebuilt (torch + Transformer Engine + megatron-core + megatron-bridge)
# — the pip-on-DLAMI install proved intractable (two shakedowns; see
# docs/MEGATRON.md). sky runs the task in this container via `image_id:
# docker:...`; the host still supplies the GPU kernel driver, so B200 needs
# sky's default AWS docker host AMI to carry driver >=570 open-kernel (the one
# integration unknown that still needs a live B200 check). NeMo images ship
# megatron-bridge; pull requires nvcr.io auth (an NGC key on the host).
MEGATRON_IMAGE = "docker:nvcr.io/nvidia/nemo:25.09"


def learner_image_for(args, spec: ClusterSpec, learner_id: int | None = None):
    """The image for a learner cluster: explicit flag > megatron container >
    internal override table > None (provider default + setup-time remediation)."""
    explicit = parse_image_spec(getattr(args, "learner_image", None))
    if explicit is not None:
        if isinstance(explicit, dict) and learner_id is not None:
            for key in (str(learner_id), f"l{learner_id}"):
                if key in explicit:
                    return explicit[key]
        return explicit
    if getattr(args, "island_backend", "torch") == "megatron":
        return MEGATRON_IMAGE
    resolver = GPU_IMAGE_OVERRIDES.get((spec.cloud, spec.gpu))
    if resolver is None or not spec.region:
        return None
    image = resolver(spec.region)
    if image:
        print(f"[launcher] {spec.gpu} learner: pinning image {image} ({spec.region})")
    return image


def parse_image_spec(value: str | None):
    """--learner-image: a single image id/tag applied everywhere, or
    comma-separated region=id pairs -> the region dict sky expects. Numeric
    keys are learner ids and are resolved before region keys; this covers
    providers where a saved OS volume is not a reusable image."""
    if not value:
        return None
    if "=" not in value:
        return value
    images = {}
    for pair in value.split(","):
        key, _, image = pair.partition("=")
        if not key or not image:
            raise ValueError(
                f"bad --learner-image entry {pair!r}; expected region=image-id "
                "or learner-id=image-id"
            )
        images[key.strip()] = image.strip()
    return images


# The Megatron island backend needs, on top of the cu128 torch from
# TORCH_SETUP: Transformer Engine (FP8 + MoE grouped GEMM + Blackwell),
# megatron-core, and megatron-bridge (HF->mcore import + LoRA-on-MoE). Two
# fragilities the research flagged: (1) mcore/bridge's [te] extra hard-codes
# the CUDA-13 TE flavor, so install core_cu12 explicitly against cu128 torch;
# (2) bridge pins a narrow transformers range, so let it resolve transformers
# rather than our 5.13.0 pin. apex is no longer required (TE provides the
# fused kernels). Production should instead pin an NGC PyTorch image with all
# of this prebuilt via --learner-image; this pip path is the from-scratch
# fallback and is best-effort (a failure disables --island-backend megatron,
# it does not abort a torch-backend island).
# The megatron stack on the DLAMI (reverse-engineered from the mega1 shakedown
# + a free local repro). Each step addresses a real failure mode:
#  * Transformer Engine: the prebuilt `transformer-engine-cu12` wheel — the
#    `[pytorch]` extra source-builds and its subprocess can't see torch.
#  * megatron-bridge: no wheel, must build from source, which needs (a) torch
#    importable at build time -> `--no-build-isolation`, (b) `wheel`+setuptools
#    in the env, and (c) `nvcc` on PATH (its setup.py shells out to nvcc for
#    the CUDA "bare metal version"; without it the build NameErrors). The
#    DLAMI ships CUDA at /usr/local/cuda.
# A prebuilt NGC/NeMo image via --learner-image is still the more robust path;
# this makes the DLAMI work without one. See docs/MEGATRON.md.
MEGATRON_SETUP = (
    "export PATH=/usr/local/cuda/bin:$PATH; "
    "pip install -q wheel setuptools packaging && "
    "pip install -q megatron-core transformer-engine-cu12 && "
    "pip install -q --no-build-isolation megatron-bridge "
    "|| echo '[yeto-setup] megatron stack install failed; --island-backend megatron unavailable' >&2"
)

PROTENIX_SETUP = (
    "pip install -q 'protenix>=2.0.0' "
    "|| echo '[yeto-setup] protenix install failed; Protenix adapter unavailable' >&2"
)
PROTENIX_DIFFUSION_ADAPTER = "yeto.diffusion.adapters.protenix:make_adapter"
HUNYUAN3D_ROOT = "~/Hunyuan3D-2.1"
HUNYUAN3D_SETUP = (
    f"if [ ! -d {HUNYUAN3D_ROOT} ]; then git clone --depth 1 "
    f"https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git {HUNYUAN3D_ROOT}; fi && "
    f"pip install -q -r {HUNYUAN3D_ROOT}/requirements.txt && "
    f"export YETO_HUNYUAN3D_ROOT={HUNYUAN3D_ROOT} "
    "|| echo '[yeto-setup] Hunyuan3D setup failed; Hunyuan3D adapter unavailable' >&2"
)
HUNYUAN3D_DIFFUSION_ADAPTER = "yeto.diffusion.adapters.hunyuan3d:make_adapter"
ALPHAFOLD3_DIFFUSION_ADAPTER = "yeto.diffusion.adapters.alphafold3:make_adapter"

def causal_kernel_setup_steps(args) -> list[str]:
    """Pinned remote installs selected explicitly for a causal torch learner."""
    from .kernel_deps import (
        BITSANDBYTES_MIN_VERSION,
        FLASH_ATTN_VERSION,
        LIGER_KERNEL_VERSION,
        NINJA_VERSION,
        PACKAGING_VERSION,
        PEFT_VERSION,
    )

    steps: list[str] = []
    if getattr(args, "base_quantization", "none") == "nf4":
        steps.append(f"pip install -q 'bitsandbytes>={BITSANDBYTES_MIN_VERSION}'")
    if getattr(args, "kernel_backend", "native") == "liger":
        steps.append(
            f"pip install -q 'liger-kernel=={LIGER_KERNEL_VERSION}' "
            f"'peft=={PEFT_VERSION}'"
        )
    if getattr(args, "attention_backend", "auto") == "flash-attn-2":
        steps.extend(
            [
                f"pip install -q 'ninja=={NINJA_VERSION}' 'packaging=={PACKAGING_VERSION}'",
                f"MAX_JOBS=${{MAX_JOBS:-8}} pip install -q --no-build-isolation "
                f"'flash-attn=={FLASH_ATTN_VERSION}'",
            ]
        )
    return steps

DIFFUSION_SAMPLE_ADAPTER_DIR = "~/yeto-adapter"
DIFFUSION_SAMPLE_OUTPUT_DIR = "~/yeto-output"
RL_INITIAL_ADAPTER_PATH = "~/yeto-rl-initial-adapter"
RL_HEAD_INITIAL_ADAPTER_PATH = "~/yeto-rl-initial-adapter-src"


def _rl_checkpoint_mount(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not (value.startswith("~/") or value.startswith("/"))
        or value.endswith("/")
        or ".." in path.parts
        or str(path.parent) in {".", "~", "/"}
    ):
        raise ValueError(
            "Spot RL --rl-completed-groups-path must be a file in an "
            "absolute or ~/ subdirectory"
        )
    return str(path.parent)


def _rl_checkpoint_storage_name(cluster_prefix: str, learner_id: int) -> str:
    stem = re.sub(r"[^a-z0-9-]+", "-", cluster_prefix.lower()).strip("-") or "yeto"
    suffix = f"-{hashlib.sha256(cluster_prefix.encode()).hexdigest()[:8]}-rl-{learner_id}"
    return stem[: 63 - len(suffix)].rstrip("-") + suffix


DOCKER_LOGIN_UNSET = (
    "unset SKYPILOT_DOCKER_USERNAME SKYPILOT_DOCKER_PASSWORD SKYPILOT_DOCKER_SERVER\n"
)


# fix-verda-provider C block (design D6): clouds whose sky adapter refuses
# `image_id: docker:` (sky/clouds/verda.py: DOCKER_IMAGE unsupported). The
# island is launched as a bare VM task instead; the host-side setup logs in
# to the registry and pulls the digest-pinned image, and the host-side run
# executes the ORIGINAL setup + run inside `docker run --gpus all --net=host`
# with the same ~ paths bind-mounted (sky's ssh user on Verda is root, so
# ~ is /root on both sides). Environment variables are forwarded by NAME
# (`-e KEY`), so no value ever appears in a script or a log. Other clouds
# keep sky's own docker runtime: nothing below runs for them.
IN_VM_DOCKER_CLOUDS = frozenset({"verda"})
IN_VM_DOCKER_DIR = "~/yeto-island"
# Host paths bind-mounted into the container at the same path under /root.
IN_VM_DOCKER_MOUNTS = (
    "~/sky_workdir",
    "~/yeto-output",
    "~/yeto-rl",
    "~/.cache/huggingface",
    RL_INITIAL_ADAPTER_PATH,
)
# sky's own variables the task scripts read (set by sky in the host job env).
IN_VM_DOCKER_SKY_ENVS = (
    "SKYPILOT_NODE_IPS",
    "SKYPILOT_NODE_RANK",
    "SKYPILOT_NUM_NODES",
    "SKYPILOT_NUM_GPUS_PER_NODE",
    "SKYPILOT_TASK_ID",
    "SKYPILOT_CLUSTER_INFO",
)
_IN_VM_SETUP_EOF = "YETO_ISLAND_SETUP_EOF"
_IN_VM_RUN_EOF = "YETO_ISLAND_RUN_EOF"
# Inside the container: same conveniences sky's docker runtime provides
# (root without sudo, non-interactive apt) plus git trusting the mounted
# checkouts.
IN_VM_DOCKER_PRELUDE = (
    "#!/bin/bash\n"
    "export HOME=/root DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1\n"
    "command -v sudo >/dev/null 2>&1 || { printf '#!/bin/sh\\nexec \"$@\"\\n' > /usr/local/bin/sudo"
    " && chmod +x /usr/local/bin/sudo; }\n"
    "git config --global --add safe.directory '*' 2>/dev/null || true\n"
    "mkdir -p ~/sky_workdir && cd ~/sky_workdir\n"
)
# Installs the NVIDIA container toolkit when the VM image lacks it (Verda's
# default image ships docker + driver; the toolkit is checked, not assumed).
IN_VM_DOCKER_TOOLKIT_SETUP = (
    'DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"\n'
    "$DOCKER info 2>/dev/null | grep -qi nvidia || {\n"
    "  echo '[yeto-island] installing nvidia-container-toolkit' >&2\n"
    "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
    "sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\n"
    "  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | "
    "sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
    "sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null\n"
    "  sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit\n"
    "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker\n"
    "  $DOCKER info 2>/dev/null | grep -qi nvidia || { echo '[yeto-island] docker has no nvidia runtime' >&2; exit 1; }\n"
    "}\n"
)


def in_vm_docker_image(rl_image: str) -> str:
    """The plain image reference for `docker pull` (sky's `docker:` prefix off)."""
    ref = (rl_image or "").strip()
    return ref[len("docker:"):] if ref.startswith("docker:") else ref


def in_vm_docker_setup(rl_image: str, *, login: bool) -> str:
    """Host-side setup for an IN_VM_DOCKER cloud: toolkit check, registry
    login from the SKYPILOT_DOCKER_* secrets sky exported (never echoed),
    then `docker pull` of the digest-pinned image. Ends with the login
    variables unset, like every other yeto task script."""
    image = in_vm_docker_image(rl_image)
    lines = ["set -e", IN_VM_DOCKER_TOOLKIT_SETUP.rstrip("\n")]
    if login:
        lines.append(
            'if [ -n "${SKYPILOT_DOCKER_PASSWORD:-}" ]; then '
            'printf \'%s\' "$SKYPILOT_DOCKER_PASSWORD" | '
            '$DOCKER login "${SKYPILOT_DOCKER_SERVER:-ghcr.io}" -u "$SKYPILOT_DOCKER_USERNAME" --password-stdin; fi'
        )
        lines.append(DOCKER_LOGIN_UNSET.rstrip("\n"))
    lines.append('echo "[yeto-island] docker pull start $(date -u +%FT%TZ)"')
    lines.append(f"$DOCKER pull -q {shlex.quote(image)}")
    lines.append('echo "[yeto-island] docker pull done $(date -u +%FT%TZ)"')
    lines.append(f"$DOCKER run --rm --gpus all {shlex.quote(image)} nvidia-smi -L")
    return "\n".join(lines) + "\n"


def in_vm_docker_run(rl_image: str, setup: str, run: str, env_names, learner_id: int) -> str:
    """Host-side run for an IN_VM_DOCKER cloud: writes the original setup and
    run scripts to IN_VM_DOCKER_DIR (0600) and executes them in one
    `docker run` with the island's ~ paths mounted, GPUs, host network/IPC
    and sky's shm/capability options. The container's exit code is the
    job's (exec). `env_names` are forwarded by name only."""
    image = in_vm_docker_image(rl_image)
    for marker, body in ((_IN_VM_SETUP_EOF, setup), (_IN_VM_RUN_EOF, run)):
        if marker in body:
            raise ValueError(f"island script contains the heredoc marker {marker}")
    names = list(dict.fromkeys(list(IN_VM_DOCKER_SKY_ENVS) + sorted(env_names)))
    env_flags = " ".join(f"-e {n}" for n in names)
    mounts = " ".join(
        f'-v "$HOME/{m[2:]}:/root/{m[2:]}"' for m in IN_VM_DOCKER_MOUNTS
    )
    mkdirs = " ".join(m for m in IN_VM_DOCKER_MOUNTS) + f" {IN_VM_DOCKER_DIR}"
    cname = f"yeto-island-{learner_id}"
    inner = (
        f"bash /root/{IN_VM_DOCKER_DIR[2:]}/setup.sh && "
        'echo "[yeto-island] in-VM docker setup done $(date -u +%FT%TZ)" && '
        f"exec bash /root/{IN_VM_DOCKER_DIR[2:]}/run.sh"
    )
    return (
        "set -e\n"
        'DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"\n'
        f"mkdir -p {mkdirs}\n"
        f"cat > {IN_VM_DOCKER_DIR}/setup.sh <<'{_IN_VM_SETUP_EOF}'\n"
        f"{IN_VM_DOCKER_PRELUDE}{setup}\n"
        f"{_IN_VM_SETUP_EOF}\n"
        f"cat > {IN_VM_DOCKER_DIR}/run.sh <<'{_IN_VM_RUN_EOF}'\n"
        f"{IN_VM_DOCKER_PRELUDE}{run}\n"
        f"{_IN_VM_RUN_EOF}\n"
        f"chmod 0600 {IN_VM_DOCKER_DIR}/setup.sh {IN_VM_DOCKER_DIR}/run.sh\n"
        f"$DOCKER rm -f {cname} >/dev/null 2>&1 || true\n"
        "SHM=$(awk '/MemTotal/ {printf \"%dm\", $2/2048}' /proc/meminfo)\n"
        'echo "[yeto-island] in-VM docker run start $(date -u +%FT%TZ)"\n'
        f"exec $DOCKER run --rm --name {cname} --gpus all --net=host --ipc=host --shm-size=$SHM "
        "--cap-add=SYS_ADMIN --cap-add=SYS_RESOURCE --ulimit memlock=-1:-1 "
        f"{env_flags} {mounts} "
        f'-v "$HOME/{IN_VM_DOCKER_DIR[2:]}:/root/{IN_VM_DOCKER_DIR[2:]}" '
        f"-w /root/sky_workdir --entrypoint /bin/bash {shlex.quote(image)} -c {shlex.quote(inner)}\n"
    )


def docker_config_login(image_ref: str | None, path: str | Path | None = None) -> dict[str, str] | None:
    """The ``~/.docker/config.json`` ``auths`` entry for ``image_ref``'s
    registry as a SKYPILOT_DOCKER_* triple, or None when the file or the
    entry is missing (credential helpers are not consulted)."""
    from .modal_runner import DOCKER_LOGIN_ENV_VARS, registry_host

    if not image_ref:
        return None
    path = Path(path) if path else Path.home() / ".docker" / "config.json"
    try:
        auths = json.loads(path.read_text(encoding="utf-8")).get("auths") or {}
    except (OSError, ValueError):
        return None
    host = registry_host(image_ref)
    entry = next((auths[k] for k in (host, f"https://{host}", f"https://{host}/v1/") if isinstance(auths.get(k), dict)), None)
    if not entry or not entry.get("auth"):
        return None
    import base64

    try:
        user, password = base64.b64decode(entry["auth"]).decode().split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return None
    if not user or not password:
        return None
    return dict(zip(DOCKER_LOGIN_ENV_VARS, (user, password, host)))


def registry_login_for(args, environ=None) -> dict[str, str] | None:
    """The private-registry login a launch injects for ``--rl-image``, or
    None (the default: a public image, nothing injected anywhere -- not the
    island task secrets, not the Modal pull secret, not the head job).

    Injection happens only when (a) the SKYPILOT_DOCKER_* variables for
    the image's registry are in ``environ`` (ports engine only, as before),
    or (b) ``--rl-image-private`` was given: then the environment login is
    used if present, else the registry's ``~/.docker/config.json`` entry;
    neither being available is an error rather than a silent public pull."""
    from .modal_runner import registry_credentials

    from .rl import default_rl_image

    environ = os.environ if environ is None else environ
    engine = getattr(args, "rl_engine", "ports") or "ports"
    # before prepare_launch_args (the head CLI) --rl-image may still be unset
    image = getattr(args, "rl_image", None) or default_rl_image(engine)
    explicit = bool(getattr(args, "rl_image_private", False))
    if not explicit and engine != "ports":
        return None
    login = registry_credentials(image, environ)
    if login is None and explicit:
        login = docker_config_login(image)
        if login is None:
            raise ValueError(
                f"--rl-image-private: no registry login for {image!r}: set "
                "SKYPILOT_DOCKER_USERNAME/PASSWORD/SERVER or add the registry to ~/.docker/config.json"
            )
    return login


def _sky_docker_login_config(login: dict[str, str]):
    """SkyPilot's DockerLoginConfig for a SKYPILOT_DOCKER_* triple."""
    from sky.provision.docker_utils import DockerLoginConfig

    return DockerLoginConfig.from_env_vars(login)


def _miles_source_setup(rl_engine: str = "ports", overlay: str | None = None) -> tuple[str, str]:
    """Return the (miles_setup, sglang_setup) remote steps for ``rl_engine``.

    ``overlay`` (ports only, default None = off): a :mod:`yeto.rl.adapters.miles.overlay`
    patch applied to the image's ~/miles after the checkout checks.
    """

    from .rl import (
        MILES_BASE_COMMIT,
        MILES_BUNDLE_PATH,
        MILES_BUNDLE_SHA256,
        MILES_COMMIT,
        MILES_NEXT_COMMIT,
        MILES_NEXT_IMAGE_MANIFEST,
        MILES_NEXT_IMAGE_SGLANG_ROOT,
        MILES_NEXT_REPOSITORY,
        MILES_PEFT_VERSION,
        MILES_REPOSITORY,
        SGLANG_COMMIT,
        SGLANG_NEXT_COMMIT,
        SGLANG_NEXT_REPOSITORY,
        SGLANG_REPOSITORY,
    )

    if overlay is not None and rl_engine != "ports":
        raise ValueError("a Miles overlay needs the ports engine")
    if rl_engine == "ports":
        from .rl.miles_overlay import overlay_setup

        def checkout(
            path: str, repository: str, commit: str, refreshed_flag: str = ""
        ) -> str:
            repo = shlex.quote(repository)
            # ``refreshed_flag``: shell variable set to 1 when the checkout
            # had to move to the pin (so the caller knows the image's
            # install of it can no longer be trusted as-is).
            mark = f"{refreshed_flag}=1\n" if refreshed_flag else ""
            return (
                f"if [ ! -d {path}/.git ]; then git clone --no-checkout "
                f"{repo} {path}; fi\n"
                f"git -C {path} remote set-url origin {repo}\n"
                f'test "$(git -C {path} config --get remote.origin.url)" = '
                f"{repo}\n"
                # MILES_NEXT_IMAGE already has the pinned fork checked out
                # at ~/miles (/root/miles); only fetch when it is not HEAD.
                f'if [ "$(git -C {path} rev-parse HEAD 2>/dev/null)" != '
                f"{commit} ]; then\n"
                f"git -C {path} fetch --depth 1 origin {commit}\n"
                f"git -C {path} checkout --detach {commit}\n"
                + mark
                + "fi\n"
                f'test "$(git -C {path} rev-parse HEAD)" = {commit}\n'
                f'test "$(git -C {path} rev-parse --abbrev-ref HEAD)" = HEAD\n'
                f'test -z "$(git -C {path} status --porcelain '
                '--untracked-files=all)"\n'
            )

        # MILES_NEXT_IMAGE ships the pinned SGLang fork installed editable
        # at /sgl-workspace/sglang: ~/sglang (first on the island's
        # PYTHONPATH) becomes a link to it.  Any other image, or an existing
        # ~/sglang directory, gets the clone + install at ~/sglang as before.
        image_root = shlex.quote(MILES_NEXT_IMAGE_SGLANG_ROOT)
        sglang_in_image = (
            f'{{ [ ! -e ~/sglang ] || [ "$(readlink ~/sglang)" = {image_root} ]; }} && '
            f'[ "$(git -C {image_root} rev-parse HEAD 2>/dev/null)" = '
            f"{SGLANG_NEXT_COMMIT} ] && "
            f'[ "$(git -C {image_root} config --get remote.origin.url)" = '
            f"{shlex.quote(SGLANG_NEXT_REPOSITORY)} ] && "
            f'[ -z "$(git -C {image_root} status --porcelain '
            '--untracked-files=all)" ] && '
            "python3 -c 'import os, sys, sglang; sys.exit(0 if os.path.realpath("
            f'sglang.__file__).startswith("{MILES_NEXT_IMAGE_SGLANG_ROOT}/python/'
            "\") else 1)'"
        )
        # MILES_NEXT_IMAGE also ships the pinned Miles fork installed
        # editable at /root/miles (= ~/miles) and records it in
        # MILES_NEXT_IMAGE_MANIFEST.  When ~/miles was already at the pin
        # (no fetch above) and the manifest names that commit at that path,
        # the image's install is kept: the editable re-install is skipped
        # and only a missing/other peft is (re)installed.  Any other image,
        # a moved checkout, or a manifest mismatch gets the full install.
        manifest = shlex.quote(MILES_NEXT_IMAGE_MANIFEST)
        miles_in_image = (
            '[ "$MILES_REFRESHED" = 0 ] && '
            f"python3 -c 'import json, os, sys; "
            f'm = json.load(open("{MILES_NEXT_IMAGE_MANIFEST}"))["miles"]; '
            f'sys.exit(0 if m["commit"] == "{MILES_NEXT_COMMIT}" and '
            'os.path.realpath(m["path"]) == '
            "os.path.realpath(os.path.expanduser(\"~/miles\")) else 1)' "
            "2>/dev/null"
        )
        peft_ok = (
            "python3 -c 'import peft, sys; "
            f"sys.exit(0 if peft.__version__ == \"{MILES_PEFT_VERSION}\" else 1)' "
            "2>/dev/null"
        )
        return (
            "set -e\n"
            "MILES_REFRESHED=0\n"
            + checkout(
                "~/miles", MILES_NEXT_REPOSITORY, MILES_NEXT_COMMIT, "MILES_REFRESHED"
            )
            + f"if {miles_in_image}; then\n"
            f"echo \"[yeto-setup] image provides miles {MILES_NEXT_COMMIT} "
            f'({manifest}); editable install kept"\n'
            f"{peft_ok} || python3 -m pip install -q 'peft=={MILES_PEFT_VERSION}'\n"
            "else\n"
            "python3 -m pip install -q --no-deps -e ~/miles "
            f"'peft=={MILES_PEFT_VERSION}'\n"
            "fi"
            + (f"\n{overlay_setup(overlay)}" if overlay else ""),
            f"if {sglang_in_image}; then\n"
            f"ln -sfn {image_root} ~/sglang\n"
            f"echo '[yeto-setup] image provides sglang {SGLANG_NEXT_COMMIT}'\n"
            "else\n"
            + checkout("~/sglang", SGLANG_NEXT_REPOSITORY, SGLANG_NEXT_COMMIT)
            + "python3 -m pip install -q --no-deps -e ~/sglang/python\n"
            "fi",
        )
    if rl_engine != "legacy":
        raise ValueError(f"unknown rl_engine {rl_engine!r}")
    miles_setup = (
        "set -e\n"
        f"MILES_BUNDLE=~/sky_workdir/{MILES_BUNDLE_PATH}\n"
        'test -f "$MILES_BUNDLE" && test ! -L "$MILES_BUNDLE"\n'
        f"printf '%s  %s\\n' {MILES_BUNDLE_SHA256} \"$MILES_BUNDLE\" "
        "| sha256sum --check -\n"
        f"if [ ! -d ~/miles/.git ]; then git clone --no-checkout "
        f"{shlex.quote(MILES_REPOSITORY)} ~/miles; fi\n"
        # An image may ship its own Miles clone from another remote (the
        # public radixark/miles images do); the runtime verifier requires
        # the pinned repository as origin.
        f"git -C ~/miles remote set-url origin {shlex.quote(MILES_REPOSITORY)}\n"
        f"git -C ~/miles fetch --depth 1 origin {MILES_BASE_COMMIT}\n"
        f"git -C ~/miles checkout --detach {MILES_BASE_COMMIT}\n"
        'git -C ~/miles bundle verify "$MILES_BUNDLE" >/dev/null\n'
        f'git -C ~/miles fetch "$MILES_BUNDLE" {MILES_COMMIT}\n'
        f"git -C ~/miles checkout --detach {MILES_COMMIT}\n"
        f'test "$(git -C ~/miles rev-parse HEAD)" = {MILES_COMMIT}\n'
        'test -z "$(git -C ~/miles status --porcelain --untracked-files=all)"\n'
        f"python3 -m pip install -q --no-deps -e ~/miles "
        f"'peft=={MILES_PEFT_VERSION}'"
    )
    sglang_setup = (
        f"if [ ! -d ~/sglang/.git ]; then git clone --no-checkout "
        f"{shlex.quote(SGLANG_REPOSITORY)} ~/sglang; fi\n"
        f"git -C ~/sglang fetch --depth 1 {shlex.quote(SGLANG_REPOSITORY)} "
        f"{SGLANG_COMMIT}\n"
        f"git -C ~/sglang checkout --detach {SGLANG_COMMIT}\n"
        "python3 -m pip install -q --no-deps -e ~/sglang/python"
    )
    return miles_setup, sglang_setup


# --- Stock Codex run bundle (ports engine; design R-SCOPE / T3 G2) ---------------
#
# The legacy direct SSH harness freezes the attested Codex artifact into the run
# bundle and ``--volume``-mounts it at /opt/yeto/codex (ssh_harness.py).  The
# sky / Modal launch paths do the same from a local bundle directory
# (``scripts/fetch_codex_bundle.py``): the launcher attests it with the same
# ``_codex_harness_contract`` the SSH harness uses, mounts the directory at
# ``CODEX_CONTAINER_DIR`` (sky file_mounts / Modal add_local_dir) and injects
# the ``YETO_CODEX_*`` environment the container preflights check
# (``yeto.rl.adapters.miles.island_entry._preflight_codex_harness`` expected_env and
# ``codex_harness_agent._attest_runtime``); every value comes from the contract.
CODEX_BUNDLE_DIR_ENV = "YETO_CODEX_BUNDLE_DIR"
CODEX_CONTAINER_DIR = "/opt/yeto/codex"
CODEX_BUNDLE_FILES = (
    "codex_harness_binary",
    "codex_package_manifest",
    "codex_app_server_schema",
)
# Harness hook / provider env the ports island preflight resolves
# (entry.HARNESS_PREFLIGHT_ENV, preflight.ENVIRONMENT_PROVIDER_ENV); the
# provider must come from the launching environment (no in-tree default).
HARNESS_PREFLIGHT_ENV = "YETO_HARNESS_PREFLIGHT"
HARNESS_ENVIRONMENT_PROVIDER_ENV = "YETO_HARNESS_ENVIRONMENT_PROVIDER"
HARNESS_PASSTHROUGH_ENV = (
    HARNESS_ENVIRONMENT_PROVIDER_ENV,
    "TBENCH_REWARD_HMAC_KEY",
    # tb2_provider Modal Sandbox backend: the island creates task sandboxes
    # itself, so it needs the Modal token; the TB2 knobs/faults ride along.
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "OPENENV_RUN_ID",
    "SECRLENV_MAX_TURNS",
)
HARNESS_PASSTHROUGH_ENV_PREFIXES = ("YETO_HARNESS_TB2_",)
# tb2_provider's Modal Sandbox backend runs inside the island (rollout
# workers create the task sandboxes), so the island python needs the Modal
# client.  The ports image does not ship it; install it in setup without
# touching the image's protobuf 7 (modal pins <7 but works; verified in-image
# on ghcr 12fcd9e5: Sandbox create/exec/terminate, ray/sglang/miles import).
MODAL_SANDBOX_PROVIDER = "yeto.rl.harness.codex.tb2_provider:modal_provider"
# rl-agentic-reward-env: same TB2 provider, sandboxes from prebaked images.
PREBAKED_MODAL_SANDBOX_PROVIDER = "yeto.cloud.modal_reward_env:modal_provider"
MODAL_SANDBOX_PROVIDERS = frozenset({MODAL_SANDBOX_PROVIDER, PREBAKED_MODAL_SANDBOX_PROVIDER})
MODAL_CLIENT_SETUP = (
    # --ignore-installed: in a Modal Function container the setup shell
    # already sees Modal's runtime copies (/pkg, /__modal/deps), so a plain
    # install reports "already satisfied" and writes nothing into the island
    # python's site-packages, which is all its own processes can import
    # (codex-smoke-20261003-8/-9).
    "python3 -m pip install -q --no-deps --ignore-installed "
    "'modal==1.5.5' 'grpclib>=0.4.7,<0.4.10' 'synchronicity~=0.12.5' cbor2 toml "
    "types-certifi types-toml watchfiles\n"
    # Verify the packages physically landed in site-packages and import under
    # the run script's PYTHONPATH (what the learner and Ray workers see).
    'PYTHONPATH="$HOME/miles:$HOME/sglang/python:$HOME/sky_workdir${PYTHONPATH:+:$PYTHONPATH}" '
    "python3 -c 'import os, sys, sysconfig; import modal, grpclib, synchronicity; "
    "sp = sysconfig.get_paths()[\"purelib\"]; "
    "missing = [m for m in (\"modal\", \"grpclib\", \"synchronicity\") if not os.path.isdir(os.path.join(sp, m))]; "
    "sys.exit(f\"not in {sp}: {missing}\" if missing else 0)' "
    "|| { echo '[yeto-setup] Modal client unusable in the island python' >&2; exit 1; }"
)
CODEX_COMPACTION_ENV = "YETO_CODEX_COMPACTION_ENABLED"


def codex_bundle_paths(bundle_dir: str) -> dict[str, str]:
    """The three attested files inside a fetched bundle dir, keyed like the SSH
    harness CLI (``--codex-harness-binary`` ...)."""
    from .rl import CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH, CODEX_CONTAINER_BINARY_PATH

    root = Path(bundle_dir).expanduser()
    return {
        "codex_harness_binary": str(root / os.path.basename(CODEX_CONTAINER_BINARY_PATH)),
        "codex_package_manifest": str(root / "codex-package.json"),
        "codex_app_server_schema": str(
            root / os.path.basename(CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH)
        ),
    }


def codex_bundle_contract(args, bundle_dir: str) -> dict:
    """Attest ``bundle_dir`` against the Yeto pins (same checks as the SSH harness)."""
    from argparse import Namespace

    from .rl.ssh_harness import HarnessError, _codex_harness_contract

    if not os.path.isdir(os.path.expanduser(bundle_dir)):
        raise ValueError(f"{CODEX_BUNDLE_DIR_ENV}={bundle_dir!r} is not a directory")
    try:
        return _codex_harness_contract(Namespace(**codex_bundle_paths(bundle_dir)), args)
    except HarnessError as exc:
        raise ValueError(f"Codex bundle {bundle_dir}: {exc}") from exc


def codex_container_env(contract: dict) -> dict[str, str]:
    """``YETO_CODEX_*`` for the island container, mirroring the SSH harness
    ``codex_env`` template and the learner's expected_env, from the contract."""
    from .rl.ssh_harness import _canonical_json, _plan_digest

    backend = contract["backend"]
    env = {
        "YETO_CODEX_BINARY_PATH": contract["container_binary_path"],
        "YETO_CODEX_BINARY_SHA256": contract["binary_sha256"],
        "YETO_CODEX_BINARY_SIZE_BYTES": str(contract["binary_size_bytes"]),
        "YETO_CODEX_VERSION": contract["cli_version"],
        "YETO_CODEX_APP_SERVER_PROTOCOL_REVISION": contract["app_server_protocol_revision"],
        "YETO_CODEX_APP_SERVER_SCHEMA_SHA256": contract["app_server_schema_sha256"],
        "YETO_CODEX_BASE_INSTRUCTIONS_SHA256": contract["base_instructions_sha256"],
        "YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": contract["terminal_exec_tool_schema_sha256"],
        "YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256": contract["submit_tool_schema_sha256"],
        "YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256": contract["dynamic_tools_schema_sha256"],
        "YETO_CODEX_REASONING_EFFORT": contract["reasoning_effort"],
        "YETO_CODEX_BACKEND_MAX_TOKENS": str(backend["max_tokens"]),
        "YETO_CODEX_BACKEND_REASONING_EFFORT": backend["reasoning_effort"],
        "YETO_CODEX_BACKEND_THINKING": backend["thinking"]["type"],
        "YETO_CODEX_CHAT_TEMPLATE": backend["chat_template"],
        "YETO_CODEX_CHAT_TEMPLATE_KWARGS": _canonical_json(backend["chat_template_kwargs"]),
        "YETO_CODEX_TITO_ALLOWED_APPEND_ROLES": "tool,user",
    }
    env.update({str(k): str(v) for k, v in contract.get("openenv_identity_env", {}).items()})
    env["YETO_CODEX_HARNESS_CONTRACT_SHA256"] = _plan_digest(contract)
    return env


def rl_gae_variant(args) -> str | None:
    """``advantage.gae_variant`` of the prepared RL algorithm spec (None if absent)."""
    raw = getattr(args, "rl_algorithm_spec_json", None)
    if getattr(args, "training_mode", "sft") != "rl" or not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    value = ((payload or {}).get("advantage") or {}).get("gae_variant")
    return None if value is None else str(value)


def compactionrl_launch_env(args, environ, harness_compacts: bool) -> dict[str, str]:
    """Rollout-worker env for CompactionRL (design D8; progress "S13 Codex 压缩接线").

    A ``cross_segment_per_sample`` spec switches ``YETO_CODEX_COMPACTIONRL=1`` on
    (and forwards ``YETO_CODEX_COMPACTIONRL_T_COMP`` from the launching
    environment when set); the island preflight forwards every ``YETO_CODEX_*``
    to the Ray rollout workers.  Any other spec returns {} (env unchanged).
    Raises when spec, harness and the launching environment disagree
    (``compactionrl.check_rollout_compaction``)."""
    from .rl.algos import compactionrl as crl

    variant = rl_gae_variant(args)
    raw_switch = environ.get(crl.COMPACTION_SWITCH_ENV)
    t_comp = environ.get(crl.COMPACTION_T_COMP_ENV)
    if variant is None and not raw_switch and not t_comp:
        return {}
    try:
        explicit = crl.compaction_switch(raw_switch)
        wanted = variant == crl.COMPACTED_GAE_VARIANT
        if raw_switch not in (None, "") and explicit != wanted:
            if wanted:
                raise ValueError(
                    f"{crl.COMPACTION_SWITCH_ENV}={raw_switch!r} in the launching environment "
                    f"contradicts the CompactionRL spec ({crl.COMPACTED_GAE_VARIANT})"
                )
        crl.check_rollout_compaction(
            variant, wanted or explicit, harness_compacts=harness_compacts, t_comp=t_comp
        )
    except ValueError as error:
        raise ValueError(f"CompactionRL launch: {error}") from error
    if not wanted:
        return {}
    env = {crl.COMPACTION_SWITCH_ENV: "1"}
    if t_comp:
        env[crl.COMPACTION_T_COMP_ENV] = str(t_comp).strip()
    return env


def codex_harness_launch(args, environ=None) -> tuple[str, dict[str, str], dict[str, str]] | None:
    """(learner flags, container envs, file_mounts) for a signed Codex agent on
    the ports engine, or None when the run is not a Codex run.

    Fails closed: no bundle dir, a bundle that does not match the pins, no
    environment provider, or trainable compaction switched on in the launching
    environment (R-D5a) all raise before any cloud resource is requested."""
    from .rl import CODEX_OPENENV_AGENT, SIGNED_CODEX_AGENTS
    from .rl.harness.codex.preflight import HARNESS_PREFLIGHT_SPEC, _FALSE

    environ = os.environ if environ is None else environ
    custom_agent = getattr(args, "custom_agent_function_path", None)
    from .rl.synthetic_segments import SYNTHETIC_SEGMENTS_REWARD

    # TEST ONLY (task 6.4 G1): the synthetic two-segment reward stands in for a
    # compacting harness; it is accepted only with cross_segment_per_sample.
    synthetic = getattr(args, "reward_function", None) == SYNTHETIC_SEGMENTS_REWARD
    if synthetic and rl_gae_variant(args) != "cross_segment_per_sample":
        raise ValueError(f"{SYNTHETIC_SEGMENTS_REWARD} is only valid with gae_variant cross_segment_per_sample")
    compaction_env = compactionrl_launch_env(
        args, environ, custom_agent == CODEX_OPENENV_AGENT or (synthetic and custom_agent is None))
    if custom_agent not in SIGNED_CODEX_AGENTS:
        return None
    if getattr(args, "rl_engine", "ports") != "ports":
        raise ValueError(
            "the signed Codex harness requires the direct SSH harness so its "
            "Linux binary can be attested and frozen into the run bundle"
        )
    bundle_dir = getattr(args, "codex_bundle_dir", None) or environ.get(CODEX_BUNDLE_DIR_ENV)
    if not bundle_dir:
        raise ValueError(
            f"the signed Codex harness on the ports engine needs {CODEX_BUNDLE_DIR_ENV} "
            "(a bundle from scripts/fetch_codex_bundle.py) to mount at "
            f"{CODEX_CONTAINER_DIR}"
        )
    if environ.get(CODEX_COMPACTION_ENV, "").strip().lower() not in _FALSE:
        raise ValueError(
            f"{CODEX_COMPACTION_ENV} is set in the launching environment; the fork pin "
            "session server has no trainable compaction (R-D5a)"
        )
    contract = codex_bundle_contract(args, bundle_dir)
    flags = f" --codex-harness-contract {shlex.quote(_json_compact(contract))}"
    envs = codex_container_env(contract)
    if custom_agent == CODEX_OPENENV_AGENT:
        envs[HARNESS_PREFLIGHT_ENV] = environ.get(HARNESS_PREFLIGHT_ENV) or HARNESS_PREFLIGHT_SPEC
        if not environ.get(HARNESS_ENVIRONMENT_PROVIDER_ENV):
            raise ValueError(
                f"{HARNESS_ENVIRONMENT_PROVIDER_ENV}=module:callable is required: the "
                "Codex Terminal-Bench preflight fails closed without an environment provider"
            )
        for name in HARNESS_PASSTHROUGH_ENV:
            if environ.get(name):
                envs[name] = environ[name]
        for name, value in environ.items():
            if name.startswith(HARNESS_PASSTHROUGH_ENV_PREFIXES) and value:
                envs[name] = value
    envs.update(compaction_env)  # {} unless the spec is CompactionRL
    mounts = {CODEX_CONTAINER_DIR: str(Path(bundle_dir).expanduser().resolve())}
    return flags, envs, mounts


def _json_compact(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# Per-node vCPU floor for RL (miles/ports) islands. Ray must fit every CPU
# actor next to the GPU workers; on a 1-GPU island miles requests roughly:
# InferenceController 1 + TrainerController 1 + RolloutExecutor 1 +
# registration reporter 1 (inference deploys) + MegatronTrainRayActor 0.4/GPU
# + RayWorkerManager helpers 0.2 each + CommandActor/router 0.001 each, i.e.
# ~4+ CPUs before sglang's own processes. A 4-vCPU node (AWS g5.xlarge /
# g6e.xlarge) left Ray with "Pending Demands: {'CPU': 1.0}" and the job hung
# (aws-g0-20261004c). 8+ keeps headroom; Nebius' smallest L40S (8 vCPU) and
# H100/H200 shapes (16/128 vCPU) already satisfy it, so they are unchanged.
# Applies per node, so multi-node islands get the same floor. An explicit
# --learner-cpus or --learner-instance-type wins. Pre-provisioned pools
# (ssh/kubernetes) and Modal size themselves and are left untouched.
RL_ISLAND_MIN_CPUS = "8+"


MODEL_STORE_MOUNT = "/mnt/yeto-models"


def model_store_filesystem(uri, cloud, region):
    """Nebius filesystem id for ``--model-store nebius-fs://<id>`` on a
    ``nebius@region`` island, else None (with a WARNING when a store was
    asked for but cannot be attached there)."""
    if not uri:
        return None
    scheme, sep, fs_id = uri.partition("://")
    if scheme != "nebius-fs" or not sep or not fs_id.startswith("computefilesystem-"):
        raise ValueError(f"--model-store: expected nebius-fs://computefilesystem-..., got {uri!r}")
    if cloud != "nebius" or not region:
        print(f"[launcher] WARNING: --model-store {uri} needs a nebius@<region> island "
              f"(got {cloud}@{region}); models download from the Hub")
        return None
    return fs_id.rstrip("/")


def model_store_sky_config(fs_id, region):
    """Client-side sky config override that attaches the store filesystem.
    sky 0.13 reads Nebius filesystems only from config
    (nebius.region_configs.<region>.filesystems, sky/clouds/nebius.py), and
    mounts each one with virtiofs during node setup."""
    from sky import skypilot_config

    regional = dict(skypilot_config.get_nested(("nebius", "region_configs", region), {}) or {})
    regional["filesystems"] = [
        # READ_WRITE: Nebius rejects READ_ONLY filesystem attachments
        # ("readOnly mode attachment is not supported for filesystems yet",
        # 2026-10-05).  Nothing writes here: HF_HUB_CACHE points at the store
        # only on a completed-marker hit, when every file is already cached.
        {"filesystem_id": fs_id, "attach_mode": "READ_WRITE", "mount_path": MODEL_STORE_MOUNT}
    ]
    return {"nebius": {"region_configs": {region: regional}}}


def model_store_env(model, revision, mount=MODEL_STORE_MOUNT):
    """Shell: point HF_HUB_CACHE at the store when it holds a COMPLETED
    snapshot of exactly ``model@revision`` (marker written by
    scripts/populate_nebius_model_store.sh after a size check); otherwise
    WARN and leave the Hub download path untouched.  Never fails."""
    if not revision:
        return (f"echo '[yeto-model-store] WARNING: no --model-revision; not using "
                f"{mount}, downloading {model} from the Hub' >&2")
    marker = f"{mount}/yeto-complete/{model.replace('/', '--')}@{revision}.json"
    snap = f"{mount}/hub/models--{model.replace('/', '--')}/snapshots/{revision}"
    return (
        f"if [ -f {shlex.quote(marker)} ] && [ -d {shlex.quote(snap)} ]; then "
        f"export HF_HUB_CACHE={mount}/hub YETO_MODEL_STORE_HIT=1; "
        f"else echo '[yeto-model-store] WARNING: {mount} has no completed {model}@{revision}; "
        "downloading from the Hub' >&2; fi"
    )


def modal_volume_store_env(model, revision, mount=MODEL_STORE_MOUNT, cache="/root/yeto-hub"):
    """Shell: serve ``model@revision`` from a Modal model Volume mounted at
    ``mount`` (``hf/<name>/<rev[:8]>/`` + MANIFEST.json, written by
    infra-drafts/fn-modal/dl_hf.py) through a symlinked HF hub cache, so the
    learner's snapshot_download finds every file locally.  Never fails: on a
    mismatch it WARNs and leaves the Hub download path untouched."""
    if not revision or len(revision) != 40:
        return (f"echo '[yeto-model-volume] WARNING: needs a full 40-hex --model-revision; "
                f"downloading {model} from the Hub' >&2")
    snap = f"{mount}/hf/{model.split('/')[-1]}/{revision[:8]}"
    repo_dir = f"{cache}/models--{model.replace('/', '--')}"
    check = (f"python3 -c \"import json,sys; m=json.load(open(sys.argv[1])); "
             f"sys.exit(0 if m.get('revision')==sys.argv[2] and m.get('repo')==sys.argv[3] and m.get('all_ok') else 1)\" "
             f"{shlex.quote(snap + '/MANIFEST.json')} {revision} {shlex.quote(model)}")
    return (
        f"if [ -f {shlex.quote(snap + '/MANIFEST.json')} ] && {check}; then "
        f"mkdir -p {shlex.quote(repo_dir)}/snapshots {shlex.quote(repo_dir)}/refs && "
        f"ln -sfn {shlex.quote(snap)} {shlex.quote(repo_dir + '/snapshots/' + revision)} && "
        f"echo {revision} > {shlex.quote(repo_dir + '/refs/main')} && "
        f"export HF_HUB_CACHE={cache} YETO_MODEL_STORE_HIT=1 && "
        f"echo '[yeto-model-volume] serving {model}@{revision} from {snap}' >&2; "
        f"else echo '[yeto-model-volume] WARNING: {snap} has no matching MANIFEST; "
        "downloading from the Hub' >&2; fi"
    )


def nebius_baked_image_id(image, cloud, region, baked=None):
    """``image_id`` for an RL island: the docker image alone, or on Nebius a
    ``{region: computeimage-..., "docker": image}`` dict when a VM image with
    exactly that docker digest pre-pulled was baked for ``region``
    (``NEBIUS_BAKED_IMAGES``; COLDSTART-PLAN.md #3).  A region without an
    explicit ``@region`` keeps the plain image: a region-keyed dict would pin
    sky to the baked regions.  A baked entry for a *different* digest is never
    used (it would add a non-default base disk and still pull everything); it
    only produces a WARNING so the stale bake is noticed."""
    if baked is None:
        from yeto.rl import NEBIUS_BAKED_IMAGES as baked
    if cloud != "nebius" or not region or not isinstance(image, str):
        return image
    if os.environ.get("YETO_NEBIUS_NO_BAKED_IMAGE"):  # diagnostic opt-out: stock VM image + docker pull
        print(f"[launcher] nebius/{region}: YETO_NEBIUS_NO_BAKED_IMAGE set; using the stock VM image")
        return image
    if not image.startswith("docker:") or "@sha256:" not in image:
        return image
    digest = "sha256:" + image.rsplit("@sha256:", 1)[1]
    vm_image = baked.get(digest, {}).get(region)
    if vm_image:
        print(f"[launcher] nebius/{region}: VM image {vm_image} has {digest[:19]} pre-pulled")
        return {region: vm_image, "docker": image}
    stale = sorted(d for d, regions in baked.items() if region in regions)
    if stale:
        print(
            f"[launcher] WARNING: nebius/{region} has baked VM images only for "
            f"{', '.join(d[:19] for d in stale)}, not {digest[:19]}; using the stock "
            "image and a full docker pull (re-run scripts/bake_nebius_image.sh)"
        )
    return image


def rl_island_cpus(args, cloud: str | None = None):
    if getattr(args, "learner_cpus", None):
        return args.learner_cpus
    if getattr(args, "learner_instance_type", None):
        return None
    if cloud in ("ssh", "kubernetes", "modal"):
        return None
    return RL_ISLAND_MIN_CPUS


def island_uses_ports_megatron(args) -> bool:
    """Whether the island image ships the editable Megatron-LM at
    PORTS_MEGATRON_PATH: the ports engine always (its default and any
    override are ports images), the legacy engine only when it is pointed at
    the ports image (MILES_NEXT_IMAGE) instead of its own MILES_IMAGE."""
    if getattr(args, "rl_engine", "ports") == "ports":
        return True
    from .rl import MILES_NEXT_IMAGE

    image = (getattr(args, "rl_image", None) or "").strip()

    def strip(ref):
        return ref[len("docker:"):] if ref.startswith("docker:") else ref

    return bool(image) and strip(image) == strip(MILES_NEXT_IMAGE)


def make_miles_island_task(
    args,
    spec: ClusterSpec,
    learner_id: int,
    num_learners: int,
    syncer_addr: str,
):
    """Create one Ray/Miles island from the pinned Miles checkout."""

    import sky

    from .datasource import learner_data_arg, learner_file_mounts
    from .models import resolve
    from .provenance import is_local_reference

    if not getattr(args, "source_sha256", None) or not getattr(
        args, "reward_sha256", None
    ):
        raise ValueError("RL task requires prepared source and reward provenance")
    codex_launch = codex_harness_launch(args)  # None unless a signed Codex agent
    # --rl-island-use-gpus-per-node: island rules see M GPUs/node; resources stay physical.
    island = rl_island_spec(args, spec)

    flags = (
        f" --model {shlex.quote(args.model)}"
        f" --rl-model-recipe {shlex.quote(args.rl_model_recipe)}"
        f" --data {shlex.quote(learner_data_arg(args.data))}"
        + (
            " --rl-single-island-no-sync"
            if getattr(args, "rl_single_island_no_sync", False)
            else " --syncer $SYNCER_ADDR"
            + (" --critic-syncer $CRITIC_SYNCER_ADDR" if rl_needs_critic(args) else "")
        )
        + (" --rl-echo-events" if _echoes_events(args, spec) else "")
        + " --learner-id $LEARNER_ID"
        f" --reward-function {shlex.quote(args.reward_function)}"
        f" --reward-sha256 {shlex.quote(args.reward_sha256)}"
        f" --source-sha256 {shlex.quote(args.source_sha256)}"
        f" --global-rounds {args.total_steps}"
        f" --sync-preset {args.rl_sync_preset}"
        f" --fragments {args.fragments}"
        f" --pipeline {args.pipeline}"
        f" --local-horizon {args.local_rl_rounds_per_sync}"
        f" --total-fragment-steps {args.rl_total_fragment_steps}"
        f" --groups-per-round {args.rollout_batch_size}"
        f" --samples-per-group {args.n_samples_per_prompt}"
        f" --over-sampling-batch-size {args.over_sampling_batch_size}"
        f" --rl-distributed-timeout-minutes {args.rl_distributed_timeout_minutes}"
        f" --optimizer-steps {int(getattr(args, 'rl_optimizer_steps', 1) or 1)}"
        f" --rollout-max-response-len {args.rollout_max_response_len}"
        f" --completed-groups-path {shlex.quote(args.rl_completed_groups_path)}"
        f" --event-tape ~/yeto-output/rl-island-{learner_id}.jsonl"
        f" --actor-num-nodes {rl_trainer_shape(args, island)[0]}"
        f" --actor-num-gpus-per-node {rl_trainer_shape(args, island)[1]}"
        + (f" --rl-island-gpus-per-node {island.gpus_per_node}" if island.num_nodes > 1 else "")
        + rl_island_bundle_map_flag(args, island)
        + (rl_cross_node_flags(args) if spec.num_nodes > 1 else "")
        + f" --tensor-parallel {args.tensor_parallel}"
        f" --pipeline-parallel {args.pipeline_parallel}"
        f" --rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine}"
        f" --sglang-mem-fraction-static {args.sglang_mem_fraction_static}"
        f" --lora-r {args.lora_r}"
        f" --lora-targets {args.lora_targets}"
        f" --inner-lr {args.inner_lr}"
        f" --seq-len {args.seq_len}"
        f" --seed {args.seed}"
        f" --wan-streams {args.wan_streams}"
        " --miles-root ~/miles"
    )
    if getattr(args, "wandb", False):
        flags += " --wandb"
        flags += f" --wandb-project {shlex.quote(args.wandb_project)}"
        flags += f" --wandb-mode {args.wandb_mode}"
        if getattr(args, "wandb_entity", None):
            flags += f" --wandb-entity {shlex.quote(args.wandb_entity)}"
    if getattr(args, "rollout_model", None):
        flags += f" --rollout-model {shlex.quote(args.rollout_model)}"
        flags += (
            " --rollout-model-revision "
            f"{shlex.quote(args.rollout_model_revision)}"
        )
    if getattr(args, "expert_full_count", 0):
        flags += f" --expert-full-count {args.expert_full_count}"
        flags += f" --expert-full-lr {args.expert_full_lr}"
        flags += (
            " --expert-selection-sha256 "
            f"{shlex.quote(args.expert_selection_sha256)}"
        )
        flags += (
            " --expert-selection-contract-sha256 "
            f"{shlex.quote(args.expert_selection_contract_sha256)}"
        )
    dynamic_filter = getattr(args, "dynamic_sampling_filter_path", None)
    if dynamic_filter:
        flags += (
            " --dynamic-sampling-filter-path "
            f"{shlex.quote(dynamic_filter)}"
        )
    if args.dynamic_sampling_max_replacements is not None:
        flags += (
            " --dynamic-sampling-max-replacements "
            f"{args.dynamic_sampling_max_replacements}"
        )
    if getattr(args, "secrlenv_max_infrastructure_replacements", None) is not None:
        flags += (
            " --secrlenv-max-infrastructure-replacements "
            f"{args.secrlenv_max_infrastructure_replacements}"
        )
    if args.rl_offload_train:
        flags += " --rl-offload-train"
    # Always explicit: the learner's own default is ports, so a legacy run
    # must say so (and an older remote learner must not guess).
    flags += f" --rl-engine {getattr(args, 'rl_engine', 'ports')}"
    algorithm_prelude, algorithm_flags = _ports_algorithm_flags(args)
    flags += algorithm_flags
    if args.expert_parallel is not None:
        flags += f" --expert-parallel {args.expert_parallel}"
    for flag, name in (
        ("--rl-prompt-column", "rl_prompt_column"),
        ("--rl-label-column", "rl_label_column"),
        ("--sglang-tp-size", "sglang_tp_size"),
        ("--sglang-dp-size", "sglang_dp_size"),
        ("--sglang-ep-size", "sglang_ep_size"),
        ("--sglang-attention-backend", "sglang_attention_backend"),
        ("--sglang-page-size", "sglang_page_size"),
        ("--sglang-max-running-requests", "sglang_max_running_requests"),
        ("--sglang-chunked-prefill-size", "sglang_chunked_prefill_size"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            flags += f" {flag} {shlex.quote(str(value))}"
    if args.use_rollout_routing_replay:
        flags += " --use-rollout-routing-replay"
    if getattr(args, "apply_chat_template_kwargs", None):
        chat_kwargs = json.dumps(
            args.apply_chat_template_kwargs, sort_keys=True, separators=(",", ":")
        )
        flags += f" --apply-chat-template-kwargs {shlex.quote(chat_kwargs)}"
    if not getattr(args, "sglang_deterministic_inference", True):
        flags += " --no-sglang-deterministic-inference"
    if args.custom_generate_function_path:
        flags += (
            " --custom-generate-function-path "
            f"{shlex.quote(args.custom_generate_function_path)}"
        )
    if getattr(args, "custom_agent_function_path", None):
        flags += (
            " --custom-agent-function-path "
            f"{shlex.quote(args.custom_agent_function_path)}"
        )
    if getattr(args, "agent_max_seq_len", None) is not None:
        flags += f" --agent-max-seq-len {args.agent_max_seq_len}"
    if args.use_session_server:
        flags += " --use-session-server"
        if args.session_server_ip:
            flags += f" --session-server-ip {shlex.quote(args.session_server_ip)}"
        if args.session_server_port:
            ports = " ".join(str(port) for port in args.session_server_port)
            flags += f" --session-server-port {ports}"
        if args.tito_model:
            flags += f" --tito-model {shlex.quote(args.tito_model)}"
        if getattr(args, "codex_backend_profile", None):
            flags += (
                " --codex-backend-profile "
                f"{shlex.quote(args.codex_backend_profile)}"
            )
        if getattr(args, "codex_exclude_reasoning_from_loss", False):
            flags += " --codex-exclude-reasoning-from-loss"
        if getattr(args, "codex_reasoning_effort", None):
            # The learner's _preflight_codex_harness re-validates the stock
            # profile (xhigh) and fails closed without it.
            flags += (
                " --codex-reasoning-effort "
                f"{shlex.quote(args.codex_reasoning_effort)}"
            )
        if getattr(args, "tito_allowed_append_roles", None):
            roles = " ".join(
                shlex.quote(role) for role in args.tito_allowed_append_roles
            )
            flags += f" --tito-allowed-append-roles {roles}"
    if args.model_revision:
        flags += f" --model-revision {shlex.quote(args.model_revision)}"
    if args.data_revision:
        flags += f" --data-revision {shlex.quote(args.data_revision)}"
    if args.trust_remote_code:
        flags += " --trust-remote-code"
    if getattr(args, "rl_initial_adapter", None) is not None:
        flags += (
            f" --initial-adapter {RL_INITIAL_ADAPTER_PATH}"
            " --initial-adapter-sha256 "
            f"{shlex.quote(args.rl_initial_adapter_sha256)}"
        )
    from .rl.miles_overlay import resolve_overlay

    if getattr(_rl_backend_module(args, "launch_flags"), "NEEDS_MILES_SOURCE", True):
        miles_setup, sglang_setup = _miles_source_setup(
            getattr(args, "rl_engine", "ports"), resolve_overlay(args)
        )
    else:  # rl-verl-backend: the engine image carries its own pinned source
        miles_setup = sglang_setup = ": # no Miles/SGLang checkout for this backend"
    model = resolve(args.model)
    if is_local_reference(model):
        prefetch = ": # local model; no Hub prefetch"
    else:
        revision = (
            f" --revision {shlex.quote(args.model_revision)}"
            if args.model_revision
            else ""
        )
        prefetch = (
            f"(nohup huggingface-cli download {shlex.quote(model)}{revision} "
            ">/tmp/hf-prefetch.log 2>&1 &) || true"
        )
    worker_model_fetch = ""
    if spec.num_nodes > 1 and not is_local_reference(model):
        # Same call and cache as the learner's own snapshot_download on the
        # head (python3 of the image; `huggingface-cli` is not on the run
        # shell's PATH there: s1-mn-20261003f), so the two nodes resolve the
        # same snapshot directory.
        fetch_py = (
            "from huggingface_hub import snapshot_download; "
            f"print(snapshot_download(repo_id={model!r}, revision={args.model_revision!r}))"
        )
        worker_model_fetch = (
            f"  python3 -c {shlex.quote(fetch_py)} >/tmp/hf-prefetch-worker.log 2>&1 "
            "|| { echo '[yeto-island] worker could not fetch the model snapshot:' >&2; "
            "tail -5 /tmp/hf-prefetch-worker.log >&2; exit 1; }\n"
        )
    file_mounts = dict(learner_file_mounts(args.data))
    if getattr(args, "rl_initial_adapter", None) is not None:
        file_mounts[RL_INITIAL_ADAPTER_PATH] = os.path.expanduser(
            args.rl_initial_adapter
        )
    local_token = os.path.expanduser(HF_TOKEN_PATH)
    if os.path.isfile(local_token):
        file_mounts[HF_TOKEN_PATH] = local_token
    if codex_launch is not None:
        codex_flags, codex_envs, codex_mounts = codex_launch
        flags += codex_flags
        file_mounts.update(codex_mounts)
    envs = {
        "SYNCER_ADDR": syncer_addr,
        **({"CRITIC_SYNCER_ADDR": critic_syncer_address(syncer_addr)}
           if rl_needs_critic(args) and syncer_addr != "none" else {}),
        "LEARNER_ID": str(learner_id),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Megatron refuses TP>1 or CP>1 without this; it is exported before
        # `ray start` so every Ray worker inherits it.  Harmless at TP1.
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # No NVTE_*_ATTN pins here: the learner passes --attention-backend
        # per recipe (unfused for the generic provider, flash for
        # gated-delta-net hybrids and DeepSeek V4) and Megatron sets the
        # three NVTE variables itself from that choice; a pin that
        # disagrees makes it assert at model construction.
        "CYBERGYM_URL": args.cybergym_url,
        "CYBERGYM_AGENT_ID": args.cybergym_agent_id,
        "CYBERGYM_TIMEOUT": str(args.cybergym_timeout),
    }
    from yeto.rl.profiles import qwen3_8_next as _fn
    if str(args.model or "").rstrip("/").rsplit("/", 1)[-1] in _fn.MODEL_NAMES.values():
        # run_qwen3_8_next.py extra_env_vars, exported before `ray start`
        envs.update(_fn.PORTS_RUNTIME_ENV)
    if args.rl_model_recipe == "deepseek-v4-flash":
        envs.update(
            {
                "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
                "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
                "SGLANG_OPT_FP8_WO_A_GEMM": "0",
                "SGLANG_OPT_FUSE_WQA_WKV": "0",
            }
        )
    if getattr(args, "expert_full_count", 0):
        envs.update(
            {
                "YETO_DSV4_EXPERT_CLONE": "1",
                "YETO_DSV4_EXPERT_FULL": "1",
                "YETO_DSV4_EXPERT_FULL_COUNT": str(args.expert_full_count),
                "YETO_DSV4_EXPERT_FULL_LR": str(args.expert_full_lr),
                "NVTE_GROUPED_LINEAR_SINGLE_PARAM": "0",
            }
        )
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if os.environ.get("CYBERGYM_API_KEY"):
        envs["CYBERGYM_API_KEY"] = os.environ["CYBERGYM_API_KEY"]
    for name in ("CYBERGYM_REWARD_SCHEME", "CYBERGYM_REWARD_VIEW"):
        if os.environ.get(name):
            envs[name] = os.environ[name]
    if codex_launch is not None:
        envs.update(codex_envs)
    envs.update(elastic_debug_delay_env(args))  # 0.25 ({} unless the test switch is given)
    envs.update(elastic_debug_pause_env(args))  # link-pause test switch ({} by default)
    if getattr(args, "wandb", False):
        # RL islands join the same fleet group as the syncer's tape run.
        envs["YETO_RUN_GROUP"] = args.cluster_prefix
        if os.environ.get("WANDB_API_KEY"):
            envs["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    setup_steps = [WAN_TUNING, HF_TOKEN_ENV, miles_setup, sglang_setup]
    if getattr(args, "wandb", False):
        setup_steps.append(
            "pip install -q wandb || echo '[yeto-setup] wandb install failed; "
            "the island will train without telemetry' >&2"
        )
    if getattr(args, "rl_initial_adapter", None) is not None:
        setup_steps.append(f"chmod -R a-w {RL_INITIAL_ADAPTER_PATH}")
    if codex_launch is not None and envs.get(HARNESS_ENVIRONMENT_PROVIDER_ENV) in MODAL_SANDBOX_PROVIDERS:
        setup_steps.append(MODAL_CLIENT_SETUP)
    store_fs = model_store_filesystem(getattr(args, "model_store", None), spec.cloud, spec.region)
    store_env = ""
    if (spec.cloud == "modal" and getattr(args, "modal_model_volume", None)
            and not is_local_reference(model)):
        store_env = modal_volume_store_env(model, args.model_revision) + "\n"
        setup_steps.append(store_env.rstrip("\n"))
        prefetch = f'[ -n "$YETO_MODEL_STORE_HIT" ] || {prefetch}'
    elif store_fs and not is_local_reference(model):
        store_env = model_store_env(model, args.model_revision) + "\n"
        setup_steps.append(store_env.rstrip("\n"))
        prefetch = f'[ -n "$YETO_MODEL_STORE_HIT" ] || {prefetch}'
    setup_steps.append(prefetch)
    # Ports images (radixark/miles) ship their own Miles at /root/miles on
    # PYTHONPATH; the pinned fork checkout must shadow it.  Legacy unchanged.
    island_pythonpath = (
        "$HOME/miles:" if getattr(args, "rl_engine", "ports") == "ports" else ""
    )
    # Ports images install Megatron-LM with `pip install -e .`, whose editable
    # finder maps only megatron.core and megatron.training; megatron.
    # post_training (imported by megatron.training.get_model whenever
    # nvidia-modelopt is importable, i.e. every raw/native-provider run such
    # as Flash-Next) is then unresolvable (S11 try25 fnA).  Miles' own
    # launchers always put the checkout on PYTHONPATH; do the same, after
    # our own sources so it shadows nothing of ours.  The driver's
    # PYTHONPATH reaches every Ray actor through the job runtime_env.
    # S14 G4/G5 (s14-dlr-legacy-20261007b/c): the legacy engine run in the
    # ports image (--rl-image MILES_NEXT_IMAGE) needs the same checkout on
    # PYTHONPATH, so it is added whenever the island image is the ports
    # image, whatever the engine; legacy in its own MILES_IMAGE is unchanged.
    # Necessary but NOT sufficient for legacy there: agentenv/miles imports
    # megatron.training.tokenizer.tokenizer._vocab_size_with_padding, which
    # the image's Megatron-LM (core 0.19, used by the ports fork via
    # megatron.core.tokenizers.utils.build_tokenizer) no longer has (run c
    # failed identically with the path present).  Legacy needs MILES_IMAGE.
    island_megatron_path = (
        f":{PORTS_MEGATRON_PATH}" if island_uses_ports_megatron(args) else ""
    )
    # Private --rl-image (registry_login_for: SKYPILOT_DOCKER_* in the
    # environment, or --rl-image-private): the login goes into the task SECRETS, SkyPilot's supported
    # form: every Task load re-derives the DockerLoginConfig from them
    # (sky/task.py _with_docker_login_config). A DockerLoginConfig placed in
    # Resources does not survive sky 0.13's YAML round trip (Resources.
    # from_yaml_config keeps a dict, the next to_yaml_config calls
    # dataclasses.asdict on it: "asdict() should be called on dataclass
    # instances"; B1 nsmoke). SkyPilot exports secrets into setup/run, so both
    # scripts unset them first. Default (public image): nothing injected.
    registry_login = registry_login_for(args)
    login_unset = DOCKER_LOGIN_UNSET if registry_login else ""
    setup_script = login_unset + "\n".join(setup_steps)
    run_script = (
            f"{login_unset}{HF_TOKEN_ENV}\n"
            f"{store_env}"
            "set -e\n"
            "cd ~/sky_workdir\n"
            f"{island_pre_run_block(getattr(args, 'rl_island_pre_run_script', None))}"
            f"{multinode_env_prelude(spec.cloud, spec.num_nodes)}"
            'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
            # The island's Ray lives in its own temp dir so that cleanup can
            # target it by path.  A whole-machine `ray stop` would also kill
            # SkyPilot's runtime Ray (port 6380, /tmp/ray), after which every
            # status refresh marks the cluster INIT and the head relaunches
            # it forever.  Ray processes carry their session dir on the
            # command line, so pkill by that path never touches sky's.
            f"{rl_island_gpu_env(args, spec)}"
            'MILES_RAY_DIR="$HOME/miles-ray"\n'
            # Same escalation as `ray stop --force` (TERM, short grace, KILL)
            # but scoped to this dir: gcs_server, raylet and the autoscaler
            # monitor ignore a bare SIGTERM for 30 s or more.
            "stop_miles_ray() {\n"
            '  pkill -f "$MILES_RAY_DIR/" >/dev/null 2>&1 || true\n'
            "  for _ in 1 2 3 4 5 6 7 8 9 10; do\n"
            '    pgrep -f "$MILES_RAY_DIR/" >/dev/null 2>&1 || return 0\n'
            "    sleep 1\n"
            "  done\n"
            '  pkill -KILL -f "$MILES_RAY_DIR/" >/dev/null 2>&1 || true\n'
            "}\n"
            "stop_miles_ray\n"
            'if [ "$SKYPILOT_NODE_RANK" = "0" ]; then\n'
            "  ray start --head --node-ip-address=\"$MASTER_ADDR\" "
            # Dashboard on: Miles' --pin-rollout-manager-to-head lists
            # nodes through Ray's state API, which the dashboard serves.
            # On a multi-node island the dashboard must listen on the node
            # ip, not 127.0.0.1: Ray registers the dashboard address in GCS
            # as given, and a Miles manager actor scheduled on a worker node
            # then asks 127.0.0.1:8265 and fails (rl-multinode-island G1,
            # 2026-10-03: ServerUnavailable in compute_ray_pin_head_options).
            '--port=6379 --include-dashboard=true --temp-dir="$MILES_RAY_DIR"'
            + rl_island_ray_gpus(args, spec)
            + (" --dashboard-host=0.0.0.0" if spec.num_nodes > 1 else "") + "\n"
            "  trap stop_miles_ray EXIT\n"
            # Miles calls ray.init(address="auto"), which reads RAY_ADDRESS
            # first and otherwise sky's /tmp/ray/ray_current_cluster file.
            f"{algorithm_prelude}"
            '  RAY_ADDRESS="$MASTER_ADDR:6379" '
            f"PYTHONPATH={island_pythonpath}$HOME/sglang/python:$HOME/sky_workdir{island_megatron_path}"
            "${PYTHONPATH:+:$PYTHONPATH} "
            f"{getattr(args, 'rl_learner_launch_prefix', '')}python3 -m {_rl_backend_module(args, 'launch_flags').ISLAND_ENTRY_MODULE}{flags}"
            f"{_island_post_cmd(args)}\n"
            "else\n"
            # rl-multinode-island D1: the trap is armed before the join loop so a
            # worker killed while joining still cleans its Miles Ray; the join is
            # bounded (head never came up -> the node exits non-zero instead of
            # looping forever).
            "  trap stop_miles_ray EXIT\n"
            # rl-multinode-island G1 (2026-10-03): the learner (head) resolves
            # the model to its own HF-cache snapshot path and hands that path
            # to the rollout engines; an engine on a worker node needs the
            # same snapshot in the worker's cache.  The setup prefetch is a
            # background download, so wait for it here (foreground, same
            # revision, idempotent) before joining the Ray; a failed fetch
            # ends this node non-zero instead of a later engine start error.
            f"{worker_model_fetch}"
            "  for _ in $(seq 1 150); do "
            'ray start --address="$MASTER_ADDR:6379" --temp-dir="$MILES_RAY_DIR"'
            f"{_worker_node_ip_flag(spec.cloud)}"
            f"{rl_island_ray_gpus(args, spec)} && break; "
            "sleep 2; done\n"
            "  ray status --address=\"$MASTER_ADDR:6379\" >/dev/null 2>&1 "
            "|| { echo '[yeto-island] worker could not join the Ray head' >&2; exit 1; }\n"
            "  while ray status --address=\"$MASTER_ADDR:6379\" "
            ">/dev/null 2>&1; do sleep 5; done\n"
            "fi"
    )
    in_vm_docker = spec.cloud in IN_VM_DOCKER_CLOUDS
    if in_vm_docker:
        # C block: sky's docker runtime is unavailable here; the same setup
        # and run execute inside `docker run` of the same pinned image.
        setup_script, run_script = (
            in_vm_docker_setup(args.rl_image, login=bool(registry_login)),
            in_vm_docker_run(args.rl_image, setup_script, run_script,
                             [*envs.keys(), *island_hmac_secret(args).keys()], learner_id),
        )
    island_secrets = {**dict(registry_login or {}), **island_hmac_secret(args)}
    task = sky.Task(
        name=f"yeto-rl-island-{learner_id}",
        setup=setup_script,
        **({"secrets": island_secrets} if island_secrets else {}),
        run=run_script,
        envs=envs,
        num_nodes=spec.num_nodes,
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts or None,
    )
    resources = {
        "infra": f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud,
        "accelerators": spec.accelerators,
        "cpus": rl_island_cpus(args, spec.cloud),
        "instance_type": args.learner_instance_type,
        "use_spot": args.spot,
        "disk_size": args.disk_size,
    }
    if not in_vm_docker:  # Verda: the VM image stays the cloud default; the container is ours
        resources["image_id"] = nebius_baked_image_id(args.rl_image, spec.cloud, spec.region)
    if spec.num_nodes > 1 and getattr(args, "rl_island_network_tier", "auto") != "none":
        tier = multinode_network_tier(spec.cloud, spec.gpu, spec.gpus_per_node)
        if tier:
            resources["network_tier"] = tier
    if spec.cloud != "modal":  # Modal islands take only run + envs (see build_modal_island_config)
        task.set_resources(sky.Resources(**resources))
    storage_mounts = {}
    if args.spot:
        checkpoint_mount = _rl_checkpoint_mount(args.rl_completed_groups_path)
        storage_mounts[checkpoint_mount] = sky.Storage(
            name=_rl_checkpoint_storage_name(args.cluster_prefix, learner_id),
            persistent=False,
            mode=sky.StorageMode.MOUNT,
            sync_on_reconstruction=True,
        )
    if getattr(args, "rl_checkpoint_store", None) and not getattr(args, "rl_elastic", False):
        store = rl_checkpoint_store_plan(args)  # resume store: same bucket mount on sky
        if store is not None and store[1] is not None:
            storage_mounts[ELASTIC_CHECKPOINT_STORE_MOUNT] = sky.Storage(
                source=store[1], mode=sky.StorageMode.MOUNT, persistent=True)
    if getattr(args, "rl_elastic", False):
        # rl-multinode-island Q4 (C5): the consistent checkpoint (journal + cuts) must
        # outlive node0 for a rebuild on other machines; a bucket URI is mounted on
        # every island node, a shared path is left to the deployment (fail open with
        # a warning: single-node and node0-local behavior is unchanged).
        store = rl_checkpoint_store_plan(args)
        if store is not None and store[1] is not None:
            storage_mounts[ELASTIC_CHECKPOINT_STORE_MOUNT] = sky.Storage(
                source=store[1], mode=sky.StorageMode.MOUNT, persistent=True)
        elif store is None and spec.num_nodes > 1:
            print(NODE0_LOCAL_CHECKPOINT_WARNING, file=sys.stderr)
    if storage_mounts:
        task.set_storage_mounts(storage_mounts)
    return task


def _is_protenix_request(args) -> bool:
    model = getattr(args, "model", "")
    adapter = getattr(args, "diffusion_adapter", "") or ""
    return model in {"protenix", "protenix-v2"} or "protenix" in adapter


def _is_hunyuan3d_request(args) -> bool:
    model = getattr(args, "model", "")
    adapter = getattr(args, "diffusion_adapter", "") or ""
    return model in {"hunyuan3d-21"} or "hunyuan3d" in adapter


def _is_alphafold3_request(args) -> bool:
    model = getattr(args, "model", "")
    adapter = getattr(args, "diffusion_adapter", "") or ""
    return model == "alphafold3" or "alphafold3" in adapter


def make_learner_task(args, spec: ClusterSpec, learner_id: int, num_learners: int, syncer_addr: str):
    import sky

    from .adapter_lifecycle import (
        learner_parent_arg,
        learner_parent_mounts,
        selected_parent,
    )
    from .datasource import learner_data_arg, learner_file_mounts
    from .models import resolve_model_kind

    model_kind = resolve_model_kind(args.model, getattr(args, "model_kind", "auto"))
    if model_kind != "diffusion" and getattr(args, "diffusion_adapter", None):
        raise ValueError("--diffusion-adapter applies only to diffusion models")
    loss_function = args.loss_function
    if model_kind == "diffusion" and loss_function == "cross_entropy":
        loss_function = "flow_matching"

    backend = getattr(args, "island_backend", "torch")
    attention_backend = getattr(args, "attention_backend", "auto")
    kernel_backend = getattr(args, "kernel_backend", "native")
    data_format = getattr(args, "data_format", "auto")
    base_quantization = getattr(args, "base_quantization", "none")
    parent_mode, parent_source = selected_parent(args)
    if parent_source is not None:
        if model_kind != "causal-lm":
            raise ValueError(
                "--resume-from/--branch-from apply only to causal-LM models"
            )
        if backend != "torch":
            raise ValueError(
                "--resume-from/--branch-from require --island-backend torch"
            )
        if args.tuning != "lora":
            raise ValueError(
                "--resume-from/--branch-from require --tuning lora"
            )
        if not getattr(args, "adapter_sha256", None):
            raise ValueError(
                "--resume-from/--branch-from must be attested before learner launch"
            )
    if model_kind != "causal-lm" and data_format != "auto":
        raise ValueError("--data-format applies only to causal-LM models")
    if model_kind != "causal-lm" and (
        attention_backend != "auto" or kernel_backend != "native"
    ):
        raise ValueError(
            "--attention-backend and --kernel-backend apply only to causal-LM models"
        )
    if model_kind == "causal-lm" and backend != "torch":
        if attention_backend != "auto" or kernel_backend != "native":
            raise ValueError(
                "--attention-backend and --kernel-backend are supported only "
                "by the torch causal-LM island backend"
            )
    if base_quantization != "none":
        if model_kind != "causal-lm":
            raise ValueError("--base-quantization applies only to causal-LM models")
        if backend != "torch":
            raise ValueError(
                "--base-quantization nf4 is supported only by the torch "
                "causal-LM island backend"
            )
        if args.tuning != "lora":
            raise ValueError("--base-quantization nf4 requires --tuning lora")
        if args.shard != "ddp":
            raise ValueError("--base-quantization nf4 requires --shard ddp")
        if kernel_backend != "native":
            raise ValueError(
                "--base-quantization nf4 requires --kernel-backend native"
            )
    if kernel_backend == "liger" and loss_function != "cross_entropy":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE supports only the built-in "
            "cross_entropy loss"
        )
    if kernel_backend == "liger" and args.tuning != "lora":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE is production-approved only "
            "for --tuning lora"
        )
    if kernel_backend == "liger" and args.shard != "ddp":
        raise ValueError(
            "--kernel-backend liger fused-linear-CE is production-approved only "
            "for --shard ddp until FSDP has separate CUDA parity evidence"
        )

    # Flags shared by all learners. The DiLoCo sync, LoRA, and data source
    # shape are identical; the per-task forward/loss loop differs.
    common_flags = (
        f" --model {shlex.quote(args.model)}"
        f" --data {shlex.quote(learner_data_arg(args.data))}"
        f" --syncer $SYNCER_ADDR"
        f" --learner-id $LEARNER_ID"
        f" --num-learners {num_learners}"
        f" --loss-function {shlex.quote(loss_function)}"
        f" --tuning {args.tuning}"
        f" --lora-r {args.lora_r}"
        f" --lora-alpha {getattr(args, 'lora_alpha', 32)}"
        f" --lora-targets {getattr(args, 'lora_targets', 'auto')}"
        f" --micro-batch-size {args.micro_batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --inner-lr {args.inner_lr}"
        f" --fragments {args.fragments}"
        f" --fragment-pattern {args.fragment_pattern}"
        f" --matrix-merge {getattr(args, 'matrix_merge', 'rda')}"
        f" --merge-alpha {args.merge_alpha}"
        f" --wire-dtype {args.wire_dtype}"
        f" --wan-streams {args.wan_streams}"
        f" --output-dir ~/yeto-output"
    )
    parent_arg = learner_parent_arg(args)
    if parent_mode is not None and parent_arg is not None:
        common_flags += f" --{parent_mode}-from {shlex.quote(parent_arg)}"
        common_flags += (
            f" --adapter-sha256 {shlex.quote(args.adapter_sha256)}"
        )
    if getattr(args, "model_revision", None):
        common_flags += f" --model-revision {shlex.quote(args.model_revision)}"
    if getattr(args, "data_revision", None):
        common_flags += f" --data-revision {shlex.quote(args.data_revision)}"
    if getattr(args, "trust_remote_code", False):
        common_flags += " --trust-remote-code"
    if getattr(args, "allow_unsafe_pickled_loss", False):
        common_flags += " --allow-unsafe-pickled-loss"
    if getattr(args, "loss_sha256", None):
        common_flags += f" --loss-sha256 {shlex.quote(args.loss_sha256)}"
    if getattr(args, "source_sha256", None):
        common_flags += f" --source-sha256 {shlex.quote(args.source_sha256)}"
    if getattr(args, "wandb", False):
        common_flags += " --wandb"
        common_flags += f" --wandb-project {shlex.quote(args.wandb_project)}"
        common_flags += f" --wandb-mode {args.wandb_mode}"
        if getattr(args, "wandb_entity", None):
            common_flags += f" --wandb-entity {shlex.quote(args.wandb_entity)}"
    for name in (
        "model_requested_identifier",
        "model_requested_revision",
        "data_requested_identifier",
        "data_requested_revision",
    ):
        value = getattr(args, name, None)
        if value is not None:
            common_flags += (
                f" --{name.replace('_', '-')} {shlex.quote(str(value))}"
            )
    learner_flags = common_flags
    if model_kind == "causal-lm":
        learner_flags += (
            f" --train-on {args.train_on}"
            f" --assistant-mask-mode {getattr(args, 'assistant_mask_mode', 'native')}"
            f" --data-format {getattr(args, 'data_format', 'auto')}"
            f" --seed {getattr(args, 'seed', 0)}"
            f" --seq-len {args.seq_len}"
            f" --tokenize {args.tokenize}"
            f" --stream-workers {args.stream_workers}"
        )
        if backend == "torch":
            learner_flags += (
                f" --base-quantization {base_quantization}"
                f" --attention-backend {attention_backend}"
                f" --kernel-backend {kernel_backend}"
            )
    else:
        if getattr(args, "island_backend", "torch") != "torch":
            raise ValueError("diffusion model-kind uses the torch island backend, not megatron")
        learner_flags += (
            f" --shard {args.shard}"
            f" --image-column {shlex.quote(args.image_column)}"
            f" --video-column {shlex.quote(args.video_column)}"
            f" --prompt-column {shlex.quote(args.prompt_column)}"
            f" --latent-column {shlex.quote(args.latent_column)}"
            f" --text-embeds-column {shlex.quote(args.text_embeds_column)}"
            f" --text-attention-mask-column {shlex.quote(args.text_attention_mask_column)}"
            f" --pooled-text-embeds-column {shlex.quote(args.pooled_text_embeds_column)}"
            f" --resize-mode {shlex.quote(getattr(args, 'resize_mode', 'stretch'))}"
            f" --stream-workers {args.stream_workers}"
        )
        diffusion_adapter = getattr(args, "diffusion_adapter", None)
        if _is_protenix_request(args) and not diffusion_adapter:
            diffusion_adapter = PROTENIX_DIFFUSION_ADAPTER
        if _is_hunyuan3d_request(args) and not diffusion_adapter:
            diffusion_adapter = HUNYUAN3D_DIFFUSION_ADAPTER
        if _is_alphafold3_request(args) and not diffusion_adapter:
            diffusion_adapter = ALPHAFOLD3_DIFFUSION_ADAPTER
        if diffusion_adapter:
            learner_flags += f" --diffusion-adapter {shlex.quote(diffusion_adapter)}"
            if getattr(args, "diffusion_adapter_sha256", None):
                learner_flags += (
                    " --diffusion-adapter-sha256 "
                    f"{shlex.quote(args.diffusion_adapter_sha256)}"
                )
        if getattr(args, "diffusion_seed", None) is not None:
            learner_flags += f" --seed {args.diffusion_seed}"
        if getattr(args, "cache_latents", False):
            learner_flags += " --cache-latents"
        if getattr(args, "cache_text_embeds", False):
            learner_flags += " --cache-text-embeds"
        if getattr(args, "bucket_by_shape", False):
            learner_flags += " --bucket-by-shape"
        if getattr(args, "diffusion_loss_weighting", "none") != "none":
            learner_flags += f" --diffusion-loss-weighting {args.diffusion_loss_weighting}"
            learner_flags += f" --diffusion-min-snr-gamma {args.diffusion_min_snr_gamma}"
        if args.height:
            learner_flags += f" --height {args.height}"
        if args.width:
            learner_flags += f" --width {args.width}"
        if args.num_frames:
            learner_flags += f" --num-frames {args.num_frames}"
        if getattr(args, "fps", None):
            learner_flags += f" --fps {args.fps}"
    if args.max_rows:
        learner_flags += f" --max-rows {args.max_rows}"

    if model_kind == "diffusion":
        entrypoint = "yeto.diffusion.learner"
        setup_steps = [
            WAN_TUNING,
            NVME_SETUP,
            NVME_ENV,
            HF_TOKEN_ENV,
            TORCH_SETUP,
            "pip install -q -r requirements.txt",
            "pip install -q 'diffusers>=0.35' safetensors pillow 'imageio[ffmpeg]' 'bitsandbytes>=0.46.1'",
        ]
        if _is_protenix_request(args):
            setup_steps.append(PROTENIX_SETUP)
        if _is_hunyuan3d_request(args):
            setup_steps.append(HUNYUAN3D_SETUP)
    elif backend == "megatron":
        gpus = spec.num_nodes * spec.gpus_per_node
        tp = max(1, getattr(args, "tensor_parallel", 1))
        pp = max(1, getattr(args, "pipeline_parallel", 1))
        ep = getattr(args, "expert_parallel", None) or max(1, gpus // (tp * pp))
        learner_flags += (
            f" --island-backend megatron"
            f" --expert-parallel {ep}"
            f" --tensor-parallel {tp}"
            f" --pipeline-parallel {pp}"
        )
        entrypoint = "yeto.megatron.learner"
        # Inside the NGC container the whole training stack (torch, TE,
        # megatron-core, bridge) is already present, so skip TORCH_SETUP and
        # MEGATRON_SETUP. NVME_SETUP is a host RAID operation that can't run in
        # a container, so it's skipped too (HF cache lands on the container's
        # disk — slower download, but correct; wiring the host instance-store
        # through to the container is a follow-up). Only yeto's pure-python
        # deps that the container may lack are added, --no-deps so they never
        # perturb the container's pinned torch/TE/transformers.
        setup_steps = [
            WAN_TUNING,
            HF_TOKEN_ENV,
            "pip install -q --no-deps datasets peft hf_transfer cloudpickle sentencepiece",
        ]
    else:
        learner_flags += f" --shard {args.shard}"
        entrypoint = "yeto.learner"
        setup_steps = [WAN_TUNING, NVME_SETUP, NVME_ENV, HF_TOKEN_ENV, TORCH_SETUP,
                       "pip install -q -r requirements.txt"]
        setup_steps.extend(causal_kernel_setup_steps(args))

    if getattr(args, "wandb", False):
        # Telemetry is never worth losing an island over: a failed install
        # is reported and the learner trains on (yeto.wandb_logger falls
        # back to a no-op sink when the import fails).
        setup_steps.append(
            "pip install -q wandb || echo '[yeto-setup] wandb install failed; "
            "the island will train without telemetry' >&2"
        )

    run = (
        f"{NVME_ENV}\n"
        f"{HF_TOKEN_ENV}\n"
        'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
        "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
        "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
        "--master_addr=$MASTER_ADDR --master_port=29500 "
        f"-m {entrypoint}{learner_flags}"
    )
    envs = {
        "SYNCER_ADDR": syncer_addr,
        "LEARNER_ID": str(learner_id),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    }
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    if getattr(args, "wandb", False):
        # Every island joins the fleet's W&B group under the run's name, so
        # the islands and the syncer's tape run land on one comparison view.
        envs["YETO_RUN_GROUP"] = args.cluster_prefix
        if os.environ.get("WANDB_API_KEY"):
            envs["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    if spec.num_nodes > 1:
        # Surface NCCL's chosen transport in the job logs so an EFA-less
        # fallback to TCP sockets is visible, not silent.
        envs["NCCL_DEBUG"] = "INFO"
    if _is_hunyuan3d_request(args):
        envs["YETO_HUNYUAN3D_ROOT"] = HUNYUAN3D_ROOT
    # Non-HF --data sources (local paths, s3://, gs://, ...) ride sky's
    # file_mounts onto every learner; see yeto/datasource.py.
    file_mounts = dict(learner_file_mounts(args.data))
    file_mounts.update(learner_parent_mounts(args))
    file_mounts = file_mounts or None
    if args.loss_function.startswith("pickle:"):
        # The pickled loss is gitignored, so the workdir sync skips it;
        # mount it into the workdir explicitly.
        file_mounts = file_mounts or {}
        loss_path = pickled_loss_path(args.loss_function)
        file_mounts[f"~/sky_workdir/{loss_path.name}"] = str(loss_path)
    # Ride the launching machine's HF token onto every learner: anonymous
    # Hub quota is half the authenticated one and shared per-IP, and a
    # gated/private --model needs the token outright. HF_TOKEN_ENV then
    # copies it wherever NVME_ENV points HF_HOME.
    local_token = os.path.expanduser(HF_TOKEN_PATH)
    if os.path.isfile(local_token):
        file_mounts = file_mounts or {}
        file_mounts[HF_TOKEN_PATH] = local_token
    # Kick the weight download off in the background at the END of setup:
    # it overlaps sky's remaining bookkeeping and races ahead of the run
    # command, which then finds a warm (or warming — hf resumes) cache.
    # hf_transfer multi-streams the download; NVMe absorbs it at GB/s.
    from .models import resolve
    from .provenance import is_local_reference

    repo = resolve(args.model)
    if _is_protenix_request(args):
        prefetch = "true  # Protenix uses native model names/checkpoints, not HF prefetch"
    elif _is_alphafold3_request(args):
        prefetch = "true  # AlphaFold3 requires local authorized parameters, not HF prefetch"
    elif is_local_reference(repo):
        prefetch = ": # local model; no Hub prefetch"
    else:
        revision_flag = (
            f" --revision {shlex.quote(args.model_revision)}"
            if getattr(args, "model_revision", None)
            else ""
        )
        prefetch = (
            f"(nohup huggingface-cli download {shlex.quote(repo)}{revision_flag} "
            ">/tmp/hf-prefetch.log 2>&1 &) || true"
        )
    run = (
        f"{NVME_ENV}\n"
        f"{HF_TOKEN_ENV}\n"
        'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
        "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
        "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
        "--master_addr=$MASTER_ADDR --master_port=29500 "
        f"-m {entrypoint}{learner_flags}"
    )
    task = sky.Task(
        name=f"yeto-learner-{learner_id}",
        setup="\n".join(setup_steps + [prefetch]),
        run=run,
        envs=envs,
        num_nodes=spec.num_nodes,
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts,
    )
    infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
    resources_kwargs = {}
    image = learner_image_for(args, spec, learner_id)
    if image is not None:
        resources_kwargs["image_id"] = image
    if spec.num_nodes > 1:
        # Multi-node learner: inner DDP all-reduce crosses the node fabric,
        # so request the cloud's RDMA-class interconnect (EFA on AWS,
        # GPUDirect on GCP). Single-node clusters stay on NVLink and don't
        # need it. On AWS this also swaps in the EFA-ready DLAMI; SkyPilot
        # installs no EFA software itself, and if the pinned AMI is missing
        # NCCL silently falls back to TCP — NCCL_DEBUG below makes the
        # chosen transport visible in the job logs (look for
        # "NET/OFI Selected Provider is efa").
        tier = multinode_network_tier(spec.cloud, spec.gpu, spec.gpus_per_node)
        if tier:
            resources_kwargs["network_tier"] = tier
    if spec.cloud == "modal":
        # Not a sky cloud: build_modal_island_config reads only this task's
        # run script and envs; the Modal runner sizes the container itself.
        return task
    task.set_resources(
        sky.Resources(
            infra=infra,
            accelerators=spec.accelerators,
            cpus=args.learner_cpus,
            instance_type=args.learner_instance_type,
            use_spot=args.spot,
            disk_size=args.disk_size,
            **resources_kwargs,
        )
    )
    return task


def _diffusion_sample_adapter_mount(adapter_dir: str) -> tuple[str, dict[str, str]]:
    from .datasource import _is_cloud_url

    if _is_cloud_url(adapter_dir):
        return DIFFUSION_SAMPLE_ADAPTER_DIR, {DIFFUSION_SAMPLE_ADAPTER_DIR: adapter_dir}
    path = os.path.expanduser(adapter_dir)
    if os.path.exists(path):
        return DIFFUSION_SAMPLE_ADAPTER_DIR, {DIFFUSION_SAMPLE_ADAPTER_DIR: path}
    raise ValueError("--adapter-dir must be an existing local path or a cloud URI")


def _add_flag(cmd: str, name: str, value) -> str:
    if value is None:
        return cmd
    return f"{cmd} --{name} {shlex.quote(str(value))}"


def make_diffusion_sample_task(args, spec: ClusterSpec):
    import sky

    from .datasource import learner_data_arg, learner_file_mounts

    adapter_arg, file_mounts = _diffusion_sample_adapter_mount(args.adapter_dir)
    sample_cmd = (
        "python3 -m yeto.diffusion.sample"
        f" --adapter-dir {shlex.quote(adapter_arg)}"
        f" --dtype {shlex.quote(args.dtype)}"
        f" --num-inference-steps {int(args.num_inference_steps)}"
        f" --fps {int(args.fps)}"
    )
    if args.data:
        sample_cmd += (
            f" --data {shlex.quote(learner_data_arg(args.data))}"
            f" --output-dir {shlex.quote(DIFFUSION_SAMPLE_OUTPUT_DIR)}"
            f" --prompt-column {shlex.quote(args.prompt_column)}"
        )
        if args.seed_column:
            sample_cmd += f" --seed-column {shlex.quote(args.seed_column)}"
        if args.max_rows is not None:
            sample_cmd += f" --max-rows {int(args.max_rows)}"
        file_mounts.update(learner_file_mounts(args.data))
    else:
        sample_cmd += (
            f" --prompt {shlex.quote(args.prompt)}"
            f" --output {shlex.quote(DIFFUSION_SAMPLE_OUTPUT_DIR + '/sample.png')}"
        )
    for name in (
        "model",
        "model_revision",
        "source_sha256",
        "diffusion_adapter",
        "diffusion_adapter_sha256",
        "model_requested_identifier",
        "model_requested_revision",
        "data_requested_identifier",
        "data_requested_revision",
        "guidance_scale",
        "height",
        "width",
        "num_frames",
        "seed",
    ):
        sample_cmd = _add_flag(sample_cmd, name.replace("_", "-"), getattr(args, name, None))
    if getattr(args, "data_revision", None):
        sample_cmd = _add_flag(sample_cmd, "data-revision", args.data_revision)
    if getattr(args, "trust_remote_code", False):
        sample_cmd += " --trust-remote-code"
    if getattr(args, "allow_unattested_legacy_adapter", False):
        sample_cmd += " --allow-unattested-legacy-adapter"

    local_token = os.path.expanduser(HF_TOKEN_PATH)
    if os.path.isfile(local_token):
        file_mounts[HF_TOKEN_PATH] = local_token
    setup_steps = [
        WAN_TUNING,
        NVME_SETUP,
        NVME_ENV,
        HF_TOKEN_ENV,
        TORCH_SETUP,
        "pip install -q -r requirements.txt",
        "pip install -q 'diffusers>=0.35' safetensors pillow 'imageio[ffmpeg]' 'bitsandbytes>=0.46.1'",
    ]
    run = f"{NVME_ENV}\n{HF_TOKEN_ENV}\nmkdir -p {DIFFUSION_SAMPLE_OUTPUT_DIR}\n{sample_cmd}"
    envs = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    if os.environ.get("HF_TOKEN"):
        envs["HF_TOKEN"] = os.environ["HF_TOKEN"]
    task = sky.Task(
        name="yeto-diffusion-sample",
        setup="\n".join(setup_steps),
        run=run,
        envs=envs,
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts or None,
    )
    infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
    resources_kwargs = {}
    image = learner_image_for(args, spec)
    if image is not None:
        resources_kwargs["image_id"] = image
    task.set_resources(
        sky.Resources(
            infra=infra,
            accelerators=spec.accelerators,
            cpus=getattr(args, "learner_cpus", None),
            instance_type=getattr(args, "learner_instance_type", None),
            use_spot=args.spot,
            disk_size=args.disk_size,
            **resources_kwargs,
        )
    )
    return task


def _status_terminal(status) -> bool:
    if status is None:
        return False
    is_terminal = getattr(status, "is_terminal", None)
    if callable(is_terminal):
        return bool(is_terminal())
    text = str(status)
    return any(s in text for s in ("SUCCEEDED", "FAILED", "CANCELLED", "STOPPED"))


def _status_succeeded(status) -> bool:
    return status is not None and "SUCCEEDED" in str(status)


def _wait_for_terminal_job(cluster: str, job_id: int, poll_interval: int = 30):
    ops = SkySDKOps()
    while True:
        status = ops.job_status(cluster, job_id)
        if _status_terminal(status):
            return status
        ops.sleep(max(1, poll_interval))


def run_diffusion_sample(args) -> int:
    from .datasource import kind as data_kind
    from .models import resolve
    from .provenance import (
        file_sha256,
        python_spec_path,
        python_spec_sha256,
        resolve_reference,
        verify_source_tree_sha256,
    )

    args.source_sha256 = verify_source_tree_sha256(
        getattr(args, "source_sha256", None)
    )
    if getattr(args, "diffusion_adapter", None):
        expected_adapter_sha256 = getattr(
            args, "diffusion_adapter_sha256", None
        )
        adapter_path = python_spec_path(args.diffusion_adapter, base_dir=REPO_ROOT)
        try:
            relative_adapter_path = adapter_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"diffusion adapter source {adapter_path} is outside the synced "
                "Yeto workdir; copy it into the repository before sampling"
            ) from exc
        target, separator, factory_name = args.diffusion_adapter.partition(":")
        if target.endswith(".py") or os.path.sep in target:
            args.diffusion_adapter = (
                f"{relative_adapter_path.as_posix()}{separator}{factory_name}"
            )
            actual_adapter_sha256 = file_sha256(adapter_path)
        else:
            actual_adapter_sha256 = python_spec_sha256(
                args.diffusion_adapter
            )
        if (
            expected_adapter_sha256 is not None
            and actual_adapter_sha256 != expected_adapter_sha256.lower()
        ):
            raise ValueError(
                "diffusion adapter SHA256 does not match the expected sampler "
                "attestation"
            )
        args.diffusion_adapter_sha256 = actual_adapter_sha256

    if getattr(args, "model", None):
        model_record = resolve_reference(
            resolve(args.model),
            getattr(args, "model_revision", None),
            repo_type="model",
            original_identifier=args.model,
        )
        args.model_revision = model_record["resolved_revision"]
        args.model_requested_identifier = model_record["requested_identifier"]
        args.model_requested_revision = model_record["requested_revision"]
    if getattr(args, "data", None):
        kind = data_kind(args.data)
        if kind == "hf":
            data_record = resolve_reference(
                args.data,
                getattr(args, "data_revision", None),
                repo_type="dataset",
            )
            args.data_revision = data_record["resolved_revision"]
            args.data_requested_identifier = data_record["requested_identifier"]
            args.data_requested_revision = data_record["requested_revision"]
        elif getattr(args, "data_revision", None) is not None:
            raise ValueError("--data-revision applies only to a Hugging Face prompt dataset")
        else:
            args.data_requested_identifier = args.data
            args.data_requested_revision = None
    specs = parse_gpu_spec(args.gpu)
    if len(specs) != 1:
        raise ValueError("diffusion sampling expects exactly one --gpu cluster")
    spec = specs[0]
    if spec.num_nodes != 1 or spec.total_gpus != 1:
        raise ValueError("diffusion sampling currently expects one single-GPU node")

    import sky

    cluster = f"{args.cluster_prefix}-sample"
    task = make_diffusion_sample_task(args, spec)
    output = getattr(args, "output", None)
    local_dest = (
        os.path.expanduser(output)
        if output and delivery.kind(output) == "local"
        else os.path.expanduser(DIFFUSION_SAMPLE_OUTPUT_DIR)
    )
    try:
        print(f"[launcher] launching diffusion sampler on {spec} as {cluster}")
        job_id, _handle = sky.stream_and_get(
            sky.launch(task, cluster_name=cluster, retry_until_up=args.retry_until_up)
        )
        threading.Thread(target=_tail, args=(cluster, job_id, "sample"), daemon=True).start()
        status = _wait_for_terminal_job(
            cluster, job_id, getattr(args, "controller_poll", 30)
        )
        if not _status_succeeded(status):
            print(f"[launcher] diffusion sample job ended as {status}", file=sys.stderr)
            return 1
        os.makedirs(local_dest, exist_ok=True)
        subprocess.run(delivery.fetch_cmd(cluster, local_dest), check=True)
        print(f"[launcher] diffusion samples fetched to {local_dest}")
        if delivery.is_remote(output):
            delivery.deliver(output, local_dest)
            print(f"[launcher] diffusion samples uploaded to {output}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"[launcher] fetching {cluster}:~/yeto-output failed ({e})", file=sys.stderr)
        return 2
    finally:
        if args.keep:
            print(f"[launcher] keeping cluster: {cluster}")
        else:
            print(f"[launcher] tearing down {cluster}")
            terminate_and_verify(sky, cluster)


def sky_cluster_name(name: str) -> str:
    """Every cluster name yeto hands sky is lower case.

    Verda hostnames are lower case, and sky 0.13's Verda status query
    matches the *display* name against them (its query_instances takes
    its arguments one slot off); an upper-case region such as FIN-03 in
    the name then reads as "no instances", sky drops the record, and the
    same-name relaunch that follows deletes the running node. Lower-case
    names break that chain at the first link (design D1)."""
    return name.lower()


def learner_cluster_names(prefix: str, specs: list[ClusterSpec]) -> list[str]:
    """Deterministic learner cluster names for a run: computable from the
    launch args alone, so the CLI can record them before provisioning.
    Modal islands always end in `-modal` (whatever their region hint) so
    every consumer — controller, registry, `yeto down` — can route them
    by name."""
    from .modal_runner import modal_island_name

    return [
        modal_island_name(prefix, m)
        if spec.cloud == "modal"
        else sky_cluster_name(f"{prefix}-l{m}-{spec.region or spec.cloud}")
        for m, spec in enumerate(specs)
    ]


def build_modal_island_config(args, spec: ClusterSpec, learner_id: int, task, syncer_addr: str):
    """Turn the sky task the factory built for this island into the Modal
    island config: same run script, same envs (syncer address swapped for
    the Modal-reachable one), image/volume per training mode."""
    from .modal_runner import (
        ModalIslandConfig,
        image_ref_from_rl_image,
        modal_app_name,
    )

    rl = getattr(args, "training_mode", "sft") == "rl"
    registry_login = registry_login_for(args) if rl else None
    envs = dict(getattr(task, "envs", None) or {})
    envs["SYNCER_ADDR"] = syncer_addr
    if "CRITIC_SYNCER_ADDR" in envs and syncer_addr != "none":
        # The critic syncer shares the actor syncer's host: swap it too, or a
        # Modal island behind --syncer-public-addr dials the private address.
        envs["CRITIC_SYNCER_ADDR"] = critic_syncer_address(syncer_addr)
    if rl and getattr(args, "rl_engine", "ports") != "ports":
        # Legacy Miles' own router launch misses its 30 s deadline on Modal's
        # CPUs (see yeto.rl.adapters.miles.island_entry.start_external_sglang_router).  Upstream
        # Miles (ports) launches its router as a Ray worker with a 120 s
        # budget and has no external router mode.
        envs["YETO_RL_EXTERNAL_ROUTER"] = "1"
    for item in getattr(args, "modal_env", None) or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key:
            raise ValueError(f"--modal-env takes KEY=VALUE, got {item!r}")
        envs[key] = value
    if rl:  # 0.20: Modal ships cfg.envs as a modal.Secret
        envs.update(island_hmac_secret(args))
    token_path = os.path.expanduser(HF_TOKEN_PATH)
    if "HF_TOKEN" not in envs and os.path.isfile(token_path):
        with open(token_path, encoding="utf-8") as f:
            envs["HF_TOKEN"] = f.read().strip()
    store_volume = modal_checkpoint_store(args) if rl else None
    if store_volume is not None:
        from .cloud.modal_ckpt_store import MODAL_VOLUME_ENV, STORE_IMPL
        from .rl.engine.resume import STORE_IMPL_ENV

        envs[MODAL_VOLUME_ENV] = store_volume[0]  # the learner commits this Volume
        envs[STORE_IMPL_ENV] = STORE_IMPL  # the island builds the Modal store from yeto.cloud
    volume_name = volume_mount = None
    if rl and getattr(args, "spot", False):
        volume_name = _rl_checkpoint_storage_name(args.cluster_prefix, learner_id)
        volume_mount = _rl_checkpoint_mount(args.rl_completed_groups_path).replace("~", "/root", 1)
    # Codex run bundle: the sky task mounts it at CODEX_CONTAINER_DIR; Modal
    # mounts the same directory through the image (add_local_dir).
    all_mounts = dict(getattr(task, "file_mounts", None) or {})
    codex_dir = all_mounts.get(CODEX_CONTAINER_DIR)
    # Every other sky file_mount (local prompt file, initial adapter, HF token
    # file) is mounted into the container the same way; `~` is /root in Modal.
    extra_mounts = {
        ("/root" + target[1:] if target.startswith("~") else target): str(source)
        for target, source in all_mounts.items()
        if target != CODEX_CONTAINER_DIR and not str(source).startswith(("s3://", "gs://", "r2://"))
    }
    requirements: tuple[str, ...] = ()
    if not rl:
        req_file = REPO_ROOT / "requirements.txt"
        requirements = ("torch",) + tuple(
            line.strip()
            for line in req_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
    return ModalIslandConfig(
        app_name=modal_app_name(args.cluster_prefix),
        learner_id=learner_id,
        training_mode="rl" if rl else "sft",
        gpu=spec.gpu,
        gpus_per_node=spec.gpus_per_node,
        num_nodes=spec.num_nodes,
        run_script=str(getattr(task, "run", "") or ""),
        envs={k: str(v) for k, v in envs.items() if v is not None},
        region=spec.region,
        gpu_exact=bool(getattr(args, "modal_gpu_exact", False)),
        registry_login=bool(registry_login),
        registry_creds=registry_login,
        image_ref=image_ref_from_rl_image(args.rl_image) if rl else None,
        setup_script=str(getattr(task, "setup", "") or "") if rl else None,
        pip_requirements=requirements,
        volume_name=volume_name,
        volume_mount=volume_mount,
        codex_dir=codex_dir,
        codex_mount=CODEX_CONTAINER_DIR if codex_dir else None,
        extra_mounts=extra_mounts,
        checkpoint_store_volume_name=None if store_volume is None else store_volume[0],
        checkpoint_store_mount=None if store_volume is None else MODAL_CHECKPOINT_STORE_MOUNT,
        tape_volume_name=getattr(args, "modal_tape_volume", None) or None,
        model_volume_name=(getattr(args, "modal_model_volume", None) or None) if rl else None,
        model_volume_mount=MODEL_STORE_MOUNT if rl and getattr(args, "modal_model_volume", None) else None,
        cpu_override=getattr(args, "modal_cpu", None),
        memory_gib_override=getattr(args, "modal_memory_gib", None),
        tape_subdir=(modal_tape_subdir(args.cluster_prefix, learner_id)
                     if getattr(args, "modal_tape_volume", None) else None),
        workdir=str(REPO_ROOT),
        # Opt-in overrides (default: Modal runner defaults, 10 retries / 24 h).
        # Acceptance runs pass --modal-retries 0 so a learner exit is final
        # (no re-run billing) and --modal-timeout-s as the Modal-side hard stop.
        # --no-island-relaunch ("a failed island is never relaunched") forces the
        # platform retries to 0 too: Modal's own retry re-runs a failed island
        # container by itself (up to 10 times) and would silently undo the flag --
        # a second paid container, replaying from the last checkpoint, that the
        # launcher neither tracks nor tears down. Wins over an explicit --modal-retries N.
        **({"retries": 0} if getattr(args, "no_island_relaunch", False)
           else {"retries": int(args.modal_retries)}
           if getattr(args, "modal_retries", None) is not None else {}),
        **({"timeout_s": int(args.modal_timeout_s)}
           if getattr(args, "modal_timeout_s", None) is not None else {}),
    )


def modal_tape_subdir(cluster_prefix: str, learner_id: int) -> str:
    """Per-run, per-island directory of the Modal tape Volume."""
    from .modal_runner import modal_app_name

    return f"{modal_app_name(cluster_prefix)}/l{learner_id}"


def _modal_tape_run_dir(args):
    # A module-level import: `run()` binds `runs` only part-way through, and
    # its finally block (where the tape is pulled) also runs on early failures.
    from . import runs as _runs

    return _runs.run_dir(args.cluster_prefix)


def pull_modal_tapes(modal_ops, modal_cfgs: dict, run_dir) -> dict[str, str]:
    """Copy each Modal island's tape Volume subdir into
    ``<run_dir>/modal-tape/<island>/``; returns {island: "ok" | error}.
    Never raises: a missing tape is reported, not fatal to teardown."""
    out: dict[str, str] = {}
    for name, cfg in modal_cfgs.items():
        vol, sub = getattr(cfg, "tape_volume_name", None), getattr(cfg, "tape_subdir", None)
        if not vol or not sub:
            continue
        dest = Path(run_dir) / "modal-tape" / name
        try:
            modal_ops.pull_tape(vol, sub, str(dest))
            files = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
            out[name] = "ok" if files else "empty"
            print(f"[launcher] Modal tape of {name}: {vol}:{sub} -> {dest} ({len(files)} file(s))")
        except Exception as e:  # noqa: BLE001
            out[name] = f"error: {e}"
            print(f"[launcher] WARN: could not pull the Modal tape of {name} from "
                  f"{vol}:{sub}: {e}", file=sys.stderr)
    return out


def require_modal_for_gpu_exact(args, specs: list[ClusterSpec]) -> None:
    """--modal-gpu-exact only means something on Modal islands; refuse it
    rather than silently launching unpinned learners elsewhere."""
    if getattr(args, "modal_gpu_exact", False):
        other = sorted({s.cloud for s in specs if s.cloud != "modal"})
        if other:
            raise ValueError(
                f"--modal-gpu-exact applies only to Modal islands; --gpu also has {other}"
            )


def warn_if_model_wont_fit(args, specs: list[ClusterSpec]) -> None:
    weight_gb = MODEL_WEIGHT_GB.get(args.model)
    if weight_gb is None:
        return
    for spec in specs:
        vram = GPU_MEM_GB.get(spec.gpu, 0) * spec.total_gpus
        if vram < weight_gb:
            print(
                f"[launcher] WARNING: {spec} has ~{vram} GB VRAM but {args.model} "
                f"needs ~{weight_gb} GB for frozen bf16 weights alone — expect OOM.",
                file=sys.stderr,
            )


from .rl.event_echo import TapeCollector as EventCollector  # noqa: E402  (no-sync tapes)


def wait_for_tapes(collectors: dict, names, threads, limit: float, *,
                   clock=time.monotonic, sleep=time.sleep, poll: float = 0.2) -> str:
    """Bounded, event-based wait before the tapes are judged complete.

    Returns ``"finalized"`` as soon as every expected island's collector holds its
    ``rl_learner_finalized`` record, ``"streams_ended"`` when every log stream
    ended (nothing more can arrive), else ``"deadline"`` after ``limit`` s.
    """
    deadline = clock() + limit
    while True:
        if set(collectors) >= set(names) and all(c.finalized for c in collectors.values()):
            return "finalized"
        if not any(t.is_alive() for t in threads):
            return "streams_ended"
        if clock() >= deadline:
            return "deadline"
        sleep(poll)


def _tail_modal(modal_ops, call_id: str, prefix: str, collector=None, guard=None) -> int:
    """Stream a Modal island's container logs (the Modal twin of _tail).
    `guard` (modal_runner.ContainerIdGuard) sees every line to catch container changes."""
    while True:
        try:
            for line in modal_ops.stream_logs(call_id):
                # a Modal log entry may hold several lines: prefix each one
                for part in str(line).rstrip("\n").split("\n"):
                    print(f"[{prefix}] {part.rstrip()}", flush=True)
                if collector is not None:
                    collector.feed(line)
                if guard is not None:
                    guard.feed(line)
            return 0
        except Exception as e:  # transient stream drops: reconnect
            print(f"[{prefix}] log stream error: {e}; retrying", flush=True)
            time.sleep(5)


ECHO_TAPE_RECOVER_TIMEOUT_S = 180.0


def _recover_echo_tape(cluster: str, collector, *, run=None, timeout: float = ECHO_TAPE_RECOVER_TIMEOUT_S) -> bool:
    """A no-sync sky island whose echoed tape stayed incomplete: fetch the
    island's own tape files (``~/yeto-output/rl-island-*.jsonl``, the source
    the echo mirrors) over the ssh alias sky wrote and complete the local tape
    from them. The S1 2x1 L40S runs (2026-10-03/04) showed ``sky.tail_logs``
    delivering the setup lines but no run-phase line at all (root cause: the
    heartbeat ``None`` ended ``_tail``'s loop, see there; fixed, kept as the
    fallback), so the stream alone cannot be the only source. Returns whether the tape is finalized;
    any fetch failure leaves it incomplete (fail closed) and is printed."""
    import tempfile

    run = run or subprocess.run
    with tempfile.TemporaryDirectory(prefix="yeto-tape-") as tmp:
        cmd = ["rsync", "-az", f"{cluster}:yeto-output/rl-island-*.jsonl", f"{tmp}/"]
        try:
            run(cmd, check=True, timeout=timeout, capture_output=True)
        except Exception as e:  # noqa: BLE001 - subprocess/timeout: report, stay incomplete
            print(f"[launcher] {cluster}: tape recovery fetch failed ({e})", file=sys.stderr)
            return False
        files = sorted(Path(tmp).glob("rl-island-*.jsonl"))
        if not files:
            print(f"[launcher] {cluster}: tape recovery found no rl-island-*.jsonl", file=sys.stderr)
            return False
        before = collector.count
        complete = False
        for f in files:
            complete = collector.recover_from_file(f) or complete
        print(f"[launcher] {cluster}: tape recovered from {[f.name for f in files]}: "
              f"+{collector.count - before} record(s), finalized={complete}")
        return complete


def settle_echo_tapes(collectors: dict, names, modal_cfgs, events_dir, *,
                      recover_sky=None, modal_tape_dir=None) -> list[str]:
    """Judge every expected island's echoed tape after the run: close it (fail
    closed: nothing is written afterwards) and, when the log stream did not
    deliver the ``rl_learner_finalized`` record, complete it from the island's
    own tape file before calling it incomplete. Returns the names whose tape is
    still incomplete (``.incomplete`` marker written) -- the launcher's exit 3.

    S14/A19 (FINAL-REPORT-S7 §6.1, -5r1 r7): a SUCCEEDED job ended with exit 3
    because ``sky.tail_logs`` stopped delivering lines part-way (58 of 146
    records, stream silent after a learner restart) and that run predated the
    island-file recovery. The island file is now consulted for every unfinalized
    tape: a streamed-but-cut one, one that never streamed at all (previously
    marked incomplete without looking), and a Modal island's tape pulled from
    its Volume (``modal_tape_dir/<island>/**/rl-island-*.jsonl``). A tape whose
    island file also lacks the finalized record stays incomplete."""
    recover_sky = recover_sky or _recover_echo_tape
    incomplete: list[str] = []
    for name in sorted(set(names) | set(collectors)):
        collector = collectors.get(name)
        streamed = collector is not None
        if collector is None:  # never streamed: nothing received over the log stream
            collector = EventCollector(Path(events_dir) / f"{name}.jsonl", fresh=False)
        complete = collector.close()
        if streamed:
            print(
                f"[launcher] {name}: {collector.count} event(s) -> {collector.path}; "
                f"{collector.discarded} malformed prefixed line(s) discarded"
            )
        else:
            print(f"[launcher] {name}: no log stream line received; tape judged from the "
                  f"island's own tape file", file=sys.stderr)
        if not complete:
            if name in modal_cfgs:
                files = (sorted(Path(modal_tape_dir, name).rglob("rl-island-*.jsonl"))
                         if modal_tape_dir is not None else [])
                if not files:
                    print(f"[launcher] {name}: no pulled Modal tape file to recover from",
                          file=sys.stderr)
                before = collector.count
                for f in files:
                    try:
                        complete = collector.recover_from_file(f) or complete
                    except Exception as e:  # noqa: BLE001 - unreadable file: stay incomplete
                        print(f"[launcher] {name}: tape recovery from {f} failed ({e})",
                              file=sys.stderr)
                if files:
                    print(f"[launcher] {name}: tape recovered from {[f.name for f in files]}: "
                          f"+{collector.count - before} record(s), finalized={complete}")
            else:
                # sky island: the tape file itself is reachable, complete from it
                complete = recover_sky(name, collector)
        if not complete:
            incomplete.append(name)
            print(
                f"[launcher] WARN: {name} event tape has no rl_learner_finalized "
                f"record; marked {collector.incomplete_marker}",
                file=sys.stderr,
            )
    return incomplete


_ISLAND_NAME_RE = re.compile(r"-l(\d+)-")


def syncer_pool_departures(tape_path, offset: int = 0) -> dict[int, str]:
    """Elastic islands the syncer formally removed from the pool and never re-admitted.

    Reads the syncer event tape (from ``offset``: this run's part only) and returns
    ``{island_id: reason}`` for every island whose last pool record is ``pool_leave``
    (``lease_expired``, ``requested``, ...). User ruling S16 §6.6: such an island owes no
    ``rl_learner_finalized`` record; the run summary reports it as a mid-run departure
    instead of the launcher's exit 3. Unreadable tape -> {} (fail closed: exit 3 stays)."""
    last: dict[int, tuple[str, str]] = {}
    try:
        with open(tape_path, encoding="utf-8") as handle:
            handle.seek(offset)
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("kind")
                if kind in ("pool_join", "pool_leave") and isinstance(rec.get("island_id"), int):
                    last[rec["island_id"]] = (kind, str(rec.get("reason") or "unknown"))
    except OSError:
        return {}
    return {i: reason for i, (kind, reason) in last.items() if kind == "pool_leave"}


def excuse_departed_islands(incomplete: list[str], departures: dict[int, str]) -> tuple[list[str], list[dict]]:
    """Split the unfinalized island tapes into those still owing ``rl_learner_finalized``
    (exit 3) and those the syncer removed from the pool (reported, not an error)."""
    still, left = [], []
    for name in incomplete:
        m = _ISLAND_NAME_RE.search(name)
        island = int(m.group(1)) if m else None
        if island is not None and island in departures:
            left.append({"island": island, "name": name, "reason": departures[island]})
        else:
            still.append(name)
    return still, left


def _tail(cluster: str, job_id: int, prefix: str, collector=None) -> int:
    """Stream a sky job's log into the launcher output (and the tape collector).

    Root cause of "sky.tail_logs delivers the setup lines but no run-phase
    line" (S1 2x1 L40S 2026-10-03/04; -5r1 r7 stream silent after the learner
    restart), read from skypilot 0.13.0: the API server's ``log_streamer``
    appends a ``<heartbeat></heartbeat>`` control payload after every 30 s
    without a new log line (sky/server/stream_utils.py, ``_HEARTBEAT_INTERVAL``),
    and the client's ``decode_rich_status`` (sky/utils/rich_utils.py) yields
    ``None`` for EVERY control payload when it runs off the main thread --
    which this tail thread does. The sdk docstring says ``None`` means "the log
    has been completely streamed", so the old loop broke on the first quiet
    30 s (model load, generation) and the thread ended silently while the job
    ran on. The end of the stream is the iterator being exhausted; ``None`` is
    a control message and is skipped."""
    import sky

    while True:
        try:
            it = sky.tail_logs(cluster, job_id, follow=True, preload_content=False)
            for line in it:
                if line is None:  # rich-status/heartbeat control payload, not the end
                    continue
                print(f"[{prefix}] {line.rstrip()}", flush=True)
                if collector is not None:
                    collector.feed(line)
            return 0
        except Exception as e:  # transient stream drops: reconnect
            print(f"[{prefix}] log stream error: {e}; retrying", flush=True)
            time.sleep(5)


class SkySDKOps:
    """Thin adapter over the sky SDK: the only surface FleetController needs.

    Tests inject a fake with the same methods. Any sky call here may raise
    (e.g. the cluster no longer exists); the controller treats exceptions
    from job_status/cluster_up as "cluster gone".
    """

    def job_status(self, cluster: str, job_id: int):
        import sky

        return sky.get(sky.job_status(cluster, [job_id])).get(job_id)

    def job_alive(self, cluster: str, job_id: int) -> bool:
        """Whether `job_id` is still queued, setting up or running on `cluster`.

        `sky.queue` reads the cluster's job table without the UP gate that
        `sky.job_status` applies, so it still answers while the cluster's
        health reads INIT.  Any failure to read the queue counts as "not
        alive": the caller then falls back to its usual failure handling.
        """
        import sky

        try:
            records = sky.get(sky.queue(cluster))
        except Exception:
            return False
        for record in records:
            if isinstance(record, dict):
                rid, status = record.get("job_id"), record.get("status")
            else:
                rid = getattr(record, "job_id", None)
                status = getattr(record, "status", None)
            if rid != job_id:
                continue
            return status is not None and not status.is_terminal()
        return False

    def cluster_up(self, cluster: str) -> bool:
        import sky

        records = sky.get(sky.status(cluster_names=[cluster]))
        if not records:
            return False
        record = records[0]
        status = (
            record.get("status")
            if isinstance(record, dict)
            else getattr(record, "status", None)
        )
        return status == sky.ClusterStatus.UP

    def rl_strict_failure(self, cluster: str, job_id: int) -> str | None:
        """Return a strict RL event from a failed job's existing log."""

        import sky

        try:
            lines = sky.tail_logs(
                cluster, job_id, follow=False, preload_content=False
            )
            for line in lines:
                text = str(line).strip()
                if (
                    "[yeto-rl-strict-failure]" in text
                    or "RL strict failure " in text
                    or "StrictRlInvariantError:" in text
                ):
                    return text
        except Exception:
            return None
        return None

    def relaunch(self, task, cluster: str):
        """Re-provision `cluster` (same spec) and submit `task` as a new job.

        Blocking; returns the new job id, or None if provisioning failed.
        """
        import sky

        try:
            job_id, _handle = sky.get(sky.launch(task, cluster_name=cluster))
            return job_id
        except Exception as e:
            print(f"[launcher] relaunch of {cluster} failed: {e}", file=sys.stderr)
            return None

    def __init__(self, nodes_by_cluster=None):
        # rl-multinode-island D10: the controller's failure-path teardown (job FAILED ->
        # recovery refused -> _down) must verify EVERY node instance, like run()'s final
        # teardown; without the mapping a cluster is treated as single-node.
        self.nodes_by_cluster = dict(nodes_by_cluster or {})

    def down(self, cluster: str) -> None:
        import sky

        nodes = getattr(self, "nodes_by_cluster", None) or {}
        teardown_island(sky, cluster, num_nodes=nodes.get(cluster, 1))

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class LocalSyncer:
    """The syncer as a subprocess of the head node's controller job.

    In head controller mode the syncer binary is file-mounted onto the head
    VM and runs right next to the controller, instead of on a separate
    cluster. Its stdout/stderr go to a log file (appended across restarts)
    that a background thread forwards into this process's stdout with a
    "[syncer]" prefix, so the head job's log stream carries the syncer's
    output. `probe`/`restart` plug into FleetController: a dead subprocess
    is simply restarted, and --resume (already in the command line) makes
    it pick up from its on-disk checkpoint.
    """

    def __init__(
        self,
        args,
        num_learners: int,
        binary: str = "~/yeto-syncer",
        log_file: str = "~/yeto-syncer.log",
    ):
        self.command = syncer_command(args, num_learners, binary=binary)
        self.binary = os.path.expanduser(binary)
        self.log_file = os.path.expanduser(log_file)
        self.event_tape = os.path.expanduser(syncer_event_tape(args))
        self.proc: subprocess.Popen | None = None
        self.tape_forwarder = None
        self.tape_run = None
        # The log file persists across controller jobs on a reused head;
        # forward only what this controller's syncer writes, not history.
        self._log_offset = os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0
        self._event_offset = (
            os.path.getsize(self.event_tape) if os.path.exists(self.event_tape) else 0
        )

    def start(self) -> None:
        if os.path.exists(self.binary):
            os.chmod(self.binary, 0o755)
        log_f = open(self.log_file, "ab")
        try:
            # shell=True so the ~ paths in the command expand. The shell may
            # fork rather than exec the binary, so terminate() on the Popen
            # pid alone can orphan the syncer, which then holds the port
            # across controller restarts; start_new_session gives the whole
            # tree its own process group for stop() to kill.
            self.proc = subprocess.Popen(
                self.command,
                shell=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            log_f.close()  # Popen holds its own duplicate of the fd
        print(f"[launcher] syncer subprocess started (pid {self.proc.pid})", flush=True)

    def probe(self) -> str | _RlStrictFailure | None:
        """None if the subprocess is healthy, else a reason string.

        Exit code 0 means the syncer completed its total steps — terminal
        success, not a failure to recover from (restarting would resume at
        the final step, instantly re-complete, and loop until the learners
        report done)."""
        if self.proc is None:
            return "syncer subprocess was never started"
        code = self.proc.poll()
        if code is None or code == 0:
            return None
        strict_failure = self._strict_failure()
        if strict_failure is not None:
            return strict_failure
        return f"syncer subprocess exited with code {code}"

    def finished(self) -> bool:
        """0.26: the syncer ended its run normally (exit 0, or its elastic
        status.json says state=finished)."""
        if self.proc is not None and self.proc.poll() == 0:
            return True
        from .rl.engine.island_status import read_syncer_status

        status = read_syncer_status(self.event_tape)
        return bool(status) and status.get("state") == "finished"

    def _strict_failure(self) -> "_RlStrictFailure | None":
        try:
            with open(self.event_tape, encoding="utf-8") as handle:
                handle.seek(self._event_offset)
                for line in handle:
                    event = json.loads(line)
                    if event.get("event") == "rl_strict_failure":
                        return _RlStrictFailure(
                            f"{event.get('metric', 'strict_failure')}: "
                            f"{event.get('error', 'syncer strict invariant failed')}"
                        )
        except (OSError, ValueError, TypeError):
            return None
        return None

    def restart(self) -> None:
        self.start()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._signal_tree(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_tree(signal.SIGKILL)
        if self.tape_forwarder is not None:
            self.tape_forwarder.stop()
            self.tape_forwarder = None
        if self.tape_run is not None:
            self.tape_run.finish()
            self.tape_run = None

    def start_tape_forwarder(self, args) -> None:
        """Tail the syncer's event tape into a W&B run.

        The syncer is Rust and stays that way: its merge records already
        carry quorum/grace timings, per-island staleness, and per-island
        contribution, so a reader on this host turns them into the fleet's
        syncer-side curves with no change to the hot path. Survives a
        syncer restart because the tape is appended to, not rewritten.
        """
        if not getattr(args, "wandb", False):
            return
        from . import wandb_logger
        from .wandb_tape import OffsetStore, TapeForwarder, offset_path_for

        run = wandb_logger.init(
            args,
            job_type="syncer",
            name="syncer",
            step_metrics={"sync/*": "global_step", "learner/*": "global_step"},
            config_extra={"syncer_command": self.command},
        )
        if not run.enabled:
            return
        self.tape_run = run
        self.tape_forwarder = TapeForwarder(
            run,
            self.event_tape,
            # A reused head keeps the previous controller job's tape; the
            # stored offset stops this one from re-logging its merges.
            offsets=OffsetStore(offset_path_for(self.event_tape)),
        )
        self.tape_forwarder.start()

    def _signal_tree(self, sig: int) -> None:
        """Signal the syncer's whole process group (see start()), falling
        back to the direct child."""
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.send_signal(sig)
            except OSError:
                pass

    def start_log_forwarder(self) -> None:
        """Tail the syncer's log file into our stdout, forever (daemon)."""

        def _forward():
            while not os.path.exists(self.log_file):
                time.sleep(0.5)
            with open(self.log_file, "r", errors="replace") as f:
                f.seek(self._log_offset)
                while True:
                    line = f.readline()
                    if line:
                        print(f"[syncer] {line.rstrip()}", flush=True)
                    else:
                        time.sleep(0.5)

        threading.Thread(target=_forward, daemon=True).start()


# FleetController states.
RUNNING = "running"
RECOVERING = "recovering"
DONE = "done"
ABANDONED = "abandoned"


@dataclass(frozen=True)
class _RlStrictFailure:
    reason: str


class _RelaunchAttempt:
    """Result slot for one background relaunch attempt."""

    def __init__(self):
        self.result = None  # new job id, or None if provisioning failed
        self.finished = False
        self.thread = None
        self.blocked = None  # reason the relaunch was refused (instance alive)
        self.name = None  # cluster name the relaunch used (may be a new one)


# Fixed-roster RL islands: consecutive relaunches allowed within one recovery
# window. A window closes (budget and count reset) once a relaunched island
# has stayed healthy for RECOVERY_STABLE_S; the time budget counts only the
# time spent recovering in the window.
FIXED_ROSTER_MAX_RELAUNCHES = 2
RECOVERY_STABLE_S = 300.0
# How long a failed-looking island is given for its rl_learner_finalized record
# to arrive over the log stream before recovery starts.
FINALIZE_GRACE_S = 60.0
# Bound on waiting for an in-flight relaunch before abandoning (so a late
# sky.launch cannot re-create a cluster after teardown).
RELAUNCH_JOIN_S = 120.0


class RunStalled(RuntimeError):
    """No island produced an event for the stall timeout (and not all finalized)."""


# Exit code when the run stalled (no island event for --rl-stall-timeout).
RUN_STALLED_EXIT = 6
DEFAULT_RL_STALL_TIMEOUT_S = 900.0


class FixedRosterIslandAbandoned(RuntimeError):
    """A fixed-roster RL island could not be recovered: the run cannot finish."""
# run() exit code when a fixed-roster island failed for good (distinct from
# 2 = artifact not fetchable and 3 = incomplete event tape).
ISLAND_FAILED_EXIT = 4
FAILED_RUN_DRAIN_S = 20.0
# Modal app not confirmed stopped (state stopped, 0 tasks) after teardown.
TEARDOWN_UNVERIFIED_EXIT = 5
# Modal moved an island to a different container mid-run (preempted/rescheduled and the
# run script re-ran): the run is failed and the Modal app stopped. Registered exit codes:
# 1 failed, 2 artifact not fetchable, 3 incomplete tape, 4 island failed, 5 teardown
# unverified, 6 stalled, 7 Modal container id changed.
CONTAINER_CHANGED_EXIT = 7
MODAL_STOP_VERIFY_ATTEMPTS = 5
MODAL_STOP_VERIFY_DELAY_S = 5.0


def _list_modal_apps() -> list[dict]:
    """Rows of ``modal app list --json`` (app_id, description, state, tasks,
    created_at, stopped_at). Raises when the listing fails."""

    import subprocess as _subprocess

    proc = _subprocess.run(
        [sys.executable, "-m", "modal", "app", "list", "--json"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"modal app list failed: {(proc.stdout + proc.stderr).strip() or proc.returncode}")
    return json.loads(proc.stdout or "[]")


def _created_unix(row) -> float | None:
    from datetime import datetime

    value = row.get("created_at")
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _verify_modal_app_stopped(modal_ops, args, *, run_started_unix: float | None = None) -> bool:
    """After stop_app, confirm on the provider that nothing of this app runs.

    ``modal app list`` can hold several rows with the same app name (earlier
    runs). EVERY row with this name must be stopped with 0 tasks (bounded
    retries); the rows created at/after this run's start are this run's app,
    and their app ids are recorded in <run dir>/teardown.json.
    """

    from . import runs

    checks, confirmed, ours = [], False, []
    for attempt in range(MODAL_STOP_VERIFY_ATTEMPTS):
        try:
            rows = [r for r in _list_modal_apps() if r.get("description") == modal_ops.app_name]
        except Exception as e:  # noqa: BLE001 - an unreadable listing is "unverified"
            checks.append(f"error: {e}")
        else:
            ours = [r.get("app_id") for r in rows
                    if run_started_unix is not None
                    and (_created_unix(r) or 0) >= run_started_unix - 1]
            summary = [{"app_id": r.get("app_id"), "state": str(r.get("state", "")).lower(),
                        "tasks": r.get("tasks"), "created_at": r.get("created_at")}
                       for r in rows]
            checks.append(summary)
            if all(s["state"] == "stopped" and int(s["tasks"] or 0) == 0 for s in summary):
                confirmed = True  # also when no row is listed any more
                break
        if attempt + 1 < MODAL_STOP_VERIFY_ATTEMPTS:
            time.sleep(MODAL_STOP_VERIFY_DELAY_S)
    record = {"provider": "modal", "app": modal_ops.app_name, "this_run_app_ids": ours,
              "confirmed_stopped": confirmed, "checks": checks}
    try:
        path = runs.run_dir(args.cluster_prefix) / "teardown.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"[launcher] could not write the teardown record: {e}", file=sys.stderr)
    if not confirmed:
        print(f"[launcher] WARN: Modal app {modal_ops.app_name} not confirmed stopped after "
              f"{len(checks)} check(s) (every row with this name must be stopped with 0 "
              f"tasks; last: {checks[-1]}); stop it by hand: modal app stop "
              f"{modal_ops.app_name} (exit {TEARDOWN_UNVERIFIED_EXIT})", file=sys.stderr)
    return confirmed


def effective_recover_timeout(args) -> float:
    """The fleet controller's learner relaunch budget.

    ``--no-island-relaunch`` or ``--modal-retries 0`` (a learner exit is final,
    e.g. acceptance runs) -> 0: a failed island is torn down, never relaunched
    by the launcher, so no second paid container can start before the app
    stops. Otherwise ``--recover-timeout`` unchanged (all clouds share this
    loop: sky islands relaunch through the same FleetController).
    """
    if getattr(args, "no_island_relaunch", False):
        return 0
    if getattr(args, "modal_retries", None) == 0 and not getattr(args, "modal_launcher_relaunch", False):
        return 0
    return args.recover_timeout


class FleetController:
    """Supervises the syncer + learner fleet after the initial launch.

    Per-learner state machine (evaluated once per poll):

        running    --(job terminal-not-SUCCEEDED, or cluster not UP)--> recovering
        recovering --(relaunch returns a new job id)------------------> running
        recovering --(recover_timeout exceeded)-----------------------> abandoned
        running    --(job SUCCEEDED)----------------------------------> done

    An abandoned learner's cluster is torn down immediately (even with
    --keep) and the run continues with the remaining fleet: the syncer's
    quorum design tolerates missing learners, and a learner that comes back
    later is caught up by the syncer's full rebroadcast. The syncer follows
    the same running/recovering cycle but is never abandoned — past the
    timeout it keeps retrying and logs an error every poll (its relaunch
    resumes from the on-VM checkpoint via the --resume flag already in its
    run command). recover_timeout <= 0 disables recovery: a failed learner
    is torn down on the spot.

    The syncer can be supervised in one of two ways:

    * as a cluster (local controller mode): pass ``syncer=(name, task,
      job_id)`` and it goes through the running/recovering cycle above,
      relaunched via sky like a learner but never abandoned;
    * as a subprocess of this process (head controller mode): pass
      ``syncer_probe``/``syncer_restart`` callables instead — ``probe()``
      returns None while healthy or a reason string when dead, at which
      point ``restart()`` is invoked (the syncer resumes from its local
      checkpoint). A local syncer is likewise never abandoned, and there
      is no syncer cluster to tear down.

    At most one relaunch attempt is in flight per cluster, each in a
    background thread so one slow re-provision never blocks polling the
    others; `thread_cls` exists so tests can substitute a synchronous stub.
    """

    def __init__(
        self,
        learners: dict,
        syncer: tuple | None,
        sky_ops,
        poll_interval: float,
        recover_timeout: float,
        on_relaunch=None,
        thread_cls=threading.Thread,
        syncer_probe=None,
        syncer_restart=None,
        fixed_roster: bool = False,
        finalized_probe=None,
        progress_probe=None,
        stall_timeout: float = 0.0,
        stop_flag=None,
        no_recover=(),
        instance_guard=None,
        on_rename=None,
        fleet_log=None,
        keep_abandoned: bool = False,
        syncer_finished_probe=None,
        left_pool_probe=None,
        elastic: bool = False,
        no_island_relaunch: bool = False,
    ):
        """`learners` maps cluster name -> (task, job_id); `syncer` is
        (name, task, job_id) for a cluster syncer, or None with
        `syncer_probe`/`syncer_restart` callables for a subprocess syncer.
        `on_relaunch(name, new_job_id)` is called after every successful
        cluster relaunch (production spawns a new log tail)."""
        self.ops = sky_ops
        # 0.26: () -> bool, the syncer finished its run (no island is relaunched
        # after that; a late island's failure then counts as a normal end).
        self.syncer_finished_probe = syncer_finished_probe
        # 0.27: name -> bool, the island left the elastic pool for good (re-JOIN
        # exhausted, learner exit 7); elastic only.
        self.left_pool_probe = left_pool_probe
        self.elastic = elastic
        self.no_island_relaunch = no_island_relaunch
        # B12: True (--keep-abandoned) leaves an abandoned learner's cluster up for
        # post-mortem instead of tearing it down; the final teardown still honors --keep.
        self.keep_abandoned = keep_abandoned
        # Path of the run's STOP flag file (``runs.stop_flag_path``; ``yeto stop-run``
        # writes it): once it exists no island is relaunched any more.
        self.stop_flag = stop_flag
        self.poll_interval = poll_interval
        self.recover_timeout = recover_timeout
        self.on_relaunch = on_relaunch
        self.thread_cls = thread_cls
        self.fixed_roster = fixed_roster
        # name -> bool: the island's tape already holds rl_learner_finalized
        # (training complete); a non-zero exit after that is a shutdown-phase
        # error, not an island failure.
        self.finalized_probe = finalized_probe
        # () -> int: events received from the islands so far (log echo). With
        # stall_timeout > 0 a run with no new event for that long, and not every
        # learner finalized, is stalled (e.g. a dead syncer connection).
        self.progress_probe = progress_probe
        self.stall_timeout = stall_timeout
        self._progress = None
        self._progress_at = None
        self.learners = {
            name: self._make_record(name, task, job_id)
            for name, (task, job_id) in learners.items()
        }
        if syncer_probe is not None:
            if syncer_restart is None:
                raise ValueError("syncer_probe requires syncer_restart")
            self.syncer = None
            self.syncer_probe = syncer_probe
            self.syncer_restart = syncer_restart
        elif syncer is None:
            # --rl-single-island-no-sync: no syncer of any kind to supervise.
            self.syncer = None
            self.syncer_probe = self.syncer_restart = None
        else:
            syncer_name, syncer_task, syncer_job = syncer
            self.syncer = self._make_record(syncer_name, syncer_task, syncer_job)
            self.syncer_probe = self.syncer_restart = None
        self.downed_clusters: set = set()
        # Learners whose recovery is disabled regardless of recover_timeout
        # (Verda without the verified sky patch: a same-name relaunch there
        # can delete the running node).
        self.no_recover = set(no_recover)
        # Cloud-side check before a relaunch (VerdaInstanceGuard): None, or
        # an object whose check(name) returns ("alive", why) | ("gone", new
        # name) | ("unknown", why) | None (not guarded).
        self.instance_guard = instance_guard
        self.on_rename = on_rename
        # dashboard.fleet.FleetLog (fleet-dashboard 2.3): island lifecycle and
        # periodic cost_tick into the run's fleet.jsonl; best-effort, optional.
        self.fleet_log = fleet_log

    def _fleet(self, kind: str, name: str | None = None, *, force: bool = False, **fields) -> None:
        if self.fleet_log is None:
            return
        try:
            if kind == "cost_tick":
                if force:
                    self.fleet_log.cost_tick()
                else:
                    self.fleet_log.maybe_tick()
            else:
                self.fleet_log.event(kind, name, **fields)
        except Exception as e:  # noqa: BLE001 - telemetry never breaks supervision
            print(f"[launcher] fleet.jsonl: {e}", file=sys.stderr)

    @staticmethod
    def _make_record(name, task, job_id):
        return {
            "name": name,
            "task": task,
            "job_id": job_id,
            "state": RUNNING,
            "failed_at": None,
            "attempt": None,
            "exit": None,
        }

    def run(self) -> dict:
        """Poll until every learner is done or abandoned.

        Returns {learner name: final status string}; raises RuntimeError
        (after downing the syncer) if every learner was abandoned.
        """
        for name in self.learners:
            self._fleet("island_ready", name, initial=True)
        while True:
            self._fleet("cost_tick")
            if self.syncer is not None:
                self._poll(self.syncer, is_syncer=True)
            elif self.syncer_probe is not None:
                self._poll_local_syncer()
            for rec in self.learners.values():
                self._poll(rec, is_syncer=False)
            if all(r["state"] in (DONE, ABANDONED) for r in self.learners.values()):
                break
            self._check_stall()
            self.ops.sleep(self.poll_interval)
        exit_codes = {name: rec["exit"] for name, rec in self.learners.items()}
        for name, rec in self.learners.items():
            if rec["state"] == DONE:
                self._fleet("island_stop", name, reason=str(rec["exit"]))
        self._fleet("cost_tick", force=True)
        print(f"[launcher] learner jobs finished: {exit_codes}")
        if not any(rec["state"] == DONE for rec in self.learners.values()):
            if self.syncer is not None:
                print(
                    "[launcher] ERROR: all learners abandoned; tearing down the syncer",
                    file=sys.stderr,
                )
                self._down(self.syncer["name"])
            else:
                print(
                    "[launcher] ERROR: all learners abandoned",
                    file=sys.stderr,
                )
            raise RuntimeError("all learners abandoned; nothing left to train")
        return exit_codes

    def _poll_local_syncer(self) -> None:
        """Subprocess syncer: never abandoned — a dead process is restarted
        on the spot (it resumes from its local checkpoint)."""
        reason = self.syncer_probe()
        if reason is None:
            return
        if self._all_learners_finalized():
            print(f"[launcher] syncer: {reason} after every learner finalized; "
                  "not restarting", file=sys.stderr)
            return
        if isinstance(reason, _RlStrictFailure):
            raise RuntimeError(f"strict RL syncer failed: {reason.reason}")
        print(
            f"[launcher] syncer: {reason}; restarting the local syncer "
            "(resumes from its checkpoint)",
            file=sys.stderr,
        )
        try:
            self.syncer_restart()
        except Exception as e:
            print(
                f"[launcher] syncer restart failed: {e}; retrying next poll",
                file=sys.stderr,
            )

    def _poll(self, rec, is_syncer: bool) -> None:
        if rec["state"] == RUNNING:
            verdict, status = self._probe(rec)
            if verdict is None:
                recovered_at = rec.get("recovered_at")
                if recovered_at is not None and self.ops.now() - recovered_at >= RECOVERY_STABLE_S:
                    # stable again: close the recovery window
                    rec.pop("recovered_at", None)
                    rec["failures"] = 0
                    rec["recovering_s"] = 0.0
                return  # healthy
            if verdict == "succeeded":
                rec["state"] = DONE
                rec["exit"] = str(status)
                print(f"[launcher] {rec['name']} job finished: {status}")
            elif is_syncer and self._all_learners_finalized():
                rec["state"] = DONE  # training is over; nothing to recover
                rec["exit"] = f"stopped after every learner finalized ({status})"
                print(f"[launcher] syncer {verdict} after every learner finalized; "
                      "not recovering", file=sys.stderr)
            elif not is_syncer and self._finalized(rec):
                # Training completed (rl_learner_finalized seen): an error while
                # shutting down (e.g. KeyboardInterrupt in ray.shutdown after
                # the syncer stopped) is neither a failure nor a reason to
                # relaunch.
                rec["state"] = DONE
                rec["exit"] = f"SUCCEEDED (finalized; shutdown ended as {status})"
                print(f"[launcher] WARN: {rec['name']} finalized, then {verdict}; "
                      "counted as succeeded, no recovery", file=sys.stderr)
            elif not is_syncer and self._syncer_finished():
                rec["state"] = DONE  # 0.26: the run is over; a late island's error is not a failure
                rec["exit"] = f"SUCCEEDED (syncer finished; island ended as {status})"
                self._fleet("island_done", rec["name"], reason="syncer finished")
                print(f"[launcher] {rec['name']}: {verdict} after the syncer finished; "
                      "counted as a normal end, not relaunched", file=sys.stderr)
            elif not is_syncer and self.elastic and self._left_pool(rec):
                # 0.27: re-JOIN exhausted (exit 7): the island left the elastic pool.
                # Not the fixed-roster abandon path (exit 4): relaunch unless
                # --no-island-relaunch, and the run goes on without it.
                self._fleet("island_lost", rec["name"], reason="left elastic pool (re-JOIN exhausted)")
                if self.no_island_relaunch:
                    rec["state"] = DONE
                    rec["exit"] = f"LEFT_POOL (re-JOIN exhausted; {status})"
                    print(f"[launcher] {rec['name']}: left the elastic pool (re-JOIN exhausted); "
                          "--no-island-relaunch: not relaunched", file=sys.stderr)
                else:
                    rec["left_pool"] = True
                    self._enter_recovering(rec, "left the elastic pool (re-JOIN exhausted)", is_syncer)
            elif not is_syncer and self._await_finalized(rec):
                return  # grace: the finalized record may still be in the log stream
            else:
                strict_failure = self._strict_failure(rec)
                if strict_failure is not None:
                    raise RuntimeError(
                        f"strict RL job {rec['name']} failed: {strict_failure}"
                    )
                self._enter_recovering(rec, verdict, is_syncer)
        elif rec["state"] == RECOVERING:
            self._drive_recovery(rec, is_syncer)

    def _syncer_finished(self) -> bool:
        if self.syncer is not None and self.syncer["state"] == DONE \
                and "SUCCEEDED" in str(self.syncer.get("exit", "")):
            return True
        if self.syncer_finished_probe is None:
            return False
        try:
            return bool(self.syncer_finished_probe())
        except Exception:
            return False

    def _left_pool(self, rec) -> bool:
        if self.left_pool_probe is None:
            return False
        try:
            return bool(self.left_pool_probe(rec["name"]))
        except Exception:
            return False

    def _check_stall(self) -> None:
        if self.progress_probe is None or self.stall_timeout <= 0:
            return
        try:
            progress = int(self.progress_probe())
        except Exception:
            return
        now = self.ops.now()
        if progress == 0:
            return  # islands still starting (image pull, model load): no clock yet
        if self._progress is None or progress != self._progress:
            self._progress, self._progress_at = progress, now
            return
        if now - self._progress_at >= self.stall_timeout and not self._all_learners_finalized():
            message = (f"no island event for {now - self._progress_at:.0f}s "
                       f"(stall timeout {self.stall_timeout:.0f}s, {progress} events so far)")
            print(f"[launcher] ERROR: run stalled: {message}", file=sys.stderr)
            raise RunStalled(message)

    def _all_learners_finalized(self) -> bool:
        return self.finalized_probe is not None and all(
            r["state"] == DONE or self._finalized(r) for r in self.learners.values()
        )

    def _finalized(self, rec) -> bool:
        if self.finalized_probe is None:
            return False
        try:
            return bool(self.finalized_probe(rec["name"]))
        except Exception:
            return False

    def _await_finalized(self, rec) -> bool:
        """Hold a failure verdict for FINALIZE_GRACE_S while the log stream may
        still deliver the island's rl_learner_finalized record."""

        if self.finalized_probe is None:
            return False
        first = rec.setdefault("failure_seen_at", self.ops.now())
        if self.ops.now() - first < FINALIZE_GRACE_S:
            return True
        rec.pop("failure_seen_at", None)
        return False

    def _probe(self, rec):
        """Classify a running cluster: (None, status) if healthy,
        ("succeeded", status), or (failure reason, status)."""
        name, job_id = rec["name"], rec["job_id"]
        try:
            status = self.ops.job_status(name, job_id)
        except Exception as e:
            # sky refuses job_status while the cluster's health reads INIT
            # (e.g. its runtime Ray is unreachable).  That is not evidence
            # the job is gone: ask the job table before treating it as lost.
            if self._job_alive(name, job_id):
                return None, None
            return f"job status unavailable ({e})", None
        if status is not None and status.is_terminal():
            if "SUCCEEDED" in str(status):
                return "succeeded", status
            return f"job ended as {status}", status
        try:
            up = self.ops.cluster_up(name)
        except Exception as e:
            return f"cluster status unavailable ({e})", status
        if not up and not self._job_alive(name, job_id):
            return "cluster is not UP (preempted or deleted)", status
        return None, status

    def _job_alive(self, name: str, job_id) -> bool:
        probe = getattr(self.ops, "job_alive", None)
        if probe is None:
            return False
        try:
            return bool(probe(name, job_id))
        except Exception:
            return False

    def _strict_failure(self, rec) -> str | None:
        if not self.fixed_roster:
            return None
        probe = getattr(self.ops, "rl_strict_failure", None)
        if probe is None:
            return None
        try:
            return probe(rec["name"], rec["job_id"])
        except Exception:
            return None

    def _enter_recovering(self, rec, reason: str, is_syncer: bool) -> None:
        rec["state"] = RECOVERING
        rec["failed_at"] = self.ops.now()
        if not is_syncer:
            self._fleet("island_lost", rec["name"], reason=str(reason))
        # Fixed-roster RL: the recovery budget is cumulative over relaunches
        # (an island that fails again right after every relaunch must not be
        # relaunched forever while the rest of the fleet waits on the syncer).
        rec["failures"] = rec.get("failures", 0) + 1  # within this recovery window
        rec.pop("recovered_at", None)
        print(
            f"[launcher] {rec['name']}: {reason}; starting recovery "
            f"(timeout {self.recover_timeout}s)",
            file=sys.stderr,
        )
        if not is_syncer and (self.recover_timeout <= 0 or rec["name"] in self.no_recover):
            if rec["name"] in self.no_recover:
                print(
                    f"[launcher] {rec['name']}: auto-recovery disabled for this island "
                    "(sky Verda provisioner unpatched); not relaunching",
                    file=sys.stderr,
                )
            self._abandon(rec, 0.0)
            return
        self._drive_recovery(rec, is_syncer)

    def _drive_recovery(self, rec, is_syncer: bool) -> None:
        if not is_syncer and self._syncer_finished():
            rec["state"] = DONE  # 0.26: never relaunch an island into a finished run
            rec["exit"] = "SUCCEEDED (syncer finished during recovery; not relaunched)"
            print(f"[launcher] {rec['name']}: the syncer finished; recovery stopped", file=sys.stderr)
            return
        attempt = rec["attempt"]
        if attempt is not None and attempt.finished:
            rec["attempt"] = None
            if attempt.blocked is not None and not is_syncer:
                print(
                    f"[launcher] {rec['name']}: NOT relaunching — {attempt.blocked}",
                    file=sys.stderr,
                )
                self._abandon(rec, self.ops.now() - rec["failed_at"])
                return
            if attempt.result is not None and attempt.name and attempt.name != rec["name"]:
                old = rec["name"]
                rec["name"] = attempt.name
                print(f"[launcher] {old} relaunched under a new cluster name: {attempt.name}")
                if self.on_rename is not None:
                    try:
                        self.on_rename(old, attempt.name)
                    except Exception as e:  # noqa: BLE001 - bookkeeping only
                        print(f"[launcher] on_rename hook failed: {e}", file=sys.stderr)
            if attempt.result is not None:
                rec["job_id"] = attempt.result
                rec["state"] = RUNNING
                # time spent recovering counts toward the window's budget
                rec["recovering_s"] = rec.get("recovering_s", 0.0) + (
                    self.ops.now() - rec["failed_at"])
                rec["failed_at"] = None
                rec["recovered_at"] = self.ops.now()
                print(
                    f"[launcher] {rec['name']} recovered: relaunched as job "
                    f"{attempt.result}"
                )
                if not is_syncer:
                    self._fleet("island_ready", rec["name"], relaunched=True)
                if self.on_relaunch is not None:
                    self.on_relaunch(rec["name"], attempt.result)
                return
            print(
                f"[launcher] relaunch attempt for {rec['name']} failed; will retry",
                file=sys.stderr,
            )
        elapsed = self.ops.now() - rec["failed_at"]
        if self.fixed_roster and not is_syncer and not rec.get("left_pool"):
            elapsed += rec.get("recovering_s", 0.0)  # this window only
            if rec.get("failures", 0) > FIXED_ROSTER_MAX_RELAUNCHES:
                self._abandon(rec, elapsed)
                return
        if self.recover_timeout <= 0 or elapsed > self.recover_timeout:
            if is_syncer:
                # The syncer is never abandoned: without it no learner can
                # make outer progress, so keep trying and complain loudly.
                print(
                    f"[launcher] ERROR: syncer unrecovered for {elapsed:.0f}s "
                    f"(recover timeout {self.recover_timeout}s exceeded); "
                    "still retrying — learners cannot sync until it returns",
                    file=sys.stderr,
                )
            else:
                self._abandon(rec, elapsed)
                return
        if rec["attempt"] is None:
            if self._stop_requested(rec, "recovery"):
                if not is_syncer:  # the syncer is never abandoned; it just is not relaunched
                    self._abandon(rec, elapsed, reason="STOP flag")
                return
            rec["attempt"] = self._start_relaunch(rec)

    def _stop_requested(self, rec, where: str) -> bool:
        flag = self.stop_flag
        if flag is None or not os.path.exists(flag):
            return False
        print(f"[launcher] {rec['name']}: STOP flag {flag} present; not relaunching "
              f"({where} ends here)", file=sys.stderr)
        return True

    def _start_relaunch(self, rec) -> _RelaunchAttempt:
        attempt = _RelaunchAttempt()
        name, task = rec["name"], rec["task"]

        def _run():
            target = name
            if self.instance_guard is not None:
                try:
                    verdict = self.instance_guard.check(name)
                except Exception as e:  # noqa: BLE001
                    verdict = ("unknown", f"instance check raised: {e}")
                if verdict is not None:
                    kind, detail = verdict
                    if kind == "alive":
                        attempt.blocked = detail
                        attempt.finished = True
                        return
                    if kind == "unknown":
                        # Could not tell whether the old node is gone: a
                        # relaunch now could delete it. Try again next poll.
                        print(f"[launcher] {name}: {detail}; relaunch deferred", file=sys.stderr)
                        attempt.finished = True
                        return
                    target = detail  # "gone": relaunch under a fresh name
            attempt.name = target
            try:
                if self._stop_requested(rec, "relaunch"):  # set after the check in the poll loop
                    attempt.result = None
                    return
                attempt.result = self.ops.relaunch(task, target)
                if attempt.result is not None and self.instance_guard is not None:
                    after = getattr(self.instance_guard, "after_relaunch", None)
                    if after is not None:
                        try:
                            after(target)
                        except Exception as e:  # noqa: BLE001
                            print(f"[launcher] recording ids of {target} failed: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[launcher] relaunch of {name} raised: {e}", file=sys.stderr)
                attempt.result = None
            finally:
                attempt.finished = True
            if attempt.result is not None and rec["state"] == ABANDONED:
                # Abandoned while this attempt was in flight, but the
                # relaunch re-provisioned the cluster anyway: tear it back
                # down so nothing is left running unattended.
                if not self.keep_abandoned:
                    self._down(target, force=True)

        thread = self.thread_cls(target=_run, daemon=True)
        attempt.thread = thread
        thread.start()
        return attempt

    def _abandon(self, rec, elapsed: float, reason: str | None = None) -> None:
        rec["state"] = ABANDONED
        self._fleet("island_stop", rec["name"], reason=f"abandoned: {reason or ''}".strip())
        rec["exit"] = (f"ABANDONED ({reason}) after {elapsed:.0f}s" if reason
                       else f"ABANDONED after {elapsed:.0f}s")
        attempt = rec.get("attempt")
        if attempt is not None and attempt.thread is not None and not attempt.finished:
            # let an in-flight relaunch finish (bounded) so its cluster is torn
            # down here (the thread downs it when it sees ABANDONED), not left
            # behind after the process exits
            attempt.thread.join(RELAUNCH_JOIN_S)
            if not attempt.finished:
                print(f"[launcher] WARN: relaunch of {rec['name']} still in flight after "
                      f"{RELAUNCH_JOIN_S:.0f}s; verify the provider for leftovers",
                      file=sys.stderr)
        if self.keep_abandoned:
            print(f"[launcher] keeping abandoned cluster {rec['name']} (--keep-abandoned)")
        else:
            self._down(rec["name"])
        if self.fixed_roster and not rec.get("left_pool"):  # 0.27: elastic pool leave is not 4
            message = (
                f"fixed-roster learner {rec['name']} could not recover"
                f"{' (' + reason + ')' if reason else ''} "
                f"({rec.get('failures', 0)} failure(s) in the window, "
                f"{elapsed:.0f}s recovering; timeout {self.recover_timeout}s)"
            )
            print(f"[launcher] ERROR: {message}", file=sys.stderr)
            raise FixedRosterIslandAbandoned(message)
        remaining = sum(1 for r in self.learners.values() if r["state"] != ABANDONED)
        print(
            f"[launcher] LEARNER {rec['name']} ABANDONED after {elapsed:.0f}s "
            f"(could not recover within {self.recover_timeout}s); "
            f"fleet continues with {remaining} learner(s)",
            file=sys.stderr,
        )

    def _down(self, name: str, force: bool = False) -> None:
        if name in self.downed_clusters and not force:
            return
        self.downed_clusters.add(name)
        print(f"[launcher] tearing down {name}")
        try:
            self.ops.down(name)
        except Exception as e:
            print(f"[launcher] teardown of {name} failed: {e}", file=sys.stderr)


def _cloud_live_instances_probe(cluster: str):
    """A zero-arg callable that queries the CLOUD (not sky's state DB) for a
    cluster's non-terminated instances, returning their ids. Captured BEFORE
    sky.down, because down deletes the cluster record we need to build the
    query. Returns None when the cluster can't be cloud-verified — never
    provisioned, or a cloud whose sky implementation predates the provision
    status API — in which case callers trust sky.down. Cloud-agnostic: sky's
    provision.query_instances routes to each cloud's own implementation.
    """
    try:
        from sky import clouds, global_user_state
        from sky import provision as provision_lib

        record = global_user_state.get_cluster_from_name(cluster)
        if record is None or record.get("handle") is None:
            return None
        handle = record["handle"]
        cloud = handle.launched_resources.cloud
        # sky's StatusVersion enum defines only ``>=``; ``<`` raises, which
        # used to trip the except below and silently disable verification
        # for every cloud ("cannot set up cloud verification ... '<' not
        # supported").
        if cloud is None or not (cloud.STATUS_VERSION >= clouds.StatusVersion.SKYPILOT):
            return None
        cloud_name = repr(cloud)
        name = handle.cluster_name
        name_on_cloud = handle.cluster_name_on_cloud
        provider_config = global_user_state.get_cluster_yaml_dict(handle.cluster_yaml)["provider"]

        def live():
            found = provision_lib.query_instances(
                cloud_name, name, name_on_cloud, provider_config,
                non_terminated_only=True,
            )
            # A None status means terminated/terminating; a real status means
            # the instance is still up (or stopped) — i.e. an orphan.
            return [iid for iid, (st, _reason) in found.items() if st is not None]

        return live
    except Exception as e:  # any sky-internals drift -> fall back to trusting down
        print(f"[launcher] cannot set up cloud verification for {cluster}: {e}", file=sys.stderr)
        return None


# Diagnostics fetched before an island is torn down: (label, remote glob or
# None for `sky logs`, local file name). Best effort, bounded.
TEARDOWN_DIAG_DIR = "~/yeto-diag"
TEARDOWN_DIAG_TIMEOUT_S = 120


def teardown_diagnostic_commands(cluster: str, dest: str, head: bool = False) -> list[tuple[str, list[str], str | None]]:
    """(label, argv, stdout file or None) for everything worth keeping
    before `cluster` goes away: its job log and event tape(s); on the head
    also sky's API server log and the cluster's recorded events."""
    cmds: list[tuple[str, list[str], str | None]] = [
        ("job_log", ["sky", "logs", cluster, "--no-follow"], os.path.join(dest, "job.log")),
        (
            "event_tape",
            ["rsync", "-a", "-e", "ssh -o ConnectTimeout=20 -o BatchMode=yes",
             f"{cluster}:~/yeto-output/*.jsonl", dest + "/"],
            None,
        ),
    ]
    if head:
        cmds.append(("sky_server_log", ["cp", os.path.expanduser("~/.sky/api_server/server.log"), dest + "/"], None))
        cmds.append(
            (
                "cluster_events",
                ["python3", "-c",
                 "import json,sys\n"
                 "from sky import global_user_state as g\n"
                 "f=getattr(g,'get_cluster_events',None)\n"
                 "print(json.dumps(f(sys.argv[1], None) if f else 'unsupported', default=str, indent=1))",
                 cluster],
                os.path.join(dest, "cluster_events.json"),
            )
        )
    return cmds


def collect_teardown_diagnostics(
    cluster: str, dest_root: str | None = None, *, head: bool = False,
    run=subprocess.run, timeout: float = TEARDOWN_DIAG_TIMEOUT_S,
) -> dict[str, str]:
    """Best-effort pull of an island's logs before its teardown.

    Never raises: every command gets `timeout` seconds and a failure or
    timeout is only a warning, so the teardown that follows always runs.
    Returns {label: "ok" | "timeout" | "failed: ..."}."""
    dest = os.path.join(os.path.expanduser(dest_root or TEARDOWN_DIAG_DIR), cluster)
    out: dict[str, str] = {}
    try:
        os.makedirs(dest, exist_ok=True)
    except OSError as e:
        print(f"[launcher] WARNING: no diagnostics dir for {cluster}: {e}", file=sys.stderr)
        return {"dir": f"failed: {e}"}
    for label, argv, stdout_file in teardown_diagnostic_commands(cluster, dest, head=head):
        fh = None
        try:
            fh = open(stdout_file, "wb") if stdout_file else None
            proc = run(argv, stdout=fh or subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout, check=False)
            code = getattr(proc, "returncode", 0)
            out[label] = "ok" if code == 0 else f"failed: exit {code}"
        except subprocess.TimeoutExpired:
            out[label] = "timeout"
        except Exception as e:  # noqa: BLE001 - diagnostics must never block teardown
            out[label] = f"failed: {e}"
        finally:
            if fh is not None:
                fh.close()
        if out[label] != "ok":
            print(f"[launcher] WARNING: {cluster}: could not fetch {label} before teardown ({out[label]})",
                  file=sys.stderr)
    print(f"[launcher] {cluster}: pre-teardown diagnostics in {dest}: {out}")
    return out


# Verda islands of this process's run: cluster name -> VerdaInstanceGuard
# (holds their instance ids). terminate_and_verify proves their teardown
# at Verda by id instead of trusting sky.
VERDA_ISLANDS: dict = {}


def _verda_teardown_check(cluster: str):
    """Verda-by-id verification for a Verda island whose ids are on record;
    None (sky's own cloud probe decides) when there are none — an empty id
    set must never read as "confirmed gone"."""
    guard = VERDA_ISLANDS.get(cluster)
    if guard is None:
        return None
    ids = guard.ids.get(cluster)
    if not ids:
        print(f"[launcher] WARNING: {cluster}: no Verda instance ids on record; "
              "falling back to sky's cloud probe", file=sys.stderr)
        return None
    from .verda_ops import verify_teardown

    return lambda: verify_teardown(guard.api, ids, guard.on_cloud.get(cluster))


def collect_diagnostics_parallel(clusters, *, head: bool = False, total_timeout: float = 300.0,
                                 collect=None) -> dict:
    """Pre-teardown diagnostics for every island at once, bounded in total:
    islands still collecting after `total_timeout` are left behind (their
    threads are daemons) and the teardown proceeds."""
    if not clusters or os.environ.get("YETO_TEARDOWN_DIAG", "1") == "0":
        return {}
    collect = collect or collect_teardown_diagnostics
    out: dict = {}

    def one(c):
        try:
            out[c] = collect(c, head=head)
        except Exception as e:  # noqa: BLE001
            out[c] = {"error": str(e)}

    threads = [threading.Thread(target=one, args=(c,), daemon=True) for c in clusters]
    for t in threads:
        t.start()
    deadline = time.monotonic() + total_timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    late = [c for c in clusters if c not in out]
    if late:
        print(f"[launcher] WARNING: diagnostics still running for {late} after {total_timeout:.0f}s; "
              "tearing down anyway", file=sys.stderr)
    return out


def prepare_verda_islands(names: list[str], on_instance_ids=None) -> dict:
    """Per-run Verda setup on this machine: the full local sky catalog, the
    patch status (unpatched -> no auto-recovery for these islands, D4) and
    the instance-id guard used by recovery and teardown."""
    from .shape.providers import write_verda_sky_catalog
    from .sky_patches import verda as verda_patch
    from .verda_ops import VerdaApi, VerdaInstanceGuard

    try:
        print(f"[launcher] verda: full sky catalog written to {write_verda_sky_catalog()}")
    except Exception as e:  # noqa: BLE001 - sky's own catalog still works, just thinner
        print(f"[launcher] WARNING: could not write the Verda sky catalog: {e}", file=sys.stderr)
    ok, why = verda_patch.verified_for_current_sky()
    pth_written = None
    if ok and remote_sky_api_server():
        # Provisioning runs in a server we cannot patch from here.
        ok, why = False, f"sky API server is remote ({remote_sky_api_server()})"
    if ok:
        from .sky_patches import ensure_local_pth

        try:
            pth, fresh = ensure_local_pth(str(REPO_ROOT))
            if fresh:
                # A server already running keeps the unpatched code: treat
                # the patch as not in effect for this run (2.5).
                pth_written = pth
                ok, why = False, (
                    f"patch hook {pth} was only now installed; a sky API server started "
                    "before it runs unpatched (install it persistently with "
                    "`python -m yeto.sky_patches install`, then `sky api stop`)"
                )
        except OSError as e:
            ok, why = False, f"cannot install the patch hook for sky's API server ({e})"
    no_recover = set() if ok else set(names)
    if not ok:
        print(
            f"[launcher] WARNING: sky's Verda provisioner is NOT patched ({why}); "
            f"auto-recovery is disabled for {', '.join(names)} (recover_timeout=0) — "
            "a same-name relaunch there can delete the running node",
            file=sys.stderr,
        )
    guard = VerdaInstanceGuard(VerdaApi(), names, on_ids=on_instance_ids)
    for n in names:
        VERDA_ISLANDS[n] = guard
    return {"guard": guard, "no_recover": no_recover, "pth_written": pth_written}


def remote_sky_api_server() -> str | None:
    """The sky API server endpoint when it is not on this machine, else None."""
    endpoint = os.environ.get("SKYPILOT_API_SERVER_ENDPOINT")
    try:
        from sky.server import common as server_common

        endpoint = server_common.get_server_url()
        if server_common.is_api_server_local(endpoint):
            return None
        return endpoint
    except Exception:  # noqa: BLE001 - sky absent or API drift: judge the env var
        if not endpoint:
            return None
        from urllib.parse import urlparse

        host = urlparse(endpoint).hostname or ""
        return None if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} else endpoint


def verda_launch_candidates(spec, args, availability, demoted) -> list[dict]:
    from .shape.providers import VerdaSignals, verda_any_of, verda_candidates

    types = VerdaSignals(cache=None)._fetch_types()
    per_node = spec.gpus_per_node
    cands = verda_candidates(
        spec.gpu, per_node, types, availability,
        regions=[spec.region] if spec.region else None,
        use_spot=bool(getattr(args, "spot", False)), demoted=demoted,
    )
    return verda_any_of(cands, spec.gpu, per_node)


def _verda_copy_override(sky, cand: dict) -> dict:
    """Resources.copy() keeps the base's cloud/region; an `infra=` override
    on top of those raises "Cannot specify both infra and cloud, region, or
    zone". Translate the candidate's infra into explicit cloud/region/zone."""
    out = dict(cand)
    infra = out.pop("infra", None)
    if infra:
        cloud, _, rest = infra.partition("/")
        region, _, zone = rest.partition("/")
        out["cloud"] = sky.CLOUD_REGISTRY.from_str(cloud)
        out["region"] = region or None
        out["zone"] = zone or None
    return out


def launch_verda_island(sky, task, name: str, spec, args, *, sleep=None):
    """Launch one Verda island over live-stock candidates (D5)."""
    from .shape.providers import VerdaSignals, launch_with_verda_candidates

    sig = VerdaSignals(cache=None, use_spot=bool(getattr(args, "spot", False)))
    base = next(iter(task.resources))

    def launch(cands):
        task.set_resources([base.copy(**_verda_copy_override(sky, c)) for c in cands])
        print(f"[launcher] {name}: Verda candidates {[c['instance_type'] + '@' + c['infra'] for c in cands]}")
        return sky.stream_and_get(sky.launch(task, cluster_name=name, retry_until_up=False))

    return launch_with_verda_candidates(
        launch,
        sig._fetch_availability,
        lambda avail, demoted: verda_launch_candidates(spec, args, avail, demoted),
        sleep=sleep,
    )


def _worker_node_ip_flag(cloud: str) -> str:
    """Modal only (CLOUD-OPTIONS-S16 A.5 item 4): a Modal container has both an
    i6pn IPv6 and an IPv4 interface, so a worker lets Ray pick and may register an
    unreachable address.  Pin it to this rank's IPv4 from SKYPILOT_NODE_IPS
    (modal_runner maps the IPv4 list, rank order).  Other clouds: unchanged argv."""
    if cloud != "modal":
        return ""
    return (' --node-ip-address="$(echo "$SKYPILOT_NODE_IPS" '
            '| sed -n "$((SKYPILOT_NODE_RANK + 1))p")"')


def _island_post_cmd(args) -> str:
    """``--rl-island-post-cmd``: shell appended to the rank-0 learner line.

    The learner's exit code is captured with ``|| LRC=$?`` (so ``set -e``
    does not end the script first), the command runs with its output in
    ``~/yeto-output/post-cmd.txt`` (the Modal tape mirror picks ``.txt`` up),
    and the script exits with the learner's code, so the island verdict is
    unchanged.  Empty when the flag is unset (every other run unchanged).
    """
    cmd = getattr(args, "rl_island_post_cmd", None)
    if not cmd:
        return ""
    return (
        " || LRC=$?\n"
        "  mkdir -p \"$HOME/yeto-output\"\n"
        f"  ( set +e; {cmd} ) > \"$HOME/yeto-output/post-cmd.txt\" 2>&1; "
        "echo \"[yeto-island] post-cmd rc=$?\"\n"
        "  exit ${LRC:-0}"
    )


def _rename_hook(clusters: list, on_clusters, fixed: list):
    """FleetController on_rename: keep the teardown list, the Verda id guard
    and the run registry on the island's new cluster name."""

    def hook(old: str, new: str) -> None:
        if old in clusters:
            clusters[clusters.index(old)] = new
        else:
            clusters.append(new)
        if old in VERDA_ISLANDS:
            VERDA_ISLANDS[new] = VERDA_ISLANDS[old]
        if on_clusters is not None:
            on_clusters(fixed + [c for c in clusters if c not in fixed])

    return hook


def teardown_island(sky, cluster: str, *, head: bool = False, collect=collect_teardown_diagnostics,
                    num_nodes: int = 1) -> bool:
    """The launcher's teardown of one of its own clusters: diagnostics
    first, then terminate_and_verify (Verda-verified for Verda islands)."""
    if collect is not None and os.environ.get("YETO_TEARDOWN_DIAG", "1") != "0":
        try:
            collect(cluster, head=head)
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] WARNING: diagnostics for {cluster} failed: {e}", file=sys.stderr)
    return terminate_and_verify(sky, cluster, num_nodes=num_nodes, verda_check=_verda_teardown_check(cluster))


def terminate_and_verify(
    sky, cluster, *, probe="auto", attempts=4, sleep_fn=time.sleep, down=None, num_nodes=1,
    verda_check=None,
) -> bool:
    """sky.down a cluster and CONFIRM at the cloud level that no instance
    survives, retrying the down while the cloud still reports live ones.

    sky.down has been observed to report success while a spot instance
    lingers (state DB and cloud diverge). In head-controller mode the head's
    sky is the ONLY thing that can reach the learner clusters, so a silent
    orphan is unrecoverable once the head is gone — hence verify here, before
    the head relinquishes control. Returns True iff the cluster is confirmed
    gone. When the cloud can't be queried we fall back to sky.down's own
    result: a clean down, or "does not exist" (sky never had it), counts;
    any other down error does not — that is exactly the case that used to
    print "teardown failed" and then claim the run was down.

    `down` overrides the sky.down call (the CLI routes it through its own
    patchable hook); `probe` is captured before the first down because
    sky.down deletes the record the probe is built from.

    `verda_check` (Verda islands) replaces sky's probe with Verda's own
    answer by instance id, including the OS volumes (trash included): it
    returns (ok, remaining) and the teardown counts only when ok.
    """
    if verda_check is not None:
        try:
            down = down or (lambda: sky.get(sky.down(cluster)))
            down()
        except Exception as e:
            print(f"[launcher] sky.down({cluster}) error: {e}; verifying at Verda", file=sys.stderr)
        try:
            ok, remaining = verda_check()
        except Exception as e:  # noqa: BLE001
            ok, remaining = False, [f"Verda verification failed: {e}"]
        if not ok:
            print(
                f"[launcher] {cluster}: teardown NOT complete at Verda; remaining: {remaining}",
                file=sys.stderr,
            )
        else:
            print(f"[launcher] {cluster}: Verda confirms instances and volumes deleted")
        return ok
    if probe == "auto":
        probe = _cloud_live_instances_probe(cluster)
    down = down or (lambda: sky.get(sky.down(cluster)))
    if int(num_nodes or 1) > 1:
        return _terminate_and_verify_nodes(cluster, probe=probe, attempts=attempts,
                                           sleep_fn=sleep_fn, down=down, num_nodes=int(num_nodes))

    def _down():
        try:
            down()
            return None
        except Exception as e:
            print(f"[launcher] sky.down({cluster}) error: {e}", file=sys.stderr)
            return e

    err = _down()
    if probe is None:
        return err is None or "does not exist" in str(err)
    for i in range(attempts):
        try:
            live = probe()
        except Exception as e:
            print(f"[launcher] cloud verify of {cluster} failed ({e}); trusting sky.down",
                  file=sys.stderr)
            return True
        if not live:
            return True
        print(
            f"[launcher] {cluster}: {len(live)} instance(s) still live after down "
            f"({','.join(map(str, live))}); retrying teardown ({i + 1}/{attempts})",
            file=sys.stderr,
        )
        sleep_fn(min(30, 5 * (i + 1)))
        _down()
    try:
        return not probe()
    except Exception:
        return True


def _terminate_and_verify_nodes(cluster, *, probe, attempts, sleep_fn, down, num_nodes) -> bool:
    """rl-multinode-island D10: a multi-node island is confirmed gone only when the
    cloud confirms EVERY node instance. The instance ids are captured before the
    first down; no cloud probe, a failing probe or any instance still live after
    the retries -> False, with the unconfirmed instance ids listed. sky.down's own
    result is never trusted for more than one node."""
    if probe is None:
        print(f"[launcher] {cluster}: {num_nodes}-node island cannot be cloud-verified; "
              "NOT confirmed down (check the cloud console)", file=sys.stderr)
        return False
    try:
        before = list(probe())
    except Exception as e:  # noqa: BLE001 - verification must not rely on sky's DB
        print(f"[launcher] {cluster}: cloud probe before down failed ({e}); NOT confirmed",
              file=sys.stderr)
        return False
    if len(before) != num_nodes:
        print(f"[launcher] {cluster}: cloud shows {len(before)} instance(s) for a {num_nodes}-node "
              f"island ({','.join(map(str, before))}); verifying what the cloud reports", file=sys.stderr)
    live = before
    for i in range(attempts + 1):
        try:
            down()
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] sky.down({cluster}) error: {e}", file=sys.stderr)
        try:
            live = list(probe())
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] {cluster}: cloud verify failed ({e}); NOT confirmed", file=sys.stderr)
            return False
        if not live:
            for iid in before:
                print(f"[launcher] {cluster}: node instance {iid} confirmed terminated")
            if not before:
                print(f"[launcher] {cluster}: cloud reports no instances (confirmed)")
            return True
        if i < attempts:
            print(f"[launcher] {cluster}: {len(live)} node instance(s) still live "
                  f"({','.join(map(str, live))}); retrying teardown ({i + 1}/{attempts})", file=sys.stderr)
            sleep_fn(min(30, 5 * (i + 1)))
    print(f"[launcher] {cluster}: UNCONFIRMED node instance(s) {','.join(map(str, live))} "
          "still live after teardown; delete them in the cloud console or rerun `yeto down`",
          file=sys.stderr)
    return False


def run(args, on_clusters=None, local_syncer=None, on_instance_ids=None) -> int:
    """Provision and supervise the fleet; returns the run's exit code.

    `on_clusters`, if given, is called once with the full list of cluster
    names for this run (syncer first). Names are deterministic from the
    args, so the callback fires before provisioning starts — callers (the
    CLI's run registry) can record them for status/teardown even if the
    launch dies mid-provision. Optional and best-effort: existing callers
    need not pass it, and a failing hook never aborts the run.

    `local_syncer` switches on head controller mode: this process is
    running ON the head VM, the syncer is the given LocalSyncer subprocess
    (already started by the caller), and no separate syncer cluster is
    launched — learners connect to this host's public IP
    ($SYNCER_PUBLIC_IP, injected by the submitting CLI).

    `on_instance_ids(cluster, ids)`, if given, receives the Verda instance
    ids of each Verda island once it is up (the CLI saves them in the run
    record; recovery and teardown are checked against them).
    """
    import sky

    if getattr(args, "rl_single_island_no_sync", False) and (
        getattr(args, "training_mode", "sft") != "rl"
    ):
        # Checked first: the flag must never turn an SFT launch syncer-less.
        raise ValueError("--rl-single-island-no-sync requires --training-mode rl")
    from . import sky_patches

    sky_patches.install()
    prepare_launch_args(args)
    _write_run_manifest(args)
    head_mode = local_syncer is not None
    # --rl-single-island-no-sync: one ports island, no syncer at all
    # (validated in _prepare_ports_algorithm before any cloud work).
    no_sync = bool(getattr(args, "rl_single_island_no_sync", False))
    if no_sync and head_mode:
        raise ValueError("--rl-single-island-no-sync has no syncer; do not run it as a head")
    specs = parse_gpu_spec(args.gpu)
    nodes_by_cluster = dict(zip(learner_cluster_names(args.cluster_prefix, specs),
                                (s.num_nodes for s in specs)))  # rl-multinode-island D10
    no_sync_incomplete: list[str] = []
    # Islands whose tape travels over the log stream (no fetchable
    # ~/yeto-output): every Modal RL ports island, and a no-sync island.
    echo_names = {
        name for name, spec in zip(learner_cluster_names(args.cluster_prefix, specs), specs)
        if _echoes_events(args, spec)
    }
    events_dir = _no_sync_events_dir(args) if echo_names else None
    if echo_names:
        # Refuse to mix runs in one tape, before anything is provisioned.
        existing = sorted(str(p) for p in events_dir.glob("*.jsonl*")) if events_dir.exists() else []
        if existing:
            raise ValueError(
                f"event tapes already exist for run {args.cluster_prefix!r}: {existing}; "
                "use another --cluster-prefix or remove them"
            )
    # External learners (machines sky cannot provision — e.g. Macs running
    # yeto.mlx.learner) get the ids AFTER the cloud learners; the syncer
    # counts them in --learners and its port is already public, so they
    # simply dial in with the printed join command.
    external = max(0, getattr(args, "external_learners", 0) or 0)
    num_learners = len(specs) + external
    require_modal_for_gpu_exact(args, specs)
    warn_if_model_wont_fit(args, specs)
    prefix = args.cluster_prefix
    syncer_cluster = None if head_mode or no_sync else sky_cluster_name(f"{prefix}-syncer")
    learner_names = learner_cluster_names(prefix, specs)
    if on_clusters is not None:
        try:
            on_clusters(([] if syncer_cluster is None else [syncer_cluster]) + learner_names)
        except Exception as e:
            print(f"[launcher] on_clusters hook failed: {e}", file=sys.stderr)
    clusters: list[str] = []
    controller = None
    teardown_unverified = False
    run_started_unix = time.time()  # Modal rows created after this are this run's app
    verda_names = [n for n, sp in zip(learner_names, specs) if sp.cloud == "verda"]
    verda = prepare_verda_islands(verda_names, on_instance_ids) if verda_names else None
    # Read by the finally block (Modal stop / tape pull) even on an early failure.
    modal_ops = None
    modal_cfgs: dict[str, object] = {}

    try:
        # 1. Syncer: a subprocess on this host (head mode) or its own VM.
        if no_sync:
            syncer_task = syncer_job = None
            syncer_addr = "none"  # the island command carries no --syncer
            print("[launcher] --rl-single-island-no-sync: no syncer, no outer sync")
        elif head_mode:
            syncer_task = syncer_job = None
            syncer_addr = f"{os.environ['SYNCER_PUBLIC_IP']}:{SYNCER_PORT}"
            print(f"[launcher] syncer runs on this head node at {syncer_addr}")
        else:
            if getattr(args, "wandb", False):
                print(
                    "[launcher] --wandb: the syncer cluster carries its own "
                    "event-tape forwarder (log: /tmp/yeto-tape-wandb.log)"
                )
            print(f"[launcher] launching syncer cluster {syncer_cluster} in {args.syncer_region}")
            syncer_task = make_syncer_task(args, num_learners)
            rid = sky.launch(syncer_task, cluster_name=syncer_cluster)
            syncer_job, syncer_handle = sky.stream_and_get(rid)
            clusters.append(syncer_cluster)
            syncer_addr = f"{syncer_handle.head_ip}:{SYNCER_PORT}"
            print(f"[launcher] syncer up at {syncer_addr}")
            probe_syncer_ports(args, str(syncer_handle.head_ip))
        if not no_sync and island_scheduling_mode(args) == "elastic" \
                and getattr(args, "rl_syncer_epoch", None) is None:
            # 0.17: the islands JOIN with the term the syncer actually runs.
            resolve_island_syncer_epoch(args, syncer_status_reader(
                local_tape=syncer_event_tape(args) if head_mode else None,
                ssh_host=None if head_mode else syncer_cluster))

        if external:
            for x in range(external):
                external_provenance_flags = ""
                if getattr(args, "model_revision", None):
                    external_provenance_flags += (
                        f" --model-revision {shlex.quote(args.model_revision)}"
                    )
                if getattr(args, "data_revision", None):
                    external_provenance_flags += (
                        f" --data-revision {shlex.quote(args.data_revision)}"
                    )
                if getattr(args, "trust_remote_code", False):
                    external_provenance_flags += " --trust-remote-code"
                if getattr(args, "source_sha256", None):
                    external_provenance_flags += (
                        f" --source-sha256 {shlex.quote(args.source_sha256)}"
                    )
                for flag_name in (
                    "model_requested_identifier",
                    "model_requested_revision",
                    "data_requested_identifier",
                    "data_requested_revision",
                ):
                    flag_value = getattr(args, flag_name, None)
                    if flag_value is not None:
                        external_provenance_flags += (
                            f" --{flag_name.replace('_', '-')} "
                            f"{shlex.quote(str(flag_value))}"
                        )
                print(
                    f"[launcher] external learner slot {len(specs) + x}: join with\n"
                    f"    python -m yeto.mlx.learner --model {shlex.quote(args.model)} "
                    f"--data {shlex.quote(args.data)} --syncer {shlex.quote(syncer_addr)} "
                    f"--learner-id {len(specs) + x} --num-learners {num_learners} "
                    f"--data-format {getattr(args, 'data_format', 'auto')} "
                    f"--train-on {args.train_on} "
                    f"--assistant-mask-mode "
                    f"{getattr(args, 'assistant_mask_mode', 'native')} "
                    f"--seed {getattr(args, 'seed', 0)} "
                    f"--tuning {args.tuning} --lora-r {args.lora_r} "
                    f"--lora-targets {getattr(args, 'lora_targets', 'auto')} "
                    f"--seq-len {args.seq_len} --fragments {args.fragments} "
                    f"--fragment-pattern {args.fragment_pattern} "
                    f"--merge-alpha {args.merge_alpha} --wire-dtype {args.wire_dtype}"
                    f"{external_provenance_flags}"
                )
            print(
                f"[launcher] the syncer will wait for all {num_learners} learners "
                f"({len(specs)} cloud + {external} external) before training starts"
            )

        # 2. Learners, in parallel. Modal entries are not sky clusters: the
        #    factory still builds their task (run script, envs) and the
        #    Modal runner executes it in a GPU container.
        tasks = {}
        rids = {}
        # Bound before any island is built: the failure path below stops the
        # Modal app when modal_ops is set, and must not NameError otherwise.
        modal_island_ops = None
        modal_ops = None
        task_factory = (
            make_miles_island_task
            if getattr(args, "training_mode", "sft") == "rl"
            else make_learner_task
        )
        modal_addr = None
        if no_sync:
            modal_addr = syncer_addr  # nothing to reach
        elif any(spec.cloud == "modal" for spec in specs):
            from .modal_runner import resolve_syncer_for_modal

            # Fails BEFORE any Modal container starts when the syncer is
            # not reachable from Modal's network.
            modal_addr = resolve_syncer_for_modal(
                syncer_addr, getattr(args, "syncer_public_addr", None)
            )
        for m, spec in enumerate(specs):
            name = learner_names[m]
            task = task_factory(args, spec, m, num_learners, syncer_addr)
            if spec.cloud == "modal":
                cfg = build_modal_island_config(args, spec, m, task, modal_addr)
                print(f"[launcher] {name} dials {island_syncer_addrs(cfg.envs)}", flush=True)
                tasks[name] = cfg
                modal_cfgs[name] = cfg
                continue
            tasks[name] = task
            print(f"[launcher] {name} dials {island_syncer_addrs(getattr(task, 'envs', None))}", flush=True)
            alloc = (rl_island_spec(args, spec) if getattr(args, "training_mode", "sft") == "rl"
                     else spec)
            print(f"[launcher] launching learner {m} on {spec} as {name}"
                  + (f" (physical {spec.gpus_per_node} GPUs/node billed, island allocated "
                     f"{alloc.gpus_per_node}/node = GPUs 0..{alloc.gpus_per_node - 1})"
                     if alloc is not spec else ""))
            if spec.cloud == "verda":
                # Launched in resolve(): live stock -> ordered any_of, with
                # refresh + bounded backoff on capacity failures.
                rids[name] = (m, ("verda", spec))
                continue
            store_fs = (model_store_filesystem(getattr(args, "model_store", None), spec.cloud, spec.region)
                        if getattr(args, "training_mode", "sft") == "rl" else None)
            if store_fs:
                from sky import skypilot_config

                store_cfg = skypilot_config.override_skypilot_config(
                    model_store_sky_config(store_fs, spec.region))
            else:
                store_cfg = contextlib.nullcontext()
            with store_cfg:  # the request snapshots the client config at submit
                rids[name] = (
                    m,
                    sky.launch(task, cluster_name=name, retry_until_up=args.retry_until_up),
                )

        results = {}
        errors = {}

        def resolve(name: str, m: int, rid) -> None:
            try:
                if isinstance(rid, tuple) and rid and rid[0] == "verda":
                    results[name] = launch_verda_island(sky, tasks[name], name, rid[1], args)
                    verda["guard"].record(
                        name, getattr(results[name][1], "cluster_name_on_cloud", None)
                    )
                else:
                    results[name] = sky.stream_and_get(rid)
            except Exception as e:
                errors[name] = e

        threads = [
            threading.Thread(target=resolve, args=(n, m, r), daemon=True)
            for n, (m, r) in rids.items()
        ]
        for t in threads:
            t.start()
        if modal_cfgs:
            from .modal_runner import ModalIslandOps, ModalOps, modal_app_name

            modal_ops = ModalOps(modal_app_name(prefix))
            for cfg in modal_cfgs.values():
                modal_ops.define(cfg)
            print(f"[launcher] deploying Modal app {modal_ops.app_name} ({len(modal_cfgs)} island(s))")
            modal_ops.deploy()
            modal_island_ops = ModalIslandOps(modal_ops)
            for name, cfg in modal_cfgs.items():
                print(f"[launcher] launching learner {cfg.learner_id} on Modal as {name}")
                call_id = modal_island_ops.relaunch(cfg, name)
                if call_id is None:
                    errors[name] = RuntimeError("Modal refused to start the island")
                else:
                    results[name] = (call_id, None)
        for t in threads:
            t.join()
        for name in list(rids) + list(modal_cfgs):
            if name not in errors:
                clusters.append(name)
        if errors:
            for name, e in errors.items():
                print(f"[launcher] ERROR launching {name}: {e}", file=sys.stderr)
            raise RuntimeError(f"{len(errors)} learner cluster(s) failed to provision")

        # 3. Stream logs while the fleet controller polls health, recovers
        #    failed/preempted clusters, and abandons learners that stay down
        #    past --recover-timeout.
        def spawn_tail(name: str, job_id) -> None:
            label = "syncer" if name == syncer_cluster else name
            collector = None
            if name in echo_names:
                # No syncer tape and (on Modal) no fetchable ~/yeto-output: the
                # island echoes its tape into the log; rebuild it locally.
                collector = event_collectors.get(name)
                if collector is None:  # a relaunch keeps appending to the same tape
                    collector = event_collectors[name] = EventCollector(
                        events_dir / f"{name}.jsonl", fresh=False
                    )
            if modal_ops is not None and name in modal_cfgs:
                from .modal_runner import ContainerIdGuard

                guard = ContainerIdGuard(on_change=on_container_change)
                container_guards.append(guard)
                thread = threading.Thread(
                    target=_tail_modal, args=(modal_ops, job_id, label, collector), kwargs={"guard": guard},
                    daemon=True
                )
            else:
                thread = threading.Thread(
                    target=_tail, args=(name, job_id, label, collector), daemon=True
                )
            thread.start()
            tail_threads.append(thread)

        tail_threads: list[threading.Thread] = []
        event_collectors: dict[str, EventCollector] = {}
        container_guards: list = []

        resume_on_new_container = resumes_in_new_container(args)

        def on_container_change(message: str) -> None:
            if resume_on_new_container:
                # rl-resume-from-checkpoint: the Volume store carries LATEST; the re-run
                # container resumes from the newest complete cut instead of starting over.
                print(f"[launcher] WARNING: {message}; the island resumes from its "
                      "modal-volume checkpoint store (no failure)", file=sys.stderr, flush=True)
                if events_dir is not None:
                    try:
                        with open(Path(events_dir) / "launcher-errors.jsonl", "a") as fh:
                            fh.write(json.dumps({"event": "modal_container_changed", "level": "warning",
                                                 "resume": True, "message": message,
                                                 "time": time.time()}) + "\n")
                    except OSError:
                        pass
                return
            print(f"[launcher] ERROR: {message}; failing the run (exit {CONTAINER_CHANGED_EXIT}) "
                  "and stopping the Modal app", file=sys.stderr, flush=True)
            if events_dir is not None:
                try:
                    with open(Path(events_dir) / "launcher-errors.jsonl", "a") as fh:
                        fh.write(json.dumps({"event": "modal_container_changed", "level": "error",
                                             "message": message, "time": time.time()}) + "\n")
                except OSError:
                    pass
            try:
                modal_ops.stop_app()
            except Exception as e:  # noqa: BLE001 - teardown verification reports the rest
                print(f"[launcher] modal app stop failed: {e}", file=sys.stderr, flush=True)

        def container_changed() -> bool:
            return not resume_on_new_container and any(g.tripped for g in container_guards)

        if syncer_cluster is not None:
            spawn_tail(syncer_cluster, syncer_job)
        for name, (job_id, _handle) in results.items():
            spawn_tail(name, job_id)

        from . import runs
        from .modal_runner import RoutingOps

        sky_sdk_ops = SkySDKOps()
        # rl-multinode-island D10: the failure-path teardown verifies every node instance
        sky_sdk_ops.nodes_by_cluster = dict(nodes_by_cluster)
        controller = FleetController(
            learners={name: (tasks[name], job_id) for name, (job_id, _h) in results.items()},
            syncer=None if syncer_cluster is None else (syncer_cluster, syncer_task, syncer_job),
            sky_ops=RoutingOps(sky_sdk_ops, modal_island_ops),
            poll_interval=args.controller_poll,
            recover_timeout=effective_recover_timeout(args),
            on_relaunch=spawn_tail,
            syncer_probe=local_syncer.probe if head_mode else None,
            syncer_restart=local_syncer.restart if head_mode else None,
            fixed_roster=getattr(args, "training_mode", "sft") == "rl",
            stop_flag=runs.stop_flag_path(args.cluster_prefix),
            finalized_probe=(
                (lambda name: name in event_collectors and event_collectors[name].finalized)
                if echo_names else None
            ),
            progress_probe=(
                (lambda: sum(c.count for c in list(event_collectors.values())))
                if echo_names else None
            ),
            stall_timeout=float(
                getattr(args, "rl_stall_timeout", None) or DEFAULT_RL_STALL_TIMEOUT_S
            ) if getattr(args, "rl_stall_timeout", None) != 0 else 0.0,
            keep_abandoned=bool(getattr(args, "keep_abandoned", False)),
            no_recover=verda["no_recover"] if verda else (),
            instance_guard=verda["guard"] if verda else None,
            on_rename=_rename_hook(clusters, on_clusters, [] if head_mode else [syncer_cluster]),
            fleet_log=_dashboard_fleet_log(args, tasks, results),
            syncer_finished_probe=local_syncer.finished if head_mode else None,
            left_pool_probe=(
                (lambda name: name in event_collectors and event_collectors[name].left_pool)
                if echo_names else None
            ),
            elastic=island_scheduling_mode(args) == "elastic",
            no_island_relaunch=bool(getattr(args, "no_island_relaunch", False)),
        )
        def drain_tapes(limit: float = NO_SYNC_EVENT_DRAIN_S) -> None:
            if echo_names:
                # The island's last events (finalization) must be on disk before
                # teardown: the log streams end when the island exits; bounded wait.
                wait_for_tapes(event_collectors, echo_names, tail_threads, limit)
                modal_tape_dir = None
                if modal_ops is not None and any(n in modal_cfgs for n in echo_names):
                    # Modal islands: the tape Volume (committed every 30 s and at
                    # exit) is the island-side source the echo mirrors.
                    modal_tape_dir = _modal_tape_run_dir(args) / "modal-tape"
                    pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
                no_sync_incomplete.extend(settle_echo_tapes(
                    event_collectors, echo_names, modal_cfgs, events_dir,
                    modal_tape_dir=modal_tape_dir))

        try:
            exit_codes = controller.run()
        except RunStalled as error:
            print(f"[launcher] ERROR: {error}; stopping the run (exit {RUN_STALLED_EXIT})",
                  file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT if container_changed() else RUN_STALLED_EXIT
        except FixedRosterIslandAbandoned as error:
            # A fixed-roster island failed for good: the run cannot finish.
            # Secure the tapes (bounded), then the finally block tears every
            # recorded resource down. Other controller errors (strict syncer /
            # strict RL job failure, all learners abandoned, bugs) propagate.
            print(f"[launcher] ERROR: {error}; stopping the run (exit "
                  f"{ISLAND_FAILED_EXIT})", file=sys.stderr)
            # surviving islands still stream (blocked on the syncer): short wait
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            if no_sync_incomplete:
                print(f"[launcher] event tape incomplete for {no_sync_incomplete} "
                      "(expected after an island failure)", file=sys.stderr)
            return CONTAINER_CHANGED_EXIT if container_changed() else ISLAND_FAILED_EXIT
        except Exception as error:
            if not container_changed():
                raise
            # the app stop that follows a container change makes the controller error out
            print(f"[launcher] controller ended after a Modal container change: {error}",
                  file=sys.stderr)
            drain_tapes(min(NO_SYNC_EVENT_DRAIN_S, FAILED_RUN_DRAIN_S))
            return CONTAINER_CHANGED_EXIT
        drain_tapes()
        if container_changed():
            return CONTAINER_CHANGED_EXIT
        if (no_sync_incomplete and head_mode and island_scheduling_mode(args) == "elastic"):
            # S16 §6.6: an island the syncer removed from the pool (lease expiry,
            # requested leave) owes no rl_learner_finalized; report it, do not exit 3.
            departures = syncer_pool_departures(
                local_syncer.event_tape, getattr(local_syncer, "_event_offset", 0))
            still, left = excuse_departed_islands(no_sync_incomplete, departures)
            no_sync_incomplete[:] = still
            for item in left:
                print(f"[launcher] run summary: island {item['island']} ({item['name']}) "
                      f"left the pool mid-run (syncer pool_leave reason={item['reason']}); "
                      "no rl_learner_finalized required", file=sys.stderr)
            if left:
                try:
                    from . import runs as _runs
                    _runs.update_run(args.cluster_prefix, islands_left_pool=left)
                except Exception as e:  # noqa: BLE001 - summary is best effort
                    print(f"[launcher] WARN: could not record islands_left_pool ({e})",
                          file=sys.stderr)
        failed = [n for n, s in exit_codes.items() if "SUCCEEDED" not in s]
        if no_sync_incomplete:
            print(
                f"[launcher] event tape incomplete for {no_sync_incomplete}; "
                f"exit {NO_SYNC_INCOMPLETE_EXIT}",
                file=sys.stderr,
            )
            if no_sync:  # nothing else to secure: no syncer checkpoint
                return NO_SYNC_INCOMPLETE_EXIT

        # Secure the artifact BEFORE the finally block tears learners down:
        # fetch ~/yeto-output from the winning learner onto this machine
        # (the head, or the local worker), then deliver to --output.
        done = [n for n, s in exit_codes.items() if "SUCCEEDED" in s]
        if not done:
            print("[launcher] no learner succeeded; recover from the syncer "
                  "checkpoint with yeto-export", file=sys.stderr)
            return 1
        rl_mode = getattr(args, "training_mode", "sft") == "rl"
        # Prefer a sky learner as the artifact source (rsync); a Modal
        # island's ~/yeto-output is not reachable that way.
        sky_done = [n for n in done if n not in modal_cfgs]
        source = (
            syncer_cluster
            if rl_mode and syncer_cluster is not None
            else next((n for n in sky_done if "-l0-" in n), (sky_done or done)[0])
        )
        output = getattr(args, "output", None)
        local_dest = (
            os.path.expanduser(output)
            if output and delivery.kind(output) == "local" and not head_mode
            else os.path.expanduser("~/yeto-output")
        )
        os.makedirs(local_dest, exist_ok=True)
        if rl_mode and head_mode:
            print(f"[launcher] committed RL checkpoint retained at {local_dest}")
        elif source in modal_cfgs and no_sync:
            print(
                f"[launcher] --rl-single-island-no-sync island ran on Modal ({source}); "
                "its ~/yeto-output is not fetchable over ssh and there is no syncer "
                "checkpoint -- use the streamed island log / event tape as the evidence",
                file=sys.stderr,
            )
            # S17 N16: the run itself is judged by the island outcome (its tape
            # was already checked complete above).  Only an explicit --output
            # the launcher cannot honour stays exit 2.
            if output:
                print(f"[launcher] --output {output} cannot be delivered from a "
                      "Modal no-sync island; exit 2", file=sys.stderr)
                return 2
            return 1 if failed else 0
        elif source in modal_cfgs:
            print(
                f"[launcher] every successful learner ran on Modal ({source}); its "
                "~/yeto-output is not fetchable over ssh — recover the model from "
                "the syncer checkpoint with yeto-export",
                file=sys.stderr,
            )
            return 2
        else:
            try:
                subprocess.run(delivery.fetch_cmd(source, local_dest), check=True)
                artifact = "committed RL checkpoint" if rl_mode else "fine-tuned model"
                print(f"[launcher] {artifact} fetched to {local_dest}")
            except subprocess.CalledProcessError as e:
                print(
                    f"[launcher] fetching {source}:~/yeto-output failed ({e}); "
                    "recover from the syncer checkpoint with yeto-export",
                    file=sys.stderr,
                )
                return 2
        if delivery.is_remote(output):
            try:
                delivery.deliver(output, local_dest)
                print(f"[launcher] output uploaded to {output}")
            except Exception as e:
                print(f"[launcher] upload to {output} failed: {e}; the model "
                      f"remains at {local_dest}", file=sys.stderr)
                return 2
        if failed:
            return 1
        # A synced run still fetches/delivers its checkpoint first, then
        # fails closed on an incomplete island tape.
        return NO_SYNC_INCOMPLETE_EXIT if no_sync_incomplete else 0
    finally:
        if verda is not None and verda.get("pth_written"):
            from .sky_patches import remove_local_pth

            remove_local_pth(verda["pth_written"])
        # Clusters the controller already tore down (abandoned learners, or
        # the syncer after a total loss) are skipped — even with --keep.
        downed = controller.downed_clusters if controller is not None else set()
        remaining = [c for c in clusters if c not in downed]
        if modal_ops is not None and modal_cfgs:
            # Containers commit their tape every 30 s and at exit; pulled
            # here (before the app stop) and again after it below.
            pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
        if args.keep:
            print(f"[launcher] keeping clusters: {remaining}")
        else:
            unverified = []
            collect_diagnostics_parallel(
                [n for n in remaining if n not in modal_cfgs], head=head_mode
            )
            for name in remaining:
                print(f"[launcher] tearing down {name}")
                if name in modal_cfgs:
                    try:
                        if modal_island_ops is not None:
                            modal_island_ops.down(name)
                    except Exception as e:  # noqa: BLE001 - app stop below is the backstop
                        print(f"[launcher] cancel of {name} failed: {e}", file=sys.stderr)
                    continue
                if not teardown_island(sky, name, head=head_mode, collect=None,
                                       num_nodes=nodes_by_cluster.get(name, 1)):
                    unverified.append(name)
            if modal_ops is not None:
                # Belt and braces: stop the whole per-run Modal app so no
                # function call of this run outlives the launcher.
                try:
                    modal_ops.stop_app()
                except Exception as e:  # noqa: BLE001
                    print(f"[launcher] Modal app stop failed: {e}", file=sys.stderr)
                if not _verify_modal_app_stopped(modal_ops, args,
                                                 run_started_unix=run_started_unix):
                    teardown_unverified = True
                if modal_cfgs:
                    # The final at-exit commits land once the containers are gone.
                    pull_modal_tapes(modal_ops, modal_cfgs, _modal_tape_run_dir(args))
            if unverified:
                # The head must NOT self-terminate: it is the only thing that
                # can still reach these orphaned learner clusters via sky.
                args._teardown_incomplete = True
                print(
                    f"[launcher] WARNING: could not confirm termination of "
                    f"{unverified}; leaving the head up so they stay reachable. "
                    f"Terminate them from the cloud console, then: yeto down {prefix}",
                    file=sys.stderr,
                )
        if head_mode:
            # The head VM cannot tear itself down here (the teardown would
            # kill this very process mid-flight). With a remote --output the
            # caller (cmd_head) self-terminates via the EC2 API after this
            # returns cleanly; otherwise the head stays up so the fetched
            # model and syncer checkpoint remain reachable.
            head_cluster = sky_cluster_name(f"{prefix}-head")
            if args.keep:
                print(
                    f"[launcher] run finished; clusters left up: "
                    f"{remaining + [head_cluster]}; tear everything down "
                    f"with: yeto down {prefix}",
                    flush=True,
                )
            elif delivery.is_remote(getattr(args, "output", None)):
                print("[launcher] run finished; head will self-terminate "
                      "after delivery", flush=True)
            else:
                print(
                    f"[launcher] run finished; model + checkpoint live on the "
                    f"head — tear it down with: yeto down {prefix}",
                    flush=True,
                )
        if teardown_unverified and sys.exc_info()[0] is None:
            # the run's own outcome is secondary to a possibly still-running app
            return TEARDOWN_UNVERIFIED_EXIT  # noqa: B012


NO_SYNC_EVENT_DRAIN_S = 120.0
# no-sync run whose island tape lacks rl_learner_finalized (differs from 2:
# "artifact not fetchable").
NO_SYNC_INCOMPLETE_EXIT = 3


def _echoes_events(args, spec) -> bool:
    """Whether this island's tape is echoed to its log and rebuilt locally."""

    return (
        getattr(args, "training_mode", "sft") == "rl"
        and getattr(args, "rl_engine", "ports") == "ports"
    )  # every ports RL island (sky too): rl_learner_finalized tells the launcher
    # that a later non-zero exit is a shutdown error, not an island failure


def _miles_overlay_manifest(args, engine: str) -> dict:
    """``miles_overlay`` for the run manifest; absent (old manifests unchanged) when off."""

    if engine != "ports":
        return {}
    from .rl.miles_overlay import overlay_record, resolve_overlay

    record = overlay_record(resolve_overlay(args))
    return {"miles_overlay": record} if record else {}


def _write_run_manifest(args) -> dict | None:
    """Record the engine pins actually used (launch.log + <run dir>/run_manifest.json)."""

    if getattr(args, "training_mode", "sft") != "rl":
        return None
    from . import rl as _rl
    from . import runs

    engine = getattr(args, "rl_engine", "ports")
    manifest = {
        "rl_engine": engine,
        "rl_image": getattr(args, "rl_image", None),
        "miles_commit": _rl.MILES_NEXT_COMMIT if engine == "ports" else _rl.MILES_COMMIT,
        "sglang_commit": getattr(_rl, "SGLANG_NEXT_COMMIT", None) if engine == "ports" else None,
        "source_sha256": getattr(args, "source_sha256", None),
        "cluster_prefix": args.cluster_prefix,
        **_miles_overlay_manifest(args, engine),
        "written_unix": time.time(),
    }
    print(f"[launcher] RL engine {engine}: image {manifest['rl_image']}, "
          f"miles {manifest['miles_commit']}, sglang {manifest['sglang_commit']}", flush=True)
    if manifest.get("miles_overlay"):
        print(f"[launcher] Miles code = {manifest['miles_overlay']['summary']} "
              "(image manifest does NOT match the running Miles code)", flush=True)
    try:
        path = runs.run_dir(args.cluster_prefix) / "run_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as e:
        print(f"[launcher] could not write the run manifest: {e}", file=sys.stderr)
    return manifest


def _dashboard_fleet_log(args, tasks, results):
    """fleet-dashboard 2.3: the run's ``fleet.jsonl`` writer, or None if it
    cannot be built (never blocks a launch)."""
    try:
        from . import runs
        from .dashboard.fleet import from_env

        return from_env(runs.run_dir(args.cluster_prefix) / "fleet.jsonl",
                        {name: tasks[name] for name in results})
    except Exception as e:  # noqa: BLE001
        print(f"[launcher] fleet.jsonl disabled: {e}", file=sys.stderr)
        return None


def _no_sync_events_dir(args) -> Path:
    """``<run dir>/events`` (yeto run registry), for --rl-single-island-no-sync."""

    from . import runs

    return runs.run_dir(args.cluster_prefix) / "events"


def dry_run_plan(args) -> dict:
    """``yeto launch --dry-run``: what a launch would request, creating nothing.

    Runs on prepared args (``prepare_launch_args`` already validated the
    request, including the ports algorithm/capability checks) and builds the
    island tasks in memory only to read the learner command; no sky/Modal
    call is made.
    """

    no_sync = bool(getattr(args, "rl_single_island_no_sync", False))
    rl = getattr(args, "training_mode", "sft") == "rl"
    if no_sync and not rl:
        raise ValueError("--rl-single-island-no-sync requires --training-mode rl")
    head = getattr(args, "controller", "local") == "head"
    if no_sync and head:
        raise ValueError("--rl-single-island-no-sync has no syncer; use --controller local")
    specs = parse_gpu_spec(args.gpu)
    external = max(0, getattr(args, "external_learners", 0) or 0)
    islands = []
    for learner_id, spec in enumerate(specs):
        entry = {
            "learner_id": learner_id,
            "cloud": spec.cloud,
            "region": spec.region,
            "gpu": spec.gpu,
            "num_nodes": spec.num_nodes,
            "gpus_per_node": spec.gpus_per_node,
            "total_gpus": spec.total_gpus,
        }
        if rl and rl_island_spec(args, spec) is not spec:
            entry["allocated_gpus_per_node"] = rl_island_spec(args, spec).gpus_per_node
        if rl:
            task = make_miles_island_task(
                args, spec, learner_id, len(specs) + external, "$SYNCER_ADDR"
            )
            entry["learner_command"] = next(
                (line.strip() for line in task.run.splitlines() if _rl_backend_module(args, "launch_flags").ISLAND_ENTRY_MODULE in line),
                None,
            )
        islands.append(entry)
    return {
        "dry_run": True,
        "training_mode": getattr(args, "training_mode", "sft"),
        "rl_engine": getattr(args, "rl_engine", None) if rl else None,
        "controller": "head" if head else "local",
        "islands": len(specs) + external,
        "external_learners": external,
        "total_gpus": sum(s.total_gpus for s in specs),
        "syncer": None if no_sync else ("head VM" if head else f"{args.cluster_prefix}-syncer"),
        # 4.2.1: present only for a critic algorithm, so every critic-free plan is unchanged
        **({"critic_syncer": {
            "port": CRITIC_SYNCER_PORT,
            "address": "$CRITIC_SYNCER_ADDR",
            "layout": "critic_layout_hash",
            "command": critic_syncer_command(args, len(specs) + external),
            "checkpoint": RL_CRITIC_SYNCER_CHECKPOINT,
            "event_tape": RL_CRITIC_SYNCER_EVENT_TAPE,
        }} if rl and not no_sync and rl_needs_critic(args) else {}),
        "outer_sync": not no_sync,
        "algorithm_spec_sha256": getattr(args, "rl_expected_algorithm_sha256", None),
        "algorithm_spec": getattr(args, "rl_algorithm_spec_json", None),
        "unverified_mechanisms": list(getattr(args, "rl_allow_unverified_mechanism", None) or ()),
        "clusters": [] if not islands else learner_cluster_names(args.cluster_prefix, specs),
        "island_requests": islands,
    }
