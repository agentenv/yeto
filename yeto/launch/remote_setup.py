"""Shell snippets that run on a launched node during setup or before its run."""

from __future__ import annotations

from pathlib import Path


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
HUNYUAN3D_ROOT = "~/Hunyuan3D-2.1"
HUNYUAN3D_SETUP = (
    f"if [ ! -d {HUNYUAN3D_ROOT} ]; then git clone --depth 1 "
    f"https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git {HUNYUAN3D_ROOT}; fi && "
    f"pip install -q -r {HUNYUAN3D_ROOT}/requirements.txt && "
    f"export YETO_HUNYUAN3D_ROOT={HUNYUAN3D_ROOT} "
    "|| echo '[yeto-setup] Hunyuan3D setup failed; Hunyuan3D adapter unavailable' >&2"
)

def causal_kernel_setup_steps(args) -> list[str]:
    """Pinned remote installs selected explicitly for a causal torch learner."""
    from ..kernel_deps import (
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
