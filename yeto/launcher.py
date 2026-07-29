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

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import delivery
from .gpu_spec import ClusterSpec, parse_gpu_spec
from .models import MODEL_WEIGHT_GB

SYNCER_PORT = 29400
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
GPU_MEM_GB = {"A100": 40, "A100-80GB": 80, "H100": 80, "H200": 141, "B200": 180, "L4": 24, "A10G": 24, "T4": 16, "V100": 16, "L40S": 48}


def build_syncer_binary() -> Path:
    binary = REPO_ROOT / "syncer/target/release/yeto-syncer"
    print("[launcher] building syncer (cargo build --release)...")
    subprocess.run(["cargo", "build", "--release"], cwd=REPO_ROOT / "syncer", check=True)
    return binary


def syncer_command(args, num_learners: int, binary: str = "~/yeto-syncer") -> str:
    """The syncer invocation shared by the syncer-cluster task (local
    controller mode) and the head-node subprocess (head controller mode).
    --resume makes any restart pick up from the on-disk checkpoint."""
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
        f" --checkpoint-path ~/yeto-state.ckpt --resume"
        f" --mark-final-checkpoint"
        f" --event-tape ~/yeto-tape.jsonl"
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


def make_syncer_task(args, num_learners: int):
    import platform

    import sky

    cross = (platform.system(), platform.machine()) != ("Linux", "x86_64")
    if cross:
        print("[launcher] non-x86-Linux submitter: building the syncer on the syncer VM")
        task = sky.Task(
            name="yeto-syncer",
            setup=WAN_TUNING + "\n" + SYNCER_REMOTE_BUILD,
            run=syncer_command(args, num_learners),
            workdir=str(REPO_ROOT),
        )
        infra = args.syncer_region if "/" in args.syncer_region else f"aws/{args.syncer_region}"
        task.set_resources(
            sky.Resources(
                infra=infra,
                cpus="8+",
                memory=f"{args.syncer_memory}+",
                ports=[SYNCER_PORT],
                use_spot=False,
            )
        )
        return task

    binary = build_syncer_binary()
    cmd = "chmod +x ~/yeto-syncer && " + syncer_command(args, num_learners)
    task = sky.Task(
        name="yeto-syncer",
        setup=WAN_TUNING,
        run=cmd,
        file_mounts={"~/yeto-syncer": str(binary)},
    )
    # --syncer-region accepts "region" (AWS assumed) or "cloud/region".
    infra = args.syncer_region if "/" in args.syncer_region else f"aws/{args.syncer_region}"
    task.set_resources(
        sky.Resources(
            infra=infra,
            cpus="8+",
            memory=f"{args.syncer_memory}+",
            ports=[SYNCER_PORT],
            use_spot=False,
        )
    )
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


def prepare_launch_args(args) -> None:
    """Resolve immutable inputs and executable artifacts before cloud spend."""

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
    payload = pin_runtime_provenance(args)
    args.model_requested_identifier = payload["model"]["requested_identifier"]
    args.model_requested_revision = payload["model"]["requested_revision"]
    if "dataset" in payload:
        args.data_requested_identifier = payload["dataset"]["requested_identifier"]
        args.data_requested_revision = payload["dataset"]["requested_revision"]
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


def causal_kernel_setup_steps(args) -> list[str]:
    """Pinned remote installs selected explicitly for a causal torch learner."""
    from .kernel_deps import (
        FLASH_ATTN_VERSION,
        LIGER_KERNEL_VERSION,
        NINJA_VERSION,
        PACKAGING_VERSION,
        PEFT_VERSION,
    )

    steps: list[str] = []
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


def make_learner_task(args, spec: ClusterSpec, learner_id: int, num_learners: int, syncer_addr: str):
    import sky

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
        f" --lora-targets {getattr(args, 'lora_targets', 'auto')}"
        f" --micro-batch-size {args.micro_batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --inner-lr {args.inner_lr}"
        f" --fragments {args.fragments}"
        f" --fragment-pattern {args.fragment_pattern}"
        f" --merge-alpha {args.merge_alpha}"
        f" --wire-dtype {args.wire_dtype}"
        f" --wan-streams {args.wan_streams}"
        f" --output-dir ~/yeto-output"
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
        if getattr(args, "diffusion_adapter", None):
            learner_flags += f" --diffusion-adapter {shlex.quote(args.diffusion_adapter)}"
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
    if spec.num_nodes > 1:
        # Surface NCCL's chosen transport in the job logs so an EFA-less
        # fallback to TCP sockets is visible, not silent.
        envs["NCCL_DEBUG"] = "INFO"
    # Non-HF --data sources (local paths, s3://, gs://, ...) ride sky's
    # file_mounts onto every learner; see yeto/datasource.py.
    file_mounts = dict(learner_file_mounts(args.data)) or None
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

    repo = resolve(args.model)
    from .provenance import is_local_reference

    if is_local_reference(repo):
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
        resources_kwargs["network_tier"] = "best"
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


def learner_cluster_names(prefix: str, specs: list[ClusterSpec]) -> list[str]:
    """Deterministic learner cluster names for a run: computable from the
    launch args alone, so the CLI can record them before provisioning."""
    return [f"{prefix}-l{m}-{spec.region or spec.cloud}" for m, spec in enumerate(specs)]


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


def _tail(cluster: str, job_id: int, prefix: str) -> int:
    import sky

    while True:
        try:
            it = sky.tail_logs(cluster, job_id, follow=True, preload_content=False)
            for line in it:
                if line is None:
                    break
                print(f"[{prefix}] {line.rstrip()}", flush=True)
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

    def down(self, cluster: str) -> None:
        import sky

        terminate_and_verify(sky, cluster)

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
        self.proc: subprocess.Popen | None = None
        # The log file persists across controller jobs on a reused head;
        # forward only what this controller's syncer writes, not history.
        self._log_offset = os.path.getsize(self.log_file) if os.path.exists(self.log_file) else 0

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

    def probe(self) -> str | None:
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
        return f"syncer subprocess exited with code {code}"

    def restart(self) -> None:
        self.start()

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self._signal_tree(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_tree(signal.SIGKILL)

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


class _RelaunchAttempt:
    """Result slot for one background relaunch attempt."""

    def __init__(self):
        self.result = None  # new job id, or None if provisioning failed
        self.finished = False


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
    ):
        """`learners` maps cluster name -> (task, job_id); `syncer` is
        (name, task, job_id) for a cluster syncer, or None with
        `syncer_probe`/`syncer_restart` callables for a subprocess syncer.
        `on_relaunch(name, new_job_id)` is called after every successful
        cluster relaunch (production spawns a new log tail)."""
        self.ops = sky_ops
        self.poll_interval = poll_interval
        self.recover_timeout = recover_timeout
        self.on_relaunch = on_relaunch
        self.thread_cls = thread_cls
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
        else:
            syncer_name, syncer_task, syncer_job = syncer
            self.syncer = self._make_record(syncer_name, syncer_task, syncer_job)
            self.syncer_probe = self.syncer_restart = None
        self.downed_clusters: set = set()

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
        while True:
            if self.syncer is not None:
                self._poll(self.syncer, is_syncer=True)
            else:
                self._poll_local_syncer()
            for rec in self.learners.values():
                self._poll(rec, is_syncer=False)
            if all(r["state"] in (DONE, ABANDONED) for r in self.learners.values()):
                break
            self.ops.sleep(self.poll_interval)
        exit_codes = {name: rec["exit"] for name, rec in self.learners.items()}
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
                return  # healthy
            if verdict == "succeeded":
                rec["state"] = DONE
                rec["exit"] = str(status)
                print(f"[launcher] {rec['name']} job finished: {status}")
            else:
                self._enter_recovering(rec, verdict, is_syncer)
        elif rec["state"] == RECOVERING:
            self._drive_recovery(rec, is_syncer)

    def _probe(self, rec):
        """Classify a running cluster: (None, status) if healthy,
        ("succeeded", status), or (failure reason, status)."""
        name, job_id = rec["name"], rec["job_id"]
        try:
            status = self.ops.job_status(name, job_id)
        except Exception as e:
            return f"job status unavailable ({e})", None
        if status is not None and status.is_terminal():
            if "SUCCEEDED" in str(status):
                return "succeeded", status
            return f"job ended as {status}", status
        try:
            up = self.ops.cluster_up(name)
        except Exception as e:
            return f"cluster status unavailable ({e})", status
        if not up:
            return "cluster is not UP (preempted or deleted)", status
        return None, status

    def _enter_recovering(self, rec, reason: str, is_syncer: bool) -> None:
        rec["state"] = RECOVERING
        rec["failed_at"] = self.ops.now()
        print(
            f"[launcher] {rec['name']}: {reason}; starting recovery "
            f"(timeout {self.recover_timeout}s)",
            file=sys.stderr,
        )
        if not is_syncer and self.recover_timeout <= 0:
            self._abandon(rec, 0.0)
            return
        self._drive_recovery(rec, is_syncer)

    def _drive_recovery(self, rec, is_syncer: bool) -> None:
        attempt = rec["attempt"]
        if attempt is not None and attempt.finished:
            rec["attempt"] = None
            if attempt.result is not None:
                rec["job_id"] = attempt.result
                rec["state"] = RUNNING
                rec["failed_at"] = None
                print(
                    f"[launcher] {rec['name']} recovered: relaunched as job "
                    f"{attempt.result}"
                )
                if self.on_relaunch is not None:
                    self.on_relaunch(rec["name"], attempt.result)
                return
            print(
                f"[launcher] relaunch attempt for {rec['name']} failed; will retry",
                file=sys.stderr,
            )
        elapsed = self.ops.now() - rec["failed_at"]
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
            rec["attempt"] = self._start_relaunch(rec)

    def _start_relaunch(self, rec) -> _RelaunchAttempt:
        attempt = _RelaunchAttempt()
        name, task = rec["name"], rec["task"]

        def _run():
            try:
                attempt.result = self.ops.relaunch(task, name)
            except Exception as e:
                print(f"[launcher] relaunch of {name} raised: {e}", file=sys.stderr)
                attempt.result = None
            finally:
                attempt.finished = True
            if attempt.result is not None and rec["state"] == ABANDONED:
                # Abandoned while this attempt was in flight, but the
                # relaunch re-provisioned the cluster anyway: tear it back
                # down so nothing is left running unattended.
                self._down(name, force=True)

        thread = self.thread_cls(target=_run, daemon=True)
        thread.start()
        return attempt

    def _abandon(self, rec, elapsed: float) -> None:
        rec["state"] = ABANDONED
        rec["exit"] = f"ABANDONED after {elapsed:.0f}s"
        self._down(rec["name"])
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
        if cloud is None or cloud.STATUS_VERSION < clouds.StatusVersion.SKYPILOT:
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


def terminate_and_verify(sky, cluster, *, probe="auto", attempts=4, sleep_fn=time.sleep) -> bool:
    """sky.down a cluster and CONFIRM at the cloud level that no instance
    survives, retrying the down while the cloud still reports live ones.

    sky.down has been observed to report success while a spot instance
    lingers (state DB and cloud diverge). In head-controller mode the head's
    sky is the ONLY thing that can reach the learner clusters, so a silent
    orphan is unrecoverable once the head is gone — hence verify here, before
    the head relinquishes control. Returns True iff the cluster is confirmed
    gone, or can't be cloud-verified (then we trust sky.down).
    """
    if probe == "auto":
        probe = _cloud_live_instances_probe(cluster)

    def _down():
        try:
            sky.get(sky.down(cluster))
        except Exception as e:
            print(f"[launcher] sky.down({cluster}) error: {e}", file=sys.stderr)

    _down()
    if probe is None:
        return True
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


def run(args, on_clusters=None, local_syncer=None) -> int:
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
    """
    import sky

    prepare_launch_args(args)
    head_mode = local_syncer is not None
    specs = parse_gpu_spec(args.gpu)
    # External learners (machines sky cannot provision — e.g. Macs running
    # yeto.mlx.learner) get the ids AFTER the cloud learners; the syncer
    # counts them in --learners and its port is already public, so they
    # simply dial in with the printed join command.
    external = max(0, getattr(args, "external_learners", 0) or 0)
    num_learners = len(specs) + external
    warn_if_model_wont_fit(args, specs)
    prefix = args.cluster_prefix
    syncer_cluster = None if head_mode else f"{prefix}-syncer"
    learner_names = learner_cluster_names(prefix, specs)
    if on_clusters is not None:
        try:
            on_clusters(([] if head_mode else [syncer_cluster]) + learner_names)
        except Exception as e:
            print(f"[launcher] on_clusters hook failed: {e}", file=sys.stderr)
    clusters: list[str] = []
    controller = None

    try:
        # 1. Syncer: a subprocess on this host (head mode) or its own VM.
        if head_mode:
            syncer_task = syncer_job = None
            syncer_addr = f"{os.environ['SYNCER_PUBLIC_IP']}:{SYNCER_PORT}"
            print(f"[launcher] syncer runs on this head node at {syncer_addr}")
        else:
            print(f"[launcher] launching syncer cluster {syncer_cluster} in {args.syncer_region}")
            syncer_task = make_syncer_task(args, num_learners)
            rid = sky.launch(syncer_task, cluster_name=syncer_cluster)
            syncer_job, syncer_handle = sky.stream_and_get(rid)
            clusters.append(syncer_cluster)
            syncer_addr = f"{syncer_handle.head_ip}:{SYNCER_PORT}"
            print(f"[launcher] syncer up at {syncer_addr}")

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

        # 2. Learners, in parallel.
        tasks = {}
        rids = {}
        for m, spec in enumerate(specs):
            name = learner_names[m]
            task = make_learner_task(args, spec, m, num_learners, syncer_addr)
            tasks[name] = task
            print(f"[launcher] launching learner {m} on {spec} as {name}")
            rids[name] = (
                m,
                sky.launch(task, cluster_name=name, retry_until_up=args.retry_until_up),
            )

        results = {}
        errors = {}

        def resolve(name: str, m: int, rid) -> None:
            try:
                results[name] = sky.stream_and_get(rid)
            except Exception as e:
                errors[name] = e

        threads = [
            threading.Thread(target=resolve, args=(n, m, r), daemon=True)
            for n, (m, r) in rids.items()
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for name in rids:
            if name not in errors:
                clusters.append(name)
        if errors:
            for name, e in errors.items():
                print(f"[launcher] ERROR launching {name}: {e}", file=sys.stderr)
            raise RuntimeError(f"{len(errors)} learner cluster(s) failed to provision")

        # 3. Stream logs while the fleet controller polls health, recovers
        #    failed/preempted clusters, and abandons learners that stay down
        #    past --recover-timeout.
        def spawn_tail(name: str, job_id: int) -> None:
            label = "syncer" if name == syncer_cluster else name
            threading.Thread(target=_tail, args=(name, job_id, label), daemon=True).start()

        if not head_mode:
            spawn_tail(syncer_cluster, syncer_job)
        for name, (job_id, _handle) in results.items():
            spawn_tail(name, job_id)

        controller = FleetController(
            learners={name: (tasks[name], job_id) for name, (job_id, _h) in results.items()},
            syncer=None if head_mode else (syncer_cluster, syncer_task, syncer_job),
            sky_ops=SkySDKOps(),
            poll_interval=args.controller_poll,
            recover_timeout=args.recover_timeout,
            on_relaunch=spawn_tail,
            syncer_probe=local_syncer.probe if head_mode else None,
            syncer_restart=local_syncer.restart if head_mode else None,
        )
        exit_codes = controller.run()
        failed = [n for n, s in exit_codes.items() if "SUCCEEDED" not in s]

        # Secure the artifact BEFORE the finally block tears learners down:
        # fetch ~/yeto-output from the winning learner onto this machine
        # (the head, or the local worker), then deliver to --output.
        done = [n for n, s in exit_codes.items() if "SUCCEEDED" in s]
        if not done:
            print("[launcher] no learner succeeded; recover from the syncer "
                  "checkpoint with yeto-export", file=sys.stderr)
            return 1
        source = next((n for n in done if "-l0-" in n), done[0])
        output = getattr(args, "output", None)
        local_dest = (
            os.path.expanduser(output)
            if output and delivery.kind(output) == "local" and not head_mode
            else os.path.expanduser("~/yeto-output")
        )
        os.makedirs(local_dest, exist_ok=True)
        try:
            subprocess.run(delivery.fetch_cmd(source, local_dest), check=True)
            print(f"[launcher] fine-tuned model fetched to {local_dest}")
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
        return 1 if failed else 0
    finally:
        # Clusters the controller already tore down (abandoned learners, or
        # the syncer after a total loss) are skipped — even with --keep.
        downed = controller.downed_clusters if controller is not None else set()
        remaining = [c for c in clusters if c not in downed]
        if args.keep:
            print(f"[launcher] keeping clusters: {remaining}")
        else:
            unverified = []
            for name in remaining:
                print(f"[launcher] tearing down {name}")
                if not terminate_and_verify(sky, name):
                    unverified.append(name)
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
            head_cluster = f"{prefix}-head"
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
