"""SkyPilot orchestration: one syncer VM + one cluster per learner.

Flow:
  1. build the Rust syncer locally (release) — the binary is file-mounted to
     a cheap CPU VM whose TCP port is opened to the learners;
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
from .models import MODEL_WEIGHT_GB  # single source; see yeto/models.py


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
        f" --delta-correction {args.delta_correction}"
        f" --total-steps {args.total_steps}"
        f" --outer-lr {args.outer_lr}"
        f" --outer-momentum {args.outer_momentum}"
        f" --checkpoint-path ~/yeto-state.ckpt --resume"
        f" --event-tape ~/yeto-tape.jsonl"
    )


def make_syncer_task(args, num_learners: int):
    import sky

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


def resolve_loss_function(loss_function) -> str:
    """Return the --loss-function string to pass to learners.

    A callable or a ``custom:<file.py>`` spec is loaded here (failing fast
    before any cloud spend), pickled by value into the workdir, and shipped
    to learners as ``pickle:.yeto_loss.pkl``. Named losses pass through.
    """
    from .losses import dump_pickled_loss, load_custom_loss

    if callable(loss_function):
        fn = loss_function
    elif isinstance(loss_function, str) and loss_function.startswith("custom:"):
        fn = load_custom_loss(loss_function)
    else:
        return loss_function
    dump_pickled_loss(fn, REPO_ROOT / PICKLED_LOSS_FILE)
    return f"pickle:{PICKLED_LOSS_FILE}"


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
# which never binds to SM100 silicon. Extend here as new GPU generations
# outpace provider image pins; an explicit --learner-image always wins,
# and a resolver returning None degrades to the setup-time driver install.
GPU_IMAGE_OVERRIDES: dict[tuple[str, str], object] = {
    ("aws", "B200"): resolve_blackwell_image,
}


def learner_image_for(args, spec: ClusterSpec):
    """The image for a learner cluster: explicit flag > internal override
    table > None (provider default + setup-time remediation)."""
    explicit = parse_image_spec(getattr(args, "learner_image", None))
    if explicit is not None:
        return explicit
    resolver = GPU_IMAGE_OVERRIDES.get((spec.cloud, spec.gpu))
    if resolver is None or not spec.region:
        return None
    image = resolver(spec.region)
    if image:
        print(f"[launcher] {spec.gpu} learner: pinning image {image} ({spec.region})")
    return image


def parse_image_spec(value: str | None):
    """--learner-image: a single image id/tag applied everywhere, or
    comma-separated region=id pairs -> the region dict sky expects."""
    if not value:
        return None
    if "=" not in value:
        return value
    images = {}
    for pair in value.split(","):
        region, _, image = pair.partition("=")
        if not region or not image:
            raise ValueError(
                f"bad --learner-image entry {pair!r}; expected region=image-id"
            )
        images[region.strip()] = image.strip()
    return images


def make_learner_task(args, spec: ClusterSpec, learner_id: int, num_learners: int, syncer_addr: str):
    import sky

    from .datasource import learner_data_arg, learner_file_mounts

    learner_flags = (
        f" --model {shlex.quote(args.model)}"
        f" --data {shlex.quote(learner_data_arg(args.data))}"
        f" --syncer $SYNCER_ADDR"
        f" --learner-id $LEARNER_ID"
        f" --num-learners {num_learners}"
        f" --loss-function {args.loss_function}"
        f" --train-on {args.train_on}"
        f" --shard {args.shard}"
        f" --tuning {args.tuning}"
        f" --lora-r {args.lora_r}"
        f" --lora-targets {getattr(args, 'lora_targets', 'auto')}"
        f" --seq-len {args.seq_len}"
        f" --micro-batch-size {args.micro_batch_size}"
        f" --grad-accum {args.grad_accum}"
        f" --inner-lr {args.inner_lr}"
        f" --fragments {args.fragments}"
        f" --fragment-pattern {args.fragment_pattern}"
        f" --merge-alpha {args.merge_alpha}"
        f" --tokenize {args.tokenize}"
        f" --stream-workers {args.stream_workers}"
        f" --wire-dtype {args.wire_dtype}"
        f" --wan-streams {args.wan_streams}"
        f" --output-dir ~/yeto-output"
    )
    if args.max_rows:
        learner_flags += f" --max-rows {args.max_rows}"
    run = (
        f"{NVME_ENV}\n"
        f"{HF_TOKEN_ENV}\n"
        'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
        "torchrun --nnodes=$SKYPILOT_NUM_NODES --node_rank=$SKYPILOT_NODE_RANK "
        "--nproc_per_node=$SKYPILOT_NUM_GPUS_PER_NODE "
        "--master_addr=$MASTER_ADDR --master_port=29500 "
        f"-m yeto.learner{learner_flags}"
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
        file_mounts[f"~/sky_workdir/{PICKLED_LOSS_FILE}"] = str(REPO_ROOT / PICKLED_LOSS_FILE)
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
    prefetch = (
        f"(nohup huggingface-cli download {shlex.quote(repo)} "
        ">/tmp/hf-prefetch.log 2>&1 &) || true"
    )
    task = sky.Task(
        name=f"yeto-learner-{learner_id}",
        setup="\n".join(
            [
                WAN_TUNING,
                NVME_SETUP,
                NVME_ENV,
                HF_TOKEN_ENV,
                TORCH_SETUP,
                "pip install -q -r requirements.txt",
                prefetch,
            ]
        ),
        run=run,
        envs=envs,
        num_nodes=spec.num_nodes,
        workdir=str(REPO_ROOT),
        file_mounts=file_mounts,
    )
    infra = f"{spec.cloud}/{spec.region}" if spec.region else spec.cloud
    resources_kwargs = {}
    image = learner_image_for(args, spec)
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

        sky.get(sky.down(cluster))

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

    head_mode = local_syncer is not None
    specs = parse_gpu_spec(args.gpu)
    num_learners = len(specs)
    args.loss_function = resolve_loss_function(args.loss_function)
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
            for name in remaining:
                print(f"[launcher] tearing down {name}")
                try:
                    sky.get(sky.down(name))
                except Exception as e:
                    print(f"[launcher] teardown of {name} failed: {e}", file=sys.stderr)
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
