#!/usr/bin/env bash
# Prepare a deterministic tiny NAVA smoke run bundle and S3 label subset.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

write_env_file() {
  cat > "${RUN_DIR}/env.sh" <<ENV
# Source this file before the run scripts if you want these defaults.
: "\${RUN_DIR:=${RUN_DIR}}"; export RUN_DIR
: "\${NAVA_ROOT:=${NAVA_ROOT}}"; export NAVA_ROOT
: "\${NAVA_SMOKE_BACKEND:=${NAVA_SMOKE_BACKEND}}"; export NAVA_SMOKE_BACKEND
: "\${NAVA_ASSETS_DIR:=${NAVA_ASSETS_DIR}}"; export NAVA_ASSETS_DIR
: "\${LABEL_INPUT:=${LABEL_INPUT}}"; export LABEL_INPUT
: "\${NAVA_S3_REGION:=${NAVA_S3_REGION}}"; export NAVA_S3_REGION
: "\${NUM_LEARNERS:=${NUM_LEARNERS}}"; export NUM_LEARNERS
: "\${SYNC_PORT:=${SYNC_PORT}}"; export SYNC_PORT
: "\${SYNC_QUORUM:=${SYNC_QUORUM}}"; export SYNC_QUORUM
: "\${TOTAL_STEPS:=${TOTAL_STEPS}}"; export TOTAL_STEPS
: "\${FRAGMENTS:=${FRAGMENTS}}"; export FRAGMENTS
: "\${NAVA_GPUS_PER_NODE:=${NAVA_GPUS_PER_NODE}}"; export NAVA_GPUS_PER_NODE
: "\${NAVA_BATCH_SIZE:=${NAVA_BATCH_SIZE}}"; export NAVA_BATCH_SIZE
: "\${NAVA_GRAD_ACCUM:=${NAVA_GRAD_ACCUM}}"; export NAVA_GRAD_ACCUM
: "\${NAVA_LOCAL_STEPS:=${NAVA_LOCAL_STEPS}}"; export NAVA_LOCAL_STEPS
: "\${NAVA_SAVE_EVERY:=${NAVA_SAVE_EVERY}}"; export NAVA_SAVE_EVERY
: "\${NAVA_LR:=${NAVA_LR}}"; export NAVA_LR
: "\${NAVA_WEIGHT_DECAY:=${NAVA_WEIGHT_DECAY}}"; export NAVA_WEIGHT_DECAY
: "\${NAVA_WARMUP_STEPS:=${NAVA_WARMUP_STEPS}}"; export NAVA_WARMUP_STEPS
: "\${NAVA_NUM_WORKERS:=${NAVA_NUM_WORKERS}}"; export NAVA_NUM_WORKERS
: "\${NAVA_IO_WORKERS:=${NAVA_IO_WORKERS}}"; export NAVA_IO_WORKERS
: "\${MAX_OUTPUT_ROWS:=${MAX_OUTPUT_ROWS}}"; export MAX_OUTPUT_ROWS
: "\${MIN_DURATION:=${MIN_DURATION}}"; export MIN_DURATION
: "\${MAX_DURATION:=${MAX_DURATION}}"; export MAX_DURATION
: "\${NAVA_MODALITY:=${NAVA_MODALITY}}"; export NAVA_MODALITY
: "\${WIRE_DTYPE:=${WIRE_DTYPE}}"; export WIRE_DTYPE
: "\${WAN_STREAMS:=${WAN_STREAMS}}"; export WAN_STREAMS
: "\${LORA_R:=${LORA_R}}"; export LORA_R
: "\${LORA_ALPHA:=${LORA_ALPHA}}"; export LORA_ALPHA
: "\${LORA_TARGETS:=${LORA_TARGETS}}"; export LORA_TARGETS
: "\${SMOKE_LATENT_DIM:=${SMOKE_LATENT_DIM}}"; export SMOKE_LATENT_DIM
: "\${SMOKE_SEQ_LEN:=${SMOKE_SEQ_LEN}}"; export SMOKE_SEQ_LEN
ENV
}

check_python_deps() {
  python3 - <<'PY'
import importlib.util
missing = [m for m in ("yaml", "torch") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing Python modules: " + ", ".join(missing))
PY
}

write_mock_nava_root() {
  if [[ "${NAVA_SMOKE_BACKEND}" != "mock" ]]; then
    return
  fi
  mkdir -p "${NAVA_ROOT}/nava_src/data" "${NAVA_ROOT}/nava_src/utils"
  touch "${NAVA_ROOT}/nava_src/__init__.py" "${NAVA_ROOT}/nava_src/data/__init__.py" "${NAVA_ROOT}/nava_src/utils/__init__.py"

  cat > "${NAVA_ROOT}/nava_src/pipeline_smoke.py" <<'PY'
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmokeBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.cross_attn = nn.Linear(dim, dim)
        self.ffn = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.self_attn(x) + self.cross_attn(x)
        return self.norm(x + self.ffn(F.silu(h)))


class SmokeBackbone(nn.Module):
    def __init__(self, dim, double_layers, single_layers):
        super().__init__()
        self.double_blocks = nn.ModuleList([SmokeBlock(dim) for _ in range(double_layers)])
        self.single_blocks = nn.ModuleList([SmokeBlock(dim) for _ in range(single_layers)])
        self.double_final_blocks = nn.ModuleList([SmokeBlock(dim)])

    def forward(self, x):
        for block in self.double_blocks:
            x = block(x)
        for block in self.single_blocks:
            x = block(x)
        for block in self.double_final_blocks:
            x = block(x)
        return x


class SmokeModel(nn.Module):
    def __init__(self, dim=128, double_layers=1, single_layers=1):
        super().__init__()
        self.backbone = SmokeBackbone(dim, double_layers, single_layers)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        return self.out(self.backbone(x))


class SmokePipeline:
    """Small NAVA-compatible smoke pipeline with no text encoder or VAE."""

    def __init__(self, model, dim):
        self.model = model
        self.dim = dim
        self.audio_vae = None
        self.video_vae = None
        self.image_vae = None

    @classmethod
    def create(cls, model_id="", use_bf16=True, audio_latent_ch=20, video_latent_ch=48,
               lambda_ddpm=5.0, cfg=None, device="cpu"):
        cfg = cfg or {}
        dim = int(cfg.get("smoke_latent_dim", 128))
        model = SmokeModel(
            dim=dim,
            double_layers=int(cfg.get("smoke_double_layers", 1)),
            single_layers=int(cfg.get("smoke_single_layers", 1)),
        )
        return cls(model, dim)

    def switch_training_mode(self):
        return None

    def forward(self, batch, global_step=None):
        param = next(self.model.parameters())
        vals = batch.get("video_latents") or batch.get("audio_latents") or batch.get("image_latents")
        xs = []
        if vals is not None:
            for value in vals:
                if value is None:
                    continue
                tensor = value.to(device=param.device, dtype=param.dtype).reshape(-1, self.dim)
                xs.append(tensor.mean(dim=0))
        if not xs:
            xs.append(torch.zeros(self.dim, device=param.device, dtype=param.dtype))
        x = torch.stack(xs, dim=0)
        pred = self.model(x)
        target = torch.tanh(x.float()).to(pred.dtype)
        loss = F.mse_loss(pred.float(), target.float())
        return loss, {"loss": float(loss.detach().cpu())}
PY

  cat > "${NAVA_ROOT}/nava_src/data/dataset_train.py" <<'PY'
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import IterableDataset


@dataclass
class DistInfo:
    world_rank: int = 0
    world_size: int = 1


class AudioVideoDataset(IterableDataset):
    """Synthetic latent dataset for Yeto smoke tests.

    It reads the NAVA JSONL manifest for sharding/resume coverage, but never
    opens media files and never calls a text encoder or VAE.
    """

    is_cycle = True

    def __init__(self, jsonl_or_src_list, batch_size=1, src_id2ratios=None, dist_info=None,
                 video_tgt_frames=8, **kwargs):
        super().__init__()
        self.batch_size = int(batch_size)
        self.dist_info = dist_info or DistInfo()
        self.dim = int(kwargs.get("smoke_latent_dim") or 128)
        self.seq_len = int(kwargs.get("smoke_seq_len") or max(2, video_tgt_frames))
        paths = []
        items = jsonl_or_src_list if isinstance(jsonl_or_src_list, list) else [["default", jsonl_or_src_list]]
        for item in items:
            paths.append(item[-1])
        self.records = []
        for path in paths:
            with open(Path(path), "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self.records.append(json.loads(line))
        if not self.records:
            raise ValueError("smoke dataset has no records")

    def __iter__(self):
        idx = 0
        rank = int(self.dist_info.world_rank)
        world = max(1, int(self.dist_info.world_size))
        while True:
            batch = []
            while len(batch) < self.batch_size:
                rec = self.records[idx % len(self.records)]
                idx += 1
                if ((idx - 1) % world) != rank:
                    continue
                text_list = rec.get("text_list") or [{"text": "smoke"}]
                text = text_list[0].get("text", "smoke")
                gen = torch.Generator().manual_seed(20260705 + rank * 100000 + idx)
                latents = torch.randn(self.seq_len, self.dim, generator=gen)
                batch.append(
                    {
                        "captions": text,
                        "video_latents": latents,
                        "audio_latents": None,
                        "image_latents": None,
                        "spk_embs": [],
                        "data_state": torch.tensor([rank, idx], dtype=torch.long),
                    }
                )
            yield batch


def collate_fn(batch):
    return {
        "captions": [b.get("captions", "smoke") for b in batch],
        "video_latents": [b.get("video_latents") for b in batch],
        "audio_latents": None,
        "image_latents": None,
        "spk_embs": [[] for _ in batch],
        "data_state": torch.stack([b["data_state"] for b in batch], dim=0),
        "t_h_w_list": [(b["video_latents"].shape[0], 1, 1) for b in batch],
    }


def collate_fn_batch(batchs):
    return [collate_fn(batch) for batch in batchs]
PY

  cat > "${NAVA_ROOT}/nava_src/utils/scheduler.py" <<'PY'
import math


class WarmupCosineAnnealingLR:
    def __init__(self, optimizer, warmup_steps=0, max_steps=1, eta_min=0.0):
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.max_steps = max(1, int(max_steps))
        self.eta_min = float(eta_min)
        self.step_count = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.step_count += 1
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            if self.warmup_steps and self.step_count <= self.warmup_steps:
                lr = base_lr * self.step_count / self.warmup_steps
            else:
                progress = min(1.0, self.step_count / self.max_steps)
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + math.cos(math.pi * progress))
            group["lr"] = lr

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.step_count = int(state.get("step_count", 0))
        self.base_lrs = list(state.get("base_lrs", self.base_lrs))
PY
}

write_smoke_config() {
  if [[ "${NAVA_SMOKE_BACKEND}" == "mock" ]]; then
    RUN_DIR="${RUN_DIR}" SMOKE_CONFIG="${SMOKE_CONFIG}" \
    SMOKE_LATENT_DIM="${SMOKE_LATENT_DIM}" SMOKE_SEQ_LEN="${SMOKE_SEQ_LEN}" \
    SMOKE_DOUBLE_LAYERS="${SMOKE_DOUBLE_LAYERS}" SMOKE_SINGLE_LAYERS="${SMOKE_SINGLE_LAYERS}" \
    NAVA_BATCH_SIZE="${NAVA_BATCH_SIZE}" NAVA_GRAD_ACCUM="${NAVA_GRAD_ACCUM}" \
    NAVA_LOCAL_STEPS="${NAVA_LOCAL_STEPS}" NAVA_LR="${NAVA_LR}" NAVA_WEIGHT_DECAY="${NAVA_WEIGHT_DECAY}" \
    NAVA_WARMUP_STEPS="${NAVA_WARMUP_STEPS}" NAVA_NUM_WORKERS="${NAVA_NUM_WORKERS}" NAVA_IO_WORKERS="${NAVA_IO_WORKERS}" \
    NAVA_MODALITY="${NAVA_MODALITY}" MIN_DURATION="${MIN_DURATION}" MAX_DURATION="${MAX_DURATION}" \
    python3 - <<'PY'
import os
from pathlib import Path

import yaml

cfg = {
    "pipeline": "nava_src.pipeline_smoke.SmokePipeline",
    "model_type": "NAVA",
    "modality": "video",
    "use_bf16": True,
    "audio_latent_ch": 20,
    "video_latent_ch": int(os.environ["SMOKE_LATENT_DIM"]),
    "lambda_ddpm": 1.0,
    "smoke_latent_dim": int(os.environ["SMOKE_LATENT_DIM"]),
    "smoke_seq_len": int(os.environ["SMOKE_SEQ_LEN"]),
    "smoke_double_layers": int(os.environ["SMOKE_DOUBLE_LAYERS"]),
    "smoke_single_layers": int(os.environ["SMOKE_SINGLE_LAYERS"]),
    "batch_size": int(os.environ["NAVA_BATCH_SIZE"]),
    "lr": float(os.environ["NAVA_LR"]),
    "weight_decay": float(os.environ["NAVA_WEIGHT_DECAY"]),
    "warmup_steps": int(os.environ["NAVA_WARMUP_STEPS"]),
    "max_steps": int(os.environ["NAVA_LOCAL_STEPS"]),
    "num_workers": int(os.environ["NAVA_NUM_WORKERS"]),
    "grad_accum_steps": int(os.environ["NAVA_GRAD_ACCUM"]),
    "log_every": 1,
    "max_grad_norm": 1.0,
    "use_ema": False,
    "cpu_offload": False,
    "data": {
        "queue_size": 1,
        "io_workers": int(os.environ["NAVA_IO_WORKERS"]),
        "modal_prob": {
            "text_to_audio": 0.0,
            "text_to_video": 1.0,
            "text_to_image": 0.0,
            "text_to_av": 0.0,
        },
        "use_local_vae": False,
        "use_precomputed": True,
        "min_audio_duration": float(os.environ["MIN_DURATION"]),
        "max_audio_duration": float(os.environ["MAX_DURATION"]),
        "video_fps": 8,
        "video_min_frames": 1,
        "video_max_frames": 32,
        "video_tgt_frames": int(os.environ["SMOKE_SEQ_LEN"]),
        "add_spk_emb": False,
        "spk_emb_prob": 0.0,
        "use_speech_special_token": False,
    },
    "model": {
        "ckpt_dir": "",
        "audio_vae_ckpt_dir": "",
        "num_train_timesteps": 32,
    },
}
Path(os.environ["SMOKE_CONFIG"]).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"wrote mock smoke config {os.environ['SMOKE_CONFIG']} without text encoder or VAE")
PY
    return
  fi

  RUN_DIR="${RUN_DIR}" NAVA_ROOT="${NAVA_ROOT}" NAVA_ASSETS_DIR="${NAVA_ASSETS_DIR}" \
  SMOKE_CONFIG="${SMOKE_CONFIG}" SMOKE_JOINT_CONFIG="${SMOKE_JOINT_CONFIG}" \
  SMOKE_DIM="${SMOKE_DIM}" SMOKE_FFN_DIM="${SMOKE_FFN_DIM}" SMOKE_HEADS="${SMOKE_HEADS}" \
  SMOKE_DOUBLE_LAYERS="${SMOKE_DOUBLE_LAYERS}" SMOKE_SINGLE_LAYERS="${SMOKE_SINGLE_LAYERS}" \
  SMOKE_IMAGE_SIZE="${SMOKE_IMAGE_SIZE}" SMOKE_VIDEO_FPS="${SMOKE_VIDEO_FPS}" \
  SMOKE_VIDEO_TGT_FRAMES="${SMOKE_VIDEO_TGT_FRAMES}" \
  NAVA_BATCH_SIZE="${NAVA_BATCH_SIZE}" NAVA_GRAD_ACCUM="${NAVA_GRAD_ACCUM}" \
  NAVA_LOCAL_STEPS="${NAVA_LOCAL_STEPS}" NAVA_LR="${NAVA_LR}" NAVA_WEIGHT_DECAY="${NAVA_WEIGHT_DECAY}" \
  NAVA_WARMUP_STEPS="${NAVA_WARMUP_STEPS}" NAVA_NUM_WORKERS="${NAVA_NUM_WORKERS}" NAVA_IO_WORKERS="${NAVA_IO_WORKERS}" \
  NAVA_MODALITY="${NAVA_MODALITY}" \
  MIN_DURATION="${MIN_DURATION}" MAX_DURATION="${MAX_DURATION}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

import yaml

run_dir = Path(os.environ["RUN_DIR"])
nava_root = Path(os.environ["NAVA_ROOT"]).expanduser().resolve()
base_cfg = nava_root / "configs" / "nava.yaml"
if not base_cfg.exists():
    raise SystemExit(f"NAVA config not found: {base_cfg}")

smoke_dim = int(os.environ["SMOKE_DIM"])
smoke_heads = int(os.environ["SMOKE_HEADS"])
if smoke_dim % smoke_heads or (smoke_dim // smoke_heads) % 2:
    raise SystemExit("SMOKE_DIM must be divisible by SMOKE_HEADS with an even head_dim")

double_layers = int(os.environ["SMOKE_DOUBLE_LAYERS"])
single_layers = int(os.environ["SMOKE_SINGLE_LAYERS"])
joint = {
    "patch_size": [1, 2, 2],
    "model_type": "ti2v",
    "dim": smoke_dim,
    "ffn_dim": int(os.environ["SMOKE_FFN_DIM"]),
    "freq_dim": 128,
    "num_heads": smoke_heads,
    "num_layers": double_layers + single_layers,
    "num_double_layers": double_layers,
    "num_single_layers": single_layers,
    "num_double_final_layers": 0,
    "vid_in_dim": 48,
    "vid_out_dim": 48,
    "audio_in_dim": 20,
    "audio_out_dim": 20,
    "text_len": 512,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}
Path(os.environ["SMOKE_JOINT_CONFIG"]).write_text(json.dumps(joint, indent=2) + "\n", encoding="utf-8")

cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
modality = os.environ["NAVA_MODALITY"]
cfg_modality = {
    "text_to_audio": "audio",
    "text_to_video": "video",
    "text_to_image": "image",
    "text_to_av": "audio_video",
}[modality]
cfg["model_type"] = "NAVA"
cfg["modality"] = cfg_modality
cfg["use_bf16"] = True
cfg["audio_latent_ch"] = 20
cfg["video_latent_ch"] = 48
cfg["image_size"] = int(os.environ["SMOKE_IMAGE_SIZE"])
cfg["log_width"] = int(os.environ["SMOKE_IMAGE_SIZE"])
cfg["log_height"] = int(os.environ["SMOKE_IMAGE_SIZE"])
cfg["use_loss_reweight"] = False
cfg["masking_modality_prob"] = 0.0
cfg["i2v_mode_prob"] = 0.0
cfg["cpu_offload"] = True
cfg["no_split_norm_ffn"] = True

data = cfg.setdefault("data", {})
data.update(
    {
        "data_filelist": str(run_dir / "unused.list"),
        "data_weights": str(run_dir / "unused.weight"),
        "use_length_buckets": False,
        "enable_ddp_bucket_sync": False,
        "queue_size": 1,
        "io_workers": int(os.environ["NAVA_IO_WORKERS"]),
        "modal_prob": {
            "text_to_audio": 1.0 if modality == "text_to_audio" else 0.0,
            "text_to_video": 1.0 if modality == "text_to_video" else 0.0,
            "text_to_image": 0.0,
            "text_to_av": 1.0 if modality == "text_to_av" else 0.0,
        },
        "use_local_vae": True,
        "use_precomputed": False,
        "audio_tokens_per_sec": 25,
        "min_audio_duration": float(os.environ["MIN_DURATION"]),
        "max_audio_duration": float(os.environ["MAX_DURATION"]),
        "video_fps": int(os.environ["SMOKE_VIDEO_FPS"]),
        "video_min_frames": 1,
        "video_max_frames": int(float(os.environ["MAX_DURATION"]) * int(os.environ["SMOKE_VIDEO_FPS"])),
        "video_tgt_frames": int(os.environ["SMOKE_VIDEO_TGT_FRAMES"]),
        "add_spk_emb": False,
        "spk_emb_prob": 0.0,
        "use_speech_special_token": False,
    }
)

model = cfg.setdefault("model", {})
model.update(
    {
        "joint_config": str(Path(os.environ["SMOKE_JOINT_CONFIG"])),
        "ckpt_dir": os.environ["NAVA_ASSETS_DIR"],
        "audio_vae_ckpt_dir": os.environ["NAVA_ASSETS_DIR"],
        "num_train_timesteps": 32,
        "gradient_checkpointing": True,
        "gradient_checkpointing_offload": False,
        "gradient_checkpoint_every_n": 1,
    }
)

cfg.update(
    {
        "batch_size": int(os.environ["NAVA_BATCH_SIZE"]),
        "lr": float(os.environ["NAVA_LR"]),
        "weight_decay": float(os.environ["NAVA_WEIGHT_DECAY"]),
        "warmup_steps": int(os.environ["NAVA_WARMUP_STEPS"]),
        "max_steps": int(os.environ["NAVA_LOCAL_STEPS"]),
        "save_every": 1,
        "out_dir": str(run_dir / "nava_native_out"),
        "num_workers": int(os.environ["NAVA_NUM_WORKERS"]),
        "prefetch_factor": 2,
        "grad_accum_steps": int(os.environ["NAVA_GRAD_ACCUM"]),
        "log_every": 1,
        "log_cases_every": 1000000,
        "log_sample_steps": 1,
        "max_grad_norm": 1.0,
        "amp_dtype": "bf16",
    }
)
Path(os.environ["SMOKE_CONFIG"]).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"wrote {os.environ['SMOKE_JOINT_CONFIG']}")
print(f"wrote {os.environ['SMOKE_CONFIG']}")
PY
}

write_tiny_checkpoint() {
  if [[ -f "${SMOKE_BASE_CKPT}" && "${RECREATE_CKPT:-0}" != "1" ]]; then
    echo "using existing ${SMOKE_BASE_CKPT}"
    return
  fi
  if [[ "${NAVA_SMOKE_BACKEND}" == "mock" ]]; then
    NAVA_ROOT="${NAVA_ROOT}" SMOKE_CONFIG="${SMOKE_CONFIG}" SMOKE_BASE_CKPT="${SMOKE_BASE_CKPT}" \
    SMOKE_SEED="${SMOKE_SEED}" python3 - <<'PY'
import hashlib
import os
import sys
from pathlib import Path

import torch
import yaml

nava_root = Path(os.environ["NAVA_ROOT"]).expanduser().resolve()
sys.path.insert(0, str(nava_root))
from nava_src.pipeline_smoke import SmokePipeline  # noqa: E402

torch.manual_seed(int(os.environ["SMOKE_SEED"]))
cfg = yaml.safe_load(Path(os.environ["SMOKE_CONFIG"]).read_text(encoding="utf-8"))
pipe = SmokePipeline.create(cfg=cfg, device="cpu")
ckpt = Path(os.environ["SMOKE_BASE_CKPT"])
tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
torch.save({"state_dict": pipe.model.state_dict(), "yeto_smoke_seed": int(os.environ["SMOKE_SEED"])}, tmp)
tmp.replace(ckpt)
h = hashlib.sha256()
with ckpt.open("rb") as f:
    for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
        h.update(chunk)
print(f"wrote mock {ckpt} sha256={h.hexdigest()}")
PY
    return
  fi
  NAVA_ROOT="${NAVA_ROOT}" SMOKE_CONFIG="${SMOKE_CONFIG}" SMOKE_BASE_CKPT="${SMOKE_BASE_CKPT}" \
  SMOKE_SEED="${SMOKE_SEED}" python3 - <<'PY'
import hashlib
import os
import sys
from pathlib import Path

import torch
import yaml

nava_root = Path(os.environ["NAVA_ROOT"]).expanduser().resolve()
sys.path.insert(0, str(nava_root))
from nava_src.model_nava import NAVA  # noqa: E402

torch.manual_seed(int(os.environ["SMOKE_SEED"]))
cfg = yaml.safe_load(Path(os.environ["SMOKE_CONFIG"]).read_text(encoding="utf-8"))
model = NAVA(lambda_ddpm=cfg.get("lambda_ddpm", 1.0), target_dtype=torch.bfloat16, config=cfg)
ckpt = Path(os.environ["SMOKE_BASE_CKPT"])
tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
torch.save({"state_dict": model.state_dict(), "yeto_smoke_seed": int(os.environ["SMOKE_SEED"])}, tmp)
tmp.replace(ckpt)
h = hashlib.sha256()
with ckpt.open("rb") as f:
    for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
        h.update(chunk)
print(f"wrote {ckpt} sha256={h.hexdigest()}")
PY
}

prepare_labels_subset() {
  if [[ -f "${SMOKE_DATA_JSONL}" && "${RECREATE_LABELS:-0}" != "1" ]]; then
    echo "using existing ${SMOKE_DATA_JSONL}"
    return
  fi
  local args=(
    --input "${LABEL_INPUT}"
    --output "${SMOKE_DATA_JSONL}"
    --report "${SMOKE_REPORT}"
    --caption-field "${CAPTION_FIELD}"
    --quality "${QUALITY}"
    --min-duration "${MIN_DURATION}"
    --max-duration "${MAX_DURATION}"
    --max-output-rows "${MAX_OUTPUT_ROWS}"
    --cache-dir "${YETO_NAVA_DATA_CACHE}"
  )
  if [[ "${FFPROBE}" == "1" ]]; then
    args+=(--ffprobe)
  fi
  python3 -m yeto.nava.labels "${args[@]}"
  local rows
  rows="$(wc -l < "${SMOKE_DATA_JSONL}" | tr -d ' ')"
  if (( rows < MIN_OUTPUT_ROWS )); then
    echo "ERROR: only ${rows} rows in ${SMOKE_DATA_JSONL}; lower filters or inspect ${SMOKE_REPORT}" >&2
    return 1
  fi
  echo "prepared ${rows} NAVA rows at ${SMOKE_DATA_JSONL}"
}

build_syncer() {
  if [[ "${BUILD_SYNCER:-1}" != "1" ]]; then
    return
  fi
  (cd "${YETO_ROOT}/syncer" && cargo build --release)
}

main() {
  echo "Preparing NAVA L4 smoke bundle"
  print_smoke_env
  check_python_deps
  write_mock_nava_root
  if [[ "${NAVA_SMOKE_BACKEND}" != "mock" ]]; then
    require_file "${NAVA_ROOT}/nava_src/model_nava.py" "NAVA checkout"
  fi
  write_env_file
  write_smoke_config
  write_tiny_checkpoint
  prepare_labels_subset
  build_syncer
  cat <<NEXT

Prepared run bundle in ${RUN_DIR}
Copy this directory to the same absolute path on server 1 before starting learner 1, for example:
  rsync -a ${RUN_DIR}/ SERVER1:${RUN_DIR}/

Next:
  server 0: SYNCER_ADDR=SERVER0_IP:${SYNC_PORT} scripts/nava_l4_smoke/01_run_syncer.sh
  server 0: LEARNER_ID=0 SYNCER_ADDR=SERVER0_IP:${SYNC_PORT} scripts/nava_l4_smoke/02_run_learner.sh
  server 1: LEARNER_ID=1 SYNCER_ADDR=SERVER0_IP:${SYNC_PORT} scripts/nava_l4_smoke/02_run_learner.sh
NEXT
}

main "$@"
