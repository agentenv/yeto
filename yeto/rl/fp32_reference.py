"""CPU fp32 reference LoRA gradient for the teacher-forcing tier (rl-engine-ports 6.2/6.3, design D12).

Tier 2 of the legacy-vs-ports equivalence check anchors the LoRA gradient on an
fp32 reference instead of comparing the two bf16 engines with each other: two
independent bf16 errors of ~8-14 % each compose into a larger cross-path
distance, so a cross-path bound either rejects numerically sound engines or
has to be so loose that it misses real bugs. The reference here is computed
on CPU in float32 from exactly the inputs both engines trained on:

* the base model at a pinned revision (``trust_remote_code`` is never enabled);
* the initial LoRA the engines started from (round audit ``round-<n>.base.f32``
  + its index ``round-<n>.json``);
* the replayed batch (``rollouts/island-<i>/<rollout_id>.pt`` recorded by the
  reference run, the same file ``yeto.rl.teacher_forcing`` replays);
* the engines' GRPO loss: group-normalized advantages (``reward - mean``,
  divided by ``std + 1e-6`` when ``grpo_std_normalization`` and std > 0, the
  unbiased std Miles uses), the PPO clipped surrogate with
  ``old_log_probs = log_probs.detach()`` (first on-policy step, so ratio == 1),
  per-sample token mean over the loss mask, summed over samples and divided by
  the number of samples (``global_batch_size``; ``calculate_per_token_loss``
  off). The returned gradient is the raw pre-clip gradient, matching
  ``yeto.rl.grad_audit`` semantics ("pre-clip, post-DP-reduce, loss-scale
  removed"); the grad-clip coefficient is reported separately.

``aggregation="token_mean"`` reproduces the per-token aggregation bug for
tests and diagnosis only; it is never the reference.

torch / transformers / peft are imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SAMPLE_MEAN = "sample_mean"
TOKEN_MEAN = "token_mean"
STD_EPS = 1e-6  # miles/ray/rollout/train_data_conversion.py _normalize_rewards_by_rollout
_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ReferenceInputError(RuntimeError):
    """A required input (model, initial LoRA, replay batch) is missing or inconsistent."""


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ReplaySample:
    tokens: list[int]
    response_length: int
    loss_mask: list[int]
    reward: float
    group: Any
    advantage: float = 0.0


@dataclass
class LossConfig:
    """The engine's GRPO loss settings (defaults = the rl-engine-ports TF runs)."""

    rewards_normalization: bool = True
    grpo_std_normalization: bool = True
    eps_clip: float = 0.2
    eps_clip_high: float = 0.2
    aggregation: str = SAMPLE_MEAN
    num_samples: int | None = None  # loss denominator; default len(batch)
    clip_grad: float = 1.0
    lora_alpha: float | None = None  # default: rank (scaling 1)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_replay(path: str | os.PathLike) -> list[ReplaySample]:
    """Samples of one recorded ``--save-debug-rollout-data`` dump."""
    import torch

    dump = torch.load(str(path), map_location="cpu", weights_only=True)
    samples = dump["samples"] if isinstance(dump, dict) else dump
    out = []
    for s in samples:
        if s.get("remove_sample"):
            continue
        tokens = [int(t) for t in s["tokens"]]
        r = int(s["response_length"])
        mask = s.get("loss_mask")
        mask = [1] * r if mask is None else [int(m) for m in mask]
        if len(mask) != r or r <= 0 or r >= len(tokens):
            raise ReferenceInputError(f"{path}: bad response_length/loss_mask ({r}, {len(mask)})")
        reward = s["reward"]
        if isinstance(reward, dict):
            raise ReferenceInputError(f"{path}: dict rewards need a reward key")
        out.append(ReplaySample(tokens, r, mask, float(reward), s.get("group_index")))
    if not out:
        raise ReferenceInputError(f"{path}: no samples")
    return out


def group_advantages(samples: list[ReplaySample], cfg: LossConfig) -> list[ReplaySample]:
    """GRPO advantages in place: per group ``r - mean`` (/ ``std + 1e-6``)."""
    import torch

    groups: dict[Any, list[ReplaySample]] = {}
    for s in samples:
        groups.setdefault(s.group, []).append(s)
    for members in groups.values():
        r = torch.tensor([m.reward for m in members], dtype=torch.float)
        if not cfg.rewards_normalization:
            adv = r
        else:
            adv = r - r.mean()
            if cfg.grpo_std_normalization and len(members) > 1:
                std = r.std()
                if std > 0:
                    adv = adv / (std + STD_EPS)
        for m, a in zip(members, adv.tolist()):
            m.advantage = float(a)
    return samples


def read_initial_lora(base_f32: str | os.PathLike, meta_json: str | os.PathLike):
    """``{canonical name: float32 tensor}`` from a round audit base + index."""
    import numpy as np
    import torch

    meta = json.loads(Path(meta_json).read_text())
    flat = np.fromfile(str(base_f32), dtype="<f4")
    out, off = {}, 0
    for spec in meta["specs"]:
        n = int(spec["numel"])
        out[spec["name"]] = torch.from_numpy(flat[off:off + n].copy()).view(*spec["shape"])
        off += n
    if off != flat.size:
        raise ReferenceInputError(f"{base_f32}: {flat.size} values, index covers {off}")
    want = meta.get("base_f32_sha256")
    got = sha256_file(base_f32)
    if want and want != got:
        raise ReferenceInputError(f"{base_f32}: sha256 {got} != index {want}")
    return out, meta, got


def resolve_model_dir(model: str, revision: str, *, local_path: str | None = None,
                      cache_dir: str | None = None, local_files_only: bool = True) -> Path:
    """Snapshot directory of ``model`` at the pinned ``revision``."""
    if not _REVISION.match(revision or ""):
        raise ReferenceInputError(f"model revision must be a 40-hex commit, got {revision!r}")
    if local_path:
        p = Path(local_path)
        if not (p / "config.json").is_file():
            raise ReferenceInputError(f"{p}: no config.json")
        if p.name != revision:
            raise ReferenceInputError(f"{p}: snapshot directory name != revision {revision}")
        return p
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, revision=revision, cache_dir=cache_dir,
                                      local_files_only=local_files_only))
    except Exception as error:  # noqa: BLE001 - surfaced as missing input
        raise ReferenceInputError(f"model {model}@{revision} unavailable: {error}") from error


def model_file_hashes(model_dir: Path) -> dict[str, str]:
    files = sorted([*model_dir.glob("*.safetensors"), model_dir / "config.json"])
    return {f.name: sha256_file(f.resolve()) for f in files if f.is_file()}


def load_base_model(model_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False)
    return model


# ---------------------------------------------------------------------------
# LoRA + loss
# ---------------------------------------------------------------------------
_LORA = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")


def attach_lora(model, initial: dict[str, Any], *, lora_alpha: float | None = None):
    """PEFT LoRA on ``model`` initialised to ``initial``; returns (peft model, {name: param})."""
    import torch
    from peft import LoraConfig, get_peft_model

    targets, rank = set(), None
    for name, t in initial.items():
        m = _LORA.match(name)
        if not m:
            raise ReferenceInputError(f"not a canonical PEFT LoRA name: {name}")
        targets.add(m.group(1).split(".")[-1])
        r = t.shape[0] if m.group(2) == "A" else t.shape[1]
        if rank is not None and r != rank:
            raise ReferenceInputError("mixed LoRA ranks are not supported")
        rank = r
    cfg = LoraConfig(r=rank, lora_alpha=lora_alpha if lora_alpha is not None else rank,
                     lora_dropout=0.0, target_modules=sorted(targets), bias="none")
    for p in model.parameters():
        p.requires_grad_(False)
    peft_model = get_peft_model(model, cfg)
    by_canon = {}
    for pname, p in peft_model.named_parameters():
        canon = pname.replace(".lora_A.default.weight", ".lora_A.weight").replace(
            ".lora_B.default.weight", ".lora_B.weight")
        if canon in initial:
            by_canon[canon] = p
    if set(by_canon) != set(initial):
        missing = sorted(set(initial) - set(by_canon))[:3]
        extra = sorted(set(by_canon) - set(initial))[:3]
        raise ReferenceInputError(f"LoRA layout mismatch: missing {missing}, extra {extra}")
    with torch.no_grad():
        for name, p in by_canon.items():
            if tuple(p.shape) != tuple(initial[name].shape):
                raise ReferenceInputError(f"{name}: shape {tuple(p.shape)} != {tuple(initial[name].shape)}")
            p.copy_(initial[name])
            p.requires_grad_(True)
            p.grad = None
    peft_model.train()
    return peft_model, by_canon


def sample_loss(logits, sample: ReplaySample, cfg: LossConfig):
    """Unnormalised per-sample policy loss and its mask sum (Miles policy_loss, ratio at step 1)."""
    import torch

    r = sample.response_length
    tok = torch.tensor(sample.tokens[-r:], dtype=torch.long)
    lp = torch.log_softmax(logits[-r - 1:-1].float(), -1).gather(-1, tok[:, None])[:, 0]
    old = lp.detach()
    ratio = torch.exp(lp - old)
    adv = torch.full_like(lp, sample.advantage)
    pg = torch.maximum(-adv * ratio,
                       -adv * torch.clamp(ratio, 1 - cfg.eps_clip, 1 + cfg.eps_clip_high))
    mask = torch.tensor(sample.loss_mask, dtype=lp.dtype)
    return (pg * mask).sum(), mask.sum()


def reference_gradient(model, params: dict[str, Any], samples: list[ReplaySample],
                       cfg: LossConfig) -> dict[str, Any]:
    """Accumulate d loss / d LoRA over ``samples`` (one forward/backward per sample)."""
    import torch

    n = cfg.num_samples or len(samples)
    total_tokens = sum(float(sum(s.loss_mask)) for s in samples)
    loss_value = 0.0
    for s in samples:
        logits = model(input_ids=torch.tensor([s.tokens], dtype=torch.long)).logits[0]
        num, msum = sample_loss(logits, s, cfg)
        if cfg.aggregation == SAMPLE_MEAN:
            loss = num / msum.clamp_min(1) / n
        elif cfg.aggregation == TOKEN_MEAN:
            loss = num / max(total_tokens, 1.0)
        else:
            raise ValueError(f"unknown aggregation {cfg.aggregation!r}")
        loss.backward()
        loss_value += float(loss.detach())
    grads = {}
    for name, p in params.items():
        grads[name] = (p.grad.detach().clone() if p.grad is not None
                       else torch.zeros_like(p))
    return {"grads": grads, "loss": loss_value}


def flatten(grads: dict[str, Any], order: list[str]):
    import numpy as np

    return np.concatenate([grads[n].detach().cpu().numpy().astype("<f4").ravel() for n in order])


# ---------------------------------------------------------------------------
# One island, end to end
# ---------------------------------------------------------------------------
def compute_island(*, replay: str, base_f32: str, base_meta: str, model: str, revision: str,
                   cfg: LossConfig, model_path: str | None = None, cache_dir: str | None = None,
                   local_files_only: bool = True, num_threads: int | None = None,
                   loaded_model=None) -> dict[str, Any]:
    """fp32 reference gradient for one island plus the provenance of every input.

    Raises :class:`ReferenceInputError` when an input is missing/inconsistent.
    """
    import numpy as np
    import torch

    for label, p in (("replay batch", replay), ("initial LoRA", base_f32),
                     ("initial LoRA index", base_meta)):
        if not p or not Path(p).is_file():
            raise ReferenceInputError(f"{label} missing: {p}")
    if not _REVISION.match(revision or ""):
        raise ReferenceInputError(f"model revision must be a 40-hex commit, got {revision!r}")
    if num_threads:
        torch.set_num_threads(num_threads)
    samples = group_advantages(load_replay(replay), cfg)
    initial, meta, base_sha = read_initial_lora(base_f32, base_meta)
    if meta.get("base_model_revision") and meta["base_model_revision"] != revision:
        raise ReferenceInputError(
            f"audit base_model_revision {meta['base_model_revision']} != {revision}")
    model_dir = resolve_model_dir(model, revision, local_path=model_path, cache_dir=cache_dir,
                                  local_files_only=local_files_only)
    base = loaded_model if loaded_model is not None else load_base_model(model_dir)
    peft_model, params = attach_lora(base, initial, lora_alpha=cfg.lora_alpha)
    try:
        out = reference_gradient(peft_model, params, samples, cfg)
    finally:
        peft_model.unload()  # restores ``base`` so a caller can reuse it for the next island
    order = sorted(params)
    flat = flatten(out["grads"], order)
    norm = float(np.linalg.norm(flat.astype(np.float64)))
    return {
        "flat": flat, "order": order,
        "shapes": {n: list(initial[n].shape) for n in order},
        "grad_norm": norm, "loss": out["loss"],
        "clip_coefficient": min(1.0, cfg.clip_grad / (norm + 1e-6)) if cfg.clip_grad else 1.0,
        "num_samples": len(samples), "loss_tokens": int(sum(sum(s.loss_mask) for s in samples)),
        "provenance": {
            "replay": {"path": str(Path(replay).resolve()), "sha256": sha256_file(replay)},
            "initial_lora": {"path": str(Path(base_f32).resolve()), "sha256": base_sha,
                             "index": str(Path(base_meta).resolve()),
                             "layout_hash": meta.get("layout_hash")},
            "model": {"id": model, "revision": revision, "dir": str(model_dir),
                      "files_sha256": model_file_hashes(model_dir)},
            "loss": {"aggregation": cfg.aggregation, "eps_clip": cfg.eps_clip,
                     "eps_clip_high": cfg.eps_clip_high,
                     "rewards_normalization": cfg.rewards_normalization,
                     "grpo_std_normalization": cfg.grpo_std_normalization,
                     "num_samples": cfg.num_samples or len(samples),
                     "lora_alpha": cfg.lora_alpha, "gradient": "raw pre-clip"},
            "dtype": "float32", "device": "cpu", "torch": torch.__version__,
        },
    }

