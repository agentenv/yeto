"""Resume manifest: prove that a restored optimizer continues where the cut stopped.

A-TRN report #8: restoring optimizer state on some stacks (seen on PPU) silently
resets the Adam ``step``. Bias correction and the LR schedule then restart while
every other check passes. This module is the backend-neutral guard (S25 N7.6):

1. At save time each rank records an :class:`OptimizerResumeState`: the Adam
   ``step`` of every parameter group (min and max over its parameters), the LR
   scheduler progress and the applied learning rates, and the ``exp_avg`` /
   ``exp_avg_sq`` norm and sha256 of a few fixed sampled parameters. The run
   adds the global optimizer step and the data watermark (:class:`ResumeManifest`).
2. After loading, and before the first training step, the restored state of
   every rank must equal the saved one (:func:`restore_problems`). A step that
   came back as 0 is named explicitly.
3. After the first training step the Adam step and the scheduler progress must
   have advanced by exactly the steps taken (:func:`first_step_problems`).

Any failure raises :class:`ResumeAssertionError` (a :class:`~yeto.rl.engine.cut.CutError`);
the caller refuses to resume and never falls back to a fresh optimizer.

Backends only produce the per-rank state: :func:`build_rank_state` takes plain
values; :func:`torch_optimizer_state` reads any ``torch.optim`` optimizer
(FSDP2 ``DTensor`` parameters are read as their local shard). torch is imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.rl.engine.cut import CutError, fsync_dir

RESUME_SCHEMA = "yeto.resume_manifest/v1"
RESUME_MANIFEST_NAME = "resume_manifest.json"
MOMENT_KEYS = ("exp_avg", "exp_avg_sq")
DEFAULT_SAMPLES = 4
LR_REL_TOL = 1e-12


class ResumeAssertionError(CutError):
    """The restored (or first-stepped) optimizer does not continue the saved run."""


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorFingerprint:
    norm: float  # L2 norm in float64
    sha256: str  # over the raw bytes, dtype and shape

    def to_dict(self) -> dict[str, Any]:
        return {"norm": self.norm, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TensorFingerprint":
        return cls(float(raw["norm"]), str(raw["sha256"]))


def fingerprint(tensor: Any) -> TensorFingerprint:
    """Norm and checksum of a tensor (an FSDP2 ``DTensor`` contributes its local shard)."""
    import torch

    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    flat = local.detach().to("cpu").contiguous().reshape(-1)
    digest = hashlib.sha256(f"{flat.dtype}|{tuple(local.shape)}|".encode())
    digest.update(flat.view(torch.uint8).numpy().tobytes())
    return TensorFingerprint(float(flat.to(torch.float64).norm()), digest.hexdigest())


@dataclass(frozen=True)
class GroupSteps:
    """Adam ``step`` over the parameters of one group; a healthy group has min == max."""

    min: int
    max: int

    def to_dict(self) -> dict[str, int]:
        return {"min": self.min, "max": self.max}


@dataclass(frozen=True)
class OptimizerResumeState:
    """What one rank records about its optimizer and LR scheduler."""

    adam_steps: Mapping[str, GroupSteps]  # group label -> steps
    scheduler_progress: int  # torch last_epoch, Megatron num_steps (samples)
    lrs: tuple[float, ...]  # applied LR per param group
    moments: Mapping[str, Mapping[str, TensorFingerprint]]  # sampled param -> exp_avg/exp_avg_sq

    def to_dict(self) -> dict[str, Any]:
        return {
            "adam_steps": {k: v.to_dict() for k, v in sorted(self.adam_steps.items())},
            "scheduler_progress": self.scheduler_progress,
            "lrs": list(self.lrs),
            "moments": {n: {k: f.to_dict() for k, f in sorted(m.items())} for n, m in sorted(self.moments.items())},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OptimizerResumeState":
        return cls(
            adam_steps={k: GroupSteps(int(v["min"]), int(v["max"])) for k, v in raw["adam_steps"].items()},
            scheduler_progress=int(raw["scheduler_progress"]),
            lrs=tuple(float(x) for x in raw["lrs"]),
            moments={n: {k: TensorFingerprint.from_dict(f) for k, f in m.items()} for n, m in raw["moments"].items()},
        )


@dataclass(frozen=True)
class ResumeManifest:
    """Run-level counters plus one :class:`OptimizerResumeState` per rank (keyed by shard name)."""

    global_step: int  # optimizer steps applied by this run
    data_watermark: int  # samples consumed (data cursor position)
    scheduler_increment: int  # scheduler progress per optimizer step (1 for torch, GBS for Megatron)
    ranks: Mapping[str, OptimizerResumeState] = field(default_factory=dict)
    schema: str = RESUME_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "global_step": self.global_step,
            "data_watermark": self.data_watermark,
            "scheduler_increment": self.scheduler_increment,
            "ranks": {k: v.to_dict() for k, v in sorted(self.ranks.items())},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResumeManifest":
        if raw.get("schema") != RESUME_SCHEMA:
            raise ResumeAssertionError(f"unknown resume manifest schema {raw.get('schema')!r}")
        try:
            return cls(
                global_step=int(raw["global_step"]),
                data_watermark=int(raw["data_watermark"]),
                scheduler_increment=int(raw["scheduler_increment"]),
                ranks={k: OptimizerResumeState.from_dict(v) for k, v in raw["ranks"].items()},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResumeAssertionError(f"malformed resume manifest: {exc}") from exc


# --------------------------------------------------------------------------
# Building the state
# --------------------------------------------------------------------------


def sample_names(names: Iterable[str], count: int = DEFAULT_SAMPLES) -> list[str]:
    """A fixed, evenly spread subset of the sorted names (first and last always included)."""
    ordered = sorted(set(names))
    if count <= 0 or not ordered:
        return []
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[0]]
    picks = {round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)}
    return [ordered[i] for i in sorted(picks)]


def _as_int_step(value: Any) -> int:
    step = float(value.item() if hasattr(value, "item") else value)
    if not math.isfinite(step) or step != int(step):
        raise ResumeAssertionError(f"Adam step {step!r} is not a whole number")
    return int(step)


def build_rank_state(
    *,
    steps_by_group: Mapping[str, Iterable[Any]],
    scheduler_progress: int,
    lrs: Sequence[float],
    moments_by_name: Mapping[str, Mapping[str, Any]],
    samples: int = DEFAULT_SAMPLES,
) -> OptimizerResumeState:
    """Plain values -> rank state. ``moments_by_name`` maps a parameter name to its
    moment tensors; only sampled names that carry every key in :data:`MOMENT_KEYS`
    are fingerprinted. A group without any step value counts as step 0 (no state yet)."""
    adam_steps = {}
    for label, values in steps_by_group.items():
        steps = [_as_int_step(v) for v in values] or [0]
        adam_steps[str(label)] = GroupSteps(min(steps), max(steps))
    eligible = [n for n, m in moments_by_name.items() if all(k in m for k in MOMENT_KEYS)]
    moments = {n: {k: fingerprint(moments_by_name[n][k]) for k in MOMENT_KEYS} for n in sample_names(eligible, samples)}
    return OptimizerResumeState(adam_steps, int(scheduler_progress), tuple(float(x) for x in lrs), moments)


def scheduler_progress_of(scheduler: Any) -> int:
    """torch ``LRScheduler.last_epoch`` or Megatron ``OptimizerParamScheduler.num_steps``."""
    for attr in ("last_epoch", "num_steps"):
        value = getattr(scheduler, attr, None)
        if value is not None:
            return int(value)
    raise ResumeAssertionError(f"cannot read the progress of LR scheduler {type(scheduler).__name__}")


def torch_optimizer_state(
    optimizer: Any,
    named_parameters: Iterable[tuple[str, Any]],
    scheduler: Any,
    *,
    samples: int = DEFAULT_SAMPLES,
) -> OptimizerResumeState:
    """Read any ``torch.optim`` Adam-family optimizer (plain module or FSDP2)."""
    names = {id(p): n for n, p in named_parameters}
    steps: dict[str, list[Any]] = {}
    moments: dict[str, Mapping[str, Any]] = {}
    for index, group in enumerate(optimizer.param_groups):
        values = steps.setdefault(f"group{index}", [])
        if "step" in group:  # apex/TE fused Adam keeps one step per group
            values.append(group["step"])
        for p in group["params"]:
            state = optimizer.state.get(p) or {}
            if "step" in state:
                values.append(state["step"])
            if id(p) in names:
                moments[names[id(p)]] = state
    lrs = [float(g["lr"]) for g in optimizer.param_groups]
    return build_rank_state(steps_by_group=steps, scheduler_progress=scheduler_progress_of(scheduler),
                            lrs=lrs, moments_by_name=moments, samples=samples)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _lr_close(a: float, b: float, rel: float = LR_REL_TOL) -> bool:
    return a == b or abs(a - b) <= rel * max(abs(a), abs(b))


def _step_problems(rank: str, saved: Mapping[str, GroupSteps], now: Mapping[str, GroupSteps],
                   advance: int) -> list[str]:
    out = []
    if set(saved) != set(now):
        return [f"{rank}: param groups {sorted(now)} != saved {sorted(saved)}"]
    for label in sorted(saved):
        want = saved[label].max + advance
        got = now[label]
        if got.min != got.max:
            out.append(f"{rank}/{label}: Adam step differs across parameters ({got.min}..{got.max}); "
                       "part of the optimizer state was reset")
        elif got.max != want:
            reset = " (Adam step was reset)" if got.max == 0 and want > 0 else ""
            out.append(f"{rank}/{label}: Adam step {got.max} != expected {want}{reset}")
    return out


def _rank_pairs(saved: ResumeManifest, now: ResumeManifest) -> tuple[list[str], list[tuple[str, Any, Any]]]:
    if set(saved.ranks) != set(now.ranks):
        return [f"ranks {sorted(now.ranks)} != saved {sorted(saved.ranks)}"], []
    return [], [(r, saved.ranks[r], now.ranks[r]) for r in sorted(saved.ranks)]


def save_problems(manifest: ResumeManifest) -> list[str]:
    """A manifest that cannot serve as a resume reference."""
    out = []
    if manifest.global_step < 0 or manifest.data_watermark < 0 or manifest.scheduler_increment <= 0:
        out.append("global_step/data_watermark must be >= 0 and scheduler_increment > 0")
    if not manifest.ranks:
        out.append("no rank state")
    for rank, state in sorted(manifest.ranks.items()):
        for label, steps in sorted(state.adam_steps.items()):
            if steps.min != steps.max:
                out.append(f"{rank}/{label}: Adam step differs across parameters ({steps.min}..{steps.max})")
    if manifest.global_step > 0 and manifest.ranks and not any(s.moments for s in manifest.ranks.values()):
        # a DistOpt rank may own no range; the run as a whole must have sampled moments
        out.append(f"no rank has a parameter with exp_avg/exp_avg_sq after {manifest.global_step} steps")
    return out


def restore_problems(saved: ResumeManifest, now: ResumeManifest) -> list[str]:
    """Restored state vs the saved one; every item must match exactly (LR within 1e-12 relative)."""
    out = []
    if now.global_step != saved.global_step:
        out.append(f"global step {now.global_step} != saved {saved.global_step}")
    if now.data_watermark != saved.data_watermark:
        out.append(f"data watermark {now.data_watermark} != saved {saved.data_watermark}")
    if now.scheduler_increment != saved.scheduler_increment:
        out.append(f"scheduler increment {now.scheduler_increment} != saved {saved.scheduler_increment}")
    problems, pairs = _rank_pairs(saved, now)
    out += problems
    for rank, s, n in pairs:
        out += _step_problems(rank, s.adam_steps, n.adam_steps, 0)
        if n.scheduler_progress != s.scheduler_progress:
            reset = " (scheduler was reset)" if n.scheduler_progress == 0 and s.scheduler_progress > 0 else ""
            out.append(f"{rank}: scheduler progress {n.scheduler_progress} != saved {s.scheduler_progress}{reset}")
        if len(n.lrs) != len(s.lrs) or not all(_lr_close(a, b) for a, b in zip(n.lrs, s.lrs)):
            out.append(f"{rank}: learning rates {list(n.lrs)} != saved {list(s.lrs)}")
        for name in sorted(s.moments):
            if name not in n.moments:
                out.append(f"{rank}: sampled parameter {name!r} has no exp_avg/exp_avg_sq after restore")
                continue
            for key in MOMENT_KEYS:
                if n.moments[name][key] != s.moments[name][key]:
                    out.append(f"{rank}: {name}.{key} norm {n.moments[name][key].norm!r} / checksum differs "
                               f"from saved (norm {s.moments[name][key].norm!r})")
    return out


def first_step_problems(saved: ResumeManifest, now: ResumeManifest, *, steps: int = 1,
                        expected_lrs: Mapping[str, Sequence[float]] | None = None) -> list[str]:
    """After ``steps`` optimizer steps on the restored state: every counter advanced by exactly ``steps``."""
    out = []
    if now.global_step != saved.global_step + steps:
        out.append(f"global step {now.global_step} != saved {saved.global_step} + {steps}")
    problems, pairs = _rank_pairs(saved, now)
    out += problems
    for rank, s, n in pairs:
        out += _step_problems(rank, s.adam_steps, n.adam_steps, steps)
        want = s.scheduler_progress + steps * saved.scheduler_increment
        if n.scheduler_progress != want:
            out.append(f"{rank}: scheduler progress {n.scheduler_progress} != expected {want}")
        lrs = (expected_lrs or {}).get(rank)
        if lrs is not None and (len(lrs) != len(n.lrs) or not all(_lr_close(a, b) for a, b in zip(n.lrs, lrs))):
            out.append(f"{rank}: learning rates {list(n.lrs)} != scheduled {list(lrs)}")
    return out


def require(problems: Sequence[str], what: str) -> None:
    if problems:
        raise ResumeAssertionError(f"{what}: " + "; ".join(problems))


def require_saveable(manifest: ResumeManifest) -> None:
    require(save_problems(manifest), "refusing to record a resume manifest")


def require_restored(saved: ResumeManifest, now: ResumeManifest) -> None:
    require(restore_problems(saved, now), "restored optimizer does not continue the cut; refusing to resume")


def require_first_step(saved: ResumeManifest, now: ResumeManifest, *, steps: int = 1,
                       expected_lrs: Mapping[str, Sequence[float]] | None = None) -> None:
    require(first_step_problems(saved, now, steps=steps, expected_lrs=expected_lrs),
            "first step after resume did not advance the optimizer by exactly the steps taken")


# --------------------------------------------------------------------------
# File form (backends whose checkpoint has no cut manifest, e.g. FSDP2)
# --------------------------------------------------------------------------


def write_resume_manifest(directory: str | os.PathLike[str], manifest: ResumeManifest) -> Path:
    """Write ``resume_manifest.json`` atomically; call it after the checkpoint itself is complete."""
    require_saveable(manifest)
    final = Path(directory) / RESUME_MANIFEST_NAME
    tmp = final.with_name(RESUME_MANIFEST_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest.to_dict(), fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    fsync_dir(final.parent)
    return final


def read_resume_manifest(directory: str | os.PathLike[str]) -> ResumeManifest:
    path = Path(directory) / RESUME_MANIFEST_NAME
    if not path.is_file():
        raise ResumeAssertionError(f"checkpoint {directory} has no {RESUME_MANIFEST_NAME}; refusing to resume")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResumeAssertionError(f"{path} is not JSON: {exc}") from exc
    return ResumeManifest.from_dict(raw)
