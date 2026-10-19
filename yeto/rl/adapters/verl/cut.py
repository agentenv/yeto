"""verl / FSDP2 cut: save at a safe point, restore into a fresh trainer, prove the
optimizer continues (S25 N7.6; same cut semantics as the Miles path).

A verl cut is a neutral :class:`~yeto.rl.engine.cut.CutManifest` (same
``progress`` / ``data`` / ``ledger`` / ``outer`` sections, same ``resume``
section, same immutability and checksum rules) over one yeto-owned file per
FSDP rank. verl itself is not changed and its own checkpoint manager is not
used: a rank file holds exactly the train state yeto resumes, so the CPU test
(FSDP2 over gloo) runs the same code as a GPU rank.

Rank file ``rank<NNNNN>.pt`` (``torch.save``):

* ``model`` -- trainable parameters (frozen base weights are skipped: LoRA), as
  the rank's local FSDP2 shards (``DTensor.to_local``) on CPU;
* ``optim`` -- the optimizer state dict keyed by parameter name, DTensor
  entries stored as local shards; ``dtensor`` lists which entries were DTensors;
* ``scheduler`` -- ``lr_scheduler.state_dict()``; ``rng`` -- CPU (and CUDA) RNG.

Restore needs the same world size (FSDP2 shards are per rank; a DP change is
refused before any rank writes). Each rank checks its file checksum, loads it
into the freshly built module/optimizer (DTensors rebuilt with the fresh
module's mesh and placements), and reports a digest of the loaded local state
(must equal the one recorded at save) plus its resume state.

Units (verl differs from Miles here): the run step is one trainer round (one
``update_actor`` call). verl steps the LR scheduler once per round and the
optimizer once per PPO mini-batch, so ``CutProgress`` uses
``global_batch_size=1`` (scheduler progress == rounds) and the first step after
a restore expects every Adam step to advance by ``optimizer_steps_per_round``.

Driver-side functions are pure (lists of rank results in, manifest/problems
out); :class:`VerlCutTrainer` binds them to a ``call_ranks`` callable (verl:
``VerlIsland.call_workers``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from yeto.rl.engine.cut import (
    CutContext,
    CutError,
    CutFile,
    CutManifest,
    RestoreExpectation,
    commit_manifest,
    context_problems,
    cut_dir,
    fsync_dir,
    fsync_file,
    sha256_file,
    verify_cut,
)
from yeto.rl.engine.resume_manifest import (
    OptimizerResumeState,
    ResumeAssertionError,
    ResumeManifest,
    require,
    require_data_watermark,
    require_first_step,
    require_saveable,
    restore_problems,
    scheduler_progress_of,
)

from .resume_check import TrainHandles, handles_rank_state, rank_label, train_handles

RANK_FILE_SCHEMA = "yeto.verl_cut_rank/v1"
RNG_POLICY = "restore_per_rank"
DATA_CURSOR_FIELDS = ("sample_offset",)  # verl data cursor (data_cursor.VerlDataCursor)


def rank_file(rank: int) -> str:
    return f"rank{int(rank):05d}.pt"


def layout_for(world: int) -> dict[str, int]:
    """FSDP2 actor layout: every rank is a data-parallel shard."""
    return {"world": int(world), "dp": int(world)}


# --------------------------------------------------------------------------
# Rank side (inside each FSDP rank)
# --------------------------------------------------------------------------


def _state_options() -> Any:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(full_state_dict=False, ignore_frozen_params=True)


def _is_dtensor(value: Any) -> bool:
    return hasattr(value, "to_local") and hasattr(value, "placements")


def _to_local(value: Any) -> Any:
    import torch

    if _is_dtensor(value):
        value = value.to_local()
    return value.detach().to("cpu").clone() if isinstance(value, torch.Tensor) else value


def _local_state(handles: TrainHandles) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """(model, optim, dtensor keys): the rank's local trainable state, CPU tensors only."""
    from torch.distributed.checkpoint.state_dict import get_state_dict

    model_sd, optim_sd = get_state_dict(handles.module, handles.optimizer, options=_state_options())
    dtensors = sorted(f"model/{k}" for k, v in model_sd.items() if _is_dtensor(v))
    model = {k: _to_local(v) for k, v in model_sd.items()}
    state = {}
    for name, entry in optim_sd.get("state", {}).items():
        state[name] = {k: _to_local(v) for k, v in entry.items()}
        dtensors += sorted(f"optim/{name}/{k}" for k, v in entry.items() if _is_dtensor(v))
    optim = {"state": state, "param_groups": [dict(g) for g in optim_sd.get("param_groups", [])]}
    return model, optim, dtensors


def _digest(model: Mapping[str, Any], optim: Mapping[str, Any]) -> str:
    """sha256 over the local trainable state (names, dtypes, shapes, bytes; group hyper-parameters)."""
    import torch

    h = hashlib.sha256()

    def tensor(name: str, t: Any) -> None:
        if not isinstance(t, torch.Tensor):
            h.update(f"{name}={t!r};".encode())
            return
        flat = t.detach().to("cpu").contiguous().reshape(-1)
        h.update(f"{name}|{flat.dtype}|{tuple(t.shape)}|".encode())
        h.update(flat.view(torch.uint8).numpy().tobytes())

    for name in sorted(model):
        tensor(f"model/{name}", model[name])
    for name in sorted(optim["state"]):
        for key in sorted(optim["state"][name]):
            tensor(f"optim/{name}/{key}", optim["state"][name][key])
    h.update(json.dumps(optim["param_groups"], sort_keys=True, default=repr).encode())
    return h.hexdigest()


def _rng_state() -> dict[str, Any]:
    import torch

    rng = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state()
    return rng


def _set_rng_state(rng: Mapping[str, Any]) -> None:
    import torch

    torch.set_rng_state(rng["cpu"])
    if "cuda" in rng:
        if not torch.cuda.is_available():
            raise CutError("cut rank file carries a CUDA RNG state but this rank has no CUDA device")
        torch.cuda.set_rng_state(rng["cuda"])


def _rank_and_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _rank_summary(handles: TrainHandles, *, rank: int, world: int, digest: str) -> dict[str, Any]:
    resume = handles_rank_state(handles)
    return {
        "coord": {"rank": rank, "world": world},
        "scheduler_samples": scheduler_progress_of(handles.scheduler),
        "has_optimizer_state": any(s.max > 0 for s in resume.adam_steps.values()) or bool(resume.moments),
        "has_rng": True,
        "state_digest": digest,
        "resume": resume.to_dict(),
    }


def save_rank(handles: TrainHandles, *, directory: str, cut_id: str) -> dict[str, Any]:
    """Write this rank's file (fsynced) and return its summary for the manifest."""
    import torch

    rank, world = _rank_and_world()
    model, optim, dtensors = _local_state(handles)
    name = rank_file(rank)
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(name + ".tmp")
    torch.save({"schema": RANK_FILE_SCHEMA, "cut_id": cut_id, "rank": rank, "world": world,
                "model": model, "optim": optim, "dtensor": dtensors,
                "scheduler": handles.scheduler.state_dict(), "rng": _rng_state()}, tmp)
    fsync_file(tmp)
    os.replace(tmp, path)
    fsync_dir(path.parent)
    summary = _rank_summary(handles, rank=rank, world=world, digest=_digest(model, optim))
    return {"path": name, "sha256": sha256_file(path), "bytes": path.stat().st_size, **summary}


def _rebuild(local: Any, template: Any) -> Any:
    """A saved local shard as a DTensor shaped like ``template`` (the fresh module's entry)."""
    from torch.distributed.tensor import DTensor

    local = local.to(template.to_local().device)
    return DTensor.from_local(local, template.device_mesh, template.placements, run_check=False,
                              shape=template.shape, stride=template.stride())


def restore_rank(handles: TrainHandles, *, directory: str, files: Sequence[Mapping[str, Any]],
                 cut_id: str) -> dict[str, Any]:
    """Load this rank's file into a freshly built module/optimizer; return the summary
    of the loaded state (digest + resume state) for the driver-side comparison."""
    import torch
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    rank, world = _rank_and_world()
    entry = next((CutFile.from_dict(f) for f in files if f["path"] == rank_file(rank)), None)
    if entry is None:
        raise CutError(f"cut {cut_id!r} has no file for rank {rank}")
    if len(files) != world:
        raise CutError(f"cut has {len(files)} rank files, this run has {world} ranks (DP change unsupported)")
    path = Path(directory) / entry.path
    if sha256_file(path) != entry.sha256:
        raise CutError(f"rank {rank}: {entry.path} checksum differs from the manifest")
    if any(state for state in handles.optimizer.state.values()):
        raise CutError("restore_cut needs a freshly built trainer (the optimizer already has state)")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("schema") != RANK_FILE_SCHEMA or saved.get("cut_id") != cut_id or saved.get("rank") != rank:
        raise CutError(f"rank {rank}: {entry.path} is not this rank's file of cut {cut_id!r}")

    template, _ = get_state_dict(handles.module, handles.optimizer, options=_state_options())
    if set(template) != set(saved["model"]):
        raise CutError(f"rank {rank}: trainable parameters differ from the cut: "
                       f"{sorted(set(template) ^ set(saved['model']))[:4]}")
    marked = set(saved["dtensor"])

    def value(key: str, local: Any, like: str) -> Any:
        if key not in marked:
            return local
        if not _is_dtensor(template[like]):
            raise CutError(f"rank {rank}: {key} was sharded at save but {like} is not sharded now")
        return _rebuild(local, template[like])

    model_sd = {k: value(f"model/{k}", v, k) for k, v in saved["model"].items()}
    optim_sd = {
        "state": {n: {k: value(f"optim/{n}/{k}", v, n) for k, v in e.items()}
                  for n, e in saved["optim"]["state"].items()},
        "param_groups": saved["optim"]["param_groups"],
    }
    set_state_dict(handles.module, handles.optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd,
                   options=_state_options())
    handles.scheduler.load_state_dict(saved["scheduler"])
    _set_rng_state(saved["rng"])
    model, optim, _ = _local_state(handles)
    return {"path": entry.path, **_rank_summary(handles, rank=rank, world=world, digest=_digest(model, optim))}


def resume_rank(handles: TrainHandles) -> dict[str, Any]:
    """This rank's resume state of the live trainer (first-step check)."""
    rank, _ = _rank_and_world()
    return {"path": rank_file(rank), "resume": handles_rank_state(handles).to_dict()}




# Ray entry points: ``VerlIsland.call_workers(fn, *args)`` calls ``fn(worker, *args)``.
def remote_save_rank(worker: Any, directory: str, cut_id: str) -> dict[str, Any]:
    return save_rank(train_handles(worker), directory=directory, cut_id=cut_id)


def remote_restore_rank(worker: Any, directory: str, files: list, cut_id: str) -> dict[str, Any]:
    return restore_rank(train_handles(worker), directory=directory, files=files, cut_id=cut_id)


def remote_resume_rank(worker: Any) -> dict[str, Any]:
    return resume_rank(train_handles(worker))


def remote_rank_layout(worker: Any = None) -> dict[str, int]:  # noqa: ARG001 - call_workers shape
    _, world = _rank_and_world()
    return layout_for(world)


def common_layout(layouts: Sequence[Mapping[str, int]]) -> dict[str, int]:
    distinct = {json.dumps(dict(x), sort_keys=True) for x in layouts}
    if len(distinct) != 1:
        raise CutError(f"ranks report different layouts: {sorted(distinct)}")
    return json.loads(distinct.pop())


# --------------------------------------------------------------------------
# Driver side (pure)
# --------------------------------------------------------------------------


def _rank_states(results: Sequence[Mapping[str, Any]]) -> dict[str, OptimizerResumeState]:
    """Rank results keyed ``rank<N>`` (from the rank file name every result carries)."""
    return {rank_label(int(Path(r["path"]).stem.removeprefix("rank"))): OptimizerResumeState.from_dict(r["resume"])
            for r in results}


def optimizer_steps_per_round(config: Any) -> int:
    """Optimizer steps one verl ``update_actor`` takes: ``ppo_epochs`` x PPO mini-batches
    (``data.train_batch_size / actor.ppo_mini_batch_size``, both in prompts). Read from
    the verl config; not yet checked against a live verl run."""
    actor = config.actor_rollout_ref.actor
    epochs = int(actor.get("ppo_epochs", 1) or 1)
    batch, mini = int(config.data.train_batch_size), int(actor.ppo_mini_batch_size)
    if mini <= 0 or batch % mini:
        raise CutError(f"train_batch_size {batch} is not a whole number of ppo_mini_batch_size {mini}")
    return epochs * (batch // mini)


def cut_resume_manifest(context: CutContext, results: Sequence[Mapping[str, Any]]) -> ResumeManifest:
    manifest = ResumeManifest(global_step=int(context.progress.local_step),
                              data_watermark=int(context.data["sample_offset"]),
                              scheduler_increment=1, ranks=_rank_states(results))
    require_saveable(manifest)
    return manifest


def build_manifest(context: CutContext, results: Sequence[Mapping[str, Any]], *, epoch: int) -> CutManifest:
    """The cut manifest over every rank's summary (refuses an incomplete or inconsistent cut)."""
    worlds = {int(r["coord"]["world"]) for r in results}
    if len(worlds) != 1 or len(results) != next(iter(worlds)):
        raise CutError(f"{len(results)} rank results for world sizes {sorted(worlds)}")
    summaries = [{k: v for k, v in r.items() if k not in ("sha256", "bytes")} for r in results]
    return CutManifest(
        cut_id=context.cut_id, epoch=int(epoch),
        runtime={"backend": "verl", "backend_fingerprint": context.backend_fingerprint,
                 "layout": layout_for(next(iter(worlds))), "rng_policy": RNG_POLICY},
        progress=context.progress, algorithm=context.algorithm,
        data={**dict(context.data), "cursor_fields": list(DATA_CURSOR_FIELDS)},
        ledger=context.ledger, outer=context.outer,
        in_flight=tuple(context.in_flight or ()),
        resume=cut_resume_manifest(context, results).to_dict(),
        files=tuple(CutFile(r["path"], r["sha256"], int(r["bytes"]), dict(r["coord"]))
                    for r in sorted(results, key=lambda r: r["path"])),
        rank_summaries=tuple(sorted(summaries, key=lambda r: r["path"])),
    )


def restored_problems(manifest: CutManifest, expect: RestoreExpectation,
                      results: Sequence[Mapping[str, Any]]) -> tuple[ResumeManifest, list[str]]:
    """The cut's resume manifest and every problem of the restored ranks: file set,
    loaded-state digest, scheduler progress and the resume assertions (data
    watermark excluded: compared later with the cursor that actually landed)."""
    if not manifest.resume:
        raise CutError(f"cut {manifest.cut_id!r} has no resume manifest; refusing to resume")
    saved = ResumeManifest.from_dict(manifest.resume)
    recorded = {s["path"]: s for s in manifest.rank_summaries}
    problems = []
    if sorted(r["path"] for r in results) != sorted(recorded):
        problems.append(f"restored ranks {sorted(r['path'] for r in results)} != cut {sorted(recorded)}")
    for r in results:
        s = recorded.get(r["path"])
        if s is None:
            continue
        if r["state_digest"] != s["state_digest"]:
            problems.append(f"{r['path']}: loaded train state differs from the cut")
        if r["scheduler_samples"] != s["scheduler_samples"]:
            problems.append(f"{r['path']}: scheduler at {r['scheduler_samples']}, cut {s['scheduler_samples']}")
    now = ResumeManifest(int(expect.local_step), saved.data_watermark, saved.scheduler_increment,
                         _rank_states(results))
    return saved, problems + restore_problems(saved, now, check_data_watermark=False)


def save_cut(call_ranks: Callable[..., list], *, epoch: int, context: CutContext, layout: Mapping[str, int],
             save_fn: Callable = remote_save_rank) -> CutManifest:
    """Check the context (nothing written on a problem), write every rank file, commit the manifest."""
    problems = context_problems(
        cut_id=context.cut_id, progress=context.progress, algorithm=context.algorithm,
        data={**dict(context.data), "cursor_fields": list(DATA_CURSOR_FIELDS)}, ledger=context.ledger,
        outer=context.outer, runtime={"backend_fingerprint": context.backend_fingerprint,
                                      "layout": dict(layout), "rng_policy": RNG_POLICY},
        in_flight=tuple(context.in_flight or ()))
    if int(context.progress.global_batch_size) != 1:
        problems.append("verl cut progress counts scheduler steps: global_batch_size must be 1")
    if problems:
        raise CutError("refusing a cut: " + "; ".join(problems))
    directory = cut_dir(context.root, context.cut_id)
    results = [dict(r) for r in call_ranks(save_fn, str(directory), context.cut_id)]
    manifest = build_manifest(context, results, epoch=epoch)
    commit_manifest(context.root, manifest, check_files=context.shared_filesystem)
    return manifest


class VerlCutTrainer:
    """Cut verbs of the verl trainer group (``CuttableTrainer`` save/restore half).

    ``call_ranks(fn, *args)`` runs ``fn(worker, *args)`` on every FSDP rank and
    returns their results. After :meth:`restore_cut` the caller that restores
    the data cursor calls :meth:`confirm_resumed_data_cursor`; the trainer group
    calls :meth:`check_first_step` after the first round."""

    def __init__(self, call_ranks: Callable[..., list], *, optimizer_steps_per_round: int,
                 fns: Mapping[str, Callable] | None = None) -> None:
        if int(optimizer_steps_per_round) <= 0:
            raise ValueError("optimizer_steps_per_round must be positive")
        self.call_ranks = call_ranks
        self.optimizer_steps_per_round = int(optimizer_steps_per_round)
        self.fns = {"save": remote_save_rank, "restore": remote_restore_rank, "resume": remote_resume_rank,
                    "layout": remote_rank_layout, **dict(fns or {})}
        self.resume_reference: ResumeManifest | None = None
        self.cursor_confirmed = True

    def actual_layout(self) -> dict[str, int]:
        return common_layout(self.call_ranks(self.fns["layout"]))

    def save_cut(self, *, epoch: int, context: CutContext) -> str:
        save_cut(self.call_ranks, epoch=epoch, context=context, layout=self.actual_layout(),
                 save_fn=self.fns["save"])
        return context.cut_id

    def restore_cut(self, cut_id: str, *, epoch: int, root: str, expect: RestoreExpectation,
                    shared_filesystem: bool = True) -> CutManifest:
        """Restore into a freshly built trainer; any exception means RECOVERY_REQUIRED."""
        manifest = verify_cut(root, cut_id, expect, check_files=shared_filesystem)
        if manifest.runtime.get("backend") != "verl":
            raise CutError(f"cut {cut_id!r} was written by {manifest.runtime.get('backend')!r}, not verl")
        if manifest.epoch > epoch:
            raise CutError(f"cut epoch {manifest.epoch} is newer than the restoring epoch {epoch}")
        files = [f.to_dict() for f in manifest.files]
        results = [dict(r) for r in self.call_ranks(self.fns["restore"], str(cut_dir(root, cut_id)), files, cut_id)]
        saved, problems = restored_problems(manifest, expect, results)
        require(problems, "restored verl trainer does not continue the cut; refusing to resume")
        self.resume_reference, self.cursor_confirmed = saved, False
        return manifest

    def confirm_resumed_data_cursor(self, cursor: Mapping[str, int]) -> None:
        """The data cursor the run resumed from vs the cut's data watermark."""
        if self.resume_reference is None:
            return
        require_data_watermark(self.resume_reference, int(cursor["sample_offset"]))
        self.cursor_confirmed = True

    def check_first_step(self, rounds: int = 1) -> None:
        """After the first ``rounds`` rounds on the restored state (no-op without a pending restore)."""
        if self.resume_reference is None:
            return
        if not self.cursor_confirmed:
            raise ResumeAssertionError("first round after resume, but the restored data cursor was never "
                                       "compared with the cut's data watermark; refusing to continue")
        saved, self.resume_reference = self.resume_reference, None
        results = [dict(r) for r in self.call_ranks(self.fns["resume"])]
        now = ResumeManifest(saved.global_step + rounds, saved.data_watermark, saved.scheduler_increment,
                             _rank_states(results))
        require_first_step(saved, now, steps=rounds, optimizer_steps=rounds * self.optimizer_steps_per_round)
