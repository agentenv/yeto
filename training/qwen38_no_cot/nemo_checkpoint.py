"""Checkpoint schedule and fail-closed resume guards for the pinned CP8 recipe.

NeMo remains responsible for DCP model/optimizer I/O and all tracked-state I/O.
These guards bind a checkpoint to its actual dataset and training contract and
verify the restored scheduler, shuffled loader and rank-local RNG state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

SCHEMA = "qwen38-checkpoint-resume/v1"


def _json_copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def prepare_checkpoint_boundary(scheduler):
    """Call before each production optimizer update, without changing its cursor.

    NeMo's zero-based checkpoint predicate is checked after the optimizer update.
    Its ordinary period therefore saves completed updates 1, 5, 10, ... here;
    epoch-end, final-step and preemption saves remain owned by NeMo.
    """
    step = scheduler.step
    if type(step) is not int or step < 0:
        raise ValueError("Checkpoint schedule requires a nonnegative integer cursor")
    scheduler.ckpt_every_steps = 1 if step == 0 else 5


def resume_identity(config, training_contract):
    dataset = config["dataset"]
    manifest = Path(dataset["path_or_dataset"])
    return _json_copy({
        "training_contract": training_contract,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "dataset": dataset,
        "dataloader": config["dataloader"],
        "seed": config.get("seed", 42),
        "scheduler": {key: config["step_scheduler"].get(key) for key in
                      ("global_batch_size", "local_batch_size", "num_epochs", "max_steps")},
        "lr_scheduler": config["lr_scheduler"],
    })


class CheckpointResumeContract:
    """Attach as recipe.resume_guard BEFORE super().setup() so NeMo tracks it."""

    def __init__(self, identity):
        self.identity = _json_copy(identity)
        self.restored = False

    def state_dict(self):
        return {"schema": SCHEMA, "identity": _json_copy(self.identity)}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("Checkpoint dataset/model/runtime/training contract differs")
        self.restored = True


def state_fingerprint(value):
    """Stable digest of nested tensor state, without logging tensor values."""
    import torch
    h = hashlib.sha256()

    def walk(obj):
        if isinstance(obj, torch.Tensor):
            tensor = obj.detach().cpu().contiguous()
            h.update(b"tensor")
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            h.update(b"dict")
            for key in sorted(obj):
                walk(key)
                walk(obj[key])
        elif isinstance(obj, (list, tuple)):
            h.update(type(obj).__name__.encode())
            for item in obj:
                walk(item)
            h.update(b"end")
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            h.update(type(obj).__name__.encode())
            h.update(json.dumps(obj, allow_nan=False).encode())
            h.update(b"\0")
        else:
            raise ValueError("Unsupported checkpoint state value: " + type(obj).__name__)

    walk(value)
    return h.hexdigest()


def validate_checkpoint_directory(path, expected_identity, *, world_size=8):
    """Read only small tracked state, requiring complete DCP model/optimizer files.

    Metadata presence is a necessary check; successful DCP loading and the GPU
    model/optimizer probes are still required to validate weight restoration.
    """
    import torch
    path = Path(path).resolve(strict=True)
    name = re.fullmatch(r"epoch_(\d+)_step_(\d+)", path.name)
    if not name or not path.is_dir() or (path / ".incomplete").exists():
        raise ValueError("Resume requires a complete named NeMo checkpoint")
    required = ["model/.metadata", "optim/.metadata", "step_scheduler.pt", "resume_guard.pt"]
    required += [f"dataloader/dataloader_dp_rank_{rank}.pt" for rank in range(world_size)]
    required += [f"rng/rng_global_rank_{rank}.pt" for rank in range(world_size)]
    if any(not (path / item).is_file() for item in required):
        raise ValueError("Checkpoint lacks model, optimizer or exact tracked resume state")

    def read(relative):
        return torch.load(path / relative, map_location="cpu", weights_only=True)

    CheckpointResumeContract(expected_identity).load_state_dict(read("resume_guard.pt"))
    scheduler = read("step_scheduler.pt")
    if (not isinstance(scheduler, dict) or set(scheduler) != {"step", "epoch"}
            or any(type(v) is not int for v in scheduler.values())
            or scheduler != {"step": int(name[2]) + 1, "epoch": int(name[1])}):
        raise ValueError("Checkpoint name and saved next-update cursor disagree")
    loader_states = [read(f"dataloader/dataloader_dp_rank_{rank}.pt") for rank in range(world_size)]
    yielded = [state.get("_num_yielded") for state in loader_states]
    if any(type(n) is not int or n <= 0 for n in yielded) or len(set(yielded)) != 1:
        raise ValueError("CP peers disagree on the completed loader cursor")
    return {
        "checkpoint": str(path),
        "next_step": scheduler["step"], "epoch": scheduler["epoch"],
        "loader_batches_yielded": yielded[0],
        "dataloader_sha256": [state_fingerprint(state) for state in loader_states],
        "rng_sha256": [state_fingerprint(read(f"rng/rng_global_rank_{rank}.pt"))
                       for rank in range(world_size)],
    }


def resolve_explicit_resume(checkpoint_dir, requested, expected_identity, *, world_size=8):
    """Resolve once before setup, rejecting NeMo's missing-LATEST fresh fallback."""
    if not requested:
        raise ValueError("Resume requires an explicit checkpoint path or LATEST")
    from nemo_automodel.components.checkpoint.utils import resolve_restore_from_to_checkpoint_dir
    path = resolve_restore_from_to_checkpoint_dir(checkpoint_dir, requested)
    if path is None:
        raise ValueError("Explicit resume did not resolve a complete checkpoint")
    return validate_checkpoint_directory(path, expected_identity, world_size=world_size)



def restored_loader_state(loader):
    """Canonicalize the exact sampler's deferred cursor without advancing it.

    Torchdata restores StatefulDistributedSampler.next_yielded, but its public
    state_dict still reports the old yielded until the generator starts. The
    effective pending cursor must agree with every loader counter before it can
    stand in for that one stale field; all other state remains hash-compared.
    """
    from copy import deepcopy
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
    state = loader.state_dict()
    sampler = loader.sampler
    if type(sampler) is StatefulDistributedSampler and sampler.next_yielded is not None:
        pending = sampler.next_yielded
        iterator = state.get("_sampler_iter_state", {})
        if (type(pending) is not int or pending < 0 or loader.batch_size != 1
                or iterator.get("samples_yielded") != pending
                or state.get("_sampler_iter_yielded") != pending
                or state.get("_num_yielded") != pending
                or iterator.get("sampler_state") != {"yielded": sampler.yielded}):
            raise RuntimeError("Deferred shuffled-sampler cursor disagrees with loader state")
        state = deepcopy(state)
        state["_sampler_iter_state"]["sampler_state"]["yielded"] = pending
    return state


def verify_restored_cursor(recipe, receipt):
    """Run immediately after load_checkpoint/setup, BEFORE creating its iterator."""
    import torch
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    scheduler = recipe.step_scheduler
    if (scheduler.step, scheduler.epoch) != (receipt["next_step"], receipt["epoch"]):
        raise RuntimeError("Loaded scheduler did not restore the next optimizer update")
    if not recipe.resume_guard.restored:
        raise RuntimeError("The checkpoint's training contract was not restored")
    loader = recipe.dataloader
    # This recipe is deliberately one epoch, DP1/CP8, local batch one, no workers.
    if scheduler.num_epochs != 1 or scheduler.epoch != 0 or loader.batch_size != 1 or loader.num_workers != 0:
        raise RuntimeError("Resume cursor verification requires the qualified one-epoch loader")
    expected_batches = min(scheduler.step * scheduler.grad_acc_steps, len(loader))
    if receipt["loader_batches_yielded"] != expected_batches:
        raise RuntimeError("Saved sample position disagrees with completed optimizer updates")
    restored_rng = recipe.rng.state_dict()
    if state_fingerprint(restored_rng) != receipt["rng_sha256"][rank]:
        raise RuntimeError("Loaded rank-local RNG differs from checkpoint")
    # StatefulDataLoader.state_dict() materializes its pending restored iterator
    # and marks it for reuse by the next __iter__ call. Torch's iterator setup
    # consumes a base-seed RNG draw, even with num_workers=0. Preserve the saved
    # training RNG while constructing that iterator, otherwise exact resume
    # silently shifts the next dropout/random draw relative to uninterrupted SFT.
    try:
        restored_loader = restored_loader_state(loader)
    finally:
        recipe.rng.load_state_dict(restored_rng)
    if state_fingerprint(restored_loader) != receipt["dataloader_sha256"][rank]:
        raise RuntimeError("Loaded shuffled dataloader state differs from checkpoint")
    return {"next_step": scheduler.step, "epoch": scheduler.epoch,
            "loader_batches_yielded": expected_batches,
            "dataloader_and_rng_restored": True}
