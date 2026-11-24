"""Preflight of ``--fragments`` for a Miles full-parameter run.

A Miles full-parameter island refuses a layout whose fragment count is not
``--fragments`` (``MilesChunkedFullParameterAdapter.initialize``), but it only
knows the count after it has built the model: minutes of paid boot per island.
The count follows from the model config, the trainer parallelism, the fragment
byte bound (``--rl-sync-fragment-max-bytes``, FP32) and the matrix merge, so
the launcher computes it here with the backend's offline planner (Miles:
``launch_flags.offline_full_parameter_layout``, the dense Qwen3 layout of
``models.dense_gpt_layout``) and refuses a mismatch before anything is ordered.

Only the model's ``config.json`` is read, from a local model directory or the
local Hugging Face cache; nothing is downloaded. Out of scope (no config in the
cache, a model family the offline layout does not list, shape-free layouts,
partial experts, CP or DP above 1, another backend), the check prints nothing
and the run goes on as before. At PP>1 the count is an estimate: a mismatch is
a warning, not a refusal.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def local_hf_config(model: str, revision: str | None) -> bytes | None:
    """The bytes of ``model``'s ``config.json`` from a local directory or the local
    HF cache (``revision``: a commit, branch or tag; None: main); None if not there."""
    try:
        path = Path(str(model)).expanduser()
        if path.is_dir():
            config = path / "config.json"
            return config.read_bytes() if config.is_file() else None
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415

        found = try_to_load_from_cache(str(model), "config.json", revision=revision or None)
        return Path(found).read_bytes() if isinstance(found, str) else None
    except Exception:  # noqa: BLE001 - not a repo id, no huggingface_hub, unreadable
        return None


def _trainer_islands(args: Any, per_island: list, specs: list) -> list[tuple[Any, Any]]:
    from ..rl.engine.sample_roles import SAMPLER, island_roles

    roles = island_roles(args, len(specs))
    return [(iargs, spec) for m, (iargs, spec) in enumerate(zip(per_island, specs))
            if roles.get(m) != SAMPLER]


def _in_scope(args: Any, trainers: list[tuple[Any, Any]]) -> bool:
    from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESET
    from .topology import rl_trainer_shape

    if (getattr(args, "rl_sync_preset", None) != FULL_PARAMETER_SYNC_PRESET
            or getattr(args, "tuning", "lora") != "full"
            or getattr(args, "rl_sync_layout", "shape-bound") != "shape-bound"
            or getattr(args, "rl_model_recipe", "generic") != "generic"
            or getattr(args, "rl_expert_selection", None)
            or int(getattr(args, "context_parallel", 1) or 1) != 1
            or int(getattr(args, "expert_parallel", 1) or 1) != 1
            or not trainers):
        return False
    group = int(args.tensor_parallel or 1) * int(getattr(args, "pipeline_parallel", 1) or 1)
    for iargs, spec in trainers:
        if getattr(iargs, "rl_backend", None) != getattr(args, "rl_backend", None):
            return False  # an island of another backend (--rl-island-engine)
        try:
            nodes, gpus = rl_trainer_shape(iargs, spec)
        except ValueError:
            return False
        if nodes * gpus != group:  # DP > 1 (or a shape the layout does not model)
            return False
    return True


@dataclass(frozen=True)
class LayoutSummary:
    """The full-parameter layout an island of one trainer shape builds (offline planner)."""

    tp: int
    pp: int
    fragments: int
    ranks: int
    fragment_bytes: tuple[int, ...]
    layout_hash: str | None  # None without a commit revision (dense_gpt_layout.offline_layout)
    exact: bool

    @property
    def shape(self) -> str:
        return f"TP{self.tp} x PP{self.pp}"

    def digest(self) -> str:
        """Short digest of the layout: its hash, else the SHA-256 of the fragment sizes."""
        if self.layout_hash:
            return self.layout_hash[:16]
        return hashlib.sha256(json.dumps(self.fragment_bytes).encode()).hexdigest()[:16]

    def describe(self) -> str:
        return (f"{self.shape}: {self.fragments} fragments over {self.ranks} trainer rank(s), "
                f"layout {self.digest()}" + ("" if self.exact else " (PP>1 estimate)"))


def offline_summary(args: Any, *, tp: int, pp: int) -> LayoutSummary | None:
    """The layout of a trainer of shape TP x PP from the backend's offline planner and the
    model's local ``config.json``; None when the planner or the config is not there."""
    from .rl_backend import _rl_backend_module

    plan = getattr(_rl_backend_module(args, "launch_flags"), "offline_full_parameter_layout", None)
    if plan is None:
        return None
    raw = local_hf_config(args.model, getattr(args, "model_revision", None))
    if raw is None:
        return None
    try:
        config = json.loads(raw)
    except ValueError:
        return None
    try:
        layout = plan(
            config, tp=tp, pp=pp, minimum_fragments=int(args.fragments),
            max_fragment_bytes=int(getattr(args, "rl_sync_fragment_max_bytes", 2 << 30)),
            matrix_merge=getattr(args, "rl_sync_matrix_merge", "avg"),
            model_revision=str(getattr(args, "model_revision", None) or ""),
            config_sha256=hashlib.sha256(raw).hexdigest())
    except ImportError:  # the backend's planner needs a package the control host lacks
        return None
    if layout is None:
        return None
    return LayoutSummary(tp=tp, pp=pp, fragments=layout.fragments, ranks=layout.ranks,
                         fragment_bytes=tuple(layout.fragment_bytes), layout_hash=layout.layout_hash,
                         exact=layout.exact)


def check_fragment_layout(args: Any, per_island: list, specs: list) -> None:
    """Refuse (ValueError) a ``--fragments`` the islands' full-parameter layout
    cannot have; warn on stderr when the count is an estimate (PP>1). Silent when
    the count matches or the check is out of scope (module doc)."""
    from .rl_backend import _rl_backend_module

    # the backend's offline planner (Miles: launch_flags.offline_full_parameter_layout)
    plan = getattr(_rl_backend_module(args, "launch_flags"), "offline_full_parameter_layout", None)
    trainers = _trainer_islands(args, per_island, specs)
    if plan is None or not _in_scope(args, trainers):
        return
    tp = int(args.tensor_parallel or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    layout = offline_summary(args, tp=tp, pp=pp)
    if layout is None or layout.fragments == int(args.fragments):
        return
    max_bytes = int(getattr(args, "rl_sync_fragment_max_bytes", 2 << 30))
    merge = getattr(args, "rl_sync_matrix_merge", "avg")
    detail = (f"the Miles full-parameter layout of {args.model} at TP{tp} x PP{pp} "
              f"(config.json from the local model files), FP32 fragments of at most "
              f"{max_bytes} bytes (--rl-sync-fragment-max-bytes) and matrix merge {merge} "
              f"has {layout.fragments} fragments over {layout.ranks} trainer rank(s)")
    if not layout.exact:
        print(f"[launcher] WARNING: --fragments {args.fragments} may not match: by estimate "
              f"{detail} (PP>1: Megatron's stage split is estimated); an island refuses a "
              f"mismatched layout after it has built the model", file=sys.stderr, flush=True)
        return
    raise ValueError(
        f"--fragments {args.fragments} does not match the layout the trainer islands build: "
        f"{detail}; every island would refuse it after its boot. Set --fragments {layout.fragments}")


def parse_trainer_shape(value: str) -> tuple[int, int]:
    """``TP`` or ``TPxPP`` (``--rl-fleet-trainer-shape``) as (tp, pp)."""
    parts = str(value).lower().split("x")
    if len(parts) not in (1, 2) or not all(part.isdigit() and int(part) >= 1 for part in parts):
        raise ValueError(f"--rl-fleet-trainer-shape must be TP or TPxPP (e.g. 2 or 2x1), got {value!r}")
    return int(parts[0]), int(parts[1]) if len(parts) == 2 else 1


def check_fleet_shape(args: Any) -> None:
    """A new or migrated island of a running fleet (``--rl-fleet-trainer-shape``, the
    trainer shape of the fleet's islands): refuse (ValueError, so at dry-run, before
    anything is ordered) an island whose full-parameter layout differs from the fleet's.

    Both layouts come from the ``--fragments`` preflight's offline planner; the
    refusal names both sides' fragment counts and layout digests. Without the
    planner or a local config.json the shapes themselves are compared: a
    shape-bound layout binds the trainer ranks, so another TP or PP is another
    layout. A shape-free layout (``--rl-sync-layout shape-free``) spans shapes.
    Unset: nothing is checked."""
    from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESET

    value = getattr(args, "rl_fleet_trainer_shape", None)
    if value is None:
        return
    fleet_tp, fleet_pp = parse_trainer_shape(value)
    if getattr(args, "rl_sync_preset", None) != FULL_PARAMETER_SYNC_PRESET:
        raise ValueError(f"--rl-fleet-trainer-shape applies to --rl-sync-preset {FULL_PARAMETER_SYNC_PRESET}")
    if getattr(args, "rl_sync_layout", "shape-bound") != "shape-bound":
        return
    tp = int(args.tensor_parallel or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    mine = offline_summary(args, tp=tp, pp=pp)
    fleet = mine if (tp, pp) == (fleet_tp, fleet_pp) else offline_summary(args, tp=fleet_tp, pp=fleet_pp)
    if mine is not None and fleet is not None:
        if (mine.fragments, mine.fragment_bytes, mine.layout_hash) == (
                fleet.fragments, fleet.fragment_bytes, fleet.layout_hash):
            return
        raise ValueError(
            "this island's full-parameter layout differs from the fleet's: fleet islands "
            f"{fleet.describe()}; this island {mine.describe()}. Islands of different shapes "
            f"cannot merge in one syncer session; launch it at TP{fleet_tp} x PP{fleet_pp}")
    if (tp, pp) != (fleet_tp, fleet_pp):
        raise ValueError(
            f"this island's trainer shape TP{tp} x PP{pp} differs from the fleet's TP{fleet_tp} x "
            f"PP{fleet_pp} (the layouts are not computable offline here: no planner or no local "
            "config.json); a shape-bound full-parameter layout binds the trainer ranks, so the "
            f"syncer would refuse the island. Launch it at TP{fleet_tp} x PP{fleet_pp}")
