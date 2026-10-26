"""Full-parameter weight sync to vLLM in the checkpoint's own names (S28 G-M2).

The FSDP actor streams its HF model's ``state_dict`` names. For a family whose
``transformers`` loader converts the checkpoint (``nemotron_h``: ``backbone.`` ->
``model.``, per-expert tensors stacked into one ``mixer.experts.{up,down}_proj``),
those names are not the checkpoint's, and vLLM's loader, which reads checkpoint
names, fails (n4 run i5: ``KeyError: 'layers.1.mixer.experts.up_proj'``).

:func:`checkpoint_stream` rewrites the stream with the same key map the HF export
uses (:mod:`.hf_checkpoint_keys`): a renamed tensor goes under its checkpoint name,
a stacked tensor is split back into its per-expert rows. A model with no
conversion passes through unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .hf_checkpoint_keys import (INDEX_FILE, SINGLE_FILE, checkpoint_key_map, model_state_conversion,
                                 read_safetensors_header)


def checkpoint_names(model_path: str | Path) -> list[str]:
    path = Path(model_path)
    if (path / INDEX_FILE).is_file():
        return list(json.loads((path / INDEX_FILE).read_text())["weight_map"])
    _raw, header = read_safetensors_header(path / SINGLE_FILE)
    return [name for name in header if name != "__metadata__"]


def state_targets(model_path: str | Path) -> dict[str, tuple[tuple[str, int | None], ...]] | None:
    """model state name -> its (checkpoint name, row or None), in stacking order; None when
    the checkpoint names are the state names (nothing to rewrite)."""
    rules, prefix, state = model_state_conversion(model_path)
    key_map = checkpoint_key_map(checkpoint_names(model_path), state, rules=rules, prefix=prefix)
    out: dict[str, list[tuple[str, int | None]]] = {}
    for ck_name, key in key_map.keys.items():
        out.setdefault(key.model_name, []).append((ck_name, key.index))
    if all(len(v) == 1 and v[0] == (k, None) for k, v in out.items()):
        return None
    return {k: tuple(sorted(v, key=lambda item: -1 if item[1] is None else item[1])) for k, v in out.items()}


def _lookup(targets, name: str):
    for candidate in (name, f"model.{name}", name.removeprefix("model.")):
        if candidate in targets:
            return targets[candidate]
    return None


def checkpoint_stream(stream: Iterable[tuple[str, Any]], targets) -> Iterator[tuple[str, Any]]:
    """``stream`` of (state name, tensor) as (checkpoint name, tensor)."""
    for name, tensor in stream:
        sources = None if targets is None else _lookup(targets, name)
        if sources is None:
            yield name, tensor
            continue
        for ck_name, index in sources:
            yield ck_name, (tensor if index is None else tensor[index])


def worker_stream(worker: Any, stream: Iterable[tuple[str, Any]], peft_config: Any = None):
    """The hook in verl's ``ActorRolloutRefWorker.update_weights`` (patch_verl CKPT_NAMES).

    The key map is built once per worker from ``worker.config.model.path``; one line is
    printed the first time, saying whether the stream is rewritten."""
    if peft_config is not None:  # a LoRA sync carries adapter names
        return stream
    if not hasattr(worker, "_yeto_ckpt_targets"):
        path = worker.config.model.path
        try:
            worker._yeto_ckpt_targets = state_targets(path)
            note = "identity" if worker._yeto_ckpt_targets is None else f"{len(worker._yeto_ckpt_targets)} state tensors"
        except Exception as error:  # noqa: BLE001 - report it; the vLLM loader will then fail loudly
            worker._yeto_ckpt_targets = None
            note = f"FAILED {type(error).__name__}: {error}"
        print(f"[yeto] rollout_names {path}: {note}", flush=True)
    return checkpoint_stream(stream, worker._yeto_ckpt_targets)
