"""Miles/SGLang :class:`~yeto.rl.engine.convertible.ConversionPort` (L7.2).

The calls are the ones of the upstream colocated branch of ``train.py`` that the
adapter already uses (``rollout.py`` ``_offload_after_rollout``, ``publish.py``
``_publish``, ``trainer.py`` ``onload``/``offload``):

* stop requests: ``controller.abort_all()`` (Miles puts the aborted groups back
  in its data buffer, so they are regenerated); the finished segments come from
  ``collect_inflight``, which the sampler side gives (it knows its partial
  trajectories);
* release rollout memory: ``controller.offload(tags=[weights, kv_cache,
  cuda_graph])`` (SGLang ``release_memory_occupation``, all tags);
* trainer: ``trainer.onload()`` / ``trainer.offload()`` (Megatron wake/sleep with
  ``--offload-train``);
* resume rollout: ``controller.onload_weights()``, then ``load_weights(version)``
  (the existing weight update; in this fork ``begin_weight_update`` ->
  ``update_weights_from_tensor`` -> ``end_weight_update``, checked on one GPU in
  SESSION24 L8.1), then ``controller.onload_kv()``. ``load_weights`` returns the
  engine weight checksum of the loaded version.

Not checked on GPU yet: the whole port (L7.4).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from yeto.rl.engine.convertible import InflightTrajectory

T = TypeVar("T")


def _all_memory_tags() -> list[str]:
    from sglang.srt.constants import (
        GPU_MEMORY_TYPE_CUDA_GRAPH,
        GPU_MEMORY_TYPE_KV_CACHE,
        GPU_MEMORY_TYPE_WEIGHTS,
    )

    return [GPU_MEMORY_TYPE_WEIGHTS, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH]


class MilesConversionPort:
    """``controller``: the Miles rollout manager handle of this node's engines;
    ``trainer``: the adapter trainer (``MilesTrainer``) on the same GPUs;
    ``run``: runs a coroutine to completion (the adapter's ``_run``)."""

    def __init__(self, *, controller: Any, trainer: Any,
                 run: Callable[[Awaitable[Any]], Any],
                 load_weights: Callable[[int], str | None],
                 collect_inflight: Callable[[], list[InflightTrajectory]] = list,
                 set_admission: Callable[[bool], None] | None = None,
                 memory_tags: Callable[[], list[str]] = _all_memory_tags) -> None:
        self._controller = controller
        self._trainer = trainer
        self._run = run
        self._load_weights = load_weights
        self._collect_inflight = collect_inflight
        self._set_admission = set_admission
        self._memory_tags = memory_tags

    def stop_requests(self) -> list[InflightTrajectory]:
        if self._set_admission is not None:
            self._set_admission(False)
        inflight = self._collect_inflight()  # read before abort clears the partial state
        self._run(self._controller.abort_all())
        return inflight

    def release_rollout(self) -> None:
        self._run(self._controller.offload(tags=self._memory_tags()))

    def onload_trainer(self) -> None:
        self._trainer.onload()

    def offload_trainer(self) -> None:
        self._trainer.offload()

    def resume_rollout(self, version: int) -> str | None:
        self._run(self._controller.onload_weights())
        checksum = self._load_weights(version)
        self._run(self._controller.onload_kv())
        return checksum

    def accept_requests(self) -> None:
        if self._set_admission is not None:
            self._set_admission(True)
