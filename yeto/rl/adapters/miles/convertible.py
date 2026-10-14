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
* resume rollout: ``controller.onload(tags=[weights])`` (not ``onload_weights``,
  which does nothing unless ``--offload-rollout-level`` has ``weight``, while
  the release above always frees the weights), then ``load_weights(version)``
  (the existing weight update; in this fork ``begin_weight_update`` ->
  ``update_weights_from_tensor`` -> ``end_weight_update``, checked on one GPU in
  SESSION24 L8.1), then ``controller.onload_kv()``. ``load_weights`` returns the
  engine weight checksum of the loaded version.

:func:`miles_conversion_port` wires the three callbacks from the island's
adapter objects:

* ``load_weights``: :meth:`.publish.MilesPublisher.load_version` (digest of the
  engines' common ``check_weights(checksum)`` body; compare with
  :attr:`.publish.MilesPublisher.published_checksum` of the same version);
* ``collect_inflight``: :func:`engine_requests_inflight`, the requests running
  or queued on the engines (Miles puts their groups back into its data buffer
  on ``abort_all`` and generates them again, so every one is an unfinished
  trajectory; a Miles group has no finished segment of its own to return);
* ``set_admission``: :meth:`.rollout.MilesRolloutPool.set_admission` (harness
  admission; no-op without a harness board).

Not checked on GPU yet: the whole port (L7.4).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from yeto.rl.engine.convertible import InflightTrajectory


def _weight_tags() -> list[str]:
    from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

    return [GPU_MEMORY_TYPE_WEIGHTS]


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
                 memory_tags: Callable[[], list[str]] = _all_memory_tags,
                 weight_tags: Callable[[], list[str]] = _weight_tags) -> None:
        self._controller = controller
        self._trainer = trainer
        self._run = run
        self._load_weights = load_weights
        self._collect_inflight = collect_inflight
        self._set_admission = set_admission
        self._memory_tags = memory_tags
        self._weight_tags = weight_tags

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
        self._run(self._controller.onload(tags=self._weight_tags()))
        checksum = self._load_weights(version)
        self._run(self._controller.onload_kv())
        return checksum

    def accept_requests(self) -> None:
        if self._set_admission is not None:
            self._set_admission(True)


def engine_requests_inflight(rollout: Any) -> list[InflightTrajectory]:
    """The requests running or queued on the engines of ``rollout`` (a
    :class:`.rollout.MilesRolloutPool`), one unfinished trajectory each, read from
    its load sample. Empty when the load is unknown (no router address, engine
    endpoint failed): the count is an observation only, since Miles puts the
    aborted groups back into its data buffer whatever this returns."""
    sample = rollout.load_sample()
    if sample is None or sample.get("running_requests") is None:
        return []
    count = int(sample["running_requests"]) + int(sample.get("queued_requests") or 0)
    return [InflightTrajectory(f"engine-request-{i}") for i in range(count)]


def miles_conversion_port(*, controller: Any, rollout: Any, trainer: Any, publisher: Any,
                          run: Callable[[Awaitable[Any]], Any]) -> MilesConversionPort:
    """The port of a colocated Miles island: ``controller`` is its inference
    controller, ``rollout`` its :class:`.rollout.MilesRolloutPool`, ``trainer`` its
    trainer (``onload``/``offload``), ``publisher`` its :class:`.publish.MilesPublisher`."""
    return MilesConversionPort(
        controller=controller, trainer=trainer, run=run,
        load_weights=publisher.load_version,
        collect_inflight=lambda: engine_requests_inflight(rollout),
        set_admission=rollout.set_admission)
