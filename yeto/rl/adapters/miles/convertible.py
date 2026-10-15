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
  SESSION24 L8.1), then ``controller.onload_kv()``, then ``read_checksum()``:
  the engine weight checksum of the loaded version. The read comes after the KV
  onload: in LoRA mode it serves one generate to load the adapter into the LoRA
  pool, which crashed the engine scheduler with the KV cache released (G-L7.4).

:func:`miles_conversion_port` wires the three callbacks from the island's
adapter objects:

* ``load_weights`` / ``read_checksum``: :meth:`.publish.MilesPublisher.load_version`
  and :meth:`.publish.MilesPublisher.served_checksum` (digest of the engines'
  common ``check_weights(checksum)`` body; compare with
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

import time
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
                 load_weights: Callable[[int], None],
                 read_checksum: Callable[[], str | None] = lambda: None,
                 collect_inflight: Callable[[], list[InflightTrajectory]] = list,
                 set_admission: Callable[[bool], None] | None = None,
                 memory_tags: Callable[[], list[str]] = _all_memory_tags,
                 weight_tags: Callable[[], list[str]] = _weight_tags) -> None:
        self._controller = controller
        self._trainer = trainer
        self._run = run
        self._load_weights = load_weights
        self._read_checksum = read_checksum
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
        self._load_weights(version)
        self._run(self._controller.onload_kv())
        return self._read_checksum()

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
        load_weights=publisher.load_version, read_checksum=publisher.served_checksum,
        collect_inflight=lambda: engine_requests_inflight(rollout),
        set_admission=rollout.set_admission)


# ---------------------------------------------------------------------------
# Convertible pool node (rl-trainer-start-stop L7.5)
# ---------------------------------------------------------------------------
def convertible_node_name(miles_args: Any) -> str | None:
    """This island's node name when it is a convertible pool node, else None."""
    import os

    from .buffer_rollout import CONVERTIBLE, SHARD_ENV, SampleBufferRolloutError

    config = getattr(miles_args, "yeto_rl_sample_buffer", None)
    if config is None or config.role != CONVERTIBLE:
        return None
    name = os.environ.get(SHARD_ENV)
    if not name:
        raise SampleBufferRolloutError(f"a convertible node needs {SHARD_ENV} (its node name)")
    return name


class DriverHandoffPort:
    """:class:`~yeto.rl.engine.convertible.ConversionPort` of a convertible pool node.

    The trainer of each conversion is a fresh island driver on this island's
    colocated components (one trainer incarnation). Its rounds move the GPU memory
    themselves, in the upstream colocated order: after the round is read the
    engines release their memory (``_offload_after_rollout``), the trainer wakes to
    train, sleeps again, and the engines take the new weights and their KV cache
    back (``MilesPublisher._publish``). So the port does not move memory:

    * ``stop_requests``: close admission, count the requests on the engines, abort
      them (Miles regenerates aborted groups; at a round boundary there are none);
    * ``release_rollout``: the rollout function reads the buffer from now on
      (``executor_set_mode(trainer)``); the engines release at the first round;
    * ``onload_trainer`` / ``offload_trainer``: nothing (each round wakes the trainer;
      the drained incarnation left it asleep after its last publication);
    * ``resume_rollout(version)``: the rollout function generates again, the engines
      load the trained ``version`` once more from the trainer (``load_version``, the
      L7.4 path) and their checksum is read (``served_checksum``); the node compares
      it with the checksum of the last publication;
    * ``accept_requests``: open admission.
    """

    def __init__(self, *, controller: Any, pool: Any, run: Callable[[Awaitable[Any]], Any],
                 publisher: Callable[[], Any]) -> None:
        self._controller = controller
        self._pool = pool
        self._run = run
        self._publisher = publisher  # the publisher of the current incarnation's driver

    def _set_mode(self, mode: str) -> None:
        from .buffer_rollout import executor_set_mode

        self._pool._in_executor(executor_set_mode, mode)

    def stop_requests(self) -> list[InflightTrajectory]:
        self._pool.set_admission(False)
        inflight = engine_requests_inflight(self._pool)
        self._run(self._controller.abort_all())
        return inflight

    def release_rollout(self) -> None:
        from .buffer_rollout import TRAINER

        self._set_mode(TRAINER)

    def onload_trainer(self) -> None:
        pass

    def offload_trainer(self) -> None:
        pass

    def resume_rollout(self, version: int) -> str | None:
        from .buffer_rollout import SAMPLER

        publisher = self._publisher()
        publisher.load_version(int(version))
        checksum = publisher.served_checksum()
        self._set_mode(SAMPLER)
        return checksum

    def accept_requests(self) -> None:
        self._pool.set_admission(True)


def run_convertible_node(miles_args: Any, *, build_driver: Callable[[], Any], controller: Any,
                         executor: Any, actor: Any, runner: Any,
                         environ: Any = None) -> Any:
    """Island entry of a convertible pool node: boot the engines with the base
    weights (policy version 0), then run :class:`~yeto.rl.engine.convertible_remote.ConvertibleNodeAgent`
    until the head ends the run. Returns the number of trainer incarnations run."""
    import os

    from yeto.rl.engine.convertible import ConvertibleNode, NodeClass, NodeSpec
    from yeto.rl.engine.convertible_remote import ConvertibleNodeAgent, RolloutRound
    from yeto.rl.engine.events import append_tape_event
    from yeto.rl.engine.resume import store_for
    from yeto.rl.engine.trainer_remote import (
        DEFAULT_POLL_S,
        POLL_ENV,
        CommandReader,
        IslandTrainerAgent,
        LocalStoreFiles,
        StoreCutLog,
    )

    from .publish import _default_update_weights
    from .rollout import MilesRolloutPool, RayMetadataSink, _release
    from .state_plugin import ALIGN_WEIGHT_VERSION

    env = os.environ if environ is None else environ
    name = convertible_node_name(miles_args)
    store = store_for(miles_args.yeto_rl_sample_buffer.store, environ=env)
    files = LocalStoreFiles(store)
    metadata = RayMetadataSink()
    pool = MilesRolloutPool(inference_controller=controller, rollout_executor=executor,
                            metadata=metadata, expected_policy=lambda: (0, ""), runner=runner,
                            args=miles_args)
    poll_s = float(env.get(POLL_ENV) or DEFAULT_POLL_S)
    current: dict[str, Any] = {}

    def emit(event: str, **fields: Any) -> None:
        append_tape_event(miles_args, {"event": event, **fields})

    async def boot() -> None:
        """Upstream colocated boot (``train.py``): the engines start with the trainer's
        weights, stamped policy version 0, then take their KV cache."""
        await actor.run_plugin(ALIGN_WEIGHT_VERSION, {"next_version": 0})
        await _default_update_weights()(miles_args, actor, executor, controller)
        if getattr(miles_args, "offload_rollout", False):
            await controller.onload_kv()

    rollout_ids = iter(range(1 << 62))

    def generate() -> RolloutRound:
        rollout_id = next(rollout_ids)
        started = time.monotonic()

        async def round_() -> None:
            await controller.prepare_rollout(rollout_id)
            _release(await executor.get(rollout_id))  # the rollout function wrote the shard

        runner.run(round_())
        metadata.discard()  # the groups are in the buffer; this process trains none of them
        groups = int(miles_args.rollout_batch_size)
        seconds = time.monotonic() - started
        emit("rl_convertible_rollout", node=name, rollout_id=rollout_id, groups=groups,
             seconds=round(seconds, 3))
        return RolloutRound(groups, seconds)

    def run_stint(incarnation: int, resume_from: Any, emit_event: Callable[..., None]) -> Any:
        store.reload()  # the newest cut may come from another node
        driver = current["driver"] = build_driver()
        controller_ = getattr(getattr(driver.sync, "round_cuts", None), "controller", None)
        if getattr(controller_, "store", None) is None:
            raise RuntimeError("a convertible node needs round cuts (a checkpoint store and no "
                               "outer syncer) to hand the trainer state over")
        watermark = getattr(driver.sync.round_cuts, "buffer_watermark", None)
        agent = IslandTrainerAgent(
            driver, commands=CommandReader(files, incarnation), incarnation=incarnation,
            resume_from=resume_from, emit=emit_event, cuts=StoreCutLog(files),
            watermark=watermark if callable(watermark) else None, poll_s=poll_s)
        try:
            return agent.run()
        except BaseException:
            if agent.handle is None:
                raise
            return agent.handle  # carries the error; the node reports it

    def published_checksum() -> str | None:
        driver = current.get("driver")
        return None if driver is None else driver.publisher.published_checksum

    def inflight() -> int | None:
        sample = pool.load_sample()
        if sample is None or sample.get("running_requests") is None:
            return None
        return int(sample["running_requests"]) + int(sample.get("queued_requests") or 0)

    runner.run(boot())
    emit("rl_convertible_node_start", node=name, store=miles_args.yeto_rl_sample_buffer.store,
         rounds=int(miles_args.num_rollout))
    port = DriverHandoffPort(controller=controller, pool=pool, run=runner.run,
                             publisher=lambda: current["driver"].publisher)
    agent = ConvertibleNodeAgent(
        name=name, files=files, node=ConvertibleNode(NodeSpec(name, NodeClass.CONVERTIBLE), port),
        generate=generate, run_stint=run_stint, published_checksum=published_checksum,
        inflight=inflight, emit=emit, max_rounds=int(miles_args.num_rollout), poll_s=poll_s)
    return agent.run()
