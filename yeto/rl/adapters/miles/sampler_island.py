"""Sampler and trainer islands over the sample buffer on Miles (L4.6).

* :func:`run_sampler_island`: the sampler island's entry. Miles runs with
  ``--debug-rollout-only`` (rollout components only: no trainer model, no
  trainer GPU); the rollout function writes every group to the buffer
  (``BufferWriterRolloutFn``); :class:`~yeto.rl.engine.sampler_loop.SamplerLoop`
  runs the rounds and loads each policy release with :class:`EngineReleaseLoader`.
* :func:`wire_policy_releases`: on a trainer island, every successful publish
  is also released into the store (:class:`ReleasingPublisher`): a full-parameter
  trainer exports an HF checkpoint (``actor.export_hf``), a LoRA trainer only its
  adapter (:mod:`.release_plugin`), into the release directory.

Engine weight load (the order of the publisher, ``publish.MilesPublisher``):
under the controller's update lock, every engine loads the release, then the
plain policy version is stamped and read back on every engine; any failure
aborts the lock. The rollout executor then takes the version as its current one.

* full: ``update_weights_from_disk`` of the checkpoint (with the weight version);
* LoRA: the adapter is hot-swapped under the one name every rollout request
  selects (``miles_lora``): ``unload_lora_adapter`` (after the first load) then
  ``load_lora_adapter`` from the release directory. The base weights stay. After
  the lock, one warm-up generate per engine puts the adapter into the LoRA pool,
  and the engines' ``lora:*`` weight checksums are read back
  (:attr:`EngineReleaseLoader.last_checksum`); they must equal the trainer's
  (``PolicyRelease.engine_checksum``) when the release carries it.

This module imports Miles only inside :func:`run_sampler_island`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from yeto.rl.engine.policy_release import (
    FULL,
    LORA,
    PolicyFollower,
    PolicyRelease,
    PolicyReleaseStore,
)
from yeto.rl.engine.policy_versions import engine_weight_version
from yeto.rl.engine.sample_roles import SAMPLER, TRAINER

from .buffer_rollout import SampleBufferRolloutError, executor_read_cursor, open_buffer
from .publish import (
    _engine_id,
    _maybe_await,
    _stamp_engine_version,
    default_lora_warmup,
    lora_adapter_name,
    lora_mode,
    read_engine_checksums,
    weight_view_digest,
)
from .release_plugin import WRITE_LORA_ADAPTER

SAMPLER_TRAIN_PARALLEL_CONFIG = {"dp_size": 1}
BASE_POLICY_VERSION = 0  # the engines start with the base weights: policy version 0
ADAPTER_REQUEST_TIMEOUT_S = 600.0


def _buffer_config(miles_args: Any, role: str) -> Any:
    config = getattr(miles_args, "yeto_rl_sample_buffer", None)
    return config if config is not None and config.role == role else None


def release_store(config: Any, environ: Any = None) -> PolicyReleaseStore:
    from yeto.rl.engine.resume import store_for

    return PolicyReleaseStore(store_for(config.store, environ=environ))


# ---------------------------------------------------------------------- sampler side
async def _post_engine(engine: Any, endpoint: str, payload: dict[str, Any],
                       timeout_s: float = ADAPTER_REQUEST_TIMEOUT_S) -> dict[str, Any]:
    """POST to one SGLang server directly (as ``publish.default_lora_warmup``):
    the fork's API client exposes no path-based adapter load."""
    import httpx

    url = f"{str(engine.server_url).rstrip('/')}/{endpoint}"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, json=payload,
                                     headers=getattr(engine, "_headers", None) or {})
    body = response.json() if response.content else {}
    if response.status_code != 200 or not (isinstance(body, dict) and body.get("success", True)):
        message = body.get("error_message") if isinstance(body, dict) else body
        raise SampleBufferRolloutError(f"{url}: HTTP {response.status_code}: {message}")
    return body


class EngineReleaseLoader:
    """Loads a release into every rollout engine (``PolicyFollower`` load hook).

    ``run`` drives a coroutine on the island's event loop (``LoopRunner.run``).
    ``lora``: the engines serve a LoRA adapter (the module docstring); a release
    must then carry ``weights == "lora"``. ``warmup`` and ``flatten`` replace the
    warm-up generate and the checksum flattening (tests)."""

    def __init__(self, controller: Any, executor: Any, run: Callable[[Any], Any], *,
                 lora: bool = False, post: Callable[..., Any] = _post_engine,
                 warmup: Callable[[Any, str, float], Any] = default_lora_warmup,
                 flatten: Callable[[Any], list[dict[str, Any]]] | None = None,
                 warmup_timeout_s: float = 120.0) -> None:
        self.controller = controller
        self.executor = executor
        self.run = run
        self.lora = bool(lora)
        self.post = post
        self.warmup = warmup
        self.flatten = flatten
        self.warmup_timeout_s = float(warmup_timeout_s)
        self.adapter = lora_adapter_name() if self.lora else None
        self._adapter_loaded = False  # the engines hold the adapter of an earlier release
        self.last_checksum: str | None = None  # LoRA: the engines' lora:* read-back digest

    @property
    def weights(self) -> str:
        return LORA if self.lora else FULL

    def __call__(self, release: PolicyRelease, directory: Path) -> None:
        if release.weights != self.weights:
            raise SampleBufferRolloutError(
                f"release r{release.seq}-v{release.version} holds {release.weights} weights; "
                f"this sampler serves {self.weights}")
        engines = self.run(self._load(str(directory), int(release.version)))
        if self.lora:
            self.last_checksum = self.run(self._read_back_adapter(engines))
            if release.engine_checksum is not None and self.last_checksum != release.engine_checksum:
                raise SampleBufferRolloutError(
                    f"release r{release.seq}-v{release.version}: the engines serve LoRA checksum "
                    f"{self.last_checksum}, the trainer published {release.engine_checksum}")

    def stamp(self, version: int) -> None:
        """Stamp ``version`` on the engines without loading weights (the base policy)."""
        self.run(self._load(None, int(version)))

    async def _load_weights(self, engines: list[Any], path: str, version: int) -> None:
        if not self.lora:
            stamp = engine_weight_version(version)
            await asyncio.gather(*(e.update_weights_from_disk(model_path=path, weight_version=stamp)
                                   for e in engines))
            return
        if self._adapter_loaded:
            await asyncio.gather(*(self.post(e, "unload_lora_adapter", {"lora_name": self.adapter})
                                   for e in engines))
            self._adapter_loaded = False
        await asyncio.gather(*(self.post(e, "load_lora_adapter",
                                         {"lora_name": self.adapter, "lora_path": path})
                               for e in engines))
        self._adapter_loaded = True

    async def _read_back_adapter(self, engines: list[Any]) -> str | None:
        """Warm the adapter into every engine's LoRA pool, then read the checksums."""
        await asyncio.gather(*(self.warmup(e, self.adapter, self.warmup_timeout_s) for e in engines))
        bodies = await read_engine_checksums(self.controller, self.flatten)
        return weight_view_digest(bodies, lora=True)

    async def _load(self, path: str | None, version: int) -> list[Any]:
        """Load and stamp under the update lock; returns the engines."""
        info = await self.controller.start_update_weights()
        ok = False
        try:
            engines = list(info.rollout_engines)
            if not engines:
                raise SampleBufferRolloutError("no rollout engines to load the policy into")
            if path is not None:
                await self._load_weights(engines, path, version)
            ids = [_engine_id(e, i) for i, e in enumerate(engines)]
            failed, cause = await _stamp_engine_version(engines, ids, version)
            if failed:
                raise SampleBufferRolloutError(
                    f"{len(failed)}/{len(engines)} rollout engines did not take policy "
                    f"v{version} ({cause.value}): {sorted(failed)}")
            ok = True
        finally:
            if ok:
                await self.controller.end_update_weights(
                    snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
            else:
                await self.controller.abort_update_weights()
        await _maybe_await(self.executor.set_weight_version(int(version)))
        return engines


def run_sampler_island(miles_args: Any, *, environ: Any = None) -> int:
    """Run the sampler island to its data end; returns the rounds generated."""
    from miles.ray.placement_group import create_rollout_components
    from miles.utils.async_utils import Disposer
    from miles.utils.orchestration_utils import init_orchestration_script

    from yeto.rl.engine.events import append_tape_event
    from yeto.rl.engine.sampler_loop import SamplerLoop

    from . import LoopRunner
    from .entry import connect_island_ray
    from .rollout import RayMetadataSink, _release

    config = _buffer_config(miles_args, SAMPLER)
    if config is None:
        raise SampleBufferRolloutError("run_sampler_island needs --rl-sample-role sampler")
    if not getattr(miles_args, "debug_rollout_only", False):
        raise SampleBufferRolloutError("a sampler island runs Miles with --debug-rollout-only")

    def emit(event: str, **fields: Any) -> None:
        append_tape_event(miles_args, {"event": event, **fields})

    connect_island_ray(miles_args=miles_args)
    # The rollout process's metadata hook pushes each round's group records to this
    # named actor (on a trainer island the driver creates it and consumes them).
    metadata = RayMetadataSink()
    runner = LoopRunner()
    disposer = Disposer()

    async def init() -> tuple[Any, Any]:
        await disposer.__aenter__()
        init_orchestration_script(miles_args, disposer=disposer)
        controller, executor, _ = await create_rollout_components(miles_args)
        disposer.add(controller, executor)
        # Miles post-processes each round for the trainer's data-parallel layout, which a
        # trainer actor sets; a sampler has none. The buffer trainer is DP=1.
        await executor.set_train_parallel_config(SAMPLER_TRAIN_PARALLEL_CONFIG)
        return controller, executor

    async def generate(rollout_id: int) -> None:
        await controller.prepare_rollout(rollout_id)
        _release(await executor.get(rollout_id))  # the rollout function wrote the groups
        metadata.discard()  # the groups are in the buffer; nobody here reads the records

    error: BaseException | None = None
    try:
        controller, executor = runner.run(init())
        lora = lora_mode(miles_args)
        loader = EngineReleaseLoader(controller, executor, runner.run, lora=lora)
        if not lora:  # LoRA: no adapter is served before the first release (require_release)
            loader.stamp(BASE_POLICY_VERSION)  # until the first release is loaded
        follower = PolicyFollower(release_store(config, environ), loader,
                                  version=BASE_POLICY_VERSION)
        loop = SamplerLoop(
            follower=follower, buffer=open_buffer(environ), emit=emit,
            generate=lambda rollout_id: runner.run(generate(rollout_id)),
            num_rollouts=int(miles_args.num_rollout),
            max_backlog_groups=int(config.max_backlog_groups),
            min_round_s=config.min_round_s, require_release=lora,
            loaded_fields=lambda: {"served_checksum": loader.last_checksum})
        emit("rl_sampler_start", store=config.store, num_rollouts=int(miles_args.num_rollout),
             max_backlog_groups=int(config.max_backlog_groups), weights=loader.weights)
        return loop.run()
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            runner.run(disposer.__aexit__(None if error is None else type(error), error,
                                          None if error is None else error.__traceback__))
        finally:
            runner.close()


# ---------------------------------------------------------------------- trainer side
class ReleasingPublisher:
    """A publisher that also releases each published policy for the samplers.

    ``export(version, directory)`` writes the weights in the ``weights`` format;
    ``read_position()`` is the trainer's buffer read position (None before its
    reader exists). The release records the wrapped publisher's engine weight
    checksums of this publish (``last_engine_checksums``; LoRA: the ``lora:*``
    keys) when every engine reported the same one. Every other attribute is the
    wrapped publisher's."""

    def __init__(self, inner: Any, releases: PolicyReleaseStore,
                 export: Callable[[int, Path], None],
                 read_position: Callable[[], int | None], emit: Callable[..., None],
                 *, weights: str = FULL) -> None:
        self.inner = inner
        self.releases = releases
        self.export = export
        self.read_position = read_position
        self.emit = emit
        self.weights = weights

    def _engine_checksum(self) -> str | None:
        checksums = getattr(self.inner, "last_engine_checksums", None)
        if not isinstance(checksums, dict):
            return None
        return weight_view_digest(list(checksums.values()), lora=self.weights == LORA)

    def publish(self, state: Any, **kwargs: Any) -> Any:
        result = self.inner.publish(state, **kwargs)
        version = int(state.policy_version)
        position = self.read_position()
        if position is None:  # nothing read yet in this process: keep the last known position
            latest = self.releases.latest()
            position = None if latest is None else latest.buffer_selected
        release = self.releases.release(version, state.policy_tensor_hash(),
                                        lambda directory: self.export(version, directory),
                                        buffer_selected=position, weights=self.weights,
                                        engine_checksum=self._engine_checksum())
        if release is not None:
            self.emit("rl_policy_released", seq=release.seq, policy_version=version,
                      policy_hash=release.policy_hash, buffer_selected=release.buffer_selected,
                      weights=release.weights, sha256=release.sha256,
                      engine_checksum=release.engine_checksum)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def wire_policy_releases(driver: Any, *, miles_args: Any, actor_model: Any, runner: Any,
                         environ: Any = None) -> ReleasingPublisher | None:
    """A trainer island (``--rl-sample-role trainer``) releases every publish."""
    config = _buffer_config(miles_args, TRAINER)
    if config is None:
        return None
    in_executor = getattr(driver.rollout, "_in_executor", None)
    if not callable(in_executor):
        raise SampleBufferRolloutError(f"rollout pool {type(driver.rollout).__name__} cannot "
                                       "run code in the rollout executor")

    lora = lora_mode(miles_args)

    def export(version: int, directory: Path) -> None:
        if lora:
            runner.run(actor_model.run_plugin(WRITE_LORA_ADAPTER, {"path": str(directory)}))
        else:
            runner.run(actor_model.export_hf(rollout_id=int(version), path=str(directory)))

    publisher = ReleasingPublisher(driver.publisher, release_store(config, environ), export,
                                   read_position=lambda: in_executor(executor_read_cursor),
                                   emit=driver.emit, weights=LORA if lora else FULL)
    driver.publisher = publisher
    return publisher
