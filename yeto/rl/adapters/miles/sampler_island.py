"""Sampler and trainer islands over the sample buffer on Miles (L4.6).

* :func:`run_sampler_island`: the sampler island's entry. Miles runs with
  ``--debug-rollout-only`` (rollout components only: no trainer model, no
  trainer GPU); the rollout function writes every group to the buffer
  (``BufferWriterRolloutFn``); :class:`~yeto.rl.engine.sampler_loop.SamplerLoop`
  runs the rounds and loads each policy release with :class:`EngineReleaseLoader`.
* :func:`wire_policy_releases`: on a trainer island (and on a convertible node
  while it trains), every successful publish
  is also released into the store (:class:`ReleasingPublisher`): a full-parameter
  trainer exports an HF checkpoint (``actor.export_hf``), a LoRA trainer only its
  adapter (:mod:`.release_plugin`), into the release directory. Synced trainers
  (several islands and a syncer): only learner 0 releases, with the merged
  version (:mod:`yeto.rl.engine.merged_release`).

  Staged full releases (S27 G-AT, the default on a trainer island; off with
  ``--rl-release-inline``): the trainer exports into a local staging directory
  (every rank writes a share of the files, ``release_plugin.write_full_checkpoint``)
  and goes on training while a background thread uploads it to the store
  (``PolicyReleaseStore.release_staged``). The next release waits for that
  upload, so releases stay in order and at most one is in flight; the driver
  waits for the last one before the run ends (``drain_releases``). The release
  records the trainer's buffer read position after the weights are uploaded, so
  it includes the round the trainer took meanwhile: a sampler at its backlog
  cap starts its next round with this release instead of the next one.

  The base policy: the samplers' engines start with the base weights (version
  0), so a non-synced full-parameter trainer releases nothing for version 0 while
  the store holds no release (``rl_policy_release_skipped``, reason
  ``base_policy``). Synced trainers release it (a restarted trainer restores its
  published versions from the store's releases).

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

LoRA staging (S26 E5): the adapter is copied to local disk before the update
lock. With ``fused_stage`` (``--rl-sampler-fused-stage``, default off) the copy
also computes the release digest, so the release is read from the store once
instead of twice (the follower's check, then the copy); a digest mismatch fails
the load before any engine sees the files, as the follower's check does.
``last_timings`` holds the seconds of each load phase.

This module imports Miles only inside :func:`run_sampler_island`.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from yeto.rl.engine.policy_release import (
    FULL,
    LORA,
    BackgroundReleaser,
    DeltaChain,
    PhaseTimer,
    PolicyFollower,
    PolicyRelease,
    PolicyReleaseStore,
    check_release_digest,
    copy_with_digest,
    ReleaseNotLoaded,
)
from yeto.rl.engine.merged_release import MergedReleaseGuard, is_publisher, release_version
from yeto.rl.engine.policy_versions import engine_weight_version
from yeto.rl.engine.release_delta import DeltaMirror, default_mirror
from yeto.rl.engine.sample_roles import CONVERTIBLE, SAMPLER, TRAINER
from yeto.rl.quant.schemes import BF16, FP8_FFN_CHANNEL, release_dir_for
from yeto.rl.quant.scope import QuantScopeError
from yeto.rl.quant.tiers import TIER_TRAINER_FP8, check_release_scope, describe, release_scheme

from .buffer_rollout import (
    SampleBufferRolloutError,
    executor_read_cursor,
    executor_read_shards,
    executor_set_gate_version,
    open_buffer,
)
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
from .release_plugin import WRITE_LORA_ADAPTER, full_checkpoint_writer
from .remote_code_prewarm import prewarm_remote_code

SAMPLER_TRAIN_PARALLEL_CONFIG = {"dp_size": 1}
BASE_POLICY_VERSION = 0  # the engines start with the base weights: policy version 0
STAGING_ENV = "YETO_RL_RELEASE_STAGING"  # local directory of staged releases (default: system temp)
# "miles": a staged release is written by Miles' export_hf (rank 0 writes) instead of
# every rank (release_plugin.write_full_checkpoint); a fallback that needs no code change
STAGED_EXPORT_ENV = "YETO_RL_RELEASE_EXPORT"
ADAPTER_REQUEST_TIMEOUT_S = 600.0
# A fixed greedy request each LoRA load answers (rl_sampler_policy_loaded.probe):
# its token log-probabilities show that the served adapter changed with the version.
PROBE_PROMPT = "Question: Natalia sold 48 clips in April and half as many in May. How many clips did she sell in total?\nAnswer:"
PROBE_TOKENS = 16


def _buffer_config(miles_args: Any, role: str) -> Any:
    config = getattr(miles_args, "yeto_rl_sample_buffer", None)
    return config if config is not None and config.role == role else None


def release_store(config: Any, environ: Any = None, *, scheme: str = BF16,
                  mirror: DeltaMirror | None = None) -> PolicyReleaseStore:
    from yeto.rl.engine.resume import store_for

    return PolicyReleaseStore(store_for(config.store, environ=environ),
                              release_dir=release_dir_for(scheme),
                              cached_latest=not config.release_legacy, mirror=mirror)


def _quant_tier(miles_args: Any) -> int:
    return int(getattr(miles_args, "yeto_rollout_quant_tier", 0) or 0)


def prepare_quantized_base(miles_args: Any, work_dir: Path) -> Path:
    """S26 E6 tier 2, sampler side: the engines start on the FP8 conversion of the BF16
    base (``miles_args.hf_checkpoint``), so later FP8 releases load into FP8 parameters
    of the same layout. The conversion is checked against the scope rule. Returns the
    converted directory, which replaces ``miles_args.hf_checkpoint``."""
    from yeto.rl.quant.awq import AwqSmoothing
    from yeto.rl.quant.requant import requantize_checkpoint, save_scoped_template

    base = Path(miles_args.hf_checkpoint)
    template, out = work_dir / "fp8-template", work_dir / "fp8-base"
    save_scoped_template(base, template, FP8_FFN_CHANNEL)
    stats = requantize_checkpoint(base, template, out, AwqSmoothing.none())
    check_release_scope(out)
    print(f"[rl] {describe(TIER_TRAINER_FP8)}: base {base} -> {out} {stats}", flush=True)
    miles_args.hf_checkpoint = str(out)
    return out


def scope_checked(load: Callable[[PolicyRelease, Path], None]) -> Callable[[PolicyRelease, Path], None]:
    """A ``PolicyFollower`` load hook that refuses an FP8 release outside the scope rule
    before the engines are touched (S26 E6 tier 2 post-load check, step one)."""
    def hook(release: PolicyRelease, directory: Path) -> None:
        try:
            check_release_scope(Path(directory))
        except QuantScopeError as error:
            raise ReleaseNotLoaded(f"r{release.seq}-v{release.version}: {error}",
                                   reason="release_refused:quant_scope") from error
        load(release, directory)

    return hook


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


async def probe_generate(engine: Any, adapter: str, timeout_s: float = 120.0) -> dict[str, Any]:
    """One greedy :data:`PROBE_PROMPT` request under ``adapter``, sent to the engine
    directly: the output text digest and the sum of its token log-probabilities."""
    body = await _post_engine(engine, "generate", {
        "text": PROBE_PROMPT, "lora_path": adapter, "return_logprob": True,
        "sampling_params": {"max_new_tokens": PROBE_TOKENS, "temperature": 0.0}}, timeout_s)
    body = body[0] if isinstance(body, list) else body
    logprobs = [float(t[0]) for t in body.get("meta_info", {}).get("output_token_logprobs") or []
                if t and t[0] is not None]
    text = str(body.get("text", ""))
    return {"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "text": text[:120], "logprob_sum": round(sum(logprobs), 6), "tokens": len(logprobs)}


class EngineReleaseLoader:
    """Loads a release into every rollout engine (``PolicyFollower`` load hook).

    ``run`` drives a coroutine on the island's event loop (``LoopRunner.run``).
    ``lora``: the engines serve a LoRA adapter (the module docstring); a release
    must then carry ``weights == "lora"``. ``adapter_loaded``: the engines already
    serve the adapter (a convertible node: its trainer pushed it at boot), so the
    first load unloads it first. ``warmup`` and ``flatten`` replace the warm-up
    generate and the checksum flattening (tests). ``fused_stage``: the LoRA staging
    copy checks the release digest (module doc; ``verifies_files``)."""

    def __init__(self, controller: Any, executor: Any, run: Callable[[Any], Any], *,
                 lora: bool = False, post: Callable[..., Any] = _post_engine,
                 warmup: Callable[[Any, str, float], Any] = default_lora_warmup,
                 flatten: Callable[[Any], list[dict[str, Any]]] | None = None,
                 probe: Callable[[Any, str], Any] | None = probe_generate,
                 warmup_timeout_s: float = 120.0, staging_root: str | None = None,
                 adapter_loaded: bool = False, fused_stage: bool = False,
                 timer: Callable[[], float] = time.perf_counter) -> None:
        self.controller = controller
        self.executor = executor
        self.run = run
        self.lora = bool(lora)
        self.post = post
        self.warmup = warmup
        self.flatten = flatten
        self.probe = probe
        self.last_probe: dict[str, Any] | None = None  # LoRA: the first engine's probe answer
        self.warmup_timeout_s = float(warmup_timeout_s)
        self.adapter = lora_adapter_name() if self.lora else None
        # the engines hold an adapter under ``self.adapter`` (an earlier release, or the
        # adapter a colocated trainer pushed: a convertible node)
        self._adapter_loaded = bool(adapter_loaded) and self.lora
        self.staging_root = staging_root  # local copies of loaded adapters (None: system temp)
        self._staged: Path | None = None  # the local copy the engines serve
        self.last_checksum: str | None = None  # LoRA: the engines' lora:* read-back digest
        self.fused_stage = bool(fused_stage) and self.lora
        self.timer = timer
        self._timing: PhaseTimer | None = None  # the load in progress
        self.last_timings: dict[str, float] = {}  # seconds per phase of the last load

    @property
    def weights(self) -> str:
        return LORA if self.lora else FULL

    @property
    def verifies_files(self) -> bool:
        """True: this loader checks the release digest (``PolicyFollower`` skips its check)."""
        return self.fused_stage

    @contextmanager
    def _phase(self, name: str) -> Iterator[None]:
        if self._timing is None:
            yield
            return
        with self._timing.phase(name):
            yield

    def __call__(self, release: PolicyRelease, directory: Path) -> None:
        if release.weights != self.weights:
            raise SampleBufferRolloutError(
                f"release r{release.seq}-v{release.version} holds {release.weights} weights; "
                f"this sampler serves {self.weights}")
        self._timing = PhaseTimer(self.timer)
        try:
            self._call(release, directory)
        finally:
            self.last_timings = self._timing.as_dict()
            self._timing = None

    def _call(self, release: PolicyRelease, directory: Path) -> None:
        if not self.lora:
            self.run(self._load(str(directory), int(release.version)))
            return
        staged = self._stage(release, directory)
        try:
            engines = self.run(self._load(str(staged), int(release.version)))
        except BaseException:
            if self._staged != staged:  # no engine took this copy
                shutil.rmtree(staged, ignore_errors=True)
            raise
        self.last_checksum = self.run(self._read_back_adapter(engines))
        if release.engine_checksum is not None and self.last_checksum != release.engine_checksum:
            raise SampleBufferRolloutError(
                f"release r{release.seq}-v{release.version}: the engines serve LoRA checksum "
                f"{self.last_checksum}, the trainer published {release.engine_checksum}")

    def _stage(self, release: PolicyRelease, directory: Path) -> Path:
        """A local copy of the adapter (SGLang keeps the adapter files open: open
        files on a Modal Volume make every later reload fail, T-LSMP g). With
        ``fused_stage`` the copy also checks the release digest."""
        staged = Path(tempfile.mkdtemp(prefix=f"v{int(release.version)}-", dir=self.staging_root))
        try:
            with self._phase("stage_s"):
                if self.fused_stage:
                    check_release_digest(release, copy_with_digest(Path(directory), staged))
                else:
                    shutil.copytree(directory, staged, dirs_exist_ok=True)
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        return staged

    def stamp(self, version: int) -> None:
        """Stamp ``version`` on the engines without loading weights (the base policy)."""
        self.run(self._load(None, int(version)))

    async def _load_weights(self, engines: list[Any], path: str, version: int) -> None:
        if not self.lora:
            stamp = engine_weight_version(version)
            await asyncio.gather(*(e.update_weights_from_disk(model_path=path, weight_version=stamp)
                                   for e in engines))
            return
        # ``path`` is the local copy (``_stage``); the previous copy is removed once its
        # adapter is unloaded. On a failure before the load the caller removes ``path``.
        if self._adapter_loaded:
            await asyncio.gather(*(self.post(e, "unload_lora_adapter", {"lora_name": self.adapter})
                                   for e in engines))
            self._adapter_loaded = False
            self._drop_staged()
        await asyncio.gather(*(self.post(e, "load_lora_adapter",
                                         {"lora_name": self.adapter, "lora_path": path})
                               for e in engines))
        self._adapter_loaded = True
        self._staged = Path(path)

    def _drop_staged(self) -> None:
        if self._staged is not None:
            shutil.rmtree(self._staged, ignore_errors=True)
            self._staged = None

    async def _read_back_adapter(self, engines: list[Any]) -> str | None:
        """Warm the adapter into every engine's LoRA pool, then read the checksums."""
        with self._phase("warmup_s"):
            await asyncio.gather(*(self.warmup(e, self.adapter, self.warmup_timeout_s) for e in engines))
        with self._phase("checksum_s"):
            bodies = await read_engine_checksums(self.controller, self.flatten)
        if self.probe is not None:
            with self._phase("probe_s"):
                self.last_probe = dict(await self.probe(engines[0], self.adapter))
        return weight_view_digest(bodies, lora=True)

    async def _load(self, path: str | None, version: int) -> list[Any]:
        """Load and stamp under the update lock; returns the engines."""
        with self._phase("lock_s"):
            info = await self.controller.start_update_weights()
        ok = False
        try:
            engines = list(info.rollout_engines)
            if not engines:
                raise SampleBufferRolloutError("no rollout engines to load the policy into")
            if path is not None:
                with self._phase("engine_load_s"):
                    await self._load_weights(engines, path, version)
            ids = [_engine_id(e, i) for i, e in enumerate(engines)]
            with self._phase("stamp_s"):
                failed, cause = await _stamp_engine_version(engines, ids, version)
            if failed:
                raise SampleBufferRolloutError(
                    f"{len(failed)}/{len(engines)} rollout engines did not take policy "
                    f"v{version} ({cause.value}): {sorted(failed)}")
            ok = True
        finally:
            with self._phase("unlock_s"):
                if ok:
                    await self.controller.end_update_weights(
                        snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)
                else:
                    await self.controller.abort_update_weights()
        await _maybe_await(self.executor.set_weight_version(int(version)))
        return engines


LEAVE_HOLD_S = 300.0  # after a reclaim LEAVE: wait this long for the cloud to stop the container


def run_sampler_island(miles_args: Any, *, environ: Any = None,
                       sleep: Callable[[float], None] = time.sleep) -> int:
    """Run the sampler island to its data end; returns the rounds generated.

    A replacement incarnation (S25 C1) continues after the newest round its
    island has in the buffer (``sampler_resume``), loads the newest policy
    release before its first round, and writes ``rl_sampler_join`` / a member
    record. On the reclaim marker (Modal SIGINT/SIGTERM, written by the Modal
    runner) it starts no new round, writes no more groups, writes
    ``rl_sampler_leave`` with reason PREEMPTED and waits for the cloud to stop
    the container (at most :data:`LEAVE_HOLD_S`)."""
    from miles.ray.placement_group import create_rollout_components
    from miles.utils.async_utils import Disposer
    from miles.utils.orchestration_utils import init_orchestration_script

    from yeto.rl.pool_client import LeaveReason
    from yeto.rl.engine.events import append_tape_event
    from yeto.rl.engine.resume import store_for
    from yeto.rl.engine.sampler_loop import SamplerLoop
    from yeto.rl.engine.sampler_resume import (
        RECLAIM_MARKER_ENV,
        TRAINERS_ENDED_EVENT,
        ReclaimWatcher,
        SamplerMembership,
        acknowledge_reclaim,
        clear_reclaim_files,
        install_marker_signals,
        new_incarnation,
        reclaim_requested,
        resume_point,
        stop_when_trainers_ended,
    )

    from . import LoopRunner
    from .buffer_rollout import INCARNATION_ENV, SAMPLER_ID_ENV
    from .entry import connect_island_ray
    from .rollout import RayMetadataSink, _release

    config = _buffer_config(miles_args, SAMPLER)
    if config is None:
        raise SampleBufferRolloutError("run_sampler_island needs --rl-sample-role sampler")
    if not getattr(miles_args, "debug_rollout_only", False):
        raise SampleBufferRolloutError("a sampler island runs Miles with --debug-rollout-only")
    env = os.environ if environ is None else environ

    def emit(event: str, **fields: Any) -> None:
        append_tape_event(miles_args, {"event": event, **fields})

    # one incarnation id for this process and its rollout workers (forwarded to Ray)
    sampler_id = env.get(SAMPLER_ID_ENV)
    if not sampler_id:
        raise SampleBufferRolloutError(f"{SAMPLER_ID_ENV} is not set")
    incarnation = env[INCARNATION_ENV] = new_incarnation()
    clear_reclaim_files(env)  # a restarted machine keeps its /tmp
    if env.get(RECLAIM_MARKER_ENV):  # set by the launcher on a pair's sampler island
        install_marker_signals(env)
    buffer = open_buffer(env, sampler=True)
    resume = resume_point(buffer.entries(), sampler_id)
    store = store_for(config.store, environ=environ)
    membership = SamplerMembership(store, sampler_id, incarnation, resume, emit)
    # a full-parameter synced group's sampler: no fixed rounds, until its trainers ended
    until = int(config.until_trainers_end or 0)
    trainers_ended = stop_when_trainers_ended(store, until)
    progress = {"rollout_id": None, "rounds": 0}  # the round in flight, rounds done here

    if _quant_tier(miles_args) == TIER_TRAINER_FP8:
        if lora_mode(miles_args):
            raise SampleBufferRolloutError("rollout quant tier 2 needs full-parameter releases")
        prepare_quantized_base(miles_args, Path(env.get("YETO_RL_WORK_DIR") or Path.home() / "yeto-rl"))
    connect_island_ray(miles_args=miles_args)
    # S27 G-CK: fill the remote-code module cache on every node before any engine starts
    prewarm_remote_code(miles_args, emit=lambda event: append_tape_event(miles_args, event))
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
        progress["rollout_id"] = rollout_id
        await controller.prepare_rollout(rollout_id)
        _release(await executor.get(rollout_id))  # the rollout function wrote the groups
        metadata.discard()  # the groups are in the buffer; nobody here reads the records
        progress["rounds"] += 1
        progress["rollout_id"] = None

    def leave(reason: LeaveReason, **detail: Any) -> None:
        membership.leave(reason.name, rounds=progress["rounds"],
                         policy_version=None if follower is None else follower.version,
                         in_flight_rollout_id=progress["rollout_id"], **detail)

    error: BaseException | None = None
    follower = None
    reclaimed = lambda: reclaim_requested(env)  # noqa: E731

    def leave_reclaimed() -> None:
        leave(LeaveReason.PREEMPTED)
        acknowledge_reclaim(env)  # a host hook (Nebius) lets the VM stop once the LEAVE is stored

    watcher = ReclaimWatcher(reclaimed, leave_reclaimed).start()
    dispense = {"installed": False}

    def close_dispense() -> None:
        """S27 P4: give the rows this sampler holds back to the dispenser now
        (a reclaimed sampler, a finished one), not at the end of its lease."""
        if not dispense["installed"]:
            return
        dispense["installed"] = False
        try:
            emit("rl_dispense_close", **(_in_executor(executor, runner.run, "close_source") or {}))
        except Exception as close_error:  # noqa: BLE001 - the lease returns them anyway
            emit("rl_dispense_close", error=f"{type(close_error).__name__}: {close_error}")

    try:
        controller, executor = runner.run(init())
        if getattr(miles_args, "yeto_rl_data_dispense", None):
            # S27 P4: before the first round, the data source claims its rows
            from yeto.protocol import island_hmac_key_from_env

            installed = _in_executor(executor, runner.run, "install",
                                     dict(miles_args.yeto_rl_data_dispense), island_hmac_key_from_env())
            dispense["installed"] = True
            emit("rl_dispense_install", sampler=sampler_id,
                 dispense_island_id=int(miles_args.yeto_rl_data_dispense["learner_id"]), **dict(installed))
        lora = lora_mode(miles_args)
        loader = EngineReleaseLoader(controller, executor, runner.run, lora=lora,
                                     fused_stage=bool(config.fused_stage))
        if not lora:  # LoRA: no adapter is served before the first release (require_release)
            loader.stamp(BASE_POLICY_VERSION)  # until the first release is loaded
        scheme = release_scheme(_quant_tier(miles_args))
        follower = PolicyFollower(release_store(config, environ, scheme=scheme,
                                                mirror=default_mirror(environ)),
                                  loader if scheme == BF16 else scope_checked(loader),
                                  version=BASE_POLICY_VERSION, scheme=scheme)
        loop = SamplerLoop(
            follower=follower, buffer=buffer, emit=emit,
            generate=lambda rollout_id: runner.run(generate(rollout_id)),
            num_rollouts=None if until else int(miles_args.num_rollout),
            max_backlog_groups=int(config.max_backlog_groups),
            min_round_s=config.min_round_s, require_release=lora,
            should_stop=lambda: reclaimed() or trainers_ended(),
            loaded_fields=lambda: {"served_checksum": loader.last_checksum,
                                   "probe": loader.last_probe,
                                   "load_timings": loader.last_timings,
                                   "fused_stage": loader.fused_stage})
        latest = follower.releases.latest()
        membership.join(latest_release=None if latest is None else
                        {"seq": latest.seq, "policy_version": latest.version})
        emit("rl_sampler_start", store=config.store, num_rollouts=loop.num_rollouts,
             until_trainers_end=until,
             max_backlog_groups=int(config.max_backlog_groups), weights=loader.weights,
             start_rollout_id=resume.next_rollout_id)
        if buffer.data_ended:  # an earlier incarnation finished and marked the data end
            rounds = 0
        else:
            rounds = loop.run(start=resume.next_rollout_id)
        close_dispense()
        if reclaimed():
            leave_reclaimed()  # the LEAVE is a no-op when the watcher already left
            _hold_until_stopped(sleep, LEAVE_HOLD_S)
        else:
            if trainers_ended():
                emit(TRAINERS_ENDED_EVENT, trainers=until, rounds=rounds)
            leave(LeaveReason.FINISHED)
        return rounds
    except BaseException as exc:
        error = exc
        close_dispense()
        try:
            leave(LeaveReason.PREEMPTED if reclaimed() else LeaveReason.INFRA_FAILED,
                  error=f"{type(exc).__name__}: {exc}"[:300])
        except Exception:  # noqa: BLE001 - the original error is the one to report
            pass
        raise
    finally:
        watcher.stop()
        try:
            runner.run(disposer.__aexit__(None if error is None else type(error), error,
                                          None if error is None else error.__traceback__))
        finally:
            runner.close()


def _in_executor(executor: Any, run: Callable[[Any], Any], name: str, *fn_args: Any) -> Any:
    """Run ``adapters.miles.task_dispense.<name>(executor, ...)`` inside the rollout
    executor's process (``__ray_call__``), as ``MilesRolloutPool`` does for a
    training island (S27 P4)."""
    from . import task_dispense
    from .rollout import _awaited, _executor_target

    fn = getattr(task_dispense, name)
    kind, target = _executor_target(executor)
    if kind == "ray":
        return run(_awaited(target.__ray_call__.remote(fn, *fn_args)))
    if kind == "local":
        return fn(executor, *fn_args)
    raise SampleBufferRolloutError(f"task dispenser: cannot reach the rollout executor "
                                   f"{type(executor).__name__}")


def _hold_until_stopped(sleep: Callable[[float], None], hold_s: float) -> None:
    """After a reclaim LEAVE the process stays up until the cloud stops the
    container: an early clean exit could end the cloud's input as finished
    instead of letting the cloud run the island again."""
    print(f"[sampler-island] reclaimed: waiting up to {hold_s:g} s for the container to stop",
          flush=True)
    sleep(float(hold_s))


# ---------------------------------------------------------------------- trainer side
def mark_trainer_end(miles_args: Any, *, learner_id: int, environ: Any = None) -> bool:
    """A trainer that reads the sample buffer ended its run normally: write its end
    record into its store, so the samplers that run until their trainers end stop
    (``sample_roles`` module doc). True when written; any other island writes
    nothing. A failed write is printed, not raised: the run itself is done."""
    config = _buffer_config(miles_args, TRAINER)
    if config is None:
        return False
    from yeto.rl.engine.resume import store_for
    from yeto.rl.engine.sampler_resume import write_trainer_end

    try:
        write_trainer_end(store_for(config.store, environ=environ),
                          str(int(learner_id)))
    except Exception as error:  # noqa: BLE001 - the samplers then stop at their own limits
        print(f"[sampler-island] trainer end record not written: {type(error).__name__}: {error}",
              flush=True)
        return False
    return True


class ReleasingPublisher:
    """A publisher that also releases each published policy for the samplers.

    ``export(version, directory)`` writes the weights in the ``weights`` format;
    ``read_position()`` is the trainer's buffer read position (None before its
    reader exists). The release records the wrapped publisher's engine weight
    checksums of this publish (``last_engine_checksums``; LoRA: the ``lora:*``
    keys) when every engine reported the same one. Every other attribute is the
    wrapped publisher's.

    ``staging``: stage each release in a new directory under it and upload it on
    ``releaser`` (module doc, staged full releases); None: write it into the
    store on the caller's thread. ``skip_base``: release nothing for the base
    policy while the store holds no release (module doc)."""

    def __init__(self, inner: Any, releases: PolicyReleaseStore,
                 export: Callable[[int, Path], None],
                 read_position: Callable[[], int | None], emit: Callable[..., None],
                 *, weights: str = FULL,
                 read_shards: Callable[[], dict[str, int] | None] | None = None,
                 version_of: Callable[[Any], int] | None = None,
                 guard: MergedReleaseGuard | None = None,
                 origin: Callable[[Any], dict[str, Any]] | None = None,
                 staging: Path | None = None, releaser: BackgroundReleaser | None = None,
                 skip_base: bool = False,
                 read_position_from_thread: Callable[[], int | None] | None = None,
                 read_shards_from_thread: Callable[[], dict[str, int] | None] | None = None,
                 delta: DeltaChain | None = None,
                 timer: Callable[[], float] = time.perf_counter) -> None:
        """``delta``: staged releases go out as bf16 deltas along this chain
        (``policy_release`` module doc, delta releases). ``read_shards``: the
        trainer's read position per sampler shard (None
        without merged shards), released with every policy for the samplers'
        backlog caps. Synced trainers (:mod:`yeto.rl.engine.merged_release`):
        ``version_of(state)`` is the merged version to release (default: the
        policy version), ``guard`` keeps the release versions increasing and
        ``origin(state)`` is the release record's ``origin``.
        ``read_position_from_thread`` / ``read_shards_from_thread``: the same reads,
        callable from the upload thread (default: ``read_position`` / ``read_shards``)."""
        if (staging is None) != (releaser is None):
            raise ValueError("staging and releaser go together")
        if delta is not None and staging is None:
            raise ValueError("delta releases need staged releases")
        self.delta = delta
        self.read_position_from_thread = read_position_from_thread or read_position
        self.read_shards_from_thread = read_shards_from_thread or read_shards
        self.version_of = version_of or (lambda state: int(state.policy_version))
        self.guard = guard
        self.origin = origin
        self.inner = inner
        self.releases = releases
        self.export = export
        self.read_position = read_position
        self.read_shards = read_shards
        self.emit = emit
        self.weights = weights
        self.staging = None if staging is None else Path(staging)
        self.releaser = releaser
        self.skip_base = bool(skip_base)
        self.timer = timer

    def _engine_checksum(self) -> str | None:
        checksums = getattr(self.inner, "last_engine_checksums", None)
        if not isinstance(checksums, dict):
            return None
        return weight_view_digest(list(checksums.values()), lora=self.weights == LORA)

    def publish(self, state: Any, **kwargs: Any) -> Any:
        started = self.timer()
        result = self.inner.publish(state, **kwargs)
        publish_s = round(self.timer() - started, 3)  # the trainer's own engines
        # an upload in flight ends first: the store is read below, and releases stay in order
        waited_s = 0.0 if self.releaser is None else round(self.releaser.wait(), 3)
        version = int(self.version_of(state))
        if self.guard is not None:
            latest = self.releases.latest()
            if self.guard.check(version, None if latest is None else latest.version) == "stale":
                self.emit("rl_policy_release_skipped", policy_version=version,
                          local_version=int(state.policy_version),
                          latest_version=None if latest is None else latest.version,
                          reason="not_newer_than_latest")
                return result
        if self.skip_base and version == BASE_POLICY_VERSION and self.releases.latest() is None:
            self.emit("rl_policy_release_skipped", policy_version=version,
                      local_version=int(state.policy_version), latest_version=None,
                      reason="base_policy")
            return result
        fields = {"weights": self.weights, "engine_checksum": self._engine_checksum(),
                  "origin": None if self.origin is None else self.origin(state)}
        policy_hash = state.policy_tensor_hash()
        event = {"policy_version": version, "local_version": int(state.policy_version),
                 "publish_s": publish_s}
        if self.staging is None:
            position, shards = self._positions(self.read_position, self.read_shards)
            release = self.releases.release(version, policy_hash,
                                            lambda directory: self.export(version, directory),
                                            buffer_selected=position, buffer_shards=shards, **fields)
            self._emit_released(release, event, self.releases.last_timings)
            return result
        staged = self._stage(version)
        export_started = self.timer()
        try:
            self.export(version, staged)
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        stage = {"wait_previous_s": waited_s, "stage_export_s": round(self.timer() - export_started, 3)}

        def positions() -> tuple[int | None, dict[str, int] | None]:
            return self._positions(self.read_position_from_thread, self.read_shards_from_thread)

        base = None if self.delta is None else self.delta.base()

        def upload() -> None:
            kept = False
            try:
                # S27 G-CD: the full releases of a delta chain carry each file's sha256
                release = self.releases.release_staged(version, policy_hash, staged,
                                                       position=positions, delta_base=base,
                                                       digest_full=self.delta is not None, **fields)
                kept = self.delta is not None and self.delta.advance(release, staged)
            finally:
                if not kept:
                    shutil.rmtree(staged, ignore_errors=True)
            delta = None if release is None else release.delta
            self._emit_released(release, {**event, "staged": True,
                                          "visible_at": self.releases.last_visible_at,
                                          "delta_base_seq": None if delta is None else delta["base_seq"],
                                          "delta_ratio": None if delta is None else delta["ratio"]},
                                {**stage, **self.releases.last_timings})

        self.releaser.submit(upload)
        return result

    def _positions(self, read_position: Callable[[], int | None],
                   read_shards: Callable[[], dict[str, int] | None] | None
                   ) -> tuple[int | None, dict[str, int] | None]:
        """The trainer's buffer read position (and per shard) for a release record."""
        position = read_position()
        shards = read_shards() if read_shards is not None and position is not None else None
        if position is None:  # nothing read yet in this process: keep the last known positions
            latest = self.releases.latest()
            position = None if latest is None else latest.buffer_selected
            shards = None if latest is None else latest.buffer_shards
        return position, shards

    def _stage(self, version: int) -> Path:
        """A new, empty staging directory for ``version`` (earlier ones of it removed)."""
        self.staging.mkdir(parents=True, exist_ok=True)
        for old in self.staging.glob(f"v{int(version)}-*"):
            shutil.rmtree(old, ignore_errors=True)
        return Path(tempfile.mkdtemp(prefix=f"v{int(version)}-", dir=self.staging))

    def _emit_released(self, release: PolicyRelease | None, event: dict[str, Any],
                       timings: dict[str, float]) -> None:
        if release is None:
            return
        self.emit("rl_policy_released", seq=release.seq, origin=release.origin,
                  policy_hash=release.policy_hash, buffer_selected=release.buffer_selected,
                  weights=release.weights, sha256=release.sha256,
                  engine_checksum=release.engine_checksum, released_at=release.released_at,
                  release_timings=timings, latest_source=self.releases.last_latest_source,
                  release_bytes=sum((release.files or {}).values()), **event)

    def drain_releases(self) -> None:
        """Wait for the staged release in flight (the driver calls it before the run
        ends); raises its error. No-op without staging."""
        if self.releaser is not None:
            self.releaser.drain()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class GateVersionPublisher:
    """A synced trainer's publisher (:mod:`yeto.rl.engine.merged_release`): after
    each publish, the island's buffer reader gates ``max_policy_age`` on the merged
    version (``set_gate_version(version)``). Every other attribute is the wrapped
    publisher's."""

    def __init__(self, inner: Any, set_gate_version: Callable[[int], Any],
                 version_of: Callable[[Any], int]) -> None:
        self.inner = inner
        self.set_gate_version = set_gate_version
        self.version_of = version_of

    def publish(self, state: Any, **kwargs: Any) -> Any:
        result = self.inner.publish(state, **kwargs)
        self.set_gate_version(int(self.version_of(state)))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def wire_policy_releases(driver: Any, *, miles_args: Any, actor_model: Any, runner: Any,
                         environ: Any = None) -> ReleasingPublisher | None:
    """A trainer island (``--rl-sample-role trainer``) releases every publish; so does
    the trainer of a convertible node (the other nodes follow the releases).

    Synced trainers (a buffer lane on ``miles_args.yeto_rl_sample_lane``): every
    island gates on the merged version; only the publisher (learner id 0) releases,
    with the merged version (:mod:`yeto.rl.engine.merged_release`). On its own lane
    (its samplers follow it) a trainer is the publisher of its sub-store and
    releases and gates with its local version."""
    config = _buffer_config(miles_args, TRAINER) or _buffer_config(miles_args, CONVERTIBLE)
    if config is None:
        return None
    in_executor = getattr(driver.rollout, "_in_executor", None)
    if not callable(in_executor):
        raise SampleBufferRolloutError(f"rollout pool {type(driver.rollout).__name__} cannot "
                                       "run code in the rollout executor")
    lane = getattr(miles_args, "yeto_rl_sample_lane", None)
    synced: dict[str, Any] = {}
    if lane is not None:
        def version_of(state: Any) -> int:
            return release_version(driver, state, lane)

        def origin(state: Any) -> dict[str, Any]:
            return {"kind": "local" if lane.is_own else "merged", "learner_id": lane.index,
                    "num_learners": lane.count, "local_version": int(state.policy_version)}

        def on_gap(previous: int, version: int) -> None:
            driver.emit("rl_policy_release_gap", previous_version=previous, policy_version=version)

        # S27 L11: the lane cursor log resumes a relaunched synced trainer, not
        # the data source; and it accepts groups made with the versions the
        # store's releases name (a publisher of an earlier process released them).
        driver.data_position_owner = "buffer_lane"
        releases = release_store(config, environ)
        driver.published_policy_history = lambda: [
            (r.version, r.policy_hash) for r in releases.records() if r.derived_from is None]
        driver.publisher = GateVersionPublisher(
            driver.publisher, lambda v: in_executor(executor_set_gate_version, int(v)), version_of)
        publisher_island = is_publisher(lane.index, lane.count)
        driver.emit("rl_policy_release_role", learner_id=lane.index, num_learners=lane.count,
                    publisher=publisher_island, own_lane=lane.is_own)
        if not publisher_island:
            return None
        synced = {"version_of": version_of, "guard": MergedReleaseGuard(on_gap), "origin": origin}

    lora = lora_mode(miles_args)
    env = os.environ if environ is None else environ
    # module doc: a trainer island stages its full releases unless --rl-release-inline
    staged = config.role == TRAINER and not lora and not config.release_inline
    all_ranks_write = staged and env.get(STAGED_EXPORT_ENV, "") != "miles"

    # local disk: every rank writes a share of the files
    write_full = full_checkpoint_writer(actor_model, runner, all_ranks=all_ranks_write)

    def export(version: int, directory: Path) -> None:
        if lora:
            runner.run(actor_model.run_plugin(WRITE_LORA_ADAPTER, {"path": str(directory)}))
        else:
            write_full(version, directory)

    staging = None
    if staged:
        staging = Path(env.get(STAGING_ENV) or Path(tempfile.gettempdir()) / "yeto-release-staging")
        shutil.rmtree(staging, ignore_errors=True)  # an earlier process's partial stages
    # the upload thread may not drive the driver's event loop (it may be running)
    from_thread = getattr(driver.rollout, "_in_executor_from_thread", None) or in_executor
    delta_every = int(getattr(config, "release_delta_full_every", 0) or 0)
    publisher = ReleasingPublisher(driver.publisher, release_store(config, environ), export,
                                   read_position=lambda: in_executor(executor_read_cursor),
                                   read_shards=lambda: in_executor(executor_read_shards),
                                   emit=driver.emit, weights=LORA if lora else FULL,
                                   staging=staging,
                                   releaser=None if staging is None else BackgroundReleaser(),
                                   skip_base=not lora and lane is None,
                                   read_position_from_thread=lambda: from_thread(executor_read_cursor),
                                   read_shards_from_thread=lambda: from_thread(executor_read_shards),
                                   delta=DeltaChain(delta_every) if staged and delta_every else None,
                                   **synced)
    driver.publisher = publisher
    if _quant_tier(miles_args) == TIER_TRAINER_FP8:
        start_trainer_releaser(miles_args, config, environ)
    return publisher


def start_trainer_releaser(miles_args: Any, config: Any, environ: Any = None) -> Any:
    """S26 E6 tier 2, trainer side: a thread of this driver converts every BF16 release
    into an FP8 release (``yeto.rl.quant.release.scoped_releaser``) for the sampler islands."""
    import threading

    from yeto.rl.engine.resume import store_for
    from yeto.rl.quant.release import scoped_releaser, start_scoped_releaser

    if lora_mode(miles_args):
        raise SampleBufferRolloutError("rollout quant tier 2 needs full-parameter releases")
    env = os.environ if environ is None else environ
    template = Path(env.get("YETO_RL_WORK_DIR") or Path.home() / "yeto-rl") / "fp8-template"
    releaser = scoped_releaser(store_for(config.store, environ=environ), Path(miles_args.hf_checkpoint),
                               template, FP8_FFN_CHANNEL)
    return start_scoped_releaser(releaser, threading.Event())
