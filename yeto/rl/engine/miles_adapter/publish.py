"""``Publisher`` port over upstream ``update_weights`` (task 3.5, D11).

Sequence for one full publication of ``state`` (already applied to the trainer
through :class:`~.state.MilesPolicyState`):

1. optionally re-export the trainer state and require its tensor hash to equal
   ``state`` (the publisher never publishes something other than ``state``);
2. upstream ``update_weights(args, actor, rollout_executor, controller)``;
3. under the controller's update lock (``start_update_weights`` ...
   ``end_update_weights``), stamp every engine with the policy token
   ``yeto:<version>:<policy_tensor_hash>`` (SGLang weight version) and read it
   back; any engine failing either step aborts the lock and raises
   :class:`PublicationError` -- a partial publication never yields a manifest;
4. optionally collect per-engine weight checksums (upstream ``check_weights``),
   requiring one successful report per engine;
5. return ``PublicationResult(InferencePublicationManifest(full), members)``.

``target_policy_hash`` is the version-independent ``policy_tensor_hash`` (the
hash inside the policy token and the one ``IslandDriver.publish`` checks).
``members`` come from the same source as ``MilesRolloutPool.members()``
(:func:`.rollout.running_members`), intersected with the cells covered by the
update snapshot, so the driver's "every member acknowledged" check compares
like with like.

Member publication (rl-infra-spec 3.5, fork M4 + 3.5a mechanism (a)),
:meth:`MilesPublisher.publish_members`:

1. ``wait_cells_tracked`` for the new members (started by ``add_engines``);
2. upstream ``update_weights(members=cells, expected_epoch, admit_cordoned=True)``:
   only these engines load the payload; they become Serving *cordoned* (no
   router traffic) and every other engine keeps its weights and version;
3. under the update lock limited to the members, stamp the policy token and
   read it back (as in the full path);
4. payload read-back: ``check_weights(checksum)`` over every addressable engine
   must equal, engine by engine, the per-engine checksum recorded by the last
   full publication of the same token (the engines that already serve the
   verified policy); no reference, a different shape or any difference fails
   closed and the members are never admitted;
5. ``admit_cells(cells, expected_epoch, expected_weight_version=token)``;
6. ``start_commit_weight_version(token, expected_epoch)`` ...
   ``end_commit_weight_version``: under one hold of the controller lock, every
   Serving engine reports the token at that epoch. The executor's integer
   weight version (fully-async staleness only) is left to the next full
   publication.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
import inspect
import json
from collections.abc import Callable
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest

from ..ports import PublicationCause, PublicationResult
from ..trainable_state import TrainableState
from ..policy_digest import is_resident
from . import LoopRunner
from .rollout import cells_of, member_id, policy_token, running_members


class PublicationError(RuntimeError):
    """``cause`` says why the publication was refused (A4 E1-B); ``engine_ids`` are
    the engines that disagree (payload read-back / token / failed update)."""

    def __init__(self, message: str, failed_members: frozenset[str] = frozenset(), *,
                 cause: PublicationCause = PublicationCause.OTHER,
                 engine_ids: Any = ()) -> None:
        super().__init__(message)
        self.failed_members = failed_members
        self.cause = PublicationCause(cause)
        self.engine_ids = tuple(sorted(str(e) for e in engine_ids))


# Test-only fault injection (plan-3.8-4.4-v2 §4, watchdog case): block this many
# seconds before the FIRST member ``update_weights`` of the process (see
# ``MilesPublisher._injected_block``). Unset (default) = no effect. Set by the
# launcher's ``--rl-test-inject-update-weights-block-s``.
INJECT_UPDATE_BLOCK_ENV = "YETO_RL_TEST_INJECT_UPDATE_WEIGHTS_BLOCK_S"
# Test-only (plan.md E1-B, 3.5): the FIRST member update_weights of the process
# ships a LoRA adapter perturbed by this amount (the trainer's adapter is
# perturbed for that one member-scoped update and restored right after), so the
# new engines hold different LoRA weights than the published policy and
# check_weights must refuse to admit them. (A base-checkpoint reload did not
# change the LoRA weights the checksum covers: A4 E1-B on Nebius.)
INJECT_LORA_PERTURB_ENV = "YETO_RL_TEST_INJECT_LORA_PERTURB"
# Test-only (A4 E1-B (a)): the FIRST member publication of the process stays this many
# seconds after ``end_update_weights`` registered the new members as cordoned and before
# ``check_weights`` / ``admit_cells`` (an observation window for "new engines are
# cordoned, not routed"). Only honoured in test-injection mode, i.e. when another
# YETO_RL_TEST_* injection variable is set too; otherwise ignored with a warning.
HOLD_BEFORE_CHECK_ENV = "YETO_RL_TEST_HOLD_BEFORE_CHECK_S"


def in_test_injection_mode(environ: Any = None) -> bool:
    """True when any ``YETO_RL_TEST_*`` injection (other than the hold itself) is set."""
    import os

    env = os.environ if environ is None else environ
    return any(k.startswith("YETO_RL_TEST_") and k != HOLD_BEFORE_CHECK_ENV and env[k] not in ("", "0")
               for k in env)


def hold_before_check(environ: Any = None) -> float | None:
    import os

    env = os.environ if environ is None else environ
    raw = env.get(HOLD_BEFORE_CHECK_ENV)
    if raw in (None, ""):
        return None
    value = float(raw)
    if not value > 0:
        raise ValueError(f"{HOLD_BEFORE_CHECK_ENV} must be a positive number of seconds")
    if not in_test_injection_mode(env):
        import sys

        print(f"[yeto] {HOLD_BEFORE_CHECK_ENV} ignored: not in test-injection mode "
              "(no other YETO_RL_TEST_* injection is set)", file=sys.stderr, flush=True)
        return None
    return value


def injected_update_block(environ: Any = None) -> float | None:
    import os

    raw = (os.environ if environ is None else environ).get(INJECT_UPDATE_BLOCK_ENV)
    if raw in (None, ""):
        return None
    value = float(raw)
    if not value > 0:
        raise ValueError(f"{INJECT_UPDATE_BLOCK_ENV} must be a positive number of seconds")
    return value


class TargetWorkersLostError(RuntimeError):
    """A27: a target engine's workers died while the member ``update_weights`` ran
    (the fork's liveness scan tore its cell down). ``cells`` names the dead cell(s)."""

    def __init__(self, message: str, cells: list[str]) -> None:
        super().__init__(message)
        self.cells = list(cells)


class InjectedBlockProbeError(RuntimeError):
    """The liveness probe of the injected block failed for a reason other than a
    dead/replaced target worker: NOT evidence of a kill (review M2)."""


class RayTargetLiveness:
    """Liveness of the target generation for the injected block (review M2).

    ``snapshot(cells)`` records each target worker's (name, generation) when the
    block starts; ``status()`` then reports ``"alive"`` or ``"dead: <why>"``:
    a cell with no workers (stopped), a changed worker set / generation
    (restarted by the health monitor) or a killed actor all count as the
    target generation being dead. Any other error raises
    :class:`InjectedBlockProbeError` (logged, reported separately).
    """

    def __init__(self, *, manager: Any = None, ray_module: Any = None) -> None:
        self._manager = manager
        self._ray = ray_module
        self.targets: dict[str, list[tuple[str, int]]] = {}
        self.dead_cell: str | None = None  # the cell the last "dead: ..." status was about

    def _deps(self) -> tuple[Any, Any]:
        if self._ray is None:
            import ray

            self._ray = ray
        if self._manager is None:
            from miles.utils.workers.ray_worker_manager import RayWorkerManager

            self._manager = RayWorkerManager.get_handle()
        return self._manager, self._ray

    async def snapshot(self, cells: list[str]) -> None:
        manager, _ = self._deps()
        self.targets = {c: sorted((i.name, int(i.generation))
                                  for i in await manager.get_worker_infos.remote(c))
                        for c in cells}
        empty = [c for c, t in self.targets.items() if not t]
        if empty:
            raise InjectedBlockProbeError(f"target cells {empty} have no workers at block start")

    async def status(self) -> str:
        manager, ray = self._deps()
        actor_died = getattr(getattr(ray, "exceptions", None), "RayActorError", ())
        cell = None
        try:
            for cell, targets in self.targets.items():
                now = sorted((i.name, int(i.generation))
                             for i in await manager.get_worker_infos.remote(cell))
                if not now:
                    self.dead_cell = cell
                    return f"dead: cell {cell} has no workers (stopped)"
                if now != targets:
                    self.dead_cell = cell
                    return f"dead: cell {cell} workers/generation changed {targets} -> {now}"
                for name, generation in targets:
                    handle = await manager.get_actor_handle.remote(
                        name, expected_generation=generation)
                    await asyncio.wait_for(handle.__ray_ready__.remote(), timeout=10)
        except actor_died as exc:  # type: ignore[misc]
            self.dead_cell = cell
            return f"dead: actor died ({type(exc).__name__})"
        except Exception as exc:  # noqa: BLE001 - not a kill: report it as such
            import sys

            print(f"[yeto] TEST INJECTION probe error (not a kill): {exc!r}", file=sys.stderr,
                  flush=True)
            raise InjectedBlockProbeError(f"liveness probe failed: {exc!r}") from exc
        return "alive"


def _payload_of(state: Any) -> tuple[str, int]:
    """rl-publish-fastpath: a trainer-resident state carries the payload digest the
    trainer computed (same definition as :func:`payload_digest`)."""

    return state.payload_digest if is_resident(state) else payload_digest(state)


def payload_digest(state: TrainableState) -> tuple[str, int]:
    """SHA256 and byte count of the canonical FP32 tensor payload."""

    import torch

    digest = hashlib.sha256(b"yeto-rl-publication-payload-v1\0")
    total = 0
    for name in state.tensor_names:
        value = state.tensors[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        raw = memoryview(value.numpy()).cast("B")
        digest.update(raw)
        total += raw.nbytes
    return digest.hexdigest(), total


def _member(cell_id: str) -> str:
    return member_id(cell_id)


def _engine_id(engine: Any, index: int) -> str:
    return str(getattr(engine, "server_url", None) or f"engine{index}")


def _default_update_weights() -> Callable[..., Any]:
    from miles.ray.placement_group import update_weights

    return update_weights


def lora_keys(body: dict[str, Any]) -> dict[str, Any]:
    """The adapter A/B entries of one engine's flattened checksum body. The patched
    sglang WeightChecker emits ``lora:<adapter>:<module>:<layer>:A|B``; the Miles
    flatten prefixes ``rank{r}/``. The stock checksum covers only base weights."""
    return {k: v for k, v in body.items() if k.startswith("lora:") or "/lora:" in k}


def lora_mode(args: Any) -> bool:
    """Mirror of miles.utils.lora.utils.is_lora_enabled (not importable on CPU)."""
    return (getattr(args, "lora_rank", 0) or 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def lora_adapter_name() -> str:
    """The single adapter name Miles registers on every engine (``miles_lora``)."""
    try:
        from miles.utils.lora.utils import LORA_ADAPTER_NAME

        return str(LORA_ADAPTER_NAME)
    except Exception:  # noqa: BLE001 - CPU (no miles): the fork's constant
        return "miles_lora"


async def default_lora_warmup(engine: Any, adapter: str, timeout_s: float) -> dict[str, Any]:
    """Make ``engine`` (an SGLang HTTP server, ``server_url``) load ``adapter`` into
    its LoRA memory pool: one direct ``/generate`` of a single token under
    ``lora_path=adapter``.

    Why: the fork's streamed adapter apply (``apply_streamed_adapter`` ->
    ``_load_lora_adapter_from_tensors``) stores a never-served adapter only in the
    CPU-side ``loras`` dict; the GPU pool slot is assigned lazily by
    ``prepare_lora_batch`` on the first forward that selects it. The read-back
    (``check_weights`` -> ``lora_pool_checksums``) hashes the pool, so a fresh
    member reports no ``lora:*`` keys until it has served one request. This call
    goes to the engine directly (never through the Miles router, where the member
    is still cordoned), so it is no rollout sample and enters no ledger."""
    import httpx

    url = f"{str(engine.server_url).rstrip('/')}/generate"
    headers = getattr(engine, "_headers", None) or {}
    payload = {"text": "warmup", "lora_path": adapter,
               "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}}
    started = time.time()
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    body = response.json()
    meta = (body[0] if isinstance(body, list) and body else body).get("meta_info", {}) \
        if isinstance(body, (list, dict)) else {}
    return {"url": url, "status": response.status_code, "seconds": round(time.time() - started, 3),
            "completion_tokens": meta.get("completion_tokens")}


def _default_flatten() -> Callable[[Any], list[dict[str, Any]]]:
    from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums

    return flatten_inference_engine_checksums


async def _maybe_await(value: Any) -> Any:
    """The trainer hook may be sync (tests, fakes) or a coroutine (real Miles
    policy state, which must not block the running loop)."""
    return await value if inspect.isawaitable(value) else value


class MilesPublisher:
    def __init__(
        self,
        *,
        args: Any,
        actor_model: Any,
        rollout_executor: Any,
        inference_controller: Any,
        export_trainer_state: Callable[[], TrainableState] | None = None,
        verify_engine_checksums: bool = True,
        update_weights: Callable[..., Any] | None = None,
        flatten_checksums: Callable[[Any], list[dict[str, Any]]] | None = None,
        runner: LoopRunner | None = None,
    ) -> None:
        self._args = args
        self._actor = actor_model
        self._executor = rollout_executor
        self._controller = inference_controller
        self._export = export_trainer_state
        self._verify_checksums = verify_engine_checksums
        self._update_weights = update_weights
        self._flatten = flatten_checksums
        self._runner = runner or LoopRunner()
        self.last_engine_checksums: dict[str, dict[str, Any]] | None = None
        self._published_once = False
        # (token, per-engine checksum body) of the last full publication whose
        # engines all reported the same body: the payload reference (3.5).
        self._reference: tuple[str, dict[str, Any]] | None = None
        self.track_timeout_s = 600.0
        # Test-only block before the FIRST member update_weights (INJECT_UPDATE_BLOCK_ENV).
        self._inject_block = injected_update_block()
        self.injected_blocks: list[float] = []
        # test-only: (event, **fields) -> structured journal + tape record; wired by
        # compose_island to IslandController.record_test_event. None = not recorded.
        self.event_sink: Callable[..., None] | None = None
        self._hold_before_check = hold_before_check()
        self.holds: list[tuple[float, float]] = []
        self.liveness_probe: Any = None  # RayTargetLiveness-like (snapshot/status)
        self.block_poll_s = 1.0
        # A27: the member update_weights is watched against the target cells' workers (fork
        # RayWorkerManager, not behind the InferenceController lock the update holds); a dead
        # target fails the publish (-> REBUILD_OLD) instead of waiting for the NCCL rendezvous.
        self.watch_targets = True
        self.target_watch_poll_s = 2.0
        # how long a dead target's update_weights may still fail on its own (the fork's connect
        # fails fast on a dead engine) before it is cancelled; see _update_weights_watching_targets
        self.dead_target_grace_s = 30.0
        # LoRA mode: load the adapter into each fresh member's GPU pool before the
        # read-back (see default_lora_warmup); None = the default direct /generate.
        self.lora_warmup: Callable[[Any, str, float], Any] | None = None
        self.lora_warmup_timeout_s = 120.0
        self.last_lora_warmups: list[dict[str, Any]] = []
        import os

        raw = os.environ.get(INJECT_LORA_PERTURB_ENV)
        self._inject_perturb = float(raw) if raw else None
        self.injected_perturbations: list[tuple[tuple[str, ...], float]] = []
        # (scale | None) -> None: perturb the trainer's LoRA adapter by ``scale``,
        # or restore it exactly (None); wired by compose_island
        self.perturb_trainer: Any = None

    def _check_trainer_holds(self, state: Any) -> str:
        """The trainer must still hold the policy being published (same check as
        before). rl-publish-fastpath: for a trainer-resident state the trainer re-hashes
        its current weights in place and only the digest crosses Ray."""

        tensor_hash = state.policy_tensor_hash()
        if is_resident(state):
            current = state.recheck().policy_tensor_hash
        elif self._export is not None:
            current = self._export().policy_tensor_hash()
        else:
            return tensor_hash
        if current != tensor_hash:
            raise PublicationError("trainer weights differ from the state requested for publication")
        return tensor_hash

    def publish(self, state: TrainableState, *, token_rollout_id: int | None = None) -> PublicationResult:
        tensor_hash = self._check_trainer_holds(state)
        payload_hash, payload_bytes = _payload_of(state)
        token = policy_token(
            state.policy_version if token_rollout_id is None else token_rollout_id, tensor_hash
        )
        members, checksums = self._runner.run(self._publish(token))
        self.last_engine_checksums = checksums
        bodies = list((checksums or {}).values())
        self._reference = (
            (token, bodies[0]) if bodies and all(b == bodies[0] for b in bodies) else None
        )
        manifest_hash = hashlib.sha256(
            json.dumps(
                {
                    "token": token,
                    "policy_tensor_hash": tensor_hash,
                    "payload_hash": payload_hash,
                    "members": sorted(members),
                    "engine_checksums": checksums,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        manifest = InferencePublicationManifest(
            publication_mode="full",
            base_policy_version=None,
            target_policy_version=state.policy_version,
            target_policy_hash=tensor_hash,
            target_manifest_hash=manifest_hash,
            payload_hash=payload_hash,
            payload_bytes=payload_bytes,
            complete=True,
        )
        return PublicationResult(manifest=manifest, members=frozenset(members))

    async def _publish(self, token: str) -> tuple[frozenset[str], dict[str, Any] | None]:
        update_weights = self._update_weights or _default_update_weights()
        offload_rollout = bool(getattr(self._args, "offload_rollout", False))
        try:
            # Upstream train.py: weights come back before every update except
            # the first (engines start resident); KV comes back after it.
            if offload_rollout and self._published_once:
                await self._controller.onload_weights()
            await update_weights(self._args, self._actor, self._executor, self._controller)
            if offload_rollout:
                await self._controller.onload_kv()
        except Exception as exc:
            raise PublicationError(f"upstream update_weights failed: {exc}",
                                   cause=PublicationCause.UPDATE_FAILED) from exc
        self._published_once = True

        info = await self._controller.start_update_weights()
        ok = False
        try:
            engines = list(info.rollout_engines)
            updated = frozenset(_member(c) for c in info.snapshot_cell_id_to_hashes)
            members = updated & await running_members(self._controller)
            if not engines or not members:
                raise PublicationError("no rollout engines are eligible for publication")
            ids = [_engine_id(e, i) for i, e in enumerate(engines)]
            results = await asyncio.gather(
                *(e.update_weight_version(token) for e in engines), return_exceptions=True
            )
            failed = {ids[i] for i, r in enumerate(results) if isinstance(r, BaseException)}
            cause = PublicationCause.UPDATE_FAILED
            if not failed:
                versions = await asyncio.gather(
                    *(e.get_weight_version() for e in engines), return_exceptions=True
                )
                failed = {
                    ids[i]
                    for i, v in enumerate(versions)
                    if isinstance(v, BaseException) or str(v) != token
                }
                cause = PublicationCause.TOKEN_MISMATCH
            if failed:
                raise PublicationError(
                    f"{len(failed)}/{len(engines)} rollout engines did not acknowledge {token}: "
                    f"{sorted(failed)}",
                    frozenset(failed), cause=cause, engine_ids=failed,
                )
            ok = True
        finally:
            if ok:
                await self._controller.end_update_weights(
                    snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes
                )
            else:
                await self._controller.abort_update_weights()

        checksums = None
        if self._verify_checksums:
            raw = await self._controller.check_weights(action="checksum")
            try:
                bodies = (self._flatten or _default_flatten())(raw)
            except AssertionError as exc:
                raise PublicationError(f"engine weight checksum failed: {exc}",
                                       cause=PublicationCause.PAYLOAD_MISMATCH) from exc
            if len(bodies) != len(engines):
                raise PublicationError(
                    f"{len(engines) - len(bodies)} engines returned no weight checksum"
                )
            checksums = {ids[i]: dict(sorted(b.items())) for i, b in enumerate(bodies)}
        return members, checksums


    # -- member publication (3.5) ---------------------------------------------
    def publish_members(
        self,
        state: TrainableState,
        members: frozenset[str],
        *,
        epoch: int,
        token_rollout_id: int | None = None,
    ) -> PublicationResult:
        if not members:
            raise PublicationError("member publication needs at least one member")
        if bool(getattr(self._args, "offload_rollout", False)):
            raise PublicationError("member publication needs a partitioned rollout (no offload)")
        tensor_hash = self._check_trainer_holds(state)
        token = policy_token(
            state.policy_version if token_rollout_id is None else token_rollout_id, tensor_hash
        )
        if not self._verify_checksums:
            raise PublicationError("member publication needs the payload read-back (checksums)")
        if self._reference is None or self._reference[0] != token:
            raise PublicationError(
                f"no verified payload reference for {token}: publish it fully first"
            )
        payload_hash, payload_bytes = _payload_of(state)
        cells = cells_of(members)
        checksums = self._runner.run(self._publish_members(token, cells, epoch))
        manifest_hash = hashlib.sha256(
            json.dumps(
                {"token": token, "policy_tensor_hash": tensor_hash, "payload_hash": payload_hash,
                 "members": sorted(members), "epoch": epoch, "engine_checksums": checksums},
                sort_keys=True, separators=(",", ":"), default=str,
            ).encode("utf-8")
        ).hexdigest()
        manifest = InferencePublicationManifest(
            publication_mode="full",
            base_policy_version=None,
            target_policy_version=state.policy_version,
            target_policy_hash=tensor_hash,
            target_manifest_hash=manifest_hash,
            payload_hash=payload_hash,
            payload_bytes=payload_bytes,
            complete=True,
        )
        return PublicationResult(manifest=manifest, members=frozenset(members))

    def verify_serving_policy(
        self, *, epoch: int, token_rollout_id: int, state: TrainableState
    ) -> None:
        """Every Serving engine reports the policy token at ``epoch`` (locked check)."""
        token = policy_token(token_rollout_id, state.policy_tensor_hash())
        self._runner.run(self._commit_version(token, epoch))

    async def _commit_version(self, token: str, epoch: int) -> None:
        # fork context_lock: start_commit_weight_version is @acquires_lock -- on
        # failure it has already released the lock itself, so end (which
        # releases) is called only after a successful start.
        try:
            await self._controller.start_commit_weight_version(
                weight_version=token, expected_epoch=epoch
            )
        except Exception as exc:
            raise PublicationError(f"serving engines do not all report {token}: {exc}",
                                   cause=PublicationCause.TOKEN_MISMATCH) from exc
        await self._controller.end_commit_weight_version()

    def _record(self, event: str, **fields: Any) -> None:
        """Test-injection record on the journal and tape (when wired)."""
        if self.event_sink is None:
            return
        self.event_sink(event, **fields)

    async def _update_weights_watching_targets(self, update_weights: Callable[..., Any],
                                               cells: list[str], epoch: int) -> Any:
        """Run the member ``update_weights`` while watching the target cells' worker actors.

        A27 (GPU 6r1/6r2 d2): a target engine SIGKILLed during VERIFYING never joins the
        trainer's NCCL weight-update group; the fork's call then waits for the rendezvous (pg
        timeout, 30 min by default) holding the InferenceController lock, so the watchdog's kill
        of the other targets unblocks nothing and the run stalls. Here the targets are probed every
        ``target_watch_poll_s`` through the fork's ``RayWorkerManager`` (a named actor, not behind
        that lock). When one is dead the call gets ``dead_target_grace_s`` to fail on its own (with
        the fork's bounded connect it does) and is cancelled otherwise; the fork's
        ``placement_group.update_weights`` runs ``abort_update_weights`` on any BaseException,
        CancelledError included, which releases the controller lock. A probe that cannot run
        (no manager, older fork) leaves the call unwatched: the watch never invents a failure.
        Residual race (documented): a cancel landing while the fork's ``start_update_weights``
        is still acquiring the lock would leave that remote acquire running; the grace period
        makes this window practically unreachable (the acquire is sub-second once the cells are
        tracked, which ``wait_cells_tracked`` ensured before)."""
        call = update_weights(
            self._args, self._actor, self._executor, self._controller,
            members=cells, expected_epoch=epoch, admit_cordoned=True,
        )
        if not self.watch_targets:
            return await call
        import sys

        probe = self.liveness_probe or RayTargetLiveness()
        try:
            await probe.snapshot(cells)
        except Exception as exc:  # noqa: BLE001 - unwatched, never a fabricated failure
            print(f"[yeto] member update_weights of {cells} runs unwatched: target probe unavailable "
                  f"({exc!r})", file=sys.stderr, flush=True)
            return await call
        task = asyncio.ensure_future(call)
        dead: str | None = None
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=self.target_watch_poll_s)
                if task.done():
                    break
                try:
                    state = await probe.status()
                except Exception as exc:  # noqa: BLE001 - stop watching, keep waiting
                    print(f"[yeto] target probe failed; member update_weights of {cells} continues "
                          f"unwatched ({exc!r})", file=sys.stderr, flush=True)
                    return await task
                if state != "alive":
                    dead = state
                    break
            if dead is None:
                return await task
            lost = [c for c in [getattr(probe, "dead_cell", None)] if c]
            print(f"[yeto] target engine of {cells} {dead} during member update_weights; waiting up to "
                  f"{self.dead_target_grace_s}s for the call to fail, then cancelling it",
                  file=sys.stderr, flush=True)
            self._record("target_workers_lost", target_members=[member_id(c) for c in cells],
                         lost_members=[member_id(c) for c in lost], state=dead, ts=time.time())
            done, _ = await asyncio.wait({task}, timeout=self.dead_target_grace_s)
            how = "the call failed on its own"
            if not done:
                task.cancel()
                how = "the call was cancelled"
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise TargetWorkersLostError(
                f"target engine of {cells} {dead} during member update_weights ({how})", lost)
        finally:
            if not task.done():
                task.cancel()

    async def _injected_block(self, cells: list[str], seconds: float) -> None:
        """TEST ONLY (plan-3.8-4.4-v2 §4): stand in for an ``update_weights`` blocked
        on the new engines. Blocks up to ``seconds``; every ``block_poll_s`` it
        checks that the target cells' worker actors are alive and fails as soon
        as one is dead (as the real call fails when its engine is killed). It
        emulates the blocked call on the yeto side; it is not a hang inside SGLang."""
        import sys

        self.injected_blocks.append(seconds)
        print(f"[yeto] TEST INJECTION {INJECT_UPDATE_BLOCK_ENV}: blocking {seconds}s before "
              f"update_weights({cells})", file=sys.stderr, flush=True)
        members = [member_id(c) for c in cells]
        self._record("test_injection", kind="block_update", target_members=members,
                     seconds=seconds, reached_ts=time.time(), applied=True)
        outcome = "elapsed"
        try:
            probe = self.liveness_probe or RayTargetLiveness()
            await probe.snapshot(cells)
            loop = asyncio.get_running_loop()
            deadline = loop.time() + seconds
            while loop.time() < deadline:
                await asyncio.sleep(min(self.block_poll_s, max(0.0, deadline - loop.time())))
                state = await probe.status()
                if state != "alive":
                    outcome = f"target {state}"
                    raise RuntimeError(f"target engine of {cells} {state} during the injected block")
        except BaseException:
            outcome = outcome if outcome != "elapsed" else "error"
            raise
        finally:
            self._record("test_injection", kind="block_update", target_members=members,
                         seconds=seconds, released_ts=time.time(), outcome=outcome, applied=True)

    async def _hold(self, cells: list[str], seconds: float) -> None:
        """TEST ONLY: observation window of E1-B (a). Awaited, so the event loop (the
        fork's controller, health checks, the watchdog's liveness probe) keeps running."""
        import sys

        started = time.time()
        self.holds.append((started, started))
        members = [member_id(c) for c in cells]
        print(f"[yeto] TEST INJECTION {HOLD_BEFORE_CHECK_ENV}: holding {seconds}s before "
              f"check_weights/admit_cells of {cells}", file=sys.stderr, flush=True)
        self._record("test_hold", target_members=members, seconds=seconds, stage="start",
                     start_ts=started)
        try:
            await asyncio.sleep(seconds)
        finally:
            ended = time.time()
            self.holds[-1] = (started, ended)
            self._record("test_hold", target_members=members, seconds=seconds, stage="end",
                         start_ts=started, end_ts=ended)

    async def _warm_up_lora(self, engines: list[Any], ids: list[str], cells: list[str],
                            members: list[str]) -> None:
        """Load the published adapter into every member engine's LoRA pool before the
        read-back (defect 6: lazily loaded adapters are invisible to check_weights).
        A warm-up that fails leaves the adapter unverifiable: refused as
        LORA_UNVERIFIABLE (never admitted, never mistaken for a payload mismatch)."""
        adapter = lora_adapter_name()
        warmup = self.lora_warmup or default_lora_warmup
        results = await asyncio.gather(
            *(warmup(e, adapter, self.lora_warmup_timeout_s) for e in engines),
            return_exceptions=True)
        report = []
        for engine_id, r in zip(ids, results, strict=True):
            ok = not isinstance(r, BaseException)
            report.append({"engine": engine_id, "ok": ok,
                           **(dict(r) if ok and isinstance(r, dict) else {}),
                           **({} if ok else {"error": repr(r)[:300]})})
        self.last_lora_warmups = report
        self._record("lora_warmup", target_members=members, adapter=adapter, engines=report,
                     applied=all(r["ok"] for r in report))
        failed = [r["engine"] for r in report if not r["ok"]]
        if failed:
            raise PublicationError(
                f"LoRA adapter warm-up failed on {failed}: the adapter cannot be read back, "
                "new members are not admitted",
                frozenset(members), cause=PublicationCause.LORA_UNVERIFIABLE, engine_ids=failed)

    async def _publish_members(
        self, token: str, cells: list[str], epoch: int
    ) -> dict[str, Any] | None:
        update_weights = self._update_weights or _default_update_weights()
        try:
            await self._controller.wait_cells_tracked(cells, timeout_seconds=self.track_timeout_s)
            if self._inject_block is not None and not self.injected_blocks:
                await self._injected_block(cells, self._inject_block)
            perturb = self._inject_perturb is not None and not self.injected_perturbations
            if perturb:
                if self.perturb_trainer is None:
                    self._record("test_injection", kind="lora_perturb",
                                 target_members=[member_id(c) for c in cells],
                                 scale=self._inject_perturb, applied=False,
                                 error="no trainer hook")
                    raise RuntimeError("LoRA perturbation injection has no trainer hook")
                import sys

                print(f"[yeto] TEST INJECTION {INJECT_LORA_PERTURB_ENV}: member update of "
                      f"{cells} ships a LoRA adapter perturbed by {self._inject_perturb}",
                      file=sys.stderr, flush=True)
                self.injected_perturbations.append((tuple(cells), self._inject_perturb))
                try:
                    await _maybe_await(self.perturb_trainer(self._inject_perturb))
                except BaseException as exc:
                    # not applied: the record must say so (a refusal then is NOT evidence)
                    self._record("test_injection", kind="lora_perturb",
                                 target_members=[member_id(c) for c in cells],
                                 scale=self._inject_perturb, applied=False, error=repr(exc))
                    raise
                self._record("test_injection", kind="lora_perturb",
                             target_members=[member_id(c) for c in cells],
                             scale=self._inject_perturb, applied=True)
            try:
                await self._update_weights_watching_targets(update_weights, cells, epoch)
            finally:
                if perturb:
                    await _maybe_await(self.perturb_trainer(None))  # the trainer holds the published policy again
        except TargetWorkersLostError as exc:
            lost = [member_id(c) for c in exc.cells] or [member_id(c) for c in cells]
            raise PublicationError(f"member update_weights failed: {exc}", frozenset(lost),
                                   cause=PublicationCause.UPDATE_FAILED, engine_ids=lost) from exc
        except Exception as exc:
            raise PublicationError(f"member update_weights failed: {exc}",
                                   frozenset(member_id(c) for c in cells),
                                   cause=PublicationCause.UPDATE_FAILED,
                                   engine_ids=[member_id(c) for c in cells]) from exc

        info = await self._controller.start_update_weights(members=cells, expected_epoch=epoch)
        ok = False
        try:
            engines = list(info.rollout_engines)
            snapshot = sorted(info.snapshot_cell_id_to_hashes)
            if snapshot != sorted(cells) or len(engines) != len(cells):
                raise PublicationError(f"update lock covers {snapshot}, expected {sorted(cells)}")
            ids = [_engine_id(e, i) for i, e in enumerate(engines)]
            results = await asyncio.gather(
                *(e.update_weight_version(token) for e in engines), return_exceptions=True
            )
            failed = {ids[i] for i, r in enumerate(results) if isinstance(r, BaseException)}
            cause = PublicationCause.UPDATE_FAILED
            if not failed:
                versions = await asyncio.gather(
                    *(e.get_weight_version() for e in engines), return_exceptions=True
                )
                failed = {ids[i] for i, v in enumerate(versions)
                          if isinstance(v, BaseException) or str(v) != token}
                cause = PublicationCause.TOKEN_MISMATCH
            if failed:
                raise PublicationError(f"members did not acknowledge {token}: {sorted(failed)}",
                                       frozenset(failed), cause=cause, engine_ids=failed)
            ok = True
        finally:
            if ok:
                await self._controller.end_update_weights(
                    snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes
                )
            else:
                await self._controller.abort_update_weights()

        # Independent identity of the member engines of this publication (A4 judge:
        # the new engines' URLs come from the fork's UpdatableEngines, not from the
        # router listing).
        members = [member_id(c) for c in cells]
        self._record("member_engines", target_members=members, engine_urls=ids, epoch=epoch,
                     token=token)
        if lora_mode(self._args) and self._verify_checksums:
            await self._warm_up_lora(engines, ids, cells, members)

        if self._hold_before_check is not None and not self.holds:
            await self._hold(cells, self._hold_before_check)

        checksums = None
        if self._verify_checksums:
            raw = await self._controller.check_weights(action="checksum")
            try:
                bodies = (self._flatten or _default_flatten())(raw)
            except AssertionError as exc:
                raise PublicationError(f"engine weight checksum failed: {exc}",
                                       cause=PublicationCause.PAYLOAD_MISMATCH) from exc
            assert self._reference is not None
            reference = self._reference[1]
            expected_engines = len(self.last_engine_checksums or {}) + len(cells)
            if len(bodies) < len(cells) or len(bodies) > expected_engines:
                raise PublicationError(
                    f"{len(bodies)} engines reported checksums; expected up to {expected_engines}",
                    cause=PublicationCause.PAYLOAD_MISMATCH,
                )
            if lora_mode(self._args):
                # fail closed: without adapter keys in BOTH the reference and every body, a
                # perturbed adapter is indistinguishable from the published one
                ref_lora = lora_keys(reference)
                blind = [i for i, body in enumerate(bodies) if not lora_keys(body)]
                per_engine = []
                for i, body in enumerate(bodies):
                    mine = lora_keys(body)
                    per_engine.append({
                        "engine": f"engine{i}", "keys": len(body), "lora_keys": len(mine),
                        "lora_keys_differ": sum(1 for k, v in mine.items() if ref_lora.get(k) != v)
                        + sum(1 for k in ref_lora if k not in mine)})
                # target-state evidence for the judge (A4 E1-B): what each engine's pool
                # actually holds, not what the trainer sent
                self._record("lora_readback", target_members=[member_id(c) for c in cells],
                             reference_lora_keys=len(ref_lora), engines=per_engine,
                             blind=[f"engine{i}" for i in blind])
                if not ref_lora or blind:
                    raise PublicationError(
                        "LoRA mode but the engine read-back carries no adapter (lora:*) checksums "
                        f"({'reference' if not ref_lora else 'engines'} blind); the adapter cannot be "
                        "verified, new members are not admitted",
                        cause=PublicationCause.LORA_UNVERIFIABLE,
                        engine_ids=[f"engine{i}" for i in (blind or range(len(bodies)))],
                    )
            bad = [i for i, body in enumerate(bodies) if dict(sorted(body.items())) != reference]
            if bad:
                lora_bad = sorted({i for i in bad if lora_keys(bodies[i]) != lora_keys(reference)})
                raise PublicationError(
                    f"payload read-back differs from the published policy on {len(bad)} engines"
                    + (f" (adapter A/B differs on {len(lora_bad)})" if lora_bad else "")
                    + "; new members are not admitted",
                    cause=PublicationCause.PAYLOAD_MISMATCH,
                    engine_ids=[f"engine{i}" for i in bad],  # index in the check_weights read-back
                )
            checksums = {"engines": len(bodies), "reference_token": token}
        await self._controller.admit_cells(cells, expected_epoch=epoch,
                                           expected_weight_version=token)
        await self._commit_version(token, epoch)
        return checksums
