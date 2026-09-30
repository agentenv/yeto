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
import hashlib
import json
from collections.abc import Callable
from typing import Any

from yeto.rl.contracts import InferencePublicationManifest

from ..ports import PublicationResult
from ..trainable_state import TrainableState
from . import LoopRunner
from .rollout import cells_of, member_id, policy_token, running_members


class PublicationError(RuntimeError):
    def __init__(self, message: str, failed_members: frozenset[str] = frozenset()) -> None:
        super().__init__(message)
        self.failed_members = failed_members


# Test-only fault injection (plan-3.8-4.4-v2 §4, watchdog case): block this many
# seconds before the FIRST member ``update_weights`` of the process (see
# ``MilesPublisher._injected_block``). Unset (default) = no effect. Set by the
# launcher's ``--rl-test-inject-update-weights-block-s``.
INJECT_UPDATE_BLOCK_ENV = "YETO_RL_TEST_INJECT_UPDATE_WEIGHTS_BLOCK_S"
# Test-only (plan.md E1-B, 3.5): after the FIRST member update_weights of the
# process, reload ONE new engine (lowest cell id) from this checkpoint path via
# SGLang's /update_weights_from_disk, so its weights differ from the published
# policy; check_weights must then refuse to admit the new members.
INJECT_WEIGHT_OVERRIDE_ENV = "YETO_RL_TEST_INJECT_WEIGHT_OVERRIDE_PATH"


def injected_update_block(environ: Any = None) -> float | None:
    import os

    raw = (os.environ if environ is None else environ).get(INJECT_UPDATE_BLOCK_ENV)
    if raw in (None, ""):
        return None
    value = float(raw)
    if not value > 0:
        raise ValueError(f"{INJECT_UPDATE_BLOCK_ENV} must be a positive number of seconds")
    return value


async def _override_engine_weights(cell: str, model_path: str) -> None:
    """POST /update_weights_from_disk to every worker of ``cell`` (SGLang server API)."""
    import json as _json
    import urllib.request

    from miles.utils.workers.ray_worker_manager import RayWorkerManager

    manager = RayWorkerManager.get_handle()
    infos = await manager.get_worker_infos.remote(cell)
    if not infos:
        raise RuntimeError(f"weight override injection: cell {cell} has no workers")
    for info in infos:
        addr = info.self_addrs.get("primary")
        if addr is None:
            continue
        host = str(addr.host).strip("[]")
        request = urllib.request.Request(
            f"http://{host}:{addr.port}/update_weights_from_disk",
            data=_json.dumps({"model_path": model_path}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        body = await asyncio.to_thread(lambda: urllib.request.urlopen(request, timeout=600).read())
        answer = _json.loads(body or b"{}")
        if answer.get("success") is False:
            raise RuntimeError(f"weight override injection refused by {info.name}: {answer}")


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
        try:
            for cell, targets in self.targets.items():
                now = sorted((i.name, int(i.generation))
                             for i in await manager.get_worker_infos.remote(cell))
                if not now:
                    return f"dead: cell {cell} has no workers (stopped)"
                if now != targets:
                    return f"dead: cell {cell} workers/generation changed {targets} -> {now}"
                for name, generation in targets:
                    handle = await manager.get_actor_handle.remote(
                        name, expected_generation=generation)
                    await asyncio.wait_for(handle.__ray_ready__.remote(), timeout=10)
        except actor_died as exc:  # type: ignore[misc]
            return f"dead: actor died ({type(exc).__name__})"
        except Exception as exc:  # noqa: BLE001 - not a kill: report it as such
            import sys

            print(f"[yeto] TEST INJECTION probe error (not a kill): {exc!r}", file=sys.stderr,
                  flush=True)
            raise InjectedBlockProbeError(f"liveness probe failed: {exc!r}") from exc
        return "alive"


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


def _default_flatten() -> Callable[[Any], list[dict[str, Any]]]:
    from miles.utils.audit_utils.checksum_utils import flatten_inference_engine_checksums

    return flatten_inference_engine_checksums


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
        self.liveness_probe: Any = None  # RayTargetLiveness-like (snapshot/status)
        self.block_poll_s = 1.0
        import os

        self._inject_override = os.environ.get(INJECT_WEIGHT_OVERRIDE_ENV) or None
        self.injected_overrides: list[tuple[str, str]] = []
        self.weight_override_injector: Any = None  # async (cell, path) -> None

    def publish(self, state: TrainableState, *, token_rollout_id: int | None = None) -> PublicationResult:
        tensor_hash = state.policy_tensor_hash()
        if self._export is not None:
            current = self._export()
            if current.policy_tensor_hash() != tensor_hash:
                raise PublicationError("trainer weights differ from the state requested for publication")
        token = policy_token(
            state.policy_version if token_rollout_id is None else token_rollout_id, tensor_hash
        )
        payload_hash, payload_bytes = payload_digest(state)
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
            raise PublicationError(f"upstream update_weights failed: {exc}") from exc
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
            if not failed:
                versions = await asyncio.gather(
                    *(e.get_weight_version() for e in engines), return_exceptions=True
                )
                failed = {
                    ids[i]
                    for i, v in enumerate(versions)
                    if isinstance(v, BaseException) or str(v) != token
                }
            if failed:
                raise PublicationError(
                    f"{len(failed)}/{len(engines)} rollout engines did not acknowledge {token}: "
                    f"{sorted(failed)}",
                    frozenset(failed),
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
                raise PublicationError(f"engine weight checksum failed: {exc}") from exc
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
        tensor_hash = state.policy_tensor_hash()
        if self._export is not None and self._export().policy_tensor_hash() != tensor_hash:
            raise PublicationError("trainer weights differ from the state requested for publication")
        token = policy_token(
            state.policy_version if token_rollout_id is None else token_rollout_id, tensor_hash
        )
        if not self._verify_checksums:
            raise PublicationError("member publication needs the payload read-back (checksums)")
        if self._reference is None or self._reference[0] != token:
            raise PublicationError(
                f"no verified payload reference for {token}: publish it fully first"
            )
        payload_hash, payload_bytes = payload_digest(state)
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
            raise PublicationError(f"serving engines do not all report {token}: {exc}") from exc
        await self._controller.end_commit_weight_version()

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
        probe = self.liveness_probe or RayTargetLiveness()
        await probe.snapshot(cells)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        while loop.time() < deadline:
            await asyncio.sleep(min(self.block_poll_s, max(0.0, deadline - loop.time())))
            state = await probe.status()
            if state != "alive":
                raise RuntimeError(f"target engine of {cells} {state} during the injected block")

    async def _publish_members(
        self, token: str, cells: list[str], epoch: int
    ) -> dict[str, Any] | None:
        update_weights = self._update_weights or _default_update_weights()
        try:
            await self._controller.wait_cells_tracked(cells, timeout_seconds=self.track_timeout_s)
            if self._inject_block is not None and not self.injected_blocks:
                await self._injected_block(cells, self._inject_block)
            await update_weights(
                self._args, self._actor, self._executor, self._controller,
                members=cells, expected_epoch=epoch, admit_cordoned=True,
            )
        except Exception as exc:
            raise PublicationError(f"member update_weights failed: {exc}",
                                   frozenset(member_id(c) for c in cells)) from exc
        if self._inject_override and not self.injected_overrides:
            target = sorted(cells)[0]
            self.injected_overrides.append((target, self._inject_override))
            import sys

            print(f"[yeto] TEST INJECTION {INJECT_WEIGHT_OVERRIDE_ENV}: reloading {target} from "
                  f"{self._inject_override}", file=sys.stderr, flush=True)
            inject = self.weight_override_injector or _override_engine_weights
            await inject(target, self._inject_override)  # an injection failure is loud

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
            if not failed:
                versions = await asyncio.gather(
                    *(e.get_weight_version() for e in engines), return_exceptions=True
                )
                failed = {ids[i] for i, v in enumerate(versions)
                          if isinstance(v, BaseException) or str(v) != token}
            if failed:
                raise PublicationError(f"members did not acknowledge {token}: {sorted(failed)}",
                                       frozenset(failed))
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
                raise PublicationError(f"engine weight checksum failed: {exc}") from exc
            assert self._reference is not None
            reference = self._reference[1]
            expected_engines = len(self.last_engine_checksums or {}) + len(cells)
            if len(bodies) < len(cells) or len(bodies) > expected_engines:
                raise PublicationError(
                    f"{len(bodies)} engines reported checksums; expected up to {expected_engines}"
                )
            bad = [i for i, body in enumerate(bodies) if dict(sorted(body.items())) != reference]
            if bad:
                raise PublicationError(
                    f"payload read-back differs from the published policy on {len(bad)} engines; "
                    "new members are not admitted"
                )
            checksums = {"engines": len(bodies), "reference_token": token}
        await self._controller.admit_cells(cells, expected_epoch=epoch,
                                           expected_weight_version=token)
        await self._commit_version(token, epoch)
        return checksums
