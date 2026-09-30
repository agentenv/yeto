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
from .rollout import member_id, policy_token, running_members


class PublicationError(RuntimeError):
    def __init__(self, message: str, failed_members: frozenset[str] = frozenset()) -> None:
        super().__init__(message)
        self.failed_members = failed_members


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
