"""``PolicyState`` port over the actor ``run_plugin`` entry point (task 3.4, D4).

Assumes the ``michaellchung/miles`` ``yeto/ports`` patch (task 1.4)::

    TrainGroup.run_plugin(fn_path: str, kwargs: dict | None = None) -> list[Any]

which runs ``fn(actor, **kwargs)`` on every rank and returns the flattened
per-rank results. Tensors travel through Ray pickling (the default
``--worker-comm-backend``); the RPC backend would require JSON-serializable
kwargs/results and is therefore not supported by this port.
"""

from __future__ import annotations

import inspect
from typing import Any

from yeto.rl.engine.policy_digest import WeightsChanged
from yeto.rl.engine.trainable_state import LAYOUT_LORA, TrainableState, require_supported_layout
from . import LoopRunner
from .state_plugin import APPLY_STATE, EXPORT_DIGEST, EXPORT_STATE, OPTIMIZER_MODES, WEIGHTS_VERSION


class PolicyStateError(RuntimeError):
    pass


def require_run_plugin(controller_cls: Any = None) -> None:
    """Refuse ports startup when the pinned Miles lacks ``run_plugin`` (D4).

    Runs before any upstream component is created (no model is loaded). The
    current ``MILES_NEXT_COMMIT`` is still the bare upstream commit, which
    has no generic plugin entry point; the ports path cannot export or apply
    trainable state without it.
    """

    if controller_cls is None:
        try:
            from miles.ray.train.group import TrainerController as controller_cls
        except ImportError as error:
            raise PolicyStateError(f"cannot import the Miles trainer group: {error}") from error
    if not callable(getattr(controller_cls, "run_plugin", None)):
        from yeto.rl import MILES_NEXT_COMMIT

        raise PolicyStateError(
            "the pinned Miles (MILES_NEXT_COMMIT "
            f"{MILES_NEXT_COMMIT[:12]}) has no TrainerController.run_plugin; "
            "--rl-engine ports needs the michaellchung/miles yeto/ports patch "
            "(rl-engine-ports task 1.4). Use --rl-engine legacy until the pin "
            "carries it."
        )


class MilesPolicyState:
    """Export/apply the LoRA trainable state of the single-cell actor group."""

    def __init__(
        self,
        *,
        actor_model: Any,
        base_model_revision: str,
        config_hash: str,
        layout: str = LAYOUT_LORA,
        expected_layout_hash: str | None = None,
        policy_version: int = 0,
        runner: LoopRunner | None = None,
    ) -> None:
        # Startup rejection of representable-but-unimplemented layouts.
        self.layout = require_supported_layout(layout)
        self._actor = actor_model
        self._revision = base_model_revision
        self._config_hash = config_hash
        self._expected_layout_hash = expected_layout_hash
        self._run = (runner or LoopRunner()).run
        self.policy_version = policy_version

    @property
    def layout_hash(self) -> str:
        """The pinned layout hash: predicted at start-up, or learned from the
        first export (Flash-Next). Reading it before that export is an error,
        never a placeholder."""

        if self._expected_layout_hash is None:
            raise PolicyStateError("LoRA layout hash requested before the first trainable-state export")
        return self._expected_layout_hash

    @staticmethod
    def _checked(path: str, results: Any) -> list[Any]:
        if not isinstance(results, list) or not results:
            raise PolicyStateError(f"run_plugin({path}) returned no per-rank results")
        return results

    def _plugin(self, path: str, kwargs: dict[str, Any]) -> list[Any]:
        return self._checked(path, self._run(self._actor.run_plugin(path, kwargs)))

    async def _aplugin(self, path: str, kwargs: dict[str, Any]) -> list[Any]:
        """``_plugin`` for callers already inside a running event loop (the sync
        ``_run`` would raise "This event loop is already running")."""
        value = self._actor.run_plugin(path, kwargs)
        if inspect.isawaitable(value):
            value = await value
        return self._checked(path, value)

    def export(self, *, policy_version: int | None = None) -> TrainableState:
        version = self._export_version(policy_version)
        results = self._plugin(EXPORT_STATE, {"policy_version": version})
        return self._export_result(version, results)

    async def aexport(self, *, policy_version: int | None = None) -> TrainableState:
        version = self._export_version(policy_version)
        results = await self._aplugin(EXPORT_STATE, {"policy_version": version})
        return self._export_result(version, results)

    def export_digest(self, *, policy_version: int | None = None):
        """rl-publish-fastpath: the trainer exports and hashes; only the digests cross
        Ray.  Returns a :class:`TrainerResidentState` whose tensors stay in the trainer
        (``materialize`` falls back to :meth:`export`)."""

        import time

        version = self._export_version(policy_version)
        started = time.monotonic()
        results = self._plugin(EXPORT_DIGEST, self._digest_kwargs(version))
        state = self._digest_result(version, results)
        export_s = next((r.get("export_seconds") for r in results if r is not None), None)
        # Timing for the next GPU run (log only; no tape field changes): total plugin
        # round trip vs the trainer-side hashing part.
        print(f"[rl] publish-fastpath digest export v{version}: "
              f"total={time.monotonic() - started:.1f}s trainer_export={export_s}s "
              f"hash={state.digest.hash_seconds:.1f}s bytes={state.digest.payload_bytes}", flush=True)
        return state

    def weights_version(self) -> dict[str, Any]:
        """The main trainer rank's weights version ``{process_id, version}`` (D3)."""

        results = [r for r in self._plugin(WEIGHTS_VERSION, {}) if r is not None]
        if len(results) != 1:
            raise PolicyStateError(f"expected exactly one main-rank weights version, got {len(results)}")
        return dict(results[0])

    def check_holds(self, state: Any) -> None:
        """rl-publish-fastpath D3: the trainer still holds ``state`` (a trainer-resident
        handle).  Same trainer process: compare weights versions only (no export).
        Trainer process replaced since the handle was taken (rebuild / resize): its
        counter restarted, so fall back to the content check (trainer-side re-hash).
        Raises :class:`WeightsChanged` on a mismatch."""

        mark = state.weights_mark
        if mark is None:
            raise WeightsChanged("the resident state carries no trainer weights version")
        now = self.weights_version()
        if now.get("process_id") == mark.get("process_id"):
            if int(now.get("version", -1)) != int(mark.get("version", -2)):
                raise WeightsChanged(
                    f"trainer weights version {now.get('version')} != {mark.get('version')} recorded "
                    f"for policy v{state.policy_version} (weights written since the export)")
            return
        current = self._current_digest(state.policy_version)
        if current.policy_tensor_hash != state.policy_tensor_hash():
            raise WeightsChanged(
                f"rebuilt trainer holds {current.policy_tensor_hash}, "
                f"not the policy {state.policy_tensor_hash()}")

    def _digest_kwargs(self, version: int) -> dict[str, Any]:
        return {"policy_version": version, "base_model_revision": self._revision,
                "lora_config_hash": self._config_hash}

    def _current_digest(self, version: int):
        import time

        from yeto.rl.engine.policy_digest import PolicyDigest

        started = time.monotonic()
        results = [r for r in self._plugin(EXPORT_DIGEST, self._digest_kwargs(version)) if r is not None]
        if len(results) != 1:
            raise PolicyStateError(f"expected exactly one main-rank digest, got {len(results)}")
        digest = PolicyDigest.from_wire(results[0]["digest"])
        print(f"[rl] publish-fastpath recheck v{version}: total={time.monotonic() - started:.1f}s "
              f"trainer_export={results[0].get('export_seconds')}s hash={digest.hash_seconds:.1f}s", flush=True)
        return digest

    def _digest_result(self, version: int, results: list[Any]):
        from yeto.rl.engine.policy_digest import PolicyDigest, TrainerResidentState, layout_hash_of

        results = [r for r in results if r is not None]
        if len(results) != 1:
            raise PolicyStateError(f"expected exactly one main-rank digest, got {len(results)}")
        (result,) = results
        if result.get("policy_version") != version:
            raise PolicyStateError("exported policy version mismatch")
        digest = PolicyDigest.from_wire(result["digest"])
        layout_hash = layout_hash_of(digest.specs)
        if self._expected_layout_hash is not None and layout_hash != self._expected_layout_hash:
            raise ValueError(
                f"canonical LoRA layout hash changed: exported {len(digest.specs)} tensors "
                f"(expected layout hash {self._expected_layout_hash}, got {layout_hash})")
        if self._expected_layout_hash is None:
            self._expected_layout_hash = layout_hash
            print(f"[rl] learned LoRA layout from first digest export: {len(digest.specs)} tensors, "
                  f"hash={layout_hash}", flush=True)
        return TrainerResidentState(
            layout=self.layout,
            base_model_revision=self._revision,
            config_hash=self._config_hash,
            layout_hash=layout_hash,
            policy_version=version,
            digest=digest,
            materialize_fn=lambda v: self.export(policy_version=v),
            weights_mark=dict(result["weights"]) if result.get("weights") else None,
            holds_fn=self.check_holds,
        )

    def _export_version(self, policy_version: int | None) -> int:
        return self.policy_version if policy_version is None else int(policy_version)

    def _export_result(self, version: int, results: list[Any]) -> TrainableState:
        from yeto.rl.core import canonical_state_from_owned_tensors

        results = [r for r in results if r is not None]
        if len(results) != 1:
            raise PolicyStateError(f"expected exactly one main-rank export, got {len(results)}")
        (result,) = results
        if result.get("policy_version") != version:
            raise PolicyStateError("exported policy version mismatch")
        try:
            lora = canonical_state_from_owned_tensors(
                version,
                result["tensors"],
                base_model_revision=self._revision,
                lora_config_hash=self._config_hash,
                layout_hash=self._expected_layout_hash,
            )
        except ValueError as exc:
            # Make a layout mismatch diagnosable from the island log: the
            # predicted layout is printed by the learner at start-up.
            raise ValueError(
                f"{exc}: exported {len(result['tensors'])} tensors "
                f"(expected layout hash {self._expected_layout_hash}): "
                + _describe_tensors(result["tensors"])
            ) from exc
        if self._expected_layout_hash is None:
            self._expected_layout_hash = lora.layout_hash
            print(
                f"[rl] learned LoRA layout from first export: {len(lora.tensors)} tensors, "
                f"hash={lora.layout_hash}",
                flush=True,
            )
        return TrainableState.from_lora(lora)

    def apply(self, state: TrainableState, *, optimizer: str, local_step: int) -> None:
        path, kwargs, lora = self._apply_request(state, optimizer, local_step)
        self._apply_result(lora, self._plugin(path, kwargs))

    async def aapply(self, state: TrainableState, *, optimizer: str, local_step: int) -> None:
        path, kwargs, lora = self._apply_request(state, optimizer, local_step)
        self._apply_result(lora, await self._aplugin(path, kwargs))

    def _apply_request(self, state: TrainableState, optimizer: str, local_step: int):
        if optimizer not in OPTIMIZER_MODES:
            raise ValueError(f"optimizer must be one of {OPTIMIZER_MODES}, got {optimizer!r}")
        if isinstance(local_step, bool) or not isinstance(local_step, int) or local_step < 0:
            raise ValueError("local_step must be a non-negative int")
        require_supported_layout(state.layout, {self.layout})
        lora = state.to_lora()
        if lora.base_model_revision != self._revision or lora.lora_config_hash != self._config_hash:
            raise PolicyStateError("trainable state belongs to a different base model or LoRA config")
        if self._expected_layout_hash is not None and lora.layout_hash != self._expected_layout_hash:
            raise PolicyStateError("global LoRA layout hash mismatch")
        import torch

        tensors = {
            name: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
            for name, value in sorted(lora.tensors.items())
        }
        kwargs = {
            "tensors": tensors,
            "policy_version": lora.policy_version,
            "local_step": local_step,
            "optimizer": optimizer,
        }
        return APPLY_STATE, kwargs, lora

    def _apply_result(self, lora: Any, results: list[Any]) -> None:
        if any(r != results[0] for r in results):
            raise PolicyStateError(f"ranks disagree on apply result: {results!r:.300}")
        self.policy_version = lora.policy_version


def _describe_tensors(tensors: Any, limit: int = 400) -> str:
    items = []
    for name in sorted(tensors)[:limit]:
        shape = tuple(int(d) for d in getattr(tensors[name], "shape", ()))
        items.append(f"{name}{list(shape)}")
    return ", ".join(items)
