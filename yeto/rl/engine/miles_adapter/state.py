"""``PolicyState`` port over the actor ``run_plugin`` entry point (task 3.4, D4).

Assumes the ``michaellchung/miles`` ``yeto/ports`` patch (task 1.4)::

    TrainGroup.run_plugin(fn_path: str, kwargs: dict | None = None) -> list[Any]

which runs ``fn(actor, **kwargs)`` on every rank and returns the flattened
per-rank results. Tensors travel through Ray pickling (the default
``--worker-comm-backend``); the RPC backend would require JSON-serializable
kwargs/results and is therefore not supported by this port.
"""

from __future__ import annotations

from typing import Any

from ..trainable_state import LAYOUT_LORA, TrainableState, require_supported_layout
from . import LoopRunner
from .state_plugin import APPLY_STATE, EXPORT_STATE, OPTIMIZER_MODES


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

    def _plugin(self, path: str, kwargs: dict[str, Any]) -> list[Any]:
        results = self._run(self._actor.run_plugin(path, kwargs))
        if not isinstance(results, list) or not results:
            raise PolicyStateError(f"run_plugin({path}) returned no per-rank results")
        return results

    def export(self, *, policy_version: int | None = None) -> TrainableState:
        from yeto.rl.core import canonical_state_from_owned_tensors

        version = self.policy_version if policy_version is None else int(policy_version)
        results = [r for r in self._plugin(EXPORT_STATE, {"policy_version": version}) if r is not None]
        if len(results) != 1:
            raise PolicyStateError(f"expected exactly one main-rank export, got {len(results)}")
        (result,) = results
        if result.get("policy_version") != version:
            raise PolicyStateError("exported policy version mismatch")
        lora = canonical_state_from_owned_tensors(
            version,
            result["tensors"],
            base_model_revision=self._revision,
            lora_config_hash=self._config_hash,
            layout_hash=self._expected_layout_hash,
        )
        if self._expected_layout_hash is None:
            self._expected_layout_hash = lora.layout_hash
        return TrainableState.from_lora(lora)

    def apply(self, state: TrainableState, *, optimizer: str, local_step: int) -> None:
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
        results = self._plugin(
            APPLY_STATE,
            {
                "tensors": tensors,
                "policy_version": lora.policy_version,
                "local_step": local_step,
                "optimizer": optimizer,
            },
        )
        if any(r != results[0] for r in results):
            raise PolicyStateError(f"ranks disagree on apply result: {results!r:.300}")
        self.policy_version = lora.policy_version
