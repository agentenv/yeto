"""Training-driver side of the eval island (rl-eval-difficulty-buckets 5.2/5.8, D11.2/D11.4).

``StoreEvalExporter`` is assigned to ``IslandDriver.eval_export``. At an eval
version (round 0, every ``interval`` rounds, the final round) it writes the
published policy (canonical LoRA tensors + identity) and a manifest with every
file's sha256, ``policy_tensor_hash`` and ``rl/policy_token`` to the eval store,
then queues the version. It does not wait for the evaluation.

The default ``policy_files`` serializes the canonical tensors (safetensors) and
their identity; the eval island recomputes ``policy_tensor_hash`` from exactly
these bytes (:func:`verify_policy_tensor_hash`), so the check covers the
tensor content, not only the file bytes.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Mapping

from .store import EvalStore

POLICY_TENSORS = "policy.safetensors"
POLICY_IDENTITY = "policy.json"
STORE_ENV = "YETO_RL_EVAL_STORE"            # local path / Modal Volume mount of the eval store
INTERVAL_ENV = "YETO_RL_EVAL_STORE_INTERVAL"  # eval interval in rounds (default 10, D3)
DEFAULT_INTERVAL = 10


def is_eval_version(rollout_id: int, interval: int, *, final: bool = False) -> bool:
    return final or rollout_id == 0 or (interval > 0 and rollout_id % interval == 0)


def canonical_policy_files(state: Any) -> dict[str, bytes]:
    """Canonical LoRA tensors (safetensors) + identity json. Needs torch/safetensors."""
    from safetensors.torch import save

    lora = state.to_lora()
    tensors = {name: lora.tensors[name].detach().contiguous().cpu() for name in sorted(lora.tensors)}
    identity = {"layout": state.layout, "base_model_revision": state.base_model_revision,
                "config_hash": state.config_hash, "layout_hash": state.layout_hash,
                "policy_version": int(state.policy_version)}
    return {POLICY_TENSORS: save(tensors),
            POLICY_IDENTITY: json.dumps(identity, sort_keys=True).encode()}


def verify_policy_tensor_hash(files_dir: Any, manifest: Mapping[str, Any]) -> str:
    """Eval-island side: rebuild the canonical state from the stored files and
    require its ``policy_tensor_hash`` to equal the manifest's. Needs torch."""
    from pathlib import Path

    from safetensors.torch import load_file

    from yeto.rl.engine.trainable_state import TrainableState

    root = Path(files_dir)
    identity = json.loads((root / POLICY_IDENTITY).read_text())
    state = TrainableState(layout=identity["layout"], base_model_revision=identity["base_model_revision"],
                           config_hash=identity["config_hash"], layout_hash=identity["layout_hash"],
                           policy_version=int(identity["policy_version"]),
                           tensors=load_file(str(root / POLICY_TENSORS)))
    got = state.policy_tensor_hash()
    if got != manifest["policy_tensor_hash"]:
        from .store import EvalIntegrityError

        raise EvalIntegrityError(f"policy_tensor_hash {got[:12]} != manifest {str(manifest['policy_tensor_hash'])[:12]}")
    return got


class StoreEvalExporter:
    def __init__(self, store: EvalStore, *, sampling: Mapping[str, Any], interval: int = DEFAULT_INTERVAL,
                 policy_files: Callable[[Any], Mapping[str, Any]] = canonical_policy_files,
                 extra: Mapping[str, Any] | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.store = store
        self.sampling = dict(sampling)
        self.interval = int(interval)
        self.policy_files = policy_files
        self.extra = dict(extra or {})
        self.clock = clock
        self.exported: list[int] = []

    def __call__(self, rollout_id: int, state: Any, token: str, *, final: bool = False) -> dict[str, Any] | None:
        if not is_eval_version(rollout_id, self.interval, final=final) or rollout_id in self.exported:
            return None
        started = self.clock()
        files = self.policy_files(state)
        manifest = self.store.put_version(
            rollout_id, files=files, policy_tensor_hash=state.policy_tensor_hash(), policy_token=token,
            sampling=self.sampling, extra={**self.extra, "final": bool(final)})
        self.exported.append(rollout_id)
        if final:
            self.store.mark_training_finished({"final_version": int(rollout_id)})
        return {**manifest, "export_seconds": round(self.clock() - started, 3)}


def initial_policy_state(model: str, revision: str, *, rank: int, targets: str, seed: int = 0,
                         trust_remote_code: bool = True) -> Any:
    """Version 0 as training starts it: PEFT's LoRA tensor contract for ``model``,
    ``lora_A`` small seeded noise, ``lora_B`` zero (the served policy equals the base).
    Needs torch, peft, accelerate, transformers (the training image)."""
    import torch

    from yeto.rl.core import canonical_lora_config_hash, canonical_state
    from yeto.rl.engine.trainable_state import TrainableState
    from yeto.rl.export import adapter_targets, derive_peft_lora_specs

    specs = derive_peft_lora_specs(model, revision, rank=rank, targets=targets,
                                   trust_remote_code=trust_remote_code)
    gen = torch.Generator().manual_seed(int(seed))
    tensors = {s.name: (torch.zeros(s.shape) if ".lora_B." in s.name
                        else torch.randn(s.shape, generator=gen) * 0.01) for s in specs}
    state = canonical_state(0, tensors, base_model_revision=revision,
                            lora_config_hash=canonical_lora_config_hash(rank=rank,
                                                                        target_modules=adapter_targets(specs)),
                            expected_specs=specs)
    return TrainableState.from_lora(state)


def export_initial_version(store: EvalStore, model: str, revision: str, *, rank: int, targets: str,
                           sampling: Mapping[str, Any], seed: int = 0) -> dict[str, Any]:
    """Seed the store with version 0 through the training export path (smoke / first eval)."""
    from yeto.rl.engine.driver import policy_token

    state = initial_policy_state(model, revision, rank=rank, targets=targets, seed=seed)
    exporter = StoreEvalExporter(store, sampling=sampling, extra={"base_model": model, "seeded": True})
    return exporter(0, state, policy_token(0, state.policy_tensor_hash())) or {}
