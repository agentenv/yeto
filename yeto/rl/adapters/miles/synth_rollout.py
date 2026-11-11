"""TEST ONLY: synthetic long-context rollout dumps for trainer memory and speed probes.

``YETO_TEST_SYNTH_ROLLOUT`` (JSON) makes the debug-rollout replay
(``YETO_TEST_MILES_LOAD_DEBUG_ROLLOUT_DATA``) train on generated samples instead of
a recorded dump: rollout ``i`` holds ``samples`` sequences of ``lengths[i]`` tokens
(``prompt`` prompt tokens, the rest response). Every Ray node writes the same files
at the dump path template, so the rollout actor finds them wherever Ray places it.

The content is deterministic in (``seed``, length, sample index): the same length
gives byte-identical samples in every rollout and every parallel layout, so the
loss, grad norm and train log-probs of two layouts compare directly. Rollout
log-probs are 0, so Miles' ``train_rollout_logprob_abs_diff`` is the mean negative
train log-prob. Routing replay ids are uniform (token ``t`` of layer ``l`` takes
experts ``(r + j * experts / router_topk) % experts``, ``r`` random), which is the
balanced case; layer 0 (dense) is -1 as in a rollout buffer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYNTH_ENV = "YETO_TEST_SYNTH_ROLLOUT"


@dataclass(frozen=True)
class SynthSpec:
    lengths: tuple[int, ...]
    samples: int = 2
    prompt: int = 1024
    topk: int = 128
    layers: int = 93
    dense_layers: int = 1
    router_topk: int = 16
    experts: int = 896
    vocab: int = 163840
    token_low: int = 1000
    token_high: int = 100000
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.lengths:
            raise ValueError(f"{SYNTH_ENV}: lengths is empty")
        if self.samples < 1:
            raise ValueError(f"{SYNTH_ENV}: samples must be positive")
        if any(int(n) <= self.prompt for n in self.lengths):
            raise ValueError(f"{SYNTH_ENV}: every length must exceed prompt {self.prompt}")
        if self.experts % self.router_topk:
            raise ValueError(f"{SYNTH_ENV}: experts must be a multiple of router_topk")
        if self.topk < 2 or self.topk * 997 >= self.vocab:  # top-k ids t + j * 997 stay distinct
            raise ValueError(f"{SYNTH_ENV}: topk {self.topk} must be in [2, vocab / 997)")

    @classmethod
    def parse(cls, raw: str) -> "SynthSpec":
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{SYNTH_ENV} must be a JSON object")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{SYNTH_ENV}: unknown keys {sorted(unknown)}")
        data["lengths"] = tuple(int(n) for n in data.get("lengths", ()))
        return cls(**data)

    def digest(self) -> str:
        raw = json.dumps(self.__dict__, sort_keys=True, default=list).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


def synth_sample_dict(spec: SynthSpec, length: int, index: int) -> dict[str, Any]:
    """One sample of ``length`` tokens as ``Sample.to_dict()`` (Miles' dump format)."""
    import numpy as np
    from miles.utils.types import Sample

    rng = np.random.default_rng([spec.seed, int(length), int(index)])
    tokens = rng.integers(spec.token_low, spec.token_high, size=length, dtype=np.int64)
    response = length - spec.prompt
    next_tokens = tokens[spec.prompt:].astype(np.int32)
    offsets = (np.arange(spec.topk, dtype=np.int64) * 997)[None, :]
    topk_ids = ((next_tokens.astype(np.int64)[:, None] + offsets) % spec.vocab).astype(np.int32)
    topk_lp = np.full((response, spec.topk), np.log(0.5 / (spec.topk - 1)), dtype=np.float32)
    topk_lp[:, 0] = np.log(0.5)
    stride = spec.experts // spec.router_topk
    base = rng.integers(0, spec.experts, size=(length - 1, spec.layers, 1), dtype=np.int32)
    routed = (base + np.arange(spec.router_topk, dtype=np.int32)[None, None, :] * stride) % spec.experts
    routed[:, : spec.dense_layers, :] = -1
    sample = Sample(
        group_index=0,
        index=int(index),
        prompt="synthetic",
        tokens=tokens.tolist(),
        response_length=response,
        reward=1.0 if index % 2 == 0 else 0.0,
        loss_mask=[1] * response,
        rollout_log_probs=[0.0] * response,
        rollout_topk_token_ids=topk_ids,
        rollout_topk_log_probs=topk_lp,
        rollout_routed_experts=routed.astype(np.int32),
        status=Sample.Status.COMPLETED,
    )
    return sample.to_dict()


def write_synth_dumps(spec: SynthSpec, path_template: str) -> list[str]:
    """Write rollout ``i`` of ``spec`` at ``path_template.format(rollout_id=i)``; files that
    carry this spec's digest are kept. Returns the paths."""
    import torch

    paths = []
    by_length: dict[int, list[dict[str, Any]]] = {}
    for rollout_id, length in enumerate(spec.lengths):
        path = Path(path_template.format(rollout_id=rollout_id))
        marker = path.with_suffix(path.suffix + ".synth")
        paths.append(str(path))
        if path.exists() and marker.exists() and marker.read_text() == spec.digest():
            continue
        if length not in by_length:
            by_length.clear()  # keep one length in host memory at a time (1M tokens: ~7 GB/sample)
            by_length[length] = [synth_sample_dict(spec, length, i) for i in range(spec.samples)]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(rollout_id=rollout_id, metadata={"synthetic": spec.digest()},
                        samples=by_length[length]), tmp)
        os.replace(tmp, path)
        marker.write_text(spec.digest())
    return paths


def synth_spec_from_env(environ=None) -> SynthSpec | None:
    raw = (os.environ if environ is None else environ).get(SYNTH_ENV)
    return SynthSpec.parse(raw) if raw else None


def write_synth_dumps_on_every_node(spec: SynthSpec, path_template: str) -> int:
    """Run :func:`write_synth_dumps` once on every alive Ray node; returns the node count."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    import sys

    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    task = ray.remote(num_cpus=1)(write_synth_dumps)
    nodes = [n["NodeID"] for n in ray.nodes() if n.get("Alive")]
    refs = [task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)).remote(
        spec, path_template) for node in nodes]
    for paths in ray.get(refs):
        print(f"[rl] TEST ONLY: synthetic rollout dumps {paths}", flush=True)
    return len(nodes)
