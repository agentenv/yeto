"""rl-code-quality-cleanup 2.2: StrictRlBridge (legacy runtime loop) and
StrictAvgSync (ports driver) produce the same strict-avg submission.

For one base policy and one local result per island, both paths must push
byte-identical DELTA payloads with the same round metadata, and the
reference strict merge (token^2/steps weighted mean, merge.rs
learner_weight) of those payloads must be elementwise equal.  No GPU,
no syncer process.
"""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from yeto.protocol import PullRequest
from yeto.rl.bridge import StrictRlBridge
from yeto.rl.core import LocalRoundStats
from yeto.rl.engine.bridges import StrictAvgSync, _PortsStrictRuntime
from yeto.rl.engine.island_ledger import merge_weight
from yeto.rl.engine.trainable_state import TrainableState

from test_rl_core import _bridge_config, state, tensors


def _stats(island: int, tokens: int) -> LocalRoundStats:
    return LocalRoundStats(
        island_id=island, local_round_id=1, base_policy_version=0, active_groups=1,
        completed_groups=1, cancelled_groups=0, completed_trajectories=1, action_tokens=tokens,
        tool_wait_seconds=0.0, group_p50_seconds=0.1, group_p95_seconds=0.1,
        group_p99_seconds=0.1, reward_mean=1.0, reward_std=0.0, zero_variance_group_ratio=1.0,
        mean_kl=None, ess_ratio=None, clip_fraction=0.0, delta_l2_norm=0.0,
        rollout_seconds=0.1, train_seconds=0.1)


def _local(base, island: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return state(0, {k: v + torch.randn(v.shape, generator=g) * (island + 1)
                     for k, v in base.tensors.items()})


class _Runtime:
    """IslandRuntime for the legacy StrictRlBridge.run path: one fixed local result."""

    def __init__(self, base, local, stats):
        self.base, self.local, self.stats = base, local, stats

    def initialize(self):
        return self.base

    def apply_global_policy(self, _state):
        pass

    def run_local_round(self, **_kw):
        return self.stats

    def export_local_policy(self):
        return self.local

    def record_local_round(self, _stats):
        pass

    def shutdown(self):
        pass


def _legacy_push(tmp_path, island, base, local, stats):
    cfg = replace(_bridge_config(tmp_path / f"legacy-{island}.jsonl"), learner_id=island)
    bridge = StrictRlBridge(_Runtime(base, local, stats), cfg)
    pushed = []
    bridge.client = SimpleNamespace(push_fragment=lambda *v: pushed.append(v))
    bridge.current = bridge.initial
    bridge._run_round(PullRequest(0, 1, 1))
    return pushed


def _ports_push(tmp_path, island, base, local, stats):
    cfg = replace(_bridge_config(tmp_path / f"ports-{island}.jsonl"), learner_id=island)
    sync = StrictAvgSync(cfg)
    sync.bridge = StrictRlBridge(_PortsStrictRuntime(base, None), cfg)
    pushed = []
    sync.bridge.client = SimpleNamespace(push_fragment=lambda *v: pushed.append(v))
    sync.bridge.current = sync.bridge.initial
    sync.current = sync.bridge.current
    sync.permit = PullRequest(0, 1, 1)
    driver = SimpleNamespace(phase=lambda *a, **k: None,
                             export_local=lambda: TrainableState.from_lora(local))
    sync._submit(driver, rollout_id=0, stats=stats)
    return pushed


def _merge(payloads, weights):
    deltas = [torch.frombuffer(bytearray(p), dtype=torch.float32).to(torch.float64) for p in payloads]
    w = torch.tensor(weights, dtype=torch.float64)
    return (sum(wi * d for wi, d in zip(w, deltas)) / w.sum()).to(torch.float32)


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_strict_bridge_and_strict_avg_sync_submit_identical_deltas(tmp_path, seed):
    base = state(0, tensors())
    legacy, ports, weights = [], [], []
    for island, tokens in ((0, 37), (1, 91)):
        local = _local(base, island, seed)
        stats = _stats(island, tokens)
        a = _legacy_push(tmp_path, island, base, local, stats)
        b = _ports_push(tmp_path, island, base, local, stats)
        assert len(a) == len(b) == 1
        assert a[0][:-1] == b[0][:-1]  # fragment, step, attempt, base version, ...
        assert bytes(a[0][-1]) == bytes(b[0][-1])  # delta payload, byte for byte
        legacy.append(bytes(a[0][-1]))
        ports.append(bytes(b[0][-1]))
        weights.append(merge_weight(tokens, 1))
        tape_a = json.loads((tmp_path / f"legacy-{island}.jsonl").read_text())
        tape_b = json.loads((tmp_path / f"ports-{island}.jsonl").read_text())
        for key in ("rl/local_delta_norm", "sync/bytes_sent", "rl/action_tokens"):
            assert tape_a[key] == tape_b[key], key
    assert legacy[0] != legacy[1] and any(legacy[0])  # non-trivial deltas
    assert torch.equal(_merge(legacy, weights), _merge(ports, weights))
