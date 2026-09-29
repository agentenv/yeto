"""CPU tests for miles_adapter.rollout + rollout_meta_hook (task 3.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook
from yeto.rl.engine.miles_adapter.rollout import (
    DirMetadataSource,
    MilesRolloutPool,
    PolicyTokenMismatch,
    RolloutMetadataError,
    handle_from_metadata,
    policy_token,
    require_policy_tokens,
)
from yeto.rl.engine.ports import RolloutPool

H = "a" * 64
TOKEN = policy_token(3, H)


class Status(Enum):
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class Span:
    version: str


@dataclass
class Call:
    spans: list


@dataclass
class Sample:  # mirrors the upstream miles.utils.types.Sample fields we read
    index: int
    group_index: int
    rollout_id: int | None = 3
    reward: float = 0.0
    response_length: int = 5
    weight_versions: list = field(default_factory=list)
    status: Status = Status.COMPLETED
    tokens: list = field(default_factory=lambda: [1, 2, 3])
    metadata: dict = field(default_factory=dict)

    def get_reward_value(self, args):
        return self.reward


def group(gi, rewards, token=TOKEN, status=Status.COMPLETED):
    return [
        Sample(index=gi * 10 + i, group_index=gi, reward=r,
               weight_versions=[Call([Span(token)])], status=status)
        for i, r in enumerate(rewards)
    ]


def run_hooks(args, kept, filtered, sink):
    hook.record_trained_groups(args, kept)
    all_samples = sorted(kept + filtered, key=lambda g: g[0].index)
    hook.put_to_sink(hook.build_metadata(args, all_samples, sink), sink)
    for attr in ("_yeto_trained_group_keys", "_yeto_bounded_filter_state"):
        setattr(args, attr, None)


def test_hook_extracts_metadata_only(tmp_path, monkeypatch):
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{tmp_path}")
    args = SimpleNamespace(_yeto_bounded_filter_state={"rollout_id": None})
    kept = [group(0, [1.0, 0.0]), group(2, [1.0, 1.0])]
    filtered = [group(1, [0.0, 0.0], status=Status.ABORTED)]
    hook.record_trained_groups(args, kept)
    hook.extract_rollout_metadata(args, sorted(kept + filtered, key=lambda g: g[0].index), None)
    assert args._yeto_bounded_filter_state is None  # per-rollout reset
    raw = (tmp_path / "rollout-3.json").read_text()
    payload = json.loads(raw)
    assert "tokens" not in raw and "[1, 2, 3]" not in raw
    assert payload["completed"] == 2 and payload["filtered"] == 1 and payload["aborted"] == 1
    assert [g["group_id"] for g in payload["groups"]] == ["g0", "g2"]
    g0 = payload["groups"][0]
    assert g0["sample_ids"] == ["s0", "s1"] and g0["policy_token"] == TOKEN
    assert g0["reward_mean"] == 0.5 and g0["token_count"] == 10
    assert payload["trained_sample_indices"] == [0, 1, 20, 21]


def test_hook_requires_recorder():
    with pytest.raises(RuntimeError, match="record_trained_groups"):
        hook.build_metadata(SimpleNamespace(), [group(0, [1.0])])


def test_mixed_versions_in_group_never_match():
    g = group(0, [1.0, 0.0])
    g[1].weight_versions = [Call([Span("yeto:2:" + "b" * 64)])]
    rec = hook.group_record(SimpleNamespace(), g)
    assert rec["policy_token"].startswith("mixed:")


class FakeController:
    def __init__(self):
        self.calls = []

    async def prepare_rollout(self, rollout_id):
        self.calls.append(("prepare", rollout_id))

    async def abort_all(self):
        self.calls.append(("abort",))

    async def get_cell_statuses(self):
        return {"c0": SimpleNamespace(phase="Running"), "c1": SimpleNamespace(phase="Pending")}


class FakeExecutor:
    def __init__(self, args, sink, kept, filtered, pack_indices=None):
        self.args, self.sink, self.kept, self.filtered = args, sink, kept, filtered
        self.pack_indices = pack_indices

    async def get(self, rollout_id):
        run_hooks(self.args, self.kept, self.filtered, self.sink)  # "inside the rollout process"
        indices = self.pack_indices
        if indices is None:
            indices = [s.index for g in self.kept for s in g]
        return SimpleNamespace(sample_indices=indices, data_ref="REF")


def pool(tmp_path, kept, filtered=(), pack_indices=None):
    controller = FakeController()
    executor = FakeExecutor(SimpleNamespace(), f"dir:{tmp_path}", list(kept), list(filtered), pack_indices)
    return controller, MilesRolloutPool(
        inference_controller=controller,
        rollout_executor=executor,
        metadata=DirMetadataSource(tmp_path),
        expected_policy=lambda: (3, H),
    )


def test_generate_returns_handle_with_opaque_payload(tmp_path):
    controller, p = pool(tmp_path, [group(0, [1.0, 0.0]), group(1, [0.0, 1.0])])
    assert isinstance(p, RolloutPool)
    handle = p.generate(3)
    assert controller.calls == [("prepare", 3)]
    assert handle.rollout_id == 3 and handle.policy_version == 3 and handle.policy_hash == H
    assert handle.payload.data_ref == "REF"
    assert handle.completed == 2 and len(handle.groups) == 2
    assert "REF" not in repr(handle)  # payload excluded from repr
    require_policy_tokens(handle, TOKEN)
    # metadata consumed; only the policy token for the rollout-side filter stays
    assert [f.name for f in tmp_path.iterdir()] == [hook.POLICY_TOKEN_FILE]
    assert (tmp_path / hook.POLICY_TOKEN_FILE).read_text() == TOKEN


def test_injected_token_mismatch_rejected_before_training(tmp_path):
    stale = policy_token(2, "c" * 64)
    _, p = pool(tmp_path, [group(0, [1.0, 0.0]), group(1, [0.0, 1.0], token=stale)])
    handle = p.generate(3)
    with pytest.raises(PolicyTokenMismatch) as err:
        require_policy_tokens(handle, TOKEN)
    assert "g1" in str(err.value) and stale in str(err.value)
    assert [g.group_id for g in err.value.groups] == ["g1"]


def test_metadata_batch_disagreement_rejected(tmp_path):
    _, p = pool(tmp_path, [group(0, [1.0, 0.0])], pack_indices=[0, 99])
    with pytest.raises(RolloutMetadataError, match="do not match"):
        p.generate(3)


def test_wrong_rollout_and_missing_metadata(tmp_path):
    payload = {"schema": hook.METADATA_SCHEMA, "rollout_id": 4, "groups": [], "completed": 0, "aborted": 0}
    with pytest.raises(RolloutMetadataError, match="rollout 4"):
        handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    with pytest.raises(RolloutMetadataError):
        DirMetadataSource(tmp_path).take(3)


def test_abort_and_members(tmp_path):
    controller, p = pool(tmp_path, [group(0, [1.0])])
    p.abort()
    assert controller.calls[-1] == ("abort",)
    assert p.members() == frozenset({"engine:c0"})


# ------------------------------------------- group reuse on the ports path


def test_policy_token_is_the_single_snapshot_token():
    from yeto.rl.core import PolicySnapshot, parse_policy_snapshot_token
    from yeto.rl.engine.driver import policy_token as driver_token

    snapshot = PolicySnapshot(3, (0, 0), H)
    assert TOKEN == snapshot.token == driver_token(3, H)
    assert parse_policy_snapshot_token(TOKEN) == (3, H)


def test_buffer_filter_keeps_only_complete_groups_of_published_policy(tmp_path, monkeypatch):
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{tmp_path}")
    stale = policy_token(2, "c" * 64)
    current = [group(0, [1.0, 0.0]), group(1, [0.0, 1.0])]
    buffer = [
        group(5, [1.0, 1.0], token=stale),  # previous policy: dropped
        current[0],
        group(6, [1.0, 1.0], status=Status.ABORTED),  # incomplete: dropped
        group(7, [1.0]),  # wrong group size: dropped
        current[1],
    ]
    args = SimpleNamespace(n_samples_per_prompt=2)
    # No published token yet: nothing is reusable.
    assert hook.policy_buffer_filter(args, None, list(buffer), 4) == []
    DirMetadataSource(tmp_path).set_policy_token(TOKEN)
    selected = hook.policy_buffer_filter(args, None, buffer, 1)
    assert selected == [current[0]] and buffer == [current[1]]


def test_generate_publishes_token_for_rollout_side_filter(tmp_path, monkeypatch):
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{tmp_path}")
    _, p = pool(tmp_path, [group(0, [1.0, 0.0])])
    p.generate(3)
    assert hook.current_policy_token() == TOKEN


def test_ports_argv_installs_the_buffer_filter():
    from yeto.rl.engine.miles_adapter import config as mc

    assert "--buffer-filter-path" in mc.ADAPTER_OWNED_FLAGS
    assert mc.POLICY_BUFFER_FILTER_PATH.endswith("rollout_meta_hook.policy_buffer_filter")


def test_filtered_count_comes_from_metadata_not_aborted():
    """alignment A2/F5: filtered (terminal) is read from the hook metadata;
    carried_over is not tracked yet and stays None (never inferred)."""
    g = {"group_id": "g0", "sample_ids": ["s0"], "policy_token": "t", "reward_mean": 0.0,
         "reward_std": 0.0, "token_count": 1}
    payload = {"schema": hook.METADATA_SCHEMA, "rollout_id": 3, "groups": [g], "completed": 1,
               "aborted": 2, "filtered": 5}
    h = handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    assert (h.aborted, h.filtered, h.carried_over) == (2, 5, None)
    payload.pop("filtered")
    h = handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    assert h.filtered is None


def test_round_counters_align_end_to_end_through_the_pool(tmp_path):
    """R2 end to end (no per-round injection into handles): the executor runs the
    real hooks in the rollout-process order -- trained-groups filter, all-samples
    hook, then a reward-postprocess dispatcher reporting its own count -- and each
    MilesRolloutPool.generate(r) handle carries round r's count."""

    counts = {3: 24, 4: 16, 5: 24}

    class DispatchingExecutor(FakeExecutor):
        async def get(self, rollout_id):
            for g in self.kept:
                for s in g:
                    s.rollout_id = rollout_id
            result = await super().get(rollout_id)
            nonzero = counts[rollout_id]  # computed during reward post-processing
            hook.record_round_metadata(self.args, self.kept, sink=self.sink,
                                       nonzero_advantages=nonzero)
            return result

    kept = [group(0, [1.0, 0.0]), group(1, [0.0, 1.0])]
    executor = DispatchingExecutor(SimpleNamespace(), f"dir:{tmp_path}", kept, [])
    versions = iter([3, 4, 5])
    current = {}

    def expected():
        return current["v"], H

    p = MilesRolloutPool(inference_controller=FakeController(), rollout_executor=executor,
                         metadata=DirMetadataSource(tmp_path), expected_policy=expected)
    seen = {}
    for rid in (3, 4, 5):
        current["v"] = next(versions)
        seen[rid] = p.generate(rid).nonzero_advantages
    assert seen == counts


def test_dynamic_filter_round_stats_come_from_the_all_samples_metadata(tmp_path):
    """1b gap: rl_local_round dynamic_filter_* stayed 0 on ports. Real hook path: 2 groups
    trained, 2 generated groups dropped by the filter -> generated 4, dropped 2."""
    from yeto.rl.engine.driver import IslandDriver, TrainStepMetrics

    kept = [group(0, [1.0, 0.0]), group(1, [0.0, 1.0])]
    dropped = [group(2, [1.0, 1.0]), group(3, [0.0, 0.0])]
    _, p = pool(tmp_path, kept, dropped)
    handle = p.generate(3)
    assert handle.filtered == 2
    stats = IslandDriver._stats(SimpleNamespace(learner_id=0), 3, handle,
                                TrainStepMetrics(grad_norm=1.0), 1.0, 1.0)
    assert (stats.dynamic_filter_generated_groups, stats.dynamic_filter_dropped_groups,
            stats.dynamic_filter_replacement_attempts) == (4, 2, 2)
