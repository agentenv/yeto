"""s19-compaction-g1-20261010g: receipt ids per trajectory; segment counts follow the final loss masks."""

from __future__ import annotations

from types import SimpleNamespace

from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.trainer import receipt_trajectory_ids
from yeto.rl.harness.codex.codex_openenv_generate import refresh_segment_counts


def _batch():
    return SimpleNamespace(rollout_id=0, groups=[
        SimpleNamespace(group_id="g4", sample_ids=("s4", "s4", "s4")),  # 3 segments of one rollout
        SimpleNamespace(group_id="g5", sample_ids=("s5",)),
    ])


def test_receipt_ids_collapse_segments_in_order():
    assert receipt_trajectory_ids(_batch()) == ("r0:g4:s4", "r0:g5:s5")


def test_receipt_accepts_the_collapsed_ids():
    ids = receipt_trajectory_ids(_batch())
    assert len(set(ids)) == len(ids)  # LocalStepReceipt.__post_init__ refuses duplicates


def _seg(index, mask, rollout=7):
    return SimpleNamespace(rollout_id=rollout, group_index=rollout, index=rollout, loss_mask=list(mask),
                           metadata={"segment_index": index, "segment_tokens": 99, "tokens_after": 99,
                                     "gae_length": 99})


def test_refresh_follows_masks_set_after_assembly():
    segs = [_seg(1, [1, 1, 0, 1]), _seg(0, [1, 1, 1]), _seg(2, [1, 0])]  # any order
    other = SimpleNamespace(rollout_id=1, loss_mask=[1], metadata={})  # not a segment: untouched
    assert refresh_segment_counts(segs + [other]) == 3
    by = {s.metadata["segment_index"]: s.metadata for s in segs}
    assert [by[i]["segment_tokens"] for i in range(3)] == [3, 3, 1]
    assert [by[i]["tokens_after"] for i in range(3)] == [4, 1, 0]
    assert {by[i]["gae_length"] for i in range(3)} == {7}
    assert other.metadata == {}
    assert refresh_segment_counts(segs) == 0  # idempotent


def test_refresh_keeps_rollouts_apart():
    a = [_seg(0, [1, 1], rollout=1), _seg(1, [1], rollout=1)]
    b = [_seg(0, [1], rollout=2), _seg(1, [1, 1, 1], rollout=2)]
    refresh_segment_counts(a + b)
    assert [s.metadata["tokens_after"] for s in a] == [1, 0]
    assert [s.metadata["tokens_after"] for s in b] == [3, 0]


def test_hook_refreshes_after_the_placeholder_mask(monkeypatch):
    seg0, seg1 = _seg(0, [1, 1, 1]), _seg(1, [1, 1])
    seg0.metadata.update(segment_tokens=3, tokens_after=2, gae_length=5)

    def fake_mask(data, **_):  # S19 #8 masks a placeholder stop token in segment 1 after assembly
        seg1.loss_mask[-1] = 0
        return 1

    monkeypatch.setattr("yeto.rl.harness.placeholder_logprob_mask.apply_to_groups", fake_mask)
    monkeypatch.setattr("yeto.rl.algos.sample_filters.apply_sample_filters", lambda args, data: None)
    monkeypatch.setattr("yeto.rl.harness.reasoning_loss_mask.apply_from_args", lambda *a, **k: None)
    monkeypatch.setattr(hook, "_group_key", lambda group: id(group))
    hook.record_trained_groups(SimpleNamespace(), [[seg0, seg1]])
    assert seg0.metadata["tokens_after"] == 1 and seg1.metadata["gae_length"] == 4
