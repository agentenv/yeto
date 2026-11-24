"""The buffer reader trains the groups it edited.

``_BufferRound.__call__`` converts the taken groups to Miles samples first, then the
xhw guard downweights, the merged-lane trainer stamps the release cut and prunes the
groups of pruned releases -- on the decoded groups. The samples Miles trains must
carry those edits: the scaled rewards, the cut stamps, and none of the pruned groups.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_buffer_rollout import _import_plugin_with_stub_miles
from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import buffer_rollout as br

SOURCE = "src-a"


def _group(g: int, rewards) -> list[dict]:
    out = []
    for j, r in enumerate(rewards):
        d = sample(len(rewards) * g + j, g, 4)
        d.update(reward=float(r), metadata={"provenance": {"sampler": SOURCE}})
        out.append(d)
    return out


def _reader(monkeypatch, groups, *, actions=None, positions=None):
    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    monkeypatch.setattr(plugin, "record_round_metadata", lambda args, rid, **counts: None)
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    report = br.TakeReport(rollout_id=2, groups=len(groups), stale_groups=0, start=0, end=len(groups),
                           position_sources=positions or {}, source_groups={SOURCE: len(groups)})
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path=None,
                                  grpo_std_normalization=False)
    reader.groups, reader.samples_per_group = len(groups), 4
    reader.reader = SimpleNamespace(buffer=SimpleNamespace(read_cursor=0, shard_positions=lambda cursor: {}),
                                    take=lambda rid, n, current_version: (groups, report),
                                    xhw_actions=actions or {})
    return plugin, reader


def _centered(rewards):
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def test_the_xhw_downweight_reaches_the_trained_samples(monkeypatch):
    monkeypatch.setenv("YETO_RL_XHW_GUARD", "action=downweight")
    groups = [(0, _group(0, (1, 0, 0, 1))), (1, _group(1, (1, 1, 0, 0)))]
    _plugin, reader = _reader(monkeypatch, groups, actions={SOURCE: {"action": "downweight", "weight": 0.25}},
                              positions={0: SOURCE})
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    trained = out["samples"]
    # group 0 (the downweighted source): r' = c + w (r - c), so its centered rewards
    # (the GRPO advantages without std normalization) are exactly w times the original
    assert _centered([d["reward"] for d in trained[0]]) == pytest.approx(
        [0.25 * a for a in _centered([1.0, 0.0, 0.0, 1.0])])
    assert trained[0][0]["metadata"]["yeto_xhw_downweight"] == {"weight": 0.25, "reward": 1.0}
    assert trained[0][0]["metadata"]["yeto_sampler_identity"] == {"group_index": 0, "index": 0}
    assert [d["reward"] for d in trained[1]] == [1.0, 1.0, 0.0, 0.0]  # another source: untouched
    assert out["metrics"]  # the round still reports


def test_a_reward_dict_is_refused_not_left_unscaled(monkeypatch):
    monkeypatch.setenv("YETO_RL_XHW_GUARD", "action=downweight")
    group = _group(0, (1, 0, 0, 1))
    group[0]["reward"] = {"acc": 1.0}
    _plugin, reader = _reader(monkeypatch, [(0, group)], actions={SOURCE: {"action": "downweight", "weight": 0.5}},
                              positions={0: SOURCE})
    with pytest.raises(br.SampleBufferRolloutError, match="not a scalar"):
        reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))


def test_pruned_merged_groups_are_not_trained_and_kept_ones_carry_their_cut(monkeypatch):
    groups = [(0, _group(0, (1, 0, 0, 1))), (1, _group(1, (1, 1, 0, 0)))]
    plugin, reader = _reader(monkeypatch, groups)

    def stamp(group):
        if group[0]["group_index"] == 1:
            return False  # its release was pruned before this trainer saw it
        for d in group:
            d["metadata"]["cut"] = "v4"
        return True

    monkeypatch.setattr(plugin, "merged_release_mode", lambda: True)
    monkeypatch.setattr(plugin, "stamp_merged_release_cuts", stamp)
    monkeypatch.setattr(plugin, "set_proc_state", lambda *a, **kw: None)
    report = reader.reader.take(2, 2, current_version=4)[1]
    report.gate_version = 4
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    real = [g for g in out["samples"] if not br.is_padding(g[0])]
    assert [g[0]["group_index"] for g in real] == [0]
    assert all(d["metadata"]["cut"] == "v4" for d in real[0])
    # the pruned group's samples are replaced by padding (Miles' global batch stays whole)
    assert sum(len(g) for g in out["samples"]) == 2 * 4


def _stamped(group, **fields):
    for d in group:
        d["metadata"]["engine"] = {"version_fields": {"fp32_lm_head": True, "temperature": 1.0, "top_p": 1.0,
                                                      "top_k": -1, **fields}}
    return group


SC_ARGS = SimpleNamespace(loss_type="score_centering", rollout_temperature=1.0, rollout_top_p=1.0, rollout_top_k=-1)


def test_a_score_centering_trainer_leaves_out_groups_of_another_sampling(monkeypatch, capsys):
    """The reader checks each group's engine.version_fields against the score_centering
    trainer's --rollout-temperature / -top-p / -top-k: another sampling is left out and
    padded like any group that does not convert (counted, sample_alignment); a group
    without version_fields (an older sampler) is trained and counted."""
    groups = [(0, _stamped(_group(0, (1, 0, 0, 1)))), (1, _stamped(_group(1, (1, 1, 0, 0)), temperature=0.7)),
              (2, _group(2, (0, 1, 0, 1))), (3, _stamped(_group(3, (1, 0, 1, 0)), top_k=0))]
    _plugin, reader = _reader(monkeypatch, groups)
    reader.groups = 4
    reader.sampling = br.trainer_sampling(SC_ARGS)
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    real = [g for g in out["samples"] if not br.is_padding(g[0])]
    assert [g[0]["group_index"] for g in real] == [0, 2, 3]
    assert sum(len(g) for g in out["samples"]) == 4 * 4  # group 1 is padding
    assert out["metrics"]["sample_buffer/unconvertible_groups"] == 1.0
    assert out["metrics"]["sample_buffer/no_version_fields_groups"] == 1.0
    printed = capsys.readouterr().out
    assert "engine.version_fields temperature 0.7 (trainer 1.0)" in printed
    assert "1 group(s) carry no engine.version_fields" in printed


def test_version_fields_are_not_checked_without_score_centering(monkeypatch):
    assert br.trainer_sampling(SimpleNamespace(loss_type="policy_loss", rollout_temperature=0.7)) is None
    assert br.trainer_sampling(SimpleNamespace()) is None
    groups = [(0, _stamped(_group(0, (1, 0, 0, 1)), temperature=0.7, top_p=0.9))]
    converted, problems = br.miles_groups(groups, samples_per_group=4)
    assert problems == [] and len(converted) == 1
    _plugin, reader = _reader(monkeypatch, groups)
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    assert "sample_buffer/no_version_fields_groups" not in out["metrics"]


def test_check_version_fields_compares_the_trainer_sampling():
    want = br.trainer_sampling(SC_ARGS)
    assert br.check_version_fields(_stamped(_group(0, (1, 0))), want) is True
    assert br.check_version_fields(_group(0, (1, 0)), want) is False  # unstamped: let through
    assert br.check_version_fields(_stamped(_group(0, (1, 0)), top_k=0), want) is True  # 0 and -1: no limit
    for fields, needle in (({"top_p": 0.95}, "top_p 0.95"), ({"top_k": 50}, "top_k 50"),
                           ({"temperature": None}, "temperature None")):
        with pytest.raises(br.SampleBufferRolloutError, match=needle):
            br.check_version_fields(_stamped(_group(0, (1, 0)), **fields), want)


def test_with_reader_edits_equals_converting_the_edited_groups():
    groups = [(5, _group(0, (1, 0, 0, 1)))]
    converted, _ = br.miles_groups(groups, samples_per_group=4)
    edited = [(5, [dict(d, reward=0.5 * d["reward"], metadata={**d["metadata"], "x": 1}) for d in groups[0][1]])]
    expected, _ = br.miles_groups(edited, samples_per_group=4)
    assert br.with_reader_edits(converted, edited) == expected
    assert br.with_reader_edits(converted, []) == []


@pytest.mark.parametrize("args, message", [
    (SimpleNamespace(length_penalty_coeff=0.1), "length-penalty"),
    (SimpleNamespace(reward_key="acc"), "reward-key"),
])
def test_downweight_is_refused_where_it_would_not_scale_exactly(args, message):
    env = {"YETO_RL_XHW_GUARD": "action=downweight"}
    with pytest.raises(br.SampleBufferRolloutError, match=message):
        br.check_xhw_downweight(std_normalized=False, environ=env, args=args)
    br.check_xhw_downweight(std_normalized=False, environ=env, args=SimpleNamespace())
    br.check_xhw_downweight(std_normalized=False, environ={"YETO_RL_XHW_GUARD": "1"}, args=args)
