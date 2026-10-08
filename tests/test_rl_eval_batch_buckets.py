"""rl-eval-difficulty-buckets 4.1 (D5): training batch summary by difficulty."""

from types import SimpleNamespace


from yeto.rl.adapters.miles import rollout_meta_hook as hook


def _s(diff, reward, status="completed", length=10):
    meta = {} if diff is None else {"difficulty": diff}
    return SimpleNamespace(metadata=meta, reward=reward, status=status, response_length=length)


def test_by_bucket_groups_and_counts():
    samples = [_s("easy", 1.0), _s("easy", 0.0, "truncated", 30), _s("hard", 0.0), _s(None, 1.0)]
    out = hook.batch_summary_by_bucket(SimpleNamespace(), samples)
    assert sorted(out) == ["easy", "hard", "unknown"]
    assert out["easy"] == {"n": 2, "reward_mean": 0.5, "success_rate": 0.5, "truncated_frac": 0.5,
                           "resp_len_mean": 20.0}
    assert out["hard"]["success_rate"] == 0.0 and out["unknown"]["n"] == 1


def test_driver_copies_by_bucket_only_when_present():
    from yeto.rl.engine.driver import IslandDriver

    drv = SimpleNamespace(trainer=SimpleNamespace())
    metrics = SimpleNamespace(train_step=3)
    plain = SimpleNamespace(groups=[], batch_summary=None)
    assert "batch_summary_by_bucket" not in IslandDriver._train_fields(drv, plain, metrics, 1.0)
    by = {"easy": {"n": 2, "success_rate": 0.5}}
    tagged = SimpleNamespace(groups=[], batch_summary=None, batch_summary_by_bucket=by)
    assert IslandDriver._train_fields(drv, tagged, metrics, 1.0)["batch_summary_by_bucket"] == by


def _group(gi, diffs):
    return [SimpleNamespace(index=gi * 10 + i, group_index=gi, rollout_id=3, status=SimpleNamespace(value="completed"),
                            remove_sample=False, metadata=None if d is None else {"difficulty": d},
                            reward=float(i % 2), response_length=5, weight_versions=None)
            for i, d in enumerate(diffs)]


def test_build_metadata_adds_by_bucket_only_with_difficulty():
    args = SimpleNamespace(yeto_rl_observe_timeline=True)
    data = [_group(0, ["easy", "hard"]), _group(1, ["hard", "hard"])]
    hook.record_trained_groups(args, data)
    meta = hook.build_metadata(args, data)
    assert meta["batch_summary_by_bucket"]["hard"]["n"] == 3
    assert meta["batch_summary_by_bucket"]["easy"]["n"] == 1
    plain = [_group(0, [None, None])]
    hook.record_trained_groups(args, plain)
    assert "batch_summary_by_bucket" not in hook.build_metadata(args, plain)
    assert "batch_summary" in hook.build_metadata(args, plain)
