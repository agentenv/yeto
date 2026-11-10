"""rl_round_trained fields of a buffer step vs a rollout step."""

from types import SimpleNamespace

from yeto.dashboard.reducer import Reducer
from yeto.rl.engine.driver import IslandDriver


def _fields(*, buffer: bool, by_bucket=None):
    trainer = SimpleNamespace(round_metrics=lambda: {"ppo_kl": 0.0, "train_rollout_kl": 0.0125})
    group = SimpleNamespace(token_count=10, sample_ids=["a", "b"], reward_mean=0.5, reward_std=0.5)
    batch = SimpleNamespace(groups=[group], batch_summary={},
                            buffer_stale_groups=0 if buffer else None,
                            batch_summary_by_bucket=by_bucket)
    metrics = SimpleNamespace(train_step=3, grad_norm=0.5)
    return IslandDriver._train_fields(SimpleNamespace(trainer=trainer), batch, metrics, 1.0)


def test_buffer_step_reports_train_rollout_kl_as_mean_kl():
    f = _fields(buffer=True)
    assert f["mean_kl"] == 0.0125
    r = Reducer()
    r.feed({"event": "rl_round_trained", "island_id": 0, "time_unix": 1.0,
            "local_round_id": 1, "rollout_id": 1, **f})
    assert [v for _, v in r.series(r.islands["0"])["kl"]] == [0.0125]


def test_rollout_step_keeps_its_field_set_and_buckets():
    f = _fields(buffer=False, by_bucket={"easy": {"n": 1}})
    assert "mean_kl" not in f and "reward_mean" not in f
    assert f["batch_summary_by_bucket"] == {"easy": {"n": 1}}
