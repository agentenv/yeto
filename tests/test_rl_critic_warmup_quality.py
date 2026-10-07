"""rl-algo-critic-family (user decision B): stage-W value quality record + optional gate (CPU).

The reference numbers below are computed by hand / with plain python, not by the
code under test.
"""

from __future__ import annotations

import pytest

from yeto.rl import critic_warmup as cw
from yeto.rl.algos import critic  # noqa: F401  (registers critic.warmup_* fields)
from yeto.rl.algos.compactionrl import compactionrl_spec
from yeto.rl.algos.vapo import vapo_spec
from yeto.rl.engine.algorithm import AlgorithmSpec


def _spec(**gate):
    return AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                         critic={"warmup_steps": 50, **gate})


def _checkpoint(path, payload: bytes):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.bin").write_bytes(payload)
    return str(path)


def test_metrics_hand_computed():
    v, g = [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 4.0]
    q = cw.value_quality(v, g, num_bins=2)
    assert q["mse"] == pytest.approx((0 + 0 + 1 + 1) / 4)
    assert q["relative_error"] == pytest.approx(0.5 ** 0.5 / (18 / 4) ** 0.5)
    # Var(G) = mean([0,1,1,4]^2) - 1.5^2 = 4.5 - 2.25; resid [0,0,-1,1] var 0.5
    assert q["explained_variance"] == pytest.approx(1 - 0.5 / 2.25)
    # bins by value: {0,1} -> mean v 0.5, mean G 0.5; {2,3} -> 2.5 vs 2.5
    assert q["calibration_bins"] == [[2, 0.5, 0.5], [2, 2.5, 2.5]]
    assert q["calibration_error"] == 0.0 and q["calibration_max_gap"] == 0.0


def test_calibration_detects_a_biased_critic():
    g = [float(i % 5) for i in range(100)]
    v = [x + 0.3 for x in g]
    q = cw.value_quality(v, g)
    assert q["calibration_error"] == pytest.approx(0.3)
    assert q["explained_variance"] == pytest.approx(1.0)  # bias is invisible to EV
    assert q["mse"] == pytest.approx(0.09)


def test_degenerate_inputs():
    q = cw.value_quality([1.0, 1.0], [0.0, 0.0])
    assert q["explained_variance"] is None and q["relative_error"] == float("inf")
    with pytest.raises(cw.CriticWarmupError):
        cw.value_quality([1.0], [1.0, 2.0])
    with pytest.raises(cw.CriticWarmupError):
        cw.value_quality([], [])


def test_gate_fields_do_not_change_hashes_when_absent():
    for name in critic.WARMUP_GATE_FIELDS:
        assert name not in _spec().canonical_json()
        assert name not in vapo_spec().canonical_json()
    assert f'"warmup_min_explained_variance"' in _spec(warmup_min_explained_variance=0.5).canonical_json()
    assert _spec(warmup_min_explained_variance=0.5).sha256() != _spec().sha256()
    assert vapo_spec().rejections() == [] and compactionrl_spec().rejections() == []


def test_gate_fields_rejected_without_warmup():
    spec = AlgorithmSpec(advantage={"estimator": "ppo"}, execution={"needs_critic": True},
                         critic={"warmup_steps": 0, "warmup_max_value_mse": 0.1})
    assert any("warmup_steps > 0" in p for p in spec.rejections())


def test_record_only_by_default(tmp_path):
    actor = _checkpoint(tmp_path / "actor", b"a")

    def run_stage(out):
        _checkpoint(tmp_path / out, b"c")
        return {"values": [0.0, 1.0], "returns": [1.0, 0.0]}  # terrible critic

    product = cw.ensure_warmup(_spec(), actor_checkpoint=actor, cache_root=tmp_path / "cache",
                               run_stage=run_stage)
    assert product.value_quality["mse"] == 1.0
    again = cw.load_product(product.critic_checkpoint, _spec(), actor_sha256=product.actor_sha256)
    assert again.value_quality == product.value_quality


def test_gate_blocks_and_writes_no_manifest(tmp_path):
    actor = _checkpoint(tmp_path / "actor", b"a")
    spec = _spec(warmup_max_value_mse=0.5, warmup_min_explained_variance=0.0)

    def bad(out):
        _checkpoint(tmp_path / out, b"c")
        return {"values": [0.0, 1.0], "returns": [1.0, 0.0]}

    with pytest.raises(cw.CriticWarmupError, match="mse=1.0 > critic.warmup_max_value_mse"):
        cw.ensure_warmup(spec, actor_checkpoint=actor, cache_root=tmp_path / "cache", run_stage=bad)
    assert not list((tmp_path / "cache").rglob(cw.MANIFEST))

    def good(out):
        _checkpoint(tmp_path / out, b"c")
        return {"values": [0.0, 0.9, 2.1], "returns": [0.0, 1.0, 2.0]}

    product = cw.ensure_warmup(spec, actor_checkpoint=actor, cache_root=tmp_path / "cache2",
                               run_stage=good)
    assert product.value_quality["explained_variance"] > 0.9


def test_gate_without_metrics_is_refused(tmp_path):
    actor = _checkpoint(tmp_path / "actor", b"a")
    spec = _spec(warmup_max_calibration_error=0.1)
    with pytest.raises(cw.CriticWarmupError, match="no value-quality metrics"):
        cw.ensure_warmup(spec, actor_checkpoint=actor, cache_root=tmp_path / "cache",
                         run_stage=lambda out: _checkpoint(tmp_path / out, b"c") and None)
    # a product recorded without metrics passes record-only but not a later gated load
    product = cw.ensure_warmup(_spec(), actor_checkpoint=actor, cache_root=tmp_path / "c2",
                               run_stage=lambda out: _checkpoint(tmp_path / out, b"c") and None)
    assert product.value_quality is None
