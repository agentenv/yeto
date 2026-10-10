"""positive_lm_success launch check + gsm8k success flag (S19 #6 s19-vapo-g3-20261010a).

The G3 run loaded the out-of-tree ``gsm8k_reward:score`` (no success flag) with a
VAPO spec using ``positive_lm_source='success'``; the Miles fork raised on both
islands at the first critic step. S13 G1 passed with ``yeto.rl.gsm8k_reward:score``.
"""

from __future__ import annotations

import asyncio
import sys
import types

from yeto.rl.algos.critic import positive_lm_success_problems
from yeto.rl.algos.vapo import vapo_spec
from yeto.rl.engine.algorithm import launch_problems
from yeto.rl.math_reward import SUCCESS_DECLARATION_ATTR

_VALS = {"rollout_batch_size": 4, "rollout_max_response_len": 384,
         "context_parallel_size": 1, "multi_lora": False}


def _spec(source="success"):
    if source == "success":
        return vapo_spec()
    return vapo_spec(loss={"positive_lm_source": source, "positive_lm_reward_threshold": 0.0})


def _install_undeclared(monkeypatch):
    mod = types.ModuleType("_oot_gsm8k_reward")

    async def score(args, sample, **kw):  # out-of-tree copy: reward only
        return 1.0

    mod.score = score
    monkeypatch.setitem(sys.modules, "_oot_gsm8k_reward", mod)


def test_vapo_spec_uses_success_source():
    assert _spec().loss.positive_lm_source == "success"


def test_undeclared_reward_is_refused(monkeypatch):
    _install_undeclared(monkeypatch)
    vals = dict(_VALS, reward_function="_oot_gsm8k_reward:score")
    probs = positive_lm_success_problems(_spec(), vals)
    assert probs and "writes_success" in probs[0]
    assert any(p.startswith("[positive_lm_success]") for p in launch_problems(_spec(), vals))


def test_unimportable_reward_is_refused():
    probs = positive_lm_success_problems(_spec(), dict(_VALS, reward_function="no_such_mod_xyz:score"))
    assert probs and "cannot be imported" in probs[0]


def test_declared_in_package_rewards_pass():
    for rf in ("yeto.rl.gsm8k_reward:score", "yeto.rl.math_reward:reward_func"):
        assert positive_lm_success_problems(_spec(), dict(_VALS, reward_function=rf)) == []


def test_skips_without_reward_or_other_source(monkeypatch):
    _install_undeclared(monkeypatch)
    assert positive_lm_success_problems(_spec(), _VALS) == []
    vals = dict(_VALS, reward_function="_oot_gsm8k_reward:score")
    assert positive_lm_success_problems(_spec("reward"), vals) == []
    from yeto.rl.engine.algorithm import AlgorithmSpec
    assert positive_lm_success_problems(AlgorithmSpec(), vals) == []


def test_gsm8k_score_writes_success_matching_reward():
    from yeto.rl.gsm8k_reward import score
    assert getattr(score, SUCCESS_DECLARATION_ATTR) is True
    for response, want in (("so \\boxed{72}", 1.0), ("so \\boxed{71}", 0.0), ("", 0.0)):
        sample = types.SimpleNamespace(response=response, label="... #### 72", metadata={})
        value = asyncio.run(score(None, sample))
        assert value == want
        assert sample.metadata["success"] is (want == 1.0)
