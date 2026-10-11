"""RL gate criteria, structured gate-evidence collection, and the RLF Miles
reward wrapping contract (task 5.8 readiness fixes)."""
import asyncio
import importlib
from types import SimpleNamespace

import pytest

from yeto.rl.rlf import AdmissionError, rl_gate
from yeto.rl.rlf.contracts import group_advantages, policy_update_evidence
from yeto.rl.adapters.miles import rewards as miles_rewards
from yeto.rl.adapters.verl import reward_fn as verl_reward_fn
from yeto.rl.rewards.types import Trajectory


def _row(reward, advantage, success, policy_update=True):
    return {'reward': reward, 'advantage': advantage,
            'metadata': {'success': success}, 'policy_update': policy_update}


# -- rl_gate criteria (spec: 非空 reward、std>0、success_count>0、adv_abs_sum>0、update) --

def test_gate_passes_with_success_and_variance():
    out = rl_gate([_row(0.0, -0.5, False), _row(1.0, 0.5, True)])
    assert out['passed'] and out['success_count'] == 1 and out['advantage_abs_sum'] == 1.0


def test_gate_rejects_empty_rows():
    with pytest.raises(AdmissionError, match='non_empty'):
        rl_gate([])


def test_gate_rejects_all_zero_rewards():
    with pytest.raises(AdmissionError):
        rl_gate([_row(0.0, 0.0, False), _row(0.0, 0.0, False)])


def test_gate_rejects_constant_nonzero_rewards():
    with pytest.raises(AdmissionError, match='reward_std_positive'):
        rl_gate([_row(1.0, -0.5, True), _row(1.0, 0.5, True)])


def test_gate_rejects_zero_success_count():
    # std>0, advantage>0, policy update=true, but nobody succeeded.
    with pytest.raises(AdmissionError, match='success_count_positive'):
        rl_gate([_row(0.0, -0.5, False), _row(1.0, 0.5, False)])


def test_gate_rejects_zero_advantage_abs_sum():
    with pytest.raises(AdmissionError, match='advantage_abs_sum_positive'):
        rl_gate([_row(0.0, 0.0, False), _row(1.0, 0.0, True)])


def test_gate_rejects_missing_policy_update():
    with pytest.raises(AdmissionError, match='policy_update'):
        rl_gate([_row(0.0, -0.5, False, policy_update=False),
                 _row(1.0, 0.5, True, policy_update=False)])


# -- structured evidence collection --

def test_group_advantages_are_reward_minus_group_mean():
    assert group_advantages([0.0, 1.0]) == [-0.5, 0.5]
    assert group_advantages([]) == []
    assert sum(group_advantages([0.2, 0.7, 1.0, 0.0])) == pytest.approx(0.0)


def test_policy_update_evidence_detects_real_change():
    ev = policy_update_evidence({'bias': 0.0}, {'bias': 0.01}, step=3)
    assert ev.updated and ev.step == 3 and ev.changed == ('bias',)
    assert ev.before_digest != ev.after_digest


def test_policy_update_evidence_rejects_unchanged_params():
    ev = policy_update_evidence({'bias': 0.5}, {'bias': 0.5}, step=0)
    assert not ev.updated and ev.before_digest == ev.after_digest and ev.changed == ()


# -- Miles wrapping of the RLF success reward --

def _rlf_metadata(success=True, reward=1.0):
    return {'rlf_episode_id': 'ep-0', 'rlf_task_id': 'fixture-ctf-ret2win',
            'rlf_trajectory_id': 'traj-0', 'grader_evidence': {'trusted': True},
            'success': success, 'reward': reward}


def _fake_sample(metadata):
    return SimpleNamespace(metadata=dict(metadata), group_index=0, index=0,
                           prompt='p', response='r', label=None, tokens=[1],
                           rollout_log_probs=[-0.1], loss_mask=[1],
                           status=None, reward=None)


@pytest.mark.parametrize('success,reward', [(True, 1.0), (False, 0.0)])
def test_miles_wrapper_matches_neutral_rlf_reward(success, reward):
    md = _rlf_metadata(success, reward)
    neutral = miles_rewards.rlf_success_reward(Trajectory(metadata=dict(md)))
    wrapped = getattr(miles_rewards, 'rlf_success')  # module __getattr__ wraps it
    sample = _fake_sample(md)
    value = asyncio.run(wrapped(None, sample))
    assert value == neutral.value == reward
    # Same input, same output: the adapter writes exactly the neutral metadata.
    assert sample.metadata['success'] == neutral.metadata['success'] == success


def test_miles_wrapper_handles_sample_list():
    wrapped = getattr(miles_rewards, 'rlf_success')
    samples = [_fake_sample(_rlf_metadata(True, 1.0)), _fake_sample(_rlf_metadata(False, 0.0))]
    assert asyncio.run(wrapped(None, samples)) == [1.0, 0.0]
    assert [s.metadata['success'] for s in samples] == [True, False]


def test_custom_rm_path_resolves_to_wrapped_callable():
    path = miles_rewards.miles_custom_rm_path('rlf_success')
    assert path == 'yeto.rl.adapters.miles.rewards.rlf_success'
    # What Miles' load_function does: split at the last dot, getattr the module.
    module, _, attr = path.rpartition('.')
    fn = getattr(importlib.import_module(module), attr)
    assert getattr(fn, '__wrapped_neutral__') is miles_rewards.rlf_success_reward
    sample = _fake_sample(_rlf_metadata(True, 1.0))
    assert asyncio.run(fn(None, sample)) == 1.0 and sample.metadata['success'] is True


def test_raw_neutral_name_is_not_a_valid_miles_entry():
    # The unwrapped neutral attribute takes one Trajectory; Miles calls fn(args, samples).
    with pytest.raises(TypeError):
        miles_rewards.rlf_success_reward(None, _fake_sample(_rlf_metadata()))


# -- verl path behavior unchanged: bare float rejected, RLF row validation kept --

def test_verl_rlf_row_requires_trusted_metadata():
    with pytest.raises(ValueError, match='missing trusted metadata'):
        verl_reward_fn.compute_score_rlf('rlf', 'answer', extra_info={'metadata': {}})


def test_verl_rlf_row_accepts_full_metadata():
    score = verl_reward_fn.compute_score_rlf('rlf', 'answer',
                                             extra_info={'metadata': _rlf_metadata(True, 1.0)})
    assert score == 1.0


def test_verl_rlf_row_rejects_bool_reward():
    md = _rlf_metadata(True, True)
    with pytest.raises(ValueError, match='must be numeric'):
        verl_reward_fn.compute_score_rlf('rlf', 'answer', extra_info={'metadata': md})


def test_verl_unmapped_reward_still_refused():
    with pytest.raises(verl_reward_fn.UnmappedReward):
        verl_reward_fn.function_for('not-a-reward')
