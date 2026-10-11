"""Dependency-free CPU proof of provider -> Codex trajectory -> trusted reward -> RL gate.

advantage_abs_sum and policy_update are collected from real computation, not
hand-filled: advantages come from group_advantages (reward minus group mean,
the GRPO-style baseline) and policy_update comes from policy_update_evidence
digests of a toy learner parameter before/after an advantage-weighted
REINFORCE step.
"""
import json
from yeto.rl.rlf import EpisodeLease, TrustedGrade, finish_episode, from_codex_result, validate_rollout, rl_gate
from yeto.rl.rlf.contracts import group_advantages, policy_update_evidence

LR = 0.1

def run():
    entries = []
    # Distinct masked-logprob sums so the advantage-weighted gradient is real.
    samples = [
        (False, [-.2, -.1]),
        (True, [-.05, -.05]),
    ]
    for i, (passed, logprobs) in enumerate(samples):
        lease = EpisodeLease('fixture-ctf-ret2win', f'traj-{i}', secret='ephemeral')
        traj = from_codex_result({'tokens': [10 + i, 11 + i], 'logprobs': logprobs, 'loss_mask': [1, 1], 'segments': [{'segment_index': 0}]}, task_id=lease.task_id, episode_id=lease.episode_id, trajectory_id=lease.trajectory_id, policy_id='qwen3-baseline')
        ev = finish_episode(lease, TrustedGrade('completed', 1.0 if passed else 0.0, passed, {'trusted': True, 'flag_match': passed}))
        entries.append((lease, traj, ev))

    # Real advantage computation: reward minus group mean (GRPO baseline).
    advantages = group_advantages([ev.value for _, _, ev in entries])

    # Toy learner step: bias += lr * sum(adv * masked-logprob-sum).
    params = {'bias': 0.0}
    before = dict(params)
    grad = sum(
        adv * sum(lp for lp, m in zip(traj.logprobs, traj.loss_mask) if m)
        for adv, (_, traj, _) in zip(advantages, entries)
    )
    params['bias'] += LR * grad
    update = policy_update_evidence(before, params, step=0)

    rows = []
    for (lease, traj, ev), adv in zip(entries, advantages):
        row = {'reward': ev.value, 'advantage': adv,
               'metadata': {**traj.as_metadata(), **ev.metadata},
               'cleanup_ok': lease.closed, 'policy_update': update.updated}
        validate_rollout(row)
        rows.append(row)

    result = rl_gate(rows)
    result['evidence'] = {
        'advantages': advantages,
        'advantage_source': 'group_advantages(reward - group mean)',
        'policy_update': {
            'step': update.step, 'updated': update.updated,
            'before_digest': update.before_digest, 'after_digest': update.after_digest,
            'changed': list(update.changed),
        },
        'learner_params_after': params,
    }
    return result

if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
