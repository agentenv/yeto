"""Evaluation island (change rl-eval-difficulty-buckets, design D6/D11).

An inference-only island that evaluates published policy versions on a fixed
hold-out set, separately from training:

* ``guard``  -- start-time checks: training/eval overlap, hold-out sha256 (D6.c).
* ``store``  -- the durable store (a Modal Volume mount on the first version):
  per-version adapter + manifest with sha256, the eval queue, per-unit results
  keyed by ``(policy_version, task_id, trial)`` (D11.3).
* ``export`` -- training-driver side: at eval versions write the policy and
  register the version, without waiting for the evaluation (D11.4).
* ``island`` -- eval-island side: verify, load, run every unit, resume after
  preemption, emit one ``rl_eval`` per complete version (D11.3/D11.4).
* ``stats``  -- per-bucket pass rate, bootstrap standard error, paired
  difference to version 0, truncation / infra-error fractions (D4, 3.2).

Neutral core: no Modal, Miles or torch imports here; the Modal parts live in
``yeto/cloud/modal_eval_island.py``.
"""
