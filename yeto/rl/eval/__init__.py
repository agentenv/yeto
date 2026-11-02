"""Evaluation island.

An inference-only island that evaluates published policy versions on a fixed
hold-out set, separately from training:

* ``guard``  -- start-time checks: training/eval overlap, hold-out sha256.
* ``store``  -- the durable store (a Modal Volume mount on the first version):
  per-version adapter + manifest with sha256, the eval queue, per-unit results
  keyed by ``(policy_version, task_id, trial)``.
* ``export`` -- training-driver side: at eval versions write the policy and
  register the version, without waiting for the evaluation.
* ``island`` -- eval-island side: verify, load, run every unit, resume after
  preemption, emit one ``rl_eval`` per complete version.
* ``stats``  -- per-bucket pass rate, bootstrap standard error, paired
  difference to version 0, truncation / infra-error fractions.

Neutral core: no Modal, Miles or torch imports here; the Modal parts live in
``yeto/cloud/modal_eval_island.py``.
"""
