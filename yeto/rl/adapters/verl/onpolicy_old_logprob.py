"""S29 MFU-C: on-policy reuse of the old log-probs (skip verl's old_log_prob recompute).

**What verl does** (decoupled PPO, ``algorithm.rollout_correction.bypass_mode=False``):
before ``update_actor`` the trainer saves the model to the host and runs a forward-only
pass over the whole batch (``_fit_compute_log_prob`` -> ``FullyAsyncTrainer.
_compute_old_log_prob``), giving ``old_log_probs`` = log pi_old. The driver then computes
the token TIS weights ``clamp(exp(old - rollout), max=upper)``
(``rollout_corr_helper.compute_rollout_correction_and_add_to_batch``) and the actor loss
uses ``ratio = exp(log_prob - old)``.

**Why it can be skipped on this island**: with ``ppo_epochs == 1``, one PPO mini-batch
per round and ``trigger_parameter_sync_step == 1``, ``update_actor`` takes exactly one
optimizer step and its forward runs on the very weights the recompute used. So
pi_old == pi_theta (up to the numerics of a second forward), ``ratio == 1`` and the
clip never fires. Setting ``old = log_prob.detach()`` inside the loss and computing the
TIS weights from ``log_prob.detach() - rollout`` gives the same objective and the same
gradient (``d ratio = ratio * d log_prob`` at ratio 1). This is **not** verl's bypass
mode, where ``old = rollout`` and the PPO ratio becomes pi_theta / pi_rollout (with
clipping against the rollout policy and no TIS weight): that changes the algorithm.

**Modes** (:data:`ENV`, read in the trainer actor; default off):

* ``on`` -- skip the recompute (and the per-round ``save_model_to_cpu``); the worker loss
  takes ``old = log_prob.detach()`` and recomputes the TIS weights there;
* ``measure`` -- keep verl's recompute and change nothing in the loss, but log the
  per-token difference between the recomputed ``old_log_probs`` and the update pass's
  ``log_prob.detach()`` (same batch, same weights) -- the equivalence evidence;
* anything else -- off.

The worker side is a hook in ``verl/workers/utils/losses.py:ppo_loss``
(``patch_verl.ONPOLICY_OLDLP``) that does something only when the trainer put
:data:`META_KEY` into the batch meta (it reaches the loss as non-tensor data). In ``on``
mode the trainer fills ``old_log_probs`` with NaN, so a worker without the hook cannot
silently train with a placeholder (the loss and the gradient would be NaN), and the
trainer refuses the step when the hook's :data:`ACTIVE` metric did not come back.

Both modes report the same TIS statistics from the worker (sums over tokens, so the
ratio is the global token mean whatever the micro-batch split): ``rollout_is_mean`` and
``rollout_is_ratio_fraction_high`` under :data:`PREFIX`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

ENV = "YETO_ONPOLICY_OLD_LOGPROB"
META_KEY = "yeto_onpolicy_old_logprob"
PREFIX = "yeto_oplp/"
ACTIVE = PREFIX + "active"
MODES = ("on", "measure")
# compute_policy_loss_* that read old_log_prob only through the ratio exp(log_prob - old)
SUPPORTED_LOSS_MODES = ("vanilla",)


class OnPolicyReuseError(RuntimeError):
    """The island's configuration does not make old == current; reuse is refused."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    """``on`` / ``measure`` / ``off`` from :data:`ENV`."""
    value = str((os.environ if environ is None else environ).get(ENV, "")).strip().lower()
    if value in ("1", "on", "true", "yes"):
        return "on"
    return value if value in MODES else "off"


def _get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, Mapping):
        return node.get(key, default)
    getter = getattr(node, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(node, key, default)


def check_config(config: Any, *, batch_rows: int, local_trigger_step: int = 1,
                 trigger_parameter_sync_step: int | None = None) -> dict:
    """Refuse every configuration in which the update pass does not run on the
    recompute's weights, or in which ``old_log_probs`` is read by something other than
    the PPO ratio and the TIS weights. Returns the TIS settings the worker needs."""
    problems = []
    actor = _get(_get(config, "actor_rollout_ref"), "actor")
    rollout = _get(_get(config, "actor_rollout_ref"), "rollout")
    algorithm = _get(config, "algorithm")
    epochs = int(_get(actor, "ppo_epochs", 1) or 1)
    if epochs != 1:
        problems.append(f"actor.ppo_epochs={epochs} (must be 1)")
    n = int(_get(rollout, "n", 1) or 1)
    mini = int(_get(actor, "ppo_mini_batch_size", 0) or 0) * n
    if mini <= 0 or int(batch_rows) != mini:
        problems.append(f"batch of {batch_rows} rows is not exactly one PPO mini-batch of "
                        f"ppo_mini_batch_size x rollout.n = {mini} rows (must be one optimizer step)")
    trigger = trigger_parameter_sync_step
    if trigger is None:
        trigger = _get(_get(config, "async_training"), "trigger_parameter_sync_step", 1)
    if int(trigger or 1) != 1 or int(local_trigger_step) != 1:
        problems.append(f"trigger_parameter_sync_step={trigger}, local_trigger_step={local_trigger_step} "
                        "(verl recomputes old with the version-1 weights then, not the current ones)")
    loss_mode = _get(_get(actor, "policy_loss"), "loss_mode", "vanilla")
    if loss_mode not in SUPPORTED_LOSS_MODES:
        problems.append(f"actor.policy_loss.loss_mode={loss_mode!r} (supported: {SUPPORTED_LOSS_MODES})")
    if _get(algorithm, "use_kl_in_reward", False):
        problems.append("algorithm.use_kl_in_reward=True (the KL reward reads old_log_probs)")
    estimator = str(_get(algorithm, "adv_estimator", "grpo"))
    if "optimal_token_baseline" in estimator.lower():
        problems.append(f"algorithm.adv_estimator={estimator} (reads old_log_probs)")
    rc = _get(algorithm, "rollout_correction")
    if _get(rc, "bypass_mode", False):
        problems.append("rollout_correction.bypass_mode=True (bypass is a different algorithm)")
    if _get(rc, "rollout_rs", None) is not None:
        problems.append(f"rollout_correction.rollout_rs={_get(rc, 'rollout_rs')!r} (rejection "
                        "sampling edits response_mask before the loss; not reproduced)")
    if _get(rc, "rollout_is_batch_normalize", False):
        problems.append("rollout_correction.rollout_is_batch_normalize=True (needs the batch "
                        "mean before the loss; not reproduced)")
    if _get(rc, "score_centering", False):
        problems.append("rollout_correction.score_centering=True")
    level = _get(rc, "rollout_is", None)
    if level not in (None, "token", "sequence"):
        problems.append(f"rollout_correction.rollout_is={level!r}")
    if problems:
        raise OnPolicyReuseError(f"{ENV}: old-log-prob reuse refused: " + "; ".join(problems))
    threshold = _get(rc, "rollout_is_threshold", 2.0)
    return {"rollout_is": level, "rollout_is_threshold": threshold}


def meta(mode_: str, tis: Mapping[str, Any]) -> str:
    """The batch meta value (a JSON string: plain non-tensor data through TensorDict)."""
    return json.dumps({"mode": mode_, **tis}, sort_keys=True)


def nan_placeholder(batch: Any) -> None:
    """``old_log_probs`` for ``on`` mode: NaN shaped like the rollout log-probs, so the
    loss is NaN unless the worker hook replaces it (module doc)."""
    import torch

    rollout = batch.batch["rollout_log_probs"]
    batch.batch["old_log_probs"] = torch.full_like(rollout, float("nan"), dtype=torch.float32)


def drop_driver_correction(batch: Any, metrics: dict) -> None:
    """In ``on`` mode the driver's rollout correction ran on the NaN placeholder: drop
    its weights (the worker recomputes them) and its metrics (NaN, not measurements)."""
    if "rollout_is_weights" in batch.batch.keys():
        batch.batch.pop("rollout_is_weights")
    for key in [k for k in metrics if k.startswith("rollout_corr/")]:
        metrics.pop(key)


def derived_metrics(metrics: Mapping[str, Any], mode_: str) -> dict:
    """Global token means from the worker's sums (``actor/`` prefix as verl renames)."""
    def get(name):
        for key in (f"actor/{PREFIX}{name}", f"{PREFIX}{name}"):
            if key in metrics:
                return float(metrics[key])
        return None

    out = {}
    active = get("active")
    if active is None or active < 1.0:
        raise OnPolicyReuseError(f"{ENV}={mode_}: the worker loss hook did not run "
                                 "(verl not patched with patch_verl ONPOLICY_OLDLP?)")
    tok = get("tok")
    if tok:
        if (w := get("is_w_sum")) is not None:
            out[f"{PREFIX}rollout_is_mean"] = w / tok
        if (h := get("is_high_sum")) is not None:
            out[f"{PREFIX}rollout_is_ratio_fraction_high"] = h / tok
        if (d := get("diff_sum")) is not None:
            out[f"{PREFIX}old_vs_cur_abs_diff_mean"] = d / tok
    return out


def in_loss(spec: str, log_prob, old_log_prob, data, rollout_is_weights):
    """The worker hook (``ppo_loss``, after ``data.select(...).to_padded_tensor()``).

    Returns ``(old_log_prob, rollout_is_weights, stats)``; ``stats`` maps a metric name
    to ``(value, aggregation)`` with aggregation ``sum`` or ``max``."""
    import torch

    cfg = json.loads(spec)
    mode_ = cfg["mode"]
    mask = data["response_mask"].to(bool)
    current = log_prob.detach()
    if "rollout_log_probs" not in data.keys():
        raise OnPolicyReuseError(f"{ENV}={mode_}: rollout_log_probs did not reach the loss")
    rollout = data["rollout_log_probs"].to(current.dtype)
    if rollout.shape != current.shape:
        raise OnPolicyReuseError(f"{ENV}: rollout_log_probs {tuple(rollout.shape)} vs log_prob "
                                 f"{tuple(current.shape)}")
    level, threshold = cfg.get("rollout_is"), cfg.get("rollout_is_threshold", 2.0)
    stats = {"active": (1.0, "max")}
    maskf = mask.to(current.dtype)
    stats["tok"] = (float(maskf.sum()), "sum")

    if mode_ == "on":
        old_log_prob = current
        if level is not None:
            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

            with torch.no_grad():
                rollout_is_weights, _ = compute_rollout_correction_weights(
                    log_ratio=current - rollout, response_mask=maskf, rollout_is=level,
                    rollout_is_threshold=threshold, rollout_is_batch_normalize=False)
        log_ratio = current - rollout
    else:  # measure: the loss is untouched
        diff = (current - old_log_prob.to(current.dtype)).abs().masked_fill(~mask, 0.0)
        stats["diff_sum"] = (float(diff.sum()), "sum")
        stats["diff_max"] = (float(diff.max()) if diff.numel() else 0.0, "max")
        log_ratio = old_log_prob.to(current.dtype) - rollout

    if rollout_is_weights is not None:
        stats["is_w_sum"] = (float((rollout_is_weights * maskf).sum()), "sum")
        from verl.trainer.ppo.rollout_corr_helper import SAFETY_BOUND

        upper = float(str(threshold).split("_")[-1])
        raw = torch.exp(torch.clamp(log_ratio, -SAFETY_BOUND, SAFETY_BOUND))
        stats["is_high_sum"] = (float(((raw > upper) & mask).sum()), "sum")
    return old_log_prob, rollout_is_weights, stats
