"""Breakers of the trainer layer: the buffer reader, the round checks and the release drain.

Owners: the Miles sample-buffer reader (:mod:`yeto.rl.adapters.miles.buffer_rollout`), the
driver's per-round checks (:mod:`yeto.rl.engine.driver`, :mod:`yeto.rl.engine.nan_skip`,
:mod:`yeto.rl.engine.xhw_guard`) and the commanded trainer island
(:mod:`yeto.rl.engine.trainer_remote`). The trainer reads its values through
:mod:`yeto.rl.engine.trainer_breakers`.

``--rl-stall-timeout`` is ``launch.stall`` (the launch host enforces it), not an entry here.

Standard library only: defaults that live in heavy runtime modules are written as literals
here, and ``tests/test_breakers_trainer.py`` compares them with the runtime constants. The
composite env values (``YETO_RL_NAN_SKIP``, ``YETO_RL_XHW_GUARD``) are parsed by the
runtime's own parsers, imported when an alias is read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from yeto.breakers.spec import Alias, Breaker, Param

# yeto.rl.adapters.miles.buffer_rollout: WAIT_ENV / DEFAULT_WAIT_S, MAX_POLICY_AGE_ENV,
# MAX_DAMAGED_RUN_ENV / DEFAULT_MAX_DAMAGED_RUN
SAMPLE_WAIT_ENV = "YETO_SAMPLE_BUFFER_WAIT_S"
DEFAULT_SAMPLE_WAIT_S = 3600.0
POLICY_AGE_ENV = "YETO_SAMPLE_BUFFER_MAX_POLICY_AGE"
DAMAGED_RUN_ENV = "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN"
DEFAULT_DAMAGED_RUN = 8
# yeto.rl.engine.nan_skip: ENV and NanSkipPolicy's defaults
NAN_SKIP_ENV = "YETO_RL_NAN_SKIP"
NAN_SKIP_DEFAULTS = {"max_consecutive": 2, "max_fraction": 0.05, "min_rounds": 20}
# yeto.rl.engine.xhw_guard: ENV, OFF_VALUES, ACTIONS and XhwGuard's defaults
XHW_ENV = "YETO_RL_XHW_GUARD"
XHW_OFF = ("off", "0", "false", "")
XHW_ACTIONS = ("warn", "quarantine", "downweight", "stop")
XHW_DEFAULTS = {"warn_abs_diff": 0.10, "warn_kl": 0.03, "warn_clip": 0.01,
                "act_abs_diff": 0.15, "act_kl": 0.06, "act_clip": 0.03,
                "window": 3, "min_rounds": 2, "quarantine_rounds": 8,
                "weight": 0.5, "max_cond": 20.0, "confidence_z": 2.0}
# yeto.rl.engine.trainer_remote: RELEASE_DRAIN_ENV / DEFAULT_RELEASE_DRAIN_S
RELEASE_DRAIN_ENV = "YETO_TRAINER_RELEASE_DRAIN_S"
DEFAULT_RELEASE_DRAIN_S = 1800.0

STRICT = "none: a data-correctness check of the trainer (StrictRlInvariantError)"
READER = "yeto.rl.adapters.miles.buffer_rollout"
DRIVER = "yeto.rl.engine.driver"


def _damaged_run(raw: Any) -> int:
    """``int(env or DEFAULT)``: an empty value reads as the default, as the reader does."""
    text = str(raw).strip()
    return int(text) if text else DEFAULT_DAMAGED_RUN


def _release_drain(raw: Any) -> float:
    """``float(env or DEFAULT)``, as the commanded island reads it."""
    text = str(raw).strip()
    return float(text) if text else DEFAULT_RELEASE_DRAIN_S


def _nan_skip(raw: Any) -> bool:
    from yeto.rl.engine.nan_skip import parse_spec  # noqa: PLC0415 - light, and only when the env is set

    return parse_spec(raw) is not None


def _nan_key(key: str) -> Callable[[Any], Any]:
    def read(raw: Any) -> Any:
        from yeto.rl.engine.nan_skip import parse_spec  # noqa: PLC0415

        return (parse_spec(raw) or {}).get(key, NAN_SKIP_DEFAULTS[key])
    return read


def _xhw_key(key: str) -> Callable[[Any], Any]:
    def read(raw: Any) -> Any:
        from yeto.rl.engine.xhw_guard import parse_spec  # noqa: PLC0415

        return (parse_spec(raw) or {}).get(key, XHW_DEFAULTS[key])
    return read


def _xhw_action(raw: Any) -> str | None:
    from yeto.rl.engine.xhw_guard import parse_spec  # noqa: PLC0415

    return (parse_spec(raw) or {}).get("action")  # None: the env names no action (the default)


def _xhw_param(name: str, key: str, type_: type, unit: str, doc: str, **bounds: float) -> tuple[Param, Alias]:
    return (Param(name, type_, XHW_DEFAULTS[key], unit=unit, doc=doc, **bounds),
            Alias("trainer_env", XHW_ENV, name, off_values=XHW_OFF, to_canonical=_xhw_key(key), materialize=False))


_XHW = (
    _xhw_param("warn_abs_diff", "warn_abs_diff", float, "", "warn above this train/rollout log-prob abs diff", lo=0),
    _xhw_param("act_abs_diff", "act_abs_diff", float, "", "act above this abs diff", lo=0),
    _xhw_param("warn_kl", "warn_kl", float, "", "warn above this train/rollout KL", lo=0),
    _xhw_param("act_kl", "act_kl", float, "", "act above this KL", lo=0),
    _xhw_param("warn_clip", "warn_clip", float, "ratio", "warn above this IS clip fraction", lo=0, hi=1),
    _xhw_param("act_clip", "act_clip", float, "ratio", "act above this IS clip fraction", lo=0, hi=1),
    _xhw_param("window_n", "window", int, "n", "trailing rounds a source is judged on", lo=1),
    _xhw_param("min_rounds_n", "min_rounds", int, "n", "rounds of a source before it is judged", lo=1),
    _xhw_param("quarantine_rounds_n", "quarantine_rounds", int, "n",
               "rounds a quarantined source is passed over before a probe round", lo=1),
    _xhw_param("weight", "weight", float, "ratio", "advantage scale of a downweighted source", lo=0, hi=1),
    _xhw_param("max_cond", "max_cond", float, "", "a mixed estimate acts only below this condition number", lo=1),
    _xhw_param("confidence_z", "confidence_z", float, "",
               "standard errors a mixed estimate must clear the act limit by", lo=0),
)


BREAKERS: tuple[Breaker, ...] = (
    Breaker(
        name="trainer.sample_wait", layer="trainer", kind="breaker",
        params=(Param("timeout_s", float, DEFAULT_SAMPLE_WAIT_S, unit="s", lo=0,
                      doc="max seconds without a new group in the buffer while a round fills"),),
        actions=("stop",), default_action="stop", can_disable=False,
        # the launcher exports the flag's value as the env on the trainer island (the flag wins)
        aliases=(Alias("trainer_flag", "--rl-sample-wait-s", "timeout_s"),
                 Alias("trainer_env", SAMPLE_WAIT_ENV, "timeout_s", materialize=False,
                       shadowed_by=("--rl-sample-wait-s",))),
        depends_on=("D4", "D5"), legacy_events=("SampleBufferRolloutError",),
        gauge="elapsed", owner=READER,
        doc="the buffer reader ends the run when no new group arrives for timeout_s while a round fills"),
    Breaker(
        name="trainer.policy_age", layer="trainer", kind="guard",
        params=(Param("max_versions", int, None, unit="n", lo=1, nullable=True,
                      doc="published policy versions a trained group may lag"),),
        actions=("drop",), default_action="drop", default_enabled=False, enabled_by=("max_versions",),
        aliases=(Alias("trainer_flag", "--rl-max-policy-age", "max_versions", off_values=(0,)),
                 # always written by the launcher from the flag: read only where no argv is known
                 Alias("trainer_env", POLICY_AGE_ENV, "max_versions", off_values=(0,), materialize=False,
                       shadowed_by=("trainer_flag",))),
        depends_on=("D15",), legacy_events=("rl_buffer_stale_skipped",), gauge="count", owner=READER,
        doc="groups older than max_versions are consumed, not trained (0 / unset: every sample from the "
            "policy just published, no buffer trainer)"),
    Breaker(
        name="trainer.nan", layer="trainer", kind="breaker",
        params=(Param("skip", bool, False,
                      doc="skip a round with a non-finite grad_norm instead of stopping at once"),
                Param("skip_max_consecutive_n", int, NAN_SKIP_DEFAULTS["max_consecutive"], unit="n", lo=1,
                      doc="skip: stop after more skipped rounds in a row"),
                Param("skip_max_fraction", float, NAN_SKIP_DEFAULTS["max_fraction"], unit="ratio", lo=0, hi=1,
                      doc="skip: stop when skipped rounds exceed this fraction"),
                Param("skip_min_rounds_n", int, NAN_SKIP_DEFAULTS["min_rounds"], unit="n", lo=1,
                      doc="skip: rounds before the fraction is judged")),
        actions=("stop",), default_action="stop", can_disable=False,
        aliases=(Alias("trainer_env", NAN_SKIP_ENV, "skip", to_canonical=_nan_skip,
                       from_canonical=lambda on: "on" if on else ""),
                 Alias("trainer_env", NAN_SKIP_ENV, "skip_max_consecutive_n", to_canonical=_nan_key("max_consecutive"),
                       materialize=False),
                 Alias("trainer_env", NAN_SKIP_ENV, "skip_max_fraction", to_canonical=_nan_key("max_fraction"),
                       materialize=False),
                 Alias("trainer_env", NAN_SKIP_ENV, "skip_min_rounds_n", to_canonical=_nan_key("min_rounds"),
                       materialize=False)),
        depends_on=("D18",), legacy_events=("rl_strict_failure", "rl_nan_skip"), gauge="count",
        owner="yeto.rl.engine.nan_skip",
        doc="a non-finite grad_norm stops the trainer (strict); with skip, the round is skipped and the "
            "trainer stops past the skip limits (nan_skip_breaker)"),
    Breaker(
        name="trainer.xhw_guard", layer="trainer", kind="guard",
        params=tuple(p for p, _ in _XHW),
        actions=XHW_ACTIONS, default_action="quarantine", default_enabled=False,
        enabled_by=tuple(p.name for p, _ in _XHW),
        aliases=(Alias("trainer_env", XHW_ENV, None, off_values=XHW_OFF, on_values=("on", "1", "true")),
                 Alias("trainer_env", XHW_ENV, "action", off_values=XHW_OFF, to_canonical=_xhw_action,
                       materialize=False),
                 *(a for _, a in _XHW)),
        depends_on=("D17",), legacy_events=("rl_xhw_mismatch",), gauge="ratio", owner="yeto.rl.engine.xhw_guard",
        doc="per-source train/rollout log-prob mismatch: warn, quarantine or downweight a source, or stop"),
    Breaker(
        name="trainer.buffer_damaged", layer="trainer", kind="breaker",
        params=(Param("max_run_n", int, DEFAULT_DAMAGED_RUN, unit="n", lo=1,
                      doc="damaged payloads in a row (or that many, over half the take) at which a take raises"),),
        actions=("stop",), default_action="stop",
        aliases=(Alias("trainer_env", DAMAGED_RUN_ENV, "max_run_n", off_values=("0",), to_canonical=_damaged_run,
                       from_canonical=str),),
        legacy_events=("DamagedPayloadLimit",), gauge="count", owner=READER,
        doc="a systematic payload fault stops the take instead of consuming the backlog"),
    Breaker(
        name="trainer.release_drain", layer="trainer", kind="breaker",
        params=(Param("timeout_s", float, DEFAULT_RELEASE_DRAIN_S, unit="s", lo=0,
                      doc="max seconds a stopping island waits for its policy release upload"),),
        actions=("stop",), default_action="stop", can_disable=False,
        aliases=(Alias("trainer_env", RELEASE_DRAIN_ENV, "timeout_s", to_canonical=_release_drain,
                       from_canonical=lambda s: f"{s:g}"),),
        legacy_events=("RemoteTrainerError",), gauge="elapsed", owner="yeto.rl.engine.trainer_remote",
        doc="a commanded island stops without the upload still running after timeout_s (the samplers keep "
            "the previous release)"),
    # invariants (decision 1): shown, never configured, cannot be disabled
    Breaker(
        name="trainer.group_integrity", layer="trainer", kind="invariant", can_disable=False,
        actions=("drop",), default_action="drop", native=STRICT, gauge="count", owner=READER,
        legacy_events=("rl_buffer_stale_skipped",),
        doc="a group with fewer than 2 or more than n trials, or a non-finite reward or log-prob, is "
            "consumed, not trained"),
    Breaker(
        name="trainer.sample_alignment", layer="trainer", kind="invariant", can_disable=False,
        actions=("drop", "stop"), default_action="drop", native=STRICT, gauge="count", owner=READER,
        doc="a group whose samples do not convert to the trainer's form (R3 routing rows, segment "
            "layout) leaves the round; takes in a row without a convertible group stop the trainer"),
    Breaker(
        name="trainer.compaction_round", layer="trainer", kind="invariant", can_disable=False,
        actions=("stop",), default_action="stop", native=STRICT, owner=READER,
        doc="a round of compacted trials the trainer cannot train whole (DP schedule, --seq-length, "
            "segment siblings) is refused"),
    Breaker(
        name="trainer.nonfinite_reward", layer="trainer", kind="invariant", can_disable=False,
        actions=("stop",), default_action="stop", native=STRICT, owner=DRIVER,
        legacy_events=("rl_strict_failure",),
        doc="a batch with a non-finite group reward is refused before training"),
    Breaker(
        name="trainer.gradient_flow", layer="trainer", kind="invariant", can_disable=False,
        actions=("stop",), default_action="stop", native=STRICT, owner=DRIVER,
        legacy_events=("rl_strict_failure",),
        doc="grad_norm 0 while the round's advantages (or a mechanism) require a gradient"),
    Breaker(
        name="trainer.policy_hash", layer="trainer", kind="invariant", can_disable=False,
        actions=("stop",), default_action="stop", native=STRICT, owner=DRIVER,
        legacy_events=("rl_strict_failure",),
        doc="the trainer's policy differs from the cut it applied or was rebuilt from"),
    Breaker(
        name="trainer.strict_invariant", layer="trainer", kind="invariant", can_disable=False,
        actions=("stop",), default_action="stop", native=STRICT, owner=DRIVER,
        legacy_events=("rl_strict_failure",),
        doc="any other strict trainer check (layout hash, LR 0 before the final round, rollout "
            "statistics) ends the trainer abnormally"),
)
