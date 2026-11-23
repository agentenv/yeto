"""Breakers of the monitor layer: read-only checks that stop nothing.

* ``monitor.dashboard_*``: the dashboard's alert rules (:mod:`yeto.dashboard.alerts`) and
  the ``run status`` alerts built from them. The numbers live here and only here:
  ``alerts.DEFAULT_THRESHOLDS`` is built from these params, whose names are the
  threshold keys (``--thresholds file.json`` overrides them by those names);
  :data:`DASHBOARD_RULES` maps an alert's ``rule`` to its entry.
* ``monitor.health_*``: ``yeto breaker check`` over a run's tapes (:mod:`yeto.rl.breaker`);
  the limits are that module's constants.

Every entry here is an alert: the only action is ``warn``.
"""

from __future__ import annotations

from yeto.breakers.spec import Breaker, Param
from yeto.rl.breaker import LIMITS, LOGPROB_LIMITS, NAN_SKIP_MIN_ROUNDS

_DASH = "yeto.dashboard.alerts"
_HEALTH = "yeto.rl.breaker"


def _s(name: str, default: float, doc: str = "") -> Param:
    return Param(name, float, default, unit="s", lo=0, doc=doc)


def _n(name: str, default: int, doc: str = "") -> Param:
    return Param(name, int, default, unit="n", lo=0, doc=doc)


def _x(name: str, default: float, unit: str = "ratio", doc: str = "") -> Param:
    return Param(name, float, default, unit=unit, lo=0, doc=doc)


def _alert(name: str, params: tuple[Param, ...], gauge: str, doc: str, owner: str = _DASH, **kw) -> Breaker:
    return Breaker(name=f"monitor.{name}", layer="monitor", kind="alert", params=params, actions=("warn",),
                   default_action="warn", gauge=gauge, owner=owner, doc=doc, **kw)


DASHBOARD: tuple[Breaker, ...] = (
    _alert("dashboard_heartbeat", (
        _s("heartbeat_warn_s", 60.0), _s("heartbeat_severe_s", 300.0)),
        "elapsed", "an island's last heartbeat (or event) is older than this"),
    _alert("dashboard_startup", (
        # island still starting (no rl_driver_start yet: weight load, Ray, engine init;
        # FN 2x8 measured 10-15 min)
        _s("startup_warn_s", 1200.0), _s("startup_severe_s", 1800.0),
        # the actor init stage of a trainer start (model build + weight load: trainer_start_phase
        # actor_import_done -> actor_init_done, or rl_startup_step before weights_loaded) writes
        # no event for its whole length; K3.1 mbridge load measured ~22 min
        _s("startup_init_warn_s", 2700.0), _s("startup_init_severe_s", 3600.0),
        # one startup sub-step (Ray up / engines ready / weights loaded) taking longer than
        # this while heartbeats still arrive (process alive, step stuck)
        _s("startup_step_warn_s", 900.0)),
        "elapsed", "an island still starting writes no event, or one startup step is slow"),
    _alert("dashboard_missed", (
        _n("missed_consecutive_warn", 2), _n("missed_consecutive_severe", 5)),
        "count", "an island missed this many synchronous rounds in a row"),
    _alert("dashboard_quorum", (
        _n("quorum_recent_rounds", 7), _n("quorum_min_rounds", 10), _x("quorum_factor", 3.0, unit="x")),
        "ratio", "the recent median quorum latency is this many times the baseline"),
    _alert("dashboard_grad_norm_spike", (
        _n("grad_norm_window", 8), _x("grad_norm_spike_factor", 5.0, unit="x")),
        "ratio", "grad_norm above this many times the median of the previous window (NaN / Inf: critical)"),
    _alert("dashboard_clip_kl", (
        _x("clipfrac_info", 0.2), _x("kl_warn", 0.1, unit="")),
        "level", "the latest clip fraction (info) or KL (warning) above its line"),
    _alert("dashboard_budget", (
        _x("budget_warn_pct", 60.0, unit="pct"), _x("budget_severe_pct", 80.0, unit="pct")),
        "level", "the cost estimate against the dashboard's --budget", depends_on=("D12",)),
    Breaker(
        name="monitor.dashboard_sft_reference", layer="monitor", kind="semantic",
        params=(
            # SFT page reference lines (not alerts), 0 = none drawn: the expected step time
            # of the recipe, the accelerator's HBM limit, the syncer host's RSS cap;
            # "slow step" = step time above the reference x factor
            _s("sft_ref_step_s", 0.0), _x("sft_hbm_limit_gib", 0.0, unit="gib"),
            _x("sft_syncer_rss_limit_bytes", 0.0, unit="bytes"), _x("sft_slow_factor", 1.1, unit="x"),
            # SFT island status: "stopped" after no step for factor x step time p50 + slack
            _x("sft_stop_factor", 1.5, unit="x"), _s("sft_stop_slack_s", 300.0)),
        native="dashboard --thresholds", owner=_DASH,
        doc="reference lines and the stopped status of the SFT page; no alert"),
    _alert("dashboard_compaction", (
        # CompactionRL (sampler rounds' agentic.compaction), over each sampler's last
        # compaction_window_rounds rounds that carry it, once it has compaction_min_trials
        # summaries / compacted trials: summary_ok share below warn / severe; share of the
        # compacted trials truncated after compacting above warn; share of trials that
        # compacted at all above info (the context budget is small for the tasks)
        _n("compaction_window_rounds", 5), _n("compaction_min_trials", 5),
        _x("compaction_summary_ok_warn", 0.9), _x("compaction_summary_ok_severe", 0.5),
        _x("compaction_truncated_warn", 0.2), _x("compaction_frac_info", 0.5),
        # share of multi-segment trials not written (lost: drain timeout / other version /
        # aborted; dropped: group not written) minus the share of one-segment trials
        _x("compaction_long_loss_gap", 0.2),
        # trials the gateway invalidated for a fork / history rewrite of a compacted session
        # (compaction_fork_lost), as a share of the compacted trials (written + fork-lost)
        _x("compaction_fork_lost_warn", 0.05)),
        "ratio", "context compaction of agentic trials: failing summaries, truncation, lost long trials"),
    _alert("dashboard_trial_fail", (
        # agentic samplers: trials that ended over the last trial_fail_window_rounds rounds of the
        # current incarnation in which any trial ended, once trial_fail_min_trials ended: none
        # with a verdict (ok / policy_limit) is critical, an aborted (infra) share above warn
        # a warning. Mirrors the run breaker's "every group INFRA" rule on the page.
        _n("trial_fail_window_rounds", 10), _n("trial_fail_min_trials", 8), _x("trial_fail_warn", 0.5)),
        "ratio", "agentic trials ending on infrastructure"),
    _alert("dashboard_trainer_missing", (
        # an async run whose samplers are up while no trainer island is on the tapes (its
        # launch failed, or it was torn down): seconds since the first sampler event
        _s("trainer_missing_warn_s", 1200.0), _s("trainer_missing_severe_s", 1800.0)),
        "elapsed", "samplers running without a trainer"),
    _alert("dashboard_trainer_failed", (
        # trainer launches that ended with a failure leave (island_leave not STOPPED / DONE ...)
        # in the last trainer_fail_window_s: one is a warning, trainer_fail_loop or more critical
        _s("trainer_fail_window_s", 3600.0), _n("trainer_fail_loop", 2)),
        "count", "failed trainer launches in the window"),
    _alert("dashboard_stale_read_position", (
        # a sampler waiting on a full backlog (as it counts it) while the trainers' read
        # position leaves less than its cap: it reported within this many seconds
        _s("stale_read_recent_s", 300.0),),
        "elapsed", "a sampler waits on a read position the trainers have moved past"),
    _alert("dashboard_release_stalled", (
        # a trainer skipping releases because the previous upload is still in flight, with no
        # release since the first skip of the streak, for this long
        _s("release_stall_warn_s", 600.0),),
        "elapsed", "a policy release stuck uploading (critical when samplers wait on it)"),
)

# alert ``rule`` (yeto.dashboard.alerts) -> the entry whose params judge it; rules without a
# threshold (store_unreadable, recovery_required, source_lost) are not here
DASHBOARD_RULES: dict[str, str] = {
    "heartbeat": "monitor.dashboard_heartbeat",
    "startup": "monitor.dashboard_startup",
    "startup_step": "monitor.dashboard_startup",
    "missed": "monitor.dashboard_missed",
    "quorum": "monitor.dashboard_quorum",
    "grad_norm_nan": "monitor.dashboard_grad_norm_spike",
    "grad_norm_spike": "monitor.dashboard_grad_norm_spike",
    "clipfrac": "monitor.dashboard_clip_kl",
    "kl": "monitor.dashboard_clip_kl",
    "budget": "monitor.dashboard_budget",
    "compaction_summary": "monitor.dashboard_compaction",
    "compaction_truncated": "monitor.dashboard_compaction",
    "compaction_long_lost": "monitor.dashboard_compaction",
    "compaction_fork_lost": "monitor.dashboard_compaction",
    "compaction_frac": "monitor.dashboard_compaction",
    "trial_fail": "monitor.dashboard_trial_fail",
    "trainer_missing": "monitor.dashboard_trainer_missing",
    "trainer_failed": "monitor.dashboard_trainer_failed",
    "stale_read_position": "monitor.dashboard_stale_read_position",
    "release_stalled": "monitor.dashboard_release_stalled",
}


def _limit(key: str, name: str) -> Param:
    value = LIMITS[key][0]
    return Param(name, type(value), value, unit="n" if name.endswith("_n") else "ratio", lo=0)


def _health(key: str, doc: str) -> Breaker:
    direction = LIMITS[key][1]
    return _alert(f"health_{key}", (_limit(key, f"{direction}_ratio"),), "ratio", doc, owner=_HEALTH,
                  direction=direction)


HEALTH: tuple[Breaker, ...] = (
    _alert("health_logprob", tuple(
        Param(f"{hw}_{level}_abs_diff", float, LOGPROB_LIMITS[hw][i], unit="", lo=0)
        for hw in ("amd", "nvidia", "ascend") for i, level in enumerate(("warn", "act"))),
        "level", "train/rollout log-prob mismatch per sampler hardware class (amd: the in-run xhw guard's "
                 "defaults)", owner=_HEALTH, depends_on=("D17",)),
    _health("is_clipped", "sc_is_clipped, mean over the window"),
    _health("is_weight", "sc_importance_weight, mean over the window"),
    _health("stale", "stale groups / (stale + trained)"),
    _health("flips", "changed / total weights of the last release (--flips)"),
    _health("aborted", "agentic trials lost / trials"),
    _health("zero_call", "agentic trials retried before any model call / trials"),
    _alert("health_nan_skip", (
        _limit("nan_skip_consecutive", "max_consecutive_n"),
        _limit("nan_skip_fraction", "max_fraction"),
        Param("min_rounds_n", int, NAN_SKIP_MIN_ROUNDS, unit="n", lo=0,
              doc="trained rounds before the fraction is judged")),
        "ratio", "NaN-skipped rounds: in a row, and as a share", owner=_HEALTH, depends_on=("D18",)),
)

BREAKERS: tuple[Breaker, ...] = DASHBOARD + HEALTH
