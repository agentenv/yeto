"""Dependency rules D1-D21 (yeto.breakers.rules): one passing and one failing case per rule.

The rules read resolved values through a small stand-in for ``ResolvedBreakers``
(get / enabled / entry / entries / deps), so each case states exactly the values
it is about; the facts are built directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

import pytest

from yeto.breakers import rules
from yeto.breakers.rules_facts import AUTO, Backlog, Dispense, Facts, PoolFacts, TrainerFacts


@dataclass(frozen=True)
class FakeParam:
    name: str
    type: type = float
    nullable: bool = False
    lo: float | None = None
    unit: str = "s"


@dataclass(frozen=True)
class FakeSource:
    kind: str
    location: str = ""


@dataclass
class FakeEntry:
    spec: Any
    enabled: bool
    params: dict[str, Any]
    sources: dict[str, FakeSource] = field(default_factory=dict)


class FakeResolved:
    """``values``: ``{"<entry>.<param>": value}``; ``off``: disabled entries;
    ``sources``: ``{"<entry>.<param>": kind}`` (default ``default``);
    ``params``: ``{"<entry>.<param>": FakeParam}`` for D21; ``kinds``: entry kinds."""

    def __init__(self, values: dict[str, Any], *, off: tuple[str, ...] = (), sources: dict[str, str] | None = None,
                 params: dict[str, FakeParam] | None = None, kinds: dict[str, str] | None = None,
                 tolerance: float | None = None, overrides: dict[str, str] | None = None) -> None:
        self.pool = None
        self.deps = SimpleNamespace(outage_tolerance_s=tolerance, overrides=overrides or {})
        self._entries: dict[str, FakeEntry] = {}
        for path, value in values.items():
            name, param = path.rsplit(".", 1)
            entry = self._entries.get(name)
            if entry is None:
                spec = SimpleNamespace(name=name, kind=(kinds or {}).get(name, "breaker"), params=())
                entry = self._entries[name] = FakeEntry(spec, name not in off, {})
            entry.params[param] = value
            entry.sources[param] = FakeSource((sources or {}).get(path, "default"), path)
            fake = (params or {}).get(path, FakeParam(param, nullable=True))
            entry.spec.params = (*entry.spec.params, fake)
        for name in off:
            if name not in self._entries:
                self._entries[name] = FakeEntry(SimpleNamespace(name=name, kind="breaker", params=()), False, {})

    def get(self, path: str, default: Any = None) -> Any:
        name, param = path.rsplit(".", 1)
        entry = self._entries.get(name)
        return default if entry is None else entry.params.get(param, default)

    def entry(self, name: str) -> FakeEntry:
        return self._entries[name]

    def enabled(self, name: str) -> bool:
        return self._entries[name].enabled

    def entries(self, layer: str | None = None) -> list[FakeEntry]:
        return list(self._entries.values())


TRAINER = TrainerFacts(groups_per_step=8, group_size=8, steps=20, release_every=1, record_rounds=1,
                       backlog_need=32, single_buffer_trainer=True, max_policy_staleness=12, estimator="grpo",
                       backend="miles")


def agentic_pool(resolved: FakeResolved, **kw: Any) -> PoolFacts:
    base = dict(name="pool", resolved=resolved, sampling_n=8, agentic=True, agent_timeout_s=3600.0,
                provider="remote-docker", drain=True)
    return PoolFacts(**{**base, **kw})


def run(rule_id: str, resolved: FakeResolved, *, pools: tuple[PoolFacts, ...] = (),
        trainer: TrainerFacts | None = TRAINER, run_up: bool = True) -> list[Any]:
    facts = Facts(trainer=trainer, pools=pools, outage_tolerance_s=resolved.deps.outage_tolerance_s, run_up=run_up)
    return rules.evaluate(resolved, facts, only=[rule_id])


def levels(found: list[Any]) -> list[str]:
    return [f.level for f in found]


# -------------------------------------------------------------------- registry of rules

def test_every_rule_id_is_registered_once_with_decision_6_levels():
    registered = rules.rules()
    assert [r.id for r in registered] == list(rules.RULE_IDS)
    errors = {r.id for r in registered if r.max_level == "error"}
    assert errors == {"D2", "D3", "D6", "D8", "D15", "D16", "D21", "D19"}  # D19: the #608 refusal


def test_a_warn_rule_never_emits_an_error():
    rule = rules.Rule("D5", "x", "run", "warn", lambda ctx: [rules.finding("D5", "error", "m")])
    assert rules._capped(rule, rules.finding("D5", "error", "m")).level == "warn"


# -------------------------------------------------------------------- D1

def test_d1_warns_when_drain_timeout_is_never_effective():
    rb = FakeResolved({"sampler.drain.timeout_s": 2400, "sampler.drain.reload_max_defer_s": 600})
    found = run("D1", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["warn"] and "drain_timeout_s is 2400 s, above reload_max_defer_s 600 s" in found[0].message
    assert found[0].suggested == {"sampler.drain.timeout_s": 600}
    rb = FakeResolved({"sampler.drain.timeout_s": 1800, "sampler.drain.reload_max_defer_s": 1800})
    found = run("D1", rb, pools=(agentic_pool(rb, compaction=True),))  # a compacted trial is a long one
    assert levels(found) == ["warn"] and found[0].suggested == {"sampler.drain.reload_max_defer_s": 2400}


def test_d1_reports_the_effective_cut_as_info_and_skips_a_live_switch():
    rb = FakeResolved({"sampler.drain.timeout_s": 2400, "sampler.drain.reload_max_defer_s": 2400})
    found = run("D1", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["info"] and "2400 s (40 min)" in found[0].message
    assert run("D1", rb, pools=(agentic_pool(rb, drain=False),)) == []


# -------------------------------------------------------------------- D2

def test_d2_refuses_a_wall_clock_without_verifier_time_and_warns_below_300_s():
    rb = FakeResolved({"sampler.trial_wall_clock.limit_s": 3630})
    assert levels(run("D2", rb, pools=(agentic_pool(rb),))) == ["error"]
    rb = FakeResolved({"sampler.trial_wall_clock.limit_s": 3800})
    found = run("D2", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["warn"] and found[0].suggested == {"sampler.trial_wall_clock.limit_s": 3900}


def test_d2_passes_with_headroom():
    rb = FakeResolved({"sampler.trial_wall_clock.limit_s": 3900})
    assert run("D2", rb, pools=(agentic_pool(rb),)) == []


# -------------------------------------------------------------------- D3

def test_d3_refuses_an_explicit_ttl_below_wall_clock_plus_grace():
    rb = FakeResolved({"sampler.trial_wall_clock.limit_s": 7200, "sampler.drain.abandon_grace_s": 120,
                       "harness.sandbox_ttl.limit_s": 7300})
    found = run("D3", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["error"] and "reclaim a sandbox under a live trial" in found[0].message


def test_d3_passes_the_derived_agentenv_ttl_and_refuses_a_margin_below_the_grace():
    values = {"sampler.trial_wall_clock.limit_s": 7200, "sampler.drain.abandon_grace_s": 120,
              "harness.sandbox_ttl.limit_s": None, "harness.sandbox_ttl.margin_s": 900}
    rb = FakeResolved(values)
    assert run("D3", rb, pools=(agentic_pool(rb, provider="agentenv"),)) == []
    rb = FakeResolved({**values, "harness.sandbox_ttl.margin_s": 60})
    assert levels(run("D3", rb, pools=(agentic_pool(rb, provider="agentenv"),))) == ["error"]


# -------------------------------------------------------------------- D4

D4_VALUES = {"trainer.sample_wait.timeout_s": 3600, "sampler.trial_wall_clock.limit_s": 3900,
             "sampler.infra_retries.max_n": 2, "sampler.drain.timeout_s": 2400,
             "sampler.drain.reload_max_defer_s": 2400, "harness.setup_retries.max_n": 0}


def test_d4_warns_when_the_sample_wait_is_below_the_longest_group():
    rb = FakeResolved(D4_VALUES)
    found = run("D4", rb, pools=(agentic_pool(rb),))
    # 3900 + 2 x 20 (one failed start) + 2400 = 6340 s
    assert levels(found) == ["warn"] and found[0].suggested == {"trainer.sample_wait.timeout_s": 6340}


def test_d4_checks_the_declared_outage_and_passes_a_long_wait():
    rb = FakeResolved({**D4_VALUES, "trainer.sample_wait.timeout_s": 7200}, tolerance=10800)
    found = run("D4", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["warn"] and "outage_tolerance_s is 10800 s (3 h)" in found[0].message
    rb = FakeResolved({**D4_VALUES, "trainer.sample_wait.timeout_s": 7200})
    assert run("D4", rb, pools=(agentic_pool(rb),)) == []


# -------------------------------------------------------------------- D5

def test_d5_warns_when_the_stall_timer_fires_first():
    found = run("D5", FakeResolved({"launch.stall.timeout_s": 900, "trainer.sample_wait.timeout_s": 3600}))
    assert levels(found) == ["warn"] and found[0].suggested == {"launch.stall.timeout_s": 3600}


def test_d5_passes_when_the_stall_timer_covers_the_wait_or_is_off():
    assert run("D5", FakeResolved({"launch.stall.timeout_s": 7200, "trainer.sample_wait.timeout_s": 3600})) == []
    assert run("D5", FakeResolved({"launch.stall.timeout_s": 900, "trainer.sample_wait.timeout_s": 3600},
                                  off=("launch.stall",))) == []


@pytest.mark.parametrize("stall", [120, 900, 3599])
def test_d5_passes_when_a_buffer_trainer_heartbeats_while_it_waits(stall):
    """A buffer trainer emits rl_sample_wait every max(60 s, stall / 4) while it waits: two
    beats fit in the stall window, so the stall timer no longer ends a legitimate wait."""
    rb = FakeResolved({"launch.stall.timeout_s": stall, "trainer.sample_wait.timeout_s": 3600})
    assert run("D5", rb, trainer=replace(TRAINER, buffer_trainer=True)) == []


def test_d5_still_warns_without_a_heartbeat_that_covers_the_stall_window():
    rb = FakeResolved({"launch.stall.timeout_s": 100, "trainer.sample_wait.timeout_s": 3600})
    found = run("D5", rb, trainer=replace(TRAINER, buffer_trainer=True))  # a 60 s beat: one per window
    assert levels(found) == ["warn"] and found[0].suggested == {"launch.stall.timeout_s": 3600}
    rb = FakeResolved({"launch.stall.timeout_s": 900, "trainer.sample_wait.timeout_s": 3600})
    assert levels(run("D5", rb, trainer=replace(TRAINER, buffer_trainer=False))) == ["warn"]  # no buffer reader
    assert levels(run("D5", rb, trainer=None)) == ["warn"]  # the trainer argv is not known here


# -------------------------------------------------------------------- D6

SILENCE = {"run.trainer_silence.limit_s": None, "run.trainer_silence.stall_factor_n": 4,
           "run.trainer_silence.floor_s": 1800}


def test_d6_refuses_stall_off_without_an_explicit_silence_limit():
    rb = FakeResolved({"launch.stall.timeout_s": 900, **SILENCE}, off=("launch.stall",))
    found = run("D6", rb)
    assert levels(found) == ["error"] and "--rl-stall-timeout 0" in found[0].message


def test_d6_accepts_stall_off_with_an_explicit_silence_limit_and_skips_launch_alone():
    rb = FakeResolved({"launch.stall.timeout_s": 900, **SILENCE, "run.trainer_silence.limit_s": 3600},
                      off=("launch.stall",), sources={"run.trainer_silence.limit_s": "breakers"})
    assert run("D6", rb) == []
    rb = FakeResolved({"launch.stall.timeout_s": 900, **SILENCE}, off=("launch.stall",))
    assert run("D6", rb, run_up=False) == []  # yeto launch accepts --rl-stall-timeout 0


def test_d6_refuses_an_explicit_silence_below_the_stall_and_passes_the_derived_one():
    rb = FakeResolved({"launch.stall.timeout_s": 7200, **SILENCE, "run.trainer_silence.limit_s": 3600},
                      sources={"run.trainer_silence.limit_s": "breakers"})
    assert levels(run("D6", rb)) == ["error"]
    assert run("D6", FakeResolved({"launch.stall.timeout_s": 7200, **SILENCE})) == []


# -------------------------------------------------------------------- D7

def test_d7_warns_on_alert_at_stop_and_says_which_fires_first():
    rb = FakeResolved({"run.starvation.alert_s": 7200, "run.starvation.stop_s": 7200,
                       "trainer.sample_wait.timeout_s": 3600})
    found = run("D7", rb)
    assert levels(found) == ["warn", "info"] and "sample wait" in found[1].message


def test_d7_passes_alert_below_stop():
    rb = FakeResolved({"run.starvation.alert_s": 1800, "run.starvation.stop_s": 7200})
    assert run("D7", rb) == []


# -------------------------------------------------------------------- D8

def dispense_pool(rows: int, passes: int) -> PoolFacts:
    return agentic_pool(FakeResolved({}), dispense=Dispense("ns", passes, rows))


def test_d8_refuses_too_few_slots_and_warns_on_a_thin_margin():
    found = run("D8", FakeResolved({}), pools=(dispense_pool(39, 4),))
    assert levels(found) == ["error"] and "156 slots" in found[0].message
    found = run("D8", FakeResolved({}), pools=(dispense_pool(41, 4),))
    assert levels(found) == ["warn"] and "margin 2.5% < 10%" in found[0].message
    assert found[0].suggested == {"prompts.dispense.passes": 5}


def test_d8_passes_with_margin_and_skips_unknown_rows():
    assert run("D8", FakeResolved({}), pools=(dispense_pool(41, 5),)) == []
    pool = agentic_pool(FakeResolved({}), dispense=Dispense("ns", 1, None))
    assert run("D8", FakeResolved({}), pools=(pool,)) == []


def test_d8_override_downgrades_and_turns_off():
    found = run("D8", FakeResolved({}, overrides={"D8.dispense_coverage": "warn"}), pools=(dispense_pool(39, 4),))
    assert levels(found) == ["warn"] and found[0].overridden == "warn"
    assert run("D8", FakeResolved({}, overrides={"D8": "off"}), pools=(dispense_pool(39, 4),)) == []


# -------------------------------------------------------------------- D9

D9_VALUES = {"sampler.dispense_redo.max_n": 2, "sampler.infra_retries.max_n": 2, "harness.setup_retries.max_n": 0,
             "harness.setup_retries.backoff_initial_s": 5.0, "harness.setup_retries.backoff_max_s": 60.0}


def test_d9_warns_below_the_declared_outage_with_the_setup_retries_to_set():
    rb = FakeResolved(D9_VALUES, tolerance=1800)
    found = run("D9", rb, pools=(agentic_pool(rb, dispense=Dispense("ns", 1, None)),))
    assert levels(found) == ["warn"] and "about 3 min" in found[0].message
    need = found[0].suggested["harness.setup_retries.max_n"]
    assert need > 0 and "HARBOR_SETUP_RETRIES" in found[0].message
    # the suggested retries ride out the target
    from yeto.breakers.rules_timing import slot_outage_s
    ctx = rules.Ctx(rb, Facts(trainer=None), agentic_pool(rb, dispense=Dispense("ns", 1, None)))
    assert slot_outage_s(ctx, setup_retries=need) >= 1800 > slot_outage_s(ctx, setup_retries=need - 1)


def test_d9_is_a_structural_note_without_a_declared_outage():
    rb = FakeResolved(D9_VALUES)
    found = run("D9", rb, pools=(agentic_pool(rb, dispense=Dispense("ns", 1, None)),))
    assert levels(found) == ["info"] and "declare breakers.deps.outage_tolerance_s" in found[0].message
    assert run("D9", rb, pools=(agentic_pool(rb),)) == []  # no store dispenser


# -------------------------------------------------------------------- D10

def test_d10_warns_on_a_lease_below_the_write_deferral_and_notes_a_long_lease():
    rb = FakeResolved({"sampler.dispense_lease.lease_s": 600, "sampler.write_defer.limit_s": 600})
    found = run("D10", rb, pools=(agentic_pool(rb, dispense=Dispense("ns", 1, None)),))
    assert levels(found) == ["warn"] and found[0].suggested == {"sampler.dispense_lease.lease_s": 900.0}
    rb = FakeResolved({"sampler.dispense_lease.lease_s": 7200, "sampler.write_defer.limit_s": 600})
    found = run("D10", rb, pools=(agentic_pool(rb, dispense=Dispense("ns", 1, None)),))
    assert levels(found) == ["info"] and "Keep the default 900 s" in found[0].message


def test_d10_passes_the_default_lease():
    rb = FakeResolved({"sampler.dispense_lease.lease_s": 900, "sampler.write_defer.limit_s": 600})
    assert run("D10", rb, pools=(agentic_pool(rb, dispense=Dispense("ns", 1, None)),)) == []


# -------------------------------------------------------------------- D11 / D12

def test_d11_warns_on_an_explicit_override_above_the_run_budget():
    rb = FakeResolved({"run.budget.usd": 100, "launch.budget.usd": 150}, sources={"launch.budget.usd": "legacy"})
    pool_rb = FakeResolved({"pool.spend_cap.usd": 200}, sources={"pool.spend_cap.usd": "breakers"})
    found = run("D11", rb, pools=(agentic_pool(pool_rb),))
    assert levels(found) == ["warn", "warn"]
    assert found[0].suggested == {"launch.budget.usd": 100} and found[1].suggested == {"pool.spend_cap.usd": 100}


def test_d11_applies_to_a_launch_with_a_breakers_section():
    rb = FakeResolved({"run.budget.usd": 450, "launch.budget.usd": 600}, sources={"launch.budget.usd": "legacy"})
    assert levels(run("D11", rb, run_up=False)) == ["warn"]


def test_d11_passes_inherited_and_smaller_budgets():
    rb = FakeResolved({"run.budget.usd": 100, "launch.budget.usd": 100}, sources={"launch.budget.usd": "inherit"})
    assert run("D11", rb) == []
    rb = FakeResolved({"run.budget.usd": 100, "launch.budget.usd": 80}, sources={"launch.budget.usd": "legacy"})
    assert run("D11", rb) == []


def test_d12_notes_a_second_warn_line():
    rb = FakeResolved({"run.budget.usd": 100, "run.budget.warn_ratio": 0.8, "launch.budget.usd": 100, "launch.budget.warn_usd": 60},
                      sources={"launch.budget.warn_usd": "legacy"})
    found = run("D12", rb)
    assert levels(found) == ["info"] and found[0].suggested == {"launch.budget.warn_usd": 80.0}


def test_d12_passes_one_warn_ratio():
    rb = FakeResolved({"run.budget.usd": 100, "run.budget.warn_ratio": 0.8, "launch.budget.usd": 100, "launch.budget.warn_usd": 80},
                      sources={"launch.budget.warn_usd": "legacy"})
    assert run("D12", rb) == []
    rb = FakeResolved({"run.budget.usd": 100, "run.budget.warn_ratio": 0.8, "launch.budget.usd": 100,
                       "launch.budget.warn_usd": 60}, sources={"launch.budget.warn_usd": "legacy"}, off=("run.budget",))
    assert run("D12", rb) == []  # no run budget: nothing to match


# -------------------------------------------------------------------- D13 / D14

def test_d13_warns_when_modal_ends_the_trainer_before_the_wall_clock():
    rb = FakeResolved({"launch.provider_timeout.timeout_s": 3600, "run.wall_clock.hours": 6})
    found = run("D13", rb, trainer=replace(TRAINER, modal=True))
    assert levels(found) == ["warn"] and found[0].suggested == {"launch.provider_timeout.timeout_s": 21600}
    rb = FakeResolved({"launch.provider_timeout.timeout_s": 86400, "run.wall_clock.hours": 72})
    found = run("D13", rb, trainer=replace(TRAINER, modal=True))  # Modal's longest: the retries carry on
    assert levels(found) == ["info"] and "--modal-retries" in found[0].message


def test_d13_passes_a_short_run_or_a_trainer_off_modal():
    rb = FakeResolved({"launch.provider_timeout.timeout_s": 86400, "run.wall_clock.hours": 6})
    assert run("D13", rb, trainer=replace(TRAINER, modal=True)) == []
    rb = FakeResolved({"launch.provider_timeout.timeout_s": 3600, "run.wall_clock.hours": 72})
    assert run("D13", rb) == []


def test_d14_warns_when_run_up_gives_up_before_the_pool():
    rb = FakeResolved({"run.first_group.timeout_s": 3600, "pool.start_timeout.limit_s": 3600})
    found = run("D14", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["warn"] and found[0].suggested == {"run.first_group.timeout_s": 4800}


def test_d14_passes_a_long_first_group_wait():
    rb = FakeResolved({"run.first_group.timeout_s": 5400, "pool.start_timeout.limit_s": 3600})
    assert run("D14", rb, pools=(agentic_pool(rb),)) == []


# -------------------------------------------------------------------- D15 / D16

def test_d15_refuses_an_age_the_algorithm_does_not_tolerate():
    rb = FakeResolved({"trainer.policy_age.max_versions": 12})
    found = run("D15", rb, trainer=replace(TRAINER, max_policy_staleness=2))
    assert levels(found) == ["error"] and "execution.max_policy_staleness >= 12" in found[0].message


def test_d15_passes_a_tolerated_age_and_no_limit():
    assert run("D15", FakeResolved({"trainer.policy_age.max_versions": 12})) == []
    rb = FakeResolved({"trainer.policy_age.max_versions": 12}, off=("trainer.policy_age",))
    assert run("D15", rb, trainer=replace(TRAINER, max_policy_staleness=0)) == []


def test_d16_refuses_min_samples_above_n_and_below_2_for_grpo():
    rb = FakeResolved({"sampler.min_samples.min_n": 9})
    assert levels(run("D16", rb, pools=(agentic_pool(rb),))) == ["error"]
    rb = FakeResolved({"sampler.min_samples.min_n": 1})
    found = run("D16", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["error"] and "no baseline" in found[0].message


def test_d16_passes_2_of_8_and_1_for_a_critic_estimator():
    rb = FakeResolved({"sampler.min_samples.min_n": 2})
    assert run("D16", rb, pools=(agentic_pool(rb),)) == []
    rb = FakeResolved({"sampler.min_samples.min_n": 1})
    assert run("D16", rb, pools=(agentic_pool(rb),), trainer=replace(TRAINER, estimator="ppo")) == []


# -------------------------------------------------------------------- D17 / D18

def test_d17_warns_on_act_below_warn_and_notes_a_drifted_health_limit():
    rb = FakeResolved({"trainer.xhw_guard.warn_abs_diff": 0.15, "trainer.xhw_guard.act_abs_diff": 0.10,
                       "monitor.health_logprob.amd_warn_abs_diff": 0.12,
                       "monitor.health_logprob.amd_act_abs_diff": 0.10})
    assert levels(run("D17", rb)) == ["warn", "info"]


def test_d17_passes_equal_constants_and_skips_a_disabled_guard_pair_check():
    rb = FakeResolved({"trainer.xhw_guard.warn_abs_diff": 0.15, "trainer.xhw_guard.act_abs_diff": 0.10,
                       "monitor.health_logprob.amd_warn_abs_diff": 0.15,
                       "monitor.health_logprob.amd_act_abs_diff": 0.10}, off=("trainer.xhw_guard",))
    assert run("D17", rb) == []


def test_d18_notes_drifted_nan_skip_limits_only_in_skip_mode():
    values = {"trainer.nan.skip": True, "trainer.nan.skip_max_consecutive_n": 3,
              "monitor.health_nan_skip.max_consecutive_n": 2}
    assert levels(run("D18", FakeResolved(values))) == ["info"]
    assert run("D18", FakeResolved({**values, "trainer.nan.skip": False})) == []
    assert run("D18", FakeResolved({**values, "trainer.nan.skip_max_consecutive_n": 2})) == []


# -------------------------------------------------------------------- D19

def backlog_pool(cap: int | str, target: int = 2, most: int = 2, over: int = 2) -> PoolFacts:
    return agentic_pool(FakeResolved({}), backlog=Backlog(cap, target, most, over))


def test_d19_refuses_caps_that_never_reach_the_first_release_and_warns_below_the_need():
    trainer = replace(TRAINER, release_every=6, backlog_need=8 * 8 + 8)
    found = run("D19", FakeResolved({}), pools=(backlog_pool(8),), trainer=trainer)
    assert levels(found) == ["error"] and "never release" in found[0].message
    found = run("D19", FakeResolved({}), pools=(backlog_pool(32),), trainer=trainer)
    assert levels(found) == ["warn"] and found[0].suggested == {"max_backlog_groups": 36}


def test_d19_passes_enough_caps_auto_and_synced_trainers():
    trainer = replace(TRAINER, release_every=6, backlog_need=72)
    assert run("D19", FakeResolved({}), pools=(backlog_pool(36),), trainer=trainer) == []
    assert run("D19", FakeResolved({}), pools=(backlog_pool(AUTO),), trainer=trainer) == []
    assert run("D19", FakeResolved({}), pools=(backlog_pool(8),),
               trainer=replace(trainer, single_buffer_trainer=False)) == []


# -------------------------------------------------------------------- D20 / D21

def test_d20_warns_below_a_declared_outage_only():
    rb = FakeResolved({"sampler.store_read_outage.limit_s": 1800}, tolerance=3600)
    found = run("D20", rb, pools=(agentic_pool(rb),))
    assert levels(found) == ["warn"] and found[0].suggested == {"sampler.store_read_outage.limit_s": 3600}
    rb = FakeResolved({"sampler.store_read_outage.limit_s": 1800})
    assert run("D20", rb, pools=(agentic_pool(rb),)) == []


def test_d21_refuses_a_zero_or_infinite_limit_of_an_enabled_breaker():
    rb = FakeResolved({"trainer.sample_wait.timeout_s": 0, "run.first_group.timeout_s": float("inf")},
                      params={"trainer.sample_wait.timeout_s": FakeParam("timeout_s"),
                              "run.first_group.timeout_s": FakeParam("timeout_s")},
                      sources={"trainer.sample_wait.timeout_s": "breakers", "run.first_group.timeout_s": "breakers"})
    found = run("D21", rb)
    assert levels(found) == ["error", "error"]
    assert "trainer.sample_wait.timeout_s is 0" in found[0].message


def test_d21_only_warns_on_a_legacy_zero():
    rb = FakeResolved({"run.budget.usd": 0}, params={"run.budget.usd": FakeParam("usd", unit="usd")},
                      sources={"run.budget.usd": "legacy"})  # budget.usd 0 with nothing priced never trips
    found = run("D21", rb)
    assert levels(found) == ["warn"] and "accepted as before" in found[0].message


def test_d21_passes_declared_zero_nullable_null_and_disabled_or_other_kinds():
    rb = FakeResolved({"pool.lease_overdue.grace_s": 0, "sampler.drain.timeout_s": None,
                       "launch.stall.timeout_s": 0, "sampler.infra_retries.max_n": 0},
                      params={"pool.lease_overdue.grace_s": FakeParam("grace_s", lo=0)},
                      off=("launch.stall",), kinds={"sampler.infra_retries": "retry_budget"})
    assert run("D21", rb) == []


# -------------------------------------------------------------------- reporting

def test_findings_are_formatted_with_the_value_to_set():
    f = rules.finding("D5", "warn", "m", suggested={"launch.stall.timeout_s": 3600})
    lines = []
    assert rules.report([f], lines.append, prefix="preflight: ")
    assert lines == ["preflight: WARNING: D5 warn m (suggested: launch.stall.timeout_s = 3600)"]
    assert not rules.report([rules.finding("D2", "error", "x")], lines.append)


@pytest.mark.parametrize("rule_id", rules.RULE_IDS)
def test_every_rule_skips_without_inputs(rule_id):
    rb = FakeResolved({})
    assert [f for f in run(rule_id, rb, pools=(agentic_pool(rb),), trainer=None) if f.level != "info"] == []
