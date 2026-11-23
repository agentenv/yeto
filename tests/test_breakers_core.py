"""yeto.breakers core: registry checks, resolution order, conflicts, legacy spellings,
write-back, invariants, switch groups, budget inheritance, events, CLI, import weight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pytest

from yeto import breakers
from yeto.breakers import cli as bcli
from yeto.breakers import events, registry as registry_mod
from yeto.breakers.resolve import Sources, from_json, materialize, parse_section, resolve
from yeto.breakers.spec import Alias, Breaker, BreakerConfigError, Param, RegistryError, SwitchGroup


def _nan_key(key, default):
    def parse(raw):
        fields = dict(kv.split("=", 1) for kv in str(raw).split(",") if "=" in kv)
        return type(default)(fields[key]) if key in fields else default
    return parse


def _entries():
    return [
        Breaker("run.budget", "run", "breaker", default_enabled=False, enabled_by=("usd",),
                params=(Param("usd", float, None, unit="usd", lo=0, nullable=True),
                        Param("warn_ratio", float, 0.8, unit="ratio", lo=0, hi=1)),
                actions=("stop",), default_action="stop", gauge="level",
                aliases=(Alias("run_config", "budget.usd", "usd", off_values=(None,)),)),
        Breaker("run.wall_clock", "run", "breaker", implicit=True, actions=("stop",), default_action="stop",
                params=(Param("hours", float, 3.0, unit="h", lo=0),),
                aliases=(Alias("run_config", "budget.wall_h", "hours", off_values=(None,)),)),
        Breaker("run.silence", "run", "breaker", params=(Param("limit_s", float, 1800.0, unit="s", lo=0),),
                aliases=(Alias("run_config", "budget.silence_min", "limit_s", scale=60),)),
        Breaker("launch.budget", "launch", "breaker", default_enabled=False, enabled_by=("usd",),
                params=(Param("usd", float, None, unit="usd", lo=0, nullable=True, inherit_from="run.budget.usd"),),
                actions=("stop",), default_action="stop",
                aliases=(Alias("launch_flag", "--budget-usd", "usd"),)),
        Breaker("launch.idle", "launch", "breaker", params=(Param("warn_s", float, 900.0, unit="s", lo=0),),
                actions=("stop",), default_action="stop",
                aliases=(Alias("launch_flag", "--idle-warn-minutes", "warn_s", scale=60, off_values=(0,)),)),
        Breaker("trainer.sample_wait", "trainer", "breaker",
                params=(Param("timeout_s", float, 3600.0, unit="s", lo=0),), actions=("stop",),
                default_action="stop", gauge="elapsed",
                aliases=(Alias("trainer_flag", "--rl-sample-wait-s", "timeout_s"),
                         Alias("env", "YETO_SAMPLE_BUFFER_WAIT_S", "timeout_s"))),
        Breaker("trainer.nan", "trainer", "breaker", default_enabled=False,
                params=(Param("max_consecutive_n", int, 2, unit="n", lo=0),),
                actions=("stop",), default_action="stop",
                aliases=(Alias("env", "YETO_RL_NAN_SKIP", None, on_values=("max_consecutive=*",)),
                         Alias("env", "YETO_RL_NAN_SKIP", "max_consecutive_n", materialize=False,
                               to_canonical=_nan_key("max_consecutive", 2)))),
        Breaker("sampler.dispense_redo", "sampler", "retry_budget", params=(Param("max_n", int, 2, unit="n", lo=0),),
                aliases=(Alias("spec", "prompts.dispense.max_redo", "max_n"),
                         Alias("pool_dispense", "max_redo", "max_n"),
                         Alias("pool_cli_flag", "--dispense-max-redo", "max_n"))),
        Breaker("sampler.drain", "sampler", "guard", actions=("drop",), default_action="drop",
                params=(Param("reload_max_defer_s", float, 600.0, unit="s", lo=0, nullable=True),),
                aliases=(Alias("spec", "reload_max_defer_s", "reload_max_defer_s"),)),
        Breaker("sampler.token_alignment", "sampler", "invariant", can_disable=False, actions=("drop",),
                default_action="drop", native="agentic round token checks"),
        Breaker("sampler.max_prompt_len", "sampler", "semantic", native="spec agentic.max_prompt_len"),
        Breaker("pool.sampler_restarts", "pool", "retry_budget",
                params=(Param("max_n", int, 2, unit="n", lo=0),),
                aliases=(Alias("pool_elastic", "max_sampler_restarts", "max_n"),)),
        Breaker("pool.crash_loop", "pool", "breaker", actions=("replace", "halt"), default_action="replace",
                aliases=(Alias("pool_elastic", "crash_loop_before_ready", "action"),)),
        Breaker("pool.start_timeout", "pool", "breaker", params=(Param("limit_s", float, 3600.0, unit="s", lo=0),),
                aliases=(Alias("pool_elastic", "start_timeout_s", "limit_s"),)),
        Breaker("pool.engine_stall", "pool", "breaker", params=(Param("limit_s", float, 600.0, unit="s", lo=0),),
                aliases=(Alias("pool_elastic", "engine_stall_s", "limit_s", off_values=(None,)),)),
        Breaker("monitor.heartbeat", "monitor", "alert", actions=("warn",), default_action="warn",
                params=(Param("limit_s", float, 300.0, unit="s", lo=0),)),
    ]


GROUPS = (SwitchGroup("cost_guard", ("launch.budget", "launch.idle"),
                      aliases=(Alias("launch_flag", "--no-cost-guard", None, nargs=0, off_values=(True,)),)),)


@pytest.fixture
def reg(monkeypatch):
    r = registry_mod.Registry(_entries(), GROUPS)
    monkeypatch.setattr(registry_mod, "_DEFAULT", r)
    return r


def _cfg(**extra):
    cfg = {"run": "r", "store": "s3://b/r", "trainer": {"argv": ["--training-mode", "rl"]}, "pool": {"config": "p"}}
    cfg.update(extra)
    return cfg


# --- registry --------------------------------------------------------------------------------------


def test_default_registry_loads():
    registry_mod._DEFAULT = None
    r = registry_mod.default_registry()
    assert all(b.can_disable is False for b in r if b.kind == "invariant")


@pytest.mark.parametrize("bad, needle", [
    (Breaker("sampler.x", "sampler", "invariant", native="n"), "cannot be disabled"),
    (Breaker("sampler.x", "sampler", "breaker", actions=("stop",), default_action="drop"), "default_action"),
    (Breaker("trainer.x", "sampler", "breaker"), "must be 'sampler.<entry>'"),
    (Breaker("sampler.x", "sampler", "breaker", params=(Param("t", float, None),)), "nullable"),
    (Breaker("sampler.x", "sampler", "breaker", params=(Param("t", float, 1.0, inherit_from="run.nope.usd"),)),
     "inherit_from"),
    (Breaker("sampler.x", "sampler", "breaker", aliases=(Alias("spec", "x", "missing"),)), "maps no param"),
    (Breaker("sampler.x", "sampler", "breaker", aliases=(Alias("spec", "x", "action"),)), "no actions"),
    (Breaker("sampler.x", "sampler", "semantic"), "native location"),
])
def test_registry_refuses_bad_declarations(bad, needle):
    with pytest.raises(RegistryError, match=needle):
        registry_mod.Registry([bad])


def test_registry_refuses_duplicates_and_shared_write_back():
    b = Breaker("sampler.x", "sampler", "breaker", params=(Param("n", int, 1),), aliases=(Alias("spec", "k", "n"),))
    with pytest.raises(RegistryError, match="declared twice"):
        registry_mod.Registry([b, b])
    c = Breaker("sampler.y", "sampler", "breaker", params=(Param("n", int, 1),), aliases=(Alias("spec", "k", "n"),))
    with pytest.raises(RegistryError, match="written back by both"):
        registry_mod.Registry([b, c])
    ro = Breaker("sampler.y", "sampler", "breaker", params=(Param("n", int, 1),),
                 aliases=(Alias("spec", "k", "n", materialize=False),))
    assert len(registry_mod.Registry([b, ro])) == 2


def test_switch_group_members_must_be_disableable():
    inv = Breaker("sampler.inv", "sampler", "invariant", can_disable=False, native="n")
    with pytest.raises(RegistryError, match="cannot be disabled"):
        registry_mod.Registry([inv], [SwitchGroup("g", ("sampler.inv",))])


# --- resolution order --------------------------------------------------------------------------------


def test_defaults_without_anything(reg):
    rb = resolve(Sources())
    assert rb.get("trainer.sample_wait.timeout_s") == 3600.0
    assert rb.entry("trainer.sample_wait").source.kind == "default"
    assert rb.get("run.budget.enabled") is False
    assert not rb.has_section


def test_order_section_then_alias_then_inherit_then_default(reg):
    # alias only
    rb = resolve(Sources(argv=["--rl-sample-wait-s", "7200"]))
    assert rb.get("trainer.sample_wait.timeout_s") == 7200.0
    assert rb.entry("trainer.sample_wait").sources["timeout_s"].location == "trainer.argv --rl-sample-wait-s"
    # section wins over the default, inherit over the default
    rb = resolve(Sources(section={"trainer": {"sample_wait": {"timeout_s": 5000}},
                                  "run": {"budget": {"usd": 450}}}))
    assert rb.get("trainer.sample_wait.timeout_s") == 5000
    assert rb.entry("launch.budget").sources["usd"].kind == "inherit"
    assert rb.get("launch.budget.usd") == 450
    # section and alias agree: the alias is redundant
    rb = resolve(Sources(argv=["--rl-sample-wait-s=5000"], section={"trainer": {"sample_wait": {"timeout_s": 5000}}}))
    assert rb.entry("trainer.sample_wait").redundant == ("trainer.argv --rl-sample-wait-s",)


def test_scale_and_env_strings(reg):
    rb = resolve(Sources(run_config=_cfg(budget={"silence_min": 45}), env={"YETO_SAMPLE_BUFFER_WAIT_S": "900"}))
    assert rb.get("run.silence.limit_s") == 2700
    assert rb.get("trainer.sample_wait.timeout_s") == 900


def test_pool_overrides(reg):
    section = {"pool": {"start_timeout": {"limit_s": 3600}}, "pools": {"modal": {"start_timeout": {"limit_s": 1800}}}}
    assert resolve(Sources(section=section)).get("pool.start_timeout.limit_s") == 3600
    rb = resolve(Sources(section=section), pool="modal")
    assert rb.get("pool.start_timeout.limit_s") == 1800
    assert rb.entry("pool.start_timeout").sources["limit_s"].location == "breakers.pools.modal.start_timeout.limit_s"


def test_layers_filter_keeps_only_those_conflicts(reg):
    sources = Sources(argv=["--rl-sample-wait-s", "7200"], env={"YETO_SAMPLE_BUFFER_WAIT_S": "60"},
                      spec={"prompts": {"dispense": {"max_redo": 3}}})
    rb = resolve(sources, layers=("sampler",))
    assert rb.get("sampler.dispense_redo.max_n") == 3 and "trainer.sample_wait" not in rb
    with pytest.raises(BreakerConfigError):
        resolve(sources, layers=("trainer",))


# --- conflicts ---------------------------------------------------------------------------------------


def test_section_and_alias_disagree(reg):
    sources = Sources(argv=["--rl-sample-wait-s", "7200"], section={"trainer": {"sample_wait": {"timeout_s": 5000}}})
    with pytest.raises(BreakerConfigError) as err:
        resolve(sources)
    assert "breakers.trainer.sample_wait.timeout_s = 5000" in str(err.value)
    assert "trainer.argv --rl-sample-wait-s = 7200" in str(err.value)
    rb = resolve(sources, strict=False)
    assert rb.conflicts[0].locations == ("breakers.trainer.sample_wait.timeout_s", "trainer.argv --rl-sample-wait-s")


def test_two_aliases_disagree(reg):
    sources = Sources(spec={"prompts": {"dispense": {"max_redo": 4}}}, pool_cli=["--dispense-max-redo", "2"])
    with pytest.raises(BreakerConfigError, match="spec prompts.dispense.max_redo = 4 but pool argv --dispense-max-redo = 2"):
        resolve(sources)
    agree = Sources(spec={"prompts": {"dispense": {"max_redo": 4}}}, pool_dispense={"max_redo": 4})
    assert resolve(agree).get("sampler.dispense_redo.max_n") == 4


def test_enabled_false_against_alias_value(reg):
    sources = Sources(argv=["--rl-sample-wait-s", "7200"], section={"trainer": {"sample_wait": {"enabled": False}}})
    with pytest.raises(BreakerConfigError, match="turns it off but trainer.argv --rl-sample-wait-s sets timeout_s"):
        resolve(sources)


def test_switch_off_against_member_enabled(reg):
    sources = Sources(argv=["--no-cost-guard"], section={"launch": {"idle": {"enabled": True}}})
    with pytest.raises(BreakerConfigError, match="switch group cost_guard is off"):
        resolve(sources)


def test_bad_alias_value_is_a_conflict(reg):
    with pytest.raises(BreakerConfigError, match="cannot be read"):
        resolve(Sources(argv=["--rl-sample-wait-s", "soon"]))
    with pytest.raises(BreakerConfigError, match=">= 0"):
        resolve(Sources(argv=["--rl-sample-wait-s", "-5"]))
    with pytest.raises(BreakerConfigError, match="not an action of pool.crash_loop"):
        resolve(Sources(pool_elastic={"crash_loop_before_ready": "explode"}))


@pytest.mark.parametrize("section, needle", [
    ({"trainer": {"nope": {}}}, "breakers.trainer.nope: unknown trainer entry"),
    ({"trainer": {"sample_wait": {"timeout": 5}}}, "breakers.trainer.sample_wait.timeout: unknown key"),
    ({"trainers": {}}, "breakers.trainers: unknown key"),
    ({"trainer": {"sample_wait": {"timeout_s": None}}}, "null is not allowed"),
    ({"trainer": {"sample_wait": {"timeout_s": "1h"}}}, "must be a finite number"),
    ({"trainer": {"sample_wait": {"action": "drop"}}}, "not an action of trainer.sample_wait"),
    ({"trainer": {"sample_wait": {"timeout_s": "inherit"}}}, "inherits from nothing"),
    ({"sampler": {"max_prompt_len": {"n": 1}}}, "configured at spec agentic.max_prompt_len"),
    ({"switches": {"cost": False}}, "unknown switch group"),
    ({"deps": {"outage_tolerance_s": -1}}, "outage_tolerance_s"),
    ({"deps": {"overrides": {"D8": "loud"}}}, "is not one of"),
])
def test_section_structure_errors(reg, section, needle):
    with pytest.raises(BreakerConfigError, match=needle.replace(".", r"\.").replace("(", r"\(")):
        parse_section(section)


def test_section_lists_every_problem(reg):
    with pytest.raises(BreakerConfigError) as err:
        parse_section({"trainer": {"nope": {}, "sample_wait": {"x": 1}}})
    assert len(err.value.problems) == 2


# --- invariants --------------------------------------------------------------------------------------


def test_invariant_cannot_be_disabled(reg):
    with pytest.raises(BreakerConfigError, match="cannot be disabled: a data-correctness check"):
        parse_section({"sampler": {"token_alignment": {"enabled": False}}})
    rb = resolve(Sources())
    assert rb.enabled("sampler.token_alignment")
    assert rb.entry("sampler.token_alignment").spec.can_disable is False


# --- legacy off spellings ----------------------------------------------------------------------------


def test_legacy_off_spellings(reg):
    rb = resolve(Sources(argv=["--idle-warn-minutes", "0"]))
    assert rb.enabled("launch.idle") is False
    assert rb.entry("launch.idle").sources["enabled"].location == "trainer.argv --idle-warn-minutes"
    assert rb.get("launch.idle.warn_s") == 900.0  # an off spelling sets no value
    rb = resolve(Sources(run_config=_cfg(budget={"usd": None, "wall_h": None})))
    assert not rb.enabled("run.budget") and not rb.enabled("run.wall_clock")
    rb = resolve(Sources(pool_elastic={"engine_stall_s": None}))
    assert not rb.enabled("pool.engine_stall")


def test_zero_is_a_number_not_off(reg):
    rb = resolve(Sources(pool_elastic={"max_sampler_restarts": 0}))
    assert rb.enabled("pool.sampler_restarts") and rb.get("pool.sampler_restarts.max_n") == 0


def test_legacy_null_means_default_unless_nullable(reg):
    rb = resolve(Sources(run_config=_cfg(budget={"silence_min": None})))
    assert rb.get("run.silence.limit_s") == 1800.0 and rb.entry("run.silence").sources["limit_s"].kind == "default"
    rb = resolve(Sources(spec={"reload_max_defer_s": None}))
    assert rb.get("sampler.drain.reload_max_defer_s") is None
    assert rb.entry("sampler.drain").sources["reload_max_defer_s"].kind == "legacy"


def test_converted_none_is_an_explicit_value():
    b = Breaker("run.starve", "run", "alert", params=(Param("alert_s", float, 1800.0, nullable=True),),
                aliases=(Alias("run_config", "starvation.alert_min", "alert_s",
                               to_canonical=lambda m: None if m == 0 else m * 60),))
    r = registry_mod.Registry([b])
    rb = resolve(Sources(run_config=_cfg(starvation={"alert_min": 0})), registry=r)
    assert rb.get("run.starve.alert_s") is None and rb.entry("run.starve").sources["alert_s"].kind == "legacy"
    rb = resolve(Sources(run_config=_cfg(starvation={})), registry=r)
    assert rb.get("run.starve.alert_s") == 1800.0


def test_composite_env_aliases(reg):
    rb = resolve(Sources(env={"YETO_RL_NAN_SKIP": "max_consecutive=5,max_fraction=0.1"}))
    assert rb.get("trainer.nan.max_consecutive_n") == 5


def test_implicit_default(reg):
    rb = resolve(Sources(run_config=_cfg()))
    assert [e.name for e in rb.implicit()] == ["run.wall_clock"]
    assert rb.entry("run.wall_clock").source.location == "default (implicit)"
    rb = resolve(Sources(run_config=_cfg(budget={"wall_h": 6})))
    assert rb.implicit() == []


def test_action_alias(reg):
    rb = resolve(Sources(pool_elastic={"crash_loop_before_ready": "halt"}))
    assert rb.get("pool.crash_loop.action") == "halt"
    assert rb.entry("pool.crash_loop").sources["action"].location == "pool elastic crash_loop_before_ready"
    with pytest.raises(BreakerConfigError, match="action"):
        resolve(Sources(pool_elastic={"crash_loop_before_ready": "halt"},
                        section={"pool": {"crash_loop": {"action": "replace"}}}))


# --- switch groups -----------------------------------------------------------------------------------


def test_one_switch_off(reg):
    rb = resolve(Sources(argv=["--no-cost-guard"]))
    assert not rb.enabled("launch.budget") and not rb.enabled("launch.idle")
    assert rb.entry("launch.idle").sources["enabled"].kind == "switch"
    assert rb.switches() == {"cost_guard": False}
    assert rb.entry("launch.idle").switch_groups == ("cost_guard",)
    rb = resolve(Sources(section={"switches": {"cost_guard": False}}))
    assert not rb.enabled("launch.idle")
    assert rb.entry("launch.idle").sources["enabled"].location == "breakers.switches.cost_guard"


# --- budget inheritance ------------------------------------------------------------------------------


def test_budget_inherits_only_from_the_section(reg):
    # a legacy budget.usd never reaches the launcher (today's behavior)
    rb = resolve(Sources.from_run_config(_cfg(budget={"usd": 45})))
    assert rb.get("run.budget.usd") == 45 and rb.enabled("run.budget")
    assert rb.get("launch.budget.usd") is None and not rb.enabled("launch.budget")
    # written once in the section: launch inherits it and turns on
    rb = resolve(Sources.from_run_config(_cfg(breakers={"run": {"budget": {"usd": 450}}})))
    assert rb.get("launch.budget.usd") == 450 and rb.enabled("launch.budget")
    assert rb.entry("launch.budget").sources["usd"].location == "inherit run.budget.usd"
    # an explicit override keeps its own source (D11 judges it)
    rb = resolve(Sources(section={"run": {"budget": {"usd": 450}}, "launch": {"budget": {"usd": 500}}}))
    assert rb.get("launch.budget.usd") == 500 and rb.entry("launch.budget").sources["usd"].kind == "breakers"
    # "inherit" written explicitly also takes a legacy parent
    rb = resolve(Sources.from_run_config(_cfg(budget={"usd": 45}, breakers={"launch": {"budget": {"usd": "inherit"}}})))
    assert rb.get("launch.budget.usd") == 45


def test_disabled_parent_gives_none(reg):
    rb = resolve(Sources(section={"run": {"budget": {"enabled": False, "usd": 450}},
                                  "launch": {"budget": {"usd": "inherit"}}}))
    assert rb.get("launch.budget.usd") is None and not rb.enabled("launch.budget")


# --- write-back --------------------------------------------------------------------------------------


def test_no_section_is_a_no_op(reg):
    argv = ["--training-mode", "rl", "--rl-sample-wait-s", "7200"]
    sources = Sources.from_run_config(_cfg(trainer={"argv": argv}), spec={"prompts": {"dispense": {"max_redo": 4}}},
                                      pool_elastic={"max_sampler_restarts": 0}, env={"A": "1"})
    out = materialize(resolve(sources), sources)
    assert out.argv == argv and out.spec == sources.spec and out.pool_elastic == sources.pool_elastic
    assert out.env == {"A": "1"} and out.changes == [] and out.run_config == sources.run_config
    assert out.spec is not sources.spec


def test_write_back_round_trip(reg):
    section = {"trainer": {"sample_wait": {"timeout_s": 7200}}, "launch": {"idle": {"enabled": False}},
               "sampler": {"dispense_redo": {"max_n": 6}}, "run": {"silence": {"limit_s": 2700}},
               "pool": {"crash_loop": {"action": "halt"}}}
    argv = ["--training-mode", "rl", "--rl-sample-wait-s=60"]
    cfg = _cfg(trainer={"argv": ["--training-mode", "rl"]}, breakers=section)
    sources = Sources.from_run_config(cfg, spec={"prompts": {}}, pool_elastic={})
    rb = resolve(sources)
    out = materialize(rb, sources)
    assert out.argv == ["--training-mode", "rl", "--idle-warn-minutes", "0", "--rl-sample-wait-s", "7200"]
    assert out.spec == {"prompts": {"dispense": {"max_redo": 6}}}
    assert out.run_config["budget"] == {"silence_min": 45}
    assert out.pool_elastic == {"crash_loop_before_ready": "halt"}
    # the written-back knobs alone (no section) resolve to the same values
    legacy_cfg = {k: v for k, v in out.run_config.items() if k != "breakers"}
    again = resolve(Sources(run_config=legacy_cfg, argv=out.argv, spec=out.spec, pool_elastic=out.pool_elastic))
    for path in ("trainer.sample_wait.timeout_s", "launch.idle.enabled", "sampler.dispense_redo.max_n",
                 "run.silence.limit_s", "pool.crash_loop.action"):
        assert again.get(path) == rb.get(path), path
    # an existing flag is replaced in place
    sources = Sources(argv=argv, section={"trainer": {"sample_wait": {"timeout_s": 60}}})
    assert materialize(resolve(sources), sources).argv == argv


def test_write_back_switch_and_inherit(reg):
    sources = Sources(argv=[], section={"switches": {"cost_guard": False}})
    assert materialize(resolve(sources), sources).argv == ["--no-cost-guard"]
    sources = Sources(argv=[], section={"run": {"budget": {"usd": 450}}})
    assert materialize(resolve(sources), sources).argv == ["--budget-usd", "450"]


def test_write_back_unapplied(reg):
    # only a read-only native knob here: refused (strict) or listed
    sources = Sources(env={}, section={"trainer": {"nan": {"enabled": True, "max_consecutive_n": 5}}})
    with pytest.raises(BreakerConfigError, match="trainer.nan.max_consecutive_n = 5"):
        materialize(resolve(sources), sources)
    assert materialize(resolve(sources), sources, strict=False).unapplied == ["trainer.nan.max_consecutive_n = 5"]
    # the knob lives in another process: skipped; an entry without aliases: read from the resolved file
    sources = Sources(section={"sampler": {"dispense_redo": {"max_n": 6}}, "monitor": {"heartbeat": {"enabled": False}}})
    out = materialize(resolve(sources), sources)
    assert out.unapplied == [] and out.changes == []


def test_load_resolved_round_trip(reg, tmp_path):
    rb = resolve(Sources(section={"run": {"budget": {"usd": 450}}}))
    path = tmp_path / "breakers.resolved.json"
    path.write_text(json.dumps(rb.to_json()))
    back = breakers.load_resolved(path)
    assert back.get("launch.budget.usd") == 450 and back.has_section
    assert back.entry("launch.budget").sources["usd"].kind == "inherit"
    with pytest.raises(ValueError, match="schema"):
        from_json({"schema": "other"})


# --- events ------------------------------------------------------------------------------------------


@pytest.fixture
def no_reason():
    events.clear_stop_reason()
    yield
    events.clear_stop_reason()


def test_trip_event_and_stop_reason(reg, no_reason):
    seen = []
    ev = breakers.trip("trainer.sample_wait", 3601.2, 3600, emit=seen.append, scope={"island": "0"},
                       detail="rollout 37", legacy_event="SampleBufferRolloutError", now=1795400000.0)
    assert seen == [ev] and events.validate_event(ev) == []
    assert ev["layer"] == "trainer" and ev["kind"] == "breaker" and ev["unit"] == "s" and ev["action"] == "stop"
    assert ev["stop_reason"] == "breaker:trainer.sample_wait"
    assert breakers.current_stop_reason() == "breaker:trainer.sample_wait"
    breakers.trip("launch.idle", 1, 1)  # a later stop does not replace the first
    assert breakers.current_stop_reason() == "breaker:trainer.sample_wait"


def test_trip_without_stop(reg, no_reason):
    ev = breakers.trip("sampler.drain", 3, None, count=3)
    assert "stop_reason" not in ev and ev["count"] == 3 and breakers.current_stop_reason() is None
    ev = breakers.trip("trainer.sample_wait", 10, 5, state="warn")
    assert "stop_reason" not in ev
    ev = breakers.trip("trainer.sample_wait", 10, 5, stop_reason=False)
    assert "stop_reason" not in ev and breakers.current_stop_reason() is None


def test_trip_never_raises_from_emit(reg, no_reason, capsys):
    def boom(_):
        raise OSError("tape gone")
    ev = breakers.trip("unregistered.thing", 1, 0, "warn", emit=boom)
    assert ev["kind"] is None and ev["layer"] == "unregistered" and events.validate_event(ev) == []
    assert "tape gone" in capsys.readouterr().err


def test_status_event(reg):
    items = [breakers.status_item("trainer.sample_wait", 1800, 3600, source="default"),
             breakers.status_item("sampler.dispense_redo", 3, None, detail="worst slot")]
    ev = breakers.status_event("trainer", items, scope={"island": "0"}, now=1.0)
    assert events.validate_event(ev) == []
    assert ev["items"][0] == {"name": "trainer.sample_wait", "value": 1800, "limit": 3600, "unit": "s",
                              "gauge": "elapsed", "ratio": 0.5, "source": "default"}
    assert ev["items"][1]["ratio"] is None
    assert events.ratio(5, 10, "min") == 2.0 and events.ratio(1, 0) is None


def test_stop_reason_env():
    assert events.stop_reason_from_env({"YETO_STOP_REASON": "breaker:x"}) == "breaker:x"
    assert events.stop_reason_from_env({}) is None


# --- CLI ---------------------------------------------------------------------------------------------


def _parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    bcli.add_parser(sub)
    return p


def _run(argv):
    return bcli.main(_parser().parse_args(bcli.normalize_argv(argv)))


def test_normalize_keeps_the_old_spelling():
    assert bcli.normalize_argv(["breaker", "--trainer-log", "x"]) == ["breaker", "check", "--trainer-log", "x"]
    assert bcli.normalize_argv(["breaker", "show"]) == ["breaker", "show"]
    assert bcli.normalize_argv(["breaker", "--help"]) == ["breaker", "--help"]
    assert bcli.normalize_argv(["launch", "--x"]) == ["launch", "--x"]


SHOW_SNAPSHOT = """\
NAME                     LAYER    KIND          ON   ACTION   EFFECTIVE                   SOURCE
run.budget               run      breaker       on   stop     usd=450 warn_ratio=0.8      breakers.run.budget.usd
run.wall_clock           run      breaker       on   stop     hours=3                     default (implicit)                    !
run.silence              run      breaker       on   -        limit_s=1800                default
launch.budget            launch   breaker       off  stop     usd=450 (inherit run.budget.usd)  switch cost_guard (trainer.argv --no-cost-guard)
launch.idle              launch   breaker       off  stop     warn_s=900                  switch cost_guard (trainer.argv --no-cost-guard)
trainer.sample_wait      trainer  breaker       on   stop     timeout_s=7200              trainer.argv --rl-sample-wait-s (legacy)
trainer.nan              trainer  breaker       off  stop     max_consecutive_n=2         default
sampler.dispense_redo    sampler  retry_budget  on   -        max_n=2                     default
sampler.drain            sampler  guard         on   drop     reload_max_defer_s=600      default
sampler.token_alignment  sampler  invariant     on   drop     -                           (cannot disable)
pool.sampler_restarts    pool     retry_budget  on   -        max_n=2                     default
pool.crash_loop          pool     breaker       on   replace  -                           default
pool.start_timeout       pool     breaker       on   -        limit_s=3600                default
pool.engine_stall        pool     breaker       on   -        limit_s=600                 default
monitor.heartbeat        monitor  alert         on   warn     limit_s=300                 default
SWITCHES
  cost_guard           off  launch.budget, launch.idle  (switch cost_guard (trainer.argv --no-cost-guard))"""


def test_show_snapshot(reg, tmp_path, capsys):
    cfg = _cfg(trainer={"argv": ["--rl-sample-wait-s", "7200", "--no-cost-guard"]},
               breakers={"run": {"budget": {"usd": 450}}})
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    assert _run(["breaker", "show", "--config", str(path)]) == 0
    out = capsys.readouterr().out.splitlines()
    cells = [line.split() for line in out]
    want = [line.split() for line in SHOW_SNAPSHOT.splitlines()]
    assert cells == want
    header = out[0]
    assert header.index("LAYER") == len("sampler.token_alignment") + 2  # columns aligned


def test_show_json_and_exit_codes(reg, tmp_path, capsys):
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"trainer": {"sample_wait": {"timeout_s": 7200}}}))
    assert _run(["breaker", "show", "--breakers", str(path), "--json", "--layer", "trainer"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["schema"] == "yeto-breakers-resolved-v1" and {e["layer"] for e in data["entries"]} == {"trainer"}
    assert data["entries"][0]["sources"]["timeout_s"]["location"] == "breakers.trainer.sample_wait.timeout_s"
    # a conflict: exit 1
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps(_cfg(trainer={"argv": ["--rl-sample-wait-s", "60"]},
                                   breakers={"trainer": {"sample_wait": {"timeout_s": 7200}}})))
    assert _run(["breaker", "show", "--config", str(cfg)]) == 1
    assert "CONFLICTS" in capsys.readouterr().out
    # bad input: exit 2
    path.write_text(json.dumps({"trainer": {"sample_wait": {"bogus": 1}}}))
    assert _run(["breaker", "show", "--breakers", str(path)]) == 2
    assert "bogus" in capsys.readouterr().err
    assert _run(["breaker", "show", "--config", str(tmp_path / "missing.json")]) == 2


def test_show_uses_rules_findings(reg, tmp_path, capsys, monkeypatch):
    import types

    rules = types.ModuleType("yeto.breakers.rules")
    rules.RULE_IDS = ("D1",)
    rules.check = lambda rb, run_config=None, sources=None: [
        breakers.Finding("D1", "error", "drain cut never effective", ("sampler.drain",))]
    monkeypatch.setitem(sys.modules, "yeto.breakers.rules", rules)
    path = tmp_path / "b.json"
    path.write_text("{}")
    assert _run(["breaker", "show", "--breakers", str(path)]) == 1
    out = capsys.readouterr().out
    assert "DEPENDENCIES" in out and "D1   error  drain cut never effective" in out
    assert any(line.startswith("sampler.drain") and line.endswith("!D1") for line in out.splitlines())


def test_list_markdown_and_json(reg, capsys):
    assert _run(["breaker", "list", "--markdown"]) == 0
    md = capsys.readouterr().out
    assert "### trainer" in md and "| `trainer.sample_wait` | breaker | timeout_s=3600 | stop | yes |" in md
    assert "| `cost_guard` | launch.budget, launch.idle | trainer.argv --no-cost-guard |" in md
    assert _run(["breaker", "list", "--json", "--layer", "sampler"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [b["name"] for b in data["breakers"]] == ["sampler.dispense_redo", "sampler.drain",
                                                     "sampler.token_alignment", "sampler.max_prompt_len"]


# --- import weight -----------------------------------------------------------------------------------


def test_import_is_light():
    code = (
        "import sys, yeto.breakers as b\n"
        "from yeto.breakers import cli\n"
        "b.resolve(b.Sources(run_config={'run': 'r', 'budget': {'usd': 1}}), strict=False)\n"
        "heavy = {'torch', 'ray', 'miles', 'modal', 'boto3', 'sglang', 'vllm', 'numpy', 'transformers'}\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


# --- mapping params and deprecations -----------------------------------------------------------------


def _caps(raw):
    text = str(raw).strip()
    return {"*": int(text)} if text.isdigit() else json.loads(text)


@pytest.fixture
def caps_reg():
    return registry_mod.Registry([Breaker(
        "harness.sandbox_caps", "harness", "guard",
        params=(Param("max_sandboxes", dict, None, nullable=True),),
        aliases=(Alias("env", "HARBOR_MAX_SANDBOXES", "max_sandboxes", to_canonical=_caps,
                       from_canonical=lambda v: json.dumps(v, sort_keys=True), deprecated=True),))])


def test_mapping_param(caps_reg):
    rb = resolve(Sources(env={"HARBOR_MAX_SANDBOXES": "16"}), registry=caps_reg)
    caps = rb.get("harness.sandbox_caps.max_sandboxes")
    assert caps == {"*": 16}
    with pytest.raises(TypeError):
        caps["*"] = 1  # read-only
    assert rb.to_json()["entries"][0]["params"]["max_sandboxes"] == {"*": 16}
    assert resolve(Sources(env={}), registry=caps_reg).get("harness.sandbox_caps.max_sandboxes") is None
    # section value agrees with the env JSON object: redundant, not a conflict
    section = {"harness": {"sandbox_caps": {"max_sandboxes": {"remote-docker": 8, "*": 32}}}}
    rb = resolve(Sources(env={"HARBOR_MAX_SANDBOXES": '{"*": 32, "remote-docker": 8}'}, section=section),
                 registry=caps_reg)
    assert rb.entry("harness.sandbox_caps").redundant == ("env HARBOR_MAX_SANDBOXES",)
    with pytest.raises(BreakerConfigError, match="must be an object"):
        parse_section({"harness": {"sandbox_caps": {"max_sandboxes": 16}}}, caps_reg)
    sources = Sources(env={}, section=section)
    out = materialize(resolve(sources, registry=caps_reg), sources, registry=caps_reg)
    assert out.env == {"HARBOR_MAX_SANDBOXES": '{"*": 32, "remote-docker": 8}'}


def test_deprecations(caps_reg):
    rb = resolve(Sources(env={"HARBOR_MAX_SANDBOXES": "16"}), registry=caps_reg)
    assert rb.deprecations() == ["env HARBOR_MAX_SANDBOXES is deprecated (removed two minor versions after it "
                                 "was deprecated): write breakers.harness.sandbox_caps.max_sandboxes"]
    assert rb.to_json()["deprecations"] == rb.deprecations()
    assert resolve(Sources(env={}), registry=caps_reg).deprecations() == []


def test_switch_section_on_against_legacy_off(reg):
    with pytest.raises(BreakerConfigError, match="switch cost_guard: breakers.switches.cost_guard = true"):
        resolve(Sources(argv=["--no-cost-guard"], section={"switches": {"cost_guard": True}}))
    assert resolve(Sources(argv=[], section={"switches": {"cost_guard": True}})).switches() == {"cost_guard": True}


def test_check_rules_only(reg, tmp_path, capsys, monkeypatch):
    import types

    rules = types.ModuleType("yeto.breakers.rules")
    rules.RULE_IDS = ("D8",)
    rules.check = lambda rb, run_config=None, sources=None: [breakers.Finding("D8", "warn", "margin 2.5%")]
    monkeypatch.setitem(sys.modules, "yeto.breakers.rules", rules)
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps(_cfg()))
    assert _run(["breaker", "check", "--config", str(cfg)]) == 0
    assert capsys.readouterr().out.strip() == "D8   warn   margin 2.5%"
    rules.check = lambda rb, run_config=None, sources=None: [breakers.Finding("D8", "error", "short")]
    assert _run(["breaker", "check", "--config", str(cfg)]) == 1
    assert _run(["breaker", "check"]) == 2  # neither tapes nor a config
    assert "--trainer-log" in capsys.readouterr().err


def test_off_when_null():
    b = Breaker("run.starve", "run", "alert", off_when_null=("alert_s", "stop_s"),
                params=(Param("alert_s", float, 1800.0, nullable=True), Param("stop_s", float, 7200.0, nullable=True)),
                aliases=(Alias("run_config", "starvation.alert_min", "alert_s", to_canonical=lambda m: m * 60 or None),
                         Alias("run_config", "starvation.stop_min", "stop_s", to_canonical=lambda m: m * 60 or None)))
    r = registry_mod.Registry([b])
    rb = resolve(Sources(run_config=_cfg(starvation={"alert_min": 0, "stop_min": 0})), registry=r)
    assert not rb.enabled("run.starve")
    assert rb.entry("run.starve").sources["enabled"].location == "run config starvation.stop_min"
    rb = resolve(Sources(run_config=_cfg(starvation={"alert_min": 0})), registry=r)
    assert rb.enabled("run.starve") and rb.get("run.starve.alert_s") is None and rb.get("run.starve.stop_s") == 7200
    with pytest.raises(RegistryError, match="off_when_null"):
        registry_mod.Registry([Breaker("run.x", "run", "alert", off_when_null=("t",), params=(Param("t", float, 1.0),))])


# --- trainer island env and shadowed aliases ---------------------------------------------------------


@pytest.fixture
def island_reg():
    return registry_mod.Registry([
        Breaker("trainer.sample_wait", "trainer", "breaker", params=(Param("timeout_s", float, 3600.0, lo=0),),
                aliases=(Alias("trainer_flag", "--rl-sample-wait-s", "timeout_s"),
                         Alias("trainer_env", "YETO_SAMPLE_BUFFER_WAIT_S", "timeout_s",
                               shadowed_by=("--rl-sample-wait-s",)))),
        Breaker("trainer.policy_age", "trainer", "guard", params=(Param("max_versions", int, 0, lo=0),),
                aliases=(Alias("trainer_flag", "--rl-max-policy-age", "max_versions"),
                         Alias("trainer_env", "YETO_SAMPLE_BUFFER_MAX_POLICY_AGE", "max_versions",
                               shadowed_by=("trainer_flag",)))),
        Breaker("trainer.damaged", "trainer", "breaker", params=(Param("max_run_n", int, 8, lo=0),),
                aliases=(Alias("trainer_env", "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN", "max_run_n", off_values=("0",)),)),
    ])


def test_trainer_env_reads_argv_pairs(island_reg):
    argv = ["--island-env", "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN=4", "--modal-env=YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN=6"]
    rb = resolve(Sources(argv=argv, env={"YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN": "2"}), registry=island_reg)
    assert rb.get("trainer.damaged.max_run_n") == 6  # --modal-env after --island-env, both over the env
    assert rb.entry("trainer.damaged").sources["max_run_n"].location == \
        "trainer.argv --modal-env YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN"
    rb = resolve(Sources(env={"YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN": "0"}), registry=island_reg)
    assert not rb.enabled("trainer.damaged")
    assert rb.entry("trainer.damaged").sources["enabled"].location == "env YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN"


def test_shadowed_alias(island_reg):
    # the launcher derives the env from the flag: the flag wins, no conflict
    rb = resolve(Sources(argv=["--rl-sample-wait-s", "5400"], env={"YETO_SAMPLE_BUFFER_WAIT_S": "7200"}),
                 registry=island_reg)
    assert rb.get("trainer.sample_wait.timeout_s") == 5400 and rb.conflicts == ()
    # in the trainer process only the env exists
    rb = resolve(Sources(env={"YETO_SAMPLE_BUFFER_WAIT_S": "7200", "YETO_SAMPLE_BUFFER_MAX_POLICY_AGE": "4"}),
                 registry=island_reg)
    assert rb.get("trainer.sample_wait.timeout_s") == 7200 and rb.get("trainer.policy_age.max_versions") == 4
    # any trainer argv shadows the policy-age env (the launcher always writes it)
    rb = resolve(Sources(argv=[], env={"YETO_SAMPLE_BUFFER_MAX_POLICY_AGE": "4"}), registry=island_reg)
    assert rb.get("trainer.policy_age.max_versions") == 0
    with pytest.raises(RegistryError, match="shadowed_by"):
        registry_mod.Registry([Breaker("trainer.x", "trainer", "breaker", params=(Param("n", int, 1),),
                                       aliases=(Alias("env", "X", "n", shadowed_by=("--nope",)),))])


def test_trainer_env_write_back(island_reg):
    section = {"trainer": {"damaged": {"max_run_n": 3}}}
    sources = Sources(argv=["--training-mode", "rl"], env={}, section=section)
    out = materialize(resolve(sources, registry=island_reg), sources, registry=island_reg)
    assert out.argv == ["--training-mode", "rl", "--island-env", "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN=3"]
    assert out.env == {}
    argv = ["--modal-env", "YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN=3"]
    sources = Sources(argv=argv, section=section)
    assert materialize(resolve(sources, registry=island_reg), sources, registry=island_reg).argv == argv
    sources = Sources(env={}, section=section)
    out = materialize(resolve(sources, registry=island_reg), sources, registry=island_reg)
    assert out.env == {"YETO_SAMPLE_BUFFER_MAX_DAMAGED_RUN": "3"}


def test_list_marks_deprecated_aliases(caps_reg, monkeypatch, capsys):
    monkeypatch.setattr(registry_mod, "_DEFAULT", caps_reg)
    assert _run(["breaker", "list", "--markdown"]) == 0
    assert "env HARBOR_MAX_SANDBOXES (deprecated)" in capsys.readouterr().out
    assert _run(["breaker", "list"]) == 0
    assert "env HARBOR_MAX_SANDBOXES (deprecated)" in capsys.readouterr().out
