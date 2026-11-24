"""``breakers.trainer.*`` and ``breakers.launch.*`` of a run config reach the trainer and the launcher.

The trainer reads its flags (``--rl-sample-wait-s``, ``--rl-max-policy-age``) and its
island env, the launcher its own flags (``--rl-stall-timeout``, ``--budget-usd``, ...), never
the run config's ``breakers:`` section. ``load_config`` writes the section's values into
``trainer.argv`` (as ``yeto launch --breakers`` does), so the value the preflight shows, the
compiled argv and the value the launched trainer / launcher resolves are one value. A config
without such an entry keeps its argv byte for byte. ``launch.budget.usd`` inherited from
``run.budget.usd`` alone stays with run up (``enforced_by: run``).
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pytest

from yeto import cli
from yeto.breakers import cli as bcli
from yeto.cost.guard import guard_config_from_args, guard_enabled
from yeto.launch.fleet import effective_stall_timeout
from yeto.launch.modal_islands import parse_island_env
from yeto.rl import up
from yeto.rl.adapters.miles import buffer_rollout
from yeto.rl.engine import sample_roles, trainer_breakers

ROOT = Path(__file__).resolve().parents[1]
STORE = "s3://b/runs/r"
ARGV = ["--training-mode", "rl", "--gpu", "modal:8xh100", "--cluster-prefix", "r", "--model", "Qwen/Qwen3-1.7B",
        "--data", "HuggingFaceH4/MATH-500", "--rl-sample-role", "trainer", "--rl-checkpoint-store", STORE,
        "--rl-max-policy-age", "2", "--rollout-batch-size", "8", "--n-samples-per-prompt", "8"]


def _pool(tmp_path: Path) -> str:
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"reload_max_defer_s": 2400}))
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"spec": str(spec), "elastic": {"start_timeout_s": 5400}}))
    return str(path)


def _write(tmp_path: Path, *, argv=None, **sections) -> Path:
    cfg = {"run": "r", "store": STORE, "pool": {"config": _pool(tmp_path)},
           "trainer": {"argv": list(ARGV if argv is None else argv)}, **sections}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return path


def _show(path: Path, capsys) -> dict:
    """``yeto breaker show --config --json`` of the run config as the user wrote it."""
    capsys.readouterr()  # load_config's log lines
    p = argparse.ArgumentParser()
    bcli.add_parser(p.add_subparsers(dest="command"))
    bcli.main(p.parse_args(bcli.normalize_argv(["breaker", "show", "--json", "--config", str(path)])))
    return {e["name"]: e for e in json.loads(capsys.readouterr().out)["entries"]}


def _resolved_file(cfg: dict, tmp_path: Path) -> dict:
    """run up's preflight record (``breakers.resolved.json``) of the compiled config."""
    run = up.Run(cfg, state_root=tmp_path / "state")
    run.write_breakers_resolved()
    doc = json.loads((run.dir / "breakers.resolved.json").read_text())
    return {e["name"]: e for e in doc["entries"]}


def _island(argv: list[str]) -> trainer_breakers.breakers.ResolvedBreakers:
    """The trainer entries as the launched trainer resolves them: the launch parser, the
    buffer env the island entry exports (sample wait, policy age), then the island env."""
    args = cli.build_parser().parse_args(["launch", *argv])
    args.rl_resume_store = args.rl_checkpoint_store  # the island's own store flag (rl_ports)
    config = sample_roles.resolve(args, groups_per_round=8)
    env = buffer_rollout.island_env("trainer", store_root=STORE, max_policy_age=int(args.rl_max_policy_age or 0),
                                    wait_s=config.wait_s)
    env.update(parse_island_env(args))
    return trainer_breakers.resolved(env)


@pytest.mark.parametrize("section, param, value", [
    ({"sample_wait": {"timeout_s": 7200}}, "trainer.sample_wait.timeout_s", 7200),
    ({"sample_wait": {"timeout_s": 900.5}}, "trainer.sample_wait.timeout_s", 900.5),
    ({"policy_age": {"max_versions": 5}}, "trainer.policy_age.max_versions", 5),
    ({"release_drain": {"timeout_s": 600}}, "trainer.release_drain.timeout_s", 600),
    ({"buffer_damaged": {"max_run_n": 3}}, "trainer.buffer_damaged.max_run_n", 3),
    ({"nan": {"skip": True}}, "trainer.nan.skip", True),
])
def test_the_preflight_the_argv_and_the_trainer_agree(tmp_path, capsys, section, param, value):
    argv = [a for a in ARGV]
    if "policy_age" in section:  # the argv must not set it itself (that is a conflict, below)
        i = argv.index("--rl-max-policy-age")
        del argv[i:i + 2]
    path = _write(tmp_path, argv=argv, breakers={"trainer": section})
    cfg = up.load_config(str(path))
    compiled = cfg["trainer"]["argv"]
    assert compiled[:len(argv)] == argv and len(compiled) == len(argv) + 2  # one flag (or env pair) added
    name, key = param.rsplit(".", 1)
    shown = _show(path, capsys)[name]
    recorded = _resolved_file(cfg, tmp_path)[name]
    island = _island(compiled)
    assert shown["params"][key] == value and shown["sources"][key]["kind"] == "breakers"
    assert recorded["params"][key] == value and recorded["sources"][key]["kind"] == "breakers"
    assert island.get(param) == value


def test_the_sample_wait_reaches_the_buffer_reader(tmp_path):
    cfg = up.load_config(str(_write(tmp_path, breakers={"trainer": {"sample_wait": {"timeout_s": 7200}}})))
    argv = cfg["trainer"]["argv"]
    assert argv[-2:] == ["--rl-sample-wait-s", "7200"]
    args = cli.build_parser().parse_args(["launch", *argv])
    assert args.rl_sample_wait_s == 7200.0
    assert "--rl-sample-wait-s 7200.0" in sample_roles.island_flags(args)
    assert trainer_breakers.reader_limits(_island(argv)).wait_s == 7200.0


def test_a_disabled_policy_age_is_written_as_its_off_value(tmp_path):
    argv = [a for a in ARGV]
    i = argv.index("--rl-max-policy-age")
    del argv[i:i + 2]
    cfg = up.load_config(str(_write(tmp_path, argv=argv, breakers={"trainer": {"policy_age": {"enabled": False}}})))
    assert cfg["trainer"]["argv"][-2:] == ["--rl-max-policy-age", "0"]


@pytest.mark.parametrize("sections", [
    {},
    {"breakers": {"run": {"starvation": {"stop_s": 900}}}},
    # run.budget alone: launch.budget inherits it in the view, run up keeps it (enforced_by run)
    {"breakers": {"run": {"budget": {"usd": 450}}}},
    {"breakers": {"trainer": {}, "launch": {}, "switches": {}}},
    # the argv already holds the section's values: nothing to add
    {"breakers": {"trainer": {"policy_age": {"max_versions": 2}}}},
    {"breakers": {"launch": {"stall": {"timeout_s": 900}}, "switches": {"cost_guard": True}},
     "argv": [*ARGV, "--rl-stall-timeout", "900"]},
])
def test_without_a_new_value_the_argv_is_unchanged(tmp_path, sections):
    path = _write(tmp_path, **sections)
    raw = json.loads(path.read_text())
    assert up.load_config(str(path))["trainer"]["argv"] == raw["trainer"]["argv"]


@pytest.mark.parametrize("run", sorted((ROOT / "examples" / "k31").glob("run*.json")), ids=lambda p: p.name)
def test_the_k31_examples_keep_their_argv(run):
    """The customer path: the K3.1 example run configs set no breakers section."""
    raw = json.loads(run.read_text())
    assert "breakers" not in raw
    assert up.apply_argv_breakers(raw) is raw


@pytest.mark.parametrize("extra, section, match", [
    (["--rl-sample-wait-s", "900"], {"sample_wait": {"timeout_s": 1200}},
     r"breakers.trainer.sample_wait.timeout_s = 1200 but trainer.argv --rl-sample-wait-s = 900"),
    (["--rl-sample-wait-s=900"], {"sample_wait": {"timeout_s": 1200}}, r"timeout_s = 1200 but .*= 900"),
    ([], {"policy_age": {"max_versions": 5}},
     r"breakers.trainer.policy_age.max_versions = 5 but trainer.argv --rl-max-policy-age = 2"),
    (["--island-env", "YETO_TRAINER_RELEASE_DRAIN_S=300"], {"release_drain": {"timeout_s": 600}},
     r"release_drain.timeout_s = 600 but trainer.argv --island-env YETO_TRAINER_RELEASE_DRAIN_S = 300"),
    # a value no trainer knob carries (the xhw thresholds are only in YETO_RL_XHW_GUARD)
    ([], {"xhw_guard": {"warn_kl": 0.1}}, r"trainer.xhw_guard.warn_kl = 0.1: .*cannot be applied"),
])
def test_a_conflicting_or_unapplicable_value_is_refused(tmp_path, extra, section, match):
    path = _write(tmp_path, argv=[*ARGV, *extra], breakers={"trainer": section})
    with pytest.raises(ValueError, match=match):
        up.load_config(str(path))


def test_the_run_env_conflicts_with_the_section_too(tmp_path):
    path = _write(tmp_path, env={"YETO_SAMPLE_BUFFER_WAIT_S": "900"},
                  breakers={"trainer": {"sample_wait": {"timeout_s": 1200}}})
    with pytest.raises(ValueError, match=r"timeout_s = 1200 but env YETO_SAMPLE_BUFFER_WAIT_S = 900"):
        up.load_config(str(path))


# ------------------------------------------------------------------ the launch layer
def _launcher(argv: list[str]) -> dict:
    """What the launcher applies: its stall clock and its cost guard."""
    args = cli.build_parser().parse_args(["launch", *argv])
    guard = guard_config_from_args(args)
    return {"stall_s": effective_stall_timeout(args), "guard": guard_enabled(args), "budget_usd": guard.budget_usd,
            "warn_usd": guard.budget_warn_usd, "idle_warn_s": guard.idle.idle_after_s,
            "idle_stop_s": guard.idle.stop_after_s}


@pytest.mark.parametrize("section, param, value, flag, launcher", [
    ({"launch": {"stall": {"timeout_s": 1800}}}, "launch.stall.timeout_s", 1800,
     ["--rl-stall-timeout", "1800"], {"stall_s": 1800.0}),
    ({"launch": {"idle": {"warn_s": 600, "stop_after_warn_s": 1200}}}, "launch.idle.warn_s", 600,
     ["--idle-warn-minutes", "10", "--idle-stop-minutes", "20"], {"idle_warn_s": 600.0, "idle_stop_s": 1200.0}),
    ({"launch": {"budget": {"usd": 50, "warn_usd": 40}}}, "launch.budget.usd", 50,
     ["--budget-usd", "50", "--budget-warn-usd", "40"], {"budget_usd": 50.0, "warn_usd": 40.0}),
    # written "inherit": the launch takes run.budget.usd as its own limit
    ({"run": {"budget": {"usd": 450}}, "launch": {"budget": {"usd": "inherit"}}}, "launch.budget.usd", 450,
     ["--budget-usd", "450"], {"budget_usd": 450.0}),
])
def test_the_preflight_the_argv_and_the_launcher_agree(tmp_path, capsys, section, param, value, flag, launcher):
    path = _write(tmp_path, breakers=section)
    cfg = up.load_config(str(path))
    assert cfg["trainer"]["argv"] == [*ARGV, *flag]
    name, key = param.rsplit(".", 1)
    shown = _show(path, capsys)[name]
    recorded = _resolved_file(cfg, tmp_path)[name]
    assert shown["params"][key] == value and recorded["params"][key] == value
    assert "enforced_by" not in shown and "enforced_by" not in recorded
    applied = _launcher(cfg["trainer"]["argv"])
    assert {k: applied[k] for k in launcher} == launcher and applied["guard"]


def test_run_up_and_the_launcher_judge_one_stall_value(tmp_path):
    """run up's silence backstop and the launcher's RunStalled clock read the same
    launch.stall: neither runs on a value the other does not see."""
    cfg = up.load_config(str(_write(tmp_path, breakers={"launch": {"stall": {"timeout_s": 1800}}})))
    run = up.Run(cfg, state_root=tmp_path / "state")
    assert run.stall_timeout_s() == _launcher(cfg["trainer"]["argv"])["stall_s"] == 1800.0


def test_the_cost_guard_switch_is_written_as_no_cost_guard(tmp_path):
    cfg = up.load_config(str(_write(tmp_path, breakers={"switches": {"cost_guard": False}})))
    assert cfg["trainer"]["argv"] == [*ARGV, "--no-cost-guard"]
    assert not _launcher(cfg["trainer"]["argv"])["guard"]


def test_a_launch_budget_inherited_alone_is_enforced_by_run_up(tmp_path, capsys):
    """run.budget.usd alone: the view shows launch.budget.usd inherited and marks it
    enforced_by run; the launch is not given it (the launcher prices the shared resource
    ledger, pool leases included, its own way: the same limit there could stop first)."""
    path = _write(tmp_path, breakers={"run": {"budget": {"usd": 450}}, "launch": {"stall": {"timeout_s": 1800}}})
    cfg = up.load_config(str(path))
    assert cfg["trainer"]["argv"] == [*ARGV, "--rl-stall-timeout", "1800"]
    for entry in (_show(path, capsys)["launch.budget"], _resolved_file(cfg, tmp_path)["launch.budget"]):
        assert entry["params"]["usd"] == 450 and entry["sources"]["usd"]["kind"] == "inherit"
        assert entry["enforced_by"] == "run"
    assert _launcher(cfg["trainer"]["argv"])["budget_usd"] is None
    bcli.main(_show_args(path))  # the table marks the row
    assert "!enforced_by_run" in next(line for line in capsys.readouterr().out.splitlines()
                                      if line.startswith("launch.budget "))


def _show_args(path: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    bcli.add_parser(p.add_subparsers(dest="command"))
    return p.parse_args(bcli.normalize_argv(["breaker", "show", "--config", str(path)]))


def test_a_launch_budget_setting_without_its_usd_is_refused(tmp_path):
    path = _write(tmp_path, breakers={"run": {"budget": {"usd": 450}}, "launch": {"budget": {"warn_usd": 40}}})
    with pytest.raises(ValueError, match=r"breakers.launch.budget warn_usd needs breakers.launch.budget.usd"):
        up.load_config(str(path))


@pytest.mark.parametrize("extra, section, match", [
    (["--rl-stall-timeout", "900"], {"stall": {"timeout_s": 1800}},
     r"breakers.launch.stall.timeout_s = 1800 but trainer.argv --rl-stall-timeout = 900"),
    (["--budget-usd", "20"], {"budget": {"usd": 50}}, r"breakers.launch.budget.usd = 50 but trainer.argv --budget-usd = 20"),
])
def test_a_conflicting_launch_value_is_refused(tmp_path, extra, section, match):
    with pytest.raises(ValueError, match=match):
        up.load_config(str(_write(tmp_path, argv=[*ARGV, *extra], breakers={"launch": section})))


# ------------------------------------------------------------------ never block a stop
CONFLICT = {"trainer": {"sample_wait": {"timeout_s": 1200}}, "launch": {"stall": {"timeout_s": 1800}}}


def _conflicting(tmp_path: Path) -> Path:
    """A config an earlier yeto accepted: its section disagrees with its argv."""
    return _write(tmp_path, argv=[*ARGV, "--rl-sample-wait-s", "900", "--rl-stall-timeout", "600"], breakers=CONFLICT)


def test_run_down_and_status_open_a_conflicting_config_run_up_refuses_it(tmp_path, monkeypatch, capsys):
    """A breakers value the argv disagrees with refuses `run up` / `scale`, never `run down`:
    a refused down would leave the samplers billing. down / status warn and read the argv as
    written."""
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    path = str(_conflicting(tmp_path))
    downs = []
    monkeypatch.setattr(up.Run, "down", lambda self: downs.append(list(self.cfg["trainer"]["argv"])) or True)
    assert up.main(types.SimpleNamespace(config=path, rl_command="down")) == 0
    assert downs == [[*ARGV, "--rl-sample-wait-s", "900", "--rl-stall-timeout", "600"]]  # as written
    err = capsys.readouterr().out
    assert "WARNING" in err and "timeout_s = 1200 but trainer.argv --rl-sample-wait-s = 900" in err
    (tmp_path / "rl-up" / "r").mkdir(parents=True, exist_ok=True)  # status reads a run brought up here
    monkeypatch.setattr(up.Run, "status", lambda self: {"run": self.run})
    assert up.main(types.SimpleNamespace(config=path, rl_command="status")) == 0
    assert "WARNING" in capsys.readouterr().out
    for command in ("up", "scale"):
        rc = up.main(types.SimpleNamespace(config=path, rl_command=command, skip_preflight=False, dry_run=True,
                                           no_monitor=True, allow_after_forced_stop=False))
        assert rc == 2
        assert "timeout_s = 1200 but trainer.argv --rl-sample-wait-s = 900" in capsys.readouterr().err


def test_load_config_warns_without_strict_breakers(tmp_path):
    path = str(_conflicting(tmp_path))
    with pytest.raises(ValueError, match="but trainer.argv --rl-stall-timeout = 600"):
        up.load_config(path)
    cfg = up.load_config(path, strict_breakers=False)
    assert cfg["trainer"]["argv"] == json.loads(Path(path).read_text())["trainer"]["argv"]
    # a launch budget setting without its usd: a warning too
    bad = _write(tmp_path, breakers={"run": {"budget": {"usd": 450}}, "launch": {"budget": {"warn_usd": 40}}})
    assert up.load_config(str(bad), strict_breakers=False)["trainer"]["argv"] == ARGV


def test_rollout_status_opens_a_conflicting_config(tmp_path, monkeypatch):
    from yeto.launch.rollout import entry

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    refs, _env = entry._status_refs(types.SimpleNamespace(pool_config=None, config=str(_conflicting(tmp_path))))
    assert [r.name for r in refs] == ["pool"]


@pytest.mark.parametrize("how, strict", [
    ({"target": 3, "add": 0, "remove": [], "retire": []}, True),
    ({"target": None, "add": 2, "remove": [], "retire": []}, True),
    ({"target": None, "add": 0, "remove": ["t1"], "retire": []}, False),
    ({"target": 1, "add": 0, "remove": [], "retire": ["t1"]}, False),
    ({"target": 0, "add": 0, "remove": [], "retire": []}, False),
])
def test_a_scale_that_only_releases_nodes_is_never_blocked(tmp_path, monkeypatch, capsys, how, strict):
    """Releasing nodes saves money: like run down it warns on a conflicting breakers value."""
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    path = str(_conflicting(tmp_path))
    scaled = []
    monkeypatch.setattr(up.Run, "scale", lambda self, *a, **k: scaled.append(a) or 0)
    args = types.SimpleNamespace(config=path, rl_command="scale", mode=None, force_static=False, dry_run=False,
                                 source="manual", **how)
    assert up.breakers_strict(args) is strict
    assert up.main(args) == (2 if strict else 0)
    assert bool(scaled) is not strict
    out = capsys.readouterr()
    assert ("timeout_s = 1200 but trainer.argv --rl-sample-wait-s = 900" in out.err) is strict
    assert ("WARNING" in out.out) is not strict


def test_rollout_scale_down_opens_a_conflicting_config(tmp_path, monkeypatch):
    from yeto.launch.rollout import entry

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(up.Run, "scale", lambda self, *a, **k: 0)
    base = dict(config=str(_conflicting(tmp_path)), pool_config=None, provider=None, pool=None, add=0, mode=None,
                force_static=False, dry_run=False, source="manual")
    assert entry.scale(types.SimpleNamespace(**base, target=None, remove=["t1"], retire=[])) == 0
    assert entry.scale(types.SimpleNamespace(**base, target=3, remove=[], retire=[])) == 2
