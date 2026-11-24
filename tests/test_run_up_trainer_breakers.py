"""``breakers.trainer.*`` of a single-trainer run config reaches the trainer.

The trainer reads its flags (``--rl-sample-wait-s``, ``--rl-max-policy-age``) and its
island env, never the run config's ``breakers:`` section. ``load_config`` writes the
section's trainer values into ``trainer.argv`` (as ``yeto launch --breakers`` does), so
the value the preflight shows, the compiled argv and the value the launched trainer
resolves are one value. A config without a trainer entry keeps its argv byte for byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from yeto import cli
from yeto.breakers import cli as bcli
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
    {"breakers": {"run": {"starvation": {"stop_s": 900}}, "launch": {"stall": {"timeout_s": 3600}}}},
    {"breakers": {"trainer": {}}},
    # the argv already holds the section's values: nothing to add
    {"breakers": {"trainer": {"policy_age": {"max_versions": 2}}}},
])
def test_without_a_new_trainer_value_the_argv_is_unchanged(tmp_path, sections):
    path = _write(tmp_path, **sections)
    raw = json.loads(path.read_text())
    assert up.load_config(str(path))["trainer"]["argv"] == raw["trainer"]["argv"]


@pytest.mark.parametrize("run", sorted((ROOT / "examples" / "k31").glob("run*.json")), ids=lambda p: p.name)
def test_the_k31_examples_keep_their_argv(run):
    """The customer path: the K3.1 example run configs set no breakers section."""
    raw = json.loads(run.read_text())
    assert "breakers" not in raw
    assert up.apply_trainer_breakers(raw) is raw


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
