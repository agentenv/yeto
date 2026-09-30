"""Opt-in launcher/learner switches for rl-infra-spec 2.3 (--rl-overlap-eval)
and 3.x (--rl-elastic). Off by default: the default launch command, learner
args and miles_args are unchanged; when given they reach entry.py."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from test_rl_engine_selection import _cli, _island_task, _learner_argv
from yeto import launcher
from yeto.launcher import _prepare_rl_args
from yeto.rl import learner
from yeto.rl.engine.miles_adapter import entry
from yeto.rl.engine.miles_adapter.rollout_meta_hook import ELASTIC_METADATA_ENV

NEW_ATTRS = ("rl_overlap_eval", "rl_elastic", "rl_elastic_resources",
             "rl_elastic_attestation", "rl_elastic_initial_config", "rl_elastic_cells",
             "rl_eval_interval", "rl_eval_data", "rl_eval_dataset_name",
             "rl_eval_samples_per_prompt")
ELASTIC_LEARNER = ("--rl-elastic", "--rl-elastic-resources", "/r.json",
                   "--rl-elastic-state-dir", "/state", "--rl-elastic-initial-config", "c0",
                   "--rl-elastic-cells", "cell-a, cell-b")


def test_default_launch_command_is_unchanged(monkeypatch):
    args = _cli()
    assert not args.rl_overlap_eval and not args.rl_elastic
    assert all(getattr(args, name) in (False, None) for name in NEW_ATTRS)
    _prepare_rl_args(args)
    run = _island_task(args, monkeypatch).run
    assert "--rl-overlap-eval" not in run and "--rl-elastic" not in run
    # Same command as a namespace that never had the new options.
    legacy = _cli()
    for name in NEW_ATTRS:
        delattr(legacy, name)
    _prepare_rl_args(legacy)
    assert _island_task(legacy, monkeypatch).run == run


def test_default_learner_leaves_miles_args_and_environment_untouched():
    args = learner.parse_args(_learner_argv())
    assert not args.rl_overlap_eval and not args.rl_elastic
    miles_args, environ = SimpleNamespace(), {}
    learner.apply_ports_infra_switches(args, miles_args, environ)
    assert vars(miles_args) == {} and environ == {}
    assert entry.elastic_wiring_for(miles_args, profile=None, fingerprint="f") is None


def test_learner_overlap_eval_sets_miles_args():
    args = learner.parse_args(_learner_argv(("--rl-overlap-eval",)))
    miles_args, environ = SimpleNamespace(), {}
    learner.apply_ports_infra_switches(args, miles_args, environ)
    assert vars(miles_args) == {"yeto_rl_overlap_eval": True} and environ == {}


def test_learner_elastic_sets_wiring_metadata_and_env():
    args = learner.parse_args(_learner_argv(ELASTIC_LEARNER))
    miles_args, environ = SimpleNamespace(), {}
    learner.apply_ports_infra_switches(args, miles_args, environ)
    assert miles_args.yeto_rl_elastic == {
        "resources": "/r.json", "attestation": None, "state_dir": "/state",
        "initial_config": "c0", "declared_cells": ("cell-a", "cell-b"),
    }
    assert miles_args.yeto_rl_elastic_metadata is True
    assert environ == {ELASTIC_METADATA_ENV: "1"}


@pytest.mark.parametrize("extra, message", [
    (("--rl-elastic-cells", "a"), "need --rl-elastic"),
    (("--rl-elastic",), "--rl-elastic needs"),
    (ELASTIC_LEARNER + ("--rl-engine", "legacy"), "only applies to --rl-engine ports"),
    (("--rl-overlap-eval", "--rl-engine", "legacy"), "only applies to --rl-engine ports"),
])
def test_learner_refuses_incomplete_or_legacy_switches(extra, message, capsys):
    with pytest.raises(SystemExit):
        learner.parse_args(_learner_argv(extra))
    assert message in capsys.readouterr().err


def test_entry_builds_elastic_wiring_from_miles_args(monkeypatch):
    from yeto.rl.engine.miles_adapter import elastic_wiring

    seen = {}
    monkeypatch.setattr(elastic_wiring, "build_elastic", lambda **kw: seen.update(kw) or "W")
    miles_args = SimpleNamespace(yeto_rl_elastic={
        "resources": "/r.json", "attestation": "/a.json", "state_dir": "/s",
        "initial_config": "c0", "declared_cells": ("x",)})
    assert entry.elastic_wiring_for(miles_args, profile="P", fingerprint="F") == "W"
    assert seen == {"state_dir": "/s", "resources": "/r.json", "attestation": "/a.json",
                    "profile": "P", "initial_config": "c0", "runtime_fingerprint": "F",
                    "declared_cells": ("x",)}


def test_run_ports_island_wires_elastic_before_ray_and_into_compose():
    source = inspect.getsource(entry.run_ports_island)
    built = source.index("elastic = elastic_wiring_for(")
    assert source.index("preflight(") < built < source.index("connect_island_ray()")
    assert "elastic=elastic," in source[source.index("compose_island("):]
    ports = inspect.getsource(learner._run_ports)
    assert ports.index("apply_ports_infra_switches(") < ports.index("run_ports_island(")


def test_launcher_forwards_switches(monkeypatch, tmp_path):
    resources = tmp_path / "res.json"
    resources.write_text(json.dumps({"configs": {"c0": {}}, "edges": []}))
    attestation = tmp_path / "att.json"
    attestation.write_text("{}")
    args = _cli(("--rl-placement", "fixed-partition",
                 "--rl-overlap-eval", "--rl-elastic", "--rl-elastic-resources", str(resources),
                 *_eval_flags(_heldout(tmp_path)),
                 "--rl-elastic-attestation", str(attestation),
                 "--rl-elastic-initial-config", "c0", "--rl-elastic-cells", "a,b"))
    launcher._check_ports_infra_switches(args, "ports")
    prelude, flags = launcher._ports_infra_flags(args)
    eval_part, _, flags = flags.partition(" --rl-overlap-eval")
    assert eval_part.startswith(" --eval-interval 2 --eval-data ~/yeto-rl/eval-heldout.jsonl")
    flags = " --rl-overlap-eval" + flags
    assert flags == (" --rl-overlap-eval --rl-elastic --rl-elastic-resources "
                     "~/yeto-rl/elastic_resources.json --rl-elastic-state-dir "
                     f"{launcher.ELASTIC_ISLAND_STATE_DIR} --rl-elastic-initial-config c0 "
                     "--rl-elastic-cells a,b --rl-elastic-attestation "
                     "~/yeto-rl/elastic_attestation.json")
    assert "elastic_resources.json" in prelude and "elastic_attestation.json" in prelude
    assert '{"configs":{"c0":{}},"edges":[]}' in prelude
    # The forwarded learner flags parse on the island side.
    island = [t.replace("~/yeto-rl", "/y") for t in flags.split()]
    parsed = learner.parse_args(_learner_argv(tuple(island)))
    assert parsed.rl_overlap_eval and parsed.rl_elastic and parsed.rl_elastic_cells == "a,b"


@pytest.mark.parametrize("extra, message", [
    (("--rl-elastic-cells", "a"), "need --rl-elastic"),
    (("--rl-elastic", "--rl-elastic-cells", "a"), "--rl-elastic needs"),
])
def test_launcher_refuses_incomplete_elastic(extra, message):
    with pytest.raises(ValueError, match=message):
        launcher._check_ports_infra_switches(_cli(extra), "ports")
    with pytest.raises(ValueError, match="only applies to --rl-engine ports"):
        launcher._check_ports_infra_switches(_cli(("--rl-overlap-eval",)), "legacy")


def _elastic_files(tmp_path, attestation="{}"):
    resources = tmp_path / "res.json"
    resources.write_text(json.dumps({"configs": {"c0": {"trainer": 1, "rollout": 1}}, "edges": []}))
    att = tmp_path / "att.json"
    att.write_text(attestation)
    return str(resources), str(att)


def _heldout(tmp_path, text='{"prompt": "q", "label": "a"}\n'):
    path = tmp_path / "heldout.jsonl"
    path.write_text(text)
    return path


def _eval_flags(path, interval="2"):
    return ("--rl-eval-interval", interval, "--rl-eval-data", str(path),
            "--rl-eval-dataset-name", "held", "--rl-eval-samples-per-prompt", "4")


def test_launcher_accepts_overlap_eval_from_real_cli_with_fixed_partition(monkeypatch, tmp_path):
    """The launcher's eval source is --rl-eval-*: the same interval is checked
    locally and forwarded as the learner's --eval-interval."""
    heldout = _heldout(tmp_path)
    args = _cli(("--rl-overlap-eval", "--rl-placement", "fixed-partition",
                 "--rl-rollout-gpus", "1", "--gpu", "aws:2xa100@us-east-1") + _eval_flags(heldout))
    launcher._check_ports_infra_switches(args, "ports")
    _prepare_rl_args(args)
    run = _island_task(args, monkeypatch).run
    assert "--rl-placement fixed-partition" in run and "--rl-overlap-eval" in run
    assert " --eval-interval 2 --eval-data ~/yeto-rl/eval-heldout.jsonl" in run
    import hashlib
    assert f"--eval-data-sha256 {hashlib.sha256(heldout.read_bytes()).hexdigest()}" in run
    assert "--eval-dataset-name held --eval-samples-per-prompt 4" in run


def test_launcher_eval_flags_parse_on_the_learner_and_ship_exact_bytes(tmp_path):
    import subprocess

    heldout = _heldout(tmp_path, '{"prompt": "it\'s 100% $HOME \\n"}\n{"prompt": "b"}\n')
    args = _cli(("--rl-placement", "fixed-partition") + _eval_flags(heldout, "3"))
    launcher._check_ports_infra_switches(args, "ports")
    prelude, flags = launcher._ports_infra_flags(args)
    home = tmp_path / "home"
    home.mkdir()
    subprocess.run(["bash", "-c", prelude], check=True, env={"HOME": str(home), "PATH": "/usr/bin:/bin"})
    assert (home / "yeto-rl" / "eval-heldout.jsonl").read_bytes() == heldout.read_bytes()
    island = [t.replace("~", str(home)) for t in flags.split()]
    parsed = learner.parse_args(_learner_argv(tuple(island)))
    assert parsed.eval_interval == 3 and parsed.eval_samples_per_prompt == 4
    assert parsed.eval_dataset_name == "held"
    parsed.eval_only = False
    parsed.parameter_mode = "lora"
    assert learner._verify_eval_dataset_identity(parsed) == home / "yeto-rl" / "eval-heldout.jsonl"


def test_launcher_refuses_overlap_eval_without_eval_interval():
    from yeto.rl.engine.execution_profile import ProfileError

    with pytest.raises(ProfileError, match="--eval-interval"):
        launcher._check_ports_infra_switches(
            _cli(("--rl-overlap-eval", "--rl-placement", "fixed-partition")), "ports")


@pytest.mark.parametrize("extra, message", [
    (("--rl-eval-dataset-name", "x"), "need --rl-eval-interval"),
    (("--rl-eval-interval", "2"), "--rl-eval-interval needs"),
    (("--rl-eval-interval", "0", "--rl-eval-data", "x", "--rl-eval-dataset-name", "x",
      "--rl-eval-samples-per-prompt", "1"), "positive int"),
])
def test_launcher_refuses_incomplete_eval(extra, message):
    with pytest.raises(ValueError, match=message):
        launcher._check_ports_infra_switches(_cli(extra), "ports")


def test_launcher_refuses_eval_on_legacy_and_same_file_as_data(tmp_path):
    heldout = _heldout(tmp_path)
    with pytest.raises(ValueError, match="only applies to --rl-engine ports"):
        launcher._check_ports_infra_switches(_cli(_eval_flags(heldout)), "legacy")
    args = _cli(_eval_flags(heldout))
    args.data = str(heldout)
    with pytest.raises(ValueError, match="distinct from --data"):
        launcher._check_ports_infra_switches(args, "ports")


def test_ports_lora_training_eval_resolves_and_legacy_lora_still_refused(tmp_path):
    from yeto.rl.engine.run_config import _resolve_eval

    args = SimpleNamespace(eval_interval=2, eval_only=False, rl_engine="ports",
                           eval_dataset_name="held", eval_samples_per_prompt=4)
    config = _resolve_eval(args, parameter_mode="lora", prompt_path="/p",
                           eval_prompt_path="/e", yeto_policy_sync=True)
    assert config.interval == 2 and config.prompt_path == "/e"
    with pytest.raises(ValueError, match="heldout"):
        _resolve_eval(args, parameter_mode="lora", prompt_path="/p",
                      eval_prompt_path=None, yeto_policy_sync=True)
    args.rl_engine = "legacy"
    with pytest.raises(ValueError, match="restricted"):
        _resolve_eval(args, parameter_mode="lora", prompt_path="/p",
                      eval_prompt_path="/e", yeto_policy_sync=True)


def test_launcher_refuses_overlap_eval_on_colocated_from_real_cli():
    from yeto.rl.engine.execution_profile import ProfileError

    with pytest.raises(ProfileError, match="fixed-partition"):
        launcher._check_ports_infra_switches(_cli(("--rl-overlap-eval",)), "ports")


@pytest.mark.parametrize("miles, message", [
    ({"eval_interval": None}, "--eval-interval"),
    ({"eval_interval": 5, "eval_uses_snapshots": True}, "--eval-uses-snapshots"),
])
def test_shared_check_still_refuses_known_eval_values(miles, message):
    """The island passes real miles_args values; they are still checked."""
    from yeto.rl.engine.execution_profile import ProfileError, check_overlap_eval

    with pytest.raises(ProfileError, match=message):
        check_overlap_eval(placement_kind="fixed-partition",
                           eval_uses_snapshots=miles.get("eval_uses_snapshots", False),
                           eval_interval=miles["eval_interval"])


def test_launcher_and_island_share_the_overlap_eval_check():
    source = inspect.getsource(launcher._check_ports_infra_switches)
    assert "check_overlap_eval(" in source and "check_elastic_placement(" in source
    assert "check_overlap_eval(" in inspect.getsource(entry)


def _elastic_cli(res, *extra):
    return _cli(("--rl-elastic", "--rl-elastic-resources", res,
                 "--rl-elastic-cells", "a", *extra))


def test_launcher_refuses_colocated_elastic(tmp_path):
    from yeto.rl.engine.execution_profile import ProfileError

    res, _ = _elastic_files(tmp_path)
    args = _elastic_cli(res, "--rl-elastic-initial-config", "c0")
    with pytest.raises(ProfileError, match="fixed-partition"):
        launcher._check_ports_infra_switches(args, "ports")
    launcher._check_ports_infra_switches(
        _elastic_cli(res, "--rl-elastic-initial-config", "c0",
                     "--rl-placement", "fixed-partition"), "ports")


def test_launcher_refuses_unknown_initial_config_and_bad_attestation(tmp_path):
    from yeto.rl.elastic_benchmark.manifest import ManifestError

    res, att = _elastic_files(tmp_path, attestation=json.dumps({"execution_modes": ["bogus"]}))
    args = _elastic_cli(res, "--rl-elastic-initial-config", "nope", "--rl-placement", "fixed-partition")
    with pytest.raises(ValueError, match="not a manifest config"):
        launcher._check_ports_infra_switches(args, "ports")
    args = _elastic_cli(res, "--rl-elastic-initial-config", "c0", "--rl-placement",
                        "fixed-partition", "--rl-elastic-attestation", att)
    with pytest.raises(ManifestError):
        launcher._check_ports_infra_switches(args, "ports")


def test_launcher_forwards_strict_pause_inputs_to_learner_and_syncer(tmp_path):
    resources, attestation = _elastic_files(tmp_path)
    args = _cli(("--rl-placement", "fixed-partition", "--rl-elastic",
                 "--rl-elastic-resources", resources, "--rl-elastic-initial-config", "c0",
                 "--rl-elastic-cells", "a", "--rl-elastic-quorum-timeout-s", "120",
                 "--rl-elastic-idle-flow-timeout-s", "300", "--rl-elastic-pause-margin", "0.25"))
    launcher._check_ports_infra_switches(args, "ports")
    _, flags = launcher._ports_infra_flags(args)
    assert flags.endswith(" --rl-elastic-quorum-timeout-s 120 --rl-elastic-idle-flow-timeout-s"
                          " 300.0 --rl-elastic-pause-margin 0.25")
    assert launcher._syncer_quorum_timeout(args) == " --quorum-timeout-s 120"
    island = [t.replace("~/yeto-rl", "/y") for t in flags.split()]
    parsed = learner.parse_args(_learner_argv(tuple(island)))
    miles_args = SimpleNamespace()
    learner.apply_ports_infra_switches(parsed, miles_args, {})
    assert {k: miles_args.yeto_rl_elastic[k] for k in
            ("quorum_timeout_s", "idle_flow_timeout_s", "pause_margin")} == {
        "quorum_timeout_s": 120.0, "idle_flow_timeout_s": 300.0, "pause_margin": 0.25}


def test_default_syncer_command_has_no_quorum_timeout():
    args = _cli()
    assert launcher._syncer_quorum_timeout(args) == ""
    with pytest.raises(ValueError, match="need --rl-elastic"):
        launcher._check_ports_infra_switches(
            _cli(("--rl-elastic-quorum-timeout-s", "120")), "ports")


def test_launcher_refuses_an_eval_file_over_the_inline_cap(tmp_path):
    row = '{"prompt": "' + "x" * 1000 + '"}\n'
    heldout = _heldout(tmp_path, row * (launcher.EVAL_INLINE_MAX_BYTES // len(row) + 1))
    assert launcher.EVAL_INLINE_MAX_BYTES == 96 * 1024
    with pytest.raises(ValueError, match="larger than"):
        launcher._check_ports_infra_switches(_cli(_eval_flags(heldout)), "ports")
