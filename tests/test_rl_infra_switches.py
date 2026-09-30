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
             "rl_elastic_attestation", "rl_elastic_initial_config", "rl_elastic_cells")
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
    args = _cli(("--rl-overlap-eval", "--rl-elastic", "--rl-elastic-resources", str(resources),
                 "--rl-elastic-attestation", str(attestation),
                 "--rl-elastic-initial-config", "c0", "--rl-elastic-cells", "a,b"))
    args.rl_placement, args.eval_interval = "fixed-partition", 5
    launcher._check_ports_infra_switches(args, "ports")
    prelude, flags = launcher._ports_infra_flags(args)
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


def _partitioned(args, **extra):
    args.rl_placement = "fixed-partition"
    for k, v in extra.items():
        setattr(args, k, v)
    return args


@pytest.mark.parametrize("attrs, message", [
    ({"rl_placement": "colocated", "eval_interval": 5}, "fixed-partition"),
    ({"rl_placement": "fixed-partition", "eval_interval": None}, "--eval-interval"),
    ({"rl_placement": "fixed-partition", "eval_interval": 5, "eval_uses_snapshots": True},
     "--eval-uses-snapshots"),
])
def test_launcher_refuses_overlap_eval_preconditions_locally(attrs, message):
    """Integ-s2 finding 3: same check as the island profile, before provisioning."""
    from yeto.rl.engine.execution_profile import ProfileError

    args = _cli(("--rl-overlap-eval",))
    for k, v in attrs.items():
        setattr(args, k, v)
    with pytest.raises(ProfileError, match=message):
        launcher._check_ports_infra_switches(args, "ports")


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
    launcher._check_ports_infra_switches(_partitioned(args), "ports")


def test_launcher_refuses_unknown_initial_config_and_bad_attestation(tmp_path):
    from yeto.rl.elastic_benchmark.manifest import ManifestError

    res, att = _elastic_files(tmp_path, attestation=json.dumps({"execution_modes": ["bogus"]}))
    args = _partitioned(_elastic_cli(res, "--rl-elastic-initial-config", "nope"))
    with pytest.raises(ValueError, match="not a manifest config"):
        launcher._check_ports_infra_switches(args, "ports")
    args = _partitioned(_elastic_cli(res, "--rl-elastic-initial-config", "c0",
                                     "--rl-elastic-attestation", att))
    with pytest.raises(ManifestError):
        launcher._check_ports_infra_switches(args, "ports")
