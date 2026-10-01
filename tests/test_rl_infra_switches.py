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
    miles_args = SimpleNamespace(use_miles_router=True, yeto_rl_elastic={
        "resources": "/r.json", "attestation": "/a.json", "state_dir": "/s",
        "initial_config": "c0", "declared_cells": ("x",)})
    assert entry.elastic_wiring_for(miles_args, profile="P", fingerprint="F") == "W"
    assert seen == {"state_dir": "/s", "resources": "/r.json", "attestation": "/a.json",
                    "profile": "P", "initial_config": "c0", "runtime_fingerprint": "F",
                    "declared_cells": ("x",)}


def test_run_ports_island_wires_elastic_before_ray_and_into_compose():
    stage = inspect.getsource(entry.preflight_stage)  # IR-1: pre-Ray checks live here
    built = stage.index("elastic = elastic_wiring_for(")
    assert stage.index("preflight(") < built < stage.index("hook(miles_args, launch)")
    source = inspect.getsource(entry.run_ports_island)
    assert source.index("preflight_stage(") < source.index("connect_island_ray()")
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


# ---------------------------------------------------------------- test-only start delay (A5)
def test_start_delay_injection_is_off_by_default_and_exported_when_given(tmp_path, monkeypatch):
    from yeto.rl.engine.miles_adapter.rollout import INJECT_START_DELAY_ENV

    args = _cli()
    assert args.rl_test_inject_start_delay_s is None
    _prepare_rl_args(args)
    assert INJECT_START_DELAY_ENV not in _island_task(args, monkeypatch).run
    resources, _ = _elastic_files(tmp_path)
    args = _cli(("--rl-placement", "fixed-partition", "--rl-elastic",
                 "--rl-elastic-resources", resources, "--rl-elastic-initial-config", "c0",
                 "--rl-elastic-cells", "a", "--rl-test-inject-start-delay-s", "150"))
    launcher._check_ports_infra_switches(args, "ports")
    prelude, _ = launcher._ports_infra_flags(args)
    assert f"export {INJECT_START_DELAY_ENV}=150.0\n" in prelude
    with pytest.raises(ValueError, match="need --rl-elastic"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-inject-start-delay-s", "150")), "ports")
    with pytest.raises(ValueError, match="positive"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-inject-start-delay-s", "0")), "ports")


def test_pool_sleeps_once_before_the_first_start_cells(monkeypatch):
    import asyncio

    from yeto.rl.engine.miles_adapter.rollout import (
        INJECT_START_DELAY_ENV,
        MilesRolloutPool,
        injected_start_delay,
    )

    calls = []

    class Controller:
        async def start_cells(self, cells, expected_epoch):
            calls.append(("start", tuple(cells)))

        async def wait_cells_tracked(self, cells, timeout_seconds):
            calls.append(("tracked", tuple(cells)))

    def pool():
        return MilesRolloutPool(inference_controller=Controller(), rollout_executor=None,
                                metadata=None, expected_policy=lambda: (0, "h"),
                                runner=SimpleNamespace(run=asyncio.run),
                                declared_cells=("c0", "c1", "c2"))

    monkeypatch.delenv(INJECT_START_DELAY_ENV, raising=False)
    plain = pool()
    plain._sleep = lambda s: calls.append(("sleep", s))
    plain.add_engines(1, epoch=0, members=frozenset({"engine:c1"}))
    assert calls == [("start", ("c1",)), ("tracked", ("c1",))]  # default: no sleep
    calls.clear()
    monkeypatch.setenv(INJECT_START_DELAY_ENV, "150")
    injected = pool()
    injected._sleep = lambda s: calls.append(("sleep", s))
    injected.add_engines(1, epoch=0, members=frozenset({"engine:c1"}))
    injected.add_engines(1, epoch=1, members=frozenset({"engine:c2"}))
    assert calls == [("sleep", 150.0), ("start", ("c1",)), ("tracked", ("c1",)),
                     ("start", ("c2",)), ("tracked", ("c2",))]  # once, before the first start
    assert injected.injected_start_delays == [150.0]
    with pytest.raises(ValueError, match="positive"):
        injected_start_delay({INJECT_START_DELAY_ENV: "-1"})


# ---------------------------------------------------------------- test-only update_weights block (§4)
def _publisher(monkeypatch, value):
    from yeto.rl.engine.miles_adapter.publish import INJECT_UPDATE_BLOCK_ENV, MilesPublisher

    if value is None:
        monkeypatch.delenv(INJECT_UPDATE_BLOCK_ENV, raising=False)
    else:
        monkeypatch.setenv(INJECT_UPDATE_BLOCK_ENV, value)
    order = []

    class Controller:
        async def wait_cells_tracked(self, cells, timeout_seconds):
            order.append("tracked")

    async def update_weights(*a, **k):
        order.append("update_weights")
        raise RuntimeError("stop here")

    pub = MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                         inference_controller=Controller(), update_weights=update_weights)
    pub.block_poll_s = 0.01
    return pub, order


def test_update_weights_block_is_off_by_default(monkeypatch):
    import asyncio

    from yeto.rl.engine.miles_adapter.publish import PublicationError

    pub, order = _publisher(monkeypatch, None)
    with pytest.raises(PublicationError, match="stop here"):
        asyncio.run(pub._publish_members("t", ["c2"], 1))
    assert order == ["tracked", "update_weights"] and pub.injected_blocks == []


def test_update_weights_block_fails_when_the_target_dies_and_applies_once(monkeypatch):
    import asyncio

    from yeto.rl.engine.miles_adapter.publish import PublicationError

    pub, order = _publisher(monkeypatch, "30")
    polls = []

    class Probe:
        async def snapshot(self, cells):
            polls.append(tuple(cells))

        async def status(self):
            polls.append("s")
            return "alive" if len(polls) < 3 else "dead: actor died (RayActorError)"

    pub.liveness_probe = Probe()
    with pytest.raises(PublicationError, match="dead: actor died"):
        asyncio.run(pub._publish_members("t", ["c2"], 1))
    assert order == ["tracked"] and pub.injected_blocks == [30.0] and len(polls) == 3
    # second member publication of the process: no block
    with pytest.raises(PublicationError, match="stop here"):
        asyncio.run(pub._publish_members("t", ["c3"], 2))
    assert order == ["tracked", "tracked", "update_weights"]


def test_update_weights_block_times_out_into_the_real_call(monkeypatch):
    import asyncio

    from yeto.rl.engine.miles_adapter.publish import PublicationError

    pub, order = _publisher(monkeypatch, "0.05")

    class Alive:
        async def snapshot(self, cells):
            pass

        async def status(self):
            return "alive"

    pub.liveness_probe = Alive()
    with pytest.raises(PublicationError, match="stop here"):
        asyncio.run(pub._publish_members("t", ["c2"], 1))
    assert order == ["tracked", "update_weights"]


def test_launcher_exports_the_update_weights_block_only_when_given(tmp_path, monkeypatch):
    from yeto.rl.engine.miles_adapter.publish import INJECT_UPDATE_BLOCK_ENV

    args = _cli()
    _prepare_rl_args(args)
    assert INJECT_UPDATE_BLOCK_ENV not in _island_task(args, monkeypatch).run
    resources, _ = _elastic_files(tmp_path)
    args = _cli(("--rl-placement", "fixed-partition", "--rl-elastic",
                 "--rl-elastic-resources", resources, "--rl-elastic-initial-config", "c0",
                 "--rl-elastic-cells", "a", "--rl-test-inject-update-weights-block-s", "600"))
    launcher._check_ports_infra_switches(args, "ports")
    prelude, _ = launcher._ports_infra_flags(args)
    assert f"export {INJECT_UPDATE_BLOCK_ENV}=600.0\n" in prelude
    with pytest.raises(ValueError, match="need --rl-elastic"):
        launcher._check_ports_infra_switches(
            _cli(("--rl-test-inject-update-weights-block-s", "600")), "ports")



class _Ref:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *a, **k):
        async def call():
            return self.fn(*a, **k)
        return call()


class _RayMod:
    class exceptions:
        class RayActorError(Exception):
            pass


def _liveness(infos_seq, ready=lambda: None):
    from yeto.rl.engine.miles_adapter.publish import RayTargetLiveness

    seq = list(infos_seq)
    info = lambda n, g: SimpleNamespace(name=n, generation=g)  # noqa: E731
    handle = SimpleNamespace(__ray_ready__=_Ref(lambda: ready()))
    manager = SimpleNamespace(
        get_worker_infos=_Ref(lambda cell: [info(*x) for x in (seq.pop(0) if len(seq) > 1 else seq[0])]),
        get_actor_handle=_Ref(lambda name, expected_generation: handle))
    return RayTargetLiveness(manager=manager, ray_module=_RayMod)


@pytest.mark.parametrize("later, expect", [
    ([], "no workers"),                     # the cell was stopped
    ([("w0", 4)], "generation changed"),    # the health monitor restarted it
    ([("w0", 3)], "alive"),
])
def test_liveness_probe_judges_the_recorded_target_generation(later, expect):
    import asyncio

    probe = _liveness([[("w0", 3)], later])
    asyncio.run(probe.snapshot(["c2"]))
    assert expect in asyncio.run(probe.status())


def test_liveness_probe_reports_killed_and_unknown_errors_differently():
    import asyncio

    from yeto.rl.engine.miles_adapter.publish import InjectedBlockProbeError

    def killed():
        raise _RayMod.exceptions.RayActorError("dead")

    probe = _liveness([[("w0", 3)]], ready=killed)
    asyncio.run(probe.snapshot(["c2"]))
    assert asyncio.run(probe.status()).startswith("dead: actor died")

    def weird():
        raise ConnectionError("gcs unreachable")

    probe = _liveness([[("w0", 3)]], ready=weird)
    asyncio.run(probe.snapshot(["c2"]))
    with pytest.raises(InjectedBlockProbeError, match="gcs unreachable"):
        asyncio.run(probe.status())
