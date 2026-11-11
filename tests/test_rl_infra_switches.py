"""Opt-in launcher/learner switches (--rl-overlap-eval)
and 3.x (--rl-elastic). Off by default: the default launch command, learner
args and miles_args are unchanged; when given they reach entry.py."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from test_rl_engine_selection import _cli, _island_task, _learner_argv
from yeto import launcher
from yeto.launcher import _prepare_rl_args
from yeto.rl.adapters.miles import island_entry as learner
from yeto.rl.adapters.miles import entry
from yeto.rl.adapters.miles.rollout_meta_hook import ELASTIC_METADATA_ENV

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
])
def test_learner_refuses_incomplete_switches(extra, message, capsys):
    with pytest.raises(SystemExit):
        learner.parse_args(_learner_argv(extra))
    assert message in capsys.readouterr().err


def test_entry_builds_elastic_wiring_from_miles_args(monkeypatch):
    from yeto.rl.adapters.miles import elastic_wiring

    seen = {}
    monkeypatch.setattr(elastic_wiring, "build_elastic", lambda **kw: seen.update(kw) or "W")
    miles_args = SimpleNamespace(router="miles", yeto_rl_elastic={
        "resources": "/r.json", "attestation": "/a.json", "state_dir": "/s",
        "initial_config": "c0", "declared_cells": ("x",)})
    assert entry.elastic_wiring_for(miles_args, profile="P", fingerprint="F") == "W"
    assert seen == {"state_dir": "/s", "resources": "/r.json", "attestation": "/a.json",
                    "profile": "P", "initial_config": "c0", "runtime_fingerprint": "F",
                    "declared_cells": ("x",)}


class _ComposeReached(Exception):
    pass


def drive_run_ports_island(monkeypatch, *, topology=None, elastic="ELASTIC"):
    """Run :func:`entry.run_ports_island` over stubbed Miles until it composes the
    island; returns (ordered call names, compose_island kwargs)."""
    import sys
    import types

    from yeto.rl.adapters.miles import eval_wiring
    from yeto.rl.adapters.miles import rollout as miles_rollout
    from yeto.rl.adapters.miles import state as miles_state

    calls, composed = [], {}

    def record(name, result=None):
        return lambda *a, **kw: calls.append(name) or result

    profile = SimpleNamespace(execution_mode="partitioned-serial")
    monkeypatch.setattr(miles_state, "require_run_plugin", record("require_run_plugin"))
    monkeypatch.setattr(entry, "publish_member_cell", lambda *a, **kw: None)
    monkeypatch.setattr(entry, "preflight_stage",
                        record("preflight_stage", ("FP", "CAPS", profile, elastic)))
    monkeypatch.setattr(eval_wiring, "eval_guard_preflight", lambda *a, **kw: None)
    monkeypatch.setattr(entry, "append_tape_event", lambda *a, **kw: None)
    monkeypatch.setattr(entry, "selection_event", lambda **kw: {})
    for name in ("connect_island_ray", "refuse_partial_island_preflight",
                 "reconcile_gpu_pool_preflight", "pin_placement_group_to_head",
                 "prewarm_remote_code", "wait_island_nodes"):
        monkeypatch.setattr(entry, name, record(name))
    monkeypatch.setattr(entry, "trainer_actor_import_done_unix", lambda: 1.0)
    monkeypatch.setattr(entry, "build_sync", lambda *a, **kw: ("SYNC", None))
    monkeypatch.setattr(miles_rollout, "RayMetadataSink", lambda: "SINK")  # needs a live Ray

    def compose(**kw):
        calls.append("compose_island")
        composed.update(kw)
        composed["_upstream"] = seen
        raise _ComposeReached

    monkeypatch.setattr(entry, "compose_island", compose)

    class _Actor:
        async def run_plugin(self, *a, **kw):
            return None

    seen = {}

    async def create_rollout_components(args, *, capability, trainer_handles):
        seen["rollout"] = (args, capability, trainer_handles)
        return "CONTROLLER", "EXECUTOR", None

    async def create_training_models(args, executor, *, handles):
        seen["training"] = (args, handles)
        return _Actor(), None

    class _StartupInfo:
        def __init__(self, args):
            self.args, self.all_args = args, args

        @classmethod
        def create(cls, all_args):
            return cls(all_args)

    def init_orchestration_script(startup_info, *, disposer):
        seen["startup_info"] = startup_info
        return "CAPABILITY"

    def create_trainer_handles(args, *, trainer_configs, capability):
        seen["handles"] = (args, trainer_configs, capability)
        return {"actor": "HANDLE"}

    class Disposer:
        def add(self, *items):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    modules = {
        "miles.ray.placement_group": dict(create_rollout_components=create_rollout_components,
                                          create_trainer_handles=create_trainer_handles,
                                          create_training_models=create_training_models),
        "miles.ray.rollout.eval_dispatch": dict(
            EvalDispatcher=lambda *a: SimpleNamespace(drain=lambda: None)),
        "miles.ray.specs.train": dict(compute_trainer_configs=lambda args: ["actor-config"]),
        "miles.utils.async_utils": dict(Disposer=Disposer),
        "miles.utils.orchestration_utils": dict(init_orchestration_script=init_orchestration_script,
                                                ArgvOrchestratorStartupInfo=_StartupInfo),
    }
    for name in ("miles", "miles.ray", "miles.ray.rollout", "miles.ray.specs", "miles.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    algorithm = SimpleNamespace(execution=SimpleNamespace(needs_critic=False))
    launch = SimpleNamespace(placement=SimpleNamespace(kind="fixed-partition", topology=topology))
    with pytest.raises(_ComposeReached):
        entry.run_ports_island(SimpleNamespace(), launch, algorithm, learner_id=0,
                               base_model_revision="r", lora_config_hash="h", layout_hash="l",
                               yeto_policy_sync=False)
    return calls, composed


def test_preflight_stage_builds_elastic_after_the_contract_preflight_and_before_the_hook(monkeypatch):
    """IR-1: the elastic wiring is built (and can fail) after the contract preflight
    and before the harness preflight hook, all inside the pre-Ray stage."""
    calls = []

    def record(name, result=None):
        return lambda *a, **kw: calls.append(name) or result

    monkeypatch.setattr(entry, "publish_member_cell", lambda *a, **kw: None)
    monkeypatch.setattr(entry, "ports_runtime_fingerprint", lambda launch: "FP")
    monkeypatch.setattr(entry, "miles_capabilities", lambda *a, **kw: "CAPS")
    monkeypatch.setattr(entry, "with_partitioned_serial", lambda caps: caps)
    monkeypatch.setattr(entry, "execution_profile_for", record("execution_profile_for", "PROFILE"))
    monkeypatch.setattr(entry, "expected_algorithm_sha256", lambda args: None)
    monkeypatch.setattr(entry, "preflight", record("preflight"))
    monkeypatch.setattr(entry, "elastic_wiring_for", record("elastic_wiring_for", "W"))
    monkeypatch.setattr(entry, "check_harness_reward_scope", lambda scope: None)
    out = entry.preflight_stage(SimpleNamespace(), SimpleNamespace(), "ALG", yeto_policy_sync=True,
                                harness_preflight=record("harness_preflight"))
    assert out == ("FP", "CAPS", "PROFILE", "W")
    assert calls == ["execution_profile_for", "preflight", "elastic_wiring_for", "harness_preflight"]


def test_run_ports_island_wires_elastic_before_ray_and_into_compose(monkeypatch):
    calls, composed = drive_run_ports_island(monkeypatch)
    assert calls.index("preflight_stage") < calls.index("connect_island_ray")
    assert composed["elastic"] == "ELASTIC"  # the preflight stage's wiring, not rebuilt
    assert calls[-1] == "compose_island"


def test_run_ports_island_assembles_upstream_components_in_main_order(monkeypatch):
    """Upstream train.py order: startup_info -> init_orchestration_script(startup_info)
    -> create_trainer_handles(orch, capability) -> create_rollout_components(orch,
    capability=, trainer_handles=) -> create_training_models(orch, executor, handles=);
    the publisher gets the orchestrator slice."""
    _, composed = drive_run_ports_island(monkeypatch)
    seen = composed["_upstream"]
    orch = seen["startup_info"].args
    assert seen["handles"] == (orch, ["actor-config"], "CAPABILITY")
    assert seen["rollout"] == (orch, "CAPABILITY", {"actor": "HANDLE"})
    assert seen["training"] == (orch, {"actor": "HANDLE"})
    assert composed["orchestrator_args"] is orch
    assert composed["rollout_executor"] == "EXECUTOR"


def test_learner_applies_infra_switches_before_the_island_runs(monkeypatch):
    calls = []
    monkeypatch.setattr(learner, "apply_ports_infra_switches",
                        lambda args, miles_args: calls.append("switches") or setattr(miles_args, "switched", True))
    monkeypatch.setattr(learner, "_run_test_debug_rollout_replay", lambda args, miles_args: False)
    monkeypatch.setattr(entry, "run_ports_island",
                        lambda miles_args, *a, **kw: calls.append(("island", miles_args.switched)))
    args = SimpleNamespace(event_tape="/t", learner_id=0, model_revision="r")
    miles_args = SimpleNamespace(yeto_rl_parameter_mode="full", yeto_rl_lora_config_hash="h",
                                 yeto_rl_layout_hash="l")
    learner._run_ports(args, miles_args, SimpleNamespace(), SimpleNamespace(), specs=(),
                       canonical_targets=(), yeto_policy_sync=False)
    assert calls == ["switches", ("island", True)]


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
    launcher._check_ports_infra_switches(args)
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
        launcher._check_ports_infra_switches(_cli(extra))


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
    launcher._check_ports_infra_switches(args)
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
    launcher._check_ports_infra_switches(args)
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
            _cli(("--rl-overlap-eval", "--rl-placement", "fixed-partition")))


@pytest.mark.parametrize("extra, message", [
    (("--rl-eval-dataset-name", "x"), "need --rl-eval-interval"),
    (("--rl-eval-interval", "2"), "--rl-eval-interval needs"),
    (("--rl-eval-interval", "0", "--rl-eval-data", "x", "--rl-eval-dataset-name", "x",
      "--rl-eval-samples-per-prompt", "1"), "positive int"),
])
def test_launcher_refuses_incomplete_eval(extra, message):
    with pytest.raises(ValueError, match=message):
        launcher._check_ports_infra_switches(_cli(extra))


def test_launcher_refuses_eval_on_the_training_data_file(tmp_path):
    heldout = _heldout(tmp_path)
    args = _cli(_eval_flags(heldout))
    args.data = str(heldout)
    with pytest.raises(ValueError, match="distinct from --data"):
        launcher._check_ports_infra_switches(args)


def test_ports_lora_training_eval_resolves(tmp_path):
    from yeto.rl.engine.run_config import _resolve_eval

    args = SimpleNamespace(eval_interval=2, eval_only=False, rl_engine="ports",
                           eval_dataset_name="held", eval_samples_per_prompt=4)
    config = _resolve_eval(args, prompt_path="/p",
                           eval_prompt_path="/e", yeto_policy_sync=True)
    assert config.interval == 2 and config.prompt_path == "/e"
    with pytest.raises(ValueError, match="heldout"):
        _resolve_eval(args, prompt_path="/p",
                      eval_prompt_path=None, yeto_policy_sync=True)


def test_training_eval_on_a_single_island_without_outer_sync(tmp_path):
    # The no-sync island owns its policy versions, so its driver can eval;
    # an island of a synced run without the external boundary still cannot.
    from yeto.rl.engine.run_config import _resolve_eval

    base = dict(eval_interval=1000, eval_only=False, rl_engine="ports",
                eval_dataset_name="held", eval_samples_per_prompt=8)
    config = _resolve_eval(SimpleNamespace(**base, rl_single_island_no_sync=True), prompt_path="/p",
                           eval_prompt_path="/e", yeto_policy_sync=False)
    assert config.interval == 1000 and config.prompt_path == "/e" and config.samples_per_prompt == 8
    with pytest.raises(ValueError, match="external policy boundary"):
        _resolve_eval(SimpleNamespace(**base, rl_single_island_no_sync=False), prompt_path="/p",
                      eval_prompt_path="/e", yeto_policy_sync=False)


def test_launcher_refuses_overlap_eval_on_colocated_from_real_cli():
    from yeto.rl.engine.execution_profile import ProfileError

    with pytest.raises(ProfileError, match="fixed-partition"):
        launcher._check_ports_infra_switches(_cli(("--rl-overlap-eval",)))


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


def test_launcher_and_island_share_the_overlap_eval_check(monkeypatch, tmp_path):
    """Both sides run the one shared check (spied), and an overlap-eval request on a
    colocated placement fails with the same error on the launcher and the island."""
    from yeto.rl.engine import execution_profile as ep
    from yeto.rl.engine.algorithm import AlgorithmSpec

    calls = []

    def spy(name):
        real = getattr(ep, name)
        return lambda *a, **kw: calls.append((name, a, kw)) or real(*a, **kw)

    for name in ("check_overlap_eval", "check_elastic_placement"):
        monkeypatch.setattr(ep, name, spy(name))
    with pytest.raises(ep.ProfileError) as on_launcher:
        launcher._check_ports_infra_switches(_cli(("--rl-overlap-eval",)))
    assert [c[0] for c in calls] == ["check_overlap_eval"]
    assert calls[0][2]["placement_kind"] == "colocated"
    calls.clear()
    miles_args = SimpleNamespace(yeto_rl_overlap_eval=True, eval_interval=5)
    launch = SimpleNamespace(placement=SimpleNamespace(kind="colocated"))
    with pytest.raises(ep.ProfileError) as on_island:
        entry.execution_profile_for(miles_args, launch, AlgorithmSpec(), yeto_policy_sync=True)
    assert [c[0] for c in calls] == ["check_overlap_eval"]
    assert calls[0][2] == {"placement_kind": "colocated", "eval_uses_snapshots": False, "eval_interval": 5}
    assert str(on_island.value) == str(on_launcher.value)
    calls.clear()
    res, _ = _elastic_files(tmp_path)
    with pytest.raises(ep.ProfileError, match="fixed-partition"):
        launcher._check_ports_infra_switches(_elastic_cli(res, "--rl-elastic-initial-config", "c0"))
    assert ("check_elastic_placement", ("colocated",), {}) in calls


def _elastic_cli(res, *extra):
    return _cli(("--rl-elastic", "--rl-elastic-resources", res,
                 "--rl-elastic-cells", "a", *extra))


def test_launcher_refuses_colocated_elastic(tmp_path):
    from yeto.rl.engine.execution_profile import ProfileError

    res, _ = _elastic_files(tmp_path)
    args = _elastic_cli(res, "--rl-elastic-initial-config", "c0")
    with pytest.raises(ProfileError, match="fixed-partition"):
        launcher._check_ports_infra_switches(args)
    launcher._check_ports_infra_switches(
        _elastic_cli(res, "--rl-elastic-initial-config", "c0",
                     "--rl-placement", "fixed-partition"))


def test_launcher_refuses_unknown_initial_config_and_bad_attestation(tmp_path):
    from yeto.rl.elastic_benchmark.manifest import ManifestError

    res, att = _elastic_files(tmp_path, attestation=json.dumps({"execution_modes": ["bogus"]}))
    args = _elastic_cli(res, "--rl-elastic-initial-config", "nope", "--rl-placement", "fixed-partition")
    with pytest.raises(ValueError, match="not a manifest config"):
        launcher._check_ports_infra_switches(args)
    args = _elastic_cli(res, "--rl-elastic-initial-config", "c0", "--rl-placement",
                        "fixed-partition", "--rl-elastic-attestation", att)
    with pytest.raises(ManifestError):
        launcher._check_ports_infra_switches(args)


def test_launcher_forwards_strict_pause_inputs_to_learner_and_syncer(tmp_path):
    resources, attestation = _elastic_files(tmp_path)
    args = _cli(("--rl-placement", "fixed-partition", "--rl-elastic",
                 "--rl-elastic-resources", resources, "--rl-elastic-initial-config", "c0",
                 "--rl-elastic-cells", "a", "--rl-elastic-quorum-timeout-s", "120",
                 "--rl-elastic-idle-flow-timeout-s", "300", "--rl-elastic-pause-margin", "0.25"))
    launcher._check_ports_infra_switches(args)
    _, flags = launcher._ports_infra_flags(args)
    assert flags.endswith(" --rl-elastic-quorum-timeout-s 120.0 --rl-elastic-idle-flow-timeout-s"
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
            _cli(("--rl-elastic-quorum-timeout-s", "120")))


def test_launcher_refuses_an_eval_file_over_the_inline_cap(tmp_path):
    row = '{"prompt": "' + "x" * 1000 + '"}\n'
    heldout = _heldout(tmp_path, row * (launcher.EVAL_INLINE_MAX_BYTES // len(row) + 1))
    assert launcher.EVAL_INLINE_MAX_BYTES == 96 * 1024
    with pytest.raises(ValueError, match="larger than"):
        launcher._check_ports_infra_switches(_cli(_eval_flags(heldout)))


# ---------------------------------------------------------------- test-only start delay (A5)
def test_start_delay_injection_is_off_by_default_and_exported_when_given(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles.rollout import INJECT_START_DELAY_ENV

    args = _cli()
    assert args.rl_test_inject_start_delay_s is None
    _prepare_rl_args(args)
    assert INJECT_START_DELAY_ENV not in _island_task(args, monkeypatch).run
    resources, _ = _elastic_files(tmp_path)
    args = _cli(("--rl-placement", "fixed-partition", "--rl-elastic",
                 "--rl-elastic-resources", resources, "--rl-elastic-initial-config", "c0",
                 "--rl-elastic-cells", "a", "--rl-test-inject-start-delay-s", "150"))
    launcher._check_ports_infra_switches(args)
    prelude, _ = launcher._ports_infra_flags(args)
    assert f"export {INJECT_START_DELAY_ENV}=150.0\n" in prelude
    with pytest.raises(ValueError, match="need --rl-elastic"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-inject-start-delay-s", "150")))
    with pytest.raises(ValueError, match="positive"):
        launcher._check_ports_infra_switches(_cli(("--rl-test-inject-start-delay-s", "0")))


def test_pool_sleeps_once_before_the_first_start_cells(monkeypatch):
    import asyncio

    from yeto.rl.adapters.miles.rollout import (
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
    from yeto.rl.adapters.miles.publish import INJECT_UPDATE_BLOCK_ENV, MilesPublisher

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

    from yeto.rl.adapters.miles.publish import PublicationError

    pub, order = _publisher(monkeypatch, None)
    with pytest.raises(PublicationError, match="stop here"):
        asyncio.run(pub._publish_members(1, "t", ["c2"], 1))
    assert order == ["tracked", "update_weights"] and pub.injected_blocks == []


def test_update_weights_block_fails_when_the_target_dies_and_applies_once(monkeypatch):
    import asyncio

    from yeto.rl.adapters.miles.publish import PublicationError

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
        asyncio.run(pub._publish_members(1, "t", ["c2"], 1))
    assert order == ["tracked"] and pub.injected_blocks == [30.0] and len(polls) == 3
    # second member publication of the process: no block
    with pytest.raises(PublicationError, match="stop here"):
        asyncio.run(pub._publish_members(1, "t", ["c3"], 2))
    assert order == ["tracked", "tracked", "update_weights"]


def test_update_weights_block_times_out_into_the_real_call(monkeypatch):
    import asyncio

    from yeto.rl.adapters.miles.publish import PublicationError

    pub, order = _publisher(monkeypatch, "0.05")

    class Alive:
        async def snapshot(self, cells):
            pass

        async def status(self):
            return "alive"

    pub.liveness_probe = Alive()
    with pytest.raises(PublicationError, match="stop here"):
        asyncio.run(pub._publish_members(1, "t", ["c2"], 1))
    assert order == ["tracked", "update_weights"]


def test_launcher_exports_the_update_weights_block_only_when_given(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles.publish import INJECT_UPDATE_BLOCK_ENV

    args = _cli()
    _prepare_rl_args(args)
    assert INJECT_UPDATE_BLOCK_ENV not in _island_task(args, monkeypatch).run
    resources, _ = _elastic_files(tmp_path)
    args = _cli(("--rl-placement", "fixed-partition", "--rl-elastic",
                 "--rl-elastic-resources", resources, "--rl-elastic-initial-config", "c0",
                 "--rl-elastic-cells", "a", "--rl-test-inject-update-weights-block-s", "600"))
    launcher._check_ports_infra_switches(args)
    prelude, _ = launcher._ports_infra_flags(args)
    assert f"export {INJECT_UPDATE_BLOCK_ENV}=600.0\n" in prelude
    with pytest.raises(ValueError, match="need --rl-elastic"):
        launcher._check_ports_infra_switches(
            _cli(("--rl-test-inject-update-weights-block-s", "600")))



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
    from yeto.rl.adapters.miles.publish import RayTargetLiveness

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

    from yeto.rl.adapters.miles.publish import InjectedBlockProbeError

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
