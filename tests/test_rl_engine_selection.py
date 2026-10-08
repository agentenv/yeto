"""rl-engine-ports task 5.1: ``--rl-engine`` selection, recording, rejection
matrix, and the ports composition root over stubbed upstream Miles."""

from __future__ import annotations

import json
import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from yeto import launcher
from yeto.cli import parse_args as parse_cli
from yeto.launcher import _prepare_rl_args, make_miles_island_task
from yeto.rl import learner as rl_learner
from yeto.rl.engine.selection import (
    UnsupportedPortsCombination,
    ports_rejections,
    require_ports_supported,
)

# ---------------------------------------------------------------- matrix


def test_supported_r0_combinations_pass():
    for preset in ("strict-avg", "decoupled"):
        assert ports_rejections(sync_preset=preset, lora_targets="attention") == []


def test_ppo_is_not_routed_to_legacy():
    # rl-algo-critic-family 3.1: the critic is decided by the algorithm spec,
    # the capability declaration and the critic rejections, not by routing.
    assert ports_rejections(extra_argv=("--advantage-estimator", "ppo")) == []
    assert ports_rejections(use_critic=True, advantage_estimator="ppo") == []


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (dict(sync_preset="sao-streaming-full"), "SAO"),
        (dict(extra_argv=("--sao-compaction",)), "SAO"),
        (dict(sync_preset="dense-full", parameter_mode="full"), "dense-full"),
        (dict(tuning="full"), "full-parameter"),
        (dict(model_recipe="deepseek-v4-flash"), "DeepSeek V4"),
        (dict(lora_targets="attention-routed-experts"), "DeepSeek V4"),
        (dict(expert_full_count=4), "DeepSeek V4"),
        (dict(use_critic=True), "critic"),
        (dict(extra_argv=("--use-critic",)), "critic"),
        (dict(extra_argv=("--advantage-estimator", "gspo")), "non-GRPO"),
        (dict(rollout_num_gpus=4), "fixed partition"),
        (dict(extra_argv=("--rollout-num-gpus", "4")), "fixed partition"),
        (dict(model_kind="diffusion"), "causal"),
    ],
)
def test_unsupported_combinations_are_rejected_with_legacy_hint(kwargs, reason):
    with pytest.raises(UnsupportedPortsCombination) as error:
        require_ports_supported(**kwargs)
    assert reason in str(error.value)
    assert "--rl-engine legacy" in str(error.value)


# ---------------------------------------------------------------- CLI + launcher


def _cli(extra=()):
    return parse_cli(
        [
            "--gpu", "aws:1xa100@us-east-1,aws:1xa100@us-west-2",
            "--model", "org/model", "--data", "org/data",
            "--training-mode", "rl", "--total-steps", "3",
            "--rollout-batch-size", "4", "--n-samples-per-prompt", "2",
            "--rollout-max-response-len", "128",
            "--local-rl-rounds-per-sync", "1",
            "--reward-function", "pkg.reward:score", "--trust-remote-code",
            *extra,
        ]
    )


def _island_task(args, monkeypatch):
    from tests.test_rl_launcher import _Resources, _Storage, _StorageMode, _Task

    monkeypatch.setitem(
        sys.modules,
        "sky",
        types.SimpleNamespace(
            Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode
        ),
    )
    from yeto.gpu_spec import parse_gpu_spec

    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    return make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 2, "127.0.0.1:29400")


def test_cli_default_is_ports(monkeypatch):
    args = _cli()
    assert args.rl_engine == "ports"
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    assert "--rl-engine ports" in task.run
    assert launcher._miles_source_setup("ports")[0] in task.setup


def test_explicit_legacy_is_forwarded_to_learner_and_source_setup(monkeypatch):
    args = _cli(("--rl-engine", "legacy"))
    assert args.rl_engine == "legacy"
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    # The learner's own default is ports, so legacy is always explicit.
    assert "--rl-engine legacy" in task.run
    assert launcher._miles_source_setup("legacy")[0] in task.setup


def test_default_ports_rejects_legacy_only_runs_with_a_legacy_hint():
    args = _cli(("--rl-model-recipe", "deepseek-v4-flash"))
    with pytest.raises(UnsupportedPortsCombination) as error:
        _prepare_rl_args(args)
    assert "--rl-engine legacy" in str(error.value)
    assert "default" in str(error.value)


def test_ports_is_forwarded_to_learner_and_source_setup(monkeypatch):
    args = _cli(("--rl-engine", "ports"))
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    assert "python3 -m yeto.rl.adapters.miles.island_entry" in task.run and "--rl-engine ports" in task.run
    assert launcher._miles_source_setup("ports")[0] in task.setup


def test_ports_provenance_is_recorded_only_for_ports():
    legacy = _cli(("--rl-engine", "legacy"))
    legacy._provenance = {"model": {}, "dataset": {}}
    with pytest.raises(ValueError):  # provenance fixture is not HF-pinned
        _prepare_rl_args(legacy)
    assert "rl_engine" not in legacy._provenance
    ports = _cli(("--rl-engine", "ports"))
    ports._provenance = {"model": {}, "dataset": {}}
    with pytest.raises(ValueError):
        _prepare_rl_args(ports)
    assert ports._provenance["rl_engine"] == "ports"
    default = _cli()
    default._provenance = {"model": {}, "dataset": {}}
    with pytest.raises(ValueError):
        _prepare_rl_args(default)
    assert default._provenance["rl_engine"] == "ports"  # default is recorded


def test_rl_image_default_follows_the_engine(monkeypatch):
    from yeto.rl import MILES_IMAGE, MILES_NEXT_IMAGE

    legacy = _cli(("--rl-engine", "legacy"))
    assert legacy.rl_image is None
    _prepare_rl_args(legacy)
    assert legacy.rl_image == MILES_IMAGE  # legacy default unchanged
    default = _cli()
    _prepare_rl_args(default)
    assert default.rl_image == MILES_NEXT_IMAGE
    ports = _cli(("--rl-engine", "ports"))
    _prepare_rl_args(ports)
    assert ports.rl_image == MILES_NEXT_IMAGE
    assert "ghcr.io/agentenv" not in ports.rl_image
    assert _island_task(ports, monkeypatch).resources.image_id == MILES_NEXT_IMAGE
    explicit = "docker:example/miles@sha256:" + "e" * 64
    pinned = _cli(("--rl-engine", "ports", "--rl-image", explicit))
    _prepare_rl_args(pinned)
    assert pinned.rl_image == explicit


def test_ports_island_puts_the_pinned_miles_checkout_first(monkeypatch):
    legacy = _cli(("--rl-engine", "legacy"))
    _prepare_rl_args(legacy)
    legacy_run = _island_task(legacy, monkeypatch).run
    assert "PYTHONPATH=$HOME/sglang/python:$HOME/sky_workdir${PYTHONPATH:+:$PYTHONPATH} " in legacy_run
    ports = _cli(("--rl-engine", "ports"))
    _prepare_rl_args(ports)
    ports_run = _island_task(ports, monkeypatch).run
    # The upstream image's /root/miles is on its PYTHONPATH; ours shadows it.
    assert (
        "PYTHONPATH=$HOME/miles:$HOME/sglang/python:$HOME/sky_workdir:/root/Megatron-LM"
        "${PYTHONPATH:+:$PYTHONPATH} " in ports_run
    )
    assert 'RAY_ADDRESS="$MASTER_ADDR:6379"' in ports_run


def test_legacy_island_in_the_ports_image_gets_the_megatron_checkout(monkeypatch):
    from yeto.rl import MILES_IMAGE, MILES_NEXT_IMAGE

    # S14 G4/G5 (s14-dlr-legacy-20261007b/c): legacy in MILES_NEXT_IMAGE
    # needs the image's editable Megatron-LM checkout on PYTHONPATH too (the
    # run still fails there: that Megatron dropped megatron.training.tokenizer,
    # which agentenv/miles imports; legacy belongs in MILES_IMAGE).
    legacy = _cli(("--rl-engine", "legacy", "--rl-image", MILES_NEXT_IMAGE))
    _prepare_rl_args(legacy)
    assert launcher.island_uses_ports_megatron(legacy)
    run = _island_task(legacy, monkeypatch).run
    assert (
        "PYTHONPATH=$HOME/sglang/python:$HOME/sky_workdir:/root/Megatron-LM"
        "${PYTHONPATH:+:$PYTHONPATH} " in run
    )
    assert "PYTHONPATH=$HOME/miles:" not in run  # legacy pip -e installs ~/miles
    own = _cli(("--rl-engine", "legacy"))
    _prepare_rl_args(own)
    assert own.rl_image == MILES_IMAGE
    assert not launcher.island_uses_ports_megatron(own)
    assert "/root/Megatron-LM" not in _island_task(own, monkeypatch).run
    other = _cli(("--rl-engine", "legacy", "--rl-image", "docker:example/miles@sha256:" + "e" * 64))
    _prepare_rl_args(other)
    assert not launcher.island_uses_ports_megatron(other)


def test_modal_ports_island_does_not_request_the_external_router():
    from yeto.gpu_spec import parse_gpu_spec

    task = SimpleNamespace(run="true", envs={}, setup="true")
    legacy = _cli(("--gpu", "modal:1xh100", "--cluster-prefix", "run", "--rl-engine", "legacy"))
    _prepare_rl_args(legacy)
    (spec,) = parse_gpu_spec(legacy.gpu)
    cfg = launcher.build_modal_island_config(legacy, spec, 0, task, "1.2.3.4:5000")
    assert cfg.envs["YETO_RL_EXTERNAL_ROUTER"] == "1"
    ports = _cli(("--gpu", "modal:1xh100", "--cluster-prefix", "run", "--rl-engine", "ports"))
    _prepare_rl_args(ports)
    cfg = launcher.build_modal_island_config(ports, spec, 0, task, "1.2.3.4:5000")
    assert "YETO_RL_EXTERNAL_ROUTER" not in cfg.envs
    assert cfg.image_ref == ports.rl_image.removeprefix("docker:")


def test_ports_router_mode_ignores_external_router_and_refuses_preset_address(capsys):
    args = SimpleNamespace(sglang_router_ip=None, sglang_router_port=None)
    rl_learner.require_ports_router_mode(args, environ={rl_learner.EXTERNAL_ROUTER_ENV: "1"})
    assert args.sglang_router_ip is None  # upstream resolves its own router
    assert "ignored on --rl-engine ports" in capsys.readouterr().out
    with pytest.raises(ValueError, match="external SGLang router"):
        rl_learner.require_ports_router_mode(
            SimpleNamespace(sglang_router_ip="10.0.0.1", sglang_router_port=3000), environ={}
        )


@pytest.mark.parametrize(
    "extra, reason",
    [
        (("--rl-model-recipe", "deepseek-v4-flash"), "DeepSeek V4"),
        (("--expert-full-count", "2"), "DeepSeek V4"),
        (("--lora-targets", "attention-routed-experts"), "DeepSeek V4"),
    ],
)
def test_launcher_rejects_ports_before_startup(extra, reason):
    args = _cli(("--rl-engine", "ports", *extra))
    with pytest.raises(UnsupportedPortsCombination, match=reason):
        _prepare_rl_args(args)


# ---------------------------------------------------------------- learner


def _learner_argv(extra=()):
    return [
        "--model", "org/model", "--model-revision", "a" * 40, "--data", "org/data",
        "--syncer", "127.0.0.1:29400", "--learner-id", "0",
        "--reward-function", "pkg.reward:score", "--reward-sha256", "e" * 64,
        "--source-sha256", "c" * 64, "--global-rounds", "2",
        "--groups-per-round", "2", "--samples-per-group", "2",
        "--over-sampling-batch-size", "2", "--optimizer-steps", "1",
        "--rollout-max-response-len", "16", "--completed-groups-path", "/tmp/x.pt",
        "--event-tape", "/tmp/x.jsonl", "--actor-num-nodes", "1",
        "--actor-num-gpus-per-node", "1", "--inner-lr", "1e-5", "--seq-len", "64",
        "--seed", "1", "--miles-root", "/tmp/miles",
        "--lora-r", "8", "--lora-targets", "attention", *extra,
    ]


def test_learner_default_is_ports():
    assert rl_learner.parse_args(_learner_argv()).rl_engine == "ports"
    assert rl_learner.parse_args(_learner_argv(("--rl-engine", "legacy"))).rl_engine == "legacy"


@pytest.mark.parametrize(
    "extra",
    [
        ("--parameter-mode", "full", "--sync-preset", "dense-full"),
        ("--rl-model-recipe", "deepseek-v4-flash"),
        ("--expert-full-count", "2"),
        ("--rollout-num-gpus", "2"),
    ],
)
def test_learner_parser_rejects_unsupported_ports_runs(extra, capsys):
    argv = _learner_argv(("--rl-engine", "ports", *extra))
    if "--parameter-mode" in extra:
        argv = [a for a in argv if a not in ("--lora-r", "8", "--lora-targets", "attention")]
    with pytest.raises(SystemExit):
        rl_learner.parse_args(argv)
    assert "only supported by the legacy path" in capsys.readouterr().err
    # the default (ports) rejects it too
    default = [a for a in argv if a not in ("--rl-engine", "ports")]
    with pytest.raises(SystemExit):
        rl_learner.parse_args(default)
    assert "--rl-engine legacy" in capsys.readouterr().err
    # explicit legacy accepts the same combination at parse time
    rl_learner.parse_args([*default, "--rl-engine", "legacy"])


@pytest.mark.parametrize(
    "extra_argv, reason",
    [
        (("--use-critic",), "critic"),
        (("--advantage-estimator", "gspo"), "non-GRPO"),
        (("--sao-compaction",), "SAO"),
        (("--rollout-num-gpus", "4"), "fixed partition"),
    ],
)
def test_run_miles_rejects_ports_before_loading_anything(extra_argv, reason, monkeypatch):
    args = rl_learner.parse_args(_learner_argv(("--rl-engine", "ports")))
    # Anything past the rejection would import Megatron; make that loud.
    monkeypatch.setitem(sys.modules, "megatron", None)
    with pytest.raises(UnsupportedPortsCombination, match=reason):
        rl_learner.run_miles(
            args, model_path="/nonexistent", prompt_path="/nonexistent",
            extra_argv=extra_argv,
        )


def test_run_miles_ports_refuses_a_miles_pin_without_run_plugin(monkeypatch):
    from yeto.rl.adapters.miles.state import PolicyStateError

    group = types.ModuleType("miles.ray.train.group")
    group.TrainerController = type("TrainerController", (), {})
    for name in ("miles", "miles.ray", "miles.ray.train"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.ray.train.group", group)
    monkeypatch.setitem(sys.modules, "megatron", None)
    args = rl_learner.parse_args(_learner_argv(("--rl-engine", "ports")))
    with pytest.raises(PolicyStateError, match="run_plugin"):
        rl_learner.run_miles(args, model_path="/x", prompt_path="/x")
    group.TrainerController.run_plugin = lambda self, fn, kwargs=None: []
    with pytest.raises(ImportError):  # passes the gate, then needs Megatron
        rl_learner.run_miles(args, model_path="/x", prompt_path="/x")


def test_learner_main_verifies_the_ports_pin_group(monkeypatch):
    from yeto.rl import MILES_NEXT_PINS

    calls = []

    class Stop(Exception):
        pass

    def verify(root, **kwargs):
        calls.append(kwargs)
        raise Stop

    monkeypatch.setattr(rl_learner, "verify_miles_revision", verify)
    import yeto.provenance as provenance

    monkeypatch.setattr(provenance, "verify_source_tree_sha256", lambda value: value)
    monkeypatch.setattr(provenance, "python_spec_sha256", lambda spec: "e" * 64)
    with pytest.raises(Stop):
        rl_learner.main(_learner_argv(("--rl-engine", "ports", "--data-revision", "f" * 40)))
    assert calls == [{"expected": MILES_NEXT_PINS}]
    calls.clear()
    with pytest.raises(Stop):
        rl_learner.main(_learner_argv(("--data-revision", "f" * 40)))
    assert calls == [{"expected": MILES_NEXT_PINS}]  # ports is the default
    calls.clear()
    with pytest.raises(Stop):
        rl_learner.main(
            _learner_argv(("--rl-engine", "legacy", "--data-revision", "f" * 40))
        )
    assert calls == [{"expected_source_sha256": None}]  # legacy call unchanged


# ---------------------------------------------------------------- SSH harness


def test_harness_plan_key_only_for_ports():
    from tests.test_rl_ssh_harness import _plan  # noqa: PLC0415
    from yeto.rl import ssh_harness
    from yeto.rl.ssh_harness import HarnessError, _learner_argv

    plan = _plan()  # no rl_engine key = legacy plan (digest unchanged)
    ssh_harness._validate_plan(plan)
    argv = _learner_argv(plan, 0)
    # The learner defaults to ports, so a legacy plan passes legacy explicitly.
    assert argv[argv.index("--rl-engine") + 1] == "legacy"
    assert rl_learner.parse_args(argv[3:]).rl_engine == "legacy"
    ports = {**_plan(), "rl_engine": "ports"}
    ssh_harness._validate_plan(ports)
    argv = _learner_argv(ports, 0)
    assert argv[argv.index("--rl-engine") + 1] == "ports"
    assert rl_learner.parse_args(argv[3:]).rl_engine == "ports"
    with pytest.raises(HarnessError, match="absent"):
        ssh_harness._validate_plan({**_plan(), "rl_engine": "legacy"})
    dsv4 = {**_plan(), "rl_engine": "ports"}
    dsv4["learner"] = {**dsv4["learner"], "rl_model_recipe": "deepseek-v4-flash"}
    with pytest.raises(HarnessError, match="legacy"):
        ssh_harness._validate_plan(dsv4)


# ---------------------------------------------------------------- driver fixes


def test_nonfinite_reward_is_rejected_before_training(tmp_path):
    from yeto.rl.core import StrictRlInvariantError
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine, fake_capabilities
    from yeto.rl.engine.ports import GroupMetadata

    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)})
    generate = engine.rollout.generate

    def nan_generate(rollout_id, **kw):
        batch = generate(rollout_id, **kw)
        g = batch.groups[0]
        bad = GroupMetadata(g.group_id, g.sample_ids, g.policy_token, math.nan, math.nan, 3)
        return type(batch)(
            batch.rollout_id, batch.policy_version, batch.policy_hash,
            (bad, *batch.groups[1:]), batch.completed, batch.aborted, payload=batch.payload,
        )

    engine.rollout.generate = nan_generate
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
        policy_state=engine.policy_state, publisher=engine.publisher,
        placement=engine.placement, capabilities=fake_capabilities(),
        algorithm=AlgorithmSpec(), sync=LocalOnlySync(2),
        events=EventTape(tmp_path / "ev.jsonl", 0),
    )
    with pytest.raises(StrictRlInvariantError, match="non-finite reward"):
        driver.run()
    assert not [c for c in engine.calls if c[0] == "train"]


def test_strict_apply_aligns_scheduler_in_optimizer_steps():
    from yeto.rl.core import canonical_state
    from yeto.rl.engine.bridges import StrictAvgSync

    calls = []
    driver = SimpleNamespace(apply_policy=lambda state, **kw: calls.append(kw) or state)
    sync = StrictAvgSync(SimpleNamespace(local_optimizer_steps=3))
    state = canonical_state(
        4, {"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
        base_model_revision="a" * 40, lora_config_hash="b" * 64,
    )
    sync._apply(driver, state)
    assert calls == [{"optimizer": "reset", "local_step": 12}]


# ---------------------------------------------------------------- composition root


NAME = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"


class _Engine:
    def __init__(self):
        self.server_url, self.version = "http://e0", None

    async def update_weight_version(self, token, abort_all_requests=False):
        self.version = token

    async def get_weight_version(self):
        return self.version


class _Controller:
    def __init__(self):
        self.engine, self.calls = _Engine(), []

    async def prepare_rollout(self, rollout_id):
        self.calls.append(("prepare", rollout_id))

    async def abort_all(self):
        pass

    async def get_cell_statuses(self):
        return {"c0": SimpleNamespace(phase="Running")}

    async def start_update_weights(self):
        return SimpleNamespace(rollout_engines=[self.engine], snapshot_cell_id_to_hashes={"c0": "h"})

    async def end_update_weights(self, snapshot_cell_id_to_hashes):
        self.calls.append(("end_update",))

    async def abort_update_weights(self):
        self.calls.append(("abort_update",))

    async def check_weights(self, action, **kw):
        return [{"success": True, "ranks": [{"checksums": {"w": "x"}}]}]


class _Actor:
    """Single-cell TrainGroup stub answering the yeto state plugin."""

    def __init__(self):
        self.tensor = torch.zeros(2, 4)
        self.norm = 0.0
        self.calls = []

    async def run_plugin(self, fn_path, kwargs=None):
        from yeto.rl.adapters.miles import state_plugin

        kwargs = kwargs or {}
        if fn_path == state_plugin.EXPORT_STATE:
            return [
                {"policy_version": kwargs["policy_version"], "tensors": {NAME: self.tensor.clone()}},
                None,
            ]
        if fn_path == state_plugin.EXPORT_DIGEST:  # rl-publish-fastpath
            from yeto.rl.engine.policy_digest import digest_canonical_tensors, layout_hash_of

            tensors = {NAME: self.tensor.clone()}
            specs = [(NAME, tuple(self.tensor.shape))]
            digest = digest_canonical_tensors(
                tensors, base_model_revision=kwargs["base_model_revision"],
                lora_config_hash=kwargs["lora_config_hash"], layout_hash=layout_hash_of(specs))
            return [{"policy_version": kwargs["policy_version"], "digest": digest.to_wire(),
                     "weights": state_plugin.current_weights_version()}, None]
        if fn_path == state_plugin.WEIGHTS_VERSION:
            return [state_plugin.current_weights_version(), None]
        if fn_path == state_plugin.APPLY_STATE:
            self.tensor = kwargs["tensors"][NAME].clone()
            self.calls.append(("apply", kwargs["policy_version"], kwargs["optimizer"]))
            return [{"ok": True}, {"ok": True}]
        if fn_path == state_plugin.GRAD_NORM:
            return [self.norm]
        if fn_path == state_plugin.APPLIED_LRS:
            return [[1e-5]]
        if fn_path == state_plugin.STEP_LOSSES:
            return [[]]
        raise AssertionError(fn_path)

    async def train(self, rollout_id, pack):
        self.calls.append(("train", rollout_id))
        self.tensor = self.tensor + 1.0
        self.norm = 2.0
        return [SimpleNamespace(outcome=SimpleNamespace(name="NORMAL"))]

    async def onload(self):
        self.calls.append(("onload",))

    async def offload(self):
        self.calls.append(("offload",))

    async def clear_memory(self):
        self.calls.append(("offload",))


def test_ports_composition_root_over_stubbed_upstream(tmp_path, monkeypatch):
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape
    from yeto.rl.adapters.miles import LoopRunner
    from yeto.rl.adapters.miles import rollout_meta_hook as hook
    from yeto.rl.adapters.miles.entry import compose_island, miles_capabilities
    from yeto.rl.adapters.miles.placement import MilesPlacement, PlacementRequest
    from yeto.rl.adapters.miles.rollout import DirMetadataSource
    from tests.test_rl_miles_adapter_rollout import Call, Sample, Span
    from yeto.rl.core import canonical_state

    sink = tmp_path / "sink"
    monkeypatch.setenv(hook.META_SINK_ENV, f"dir:{sink}")
    controller, actor = _Controller(), _Actor()
    rollout_args = SimpleNamespace(n_samples_per_prompt=2)
    stale = "yeto:9:" + "c" * 64

    class Executor:
        async def get(self, rollout_id):
            token = controller.engine.version  # what SGLang stamps on samples
            groups = [
                [Sample(index=10 * g + i, group_index=g, rollout_id=rollout_id,
                        reward=float(i), weight_versions=[Call([Span(token)])])
                 for i in range(2)]
                for g in range(2)
            ]
            # rollout-process group reuse: a stale buffered group is dropped
            buffer = [[Sample(index=90 + i, group_index=9, weight_versions=[Call([Span(stale)])])
                       for i in range(2)]]
            assert hook.policy_buffer_filter(rollout_args, None, buffer, 1) == []
            hook.record_trained_groups(rollout_args, groups)
            hook.extract_rollout_metadata(rollout_args, groups)
            return SimpleNamespace(sample_indices=[s.index for g in groups for s in g])

    async def update_weights(*args, **kwargs):
        pass

    miles_args = SimpleNamespace(num_steps_per_rollout=1, offload_train=True, offload_rollout=False)
    runner = LoopRunner()
    released = []
    driver = compose_island(
        miles_args=miles_args,
        launch=SimpleNamespace(placement=PlacementRequest("colocated", 1, 1, 1)),
        algorithm=AlgorithmSpec(),
        inference_controller=controller,
        rollout_executor=Executor(),
        actor_model=actor,
        learner_id=0,
        base_model_revision="0" * 40,
        lora_config_hash="1" * 64,
        layout_hash=canonical_state(
            0, {NAME: torch.zeros(2, 4)}, base_model_revision="0" * 40,
            lora_config_hash="1" * 64,
        ).layout_hash,
        sync=LocalOnlySync(2),
        progress=None,
        metadata=DirMetadataSource(sink),
        capabilities=miles_capabilities("sha256:" + "0" * 64),
        runner=runner,
        events=EventTape(tmp_path / "events.jsonl", 0),
        update_weights=update_weights,
        release_refs=lambda args, pack: released.append(pack),
        flatten_checksums=lambda raw: [{"w": "x"} for _ in raw],
        placement=MilesPlacement(
            PlacementRequest("colocated", 1, 1, 1),
            {"actor": (None, [0], [0]), "rollout": (None, [0], [0])},
            logical=True,
        ),
    )
    final = driver.run()
    # rl-publish-fastpath: the final policy stays in the trainer (digest handle);
    # read its tensors (a full export) while the loop runner is still open.
    final_tensors = final.tensors
    runner.close()

    assert torch.equal(final_tensors[NAME], torch.full((2, 4), 2.0))
    assert [c for c in actor.calls if c[0] == "train"] == [("train", 0), ("train", 1)]
    assert len(released) == 2
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    publications = [e for e in events if e["event"] == "rl_publication"]
    tokens = [e["rl/policy_token"] for e in publications]
    assert [t.split(":")[1] for t in tokens] == ["0", "1", "2"]
    # publisher token == driver token == what the rollout process filtered on
    assert controller.engine.version == tokens[-1]
    assert hook.current_policy_token() == tokens[-2]
    assert all(e["sync/publication_members"] == ["engine:c0"] for e in publications)
    local = [e for e in events if e["event"] == "rl_local_round"]
    assert [e["grad_norm"] for e in local] == [2.0, 2.0]


def test_run_miles_ports_never_starts_the_legacy_external_router(monkeypatch):
    from yeto.rl.adapters.miles.state import PolicyStateError

    group = types.ModuleType("miles.ray.train.group")
    group.TrainerController = type("TrainerController", (), {})
    # Upstream Miles: no get_host_info in http_utils.
    http_utils = types.ModuleType("miles.utils.http_utils")
    for name in ("miles", "miles.ray", "miles.ray.train", "miles.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.ray.train.group", group)
    monkeypatch.setitem(sys.modules, "miles.utils.http_utils", http_utils)
    monkeypatch.setitem(sys.modules, "megatron", None)
    monkeypatch.setenv(rl_learner.EXTERNAL_ROUTER_ENV, "1")

    def legacy_router(*_args, **_kwargs):
        raise AssertionError("ports must not start the legacy external router")

    monkeypatch.setattr(rl_learner, "start_external_sglang_router", legacy_router)
    args = rl_learner.parse_args(_learner_argv(("--rl-engine", "ports")))
    with pytest.raises(PolicyStateError, match="run_plugin"):  # got past the router
        rl_learner.run_miles(args, model_path="/x", prompt_path="/x")


def test_explicit_fixed_partition_is_a_ports_combination():
    # rl-infra-spec 2.1: allowed only when placement is requested explicitly
    assert ports_rejections(placement="fixed-partition", rollout_num_gpus=2) == []
    assert "needs --rollout-num-gpus" in " ".join(ports_rejections(placement="fixed-partition"))
    assert ports_rejections(placement="elastic")


def test_island_post_cmd_runs_after_the_learner_and_keeps_its_exit_code(monkeypatch):
    args = _cli()
    _prepare_rl_args(args)
    assert "post-cmd" not in _island_task(args, monkeypatch).run  # default: unchanged
    args = _cli(("--rl-island-post-cmd", "nvidia-smi -L; echo done"))
    _prepare_rl_args(args)
    run = _island_task(args, monkeypatch).run
    learner = next(l for l in run.splitlines() if "-m yeto.rl.adapters.miles.island_entry" in l)
    assert learner.endswith(" || LRC=$?")
    after = run.split(learner, 1)[1]
    assert '( set +e; nvidia-smi -L; echo done ) > "$HOME/yeto-output/post-cmd.txt" 2>&1' in after
    assert "exit ${LRC:-0}" in after.split("else", 1)[0]
