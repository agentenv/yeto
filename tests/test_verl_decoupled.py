"""Decoupled DiLoCo (LoRA fragments) on verl islands.

The verl island runs the engine-neutral ``DecoupledSync`` (``yeto.rl.engine.bridges``)
over its own ports; ``adapters/verl/decoupled_session.py`` only builds the session's
configuration from the verl plan. The last test runs two launcher-built verl islands
through the launcher-built Rust syncer: the real ``VerlPolicyState`` and
``VerlPublisher`` (and the real worker functions ``remote_export_lora`` /
``remote_apply_lora``) over a one-rank fake of verl's FSDP2 PEFT worker, driven by the
neutral ``IslandDriver``. CPU only; the last test needs cargo.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from free_ports import free_port
from process_groups import group_popen, kill_group
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_verl_full_parameter_sync import needs_cargo, syncer_binary  # noqa: F401 - fixture
from verl_island_fakes import ISLAND_ARGV, drive_trainer, modules, run_island_entry, stub_verl_and_ray
from yeto.rl.adapters.verl import decoupled_session as ds
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl.publish import lora_checksum
from yeto.rl.core import canonical_layout_hash, canonical_state, policy_tensor_hash
from yeto.rl.engine import progress
from yeto.rl.engine.bridges import DecoupledIslandProgress, DecoupledSync

ENTRY = "-m yeto.rl.adapters.verl.island_entry"
GPU = "modal:1xh200,modal:1xh200"
MODEL_REVISION = "a" * 40
LORA_HASH = "b" * 64
# canonical PEFT names, as the verl FSDP2 PEFT model exports them (param_names)
SHAPES = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": (2, 8),
          "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (8, 2),
          "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": (2, 12),
          "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": (16, 2)}


def _initial(version: int = 0):
    gen = torch.Generator().manual_seed(3)
    return canonical_state(version, {n: torch.randn(s, generator=gen) for n, s in SHAPES.items()},
                           base_model_revision=MODEL_REVISION, lora_config_hash=LORA_HASH)


def _prepared(*extra):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-backend", "verl", "--reward-function", "gsm8k_reward:score",
                                    "--rl-sync-preset", "decoupled", "--fragments", "2", "--pipeline", "2",
                                    "--local-rl-rounds-per-sync", "2", *extra), gpu=GPU)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _island_argv(args, learner_id: int, syncer: str) -> list[str]:
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task

    spec = parse_gpu_spec(args.gpu)[learner_id]
    task = launcher._island_task(make_miles_island_task, args, spec, learner_id, 2, syncer, {})
    line = next(line for line in task.run.splitlines() if ENTRY in line)
    return shlex.split(line.split(ENTRY, 1)[1].replace("$LEARNER_ID", str(learner_id))
                       .replace("$SYNCER_ADDR", syncer))


def _no_problems(parsed) -> list[str]:
    return ds.island_problems(parsed, megatron=False, buffer_trainer=False, max_policy_age=0,
                              rollout_ahead=0, separate=False, resume=None)


def _plan(parsed, tmp_path: Path, learner_id: int) -> dict:
    parsed.completed_groups_path = str(tmp_path / f"island-{learner_id}.pt")
    return {"learner_id": learner_id, "syncer": parsed.syncer, "model_revision": parsed.model_revision,
            "global_rounds": parsed.global_rounds, "groups_per_round": parsed.groups_per_round,
            "samples_per_group": parsed.samples_per_group, "wan_streams": 1,
            ds.PLAN_KEY: ds.plan_section(parsed, trainer_gpus=1)}


# ----------------------------------------------------------------------------- launcher -> island plan
def test_the_launcher_argv_reaches_the_decoupled_plan(tmp_path):
    args = _prepared()
    parsed, ignored = island_entry.parse_args(_island_argv(args, 1, "127.0.0.1:9"))
    assert ds.requested(parsed) and _no_problems(parsed) == []
    for flag in ("--sync-preset", "--completed-groups-path", "--source-sha256", "--reward-sha256"):
        assert flag not in ignored, flag
    section = _plan(parsed, tmp_path, 1)[ds.PLAN_KEY]
    assert (section["fragments"], section["pipeline"], section["local_horizon"]) == (2, 2, 2)
    assert section["total_fragment_steps"] == args.total_steps * 2
    assert section["matrix_merge"] == args.rl_sync_matrix_merge
    assert section["source_sha256"] == "c" * 64 and section["reward_sha256"] == "d" * 64


@pytest.mark.parametrize("kwargs, needle", [
    ({"megatron": True}, "FSDP2"), ({"buffer_trainer": True}, "buffer-trainer"),
    ({"max_policy_age": 1}, "--rl-max-policy-age 0"), ({"rollout_ahead": 1}, "--rl-rollout-ahead"),
    ({"separate": True}, "colocated"), ({"resume": {"store": "s"}}, "cuts"),
])
def test_the_island_refuses_the_shapes_the_session_does_not_cover(kwargs, needle):
    parsed, _ = island_entry.parse_args(_island_argv(_prepared(), 0, "127.0.0.1:9"))
    shape = {"megatron": False, "buffer_trainer": False, "max_policy_age": 0, "rollout_ahead": 0,
             "separate": False, "resume": None, **kwargs}
    assert any(needle in problem for problem in ds.island_problems(parsed, **shape))


def test_the_island_refuses_an_incomplete_decoupled_command():
    parsed, _ = island_entry.parse_args(_island_argv(_prepared(), 0, "127.0.0.1:9"))
    parsed.fragments, parsed.local_horizon, parsed.completed_groups_path = 1, 1, None
    problems = " | ".join(_no_problems(parsed))
    for needle in ("--fragments >= 2", "--local-horizon >= 2", "--total-fragment-steps", "--completed-groups-path"):
        assert needle in problems, needle


SYNCED = tuple(token for token in ISLAND_ARGV if token != "--rl-single-island-no-sync")
DECOUPLED_ISLAND = ("--syncer", "127.0.0.1:9", "--sync-preset", "decoupled", "--fragments", "2", "--pipeline", "2",
                    "--local-horizon", "2", "--total-fragment-steps", "2", "--completed-groups-path", "p",
                    "--source-sha256", "c" * 64, "--reward-sha256", "d" * 64)


def test_the_island_entry_plans_the_decoupled_session(monkeypatch, tmp_path):
    # eval_section runs after the session is chosen and its plan section built
    name, _ = run_island_entry(monkeypatch, tmp_path, *DECOUPLED_ISLAND, stop="eval_section", base=SYNCED)
    assert name == "eval_section"


@pytest.mark.parametrize("argv, needle", [
    (("--fragments", "1"), "a verl decoupled island needs: --fragments >= 2"),
    (("--tensor-parallel", "2", "--parameter-mode", "full"), "needs --parameter-mode lora"),
    (("--bcast-delta",), "--bcast-delta"),
])
def test_the_island_entry_refuses_what_the_session_does_not_cover(monkeypatch, tmp_path, argv, needle):
    name, message = run_island_entry(monkeypatch, tmp_path, *DECOUPLED_ISLAND, *argv, stop="eval_section",
                                     base=SYNCED)
    assert name == "refused" and needle in message, message


# ----------------------------------------------------------------------------- the session configuration
def test_the_session_reads_the_plan_and_writes_a_schema4_progress_file(tmp_path):
    from yeto.rl.core import PolicySnapshot

    parsed, _ = island_entry.parse_args(_island_argv(_prepared(), 0, "127.0.0.1:9"))
    initial = _initial()
    plan = _plan(parsed, tmp_path, 0)

    def session():
        return ds.session_args(plan, initial.specs, lora_hash=LORA_HASH, layout_hash=initial.layout_hash,
                               identity_sha256="e" * 64, compat_group="nvidia-h200")

    args = session()
    bridge = args.yeto_rl_bridge_config
    assert (bridge.num_fragments, bridge.pipeline, bridge.local_horizon) == (2, 2, 2)
    assert bridge.syncer_addr == ("127.0.0.1", 9) and bridge.canonical_layout_hash == initial.layout_hash
    assert bridge.backend_identity_sha256 == "e" * 64 and bridge.compat_group == "nvidia-h200"
    assert args.yeto_rl_sync_layout_fingerprint == ds.sync_layout_fingerprint(
        initial.specs, 2, plan[ds.PLAN_KEY]["matrix_merge"])
    # The progress file's config block is complete and the same for the same plan
    config = progress._decoupled_checkpoint_config(args)
    assert config == progress._decoupled_checkpoint_config(session())
    assert config["sync_preset"] == "decoupled" and config["layout_hash"] == initial.layout_hash
    snapshot = PolicySnapshot.create(0, initial, (0, 0))
    progress._save_decoupled_checkpoint(args, snapshot=snapshot, optimizer_steps=0, action_tokens=0,
                                        rollout_metrics={}, local_round_stats=None, completed_groups=[])
    assert progress._load_decoupled_checkpoint(session())["policy_token"] == snapshot.token


def test_run_island_builds_the_decoupled_sync_and_its_progress(monkeypatch, tmp_path):
    from yeto.rl import export

    initial = _initial()
    monkeypatch.setattr(export, "derive_peft_lora_specs", lambda *a, **k: initial.specs)
    parsed, _ = island_entry.parse_args(_island_argv(_prepared(), 0, "127.0.0.1:9"))
    plan = {**_plan(parsed, tmp_path, 0), "sync": "decoupled", "parameter_mode": "lora",
            "max_policy_age": 0, "model_path": str(tmp_path), "lora_rank": 2, "lora_targets": "all-linear",
            "out_dir": str(tmp_path), "event_tape": str(tmp_path / "tape.jsonl"), "thresholds_key": []}
    with stub_verl_and_ray():
        kwargs = drive_trainer(monkeypatch, tmp_path, "sync", plan=plan)
    assert isinstance(kwargs["sync"], DecoupledSync) and isinstance(kwargs["progress"], DecoupledIslandProgress)
    assert kwargs["sync"].args.yeto_rl_bridge_config.canonical_layout_hash == canonical_layout_hash(initial.specs)


def test_the_fully_async_path_refuses_a_decoupled_plan():
    with stub_verl_and_ray():
        (trainer,) = modules("yeto.rl.adapters.verl.trainer")
        with pytest.raises(ValueError, match="synchronous verl trainer"):
            trainer.run_fully_async_island(lambda *a: None, None, {"sync": "decoupled"}, {})


# ----------------------------------------------------------------------------- two islands, real syncer
class _PeftModule:
    """One rank of verl's FSDP2 PEFT model: ``lora_{A,B}.default.weight`` parameters that
    alias the fake engine's tensors (a single rank holds whole tensors, no DTensor)."""

    def __init__(self, engine):
        self.engine = engine

    def named_parameters(self):
        for name, tensor in self.engine.tensors.items():
            head = name[: -len(".weight")]
            yield f"{head}.default.weight", torch.nn.Parameter(tensor, requires_grad=False)


class _Optimizer:
    def __init__(self, engine):
        self.engine = engine

    @property
    def state(self):
        return {name: {"exp_avg": moment, "exp_avg_sq": moment} for name, moment in self.engine.moments.items()}


class _CheckpointManager:
    """verl's colocated ``update_weights``: the rollout side registers the trainer's
    LoRA; the read-back reports its checksum (``vllm_readback``)."""

    def __init__(self, engine, island):
        self.engine, self.island = engine, island

    def update_weights(self, _global_steps):
        version = int((self.island.readback_dir / "expected_version").read_text())
        tensors = {name: value.clone() for name, value in self.engine.tensors.items()}
        (self.island.readback_dir / f"readback-v{version}.json").write_text(json.dumps(lora_checksum(tensors)))
        state = canonical_state(version, tensors, base_model_revision=MODEL_REVISION, lora_config_hash=LORA_HASH)
        self.engine.published = (version, policy_tensor_hash(state))


def _verl_island(engine, tmp_path: Path, learner_id: int):
    from yeto.rl.adapters.verl.ports_impl import VerlIsland

    initial = _initial()
    worker = SimpleNamespace(actor=SimpleNamespace(engine=SimpleNamespace(
        module=_PeftModule(engine), optimizer=_Optimizer(engine))))

    class Island(VerlIsland):
        def call_workers(self, fn, *args):
            return [fn(worker, *args)]

    trainer = SimpleNamespace(global_steps=0)
    island = Island(trainer, learner_id=learner_id, out_dir=str(tmp_path / f"out-{learner_id}"),
                    base_model_revision=MODEL_REVISION, lora_config_hash=LORA_HASH,
                    layout_hash=initial.layout_hash, expected_specs=initial.specs, tis_upper=2.0,
                    thresholds_key=())
    trainer.checkpoint_manager = _CheckpointManager(engine, island)
    return island


def _run_verl_island(plan: dict, tmp_path: Path, learner_id: int, results: dict) -> None:
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.ports_impl import MEMBER, VerlPolicyState, VerlPublisher
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine

    try:
        initial = _initial()
        # the two islands train in different directions from the same version-zero cut
        engine = FakeEngine(tensors=dict(initial.tensors), step_delta=0.01 * (learner_id + 1),
                            members_ids=(MEMBER,))
        island = _verl_island(engine, tmp_path, learner_id)
        sync = ds.build_sync(plan, initial.specs, lora_hash=LORA_HASH, layout_hash=initial.layout_hash,
                             identity_sha256=None, compat_group=None)
        tape = tmp_path / f"island-{learner_id}.jsonl"
        driver = IslandDriver(
            learner_id=learner_id, rollout=engine.rollout, trainer=engine.trainer,
            policy_state=VerlPolicyState(island), publisher=VerlPublisher(island, timeout_s=10),
            placement=engine.placement, capabilities=verl_capabilities("sha256:" + "0" * 64),
            algorithm=AlgorithmSpec(), sync=sync, events=EventTape(tape, learner_id),
            progress=ds.progress_for(sync), max_rollouts=200)
        state = driver.run()
        results[learner_id] = {"state": state, "island": island, "tape": tape}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error


@needs_cargo
@pytest.mark.parametrize("mode", ["async", "strict"])
def test_two_launcher_built_verl_islands_run_decoupled_fragments_through_the_real_syncer(
        syncer_binary, tmp_path, monkeypatch, mode):
    """Both sync modes of the launcher: async (the default: quorum 1, base lag 2, RDA)
    and strict (the decoupled preset's full roster, outer Nesterov 0.7 / 0.9)."""
    from yeto import launcher

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-decoupled-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    monkeypatch.delenv("YETO_VERL_READBACK_DIR", raising=False)
    args = _prepared("--total-steps", "2", "--rl-sync-mode", mode)
    total = args.rl_total_fragment_steps
    port = free_port()
    command = (launcher.role_syncer_commands(args, 2, binary=str(syncer_binary))["actor"]
               .replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    plans = []
    for learner_id in range(2):
        parsed, _ignored = island_entry.parse_args(_island_argv(args, learner_id, f"127.0.0.1:{port}"))
        assert _no_problems(parsed) == []
        plans.append(_plan(parsed, tmp_path, learner_id))
    process = group_popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    results: dict = {}
    try:
        threads = [threading.Thread(target=_run_verl_island, args=(plans[i], tmp_path, i, results))
                   for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not any(thread.is_alive() for thread in threads), "an island did not finish"
        failed = {i: r for i, r in results.items() if isinstance(r, BaseException)}
        assert not failed, failed
        assert process.wait(timeout=30) == 0
    finally:
        kill_group(process)
        log = process.communicate(timeout=10)[0]
    hashes = set()
    for learner_id in range(2):
        result = results.get(learner_id)
        assert result is not None and not isinstance(result, BaseException), (result, log[-3000:])
        events = [json.loads(line) for line in result["tape"].read_text().splitlines()]
        pushes = [e for e in events if e.get("event") == "rl_fragment_push"]
        assert {p["fragment_id"] for p in pushes} == {0, 1}
        assert all(p["c_steps"] >= 2 for p in pushes)  # H = 2 local steps per push
        # every publication went through verl's update_weights and was read back
        island = result["island"]
        checks = [json.loads(line) for line in (island.out / f"verl-publish-{learner_id}.jsonl").read_text()
                  .splitlines()]
        assert checks and all(c["status"] == "VERIFIED" for c in checks)
        # the start applies the syncer's cut with a reset, broadcasts keep the optimizer
        applies = [json.loads(line) for line in (island.out / f"verl-apply-{learner_id}.jsonl").read_text()
                   .splitlines()]
        assert applies[0]["optimizer"] == "reset" and {a["optimizer"] for a in applies[1:]} == {"preserve"}
        hashes.add(result["state"].policy_tensor_hash())
        saved = torch.load(plans[learner_id][ds.PLAN_KEY]["completed_groups_path"], weights_only=True)
        assert saved["config"]["sync_preset"] == "decoupled"
        assert sorted(saved["fragment_versions"]) == [total - 1, total]
    assert len(hashes) == 1  # both islands end on the syncer's final cut
    rows = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines() if line.strip()]
    # one merge per fragment step, both islands in every merge, each fragment in turn
    assert [(r["step"], r["fragment"]) for r in rows] == [(t, (t - 1) % 2) for t in range(1, total + 1)]
    responded = [sorted(r["responded"]) for r in rows]
    if mode == "strict":  # quorum = the roster
        assert "--quorum 2" in command and "--outer-lr 0.7" in command
        assert responded == [[0, 1]] * total, responded
    else:  # quorum 1: a merge may close with the first island
        assert "--quorum 1" in command
        assert all(responded) and set().union(*responded) == {0, 1}, responded
