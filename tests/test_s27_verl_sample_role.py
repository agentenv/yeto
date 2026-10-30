"""S27 G-EW: a verl full-parameter buffer trainer and its sampler island (VERL-PARITY #14).

CPU only. The trainer runs ``trainer.run_fully_async_island`` itself (verl's own
modules are stubbed: the function only needs them at import); the trainer actor is
a fake that holds one verl actor worker (a plain torch module behind the real
worker functions of ``yeto.rl.adapters.verl.full_parameter``). The buffer, the
release store, the driver, the fully_async rollout port, the feed, the publisher,
the full-parameter policy state and the HF writer are the real ones. The sampler
is the real engine sampler island (fake engine) in a thread, as in
test_rl_verl_buffer_trainer.py.

Checks:
* the island entry and the launcher accept a full-parameter buffer trainer (pair
  and synced group), and refuse it with fixed-partition placement;
* three rounds: version 0 (the base) is not released, versions 1-3 are, each one
  the trainer's weights at that version in the model's dtype; the sampler loads
  them in order; no buffer position is trained twice;
* the stream-facing group saw each trained round.
"""
from __future__ import annotations

import importlib
import json
import shlex
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_rl_verl_buffer_trainer import TOLERANT
from test_s27_verl_full_parameter_sync import SHAPES, _Island, _Model, _step
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.adapters.verl import buffer_feed as bf
from yeto.rl.adapters.verl import island_entry, launch_flags
from yeto.rl.adapters.verl.buffer_ports import (CONSUMED_LOG, RELEASE_LOG, install_buffer_ports,
                                                release_weights_for)
from yeto.rl.engine.engine_sampler import EngineSamplerConfig, PromptItem, build_engine_sampler
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_checks import PostLoadChecks
from yeto.rl.engine.sampler_engine import SamplingParams

STORE = "modal-volume://yeto-ew/run"


# ---------------------------------------------------------------------------- the launch
def _launch(monkeypatch, extra: list[str]):
    """The island args of a verl launch (``extra``: the shape; a no-sync pair passes
    ``--tuning full``, the full-parameter preset implies it)."""
    from tests.test_rl_engine_selection import _cli
    from tests.test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task
    from yeto.rl.engine import sample_roles

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k" * 64)  # test key
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _cli(("--rl-backend", "verl", "--rl-checkpoint-store", STORE, "--rl-max-policy-age", "1",
                 "--reward-function", "gsm8k_reward:score", "--total-steps", "4", *extra))
    launcher._prepare_rl_args(args)
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    specs = parse_gpu_spec(args.gpu)
    learners = sample_roles.syncer_learner_count(args, len(specs))
    out = []
    for m in range(len(specs)):
        task = launcher._island_task(make_miles_island_task, args, specs[m], m, learners, "127.0.0.1:1", {})
        line = next(l for l in task.run.splitlines() if "-m yeto.rl.adapters.verl.island_entry" in l)
        argv = shlex.split(line.split("-m yeto.rl.adapters.verl.island_entry", 1)[1]
                           .replace("$LEARNER_ID", task.envs["LEARNER_ID"]).replace("$SYNCER_ADDR", "127.0.0.1:1"))
        out.append(island_entry.parse_args(argv)[0])
    return args, out


def test_one_launch_gives_a_full_parameter_sampler_and_buffer_trainer(monkeypatch):
    _args, (sampler, trainer) = _launch(monkeypatch, [
        "--rl-single-island-no-sync", "--tuning", "full", "--gpu", "modal:1xh200,modal:1xh200",
        "--rl-island-sample-role", "0:sampler", "--rl-island-sample-role", "1:trainer"])
    assert (sampler.rl_sample_role, trainer.rl_sample_role) == ("sampler", "trainer")
    assert sampler.parameter_mode == trainer.parameter_mode == "full"
    assert trainer.rl_max_policy_age == 1 and trainer.rl_single_island_no_sync
    section = island_entry.sample_buffer_section(trainer)
    assert section["store"] == trainer.rl_resume_store and section["lane"] is None


def test_one_launch_gives_synced_full_parameter_buffer_trainers(monkeypatch):
    """The full-parameter preset (asynchronous rounds) with --rl-sample-role trainer: the
    launcher no longer refuses it on verl; each trainer reads its lane, the sampler is a
    no-sync full-parameter island."""
    args, (t0, t1, sampler) = _launch(monkeypatch, [
        "--gpu", "modal:1xh200,modal:1xh200,modal:1xh200", "--rl-sample-role", "trainer",
        "--rl-sync-preset", "full-parameter", "--rl-sync-full-parameter-rounds", "async",
        "--rl-parameter-layout-sha256", "e" * 64, "--fragments", "4", "--rl-island-sample-role", "2:sampler"])
    assert [t.learner_id for t in (t0, t1)] == [0, 1] and t0.num_learners == 2
    assert t0.parameter_mode == t1.parameter_mode == sampler.parameter_mode == "full"
    assert t0.full_parameter_rounds == "async" and t0.parameter_layout_sha256 == "e" * 64
    assert [island_entry.sample_buffer_section(t)["lane"] for t in (t0, t1)] == ["0/2", "1/2"]
    assert sampler.rl_sample_role == "sampler" and sampler.rl_single_island_no_sync
    assert launch_flags.full_parameter_problems(args) == []


def test_the_launcher_refuses_a_buffer_trainer_on_fixed_partition_cards():
    args = types.SimpleNamespace(rl_placement="fixed-partition", rl_sample_role="trainer", rl_max_policy_age=1,
                                 rl_parameter_layout_sha256="e" * 64)
    problems = " | ".join(launch_flags.full_parameter_problems(args))
    assert "no --rl-placement fixed-partition with --rl-sample-role trainer" in problems
    assert "--rl-max-policy-age 0" not in problems  # the buffer trainer is the fully_async trainer


def test_the_island_entry_no_longer_refuses_a_full_parameter_buffer_trainer():
    import inspect

    source = inspect.getsource(island_entry.main)
    assert "if full and buffer_section is not None:" not in source
    assert "if pools is not None and buffer_section is not None:" in source  # no rollout cards to split


# ---------------------------------------------------------------------------- the trainer
@pytest.fixture
def verl_trainer(monkeypatch):
    """``yeto.rl.adapters.verl.trainer`` with verl's modules stubbed (it needs them at
    import only: the sync trainer's base class and the advantage registry)."""
    for name in ("verl", "verl.trainer", "verl.trainer.ppo", "verl.trainer.ppo.v1"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    base = types.ModuleType("verl.trainer.ppo.v1.trainer_base")
    base.register_trainer = lambda name: (lambda cls: cls)
    sync = types.ModuleType("verl.trainer.ppo.v1.trainer_sync")
    sync.PPOTrainerSync = object
    core = types.ModuleType("verl.trainer.ppo.core_algos")
    core.register_adv_est = lambda name: (lambda fn: fn)
    sys.modules["verl.trainer.ppo"].core_algos = core
    for module in (base, sync, core):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in ("yeto.rl.adapters.verl.trainer", "yeto.rl.adapters.verl.advantage"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("yeto.rl.adapters.verl.trainer")
    yield module
    for name in ("yeto.rl.adapters.verl.trainer", "yeto.rl.adapters.verl.advantage"):
        sys.modules.pop(name, None)


class FullBufferActor:
    """Stands in for YetoBufferTrainer on a full-parameter island: one FSDP2 actor
    worker behind ``yeto_call_workers`` (the stream and the HF writer read it), the
    buffer rows it keeps, and an optimizer step per ``yeto_train_step``."""

    def __init__(self, seed: int = 0):
        self.model = _Model(seed=seed)
        self.worker = _Island(self.model).worker
        self.param_version = 0
        self.kept: list = []
        self.assembled: list[list[str]] = []
        self.calls: list[tuple[str, str]] = []
        self.trained: dict[int, dict[str, torch.Tensor]] = {}

    def __call__(self, name, *args, timeout=None):
        if name in ("yeto_call_workers", "yeto_call_workers_shared"):
            fn, *rest = args
            self.calls.append((name, fn.__name__))
            return [fn(self.worker, *rest)]
        return getattr(self, name)(*args)

    def yeto_configure(self, *_):
        return None

    def yeto_add_group(self, rows):
        self.kept.append(rows)
        return len(self.kept)

    def yeto_assemble(self):
        self.assembled.append([r["non_tensor"]["uid"][0] for r in self.kept])
        n, self.kept = len(self.kept), []
        return {"samples": n, "queue_size": None}

    def yeto_train_step(self):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.add_(_step(0, self.param_version)[name])
        self.param_version += 1
        self.trained[self.param_version] = self.model.values()
        return {"metrics": {"actor/grad_norm": 0.5, "actor/pg_loss": 0.1, "actor/lr": 1e-6},
                "criteria": {}, "param_version": self.param_version}

    def yeto_param_version(self):
        return self.param_version


def _full_base(tmp_path: Path, model: _Model) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    save_file({k: v.to(torch.bfloat16) for k, v in model.values().items()}, str(base / "model.safetensors"))
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}))
    (base / "tokenizer.json").write_text("{}")
    return base


def _plan(tmp_path: Path, base: Path, *, rounds: int, groups: int) -> dict:
    from yeto.provenance import file_sha256

    return {"learner_id": 0, "sync": "none", "parameter_mode": "full", "max_policy_age": 1,
            "global_rounds": rounds, "groups_per_round": groups, "samples_per_group": 2,
            "model_path": str(base), "model_revision": "a" * 40,
            "model_config_sha256": file_sha256(base / "config.json"), "out_dir": str(tmp_path / "out"),
            "event_tape": str(tmp_path / "tape.jsonl"), "thresholds_key": ["verl", "fsdp2", "vllm-x", "h200"],
            "tis_upper": 2.0, "compat_group": "nvidia-h200", "algorithm_spec": TOLERANT.to_dict(),
            "asserted": {}, "lora_rank": 0, "lora_targets": None}


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_a_sampler_island_and_a_full_parameter_buffer_trainer_run_three_rounds(tmp_path, verl_trainer):
    rounds, groups = 3, 2
    actor = FullBufferActor(seed=0)
    base = _full_base(tmp_path, actor.model)
    initial = actor.model.values()
    store_dir = tmp_path / "store"
    store = CheckpointStore(store_dir)
    sampler_events, done = [], threading.Event()
    config = EngineSamplerConfig(sampler_id="island0", num_rollouts=40, max_backlog_groups=2,
                                 sampling=SamplingParams(max_tokens=6, n=2), max_prompt_tokens=32, poll_s=0.01)
    engine = FakeEngine(max_model_len=64)
    sampler = build_engine_sampler(
        config, engine=engine, releases=PolicyReleaseStore(store), buffer=FileSampleBuffer(store, batch_size=1),
        emit=lambda n, **f: sampler_events.append((n, f)),
        prompts=lambda r: [PromptItem(f"p{r}", f"prompt {r}", label="x")],
        reward=lambda item, c: float(len(c.token_ids) % 2),
        checks=PostLoadChecks(expected_expert_matrices=None, fixed_prompts=(), probe_pairs=(("Q", " 7"),)),
        should_stop=done.is_set, sleep=lambda s: time.sleep(0.005))
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    plan = _plan(tmp_path, base, rounds=rounds, groups=groups)
    islands = []

    def ports(island):  # buffer_trainer_runner.YetoBufferTaskRunner.run's closure
        islands.append(island)
        return install_buffer_ports(island, buffer=FileSampleBuffer(store, batch_size=1, create_index=False),
                                    releases=PolicyReleaseStore(store), shape=bf.RowShape(64, 16, 0),
                                    base_dir=base, tokenize=lambda text: [7, 8], poll_s=0.01, wait_s=30,
                                    weights=release_weights_for(island, plan))

    try:
        result = verl_trainer.run_fully_async_island(actor, None, plan, {}, buffer_ports=ports)
    finally:
        done.set()
        thread.join(timeout=10)
    island = islands[0]
    assert result["rounds_completed"] == rounds and result["final_policy_version"] == rounds
    assert island.vmap.to_dict() == {str(v): v for v in range(rounds + 1)}
    # the full-parameter pieces: the stream-facing group saw every trained round
    assert island.full_group.trained_rounds == rounds and sorted(island.full_group.receipts) == list(range(rounds))
    assert ("yeto_call_workers", "remote_manifest") in actor.calls
    # versions: v0 (the base checkpoint) is not released; v1..v3 are, in order
    releases = _lines(tmp_path / "out" / RELEASE_LOG.format(learner=0))
    assert [(r["policy_version"], r["released"]) for r in releases] == [(0, False), (1, True), (2, True), (3, True)]
    released = PolicyReleaseStore(store)
    latest = released.latest()
    assert (latest.version, latest.weights) == (rounds, "full")
    # each release holds the trainer's weights at that version in the model's dtype
    landed = load_file(str(store_dir / released.release_dir / latest.path / "model.safetensors"))
    assert sorted(landed) == sorted(SHAPES)
    for name, value in actor.trained[rounds].items():
        assert landed[name].dtype == torch.bfloat16 and torch.equal(landed[name], value.to(torch.bfloat16))
    assert not torch.equal(landed["model.norm.weight"], initial["model.norm.weight"].to(torch.bfloat16))
    assert sorted(p.name for p in (store_dir / released.release_dir / latest.path).iterdir()) == [
        "RELEASE.json", "config.json", "model.safetensors", "tokenizer.json"]
    # the sampler loaded the releases in order
    loaded = [f["policy_version"] for n, f in sampler_events if n == "rl_sampler_policy_loaded"]
    assert loaded == sorted(set(loaded)) and set(loaded) <= {1, 2, 3}
    # no group consumed twice; the trained groups are those handed to the actor
    consumed = _lines(tmp_path / "out" / CONSUMED_LOG.format(learner=0))
    positions = [r["position"] for r in consumed]
    assert positions == sorted(set(positions)) == list(range(len(positions)))
    kept = [r for r in consumed if r["kept"]]
    assert len(kept) == rounds * groups and [len(a) for a in actor.assembled] == [groups] * rounds
    events = _lines(tmp_path / "tape.jsonl")
    start = next(e for e in events if e["event"] == "rl_verl_island_start")
    assert start["parameter_mode"] == "full" and start["sample_role"] == "trainer"
    assert start["layout_hash"] == island.layout_hash and len(island.layout_hash) == 64
    assert events[-1]["event"] == "rl_learner_finalized"


def test_release_weights_follow_the_parameter_mode():
    lora = types.SimpleNamespace(full_group=None)
    assert release_weights_for(lora, {"parameter_mode": "lora"}) is None
    with pytest.raises(RuntimeError, match="after its policy state"):
        release_weights_for(lora, {"parameter_mode": "full", "model_path": "/m"})


def test_a_full_parameter_fully_async_island_with_a_rollouter_is_wired(verl_trainer):
    # S27 G-EV (VERL-PARITY #11): the refusal is gone; the rollouter path runs in
    # test_s27_verl_fully_async_rollouter.py
    import inspect

    assert "with a rollouter is not wired" not in inspect.getsource(verl_trainer)

def test_verify_every_is_wired_on_a_buffer_trainer_and_unverified_releases_say_so(tmp_path, verl_trainer):
    """S28 G-120C: --rl-verify-every N reached the plan of a buffer trainer but not its
    driver (only the synchronous and the Megatron-engine drivers wired it), so every
    round hashed the whole trainer. Now the fully_async driver takes it; the rounds it
    does not verify publish an identity label, and their releases say so."""
    rounds, groups = 3, 2
    actor = FullBufferActor(seed=0)
    base = _full_base(tmp_path, actor.model)
    store_dir = tmp_path / "store"
    store = CheckpointStore(store_dir)
    done = threading.Event()
    config = EngineSamplerConfig(sampler_id="island0", num_rollouts=40, max_backlog_groups=2,
                                 sampling=SamplingParams(max_tokens=6, n=2), max_prompt_tokens=32, poll_s=0.01)
    sampler = build_engine_sampler(
        config, engine=FakeEngine(max_model_len=64), releases=PolicyReleaseStore(store),
        buffer=FileSampleBuffer(store, batch_size=1), emit=lambda n, **f: None,
        prompts=lambda r: [PromptItem(f"p{r}", f"prompt {r}", label="x")],
        reward=lambda item, c: float(len(c.token_ids) % 2),
        checks=PostLoadChecks(expected_expert_matrices=None, fixed_prompts=(), probe_pairs=(("Q", " 7"),)),
        should_stop=done.is_set, sleep=lambda s: time.sleep(0.005))
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    plan = {**_plan(tmp_path, base, rounds=rounds, groups=groups), "verify_every": 1000}

    def ports(island):
        return install_buffer_ports(island, buffer=FileSampleBuffer(store, batch_size=1, create_index=False),
                                    releases=PolicyReleaseStore(store), shape=bf.RowShape(64, 16, 0),
                                    base_dir=base, tokenize=lambda text: [7, 8], poll_s=0.01, wait_s=30,
                                    weights=release_weights_for(island, plan))

    try:
        result = verl_trainer.run_fully_async_island(actor, None, plan, {}, buffer_ports=ports)
    finally:
        done.set()
        thread.join(timeout=10)
    assert result["rounds_completed"] == rounds
    tape = _lines(tmp_path / "tape.jsonl")
    wired = [{k: e[k] for k in ("planned", "driver_verify_every", "policy_state", "supports_unverified_rounds")}
             for e in tape if e["event"] == "rl_verify_every_wired"]
    assert wired == [{"planned": 1000, "driver_verify_every": 1000,
                      "policy_state": "VerlFullParameterLocalState", "supports_unverified_rounds": True}]
    releases = _lines(tmp_path / "out" / RELEASE_LOG.format(learner=0))
    # v0 (first) and v3 (the last round) are verified; v1 and v2 are not (N = 1000)
    assert [(r["policy_version"], r["policy_hash_verified"]) for r in releases] == [
        (0, True), (1, False), (2, False), (3, True)]
    store_releases = PolicyReleaseStore(store)
    latest = store_releases.latest()
    assert latest.version == rounds and latest.origin is None  # a verified release: record unchanged
    # the release record of an unverified round says its policy hash is an identity label
    v2 = next(r for r in releases if r["policy_version"] == 2)  # the store keeps the newest two
    record = json.loads((store_dir / store_releases.release_dir / v2["path"] / "RELEASE.json").read_text())
    assert record["origin"] == {"kind": "single", "policy_hash": "unverified-identity"}


def test_without_verify_every_every_release_is_verified(tmp_path, verl_trainer):
    """The default path is unchanged: no wiring event, every release verified."""
    rounds, groups = 2, 2
    actor = FullBufferActor(seed=1)
    base = _full_base(tmp_path, actor.model)
    store = CheckpointStore(tmp_path / "store")
    done = threading.Event()
    config = EngineSamplerConfig(sampler_id="island0", num_rollouts=40, max_backlog_groups=2,
                                 sampling=SamplingParams(max_tokens=6, n=2), max_prompt_tokens=32, poll_s=0.01)
    sampler = build_engine_sampler(
        config, engine=FakeEngine(max_model_len=64), releases=PolicyReleaseStore(store),
        buffer=FileSampleBuffer(store, batch_size=1), emit=lambda n, **f: None,
        prompts=lambda r: [PromptItem(f"p{r}", f"prompt {r}", label="x")],
        reward=lambda item, c: float(len(c.token_ids) % 2),
        checks=PostLoadChecks(expected_expert_matrices=None, fixed_prompts=(), probe_pairs=(("Q", " 7"),)),
        should_stop=done.is_set, sleep=lambda s: time.sleep(0.005))
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    plan = _plan(tmp_path, base, rounds=rounds, groups=groups)

    def ports(island):
        return install_buffer_ports(island, buffer=FileSampleBuffer(store, batch_size=1, create_index=False),
                                    releases=PolicyReleaseStore(store), shape=bf.RowShape(64, 16, 0),
                                    base_dir=base, tokenize=lambda text: [7, 8], poll_s=0.01, wait_s=30,
                                    weights=release_weights_for(island, plan))

    try:
        verl_trainer.run_fully_async_island(actor, None, plan, {}, buffer_ports=ports)
    finally:
        done.set()
        thread.join(timeout=10)
    assert not [e for e in _lines(tmp_path / "tape.jsonl") if e["event"] == "rl_verify_every_wired"]
    releases = _lines(tmp_path / "out" / RELEASE_LOG.format(learner=0))
    assert all(r["policy_hash_verified"] for r in releases)
