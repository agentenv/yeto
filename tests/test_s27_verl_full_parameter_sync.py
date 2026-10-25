"""S27 B10 phase 2 (G-DB): the verl binding of the full-parameter role stream.

CPU only.  The verl actor workers are plain ``torch`` modules behind the real
worker functions of ``yeto.rl.adapters.verl.full_parameter`` (no DTensor, no
verl, no Ray).  The last tests run two verl islands through the launcher-built
Rust syncer (needs cargo): strict sweeps and asynchronous rounds with local
init, both checked against the exact average merge.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import socket
import subprocess
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.protocol import MSG_BCAST_FRAGMENT
from yeto.rl.adapters.verl import full_parameter as fp
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.engine.full_parameter_stream import RoleStreamConfig
from yeto.rl.local_learner import ComponentIdentity

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
COMPONENT = ComponentIdentity(role="actor", model_revision="a" * 40, config_hash="b" * 64)
SHAPES = {"model.embed_tokens.weight": (12, 4), "model.layers.0.mlp.weight": (4, 6),
          "model.layers.0.norm.weight": (4,), "model.norm.weight": (4,)}
# 16 floats per fragment: the 12x4 embedding (48 floats) is split into row slices.
MAX_FRAGMENT_BYTES = 64


# ----------------------------------------------------------------------------- fake verl island
class _Model(torch.nn.Module):
    """The unwrapped training module of one verl actor worker (FP32 masters)."""

    def __init__(self, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        for name, shape in SHAPES.items():
            module = self
            *path, leaf = name.split(".")
            for part in path:
                if not hasattr(module, part):
                    module.add_module(part, torch.nn.Module())
                module = getattr(module, part)
            module.register_parameter(leaf, torch.nn.Parameter(torch.randn(shape, generator=generator)))

    def values(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters()}


class _Island:
    """``VerlIsland.call_workers`` over one in-process worker."""

    def __init__(self, model: _Model, *, drop_applies: bool = False):
        self.model = model
        self.worker = SimpleNamespace(actor=SimpleNamespace(engine=SimpleNamespace(module=model, optimizer=None)))
        self.drop_applies = drop_applies
        self.applied: list[list[str]] = []

    def call_workers(self, fn, *args):
        if fn is fp.remote_apply_fragment:
            self.applied.append(sorted(args[0]))
            if self.drop_applies:
                return [{"applied": 0}]
        return [fn(self.worker, *args)]

    call_workers_shared = call_workers


def _config(**overrides) -> RoleStreamConfig:
    layout = _layout(minimum_fragments=overrides.pop("minimum_fragments", 1))
    values = dict(
        algorithm="grpo", role="actor", component=COMPONENT, syncer_addr=("127.0.0.1", 30100),
        learner_id=0, learner_generation=0, learner_generations=(0, 0),
        total_fragment_steps=layout.fragments.num_fragments * 2,
        expected_fragments=layout.fragments.num_fragments, expected_layout_hash=layout.layout_hash,
        local_horizon=1, optimizer_steps_per_round=1, training_contract_hash="2" * 64,
        syncer_profile_hash="3" * 64, pipeline_depth=1, algorithm_spec_sha256="a1" * 32,
        max_fragment_bytes=MAX_FRAGMENT_BYTES, max_chunk_bytes=8)
    values.update(overrides)
    return RoleStreamConfig(**values)


def _layout(minimum_fragments: int = 1, matrix_merge: str = "avg"):
    return fp.plan_layout(fp.island_specs(SHAPES), component=COMPONENT, minimum_fragments=minimum_fragments,
                          max_fragment_bytes=MAX_FRAGMENT_BYTES, matrix_merge=matrix_merge)


def _init(island: _Island, config: RoleStreamConfig | None = None):
    group = vfs.VerlFullParameterGroup(island)
    adapter, cut = vfs.asyncio.run(vfs.VerlChunkedFullParameterAdapter.initialize(group, config=config or _config()))
    return group, adapter, cut


def _joined(parts) -> bytes:
    return b"".join(bytes(part) for part in parts)


# ----------------------------------------------------------------------------- the adapter
def test_a_cut_covers_every_parameter_and_round_trips_split_rows():
    island = _Island(_Model())
    _group, adapter, cut = _init(island)
    assert adapter.layout.has_split_parameters
    assert cut.local_step_generation == 0 and cut.policy_version == 0
    assert sum(flat.numel() for flat in cut.fragments) == sum(t.numel() for t in island.model.values().values())
    back = adapter.unflatten(cut, list(SHAPES))
    for name, value in island.model.values().items():
        assert torch.equal(back[name], value), name
    for fragment_id, flat in enumerate(cut.fragments):
        # The INIT bytes are the fragment's f32 payload, chunked by max_chunk_bytes.
        parts = list(adapter.fragment_parts(cut, fragment_id)())
        assert all(len(bytes(part)) <= 8 for part in parts)
        assert _joined(parts) == flat.numpy().tobytes()
        assert cut.fragment_hashes[fragment_id] == hashlib.sha256(flat.numpy().tobytes()).hexdigest()


def test_the_layout_probe_refusal_names_the_observed_layout(capsys):
    with pytest.raises(RuntimeError, match="differs from the frozen probe"):
        _init(_Island(_Model()), _config(expected_layout_hash="f" * 64))
    line = next(line for line in capsys.readouterr().out.splitlines()
                if line.startswith(vfs.LAYOUT_PROBE_LOG_PREFIX))
    assert _layout().layout_hash in line


def test_a_push_is_local_minus_anchor_and_needs_a_trained_cut():
    island = _Island(_Model())
    group, adapter, initial = _init(island)
    sink = adapter.fragment_sink(initial)
    anchor = sink.adopt(initial, 1, 0, initial.fragment_hashes[1])
    with torch.no_grad():
        for parameter in island.model.parameters():
            parameter.add_(0.25)
    local = vfs.asyncio.run(adapter.capture(group, policy_version=0, local_step_generation=1))
    delta = torch.frombuffer(bytearray(_joined(adapter.delta_parts_from_authoritative(anchor, local, 1)())),
                             dtype=torch.float32)
    assert torch.allclose(delta, torch.full_like(delta, 0.25))
    with pytest.raises(ValueError, match="local horizon"):
        adapter.delta_parts_from_authoritative(anchor, initial, 1)
    with pytest.raises(ValueError, match="provenance"):
        adapter.delta_parts_from_authoritative(anchor, local, 0)


def test_an_apply_writes_only_replaced_fragments_and_proves_them():
    island = _Island(_Model())
    group, adapter, current = _init(island)
    sink = adapter.fragment_sink(current)
    new = torch.arange(adapter.fragment_numel(2), dtype=torch.float32)
    stored = sink(2, 3, new.numpy().tobytes())
    target = adapter.assemble_mixed_target(current, target_policy_version=1, authoritative_fragments=[stored])
    assert target.fragments[0] is current.fragments[0] and target.fragments[2] is stored.flat
    vfs.asyncio.run(adapter.apply(group, target))
    assert island.applied == [sorted(fp.fragment_tensor_names(adapter.layout, 2))]
    after = vfs.asyncio.run(adapter.capture(group, policy_version=1, local_step_generation=0))
    assert after.fragment_hashes == target.fragment_hashes
    # Nothing replaced: nothing written.
    vfs.asyncio.run(adapter.apply(group, adapter.assemble_mixed_target(
        after, target_policy_version=2, authoritative_fragments=[])))
    assert len(island.applied) == 1


def test_an_apply_that_does_not_land_is_refused():
    island = _Island(_Model(), drop_applies=True)
    group, adapter, current = _init(island)
    stored = adapter.fragment_sink(current)(0, 1, torch.zeros(adapter.fragment_numel(0)).numpy().tobytes())
    target = adapter.assemble_mixed_target(current, target_policy_version=1, authoritative_fragments=[stored])
    with pytest.raises(RuntimeError, match="did not land"):
        vfs.asyncio.run(adapter.apply(group, target))
    assert adapter.resident is None  # unknown after a failed apply: the next one writes everything


def test_version_zero_must_be_this_islands_checkpoint():
    island = _Island(_Model())
    _group, adapter, cut = _init(island)
    sink = adapter.fragment_sink(cut)
    own = [sink(i, 0, cut.fragments[i].numpy().tobytes()) for i in range(len(cut.fragments))]
    adapter.verify_initial_fragments(cut, own)
    other = _init(_Island(_Model(seed=1)))[2]
    foreign = [vfs.VerlStoredFragment(i, 0, other.fragment_hashes[i], other.fragments[i])
               for i in range(len(other.fragments))]
    with pytest.raises(RuntimeError, match="differs from the checkpoint"):
        adapter.verify_initial_fragments(cut, foreign)
    with pytest.raises(RuntimeError, match="incomplete"):
        adapter.verify_initial_fragments(cut, own[:-1])


def test_the_inbound_sink_streams_out_of_order_chunks_and_keeps_replays():
    island = _Island(_Model())
    _group, adapter, cut = _init(island)
    inbound = vfs.VerlInboundChunkSink(adapter.fragment_sink(cut))
    values = torch.linspace(-1, 1, adapter.fragment_numel(1)).numpy().tobytes()

    def deliver(raw: bytes):
        transaction = inbound.begin_message(7, len(raw) + 32)
        inbound.bind_fragment(transaction, MSG_BCAST_FRAGMENT, 1, 4, len(raw))
        token = inbound.stage_chunk(transaction, raw[8:])
        inbound.consume_chunk(transaction, 0, raw[:8])
        inbound.consume_staged_chunk(transaction, 8, token, len(raw) - 8)
        return inbound.finish_message(transaction)

    first = deliver(values)
    assert first.data.flat.numpy().tobytes() == values and first.discard is not None
    assert first.payload_hash == hashlib.sha256(values).digest()
    again = deliver(values)
    assert again.data is first.data and again.discard is None
    with pytest.raises(RuntimeError, match="replay changed bytes"):
        deliver(bytes(len(values)))
    partial = inbound.begin_message(8, 64)
    inbound.bind_fragment(partial, MSG_BCAST_FRAGMENT, 1, 5, len(values))
    inbound.consume_chunk(partial, 0, values[:8])
    with pytest.raises(RuntimeError, match="incomplete"):
        inbound.finish_message(partial)
    with pytest.raises(ValueError, match="byte count"):
        inbound.bind_fragment(inbound.begin_message(9, 64), MSG_BCAST_FRAGMENT, 1, 6, 4)


@pytest.mark.parametrize("option, value", [("overlap", True), ("expert_token_weights", True),
                                           ("peer_recovery", object())])
def test_options_a_verl_island_does_not_offer_are_refused(option, value):
    config = _config()
    forced = SimpleNamespace(**{**{f: getattr(config, f) for f in config.__dataclass_fields__}, option: value})
    with pytest.raises(ValueError, match="verl 全参数岛不支持"):
        vfs.refuse_unsupported(forced)


# ----------------------------------------------------------------------------- the island plan
def _island_args(**overrides):
    values = dict(model="Qwen/Qwen3-0.6B", model_revision="C" * 40, reward_function="gsm8k_reward:score",
                  global_rounds=3, fragments=2, total_fragment_steps=6, learner_id=1, num_learners=2,
                  learner_generation=0, parameter_layout_sha256="e" * 64, syncer_profile_sha256="d" * 64,
                  full_parameter_rounds="sweep", init_mode="auto", sweep_push_dtype="f32",
                  sweep_fragment_max_bytes=2 << 30, sweep_wait_timeout_s=900.0, sweep_final_cut="download",
                  pipeline=1, local_horizon=1, rl_sync_matrix_merge="avg", wan_streams=2,
                  rl_sync_island_lease_s=None, rl_syncer_epoch=0)
    values.update(overrides)
    return argparse.Namespace(**values)


def _run_dict(**overrides) -> dict:
    from yeto.rl.adapters.verl import config as vconf

    run = vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o",
                              parameter_mode="full", lora_rank=0, lora_alpha=0)
    return replace(run, **overrides).to_dict()


@pytest.fixture
def model_dir(tmp_path) -> Path:
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    return tmp_path


def test_the_plan_section_binds_the_model_config_and_the_training_contract(model_dir):
    section = vfs.plan_section(_island_args(), _run_dict(), model_path=str(model_dir), sync="strict",
                               algorithm_spec_sha256="a" * 64)
    full = section["full_parameter"]
    assert section["model_config_sha256"] == hashlib.sha256(b'{"model_type": "qwen3"}').hexdigest()
    assert full["async_rounds"] is False and full["local_init"] is False and full["island_pool"] is None
    config = vfs.role_stream_config(full, syncer="10.0.0.1:29400", learner_id=1, model_revision="C" * 40)
    assert config.learner_generations == (0, 0) and config.total_fragment_steps == 6
    assert config.component.model_revision == "c" * 40 and config.partial_quorum is False
    # The card count, the seed and the vLLM memory share stay out of the contract.
    for local in (dict(n_gpus=4), dict(seed=7), dict(gpu_memory_utilization=0.3), dict(out_dir="/x")):
        other = vfs.plan_section(_island_args(), _run_dict(**local), model_path=str(model_dir),
                                 sync="strict", algorithm_spec_sha256="a" * 64)
        assert other["full_parameter"]["training_contract_sha256"] == full["training_contract_sha256"]
    lr = vfs.plan_section(_island_args(), _run_dict(lr=2e-6), model_path=str(model_dir), sync="strict",
                          algorithm_spec_sha256="a" * 64)
    assert lr["full_parameter"]["training_contract_sha256"] != full["training_contract_sha256"]


def test_async_rounds_default_to_local_init_and_a_pool_carries_its_lease(model_dir):
    section = vfs.plan_section(_island_args(full_parameter_rounds="async", local_horizon=2, pipeline=2,
                                            rl_sync_island_lease_s=60.0, rl_syncer_epoch=3),
                               _run_dict(), model_path=str(model_dir), sync="pool", algorithm_spec_sha256="a" * 64)
    full = section["full_parameter"]
    assert full["local_init"] is True and full["island_pool"] == {"lease_s": 60.0, "syncer_epoch": 3}
    config = vfs.role_stream_config(full, syncer="h:1", learner_id=1, model_revision="c" * 40)
    assert config.partial_quorum and config.island_lease_s == 60.0 and config.syncer_epoch == 3
    assert config.local_horizon == 2 and config.pipeline_depth == 2


@pytest.mark.parametrize("overrides, sync, match", [
    (dict(total_fragment_steps=5), "strict", "fragment budget"),
    (dict(parameter_layout_sha256=None), "strict", "parameter-layout-sha256"),
    (dict(), "pool", "needs --full-parameter-rounds async"),
    (dict(local_horizon=2), "strict", "local-horizon 1"),
    (dict(learner_id=2), "strict", "below --num-learners"),
])
def test_the_plan_section_refuses_what_the_island_cannot_run(model_dir, overrides, sync, match):
    with pytest.raises(ValueError, match=match):
        vfs.plan_section(_island_args(**overrides), _run_dict(), model_path=str(model_dir), sync=sync,
                         algorithm_spec_sha256="a" * 64)


def test_a_no_sync_island_carries_only_the_model_identity(model_dir):
    section = vfs.plan_section(_island_args(parameter_layout_sha256=None), _run_dict(),
                               model_path=str(model_dir), sync="none", algorithm_spec_sha256="a" * 64)
    assert set(section) == {"model_config_sha256"}


def test_the_no_sync_policy_state_exports_a_resident_whole_cut():
    island = _Island(_Model())
    state = vfs.VerlFullParameterLocalState(vfs.VerlFullParameterGroup(island), component=COMPONENT,
                                            max_fragment_bytes=MAX_FRAGMENT_BYTES)
    first = state.export(policy_version=3)
    assert first.policy_version == 3 and first.layout_hash == state.layout_hash
    assert first.digest.payload_bytes == 4 * sum(t.numel() for t in island.model.values().values())
    with torch.no_grad():
        island.model.model.norm.weight.add_(1.0)
    assert state.export(policy_version=3).policy_tensor_hash() != first.policy_tensor_hash()
    with pytest.raises(RuntimeError, match="never applies"):
        state.apply(first, optimizer="preserve", local_step=0)


def test_full_parameter_identity_matches_the_miles_island_rule():
    from yeto.rl.adapters.miles.island_entry import _full_parameter_identity

    assert vfs.full_parameter_identity("9" * 64) == _full_parameter_identity("9" * 64)


# ----------------------------------------------------------------------------- two islands, real syncer
needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")
TOKENS = 100  # trained tokens per round, equal on both islands (equal merge weights)


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


class _Driver:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


def _start_syncer(binary: Path, tmp_path: Path, *, fragments: int, rounds: int, extra: tuple[str, ...]):
    from yeto import launcher
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", str(fragments), "--total-steps", str(rounds), *extra),
        gpu="modal:2xh100,modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    command = launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = (command.replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return process, port, launcher.rl_syncer_profile(args, 2).sha256


def _plan(learner_id: int, port: int, profile: str, *, rounds: int, fragments: int, async_rounds: bool,
          matrix_merge: str, pipeline: int, final_cut: str) -> dict:
    layout = _layout(minimum_fragments=fragments, matrix_merge=matrix_merge)
    assert layout.fragments.num_fragments == fragments
    return {"learner_id": learner_id, "syncer": f"127.0.0.1:{port}", "model_revision": "a" * 40,
            vfs.PLAN_KEY: {
                "async_rounds": async_rounds, "num_learners": 2, "learner_generation": 0,
                "fragments": fragments, "total_fragment_steps": rounds * fragments,
                "parameter_layout_sha256": layout.layout_hash, "syncer_profile_sha256": profile,
                "model_config_sha256": "b" * 64, "training_contract_sha256": "2" * 64,
                "algorithm_spec_sha256": "a1" * 32, "local_horizon": 1, "pipeline": pipeline,
                "wait_timeout_s": 60.0, "wan_streams": 1, "push_dtype": "f32",
                "max_fragment_bytes": MAX_FRAGMENT_BYTES, "final_cut": final_cut,
                "local_init": async_rounds, "matrix_merge": matrix_merge, "island_pool": None}}


def _step(learner_id: int, round_id: int) -> dict[str, torch.Tensor]:
    """This island's local training step of one round (deterministic)."""
    generator = torch.Generator().manual_seed(1000 * learner_id + round_id)
    return {name: 0.01 * torch.randn(shape, generator=generator) for name, shape in SHAPES.items()}


def _run_island(learner_id: int, plan: dict, max_rounds: int, results: dict) -> None:
    """Train and sync until the syncer's final cut (a sweep: exactly one round
    per sweep; asynchronous rounds: as many local rounds as the pool needs)."""
    model = _Model(seed=0)  # every island loads the same checkpoint
    island = _Island(model)
    island.full_group = vfs.VerlFullParameterGroup(island)
    sync, policy_state, _stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
    driver = _Driver()
    try:
        state = sync.start(driver).state
        for round_id in range(max_rounds):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.add_(_step(learner_id, round_id)[name])
            ids = (f"l{learner_id}-r{round_id}",)
            batch_hash = hashlib.sha256(ids[0].encode()).hexdigest()
            policy_hash = state.policy_tensor_hash()
            island.full_group.record_round(
                LocalStepReceipt(algorithm="grpo", learner_id=learner_id, learner_generation=0,
                                 base_policy_version=round_id, base_policy_hash=policy_hash,
                                 input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS,
                                 optimizer_steps=1, optimizer_step_succeeded=True,
                                 parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"]),
                vfs.VerlRoundEvidence(round_id, policy_hash, batch_hash, TOKENS, ids))
            boundary = sync.boundary(driver, rollout_id=round_id, stats=None)
            state = boundary.state
            if boundary.stop:
                sync.published(driver, rollout_id=round_id, policy_hash=state.policy_tensor_hash())
                break
        sync.finish(driver)
        results[learner_id] = {"values": model.values(), "events": driver.events, "state": state,
                               "policy_layout": policy_state.layout_hash, "rounds": round_id + 1}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error
    finally:
        sync.close()


def _two_islands(binary, tmp_path, monkeypatch, *, rounds: int, fragments: int, async_rounds: bool,
                 extra: tuple[str, ...], matrix_merge: str, pipeline: int, final_cut: str = "download"):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-full-parameter-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    process, port, profile = _start_syncer(binary, tmp_path, fragments=fragments, rounds=rounds, extra=extra)
    results: dict = {}
    try:
        threads = [threading.Thread(target=_run_island, args=(
            i, _plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=async_rounds,
                     matrix_merge=matrix_merge, pipeline=pipeline, final_cut=final_cut),
            rounds if not async_rounds else 20 * rounds, results)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        assert not isinstance(results.get(learner_id), BaseException), (results.get(learner_id), log[-3000:])
        assert learner_id in results, log[-3000:]
    return results, log


def _expected_average(rounds: int) -> dict[str, torch.Tensor]:
    """Outer lr 1, no momentum, equal weights: v0 + sum over rounds of the islands' mean step."""
    out = _Model(seed=0).values()
    for round_id in range(rounds):
        for name in out:
            out[name] = out[name] + (_step(0, round_id)[name] + _step(1, round_id)[name]) / 2
    return out


@needs_cargo
@pytest.mark.parametrize("final_cut", ["download", "ack"])
def test_two_verl_islands_run_strict_sweeps_through_the_real_syncer(syncer_binary, tmp_path, monkeypatch,
                                                                     final_cut):
    rounds, fragments = 3, len(_layout().fragments.fragments)
    # An explicit sweep is the strict control profile (plain average, every island per round).
    results, _log = _two_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                 async_rounds=False, matrix_merge="avg", pipeline=1, final_cut=final_cut,
                                 extra=("--rl-sync-full-parameter-rounds", "sweep",
                                        "--rl-sync-final-cut", final_cut))
    expected = _expected_average(rounds)
    for learner_id in range(2):
        result = results[learner_id]
        # "ack": the final sweep stays with the syncer, so the islands hold the
        # merge of every round but the last plus their own last step.
        merged = expected if final_cut == "download" else {
            name: value - (_step(1 - learner_id, rounds - 1)[name] - _step(learner_id, rounds - 1)[name]) / 2
            for name, value in expected.items()}
        for name, value in merged.items():
            assert torch.allclose(result["values"][name], value, atol=1e-6), (learner_id, name)
        merges = [fields for event, fields in result["events"] if event == "rl_full_parameter_round"]
        applied = rounds if final_cut == "download" else rounds - 1  # "ack": the last merge stays in the syncer
        assert [m["sync/fragment_versions"] for m in merges][-1] == list(
            range((applied - 1) * fragments + 1, applied * fragments + 1))
        assert len(merges) == rounds == result["rounds"]
        acks = [fields for event, fields in result["events"] if event == "rl_full_parameter_final_ack"]
        assert acks == ([{"acknowledged": True}] if final_cut == "download"
                        else [{"acknowledged": True, "sync/final_cut": "ack"}])
    if final_cut == "download":
        assert results[0]["state"].policy_tensor_hash() == results[1]["state"].policy_tensor_hash()


@needs_cargo
def test_two_verl_islands_run_asynchronous_rounds_with_local_init(syncer_binary, tmp_path, monkeypatch):
    rounds, fragments = 3, len(_layout().fragments.fragments)
    # The S27 async defaults of the launcher (quorum 1, two rounds in flight, RDA
    # matrices, local init); the final cut is downloaded so both islands can be compared.
    results, log = _two_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                async_rounds=True, extra=("--rl-sync-full-parameter-rounds", "async",
                                                          "--rl-sync-final-cut", "download"),
                                matrix_merge="rda", pipeline=2)
    for learner_id in range(2):
        events = results[learner_id]["events"]
        acks = [fields for event, fields in events if event == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}], (learner_id, log[-2000:])
        merges = [fields for event, fields in events if event == "rl_full_parameter_round"]
        assert merges and merges[-1]["sync/fragment_versions"] == list(
            range((rounds - 1) * fragments + 1, rounds * fragments + 1))
    # With a partial quorum the merge order depends on timing; the final cut is
    # the syncer's and both islands downloaded the same one.
    for name in SHAPES:
        assert torch.equal(results[0]["values"][name], results[1]["values"][name]), name
