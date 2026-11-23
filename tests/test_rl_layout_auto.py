"""``--rl-parameter-layout-sha256 auto``: the syncer, not a frozen probe hash, binds the layout.

Covered here:

1. the launcher takes ``auto`` (opt-in) for the Miles full-parameter preset,
   prints where the layout is bound, passes ``auto`` to the islands, and keeps
   refusing a missing hash; the verl backend refuses ``auto`` (it plans its
   layout offline);
2. the Miles island accepts ``auto`` and builds its role stream without an
   expected layout hash;
3. real Rust syncer, engine-neutral role streams over two CPU islands:
   an ``auto`` island and an island pinned to the hash the ``auto`` island
   recorded share one session and finish the run (a recorded hash can be
   pinned later); an ``auto`` island with another layout is refused at its
   HELLO, and printed its observed layout before the HELLO.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import threading
import time

import pytest
import torch

from process_groups import kill_group
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_verl_full_parameter_sync import (
    SHAPES,
    TOKENS,
    _Driver,
    _Island,
    _layout,
    _plan,
    _start_syncer,
    local_rounds,
    needs_cargo,
    syncer_binary,  # noqa: F401 - fixture
)
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.engine.full_parameter_stream import LAYOUT_AUTO_LOG_PREFIX, LAYOUT_BOUND_EVENT
from yeto.rl.engine.selection import PARAMETER_LAYOUT_AUTO

TWO = "modal:2xh100,modal:2xh100"
FULL = ("--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
        "--fragments", "4", "--rl-sync-full-parameter-rounds", "async", "--local-rl-rounds-per-sync", "2")


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "layout-auto-test-key")


def _prepared(*extra: str):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (*FULL, *extra), gpu=TWO)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


# ------------------------------------------------------------------ 1. launcher
def test_the_launcher_takes_auto_and_passes_it_to_the_islands(capsys):
    from yeto.launcher import dry_run_plan

    args = _prepared("--rl-parameter-layout-sha256", PARAMETER_LAYOUT_AUTO)
    assert "layout hash: bound by the syncer at first HELLO; resume verifies it" in capsys.readouterr().out
    plan = dry_run_plan(args)
    for island in plan["island_requests"]:
        assert "--parameter-layout-sha256 auto" in island["learner_command"]


def test_an_explicit_hash_is_unchanged_and_a_missing_one_is_still_refused(capsys):
    args = _prepared("--rl-parameter-layout-sha256", "e" * 64)
    assert "bound by the syncer" not in capsys.readouterr().out
    assert args.rl_parameter_layout_sha256 == "e" * 64
    with pytest.raises(ValueError, match=r"--rl-parameter-layout-sha256 <64 lowercase hex> \(or auto\)"):
        _prepared()


def test_the_verl_backend_refuses_auto():
    with pytest.raises(ValueError, match="auto is Miles-only"):
        _prepared("--rl-backend", "verl", "--rl-parameter-layout-sha256", PARAMETER_LAYOUT_AUTO)


# ------------------------------------------------------------------ 2. Miles island
def test_the_miles_island_builds_its_role_stream_without_an_expected_hash():
    from yeto.launcher import dry_run_plan
    from yeto.rl.adapters.miles import island_entry

    args = _prepared("--rl-parameter-layout-sha256", PARAMETER_LAYOUT_AUTO)
    command = dry_run_plan(args)["island_requests"][0]["learner_command"]
    tail = command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    tokens = [{"$LEARNER_ID": "0", "$SYNCER_ADDR": "127.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
    parsed = island_entry.parse_args(tokens)
    assert parsed.parameter_layout_sha256 == PARAMETER_LAYOUT_AUTO
    island_entry._check_dense_full_sweep(parsed, yeto_policy_sync=True)
    config = island_entry._dense_full_sync_config(
        parsed, model_config_sha256="b" * 64, training_contract_sha256="2" * 64,
        algorithm_spec_sha256="a1" * 32).by_role()["actor"]
    assert config.expected_layout_hash is None
    assert config.expected_fragments == 4


# ------------------------------------------------------------------ 3. real syncer
RENAMED = {("model.norm2.weight" if name == "model.norm.weight" else name): shape
           for name, shape in SHAPES.items()}


class _ShapedModel(torch.nn.Module):
    """``test_verl_full_parameter_sync._Model`` with its own parameter names."""

    def __init__(self, shapes: dict[str, tuple[int, ...]]):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        for name, shape in shapes.items():
            module = self
            *path, leaf = name.split(".")
            for part in path:
                if not hasattr(module, part):
                    module.add_module(part, torch.nn.Module())
                module = getattr(module, part)
            module.register_parameter(leaf, torch.nn.Parameter(torch.randn(shape, generator=generator)))


def _run_island(learner_id: int, plan: dict, shapes: dict, results: dict) -> None:
    model = _ShapedModel(shapes)
    island = _Island(model)
    island.full_group = vfs.VerlFullParameterGroup(island)
    driver = _Driver()
    sync = None
    try:
        sync, _policy_state, stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
        state = sync.start(driver).state
        layout_hash = stream.adapter.layout.layout_hash
        for round_id in local_rounds(None):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(0.01 * (learner_id + 1))
            ids = (f"l{learner_id}-r{round_id}",)
            batch_hash = hashlib.sha256(ids[0].encode()).hexdigest()
            policy_hash = state.policy_tensor_hash()
            island.full_group.record_round(
                LocalStepReceipt(algorithm="grpo", learner_id=learner_id, learner_generation=0,
                                 base_policy_version=round_id, base_policy_hash=policy_hash,
                                 input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS,
                                 optimizer_steps=1, optimizer_step_succeeded=True,
                                 parameter_layout_hash=layout_hash),
                vfs.VerlRoundEvidence(round_id, policy_hash, batch_hash, TOKENS, ids))
            boundary = sync.boundary(driver, rollout_id=round_id, stats=None)
            state = boundary.state
            if boundary.stop:
                sync.published(driver, rollout_id=round_id, policy_hash=state.policy_tensor_hash())
                break
        sync.finish(driver)
        results[learner_id] = {"events": driver.events, "layout_hash": layout_hash,
                               "values": {n: p.detach().clone() for n, p in model.named_parameters()}}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error
    finally:
        if sync is not None:
            sync.close()


def _island_plan(learner_id: int, port: int, profile: str, fragments: int, *, layout_hash: str | None,
                 wait_timeout_s: float = 30.0) -> dict:
    plan = _plan(learner_id, port, profile, rounds=ROUNDS, fragments=fragments, async_rounds=True,
                 matrix_merge="rda", pipeline=2, final_cut="download")
    plan[vfs.PLAN_KEY]["parameter_layout_sha256"] = layout_hash
    plan[vfs.PLAN_KEY]["wait_timeout_s"] = wait_timeout_s
    return plan


def test_auto_and_the_pinned_hash_send_the_same_hello():
    """The expected hash is an island-side check only: an auto island's HELLO
    (session contract) equals the HELLO of an island pinned to the hash it
    recorded, so a checkpoint written under auto resumes with the pinned hash."""
    from yeto.rl.engine.full_parameter_stream import session_contract_hash

    layout = _layout(minimum_fragments=1, matrix_merge="rda")
    pinned = vfs.role_stream_config(_island_plan(0, 1, "3" * 64, layout.fragments.num_fragments,
                                                 layout_hash=layout.layout_hash)[vfs.PLAN_KEY],
                                    syncer="127.0.0.1:1", learner_id=0, model_revision="a" * 40)
    auto = vfs.role_stream_config(_island_plan(0, 1, "3" * 64, layout.fragments.num_fragments,
                                               layout_hash=None)[vfs.PLAN_KEY],
                                  syncer="127.0.0.1:1", learner_id=0, model_revision="a" * 40)
    assert auto.expected_layout_hash is None and pinned.expected_layout_hash == layout.layout_hash
    assert session_contract_hash(layout.layout_hash, auto, backend_identity_sha256="c" * 64) == \
        session_contract_hash(layout.layout_hash, pinned, backend_identity_sha256="c" * 64)


ROUNDS = 3
ASYNC = ("--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download")


@needs_cargo
def test_an_auto_island_and_an_island_pinned_to_its_recorded_hash_share_one_session(
        syncer_binary, tmp_path, monkeypatch, capfd):  # noqa: F811
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    layout = _layout(minimum_fragments=1, matrix_merge="rda")
    fragments = layout.fragments.num_fragments
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=ROUNDS,
                                           extra=ASYNC)
    results: dict = {}
    try:
        threads = [
            threading.Thread(target=_run_island, args=(
                0, _island_plan(0, port, profile, fragments, layout_hash=None), SHAPES, results)),
            threading.Thread(target=_run_island, args=(
                1, _island_plan(1, port, profile, fragments, layout_hash=layout.layout_hash), SHAPES, results)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=150)
        code = process.wait(timeout=30) if all(isinstance(results.get(i), dict) for i in range(2)) else None
    finally:
        kill_group(process)
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        assert isinstance(results.get(learner_id), dict), (results.get(learner_id), log[-3000:])
    assert code == 0, log[-3000:]
    # The auto island printed and recorded the layout the pinned island froze.
    bound = [fields for event, fields in results[0]["events"] if event == LAYOUT_BOUND_EVENT]
    assert bound == [{"role": "actor", "learner_id": 0, "parameter_layout_sha256": layout.layout_hash,
                      "fragments": fragments,
                      "fragment_bytes": [f.numel * 4 for f in layout.fragments.fragments]}]
    assert not [e for e, _ in results[1]["events"] if e == LAYOUT_BOUND_EVENT]  # pinned: as today
    printed = [line for line in capfd.readouterr().out.splitlines() if line.startswith(LAYOUT_AUTO_LOG_PREFIX)]
    assert [json.loads(line[len(LAYOUT_AUTO_LOG_PREFIX):])["parameter_layout_sha256"] for line in printed] == [
        layout.layout_hash]
    for name in SHAPES:
        assert torch.equal(results[0]["values"][name], results[1]["values"][name]), name


@needs_cargo
def test_an_auto_island_with_another_layout_is_refused_at_its_hello(
        syncer_binary, tmp_path, monkeypatch, capfd):  # noqa: F811
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    layout = _layout(minimum_fragments=1, matrix_merge="rda")
    fragments = layout.fragments.num_fragments
    other = vfs.fp.plan_layout(vfs.fp.island_specs(RENAMED), component=layout.components[0],
                               minimum_fragments=1, max_fragment_bytes=64, matrix_merge="rda")
    assert other.fragments.num_fragments == fragments and other.layout_hash != layout.layout_hash
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=ROUNDS,
                                           extra=ASYNC)
    results: dict = {}
    try:
        first = threading.Thread(target=_run_island, args=(
            0, _island_plan(0, port, profile, fragments, layout_hash=None, wait_timeout_s=8.0), SHAPES, results))
        first.start()
        time.sleep(5.0)  # island 0's HELLO opens the session (its first dial waits ~2 s)
        second = threading.Thread(target=_run_island, args=(
            1, _island_plan(1, port, profile, fragments, layout_hash=None), RENAMED, results))
        second.start()
        second.join(timeout=90)
        first.join(timeout=150)
    finally:
        kill_group(process)
        log = process.communicate(timeout=10)[0]
    refused = results.get(1)
    assert isinstance(refused, BaseException), (refused, log[-3000:])
    assert "session mismatch (HELLO refused" in str(refused) and "session mismatch" in log, (refused, log[-3000:])
    # Island 0 holds the session; with its peer refused it cannot finish a
    # fixed-roster run, which is not what this test is about.
    printed = {json.loads(line[len(LAYOUT_AUTO_LOG_PREFIX):])["learner_id"]:
               json.loads(line[len(LAYOUT_AUTO_LOG_PREFIX):])["parameter_layout_sha256"]
               for line in capfd.readouterr().out.splitlines() if line.startswith(LAYOUT_AUTO_LOG_PREFIX)}
    assert printed == {0: layout.layout_hash, 1: other.layout_hash}  # both in the logs
