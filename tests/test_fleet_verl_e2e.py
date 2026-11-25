"""A two-island verl fleet end to end on CPU (rl-fleet-verl-islands 2.3).

Two fake verl learners (plain ``torch`` modules behind the real verl full-parameter
stream) merge through the launcher-built Rust syncer with asynchronous rounds. Each is
also its island's buffer trainer releasing its own local versions into its sub-store
``trainer<k>/`` (``--rl-release-mode per_learner``; island 1 every 2 versions). Each
island's fake pool follows its own trainer: before every round it loads its sub-store's
``LATEST.json`` and claims prompts from the fleet's shared task ledger
(``<store>/fleet/dispense``), stamping each claim with the version it serves. Checked:
both islands release into their own sub-store only, each pool loads only its own
trainer's releases and never goes back, the syncer's merges are contiguous, and the
ledger gives every prompt out once across the fleet. Needs cargo (the syncer).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
import torch

from process_groups import group_popen, kill_group
from syncer_build import build_syncer
from syncer_port import bound_port, on_port_zero
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_verl_buffer_trainer import _State
from test_store_dispense_fleet import ROWS, _prompts, _sampler, _spec
from test_verl_fleet_islands import NONZERO, _fleet_trainer
from test_verl_full_parameter_sync import (CARGO, TOKENS, _Driver, _Island, _layout, _Model, _plan, _step,
                                           local_rounds)
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.contracts import LocalStepReceipt
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore

needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")
RELEASE_EVERY = {0: 1, 1: 2}


def _start_syncer(binary: Path, tmp_path: Path, *, fragments: int, rounds: int):
    import subprocess

    from yeto import launcher
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (
        "--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", "e" * 64,
        "--fragments", str(fragments), "--total-steps", str(rounds),
        "--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download"),
        gpu="modal:2xh100,modal:2xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    command = on_port_zero(launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"])
    command = (command.replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl"))
               .replace("~/yeto-output/", f"{tmp_path}/"))  # the dispenser's log: not the host's home
    process = group_popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return process, bound_port(process), launcher.rl_syncer_profile(args, 2).sha256


class _FollowingPool:
    """Island k's fake engine pool: serves its own trainer's newest release and claims
    prompts from the fleet's ledger (as the engine sampler does through its spec)."""

    def __init__(self, k: int, run: Path, prompts: Path):
        self.k = k
        self.releases = PolicyReleaseStore(CheckpointStore(run / f"trainer{k}"))
        self.prompts = _sampler(_spec(run, k, f"pool-i{k}-0", prompts, shared=True, groups=1), inc=f"inc-{k}")
        self.loaded: list[tuple[int, str]] = []  # (version, the release's weights)
        self.claims: list[tuple[int, int]] = []  # (slot, the version it was sampled on)
        self.ended = False

    def serving(self) -> int:
        latest = self.releases.latest()
        if latest is not None and (not self.loaded or latest.version != self.loaded[-1][0]):
            weights = (self.releases.root / latest.path / "model.safetensors").read_bytes().decode()
            self.loaded.append((latest.version, weights))
        return self.loaded[-1][0] if self.loaded else 0

    def round(self, rollout_id: int) -> None:
        if self.ended:
            return
        version = self.serving()
        state, slots = self.prompts.poll(self.prompts.groups_per_round)
        if state == "items":
            self.claims += [(slot, version) for slot in slots]
            self.prompts.written(rollout_id, slots)
        elif state in ("end", "fenced"):
            self.ended = True

    def drain(self) -> None:
        for rollout_id in range(10 * ROWS):
            if self.ended:
                return
            self.round(10_000 + rollout_id)


def _run_island(k: int, plan: dict, run: Path, scratch: Path, prompts: Path, results: dict) -> None:
    model = _Model(seed=0)
    island = _Island(model)
    island.full_group = vfs.VerlFullParameterGroup(island)
    sync, _policy_state, _stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
    driver = _Driver()
    releases: list = []
    scratch.mkdir(parents=True, exist_ok=True)
    actor, _isl, publisher, release_driver = _fleet_trainer(scratch, CheckpointStore(run / f"trainer{k}"), k,
                                                            releases, release_every=RELEASE_EVERY[k])
    pool = _FollowingPool(k, run, prompts)
    try:
        state = sync.start(driver).state
        for round_id in local_rounds(None):
            pool.round(round_id)  # the pool samples on the release it serves now
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.add_(_step(k, round_id)[name])
            ids = (f"l{k}-r{round_id}",)
            import hashlib

            batch_hash = hashlib.sha256(ids[0].encode()).hexdigest()
            policy_hash = state.policy_tensor_hash()
            island.full_group.record_round(
                LocalStepReceipt(algorithm="grpo", learner_id=k, learner_generation=0,
                                 base_policy_version=round_id, base_policy_hash=policy_hash,
                                 input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS,
                                 optimizer_steps=1, optimizer_step_succeeded=True,
                                 parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"]),
                vfs.VerlRoundEvidence(round_id, policy_hash, batch_hash, TOKENS, ids))
            boundary = sync.boundary(driver, rollout_id=round_id, stats=None)
            state = boundary.state
            # the buffer trainer's publication of its new local version (per learner, its cadence)
            local = round_id + 1
            actor.param_version = local
            release_driver.outer_version = None
            publisher.publish(_State(local, NONZERO, f"l{k}-v{local}"))
            if boundary.stop:
                sync.published(driver, rollout_id=round_id, policy_hash=state.policy_tensor_hash())
                break
        sync.finish(driver)
        publisher.drain_releases()  # the run's end: the last version skipped by the cadence goes out
        pool.serving()
        results[k] = {"events": driver.events, "releases": releases, "pool": pool, "rounds": round_id + 1}
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[k] = error
    finally:
        sync.close()


@needs_cargo
def test_a_two_island_verl_fleet_releases_per_island_pools_follow_and_the_ledger_gives_each_prompt_once(
        tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", f"verl-fleet-e2e-{os.getpid()}")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    rounds, fragments = 3, len(_layout().fragments.fragments)
    run, prompts = tmp_path / "run", _prompts(tmp_path)
    process, port, profile = _start_syncer(build_syncer(), tmp_path, fragments=fragments, rounds=rounds)
    results: dict = {}
    try:
        threads = [threading.Thread(target=_run_island, args=(
            k, _plan(k, port, profile, rounds=rounds, fragments=fragments, async_rounds=True, matrix_merge="rda",
                     pipeline=2, final_cut="download"), run, tmp_path / f"island{k}", prompts, results))
            for k in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=150)
        failed = {k: r for k, r in results.items() if isinstance(r, BaseException)}
        assert not failed, failed  # an island's own error, not the syncer's wait for it
        assert process.wait(timeout=30) == 0
    finally:
        kill_group(process)
        log = process.communicate(timeout=10)[0]
    for k in (0, 1):
        assert not isinstance(results.get(k), BaseException), (results.get(k), log[-3000:])
        assert k in results, log[-3000:]

    for k in (0, 1):
        r = results[k]
        local = r["rounds"]
        # releases: this island's own local versions into its own sub-store, on its cadence,
        # strictly increasing, the last version always out (the drain)
        released = [f["policy_version"] for e, f in r["releases"] if e == "rl_policy_released"]
        every = RELEASE_EVERY[k]
        expected = [v for v in range(1, local + 1) if v % every == 0]
        if local % every:
            expected.append(local)
        assert released == expected, (k, released, local)
        assert all(f["origin"]["kind"] == "local" for e, f in r["releases"] if e == "rl_policy_released")
        skipped = [f["policy_version"] for e, f in r["releases"] if e == "rl_policy_release_skipped"]
        assert skipped == [v for v in range(1, local + 1) if v % every], k
        own = sorted((p.name for p in (run / f"trainer{k}" / "policy-releases").iterdir() if p.is_dir()),
                     key=lambda name: int(name.split("-", 1)[0][1:]))
        # the store keeps the newest releases (keep 2): they are this island's newest
        assert [int(name.rsplit("-v", 1)[1]) for name in own] == released[-len(own):] and own
        # the pool follows its own trainer only: versions never go back, and every one it
        # loaded is a release of this island (its weights say whose)
        pool = r["pool"]
        versions = [v for v, _ in pool.loaded]
        assert versions == sorted(set(versions)) and versions[-1] == released[-1], (k, versions)
        assert set(versions) <= set(released)
        assert {weights for _, weights in pool.loaded} == {f"learner{k}"}
        assert all(v in {0, *released} for _, v in pool.claims)
        # merges: the syncer's fragment versions as this island applied them are contiguous
        merges = [f["sync/fragment_versions"] for e, f in r["events"] if e == "rl_full_parameter_round"]
        assert merges and merges[-1] == list(range((rounds - 1) * fragments + 1, rounds * fragments + 1))
        for before, after in zip(merges, merges[1:]):  # no fragment ever goes back to an older merge
            assert all(b <= a for b, a in zip(before, after)), (k, before, after)
        acks = [f for e, f in r["events"] if e == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}]
    # the syncer's merges: one per step, every step from 1 to the last, none twice
    import json

    tape = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines()]
    steps = [row["step"] for row in tape if "kind" not in row and "step" in row]
    assert steps == list(range(1, rounds * fragments + 1)), steps
    # the ledger: drain both pools to its end; every prompt went to exactly one claim
    for k in (0, 1):
        results[k]["pool"].drain()
    try:
        slots = [slot for k in (0, 1) for slot, _ in results[k]["pool"].claims]
        assert sorted(slots) == list(range(ROWS)), "every prompt once across the fleet, none twice"
        assert all(results[k]["pool"].claims for k in (0, 1)), "both islands' pools took a share"
    finally:
        for k in (0, 1):
            results[k]["pool"].prompts.close()
