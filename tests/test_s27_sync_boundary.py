"""S27 G-AY: the per-fragment round boundary, staggered first pushes, and a
final cut that islands acknowledge without downloading it.

* lazy capture: a local round's capture reads no master (its rows carry
  labels) and its receipt skips the whole-model model/master check; each
  pushed fragment gets its content descriptor when it is materialized, and
  every ``YETO_FULL_PARAMETER_FULL_CHECK_EVERY``-th round still hashes every
  fragment. Pushes, trainer values and syncer values must be bit-identical
  to the run that hashes every fragment every round;
* staggered first pushes: with H > 1 the fragments that all become pushable
  at round H wait ``fragment_id * H // P`` more rounds for their first push;
* final cut "ack": the syncer sends the version manifest only
  (``--final-cut-payload manifest``) and an island ACKs it; "download"
  keeps the f32 final cut.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest
from test_rl_full_parameter_async import CARGO, needs_cargo

from yeto.rl.adapters.miles.full_parameter import role_stream as role_stream_module
from yeto.rl.adapters.miles.full_parameter.role_stream import (
    FULL_CHECK_EVERY_ENV,
    STAGGER_FIRST_PUSH_ENV,
    first_push_phase,
    full_check_every,
)
from yeto.syncer_profile import SyncerSemanticProfile

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.fixture
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "final-cut-test-key")


# -- staggered first pushes --------------------------------------------------------


def test_first_push_phases_spread_the_startup_burst_over_h_rounds():
    fragments, horizon, pipeline = 60, 8, 20
    phases = [first_push_phase(p, fragments, horizon) for p in range(fragments)]
    assert phases[:pipeline] == [0] * 8 + [1] * 7 + [2] * 5  # the syncer's first 20 rounds
    assert max(phases) == horizon - 1 and phases == sorted(phases)
    # At most ceil(P / H) fragments share a phase: one round's share of the sweep.
    assert max(phases.count(phase) for phase in set(phases)) == -(-fragments // horizon)
    assert all(first_push_phase(p, fragments, 1) == 0 for p in range(fragments))
    assert first_push_phase(0, 1, 8) == 0


def test_the_full_check_period_is_validated(monkeypatch):
    monkeypatch.delenv(FULL_CHECK_EVERY_ENV, raising=False)
    assert full_check_every() == role_stream_module.DEFAULT_FULL_CHECK_EVERY
    monkeypatch.setenv(FULL_CHECK_EVERY_ENV, "0")
    assert full_check_every() == 0
    for bad in ("-1", "x"):
        monkeypatch.setenv(FULL_CHECK_EVERY_ENV, bad)
        with pytest.raises(ValueError, match=FULL_CHECK_EVERY_ENV):
            full_check_every()


# -- the real Miles role stream (contract fakes) ------------------------------------

try:
    import miles  # noqa: F401

    HAVE_MILES = True
except ImportError:
    HAVE_MILES = False

needs_miles = pytest.mark.skipif(not HAVE_MILES, reason="needs Miles")


def _lazy_contract_trainer():
    """The Miles lazy-export contract over ``_MilesContractTrainer``: a lazy
    export labels every row (fresh per export, never a content hash), a
    labelled row is materialized only with its content descriptor, never read
    by an apply, and a receipt may skip the model/master check."""
    import test_s27_sync_fast_miles_role_stream as fast
    from miles.ray.full_parameter_transport import FullParameterChunkedCut
    from test_e3_overlap_miles_role_stream import _build_cut

    def labelled(descriptor, nonce: bytes):
        def label(*parts) -> str:
            digest = hashlib.sha256(b"test-label\0" + nonce)
            for part in parts:
                digest.update(int(part).to_bytes(8, "little"))
            return digest.hexdigest()

        chunks = tuple(replace(chunk, payload_hash=label(descriptor.fragment_id, chunk.chunk_index))
                       for chunk in descriptor.chunks)
        return replace(descriptor, payload_hash=label(descriptor.fragment_id, 1 << 40), chunks=chunks)

    def row(cut, fragment_id):
        ((descriptor, refs),) = [(d, r) for shard in cut.shards
                                 for d, r in zip(shard.fragments, shard.chunk_refs, strict=True)
                                 if d.fragment_id == fragment_id]
        return descriptor, refs

    class LazyContractTrainer(fast._MilesContractTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.labels: set[str] = set()
            self.lazy_exports = 0
            self.full_exports = 0
            self.skipped_model_checks = 0
            self.described = 0

        async def record_full_parameter_local_step(self, *, check_model=True, **kwargs):
            receipts = await super().record_full_parameter_local_step(**kwargs)
            if not check_model:
                self.model_checks -= 1
                self.skipped_model_checks += 1
            return receipts

        async def export_full_parameter_chunked_cut(self, policy_version, local_step_generation=0, *,
                                                    max_chunk_bytes, defer_payload=False, lazy=False):
            cut = await super().export_full_parameter_chunked_cut(
                policy_version, local_step_generation, max_chunk_bytes=max_chunk_bytes,
                defer_payload=defer_payload)
            if not lazy:
                self.full_exports += defer_payload
                return cut
            assert defer_payload
            self.lazy_exports += 1
            nonce = os.urandom(8)
            cut = FullParameterChunkedCut(
                cut.policy_version, cut.local_step_generation, cut.parameter_layout_hash,
                tuple(replace(shard, fragments=tuple(labelled(d, nonce) for d in shard.fragments))
                      for shard in cut.shards))
            self.resident = fast._hashes(cut)
            self.labels |= set(self.resident.values())
            return cut

        async def materialize_full_parameter_fragments(self, cut, fragment_ids, *, return_descriptors=False):
            wanted = fast._hashes(cut)
            if not any(wanted[f] in self.labels for f in fragment_ids):
                stored = await super().materialize_full_parameter_fragments(cut, fragment_ids)
                if not return_descriptors:
                    return stored
                return {f: (row(cut, f)[0], refs) for f, refs in stored.items()}
            assert return_descriptors, "a labelled fragment needs its content descriptor"
            fresh = _build_cut(self.adapter, self.values, policy_version=cut.policy_version,
                               generation=cut.local_step_generation, ray=self.ray)
            stored = {}
            for fragment_id in fragment_ids:
                assert self.resident is not None and self.resident[fragment_id] == wanted[fragment_id], \
                    "materialized a fragment the masters no longer hold"
                descriptor, refs = row(fresh, fragment_id)
                stored[fragment_id] = (descriptor, list(refs))
                self.stored_bytes += 4 * descriptor.numel
                self.described += 1
            return stored

        async def apply_full_parameter_chunked_cut(self, cut, **kwargs):
            for shard in cut.shards:
                for descriptor in shard.fragments:
                    if descriptor.payload_hash in self.labels:
                        assert self.resident is not None and \
                            self.resident.get(descriptor.fragment_id) == descriptor.payload_hash, \
                            "an apply read a labelled row"
            return await super().apply_full_parameter_chunked_cut(cut, **kwargs)

    return LazyContractTrainer


def _bits(values):
    import torch

    return {f: value.view(torch.int32).tolist() for f, value in values.items()}


def _pushes(run):
    import torch

    return sorted((step, attempt, delta.view(torch.int32).tolist()) for step, attempt, delta in run.syncer.pushed)


@needs_miles
@pytest.mark.parametrize(("horizon", "every"), [(1, "3"), (2, "2"), (2, "0")])
def test_lazy_capture_keeps_every_value_bit_identical(monkeypatch, horizon, every):
    from test_s27_background_push import CONTRACT_FEATURES, _run_miles

    monkeypatch.setenv(FULL_CHECK_EVERY_ENV, every)
    trainer = _lazy_contract_trainer()
    hashed = _run_miles(monkeypatch, background=True, horizon=horizon, trainer_base=trainer)
    lazy = _run_miles(monkeypatch, background=True, horizon=horizon, trainer_base=trainer,
                      features=CONTRACT_FEATURES | {"lazy_export"})

    assert hashed.trainer.lazy_exports == 0 and lazy.trainer.lazy_exports > 0
    assert lazy.stops == hashed.stops and lazy.stops[-1]
    assert _pushes(lazy) == _pushes(hashed)
    assert _bits(lazy.trainer.values) == _bits(hashed.trainer.values)
    assert _bits(lazy.syncer.global_) == _bits(hashed.syncer.global_)
    rounds = lazy.stream.local_rounds
    period = int(every)
    full_rounds = rounds // period if period else 0
    assert lazy.trainer.full_exports == full_rounds
    assert lazy.trainer.lazy_exports == lazy.trainer.skipped_model_checks == rounds - full_rounds
    assert lazy.trainer.described > 0  # pushes read the content descriptors
    flagged = [fields.get("sync/capture_lazy") for event, fields in lazy.driver.events
               if event == "rl_full_parameter_round"]
    assert flagged.count(True) == rounds - full_rounds and set(flagged) <= {True, None}
    assert not any(fields.get("sync/capture_lazy") for event, fields in hashed.driver.events)


def _first_push_c_steps(run):
    first = {}
    for step, c_steps, _local in sorted(run.client.meta):
        first.setdefault((step - 1) % len(run.syncer.versions), c_steps)
    return first


@needs_miles
def test_staggered_first_pushes_keep_every_later_push_at_h(monkeypatch):
    from test_s27_background_push import _run_miles

    horizon = 2
    monkeypatch.setenv(STAGGER_FIRST_PUSH_ENV, "0")
    plain = _run_miles(monkeypatch, background=True, horizon=horizon)
    monkeypatch.delenv(STAGGER_FIRST_PUSH_ENV)
    staggered = _run_miles(monkeypatch, background=True, horizon=horizon)

    fragments = len(staggered.syncer.versions)
    assert _first_push_c_steps(plain) == {f: horizon for f in range(fragments)}
    assert _first_push_c_steps(staggered) == {
        f: horizon + first_push_phase(f, fragments, horizon) for f in range(fragments)}
    for run in (plain, staggered):
        assert run.stops[-1] and min(c_steps for _s, c_steps, _l in run.client.meta) >= horizon
        by_fragment = {}
        for step, _c_steps, local_step in sorted(run.client.meta):
            by_fragment.setdefault((step - 1) % fragments, []).append(local_step)
        for local_steps in by_fragment.values():
            assert all(b - a >= horizon for a, b in pairwise(local_steps))


@needs_miles
def test_an_ack_final_cut_acknowledges_the_manifest_without_applying_it(monkeypatch):
    from test_s27_background_push import _run_miles

    download = _run_miles(monkeypatch, background=True, horizon=1)
    ack = _run_miles(monkeypatch, background=True, horizon=1, final_cut="ack")

    for run in (download, ack):
        assert run.stops[-1] and run.stream.final_reached and run.stream.final_acknowledged
        assert run.client.acknowledged == run.stream.terminal_manifest
    assert download.stream.final_applied and not ack.stream.final_applied
    assert _bits(download.trainer.values) == _bits(download.syncer.global_)
    # The island kept its own weights: the final parameters stay with the syncer.
    assert _bits(ack.syncer.global_) == _bits(download.syncer.global_)
    assert _bits(ack.trainer.values) != _bits(ack.syncer.global_)
    assert _pushes(ack) == _pushes(download)
    ((_, fields),) = [(e, f) for e, f in ack.driver.events if e == "rl_full_parameter_final_ack"]
    assert fields == {"acknowledged": True, "sync/final_cut": "ack"}
    ((_, fields),) = [(e, f) for e, f in download.driver.events if e == "rl_full_parameter_final_ack"]
    assert fields == {"acknowledged": True}


def test_the_role_stream_refuses_an_unknown_final_cut():
    from test_s27_sync_fast_miles_role_stream import _config

    with pytest.raises(ValueError, match="final cut"):
        replace(_config(), final_cut="maybe")


# -- launcher and syncer profile ----------------------------------------------------

def _plan(extra=()):
    from test_rl_full_parameter_async import ASYNC
    from test_rl_full_parameter_async import _plan as plan

    return plan(ASYNC + tuple(extra))


def _flag(command: str, flag: str):
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1] if flag in tokens else None


@pytest.mark.parametrize(("extra", "mode"), [(("--rl-sync-mode", "async"), "ack"),
                                             (("--rl-sync-final-cut", "ack"), "ack"),
                                             (("--rl-sync-final-cut", "download"), "download")])
def test_the_launcher_resolves_the_final_cut_for_syncer_profile_and_islands(extra, mode):
    from yeto import launcher
    from yeto.launch.syncer import rl_full_parameter_final_cut

    args, plan = _plan(extra)
    assert rl_full_parameter_final_cut(args) == mode
    [actor] = plan["role_syncers"]
    assert _flag(actor["command"], "--final-cut-payload") == ("manifest" if mode == "ack" else None)
    profile = launcher.rl_syncer_profile(args, 2)
    assert profile.final_cut_manifest_only is (mode == "ack")
    for island in plan["island_requests"]:
        command = island["learner_command"]
        assert _flag(command, "--sweep-final-cut") == ("ack" if mode == "ack" else None)
        assert _flag(command, "--syncer-profile-sha256") == profile.sha256


def test_the_strict_sweep_keeps_the_download_by_default():
    from test_rl_full_parameter_async import FULL
    from test_rl_full_parameter_async import _plan as plan

    from yeto.launch.syncer import rl_full_parameter_final_cut

    args, sweep = plan(FULL)
    assert rl_full_parameter_final_cut(args) == "download"
    assert "--final-cut-payload" not in sweep["role_syncers"][0]["command"]
    args, sweep = plan(FULL + ("--rl-sync-final-cut", "ack"))
    assert rl_full_parameter_final_cut(args) == "ack"
    assert _flag(sweep["role_syncers"][0]["command"], "--final-cut-payload") == "manifest"
    # The strict sync mode is the control: its commands do not change.
    args, strict = _plan(("--rl-sync-mode", "strict"))
    assert rl_full_parameter_final_cut(args) == "download"
    assert "--final-cut-payload" not in strict["role_syncers"][0]["command"]


def test_island_eval_keeps_the_download_by_default():
    from yeto.launch.syncer import rl_full_parameter_final_cut

    args, _ = _plan(("--rl-sync-mode", "async"))
    assert rl_full_parameter_final_cut(args) == "ack"
    args.rl_eval_interval = 5
    assert rl_full_parameter_final_cut(args) == "download"
    args.rl_sync_final_cut = "ack"
    assert rl_full_parameter_final_cut(args) == "ack"


def test_the_final_cut_option_is_full_parameter_only():
    from test_rl_full_parameter_async import _launcher_args

    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-sync-final-cut", "ack"))
    with pytest.raises(ValueError, match="--rl-sync-final-cut only applies"):
        _prepare_rl_args(args)


def test_the_manifest_only_profile_matches_the_rust_vector():
    from test_rl_moe_expert_token_weights import _syncer_profile

    raw = _syncer_profile()
    off = SyncerSemanticProfile.from_mapping(raw).sha256
    assert SyncerSemanticProfile.from_mapping({**raw, "final_cut_manifest_only": False}).sha256 == off
    on = SyncerSemanticProfile.from_mapping({**raw, "final_cut_manifest_only": True}).sha256
    source = (ROOT / "syncer/src/server.rs").read_text()
    (vector,) = re.findall(r'FINAL_CUT_MANIFEST_ONLY_VECTOR: &str = "([0-9a-f]{64})"', source)
    assert on == vector != off


# -- the real syncer ------------------------------------------------------------------


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_the_real_syncer_sends_a_manifest_only_final_cut(syncer_binary, tmp_path):
    from test_rl_full_parameter_async import (
        P,
        _client,
        _delta,
        _f32,
        _initial,
        _start_async_syncer,
        _wait,
    )

    rounds = 2  # merges per fragment (launcher --total-steps)
    total_steps = rounds * P
    # A manifest-only final cut (the profile hash the clients send binds it).
    process, port, profile_hash = _start_async_syncer(
        syncer_binary, tmp_path, "--rl-sync-quorum", "2", "--rl-sync-final-cut", "ack", total_steps=rounds,
        log_file=tmp_path / "syncer.log")
    clients = [_client(learner, port, profile_hash) for learner in (0, 1)]
    seen: list[set] = [set(), set()]

    def poll():
        for learner, client in enumerate(clients):
            client.check_health()
            client.drain_updates()
            for permit in client.drain_pulls():
                if (permit.global_step, permit.round_attempt) in seen[learner]:
                    continue
                seen[learner].add((permit.global_step, permit.round_attempt))
                base = max(0, permit.global_step - P)
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt, base,
                                     permit.global_step, 1, 10, _f32(_delta(learner, permit.global_step)))
        return all(client.finalizing.is_set() for client in clients)

    try:
        for client in clients:
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=60)
        for client in clients:
            manifest = client.wait_for_final_manifest(timeout=30)
            assert manifest.global_step == total_steps
            assert manifest.versions == tuple(range(total_steps - P + 1, total_steps + 1))
            with pytest.raises(TimeoutError, match="missing fragment versions"):
                client.wait_for_final_fragments(timeout=0.5)
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients:
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    log = (tmp_path / "syncer.log").read_text()
    assert log.count("learner finalized authoritative parameters") == 2
