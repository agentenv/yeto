"""Local init against the real Rust syncer (CPU, no Ray).

Every island loaded the same base weights. Islands report per-fragment
INIT digests; learner 0 still uploads INIT (the syncer keeps exact f32
masters), but an island whose digests agree gets INIT_CONFIRM instead of the
version-0 broadcast. Merged rounds must equal the old INIT path bit for bit,
a disagreeing island must stop the run with a clear error (the default), or
with --local-init-fallback get the broadcast of learner 0's INIT plus a
warning, and a resumed syncer must broadcast its checkpointed cut instead of
confirming.
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest

from test_rl_outer_merge_real_syncer import (  # noqa: F401 - fixtures
    _deltas,
    _f32,
    _fake_sky,
    _frag_of,
    _hmac_key,
    _initial,
    _no_modal_listing,
    Reference,
    Rig,
    syncer_binary,
)
from yeto.protocol import DTYPE_F32, SessionRejectedError, SyncerClient

LAYOUT_HASH = hashlib.sha256(b"local-init-test-layout").digest()
LR, MU = 0.35, 0.6


def _digests(initial):
    return [hashlib.sha256(_f32(values)).digest() for values in initial]


class LocalRig(Rig):
    """``Rig`` whose islands may report local-init digests before start."""

    def __init__(self, binary, tmp_path, islands, *options, syncer_flags=(), **kwargs):
        # The launcher-built command adds --resume when the checkpoint exists;
        # ``syncer_flags`` go straight to the syncer binary.
        self.tape = tmp_path / "tape.jsonl"
        binary = " ".join([str(binary), *syncer_flags])
        super().__init__(binary, tmp_path, islands, "--rl-sync-outer-lr", str(LR),
                         "--rl-sync-outer-momentum", str(MU), *options, **kwargs)

    def start_local(self, initial, digests_by_island):
        """Islands in ``digests_by_island`` report digests; learner 0 sends INIT."""
        for island, digests in digests_by_island.items():
            self.clients[island].set_init_digests(LAYOUT_HASH, digests)
        for client in self.clients:
            client.start()
        time.sleep(0.5)  # let every island's digests reach the syncer first
        for fid, values in enumerate(initial):
            self.clients[0].send_init(fid, _f32(values))

    def wait_confirmed(self, island, timeout=20.0):
        client, deadline = self.clients[island], time.monotonic() + timeout
        while not client.initial_cut_confirmed():
            client.check_health()
            assert time.monotonic() < deadline, f"island {island} got no INIT_CONFIRM"
            time.sleep(0.01)
        client.mark_initial_cut_received()

    def wait_exit(self, timeout=15.0):
        deadline = time.monotonic() + timeout
        while self.process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        return self.process.poll()


@pytest.fixture
def make_local_rig(syncer_binary, tmp_path):
    rigs = []

    def make(islands, *options, workdir=None, **kwargs):
        if workdir is None:
            workdir = tmp_path / f"rig{len(rigs)}"
            workdir.mkdir()
        rigs.append(LocalRig(syncer_binary, workdir, islands, *options, **kwargs))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.close()


def _run_rounds(rig, initial, steps, *, ref=None, seed=40):
    """Push both islands' deltas for ``steps``; return every merged BCAST."""
    ref = ref or Reference(rig.layout, initial, LR, MU)
    got = {}
    for step in steps:
        fid = _frag_of(rig, step)
        deltas = _deltas(rig.layout, fid, seed + step, 2)
        base = max(step - rig.layout.num_fragments, 0)
        for island in (0, 1):
            rig.push(island, step, deltas[island], base=base)
        want = ref.step(fid, deltas, [1.0, 1.0])
        for island in (0, 1):
            got[(island, step)] = rig.bcast(island, fid, step).copy()
        np.testing.assert_allclose(got[(0, step)], want, rtol=2e-5, atol=2e-5)
    return got


def test_local_init_confirms_without_broadcast_and_merges_like_init_broadcast(make_local_rig):
    steps = range(1, 7)
    old = make_local_rig(2, total_steps=4)
    initial = _initial(old.layout)
    old.start(initial)
    reference = _run_rounds(old, initial, steps)

    rig = make_local_rig(2, total_steps=4)
    rig.start_local(initial, {0: _digests(initial), 1: _digests(initial)})
    for island in (0, 1):
        rig.wait_confirmed(island)
        assert not rig.clients[island].drain_updates(), "no version-0 broadcast in local init"
    local = _run_rounds(rig, initial, steps)
    for key, values in reference.items():  # identical bits, not just close
        assert values.tobytes() == local[key].tobytes(), key


def test_island_without_digests_still_gets_the_broadcast(make_local_rig):
    rig = make_local_rig(2, total_steps=4)
    initial = _initial(rig.layout)
    rig.start_local(initial, {0: _digests(initial)})
    rig.wait_confirmed(0)
    for fid in range(rig.layout.num_fragments):
        assert rig.bcast(1, fid, 0).tobytes() == np.asarray(initial[fid]).tobytes()
    _run_rounds(rig, initial, range(1, 3))


@pytest.mark.parametrize("bad", ["island", "init"])
def test_digest_mismatch_stops_the_run_with_a_clear_error(make_local_rig, bad):
    rig = make_local_rig(2)
    initial = _initial(rig.layout)
    digests = _digests(initial)
    other = list(digests)
    other[1] = hashlib.sha256(b"another checkpoint").digest()
    # "island": island 1 loaded other weights; "init": learner 0 uploads
    # bytes that differ from the digests both islands agreed on.
    by_island = {0: digests, 1: other} if bad == "island" else {0: other, 1: other}
    rig.start_local(initial, by_island)
    assert rig.wait_exit() not in (None, 0)
    rig.close()
    assert "init digest mismatch: fragment 1" in rig.output
    assert "same base checkpoint" in rig.output
    # The island that hears the refusal fails as a strict invariant.
    from yeto.rl.core import StrictRlInvariantError

    errors = [client._err for client in rig.clients]
    assert any(isinstance(e, SessionRejectedError) and e.metric == "init_digest_mismatch"
               for e in errors), errors
    # Only the refused island raises the strict error; the other one may just
    # see the syncer close its connection first, so check the refused one.
    refused = next(c for c in rig.clients if isinstance(c._err, SessionRejectedError))
    with pytest.raises(StrictRlInvariantError, match="init digest mismatch"):
        refused.check_health()


@pytest.mark.parametrize("bad", ["island", "init"])
def test_local_init_fallback_broadcasts_to_a_disagreeing_island(make_local_rig, bad):
    """--local-init-fallback: the run goes on; learner 0's INIT is version 0,
    an island whose digests differ from it gets the broadcast, and merged
    rounds equal the plain INIT-broadcast path bit for bit."""
    steps = range(1, 7)
    old = make_local_rig(2, total_steps=4)
    initial = _initial(old.layout)
    old.start(initial)
    reference = _run_rounds(old, initial, steps)

    rig = make_local_rig(2, total_steps=4, syncer_flags=["--local-init-fallback"])
    digests = _digests(initial)
    other = list(digests)
    other[1] = hashlib.sha256(b"another conversion path").digest()
    # "island": island 1 canonicalized the base to other bits; "init": both
    # islands agree but learner 0's INIT differs (every island falls back).
    by_island = {0: digests, 1: other} if bad == "island" else {0: other, 1: other}
    rig.start_local(initial, by_island)
    fallback = [1] if bad == "island" else [0, 1]
    if bad == "island":
        rig.wait_confirmed(0)
        assert not rig.clients[0].drain_updates()
    for island in fallback:
        for fid in range(rig.layout.num_fragments):
            assert rig.bcast(island, fid, 0).tobytes() == np.asarray(initial[fid]).tobytes()
        assert not rig.clients[island].initial_cut_confirmed()
    local = _run_rounds(rig, initial, steps)
    for key, values in reference.items():
        assert values.tobytes() == local[key].tobytes(), key
    rig.close()
    for island in fallback:  # one summary warning per disagreeing island
        assert rig.output.count(f"learner {island} reported local-init digests") == 1, rig.output
    assert "init digest mismatch: fragment 1" not in rig.output  # per-fragment detail is debug only
    records = [json.loads(line) for line in rig.tape.read_text().splitlines()]
    uploads = [r for r in records if r.get("event") == "init_digest_fallback_upload"]
    assert sorted(r["learner_id"] for r in uploads) == fallback
    assert all(r["mismatched_fragments"] == 1 for r in uploads)
    assert not any(r.get("event") in ("rl_strict_failure", "init_digest_mismatch_detected")
                   for r in records)


def test_strict_local_init_flag_keeps_the_fatal_check(make_local_rig):
    rig = make_local_rig(2, syncer_flags=["--strict-local-init"])
    initial = _initial(rig.layout)
    digests = _digests(initial)
    other = list(digests)
    other[0] = hashlib.sha256(b"another checkpoint").digest()
    rig.start_local(initial, {0: digests, 1: other})
    assert rig.wait_exit() not in (None, 0)
    rig.close()
    assert "init digest mismatch: fragment 0" in rig.output


def test_local_init_digests_parallel_equals_serial():
    from yeto.rl.engine.full_parameter_stream import local_init_digests

    rng = np.random.default_rng(3)
    fragments = [rng.standard_normal(n).astype(np.float32) for n in (1, 7, 4096, 33, 100_000)]

    def parts_of(fid):  # uneven chunks, as an adapter streams them
        raw = memoryview(fragments[fid].tobytes())
        return [raw[start:start + 1000] for start in range(0, len(raw), 1000)]

    want = [hashlib.sha256(f.tobytes()).digest() for f in fragments]
    for workers in (1, 2, 8):
        assert local_init_digests(len(fragments), parts_of, workers=workers) == want


def test_resumed_syncer_broadcasts_its_checkpoint_instead_of_confirming(make_local_rig, tmp_path):
    workdir = tmp_path / "resume"
    workdir.mkdir()
    rig = make_local_rig(2, total_steps=4, workdir=workdir)
    initial = _initial(rig.layout)
    rig.start_local(initial, {0: _digests(initial), 1: _digests(initial)})
    for island in (0, 1):
        rig.wait_confirmed(island)
    ref = Reference(rig.layout, initial, LR, MU)
    first = _run_rounds(rig, initial, range(1, 4), ref=ref)
    time.sleep(0.5)  # the round-3 checkpoint commits before its broadcast; be safe
    rig.close()
    assert (workdir / "state.ckpt").exists()

    resumed = make_local_rig(2, total_steps=4, workdir=workdir)
    assert "--resume" in resumed.command
    resumed.start_local(initial, {0: _digests(initial), 1: _digests(initial)})
    # Fragment 0 was last merged in round 3, fragment 1 in round 2.
    for island in (0, 1):
        assert resumed.bcast(island, 0, 3).tobytes() == first[(0, 3)].tobytes()
        assert resumed.bcast(island, 1, 2).tobytes() == first[(0, 2)].tobytes()
        assert not resumed.clients[island].initial_cut_confirmed()
    _run_rounds(resumed, initial, range(4, 7), ref=ref)


def test_set_init_digests_validates_its_input():
    from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout

    layout = FragmentLayout([Fragment(MERGE_AVG, [("w", 4)])])
    client = SyncerClient(("127.0.0.1", 9), 0, layout, dtype=DTYPE_F32)
    with pytest.raises(ValueError):
        client.set_init_digests(LAYOUT_HASH, [])
    with pytest.raises(ValueError):
        client.set_init_digests(b"short", [b"x" * 32])
    client.set_init_digests(LAYOUT_HASH, [b"x" * 32])
    assert not client.initial_cut_confirmed()


# ---- role stream (no syncer) ----------------------------------------------------------

class _Stored:
    def __init__(self, fragment_id, payload=b""):
        self.fragment_id, self.version, self.payload, self.released = fragment_id, 0, payload, False

    def release(self):
        self.released = True


def _fake_adapter(payloads):
    from types import SimpleNamespace

    from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout

    fragments = FragmentLayout([Fragment(MERGE_AVG, [(f"w{i}", len(p) // 4)])
                                for i, p in enumerate(payloads)])
    layout = SimpleNamespace(fragments=fragments, layout_hash="1" * 64)

    def fragment_parts(_initial, fragment_id, ray_module=None):
        data = payloads[fragment_id]
        return lambda: [data[:4], data[4:]]  # parts, as Miles streams them

    return SimpleNamespace(layout=layout, fragment_parts=fragment_parts)


def test_role_stream_keeps_local_version_zero_after_the_confirm(monkeypatch):
    from dataclasses import replace

    from test_rl_full_parameter_sync import _role_stream_config
    from yeto.rl.adapters.miles.full_parameter import role_stream as module

    payloads = [_f32([1, 2, 3]), _f32([4, 5])]
    adapter = _fake_adapter(payloads)
    stream = module.MilesFullParameterRoleStream(
        replace(_role_stream_config("actor", 30100), local_init=True))
    class AdoptingSink:
        def adopt(self, cut, fragment_id, version, wire_payload_hash):
            assert version == 0
            assert wire_payload_hash == hashlib.sha256(payloads[fragment_id]).hexdigest()
            return _Stored(fragment_id, cut[fragment_id])

    local, digests = stream._local_version_zero(adapter, payloads, AdoptingSink())
    assert [s.payload for s in local] == payloads
    assert digests == [hashlib.sha256(p).digest() for p in payloads]

    class ConfirmingClient:
        marked = False

        def check_health(self):
            pass

        def drain_updates(self):
            return []

        def initial_cut_confirmed(self):
            return True

        def mark_initial_cut_received(self):
            self.marked = True

    client = ConfirmingClient()
    assert stream._wait_for_initial_fragments(client, adapter, None, local) == local
    assert client.marked and not any(s.released for s in local)


def test_initial_cut_timeout_scales_with_the_model_and_the_init_mode():
    from yeto.rl.adapters.miles.full_parameter.role_stream import initial_cut_timeout

    x6 = 116_500_000_000  # X6: f32 Qwen3-30B-A3B
    upload = initial_cut_timeout(900.0, x6, 2, local_init=False)
    local = initial_cut_timeout(900.0, x6, 2, local_init=True)
    assert upload == pytest.approx(3 * local) and local > 3000  # 900 s timed out in X6
    assert initial_cut_timeout(900.0, 1 << 20, 2, local_init=False) == 900.0


def test_local_init_is_the_default_for_async_rounds_only():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.island_entry import _local_init

    assert _local_init(SimpleNamespace(init_mode="auto"), True)
    assert not _local_init(SimpleNamespace(init_mode="auto"), False)
    assert _local_init(SimpleNamespace(init_mode="local"), False)
    assert not _local_init(SimpleNamespace(init_mode="upload"), True)


def test_local_init_hash_is_serial_unless_opted_in(monkeypatch):
    from yeto.rl.engine import full_parameter_stream as fps

    monkeypatch.delenv(fps.LOCAL_INIT_HASH_WORKERS_ENV, raising=False)
    assert fps.local_init_hash_workers(64) == 1
    monkeypatch.setenv(fps.LOCAL_INIT_HASH_WORKERS_ENV, "4")
    assert fps.local_init_hash_workers(64) == 4
    assert fps.local_init_hash_workers(2) == 2
    monkeypatch.setenv(fps.LOCAL_INIT_HASH_WORKERS_ENV, "auto")
    monkeypatch.setattr(fps.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert fps.local_init_hash_workers(64) == 3
    monkeypatch.delattr(fps.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(fps.os, "cpu_count", lambda: 40)
    assert fps.local_init_hash_workers(64) == fps.LOCAL_INIT_HASH_AUTO_MAX
    monkeypatch.setenv(fps.LOCAL_INIT_HASH_WORKERS_ENV, "0")
    with pytest.raises(ValueError):
        fps.local_init_hash_workers(64)
