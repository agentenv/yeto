"""S27 P1: the outer-merge options against the real Rust syncer (CPU, no Ray).

The syncer runs the exact command the launcher builds for
``--rl-sync-preset decoupled`` plus the ``--rl-sync-*`` options. Two or three
fake islands (``SyncerClient``) push crafted deltas; each merged BCAST is
compared with a numpy reference of the same rule (weighted average or RDA,
then SGD with Nesterov momentum).

Round numbering: with P fragments and pipeline 1, round ``t`` merges fragment
``(t - 1) % P``. A fragment's version is the number of the round that last
merged it, so its base version at round ``t`` is ``max(t - P, 0)``.
``--rl-sync-max-base-lag`` counts fragment rounds; the launcher passes
rounds x P to the syncer's ``--max-base-lag``, which counts versions.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_algorithm_provenance import _launcher_args as _provenance_launcher_args


def _launcher_args(engine, extra=(), gpu="aws:1xa100@us-east-1"):
    """These tests pin the strict control (--rl-sync-mode strict); the async
    defaults are covered in test_rl_sync_mode.py."""
    return _provenance_launcher_args(engine, ("--rl-sync-mode", "strict", *extra), gpu=gpu)
from yeto.fragments import MERGE_AVG, MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import DTYPE_F32, SyncerClient
from yeto.rl.core import CanonicalTensorSpec, build_rl_fragment_layout

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
NUMEL = 4
pytestmark = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")
# decoupled preset defaults (outer lr 0.7, momentum 0.9) when no option is set
DEFAULT_LR, DEFAULT_MU = 0.7, 0.9


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "outer-merge-test-key")


def default_layout(merge: int = MERGE_AVG) -> FragmentLayout:
    return FragmentLayout([Fragment(merge, [(f"actor.w{i}", NUMEL)]) for i in range(2)])


def _f32(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _from(data) -> np.ndarray:
    return np.frombuffer(bytes(data), dtype=np.float32).astype(np.float64)


# ---- numpy references --------------------------------------------------------------

def np_avg(deltas, weights):
    w = np.asarray(weights, dtype=np.float64)
    return sum(wi / w.sum() * d for wi, d in zip(w, deltas))


def np_rda(deltas, weights):
    """Radial-directional average: weighted mean norm times the unit weighted mean direction."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    norms = np.array([np.linalg.norm(d) for d in deltas])
    direction = sum(wi * d / n for wi, d, n in zip(w, deltas, norms))
    return float(w @ norms) * direction / np.linalg.norm(direction)


def np_weights(mode, pushes):
    """Learner weights: ``pushes`` holds (c_steps, c_tokens) per island."""
    if mode == "equal":
        return [1.0] * len(pushes)
    return [t * t / s for s, t in pushes]


class Reference:
    """Per-fragment master + Nesterov buffer: buf = mu*buf + g; theta += lr*(g + mu*buf)."""

    def __init__(self, layout, initial, lr, mu):
        self.layout, self.lr, self.mu = layout, lr, mu
        self.theta = [np.asarray(v, dtype=np.float64).copy() for v in initial]
        self.buf = [np.zeros_like(v) for v in self.theta]

    def step(self, fid, deltas, weights):
        fragment = self.layout.fragments[fid]
        merge = np_rda if fragment.merge_mode == MERGE_RDA else np_avg
        merged, off = np.zeros_like(self.theta[fid]), 0
        for _, numel in fragment.tensors:  # RDA acts per tensor slice
            merged[off:off + numel] = merge([d[off:off + numel] for d in deltas], weights)
            off += numel
        self.buf[fid] = self.mu * self.buf[fid] + merged
        self.theta[fid] = self.theta[fid] + self.lr * (merged + self.mu * self.buf[fid])
        return self.theta[fid]


# ---- the rig ------------------------------------------------------------------------

class Rig:
    """A launcher-built decoupled syncer plus one fake island client per learner."""

    def __init__(self, binary, tmp_path, islands, *options, total_steps=1, layout=None):
        from yeto import launcher
        from yeto.launcher import _prepare_rl_args

        self.layout = layout or default_layout()
        args = _launcher_args("ports", (
            "--rl-sync-preset", "decoupled", "--fragments", str(self.layout.num_fragments),
            "--pipeline", "1", "--total-steps", str(total_steps),
            "--local-rl-rounds-per-sync", "3", *options), gpu=",".join(["modal:2xh100"] * islands))
        args.controller = "local"
        _prepare_rl_args(args)
        command = launcher.role_syncer_commands(args, islands, binary=str(binary))["actor"]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.command = (command
                        .replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
                        .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
                        .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
        self.process = subprocess.Popen(["bash", "-c", self.command], stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
        self.clients = [SyncerClient(("127.0.0.1", port), i, self.layout, dtype=DTYPE_F32,
                                     num_streams=1, connect_timeout=10, max_reconnects=0,
                                     finalization_timeout=30) for i in range(islands)]
        self.permits = [{} for _ in self.clients]
        self.updates = [{} for _ in self.clients]
        self.output = ""

    def start(self, initial):
        for client in self.clients:
            client.start()
        for fid, values in enumerate(initial):
            self.clients[0].send_init(fid, _f32(values))

    def permit(self, island, step, timeout=20.0):
        """The island's PULL for round ``step``."""
        client, deadline = self.clients[island], time.monotonic() + timeout
        while time.monotonic() < deadline:
            client.check_health()
            for pull in client.drain_pulls():
                self.permits[island][pull.global_step] = pull
            if step in self.permits[island]:
                return self.permits[island][step]
            time.sleep(0.01)
        raise TimeoutError(f"island {island} got no PULL for round {step}")

    def push(self, island, step, delta, *, base, c_steps=1, c_tokens=1):
        pull = self.permit(island, step)
        self.clients[island].push_fragment(pull.fragment_id, pull.global_step, pull.round_attempt,
                                           base, 1, c_steps, c_tokens, _f32(delta))

    def bcast(self, island, fid, version, timeout=20.0):
        """The island's BCAST of ``(fragment, version)`` as float64, or TimeoutError."""
        client, deadline = self.clients[island], time.monotonic() + timeout
        while time.monotonic() < deadline:
            client.check_health()
            for update in client.drain_updates():
                self.updates[island][(update.fragment_id, update.version)] = _from(update.data)
            if (fid, version) in self.updates[island]:
                return self.updates[island][(fid, version)]
            time.sleep(0.01)
        raise TimeoutError(f"island {island} got no BCAST of fragment {fid} v{version}")

    def close(self):
        for client in self.clients:
            client.close()
        if self.process.poll() is None:
            self.process.terminate()
        self.output, _ = self.process.communicate(timeout=15)


@pytest.fixture
def make_rig(syncer_binary, tmp_path):
    rigs = []

    def make(islands, *options, **kwargs):
        workdir = tmp_path / f"rig{len(rigs)}"  # one checkpoint and tape per syncer
        workdir.mkdir()
        rigs.append(Rig(syncer_binary, workdir, islands, *options, **kwargs))
        return rigs[-1]

    yield make
    for rig in rigs:
        rig.close()


def _initial(layout):
    return [np.arange(f.numel, dtype=np.float64) * 0.25 + fid for fid, f in enumerate(layout.fragments)]


def _deltas(layout, fid, seed, count):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=layout.fragments[fid].numel).astype(np.float32).astype(np.float64)
            for _ in range(count)]


def _check(got, want):
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def _frag_of(rig, step):
    return (step - 1) % rig.layout.num_fragments


def _base_of(rig, step):
    """The version of the fragment merged in round ``step`` before that round."""
    return max(step - rig.layout.num_fragments, 0)


# ---- K < all islands ----------------------------------------------------------------

def test_quorum_below_all_islands_proceeds_without_the_slow_island(make_rig):
    rig = make_rig(3, "--rl-sync-quorum", "2", total_steps=2)  # the last round arrives as FINAL, not BCAST
    assert "--quorum 2" in rig.command
    initial = _initial(rig.layout)
    rig.start(initial)
    ref = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU)
    for step in (1, 2):  # island 2 never pushes
        fid = _frag_of(rig, step)
        deltas = _deltas(rig.layout, fid, step, 2)
        for island in (0, 1):
            rig.push(island, step, deltas[island], base=0)
        want = ref.step(fid, deltas, [1.0, 1.0])
        for island in (0, 1, 2):  # the slow island still receives the merge
            _check(rig.bcast(island, fid, step, timeout=10), want)
    # A late push of the slow island for a committed round is dropped, not fatal.
    rig.push(2, 1, _deltas(rig.layout, 0, 99, 1)[0], base=0)
    time.sleep(0.5)
    assert rig.process.poll() is None, "a late push must not stop the syncer"


def test_default_quorum_still_waits_for_every_island(make_rig):
    rig = make_rig(3)
    assert "--quorum 3" in rig.command
    rig.start(_initial(rig.layout))
    for island in (0, 1):
        rig.push(island, 1, _deltas(rig.layout, 0, 1, 2)[island], base=0)
    with pytest.raises(TimeoutError):
        rig.bcast(0, 0, 1, timeout=2.5)
    rig.push(2, 1, np.ones(NUMEL), base=0)
    assert rig.bcast(0, 0, 1, timeout=10).shape == (NUMEL,)


# ---- grace --------------------------------------------------------------------------

GRACE_MS = 1500


def test_grace_waits_for_a_straggler_that_arrives_inside_the_window(make_rig):
    rig = make_rig(2, "--rl-sync-quorum", "1", "--rl-sync-grace-ms", str(GRACE_MS))
    initial = _initial(rig.layout)
    rig.start(initial)
    deltas = _deltas(rig.layout, 0, 3, 2)
    rig.push(0, 1, deltas[0], base=0)
    time.sleep(0.5)  # well inside the grace window
    rig.push(1, 1, deltas[1], base=0)
    want = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU).step(0, deltas, [1.0, 1.0])
    _check(rig.bcast(0, 0, 1, timeout=10), want)


def test_grace_ends_within_its_bound_and_drops_a_later_straggler(make_rig):
    rig = make_rig(2, "--rl-sync-quorum", "1", "--rl-sync-grace-ms", str(GRACE_MS))
    initial = _initial(rig.layout)
    rig.start(initial)
    deltas = _deltas(rig.layout, 0, 4, 2)
    pushed = time.monotonic()
    rig.push(0, 1, deltas[0], base=0)
    got = rig.bcast(0, 0, 1, timeout=10)
    waited = time.monotonic() - pushed
    # First round has no step-time estimate, so the window is the full cap.
    assert GRACE_MS / 1000 * 0.9 <= waited <= GRACE_MS / 1000 + 3.0, waited
    want = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU).step(0, deltas[:1], [1.0])
    _check(got, want)
    rig.push(1, 1, deltas[1], base=0)  # too late: ignored
    time.sleep(0.5)
    assert rig.process.poll() is None


# ---- max base lag -------------------------------------------------------------------

def test_max_base_lag_accepts_a_push_within_the_lag_and_rejects_one_beyond_it(make_rig):
    # Two fragments: fragment 0 is merged in rounds 1, 3, 5, so its base versions
    # are 0, 1, 3. One round of lag is 2 versions; a lag of 3 versions is refused.
    rig = make_rig(2, "--rl-sync-max-base-lag", "1", total_steps=3)
    assert "--max-base-lag 2" in rig.command
    initial = _initial(rig.layout)
    rig.start(initial)
    ref = Reference(rig.layout, initial, DEFAULT_LR, DEFAULT_MU)
    stale_base = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}  # island 1's base per round: lag 0, 0, 1, 2, 2
    for step in range(1, 6):
        fid, base = _frag_of(rig, step), _base_of(rig, step)
        deltas = _deltas(rig.layout, fid, 10 + step, 2)
        rig.push(0, step, deltas[0], base=base)
        if step == 5:  # base 0 is 3 versions behind: refused, the round keeps waiting
            rig.push(1, step, deltas[1], base=0)
            with pytest.raises(TimeoutError):
                rig.bcast(0, fid, step, timeout=2.0)
            assert rig.process.poll() is None
        rig.push(1, step, deltas[1], base=stale_base[step])
        _check(rig.bcast(0, fid, step, timeout=10), ref.step(fid, deltas, [1.0, 1.0]))


def test_max_base_lag_zero_stops_the_syncer_on_a_stale_push(make_rig):
    rig = make_rig(2, total_steps=2)
    assert "--max-base-lag 0" in rig.command
    rig.start(_initial(rig.layout))
    for step in (1, 2):
        for island in (0, 1):
            rig.push(island, step, np.ones(NUMEL), base=0)
    rig.bcast(0, 0, 1)
    rig.bcast(0, 1, 2)
    rig.push(0, 3, np.ones(NUMEL), base=1)
    rig.push(1, 3, np.ones(NUMEL), base=0)
    deadline = time.monotonic() + 10
    while rig.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert rig.process.poll() not in (None, 0)
    rig.close()
    assert "rejected_stale_updates" in rig.output


# ---- weights, outer optimizer, RDA --------------------------------------------------

def test_tokens2_over_steps_weights_match_numpy(make_rig):
    rig = make_rig(3, "--rl-sync-learner-weight", "tokens2-over-steps",
                   "--rl-sync-outer-lr", "1.0", "--rl-sync-outer-momentum", "0.0")
    assert "--learner-weight tokens2-over-steps" in rig.command
    initial = _initial(rig.layout)
    rig.start(initial)
    deltas = _deltas(rig.layout, 0, 5, 3)
    pushes = [(2, 10), (3, 30), (1, 5)]  # (c_steps, c_tokens) -> weights 50, 300, 25
    for island, (c_steps, c_tokens) in enumerate(pushes):
        rig.push(island, 1, deltas[island], base=0, c_steps=c_steps, c_tokens=c_tokens)
    weights = np_weights("tokens2-over-steps", pushes)
    want = Reference(rig.layout, initial, 1.0, 0.0).step(0, deltas, weights)
    equal = Reference(rig.layout, initial, 1.0, 0.0).step(0, deltas, [1.0] * 3)
    assert not np.allclose(want, equal, atol=1e-2), "test data must tell the two weightings apart"
    _check(rig.bcast(0, 0, 1), want)


@pytest.mark.parametrize(("lr", "mu"), [(0.35, 0.6), (1.0, 0.0)])
def test_outer_lr_and_momentum_follow_numpy_nesterov(make_rig, lr, mu):
    rig = make_rig(2, "--rl-sync-outer-lr", str(lr), "--rl-sync-outer-momentum", str(mu),
                   total_steps=4)  # rounds 1-6 are checked as BCASTs; the last one is FINAL
    initial = _initial(rig.layout)
    rig.start(initial)
    ref = Reference(rig.layout, initial, lr, mu)
    for step in range(1, 7):
        fid = _frag_of(rig, step)
        deltas = _deltas(rig.layout, fid, 20 + step, 2)
        for island in (0, 1):
            rig.push(island, step, deltas[island], base=_base_of(rig, step))
        _check(rig.bcast(1, fid, step), ref.step(fid, deltas, [1.0, 1.0]))


def _matrix_layout(matrix_merge):
    names = ["embed_tokens.lora_A.weight", "layers.0.q.lora_A.weight", "layers.0.k.lora_A.weight",
             "layers.1.q.lora_A.weight", "layers.1.k.lora_A.weight"]
    specs = [CanonicalTensorSpec(n, (2, 3), "float32", 6) for n in names]
    return build_rl_fragment_layout(specs, 2, matrix_merge=matrix_merge)


@pytest.mark.parametrize("weight", ["equal", "tokens2-over-steps"])
def test_rda_and_avg_match_their_numpy_references(make_rig, weight):
    results = {}
    pushes = [(2, 10), (3, 30), (1, 5)]
    for merge in ("avg", "rda"):
        layout = _matrix_layout(merge)
        rig = make_rig(3, "--rl-sync-matrix-merge", merge, "--rl-sync-learner-weight", weight,
                       "--rl-sync-outer-lr", "1.0", "--rl-sync-outer-momentum", "0.0",
                       total_steps=2, layout=layout)
        initial = _initial(layout)
        rig.start(initial)
        ref = Reference(layout, initial, 1.0, 0.0)
        for step in (1, 2):
            fid = _frag_of(rig, step)
            deltas = _deltas(layout, fid, 30 + step, 3)
            for island, (c_steps, c_tokens) in enumerate(pushes):
                rig.push(island, step, deltas[island], base=0, c_steps=c_steps, c_tokens=c_tokens)
            got = rig.bcast(0, fid, step)
            _check(got, ref.step(fid, deltas, np_weights(weight, pushes)))
            results[(merge, fid)] = got
    modes = {fid: [f.merge_mode for f in [_matrix_layout("rda").fragments[fid]]][0] for fid in (0, 1)}
    # Embedding fragments stay averaged even when RDA is requested.
    for fid in (0, 1):
        same = np.allclose(results[("avg", fid)], results[("rda", fid)], atol=1e-4)
        assert same == (modes[fid] == MERGE_AVG), (fid, modes)
    assert MERGE_RDA in modes.values() and MERGE_AVG in modes.values()
