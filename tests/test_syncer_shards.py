"""Fragment-sharded syncer head (S27 P4b) with real syncer processes.

N syncer shards run on localhost; fake islands (threads with a
``ShardedSyncerClient``) answer every PULL with a deterministic learner
delta. The final parameters of every shard are checked against a numpy
replay of the merges the shards recorded on their event tapes
(responders, weights, step, fragment), so the test covers routing, the
shard step/fragment plan, the streaming weighted fold and finalization.
No GPU, no torch, no Ray.
"""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from yeto.fragments import MERGE_AVG, MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import DTYPE_F32, SyncerClient
from yeto.syncer_shards import (
    ShardedSyncerClient,
    owned_range,
    owner_of,
    parse_syncer_addrs,
)

ROOT = Path(__file__).resolve().parent.parent
OUTER_LR = 0.7
MOMENTUM = 0.9


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run(["cargo", "build", "-q"], cwd=ROOT / "syncer", check=True)
    binary = ROOT / "syncer/target/debug/yeto-syncer"
    assert binary.is_file()
    return binary


def free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    try:
        for sock in socks:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in socks]
    finally:
        for sock in socks:
            sock.close()


def layout() -> FragmentLayout:
    """Five fragments of mixed merge modes and tensor counts."""
    return FragmentLayout([
        Fragment(MERGE_AVG, [("model.embed_tokens.weight", 24)]),
        Fragment(MERGE_RDA, [("model.layers.0.a", 16), ("model.layers.0.b", 5)]),
        Fragment(MERGE_RDA, [("model.layers.1.a", 33)]),
        Fragment(MERGE_RDA, [("model.layers.2.a", 7), ("model.layers.2.b", 9), ("model.layers.2.c", 4)]),
        Fragment(MERGE_RDA, [("model.layers.3.a", 12)]),
    ])


def initial(fid: int, numel: int) -> np.ndarray:
    return (np.arange(numel, dtype=np.float32) * 0.01 + fid).astype(np.float32)


def learner_delta(learner: int, fid: int, step: int, numel: int) -> np.ndarray:
    """θ_local − anchor sent by ``learner`` for (fragment, step)."""
    rng = np.random.default_rng(1000 * learner + 37 * fid + step)
    return (rng.standard_normal(numel) * 0.05).astype(np.float32)


def counters(learner: int) -> tuple[int, int]:
    """(c_steps, c_tokens): distinct tokens²/steps weights per learner."""
    return learner + 1, 100 * (learner + 2)


def test_python_ownership_mirrors_the_rust_plan():
    for count in range(1, 5):
        for fragments in range(count, 10):
            ranges = [owned_range(i, count, fragments) for i in range(count)]
            assert ranges[0].start == 0 and ranges[-1].stop == fragments
            assert all(a.stop == b.start for a, b in zip(ranges, ranges[1:]))
            assert all(len(r) > 0 for r in ranges)
            for fid in range(fragments):
                assert fid in ranges[owner_of(fid, count, fragments)]
    assert owned_range(2, 3, 7) == range(4, 7)  # same as shard.rs test
    with pytest.raises(ValueError):
        owned_range(0, 3, 2)
    assert parse_syncer_addrs("h:1,h2:2") == [("h", 1), ("h2", 2)]
    with pytest.raises(ValueError):
        parse_syncer_addrs("h")


def test_sharded_client_refuses_misrouted_fragments():
    lay = layout()
    client = SyncerClient(("127.0.0.1", 1), 0, lay, DTYPE_F32, 0, owned_fragments=range(2, 5))
    with pytest.raises(ValueError, match="owning fragments 2..4"):
        client.send_init(0, initial(0, 24).tobytes())
    with pytest.raises(ValueError, match="owning fragments"):
        client.push_fragment(1, 1, 1, 0, 1, 1, 1, np.zeros(21, np.float32).tobytes())
    with pytest.raises(ValueError):
        SyncerClient(("127.0.0.1", 1), 0, lay, DTYPE_F32, 0, owned_fragments=range(3, 9))


def launch_shards(binary: Path, tmp_path: Path, ports: list[int], *, learners: int,
                  quorum: int, total_steps: int, fold: str) -> list[subprocess.Popen]:
    procs = []
    count = len(ports)
    for index, port in enumerate(ports):
        args = [
            str(binary),
            "--port", str(port),
            "--learners", str(learners),
            "--quorum", str(quorum),
            "--grace-ms", "50",
            "--quorum-timeout-s", "20",
            "--total-steps", str(total_steps),
            "--pipeline", "2",
            "--sync-interval-steps", "0",
            "--delta-correction", "none",
            "--outer-lr", str(OUTER_LR),
            "--outer-momentum", str(MOMENTUM),
            "--learner-weight", "tokens2-over-steps",
            "--weighted-fold", fold,
            "--checkpoint-path", str(tmp_path / f"state{index}.ckpt"),
            "--checkpoint-every", "1",
            "--final-state", str(tmp_path / f"final{index}.bin"),
            "--event-tape", str(tmp_path / f"tape{index}.jsonl"),
            "--allow-unauthenticated-islands",
        ]
        if count > 1:
            args += ["--shard-index", str(index), "--shard-count", str(count)]
        procs.append(subprocess.Popen(
            args, stdout=open(tmp_path / f"syncer{index}.log", "w"), stderr=subprocess.STDOUT))
    return procs


def run_island(addrs, learner: int, lay: FragmentLayout, result: dict, errors: list) -> None:
    try:
        if len(addrs) > 1:
            client = ShardedSyncerClient(addrs, learner, lay, DTYPE_F32, num_streams=1,
                                         connect_timeout=30.0)
        else:
            client = SyncerClient(addrs[0], learner, lay, DTYPE_F32, 1, connect_timeout=30.0)
        client.start()
        if learner == 0:
            for fid, fragment in enumerate(lay.fragments):
                client.send_init(fid, initial(fid, fragment.numel).tobytes())
        versions: dict[int, int] = {}
        pending = []
        local_step = 0
        deadline = time.monotonic() + 120
        while not client.finalizing.is_set():
            assert time.monotonic() < deadline, "island did not reach finalization"
            client.check_health()
            for update in client.drain_updates():
                versions[update.fragment_id] = max(versions.get(update.fragment_id, 0), update.version)
            pending.extend(client.drain_pulls())
            still = []
            for pull in pending:
                fid = pull.fragment_id
                if fid not in versions:  # PULL overtook the initial BCAST
                    still.append(pull)
                    continue
                local_step += 1
                c_steps, c_tokens = counters(learner)
                delta = learner_delta(learner, fid, pull.global_step, lay.fragments[fid].numel)
                client.push_fragment(fid, pull.global_step, pull.round_attempt, versions[fid],
                                     local_step, c_steps, c_tokens, delta.tobytes())
            pending = still
            time.sleep(0.002)
        manifest, fragments = client.wait_for_final_fragments(60.0)
        result[learner] = {
            f.fragment_id: np.frombuffer(bytes(f.data), dtype=np.float32).copy() for f in fragments
        }
        result[f"manifest{learner}"] = manifest
        client.acknowledge_finalization(manifest)
        client.close()
    except BaseException as error:  # surfaced by the test thread
        errors.append(error)


def read_final_state(path: Path) -> list[np.ndarray]:
    data = path.read_bytes()
    (count,) = struct.unpack_from("<I", data, 0)
    off, out = 4, []
    for _ in range(count):
        (numel,) = struct.unpack_from("<Q", data, off)
        off += 8
        out.append(np.frombuffer(data, dtype="<f4", count=numel, offset=off).copy())
        off += 4 * numel
    return out


def reference_merge(fragment: Fragment, grads: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """f64 Avg/RDA of signed outer gradients, per tensor."""
    w = np.asarray(weights, dtype=np.float64)
    out, off = [], 0
    for _, numel in fragment.tensors:
        g = np.stack([x[off:off + numel].astype(np.float64) for x in grads])
        avg = (w[:, None] * g).sum(0) / w.sum()
        if fragment.merge_mode == MERGE_RDA:
            norms = np.linalg.norm(g, axis=1)
            radial = (w * norms).sum() / w.sum()
            unit = np.where(norms[:, None] > 0, g / np.where(norms > 0, norms, 1)[:, None], 0)
            direction = (w[:, None] * unit).sum(0) / w.sum()
            dn = np.linalg.norm(direction)
            avg = avg if dn < 1e-12 else direction * radial / dn
        out.append(avg)
        off += numel
    return np.concatenate(out)


def replay(lay: FragmentLayout, tapes: list[Path]) -> list[np.ndarray]:
    """Re-run every recorded merge in step order with an f64 Nesterov step."""
    params = [initial(fid, f.numel).astype(np.float64) for fid, f in enumerate(lay.fragments)]
    momentum = [np.zeros_like(p) for p in params]
    records = []
    for tape in tapes:
        for line in tape.read_text().splitlines():
            record = json.loads(line)
            if "fragment" in record and "responders" in record:
                records.append(record)
    records.sort(key=lambda r: r["step"])
    for record in records:
        fid, step = record["fragment"], record["step"]
        fragment = lay.fragments[fid]
        grads, weights = [], []
        for responder in sorted(record["responders"], key=lambda r: r["id"]):
            learner = responder["id"]
            c_steps, c_tokens = counters(learner)
            grads.append(-learner_delta(learner, fid, step, fragment.numel))
            weights.append(c_tokens * c_tokens / c_steps)
        delta = reference_merge(fragment, grads, weights)
        momentum[fid] = MOMENTUM * momentum[fid] + delta
        params[fid] = params[fid] - OUTER_LR * (delta + MOMENTUM * momentum[fid])
    return params, records


@pytest.mark.parametrize("shards,quorum,fold", [
    (1, 2, "streaming"),
    (2, 2, "streaming"),
    (3, 1, "streaming"),
    (2, 1, "buffered"),
])
def test_sharded_head_matches_numpy_replay(syncer_binary, tmp_path, shards, quorum, fold):
    lay = layout()
    learners, total_steps = 2, 14
    ports = free_ports(shards)
    procs = launch_shards(syncer_binary, tmp_path, ports, learners=learners, quorum=quorum,
                          total_steps=total_steps, fold=fold)
    addrs = [("127.0.0.1", port) for port in ports]
    result: dict = {}
    errors: list = []
    threads = [threading.Thread(target=run_island, args=(addrs, m, lay, result, errors), daemon=True)
               for m in range(learners)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(150)
        assert not errors, errors
        for proc in procs:
            assert proc.wait(timeout=60) == 0
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
        logs = "\n".join((tmp_path / f"syncer{i}.log").read_text()[-3000:] for i in range(shards))
    assert not errors, logs

    # Every step 1..T ran exactly once, on its owning shard, on the planned fragment.
    expected, records = replay(lay, [tmp_path / f"tape{i}.jsonl" for i in range(shards)])
    assert [r["step"] for r in records] == list(range(1, total_steps + 1)), logs
    for record in records:
        shard = (record["step"] - 1) % shards
        owned = owned_range(shard, shards, lay.num_fragments)
        k = (record["step"] - 1) // shards
        assert record["fragment"] == owned.start + k % len(owned)

    # Head state: each shard holds only its own fragments; together they
    # equal the numpy replay up to f32 rounding.
    for index in range(shards):
        state = read_final_state(tmp_path / f"final{index}.bin")
        owned = owned_range(index, shards, lay.num_fragments)
        for fid, values in enumerate(state):
            if fid in owned:
                np.testing.assert_allclose(values, expected[fid], rtol=2e-5, atol=2e-6)
            else:
                assert values.size == 0
    # Islands applied the same union cut.
    for learner in range(learners):
        cut = result[learner]
        assert sorted(cut) == list(range(lay.num_fragments))
        for fid, values in cut.items():
            np.testing.assert_allclose(values, expected[fid], rtol=2e-5, atol=2e-6)
        manifest = result[f"manifest{learner}"]
        assert manifest.global_step == total_steps
        assert all(v > 0 for v in manifest.versions)
    # Checkpoints are split: unowned fragments are stored empty.
    sizes = [(tmp_path / f"state{i}.ckpt").stat().st_size for i in range(shards)]
    model_bytes = 2 * 4 * sum(f.numel for f in lay.fragments)
    assert sum(sizes) - model_bytes < 400 * shards


# -- launcher: --syncer-shards / --syncer-weighted-fold ---------------------

LAUNCH_ARGS = [
    "--gpu", "aws:8xa100@us-east-2,aws:8xa100@us-west-2",
    "--model", "gemma4",
    "--model-revision", "a" * 40,
    "--data", "org/ds",
    "--data-revision", "b" * 40,
]


def test_default_launch_keeps_the_single_syncer_command():
    import yeto.cli as cli
    from yeto.launch import syncer as launch_syncer

    args = cli.parse_args(LAUNCH_ARGS)
    cmd = launch_syncer.syncer_command(args, 2)
    assert "--shard" not in cmd and "--weighted-fold" not in cmd and "pids=" not in cmd
    assert cmd.endswith("--event-tape ~/yeto-tape.jsonl --allow-unauthenticated-islands")
    assert launch_syncer.syncer_shard_addrs("10.0.0.1", args) == "10.0.0.1:29400"
    assert launch_syncer.syncer_ports(args) == [29400]
    assert launch_syncer.ufw_setup(*launch_syncer.syncer_shard_ports(args)) == launch_syncer.ufw_setup()


def test_sharded_launch_runs_one_syncer_per_shard_and_fails_fast(tmp_path):
    import os

    import yeto.cli as cli
    from yeto.launch import syncer as launch_syncer

    args = cli.parse_args(LAUNCH_ARGS + ["--syncer-shards", "3", "--fragments", "8",
                                         "--syncer-weighted-fold", "streaming"])
    cmd = launch_syncer.syncer_command(args, 2, binary="echo")
    for index in range(3):
        assert f"--port {29400 + index}" in cmd
        assert f"--shard-index {index} --shard-count 3" in cmd
    assert "--checkpoint-path ~/yeto-state.ckpt " in cmd
    assert "--checkpoint-path ~/yeto-state.shard2.ckpt" in cmd
    assert "--event-tape ~/yeto-tape.shard1.jsonl" in cmd
    assert cmd.count("--weighted-fold streaming") == 3
    assert launch_syncer.syncer_shard_addrs("h", args) == "h:29400,h:29401,h:29402"
    assert launch_syncer.syncer_ports(args) == [29400, 29401, 29402]
    assert "ufw allow 29402/tcp" in launch_syncer.ufw_setup(*launch_syncer.syncer_shard_ports(args))
    # The supervisor runs every shard and exits 0 only when all did.
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}
    done = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.count("--shard-count 3") == 3
    # One failing shard stops the job with its exit status.
    failing = launch_syncer._sharded_supervisor(["sleep 30", "exit 7", "sleep 30"])
    started = time.monotonic()
    done = subprocess.run(["bash", "-c", failing], capture_output=True, text=True, timeout=20)
    assert done.returncode == 7 and time.monotonic() - started < 10


def test_sharded_launch_refuses_unsupported_modes():
    import yeto.cli as cli
    from yeto.launch import syncer as launch_syncer

    args = cli.parse_args(LAUNCH_ARGS + ["--syncer-shards", "2"])
    args.training_mode = "rl"
    with pytest.raises(ValueError, match="training-mode rl"):
        launch_syncer.syncer_shard_count(args)
    args = cli.parse_args(LAUNCH_ARGS + ["--syncer-shards", "9", "--fragments", "8"])
    with pytest.raises(ValueError, match="--fragments >= 9"):
        launch_syncer.syncer_shard_count(args)
    args = cli.parse_args(LAUNCH_ARGS + ["--syncer-shards", "0"])
    with pytest.raises(ValueError):
        launch_syncer.syncer_shard_count(args)
