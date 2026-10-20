"""CPU micro-benchmark of the full-parameter sync boundary (S26 E3).

Runs the real Rust syncer on localhost with two in-process learners and a
synthetic f32 model, and times each strict sweep from the first push to the
last updated fragment received by every learner. It compares the serial
profile (pipeline 1, checkpoint on the scheduler) with the streamed profile
(pipeline P, ``--async-checkpoint``).

These are local CPU numbers: loopback transfer and a local NVMe checkpoint.
They show where the boundary time goes on this machine; they do not predict
GPU-cluster or WAN timings.

    python tools/bench_sync_boundary.py --floats-per-fragment 33554432 --sweeps 4
"""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from yeto.fragments import MERGE_RDA, Fragment, FragmentLayout
from yeto.protocol import DTYPE_F32, SyncerClient

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _syncer(binary: Path, workdir: Path, *, port: int, learners: int, fragments: int,
            sweeps: int, streamed: bool) -> subprocess.Popen:
    command = [
        str(binary), "--port", str(port), "--learners", str(learners),
        "--quorum", str(learners), "--grace-ms", "0",
        "--pipeline", str(fragments if streamed else 1),
        "--sync-interval-steps", "0", "--delta-correction", "none",
        "--total-steps", str(sweeps * fragments), "--policy-sweep-fragments", str(fragments),
        "--outer-lr", "1", "--outer-momentum", "0", "--max-base-lag", "0",
        "--learner-weight", "equal", "--checkpoint-path", str(workdir / "state.ckpt"),
        "--checkpoint-every", "1", "--event-tape", str(workdir / "tape.jsonl"),
        "--allow-unauthenticated-islands",  # loopback-only benchmark
    ]
    if streamed:
        command.append("--async-checkpoint")
    return subprocess.Popen(command, stdout=open(workdir / "syncer.log", "w"),
                            stderr=subprocess.STDOUT)


def _learner(learner_id: int, port: int, layout: FragmentLayout, numel: int, sweeps: int,
             timings: list[list[float]], errors: list[BaseException]) -> None:
    fragments = len(layout.fragments)
    client = SyncerClient(("127.0.0.1", port), learner_id, layout, dtype=DTYPE_F32,
                          num_streams=1, connect_timeout=30, max_reconnects=0,
                          finalization_timeout=600)
    rng = np.random.default_rng(learner_id)
    try:
        client.start()
        if learner_id == 0:
            for fid in range(fragments):
                client.send_init(fid, rng.standard_normal(numel, dtype=np.float32).tobytes())
        deltas = [rng.standard_normal(numel, dtype=np.float32).tobytes() for _ in range(fragments)]
        carried: list = []  # a PULL of the next sweep can arrive before this sweep's BCAST
        for sweep in range(1, sweeps + 1):
            pushed: set[int] = set()
            updated: set[int] = set()
            first_push = None
            while len(updated) < fragments or len(pushed) < fragments:
                client.check_health()
                permits, carried = carried + client.drain_pulls(), []
                for permit in permits:
                    if permit.fragment_id in pushed:
                        carried.append(permit)
                        continue
                    base = 0 if sweep == 1 else (sweep - 2) * fragments + permit.fragment_id + 1
                    first_push = first_push or time.monotonic()
                    client.push_fragment(permit.fragment_id, permit.global_step,
                                         permit.round_attempt, base, sweep, 1, 1,
                                         deltas[permit.fragment_id])
                    pushed.add(permit.fragment_id)
                for update in client.drain_updates():
                    if update.version > (sweep - 1) * fragments:
                        updated.add(update.fragment_id)
                if sweep == sweeps and len(pushed) == fragments:
                    break  # the terminal sweep is delivered by FINAL, not BCAST
                time.sleep(0.001)
            if sweep == sweeps:
                manifest, _ = client.wait_for_final_fragments(timeout=600)
                timings[learner_id].append(time.monotonic() - first_push)
                client.acknowledge_finalization(manifest)
            else:
                timings[learner_id].append(time.monotonic() - first_push)
    except BaseException as error:  # noqa: BLE001 - reported by the caller
        errors.append(error)
    finally:
        client.close()


def run(binary: Path, *, streamed: bool, learners: int, fragments: int, numel: int,
        sweeps: int) -> dict:
    layout = FragmentLayout([Fragment(MERGE_RDA, [(f"w{i}", numel)]) for i in range(fragments)])
    with tempfile.TemporaryDirectory(prefix="e3-bench-", dir="/tmp") as tmp:
        workdir = Path(tmp)
        port = _free_port()
        process = _syncer(binary, workdir, port=port, learners=learners, fragments=fragments,
                          sweeps=sweeps, streamed=streamed)
        timings: list[list[float]] = [[] for _ in range(learners)]
        errors: list[BaseException] = []
        threads = [threading.Thread(target=_learner, args=(i, port, layout, numel, sweeps,
                                                           timings, errors))
                   for i in range(learners)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        code = process.wait(timeout=120)
        if errors or code != 0:
            raise RuntimeError(f"benchmark failed (syncer exit {code}): {errors!r}; "
                               f"log: {(workdir / 'syncer.log').read_text()[-2000:]}")
        tape = [json.loads(line) for line in (workdir / "tape.jsonl").read_text().splitlines()
                if '"sync_ms"' in line]
    # Sweep 1 includes connection set-up; the last sweep ends with FINAL.
    steady = [max(per[i] for per in timings) for i in range(1, sweeps - 1)] or \
        [max(per[i] for per in timings) for i in range(sweeps)]
    return {
        "profile": "streamed" if streamed else "serial",
        "boundary_s_per_sweep": [round(x, 3) for x in steady],
        "boundary_s_median": round(statistics.median(steady), 3),
        "checkpoint_ms_median": statistics.median(r["checkpoint_ms"] for r in tape),
        "sync_ms_median": statistics.median(r["sync_ms"] for r in tape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, default=ROOT / "syncer/target/release/yeto-syncer")
    parser.add_argument("--learners", type=int, default=2)
    parser.add_argument("--fragments", type=int, default=2)
    parser.add_argument("--floats-per-fragment", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--sweeps", type=int, default=4)
    args = parser.parse_args()
    results = [run(args.binary, streamed=streamed, learners=args.learners,
                   fragments=args.fragments, numel=args.floats_per_fragment, sweeps=args.sweeps)
               for streamed in (False, True)]
    print(json.dumps({"note": "CPU localhost numbers; not GPU or WAN timings",
                      "learners": args.learners, "fragments": args.fragments,
                      "floats_per_fragment": args.floats_per_fragment, "results": results},
                     indent=2))


if __name__ == "__main__":
    main()
