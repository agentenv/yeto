#!/usr/bin/env python
"""Streaming per-fragment update autocorrelation (rho) from syncer probe captures.

Watches a --probe-capture-dir being populated by the syncer
(states/state_before_step_%08d.ckpt via atomic tmp+rename, candidates/*.f32,
index.jsonl), diffs consecutive state snapshots to recover the outer update
applied at each step, accumulates energy-weighted lag-1..N autocorrelation of
same-fragment consecutive updates, and deletes consumed capture files as it
goes so a full-parameter run's captures never accumulate on disk.

For a mu=0 outer optimizer the state diff at step t is exactly -eta times the
merged pseudo-gradient, so the (scale-invariant) autocorrelation of the
updates equals the pseudo-gradient lag kernel rho_k used by the two-term
amplification law. Candidates are deleted unread: rho needs only the states.

Exit: once --done-file exists and no unprocessed states remain, the summary
JSON is written and remaining capture files are removed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import time
from pathlib import Path

import numpy as np

CKPT_MAGIC = 0xD170_5A7E


def parse_state_params(path: Path) -> tuple[int, list[np.ndarray]]:
    """Return (global_step, per-fragment f32 param arrays); momentum skipped."""
    data = path.read_bytes()  # written via tmp+rename, so always complete
    off = 0
    (magic,) = struct.unpack_from("<I", data, off)
    off += 4
    if magic != CKPT_MAGIC:
        raise ValueError(f"{path}: bad checkpoint magic 0x{magic:08X}")
    (gstep,) = struct.unpack_from("<Q", data, off)
    off += 8
    (nfrag,) = struct.unpack_from("<I", data, off)
    off += 4
    frags: list[np.ndarray] = []
    for fid in range(nfrag):
        _ver, numel = struct.unpack_from("<QQ", data, off)
        off += 16
        params = np.frombuffer(data, dtype="<f4", count=numel, offset=off).copy()
        off += 4 * numel  # params
        off += 4 * numel  # momentum (skipped)
        frags.append(params)
    return gstep, frags


def state_step(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


class LagAccumulator:
    def __init__(self, max_lag: int):
        self.max_lag = max_lag
        # lag -> [sum_dot, sum_normprod, n, sum_cos, sum_cos2]
        self.acc = {k: [0.0, 0.0, 0, 0.0, 0.0] for k in range(1, max_lag + 1)}

    def add(self, lag: int, dot: float, norm_a: float, norm_b: float) -> None:
        a = self.acc[lag]
        a[0] += dot
        a[1] += norm_a * norm_b
        a[2] += 1
        if norm_a > 0 and norm_b > 0:
            c = dot / (norm_a * norm_b)
            a[3] += c
            a[4] += c * c

    def summary(self) -> dict:
        out = {}
        for k, (sd, sp, n, sc, sc2) in self.acc.items():
            if n == 0:
                out[f"lag{k}"] = {"n": 0}
                continue
            mean_cos = sc / n
            var = max(sc2 / n - mean_cos * mean_cos, 0.0)
            out[f"lag{k}"] = {
                "rho_energy": sd / sp if sp > 0 else None,
                "cos_mean": mean_cos,
                "cos_std": math.sqrt(var),
                "n": n,
            }
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--done-file",
        type=Path,
        required=True,
        help="finalize once this file exists and no unprocessed states remain",
    )
    ap.add_argument("--max-lag", type=int, default=4)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=25.0,
        help="if free disk drops below this, drop oldest unprocessed states",
    )
    args = ap.parse_args()

    states_dir = args.capture_dir / "states"
    cand_dir = args.capture_dir / "candidates"

    prev_step: int | None = None
    prev_frags: list[np.ndarray] | None = None
    # fragment id -> list of (step, update f32 array), newest last, len<=max_lag
    history: dict[int, list[tuple[int, np.ndarray]]] = {}
    per_frag: dict[int, LagAccumulator] = {}
    overall = LagAccumulator(args.max_lag)
    norms: list[list] = []
    processed = 0
    updates = 0
    gaps = 0
    dropped = 0

    def record_update(fid: int, step: int, u: np.ndarray) -> None:
        nonlocal updates
        updates += 1
        un = float(np.linalg.norm(u.astype(np.float64)))
        if len(norms) < 20000:
            norms.append([step, fid, un])
        hist = history.setdefault(fid, [])
        acc = per_frag.setdefault(fid, LagAccumulator(args.max_lag))
        u64 = u.astype(np.float64)
        for lag in range(1, args.max_lag + 1):
            if len(hist) < lag:
                break
            pstep, pu = hist[-lag]
            dot = float(np.dot(u64, pu.astype(np.float64)))
            pn = float(np.linalg.norm(pu.astype(np.float64)))
            acc.add(lag, dot, un, pn)
            overall.add(lag, dot, un, pn)
        hist.append((step, u))
        if len(hist) > args.max_lag:
            hist.pop(0)

    def flush(final: bool) -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "rho_sidecar_v1",
            "final": final,
            "states_processed": processed,
            "updates_recorded": updates,
            "step_gaps": gaps,
            "states_dropped_for_disk": dropped,
            "max_lag": args.max_lag,
            "overall": overall.summary(),
            "per_fragment": {
                str(fid): acc.summary() for fid, acc in sorted(per_frag.items())
            },
            "update_norms": norms,
        }
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(args.out)

    last_flush = time.monotonic()
    while True:
        # Candidates are never needed for rho: delete on sight.
        if cand_dir.is_dir():
            for c in cand_dir.glob("*.f32"):
                try:
                    c.unlink()
                except OSError:
                    pass
        files = (
            sorted(states_dir.glob("state_before_step_*.ckpt"), key=state_step)
            if states_dir.is_dir()
            else []
        )
        # Disk pressure: drop oldest unprocessed states rather than crash the run.
        if files and args.capture_dir.exists():
            free_gb = shutil.disk_usage(args.capture_dir).free / 1e9
            while free_gb < args.min_free_gb and len(files) > 2:
                victim = files.pop(0)
                try:
                    victim.unlink()
                except OSError:
                    pass
                dropped += 1
                free_gb = shutil.disk_usage(args.capture_dir).free / 1e9

        progress = False
        for f in files:
            step = state_step(f)
            try:
                _gstep, frags = parse_state_params(f)
            except (ValueError, OSError) as exc:
                print(f"[rho-sidecar] skipping {f.name}: {exc}", flush=True)
                try:
                    f.unlink()
                except OSError:
                    pass
                continue
            try:
                f.unlink()
            except OSError:
                pass
            processed += 1
            progress = True
            if prev_frags is not None and prev_step is not None:
                if step == prev_step + 1 and len(frags) == len(prev_frags):
                    changed = []
                    for fid, (a, b) in enumerate(zip(prev_frags, frags)):
                        if a.shape != b.shape:
                            changed = []
                            break
                        if not np.array_equal(a, b):
                            changed.append(fid)
                    # Exactly one fragment commits per syncer step; a diff
                    # touching several means we misread the stream — skip it.
                    if len(changed) == 1:
                        fid = changed[0]
                        record_update(fid, prev_step, frags[fid] - prev_frags[fid])
                    else:
                        gaps += 1
                else:
                    gaps += 1
            prev_step, prev_frags = step, frags
        if time.monotonic() - last_flush > 60:
            flush(final=False)
            last_flush = time.monotonic()
        if not progress:
            if args.done_file.exists():
                flush(final=True)
                # Clean any stragglers so the arm dir stays lean.
                if args.capture_dir.exists():
                    shutil.rmtree(args.capture_dir, ignore_errors=True)
                print(
                    f"[rho-sidecar] done: {processed} states, {updates} updates, "
                    f"{gaps} gaps, {dropped} dropped",
                    flush=True,
                )
                return 0
            time.sleep(args.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
