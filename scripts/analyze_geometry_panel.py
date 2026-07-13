#!/usr/bin/env python
"""Per-commit geometry panel for the EXP2.39 mediation-closure experiment.

For each run (cell), extract per-commit geometry from the event tape and,
when a syncer probe capture is available, from an exact replay of the
production merge:

  ||g_t||     merged-delta norm (tape `gnorm`; cross-checked against the
              RDA replay when a capture exists)
  rho_t       cos(g_t, g_{t-1}) between consecutive SAME-FRAGMENT merged
              deltas (capture replay)
  c_t, r_t    realized buffer/delta geometry of the outer Nesterov buffer
              b_{t-1} (from the pre-commit state checkpoint) against the
              merged delta: c_t = <b, g>/||g||^2, r_t = ||b - c_t g||/||g||
  agree_t     worker agreement: mean_i cos(g_t^i, mean_j g_t^j) over the
              per-learner deltas of the commit
  lambda_hat  secant curvature proxy between consecutive same-fragment
              commits: [<g_t - g_{t-1}, theta_t - theta_{t-1}>]_+ /
              ||theta_t - theta_{t-1}||^2
  r2_lambda   curvature-weighted transverse candidate r_t^2 * lambda_hat

The capture replay mirrors syncer/src/merge.rs `merge_rda` (weighted, f32
coefficient op order) and verifies itself end-to-end: the outer step
replayed from the pre-commit checkpoint (params + momentum buffer, the
configured mu/eta, and the optional --delta-norm-ref rescale) must
reproduce the next same-fragment anchor checkpoint (max |diff| reported;
bit-exact for mu=0 unscaled arms).

Run specs are repeated `--run` flags with comma-separated key=value pairs:

  --run label=innerlr-hi-h64-mu09,tape=.../tape.jsonl,eta=0.175,mu=0.9,\
        capture=.../syncer_probe,adapter=.../adapter_model.safetensors,\
        results=.../results.jsonl,delta_norm_ref=0

`capture`, `adapter`, `results`, and `delta_norm_ref` are optional; without
a capture only the tape columns are produced. Writes per-run
`<label>_per_commit.jsonl` plus `panel.json` / `panel.md` to --out
(default experiment-results/EXP2/geometry-panel/).
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CKPT_MAGIC = 0xD1705A7E  # syncer state snapshot magic, see syncer/src/state.rs
CAND_RE = re.compile(r"candidate_step_(\d{8})_fragment_(\d{4})_learner_(\d{4})\.f32$")


def parse_run_spec(spec: str) -> dict:
    out: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"malformed --run item {part!r} in {spec!r}")
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    for required in ("label", "tape", "eta", "mu"):
        if required not in out:
            raise SystemExit(f"--run spec missing {required!r}: {spec!r}")
    if "capture" in out and "adapter" not in out:
        raise SystemExit(f"--run spec with capture needs adapter=: {spec!r}")
    return out


def load_tape(path: Path) -> dict[tuple[int, int], dict]:
    """{(step, fragment): record} — completion order in the file is not
    guaranteed to be step order under pipelining."""
    records = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records[(int(rec["step"]), int(rec["fragment"]))] = rec
    return records


def load_final_loss(path: Path) -> float | None:
    loss = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("arm") not in ("base (untrained)", "baseline (sync, injected)"):
            loss = rec.get("eval_loss", loss)
    return loss


def load_layout_bounds(adapter: Path, num_fragments: int) -> list[np.ndarray]:
    """Per-fragment tensor boundaries from the exported adapter, via the same
    binpack `build_layout` path the learners used (validated downstream by the
    bit-exact step replay)."""
    from safetensors import safe_open

    from yeto.fragments import build_layout

    f = safe_open(str(adapter), "np")
    named = []
    for k in f.keys():
        numel = int(np.prod(f.get_slice(k).get_shape()))
        fq = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        named.append((fq, numel))
    layout = build_layout(named, num_fragments, "binpack")
    bounds = []
    for fr in layout.fragments:
        bounds.append(np.cumsum([0] + [n for _, n in fr.tensors]))
    return bounds


def ckpt_fragment_offsets(path: Path) -> list[tuple[int, int]]:
    """[(numel, params_byte_offset)] per fragment, parsed from the header."""
    out = []
    with open(path, "rb") as f:
        magic, _global_step, num_fragments = struct.unpack("<IQI", f.read(16))
        if magic != CKPT_MAGIC:
            raise ValueError(f"{path}: bad checkpoint magic 0x{magic:08X}")
        offset = 16
        for _ in range(num_fragments):
            f.seek(offset)
            _version, numel = struct.unpack("<QQ", f.read(16))
            out.append((numel, offset + 16))
            offset += 16 + 8 * numel
    return out


def read_fragment_state(path: Path, frag: int) -> tuple[np.ndarray, np.ndarray]:
    """(params, momentum) f32 arrays of one fragment from a state checkpoint."""
    offsets = ckpt_fragment_offsets(path)
    numel, start = offsets[frag]
    with open(path, "rb") as f:
        f.seek(start)
        params = np.frombuffer(f.read(4 * numel), dtype="<f4").copy()
        momentum = np.frombuffer(f.read(4 * numel), dtype="<f4").copy()
    return params, momentum


def merge_rda_tensor(
    anchor: np.ndarray, cands: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    """Weighted RDA over one tensor slice, mirroring merge.rs `merge_rda`
    (f32 deltas, f64 norm accumulation, f32 coef = (w/wsum/n), degenerate
    fallback to weighted direct averaging)."""
    wsum = float(sum(weights))
    deltas = [anchor - c for c in cands]  # f32, matches rust (*a - *l)
    norms = [float(np.sqrt(np.sum(d.astype(np.float64) ** 2))) for d in deltas]
    radial = sum(n * w for n, w in zip(norms, weights)) / wsum
    out = np.zeros(anchor.shape[0], dtype=np.float32)
    for d, n, w in zip(deltas, norms, weights):
        if n == 0.0:
            continue
        out += np.float32(w / wsum / n) * d
    mean_dir_norm = float(np.sqrt(np.sum(out.astype(np.float64) ** 2)))
    if mean_dir_norm < 1e-12:
        out = np.zeros(anchor.shape[0], dtype=np.float32)
        for d, w in zip(deltas, weights):
            out += np.float32(w / wsum) * d
        return out
    return out * np.float32(radial / mean_dir_norm)


def cos64(a: np.ndarray, b: np.ndarray) -> float | None:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(af, bf) / (na * nb))


def discover_capture(capture: Path) -> dict[int, dict[int, dict[int, Path]]]:
    """{step: {frag: {learner: candidate_path}}} from the capture dir."""
    groups: dict[int, dict[int, dict[int, Path]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for p in sorted((capture / "candidates").iterdir()):
        m = CAND_RE.search(p.name)
        if not m:
            continue
        step, frag, learner = (int(m.group(i)) for i in (1, 2, 3))
        groups[step][frag][learner] = p
    return groups


def index_weights(capture: Path) -> dict[tuple[int, int, int], float]:
    """{(step, frag, learner): weight} from the capture index."""
    weights = {}
    for line in (capture / "index.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        weights[(int(rec["step"]), int(rec["fragment"]), int(rec["learner_id"]))] = (
            float(rec["weight"])
        )
    return weights


def analyze_run(spec: dict, out_dir: Path) -> dict:
    label = spec["label"]
    eta = float(spec["eta"])
    mu = float(spec["mu"])
    delta_norm_ref = float(spec.get("delta_norm_ref", 0.0))
    tape = load_tape(Path(spec["tape"]))
    rows: dict[tuple[int, int], dict] = {}
    for (step, frag), rec in sorted(tape.items()):
        rows[(step, frag)] = {
            "label": label,
            "step": step,
            "fragment": frag,
            "gnorm_tape": rec["gnorm"],
            "outer_step_norm": rec["outer_step_norm"],
        }

    summary: dict = {
        "label": label,
        "eta": eta,
        "mu": mu,
        "delta_norm_ref": delta_norm_ref,
        "commits": len(rows),
        "mean_gnorm": float(np.mean([r["gnorm_tape"] for r in rows.values()])),
        "mean_outer_step_norm": float(
            np.mean([r["outer_step_norm"] for r in rows.values()])
        ),
        "capture": None,
    }
    if "results" in spec:
        summary["final_eval_loss"] = load_final_loss(Path(spec["results"]))

    if "capture" in spec:
        capture = Path(spec["capture"])
        groups = discover_capture(capture)
        weights = index_weights(capture)
        num_fragments = len(ckpt_fragment_offsets(next(iter(sorted(
            (capture / "states").iterdir())))))
        bounds = load_layout_bounds(Path(spec["adapter"]), num_fragments)
        steps = sorted(groups)
        # per-fragment history of the previous same-fragment commit:
        # (step, anchor, buffer_after_update, applied_delta, raw_merged) —
        # applied_delta is the (possibly norm-matched) delta that entered the
        # optimizer; raw_merged is used for rho/lambda so the geometry
        # metrics stay in gradient units.
        prev: dict[int, tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        verify = {"checked": 0, "exact": 0, "max_abs_diff": 0.0}
        cap_stats: dict[str, list[float]] = defaultdict(list)

        for step in steps:
            (frag,) = groups[step].keys()
            learners = groups[step][frag]
            state_path = capture / "states" / f"state_before_step_{step:08d}.ckpt"
            anchor, buf = read_fragment_state(state_path, frag)
            cands = [
                np.fromfile(learners[l], dtype="<f4") for l in sorted(learners)
            ]
            w = [weights[(step, frag, l)] for l in sorted(learners)]
            b = bounds[frag]
            merged = np.empty(anchor.shape[0], dtype=np.float32)
            for i in range(len(b) - 1):
                merged[b[i] : b[i + 1]] = merge_rda_tensor(
                    anchor[b[i] : b[i + 1]],
                    [c[b[i] : b[i + 1]] for c in cands],
                    w,
                )
            raw_norm = float(np.sqrt(np.sum(merged.astype(np.float64) ** 2)))
            # optional post-merge renormalization (mirrors state.rs op order)
            if delta_norm_ref > 0.0 and raw_norm > 0.0:
                delta = np.float32(delta_norm_ref / raw_norm) * merged
            else:
                delta = merged
            delta_norm = float(np.sqrt(np.sum(delta.astype(np.float64) ** 2)))

            # verification against the previous same-fragment commit
            row = rows.setdefault(
                (step, frag), {"label": label, "step": step, "fragment": frag}
            )
            if frag in prev:
                pstep, panchor, pbuf_after, pdelta, pmerged = prev[frag]
                # replay the previous outer step (nesterov_step op order)
                direction = pdelta + np.float32(mu) * pbuf_after
                pred = panchor - np.float32(eta) * direction
                diff = float(np.max(np.abs(pred - anchor)))
                verify["checked"] += 1
                verify["max_abs_diff"] = max(verify["max_abs_diff"], diff)
                if diff == 0.0:
                    verify["exact"] += 1
                # secant curvature proxy on same-fragment consecutive commits
                dtheta = (anchor - panchor).astype(np.float64)
                dtheta_norm_sq = float(np.dot(dtheta, dtheta))
                dgrad = merged.astype(np.float64) - pmerged.astype(np.float64)
                if dtheta_norm_sq > 0.0:
                    lam = max(float(np.dot(dgrad, dtheta)), 0.0) / dtheta_norm_sq
                    row["lambda_hat"] = lam
                    cap_stats["lambda_hat"].append(lam)
                rho = cos64(merged, pmerged)
                if rho is not None:
                    row["rho"] = rho
                    cap_stats["rho"].append(rho)

            # buffer/delta geometry of the PRODUCTION step (rescaled delta)
            if delta_norm > 0.0:
                bf = buf.astype(np.float64)
                df = delta.astype(np.float64)
                c_t = float(np.dot(bf, df)) / (delta_norm**2)
                r_t = float(np.linalg.norm(bf - c_t * df)) / delta_norm
                row["c_t"] = c_t
                row["r_t"] = r_t
                cap_stats["c_t"].append(c_t)
                cap_stats["r_t"].append(r_t)
                if "lambda_hat" in row:
                    row["r2_lambda"] = r_t * r_t * row["lambda_hat"]
                    cap_stats["r2_lambda"].append(row["r2_lambda"])

            # worker agreement against the plain mean of per-learner deltas
            learner_deltas = [anchor - c for c in cands]
            mean_delta = np.mean(
                np.stack([d.astype(np.float64) for d in learner_deltas]), axis=0
            )
            agree = [
                c
                for c in (cos64(d, mean_delta) for d in learner_deltas)
                if c is not None
            ]
            if agree:
                row["agree_mean"] = float(np.mean(agree))
                row["agree_min"] = float(np.min(agree))
                cap_stats["agree_mean"].append(row["agree_mean"])
            row["gnorm_replay"] = raw_norm
            gt = row.get("gnorm_tape")
            if gt is not None and max(gt, raw_norm) > 0:
                cap_stats["gnorm_rel_err"].append(
                    abs(raw_norm - gt) / max(gt, raw_norm)
                )
            cap_stats["gnorm_replay"].append(raw_norm)

            # nesterov buffer update for the NEXT verification/geometry step
            buf_after = np.float32(mu) * buf + delta
            prev[frag] = (step, anchor, buf_after, delta, merged)

        summary["capture"] = {
            "steps": len(steps),
            "verification": verify,
            "means": {k: float(np.mean(v)) for k, v in cap_stats.items() if v},
            "medians": {
                k: float(np.median(v)) for k, v in cap_stats.items() if v
            },
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{label}_per_commit.jsonl", "w") as f:
        for key in sorted(rows):
            f.write(json.dumps(rows[key]) + "\n")
    return summary


PANEL_COLUMNS = [
    ("mean_gnorm", "||g||"),
    ("rho", "rho"),
    ("c_t", "c_t"),
    ("r_t", "r_t"),
    ("agree_mean", "agree"),
    ("lambda_hat", "lambda^"),
    ("r2_lambda", "r^2*lam"),
]


def write_panel_md(out: Path, summaries: list[dict]) -> None:
    lines = [
        "# EXP2.39 geometry panel (per-cell means over commits)",
        "",
        "rho = cos of consecutive same-fragment merged deltas; c_t/r_t from the",
        "pre-commit Nesterov buffer vs the (possibly norm-matched) merged delta;",
        "agree = mean_i cos(g_t^i, mean); lambda^ = secant curvature proxy;",
        "verify = replayed-step vs next-anchor exactness (exact/checked, max|diff|).",
        "",
        "| cell | eta | mu | Dref | commits | ||g|| | rho | c_t | r_t | agree "
        "| lambda^ | r^2*lam | loss | verify |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        cap = s.get("capture") or {}
        means = cap.get("means", {})

        def cell(key, fmt="{:.4f}"):
            if key == "mean_gnorm":
                return fmt.format(s["mean_gnorm"])
            value = means.get(key)
            return fmt.format(value) if value is not None else "-"

        loss = s.get("final_eval_loss")
        loss_str = f"{loss:.6f}" if loss is not None else "-"
        verify = cap.get("verification")
        verify_str = (
            f"{verify['exact']}/{verify['checked']} ({verify['max_abs_diff']:.2g})"
            if verify
            else "-"
        )
        lines.append(
            f"| {s['label']} | {s['eta']} | {s['mu']} | {s['delta_norm_ref']} "
            f"| {s['commits']} | "
            + " | ".join(cell(k) for k, _ in PANEL_COLUMNS)
            + f" | {loss_str} | {verify_str} |"
        )
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="SPEC",
        help="label=..,tape=..,eta=..,mu=..[,capture=..,adapter=..,"
        "results=..,delta_norm_ref=..]",
    )
    ap.add_argument(
        "--out", type=Path, default=REPO / "experiment-results/EXP2/geometry-panel"
    )
    args = ap.parse_args()

    summaries = []
    for raw in args.run:
        spec = parse_run_spec(raw)
        print(f"[panel] {spec['label']} ...", flush=True)
        summary = analyze_run(spec, args.out)
        cap = summary.get("capture")
        if cap:
            v = cap["verification"]
            print(
                f"[panel] {spec['label']}: {summary['commits']} commits, "
                f"replay exact {v['exact']}/{v['checked']} "
                f"(max diff {v['max_abs_diff']:.3g})",
                flush=True,
            )
        summaries.append(summary)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "panel.json").write_text(json.dumps(summaries, indent=2) + "\n")
    write_panel_md(args.out / "panel.md", summaries)
    print(f"wrote {args.out}/panel.json and panel.md")


if __name__ == "__main__":
    main()
