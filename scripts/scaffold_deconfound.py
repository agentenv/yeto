#!/usr/bin/env python3
"""Fit the SCAFFOLD-lite/SGD update scale from syncer probe captures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

DEFAULT_TAPES = (
    "s3://yeto-exp-artifacts-533462777468-us-west-2/"
    "probecommit-resume-20260710/exp2-51-scaffold/"
)
UPDATE_FIELD = "applied_update_f32"


def _materialize_tapes(source: str, destination: Path) -> Path:
    if not source.startswith("s3://"):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"--tapes path does not exist: {path}")
        return path
    env = os.environ.copy()
    env.setdefault("AWS_DEFAULT_REGION", "us-west-2")
    command = ["aws", "s3", "cp", source, str(destination), "--recursive"]
    try:
        subprocess.run(command, check=True, env=env)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"could not fetch {source}: {exc}") from exc
    return destination


def _capture_roots(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("index.jsonl"))


def _choose_arm(roots: list[Path], hint: str | None, kind: str) -> Path:
    if hint:
        matches = [root for root in roots if hint in str(root)]
    elif kind == "lite":
        matches = [root for root in roots if "lite" in str(root).lower()]
    else:
        matches = [
            root
            for root in roots
            if any(word in str(root).lower() for word in ("sgd", "control"))
            and "lite" not in str(root).lower()
        ]
    if len(matches) != 1:
        listing = "\n  ".join(str(root) for root in roots) or "(none)"
        raise SystemExit(
            f"could not uniquely select the {kind} arm; use --{kind}-arm. "
            f"Capture roots:\n  {listing}"
        )
    return matches[0]


def _read_rows(root: Path) -> list[dict]:
    rows = []
    with (root / "index.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{root / 'index.jsonl'} contains no records")
    return rows


def _check_update_vectors(arms: dict[str, tuple[Path, list[dict]]]) -> None:
    missing = {
        name: sum(UPDATE_FIELD not in row for row in rows)
        for name, (_, rows) in arms.items()
    }
    if any(missing.values()):
        details = ", ".join(f"{name}={count}" for name, count in missing.items())
        print(
            "INSUFFICIENT CAPTURE: the exp2-51 probe rows do not contain the "
            f"per-commit applied update vectors u_t ({details}).",
            file=sys.stderr,
        )
        print(
            "Older syncer_probe_capture_v1 rows contain pre-merge state "
            "checkpoints and candidate endpoints, not applied updates. A fresh "
            "run with --syncer-probe-capture is required; current v1 rows add "
            f"the `{UPDATE_FIELD}` field pointing to an f32 vector for each "
            "(fragment, version) commit.",
            file=sys.stderr,
        )
        print(
            "Fallback: run a short paired H16 lite/SGD capture with that flag "
            "and measure the RMS-energy ratio from the newly captured vectors.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _updates(root: Path, rows: list[dict]) -> dict[tuple[int, int], Path]:
    updates: dict[tuple[int, int], Path] = {}
    for row in rows:
        version = row["version"] if "version" in row else row["step"]
        key = (int(row["fragment"]), int(version))
        path = Path(row[UPDATE_FIELD])
        path = path if path.is_absolute() else root / path
        previous = updates.setdefault(key, path)
        if previous != path:
            raise SystemExit(f"{root}: commit {key} names multiple update vectors")
    return updates


def _read_f32(path: Path) -> array:
    values = array("f")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def fit(lite_root: Path, lite_rows: list[dict], sgd_root: Path, sgd_rows: list[dict]) -> dict:
    lite = _updates(lite_root, lite_rows)
    sgd = _updates(sgd_root, sgd_rows)
    keys = sorted(lite.keys() & sgd.keys())
    if not keys:
        raise SystemExit("the selected arms have no aligned (fragment, version) commits")
    dot = lite_energy = sgd_energy = 0.0
    for key in keys:
        u_lite = _read_f32(lite[key])
        u_sgd = _read_f32(sgd[key])
        if len(u_lite) != len(u_sgd):
            raise SystemExit(f"commit {key} vector shapes differ")
        for lite_value, sgd_value in zip(u_lite, u_sgd):
            dot += lite_value * sgd_value
            lite_energy += lite_value * lite_value
            sgd_energy += sgd_value * sgd_value
    if sgd_energy == 0.0:
        raise SystemExit("aligned SGD updates have zero total energy")
    scale = dot / sgd_energy
    return {
        "aligned_commits": len(keys),
        "s_star": scale,
        "eta_match": 0.28 * scale,
        "r_E": (lite_energy / sgd_energy) ** 0.5,
        "sum_lite_dot_sgd": dot,
        "sum_lite_energy": lite_energy,
        "sum_sgd_energy": sgd_energy,
        "first_key": list(keys[0]),
        "last_key": list(keys[-1]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", default=DEFAULT_TAPES)
    parser.add_argument("--lite-arm", default=None, help="path substring for the lite arm")
    parser.add_argument("--sgd-arm", default=None, help="path substring for the SGD control")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="scaffold-deconfound-") as tmp:
        root = _materialize_tapes(args.tapes, Path(tmp))
        roots = _capture_roots(root)
        lite_root = _choose_arm(roots, args.lite_arm, "lite")
        sgd_root = _choose_arm(roots, args.sgd_arm, "sgd")
        arms = {
            "lite": (lite_root, _read_rows(lite_root)),
            "sgd": (sgd_root, _read_rows(sgd_root)),
        }
        _check_update_vectors(arms)
        result = fit(*arms["lite"], *arms["sgd"])
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
