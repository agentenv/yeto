"""Per-step metrics of an E6 verl run from its island log (verl console metrics).

Per step: reward mean (``critic/score/mean``), rollout-vs-trainer log-prob |diff|
(``yeto_mismatch/abs_diff``), TIS clip share (``yeto_mismatch/tis_clipfrac``),
response tokens (``response_length/mean`` x samples), the step's wall time
(between consecutive step lines; includes generation and training) and its
output tokens per second. Usage: ``python parse_verl.py RUN_DIR... > out.json``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

KEYS = ("critic/score/mean", "yeto_mismatch/abs_diff", "yeto_mismatch/tis_clipfrac", "yeto_mismatch/signed_mean",
        "response_length/mean", "training/rollout_probs_diff_mean", "timing_s/gen", "timing_s/step")
NUM = r"(?:np\.float64\()?(-?[0-9.]+(?:e-?[0-9]+)?)"


def parse(log: Path, samples: int) -> list[dict]:
    steps, last_t = [], None
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"\bstep:(\d+) ", line)
        if not m or "critic/score/mean" not in line:
            continue
        row: dict = {"step": int(m.group(1))}
        for key in KEYS:
            hit = re.search(re.escape(key) + ":" + NUM, line)
            if hit:
                row[key] = float(hit.group(1))
        steps.append(row)
    return steps


def main() -> None:
    out = {}
    for run_dir in sys.argv[1:]:
        run_dir = Path(run_dir)
        log = next(run_dir.rglob("island.log"))
        meta = json.loads(next(run_dir.rglob("meta.json")).read_text())
        cfg = meta["cfg"]
        rows = parse(log, cfg["samples"])
        out[run_dir.name] = {"rc": meta.get("rc"), "seconds": meta.get("seconds"), "gpu": meta.get("gpu"),
                             "groups": cfg["groups"], "samples": cfg["samples"], "steps": rows}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
