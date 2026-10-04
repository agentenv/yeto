"""Extract the learner flags for the E3 container from ``yeto launch ... --dry-run`` (local, no resources).

    python build_flags.py <out learner_flags.txt> -- <yeto launch args incl. --rl-single-island-no-sync
                                                     --controller local --gpu modal:2xh100 ...>

The launcher's dry-run plan carries each island's ``python3 -m yeto.rl.learner``
line; its flags are what a real island would run, so the harness gets the
production argument pipeline (learner_shim.py).
"""

from __future__ import annotations

import json
import subprocess
import sys

MARKER = "yeto.rl.learner"


def learner_flags(plan: dict) -> str:
    islands = plan.get("island_requests") or []
    if len(islands) != 1:
        raise ValueError(f"E3 runs one island, the plan has {len(islands)}")
    command = islands[0].get("learner_command") or ""
    if MARKER not in command:
        raise ValueError("dry-run plan has no learner command")
    flags = command.split(MARKER, 1)[1].strip()
    for needed in ("--rl-single-island-no-sync",):
        if needed not in flags:
            raise ValueError(f"learner flags lack {needed}")
    return flags


def main(argv: list[str]) -> int:  # pragma: no cover - runs the local launcher
    out, rest = argv[0], argv[argv.index("--") + 1:]
    proc = subprocess.run([sys.executable, "-m", "yeto.cli", "launch", *rest, "--dry-run"],
                          check=True, capture_output=True, text=True)
    open(out, "w").write(learner_flags(json.loads(proc.stdout)) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
