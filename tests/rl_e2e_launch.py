"""End-to-end helper: real ``yeto launch`` CLI -> island run command -> the learner.

Runs the island's own shell prelude (files written under ~/yeto-rl, ``export``s)
with bash under a temporary HOME, lets bash expand the learner command line and
parses it with the real ``yeto.rl.adapters.miles.island_entry.parse_args``. No cloud, no GPU.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def island_run(cli_extra, monkeypatch):
    from test_rl_engine_selection import _cli, _island_task
    from yeto.launcher import _prepare_rl_args

    args = _cli(tuple(cli_extra))
    _prepare_rl_args(args)
    return _island_task(args, monkeypatch).run


def learner_from_run(run: str, home: Path, *, env=None):
    """(learner args namespace, exported env) as the island would see them."""
    from yeto.rl import learner

    head = run.split('if [ "$SKYPILOT_NODE_RANK" = "0" ]; then', 1)[1].split("\nelse\n", 1)[0]
    body = head.split("trap stop_miles_ray EXIT\n", 1)[1]
    match = re.search(r"^\s*RAY_ADDRESS=\S+ PYTHONPATH=\S+ (?:yeto_rl_restart_loop )?python3 -m yeto\.rl\.adapters\.miles\.island_entry", body, re.M)
    assert match, "no learner command in the island run"
    prelude, command = body[: match.start()], body[match.end():]
    home.mkdir(parents=True, exist_ok=True)
    out = home / "argv.bin"
    envfile = home / "env.bin"
    script = (prelude + f"\nprintf '%s\\0' {command.strip()} > {out}\nenv -0 > {envfile}\n")
    base = {"HOME": str(home), "PATH": "/usr/bin:/bin", "SYNCER_ADDR": "127.0.0.1:1",
            "LEARNER_ID": "0", **(env or {})}
    subprocess.run(["bash", "-e", "-c", script], check=True, env=base)
    tokens = [t for t in out.read_bytes().decode().split("\0") if t]
    exported = dict(kv.split("=", 1) for kv in envfile.read_bytes().decode().split("\0") if "=" in kv)
    return learner.parse_args(tokens), exported
