"""``yeto run scale`` on the default control plane, branch by branch (the golden of
``test_run_scale_without_the_store_control_plane_is_byte_for_byte_as_before``): every
pool-scale command it spawns, its return codes, what it prints and ``state.scales``."""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest


def run_scale_branches(tmp_path: Path) -> dict[str, Any]:
    from tests.test_rollout_entry import _cfg, _two_pools  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
        mp.setenv("YETO_NC_ACCOUNTS", "1,2,3,4")
        mp.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
        mp.setattr(up.Run, "budget_problem", lambda self: None)
        mp.setattr(up.Run, "_objects", lambda self: (_ for _ in ()).throw(AssertionError("store read")))
        calls: list[list[str]] = []
        code = {"rc": 0}
        mp.setattr(subprocess, "run", lambda argv, **kw: calls.append(list(argv[1:]))
                   or subprocess.CompletedProcess(argv, code["rc"]))
        run_cfg, _, _ = _two_pools(tmp_path)
        out = io.StringIO()
        rcs = []
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            r = up.Run(run_cfg)
            rcs += [r.scale(2, retire=["t2", "t3"], pool="nc"), r.scale(4, pool="nc", force_static=True),
                    r.scale(None, add=2, pool="modal", source="auto"), r.scale(3, provider="modal"),
                    r.scale(None, remove=["t-unknown"]), r.scale(1, pool="nope"), r.scale(None, add=1)]
            code["rc"] = 3  # pool-scale refuses: its code is passed through
            rcs.append(r.scale(3, pool="nc"))
            code["rc"] = 0
            single_dir = tmp_path / "single"
            single_dir.mkdir()
            _, nc_path = _cfg(single_dir, accounts=[1, 4])
            one = up.Run({"run": "run-b", "store": "s3://b/runs/run-b", "trainer": {"argv": []},
                          "pool": {"config": nc_path}})
            rcs += [one.scale(2), one.scale(None, add=1, mode="fence"), one.scale(None, remove=["t5"])]
            scales = [json.loads(json.dumps(x.state.get("scales"), default=str)) for x in (r, one)]
        text = re.sub(r"\d{4}-\d\d-\d\dT[\d:.]+Z?", "<T>", out.getvalue()).replace(str(tmp_path), "<T>")
        scales = json.loads(re.sub(r"\d{4}-\d\d-\d\dT[\d:.]+Z?", "<T>",
                                   json.dumps(scales).replace(str(tmp_path), "<T>")))
        return {"rcs": rcs, "calls": [[str(a).replace(str(tmp_path), "<T>") for a in c] for c in calls],
                "out": text, "scales": scales}
    finally:
        mp.undo()
