"""S11 fnboot boot-only (``--rl-boot-only``): the learner runs every torch_dist-free
check (real Flash-Next provider view from the 4layer config.json + Miles alias
fixture, ports argv build, ref-load probe), writes ``~/yeto-rl/boot_only.json`` +
``FN_BOOT_ONLY_OK ref_load_present=<bool>`` and returns (exit 0) without training,
so the job SUCCEEDS and the launcher's --keep keeps the FS-attached cluster.

Only ``parse_miles_args`` / ``verify_ports_algorithm`` (need the Miles parser +
plugins, absent on CPU), ``require_run_plugin`` (Miles' Ray trainer group) and the
training entry (``_run_ports``: must never be reached) are replaced; the provider
and ``_resolve_ref_load`` are real.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_rl_fn_provider_view import REPO, _flag, _probe, snapshots  # noqa: E402,F401

MG = REPO / "tests" / "multinode_gpu"
TD = "/mnt/yeto-models/torch_dist/qwen3.8-flash-next-4layer_torch_dist"


def _boot_body(snapshot: Path, *, boot_only: bool, ref_dir: str | None) -> str:
    return f"""
        import os, tempfile, io, contextlib
        from pathlib import Path
        home = Path(tempfile.mkdtemp())
        os.environ["HOME"] = str(home)
        sys.path.insert(0, {str(REPO / 'tests')!r})
        sys.path.insert(0, {str(MG)!r})
        import _pytest.monkeypatch as _m
        from types import SimpleNamespace
        from rl_e2e_launch import island_run, learner_from_run
        from fp_fn import fnrun_cli
        from yeto.rl import learner
        from yeto.rl.adapters.miles import config as mac
        from yeto.rl.adapters.miles import state as mas

        mp = _m.MonkeyPatch()
        extra = ("--rl-boot-only",) if {boot_only!r} else ()
        run = island_run(tuple(fnrun_cli("fn8s", extra=extra)), mp)
        args, _env = learner_from_run(run, home / "lh")
        out = {{"args_boot_only": bool(getattr(args, "rl_boot_only", False)),
               "configured_ref": args.megatron_ref_load}}
        if {ref_dir!r} is not None:
            args.megatron_ref_load = {ref_dir!r}
        cap = {{}}
        def parse(launch):
            cap["argv"] = list(launch.argv)
            return SimpleNamespace()
        mp.setattr(mac, "parse_miles_args", parse)
        mp.setattr(learner, "verify_ports_algorithm", lambda *a: None)
        mp.setattr(mas, "require_run_plugin", lambda: None)
        def no_train(*a, **k):
            raise AssertionError("boot-only reached training")
        mp.setattr(learner, "_run_ports", no_train)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                learner.run_miles(args, model_path={str(snapshot)!r},
                                  prompt_path="/root/yeto-rl/prompts.jsonl",
                                  yeto_policy_sync=False)
            out["returned"] = True
        except Exception as exc:
            out["returned"] = False
            out["err"] = f"{{type(exc).__name__}}: {{exc}}"
        marker = home / "yeto-rl" / "boot_only.json"
        out["marker"] = json.loads(marker.read_text()) if marker.exists() else None
        out["stdout_lines"] = [l for l in buf.getvalue().splitlines() if "BOOT_ONLY" in l]
        out["argv"] = cap.get("argv")
        out["megatron_loaded"] = any(
            m == "megatron" or m.startswith("megatron.")
            for m, v in sys.modules.items() if v is not None)
        print(json.dumps(out))
    """


def test_boot_only_without_torch_dist_exits_cleanly(snapshots):
    out = _probe(_boot_body(snapshots["4layer"], boot_only=True, ref_dir=None))
    assert out["args_boot_only"] is True and out["configured_ref"] == TD
    assert out["returned"] is True, out.get("err")
    m = out["marker"]
    assert m["marker"] == "FN_BOOT_ONLY_OK" and m["ref_load_present"] is False
    assert m["ref_load"]["configured"] == TD
    assert "real local directory" in m["ref_load"]["reason"]
    assert m["model_recipe"] == "qwen3_8_next"
    assert m["runtime_fingerprint"].startswith("sha256:") or m["runtime_fingerprint"]
    assert "FN_BOOT_ONLY_OK ref_load_present=false recipe=qwen3_8_next" in out["stdout_lines"]
    assert out["megatron_loaded"] is False
    argv = out["argv"]
    assert _flag(argv, "--model-name") == "qwen4_exp" and _flag(argv, "--num-layers") == "4"
    assert _flag(argv, "--ref-load") == TD  # recorded verbatim, not resolved


def test_boot_only_with_torch_dist_still_does_not_train(snapshots, tmp_path):
    ref = tmp_path / "td"
    ref.mkdir()
    (ref / "latest_checkpointed_iteration.txt").write_text("release\n")
    out = _probe(_boot_body(snapshots["4layer"], boot_only=True, ref_dir=str(ref)))
    assert out["returned"] is True, out.get("err")  # _run_ports would have raised
    m = out["marker"]
    assert m["ref_load_present"] is True and m["ref_load"]["resolved"] == str(ref.resolve())
    assert "FN_BOOT_ONLY_OK ref_load_present=true recipe=qwen3_8_next" in out["stdout_lines"]


def test_default_still_fails_fast_on_missing_torch_dist(snapshots):
    out = _probe(_boot_body(snapshots["4layer"], boot_only=False, ref_dir=None))
    assert out["args_boot_only"] is False
    assert out["returned"] is False and "real local directory" in out["err"], out
    assert out["marker"] is None and out["stdout_lines"] == []


def test_launcher_rejects_boot_only_without_no_sync():
    from yeto import launcher
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="--rl-boot-only needs"):
        launcher._check_ports_infra_switches(
            SimpleNamespace(rl_boot_only=True, rl_single_island_no_sync=False), "ports")


# ------------------------------------------------------------ scripts (print-only)
def _fnrun(env_extra):
    env = {k: v for k, v in os.environ.items() if k != "BOOT_ONLY"} | env_extra
    return subprocess.run(["bash", str(MG / "fnrun.sh"), "fn8s"], env=env, check=True,
                          capture_output=True, text=True).stdout


def test_fnrun_boot_only_switch():
    assert "--rl-boot-only" not in shlex.split(_fnrun({}))
    assert "--rl-boot-only" in shlex.split(_fnrun({"BOOT_ONLY": "1"}))
    r = subprocess.run(["bash", str(MG / "fnrun.sh"), "fn32s"], capture_output=True, text=True,
                       env={**os.environ, "BOOT_ONLY": "1"})
    assert r.returncode == 64 and "only applies to fn8s" in r.stderr


@pytest.mark.parametrize("boot, expect", [("0", False), ("1", True)])
def test_s1run_dry_fn8s(boot, expect):
    env = {**os.environ, "DRY": "1", "BOOT_ONLY": boot, "THREAD_MAX": "99999"}
    r = subprocess.run(["bash", str(MG / "s1run.sh"), "fn8s", "pfx", "100", "200"],
                       env=env, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    toks = shlex.split(r.stdout.splitlines()[-1])
    assert ("--rl-boot-only" in toks) is expect and "--keep" in toks


def test_fp_fn_fingerprint_ignores_boot_only(monkeypatch):
    sys.path.insert(0, str(MG))
    try:
        import fp_fn
    finally:
        sys.path.pop(0)
    monkeypatch.delenv("BOOT_ONLY", raising=False)
    base = fp_fn.fnrun_cli("fn8s", total_steps=6)
    monkeypatch.setenv("BOOT_ONLY", "1")
    assert fp_fn.fnrun_cli("fn8s", total_steps=6) == base
    assert "--rl-boot-only" not in base


def test_chain_fnboot_uses_boot_only_and_fna_does_not():
    src = (MG / "s11h200chain.sh").read_text()
    assert subprocess.run(["bash", "-n", str(MG / "s11h200chain.sh")]).returncode == 0
    assert "fnrun fnboot ${HARD_FNBOOT:-3000} 1\n" in src
    assert "fnrun fna ${HARD_FNA:-3600}\n" in src
    assert "BOOT_ONLY=${3:-0}" in src
    assert "already trained 6 rounds" not in src
    assert "FN_BOOT_ONLY_OK ref_load_present=" in src
