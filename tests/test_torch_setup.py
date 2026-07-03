"""The learner-node torch selector, exercised as real shell with stubbed
nvidia-smi/pip/python3 — the logic that decides wheels must not regress
into the silent-wrong-wheel failure mode."""

from __future__ import annotations

import os
import stat
import subprocess

from yeto.launcher import TORCH_SETUP


def _run(tmp_path, compute_cap, driver, torch_cuda_ok_sequence):
    """Run TORCH_SETUP with stubs. torch_cuda_ok_sequence: exit codes for
    successive `python3 -c ...` probes (pre-check, post-install check)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "probe_count"
    state.write_text("0")
    log = tmp_path / "log"

    def stub(name, body):
        f = bin_dir / name
        f.write_text("#!/bin/bash\n" + body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)

    smi = ""
    if compute_cap is not None:
        smi += f'[[ "$*" == *compute_cap* ]] && echo "{compute_cap}"\n'
    if driver is not None:
        smi += f'[[ "$*" == *driver_version* ]] && echo "{driver}"\n'
    stub("nvidia-smi", smi + "exit 0\n")
    stub("pip", f'echo "pip $*" >> {log}\nexit 0\n')
    codes = " ".join(str(c) for c in torch_cuda_ok_sequence)
    stub(
        "python3",
        f'n=$(cat {state}); echo $((n+1)) > {state}\n'
        f"codes=({codes})\nexit ${{codes[$n]:-1}}\n",
    )
    proc = subprocess.run(
        ["bash", "-c", TORCH_SETUP],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    return proc, log.read_text() if log.exists() else ""


def test_blackwell_gets_cu128(tmp_path):
    proc, pips = _run(tmp_path, "10.0", "580.12.34", [1, 0])
    assert proc.returncode == 0
    assert "cu128" in pips and "cu121" not in pips


def test_ampere_on_old_driver_gets_cu121(tmp_path):
    proc, pips = _run(tmp_path, "8.0", "535.216.01", [1, 0])
    assert proc.returncode == 0
    assert "cu121" in pips and "cu128" not in pips


def test_hopper_on_new_driver_gets_cu128(tmp_path):
    proc, pips = _run(tmp_path, "9.0", "570.10.1", [1, 0])
    assert proc.returncode == 0
    assert "cu128" in pips


def test_blackwell_with_old_driver_fails_loudly(tmp_path):
    proc, pips = _run(tmp_path, "10.0", "535.216.01", [1])
    assert proc.returncode == 1
    assert "needs cu128" in proc.stderr
    assert pips == ""  # never installs a wheel that cannot work


def test_missing_nvidia_smi_fails_loudly(tmp_path):
    proc, pips = _run(tmp_path, None, None, [1])
    assert proc.returncode == 1
    assert "no GPU info" in proc.stderr and pips == ""


def test_prehistoric_driver_fails_loudly(tmp_path):
    proc, _ = _run(tmp_path, "8.0", "470.1", [1])
    assert proc.returncode == 1
    assert "predates CUDA 12" in proc.stderr


def test_functional_torch_skips_reinstall(tmp_path):
    proc, pips = _run(tmp_path, "10.0", "580.1", [0])
    assert proc.returncode == 0
    assert pips == ""  # idempotent recovery path
    assert "keeping it" in proc.stdout


def test_broken_install_fails_verification(tmp_path):
    proc, pips = _run(tmp_path, "9.0", "570.1", [1, 1])
    assert proc.returncode == 1
    assert "cannot see the GPUs" in proc.stderr
