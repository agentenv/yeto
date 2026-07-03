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
    assert "unparseable GPU compute capability" in proc.stderr and pips == ""


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


def _run_driver_scenario(tmp_path, smi_body, lspci_nvidia=True, apt_ok=True):
    """Full-stub harness for the driver-remediation path (sleep is a no-op
    so the wait loop runs fast)."""
    bin_dir = tmp_path / "dbin"
    bin_dir.mkdir()
    log = tmp_path / "dlog"

    def stub(name, body):
        f = bin_dir / name
        f.write_text("#!/bin/bash\n" + body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)

    stub("nvidia-smi", smi_body)
    stub("lspci", f"echo '00:1e.0 3D controller: {'NVIDIA' if lspci_nvidia else 'Other'} Device'\n")
    stub("sudo", f'echo "sudo $*" >> {log}\nexit {0 if apt_ok else 1}\n')
    stub("uname", "echo 6.8.0-fake\n")
    stub("sleep", "exit 0\n")
    stub("pip", f'echo "pip $*" >> {log}\nexit 0\n')
    stub("python3", "exit 1\n")  # torch never sees CUDA pre-install
    proc = subprocess.run(
        ["bash", "-c", TORCH_SETUP],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    return proc, log.read_text() if log.exists() else ""


def test_broken_driver_gets_installed_then_selects_cu128(tmp_path):
    # nvidia-smi fails until the (stubbed) install marker file appears —
    # the stub flips to healthy after sudo ran, emulating a driver install.
    marker = tmp_path / "installed"
    smi = (
        f'if [ ! -f {marker} ]; then exit 9; fi\n'
        '[[ "$*" == *compute_cap* ]] && echo "10.0"\n'
        '[[ "$*" == *driver_version* ]] && echo "580.12.1"\n'
        "exit 0\n"
    )
    proc, log = _run_driver_scenario(tmp_path, smi)
    # Rerun with stubs where the install actually "works": sudo creates a
    # marker, nvidia-smi and the torch probe both go healthy once it exists.
    bin_dir = tmp_path / "dbin"
    sudo = bin_dir / "sudo"
    sudo.write_text(f'#!/bin/bash\necho "sudo $*" >> {tmp_path / "dlog"}\ntouch {marker}\nexit 0\n')
    py = bin_dir / "python3"
    py.write_text(f'#!/bin/bash\n[ -f {marker} ] && exit 0 || exit 1\n')
    proc = subprocess.run(
        ["bash", "-c", TORCH_SETUP],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    log = (tmp_path / "dlog").read_text()
    assert proc.returncode == 0, proc.stderr
    assert "installing an open-module driver" in proc.stdout
    assert "nvidia-driver-580-open" in log or "nvidia-driver-575-open" in log
    assert "cu128" in log


def test_driver_never_recovers_fails_loudly(tmp_path):
    proc, _ = _run_driver_scenario(tmp_path, "exit 9\n")
    assert proc.returncode == 1
    assert "driver not ready after 120s" in proc.stderr


def test_garbage_smi_output_is_rejected_not_compared(tmp_path):
    smi = (
        '[[ "$*" == "-L" ]] && exit 0\n'
        'echo "NVIDIA-SMI has failed because it could not communicate"\nexit 0\n'
    )
    proc, _ = _run_driver_scenario(tmp_path, smi)
    assert proc.returncode == 1
    assert "unparseable GPU compute capability" in proc.stderr
    assert "integer expression expected" not in proc.stderr
