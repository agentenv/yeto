"""The NVMe scratch setup block, exercised as real shell with stubbed
lsblk/mdadm/mkfs/mount — striping decisions must not regress."""

from __future__ import annotations

import os
import stat
import subprocess

from yeto.launcher import NVME_ENV, NVME_SETUP


def _run(tmp_path, lsblk_out, mounted=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "log"
    if log.exists():
        log.unlink()

    def stub(name, body):
        f = bin_dir / name
        f.write_text("#!/bin/bash\n" + body)
        f.chmod(f.stat().st_mode | stat.S_IEXEC)

    stub("mountpoint", f"exit {0 if mounted else 1}\n")
    stub("lsblk", f"cat <<'OUT'\n{lsblk_out}\nOUT\n")
    stub("sudo", f'echo "sudo $*" >> {log}\nexit 0\n')  # log, never execute
    stub("whoami", "echo tester\n")
    proc = subprocess.run(
        ["bash", "-c", NVME_SETUP],
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    return proc, log.read_text() if log.exists() else ""


TWO_DEVICES = (
    "nvme0n1 Amazon Elastic Block Store\n"
    "nvme1n1 Amazon EC2 NVMe Instance Storage\n"
    "nvme2n1 Amazon EC2 NVMe Instance Storage"
)


def test_two_instance_store_devices_get_raid0(tmp_path):
    proc, log = _run(tmp_path, TWO_DEVICES)
    assert proc.returncode == 0
    assert "mdadm --create /dev/md0" in log and "--raid-devices=2" in log
    assert "/dev/nvme1n1 /dev/nvme2n1" in log  # EBS device excluded
    assert "mkfs.ext4" in log and "mount -o noatime /dev/md0 /opt/yeto-nvme" in log
    assert "striped 2" in proc.stdout


def test_single_device_skips_raid(tmp_path):
    one = "nvme0n1 Amazon Elastic Block Store\nnvme1n1 Amazon EC2 NVMe Instance Storage"
    proc, log = _run(tmp_path, one)
    assert proc.returncode == 0
    assert "mdadm" not in log
    assert "mkfs.ext4 -q -F /dev/nvme1n1" in log


def test_ebs_only_node_is_noop(tmp_path):
    proc, log = _run(tmp_path, "nvme0n1 Amazon Elastic Block Store")
    assert proc.returncode == 0
    assert log == ""
    assert "stays on the boot disk" in proc.stdout


def test_already_mounted_skips_everything(tmp_path):
    proc, log = _run(tmp_path, TWO_DEVICES, mounted=True)
    assert proc.returncode == 0
    assert log == "" and "already mounted" in proc.stdout


def test_nvme_env_gates_on_real_mount():
    # A failed stripe must never divert the HF cache to a plain directory:
    # the env only exports when /opt/yeto-nvme is an actual mountpoint,
    # and NVMe-less nodes pass straight through.
    assert NVME_ENV.startswith("mountpoint -q /opt/yeto-nvme")
    assert NVME_ENV.endswith("|| true")
    assert "HF_HOME" in NVME_ENV and "HF_HUB_CACHE" in NVME_ENV
    proc = subprocess.run(["bash", "-c", NVME_ENV + "; echo ok"], capture_output=True, text=True)
    assert proc.returncode == 0 and "ok" in proc.stdout


def test_gcp_and_azure_ephemeral_models_detected(tmp_path):
    gcp = "nvme0n1 PersistentDisk\nnvme1n1 nvme_card\nnvme2n1 nvme_card"
    proc, log = _run(tmp_path, gcp)
    assert proc.returncode == 0 and "--raid-devices=2" in log
    azure = "sda Virtual Disk\nnvme0n1 Microsoft NVMe Direct Disk"
    proc, log = _run(tmp_path, azure)
    assert proc.returncode == 0 and "mkfs.ext4 -q -F /dev/nvme0n1" in log


def test_failed_mount_reports_and_stays_on_boot_disk(tmp_path):
    # sudo stub that fails mkfs/mount: the block must degrade, not die.
    bin_dir = tmp_path / "bin2"
    bin_dir.mkdir()
    import os as _os
    import stat as _stat

    def stub(name, body):
        f = bin_dir / name
        f.write_text("#!/bin/bash\n" + body)
        f.chmod(f.stat().st_mode | _stat.S_IEXEC)

    stub("mountpoint", "exit 1\n")
    stub("lsblk", "echo 'nvme1n1 Amazon EC2 NVMe Instance Storage'\n")
    stub("sudo", '[[ "$*" == mkfs* || "$*" == mount* ]] && exit 1; exit 0\n')
    stub("whoami", "echo tester\n")
    proc = subprocess.run(
        ["bash", "-c", NVME_SETUP],
        env={**_os.environ, "PATH": f"{bin_dir}:{_os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "staying on the boot disk" in proc.stderr
