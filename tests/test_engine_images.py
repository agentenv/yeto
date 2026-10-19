"""S25 N0.2: the image catalog, the version locks and the start check."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from yeto.launch import engine_images as images
from yeto.rl.adapters.miles import pins as miles_pins
from yeto.rl.adapters.verl import pins as verl_pins

DOCKER = Path(__file__).resolve().parents[1] / "docker"
ASSERT_SCRIPT = DOCKER / "image-assert" / "assert_image_versions.py"


def _load_assert_module():
    spec = importlib.util.spec_from_file_location("assert_image_versions", ASSERT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_assert_module()


def _lock(name: str) -> dict:
    return json.loads(images.IMAGES[name].lock_path.read_text())


def _dockerfile(name: str) -> str:
    return (DOCKER / name / "Dockerfile").read_text()


# --- catalog ----------------------------------------------------------------


def test_every_catalog_image_has_a_dockerfile_and_a_lock_that_matches_it():
    for name, spec in images.IMAGES.items():
        assert (DOCKER / name / "Dockerfile").is_file(), name
        lock = _lock(name)
        assert lock["image"] == spec.repository
        assert (lock["cpu_arch"], lock["device_family"]) == (spec.cpu_arch, spec.device_family)
        assert spec.cpu_arch in images.CPU_ARCHES
        assert lock["decisions"], f"{name}: the lock must record the decisions behind its versions"


def test_canonical_cpu_arch_accepts_the_docker_spellings_and_refuses_others():
    assert images.canonical_cpu_arch("amd64") == "x86_64"
    assert images.canonical_cpu_arch("ARM64") == "aarch64"
    assert images.canonical_cpu_arch(" x86_64\n") == "x86_64"
    with pytest.raises(images.UnknownCpuArch, match="cpu_arch 'ppc64le' is unknown"):
        images.canonical_cpu_arch("ppc64le")


@pytest.mark.parametrize("reference, expected", [
    ("reg.example/yeto-ppu-train:1", "ppu-train"),
    ("reg.example/a/b/yeto-npu-sampler:tag@sha256:ab", "npu-sampler"),
    ("reg.example/yeto-ppu-train", "ppu-train"),
    ("reg.example/site-own:1", None),
    ("yeto-ppu-train-extra:1", None),
])
def test_catalog_image_recognizes_a_reference_by_repository(reference, expected):
    spec = images.catalog_image(reference)
    assert (spec.name if spec else None) == expected


def test_image_mismatch_only_judges_catalog_images():
    assert images.image_mismatch("site/own:1", cpu_arch="aarch64", device_family="npu") is None
    assert images.image_mismatch("r/yeto-npu-sampler:1", cpu_arch="aarch64", device_family="npu") is None
    assert "the node is x86_64" in images.image_mismatch(
        "r/yeto-npu-sampler:1", cpu_arch="x86_64", device_family="npu")
    assert "device_family cuda" in images.image_mismatch(
        "r/yeto-ppu-train:1", cpu_arch="x86_64", device_family="cuda")


# --- the locks against the Dockerfiles and the yeto pins --------------------


def test_ppu_lock_matches_the_dockerfile_build_args_and_the_yeto_pins():
    lock, text = _lock("ppu-train"), _dockerfile("ppu-train")
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", text, re.MULTILINE))
    assert lock["commits"]["miles"] == args["MILES_COMMIT"]
    assert lock["commits"]["verl"] == args["VERL_COMMIT"] == verl_pins.VERL_COMMIT
    assert lock["commits"]["mbridge"] == args["MBRIDGE_COMMIT"]
    assert lock["commits"]["megatron_bridge"] == args["MEGATRON_BRIDGE_COMMIT"]
    assert lock["packages"]["tilelang"]["version"] == args["TILELANG_VERSION"]
    assert lock["packages"]["peft"]["version"] == miles_pins.MILES_PEFT_VERSION
    # No commit for Megatron yet: the build demands one and records it.
    assert lock["commits"]["megatron"] is None and "MEGATRON_COMMIT is required" in text


def test_ppu_lock_keeps_the_ppu_base_versions_the_survey_read():
    packages = _lock("ppu-train")["packages"]
    assert packages["torch"]["version"] == "2.10.0"
    assert packages["triton"]["version"] == "3.6.0"
    assert packages["flash-linear-attention"]["version"] == "1.0.0"
    assert packages["sglang"]["version"] == "0.5.17"
    # TransformerEngine and tilelang must not fail the start check before the
    # first PPU probe says whether they work (N7.5).
    assert packages["transformer-engine"]["optional"] and packages["tilelang"]["optional"]
    assert packages["megatron-core"]["import"].endswith("hyper_connection")  # mHC


def test_npu_lock_matches_the_vllm_ascend_release_line():
    lock = _lock("npu-sampler")
    assert lock["packages"]["vllm-ascend"]["version"] == "0.30.0rc1"
    assert lock["packages"]["torch-npu"]["version"] == "2.10.0.post4"
    assert lock["cann"]["version"] == "9."  # 9.x in the container, not the host's 8.0.1
    assert lock["driver"]["enforce"] is False
    assert any("option A" in d for d in lock["decisions"])
    assert any("option B" in d for d in lock["decisions"])


def test_requirements_of_the_ppu_image_leave_the_base_packages_alone():
    text = (DOCKER / "ppu-train" / "requirements.txt").read_text()
    pinned = {line.split("=")[0].split(">")[0].split("[")[0].strip()
              for line in text.splitlines() if line and not line.startswith("#")}
    assert not pinned & {"torch", "triton", "sglang", "flash-linear-attention", "transformers"}
    assert "ray[default]==2.56.0" in text  # the one deliberate base change


def test_both_new_images_run_the_start_check_from_their_entrypoint():
    for name in ("ppu-train", "npu-sampler"):
        text = _dockerfile(name)
        assert 'ENTRYPOINT ["/opt/yeto/image/entrypoint.sh"]' in text, name
        assert "docker/image-assert/assert_image_versions.py" in text, name
        assert "BASE_IMAGE must be pinned by digest" in text, name
    entrypoint = (DOCKER / "image-assert" / "entrypoint.sh").read_text()
    assert "set -e" in entrypoint and 'exec "$@"' in entrypoint


# --- the start check --------------------------------------------------------


BASE_LOCK = {"image": "test", "cpu_arch": "x86_64", "python": None, "packages": {}}


def _rows(report) -> dict[str, str]:
    return {row["check"]: row["status"] for row in report.rows}


def test_start_check_passes_on_a_matching_image():
    lock = {**BASE_LOCK, "packages": {"pytest": {"version": pytest.__version__, "import": "pytest"}}}
    report = checker.run(lock, machine="x86_64")
    assert not report.failed and _rows(report)["import pytest"] == "ok"


def test_start_check_fails_on_the_wrong_arch_version_or_import():
    report = checker.run({**BASE_LOCK, "cpu_arch": "aarch64"}, machine="x86_64")
    assert [row["check"] for row in report.failed] == ["cpu_arch"]
    lock = {**BASE_LOCK, "packages": {
        "pytest": {"version": "0.0.1"},
        "no-such-dist-yeto": {"version": "1.0"},
        "pytest-asyncio": {"version": None, "import": "yeto_no_such_module"},
    }}
    failed = [row["check"] for row in checker.run(lock, machine="x86_64").failed]
    assert failed == ["package pytest", "package no-such-dist-yeto", "import yeto_no_such_module"]


def test_start_check_records_an_unpinned_version_and_tolerates_an_optional_gap():
    lock = {**BASE_LOCK, "packages": {
        "pytest": {"version": None},
        "no-such-dist-yeto": {"version": "1.0", "optional": True},
        "pytest-asyncio": {"version": None, "import": "yeto_no_such_module", "optional": True},
    }}
    report = checker.run(lock, machine="x86_64")
    assert not report.failed
    assert _rows(report)["package no-such-dist-yeto"] == "not-enforced"


def test_start_check_compares_the_public_version_only():
    assert checker.public_version("2.10.0+abc") == "2.10.0"


def test_start_check_reads_commits_cann_driver_and_required_files(tmp_path):
    (tmp_path / "commits.json").write_text(json.dumps({"miles": "abc", "verl": "def",
                                                       "megatron": "012"}))
    (tmp_path / "version.cfg").write_text("package=Ascend-cann-toolkit\nversion=9.2.0\n")
    (tmp_path / "driver.info").write_text("Version=26.0.RC1\n")
    lock = {**BASE_LOCK,
            "commits": {"miles": "abc", "verl": "xxx", "megatron": None, "absent": None},
            "cann": {"version": "9.", "version_files": [str(tmp_path / "missing"),
                                                        str(tmp_path / "version.cfg")]},
            "driver": {"min": "24.1", "version_file": str(tmp_path / "driver.info")},
            "required_files": [str(tmp_path / "commits.json"), str(tmp_path / "absent")]}
    report = checker.run(lock, machine="x86_64", commits_file=tmp_path / "commits.json")
    rows = _rows(report)
    assert rows["commit miles"] == "ok" and rows["commit verl"] == "FAIL"
    # Recorded, not pinned: present passes, absent fails (the build must write it).
    assert rows["commit megatron"] == "ok" and rows["commit absent"] == "FAIL"
    assert rows["cann"] == "ok" and rows["driver"] == "ok"
    assert rows[f"file {tmp_path / 'absent'}"] == "FAIL"


def test_start_check_fails_a_driver_below_the_minimum_unless_the_lock_says_otherwise(tmp_path):
    info = tmp_path / "driver.info"
    info.write_text("Version=24.1.0.3\n")
    lock = {**BASE_LOCK, "driver": {"min": "26.0", "version_file": str(info)}}
    assert [row["check"] for row in checker.run(lock, machine="x86_64").failed] == ["driver"]
    lock["driver"]["enforce"] = False
    assert not checker.run(lock, machine="x86_64").failed


def test_start_check_exit_code_and_report_file(tmp_path):
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps({**BASE_LOCK, "cpu_arch": "aarch64"}))
    report = tmp_path / "report.json"
    assert checker.main([str(lock_path), "--report", str(report)]) == 1
    assert json.loads(report.read_text())["rows"][0]["check"] == "cpu_arch"
    lock_path.write_text(json.dumps({**BASE_LOCK, "cpu_arch": images.canonical_cpu_arch(
        __import__("platform").machine())}))
    assert checker.main([str(lock_path)]) == 0
