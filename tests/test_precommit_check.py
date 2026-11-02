"""Rules of scripts/precommit_check.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "precommit_check", Path(__file__).resolve().parents[1] / "scripts" / "precommit_check.py")
pc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pc)


@pytest.mark.parametrize("text", [b"dial 10.1.2.3.", b"host 192.168.0.7.\n", b"(172.16.4.5)", b"10.0.0.1:8080"])
def test_private_ip_is_found_also_before_a_full_stop(text):
    assert pc.PRIVATE_IP_RE.search(text)


@pytest.mark.parametrize("text", [b"10.1.2.3.4", b"192.0.2.74.", b"v1.10.2.3", b"172.32.0.1"])
def test_private_ip_ignores_longer_dotted_numbers_and_public_ranges(text):
    assert not pc.PRIVATE_IP_RE.search(text)


@pytest.mark.parametrize("text", [b"cd /home/alice", b"/home/alice.", b"HOME=/Users/bob\n", b"/home/alice/x"])
def test_home_path_without_trailing_slash_is_personal(text):
    assert pc._personal_home(text)


@pytest.mark.parametrize("name", [b"yeto", b"sky", b"ubuntu", b"runner", b"root", b"ec2-user"])
def test_generic_account_names_are_exempt(name):
    assert not pc._personal_home(b"/home/" + name + b"/work and /home/" + name)


@pytest.mark.parametrize("name", [b"skyler", b"yetodev", b"ubuntu2"])
def test_exemption_is_exact_not_a_prefix(name):
    assert pc._personal_home(b"/home/" + name + b"/work")


def test_check_reports_a_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("ssh to 10.0.0.5.\n")
    assert any("private-network IP" in p for p in pc.check([str(f)]))


def test_lock_file_versions_are_not_private_ips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("uv.lock").write_text('version = "10.3.9.90"\n')
    assert not any("private-network IP" in p for p in pc.check(["uv.lock"]))


def test_tenant_id_is_rejected(tmp_path):
    f = tmp_path / "plan.py"
    f.write_text("TENANT = 'tenant-" + "e00a" * 6 + "'\n")
    assert any("tenant ID" in p for p in pc.check([str(f)]))


@pytest.mark.parametrize("text", [b"tenant-short", b"the tenant-id flag", b"tenant-ABCDEFGHIJKLMNOPQRSTUV"])
def test_tenant_rule_needs_a_long_lowercase_id(text):
    assert not pc.TENANT_RE.search(text)


@pytest.mark.parametrize("text", [b"image: ghcr.io/alice/miles:v1", b"docker:ghcr.io/bob-dev/r@sha256:ab"])
def test_personal_registry_is_found(text):
    assert pc._personal_registry(text)


@pytest.mark.parametrize("text", [
    b"ghcr.io/agentenv/miles", b"ghcr.io/x/yeto-release@sha256:aa", b"docker:ghcr.io/o/r:latest",
    b"ghcr.io/<owner>/yeto", b"ghcr.io/${OWNER}/yeto", b"ghcr.io/$OWNER/yeto", b"IMAGE_REF=ghcr.io/...@sha256",
    b"ghcr.io/owner/yeto", b"ghcr.io/your-org/yeto",
])
def test_organizations_and_placeholders_are_not_personal_registries(text):
    assert not pc._personal_registry(text)


def test_check_reports_personal_registry(tmp_path):
    f = tmp_path / "build.sh"
    f.write_text("REG=ghcr.io/alice\n")
    assert any("ghcr.io/alice" in p for p in pc.check([str(f)]))


@pytest.mark.parametrize("text", ["中文", "note：", "a、b", "x。"])
def test_cjk_is_rejected(tmp_path, text):
    f = tmp_path / "doc.md"
    f.write_text(text, encoding="utf-8")
    assert any("CJK" in p for p in pc.check([str(f)]))


def test_cjk_rule_ignores_english_accents_and_binary_files(tmp_path):
    text = tmp_path / "doc.md"
    text.write_text("naïve café — ± →\n", encoding="utf-8")
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\0" + "中".encode())
    assert pc.check([str(text), str(binary)]) == []


@pytest.mark.parametrize("text, tags", [
    (b"(P0 1.7b.2b)", ["P0"]), (b"S29 spot", ["S29"]), (b"tests/test_s30_vm.py", ["s30"]),
    (b"s29next island", ["s29next"]), (b"G1 and E2b", ["E2b", "G1"]), (b"P0-b", ["P0-b"]), (b"fix D17", ["D17"]),
])
def test_phase_tags_are_found(text, tags):
    assert pc.phase_tags(text) == tags


@pytest.mark.parametrize("text", [b"upload to S3", b"s3://bucket/key", b"S3_PREFIX", b"sha256 SM100 D41 G10x 3D"])
def test_phase_tags_ignore_s3_and_other_names(text):
    assert pc.phase_tags(text) == []


def test_phase_tags_warn_without_failing(tmp_path):
    f = tmp_path / "notes.py"
    f.write_text("# S29 G1 fix\n")
    warnings: list[str] = []
    assert pc.check([str(f)], warnings) == []
    assert warnings and "S29" in warnings[0]


def test_phase_tags_ignore_single_digit_lowercase_variables():
    assert pc.phase_tags(b"s0, s1 = split(x)") == []
