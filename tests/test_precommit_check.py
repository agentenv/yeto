"""Rules of scripts/precommit_check.py: private IPs and per-user home paths."""
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
