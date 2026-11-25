"""The syncer builds under an isolated HOME with the caller's own Rust toolchain
(yeto.launch.cargo_toolchain); with no toolchain at all a syncer test skips with
the reason, it never fails for want of one."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import syncer_build
from yeto.launch import cargo_toolchain, syncer

ROOT = Path(__file__).resolve().parent.parent


def _rustup_proxy(path: Path) -> None:
    """A stand-in for rustup's cargo proxy: runs only with a toolchain under RUSTUP_HOME."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/bin/sh\ntest -d "${RUSTUP_HOME:-$HOME/.rustup}/toolchains/stable" || exit 1\necho cargo 1.0\n')
    path.chmod(0o755)


def _owner(tmp_path: Path) -> Path:
    owner = tmp_path / "owner"
    (owner / ".rustup/toolchains/stable").mkdir(parents=True)
    _rustup_proxy(owner / ".cargo/bin/cargo")
    return owner


def test_a_working_environment_is_used_as_it_is(tmp_path):
    owner = _owner(tmp_path)
    tc = cargo_toolchain.resolve({"HOME": str(owner), "PATH": str(owner / ".cargo/bin")}, owner_home=str(owner))
    assert tc.ok and tc.env == {} and tc.cargo == str(owner / ".cargo/bin/cargo")


def test_an_isolated_home_takes_the_owners_toolchain(tmp_path):
    owner, empty = _owner(tmp_path), tmp_path / "empty"
    empty.mkdir()
    env = {"HOME": str(empty), "PATH": f"{owner / '.cargo/bin'}:/usr/bin:/bin"}
    tc = cargo_toolchain.resolve(env, owner_home=str(owner))
    assert tc.ok and tc.env == {"RUSTUP_HOME": str(owner / ".rustup"), "CARGO_HOME": str(owner / ".cargo")}
    assert subprocess.run([tc.cargo, "--version"], env={**env, **tc.env}).returncode == 0


def test_cargo_off_path_comes_from_the_owners_cargo_home(tmp_path):
    owner, empty = _owner(tmp_path), tmp_path / "empty"
    empty.mkdir()
    tc = cargo_toolchain.resolve({"HOME": str(empty), "PATH": "/usr/bin:/bin"}, owner_home=str(owner))
    assert tc.ok and tc.env["PATH"] == f"/usr/bin:/bin:{owner / '.cargo/bin'}"  # appended: shadows nothing
    assert tc.cargo == str(owner / ".cargo/bin/cargo")


def test_a_variable_the_caller_set_is_kept(tmp_path):
    """RUSTUP_HOME set (to a home with no toolchain) is the caller's choice: not replaced, so
    cargo does not run, and the reason names it."""
    owner, empty = _owner(tmp_path), tmp_path / "empty"
    empty.mkdir()
    env = {"HOME": str(empty), "PATH": f"{owner / '.cargo/bin'}:/usr/bin:/bin", "RUSTUP_HOME": str(empty)}
    tc = cargo_toolchain.resolve(env, owner_home=str(owner))
    assert not tc.ok and tc.env == {} and f"RUSTUP_HOME={str(empty)!r}" in tc.reason


def test_no_toolchain_says_why(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    tc = cargo_toolchain.resolve({"HOME": str(empty), "PATH": str(empty)}, owner_home=str(empty))
    assert not tc.ok and tc.cargo is None and "no Rust toolchain" in tc.reason and str(empty) in tc.reason


def test_every_test_that_runs_cargo_is_known_to_need_it():
    """A test that runs cargo itself, outside a syncer_binary fixture, a needs_cargo marker
    or a skip on a failed build, is not seen by the skip rule: with no toolchain it fails
    instead of skipping. The allowed files name cargo in a command without running it."""
    import re

    runs = re.compile(r'\[\s*(?:"cargo"|CARGO)\s*,\s*"build"')
    allowed = {"test_syncer_reaper.py", "test_rl_benchmark.py", "test_cargo_toolchain.py", "syncer_build.py"}
    seen = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        if path.name in allowed:
            continue
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if not runs.search(line):
                continue
            head = "\n".join(lines[max(0, i - 12):i])
            tail = "\n".join(lines[i:i + 4])
            fixture = re.findall(r"def (\w+)\(", head)
            known = ((fixture and fixture[-1] in ("syncer_binary", "build_syncer"))
                     or "needs_cargo" in head  # its skipif marker names cargo
                     or 'pytest.skip("needs cargo' in tail)  # skips when the build fails
            if not known:
                seen.append(f"{path.name}:{i + 1}")
    assert seen == [], seen


def test_the_launchers_build_is_unchanged_with_a_working_toolchain(monkeypatch):
    calls = []
    monkeypatch.setattr(syncer.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr(cargo_toolchain, "resolve", lambda: cargo_toolchain.Toolchain(cargo="/bin/cargo"))
    syncer.build_syncer_binary()
    assert calls == [((["cargo", "build", "--release"],), {"cwd": ROOT / "syncer", "check": True})]
    calls.clear()
    monkeypatch.setattr(cargo_toolchain, "resolve",
                        lambda: cargo_toolchain.Toolchain(env={"RUSTUP_HOME": "/o/.rustup"}, cargo="/bin/cargo"))
    syncer.build_syncer_binary()
    ((argv,), kwargs), = calls
    assert argv == ["cargo", "build", "--release"] and kwargs["env"]["RUSTUP_HOME"] == "/o/.rustup"
    assert kwargs["env"]["PATH"] == os.environ["PATH"]


def test_a_binary_is_current_only_when_no_source_is_newer(tmp_path):
    crate = tmp_path / "syncer"
    (crate / "src").mkdir(parents=True)
    for name in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
        (crate / name).write_text("x")
        os.utime(crate / name, (100, 100))
    binary = tmp_path / "yeto-syncer"
    assert not syncer_build.binary_is_current(binary, crate)
    binary.write_text("bin")
    os.utime(binary, (200, 200))
    assert syncer_build.binary_is_current(binary, crate)
    os.utime(crate / "src/main.rs", (300, 300))
    assert not syncer_build.binary_is_current(binary, crate)


def test_without_a_toolchain_a_syncer_test_skips_with_the_reason(monkeypatch):
    monkeypatch.setattr(syncer_build, "TOOLCHAIN", cargo_toolchain.Toolchain(reason="no Rust toolchain runs cargo (x)"))
    own_cargo = SimpleNamespace(fixturenames=("syncer_binary",), module=SimpleNamespace(), iter_markers=lambda _n: [])
    shared = SimpleNamespace(fixturenames=(), module=SimpleNamespace(build_syncer=syncer_build.build_syncer),
                             iter_markers=lambda _n: [])
    marked = SimpleNamespace(fixturenames=(), module=SimpleNamespace(),
                             iter_markers=lambda _n: [SimpleNamespace(kwargs={"reason": "needs cargo to build"})])
    other = SimpleNamespace(fixturenames=("tmp_path",), module=SimpleNamespace(), iter_markers=lambda _n: [])
    assert syncer_build.skip_reason(own_cargo) == "needs cargo to build the syncer: no Rust toolchain runs cargo (x)"
    assert syncer_build.skip_reason(marked) is not None and syncer_build.skip_reason(other) is None
    monkeypatch.setattr(syncer_build, "binary_is_current", lambda: True)  # a binary built before: used as built
    assert syncer_build.skip_reason(shared) is None and syncer_build.skip_reason(own_cargo) is not None
    monkeypatch.setattr(syncer_build, "binary_is_current", lambda: False)
    assert syncer_build.skip_reason(shared) is not None
    monkeypatch.setattr(syncer_build, "TOOLCHAIN", cargo_toolchain.Toolchain(cargo="/bin/cargo"))
    assert syncer_build.skip_reason(own_cargo) is None


def _pytest_under(home: Path, tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    keep = {k: v for k, v in os.environ.items() if k not in ("CARGO_HOME", "RUSTUP_HOME", "PYTEST_ADDOPTS")}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider",
                           f"--basetemp={tmp_path / 'bt'}", *args], cwd=ROOT, capture_output=True, text=True,
                          env={**keep, "HOME": str(home), **env}, timeout=900)


@pytest.mark.skipif(not cargo_toolchain.resolve(
    {k: v for k, v in os.environ.items() if k not in ("CARGO_HOME", "RUSTUP_HOME")}).ok,
    reason="needs cargo to build the syncer")
def test_real_syncer_tests_run_under_an_empty_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    p = _pytest_under(home, tmp_path, "tests/test_member_retire.py::test_drain_then_retire_on_the_real_syncer",
                      "tests/test_delta_protocol.py::test_versioned_hello_golden_bytes_and_reserved_ids")
    assert p.returncode == 0 and "2 passed" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]


def test_no_toolchain_skips_the_real_syncer_tests_instead_of_failing(tmp_path):
    """RUSTUP_HOME set to an empty directory: no toolchain runs cargo. The test whose
    module runs cargo itself skips with the reason; nothing fails."""
    home = tmp_path / "home"
    home.mkdir()
    if cargo_toolchain.resolve({**os.environ, "HOME": str(home), "RUSTUP_HOME": str(home)}).ok:
        pytest.skip("a cargo that is not rustup's proxy runs without RUSTUP_HOME")
    p = _pytest_under(home, tmp_path, "tests/test_member_retire.py::test_drain_then_retire_on_the_real_syncer",
                      RUSTUP_HOME=str(home))
    assert p.returncode == 0 and "1 skipped" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]
    assert "needs cargo to build the syncer: no Rust toolchain runs cargo" in p.stdout
