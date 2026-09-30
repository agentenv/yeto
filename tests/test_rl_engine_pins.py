"""rl-engine-ports tasks 1.1/1.2/1.5: pin groups and remote prep scripts."""

from __future__ import annotations

import hashlib
import types

import pytest

import yeto.rl as rl
from yeto.launcher import _miles_source_setup
from yeto.rl.ssh_harness import _host_setup_script, _source_checkout_script

# Captured from main cb55aca, re-verified unchanged at e21a7ff (inline setup in make_miles_island_task; the
# #64 MILES_COMMIT/bundle bump is the only change vs the c40a32c golden).
LEGACY_LAUNCHER_MILES_SETUP = (
    "set -e\n"
    "MILES_BUNDLE=~/sky_workdir/yeto/rl/vendor/miles-qwen38.bundle\n"
    'test -f "$MILES_BUNDLE" && test ! -L "$MILES_BUNDLE"\n'
    "printf '%s  %s\\n' "
    "da3464d3c389f7e2cb3e390119b3c42c2f130d94a0711f9c95608f3243826cc6 "
    '"$MILES_BUNDLE" | sha256sum --check -\n'
    "if [ ! -d ~/miles/.git ]; then git clone --no-checkout "
    "https://github.com/agentenv/miles ~/miles; fi\n"
    "git -C ~/miles remote set-url origin https://github.com/agentenv/miles\n"
    "git -C ~/miles fetch --depth 1 origin "
    "6062afe0a9d5d6471e8395dedc81c78dd9f4a84f\n"
    "git -C ~/miles checkout --detach 6062afe0a9d5d6471e8395dedc81c78dd9f4a84f\n"
    'git -C ~/miles bundle verify "$MILES_BUNDLE" >/dev/null\n'
    'git -C ~/miles fetch "$MILES_BUNDLE" '
    "ae475060fa670145aef75d678809039ae999cb97\n"
    "git -C ~/miles checkout --detach ae475060fa670145aef75d678809039ae999cb97\n"
    'test "$(git -C ~/miles rev-parse HEAD)" = '
    "ae475060fa670145aef75d678809039ae999cb97\n"
    'test -z "$(git -C ~/miles status --porcelain --untracked-files=all)"\n'
    "python3 -m pip install -q --no-deps -e ~/miles 'peft==0.20.0'"
)
LEGACY_LAUNCHER_SGLANG_SETUP = (
    "if [ ! -d ~/sglang/.git ]; then git clone --no-checkout "
    "https://github.com/agentenv/sglang ~/sglang; fi\n"
    "git -C ~/sglang fetch --depth 1 https://github.com/agentenv/sglang "
    "e1b57eb8e7749235c987cc6b1b2824ce3265369b\n"
    "git -C ~/sglang checkout --detach e1b57eb8e7749235c987cc6b1b2824ce3265369b\n"
    "python3 -m pip install -q --no-deps -e ~/sglang/python"
)
# sha256 of main cb55aca (unchanged at e21a7ff) _host_setup_script(_minimal_plan(), 4).
LEGACY_HARNESS_SETUP_SHA256 = (
    "1eec095398f2ebbebea6f4a5f8db7a38b328cbf44c83f86605168ace05a53d1b"
)


def _minimal_plan(**extra):
    plan = {
        "remote_run": "yeto-rl/run-golden",
        "docker_image": "ghcr.io/agentenv/miles@sha256:" + "8" * 64,
        "miles": {
            "repository": rl.MILES_REPOSITORY,
            "base_commit": rl.MILES_BASE_COMMIT,
            "commit": rl.MILES_COMMIT,
            "bundle_path": rl.MILES_BUNDLE_PATH,
            "bundle_sha256": rl.MILES_BUNDLE_SHA256,
        },
        "sglang": {"repository": rl.SGLANG_REPOSITORY, "commit": rl.SGLANG_COMMIT},
        "learner": {},
    }
    plan.update(extra)
    return plan


def test_next_pin_groups_are_independent_of_legacy_and_bundle_free():
    assert rl.MILES_NEXT_REPOSITORY == "https://github.com/michaellchung/miles"
    assert rl.SGLANG_NEXT_REPOSITORY == "https://github.com/michaellchung/sglang"
    for value in (rl.MILES_NEXT_COMMIT, rl.SGLANG_NEXT_COMMIT):
        assert len(value) == 40 and set(value) <= set("0123456789abcdef")
    assert rl.MILES_NEXT_COMMIT not in {rl.MILES_COMMIT, rl.MILES_BASE_COMMIT}
    assert rl.SGLANG_NEXT_COMMIT != rl.SGLANG_COMMIT
    assert rl.MILES_NEXT_IMAGE != rl.MILES_IMAGE
    assert "@sha256:" in rl.MILES_NEXT_IMAGE
    assert rl.MILES_LEGACY_PINS == (rl.MILES_REPOSITORY, rl.MILES_COMMIT)
    assert rl.MILES_NEXT_PINS == (rl.MILES_NEXT_REPOSITORY, rl.MILES_NEXT_COMMIT)
    next_names = [n for n in dir(rl) if n.startswith(("MILES_NEXT", "SGLANG_NEXT"))]
    assert not any("BUNDLE" in name for name in next_names)
    assert not any("bundle" in str(getattr(rl, name)) for name in next_names)


def test_launcher_legacy_setup_is_byte_identical():
    assert _miles_source_setup("legacy") == (
        LEGACY_LAUNCHER_MILES_SETUP,
        LEGACY_LAUNCHER_SGLANG_SETUP,
    )
    # ports is the default engine.
    assert _miles_source_setup() == _miles_source_setup("ports")


def test_harness_legacy_setup_is_byte_identical():
    script = _host_setup_script(_minimal_plan(), 4)
    assert hashlib.sha256(script.encode()).hexdigest() == LEGACY_HARNESS_SETUP_SHA256
    assert _source_checkout_script(
        _minimal_plan(rl_engine="legacy")
    ) == _source_checkout_script(_minimal_plan())


def _assert_ports_checkout(script: str, path: str) -> None:
    lowered = script.lower()
    assert "bundle" not in lowered and "agentenv" not in lowered
    for repository, commit, name in (
        (rl.MILES_NEXT_REPOSITORY, rl.MILES_NEXT_COMMIT, "miles"),
        (rl.SGLANG_NEXT_REPOSITORY, rl.SGLANG_NEXT_COMMIT, "sglang"),
    ):
        where = path.format(name=name)
        assert f"git -C {where} remote set-url origin {repository}\n" in script
        assert (
            f'test "$(git -C {where} config --get remote.origin.url)" = '
            f"{repository}\n"
        ) in script
        assert f"git -C {where} fetch --depth 1 origin {commit}\n" in script
        assert f'test "$(git -C {where} rev-parse HEAD)" = {commit}\n' in script
        assert f"git -C {where} rev-parse --abbrev-ref HEAD" in script
        assert (
            f'test -z "$(git -C {where} status --porcelain --untracked-files=all)"'
        ) in script


def test_launcher_ports_setup_fetches_forks_without_bundle():
    miles_setup, sglang_setup = _miles_source_setup("ports")
    _assert_ports_checkout(miles_setup + "\n" + sglang_setup, "~/{name}")
    assert miles_setup.startswith("set -e\n")
    with pytest.raises(ValueError, match="rl_engine"):
        _miles_source_setup("bogus")


def test_harness_ports_setup_fetches_forks_without_bundle():
    checkout = _source_checkout_script(_minimal_plan(rl_engine="ports"))
    _assert_ports_checkout(checkout, '"$RUN/{name}"')
    script = _host_setup_script(_minimal_plan(rl_engine="ports"), 4)
    assert checkout in script and "bundle" not in script.lower()
    with pytest.raises(Exception, match="rl_engine"):
        _source_checkout_script(_minimal_plan(rl_engine="bogus"))


def _fake_git(monkeypatch, tmp_path, *, commit, origin, status=""):
    miles = pytest.importorskip("yeto.rl.miles")
    responses = {
        ("rev-parse", "HEAD"): commit,
        ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD",
        ("config", "--get", "remote.origin.url"): origin,
        ("status", "--porcelain", "--untracked-files=all"): status,
    }

    def run(argv, **_kwargs):
        return types.SimpleNamespace(stdout=responses[tuple(argv[3:])] + "\n")

    monkeypatch.setattr("yeto.rl.miles.subprocess.run", run)
    monkeypatch.setattr(
        "yeto.rl.miles.importlib.import_module",
        lambda _name: types.SimpleNamespace(__file__=tmp_path / "miles/__init__.py"),
    )
    return miles.verify_miles_revision, responses


def test_verify_miles_revision_accepts_ports_group(tmp_path, monkeypatch):
    verify, _ = _fake_git(
        monkeypatch,
        tmp_path,
        commit=rl.MILES_NEXT_COMMIT,
        origin=rl.MILES_NEXT_REPOSITORY + ".git",
    )
    assert verify(tmp_path, expected=rl.MILES_NEXT_PINS) == tmp_path.resolve()
    # The legacy default rejects a ports checkout.
    with pytest.raises(RuntimeError, match="revision mismatch"):
        verify(tmp_path)


def test_verify_miles_revision_ports_rejects_commit_mismatch(tmp_path, monkeypatch):
    verify, _ = _fake_git(
        monkeypatch, tmp_path, commit=rl.MILES_COMMIT, origin=rl.MILES_NEXT_REPOSITORY
    )
    with pytest.raises(RuntimeError, match="revision mismatch"):
        verify(tmp_path, expected=rl.MILES_NEXT_PINS)


def test_verify_miles_revision_ports_rejects_origin_mismatch(tmp_path, monkeypatch):
    verify, _ = _fake_git(
        monkeypatch, tmp_path, commit=rl.MILES_NEXT_COMMIT, origin=rl.MILES_REPOSITORY
    )
    with pytest.raises(RuntimeError, match="origin mismatch"):
        verify(tmp_path, expected=rl.MILES_NEXT_PINS)


def test_verify_miles_revision_ports_rejects_dirty_tree(tmp_path, monkeypatch):
    verify, _ = _fake_git(
        monkeypatch,
        tmp_path,
        commit=rl.MILES_NEXT_COMMIT,
        origin=rl.MILES_NEXT_REPOSITORY,
        status=" M miles/__init__.py",
    )
    with pytest.raises(RuntimeError, match="not clean"):
        verify(tmp_path, expected=rl.MILES_NEXT_PINS)
