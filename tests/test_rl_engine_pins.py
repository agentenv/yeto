"""rl-engine-ports tasks 1.1/1.2/1.5: pin groups and remote prep scripts."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import yeto.rl as rl
from yeto.launcher import _miles_source_setup
from yeto.rl.ssh_harness import _host_setup_script, _source_checkout_script

def _minimal_plan(**extra):
    plan = {
        "remote_run": "yeto-rl/run-golden",
        "docker_image": "ghcr.io/agentenv/miles@sha256:" + "8" * 64,
        "rl_engine": "ports",
        "miles": {"repository": rl.MILES_NEXT_REPOSITORY, "commit": rl.MILES_NEXT_COMMIT,
                  "peft_version": rl.MILES_PEFT_VERSION},
        "sglang": {"repository": rl.SGLANG_NEXT_REPOSITORY, "commit": rl.SGLANG_NEXT_COMMIT},
        "learner": {},
    }
    plan.update(extra)
    return plan


def test_legacy_pins_and_bundle_are_gone():
    # rl-code-quality-cleanup 1.10: the agentenv fork pins went with the legacy engine.
    for name in ("MILES_COMMIT", "MILES_BASE_COMMIT", "MILES_BUNDLE_PATH", "MILES_IMAGE",
                 "MILES_LEGACY_PINS", "SGLANG_COMMIT", "SGLANG_REPOSITORY"):
        assert not hasattr(rl, name), name
    assert not list((Path(rl.__file__).parent / "vendor").glob("*.bundle"))


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
    miles_setup, sglang_setup = _miles_source_setup()
    _assert_ports_checkout(miles_setup + "\n" + sglang_setup, "~/{name}")
    assert miles_setup.startswith("set -e\n")


def test_harness_ports_setup_fetches_forks_without_bundle():
    checkout = _source_checkout_script(_minimal_plan())
    _assert_ports_checkout(checkout, '"$RUN/{name}"')
    script = _host_setup_script(_minimal_plan(), 4)
    assert checkout in script and "bundle" not in script.lower()


def _fake_git(monkeypatch, tmp_path, *, commit, origin, status=""):
    from yeto.rl.adapters.miles import revision
    responses = {
        ("rev-parse", "HEAD"): commit,
        ("rev-parse", "--abbrev-ref", "HEAD"): "HEAD",
        ("config", "--get", "remote.origin.url"): origin,
        ("status", "--porcelain", "--untracked-files=all"): status,
    }

    def run(argv, **_kwargs):
        return types.SimpleNamespace(stdout=responses[tuple(argv[3:])] + "\n")

    monkeypatch.setattr("yeto.rl.adapters.miles.revision.subprocess.run", run)
    monkeypatch.setattr(
        "yeto.rl.adapters.miles.revision.importlib.import_module",
        lambda _name: types.SimpleNamespace(__file__=tmp_path / "miles/__init__.py"),
    )
    return revision.verify_miles_revision, responses


def test_verify_miles_revision_accepts_ports_group(tmp_path, monkeypatch):
    verify, _ = _fake_git(
        monkeypatch,
        tmp_path,
        commit=rl.MILES_NEXT_COMMIT,
        origin=rl.MILES_NEXT_REPOSITORY + ".git",
    )
    assert verify(tmp_path) == tmp_path.resolve()


@pytest.mark.parametrize("origin", [
    "https://github.com/new-home/miles",
    "https://github.com/new-home/miles.git",
    # an image built before a move keeps the former origin in /root/miles
    rl.MILES_NEXT_REPOSITORY,
    rl.MILES_NEXT_REPOSITORY + ".git/",
])
def test_verify_miles_revision_accepts_current_and_former_origin(tmp_path, monkeypatch, origin):
    from yeto.rl.adapters.miles.pins import MilesRevisionPins

    moved = MilesRevisionPins("https://github.com/new-home/miles", rl.MILES_NEXT_COMMIT,
                              (rl.MILES_NEXT_REPOSITORY,))
    verify, _ = _fake_git(monkeypatch, tmp_path, commit=rl.MILES_NEXT_COMMIT, origin=origin)
    assert verify(tmp_path, expected=moved) == tmp_path.resolve()
    assert not moved.accepts_origin("https://github.com/radixark/miles")


def test_island_commit_override_keeps_the_pinned_origins():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.island_entry import miles_revision_pins

    pins = miles_revision_pins(SimpleNamespace(miles_commit="a" * 40))
    assert pins.repository == rl.MILES_NEXT_REPOSITORY
    assert pins.former_repositories == rl.MILES_NEXT_PINS.former_repositories


def test_verify_miles_revision_rejects_a_branch_checkout(tmp_path, monkeypatch):
    verify, responses = _fake_git(
        monkeypatch, tmp_path, commit=rl.MILES_NEXT_COMMIT, origin=rl.MILES_NEXT_REPOSITORY
    )
    responses[("rev-parse", "--abbrev-ref", "HEAD")] = "main"
    with pytest.raises(RuntimeError, match="not detached"):
        verify(tmp_path)


def test_verify_miles_revision_ports_rejects_commit_mismatch(tmp_path, monkeypatch):
    verify, _ = _fake_git(
        monkeypatch, tmp_path, commit="f" * 40, origin=rl.MILES_NEXT_REPOSITORY
    )
    with pytest.raises(RuntimeError, match="revision mismatch"):
        verify(tmp_path, expected=rl.MILES_NEXT_PINS)


def test_verify_miles_revision_ports_rejects_origin_mismatch(tmp_path, monkeypatch):
    verify, _ = _fake_git(
        monkeypatch, tmp_path, commit=rl.MILES_NEXT_COMMIT,
        origin="https://github.com/agentenv/miles"
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


# -- rl-code-quality-cleanup 0.7: one source for fork lineage and harness pins --
def test_fork_lineage_is_the_single_source_of_every_fork_commit_set():
    from yeto.rl.adapters.miles import entry
    from yeto.rl.adapters.miles import pins
    from yeto.rl.algos import critic_fork, loss_variants

    lineage = pins.MILES_FORK_LINEAGE
    assert len(set(lineage)) == len(lineage) and lineage[-1] == rl.MILES_NEXT_COMMIT
    # Miles declarations: 0af62f4d onwards; loss variants: 5c1b49eb onwards (one fewer).
    assert entry._PINS_0AF62F4D_PLUS == frozenset(lineage)
    assert loss_variants.FORK_COMMITS == frozenset(lineage[1:])
    assert loss_variants.FORK_COMMITS < entry._PINS_0AF62F4D_PLUS
    assert critic_fork.FORK_COMMITS == frozenset(pins.CRITIC_FORK_LINEAGE)
    assert critic_fork.CRITIC_FORK_PIN == pins.CRITIC_FORK_LINEAGE[-1]


def test_appending_a_fork_commit_reaches_every_derived_set(monkeypatch):
    from yeto.rl.adapters.miles import pins

    new = "f" * 40
    monkeypatch.setattr(pins, "MILES_FORK_LINEAGE", pins.MILES_FORK_LINEAGE + (new,))
    assert new in pins.miles_fork_lineage_from(pins.MILES_FORK_LINEAGE[0])
    assert new in pins.miles_fork_lineage_from("5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba")


def test_ssh_harness_pins_come_from_the_pin_module():
    from yeto.rl import ssh_harness
    from yeto.rl.adapters.miles import pins

    for name in ("TMS_PRELOAD_PATCH_REPOSITORY", "TMS_PRELOAD_PATCH_BASE_COMMIT",
                 "TMS_PRELOAD_PATCH_COMMIT", "TMS_PRELOAD_BASE_BINARY_SHA256"):
        assert getattr(ssh_harness, name) == getattr(pins, name)
    source = (Path(ssh_harness.__file__)).read_text()
    assert 'TMS_PRELOAD_PATCH_COMMIT = "' not in source
