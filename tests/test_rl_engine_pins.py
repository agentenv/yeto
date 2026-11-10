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


# -- rl-code-quality-cleanup 0.7 / k31 sync: lineages, features, evidence, harness pins --
def test_fork_lineage_is_the_single_source_of_every_fork_commit_set():
    from yeto.rl.adapters.miles import entry
    from yeto.rl.adapters.miles import pins
    from yeto.rl.algos import critic_fork, loss_variants

    v1 = pins.MILES_LINEAGES["ports-v1"]
    assert v1.commits is pins.MILES_FORK_LINEAGE and v1.commits[-1] == pins.MILES_PORTS_V1_COMMIT
    # Miles declarations: 0af62f4d onwards; loss variants: 5c1b49eb onwards (one fewer).
    assert entry.MILES_DECLARED_PINS["loss_aggregations:token"] == frozenset(v1.commits)
    # ... and on the k31 lineage from k31/b8 on (feature loss_variants in both tables)
    k31 = pins.MILES_LINEAGES["k31"]
    assert loss_variants.FORK_COMMITS == frozenset(v1.commits[1:]) | k31.from_commit(pins.MILES_K31_B8_COMMIT)
    assert loss_variants.FORK_COMMITS == pins.commits_with("loss_variants")
    # the critic family: the k31 lineage from k31/b8 on (no separate local chain any more)
    assert pins.CRITIC_FORK_LINEAGE == tuple(k31.commits[k31.commits.index(pins.MILES_K31_B8_COMMIT):])
    assert critic_fork.FORK_COMMITS == frozenset(pins.CRITIC_FORK_LINEAGE) == pins.commits_with("critic_family")
    assert critic_fork.CRITIC_FORK_PIN == pins.MILES_K31_B8_COMMIT == pins.CRITIC_FORK_LINEAGE[0]


def test_every_lineage_is_consistent():
    from yeto.rl.adapters.miles import pins

    seen: set[str] = set()
    for name, lineage in pins.MILES_LINEAGES.items():
        assert lineage.name == name
        assert len(set(lineage.commits)) == len(lineage.commits) >= 1
        assert not seen & set(lineage.commits)  # two lineages never share a commit
        seen |= set(lineage.commits)
        assert all(len(c) == 40 for c in (*lineage.commits, lineage.upstream_base))
        assert set(lineage.features.values()) <= set(lineage.commits)
        for feature, first in lineage.features.items():
            assert lineage.has(feature, first) and lineage.has(feature, lineage.commits[-1])
            before = lineage.commits[:lineage.commits.index(first)]
            assert not any(lineage.has(feature, c) for c in before)
    current = pins.MILES_LINEAGES[pins.CURRENT_LINEAGE]
    assert pins.MILES_NEXT_COMMIT == rl.MILES_NEXT_COMMIT and current.index(pins.MILES_NEXT_COMMIT) is not None
    # commits past the pin are reachable only through an image that names them (--miles-commit)
    later = current.commits[current.index(pins.MILES_NEXT_COMMIT) + 1:]
    assert set(later) <= set(pins.IMAGE_MILES_COMMITS.values())
    assert pins.image_miles_commit("docker:x@sha256:" + pins.MILES_K31_FULL_IMAGE.rpartition("@sha256:")[2]) \
        == pins.MILES_K31_FULL_E_COMMIT
    assert pins.MILES_NEXT_UPSTREAM_COMMIT == current.upstream_base
    assert pins.MILES_NEXT_PINS.commit == pins.MILES_NEXT_COMMIT
    assert pins.MILES_PORTS_V1_PINS.commit == pins.MILES_PORTS_V1_COMMIT


def test_pin_has_never_raises_and_is_lineage_local():
    from yeto.rl.adapters.miles import pins

    unknown = "f" * 40
    assert pins.lineage_of(unknown) is None and pins.lineage_of(None) is None
    assert pins.miles_fork_lineage_from(unknown) == frozenset()
    assert pins.pin_has("run_plugin", unknown) is False
    assert pins.pin_has("no_such_feature") is False
    # ports-v1 features stop at the ports-v1 head; the k31 lineage registers its own
    assert pins.pin_has("loss_variants", pins.MILES_PORTS_V1_COMMIT)
    assert pins.pin_has("dp_replica_full_parameter", pins.MILES_PORTS_V1_COMMIT)
    assert not pins.pin_has("loss_variants", pins.MILES_K31_F1_COMMIT)
    assert not pins.pin_has("loss_variants", pins.MILES_K31_B4_COMMIT)
    assert pins.pin_has("loss_variants", pins.MILES_K31_B8_COMMIT)
    assert not pins.pin_has("dp_replica_full_parameter", pins.MILES_K31_F1_COMMIT)
    assert pins.pin_has("run_plugin", pins.MILES_K31_F1_COMMIT) and pins.pin_has("yeto_ext")
    # the old "first commit" constants are the ports-v1 feature entries
    v1 = pins.PORTS_V1_FEATURES
    assert v1["distopt_full_parameter"] == pins.MILES_DISTOPT_FULL_PARAMETER_COMMIT
    assert v1["dp_replica_full_parameter"] == pins.MILES_DP_REPLICA_FULL_PARAMETER_COMMIT
    assert v1["dp_sharded_master"] == pins.MILES_DP_SHARDED_MASTER_COMMIT
    assert pins.miles_fork_lineage_from("5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba") == frozenset(
        pins.MILES_FORK_LINEAGE[1:])


def test_current_miles_commit_honours_the_island_override():
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import pins

    assert pins.current_miles_commit() == pins.MILES_NEXT_COMMIT
    assert pins.current_miles_commit(SimpleNamespace(miles_commit=None)) == pins.MILES_NEXT_COMMIT
    assert pins.current_miles_commit(SimpleNamespace(miles_commit="a" * 40)) == "a" * 40


def test_declared_evidence_is_lineage_bound():
    from yeto.rl.adapters.miles import entry, pins

    for mechanism, refs in entry.MILES_DECLARED_EVIDENCE.items():
        assert mechanism in entry.MILES_DECLARED
        for ref in refs:
            lineage = pins.MILES_LINEAGES[ref.lineage]
            assert ref.from_commit in lineage.commits and ref.run_id and ref.judgment_path
            assert entry.evidence_covers(ref, lineage.commits[-1])
            assert not entry.evidence_covers(ref, pins.MILES_K31_F1_COMMIT)
    # the 11 pin-bound mechanisms: declared on the ports-v1 head, not on the k31 pin
    v1 = entry.declared_by_dimension(pins.MILES_PORTS_V1_COMMIT)
    k31 = entry.declared_by_dimension(pins.MILES_K31_F1_COMMIT)
    for mechanism in entry.MILES_DECLARED_EVIDENCE:
        dimension, name = mechanism.split(":", 1)
        assert name in v1[dimension] and name not in k31.get(dimension, set())
    # the mechanisms without pin-bound evidence stay declared on both
    assert "tis" in v1["corrections"] and "tis" in k31["corrections"]
    assert entry.declared_by_dimension() == k31  # the default is the current pin


def test_ssh_harness_takes_no_tms_preload_patch():
    """The TMS preload patch belonged to the ports-v1 image; its pins stay in
    pins.py for that lineage, the harness no longer attests or mounts it."""
    from yeto.rl import ssh_harness
    from yeto.rl.adapters.miles import pins

    for name in ("TMS_PRELOAD_PATCH_REPOSITORY", "TMS_PRELOAD_PATCH_BASE_COMMIT",
                 "TMS_PRELOAD_PATCH_COMMIT", "TMS_PRELOAD_BASE_BINARY_SHA256"):
        assert getattr(pins, name)
        assert not hasattr(ssh_harness, name)
    source = Path(ssh_harness.__file__).read_text()
    assert "--tms-preload-patch" not in source and "TMS_PRELOAD" not in source


def test_sglang_k31_pin_is_the_branch_head_full_sha():
    from yeto.rl.adapters.miles import pins

    assert pins.SGLANG_K31_COMMIT == "35708f76dfb720f831f7028ddb68250b6caa8f12"
    assert pins.SGLANG_K31_BRANCH == "sglang-miles-kimi-k31"
    assert pins.SGLANG_NEXT_COMMIT == pins.SGLANG_K31_COMMIT
    assert pins.SGLANG_PORTS_V1_COMMIT == "2fa880182eefdb31e64d8ea70317be27f9f32121"
