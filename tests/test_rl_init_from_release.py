"""--rl-init-from-release: a trainer starts from its store's highest release."""

import json
from types import SimpleNamespace

import pytest

from yeto.rl.engine.init_from_release import (
    RECORD, InitFromReleaseError, check_resumed, island_model_path, main, prepare, read_record, select_release,
)
from yeto.rl.engine.merged_release import resumed_releases
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore

SEL = "a" * 64
PARTIAL = {"expert_selection_sha256": SEL, "partial_experts": {"model": "kimi_k31", "num_trainable": 2}}


def _base(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model-1.safetensors").write_bytes(b"base-1")
    (base / "model-2.safetensors").write_bytes(b"base-2")
    (base / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layers.1.experts.0.w1": "model-1.safetensors",
                        "layers.1.experts.1.w1": "model-1.safetensors",
                        "layers.2.dense": "model-2.safetensors"}}))
    return base


def _release(releases, version, tag, *, manifest=PARTIAL):
    def write(directory):
        (directory / "config.json").write_text("{}")
        (directory / "model-00001.safetensors").write_bytes(tag.encode())
        (directory / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"layers.1.experts.1.w1": "model-00001.safetensors"}}))
    return releases.release(version, f"hash-{version}-{tag}", write, manifest=manifest)


@pytest.fixture
def store(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=5)
    for version in (1, 2, 3):
        _release(releases, version, f"v{version}")
    return releases


def test_latest_is_the_highest_version_and_a_lower_one_is_refused(store):
    assert select_release(store.records(), "latest").path == "r3-v3"
    assert select_release(store.records(), "r3-v3").path == "r3-v3"
    assert select_release(store.records(), "3").path == "r3-v3"
    with pytest.raises(InitFromReleaseError, match="not the store's highest version"):
        select_release(store.records(), "r2-v2")
    with pytest.raises(InitFromReleaseError, match="no release"):
        select_release(store.records(), "r9-v9")


def test_highest_version_wins_over_latest_when_latest_went_back(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=5)
    _release(releases, 3, "a")
    _release(releases, 2, "b")  # a store written by a relaunch that reused numbers
    assert releases.latest().version == 2
    assert select_release(releases.records(), "latest").path == "r1-v3"


def test_prepare_lays_the_release_over_the_base(tmp_path, store):
    base = _base(tmp_path)
    out = tmp_path / "init"
    record = prepare(store, "latest", base, out, expected_selection_sha256=SEL)
    assert record["version"] == 3 and record["policy_hash"] == "hash-3-v3" and record["partial"]
    model = out / "model"
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    released = index["layers.1.experts.1.w1"]
    assert released.startswith("release-") and (model / released).read_bytes() == b"v3"
    assert (model / index["layers.1.experts.0.w1"]).read_bytes() == b"base-1"  # frozen: the base's
    assert (model / index["layers.2.dense"]).read_bytes() == b"base-2"
    assert json.loads((out / RECORD).read_text()) == record and not (model / RECORD).exists()
    path, again = island_model_path(out, base)
    assert path == str(model) and again == record


def test_prepare_refuses_another_selection_or_none(tmp_path, store):
    base = _base(tmp_path)
    with pytest.raises(InitFromReleaseError, match="expert selection"):
        prepare(store, "latest", base, tmp_path / "x", expected_selection_sha256="b" * 64)
    with pytest.raises(InitFromReleaseError, match="--rl-expert-selection"):
        prepare(store, "latest", base, tmp_path / "y")


def test_a_full_release_is_used_as_it_is(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    _release(releases, 4, "full", manifest=None)
    record = prepare(releases, "latest", _base(tmp_path), tmp_path / "init")
    assert not record["partial"] and record["model_dir"].endswith("r1-v4")


def test_island_refuses_a_release_laid_over_another_base(tmp_path, store):
    prepare(store, "latest", _base(tmp_path), tmp_path / "init", expected_selection_sha256=SEL)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(InitFromReleaseError, match="laid over"):
        island_model_path(tmp_path / "init", other)
    with pytest.raises(InitFromReleaseError, match="did not prepare"):
        island_model_path(tmp_path / "missing", other)


def test_resumed_numbering_must_start_at_the_loaded_release(tmp_path, store):
    record = prepare(store, "latest", _base(tmp_path), tmp_path / "init", expected_selection_sha256=SEL)
    resumed = resumed_releases(store, 2)
    check_resumed(record, resumed.base_version, resumed.history())  # v3, its hash
    with pytest.raises(InitFromReleaseError, match="resumes its numbering at v2"):
        check_resumed(record, 2, resumed.history())
    with pytest.raises(InitFromReleaseError, match="numbering at vNone"):
        check_resumed(record, None, ())
    with pytest.raises(InitFromReleaseError, match="policy hash"):
        check_resumed(record, 3, [(3, "other")])


def test_resume_single_trainer_checks_and_reports_the_loaded_release(tmp_path, store):
    from yeto.rl.adapters.miles.sampler_island import _resume_single_trainer

    record = prepare(store, "latest", _base(tmp_path), tmp_path / "init", expected_selection_sha256=SEL)
    events = []
    driver = SimpleNamespace(_max_policy_age=lambda: 2, emit=lambda name, **kw: events.append((name, kw)))
    assert _resume_single_trainer(driver, store, init_release=record) == {"resumed_base": 3}
    assert ("rl_trainer_init_from_release",
            {"release": "r3-v3", "version": 3, "policy_hash": "hash-3-v3"}) in events
    _release(store, 4, "v4")  # another trainer released meanwhile: v4 is not what was loaded
    with pytest.raises(InitFromReleaseError):
        _resume_single_trainer(driver, store, init_release=record)


def test_cli_prepares_from_the_store_path(tmp_path, store, capsys):
    base, out = _base(tmp_path), tmp_path / "init"
    argv = ["--release", "latest", "--store", str(tmp_path / "store"), "--model", str(base), "--out", str(out)]
    with pytest.raises(InitFromReleaseError, match="--rl-expert-selection"):
        main(argv, environ={})
    full = tmp_path / "full"
    _release(PolicyReleaseStore(CheckpointStore(full)), 1, "f", manifest=None)
    assert main(["--release", "r1-v1", "--store", str(full), "--model", str(base), "--out", str(out)],
                environ={}) == 0
    assert read_record(out)["path"] == "r1-v1" and "[yeto-init-release]" in capsys.readouterr().out
    with pytest.raises(InitFromReleaseError, match="no store path"):
        main(["--release", "latest", "--model", str(base), "--out", str(out)], environ={})


def test_launcher_prepares_the_release_on_every_node_before_ray():
    from yeto.launch.rl_island_task import init_release_block

    args = SimpleNamespace(rl_init_from_release="latest", model_revision="d" * 40,
                           rl_expert_selection="/local/sel.json")
    block = init_release_block(args, "moonshotai/Kimi-K3.1-Pro")
    assert block.startswith("PYTHONPATH=$HOME/sky_workdir")
    assert "-m yeto.rl.engine.init_from_release --release latest --model moonshotai/Kimi-K3.1-Pro" in block
    assert "--out ~/yeto-init-release" in block and f"--model-revision {'d' * 40}" in block
    assert "--expert-selection ~/yeto-rl-expert-selection.json" in block and "exit 1" in block
    assert init_release_block(SimpleNamespace(rl_init_from_release=None), "m") == ""


def test_prepare_replaces_an_earlier_overlay(tmp_path, store):
    base = _base(tmp_path)
    out = tmp_path / "init"
    (out / "model").mkdir(parents=True)
    (out / "model" / "release-stale.safetensors").write_bytes(b"old")
    prepare(store, "latest", base, out, expected_selection_sha256=SEL)
    model = out / "model"
    assert not (model / "release-stale.safetensors").exists()
    assert sorted(p.name for p in out.iterdir() if p.name.startswith(".model")) == []
