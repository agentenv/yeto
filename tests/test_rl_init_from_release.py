"""--rl-init-from-release: a trainer starts from its store's highest release."""

import json
from pathlib import Path
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


def test_cli_prepares_a_delta_release(tmp_path, monkeypatch):
    """--rl-release-delta-full-every: the newest release is usually a delta of the one
    before it; a relaunch (--rl-init-from-release latest) must rebuild it from its chain."""
    import numpy as np

    from yeto.rl.engine import release_delta as rd
    from yeto.rl.engine.policy_release import DeltaChain

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    rng = np.random.default_rng(0)
    words = [rng.integers(0, 1 << 16, size=50_000, dtype=np.uint16)]
    for _ in range(2):
        nxt = words[-1].copy()
        nxt[rng.choice(nxt.size, 200, replace=False)] ^= 1
        words.append(nxt)
    trainer = PolicyReleaseStore(CheckpointStore(tmp_path / "store"), keep=2)
    chain = DeltaChain(full_every=8)
    for version, w in enumerate(words, start=1):
        staged = tmp_path / "staging" / f"v{version}"
        staged.mkdir(parents=True)
        (staged / "config.json").write_text("{}")
        (staged / "model.safetensors").write_bytes(w.astype("<u2").tobytes())
        release = trainer.release_staged(version, f"h{version}", staged, position=lambda: (version, None),
                                         delta_base=chain.base())
        chain.advance(release, staged)
    assert trainer.latest().delta is not None
    base, out = _base(tmp_path), tmp_path / "init"
    # a rebuild killed mid-way by an earlier launch (the node's disk outlives it): a
    # release-sized scratch copy that nothing else ever removes
    leftovers = [out / "mirror" / ".r3-v3.4242.0f0f.partial", out / "mirror" / ".r2-v2.4242.0e0e.stale"]
    for leftover in leftovers:
        leftover.mkdir(parents=True)
        (leftover / "model.safetensors").write_bytes(b"half")
    for if_released in ([], ["--if-released"]):
        assert main(["--release", "latest", "--store", str(tmp_path / "store"), "--model", str(base),
                     "--out", str(out), *if_released], environ={}) == 0
        record = read_record(out)
        assert record["path"] == "r3-v3"
        assert (Path(record["model_dir"]) / "model.safetensors").read_bytes() == words[-1].astype("<u2").tobytes()
    assert not any(leftover.exists() for leftover in leftovers)


def test_a_failed_prepare_leaves_its_reason_for_the_launcher(tmp_path, store, monkeypatch, capsys):
    """The node's last error line (e.g. how many GB a rebuild needs, how many are
    free) reaches the run's output instead of only a generic failure."""
    from yeto.rl.engine import disk_space
    from yeto.rl.engine import init_from_release as ifr

    def full_disk(*_a, **_k):
        raise disk_space.InsufficientDiskSpace("rebuild of delta release r3-v3: needs about 210.00 GB "
                                               "on /x, only 12.50 GB free")

    monkeypatch.setattr(ifr, "prepare", full_disk)
    out = tmp_path / "init"
    argv = ["--release", "latest", "--store", str(tmp_path / "store"), "--model", str(_base(tmp_path)),
            "--out", str(out)]
    assert ifr.cli(argv, environ={}) == 1
    reason = (out / ifr.FAILURE_FILE).read_text().strip()
    assert reason.startswith("InsufficientDiskSpace: ") and "210.00 GB" in reason and "12.50 GB free" in reason
    assert f"[yeto-init-release] FAILED: {reason}" in capsys.readouterr().err
    monkeypatch.setattr(ifr, "prepare", lambda *a, **k: {"path": "r3-v3"})
    assert ifr.cli(argv, environ={}) == 0
    assert not (out / ifr.FAILURE_FILE).exists()  # a later success clears an earlier reason


def test_the_node_failure_line_names_the_reason(tmp_path):
    """The run script's failure line carries the reason init_from_release left."""
    import os
    import subprocess

    from yeto.launch.rl_island_task import init_release_block
    from yeto.rl.engine.init_from_release import FAILURE_FILE

    home, bin_dir = tmp_path / "home", tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python3"  # stands in for a prepare that failed on a full disk
    stub.write_text("#!/bin/sh\nmkdir -p \"$HOME/yeto-init-release\"\n"
                    f"echo 'InsufficientDiskSpace: needs about 210.00 GB, only 12.50 GB free' "
                    f"> \"$HOME/yeto-init-release/{FAILURE_FILE}\"\nexit 1\n")
    stub.chmod(0o755)
    block = init_release_block(SimpleNamespace(rl_init_from_release="latest", model_revision=None,
                                               rl_expert_selection=None), "m")
    env = {**os.environ, "HOME": str(home), "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    done = subprocess.run(["bash", "-c", block], env=env, capture_output=True, text=True)
    assert done.returncode == 1
    assert "could not be prepared: InsufficientDiskSpace: needs about 210.00 GB, only 12.50 GB free" in done.stderr
    stub.write_text("#!/bin/sh\nexit 1\n")  # died before it could write a reason
    done = subprocess.run(["bash", "-c", block], env={**env, "HOME": str(tmp_path / "h2")},
                          capture_output=True, text=True)
    assert done.returncode == 1 and "could not be prepared" in done.stderr


def test_a_failure_reason_carries_no_credentials(tmp_path, store, monkeypatch):
    """The reason reaches the run's output and logs: secrets in an exception message
    (a presigned URL, URL userinfo, key=value, a secret env value) are redacted, and
    a long multi-line message becomes one bounded line."""
    from yeto.rl.engine import init_from_release as ifr

    secret = "Zk3uSECRETVALUE0123456789"

    def leaky(*_a, **_k):
        raise RuntimeError(f"GET https://acct.r2.cloudflarestorage.com/b/k?X-Amz-Signature=abcdef&X-Amz-Credential=AK\n"
                           f"s3://AKID:{secret}@host/b aws_secret_access_key={secret} token: tok123456 "
                           + "x" * 2000)

    monkeypatch.setattr(ifr, "prepare", leaky)
    out = tmp_path / "init"
    argv = ["--release", "latest", "--store", str(tmp_path / "store"), "--model", str(_base(tmp_path)),
            "--out", str(out)]
    assert ifr.cli(argv, environ={"AWS_SECRET_ACCESS_KEY": secret}) == 1
    reason = (out / ifr.FAILURE_FILE).read_text()
    assert reason.count("\n") == 1 and len(reason) <= ifr.FAILURE_REASON_MAX + 1
    for leaked in (secret, "abcdef", "X-Amz-Credential", "AKID", "tok123456"):
        assert leaked not in reason
    assert reason.startswith("RuntimeError: GET https://acct.r2.cloudflarestorage.com/b/k?<redacted>")


def test_redact_text_covers_bearer_basic_and_pem_and_keeps_counts():
    """Bearer / ``Authorization: Basic`` credentials and PEM private keys are redacted;
    digit-only counts (``max_tokens=8192``) and paths stay readable for diagnosis."""
    from yeto.rl.engine.init_from_release import redact_text

    fake = "Zk3u" + "FAKEVALUE" + "0123456789"
    text = (f"Authorization: Bearer {fake}\nAuthorization: Basic {fake}==\nbearer {fake}\n"
            "-----BEGIN " + "RSA PRIVATE KEY-----\n" + f"MIIE{fake}\n{fake}\n-----END RSA PRIVATE KEY-----\n"
            "max_tokens=8192 token_count=12 path=/data/key_value.bin")
    out = redact_text(text, {})
    assert fake not in out and "MIIE" not in out
    assert "max_tokens=8192 token_count=12 path=/data/key_value.bin" in out


def test_redact_text_is_linear_on_long_keyword_dense_words():
    """No quadratic backtracking: a long word full of keywords is redacted quickly."""
    import time

    from yeto.rl.engine.init_from_release import redact_text

    t0 = time.monotonic()
    redact_text("token" * 40000 + " " + "accesskey" * 20000, {})
    assert time.monotonic() - t0 < 2.0
