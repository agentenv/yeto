"""COLDSTART-PLAN #3/#4: Nebius baked VM image (pre-pulled docker layers) and
the cloud-local model store (``--model-store nebius-fs://...``)."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from yeto import launcher
from yeto.launcher import (
    model_store_env,
    model_store_filesystem,
    model_store_sky_config,
    nebius_baked_image_id,
)
from tests.test_rl_engine_selection import _cli, _island_task
from yeto.launcher import _prepare_rl_args

D1 = "sha256:" + "1" * 64
D2 = "sha256:" + "2" * 64
IMG1 = "docker:ghcr.io/o/r@" + D1
BAKED = {D1: {"eu-north1": "computeimage-aaa"}}


def test_baked_image_used_only_on_exact_digest_and_region(capsys):
    assert nebius_baked_image_id(IMG1, "nebius", "eu-north1", BAKED) == {
        "eu-north1": "computeimage-aaa",
        "docker": IMG1,
    }
    assert "pre-pulled" in capsys.readouterr().out
    # other clouds / no explicit region / other region: plain docker image
    assert nebius_baked_image_id(IMG1, "aws", "eu-north1", BAKED) == IMG1
    assert nebius_baked_image_id(IMG1, "nebius", None, BAKED) == IMG1
    assert nebius_baked_image_id(IMG1, "nebius", "us-central1", BAKED) == IMG1
    assert capsys.readouterr().out == ""
    # an unpinned tag is never matched
    assert nebius_baked_image_id("docker:ghcr.io/o/r:latest", "nebius", "eu-north1", BAKED) \
        == "docker:ghcr.io/o/r:latest"


def test_stale_bake_falls_back_with_warning(capsys):
    img2 = "docker:ghcr.io/o/r@" + D2
    assert nebius_baked_image_id(img2, "nebius", "eu-north1", BAKED) == img2
    out = capsys.readouterr().out
    assert "WARNING" in out and D1[:19] in out and "bake_nebius_image.sh" in out


def test_island_task_uses_baked_image_dict(monkeypatch):
    import yeto.rl

    args = _cli(["--gpu", "nebius:1xl40s@eu-north1"])
    _prepare_rl_args(args)
    digest = "sha256:" + args.rl_image.rsplit("@sha256:", 1)[1]
    monkeypatch.setattr(yeto.rl, "NEBIUS_BAKED_IMAGES", {digest: {"eu-north1": "computeimage-x"}})
    task = _island_task(args, monkeypatch)
    assert task.resources.image_id == {"eu-north1": "computeimage-x", "docker": args.rl_image}
    monkeypatch.setattr(yeto.rl, "NEBIUS_BAKED_IMAGES", {})
    assert _island_task(args, monkeypatch).resources.image_id == args.rl_image


def test_model_store_uri_validation(capsys):
    fs = "computefilesystem-e00abc"
    assert model_store_filesystem(None, "nebius", "eu-north1") is None
    assert model_store_filesystem(f"nebius-fs://{fs}", "nebius", "eu-north1") == fs
    assert model_store_filesystem(f"nebius-fs://{fs}", "aws", "us-east-1") is None
    assert "WARNING" in capsys.readouterr().out
    for bad in ("s3://bucket", "nebius-fs://disk-1", fs):
        with pytest.raises(ValueError):
            model_store_filesystem(bad, "nebius", "eu-north1")


def _run(snippet):
    return subprocess.run(
        ["bash", "-c", snippet + '\necho "HUB=${HF_HUB_CACHE:-} HIT=${YETO_MODEL_STORE_HIT:-}"'],
        capture_output=True, text=True, check=True,
    )


def test_model_store_env_hit_and_fallback(tmp_path):
    rev = "c" * 40
    snap = tmp_path / "hub/models--Qwen--Qwen3-0.6B/snapshots" / rev
    snap.mkdir(parents=True)
    env = model_store_env("Qwen/Qwen3-0.6B", rev, str(tmp_path))
    miss = _run(env)
    assert "HUB= HIT=" in miss.stdout and "WARNING" in miss.stderr  # no marker yet
    (tmp_path / "yeto-complete").mkdir()
    (tmp_path / "yeto-complete" / f"Qwen--Qwen3-0.6B@{rev}.json").write_text("{}")
    hit = _run(env)
    assert f"HUB={tmp_path}/hub HIT=1" in hit.stdout and hit.stderr == ""
    other = _run(model_store_env("Qwen/Qwen3-0.6B", "d" * 40, str(tmp_path)))
    assert "HIT=\n" in other.stdout and "WARNING" in other.stderr  # revision mismatch
    norev = _run(model_store_env("Qwen/Qwen3-0.6B", None, str(tmp_path)))
    assert "HIT=\n" in norev.stdout and "WARNING" in norev.stderr


def test_model_store_sky_config_keeps_region_settings(monkeypatch):
    cfg = {"nebius": {"region_configs": {"eu-north1": {"project_id": "project-p"}}}}

    def get_nested(keys, default):
        node = cfg
        for k in keys:
            node = node.get(k, {})
        return node or default

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        skypilot_config=types.SimpleNamespace(get_nested=get_nested)))
    out = model_store_sky_config("computefilesystem-x", "eu-north1")
    region = out["nebius"]["region_configs"]["eu-north1"]
    assert region["project_id"] == "project-p"
    assert region["filesystems"] == [{"filesystem_id": "computefilesystem-x",
                                      "attach_mode": "READ_WRITE",
                                      "mount_path": launcher.MODEL_STORE_MOUNT}]


def test_island_task_model_store(monkeypatch):
    args = _cli(["--gpu", "nebius:2x1xl40s@eu-north1",
                 "--model-store", "nebius-fs://computefilesystem-x"])
    _prepare_rl_args(args)
    task = _island_task(args, monkeypatch)
    env = model_store_env("org/model", "a" * 40)
    assert env in task.setup
    assert '[ -n "$YETO_MODEL_STORE_HIT" ] || (nohup huggingface-cli download' in task.setup
    # both ranks export it before the learner / worker snapshot fetch
    assert task.run.index(env) < task.run.index("set -e") < task.run.index("snapshot_download")
    plain = _cli(["--gpu", "nebius:2x1xl40s@eu-north1"])
    _prepare_rl_args(plain)
    assert "yeto-model-store" not in _island_task(plain, monkeypatch).run
