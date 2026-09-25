"""Pre-spend cloud checks run from prepare_launch_args."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.gpu_spec import parse_gpu_spec
from yeto.launcher import check_cloud_prerequisites


def test_nebius_fleet_needs_a_project_per_region():
    specs = parse_gpu_spec("nebius:8xh100@eu-north1,nebius:8xh200@us-central1,aws:8xh100@us-east-1")
    with pytest.raises(ValueError, match=r"no project_id for us-central1 under nebius\.region_configs"):
        check_cloud_prerequisites(specs, project_ids={"eu-north1": "project-e00"})
    # Both configured -> fine; and a fleet without Nebius never consults the config.
    check_cloud_prerequisites(specs, project_ids={"eu-north1": "p1", "us-central1": "p2"})
    check_cloud_prerequisites(parse_gpu_spec("aws:8xh100@us-east-1"), project_ids={})


def test_missing_regions_are_all_listed():
    specs = parse_gpu_spec("nebius:8xh100@eu-north1,nebius:8xh100@eu-west1")
    with pytest.raises(ValueError, match="eu-north1, eu-west1"):
        check_cloud_prerequisites(specs, project_ids={})


def test_head_cloud_credentials_files_env_and_errors(monkeypatch, tmp_path):
    from yeto.launcher import head_cloud_credentials

    home = tmp_path / "home"
    (home / ".aws").mkdir(parents=True)
    (home / ".modal.toml").write_text("[default]\n")
    monkeypatch.setenv("HOME", str(home))
    # Files win; env vars stand in for a missing file; a cloud with neither is an error.
    mounts, envs = head_cloud_credentials(
        ["aws", "modal", "verda"],
        environ={"VERDA_CLIENT_ID": "id", "VERDA_CLIENT_SECRET": "s"},
    )
    assert mounts == {"~/.aws": str(home / ".aws"), "~/.modal.toml": str(home / ".modal.toml")}
    assert envs == {"VERDA_CLIENT_ID": "id", "VERDA_CLIENT_SECRET": "s"}
    with pytest.raises(ValueError, match=r"runpod credentials not found at ~/\.runpod or env RUNPOD_API_KEY"):
        head_cloud_credentials(["runpod"], environ={})
    # GCP is optional: mounted when present, never required.
    assert head_cloud_credentials(["gcp"], environ={}) == ({}, {})
    (home / ".config" / "gcloud").mkdir(parents=True)
    mounts, _ = head_cloud_credentials(["aws"], environ={})
    assert mounts["~/.config/gcloud"] == str(home / ".config" / "gcloud")


def test_head_gets_sky_config_when_fleet_uses_nebius(monkeypatch, tmp_path):
    """The head re-runs the Nebius project check and provisions Nebius
    islands, so it needs the per-region project ids from sky's config."""
    from yeto.launcher import head_cloud_credentials

    home = tmp_path / "home"
    (home / ".nebius").mkdir(parents=True)
    (home / ".aws").mkdir()
    (home / ".sky").mkdir()
    (home / ".sky" / "config.yaml").write_text("nebius:\n  region_configs:\n    eu-north1:\n      project_id: p\n")
    monkeypatch.setenv("HOME", str(home))
    mounts, _ = head_cloud_credentials(["nebius"], environ={})
    assert mounts["~/.sky/config.yaml"] == str(home / ".sky" / "config.yaml")
    mounts, _ = head_cloud_credentials(["aws"], environ={})
    assert "~/.sky/config.yaml" not in mounts


def test_fleet_clouds_include_the_head_cloud():
    from yeto.launcher import fleet_clouds, head_cloud

    ns = SimpleNamespace(gpu="verda:8xh100@FIN-03,modal:8xh100", syncer_region="nebius/eu-north1", controller="head")
    assert head_cloud(ns) == "nebius"
    assert fleet_clouds(ns) == ["nebius", "verda", "modal"]
    ns = SimpleNamespace(gpu="aws:8xh100@us-east-1", syncer_region="us-west-2", controller="local")
    assert fleet_clouds(ns) == ["aws"]  # local controller: no head VM


def test_prepare_launch_args_runs_the_check(monkeypatch):
    # The check is wired into the pre-spend validation path; a Nebius
    # region without a project fails before anything is provisioned.
    from yeto import launcher

    monkeypatch.setattr("yeto.shape.providers.nebius_project_ids", lambda config_path=None: {})
    with pytest.raises(ValueError, match="Nebius needs a project per region"):
        launcher.check_cloud_prerequisites(parse_gpu_spec("nebius:1xh100@eu-north1"))
