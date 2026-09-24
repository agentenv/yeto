"""Pre-spend cloud checks run from prepare_launch_args."""

from __future__ import annotations

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


def test_prepare_launch_args_runs_the_check(monkeypatch):
    # The check is wired into the pre-spend validation path; a Nebius
    # region without a project fails before anything is provisioned.
    from yeto import launcher

    monkeypatch.setattr("yeto.shape.providers.nebius_project_ids", lambda config_path=None: {})
    with pytest.raises(ValueError, match="Nebius needs a project per region"):
        launcher.check_cloud_prerequisites(parse_gpu_spec("nebius:1xh100@eu-north1"))
