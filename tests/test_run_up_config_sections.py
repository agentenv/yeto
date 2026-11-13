"""``yeto run up`` rejects a malformed first_group / budget / dashboard / rollout section
when it loads the run config, before anything is ordered (not mid-run, not silently)."""

from __future__ import annotations

import json

import pytest

from yeto.rl import up


def _write(tmp_path, **extra):
    cfg = {"run": "r", "store": "s3://b/p", "trainer": {"argv": []}, "pool": {"config": "p.json"}}
    cfg.update(extra)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(cfg))
    return str(path)


def test_the_guide_template_sections_load(tmp_path):
    up.load_config(_write(tmp_path, first_group={"timeout_s": 5400, "poll_s": 60}, dashboard={"port": 8795},
                          budget={"usd": 0, "wall_h": 3.0, "nc_usd_per_h": 0, "silence_min": 30,
                                  "release_nc_on_breach": False, "release_nc_on_finish": True},
                          rollout={"stall_alert_s": 0, "drain_on_exit": True}))
    up.load_config(_write(tmp_path, budget={"usd": None, "wall_h": None}, dashboard={"prices": "x.json"}))


@pytest.mark.parametrize("extra, field", [
    ({"first_group": {"timeout_s": None}}, "first_group.timeout_s"),
    ({"first_group": {"timeout_s": "90 min"}}, "first_group.timeout_s"),
    ({"first_group": {"timeout_s": -1}}, "first_group.timeout_s"),
    ({"first_group": {"poll_s": 0}}, "first_group.poll_s"),
    ({"first_group": {"timeout_min": 90}}, "first_group"),
    ({"first_group": [5400]}, "first_group"),
    ({"budget": {"usd_cap": 450}}, "budget"),
    ({"budget": {"silence_min": "30"}}, "budget.silence_min"),
    ({"budget": {"silence_min": -5}}, "budget.silence_min"),
    ({"budget": {"nc_usd_per_h": "24"}}, "budget.nc_usd_per_h"),
    ({"budget": {"nc_usd_per_h": -1}}, "budget.nc_usd_per_h"),
    ({"budget": {"release_nc_on_breach": "false"}}, "budget.release_nc_on_breach"),
    ({"budget": {"release_nc_on_finish": 1}}, "budget.release_nc_on_finish"),
    ({"dashboard": {"port": None}}, "dashboard.port"),
    ({"dashboard": {"port": 0}}, "dashboard.port"),
    ({"dashboard": {"prot": 8795}}, "dashboard"),
    ({"rollout": {"stall_alert_s": None}}, "rollout.stall_alert_s"),
    ({"rollout": {"stall_alert_s": "30m"}}, "rollout.stall_alert_s"),
    ({"rollout": {"stall_alert_s": -1}}, "rollout.stall_alert_s"),
])
def test_a_bad_section_value_is_refused_at_load(tmp_path, extra, field):
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        up.load_config(_write(tmp_path, **extra))
