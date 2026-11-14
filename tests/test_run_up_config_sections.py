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


def test_a_retired_budget_key_still_loads(tmp_path):
    """``budget.silence_grace_min`` (read by an earlier run up, still in existing run configs)
    is ignored, not refused: the run's next ``run up`` must load its config."""
    cfg = up.load_config(_write(tmp_path, budget={"silence_min": 30, "silence_grace_min": 50}))
    assert cfg["budget"]["silence_min"] == 30
    with pytest.raises(ValueError, match="silence_grace_mni"):
        up.load_config(_write(tmp_path, budget={"silence_grace_mni": 50}))


def test_the_starvation_section_loads_and_is_checked(tmp_path):
    """``starvation`` (read by the monitor and the preflight) is a known run config key:
    the guide's section loads, a bad value is refused at load like the other sections."""
    cfg = up.load_config(_write(tmp_path, starvation={"alert_min": 30, "stop_min": 120, "unhealthy_min": 10,
                                                      "check_s": 300, "stop_grace_min": 20,
                                                      "trainer_wait_max_min": 120}))
    assert cfg["starvation"]["stop_min"] == 120
    for bad, field in (({"stop_min": -1}, "stop_min"), ({"alert_mins": 30}, "alert_mins"),
                       ({"alert_min": 60, "stop_min": 30}, "stop_min"), ([30], "starvation")):
        with pytest.raises(ValueError, match=field):
            up.load_config(_write(tmp_path, starvation=bad))


def test_the_controller_value_is_checked_at_load(tmp_path):
    """``controller`` is refused at load when misspelled, not only once ``run up`` reaches the
    pool step (``run status`` / ``scale`` / ``down`` never got there)."""
    for mode in ("subprocess", "in_process"):
        assert up.load_config(_write(tmp_path, controller=mode))["controller"] == mode
    with pytest.raises(ValueError, match="in-process"):
        up.load_config(_write(tmp_path, controller="in-process"))


@pytest.mark.parametrize("extra, field", [
    # a misspelled price: the spend would count the trainer at $0/h and budget.usd never trip
    ({"trainer": {"argv": [], "usd_per_hour": 136}}, "trainer"),
    ({"trainer": {"argv": [], "usd_per_h": "136"}}, r"trainer\.usd_per_h"),
    ({"trainer": {"argv": "--training-mode rl"}}, r"trainer\.argv"),
    ({"trainer": {"args": []}}, "trainer"),
    ({"trainer": {}}, r"trainer\.argv"),
    ({"trainer": {"argv": [], "progress_paths": "tape.jsonl"}}, r"trainer\.progress_paths"),
    # a misspelled container cap: the sandbox container alert would be off
    ({"sandbox": {"max_run_container": 64}}, "sandbox"),
    ({"sandbox": {"max_run_containers": "64"}}, r"sandbox\.max_run_containers"),
    ({"pool": {"config": "p.json", "interval": 30}}, "pool"),
    ({"pool": {"config": "p.json", "interval_s": 0}}, r"pool\.interval_s"),
    ({"preflight": {"command": [["true"]]}}, "preflight"),
])
def test_a_bad_trainer_sandbox_pool_preflight_key_is_refused_at_load(tmp_path, extra, field):
    with pytest.raises(ValueError, match=field):
        up.load_config(_write(tmp_path, **extra))


def test_the_documented_trainer_sandbox_pool_preflight_keys_load(tmp_path):
    up.load_config(_write(tmp_path, trainer={"argv": ["--x"], "usd_per_h": 136, "progress_paths": ["t"]},
                          pool={"config": "p.json", "interval_s": 30},
                          sandbox={"envs": "e.json", "docker_host": ["ssh://h"], "max_run_containers": 64,
                                   "max_containers_per_sampler": 20},
                          preflight={"commands": [["true"]]}))
