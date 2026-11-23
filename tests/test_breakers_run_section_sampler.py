"""A run's ``breakers.sampler`` / ``breakers.harness`` settings reach the sampler.

``yeto run up`` writes them into the pool's ``sampler.breakers`` knob (a copy of the pool
config in the run's state directory); every rendered sampler spec carries them as its
``breakers`` key, the sampler resolves them from there and exports the harness ones to the
harness's env variables. The spec file itself is never copied or changed (its relative
paths keep resolving against its own directory). Without a section nothing changes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from yeto import breakers
from yeto.breakers.registry import default_registry
from yeto.launch.nc import cli as nc_cli
from yeto.launch.nc import tasks
from yeto.launch.rollout import pools as rollout_pools
from yeto.rl.engine.sampler_breakers import apply_harness_env, spec_breakers

ROOT = Path(__file__).resolve().parents[1]
K31 = ROOT / "examples" / "k31"


def _kit(tmp_path: Path, **pool_edits) -> tuple[dict, Path, Path]:
    """The K3.1 kit (run config, pool, agentic sampler spec) in ``tmp_path``, placeholders filled."""
    spec = tmp_path / "sampler-spec.json"
    shutil.copy(K31 / "sampler-spec.nc-mi355x-vllm-agentic-262k.json", spec)
    pool = json.loads((K31 / "pool.nc-mi355x-agentic-262k.json").read_text())
    pool.update(spec=str(spec), run_prefix="k31/run1", agentic_env_file=None, **pool_edits)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(pool, indent=1))
    run = json.loads((K31 / "run.agentic-262k.json").read_text())
    run["pool"] = {"config": str(pool_path), "interval_s": 30}
    run.pop("env", None)
    return run, pool_path, spec


def _rendered(ref) -> dict:
    """The spec one member of the pool is started with (the pool commands' render)."""
    cfg = ref.load()
    spec = json.loads(Path(cfg["spec"]).read_text().replace("__RUN_PREFIX__", cfg.get("run_prefix", "")))
    return tasks.apply_sampler_overrides(spec, **nc_cli._overrides(cfg.get("sampler") or {}))


def test_k31_kit_without_a_section_keeps_its_pool_and_spec_byte_identical(tmp_path):
    run, pool_path, spec = _kit(tmp_path)
    assert "breakers" not in run
    pool_bytes, spec_bytes = pool_path.read_bytes(), spec.read_bytes()
    (ref,) = rollout_pools.resolve(run, tmp_path / "state")
    assert ref.config == str(pool_path) and ref.spec_carried == ()
    assert not (tmp_path / "state").exists()
    assert pool_path.read_bytes() == pool_bytes and spec.read_bytes() == spec_bytes
    assert ref.load()["spec"] == str(spec)
    rendered = _rendered(ref)
    assert "breakers" not in rendered
    assert rendered == tasks.apply_sampler_overrides(json.loads(spec.read_text().replace(
        "__RUN_PREFIX__", "k31/run1")), **nc_cli._overrides(json.loads(pool_bytes)["sampler"]))


def test_a_sampler_setting_reaches_the_sampler_spec(tmp_path):
    run, pool_path, spec = _kit(tmp_path)
    spec_bytes = spec.read_bytes()
    run["breakers"] = {"sampler": {"infra_streak": {"action": "pause", "max_n": 5}}}
    (ref,) = rollout_pools.resolve(run, tmp_path / "state")
    assert ref.config == str(tmp_path / "state" / "pools" / "pool.json")
    assert set(ref.spec_carried) == {"sampler.infra_streak.action", "sampler.infra_streak.max_n"}
    doc = ref.load()
    assert doc["spec"] == str(spec) and spec.read_bytes() == spec_bytes  # the spec file is never copied
    assert doc["sampler"]["breakers"] == {"sampler": {"infra_streak": {"action": "pause", "max_n": 5}}}
    assert doc["sampler"]["dispense"] == json.loads(pool_path.read_text())["sampler"]["dispense"]
    seen = spec_breakers(_rendered(ref))
    assert seen.entry("sampler.infra_streak").action == "pause"
    assert seen.get("sampler.infra_streak.max_n") == 5


def test_a_harness_setting_reaches_the_harness_env(tmp_path):
    run, _pool, _spec = _kit(tmp_path)
    run["breakers"] = {"harness": {"setup_retries": {"max_n": 6}}}
    (ref,) = rollout_pools.resolve(run, tmp_path / "state")
    assert ref.spec_carried == ("harness.setup_retries.max_n",)
    rendered = _rendered(ref)
    environ: dict[str, str] = {}
    assert apply_harness_env(rendered, environ) == ["env HARBOR_SETUP_RETRIES 6"]
    assert environ == {"HARBOR_SETUP_RETRIES": "6"}
    from yeto.rl.harness.harbor import breaker_settings

    assert breaker_settings.setting("harness.setup_retries.max_n", environ) == 6


def test_inherited_and_switched_values_are_carried_resolved():
    resolved = breakers.resolve(breakers.Sources(section={"harness": {"reach_check": {"enabled": False}}}),
                                layers=("sampler", "harness"))
    assert rollout_pools.spec_carried(resolved) == ("harness.reach_check.enabled",)


def test_without_a_section_the_harness_env_is_left_alone():
    environ = {"HARBOR_SETUP_RETRIES": "3"}
    assert apply_harness_env({"agentic": {}}, environ) == [] and environ == {"HARBOR_SETUP_RETRIES": "3"}


def test_a_harness_value_against_the_sampler_env_is_refused():
    with pytest.raises(ValueError, match="HARBOR_SETUP_RETRIES|setup_retries"):
        apply_harness_env({"breakers": {"harness": {"setup_retries": {"max_n": 6}}}},
                          {"HARBOR_SETUP_RETRIES": "3"})


def test_a_sampler_setting_against_the_spec_native_key_is_refused(tmp_path):
    run, _pool, spec = _kit(tmp_path)
    doc = json.loads(spec.read_text())
    doc["agentic"]["infra_streak"] = {"action": "warn"}
    spec.write_text(json.dumps(doc))
    run["breakers"] = {"sampler": {"infra_streak": {"action": "pause"}}}
    with pytest.raises(ValueError, match="pool 'pool'.*infra_streak"):
        rollout_pools.resolve(run, tmp_path / "state")
    assert not (tmp_path / "state" / "pools").exists()


def test_a_sampler_setting_against_the_spec_section_is_refused(tmp_path):
    run, _pool, spec = _kit(tmp_path)
    doc = json.loads(spec.read_text())
    doc["breakers"] = {"sampler": {"infra_streak": {"max_n": 3}}}
    spec.write_text(json.dumps(doc))
    run["breakers"] = {"sampler": {"infra_streak": {"max_n": 5}}}
    with pytest.raises(ValueError, match=r"breakers\.sampler\.infra_streak\.max_n: 3 in the spec, 5"):
        rollout_pools.resolve(run, tmp_path / "state")
    run["breakers"] = {"sampler": {"infra_streak": {"max_n": 3}}}  # repeating the spec's value is fine
    (ref,) = rollout_pools.resolve(run, tmp_path / "state")
    assert spec_breakers(_rendered(ref)).get("sampler.infra_streak.max_n") == 3


def test_a_harness_setting_against_the_agentic_env_file_is_refused(tmp_path):
    env_file = tmp_path / "agentic.env"
    env_file.write_text("HARBOR_SETUP_RETRIES=3\n")
    run, _pool, _spec = _kit(tmp_path)
    pool = json.loads(Path(run["pool"]["config"]).read_text())
    pool["agentic_env_file"] = str(env_file)
    Path(run["pool"]["config"]).write_text(json.dumps(pool))
    run["breakers"] = {"harness": {"setup_retries": {"max_n": 6}}}
    with pytest.raises(ValueError, match="pool 'pool'.*setup_retries"):
        rollout_pools.resolve(run, tmp_path / "state")


def test_a_pool_without_a_spec_cannot_carry_sampler_settings(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"target": 1, "shard_base": "k31"}))
    cfg = {"pool": {"config": str(path)}, "breakers": {"sampler": {"infra_streak": {"max_n": 5}}}}
    with pytest.raises(ValueError, match="pool 'pool'.*no spec"):
        rollout_pools.resolve(cfg, tmp_path / "state")


def test_an_unwritable_state_dir_fails_loudly(tmp_path):
    run, _pool, _spec = _kit(tmp_path)
    run["breakers"] = {"sampler": {"infra_streak": {"max_n": 5}}}
    state = tmp_path / "state"
    state.write_text("a file, not a directory")
    with pytest.raises(ValueError, match="pool 'pool'.*cannot write"):
        rollout_pools.resolve(run, state)


def test_every_sampler_and_harness_setting_is_carried(tmp_path):
    """No sampler / harness setting of the section is dropped: each one is either carried in the
    sampler spec or refused (the silent drop this guards against)."""
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"target": 1, "shard_base": "k31", "spec": str(spec)}))
    checked = 0
    for b in default_registry():
        if b.layer not in rollout_pools.SPEC_LAYERS or not b.configurable:
            continue
        for p in b.params:
            if p.default is None:
                continue
            cfg = {"pool": {"config": str(path)}, "breakers": {b.layer: {b.entry: {p.name: p.default}}}}
            (ref,) = rollout_pools.resolve(cfg, tmp_path / f"state-{b.entry}-{p.name}")
            assert f"{b.name}.{p.name}" in ref.spec_carried, (b.name, p.name)
            assert spec_breakers(_rendered(ref)).get(f"{b.name}.{p.name}") == p.default
            checked += 1
    assert checked > 10


def test_the_pool_knob_is_validated():
    assert nc_cli._overrides({"breakers": {"sampler": {"infra_streak": {"max_n": 5}}}})["breakers"]
    with pytest.raises(ValueError, match="pool sampler config: breakers"):
        nc_cli._overrides({"breakers": {"sampler": {"nope": {"max_n": 5}}}})


def test_merge_breakers_repeats_but_never_contradicts():
    base = {"sampler": {"infra_streak": {"max_n": 5}}}
    assert tasks.merge_breakers(base, {"sampler": {"infra_streak": {"max_n": 5, "action": "pause"}}}) == {
        "sampler": {"infra_streak": {"max_n": 5, "action": "pause"}}}
    assert base == {"sampler": {"infra_streak": {"max_n": 5}}}
    with pytest.raises(ValueError, match="max_n"):
        tasks.merge_breakers(base, {"sampler": {"infra_streak": {"max_n": 6}}})
    with pytest.raises(ValueError, match="enabled"):
        tasks.merge_breakers({"harness": {"reach_check": {"enabled": True}}},
                             {"harness": {"reach_check": {"enabled": 1}}})
