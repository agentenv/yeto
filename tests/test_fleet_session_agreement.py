"""A fleet launched island by island: the syncer launch and every island launch (the
run config's islands, an island added at run time, a moved island) compile the same
syncer semantic profile and the same HELLO-bound session flags, so no island is refused
at HELLO for a profile the syncer does not run ("HELLO syncer semantic profile does not
match the running server"). Each launch derives them from its own argv: one --gpu entry
for an island launch, every declared trainer's for the syncer launch."""

from __future__ import annotations

import json
import re

import pytest

from test_fleet_island_launches import _prepared, example_doc
from yeto.launch.rl_island_task import _rl_trainable_island_flags
from yeto.launch.syncer import rl_syncer_profile
from yeto.launch.topology import rl_island_count
from yeto.rl import fleet, fleet_islands, island_launches, up
from yeto.rl.engine.sample_roles import syncer_learner_count

RECORD = {"addr": "10.0.0.9:29400", "syncer_epoch": 0}


@pytest.fixture(autouse=True)
def _island_env(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-t5b-test-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")


def _session(args) -> tuple[dict, str]:
    """The launch's syncer semantic profile (every field) and its HELLO-bound island flags
    (learner id masked: the one field that is the island's own)."""
    learners = syncer_learner_count(args, rl_island_count(args))
    profile = rl_syncer_profile(args, learners)
    flags = _rl_trainable_island_flags(args, 0, learners)
    return dict(profile.__dict__), re.sub(r"--learner-id \d+", "", flags)


def _islands(cfg) -> dict[str, list[str]]:
    plan = fleet.plan_of(cfg)
    out = {m["name"]: island_launches.launch_argv(m, syncer=RECORD)
           for m in island_launches.trainer_members(cfg, plan)}
    entry = dict(cfg["islands"][-1]["samplers"][0], name="sglang-north")
    added, _ = fleet_islands.compile_island(cfg, plan, k=2, name="north", gpu="modal:1x2xh100", usd_per_h=8.0,
                                            samplers=[entry])
    out["north (added)"] = island_launches.launch_argv(added, syncer=RECORD)
    moved = island_launches.trainer_members(cfg, plan)[-1]
    out[f"{moved['name']} (moved)"] = island_launches.launch_argv(
        {**moved, "gpu": "aws:2xh100@us-east-2"}, incarnation=1, syncer=RECORD)
    return out


@pytest.mark.parametrize("islands", [2, 1])
def test_every_island_launch_compiles_the_syncer_launch_s_profile_and_session(tmp_path, islands):
    """Two islands: the outer Nesterov (lr 0.7, momentum 0.9) on every launch; one: the
    identity merge (1.0, 0.0) on every launch, an island added later too."""
    doc = example_doc(tmp_path)
    doc["islands"] = doc["islands"][:islands]
    path = tmp_path / "run.json"
    path.write_text(json.dumps(doc))
    cfg = up.load_config(str(path))
    syncer_args = _prepared([*cfg["trainer"]["argv"], "--controller", "local"])
    syncer_profile, syncer_flags = _session(syncer_args)
    assert syncer_profile["learners"] == islands
    assert syncer_profile["outer_momentum"] == (pytest.approx(0.9) if islands == 2 else 0.0)
    for name, argv in _islands(cfg).items():
        island_profile, island_flags = _session(_prepared(argv))
        diff = {k: (v, island_profile.get(k)) for k, v in syncer_profile.items() if island_profile.get(k) != v}
        assert not diff, f"island {name}: (syncer, island) {diff}"
        assert island_flags == syncer_flags, name
