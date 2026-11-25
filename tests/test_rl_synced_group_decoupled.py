"""A synced group (``--rl-sample-role trainer --rl-island-sample-role M:sampler``)
with the decoupled preset (dry run; no cloud, no Ray).

Its trainers have one version per fragment, so the group needs
``--rl-sampler-follow trainer``: each trainer and its own sampler share the
store sub-directory ``trainer<k>``, the trainer releases its local versions, and
the sampler is a LoRA no-sync sampler (strict-avg, one fragment) that generates
until its trainer ended. Without the rewrite the sampler kept the decoupled
preset next to ``--rl-single-island-no-sync``, which the island entry refuses:
the sampler island died at start-up while the trainers kept running (seen on a
two-trainer Modal run). The refusal without ``--rl-sampler-follow trainer`` is
checked at launch, before any GPU starts."""

from __future__ import annotations

import shlex
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles
from yeto.rl.engine.merged_release import OWN_LANE

GPU = "modal:1xh100,modal:1xh100,modal:1xl40s,modal:1xl40s"
STORE = "modal-volume://yeto-dl/run"
MOUNT = "/root/yeto-checkpoint-store/run"
GROUP = ("--tuning", "lora", "--lora-r", "16", "--lora-targets", "attention",
         "--rl-max-policy-age", "2",
         "--rl-algorithm-spec", "examples/k31/algorithm-spec.async-age2-tis.json",
         "--rl-sample-role", "trainer", "--rl-island-sample-role", "2:sampler",
         "--rl-island-sample-role", "3:sampler",
         "--rl-checkpoint-store", STORE, "--cluster-prefix", "dl-group")
DECOUPLED = ("--rl-sync-preset", "decoupled", "--fragments", "4", "--pipeline", "2",
             "--local-rl-rounds-per-sync", "4")
FOLLOW = ("--rl-sampler-follow", "trainer")


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "dl-test-key")


def _prepared(*flags, gpu=GPU):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (*flags, *GROUP), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _islands(args):
    """[(island request, parsed island args)] of the dry-run plan."""
    from yeto.launcher import dry_run_plan

    plan = dry_run_plan(args)
    trainers = [i["learner_id"] for i in plan["island_requests"] if i["sample_role"] == "trainer"]
    out = []
    for island in plan["island_requests"]:
        learner = trainers.index(island["learner_id"]) if island["learner_id"] in trainers else 0
        tail = island["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
        tokens = [{"$LEARNER_ID": str(learner), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t)
                  for t in shlex.split(tail)]
        out.append((island, island_entry.parse_args(tokens)))
    return plan, out


def test_decoupled_synced_group_needs_the_samplers_to_follow_their_trainer():
    with pytest.raises(ValueError, match="needs --rl-sampler-follow trainer"):
        _prepared(*DECOUPLED)


def test_decoupled_synced_group_refuses_an_initial_adapter_and_other_backends(monkeypatch):
    base = dict(rl_sampler_follow="trainer", rl_backend="miles", rl_initial_adapter=None)
    sample_roles.check_decoupled_group(SimpleNamespace(**base))
    with pytest.raises(ValueError, match="does not take --rl-initial-adapter"):
        sample_roles.check_decoupled_group(SimpleNamespace(**{**base, "rl_initial_adapter": "/a"}))
    # a backend whose trainers do not release their own versions (per_learner_release)
    from yeto.rl.engine import backends

    real = backends.capabilities
    monkeypatch.setattr(backends, "capabilities", lambda name=None: frozenset() if name == "verl" else real(name))
    with pytest.raises(ValueError, match=r"per_learner_release: --rl-backend miles only\); --rl-backend verl does not"):
        sample_roles.check_decoupled_group(SimpleNamespace(**{**base, "rl_backend": "verl"}))


def test_decoupled_follow_group_islands_parse_and_pair_by_sub_store():
    args = _prepared(*DECOUPLED, *FOLLOW)
    assert sample_roles.sampler_bindings(args, 4) == {2: 0, 3: 1}
    plan, islands = _islands(args)
    assert plan["outer_sync"]
    assert [i["sample_role"] for i, _ in islands] == ["trainer", "trainer", "sampler", "sampler"]
    syncer = plan["role_syncers"][0]["command"]
    assert "--learners 2 " in syncer  # the trainers only
    assert f"--total-steps {4 * int(args.total_steps)} " in syncer
    (_, t0), (_, t1), (_, s0), (_, s1) = islands
    for k, (trainer, sampler) in enumerate(((t0, s0), (t1, s1))):
        substore = f"{MOUNT}/trainer{k}"
        # the trainer: a decoupled island of the syncer reading its own sub-store
        assert trainer.sync_preset == "decoupled" and trainer.fragments == 4
        assert trainer.pipeline == 2 and trainer.local_horizon == 4
        assert trainer.rl_sample_store == substore and trainer.rl_sample_lanes == 1
        assert sample_roles.buffer_lane(trainer) == OWN_LANE
        assert trainer.syncer == "10.0.0.1:29400"
        # its sampler: a LoRA no-sync sampler over the same sub-store, same adapter shape
        assert sampler.rl_single_island_no_sync and sampler.sync_preset == "strict-avg"
        assert sampler.fragments == 1 and sampler.pipeline == 1 and sampler.local_horizon == 1
        assert sampler.parameter_mode == "lora"
        assert (sampler.lora_r, sampler.lora_targets) == (trainer.lora_r, trainer.lora_targets) == (16, "attention")
        assert sampler.rl_resume_store == substore and sampler.learner_id == 0
        assert sampler.initial_adapter is None
        # the island entry accepts it (it refused the decoupled preset on a no-sync island)
        island_entry._check_single_island_no_sync(sampler)
        # it runs until its own trainer ended (the trainers run until the syncer stops them)
        assert sampler.rl_sampler_until_trainers_end == 1
        assert sample_roles.resolve(sampler, groups_per_round=4).until_trainers_end == 1
        assert sample_roles.resolve(trainer, groups_per_round=4).store == substore


def test_decoupled_follow_samplers_stop_on_their_trainer(capsys):
    args = _prepared(*DECOUPLED, *FOLLOW)
    assert "generate until its own trainer ended" in capsys.readouterr().err
    assert sample_roles.sampler_stop_trainers(args, 4) == {2: 1, 3: 1}


def test_strict_avg_synced_group_is_unchanged():
    args = _prepared("--rl-sync-preset", "strict-avg")
    sampler = sample_roles.island_args(args, 2, 4)
    assert sampler.rl_sample_role == "sampler" and sampler.rl_sync_preset == "strict-avg"
    assert sample_roles.sampler_stop_trainers(args, 4) == {}


# ------------------------------------------------ DecoupledSync on a synced trainer
def _sync(lane):
    from yeto.rl.engine.bridges import DecoupledSync

    sync = DecoupledSync(SimpleNamespace(yeto_rl_sample_lane=lane))
    sync._driver = SimpleNamespace(outer_version=None)
    return sync


def test_a_decoupled_buffer_trainer_states_the_merged_version_of_its_cut():
    from yeto.rl.engine.full_parameter_sync import merged_version

    sync = _sync(OWN_LANE)
    # P = 4: fragment f is at global steps f+1, f+1+4, ...; (5, 2, 7, 4) has every
    # fragment merged once, fragment 0 and 2 twice
    sync._publish_outer_version((5, 2, 7, 4))
    assert sync._driver.outer_version == 1 == merged_version((5, 2, 7, 4))
    assert sync._driver.fragment_versions == (5, 2, 7, 4)
    sync._publish_outer_version((0, 0, 0, 0))
    assert sync._driver.outer_version == 0


def test_a_colocated_decoupled_island_keeps_no_outer_version():
    sync = _sync(None)
    sync._publish_outer_version((5, 2, 7, 4))
    assert sync._driver.outer_version is None and not hasattr(sync._driver, "fragment_versions")
