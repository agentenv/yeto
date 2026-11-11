"""A synced group (``--rl-sample-role trainer --rl-island-sample-role M:sampler``)
with the decoupled preset is refused at launch, before any GPU starts.

Its sampler islands run as no-sync islands, and the Miles island entry refuses
``--rl-single-island-no-sync`` together with ``--sync-preset decoupled``: without
the launch check the sampler island died at start-up while the trainers kept
running (seen on a two-trainer Modal run)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.adapters.miles import island_entry
from yeto.rl.engine import sample_roles

GPU = "modal:1xh100,modal:1xh100,modal:1xh100"
GROUP = ("--tuning", "lora", "--rl-max-policy-age", "2",
         "--rl-algorithm-spec", "examples/k31/algorithm-spec.async-age2-tis.json", "--rl-sample-role", "trainer", "--rl-island-sample-role", "2:sampler",
         "--rl-checkpoint-store", "modal-volume://yeto-dl/run", "--cluster-prefix", "dl-group")


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "dl-test-key")


DECOUPLED = ("--rl-sync-preset", "decoupled", "--fragments", "2", "--local-rl-rounds-per-sync", "4")


def _prepared(preset_flags):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (*preset_flags, *GROUP), gpu=GPU)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def test_decoupled_synced_group_is_refused_at_launch():
    with pytest.raises(ValueError, match="decoupled does not combine with a synced group"):
        _prepared(DECOUPLED)


def test_the_sampler_island_command_is_what_the_island_entry_refuses():
    """Why the launch check exists: the sampler's island args keep the decoupled
    preset next to --rl-single-island-no-sync, a combination the entry refuses."""
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-sync-preset", "strict-avg", *GROUP), gpu=GPU)
    args.controller = "local"
    _prepare_rl_args(args)
    args.rl_sync_preset = args.sync_preset = "decoupled"
    sampler = sample_roles.island_args(args, 2, 3)
    assert sampler.rl_single_island_no_sync and sampler.rl_sync_preset == "decoupled"
    with pytest.raises(ValueError, match="cannot be combined with: --sync-preset decoupled"):
        island_entry._check_single_island_no_sync(SimpleNamespace(
            rl_single_island_no_sync=True, syncer=None, num_learners=1, learner_id=0,
            sync_preset=sampler.rl_sync_preset, initial_adapter=None))


def test_strict_avg_synced_group_still_launches():
    args = _prepared(("--rl-sync-preset", "strict-avg"))
    assert sample_roles.island_args(args, 2, 3).rl_sample_role == "sampler"
