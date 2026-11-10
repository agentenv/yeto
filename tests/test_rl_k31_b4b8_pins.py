"""Miles k31/b4 (elastic) and k31/b8 (loss variants / critic): pins, features, probes
and the API each pin selects.

The k31 lineage is linear (F1 < B2 < B3B7 < B4 < B8); the B4 verbs are taken only
on a pin carrying them and the ports-v1 pin keeps the fork-era ones. The default
GRPO argv (test_decoupling_golden) does not move.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import config as mc
from yeto.rl.adapters.miles import entry, pins
from yeto.rl.adapters.miles import runtime_manifest as rm

from test_rl_miles_adapter_config import make_config

F1 = "4a839d8a663eead89ac4815b03304098321157f4"
B2 = "f9454517e81f6b41087bcc85a605258d80fb6b9c"
B3B7 = "a07553f0a4f361b1618a707b416b690c5666e03d"
B4 = "96437f3732a1232bb8a99c8a6d18f6fd2a37dc24"
B8 = "efc83590dcb8daf9b230c71b45ffb4f1e6b54cb3"
GPUA = "fc15023cb346d5205984c9e68957b673b00e4b9a"  # k31/gpu-a-b8: B300 bring-up fixes on B8
FULL = "63834eaa53b94f449a0e75ed86d6b597528817a9"  # k31/full: GPUA + routed-expert QAT
STUB = "ecb33cbaa2b8eac25de24f443e3861113cbbd2ff"  # k31/full-stub: FULL + stub-optimizer skip
FULL_B = "83d5db6b60677e9c8372ee505bdd6bb5713047e7"
FULL_C = "690ebe4910337cf0a96542e9d057c1baaee5ae16"
FULL_D = "c351a33c643326c478a9edaf9bae6c43299508fb"
FULL_E = "1f29e43043e4c9aace8287a445eca3851e8b52c7"  # k31/integ-trainer: the full-form image's Miles
# branch heads the rebase replaced; nothing may pin them any more
REPLACED = ("a019deded8cbb9bc10ad5ef55ea8ad3088123c61", "640de460", "fa1a4607", "daf21371", "033f97e7")
B4_FEATURES = ("placement_map", "membership_epoch", "router_cordon", "member_publish", "trainer_rebuild")
B8_FEATURES = ("loss_variants", "gae_variants", "critic_family")


def _pin(monkeypatch, commit):
    monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", commit)
    monkeypatch.setattr(mc, "MILES_NEXT_COMMIT", commit)


def test_k31_lineage_is_linear_and_ends_at_full():
    assert (pins.MILES_K31_F1_COMMIT, pins.MILES_K31_B2_COMMIT, pins.MILES_K31_B3B7_COMMIT,
            pins.MILES_K31_B4_COMMIT, pins.MILES_K31_B8_COMMIT) == (F1, B2, B3B7, B4, B8)
    assert pins.MILES_LINEAGES["k31"].commits == (F1, B2, B3B7, B4, B8, GPUA, FULL, STUB, FULL_B, FULL_C, FULL_D, FULL_E)
    assert pins.MILES_NEXT_COMMIT == FULL_D == pins.MILES_NEXT_PINS.commit
    from yeto import rl

    assert rl.MILES_NEXT_COMMIT == FULL_D and rl.MILES_K31_B4_COMMIT == B4 and rl.MILES_K31_B8_COMMIT == B8
    root = Path(pins.__file__).resolve().parents[3]
    for path in (root / "yeto").rglob("*.py"):
        text = path.read_text()
        assert not [old for old in REPLACED if old in text], path


def test_b4_and_b8_features_start_at_their_batch():
    for features, first in ((B4_FEATURES, B4), (B8_FEATURES, B8)):
        for feature in features:
            assert pins.K31_FEATURES[feature] == first
            assert pins.pin_has(feature) and pins.pin_has(feature, B8)
            assert not pins.pin_has(feature, B3B7) and not pins.pin_has(feature, F1)
    assert not pins.pin_has("loss_variants", B4)
    # every earlier k31 feature carries forward to the head
    for feature, first in pins.K31_FEATURES.items():
        assert pins.pin_has(feature, B8), feature
    # the frozen ports-v1 entries keep their values
    assert pins.PORTS_V1_FEATURES["loss_variants"] == "5c1b49ebccbc7508c1d9ef89eacc2db3e448b6ba"
    assert pins.PORTS_V1_FEATURES["elastic_m1_m6"] == "0af62f4d48ed6a5b185c257578d8f7e22312aa87"
    assert not pins.pin_has("member_publish", pins.MILES_PORTS_V1_COMMIT)


def test_loss_variant_and_critic_commit_sets_follow_the_lineages():
    from yeto.rl.algos import critic_fork, loss_variants

    k31_from_b8 = pins.MILES_LINEAGES["k31"].from_commit(B8)
    assert loss_variants.fork_supports_variants(B8) and loss_variants.fork_supports_variants()
    assert loss_variants.fork_supports_variants(pins.MILES_PORTS_V1_COMMIT)  # ports-v1 unchanged
    assert not loss_variants.fork_supports_variants(B4)
    assert k31_from_b8 <= loss_variants.FORK_COMMITS
    assert critic_fork.FORK_COMMITS == k31_from_b8 and critic_fork.CRITIC_FORK_PIN == B8
    assert not critic_fork.fork_carries_critic_family(B4)


B4_PROBES = {
    "fork_m1_placement_map": ("miles.ray.placement_group", "parse_placement_map"),
    "fork_m2_start_stop_cells": ("miles.ray.rollout.inference_controller", "InferenceController.start_cells"),
    "fork_m3_router_cordon": ("miles.router.router", "MilesRouter.cordon_worker"),
    "fork_m4_member_publish": ("miles.ray.rollout.inference_controller",
                               "InferenceController._start_member_update_weights"),
    "fork_m6_rebuild_trainer": ("miles.ray.trainer_rebuild", "rebuild_training_models"),
}


def test_runtime_manifest_probes_the_b4_verbs():
    for name, target in B4_PROBES.items():
        assert rm.INTERFACES[name] == target
    assert rm.CAPABILITY_REQUIREMENTS["Publisher.publish(members)"] == ("fork_m4_member_publish",)
    assert rm.CAPABILITY_REQUIREMENTS["Placement.reconfigure"] == ("fork_m1_placement_map",
                                                                    "fork_m6_rebuild_trainer")


def test_b4_probes_resolve_in_a_b4_checkout():
    pytest.importorskip("ray")
    pytest.importorskip("miles.ray.trainer_rebuild")
    from yeto.rl.engine.runtime_manifest import _probe

    assert [n for n, (m, a) in B4_PROBES.items() if not _probe(m, a)] == []


# -- elastic flags are refused on a k31 pin before B4 -----------------------------


def test_elastic_router_flag_needs_a_pin_with_router_cordon(monkeypatch):
    from yeto.rl.engine.algorithm import AlgorithmSpec

    cfg = dataclasses.replace(make_config(), use_miles_router=True)
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert argv[argv.index("--router") + 1] == "miles"
    _pin(monkeypatch, B3B7)
    with pytest.raises(mc.MilesConfigError, match="k31/b4"):
        mc.translate_run_config(cfg, AlgorithmSpec())
    _pin(monkeypatch, pins.MILES_PORTS_V1_COMMIT)  # the fork's M3
    assert "--router" in list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)


def test_placement_map_needs_a_pin_with_the_map(monkeypatch):
    mc._require_elastic_pin("placement_map", "map")
    _pin(monkeypatch, B3B7)
    with pytest.raises(mc.MilesConfigError, match="placement_map"):
        mc._require_elastic_pin("placement_map", "an explicit bundle map (--yeto-placement-map)")
    _pin(monkeypatch, pins.MILES_PORTS_V1_COMMIT)
    mc._require_elastic_pin("placement_map", "map")


# -- member publish / commit: the k31/b4 verbs -------------------------------------


class _Engine:
    def __init__(self, url):
        self.server_url, self.version = url, None

    async def update_weight_version(self, v, abort_all_requests=False):
        self.version = v

    async def get_weight_version(self):
        return self.version


class _Controller:
    """The k31/b4 InferenceController surface the member publish touches."""

    def __init__(self):
        self.log = []
        self.engines = {c: _Engine(f"http://{c}") for c in ("c0", "c1", "c2")}

    async def wait_cells_tracked(self, cell_ids, *, timeout_seconds):
        self.log.append(("wait_cells_tracked", tuple(cell_ids)))

    async def start_update_weights(self, model_id=None, receiver_pool="rollout", members=None,
                                   expected_epoch=None, admit_cordoned=False):
        cells = list(members)
        self.log.append(("start_update_weights", tuple(cells), expected_epoch))
        return SimpleNamespace(rollout_engines=[self.engines[c] for c in cells],
                               snapshot_cell_id_to_hashes={c: "h" for c in cells})

    async def end_update_weights(self, snapshot_cell_id_to_hashes, failed_cell_ids, **kw):
        self.log.append(("end_update_weights", tuple(failed_cell_ids)))

    async def abort_update_weights(self):
        self.log.append(("abort_update_weights",))

    async def admit_cells(self, cell_ids, *, expected_epoch, expected_weight_version):
        self.log.append(("admit_cells", tuple(cell_ids), expected_epoch, expected_weight_version))

    async def start_commit_weight_version(self, *, weight_version, expected_epoch, model_id=None):
        self.log.append(("start_commit_weight_version", weight_version, expected_epoch))  # ports-v1 only

    async def end_commit_weight_version(self):
        self.log.append(("end_commit_weight_version",))


class _Actor:
    def __init__(self, log):
        self.log = log

    async def run_plugin(self, path, kwargs):
        self.log.append(("run_plugin", path.rsplit(".", 1)[-1], kwargs))
        return [None]


def _publisher(controller, **kw):
    from yeto.rl.adapters.miles.publish import MilesPublisher

    pub = MilesPublisher(args=SimpleNamespace(), actor_model=_Actor(controller.log), rollout_executor="executor",
                         inference_controller=controller, verify_engine_checksums=False, **kw)
    pub.watch_targets = False
    return pub


def _fake_member_publish(log, failed=()):
    class MemberPublishFailedError(RuntimeError):
        def __init__(self, cells):
            self.failed_cell_ids = tuple(cells)
            super().__init__(f"member cells {list(cells)} failed")

    async def update_member_weights(actor, controller, *, members, expected_epoch, admit_cordoned=False):
        log.append(("update_member_weights", type(actor).__name__, tuple(members), expected_epoch, admit_cordoned))
        if failed:
            raise MemberPublishFailedError(failed)
        return 7

    async def commit_weight_version(executor, controller, *, weight_version, expected_epoch):
        log.append(("commit_weight_version", executor, weight_version, expected_epoch))

    return update_member_weights, commit_weight_version


def test_member_publish_uses_update_member_weights_and_commit_weight_version():
    ctl = _Controller()
    update, commit = _fake_member_publish(ctl.log)
    pub = _publisher(ctl, update_member_weights=update, commit_weight_version=commit)
    asyncio.run(pub._publish_members(1, "tok", ["c1", "c2"], 3))
    pub.verify_serving_policy(epoch=3, token_rollout_id=1, state=None)
    assert ctl.log == [
        ("wait_cells_tracked", ("c1", "c2")),
        ("run_plugin", "align_weight_version", {"next_version": 1}),
        ("update_member_weights", "_Actor", ("c1", "c2"), 3, True),
        ("start_update_weights", ("c1", "c2"), 3),
        ("end_update_weights", ()),
        ("admit_cells", ("c1", "c2"), 3, "1"),
        ("commit_weight_version", "executor", 1, 3),
        ("commit_weight_version", "executor", 1, 3),
    ]


def test_failed_members_name_only_the_cells_that_left_service():
    from yeto.rl.adapters.miles.publish import PublicationError

    ctl = _Controller()
    update, commit = _fake_member_publish(ctl.log, failed=("c2",))
    pub = _publisher(ctl, update_member_weights=update, commit_weight_version=commit)
    with pytest.raises(PublicationError) as raised:
        asyncio.run(pub._publish_members(1, "tok", ["c1", "c2"], 3))
    assert raised.value.failed_members == {"engine:c2"} and raised.value.engine_ids == ("engine:c2",)


def test_the_pin_selects_the_member_publish_verbs(monkeypatch):
    from yeto.rl.adapters.miles import publish

    ctl = _Controller()
    update, commit = _fake_member_publish(ctl.log)
    legacy = []

    async def update_weights(*args, **kw):  # ports-v1 placement_group.update_weights(members=)
        legacy.append(kw)

    monkeypatch.setattr(publish, "_default_update_member_weights", lambda: update)
    monkeypatch.setattr(publish, "_default_commit_weight_version", lambda: commit)
    monkeypatch.setattr(publish, "_default_update_weights", lambda: update_weights)
    asyncio.run(_publisher(ctl)._publish_members(1, "tok", ["c2"], 1))  # the k31 head: B4 verbs
    assert ("update_member_weights", "_Actor", ("c2",), 1, True) in ctl.log and legacy == []
    assert ctl.log[-1] == ("commit_weight_version", "executor", 1, 1)
    assert "start_commit_weight_version" not in [e[0] for e in ctl.log]
    _pin(monkeypatch, pins.MILES_PORTS_V1_COMMIT)
    ctl.log.clear()
    asyncio.run(_publisher(ctl)._publish_members(1, "tok", ["c2"], 1))
    assert legacy == [{"members": ["c2"], "expected_epoch": 1, "admit_cordoned": True}]
    assert not [e for e in ctl.log if e[0] in ("update_member_weights", "commit_weight_version")]
    assert ctl.log[-2:] == [("start_commit_weight_version", 1, 1), ("end_commit_weight_version",)]


# -- trainer rebuild: miles.ray.trainer_rebuild on k31/b4+ -----------------------


def test_rebuild_comes_from_the_trainer_rebuild_module_on_b4(monkeypatch):
    from yeto.rl.adapters.miles import trainer_rebuild as tr

    rebuilt, legacy = types.ModuleType("miles.ray.trainer_rebuild"), types.ModuleType("miles.ray.placement_group")
    rebuilt.rebuild_training_models, legacy.rebuild_training_models = object(), object()
    ray = types.ModuleType("miles.ray")
    ray.trainer_rebuild, ray.placement_group = rebuilt, legacy
    for module in (types.ModuleType("miles"), ray, rebuilt, legacy):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delenv("YETO_RL_TEST_INJECT_REBUILD_FAIL", raising=False)
    assert tr.rebuild_module() is rebuilt and tr._default_rebuild() is rebuilt.rebuild_training_models
    _pin(monkeypatch, pins.MILES_PORTS_V1_COMMIT)
    assert tr.rebuild_module() is legacy and tr._default_rebuild() is legacy.rebuild_training_models


# -- B8: the 11 pin-bound mechanisms need their k31 GPU re-run ---------------------

PIN_BOUND = {
    "losses:cispo", "losses:sapo", "losses:gmpo", "features:dual_clip", "features:kl_unbiased",
    "features:clip_higher", "loss_aggregations:token", "features:over_sampling", "features:overlong_filter",
    "corrections:opsm_rollout", "features:rollout_logprobs_as_old",
}


def test_pin_bound_mechanisms_are_marked_for_k31_reverification():
    assert set(entry.K31_REVERIFY) == set(entry.MILES_DECLARED_EVIDENCE) == PIN_BOUND
    assert set(entry.needs_k31_reverification()) == PIN_BOUND  # the k31 head: none re-run yet
    assert entry.needs_k31_reverification(pins.MILES_PORTS_V1_COMMIT) == {}
    assert entry.needs_k31_reverification("f" * 40) == {}
    declared = entry.declared_by_dimension(B8)
    for mechanism in PIN_BOUND:
        dimension, name = mechanism.split(":", 1)
        assert name not in declared.get(dimension, set()), mechanism
    assert "yeto/dual_clipfrac" in entry.MILES_DECLARED["features:dual_clip"]
    assert "yeto/dual_clipfrac" in entry.K31_REVERIFY["features:dual_clip"]


@pytest.mark.parametrize("extra, mechanisms", [
    ("--eps-clip-c 1.01", ("features:dual_clip",)),
    ("--policy-loss-variant cispo --calculate-per-token-loss --eps-clip 0.2 --eps-clip-high 0.28",
     ("losses:cispo", "loss_aggregations:token", "features:clip_higher")),
    ("--eps-clip-high 0.28", ("features:clip_higher",)),
])
def test_launcher_refuses_them_on_k31_like_any_unverified_mechanism(extra, mechanisms):
    from yeto.rl.adapters.miles import algorithm_flags as af

    refused = af.dry_run(["--dry-run", "--extra", extra])
    assert refused["verdict"] == "rejected"
    assert all(m.split(":", 1)[1] in refused["error"] for m in mechanisms), refused["error"]
    allow = [x for m in mechanisms for x in ("--rl-allow-unverified-mechanism", m)]
    allowed = af.dry_run(["--dry-run", "--extra", extra, *allow])
    assert allowed["verdict"] == "accepted", allowed.get("error")
    # the ports-v1 head declares them from its GPU evidence
    assert not set(mechanisms) & set(entry.needs_k31_reverification(pins.MILES_PORTS_V1_COMMIT))


def test_default_grpo_needs_none_of_them():
    from yeto.rl.adapters.miles import algorithm_flags as af

    assert af.dry_run(["--dry-run"])["verdict"] == "accepted"
