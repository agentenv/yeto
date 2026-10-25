"""S27 G-EU (VERL-PARITY #25, #26): the publication checks on a verl island.

#26 ``--rl-verify-every N`` (a no-sync island, Miles S27 C5): the full-parameter
policy states (FSDP2 local cut, Megatron) hash the policy on the first, every
N-th, the last and every cut round only; the other rounds publish the trainer
weights version as the identity. #25 ``--rl-verify-every-round``: a
full-parameter verl island refuses it (no vLLM read-back); a LoRA island
verifies every publication anyway. verl is not installed here."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import train_options as to
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import IslandDriver
from yeto.rl.engine.policy_digest import UNVERIFIED_PREFIX, WeightsChanged  # noqa: F401


def _args(**over):
    base = dict(rl_verify_every_round=False, rl_verify_every=None, rl_single_island_no_sync=True)
    return SimpleNamespace(**{**base, **over})


# --------------------------------------------------------------------- the plan keys


def test_verify_plan_maps_refuses_and_records():
    assert to.verify_plan(_args(), full=True) == {}
    assert to.verify_plan(_args(rl_verify_every=4), full=True) == {"verify_every": 4}
    with pytest.raises(ValueError, match="VERL-PARITY #25"):
        to.verify_plan(_args(rl_verify_every_round=True), full=True)
    assert to.verify_plan(_args(rl_verify_every_round=True), full=False) == {"verify_every_round": True}
    lora = to.verify_plan(_args(rl_verify_every=4), full=False)
    assert "verify_every" not in lora and "nothing is skipped" in lora["verify_every_note"]
    with pytest.raises(ValueError, match="no-sync"):  # run_flags.check_verify_every: a synced island
        to.verify_plan(_args(rl_verify_every=4, rl_single_island_no_sync=False), full=True)
    with pytest.raises(ValueError, match="N >= 1"):
        to.verify_plan(_args(rl_verify_every=0), full=True)


def test_the_verl_island_parses_both_flags():
    from yeto.rl.adapters.verl import island_entry

    argv = ["--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "0",
            "--reward-function", "r", "--global-rounds", "1", "--groups-per-round", "1",
            "--samples-per-group", "1", "--rollout-max-response-len", "1", "--event-tape", "t",
            "--inner-lr", "1", "--seq-len", "1", "--seed", "1",
            "--rl-verify-every", "3", "--rl-verify-every-round"]
    parsed, ignored = island_entry.parse_args(argv)
    assert parsed.rl_verify_every == 3 and parsed.rl_verify_every_round
    assert not [t for t in ignored if t.startswith("--rl-verify")], "never dropped silently"


def test_wire_verify_sets_the_driver_and_refuses_a_state_that_cannot_skip():
    driver, state = SimpleNamespace(), SimpleNamespace(allow_unverified_rounds=lambda: calls.append(1))
    calls: list = []
    to.wire_verify(driver, state, {"verify_every": 3, "verify_every_round": True})
    assert driver.verify_every == 3 and driver.verify_every_round and calls == [1]
    untouched = SimpleNamespace()
    to.wire_verify(untouched, object(), {})
    assert not vars(untouched)
    with pytest.raises(ValueError, match="cannot skip"):
        to.wire_verify(SimpleNamespace(), object(), {"verify_every": 2})


# ------------------------------------------------------- the FSDP2 local policy state


class _Cut:
    def __init__(self, version: int, content: str) -> None:
        self.layout_hash, self.policy_hash, self.content_hash = "l" * 64, f"p{content}".ljust(64, "0"), content


class _Adapter:
    expected_parameter_scalar_count = 10

    def __init__(self) -> None:
        self.captures = 0
        self.content = "c0"

    async def capture(self, group, *, policy_version, local_step_generation):
        self.captures += 1
        return _Cut(policy_version, self.content)


def _local_state():
    from yeto.rl.adapters.verl.full_parameter_stream import VerlFullParameterLocalState, _LoopRunner

    group = SimpleNamespace(trained_rounds=0)
    state = VerlFullParameterLocalState(group, component=SimpleNamespace(model_revision="a" * 40,
                                                                         config_hash="c" * 64),
                                        max_fragment_bytes=1 << 20)
    state.adapter = _Adapter()
    state.runner = _LoopRunner()
    return state, group


def _driver(state, every: int, rounds: int):
    return SimpleNamespace(policy_state=state, verify_every_round=False, verify_every=every,
                           sync=LocalOnlySync(rounds), published_version=None)


def test_the_local_state_skips_the_whole_cut_on_the_rounds_the_driver_does_not_verify():
    state, group = _local_state()
    assert not state.supports_unverified_rounds
    first = state.export_digest(policy_version=0)
    assert first.verified and first.policy_version == 0 and state.adapter.captures == 1
    assert first.policy_tensor_hash() == state.export().with_version(0).policy_tensor_hash(), (
        "the resident export equals the driver's fallback (export().with_version)")
    first.check_holds()
    group.trained_rounds += 1
    with pytest.raises(WeightsChanged):
        first.check_holds()
    assert state.export_identity(policy_version=1) is not None  # after a verified cut
    fresh, _ = _local_state()
    assert fresh.export_identity(policy_version=1) is None, "no verified cut yet: the driver verifies"
    # the driver's choice (IslandDriver.export_local_resident + verify_publication)
    state.allow_unverified_rounds()
    driver = _driver(state, every=3, rounds=7)
    seen = {}
    for version in range(1, 8):
        driver.published_version = version - 1
        captures = state.adapter.captures
        resident = IslandDriver.export_local_resident(driver, policy_version=version)
        seen[version] = (resident.verified, state.adapter.captures - captures)
        if not resident.verified:
            assert resident.policy_tensor_hash() != first.policy_tensor_hash()
            assert resident.digest.payload_bytes == first.digest.payload_bytes
            resident.check_holds()
        group.trained_rounds += 1
    # every 3rd version and the last round's (v7) are hashed; the others publish the identity
    assert seen == {1: (False, 0), 2: (False, 0), 3: (True, 1), 4: (False, 0), 5: (False, 0),
                    6: (True, 1), 7: (True, 1)}


def test_without_the_option_every_round_is_a_whole_cut_as_before():
    state, _ = _local_state()
    driver = _driver(state, every=None, rounds=4)
    driver.verify_every = None
    for version in range(1, 4):
        driver.published_version = version - 1
        assert IslandDriver.export_local_resident(driver, policy_version=version).verified
    assert state.adapter.captures == 3


# ------------------------------------------------------------ the Megatron policy state


def test_the_megatron_state_publishes_the_identity_on_unverified_rounds(monkeypatch):
    from yeto.rl.adapters.verl import megatron_ports as mp
    from yeto.rl.engine.policy_digest import PolicyDigest

    island = SimpleNamespace(process_id="p", trained_rounds=0, base_model_revision="a" * 40,
                             lora_config_hash="c" * 64, layout_hash="l" * 64, call_workers=lambda fn: [])
    digests = []
    monkeypatch.setattr(mp, "combine_digests", lambda records: digests.append(1) or PolicyDigest(
        "h" * 64, "h" * 64, 99, (("w", (2,)),)))
    monkeypatch.setattr(mp, "layout_hash_of", lambda specs: "l" * 64)
    state = mp.MegatronPolicyState(island)
    assert state.export_identity(policy_version=1) is None
    verified = state.export_digest(policy_version=0)
    state.allow_unverified_rounds()
    assert state.supports_unverified_rounds
    island.trained_rounds = 1
    identity = state.export_identity(policy_version=1)
    assert not identity.verified and identity.digest.payload_bytes == 99 and digests == [1]
    assert identity.policy_tensor_hash() != verified.policy_tensor_hash()
    identity.check_holds()
    island.trained_rounds = 2
    with pytest.raises(WeightsChanged):
        identity.check_holds()
    assert asyncio  # keep the import (the FSDP2 test's runner uses an event loop)


# ------------------------------------------------- #35: the debug dump and the island guard


def test_the_debug_dump_option_maps_and_reaches_the_trainer():
    assert to.debug_plan(SimpleNamespace(rl_debug_dump=False)) == {}
    assert to.debug_plan(SimpleNamespace(rl_debug_dump=True)) == {"debug_dump": True}
    island = SimpleNamespace()
    to.wire_debug(island, {"debug_dump": True})
    assert island.debug_dump is True
    to.wire_debug(island, {})
    assert island.debug_dump is False


def test_the_debug_dump_writes_what_verl_holds_and_never_fails_the_round(tmp_path):
    from yeto.rl.adapters.verl.ports_impl import DEBUG_DUMP_FIELDS, debug_dump

    events, saved = [], {}
    island = SimpleNamespace(out=tmp_path, learner_id=1, trainer=SimpleNamespace(global_steps=4),
                             emit=lambda event, **f: events.append((event, f)))
    held = {"uid": ["a", "b"], "prompts": [1, 2], "responses": [3, 4], "rm_scores": [0.0, 1.0]}

    def get(fields):
        (name,) = fields
        if name not in held:
            raise KeyError(name)
        return {name: held[name]}

    record = debug_dump(island, None, 3, get=get, save=lambda obj, path: saved.update(obj=obj, path=path))
    assert saved["path"].name == "verl-debug-1-r0003.pt"
    assert saved["obj"]["uid"] == ["a", "b"] and saved["obj"]["rollout_id"] == 3
    assert record["fields"] == sorted(held) and len(record["missing"]) == len(DEBUG_DUMP_FIELDS) - len(held)
    assert events[0][0] == "rl_verl_debug_dump"
    broken = debug_dump(island, None, 4, get=get, save=lambda obj, path: 1 / 0)
    assert broken["error"].startswith("ZeroDivisionError"), "archival: recorded, never raised"


def test_a_full_parameter_island_refuses_the_options_it_cannot_honour_even_without_the_launcher():
    from yeto.rl.adapters.verl import launch_flags

    to.refuse_unmapped_argv(["--rl-profile-trainer"], full=False)  # LoRA islands: unchanged
    to.refuse_unmapped_argv(["--source-sha256", "x"], full=True)  # Miles-only flags: as before
    with pytest.raises(ValueError, match="rl-async-buffer-rounds") as error:
        to.refuse_unmapped_argv(["--rl-async-buffer-rounds", "2", "--rl-profile-trainer"], full=True)
    assert launch_flags.NOT_MAPPED_REASONS["--rl-profile-trainer"] in str(error.value)
    assert set(launch_flags.NOT_APPLICABLE_ON_FULL_PARAMETER) <= set(launch_flags.NOT_MAPPED_REASONS)
    assert "--rl-debug-dump" not in launch_flags.NOT_MAPPED_REASONS, "mapped (debug_plan)"
