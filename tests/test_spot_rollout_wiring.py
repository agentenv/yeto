"""S27 G-EA wiring: the spot-rollout options are off by default (commands,
island argv and environment unchanged) and reach the syncer and islands when
given."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest

from test_rl_engine_selection import _cli, _island_task, _learner_argv
from test_rl_full_parameter_async import ASYNC, _plan
from yeto import launcher
from yeto.launcher import _prepare_rl_args
from yeto.rl.adapters.miles import island_entry as learner

NEW_ATTRS = ("rl_spot_subscribers", "rl_sample_relay", "rl_sample_relay_prefetch")


def test_default_island_command_is_unchanged(monkeypatch):
    args = _cli()
    assert args.rl_spot_subscribers == 0 and not args.rl_sample_relay
    assert args.rl_sample_relay_prefetch is None
    _prepare_rl_args(args)
    run = _island_task(args, monkeypatch).run
    assert "--rl-spot-subscribers" not in run and "--rl-sample-relay" not in run
    legacy = _cli()
    for name in NEW_ATTRS:
        delattr(legacy, name)
    _prepare_rl_args(legacy)
    assert _island_task(legacy, monkeypatch).run == run


def test_default_syncer_command_is_unchanged():
    args, _ = _plan(ASYNC)
    commands = launcher.role_syncer_commands(args, 2)
    for name in NEW_ATTRS:
        delattr(args, name)
    assert launcher.role_syncer_commands(args, 2) == commands
    assert "--max-subscribers" not in commands["actor"] and "--sample-relay" not in commands["actor"]


def test_spot_options_reach_the_actor_syncer_and_the_islands(monkeypatch):
    args, _ = _plan(ASYNC + ("--rl-spot-subscribers", "3", "--rl-sample-relay",
                             "--rl-sample-relay-prefetch", "16"))
    command = launcher.role_syncer_commands(args, 2)["actor"]
    assert command.endswith(" --max-subscribers 3 --sample-relay")
    flags = shlex.split(command)
    assert flags.count("--max-subscribers") == 1
    cli = _cli(("--rl-spot-subscribers", "2", "--rl-sample-relay"))
    _prepare_rl_args(cli)
    run = _island_task(cli, monkeypatch).run
    assert "--rl-spot-subscribers 2 --rl-sample-relay" in run


def test_spot_options_refuse_a_sharded_head():
    with pytest.raises(ValueError, match="unsharded syncer"):
        _plan(ASYNC + ("--rl-spot-subscribers", "1", "--syncer-shards", "2"))


def test_default_learner_environment_is_untouched():
    args = learner.parse_args(_learner_argv())
    assert args.rl_spot_subscribers == 0 and not args.rl_sample_relay
    miles_args, environ = SimpleNamespace(), {}
    learner.apply_sample_role(args, miles_args, environ)
    assert vars(miles_args) == {} and environ == {}


def test_spot_subscribers_make_island_zero_publish_its_layout():
    from yeto.rl.spot_rollout.layout_manifest import PUBLISH_ENV

    args = learner.parse_args(_learner_argv(("--rl-spot-subscribers", "2")))
    environ: dict[str, str] = {}
    learner.apply_sample_role(args, SimpleNamespace(), environ)
    assert environ == {PUBLISH_ENV: "1"}


def test_relay_trainer_reads_a_local_buffer_filled_by_the_pump(tmp_path):
    from yeto.rl.adapters.miles.buffer_rollout import BUFFER_ROOT_ENV, ISLAND_ENVS
    from yeto.rl.spot_rollout.relay_buffer import RELAY_ENVS

    args = SimpleNamespace(syncer="10.1.2.3:29400", learner_id=1, groups_per_round=32,
                           rl_sample_relay_prefetch=None)
    env = learner.sample_relay_env(args, tmp_root=str(tmp_path))
    assert env == {BUFFER_ROOT_ENV: str(tmp_path / "yeto-relay-buffer" / "island1"),
                   "YETO_SAMPLE_RELAY_ADDR": "10.1.2.3:29400", "YETO_SAMPLE_RELAY_CHANNEL": "1",
                   "YETO_SAMPLE_RELAY_PREFETCH": "64"}
    assert set(RELAY_ENVS) <= set(ISLAND_ENVS)  # forwarded to the rollout workers


def test_pump_is_off_without_the_relay_address(tmp_path):
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.spot_rollout.relay_buffer import maybe_start_relay_pump

    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "b"), batch_size=1)
    assert maybe_start_relay_pump(buffer, {}) is None


def test_layout_publication_is_off_by_default_and_for_other_islands():
    from yeto.rl.spot_rollout.layout_manifest import PUBLISH_ENV, maybe_publish

    from test_spot_rollout_syncer import layout

    assert maybe_publish(("127.0.0.1", 1), 0, layout(), environ={}) is None
    assert maybe_publish(("127.0.0.1", 1), 1, layout(), environ={PUBLISH_ENV: "1"}) is None


def test_layout_record_round_trip_and_fingerprint_check():
    from yeto.rl.spot_rollout import layout_manifest

    from test_spot_rollout_syncer import layout

    lay = layout()
    decoded, meta = layout_manifest.decode(layout_manifest.encode(lay, {"role": "actor"}))
    assert decoded == lay and meta == {"role": "actor"}
    record = layout_manifest.layout_record(lay)
    record["fragments"][0]["tensors"][0][1] += 1
    with pytest.raises(layout_manifest.LayoutManifestError, match="fingerprint"):
        layout_manifest.layout_from_record(json.loads(json.dumps(record)))


def test_wire_constants_match_the_syncer():
    from pathlib import Path

    from yeto import protocol
    from yeto.rl.spot_rollout import relay

    root = Path(__file__).resolve().parents[1] / "syncer" / "src"
    rust = (root / "sample_relay.rs").read_text() + (root / "protocol.rs").read_text() \
        + (root / "subscriber.rs").read_text()
    for name in ("MSG_RELAY_HELLO", "MSG_RELAY_WELCOME", "MSG_RELAY_PUT", "MSG_RELAY_PUT_ACK",
                 "MSG_RELAY_TAKE", "MSG_RELAY_ITEMS", "MSG_RELAY_STATUS"):
        assert f"pub const {name}: u8 = {getattr(relay, name)};" in rust
    assert f"pub const MSG_SUBSCRIBE: u8 = {protocol.MSG_SUBSCRIBE};" in rust
    assert "pub const SUBSCRIBER_ID_BIT: u32 = 0x8000_0000;" in rust
    assert 'b"yeto-subscribe-mac-v1\\0"' in rust and protocol.SUBSCRIBE_MAC_DOMAIN == b"yeto-subscribe-mac-v1\0"
    assert 'b"yeto-relay-hello-mac-v1\\0"' in rust and relay.RELAY_MAC_DOMAIN == b"yeto-relay-hello-mac-v1\0"
    for role in ("ROLE_PRODUCER", "ROLE_CONSUMER", "ROLE_BOARD"):
        assert f"pub const {role}: u8 = {getattr(relay, role)};" in rust
