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

NEW_ATTRS = ("rl_spot_subscribers", "rl_sample_relay", "rl_sample_relay_prefetch", "rl_spot_syncer",
             "rl_spot_channel")


def test_default_island_command_is_unchanged(monkeypatch):
    args = _cli()
    assert args.rl_spot_subscribers == 0 and not args.rl_sample_relay
    assert args.rl_sample_relay_prefetch is None
    _prepare_rl_args(args)
    run = _island_task(args, monkeypatch).run
    assert "--rl-spot-subscribers" not in run and "--rl-sample-relay" not in run
    assert "--rl-spot-syncer" not in run
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
    # S27 dispatch merge: the shared member table (stream, incarnation trailer, fences)
    from yeto.rl.spot_rollout import subscriber

    assert f"pub const RELAY_VERSION: u16 = {relay.RELAY_VERSION};" in rust and relay.RELAY_VERSION == 2
    assert relay._HELLO.size == 2 + 1 + 4 + 4 + 8 + 8
    assert f'pub const FENCED_PREFIX: &str = "{relay.FENCED_PREFIX}";' in rust
    assert f'pub const FENCED_PREFIX: &str = "{subscriber.FENCED_PREFIX}";' in rust
    assert 'pub const INCARNATION_MAGIC: &[u8; 4] = b"YSI1";' in rust
    assert protocol.SUBSCRIBE_INCARNATION_MAGIC == b"YSI1"
    members = (root / "members.rs").read_text()
    from yeto.rl.engine.task_dispense import SUPERSEDED_MARK

    assert f'Retired::Superseded => "{SUPERSEDED_MARK[1:-1]}"' in members


def test_one_sampler_process_has_one_member_incarnation_on_every_channel(monkeypatch):
    """The relay producer and the subscriber of a sampler process take the
    member incarnation its task dispenser takes (the process incarnation)."""
    from yeto.rl.engine import task_dispense as td
    from yeto.rl.spot_rollout.relay import RelayProducer
    from yeto.rl.spot_rollout.relay_buffer import relay_sampler_buffer
    from yeto.rl.spot_rollout.subscriber import SubscriberClient

    from test_spot_rollout_syncer import layout

    monkeypatch.setenv(td.SAMPLER_INCARNATION_ENV, "ab12cd34")
    expected = td.member_incarnation()
    assert expected == td.member_incarnation({td.SAMPLER_INCARNATION_ENV: "ab12cd34"})
    assert td.member_incarnations()() == expected
    producer = RelayProducer(("127.0.0.1", 1), 0, td.sampler_holder_id("s"))
    other = RelayProducer(("127.0.0.1", 1), 0, td.sampler_holder_id("s"))
    assert producer.incarnation == other.incarnation == expected
    assert producer.conn.stream != other.conn.stream  # two writers never share a sequence
    buffer = relay_sampler_buffer({"YETO_SAMPLE_RELAY_ADDR": "127.0.0.1:1",
                                   td.SAMPLER_INCARNATION_ENV: "ab12cd34"})
    assert buffer.producer.incarnation == expected
    client = SubscriberClient(("127.0.0.1", 1), td.sampler_holder_id("s"), layout(), num_streams=1)
    assert client.incarnation == expected
    assert client._hello_body_suffix() == b"YSI1" + expected.to_bytes(8, "little")


def test_spot_sampler_island_gets_the_subscriber_and_relay_env(monkeypatch):
    from yeto.rl.spot_rollout.relay_buffer import RELAY_ADDR_ENV, RELAY_CHANNEL_ENV, SPOT_SYNCER_ENV

    args = SimpleNamespace(rl_spot_syncer="head.example:29400", rl_spot_channel=1)
    assert learner.spot_sampler_env(args) == {SPOT_SYNCER_ENV: "head.example:29400",
                                              RELAY_ADDR_ENV: "head.example:29400",
                                              RELAY_CHANNEL_ENV: "1"}
    cli = _cli(("--rl-spot-syncer", "head.example:29400", "--rl-spot-channel", "1"))
    _prepare_rl_args(cli)
    run = _island_task(cli, monkeypatch).run
    assert "--rl-spot-syncer head.example:29400 --rl-spot-channel 1" in run


def test_a_sampler_buffer_goes_to_the_relay_only_with_the_relay_env(tmp_path):
    from yeto.rl.adapters.miles.buffer_rollout import BUFFER_ROOT_ENV, SAMPLER_ID_ENV, open_buffer
    from yeto.rl.engine.sample_buffer import FileSampleBuffer
    from yeto.rl.spot_rollout.relay_buffer import RelaySampleBuffer
    from yeto.rl.spot_rollout.release_store import subscriber_id_for

    env = {BUFFER_ROOT_ENV: str(tmp_path / "store"), SAMPLER_ID_ENV: "spot-a"}
    assert isinstance(open_buffer(env, sampler=True), FileSampleBuffer)
    relay = open_buffer({**env, "YETO_SAMPLE_RELAY_ADDR": "127.0.0.1:1", "YETO_SAMPLE_RELAY_CHANNEL": "2"},
                        sampler=True)
    assert isinstance(relay, RelaySampleBuffer)
    assert relay.producer.conn.channel == 2 and relay.producer.conn.client == subscriber_id_for("spot-a")
    # The trainer side (reader) never gets the relay buffer.
    reader = open_buffer({**env, "YETO_SAMPLE_RELAY_ADDR": "127.0.0.1:1"}, reader=True)
    assert isinstance(reader, FileSampleBuffer)


def test_subscriber_ids_are_the_dispenser_sampler_holder_ids():
    import hashlib

    from yeto.rl.spot_rollout.release_store import subscriber_id_for

    sid = subscriber_id_for("spot-a")
    digest = hashlib.sha256(b"yeto-sampler:spot-a").digest()
    assert sid == (1 << 31) | (int.from_bytes(digest[:4], "big") & 0x7FFFFFFF)
    assert sid >> 31 == 1 and subscriber_id_for("spot-b") != sid


def test_materialize_joins_row_slices_and_strips_the_wire_prefix(tmp_path):
    import torch
    from safetensors.torch import load_file

    from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
    from yeto.rl.eval.cut_to_hf import MaterializeError
    from yeto.rl.spot_rollout.materialize import identity_converter, materialize
    from yeto.rl.spot_rollout.subscriber import WeightCut

    shard = "tp0-of-1.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1"
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    top, bottom = f"actor::{shard}::decoder.w#rows=0:2", f"actor::{shard}::decoder.w#rows=2:4"
    lay = FragmentLayout([Fragment(MERGE_AVG, [(top, 6)], identity_shapes={top: (2, 3)}),
                          Fragment(MERGE_AVG, [(bottom, 6)], identity_shapes={bottom: (2, 3)})])

    def raw(t):
        return t.to(torch.bfloat16).flatten().view(torch.int16).numpy().tobytes()

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "x", "hidden_size": 3, "num_attention_heads": 1,
                                                  "vocab_size": 4, "num_hidden_layers": 1}))
    (base / "tokenizer.json").write_text("{}")
    cut = WeightCut((1, 2), (raw(weight[:2]), raw(weight[2:])))
    out = materialize(lay, cut, base_model=base, out_root=tmp_path / "out", convert=identity_converter)
    tensors = load_file(str(out / "model.safetensors"))
    assert sorted(tensors) == ["decoder.w"] and tensors["decoder.w"].float().tolist() == weight.tolist()
    assert (out / "tokenizer.json").is_file() and (out / "config.json").is_file()
    # With base weights in the directory, the HF names must equal them (G-EE): a
    # missing weight would keep its old value in the engine after the reload.
    from safetensors.torch import save_file

    save_file({"decoder.w": weight.to(torch.bfloat16)}, str(base / "model.safetensors"))
    named = materialize(lay, cut, base_model=base, out_root=tmp_path / "named", convert=identity_converter)
    assert sorted(load_file(str(named / "model.safetensors"))) == ["decoder.w"]
    save_file({"decoder.w": weight.to(torch.bfloat16), "decoder.v": weight.to(torch.bfloat16)},
              str(base / "model.safetensors"))
    with pytest.raises(MaterializeError, match=r"missing \['decoder.v'\]"):
        materialize(lay, cut, base_model=base, out_root=tmp_path / "missing", convert=identity_converter)
    (base / "model.safetensors").unlink()
    # A sharded trainer's layout (two shard ids) is refused: it needs param_layout.
    other = f"actor::tp1-of-2.pp0-of-1.ep0-of-1.cp0-of-1.dp0-of-1::decoder.w#rows=2:4"
    two = FragmentLayout([lay.fragments[0], Fragment(MERGE_AVG, [(other, 6)], identity_shapes={other: (2, 3)})])
    with pytest.raises(MaterializeError, match="param_layout"):
        materialize(two, WeightCut((1, 4), cut.fragments), base_model=base, out_root=tmp_path / "out2",
                    convert=identity_converter)
