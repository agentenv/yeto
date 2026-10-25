"""S27 P3 (data mix, task label affinity) and P4 (sampler islands) of the task
dispenser: wire golden bytes, the stream table, the launch plan and checks,
and stand-in islands and samplers against the real Rust syncer (no GPU, no
Ray, no Miles)."""

from __future__ import annotations

import json
import random
import shlex
import threading
import time
from fractions import Fraction
from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_buffer_rollout import _import_plugin_with_stub_miles
from test_s27_dispense import FP, KEY, Syncer, crash, syncer_binary  # noqa: F401
from test_s27_dispense_wiring import FakeMilesSource, FakeSample, _flag
from tests.buffer_rollout_procs import sample as buffer_sample
from tools.eval.dispense_check import check, check_tapes, prompt_table
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.engine import task_dispense as td
from yeto.rl.engine.data_split import DATA_ROW_KEY, DISPENSE, pass_order, row_runs

TWO_ISLANDS = "modal:1xh100,modal:1xh100"
THREE_ISLANDS = "modal:1xh100,modal:1xh100,modal:1xh100"


# ------------------------------------------------------------------ golden


def test_stream_hello_matches_the_rust_golden_vector():
    """Same bytes as ``dispense_service::tests::wire_golden_vectors_with_streams``."""
    body = td.encode_task_hello_streams(3, 0x0102030405060708, b"\xab" * 32,
                                        [(10, 7, "a/web"), (5, 3, "")], [1])
    assert body.hex() == ("0200030000000807060504030201" + "ab" * 32 + "02000a000000000000000700000000"
                          "0000000500612f77656205000000000000000300000000000000000001000100")
    sealed = td.seal_task_hello(body, b"k")
    assert sealed[len(body):].hex() == "ae4b9a4a78ae1aa7fbc05b27d614d78767b3464bc01b72ba7e2a30d6a83a9e97"
    welcome = bytes.fromhex("0200e0930400000000000000000000000000000001000001090000000000000006"
                            "000000000000002800000000000000")
    assert td.decode_welcome(welcome, td.TASK_PROTOCOL_VERSION_STREAMS).superseded == 9
    with pytest.raises(td.DispenseError, match="protocol 2"):
        td.decode_welcome(welcome)  # a P2 island refuses a version-2 answer


def test_the_plain_data_still_speaks_protocol_1(syncer_binary, tmp_path):
    """Default unchanged: no mix and no labels -> the P2 TASK_HELLO bytes."""
    sent = []
    real = td.DispenseConnection._send

    def spy(self, msg_type, payload):
        sent.append((msg_type, payload))
        return real(self, msg_type, payload)

    syncer = Syncer(syncer_binary, tmp_path)
    try:
        td.DispenseConnection._send = spy
        conn = td.DispenseConnection(("127.0.0.1", syncer.port), learner_id=0, incarnation=5, rows=12,
                                     fingerprint=FP, key=KEY, streams=[(12, 1, "")])
        conn.close()
    finally:
        td.DispenseConnection._send = real
        syncer.stop()
    assert sent[0][1][:2] == b"\x01\x00" and len(sent[0][1]) == 54 + 32
    assert not any(r["kind"] == "dispense_open" and r.get("accept") for r in syncer.records())


# ------------------------------------------------------------- stream table


def test_stream_table_splits_by_source_and_label_and_weights_them():
    plain = td.StreamTable.build([(0, "")] * 4, FP)
    assert plain.single and plain.decl() == [(4, 1, "")]
    labels = td.StreamTable.build([(0, "web"), (0, "pwn"), (0, "web"), (0, ""), (0, "web")], FP)
    assert not labels.single
    assert [(s.label, s.rows, s.weight) for s in labels.streams] == [
        ("", (3,), 1), ("pwn", (1,), 1), ("web", (0, 2, 4), 3)], "sorted, weighted by rows"
    assert labels.accept(["web"]) == [2] and labels.accept(["-", "pwn"]) == [0, 1]
    assert labels.accept(["web", "pwn", "-"]) is None and labels.accept(None) is None
    with pytest.raises(ValueError, match="no row of the data has task label"):
        labels.accept(["crypto"])
    mix = td.StreamTable.build([(0, "")] * 6 + [(1, "a"), (1, "b")], FP, mix=["0.7", "0.3"])
    shares = [Fraction(s.weight, sum(t.weight for t in mix.streams)) for s in mix.streams]
    assert [s.name for s in mix.streams] == ["s0", "s1/a", "s1/b"]
    assert all(abs(float(a - b)) < 1e-12 for a, b in zip(shares, [Fraction(7, 10), Fraction(3, 20), Fraction(3, 20)]))
    with pytest.raises(ValueError, match="a source without rows"):
        td.StreamTable.build([(0, "")], FP, mix=["1", "1"])
    round_trip = td.StreamTable.from_config(json.loads(json.dumps(mix.to_config())))
    assert round_trip.decl() == mix.decl() and [s.rows for s in round_trip.streams] == [s.rows for s in mix.streams]
    for bad in ("", "a,b", "x" * 49, td.UNLABELED):
        with pytest.raises(ValueError):
            td.check_label(bad)


def test_position_map_keeps_p2_and_maps_streams_to_their_rows():
    one = td.PositionMap(10, seed=3)
    assert [one.row(p) for p in range(10)] == list(range(10))
    assert [one.row(p) for p in range(10, 20)] == pass_order(10, seed=3, data_pass=1)
    table = td.StreamTable.build([(0, "web"), (0, "pwn"), (0, "web"), (0, "pwn"), (0, "web")], FP)
    pmap = td.PositionMap(seed=3, table=table)
    web = [pmap.row(td.position_of(1, k)) for k in range(6)]
    assert web[:3] == [0, 2, 4] and sorted(web[3:]) == [0, 2, 4], "pass 1 reshuffles the stream"
    assert [pmap.row(td.position_of(0, k)) for k in range(2)] == [1, 3]
    assert pmap.data_pass(td.position_of(1, 3)) == 1 and pmap.rows == 5
    assert pass_order(5, seed=3, data_pass=1, stream=0) == pass_order(5, seed=3, data_pass=1)
    with pytest.raises(ValueError, match="stream 7"):
        pmap.row(td.position_of(7, 0))


def test_island_task_labels_spec():
    assert td.parse_island_labels("l0=web;l1=pwn, crypto;2=*;L3=-") == {
        0: ["web"], 1: ["crypto", "pwn"], 2: None, 3: ["-"]}
    for bad in ("l0", "x0=web", "l0=web;l0=pwn", "l0=", "l0=*,web", "l0=a=b", ""):
        with pytest.raises(ValueError):
            td.parse_island_labels(bad)
    labels = {0: ["web"], 1: ["pwn"]}
    assert td.unmatched_labels({"web", "pwn", ""}, [0, 1], labels) == [""]
    assert td.unmatched_labels({"web", "pwn", ""}, [0, 1, 2], labels) == [], "island 2 takes every row"
    assert td.unmatched_labels({"web", ""}, [0, 1], {0: ["web"], 1: ["-"]}) == []


# ------------------------------------------------- stand-in islands, real syncer


def _streams_source(port, learner_id, table, *, block, accept=None, prefetch=1):
    connect = td.tcp_connector(("127.0.0.1", port), learner_id=learner_id, rows=table.total_rows,
                               fingerprint=table.fingerprint, key=KEY, timeout_s=5.0,
                               streams=table.decl(), accept=accept)
    return td.TaskSource(connect=connect, block=block, prefetch_blocks=prefetch, retry_s=0.1).start()


def test_two_islands_train_the_mix_within_one_block(syncer_binary, tmp_path):
    """--data-mix 0.7/0.3: whatever their speeds, each island's trained rows are
    within one block of the weights, and the fresh positions given out are within
    one position of them at every grant (the checker's mix_error)."""
    keys = [(0, "")] * 50 + [(1, "")] * 20
    table = td.StreamTable.build(keys, FP, mix=["0.7", "0.3"])
    syncer = Syncer(syncer_binary, tmp_path)
    block = 10
    try:
        sources = [_streams_source(syncer.port, i, table, block=block) for i in range(2)]
        trained = {0: [], 1: []}
        rng = random.Random(4)
        for _ in range(30):
            island = 0 if rng.random() < 0.7 else 1  # island 0 is faster
            got = sources[island].take(block)
            sources[island].report(trained=got)
            trained[island].extend(got)
        for s in sources:
            s.close()
        for island, positions in trained.items():
            share = sum(1 for p in positions if td.stream_of(p) == 0) / len(positions)
            assert abs(share - 0.7) * len(positions) <= block, (island, share, len(positions))
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["mix_error"] < 1.0, summary["mix_error"]
        assert set(summary["streams"]) == {"s0", "s1"}
        assert summary["streams"]["s0"]["weight_share"] == pytest.approx(0.7)
        assert summary["islands"]["0"]["trained_by_stream"]["s0"] > 0
    finally:
        syncer.stop()


def test_label_affinity_gives_each_island_only_its_labels(syncer_binary, tmp_path):
    keys = [(0, "web")] * 12 + [(0, "pwn")] * 8 + [(0, "")] * 4
    table = td.StreamTable.build(keys, FP)
    accepts = {0: table.accept(["web"]), 1: table.accept(["pwn", "-"]), 2: None}
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1")
    try:
        sources = {i: _streams_source(syncer.port, i, table, block=4, accept=a) for i, a in accepts.items()}
        allowed = {0: {"web"}, 1: {"pwn", ""}, 2: {"web", "pwn", ""}}
        seen = {i: [] for i in sources}
        ended = set()
        while len(ended) < 3:
            for i, src in sources.items():
                if i in ended:
                    continue
                try:
                    # nothing fresh for this island while another holds rows: return short
                    got = src.take(4, timeout_s=10, drained_ok=True)
                except td.DispenseEnded:
                    ended.add(i)
                    continue
                if not got:
                    time.sleep(0.05)
                for p in got:
                    assert table.streams[td.stream_of(p)].label in allowed[i], (i, p)
                seen[i].extend(got)
                src.report(trained=got)
        for src in sources.values():
            src.close()
        everything = sorted(p for got in seen.values() for p in got)
        assert len(everything) == len(set(everything)) == 24, "one pass, every row once"
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["islands"]["0"]["accept"] == accepts[0]
        assert {k: v["trained"] for k, v in summary["streams"].items()} == {"stream0": 4, "pwn": 8, "web": 12}
    finally:
        syncer.stop()


def test_an_island_with_another_stream_table_is_refused(syncer_binary, tmp_path):
    table = td.StreamTable.build([(0, "web"), (0, "pwn")], FP)
    other = td.StreamTable.build([(0, "web"), (0, "web")], FP)
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        _streams_source(syncer.port, 0, table, block=1).close()
        with pytest.raises(td.DispenseMismatch, match="other streams or weights"):
            _streams_source(syncer.port, 1, other, block=1)
        with pytest.raises(td.DispenseError, match="accepts stream 5"):
            _streams_source(syncer.port, 2, table, block=1, accept=[5])
    finally:
        syncer.stop()


def test_miles_hook_with_label_streams_draws_only_the_islands_rows(syncer_binary, tmp_path):
    """The island side end to end: prompt file -> StreamTable -> config -> hook."""
    from yeto.rl.adapters.miles import task_dispense as hook
    from yeto.rl.adapters.miles.island_entry import _dispense_streams, prepare_prompt_data

    data = tmp_path / "d.jsonl"
    labels = ["web", "pwn", "web", "pwn", "web", "web"]
    data.write_text("".join(json.dumps({"prompt": f"q{i}", "task_label": lab}) + "\n"
                            for i, lab in enumerate(labels)))
    prompts = prepare_prompt_data(str(data), None, tmp_path / "p.jsonl", number_rows=True)
    rows, digest = td.file_fingerprint(prompts)
    island = SimpleNamespace(rl_dispense_mix=None, rl_dispense_by_label=True, rl_dispense_labels="pwn")
    extra = _dispense_streams(island, prompts, rows, digest)
    assert extra["accept"] == [0] and [s["label"] for s in extra["stream_record"]] == ["pwn", "web"]
    assert _dispense_streams(SimpleNamespace(rl_dispense_mix=None, rl_dispense_by_label=False,
                                             rl_dispense_labels=None), prompts, rows, digest) == {}
    with pytest.raises(ValueError, match="no row of the data has task label"):
        _dispense_streams(SimpleNamespace(rl_dispense_mix=None, rl_dispense_by_label=True,
                                          rl_dispense_labels="crypto"), prompts, rows, digest)
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        fake = FakeMilesSource(rows)
        config = {"address": f"127.0.0.1:{syncer.port}", "learner_id": 1, "rows": rows,
                  "fingerprint": digest.hex(), "seed": 2, "block": 2, "prefetch_blocks": 1, **extra}
        hook._CURRENT = None
        hook.install(SimpleNamespace(data_source=fake), config, KEY)
        drawn = [g[0].metadata[DATA_ROW_KEY] for _ in range(3) for g in fake.get_samples(2)]
        assert sorted(set(drawn)) == [1, 3] and len(drawn) == 6, "pwn rows only, passes after the first"
        hook.current().tasks.close()
    finally:
        hook._CURRENT = None
        syncer.stop()


# ---------------------------------------------------------------- the launch


def _launch(tmp_path, monkeypatch, gpu, *extra, data=None, mix=None):
    from yeto import runs
    from yeto.launch.data_mix import materialize_data_mix

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    args = _launcher_args("ports", tuple(extra), gpu=gpu)
    args.controller = "local"
    if data is not None:
        args.data, args.data_revision = str(data), None
    if mix is not None:
        args.data, args.data_revision, args.data_mix = None, None, mix
        materialize_data_mix(args)
    return args


def _commands(args):
    from yeto import launcher

    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    return plan, [i["learner_command"] for i in plan["island_requests"]]


def _rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_data_mix_becomes_one_prompt_file_and_the_islands_get_the_weights(tmp_path, monkeypatch, capsys):
    a = _rows(tmp_path / "a.jsonl", [{"prompt": f"a{i}"} for i in range(5)])
    b = _rows(tmp_path / "b.jsonl", [{"prompt": f"b{i}", "metadata": {"k": i}} for i in range(3)])
    args = _launch(tmp_path, monkeypatch, TWO_ISLANDS, "--rl-data-split", "dispense",
                   mix=f"{a}=0.7,{b}=0.3")
    mixed = [json.loads(line) for line in open(args.data)]
    assert [r["metadata"][td.DATA_SOURCE_KEY] for r in mixed] == [0] * 5 + [1] * 3
    assert mixed[5]["metadata"]["k"] == 0 and args.rl_dispense_mix == "0.7,0.3" and args.data_mix is None
    assert [s["rows"] for s in args.data_mix_record["sources"]] == [5, 3]
    plan, commands = _commands(args)
    for command in commands:
        assert _flag(command, "--rl-dispense-mix") == "0.7,0.3"
        assert "--rl-dispense-by-label" not in shlex.split(command)
    assert "data mix (2 sources" in capsys.readouterr().err
    table = td.StreamTable.from_prompts(args.data, mix=["0.7", "0.3"])
    assert [len(s.rows) for s in table.streams] == [5, 3]
    # the mix needs the dispenser
    with pytest.raises(ValueError, match="--data-mix needs --rl-data-split dispense"):
        _commands(_launch(tmp_path, monkeypatch, TWO_ISLANDS, mix=f"{a}=1,{b}=1"))
    with pytest.raises(ValueError, match="two or more"):
        _launch(tmp_path, monkeypatch, TWO_ISLANDS, mix=f"{a}=1")
    with pytest.raises(ValueError, match="positive"):
        _launch(tmp_path, monkeypatch, TWO_ISLANDS, mix=f"{a}=0,{b}=1")


def test_data_mix_and_data_exclude_each_other_on_the_command_line(capsys):
    from yeto.cli import parse_args

    base = ["--gpu", "modal:1xh100", "--model", "org/m", "--training-mode", "rl"]
    assert parse_args([*base, "--data-mix", "a=1,b=1"]).data is None
    with pytest.raises(SystemExit):
        parse_args([*base, "--data", "org/d", "--data-mix", "a=1,b=1"])
    with pytest.raises(SystemExit):
        parse_args(base)
    capsys.readouterr()


def test_task_labels_reach_the_islands_and_bad_specs_fail_at_launch(tmp_path, monkeypatch, capsys):
    data = _rows(tmp_path / "d.jsonl", [{"prompt": f"q{i}", "task_label": lab}
                                       for i, lab in enumerate(["web", "pwn", "web", "pwn"])])
    args = _launch(tmp_path, monkeypatch, THREE_ISLANDS, "--rl-data-split", "dispense",
                   "--rl-island-task-labels", "l0=web;l1=pwn", data=data)
    _, commands = _commands(args)
    assert [_flag(c, "--rl-dispense-labels") for c in commands] == ["web", "pwn", None]
    assert all("--rl-dispense-by-label" in shlex.split(c) for c in commands)
    err = capsys.readouterr().err
    assert "task labels {'pwn': 2, 'web': 2}" in err and "rl-sync-learner-weight equal" in err
    cases = [
        (("--rl-data-split", "dispense", "--rl-island-task-labels", "l0=web;l1=web"), TWO_ISLANDS,
         r"no island takes the rows with task label\(s\) \['pwn'\]"),
        (("--rl-data-split", "dispense", "--rl-island-task-labels", "l0=web,crypto"), TWO_ISLANDS,
         r"no row of --data has task label\(s\) \['crypto'\]"),
        (("--rl-data-split", "dispense", "--rl-island-task-labels", "l5=web"), TWO_ISLANDS,
         r"names island\(s\) \[5\]"),
        (("--rl-island-task-labels", "l0=web;l1=pwn"), TWO_ISLANDS, "needs --rl-data-split dispense"),
        (("--rl-data-split", "dispense", "--rl-dispense-labels", "web"), TWO_ISLANDS, "set by the launcher"),
        (("--rl-data-split", "dispense", "--rl-dispense-mix", "1,1"), TWO_ISLANDS, "set by the launcher"),
    ]
    for extra, gpu, message in cases:
        with pytest.raises(ValueError, match=message):
            _commands(_launch(tmp_path, monkeypatch, gpu, *extra, data=data))


# ------------------------------------------------------------- P4 launch


def _group(tmp_path, monkeypatch, *extra):
    from test_rl_synced_group_full_parameter import GROUP, GROUP_GPU

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "p4-test-key")
    return _launch(tmp_path, monkeypatch, GROUP_GPU, *GROUP, *extra)


def test_a_synced_groups_samplers_claim_from_its_syncer(tmp_path, monkeypatch, capsys):
    from yeto.launch.syncer import syncer_command

    default = _group(tmp_path, monkeypatch)
    _, commands = _commands(default)
    assert [_flag(c, "--rl-sampler-data-slot") for c in commands[2:]] == ["0/2", "1/2"], "default unchanged"
    assert not any("--rl-data-dispense" in shlex.split(c) for c in commands)
    assert "--dispense" not in syncer_command(default, 2)
    args = _group(tmp_path, monkeypatch, "--rl-data-split", "dispense")
    plan, commands = _commands(args)
    for command in commands[2:]:
        tokens = shlex.split(command)
        assert "--rl-data-dispense" in tokens and _flag(command, "--rl-dispense-addr") == "$SYNCER_ADDR"
        assert _flag(command, "--rl-sampler-data-slot") is None
    for command in commands[:2]:
        assert "--rl-data-dispense" not in shlex.split(command), "buffer trainers read the buffer"
    assert plan["data_split"]["mode"] == DISPENSE and plan["data_split"]["samplers"] == [2, 3]
    assert " --dispense " in syncer_command(args, 2)
    assert "sampler islands 2, 3 claim" in capsys.readouterr().err
    # the sampler's island args parse and its config names a sampler island id
    from test_rl_synced_group_full_parameter import _islands

    sampler = [a for i, a in _islands(plan) if i["sample_role"] == "sampler"][0]
    assert sampler.rl_data_dispense and sampler.rl_dispense_addr == "10.0.0.1:29400"
    labelled = _group(tmp_path, monkeypatch, "--rl-data-split", "dispense", "--rl-data-passes", "1")
    with pytest.raises(ValueError, match="sampler islands do not end"):
        _commands(labelled)


def test_sampler_dispense_is_refused_without_a_syncer_and_a_sampler_launch_names_one(tmp_path, monkeypatch):
    from test_rl_sampler_trainer_pair_launch import PAIR, PAIR_GPU, SPEC

    with pytest.raises(ValueError, match="store-coupled pair has no syncer"):
        _commands(_launch(tmp_path, monkeypatch, PAIR_GPU, *PAIR, "--rl-data-split", "dispense"))
    alone = ("--rl-sample-role", "sampler", "--rl-single-island-no-sync", "--tuning", "full",
             "--rl-algorithm-spec", SPEC, "--rl-checkpoint-store", "modal-volume://p4/run",
             "--rl-data-split", "dispense")
    with pytest.raises(ValueError, match="give --rl-dispense-addr"):
        _commands(_launch(tmp_path, monkeypatch, "modal:1xh100", *alone))
    args = _launch(tmp_path, monkeypatch, "modal:1xh100", *alone, "--rl-sampler-node", "spot7",
                   "--rl-dispense-addr", "203.0.113.5:29400")
    plan, commands = _commands(args)
    assert _flag(commands[0], "--rl-dispense-addr") == "203.0.113.5:29400"
    assert plan["data_split"]["samplers"] == [0]
    from yeto.launch.data_split_plan import dispense_syncer_flags
    from yeto.launch.dispense_evidence import dispense_planned

    assert dispense_syncer_flags(args) == "" and not dispense_planned(args), "it runs no dispenser"
    with pytest.raises(ValueError, match="needs --rl-data-split dispense"):
        _commands(_launch(tmp_path, monkeypatch, "modal:1xh100", *alone[:-2],
                          "--rl-dispense-addr", "203.0.113.5:29400"))


def test_a_sampler_island_config_uses_its_sampler_id_and_dispenser(tmp_path):
    from yeto.rl.adapters.miles.island_entry import _dispense_config, _island_dispense

    prompts = _rows(tmp_path / "p.jsonl", [{"messages": [], "metadata": {DATA_ROW_KEY: i}} for i in range(4)])
    args = SimpleNamespace(rl_data_dispense=True, rl_data_slot=None, data_sha256=None, eval_only=False,
                           rl_sample_role="sampler", rl_sampler_data_slot=None,
                           rl_dispense_addr="10.0.0.1:29400", rl_sampler_node="island2", learner_id=0,
                           rl_dispense_prefetch_blocks=2, rl_dispense_mix=None, rl_dispense_by_label=False,
                           rl_dispense_labels=None, syncer=None)
    assert _island_dispense(args)
    miles = SimpleNamespace(fully_async=False, over_sampling_batch_size=None, rollout_batch_size=8, seed=3)
    config = _dispense_config(args, miles, prompts)
    assert config["address"] == "10.0.0.1:29400" and config["block"] == 8
    assert config["learner_id"] == td.sampler_holder_id("island2") >= td.SAMPLER_HOLDER_BASE
    with pytest.raises(ValueError, match="needs --rl-dispense-addr"):
        _island_dispense(SimpleNamespace(**{**vars(args), "rl_dispense_addr": None}))
    with pytest.raises(ValueError, match="exclude each other"):
        _island_dispense(SimpleNamespace(**{**vars(args), "rl_sampler_data_slot": "0/2"}))


# --------------------------------------------- P4 stand-in samplers, real syncer


class BufferSample(FakeSample):
    """A generated sample as the buffer writer sees it (status, payload dict)."""

    status = "completed"
    version = 1

    def to_dict(self):
        out = buffer_sample(self.index, self.group_index, self.version)
        out["metadata"] = dict(self.metadata)
        return out


class BufferMilesSource(FakeMilesSource):
    def __init__(self, rows):
        super().__init__(rows)
        samples = [BufferSample(i) for i in range(rows)]
        self.dataset.origin_samples = samples
        self.dataset.samples = list(samples)


class StandInSampler:
    """One sampler island process: the hook in its data source, the buffer writer
    of ``BufferWriterRolloutFn`` (the real ``_write``), its own buffer shard."""

    def __init__(self, plugin, hook, port, name, rows, root, shards):
        from yeto.rl.engine.sampler_resume import new_incarnation

        self.hook, self.name = hook, name
        self.fake = BufferMilesSource(rows)
        config = {"address": f"127.0.0.1:{port}", "learner_id": td.sampler_holder_id(name), "rows": rows,
                  "fingerprint": FP.hex(), "seed": 9, "block": 4, "prefetch_blocks": 1}
        hook._CURRENT = None
        hook.install(SimpleNamespace(data_source=self.fake), config, KEY)
        self.wrapped = hook.current()
        env = {br.BUFFER_ROOT_ENV: str(root), br.SAMPLER_ID_ENV: name, br.SHARDS_ENV: ",".join(shards)}
        self.fn = plugin.BufferWriterRolloutFn.__new__(plugin.BufferWriterRolloutFn)
        self.fn._rounds = br.SamplerRoundWriter(br.open_buffer(env, sampler=True), name,
                                                incarnation=new_incarnation())
        self.rollout_id = self.fn._rounds.resume.next_rollout_id

    def draw(self):
        return self.fake.get_samples(4)

    def write(self, groups, version):
        for group in groups:
            for s in group:
                s.version = version
        self.hook._CURRENT = self.wrapped  # one rollout executor per process
        self.fn._write(self.rollout_id, groups, None)
        self.rollout_id += 1

    def round(self, version):
        groups = self.draw()
        self.write(groups, version)
        return [g[0].metadata[td.DISPENSE_POSITION_KEY] for g in groups]


def test_samplers_share_rows_survive_preemption_and_accept_a_new_sampler(
        syncer_binary, tmp_path, monkeypatch):
    """Two stand-in sampler islands claim from the real syncer, write their groups
    into their buffer shards; sampler s1 is preempted mid-round (relaunched with
    the same sampler id), sampler s0 is preempted for good (its lease ends), a
    new sampler s2 joins; a buffer trainer passes over the oldest groups."""
    from yeto.rl.adapters.miles import task_dispense as hook

    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(Status=SimpleNamespace(ABORTED="aborted")))
    live: list[StandInSampler] = []

    def written_by_owner(rollout_id, positions):  # each process has its own rollout executor
        owner = next(s for s in reversed(live) if set(positions) <= s.wrapped.tasks.taken())
        return owner.wrapped.written(rollout_id, positions)

    monkeypatch.setattr(plugin, "dispense_written", written_by_owner)
    root, shards, rows = tmp_path / "store", ["s0", "s1", "s2"], 48
    syncer = Syncer(syncer_binary, tmp_path, "--dispense-passes", "1", lease_s=1.5)
    try:
        s0 = StandInSampler(plugin, hook, syncer.port, "s0", rows, root, shards)
        s1 = StandInSampler(plugin, hook, syncer.port, "s1", rows, root, shards)
        live.extend([s0, s1])
        written = s0.round(1) + s1.round(1)
        assert len(set(written)) == 8, "two samplers draw disjoint rows"
        # s1 is preempted after drawing a round, before writing it; relaunched as s1
        lost = [g[0].metadata[td.DISPENSE_POSITION_KEY] for g in s1.draw()]
        crash(s1.wrapped.tasks)
        s1 = StandInSampler(plugin, hook, syncer.port, "s1", rows, root, shards)
        live.append(s1)
        again = s1.round(2)
        assert sorted(again) == sorted(lost), "the relaunched sampler takes them over at once"
        written += again
        # s0 is preempted for good holding a drawn round: after its lease the rows
        # go back first in line, to whichever sampler claims next
        gone = [g[0].metadata[td.DISPENSE_POSITION_KEY] for g in s0.draw()]
        crash(s0.wrapped.tasks)
        live.remove(s0)
        time.sleep(2.0)
        # a sampler that was not there at the start just claims
        s2 = StandInSampler(plugin, hook, syncer.port, "s2", rows, root, shards)
        live.append(s2)
        first_s2 = s2.round(2)
        assert first_s2 and not set(first_s2) & set(written), "the new sampler's rows are new to the buffer"
        written += first_s2
        lock = threading.Lock()

        def run(sampler):
            while True:
                try:
                    got = sampler.round(3)
                except td.DispenseEnded:
                    return
                with lock:
                    written.extend(got)
                time.sleep(0.005 if sampler is s1 else 0.02)

        threads = [threading.Thread(target=run, args=(x,)) for x in (s1, s2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads)
        for sampler in (s1, s2):
            hook._CURRENT = sampler.wrapped
            hook.close_source(None)
        assert sorted(written) == list(range(rows)), "every row in the buffer exactly once"
        assert set(gone) <= set(written[12:]), "the expired sampler's rows were written after its lease"
        # the buffer trainer: groups of version 1 are too stale for version 4 (age 2)
        reader_env = {br.BUFFER_ROOT_ENV: str(root), br.SHARDS_ENV: ",".join(shards)}
        reader = br.BatchReader(br.open_buffer(reader_env, batch_size=4, reader=True),
                                max_policy_age=2, wait_s=0.0, poll_s=0.0)
        trained_rows, stale = [], []
        for rid in range(100):
            try:
                taken, report = reader.take(rid, 4, current_version=4)
            except br.SampleBufferRolloutError:
                break
            trained_rows += [g[0]["metadata"][DATA_ROW_KEY] for _, g in taken]
            stale += report.stale_positions
            if len(trained_rows) + len(stale) >= rows:
                break
        assert sorted(stale) == sorted(written[:8]), "the version-1 groups, by dispenser position"
        summary = check(syncer.records())
        assert summary["ok"], summary["problems"]
        assert summary["trained"] == rows and summary["queued_at_end"] == 0
        sampler_ids = {str(td.sampler_holder_id(n)) for n in shards}
        assert set(summary["islands"]) == sampler_ids
        assert summary["islands"][str(td.sampler_holder_id("s1"))]["incarnations"] == 2
        reasons = {r["reason"] for r in syncer.records() if r["kind"] == "dispense_reclaim"}
        assert reasons == {"superseded", "lease_expired"}
        # the trainer tape against the samplers' reports (S27 P4 checker mode)
        tape = [{"island_id": 0, "event": "rl_driver_start"},
                {"island_id": 0, "event": "rl_round_data_rows", "rollout_id": 0,
                 "rows": row_runs(trained_rows)},
                {"island_id": 0, "event": "rl_buffer_stale_skipped", "rollout_id": 0,
                 "stale_positions": row_runs(stale)}]
        info = {}
        assert check_tapes(summary, tape, seed=9, info=info) == []
        assert info["stale_positions"] == 8 and info["sampler_rows_reported"] == rows
        bad = tape[:1] + [{**tape[1], "rows": row_runs([999])}]
        assert check_tapes(summary, bad, seed=9) and "reported by no sampler" in check_tapes(summary, bad, seed=9)[0]
    finally:
        hook._CURRENT = None
        syncer.stop()


def test_stale_positions_reach_the_round_metadata(monkeypatch):
    from yeto.rl.adapters.miles.rollout_meta_hook import record_round_metadata

    sent = []
    monkeypatch.setattr("yeto.rl.adapters.miles.rollout_meta_hook.put_to_sink",
                        lambda record, sink: sent.append(record))
    record_round_metadata(None, 3, buffer_stale_groups=2, buffer_stale_positions=[[4, 6, 2]])
    assert sent[0]["buffer_stale_positions"] == [[4, 6, 2]]
    with pytest.raises(RuntimeError, match="list of int runs"):
        record_round_metadata(None, 3, buffer_stale_positions=[4])


def test_the_checker_tape_check_with_streams_needs_the_prompt_rows(syncer_binary, tmp_path):
    data = _rows(tmp_path / "d.jsonl", [{"prompt": f"q{i}", "task_label": "web" if i % 3 else "pwn"}
                                       for i in range(9)])
    table = prompt_table(str(data), mix=None, by_label=True)
    syncer = Syncer(syncer_binary, tmp_path)
    try:
        src = _streams_source(syncer.port, 0, table, block=3)
        got = src.take(6)
        src.report(trained=got)
        src.close()
        summary = check(syncer.records())
        pmap = td.PositionMap(seed=1, table=table)
        tape = [{"island_id": 0, "event": "rl_round_data_rows", "rollout_id": 0,
                 "rows": row_runs(pmap.row(p) for p in got)}]
        assert "needs --prompts" in check_tapes(summary, tape, seed=1)[0]
        assert check_tapes(summary, tape, seed=1, table=table) == []
    finally:
        syncer.stop()


def test_the_sampler_entry_installs_and_closes_the_hook_in_the_rollout_executor(syncer_binary, tmp_path):
    """``sampler_island._in_executor`` reaches the executor the way
    ``MilesRolloutPool`` does (``__ray_call__`` on the actor handle)."""
    import asyncio

    from yeto.rl.adapters.miles import task_dispense as hook
    from yeto.rl.adapters.miles.sampler_island import _in_executor

    syncer = Syncer(syncer_binary, tmp_path)
    try:
        actor = SimpleNamespace(data_source=FakeMilesSource(8))
        handle = SimpleNamespace(__ray_call__=SimpleNamespace(remote=lambda fn, *a: fn(actor, *a)))
        executor = SimpleNamespace(_actor_handle=handle)  # miles RayWorkerHandle
        config = {"address": f"127.0.0.1:{syncer.port}", "learner_id": td.sampler_holder_id("island3"),
                  "rows": 8, "fingerprint": FP.hex(), "seed": 0, "block": 2, "prefetch_blocks": 1}
        hook._CURRENT = None
        out = _in_executor(executor, asyncio.run, "install", config, KEY)
        assert out["installed"]
        actor.data_source.get_samples(2)
        closed = _in_executor(executor, asyncio.run, "close_source")
        assert closed["released"] >= 2 and hook.current() is None
        summary = check(syncer.records())
        # the drawn and the prefetched rows went back at once, not at the end of the lease
        assert summary["ok"] and summary["held_at_end"] == 0
        assert summary["queued_at_end"] == summary["given_out"] >= 2
    finally:
        hook._CURRENT = None
        syncer.stop()
