"""Expert token weights: experts merge by the tokens each island routed to them.

Covers the island side (router counter on a stand-in router, routed-token
ledger, per-tensor tags), the wire (tag block, HELLO flag, unchanged bytes
when off) and the real syncer merging a tiny Qwen3-MoE layout against a NumPy
reference.
"""

from __future__ import annotations

import struct
import subprocess
import time

import numpy as np
import pytest
from test_rl_moe_full_parameter_layout import (  # noqa: F401 - fixture
    TINY,
    avg,
    free_port,
    layout,
    qwen3_moe_specs,
    rda,
    syncer_binary,
)
from test_rl_sao_streaming_runtime import _syncer_profile

from yeto.fragments import MERGE_AVG
from yeto.protocol import (
    DENSE_TENSOR_TAG,
    DTYPE_F32,
    MSG_PUSH_FRAGMENT,
    SyncerClient,
    encode_expert_tags,
    encode_hello,
)
from yeto.recovery_protocol import encode_hello_ext
from yeto.rl.moe_layout import RoutedTokens, expert_of, fragment_expert_tags
from yeto.syncer_profile import SyncerSemanticProfile

# --- island side -------------------------------------------------------------


def test_routed_tokens_accumulate_and_take_intervals():
    first = RoutedTokens.from_mapping({0: [1, 2, 0, 3], 1: [4, 0, 0, 0]})
    later = first + RoutedTokens.from_mapping({1: [0, 5, 0, 1]})
    assert later.as_dict() == {0: (1, 2, 0, 3), 1: (4, 5, 0, 1)}
    assert (later - first).as_dict() == {0: (0, 0, 0, 0), 1: (0, 5, 0, 1)}
    assert (later - RoutedTokens()).as_dict() == later.as_dict()
    with pytest.raises(ValueError, match="backwards"):
        first - later
    with pytest.raises(ValueError, match="lost layers"):
        RoutedTokens() - first
    with pytest.raises(ValueError, match="different expert counts"):
        RoutedTokens.from_mapping({0: [1, 2], 1: [1, 2, 3]})


def test_fragment_tags_name_each_expert_and_leave_dense_tensors_alone():
    names = [
        "actor::tp0-of-2::module.module.decoder.layers.1.mlp.experts.linear_fc2.weight3",
        "actor::tp0-of-2::module.module.decoder.layers.1.mlp.router.weight",
        "actor::tp0-of-2::module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0",
    ]
    assert [expert_of(name) for name in names] == [(1, 3), None, (0, 0)]
    interval = RoutedTokens.from_mapping({0: [6, 2, 0, 0], 1: [1, 1, 1, 0]})
    assert fragment_expert_tags(names, interval) == ((0, 3, 4), DENSE_TENSOR_TAG, (6, 8, 4))
    with pytest.raises(ValueError, match="no routed-token count"):
        fragment_expert_tags(names, RoutedTokens.from_mapping({0: [1, 1, 1, 1]}))


def test_router_counter_counts_training_forwards_once_and_drops_padding():
    torch = pytest.importorskip("torch")
    from yeto.rl.adapters.miles import routing_plugin

    class StandInRouter(torch.nn.Module):
        """Megatron TopKRouter call shape: (hidden, padding_mask, ...) -> (probs, map)."""

        def __init__(self, layer_number: int, experts: int, topk: int) -> None:
            super().__init__()
            self.layer_number = layer_number
            self.gate = torch.nn.Linear(4, experts, bias=False)
            self.topk = topk

        def forward(self, hidden, padding_mask=None, input_ids=None, packed_seq_params=None):
            logits = self.gate(hidden)
            top = logits.topk(self.topk, dim=-1).indices
            routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, top, True)
            return logits.softmax(-1) * routing_map, routing_map

    torch.manual_seed(0)
    model = [torch.nn.ModuleList([StandInRouter(1, 4, 2), StandInRouter(2, 4, 2)])]
    assert routing_plugin.install_hooks(model, StandInRouter) == 2
    assert routing_plugin.install_hooks(model, StandInRouter) == 2  # idempotent
    first, second = model[0]
    hidden = torch.randn(6, 4)
    padding = torch.tensor([False, False, False, False, True, True])

    with torch.no_grad():  # log-probability pass: not counted
        first(hidden, padding)
    _, routing_map = first(hidden, padding)
    expected = routing_map[:4].sum(0).tolist()
    _, routing_map = second(hidden)
    drained = routing_plugin.drain(model, StandInRouter)
    assert drained == {"0": expected, "1": routing_map.sum(0).tolist()}
    assert sum(drained["0"]) == 4 * 2  # 4 unpadded tokens, top-2
    assert routing_plugin.drain(model, StandInRouter) == {"0": [0] * 4, "1": [0] * 4}


# --- wire --------------------------------------------------------------------


def test_off_keeps_hello_and_push_bytes():
    specs = qwen3_moe_specs(**TINY, tp=2, ep=2)
    lay = layout(specs, 6).fragments
    plain = encode_hello(0, DTYPE_F32, lay, 1, 1)
    assert encode_hello(0, DTYPE_F32, lay, 1, 1, expert_tokens=False) == plain
    assert encode_hello_ext(None, False, False) == b""
    flagged = encode_hello(0, DTYPE_F32, lay, 1, 1, expert_tokens=True)
    assert flagged[: len(plain)] == plain
    assert flagged[len(plain):] == b"YEX1" + bytes([4]) + bytes(8)

    def pushed(expert_tokens: bool, **kwargs) -> bytes:
        client = SyncerClient(("127.0.0.1", 1), 0, lay, dtype=DTYPE_F32, num_streams=1,
                              max_reconnects=0, expert_tokens=expert_tokens)
        sent = []
        client._send_large = lambda msg_type, payload: sent.append((msg_type, payload))
        numel = lay.fragments[0].numel
        client.push_fragment(0, 1, 1, 0, 1, 1, 10, bytes(4 * numel), **kwargs)
        assert sent[0][0] == MSG_PUSH_FRAGMENT
        return sent[0][1]

    head = struct.pack("<IIQIQQIQ", 0, 0, 1, 1, 0, 1, 1, 10)
    assert pushed(False) == head + bytes(4 * lay.fragments[0].numel)
    tags = [DENSE_TENSOR_TAG] * len(lay.fragments[0].tensors)
    with_tags = pushed(True, expert_tags=tags)
    assert with_tags[len(head):len(head) + 4] == struct.pack("<I", len(tags))
    assert len(with_tags) == len(head) + 4 + 20 * len(tags) + 4 * lay.fragments[0].numel
    with pytest.raises(ValueError, match="exactly when"):
        pushed(False, expert_tags=tags)
    with pytest.raises(ValueError, match="exactly when"):
        pushed(True)


def test_tag_block_refuses_inconsistent_counts():
    assert encode_expert_tags([(0, 0, 0), (3, 4, 2)], 2)[:4] == struct.pack("<I", 2)
    with pytest.raises(ValueError, match="cover 1 tensors"):
        encode_expert_tags([(0, 0, 0)], 2)
    with pytest.raises(ValueError, match="dense tensor"):
        encode_expert_tags([(1, 1, 0)], 1)
    with pytest.raises(ValueError, match="routed 5 of 4"):
        encode_expert_tags([(5, 4, 2)], 1)


def test_profile_hash_appends_only_when_on():
    # Same vector as server.rs semantic_profile_hash_matches_the_python_canonical_vector.
    raw = _syncer_profile()
    off = SyncerSemanticProfile.from_mapping(raw).sha256
    assert SyncerSemanticProfile.from_mapping({**raw, "expert_token_weights": False}).sha256 == off
    assert SyncerSemanticProfile.from_mapping({**raw, "expert_token_weights": True}).sha256 == (
        "f51de8a095c0fa428c5483c7921da805bb8b3cd2736fe19fdd03f47e4f733913"
    )


# --- real syncer -------------------------------------------------------------


def expert_scale(tag: tuple[int, int, int]) -> float:
    """NumPy-side reference of the syncer's per-tensor scale."""
    routed, layer_routed, experts = tag
    if experts == 0:
        return 1.0
    return 0.0 if layer_routed == 0 else routed * experts / layer_routed


def start_syncer(binary, port: int, steps: int, tmp_path, *extra: str) -> subprocess.Popen:
    return subprocess.Popen(
        [str(binary), "--port", str(port), "--learners", "2", "--quorum", "2",
         "--grace-ms", "20", "--quorum-timeout-s", "10", "--total-steps", str(steps),
         "--pipeline", "1", "--sync-interval-steps", "0", "--delta-correction", "none",
         "--outer-lr", "1", "--outer-momentum", "0", "--learner-weight", "tokens2-over-steps",
         "--final-state", str(tmp_path / "final.bin"), *extra],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


@pytest.mark.timeout(60)
def test_two_islands_merge_experts_by_routed_tokens(syncer_binary, tmp_path):
    specs = qwen3_moe_specs(**TINY, tp=2, ep=2)
    lay = layout(specs, 6)
    num = lay.fragments.num_fragments
    rng = np.random.default_rng(11)
    init = {s.wire_name: rng.standard_normal(s.numel).astype(np.float32) for s in specs}
    deltas = [
        {s.wire_name: (0.01 * rng.standard_normal(s.numel)).astype(np.float32) for s in specs}
        for _ in range(2)
    ]
    # Interval counts (top-2 of 4 experts). Island 1 never routes to expert 3
    # of layer 0; island 0 sends layer 1 mostly to expert 0.
    routed = [
        RoutedTokens.from_mapping({0: [5, 5, 5, 5], 1: [14, 2, 2, 2]}),
        RoutedTokens.from_mapping({0: [30, 20, 10, 0], 1: [15, 15, 15, 15]}),
    ]
    tokens = (10, 30)
    island_weights = [t * t / 1 for t in tokens]  # tokens^2 / steps
    tags = [
        {fid: fragment_expert_tags([n for n, _ in lay.fragments.fragments[fid].tensors], r)
         for fid in range(num)}
        for r in routed
    ]

    def fragment_bytes(values, fid: int) -> bytes:
        return np.concatenate([values[n] for n, _ in lay.fragments.fragments[fid].tensors]).tobytes()

    port = free_port()
    proc = start_syncer(syncer_binary, port, num, tmp_path, "--expert-token-weights")
    clients = [
        SyncerClient(("127.0.0.1", port), island, lay.fragments, dtype=DTYPE_F32,
                     num_streams=1, connect_timeout=10, max_reconnects=0,
                     finalization_timeout=30, expert_tokens=True)
        for island in range(2)
    ]
    try:
        for client in clients:
            client.start()
        for fid in range(num):
            clients[0].send_init(fid, fragment_bytes(init, fid))
        pushed = [set(), set()]
        deadline = time.monotonic() + 30
        while any(len(p) < num for p in pushed):
            assert time.monotonic() < deadline, "syncer did not permit every fragment"
            for island, client in enumerate(clients):
                client.check_health()
                for permit in client.drain_pulls():
                    fid = permit.fragment_id
                    client.push_fragment(
                        fid, permit.global_step, permit.round_attempt, 0, 1, 1, tokens[island],
                        fragment_bytes(deltas[island], fid), expert_tags=tags[island][fid],
                    )
                    pushed[island].add(fid)
            time.sleep(0.01)
        _, finals = clients[0].wait_for_final_fragments(timeout=30)
        checked = set()
        for final in finals:
            fragment = lay.fragments.fragments[final.fragment_id]
            got = np.frombuffer(bytes(final.data), dtype=np.float32)
            offset = 0
            for index, (name, numel) in enumerate(fragment.tensors):
                weights = [w * expert_scale(tags[m][final.fragment_id][index])
                           for m, w in enumerate(island_weights)]
                merge = avg if fragment.merge_mode == MERGE_AVG else rda
                expected = init[name] + merge([d[name] for d in deltas], weights)
                np.testing.assert_allclose(got[offset:offset + numel], expected, rtol=1e-5, atol=1e-6)
                if expert_of(name) == (0, 3):
                    # Island 1 never routed here: island 0's update alone.
                    np.testing.assert_allclose(got[offset:offset + numel] - init[name],
                                               deltas[0][name], rtol=1e-4, atol=5e-7)
                    checked.add(name)
                offset += numel
        assert len(checked) == 2  # both projections of expert 3, layer 0
    finally:
        for client in clients:
            client.close()
        stop(proc)


@pytest.mark.timeout(30)
def test_syncer_refuses_an_island_without_counts(syncer_binary, tmp_path):
    lay = layout(qwen3_moe_specs(**TINY, tp=2, ep=2), 6)
    port = free_port()
    proc = start_syncer(syncer_binary, port, lay.fragments.num_fragments, tmp_path,
                        "--expert-token-weights")
    client = SyncerClient(("127.0.0.1", port), 0, lay.fragments, dtype=DTYPE_F32,
                          num_streams=1, connect_timeout=10, max_reconnects=0)
    try:
        client.start()
        deadline = time.monotonic() + 15
        with pytest.raises(Exception, match="expert token"):
            while time.monotonic() < deadline:
                client.check_health()
                time.sleep(0.02)
    finally:
        client.close()
        stop(proc)


# --- launcher and role stream ------------------------------------------------


def test_launcher_turns_the_option_on_for_syncer_profile_and_islands():
    from test_rl_full_parameter_async import ASYNC, _flag, _plan  # noqa: F401

    from yeto import launcher

    args, plan = _plan(ASYNC + ("--rl-sync-expert-token-weights",))
    assert "--expert-token-weights" in plan["role_syncers"][0]["command"].split()
    profile = launcher.rl_syncer_profile(args, 2)
    assert profile.expert_token_weights
    off_args, off_plan = _plan(ASYNC)
    assert "--expert-token-weights" not in off_plan["role_syncers"][0]["command"]
    assert profile.sha256 != launcher.rl_syncer_profile(off_args, 2).sha256
    for island in plan["island_requests"]:
        command = island["learner_command"]
        assert "--sweep-expert-token-weights" in command.split()
        assert _flag(command, "--syncer-profile-sha256") == profile.sha256
    assert all("--sweep-expert-token-weights" not in island["learner_command"]
               for island in off_plan["island_requests"])


def test_launcher_and_island_refuse_the_option_on_the_sweep(monkeypatch):
    from test_rl_full_parameter_async import _launcher_args  # strict control
    from test_rl_full_parameter_async import FULL, TWO_ISLANDS, _island_args

    from yeto.launcher import _prepare_rl_args
    from yeto.rl.adapters.miles.island_entry import _check_dense_full_sweep

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    with pytest.raises(ValueError, match="expert-token-weights only with"):
        _prepare_rl_args(_launcher_args("ports", FULL + ("--rl-sync-expert-token-weights",),
                                        gpu=TWO_ISLANDS))
    _check_dense_full_sweep(_island_args(sweep_expert_token_weights=True), yeto_policy_sync=True)
    with pytest.raises(ValueError, match="needs asynchronous rounds"):
        _check_dense_full_sweep(_island_args(full_parameter_rounds="sweep", local_horizon=1,
                                             sweep_expert_token_weights=True),
                                yeto_policy_sync=True)


@pytest.mark.asyncio
async def test_role_stream_sums_rank_counts_and_tags_by_interval():
    from dataclasses import replace

    from test_rl_full_parameter_sync import _role_stream_config

    from yeto.rl.adapters.miles.full_parameter.role_stream import MilesFullParameterRoleStream
    from yeto.rl.adapters.miles.routing_plugin import DRAIN_ROUTED_TOKEN_COUNTS

    class Group:
        def __init__(self):
            self.drains = [
                [{"0": [1, 0, 2, 1]}, {"0": [0, 3, 0, 1]}],  # two ranks, round 1
                [{"0": [4, 0, 0, 0]}, {}],                   # round 2
            ]

        async def run_plugin(self, fn_path, kwargs=None):
            assert fn_path == DRAIN_ROUTED_TOKEN_COUNTS
            return self.drains.pop(0)

    stream = MilesFullParameterRoleStream(
        replace(_role_stream_config("actor", 30100), expert_token_weights=True))
    stream.group = Group()
    anchor = await stream._drain_routed_tokens()
    assert anchor.as_dict() == {0: (1, 3, 2, 2)}
    later = anchor + await stream._drain_routed_tokens()
    names = ["actor::s::decoder.layers.0.mlp.experts.linear_fc1.weight0",
             "actor::s::decoder.layers.0.mlp.experts.linear_fc1.weight1"]
    # The push after the anchor carries only round 2's routing.
    assert fragment_expert_tags(names, later - anchor) == ((4, 4, 4), (0, 4, 4))
