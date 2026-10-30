"""S28 G-LHASH: the light policy digest of a verl Megatron full-parameter island.

Invariants of ``adapters/verl/local_digest`` (module doc): same weights give the
same digest; one changed element changes it; the per-rank rows combine the same
way in any order, tile every parameter exactly, and carry the topology, so a
different TP/PP split gives a different (but self-consistent) digest.  And the
driver runs the full check on the first, every cut and the last round only."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.adapters.verl import local_digest as ld  # noqa: E402
from yeto.rl.engine.bridges import LocalOnlySync  # noqa: E402
from yeto.rl.engine.driver import IslandDriver  # noqa: E402

TOPO = {"tp": 1, "pp": 1, "ep": 1, "cp": 1, "dp": 1, "chunks": 1}


def _raw(t):
    return memoryview(t.detach().reshape(-1).contiguous().view(torch.uint8).numpy())


def _record(shards, *, topology=TOPO, algorithm="sha256", threads=1):
    """One rank's record from ``[(key, start, stop, numel, fp32 tensor)]``."""
    digests = ld.hash_shards(algorithm, [("float32", s.numel(), _raw(s)) for *_k, s in shards], threads=threads)
    rows = [[k, a, b, n, d] for (k, a, b, n, _s), d in zip(shards, digests)]
    return {"rank": 0, "algorithm": algorithm, "topology": topology, "rows": rows, "bytes": 0, "seconds": 0.0}


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"a.weight": torch.randn(8, 6, generator=g), "b.bias": torch.randn(5, generator=g)}


def _split(weights, tp: int, *, dp: int = 1):
    """Records of ``tp`` ranks (row split of a.weight, b.bias replicated), each DP-sharded
    into ``dp`` flat ranges, as a DistributedOptimizer would."""
    records = []
    for t in range(tp):
        for d in range(dp):
            shards = []
            for name, value in weights.items():
                local = value.chunk(tp, dim=0)[t] if name == "a.weight" else value
                flat = local.reshape(-1)
                bounds = [round(i * flat.numel() / dp) for i in range(dp + 1)]
                start, stop = bounds[d], bounds[d + 1]
                shards.append((f"0:{name}@tp{t}.pp0", start, stop, flat.numel(), flat[start:stop]))
            records.append(_record(shards, topology={**TOPO, "tp": tp, "dp": dp}))
    return records


# ----------------------------------------------------------------------------- hashing
def test_same_weights_give_the_same_digest_and_one_changed_element_changes_it(monkeypatch):
    monkeypatch.setattr(ld, "BLOCK_BYTES", 16)  # several blocks per shard
    w = _weights()
    first = ld.combine(_split(w, 2, dp=2))["digest"]
    assert first == ld.combine(_split(_weights(), 2, dp=2))["digest"]
    assert len(first) == 64
    changed = _weights()
    changed["a.weight"][7, 5] = torch.nextafter(changed["a.weight"][7, 5], torch.tensor(1e9))
    assert ld.combine(_split(changed, 2, dp=2))["digest"] != first


def test_the_digest_does_not_depend_on_threads_or_rank_order(monkeypatch):
    monkeypatch.setattr(ld, "BLOCK_BYTES", 32)
    w = _weights()
    one = [_record([(f"0:{n}@tp0.pp0", 0, v.numel(), v.numel(), v.reshape(-1)) for n, v in w.items()], threads=1)]
    many = [_record([(f"0:{n}@tp0.pp0", 0, v.numel(), v.numel(), v.reshape(-1)) for n, v in w.items()], threads=8)]
    assert ld.combine(one)["digest"] == ld.combine(many)["digest"]
    records = _split(w, 2, dp=2)
    assert ld.combine(records)["digest"] == ld.combine(list(reversed(records)))["digest"]


@pytest.mark.parametrize("prefer", [("blake3",), ("xxh3_128",), ("sha256",)])
def test_every_available_algorithm_is_deterministic_and_named(prefer):
    try:
        name = ld.algorithm(prefer)
    except ValueError:
        pytest.skip(f"{prefer[0]} is not installed")
    records = [_record([("0:x@tp0.pp0", 0, 4, 4, torch.arange(4.0))], algorithm=name)]
    a, b = ld.combine(records), ld.combine(records)
    assert a == b and a["algorithm"] == name
    assert ld.algorithm(("nope-not-a-hash", "sha256")) == "sha256"


# ----------------------------------------------------------------------------- combine rules
def test_different_splits_are_self_consistent_but_never_equal():
    w = _weights()
    digests = {}
    for tp, dp in ((1, 1), (2, 1), (1, 3), (2, 2)):
        result = ld.combine(_split(w, tp, dp=dp))
        assert result["digest"] == ld.combine(_split(w, tp, dp=dp))["digest"]
        assert result["topology"]["tp"] == tp and result["topology"]["dp"] == dp
        digests[(tp, dp)] = result["digest"]
    assert len(set(digests.values())) == 4, "the topology is part of the light digest"


def test_combine_refuses_gaps_overlaps_divergence_and_mixed_ranks():
    w = _weights()
    records = _split(w, 1, dp=2)
    with pytest.raises(ld.LightDigestError, match="cover"):
        ld.combine(records[:1])  # dp rank 1's ranges missing
    overlap = _split(w, 1, dp=2)
    overlap[1]["rows"][0][1] -= 1
    with pytest.raises(ld.LightDigestError, match="gap or overlaps"):
        ld.combine(overlap)
    replica = _split(w, 1)[0]
    diverged = {**replica, "rows": [[*row[:4], "f" * 64] for row in replica["rows"]]}
    with pytest.raises(ld.LightDigestError, match="diverged"):
        ld.combine([replica, diverged])
    assert ld.combine([replica, replica])["digest"] == ld.combine([replica])["digest"]  # same replica twice
    with pytest.raises(ld.LightDigestError, match="topologies"):
        ld.combine([replica, {**replica, "topology": {**TOPO, "tp": 2}}])
    with pytest.raises(ld.LightDigestError, match="algorithms"):
        ld.combine([replica, {**replica, "algorithm": "blake3"}])


# ----------------------------------------------------------------------------- worker side
def _patch_worker(monkeypatch, module, coords, optimizer=None):
    from yeto.rl.adapters.verl import megatron_full_parameter as mfp
    from yeto.rl.adapters.verl import megatron_ports as mp

    monkeypatch.setattr(mfp, "_engine", lambda wd: (None, [module], optimizer))
    monkeypatch.setattr(mfp, "_rank", lambda: 0)
    monkeypatch.setattr(mp, "_parallel_coordinates", lambda: dict(coords))
    monkeypatch.setattr(ld, "_topology", lambda chunks: {**TOPO, "chunks": chunks})


def test_the_worker_hashes_complete_fp32_masters_on_dp_rank_zero_only(monkeypatch):
    layer = torch.nn.Linear(3, 2)
    expert = torch.nn.Parameter(torch.arange(4.0).reshape(2, 2))
    expert.allreduce = False
    layer.register_parameter("experts_w", expert)
    _patch_worker(monkeypatch, layer, dict(tp=0, pp=0, ep=1, dp=1, edp=0))
    record = ld.remote_light_digest(None, threads=2, prefer=("sha256",))
    assert [row[0] for row in record["rows"]] == ["0:experts_w@tp0.pp0.ep1"]
    assert record["bytes"] == 16 and record["algorithm"] == "sha256"
    _patch_worker(monkeypatch, layer, dict(tp=0, pp=0, ep=0, dp=0, edp=0))
    first = ld.combine([ld.remote_light_digest(None, prefer=("sha256",))])
    assert first["rows"] == 3
    with torch.no_grad():
        layer.weight[0, 0] += 1.0
    assert ld.combine([ld.remote_light_digest(None, prefer=("sha256",))])["digest"] != first["digest"]
    with torch.no_grad():
        layer.bias[0] = float("nan")
    with pytest.raises(ld.LightDigestError, match="NaN"):
        ld.remote_light_digest(None, prefer=("sha256",))


def test_the_worker_hashes_its_distributed_optimizer_ranges(monkeypatch):
    from yeto.rl import megatron_masters as mm

    layer = torch.nn.Linear(4, 2, bias=False).to(torch.bfloat16)
    master = torch.arange(8.0)
    records = []
    for dp, (start, stop) in enumerate(((0, 5), (5, 8))):
        _patch_worker(monkeypatch, layer, dict(tp=0, pp=0, ep=0, dp=dp, edp=dp))
        monkeypatch.setattr(mm, "distributed_ranges",
                            lambda opt, s=start, e=stop: (True, {id(layer.weight): ("leaf", layer.weight, s, e)}))
        monkeypatch.setattr(mm, "_main_shard", lambda leaf, p, s=start, e=stop: master[s:e])
        records.append(ld.remote_light_digest(None, prefer=("sha256",)))
    assert [r["rows"][0][1:4] for r in records] == [[0, 5, 8], [5, 8, 8]]
    whole = ld.combine(records)
    assert whole["rows"] == 2 and whole["bytes"] == 32
    with pytest.raises(ld.LightDigestError, match="cover"):
        ld.combine(records[:1])


# ----------------------------------------------------------------------------- policy state + driver
class _Adapter:
    expected_parameter_scalar_count = 10
    layout = SimpleNamespace(layout_hash="l" * 64)

    def __init__(self):
        self.full = 0

    async def fragment_digests(self, group):
        self.full += 1
        return ("a" * 64,)


class _Group:
    def __init__(self):
        self.trained_rounds = 0
        self.light = 0
        self.content = "1"

    async def light_digest(self):
        self.light += 1
        return {"digest": (self.content * 64)[:64], "algorithm": "sha256", "rows": 1, "bytes": 40, "seconds": 0.1}


def _state(group, **kw):
    from yeto.rl.adapters.verl.full_parameter_stream import VerlFullParameterLocalState, _LoopRunner

    state = VerlFullParameterLocalState(group, component=SimpleNamespace(model_revision="a" * 40, config_hash="c" * 64),
                                        max_fragment_bytes=1 << 20, **kw)
    state.adapter = _Adapter()
    state.runner = _LoopRunner()
    return state


def _publish(state, group, rounds, *, every=None, cut_due=None):
    sync = LocalOnlySync(rounds)
    if cut_due is not None:
        sync.cut_due = cut_due
    driver = SimpleNamespace(policy_state=state, verify_every_round=False, verify_every=every, sync=sync,
                             published_version=None)
    seen = {}
    for version in range(0, rounds + 1):
        full, light = state.adapter.full, group.light
        resident = IslandDriver.export_local_resident(driver, policy_version=version)
        seen[version] = "full" if state.adapter.full > full else ("light" if group.light > light else "none")
        if seen[version] == "light":
            assert not resident.verified and len(resident.policy_tensor_hash()) == 64
            resident.check_holds()
        driver.published_version = version
        group.trained_rounds += 1
    return seen


def test_the_full_check_runs_on_the_first_cut_and_last_rounds_only(monkeypatch):
    monkeypatch.delenv("YETO_RL_LIGHT_DIGEST", raising=False)
    group = _Group()
    state = _state(group)
    assert state.light and state.supports_unverified_rounds
    seen = _publish(state, group, 6, cut_due=lambda v: v == 3)
    assert seen == {0: "full", 1: "light", 2: "light", 3: "full", 4: "light", 5: "light", 6: "full"}
    group2 = _Group()
    assert _publish(_state(group2), group2, 6, every=2)[4] == "full", "--rl-verify-every still adds its rounds"


def test_the_light_policy_hash_follows_the_weights():
    group = _Group()
    state = _state(group, light_digest=True)
    state.export_digest(policy_version=0)
    a = state.export_identity(policy_version=1).policy_tensor_hash()
    assert a == state.export_identity(policy_version=1).policy_tensor_hash()
    group.content = "2"
    assert state.export_identity(policy_version=1).policy_tensor_hash() != a


def test_the_switch_off_and_a_group_without_light_digest_keep_the_full_check(monkeypatch):
    monkeypatch.setenv("YETO_RL_LIGHT_DIGEST", "0")
    group = _Group()
    state = _state(group)
    assert not state.light and not state.supports_unverified_rounds
    assert set(_publish(state, group, 3).values()) == {"full"}
    monkeypatch.delenv("YETO_RL_LIGHT_DIGEST")
    plain = SimpleNamespace(trained_rounds=0, light=0)  # the FSDP2 group: no light_digest
    state = _state(plain)
    assert not state.light
    assert asyncio  # the runner uses an event loop
