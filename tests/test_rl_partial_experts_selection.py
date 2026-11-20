"""expert_selection.json generation and validation (INTERFACES 2.3)."""

from __future__ import annotations

import json
import sys

import numpy
import pytest

from yeto.rl.adapters.miles.models.partial_experts import selection as sel
from yeto.rl.adapters.miles.models.partial_experts.attest import canonical_sha256
from yeto.rl.engine.buffer_payload import encode_group

N, K, EP, LAYERS, TOPK = 64, 16, 4, 3, 4  # a small K3.1-shaped model: MoE layers 1..3


def _skewed_counts(seed: int = 0) -> sel.Counts:
    rng = numpy.random.default_rng(seed)
    counts: sel.Counts = {}
    for layer in range(1, LAYERS + 1):
        # heavy-tailed load: a different hot set per layer
        weights = rng.pareto(1.5, N) + 1e-3
        weights = numpy.roll(weights, layer * 7)
        counts[layer] = [int(x) for x in numpy.round(weights / weights.sum() * 100_000)]
    return counts


def _routed_from_counts(counts: sel.Counts, *, tokens: int, seed: int = 0) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    layers = sorted(counts)
    out = numpy.zeros((tokens, len(layers), TOPK), dtype=numpy.uint16)
    for j, layer in enumerate(layers):
        p = numpy.asarray(counts[layer], dtype=float)
        p /= p.sum()
        for t in range(tokens):
            out[t, j] = rng.choice(N, size=TOPK, replace=False, p=p)
    return out


# -- counting -----------------------------------------------------------------


def test_counts_from_routed_experts_is_per_layer_with_offset():
    routed = numpy.zeros((5, 2, TOPK), dtype=numpy.uint16)
    routed[:, 0, :] = 3  # MoE layer 0 -> HF layer 1
    routed[:, 1, :] = numpy.arange(TOPK)
    counts = sel.counts_from_routed_experts(routed, num_experts=N, moe_layer_offset=1)
    assert set(counts) == {1, 2}
    assert counts[1][3] == 5 * TOPK and sum(counts[1]) == 5 * TOPK
    assert counts[2][:TOPK] == [5] * TOPK
    with pytest.raises(sel.SelectionError, match="outside"):
        sel.counts_from_routed_experts(numpy.full((1, 1, 1), N, dtype=numpy.uint16), num_experts=N, moe_layer_offset=1)
    with pytest.raises(sel.SelectionError, match="T, L, top_k"):
        sel.counts_from_routed_experts(numpy.zeros((2, 2), dtype=numpy.uint16), num_experts=N, moe_layer_offset=1)


def test_counts_skip_pads_uint16_and_signed():
    routed = numpy.full((3, 2, TOPK), sel.ROUTED_EXPERTS_PAD, dtype=numpy.uint16)
    routed[0, 0, :] = 7
    counts = sel.counts_from_routed_experts(routed, num_experts=N, moe_layer_offset=1)
    assert counts[1][7] == TOPK and sum(counts[1]) == TOPK and sum(counts[2]) == 0
    signed = numpy.full((2, 1, TOPK), -1, dtype=numpy.int32)
    signed[1, 0, :] = 5
    counts = sel.counts_from_routed_experts(signed, num_experts=N, moe_layer_offset=1)
    assert counts[1][5] == TOPK and sum(counts[1]) == TOPK
    # an all-pad array contributes nothing but still names its layers
    counts = sel.counts_from_routed_experts(numpy.full((1, 2, TOPK), -1, dtype=numpy.int32), num_experts=N, moe_layer_offset=1)
    assert set(counts) == {1, 2} and not any(sum(row) for row in counts.values())


def test_counts_from_payloads_reads_uint16_and_legacy_uint8(tmp_path):
    counts = _skewed_counts()
    routed = _routed_from_counts(counts, tokens=20)
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(encode_group([{"tokens": [1], sel.ROUTED_EXPERTS_FIELD: routed[:12]}, {"tokens": [2]}]))
    # an older writer stored uint8 ids (fits 64 experts); the reader must accept it
    b.write_bytes(encode_group([{sel.ROUTED_EXPERTS_FIELD: routed[12:].astype(numpy.uint8)}]))
    got, tokens = sel.counts_from_payloads([a, b], num_experts=N, moe_layer_offset=1)
    assert tokens == 20
    expected = sel.counts_from_routed_experts(routed, num_experts=N, moe_layer_offset=1)
    assert got == expected
    with pytest.raises(sel.SelectionError, match="no sample carried"):
        empty = tmp_path / "e.json"
        empty.write_bytes(encode_group([{"tokens": [1]}]))
        sel.counts_from_payloads([empty], num_experts=N, moe_layer_offset=1)


def test_counts_from_payloads_prefers_the_buffer_reader(tmp_path, monkeypatch):
    """With S1's ``buffer_payload.rollout_routed_experts(sample)`` present, it normalizes the field."""

    from yeto.rl.engine import buffer_payload

    seen = []

    def reader(sample):
        seen.append(sorted(sample))
        return numpy.full((2, 1, TOPK), 9, dtype=numpy.uint16)

    monkeypatch.setattr(buffer_payload, "rollout_routed_experts", reader, raising=False)
    path = tmp_path / "g.json"
    path.write_bytes(encode_group([{"tokens": [1]}]))
    counts, tokens = sel.counts_from_payloads([path], num_experts=N, moe_layer_offset=1)
    assert seen == [["tokens"]] and tokens == 2 and counts == {1: [0] * 9 + [2 * TOPK] + [0] * (N - 10)}


def test_counts_from_routing_plugin_and_merge():
    plugin = {"1": [1] * N, "2": [2] * N}
    counts = sel.counts_from_routing_plugin(plugin, num_experts=N)
    assert counts == {1: [1] * N, 2: [2] * N}
    merged = sel.merge_counts(counts, {1: [3] * N})
    assert merged[1] == [4] * N and merged[2] == [2] * N
    with pytest.raises(sel.SelectionError, match="expected 64"):
        sel.counts_from_routing_plugin({"1": [1] * 3}, num_experts=N)


# -- selection ----------------------------------------------------------------


def test_top_k_is_per_layer_ties_to_lower_id():
    row = [5, 9, 9, 1, 9, 0]
    assert sel.select_top_k(row, 3) == [1, 2, 4]
    assert sel.select_top_k(row, 4) == [0, 1, 2, 4]
    with pytest.raises(sel.SelectionError):
        sel.select_top_k(row, 7)


def test_selection_from_counts_shape_and_slots():
    counts = _skewed_counts()
    s = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP,
                                        source={"prompts_sha256": "a" * 64, "tokens": 100})
    assert set(s.trainable) == {1, 2, 3}
    for layer, ids in s.trainable.items():
        assert len(ids) == K and list(ids) == sorted(set(ids))
        assert list(ids) == sel.select_top_k(counts[layer], K)  # each layer selects its own
        order = s.slot_orders[layer]
        assert sorted(order) == list(range(N))
        n_local, k_local = N // EP, K // EP
        for rank in range(EP):
            local = order[rank * n_local:(rank + 1) * n_local]
            assert set(local[:k_local]) <= set(ids)
            assert not set(local[k_local:]) & set(ids)
    assert len({tuple(v) for v in s.trainable.values()}) > 1  # layers differ
    doc = s.to_json()
    assert doc["schema"] == 1 and doc["num_experts"] == N and doc["num_trainable"] == K and doc["ep_size"] == EP
    assert doc["source"]["method"] == "routing_frequency" and doc["source"]["tokens"] == 100
    assert doc["sha256"] == canonical_sha256(doc["layers"])
    sel.validate_document(doc)
    assert sel.ExpertSelection.from_json(json.loads(json.dumps(doc))).sha256 == s.sha256


def test_hot_experts_are_spread_round_robin_over_ranks():
    counts = _skewed_counts()
    s = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP)
    for layer, order in s.slot_orders.items():
        load = counts[layer]
        n_local, k_local = N // EP, K // EP
        per_rank = [sum(load[i] for i in order[r * n_local:r * n_local + k_local]) for r in range(EP)]
        heaviest = sorted((load[i] for i in s.trainable[layer]), reverse=True)
        # no rank holds the k_local heaviest together: the bit-reversed deal spreads them
        assert max(per_rank) < sum(heaviest[:k_local])


def test_coverage_metric_per_layer_and_summary():
    counts = _skewed_counts()
    s = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP)
    cov = s.coverage()
    assert set(cov) == {"1", "2", "3", "mean", "min"}
    for layer in (1, 2, 3):
        row = counts[layer]
        assert cov[str(layer)] == pytest.approx(sum(row[i] for i in s.trainable[layer]) / sum(row))
        assert 0.0 < cov[str(layer)] <= 1.0
    assert cov["min"] <= cov["mean"] <= 1.0
    # top-16 of a heavy-tailed load covers much more than 16/64 of the tokens
    assert cov["min"] > K / N
    held_out = {1: [1] * N}
    assert s.coverage(held_out) == {"1": pytest.approx(K / N), "mean": pytest.approx(K / N), "min": pytest.approx(K / N)}


def test_sha256_equals_miles_f2_canonicalization():
    """Verbatim copy of Miles F2 ``configs/partial_experts.canonical_layers_sha256``: both sides must agree."""

    import hashlib

    def f2_canonical_layers_sha256(layers):
        canonical = {
            str(k): {"trainable": [int(x) for x in v["trainable"]], "slot_order": [int(x) for x in v["slot_order"]]}
            for k, v in layers.items()
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    doc = sel.ExpertSelection.from_counts(_skewed_counts(), model="toy", num_trainable=K, ep_size=EP).to_json()
    assert doc["sha256"] == f2_canonical_layers_sha256(doc["layers"])
    # F2 validates per EP rank: first k_local local slots trainable, the rest frozen
    n_local, k_local = N // EP, K // EP
    for layer in doc["layers"].values():
        trainable = set(layer["trainable"])
        for rank in range(EP):
            local = layer["slot_order"][rank * n_local:(rank + 1) * n_local]
            assert [e in trainable for e in local] == [True] * k_local + [False] * (n_local - k_local)


def test_layer_set_check():
    s = sel.ExpertSelection.from_counts(_skewed_counts(), model="toy", num_trainable=K, ep_size=EP)
    s.check_layers([1, 2, 3])
    with pytest.raises(sel.SelectionError, match="model MoE layers"):
        s.check_layers([1, 2, 3, 4])
    with pytest.raises(sel.SelectionError, match="model MoE layers"):
        s.check_layers([0, 1, 2])


# -- EP change ----------------------------------------------------------------


@pytest.mark.parametrize("new_ep", [1, 2, 8, 16])
def test_ep_change_keeps_trainable_and_recomputes_slot_order(new_ep):
    s = sel.ExpertSelection.from_counts(_skewed_counts(), model="toy", num_trainable=K, ep_size=EP)
    t = s.with_ep_size(new_ep)
    assert t.trainable == s.trainable
    assert t.trainable_sha256 == s.trainable_sha256
    assert t.ep_size == new_ep
    sel.validate_document(t.to_json())
    if new_ep != EP:
        assert t.sha256 != s.sha256  # slot_order is part of the interface sha
    # trainable index t -> id is EP independent: rank chunks concatenate to the same order
    def trainable_index_order(x):
        n_local, k_local = N // x.ep_size, K // x.ep_size
        return {layer: [o[r * n_local + j] for r in range(x.ep_size) for j in range(k_local)]
                for layer, o in x.slot_orders.items()}
    assert trainable_index_order(t) == trainable_index_order(s)
    with pytest.raises(sel.SelectionError, match="must divide"):
        s.with_ep_size(3)


def test_round_trip_through_file_and_sha_check(tmp_path):
    s = sel.ExpertSelection.from_counts(_skewed_counts(), model="toy", num_trainable=K, ep_size=EP)
    path = tmp_path / "expert_selection.json"
    sha = s.write(path)
    loaded = sel.ExpertSelection.load_file(path, expected_sha256=sha)
    assert loaded.to_json() == s.to_json()
    with pytest.raises(sel.SelectionError, match="SHA256 mismatch"):
        sel.ExpertSelection.load_file(path, expected_sha256="0" * 64)


# -- validation failures ------------------------------------------------------


def _doc():
    return sel.ExpertSelection.from_counts(_skewed_counts(), model="toy", num_trainable=K, ep_size=EP).to_json()


def _mutated(mutate):
    doc = _doc()
    mutate(doc)
    return doc


def _swap_trainable_into_frozen(doc):
    layer = doc["layers"]["1"]
    order = layer["slot_order"]
    n_local, k_local = N // EP, K // EP
    order[0], order[k_local] = order[k_local], order[0]  # trainable slot <-> frozen slot on rank 0
    doc["sha256"] = canonical_sha256(doc["layers"])


def _wrong_length(doc):
    doc["layers"]["1"]["trainable"] = doc["layers"]["1"]["trainable"][:-1]
    doc["sha256"] = canonical_sha256(doc["layers"])


def _not_permutation(doc):
    doc["layers"]["2"]["slot_order"][0] = doc["layers"]["2"]["slot_order"][1]
    doc["sha256"] = canonical_sha256(doc["layers"])


def _missing_layer_from_slots(doc):
    layer = doc["layers"]["3"]
    ids = layer["trainable"]
    # keep the trainable list but place a frozen expert where a trainable one belongs on rank 1
    order = layer["slot_order"]
    n_local, k_local = N // EP, K // EP
    frozen = next(i for i in range(N) if i not in ids)
    pos_frozen = order.index(frozen)
    order[n_local], order[pos_frozen] = order[pos_frozen], order[n_local]
    doc["sha256"] = canonical_sha256(doc["layers"])


@pytest.mark.parametrize(("mutate", "match"), [
    (lambda d: d.update(schema=2), "schema"),
    (lambda d: d.pop("sha256"), "lacks 'sha256'"),
    (lambda d: d.update(sha256="0" * 64), "does not match"),
    (lambda d: d.update(sha256="zz"), "lowercase SHA256"),
    (lambda d: d.update(ep_size=3), "must divide"),
    (lambda d: d.update(num_trainable=K + 1), "must divide|unique ascending"),
    (_wrong_length, f"must be {K} unique ascending"),
    (lambda d: d["layers"]["1"]["trainable"].__setitem__(0, -1), "unique ascending|outside"),
    (_not_permutation, "permutation"),
    (_swap_trainable_into_frozen, "frozen expert in a trainable slot|trainable expert in a frozen slot"),
    (_missing_layer_from_slots, "frozen expert in a trainable slot|trainable expert in a frozen slot|cover"),
    (lambda d: d.update(load={"1": [0] * N}), "cover exactly"),
    (lambda d: d["load"].__setitem__("1", [0] * 3), "expected 64"),
    (lambda d: d.update(trainable_sha256="1" * 64), "trainable_sha256"),
    (lambda d: d["layers"].__setitem__("x", d["layers"]["1"]), "not an integer"),
])
def test_validation_failures(mutate, match):
    with pytest.raises(sel.SelectionError, match=match):
        sel.validate_document(_mutated(mutate))


# -- synthetic modes ----------------------------------------------------------


def test_uniform_and_random_modes():
    layers = range(1, LAYERS + 1)
    u = sel.ExpertSelection.synthetic("uniform", model="toy", num_experts=N, num_trainable=K, ep_size=EP, layers=layers)
    assert all(ids == tuple(range(0, N, N // K)) for ids in u.trainable.values())
    assert u.source == {"method": "uniform", "tokens": 0} and u.load is None
    sel.validate_document(u.to_json())
    r = sel.ExpertSelection.synthetic("random", model="toy", num_experts=N, num_trainable=K, ep_size=EP, layers=layers, seed=3)
    r2 = sel.ExpertSelection.synthetic("random", model="toy", num_experts=N, num_trainable=K, ep_size=EP, layers=layers, seed=3)
    assert r.to_json() == r2.to_json()  # deterministic in the seed
    assert len({ids for ids in r.trainable.values()}) == LAYERS  # independent per layer
    assert all(len(ids) == K for ids in r.trainable.values())
    sel.validate_document(r.to_json())
    with pytest.raises(sel.SelectionError, match="coverage needs"):
        r.coverage()
    with pytest.raises(sel.SelectionError, match="mode must be"):
        sel.ExpertSelection.synthetic("hot", model="toy", num_experts=N, num_trainable=K, ep_size=EP, layers=layers)


def test_k31_shape_uniform_selection_is_valid():
    s = sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=896, num_trainable=128,
                                      ep_size=8, layers=range(1, 93))
    doc = s.to_json()
    sel.validate_document(doc)
    assert len(doc["layers"]) == 92 and all(len(v["trainable"]) == 128 for v in doc["layers"].values())
    assert all(len(v["slot_order"]) == 896 for v in doc["layers"].values())


# -- CLI ----------------------------------------------------------------------


def test_cli_payload_to_file_and_validate(tmp_path, capsys):
    counts = _skewed_counts()
    routed = _routed_from_counts(counts, tokens=30)
    payload = tmp_path / "g.json"
    payload.write_bytes(encode_group([{sel.ROUTED_EXPERTS_FIELD: routed}]))
    out = tmp_path / "expert_selection.json"
    rc = sel.main(["--model", "toy", "--num-experts", str(N), "--num-trainable", str(K), "--ep-size", str(EP),
                   "--moe-layer-offset", "1", "--payload", str(payload), "--output", str(out),
                   "--prompts-sha256", "b" * 64])
    assert rc == 0
    summary = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    doc = json.loads(out.read_text())
    assert doc["source"] == {"method": "routing_frequency", "prompts_sha256": "b" * 64, "tokens": 30}
    assert summary["sha256"] == doc["sha256"] and 0 < summary["coverage_min"] <= summary["coverage_mean"] <= 1
    assert sel.main(["--validate", str(out)]) == 0
    # re-derive for EP 2
    out2 = tmp_path / "ep2.json"
    assert sel.main(["--input", str(out), "--ep-size", "2", "--output", str(out2)]) == 0
    doc2 = json.loads(out2.read_text())
    assert doc2["ep_size"] == 2 and doc2["trainable_sha256"] == doc["trainable_sha256"]
    # a corrupted file fails validation with exit 1
    doc["sha256"] = "0" * 64
    out.write_text(json.dumps(doc))
    assert sel.main(["--validate", str(out)]) == 1
    assert "invalid" in capsys.readouterr().err


def test_cli_counts_and_synthetic(tmp_path):
    counts_path = tmp_path / "counts.json"
    counts_path.write_text(json.dumps({str(k): v for k, v in _skewed_counts().items()}))
    out = tmp_path / "c.json"
    assert sel.main(["--model", "toy", "--num-experts", str(N), "--num-trainable", str(K), "--ep-size", str(EP),
                     "--counts", str(counts_path), "--output", str(out), "--no-load"]) == 0
    doc = json.loads(out.read_text())
    assert "load" not in doc and doc["source"]["counter"] == "routing_plugin"
    smoke = tmp_path / "u.json"
    assert sel.main(["--model", "toy", "--mode", "uniform", "--num-moe-layers", "3", "--num-experts", str(N),
                     "--num-trainable", str(K), "--ep-size", str(EP), "--output", str(smoke)]) == 0
    assert sel.ExpertSelection.load_file(smoke).source["method"] == "uniform"


def test_module_is_runnable_as_main(tmp_path):
    import subprocess

    out = tmp_path / "u.json"
    result = subprocess.run(
        [sys.executable, "-m", "yeto.rl.adapters.miles.models.partial_experts.selection", "--model", "toy",
         "--mode", "uniform", "--num-moe-layers", "2", "--num-experts", "32", "--num-trainable", "8",
         "--ep-size", "2", "--output", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stderr.strip().splitlines()[-1])["layers"] == 2
    sel.validate_document(json.loads(out.read_text()))


def test_cli_refuses_wrong_layer_set_for_registered_model(tmp_path):
    out = tmp_path / "k.json"
    with pytest.raises(sel.SelectionError, match="model MoE layers"):
        sel.main(["--model", "kimi_k31", "--mode", "uniform", "--num-moe-layers", "3",
                  "--num-experts", "896", "--num-trainable", "128", "--ep-size", "8", "--output", str(out)])
    assert sel.main(["--model", "kimi_k31", "--mode", "uniform", "--num-moe-layers", "3", "--skip-layer-check",
                     "--num-experts", "896", "--num-trainable", "128", "--ep-size", "8", "--output", str(out)]) == 0
    assert sel.main(["--model", "kimi_k31", "--mode", "random", "--seed", "1",
                     "--num-experts", "896", "--num-trainable", "128", "--ep-size", "8", "--output", str(out)]) == 0


# -- lpt placement --------------------------------------------------------------


def _assert_valid_slots(s):
    n_local, k_local = N // s.ep_size, K // s.ep_size
    for layer, order in s.slot_orders.items():
        assert sorted(order) == list(range(N))
        for rank in range(s.ep_size):
            local = order[rank * n_local:(rank + 1) * n_local]
            assert set(local[:k_local]) <= set(s.trainable[layer])
            assert not set(local[k_local:]) & set(s.trainable[layer])
            assert list(local[k_local:]) == sorted(local[k_local:])
    sel.validate_document(s.to_json())


def test_lpt_placement_keeps_trainable_and_balances_rank_load():
    counts = _skewed_counts()
    bitrev = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP)
    lpt = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP, placement="lpt")
    _assert_valid_slots(lpt)
    assert lpt.trainable == bitrev.trainable and lpt.trainable_sha256 == bitrev.trainable_sha256
    assert lpt.sha256 != bitrev.sha256
    assert lpt.source["placement"] == "lpt" and "placement" not in bitrev.source
    assert bitrev.placement == "bitrev" and lpt.placement == "lpt"
    for layer in counts:
        before = sel.rank_loads(bitrev.slot_orders[layer], counts[layer], EP)
        after = sel.rank_loads(lpt.slot_orders[layer], counts[layer], EP)
        assert sum(after) == sum(before)
        assert max(after) <= max(before)
        # LPT bound: the max rank exceeds the mean by at most one expert's load
        assert max(after) - sum(after) / EP <= max(counts[layer]) + 1e-9
    assert lpt.ep_imbalance()["mean"] < bitrev.ep_imbalance()["mean"]
    # deterministic
    again = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP, placement="lpt")
    assert again.sha256 == lpt.sha256


def test_lpt_splits_a_rank_that_bitrev_overloads():
    # a few hot frozen experts that snake order puts together on top of a hot trainable rank
    load = [1] * N
    trainable = list(range(K))
    for i in trainable:
        load[i] = 50
    load[K] = load[K + 1] = 400  # two hot frozen experts
    counts = {1: load}
    lpt = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP, placement="lpt")
    loads = sel.rank_loads(lpt.slot_orders[1], load, EP)
    owners = {lpt.slot_orders[1].index(K) // (N // EP), lpt.slot_orders[1].index(K + 1) // (N // EP)}
    assert len(owners) == 2
    assert max(loads) - min(loads) <= 400


def test_lpt_with_ep_size_keeps_the_trainable_index_order_and_switching_re_deals():
    counts = _skewed_counts(3)
    lpt = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP, placement="lpt")
    for new_ep in (2, EP):
        t = lpt.with_ep_size(new_ep)
        _assert_valid_slots(t)
        assert t.placement == "lpt" and t.trainable_sha256 == lpt.trainable_sha256
        assert {layer: t.trainable_index_order(layer) for layer in counts} == \
            {layer: lpt.trainable_index_order(layer) for layer in counts}
    assert lpt.with_ep_size(EP).sha256 == lpt.sha256
    # bitrev -> lpt on an existing document: same trainable set, new slots
    bitrev = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP)
    converted = bitrev.with_ep_size(EP, placement="lpt")
    assert converted.sha256 == lpt.sha256
    back = converted.with_ep_size(EP, placement="bitrev")
    assert back.sha256 == bitrev.sha256 and "placement" not in back.source
    no_load = sel.ExpertSelection.from_counts(counts, model="toy", num_trainable=K, ep_size=EP, keep_load=False)
    with pytest.raises(sel.SelectionError, match="needs the selection's load"):
        no_load.with_ep_size(EP, placement="lpt")
    with pytest.raises(sel.SelectionError, match="placement must be one of"):
        bitrev.with_ep_size(EP, placement="snake")


def test_lpt_needs_load_and_rejects_a_foreign_hot_order():
    with pytest.raises(sel.SelectionError, match="needs the per-expert load"):
        sel.slot_order(list(range(K)), num_experts=N, ep_size=EP, load=None, placement="lpt")
    with pytest.raises(sel.SelectionError, match="not an order of the trainable set"):
        sel.slot_order(list(range(K)), num_experts=N, ep_size=EP, load=[1] * N, placement="lpt",
                       hot_order=list(range(1, K + 1)))
    with pytest.raises(sel.SelectionError, match="only for placement 'lpt'"):
        sel.slot_order(list(range(K)), num_experts=N, ep_size=EP, load=[1] * N, hot_order=list(range(K)))


def test_cli_placement_lpt_from_counts_and_input(tmp_path, capsys):
    counts_path = tmp_path / "counts.json"
    counts_path.write_text(json.dumps({str(k): v for k, v in _skewed_counts().items()}))
    base = ["--model", "toy", "--num-experts", str(N), "--num-trainable", str(K), "--ep-size", str(EP)]
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    assert sel.main(base + ["--counts", str(counts_path), "--output", str(a)]) == 0
    assert sel.main(base + ["--counts", str(counts_path), "--placement", "lpt", "--output", str(b)]) == 0
    assert sel.main(base + ["--input", str(a), "--placement", "lpt", "--output", str(c)]) == 0
    summaries = [json.loads(line) for line in capsys.readouterr().err.strip().splitlines()]
    da, db, dc = (json.loads(p.read_text()) for p in (a, b, c))
    assert db["sha256"] == dc["sha256"] != da["sha256"]
    assert db["trainable_sha256"] == da["trainable_sha256"]
    assert db["source"]["placement"] == "lpt" and "placement" not in da["source"]
    assert summaries[1]["ep_imbalance_mean"] < summaries[0]["ep_imbalance_mean"]
    # --input without --placement keeps the document's placement
    d = tmp_path / "d.json"
    assert sel.main(base + ["--input", str(b), "--output", str(d)]) == 0
    assert json.loads(d.read_text())["sha256"] == db["sha256"]
