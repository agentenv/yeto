"""G3: candidate edges are filtered by kind; trainer-dp only after 4.6/4.7 (A4)."""
import dataclasses
from types import SimpleNamespace as NS

from yeto.rl.elastic_benchmark.capabilities import Attestation
from yeto.rl.engine.recommend import candidate_edges_from_attestation as cands

CFG = {n: NS(rollout=r, rollout_engine_gpus=1) for n, r in (("A", 1), ("B", 2), ("C", 2), ("D", 2))}
RO, DP, RT, SS = ("A", "B", "rollout-only"), ("A", "C", "trainer-dp"), ("A", "D", "role-transfer"), \
    ("B", "A", "standby-scale")
SHA = "a" * 64


def _att(*, algos=()):
    return dataclasses.replace(Attestation.none(), certified_edges=frozenset({RO, DP, RT, SS}),
                               auto_controller=True, edge_algorithms=tuple(algos))


def _targets(edges):
    return sorted((e.source, e.target) for e in edges)


def test_only_rollout_only_without_trainer_certificate():
    assert _targets(cands(_att(), CFG)) == [("A", "B")]


def test_trainer_dp_after_algorithm_certificate():
    att = _att(algos=[(DP, frozenset({SHA}))])
    assert _targets(cands(att, CFG)) == [("A", "B"), ("A", "C")]
    assert _targets(cands(att, CFG, algorithm_sha256=SHA)) == [("A", "B"), ("A", "C")]
    assert _targets(cands(att, CFG, algorithm_sha256="b" * 64)) == [("A", "B")]


def test_kinds_parameter_restricts_to_rollout_edges():
    att = _att(algos=[(DP, frozenset({SHA}))])
    assert _targets(cands(att, CFG, kinds=("rollout-only",))) == [("A", "B")]
