"""rl-identity-tiering L9.1/L9.2: semantic hash, dual write, tolerant PluginRef."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.provenance import CODE_PACK_MODE_SOURCE_TREE
from yeto.rl.algos import mismatch_correction as mc
from yeto.rl.engine.algorithm import (
    AlgorithmSpec,
    AlgorithmSpecError,
    PluginRef,
    plugin_source_sha256,
)
from yeto.rl.engine.driver import IslandDriver

OBSERVE = mc.OBSERVE_PATH
STALE = "1" * 64


def _custom(function: dict, **correction) -> AlgorithmSpec:
    return AlgorithmSpec(correction={"method": "custom", "mismatch_metrics": True,
                                     "function": function, **correction})


def test_plugin_source_change_keeps_semantic_hash():
    current = _custom({"path": OBSERVE, "sha256": plugin_source_sha256(OBSERVE)})
    changed = _custom({"path": OBSERVE, "sha256": STALE})
    unpinned = _custom({"path": OBSERVE})
    assert current.sha256() != changed.sha256()  # the full spec hash still sees the source
    assert current.semantic_sha256() == changed.semantic_sha256() == unpinned.semantic_sha256()
    # top-level plugins list: the path stays, the hash goes
    listed = AlgorithmSpec(plugins=[{"path": OBSERVE, "sha256": STALE}])
    assert listed.semantic_dict()["spec"]["plugins"] == [{"path": OBSERVE}]


def test_configuration_change_changes_semantic_hash():
    base = _custom({"path": OBSERVE})
    assert base.semantic_sha256() != AlgorithmSpec().semantic_sha256()
    assert AlgorithmSpec(kl_coef=0.01).semantic_sha256() != AlgorithmSpec().semantic_sha256()
    other_path = _custom({"path": mc.ICEPOP_PATH})
    assert other_path.semantic_sha256() != base.semantic_sha256()


def test_plugin_ref_without_sha256_keeps_old_canonical_forms():
    ref = PluginRef.from_dict({"path": OBSERVE})
    assert ref.sha256 is None and ref.to_dict() == {"path": OBSERVE}
    assert ref.verify(import_callable=True) is None
    pinned = PluginRef.from_path(OBSERVE)
    assert pinned.to_dict() == {"path": OBSERVE, "sha256": plugin_source_sha256(OBSERVE)}
    with pytest.raises(AlgorithmSpecError, match="optional sha256"):
        PluginRef.from_dict({"sha256": STALE})
    with pytest.raises(AlgorithmSpecError, match="optional sha256"):
        PluginRef.from_dict({"path": OBSERVE, "sha256": STALE, "extra": 1})
    with pytest.raises(AlgorithmSpecError, match="64 hex"):
        PluginRef.from_dict({"path": OBSERVE, "sha256": "abc"})


def test_old_spec_file_with_stale_plugin_hash_loads_and_warns(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(_custom({"path": OBSERVE, "sha256": STALE}).to_dict()))
    spec = AlgorithmSpec.from_json_file(str(path))
    warnings = spec.verify_plugins()
    assert len(warnings) == 1
    assert f"spec {STALE}, runtime {plugin_source_sha256(OBSERVE)}" in warnings[0]


def test_driver_start_code_identity():
    spec = _custom({"path": OBSERVE})
    without = SimpleNamespace(algorithm=spec, code_pack_digest=None)
    assert IslandDriver._code_identity(without) == {}  # rl_driver_start unchanged
    with_digest = SimpleNamespace(algorithm=spec, code_pack_digest="a" * 64)
    assert IslandDriver._code_identity(with_digest) == {
        "semantic_sha256": spec.semantic_sha256(),
        "code_pack_digest": "a" * 64,
        "code_pack_mode": CODE_PACK_MODE_SOURCE_TREE,
    }
