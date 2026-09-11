"""Focused CPU tests for the one-off immutable dataset/runtime bridge."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from monitoring import attest_masked_native_gap_v4_dataset_runtime_compatibility_20260911 as bridge
from monitoring import prepare_cot_masked_native_gap_v4_n3_20260910 as prepare_v4
from monitoring import launch_cot_masked_native_gap_v4_n3_20260910 as launch_v4

ROOT=Path(__file__).resolve().parents[2]
CONVERSION_DEPENDENCIES=bridge._declared_conversion_dependencies(ROOT)

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _freeze(root):
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _thaw(root):
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)
    root.chmod(0o755)


def _code_manifest(root, contents):
    records=[]
    for name,value in sorted(contents.items()):
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value)
        records.append({"path":name,"bytes":len(value),
                        "sha256":hashlib.sha256(value).hexdigest()})
    raw=(json.dumps({"schema":bridge.CODE_SCHEMA,"files":records},
                    sort_keys=True,separators=(",",":"))+"\n").encode()
    (root/"code-manifest.json").write_bytes(raw)
    _freeze(root)
    return hashlib.sha256(raw).hexdigest(),{row["path"]:row for row in records}


@pytest.fixture
def compatibility_fixture(tmp_path,monkeypatch):
    monkeypatch.setenv("YETA_TRAINING_IMAGE",bridge.IMAGE)
    old=tmp_path/"old-code";new=tmp_path/"new-code";old.mkdir();new.mkdir()
    common={name:(ROOT/name).read_bytes() for name in CONVERSION_DEPENDENCIES}
    old_contents={**common,**{name:("old:"+name).encode()
                             for name in bridge.EXPECTED_MODIFIED}}
    new_contents={**common,**{name:("new:"+name).encode()
                             for name in bridge.EXPECTED_MODIFIED}}
    new_contents[bridge.SELF]=Path(bridge.__file__).read_bytes()
    old_sha,old_records=_code_manifest(old,old_contents)
    new_sha,new_records=_code_manifest(new,new_contents)
    probe_dir=tmp_path/"cp-probe";nested_dir=probe_dir/"probe-output";nested_dir.mkdir(parents=True)
    nested={"schema":bridge.CP_NESTED_SCHEMA,"status":"passed",
        "runtime_image":bridge.IMAGE,"world_size":8,"cp_size":8,"backend":"nccl",
        "patched_guard_sha256":new_records[
            "training/qwen38_native_gap_v3/masked_train.py"]["sha256"],
        "probe_sha256":"p"*64,
        "primary_contract":"full-input-then-in-forward-round-robin",
        "aux_contract":"deferred-context-entry-round-robin",
        "assertions":{"deferred_labels":True,"in_forward_primary":True},
        "rank_records":[{"rank":rank} for rank in range(8)]}
    nested_path=nested_dir/"receipt.json";nested_path.write_text(json.dumps(nested))
    nested_sha=_sha(nested_path)
    host={"schema":bridge.CP_PROBE_SCHEMA,"status":"passed",
        "runtime_image_id":bridge.IMAGE,"world_size":8,"backend":"nccl",
        "docker_exit_code":0,"dataset_accessed":False,"model_loaded":False,
        "gpu_compute_processes_after":[],"runner_inactive_after":True,
        "transient_container_present_after_run":False,
        "patched_masked_train_sha256":nested["patched_guard_sha256"],
        "probe_receipt_path":str(nested_path),"probe_receipt_sha256":nested_sha,
        "probe_source_sha256":nested["probe_sha256"],"probe_receipt":nested}
    probe=probe_dir/"host-receipt.json";probe.write_text(json.dumps(host))
    probe_sha=_sha(probe);_freeze(probe_dir)
    monkeypatch.setattr(bridge,"CP_PROBE_PATH",probe)
    monkeypatch.setattr(bridge,"CP_PROBE_SHA256",probe_sha)
    monkeypatch.setattr(bridge,"CP_NESTED_SHA256",nested_sha)
    dataset=tmp_path/"cot-masked-native-gap-v4-fixture";dataset.mkdir()
    dependencies={name:old_records[name]["sha256"]
                  for name in CONVERSION_DEPENDENCIES}
    manifest={"preparation_sha256":dependencies[
        "training/qwen38_native_gap_v3/prepare_masked_full.py"],
        "conversion_dependency_sha256":dependencies}
    (dataset/"manifest.json").write_text(json.dumps(manifest))
    (dataset/"index.json").write_text("{}")
    (dataset/"COMPLETE.json").write_text("{}")
    inventory=[{"path":path.name,"bytes":path.stat().st_size,"sha256":_sha(path)}
               for path in sorted(dataset.iterdir())]
    receipt=dataset.with_name(dataset.name+".build-receipt.json")
    receipt.write_text(json.dumps({
        "schema":bridge.BUILD_SCHEMA,"status":"complete","mode":"final",
        "output":str(dataset),"runtime_image":bridge.IMAGE,
        "code_manifest_sha256":old_sha,"immutable_mode":"files0444_dirs0555",
        "atomic_direct_dataset_publication":True,"tree_inventory":inventory,
        "files":len(inventory),"bytes":sum(row["bytes"] for row in inventory),
        "tree_sha256":bridge._canonical_sha(inventory)}))
    receipt.chmod(0o444);_freeze(dataset)
    output=tmp_path/"compatibility.json"
    values={"dataset":dataset,"build_receipt":receipt,
        "build_receipt_sha256":_sha(receipt),"old_code_root":old,
        "old_code_manifest_sha256":old_sha,"new_code_root":new,
        "new_code_manifest_sha256":new_sha,"cp_guard_probe":probe,
        "cp_guard_probe_sha256":probe_sha,"output":output}
    try:
        yield values
    finally:
        for root in (old,new,dataset,probe_dir):_thaw(root)
        if output.exists():output.chmod(0o644)


def test_runtime_compatibility_attests_and_reverifies_exact_bytes(
        compatibility_fixture,monkeypatch):
    values=compatibility_fixture
    result=bridge.attest(**values)
    assert result["status"]=="passed"
    assert result["added_runtime_files"]==sorted(bridge.EXPECTED_ADDED)
    assert result["modified_runtime_files"]==sorted(bridge.EXPECTED_MODIFIED)
    assert result["removed_runtime_files"]==[]
    assert result["conversion_dependency_count"]==len(CONVERSION_DEPENDENCIES)
    assert values["output"].stat().st_mode&0o222==0
    # CPU attestation runs in the exact image; the host launcher must be able
    # to independently rehash the signed result without pretending to be it.
    monkeypatch.delenv("YETA_TRAINING_IMAGE")
    record=bridge.verify_receipt(receipt_path=values["output"],
        receipt_sha256=_sha(values["output"]),dataset=values["dataset"],
        build_receipt=values["build_receipt"],
        build_receipt_sha256=values["build_receipt_sha256"],
        new_code_root=values["new_code_root"],
        new_code_manifest_sha256=values["new_code_manifest_sha256"],
        cp_guard_probe=values["cp_guard_probe"],
        cp_guard_probe_sha256=values["cp_guard_probe_sha256"])
    assert record["dataset_tree_sha256"]==result["dataset_tree_sha256"]


def test_runtime_compatibility_rejects_dataset_mutation(compatibility_fixture):
    values=compatibility_fixture
    path=values["dataset"]/"index.json";path.chmod(0o644);path.write_text("[]");path.chmod(0o444)
    with pytest.raises(ValueError,match="Dataset member differs"):
        bridge.attest(**values)


def test_runtime_compatibility_rejects_wrong_cp_probe_identity(compatibility_fixture):
    values=compatibility_fixture;values["cp_guard_probe_sha256"]="0"*64
    with pytest.raises(ValueError,match="eight-rank CP guard probe"):
        bridge.attest(**values)


def test_runtime_compatibility_rejects_conversion_dependency_change(compatibility_fixture):
    values=compatibility_fixture;root=values["new_code_root"]
    _thaw(root)
    name=CONVERSION_DEPENDENCIES[0];path=root/name
    path.write_text("changed conversion code")
    manifest=json.loads((root/"code-manifest.json").read_text())
    for row in manifest["files"]:
        if row["path"]==name:
            row.update(bytes=path.stat().st_size,sha256=_sha(path))
    (root/"code-manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
    values["new_code_manifest_sha256"]=_sha(root/"code-manifest.json");_freeze(root)
    with pytest.raises(ValueError,match="conversion dependency changed"):
        bridge.attest(**values)


def test_runtime_compatibility_rejects_unreviewed_runtime_change(compatibility_fixture):
    values=compatibility_fixture;root=values["new_code_root"]
    _thaw(root);path=root/"monitoring/unreviewed.py";path.write_text("unreviewed = True\n")
    manifest=json.loads((root/"code-manifest.json").read_text())
    manifest["files"].append({"path":"monitoring/unreviewed.py","bytes":path.stat().st_size,
                              "sha256":_sha(path)})
    manifest["files"].sort(key=lambda row:row["path"])
    (root/"code-manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n")
    values["new_code_manifest_sha256"]=_sha(root/"code-manifest.json");_freeze(root)
    with pytest.raises(ValueError,match="exact reviewed runtime-only patch"):
        bridge.attest(**values)


def test_runtime_compatibility_receipt_rehash_rejects_post_attestation_change(
        compatibility_fixture):
    values=compatibility_fixture;bridge.attest(**values)
    root=values["new_code_root"]
    _thaw(root);path=root/next(iter(bridge.EXPECTED_MODIFIED));path.write_text("changed later")
    _freeze(root)
    with pytest.raises(ValueError,match="code member changed"):
        bridge.verify_receipt(receipt_path=values["output"],
            receipt_sha256=_sha(values["output"]),dataset=values["dataset"],
            build_receipt=values["build_receipt"],
            build_receipt_sha256=values["build_receipt_sha256"],
            new_code_root=root,new_code_manifest_sha256=values["new_code_manifest_sha256"],
            cp_guard_probe=values["cp_guard_probe"],
            cp_guard_probe_sha256=values["cp_guard_probe_sha256"])


def test_runtime_compatibility_receipt_tamper_is_rejected(compatibility_fixture):
    values=compatibility_fixture;bridge.attest(**values)
    values["output"].chmod(0o600)
    record=json.loads(values["output"].read_text());record["exact_runtime_only_delta"]=False
    values["output"].write_text(json.dumps(record));values["output"].chmod(0o400)
    with pytest.raises(ValueError,match="receipt changed"):
        bridge.verify_receipt(receipt_path=values["output"],receipt_sha256="0"*64,
            dataset=values["dataset"],build_receipt=values["build_receipt"],
            build_receipt_sha256=values["build_receipt_sha256"],
            new_code_root=values["new_code_root"],
            new_code_manifest_sha256=values["new_code_manifest_sha256"],
            cp_guard_probe=values["cp_guard_probe"],
            cp_guard_probe_sha256=values["cp_guard_probe_sha256"])


def test_cpu_preflight_requires_bridge_only_for_code_mismatch(monkeypatch,tmp_path):
    publication={"code_manifest_sha256":"a"*64,"tree_sha256":"t"*64}
    common=dict(dataset=tmp_path/"data",build_receipt=tmp_path/"receipt",
                build_receipt_sha256="b"*64,publication=publication,
                code=tmp_path/"code",code_manifest_sha256="c"*64,
                cp_guard_probe=None,cp_guard_probe_sha256=None)
    with pytest.raises(ValueError,match="requires a compatibility receipt"):
        prepare_v4._runtime_compatibility(**common,receipt=None,receipt_sha256=None)
    expected={"old_code_manifest_sha256":"a"*64,"dataset_tree_sha256":"t"*64}
    monkeypatch.setattr(prepare_v4.compatibility,"verify_receipt",lambda **_:expected)
    with_probe={**common,"cp_guard_probe":tmp_path/"probe",
                "cp_guard_probe_sha256":"p"*64}
    assert prepare_v4._runtime_compatibility(**with_probe,receipt=tmp_path/"compat",
        receipt_sha256="d"*64)==expected
    same={**common,"code_manifest_sha256":"a"*64}
    assert prepare_v4._runtime_compatibility(**same,receipt=None,receipt_sha256=None) is None
    with pytest.raises(ValueError,match="identical code"):
        prepare_v4._runtime_compatibility(**same,receipt=tmp_path/"compat",
                                          receipt_sha256="d"*64)


def test_launcher_reverifies_mismatched_build_runtime_bridge(monkeypatch,tmp_path):
    dataset=tmp_path/"dataset";dataset.mkdir();code=tmp_path/"code";code.mkdir()
    config_dir=tmp_path/"config";config_dir.mkdir()
    build_receipt=tmp_path/"dataset.build-receipt.json";build_receipt.write_text("{}")
    bridge_path=config_dir/"dataset-runtime-compatibility.json";bridge_path.write_text("{}")
    bridge_path.chmod(0o444)
    contract={"schema":bridge.SCHEMA,"receipt_path":str(bridge_path),
        "receipt_sha256":"d"*64,"dataset_tree_sha256":"t"*64,
        "dataset_build_receipt_sha256":"r"*64,
        "build_code_manifest_sha256":"a"*64,
        "runtime_code_manifest_sha256":"b"*64,
        "conversion_dependency_map_sha256":"m"*64,
        "cp_guard_probe_path":str(tmp_path/"cp-probe.json"),
        "cp_guard_probe_sha256":"p"*64,
        "cp_guard_nested_receipt_sha256":"n"*64}
    cpu={"code_manifest_sha256":"b"*64,"build_receipt_sha256":"r"*64,
         "runtime_compatibility":contract}
    config={"dataset_contract":{"runtime_compatibility":contract}}
    publication={"code_manifest_sha256":"a"*64,"tree_sha256":"t"*64}
    plan={"runtime_compatibility_sha256":"d"*64,
          "cp_guard_probe_path":contract["cp_guard_probe_path"],
          "cp_guard_probe_sha256":contract["cp_guard_probe_sha256"]}
    expected={"conversion_dependency_map_sha256":"m"*64}
    calls=[]
    monkeypatch.setattr(launch_v4.compatibility,"verify_receipt",
                        lambda **kwargs:(calls.append(kwargs) or expected))
    assert launch_v4._verify_runtime_compatibility(plan=plan,cpu=cpu,config=config,
        publication=publication,dataset=dataset,build_receipt=build_receipt,code=code,
        config_dir=config_dir)==expected
    assert calls and calls[0]["receipt_path"]==bridge_path
    plan["runtime_compatibility_sha256"]="e"*64
    with pytest.raises(ValueError,match="binding changed"):
        launch_v4._verify_runtime_compatibility(plan=plan,cpu=cpu,config=config,
            publication=publication,dataset=dataset,build_receipt=build_receipt,
            code=code,config_dir=config_dir)
