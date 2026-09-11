import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from cot_filler.core import canonical, digest
from cot_filler.corpus_worker import gap_at
from .test_render import fixture
from .test_replay import rollout_job, ASSETS
from .prepare_full import entries_for_source, load_freeze, prepare, split_for_group


def freeze_fixture(tmp_path, *, missing_group=False, low_quality=False, invalid_zero_tool=False):
    source, entries, config = fixture()
    source["metadata"] = {"selection": {"capture_id": source["trace_id"], "group_id": None if missing_group else "original-group",
        "quality_score": 3 if low_quality else 4, "overall_confidence_score": 3}}
    gap = gap_at(source, 1)
    entry = entries[0];entry["gap"] = gap
    for block in (entry["candidate"], entry["generation_receipt"]):
        block["gap_id"] = gap["id"]
        meta = json.loads(block["data"]);meta["prompt_hash"] = gap["prompt_hash"];block["data"] = canonical(meta)
    entry["candidate"].update({k: gap[k] for k in ("prefix_hash", "lookahead_hash", "prompt_hash")})
    zero = json.loads(canonical(source));zero["trace_id"] = "zero-filled-original"
    zero["metadata"]["selection"]["capture_id"] = zero["trace_id"]
    if invalid_zero_tool:
        zero["events"].append({"event_id": "bad-tool", "kind": "tool_call", "role": "assistant", "content": "",
            "name": "read", "call_id": "bad-call", "data": {"block": {"type": "function_call", "name": "read",
            "call_id": "bad-call", "arguments": {"unsafe parameter key": "value"}}}})
    sources = [source, zero]
    source_path = tmp_path / "sources.jsonl"
    raw_rows = [canonical(s).encode() + b"\n" for s in sources]
    source_path.write_bytes(b"".join(raw_rows))
    dbpath = tmp_path / "snapshot.sqlite3"
    db = sqlite3.connect(dbpath)
    db.executescript("""
        CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE sources(id INTEGER PRIMARY KEY,trace_id TEXT,offset INTEGER,length INTEGER,row_sha256 TEXT,source_digest TEXT);
        CREATE TABLE gaps(ordinal INTEGER PRIMARY KEY,id TEXT,source_id INTEGER,event_index INTEGER,event_id TEXT,state TEXT,invocation TEXT,updated TEXT);
        CREATE TABLE candidates(gap_id TEXT,text TEXT,data TEXT,prefix_hash TEXT,lookahead_hash TEXT,prompt_hash TEXT,created TEXT);
        CREATE TABLE responses(id INTEGER PRIMARY KEY,gap_id TEXT,stage TEXT,text TEXT,data TEXT,created TEXT);
    """)
    offset = 0
    for i, (s, raw) in enumerate(zip(sources, raw_rows), 1):
        db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?)", (i, s["trace_id"], offset, len(raw), hashlib.sha256(raw).hexdigest(), digest(s)))
        offset += len(raw)
    db.execute("INSERT INTO gaps VALUES(?,?,?,?,?,?,?,?)", (1, gap["id"], 1, 1, "a", "review_pending", None, "now"))
    c, r = entry["candidate"], entry["generation_receipt"]
    db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?)", tuple(c[k] for k in ("gap_id", "text", "data", "prefix_hash", "lookahead_hash", "prompt_hash")) + ("now",))
    db.execute("INSERT INTO responses VALUES(?,?,?,?,?,?)", (1, r["gap_id"], r["stage"], r["text"], r["data"], "now"))
    identity = {"source_path": str(source_path), "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(), "generator_config": config}
    db.execute("INSERT INTO meta VALUES(?,?)", ("identity", canonical(identity)));db.commit();db.close()
    job = rollout_job(tmp_path)
    manifest = tmp_path / "replay-jobs.jsonl";manifest.write_text(canonical(job) + "\n")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {"schema": "qwen38-frozen-generation-journal/v1", "snapshot_path": str(dbpath),
        "snapshot_sha256": sha(dbpath), "original_journal": str(tmp_path / "original.sqlite3"),
        "original_identity_sha256": digest(identity), "counts": {"sources": 2, "valid_candidates": 1},
        "gap_states": {"review_pending": 1}, "invalid_candidate_count_preserved": 0,
        "source_path": str(source_path), "source_sha256": identity["source_sha256"],
        "source_bytes": source_path.stat().st_size,
        "replay_manifests": [{"path": str(manifest), "sha256": sha(manifest)}]}
    path = tmp_path / "receipt.json";path.write_text(canonical(receipt))
    return path


def test_real_full_export_keeps_zero_filled_and_replay_with_exact_manifest(tmp_path):
    receipt = freeze_fixture(tmp_path)
    result = prepare(freeze_receipt=receipt, tokenizer_dir=str(ASSETS), output=tmp_path / "out", workers=1, shard_rows=1)
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert sum(manifest["counts"].values()) == 3
    assert manifest["unreviewed_gap_count"] == 1
    assert manifest["all_frozen_generations_included_before_cutoff"] is True
    assert manifest["reviewed"] is False and manifest["future_information_checked"] is False
    assert sum(result["index"]["rows"].values()) == 3
    assert (tmp_path / "out/COMPLETE.json").exists()
    feed = [json.loads(line) for line in (tmp_path / "out/finalized-shards.jsonl").read_text().splitlines()]
    assert len(feed) == 3
    assert all(hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest() == s["sha256"] for s in feed)
    assert all(Path(s["path"]).stat().st_mode & 0o222 == 0 for s in feed)
    rows = [json.loads(Path(s["path"]).read_text()) for v in manifest["splits"].values() for s in v]
    assert sorted(r["metadata"]["provenance"]["generated_cot"]["filled_gaps"] for r in rows) == [0, 0, 1]
    assert all(split_for_group(r["group_id"]) == split for split, shards in manifest["splits"].items()
               for shard in shards for r in [json.loads(Path(shard["path"]).read_text())])


def test_changed_snapshot_refused(tmp_path):
    path = freeze_fixture(tmp_path)
    receipt = json.loads(path.read_text());receipt["snapshot_sha256"] = "0" * 64;path.write_text(canonical(receipt))
    with pytest.raises(ValueError, match="immutable generation snapshot"):
        load_freeze(path)


def test_exact_baseline_session_split_rule():
    from training.qwen38_no_cot.prepare_data import SEED
    for group in ("original-group", "session:abc", "session:def"):
        expected = "validation" if int(hashlib.sha256((SEED + ":split:" + group).encode()).hexdigest()[:8], 16) % 100 == 0 else "train"
        assert split_for_group(group) == expected


def test_all_train_preserves_unknown_session_generations_without_mutating_source(tmp_path):
    receipt = freeze_fixture(tmp_path, missing_group=True)
    before = (tmp_path / "sources.jsonl").read_bytes()
    result = prepare(freeze_receipt=receipt, tokenizer_dir=str(ASSETS), output=tmp_path / "out",
                     workers=1, shard_rows=1, all_train=True)
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert set(manifest["splits"]) == {"train"}
    assert manifest["all_train"] is True and manifest["internal_validation"] is False
    assert manifest["session_grouping_complete"] is False
    assert manifest["unresolved_session_sources"] == 2 and manifest["unresolved_session_generated_gaps"] == 1
    assert manifest["unreviewed_gap_count"] == 1
    rows = [json.loads(Path(shard["path"]).read_text()) for shard in manifest["splits"]["train"]]
    for row in rows:
        if row["source"] == "trace":
            p = row["metadata"]["provenance"]
            assert p["session_identity_verified"] is False and p["group_basis"] == "capture_id"
            assert p["original_session_group_id"] is None
            assert row["group_id"] == "capture-only:" + p["generated_cot"]["trace_id"]
    assert (tmp_path / "sources.jsonl").read_bytes() == before


@pytest.mark.parametrize("all_train,low_quality", [(False, False), (True, True)])
def test_train_only_neither_weakens_quality_nor_default_missing_session_guard(tmp_path, all_train, low_quality):
    receipt = freeze_fixture(tmp_path, missing_group=True, low_quality=low_quality)
    with pytest.raises(ValueError, match="original-trace/generation exclusions"):
        prepare(freeze_receipt=receipt, tokenizer_dir=str(ASSETS), output=tmp_path / "out",
                workers=1, shard_rows=1, all_train=all_train)
    assert not (tmp_path / "out/COMPLETE.json").exists()


def exclusion_fixture(tmp_path, freeze_path):
    receipt = json.loads(freeze_path.read_text())
    baseline = tmp_path / "baseline-exclusions.jsonl"
    baseline.write_text(canonical({"identity": "zero-filled-original", "source": "trace", "reason": "UnsupportedMessage"}) + "\n")
    native = tmp_path / "native-audit.json"
    native.write_text(canonical({"snapshot_sha256": receipt["snapshot_sha256"], "source_sha256": receipt["source_sha256"],
        "all_invalid_prior_baseline_excluded": True, "invalid_sources": 1, "invalid_source_generated_gaps": 0,
        "bad_sources": [{"identity": "zero-filled-original", "source_id": 2, "reason": "UnsupportedMessage",
                         "prior_baseline_reason": "UnsupportedMessage", "generated_gaps": 0}]}))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    policy = tmp_path / "source-policy.json"
    policy.write_text(canonical({"schema": "qwen38-baseline-native-source-exclusions/v1",
        "snapshot_sha256": receipt["snapshot_sha256"], "source_sha256": receipt["source_sha256"],
        "native_audit_path": str(native), "native_audit_sha256": sha(native),
        "baseline_exclusions_path": str(baseline), "baseline_exclusions_sha256": sha(baseline)}))
    return policy, sha(policy)


def test_only_exact_rechecked_prior_native_failure_can_be_excluded(tmp_path):
    receipt = freeze_fixture(tmp_path, invalid_zero_tool=True)
    policy, signature = exclusion_fixture(tmp_path, receipt)
    result = prepare(freeze_receipt=receipt, tokenizer_dir=str(ASSETS), output=tmp_path / "out", workers=1,
                     all_train=True, source_exclusions=policy, source_exclusions_sha256=signature)
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["excluded_source_count"] == 1 and manifest["eligible_source_count"] == 1
    assert manifest["all_frozen_generations_accounted_for"] is True
    assert manifest["included_valid_candidate_count"] == 1 and manifest["excluded_valid_candidate_count"] == 0
    exclusions = [json.loads(line) for line in (tmp_path / "out/exclusions.jsonl").read_text().splitlines()]
    assert len(exclusions) == 1 and exclusions[0]["preaudited_known_baseline_exclusion"] is True


def test_policy_cannot_exclude_actually_renderable_trace(tmp_path):
    receipt = freeze_fixture(tmp_path)
    policy, signature = exclusion_fixture(tmp_path, receipt)
    with pytest.raises(ValueError, match="natively renderable source cannot be excluded"):
        prepare(freeze_receipt=receipt, tokenizer_dir=str(ASSETS), output=tmp_path / "out", workers=1,
                all_train=True, source_exclusions=policy, source_exclusions_sha256=signature)
