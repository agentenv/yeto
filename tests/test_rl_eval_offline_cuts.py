"""S27 R-5: offline evaluation of round cuts on a held-out set (CPU, fake engine)."""

from __future__ import annotations

import csv
import json
import os

import pytest

from yeto.rl.engine.cut import (AlgorithmIdentity, CutFile, CutManifest, CutProgress, commit_manifest, cut_dir,
                                sha256_file)
from yeto.rl.engine.resume import ROUND_CUTS
from yeto.rl.eval.offline_cuts import (EvalSpec, HeldOutSet, OfflineEvalError, SglangHttpEngine, discover_cuts,
                                       evaluate_run, main, request_seed)
from yeto.rl.rewards.registry import load_reward

ALGO = AlgorithmIdentity("a" * 64)


def _commit_cut(store, cut_id, *, step, mtime, outer=None, created=None):
    root = store / ROUND_CUTS
    d = cut_dir(root, cut_id)
    d.mkdir(parents=True, exist_ok=True)
    shard = d / "trainer_tp0_pp0_dp0.pt"
    shard.write_bytes(b"w" * 8)
    m = CutManifest(
        cut_id=cut_id, epoch=1,
        runtime={"backend_fingerprint": "fake", "layout": {"world": 1}, "rng_policy": "exact"},
        progress=CutProgress(local_step=step, scheduler_samples=step * 4, global_batch_size=4,
                             next_rollout_id=step, policy_version=step, policy_hash=f"{step:064d}"),
        algorithm=ALGO,
        data={"sample_offset": 0, "epoch_id": 0, "sample_group_index": 0, "sample_index": 0},
        ledger={"carried_over": 0, "ready_unconsumed": 0},
        outer={"settled": True, **({"outer_version": outer} if outer is not None else {})},
        files=(CutFile(shard.name, sha256_file(shard), 8, {"tp": 0}),),
        rank_summaries=({"path": shard.name, "scheduler_samples": step * 4, "has_optimizer_state": True,
                         "has_rng": True, "state_digest": "s", "rng_digest": "r"},),
        created_unix=created)
    commit_manifest(root, m)
    os.utime(d / "manifest.json", (mtime, mtime))


def _heldout(tmp_path, n=4):
    path = tmp_path / "heldout.jsonl"
    path.write_text("".join(json.dumps({"id": f"q{i}", "prompt": f"what is {i}+{i}?", "label": str(2 * i)}) + "\n"
                            for i in range(n)))
    return path


class FakeEngine:
    """Answers correctly for the first ``correct[cut_id]`` prompts (by task order)."""

    def __init__(self, correct):
        self.correct = correct
        self.loaded = None
        self.loads = []
        self.seen_seeds = []

    def load(self, cut):
        self.loaded = cut.cut_id
        self.loads.append(cut.cut_id)

    def generate(self, requests, sampling):
        self.seen_seeds.append([r.seed for r in requests])
        out = []
        for r in requests:
            i = int(r.task_id[1:])
            ok = i < self.correct[self.loaded]
            out.append(f"answer \\boxed{{{2 * i if ok else -1}}}")
        return out


@pytest.fixture
def run(tmp_path):
    store = tmp_path / "store"
    _commit_cut(store, "r000002-inc0", step=2, mtime=1000.0)
    _commit_cut(store, "r000004-inc0", step=4, mtime=1300.0, outer=7)
    # an interrupted save: shards, no manifest
    (store / ROUND_CUTS / "r000006-inc0").mkdir(parents=True)
    return store


def _spec(tmp_path, **kw):
    held = HeldOutSet.load(_heldout(tmp_path), id_key="id")
    return EvalSpec(held, load_reward("gsm8k"), {"temperature": 0.0}, **kw)


def test_discover_skips_uncommitted_and_sorts(run):
    skipped = []
    cuts = discover_cuts(run, skipped=skipped)
    assert [c.cut_id for c in cuts] == ["r000002-inc0", "r000004-inc0"]
    assert [c.train_step for c in cuts] == [2, 4]
    assert cuts[1].outer_version == 7 and cuts[0].outer_version is None
    assert cuts[0].wall_time == 1000.0
    assert [s["cut_id"] for s in skipped] == ["r000006-inc0"]


def test_curve_vs_step_and_wall_clock(run, tmp_path):
    engine = FakeEngine({"r000002-inc0": 1, "r000004-inc0": 3})
    out = tmp_path / "out"
    curve = evaluate_run(discover_cuts(run), _spec(tmp_path, samples_per_prompt=2), engine, out, run_start=900.0)
    pts = curve["points"]
    assert [p["train_step"] for p in pts] == [2, 4]
    assert [p["mean_reward"] for p in pts] == [0.25, 0.75]
    assert [p["elapsed_s"] for p in pts] == [100.0, 400.0]
    assert pts[0]["n_prompts"] == 4 and pts[0]["n_samples"] == 8
    assert pts[0]["stderr"] is not None
    # same seeds for every cut: the sampling noise source is fixed across cuts
    assert engine.seen_seeds[0] == engine.seen_seeds[1]
    assert engine.seen_seeds[0][:2] == [request_seed("q0", 0, 0), request_seed("q0", 1, 0)]
    rows = list(csv.DictReader((out / "curve.csv").open()))
    assert [r["cut_id"] for r in rows] == ["r000002-inc0", "r000004-inc0"]
    assert rows[1]["outer_version"] == "7" and rows[0]["outer_version"] == ""
    saved = json.loads((out / "cuts" / "r000004-inc0.json").read_text())
    assert len(saved["samples"]) == 8 and saved["ruler_sha256"] == curve["ruler_sha256"]
    assert json.loads((out / "curve.json").read_text()) == curve


def test_rerun_reuses_same_ruler_and_redoes_changed_ruler(run, tmp_path):
    out = tmp_path / "out"
    evaluate_run(discover_cuts(run), _spec(tmp_path), FakeEngine({"r000002-inc0": 1, "r000004-inc0": 3}), out)
    again = FakeEngine({"r000002-inc0": 0, "r000004-inc0": 0})
    curve = evaluate_run(discover_cuts(run), _spec(tmp_path), again, out)
    assert again.loads == [] and [p["mean_reward"] for p in curve["points"]] == [0.25, 0.75]
    changed = evaluate_run(discover_cuts(run), _spec(tmp_path, seed=1), again, out)
    assert again.loads == ["r000002-inc0", "r000004-inc0"]
    assert [p["mean_reward"] for p in changed["points"]] == [0.0, 0.0]
    assert changed["run_start_source"] == "earliest_cut" and changed["points"][0]["elapsed_s"] == 0.0


def test_heldout_checks(tmp_path):
    path = _heldout(tmp_path)
    with pytest.raises(OfflineEvalError, match="sha256"):
        HeldOutSet.load(path, expect_sha256="0" * 64)
    assert len(HeldOutSet.load(path, limit=2).items) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "x"}) + "\n")
    with pytest.raises(OfflineEvalError, match="missing"):
        HeldOutSet.load(bad)


def test_engine_response_count_mismatch_fails(run, tmp_path):
    class Short(FakeEngine):
        def generate(self, requests, sampling):
            return super().generate(requests, sampling)[:-1]

    with pytest.raises(OfflineEvalError, match="responses"):
        evaluate_run(discover_cuts(run), _spec(tmp_path), Short({"r000002-inc0": 1, "r000004-inc0": 1}),
                     tmp_path / "out")


def test_sglang_engine_protocol(run):
    calls = []

    def post(url, payload, timeout):
        calls.append((url.rsplit("/", 1)[1], payload))
        if url.endswith("/update_weights_from_disk"):
            return {"success": True}
        return [{"text": f"t{i}"} for i in range(len(payload["text"]))]

    cut = discover_cuts(run)[0]
    eng = SglangHttpEngine("http://x:1/", lambda c: f"/hf/{c.cut_id}", batch_size=2, post=post)
    eng.load(cut)
    held = [type("R", (), {"prompt": f"p{i}", "seed": i})() for i in range(3)]
    assert eng.generate(held, {"temperature": 0.0}) == ["t0", "t1", "t0"]
    assert calls[0] == ("update_weights_from_disk", {"model_path": "/hf/r000002-inc0"})
    assert calls[1][1]["sampling_params"][1] == {"temperature": 0.0, "sampling_seed": 1}
    assert len(calls) == 3  # one load + two batches

    bad = SglangHttpEngine("http://x", lambda c: "/m", post=lambda *a: {"success": False, "message": "no"})
    with pytest.raises(OfflineEvalError, match="could not load"):
        bad.load(cut)


def test_cli_end_to_end(run, tmp_path, monkeypatch):
    import yeto.rl.eval.offline_cuts as mod

    monkeypatch.setattr(mod, "_http_post_json", lambda url, payload, timeout: (
        {"success": True} if url.endswith("update_weights_from_disk")
        else [{"text": "\\boxed{0}"} for _ in payload["text"]]))
    monkeypatch.setattr(mod, "_load_callable", lambda ref: (lambda cut: f"/hf/{cut.cut_id}"))
    out = tmp_path / "cli"
    assert main(["--store", str(run), "--heldout", str(_heldout(tmp_path)), "--reward", "gsm8k",
                 "--out", str(out), "--sglang-url", "http://x", "--materialize", "conv:to_hf"]) == 0
    curve = json.loads((out / "curve.json").read_text())
    assert [p["mean_reward"] for p in curve["points"]] == [0.25, 0.25]  # only q0 (label 0) is right
    assert curve["skipped_cuts"][0]["cut_id"] == "r000006-inc0"


def test_created_unix_preferred_over_mtime(tmp_path):
    store = tmp_path / "s"
    _commit_cut(store, "r000001-a", step=1, mtime=5000.0, created=4000.0)
    _commit_cut(store, "r000002-a", step=2, mtime=6000.0)
    a, b = discover_cuts(store)
    assert (a.wall_time, a.wall_time_source) == (4000.0, "manifest_created_unix")
    assert (b.wall_time, b.wall_time_source) == (6000.0, "manifest_mtime")
