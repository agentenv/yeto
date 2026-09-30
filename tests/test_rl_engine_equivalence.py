"""Tests for scripts/rl_engine_equivalence.py and yeto.rl.teacher_forcing (rl-engine-ports 6.1-6.3)."""

from __future__ import annotations

import asyncio
import enum
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rl_engine_equivalence",
                                               ROOT / "scripts" / "rl_engine_equivalence.py")
eq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eq)

EVIDENCE = ROOT / "openspec" / "changes" / "rl-engine-ports" / "evidence"


def _row(reward=0.5, loss=0.1, gn=1.0, groups=4, tokens=100):
    return {"reward_mean": reward, "loss": loss, "grad_norm": gn,
            "completed_groups": groups, "action_tokens": tokens}


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def test_extract_rounds_keys_hash_by_policy_version():
    events = [
        {"event": "rl_policy_apply", "island_id": 1, "policy_version": 0,
         "sync/global_policy_hash": "h0"},
        {"event": "rl_local_round", "island_id": 1, "local_round_id": 1, "reward_mean": 0.2,
         "loss": 0.3, "grad_norm": 4.0, "completed_groups": 2, "action_tokens": 9},
        {"event": "rl_policy_apply", "island_id": 1, "policy_version": 1,
         "sync/global_policy_hash": "h1"},
        {"event": "rl_driver_phase", "phase": "train"},
    ]
    assert eq.extract_rounds(events) == {(1, 1): {**_row(0.2, 0.3, 4.0, 2, 9),
                                                  "post_sync_hash": "h1"}}


def test_extract_hashes_skips_partial_fragment_applies_but_keeps_final_cut():
    ev = lambda i, v, h, partial: {"event": "rl_policy_apply", "island_id": i,  # noqa: E731
                                   "policy_version": v, "sync/global_policy_hash": h,
                                   "partial_fragment_apply": partial}
    events = [ev(0, 0, "a", False), ev(0, 5, "x", True), ev(0, 8, "f", True),
              ev(1, 0, "a", False), ev(1, 5, "y", True), ev(1, 8, "f", True)]
    hashes = eq.extract_hashes(events)
    assert hashes == {(0, 0): "a", (0, 8): "f", (1, 0): "a", (1, 8): "f"}
    assert eq.hash_check(hashes)["passed"]


def _step_line(step, loss, gn):
    return (f"(MegatronTrainRayActor pid=1) [t actor_cell0_rank0] log_utils.py:460 - step {step}: "
            f"{{'train/loss': {loss}, 'train/grad_norm': {gn}}}")


def _write_arm(arm: Path, rows: dict, *, hashes=None, log_metrics=True, audit=None):
    """Minimal benchmark_rl arm: island-i/events.jsonl (+ miles.log, + audit)."""

    for island in sorted({i for i, _ in rows}):
        d = arm / f"island-{island}"
        d.mkdir(parents=True, exist_ok=True)
        lines, log = [], ["noise line"]
        for step, r in enumerate(sorted(r for i, r in rows if i == island)):
            row = dict(rows[(island, r)])
            ev = {"event": "rl_local_round", "island_id": island, "local_round_id": r,
                  **{m: row[m] for m in ("reward_mean", "completed_groups", "action_tokens")},
                  "loss": None if log_metrics else row["loss"],
                  "grad_norm": None if log_metrics else row["grad_norm"]}
            lines.append(ev)
            log.append(_step_line(step, row["loss"], row["grad_norm"]))
            h = (hashes or {}).get((island, r), f"hash-{r}")
            lines.append({"event": "rl_policy_apply", "island_id": island, "policy_version": r,
                          "sync/global_policy_hash": h})
        (d / "events.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
        if log_metrics:
            (d / "miles.log").write_text("\n".join(log) + "\n")
        if audit is not None and island in audit:
            base, delta, layout, *grads = audit[island]
            a = d / "audit"
            a.mkdir()
            if grads and grads[0] is not None:
                import torch

                from yeto.rl import grad_audit

                grad_audit.write(a, "round-00000001",
                                 {n: torch.tensor(v, dtype=torch.float32) for n, v in grads[0].items()},
                                 {"engine": "test", "grad_norm_l2": 1.0})
            np.asarray(base, dtype="<f4").tofile(a / "round-00000001.base.f32")
            np.asarray(delta, dtype="<f4").tofile(a / "round-00000001.delta.f32")
            (a / "round-00000001.json").write_text(json.dumps({"layout_hash": layout}))
    return arm


def test_load_arm_fills_loss_and_grad_norm_from_miles_log(tmp_path):
    arm = _write_arm(tmp_path / "work" / "seed-17" / "yeto-federated-m2",
                     {(0, 1): _row(gn=0.6), (0, 2): _row(gn=0.0, loss=0.0)})
    loaded = eq.load_arm(arm)
    assert loaded["rounds"][(0, 1)]["grad_norm"] == 0.6
    assert loaded["rounds"][(0, 2)]["loss"] == 0.0
    assert loaded["filled_from_miles_log"]["0/2/grad_norm"] == "miles.log step 1"
    assert eq.discover_arms(tmp_path) == {17: arm}
    assert eq.discover_arms(arm) == {17: arm}


def test_miles_steps_ignores_non_trainer_lines():
    text = "\n".join([_step_line(0, 1e-8, 0.5),
                      "(RolloutManager) log_utils.py:1 - step 0: {'train/loss': 9}"])
    assert eq.miles_steps(text) == {0: {"train/loss": 1e-8, "train/grad_norm": 0.5}}


# ---------------------------------------------------------------------------
# tier 1
# ---------------------------------------------------------------------------
def test_round1_exact_counts_and_grad_norm_relative_bound():
    legacy = {(0, 1): _row(gn=1.0), (1, 1): _row(gn=2.0)}
    ok = eq.round1_check(legacy, {(0, 1): _row(gn=1.029), (1, 1): _row(gn=1.95)})
    assert ok["passed"]
    bad = eq.round1_check(legacy, {(0, 1): _row(gn=1.031), (1, 1): _row(gn=2.0, tokens=101)})
    failed = {(r["island"], r["metric"]) for r in bad["rows"] if not r["pass"]}
    assert failed == {(0, "grad_norm"), (1, "action_tokens")}
    missing = eq.round1_check(legacy, {(0, 1): _row()})
    assert not missing["passed"]


def test_round1_on_committed_evidence_matches_the_baseline_report():
    legacy = eq.load_arm(next(iter(eq.discover_arms(
        EVIDENCE / "2026-09-29-legacy-baseline-v2" / "legacy-0").values())))
    ports = eq.load_arm(next(iter(eq.discover_arms(
        EVIDENCE / "2026-09-29-strict2-federated").values())))
    r1 = eq.round1_check(legacy["rounds"], ports["rounds"])
    assert r1["passed"]
    rel = sorted(r["rel_diff"] for r in r1["rows"] if r["metric"] == "grad_norm")
    assert 0.005 < rel[0] < 0.006 and 0.010 < rel[1] < 0.011  # 0.6% / 1.0%
    assert eq.hash_check(legacy["hashes"])["passed"] and eq.hash_check(ports["hashes"])["passed"]


# ---------------------------------------------------------------------------
# tier 3
# ---------------------------------------------------------------------------
def _seeds(values_by_seed, metric="grad_norm"):
    return {s: {(0, 1): _row(gn=99.0), (0, 2): {**_row(), metric: v}, (1, 2): {**_row(), metric: v}}
            for s, v in values_by_seed.items()}


def test_seed_summary_averages_rounds_from_two_and_skips_round_one():
    rounds = {(0, 1): _row(gn=100.0), (0, 2): _row(gn=1.0), (1, 2): _row(gn=2.0), (0, 3): _row(gn=3.0)}
    assert eq.seed_summary(rounds)["grad_norm"] == pytest.approx(2.0)
    assert eq.seed_summary({(0, 1): _row()})["grad_norm"] is None


def test_permutation_test_exact_enumeration():
    x, y = [1, 2, 3, 4, 5], [6, 7, 8, 9, 10]
    assert eq.permutation_test(x, y) == pytest.approx(2 / 252)  # fully separated: minimum p
    assert eq.permutation_test(x, list(x)) == 1.0
    p = eq.permutation_test([1, 2, 3, 4, 6], [5, 7, 8, 9, 10])
    assert 2 / 252 < p < 0.05
    assert eq.effect_size([0, 2], [2, 4]) == pytest.approx(2 / 2 ** 0.5)
    assert eq.effect_size([1, 1], [1, 1]) == 0.0


def test_distribution_permutation_pass_and_fail():
    legacy = _seeds({17: 0.1, 18: 0.5, 19: 0.3, 20: 0.2, 21: 0.4})
    mixed = eq.distribution_check(legacy, _seeds({17: 0.15, 18: 0.45, 19: 0.35, 20: 0.25, 21: 0.6}))
    assert mixed["passed"]
    row = next(r for r in mixed["rows"] if r["metric"] == "grad_norm")
    assert row["legacy_values"][17] == 0.1 and row["p_value"] > 0.0125
    assert row["legacy_mean"] == pytest.approx(0.3)
    shifted = eq.distribution_check(legacy, _seeds({17: 0.6, 18: 0.7, 19: 0.8, 20: 0.9, 21: 1.0}))
    assert not shifted["passed"]
    bad = [r for r in shifted["rows"] if not r["pass"]]
    assert [r["metric"] for r in bad] == ["grad_norm"]
    assert bad[0]["p_value"] == pytest.approx(2 / 252) and bad[0]["effect_size"] > 3
    st = shifted["stats"]
    assert st["alpha_per_metric"] == pytest.approx(0.0125) and st["splits"] == 252
    assert st["reject_only_if_fully_separated"] and 2.5 <= st["min_effect_80pct_power_sd"] <= 3.5


def test_distribution_edges_missing_values_and_too_few_seeds():
    legacy = _seeds({17: 0.1, 18: 0.5, 19: 0.3})
    ports = _seeds({17: 0.2, 18: 0.2, 19: 0.2})
    ports[19][(0, 2)]["loss"] = None
    res = eq.distribution_check(legacy, ports)
    assert not res["passed"] and any(r.get("note") for r in res["rows"])
    few = eq.distribution_check(_seeds({17: 0.1, 18: 0.2}), ports)
    assert not few["passed"] and few["incomplete"] and "need >= 3" in few["error"]
    round1_only = eq.distribution_check({s: {(0, 1): _row()} for s in (1, 2, 3)},
                                        {s: {(0, 1): _row()} for s in (1, 2, 3)})
    assert not round1_only["passed"]
    # 3 vs 3: min p = 2/20 = 0.1 > 0.0125, the test can never fail -> reported
    assert eq.distribution_check(legacy, _seeds({17: 9, 18: 9, 19: 9}))["stats"]["min_effect_80pct_power_sd"] is None


# ---------------------------------------------------------------------------
# tier 2 (teacher forcing report parsing)
# ---------------------------------------------------------------------------
GRADS = {"base_model.model.l0.q_proj.lora_A.weight": [[0.3, -0.2], [0.1, 0.05]],
         "base_model.model.l0.q_proj.lora_B.weight": [[1e-4, -2e-4]]}


def _tf_arms(tmp_path, *, ports_gn=1.01, ports_delta=None, ports_base=None, ports_tokens=100,
             ports_audit=True, loss_p=1e-8, ports_grads=None, grads=True, legacy_grads=None):
    base = [0.5, -0.25, 1.0, 2.0]
    delta = [0.01, -0.02, 0.0, 0.03]
    lgrads = (legacy_grads or GRADS) if grads else None
    pgrads = (ports_grads or GRADS) if grads else None
    ref = _write_arm(tmp_path / "ref" / "work" / "seed-17" / "a",
                     {(i, 1): _row(gn=1.0, loss=1.2e-8) for i in (0, 1)})
    lg = _write_arm(tmp_path / "tfl" / "work" / "seed-17" / "a",
                    {(i, 1): _row(gn=1.0, loss=1.2e-8) for i in (0, 1)},
                    audit={i: (base, delta, "L", lgrads) for i in (0, 1)})
    pt = _write_arm(tmp_path / "tfp" / "work" / "seed-17" / "a",
                    {(i, 1): _row(gn=ports_gn, loss=loss_p, tokens=ports_tokens) for i in (0, 1)},
                    audit=({i: (ports_base or base, ports_delta or delta, "L", pgrads) for i in (0, 1)}
                           if ports_audit else None))
    return eq.load_tf_arm(ref), eq.load_tf_arm(lg), eq.load_tf_arm(pt)


def _fp32_ref(grads=GRADS, islands=(0, 1)):
    """An fp32 reference in the shape ``yeto.rl.fp32_reference.compute_island`` returns."""
    order = sorted(grads)
    flat = np.concatenate([np.asarray(grads[n], dtype="<f4").ravel() for n in order])
    ref = {"flat": flat, "order": order,
           "shapes": {n: list(np.asarray(grads[n]).shape) for n in order},
           "grad_norm": float(np.linalg.norm(flat)), "loss": 0.0,
           "provenance": {"replay": {"path": "r.pt", "sha256": "0"},
                          "initial_lora": {"path": "b.f32", "sha256": "1", "layout_hash": "L"},
                          "model": {"id": "org/m", "revision": "a" * 40, "dir": "d",
                                    "files_sha256": {"config.json": "2"}},
                          "loss": {"aggregation": "sample_mean"}, "dtype": "float32",
                          "device": "cpu", "torch": "t"}}
    return {i: ref for i in islands}


def _tfc(tmp_path, *, fp32=GRADS, require_update=True, **kwargs):
    return eq.teacher_forcing_check(*_tf_arms(tmp_path, **kwargs), require_update=require_update,
                                    fp32=(_fp32_ref(fp32) if isinstance(fp32, dict)
                                          and all(isinstance(k, str) for k in fp32) else fp32))


def test_teacher_forcing_passes_within_thresholds(tmp_path):
    near = [0.0101, -0.0201, 0.0, 0.0301]  # rel L2 ~0.5%
    res = _tfc(tmp_path, ports_delta=near)
    assert res["passed"], [r for r in res["rows"] if r["gate"] and not r["pass"]]
    upd = [r for r in res["rows"] if r["check"] == "LoRA update rel L2 (info)"]
    assert len(upd) == 2 and upd[0]["value"] < 0.01 and upd[0]["cosine"] > 0.999
    assert not upd[0]["gate"]
    g = [r for r in res["rows"] if r["check"] == "LoRA gradient relL2 vs fp32"]
    assert len(g) == 2 and g[0]["ports"] == 0.0 and g[0]["gate"] and g[0]["pass"]
    cross = next(r for r in res["rows"] if r["check"] == "cross-path LoRA gradient rel L2 (info)")
    assert cross["gate"] is False
    loss = next(r for r in res["rows"] if r["check"] == "loss")
    assert loss["pass"] and loss["value"] == pytest.approx(2e-9)


@pytest.mark.parametrize("kwargs,failing", [
    ({"ports_gn": 1.04}, "grad_norm"),
    ({"ports_grads": {**GRADS, "base_model.model.l0.q_proj.lora_A.weight": [[0.33, -0.2], [0.1, 0.05]]}},
     "LoRA gradient relL2 vs fp32"),
    ({"ports_grads": {"base_model.model.l0.q_proj.lora_A.weight": [[0.3, -0.2], [0.1, 0.05]],
                      "base_model.model.l0.k_proj.lora_B.weight": [[1e-4, -2e-4]]}},
     "LoRA gradient tensors aligned with fp32 reference"),
    ({"grads": False}, "LoRA gradient (grad audit f32)"),
    ({"ports_base": [0.5, -0.25, 1.0, 2.1]}, "initial LoRA rel L2"),
    ({"ports_tokens": 99}, "fidelity ports-TF action_tokens == reference"),
    ({"loss_p": 5e-6}, "loss"),
    ({"ports_audit": False}, "LoRA base/update (audit f32)"),
])
def test_teacher_forcing_failures(tmp_path, kwargs, failing):
    res = _tfc(tmp_path, **kwargs)
    assert not res["passed"]
    assert failing in {r["check"] for r in res["rows"] if r["gate"] and not r["pass"]}


def test_teacher_forcing_decoupled_without_audit_reports_update_ungated(tmp_path):
    res = _tfc(tmp_path, ports_audit=False, require_update=False)
    assert res["passed"]
    row = next(r for r in res["rows"] if r["check"] == "LoRA base/update (audit f32)")
    assert row["gate"] is False and row["pass"] is None
    res = eq.teacher_forcing_check(*_tf_arms(tmp_path / "g", grads=False), require_update=False)
    assert res["passed"]
    row = next(r for r in res["rows"] if r["check"] == "LoRA gradient (grad audit f32)")
    assert row["gate"] is False


def _scaled(grads, noise):
    """GRADS with a deterministic perturbation of relative size ``noise`` per element."""
    out = {}
    for k, (n, v) in enumerate(sorted(grads.items())):
        a = np.asarray(v, dtype=np.float64)
        signs = np.where((np.arange(a.size).reshape(a.shape) + k) % 2 == 0, 1.0, -1.0)
        out[n] = (a * (1 + noise * signs)).tolist()
    return out


def test_fp32_anchor_passes_when_ports_is_as_close_as_legacy(tmp_path):
    # Legacy 8 % and ports 12 % away from fp32 (cross-path ~20 %): passes option A.
    res = _tfc(tmp_path, fp32=_scaled(GRADS, 0.0), grads=True,
               ports_grads=_scaled(GRADS, -0.12), legacy_grads=_scaled(GRADS, 0.08))
    rows = {r["check"]: r for r in res["rows"] if r["island"] == 0}
    assert res["passed"], [r for r in res["rows"] if r["gate"] and not r["pass"]]
    rel = rows["LoRA gradient relL2 vs fp32"]
    assert rel["legacy"] == pytest.approx(0.08, rel=1e-3) and rel["ports"] == pytest.approx(0.12, rel=1e-3)
    assert rows["cross-path LoRA gradient rel L2 (info)"]["value"] > 0.18


def test_fp32_anchor_fails_when_ports_is_farther_than_legacy_plus_margin(tmp_path):
    res = _tfc(tmp_path, fp32=GRADS, legacy_grads=_scaled(GRADS, 0.08),
               ports_grads=_scaled(GRADS, 0.14))
    assert not res["passed"] and not res["incomplete"]
    bad = {r["check"] for r in res["rows"] if r["gate"] and not r["pass"]}
    assert "LoRA gradient relL2 vs fp32" in bad


def test_fp32_anchor_cosine_margin():
    ok = {"legacy": {"rel_l2": 0.1, "cosine": 0.995}, "ports": {"rel_l2": 0.1, "cosine": 0.9901}}
    assert eq.fp32_gate(ok) == {"rel_ok": True, "cos_ok": True}
    bad = {"legacy": {"rel_l2": 0.1, "cosine": 0.995}, "ports": {"rel_l2": 0.1, "cosine": 0.9899}}
    assert eq.fp32_gate(bad)["cos_ok"] is False


def test_fp32_reference_missing_or_failed_is_incomplete(tmp_path):
    for fp32 in (None, {0: {"error": "ReferenceInputError: replay batch missing"},
                        1: {"error": "x"}}):
        res = _tfc(tmp_path / str(bool(fp32)), fp32=fp32)
        assert not res["passed"] and res["incomplete"], res["rows"]
        tiers = eq.evaluate(preset="strict-avg", legacy_runs={}, ports_runs={}, primary_seed=17,
                            tf=res)
        assert tiers["tiers"]["2_teacher_forcing"]["incomplete"]
    # A real gate failure next to a missing reference is a FAIL, not INCOMPLETE.
    res = _tfc(tmp_path / "f", fp32=None, ports_gn=1.5)
    assert not res["passed"] and not res["incomplete"]


def test_miles_loss_args_and_loss_config(tmp_path):
    log = tmp_path / "miles.log"
    log.write_text("\n".join([
        "  advantage_estimator ....... grpo", "  eps_clip .......... 0.2",
        "  eps_clip_high ...... 0.28", "  global_batch_size ....... 32",
        "  global_batch_size ....... 99", "  calculate_per_token_loss ...... False",
        "  lora_alpha ....... 16", "  grpo_std_normalization ..... True", "noise"]))
    a = eq.miles_loss_args(log)
    assert a["global_batch_size"] == "32" and a["eps_clip_high"] == "0.28"
    cfg = eq.fp32_loss_config(a, dict(a))
    assert cfg.num_samples == 32 and cfg.eps_clip_high == 0.28 and cfg.lora_alpha == 16.0
    assert "differ" in eq.fp32_loss_config(a, {**a, "eps_clip": "0.3"})
    assert "not modelled" in eq.fp32_loss_config({**a, "calculate_per_token_loss": "True"},
                                                 {**a, "calculate_per_token_loss": "True"})


def test_compute_fp32_references_missing_replay_is_reported(tmp_path):
    _tf_arms(tmp_path)
    ref, lg, pt = (eq._one_arm(str(tmp_path / n)) for n in ("ref", "tfl", "tfp"))
    args = eq.parse_args(["analyze", "--legacy", "x", "--ports", "y",
                          "--fp32-model", "org/m", "--fp32-revision", "a" * 40])
    out = eq.compute_fp32_references(args, ref, lg, pt, eq.load_tf_arm(lg))
    assert set(out) == {0, 1} and all("error" in v for v in out.values())


def test_teacher_forcing_update_is_reported_not_gated(tmp_path):
    # Adam step 1 ~ lr*sign(g): a large update distance with matching gradients passes (D12).
    res = _tfc(tmp_path, ports_delta=[0.01, 0.02, 0.0, -0.03])
    assert res["passed"], [r for r in res["rows"] if r["gate"] and not r["pass"]]
    rows = {r["check"]: r for r in res["rows"] if r["island"] == 0}
    assert rows["LoRA update rel L2 (info)"]["value"] > 1.0
    assert rows["LoRA update sign-flip fraction (info)"]["value"] == pytest.approx(0.5)
    assert rows["LoRA update norm ratio ports/legacy (info)"]["value"] == pytest.approx(1.0)


def test_grad_compare_cosine_gate_and_worst_tensors(tmp_path):
    # Concatenated metrics over canonical-name order; the worst tensor is listed first.
    import torch

    from yeto.rl import grad_audit

    def put(d, tensors):
        grad_audit.write(d, "round-00000001", {n: torch.tensor(v) for n, v in tensors.items()}, {})
        return d / "round-00000001.grad.f32"

    a = put(tmp_path / "a", {"base_model.model.x.lora_A.weight": [1.0, 0.0],
                             "base_model.model.y.lora_B.weight": [0.0, 1.0]})
    b = put(tmp_path / "b", {"base_model.model.x.lora_A.weight": [1.0, 0.02],
                             "base_model.model.y.lora_B.weight": [0.0, 1.0]})
    g = eq.grad_compare(a, b)
    assert g["tensors"] == 2 and g["numel"] == 4
    assert g["rel_l2"] == pytest.approx(0.02 / 2 ** 0.5, rel=1e-5)
    assert g["worst"][0]["name"] == "base_model.model.x.lora_A.weight"
    assert eq.thresholds()["tf_lora_update"].startswith("reported only")


def test_analyze_cli_with_teacher_forcing_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(eq, "compute_fp32_references", lambda *a, **k: _fp32_ref())
    ref, lg, pt = (tmp_path / n for n in ("ref", "tfl", "tfp"))
    _tf_arms(tmp_path)
    legacy_dirs, ports_dirs = [], []
    for s, off in zip((17, 18, 19), (-1, 0, 1)):
        rows = {(i, r): _row(reward=0.5 + 0.1 * off * (r > 1), gn=1.0 + 0.1 * off * (r > 1),
                             tokens=100 + off * (r > 1)) for i in (0, 1) for r in (1, 2)}
        legacy_dirs.append(str(_write_arm(tmp_path / f"l{s}" / "work" / f"seed-{s}" / "a", rows)))
        prow = {(i, r): _row(gn=1.01 if r == 1 else 1.0) for i in (0, 1) for r in (1, 2)}
        ports_dirs.append(str(_write_arm(tmp_path / f"p{s}" / "work" / f"seed-{s}" / "a", prow)))
    out = tmp_path / "out"
    rc = eq.main(["analyze", "--legacy", *legacy_dirs, "--ports", *ports_dirs,
                  "--tf-reference", str(ref), "--tf-legacy", str(lg), "--tf-ports", str(pt),
                  "--out-dir", str(out)])
    data = json.loads((out / "report.json").read_text())
    assert rc == 0, {k: v["passed"] for k, v in data["tiers"].items()}
    assert data["verdict"] == "PASS"
    text = (out / "report.md").read_text()
    assert "## Result: PASS" in text and "power limitation" in text
    assert text.index("## Thresholds") < text.index("## Result")


# ---------------------------------------------------------------------------
# verdict, plan, fake
# ---------------------------------------------------------------------------
def _runs(rows_by_seed):
    return {s: {"dir": "x", "rounds": rows, "hashes": {(0, 1): "h", (1, 1): "h"},
                "filled_from_miles_log": {}} for s, rows in rows_by_seed.items()}


def test_evaluate_verdicts():
    rows = {(0, 1): _row(), (0, 2): _row()}
    runs = _runs({17: rows, 18: rows, 19: rows})
    tf_ok = {"passed": True, "rows": []}
    assert eq.evaluate(preset="strict-avg", legacy_runs=runs, ports_runs=runs,
                       primary_seed=17, tf=tf_ok)["verdict"] == "PASS"
    assert eq.evaluate(preset="strict-avg", legacy_runs=runs, ports_runs=runs,
                       primary_seed=17, tf=None)["verdict"] == "INCOMPLETE"
    bad = _runs({17: {(0, 1): _row(tokens=1), (0, 2): _row()}, 18: rows, 19: rows})
    assert eq.evaluate(preset="strict-avg", legacy_runs=runs, ports_runs=bad,
                       primary_seed=17, tf=None)["verdict"] == "FAIL"
    # decoupled: round 1 is reported, not gated; PEFT tier is required
    dec = eq.evaluate(preset="decoupled", legacy_runs=runs, ports_runs=bad, primary_seed=17,
                      tf=tf_ok, extras={"passed": True, "peft": []})
    assert not dec["tiers"]["1_round1"]["gated"] and dec["verdict"] == "PASS"


def test_plan_seeds_and_teacher_forcing_commands(tmp_path, capsys):
    assert eq.main(["plan", "--launch-args", "--model m --global-rounds 3",
                    "--seeds", "17,18,19", "--teacher-forcing", "--out-dir", str(tmp_path)]) == 0
    plan = json.loads(capsys.readouterr().out)
    names = [s["name"] for s in plan["steps"]]
    assert names == ["legacy-s17", "legacy-s18", "legacy-s19", "ports-s17", "ports-s18",
                     "ports-s19", "tf-legacy", "tf-ports"]
    for s in plan["steps"]:
        cmd = s["command"]
        assert cmd[cmd.index("--rl-engine") + 1] == s["engine"]
        assert cmd[cmd.index("--seeds") + 1] == str(s["seed"])
    tf = plan["steps"][-1]
    assert tf["command"][tf["command"].index("--custom-generate-function-path") + 1] == \
        "yeto.rl.teacher_forcing.replay_generate"
    assert tf["command"][len(tf["command"]) - tf["command"][::-1].index("--global-rounds")] == "1"
    assert tf["env"]["YETO_RL_REPLAY_ROLLOUTS"].endswith("legacy-s17/work/seed-17/*/rollouts/island-*/0.pt")
    assert all(s["env"].get("YETO_RL_AUDIT_GRADS") == "1" for s in plan["steps"][-2:])
    assert not any(s["env"] for s in plan["steps"][:-2])
    assert not list(tmp_path.iterdir())
    with pytest.raises(SystemExit):
        eq.parse_args(["plan", "--launch-args", "x", "--seeds", "18", "--primary-seed", "17"])


def test_fake_mode_produces_labeled_layered_report(tmp_path):
    out = tmp_path / "eqv"
    assert eq.main(["fake", "--out-dir", str(out), "--rounds", "3"]) == 0
    text = (out / "report.md").read_text()
    assert text.startswith("# FAKE REPORT")
    assert "## Result: INCOMPLETE (FAKE)" in text  # teacher forcing is not faked
    data = json.loads((out / "report.json").read_text())
    tiers = data["tiers"]
    assert tiers["1_round1"]["passed"] and tiers["3_distribution"]["passed"]
    assert tiers["4_hash"]["passed"] and tiers["2_teacher_forcing"]["error"] == "not run"
    assert {r["metric"] for r in tiers["3_distribution"]["rows"]} == set(eq.DIST_METRICS)


# ---------------------------------------------------------------------------
# replay generate (teacher forcing entry) + benchmark passthrough
# ---------------------------------------------------------------------------
from yeto.rl import teacher_forcing as tfm  # noqa: E402


class _Status(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"


def _sample(prompt, index):
    return SimpleNamespace(prompt=prompt, index=index, status=_Status.PENDING, tokens=[],
                           response="", response_length=0, loss_mask=None,
                           rollout_log_probs=None, reward=None, metadata={},
                           weight_versions=[])


def _rec(prompt, index, resp):
    return {"prompt": prompt, "index": index, "tokens": [1, 2, 3], "response": resp,
            "response_length": 2, "loss_mask": [1, 1], "rollout_log_probs": [-0.1, -0.2],
            "reward": 1.0, "status": "completed", "weight_versions": ["old"]}


def test_replay_index_exact_then_occurrence_then_miss():
    idx = tfm.ReplayIndex([_rec("p", 0, "a"), _rec("p", 1, "b"), _rec([{"r": "u"}], 2, "c")])
    assert idx.take("p", 1)["response"] == "b"
    assert idx.take("p", 7)["response"] == "a"  # index differs -> first unused occurrence
    assert idx.take([{"r": "u"}], 2)["response"] == "c"
    with pytest.raises(tfm.ReplayMiss):
        idx.take("p", 0)
    with pytest.raises(tfm.ReplayMiss):
        idx.take("unknown", 0)
    assert idx.hits == {"exact": 2, "occurrence": 1}


def test_apply_record_restamps_policy_token_per_engine():
    s = tfm.apply_record(_sample("p", 0), _rec("p", 0, "a"), ports_token="yeto:0:abc",
                         legacy_token=None)
    assert s.response == "a" and s.tokens == [1, 2, 3] and s.status is _Status.COMPLETED
    assert s.weight_versions == [] and s.metadata["weight_version"] == "yeto:0:abc"
    s = tfm.apply_record(_sample("p", 0), _rec("p", 0, "a"), ports_token=None, legacy_token="L")
    assert s.weight_versions == ["L"] and s.metadata["yeto_teacher_forced"]
    with pytest.raises(tfm.ReplayMiss):
        tfm.apply_record(_sample("q", 0), _rec("p", 0, "a"), ports_token=None, legacy_token=None)


def test_replay_generate_loads_recorded_dump(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    d = tmp_path / "rollouts" / "island-0"
    d.mkdir(parents=True)
    torch.save({"rollout_id": 0, "metadata": {}, "samples": [_rec("p", 0, "a")]}, d / "0.pt")
    monkeypatch.setenv(tfm.REPLAY_ENV, str(tmp_path / "rollouts" / "island-*" / "0.pt"))
    monkeypatch.setattr(tfm, "_INDEX", None)
    monkeypatch.setattr(tfm, "_ports_policy_token", lambda: None)
    out = asyncio.run(tfm.replay_generate(SimpleNamespace(), _sample("p", 0), {}))
    assert out.response == "a" and out.weight_versions == ["old"]
    with pytest.raises(RuntimeError):
        asyncio.run(tfm.replay_generate(SimpleNamespace(), _sample("p", 1), {}, evaluation=True))


def test_benchmark_forwards_custom_generate_function_path(tmp_path):
    spec = importlib.util.spec_from_file_location("benchmark_rl_eq", ROOT / "scripts" / "benchmark_rl.py")
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)
    args = bench.build_parser().parse_args([
        "--model", "org/model", "--model-revision", "a" * 40, "--data", "org/data",
        "--data-revision", "b" * 40, "--reward-function", "pkg.reward:score",
        "--custom-generate-function-path", "yeto.rl.teacher_forcing.replay_generate"])
    args._active_seed = 17
    prompt = tmp_path / "prompts.jsonl"
    prompt.write_text("{}\n", encoding="utf-8")
    payload = bench.worker_payload(
        args, bench.WorkerSpec(0, 1, 1, prompt, False),
        arm=bench.select_arms("1", 1, 1, kinds=("native",))[0], run_dir=tmp_path,
        model_path=tmp_path / "model", syncer=None, reward_sha256="c" * 64)
    assert payload["arguments"]["custom_generate_function_path"] == \
        "yeto.rl.teacher_forcing.replay_generate"
