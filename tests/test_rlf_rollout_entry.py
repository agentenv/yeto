"""RLF rollout entry points (W2): catalog -> prompt data, identity producer,
Modal provider factory contract, compaction profile registration, RLF grade
interface, and RLF metadata preservation across compaction segments.

CPU only: fake samples/fake VM, no Ray, no Miles, no Modal, no torch.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace
from typing import Any

import pytest

from yeto.rl.compaction import T_COMP
from yeto.rl.harness.codex import codex_harness_agent as harness
from yeto.rl.harness.codex import codex_openenv_generate as generate_wrapper
from yeto.rl.harness.codex import compaction_bridge as cb
from yeto.rl.rlf.catalog import (
    TaskCatalog, import_catalog, prompt_rows, rlf_identity, task_statement, write_prompt_data,
)
from yeto.rl.rlf.contracts import AdmissionError, EpisodeLease, TrustedGrade
from yeto.rl.rlf.preflight import RolloutRejected, trusted_grade_from_evaluation


# --- helpers -----------------------------------------------------------------

def _package(root, task_id="fixture-ctf-ret2win"):
    (root / "public").mkdir(parents=True)
    (root / "public" / "README.md").write_text("# ret2win\nConnect to the target and obtain the flag.")
    (root / "task.yaml").write_text(f"task_id: {task_id}\ngrader_mode: flag\n")
    (root / "MANIFEST.json").write_text(json.dumps({"task_id": task_id, "grader_mode": "flag"}))


def _catalog(tmp_path):
    _package(tmp_path / "pkg")
    return import_catalog([tmp_path / "pkg"], evidence={"runtime_quality_gate": {"healthy": True}})


def _sample(n_prompt, n_gen, version="pv", **meta):
    tokens = list(range(n_prompt + n_gen))
    span = SimpleNamespace(version=version, abs_start=n_prompt, abs_end=n_prompt + n_gen)
    return SimpleNamespace(tokens=tokens, loss_mask=[1] * n_gen, rollout_log_probs=[-0.1] * n_gen,
                           response_length=n_gen, weight_versions=[SimpleNamespace(spans=[span])],
                           metadata=dict(meta), status=None)


# --- breakpoint 2: catalog -> rollout prompt data ----------------------------

def test_catalog_prompt_rows_feed_miles_prompt_data(tmp_path):
    catalog = _catalog(tmp_path)
    rows = prompt_rows(catalog)
    assert len(rows) == 1
    row = rows[0]
    md = row["metadata"]
    # harness task_identity reads metadata.task_id; the RLF producer reads rlf_task_id.
    assert md["task_id"] == "fixture-ctf-ret2win" == md["rlf_task_id"]
    assert md["grader_mode"] == "flag" and md["benchmark"] == "rlf"
    assert md["package_digest"] == catalog.get("fixture-ctf-ret2win").package_digest
    assert "obtain the flag" in md["prompt"] and row["prompt"] == md["prompt"]
    out = write_prompt_data(catalog, tmp_path / "prompt.jsonl", repeat=2)
    loaded = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(loaded) == 2 and loaded[0] == row


def test_catalog_refuses_fabricated_statement_and_unknown_task(tmp_path):
    root = tmp_path / "pkg"
    _package(root)
    (root / "public" / "README.md").unlink()
    catalog = import_catalog([root], evidence={"runtime_quality_gate": {"healthy": True}})
    with pytest.raises(AdmissionError, match="task statement"):
        prompt_rows(catalog)
    with pytest.raises(KeyError):
        prompt_rows(catalog, ["no-such-task"])


def test_catalog_episode_produces_rlf_identity(tmp_path):
    catalog = _catalog(tmp_path)
    lease = catalog.episode("fixture-ctf-ret2win")
    assert isinstance(lease, EpisodeLease) and lease.task_id == "fixture-ctf-ret2win"
    assert lease.episode_id and lease.trajectory_id
    other = catalog.episode("fixture-ctf-ret2win")
    assert lease.episode_id != other.episode_id  # episodes are never shared
    identity = rlf_identity(lease, policy_id="pv")
    assert identity == {"rlf_task_id": "fixture-ctf-ret2win", "rlf_episode_id": lease.episode_id,
                        "rlf_trajectory_id": lease.trajectory_id, "policy_id": "pv"}


# --- breakpoint 1: identity producer on the real generate path ---------------

def test_bookkeeping_produces_rlf_identity_for_catalog_rows():
    inp = SimpleNamespace(group_index=3, index=5,
                          metadata={"rlf_task_id": "fixture-ctf-ret2win"})
    chains = [_sample(4, 6), _sample(4, 6)]
    generate_wrapper.apply_trajectory_bookkeeping(inp, chains, expected_version="pv")
    md = [c.metadata for c in chains]
    assert all(m["rlf_task_id"] == "fixture-ctf-ret2win" for m in md)
    assert len({m["rlf_episode_id"] for m in md}) == 1  # one episode per rollout
    assert len({m["rlf_trajectory_id"] for m in md}) == 1
    assert all(m["policy_id"] == "pv" for m in md)
    assert all(c.status is None for c in chains)  # versions match: no abort


def test_bookkeeping_keeps_existing_identity_and_ignores_non_rlf_rows():
    inp = SimpleNamespace(group_index=3, index=5, metadata={
        "rlf_task_id": "fixture-ctf-ret2win", "rlf_episode_id": "ep-fixed",
        "rlf_trajectory_id": "traj-fixed", "policy_id": "pv-old"})
    chain = _sample(4, 6)
    generate_wrapper.apply_trajectory_bookkeeping(inp, [chain], expected_version="pv")
    assert chain.metadata["rlf_episode_id"] == "ep-fixed"
    assert chain.metadata["rlf_trajectory_id"] == "traj-fixed"
    assert chain.metadata["policy_id"] == "pv-old"  # never overwritten
    plain_inp = SimpleNamespace(group_index=3, index=5, metadata={"task_id": "tb2-task"})
    plain = _sample(4, 6)
    generate_wrapper.apply_trajectory_bookkeeping(plain_inp, [plain], expected_version="pv")
    assert not any(k in plain.metadata for k in generate_wrapper.RLF_IDENTITY_KEYS)


# --- breakpoint: rlf_modal_provider factory contract --------------------------

def test_rlf_modal_provider_matches_preflight_factory_contract(monkeypatch):
    from yeto.cloud import modal_reward_env as mre
    from yeto.rl.rlf import modal as rlf_modal

    class FakeSandbox:
        object_id = "sb-fake"

        def terminate(self):
            pass

    calls = {}

    def fake_acquire_vm(task_id, episode_id, *, config=None, image=None, trajectory_id=None):
        calls.update(task_id=task_id, episode_id=episode_id, trajectory_id=trajectory_id)
        return rlf_modal.RlfVmLease(FakeSandbox(), task_id=task_id, episode_id=episode_id,
                                    trajectory_id=trajectory_id)

    monkeypatch.setattr(rlf_modal, "acquire_vm", fake_acquire_vm)
    factory = mre.rlf_modal_provider
    provider = factory(SimpleNamespace())  # preflight calls factory(miles_args)
    assert hasattr(provider, "acquire")  # resolve_environment_provider's only check
    lease = asyncio.run(provider.acquire("fixture-ctf-ret2win", "traj-1"))
    assert lease.task_id == "fixture-ctf-ret2win"
    assert lease.episode_id and lease.trajectory_id == "traj-1"
    assert calls["trajectory_id"] == "traj-1" and calls["episode_id"] == lease.episode_id
    lease.teardown()  # idempotent cleanup path stays intact
    assert lease.closed


# --- compaction profile registration (5.7) ------------------------------------

def test_qwen3_cyber_profile_is_registered_and_selectable():
    assert "qwen3_cyber" in cb.compaction_profiles()
    cfg = cb.compaction_config(24_576, env={cb.PROFILE_ENV: "qwen3_cyber"})
    assert (cfg.context_budget, cfg.t_comp, cfg.max_compactions) == (24_576, 4_096, 3)
    with pytest.raises(harness.CodexHarnessError, match="context_budget"):
        cb.compaction_config(32_768, env={cb.PROFILE_ENV: "qwen3_cyber"})
    with pytest.raises(harness.CodexHarnessError, match="not a registered"):
        cb.compaction_config(24_576, env={cb.PROFILE_ENV: "no-such-profile"})


def test_compaction_config_default_derivation_unchanged():
    cfg = cb.compaction_config(32_768, env={})
    assert cfg.context_budget == 32_768 and cfg.t_comp == T_COMP
    cfg = cb.compaction_config(32_768, env={cb.T_COMP_ENV: "4096"})
    assert cfg.t_comp == 4_096
    with pytest.raises(harness.CodexHarnessError):
        cb.compaction_config(32_768, env={cb.T_COMP_ENV: "x"})


# --- trusted finish: RLF grade interface (breakpoint 5, interface only) -------

def test_trusted_grade_from_evaluation_prefers_rlf_and_falls_back():
    assert trusted_grade_from_evaluation({"passed": True, "testsh_rc": 0}) is None  # TB2 HMAC fallback
    payload = {"status": "completed", "reward": 1.0, "passed": True,
               "grader_evidence": {"flag": "matched"}}
    grade = trusted_grade_from_evaluation({"rlf_grade": payload, "log": "..."})
    assert isinstance(grade, TrustedGrade) and grade.passed and grade.reward == 1.0
    assert grade.grader_evidence == {"flag": "matched"}
    direct = trusted_grade_from_evaluation(payload)  # a bare full-grade mapping also parses
    assert direct is not None and direct.status == "completed"
    with pytest.raises(RolloutRejected, match="malformed"):
        trusted_grade_from_evaluation({"rlf_grade": {"status": "completed"}})


# --- RLF metadata survives compaction segment splitting ------------------------

@dataclasses.dataclass(frozen=True)
class _Out:
    samples: Any


def test_compaction_segments_preserve_rlf_identity_and_reward_metadata():
    seg0_meta = {
        "expected_policy_version": "pv", "session_server_id": "10.0.0.1:9",
        "exit_status": "completed",
        "rlf_task_id": "fixture-ctf-ret2win", "rlf_episode_id": "ep-1",
        "rlf_trajectory_id": "traj-1", "policy_id": "pv",
        "success": True, "reward": 1.0, "grader_evidence": {"flag": "matched"},
        generate_wrapper.SESSIONS_KEY: ["a", "b"],
        "agent_metrics": {cb.METRICS_KEY: {"compactions": 1}},
    }
    seg0 = _sample(10, 30, **seg0_meta)
    s1 = _sample(8, 20)
    inp = SimpleNamespace(sample=SimpleNamespace(group_index=7, index=11,
                                                 metadata={"expected_policy_version": "pv"}))

    async def upstream(_input):
        return _Out(samples=seg0)

    async def collect(_input, router, sid):
        return (([s1], {"seg": 1}) if sid == "a" else ([], {}))

    out = asyncio.run(generate_wrapper.generate(inp, upstream=upstream, collect=collect))
    assert out.samples == [seg0, s1]
    for sample in out.samples:
        md = sample.metadata
        assert md["rlf_task_id"] == "fixture-ctf-ret2win"
        assert md["rlf_episode_id"] == "ep-1" and md["rlf_trajectory_id"] == "traj-1"
        assert md["policy_id"] == "pv"
        assert md["success"] is True and md["grader_evidence"] == {"flag": "matched"}
    assert [s.metadata["segment_index"] for s in out.samples] == [0, 1]
    assert s1.metadata["seg"] == 1
