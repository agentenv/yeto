"""Replay of real async GRPO tapes (sanitized: numbers and enum strings only,
times shifted): a 2-trainer / 2-sampler buffer run on Modal (one trainer was
stopped mid-run) and an NC engine sampler that writes no island_id."""

from __future__ import annotations

from pathlib import Path

import pytest

from yeto.dashboard.export import export_view, render_page
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all

FX = Path(__file__).parent / "fixtures" / "dashboard" / "async_grpo_replay"


@pytest.fixture(scope="module")
def p7():
    r = Reducer(run="replay")
    load_all(r, [str(FX / "p7")])
    return export_view(r, generated_at=0.0)


def _card(view, iid):
    return next(c for c in view["overview"]["islands"] if c["id"] == iid)


def test_roles_and_statuses(p7):
    assert {c["id"]: c["role"] for c in p7["overview"]["islands"]} == {
        "0": "trainer", "1": "trainer", "2": "sampler", "3": "sampler"}
    assert _card(p7, "0")["status"] == "done"
    assert _card(p7, "1")["status"] == "left"  # stopped mid-run (island_leave STOPPED)
    assert _card(p7, "2")["status"] == "done" and _card(p7, "3")["status"] == "done"
    assert p7["overview"]["alerts"] == []
    assert p7["overview"]["global_status"]["level"] == "ok"


@pytest.mark.parametrize("key,n", [("kl_tr", 24), ("tis_clip", 24), ("resp_len", 24), ("tok_s", 24)])
def test_trainer_series(p7, key, n):
    assert len(p7["overview"]["series"]["0"][key]) == n


def test_trainer_rounds_phases_and_logprob(p7):
    rounds = p7["page"]["islands"]["0"]["rounds"]
    assert len(rounds) == 24
    assert all(r["logprob_diff"] is not None for r in rounds)
    assert all({"R", "T"} <= set(r["dur"]) for r in rounds)  # derived from rl_driver_phase
    assert all(r["resp_derived"] for r in rounds)


def test_old_tape_has_no_reward_or_grad_norm(p7):
    """This run predates the buffer trainer's reward / grad_norm fields: the page
    says n/a for them instead of inventing a value."""
    s = p7["overview"]["series"]["0"]
    assert s["reward"] == [] and s["grad_norm"] == []


def test_samplers_backlog_releases(p7):
    a = p7["page"]["async"]
    assert a["samplers"]["2"]["n_rounds"] == 40 and a["samplers"]["3"]["n_rounds"] == 39
    assert a["samplers"]["2"]["lag_hist"] == {"0": 40}
    assert a["trainers"]["0"]["age_hist"] == {"0": 40, "1": 56, "2": 96}
    assert a["trainers"]["0"]["stale_groups"] == 25
    b = a["backlog"]
    assert b["written"] == 632 and b["read"] == 401 and b["points"][-1][3] == 231
    rel = a["releases"]
    assert [r["policy_version"] for r in rel] == [0, 1, 2, 3, 4, 5]
    assert all(set(r["loads"]) == {"2", "3"} for r in rel)
    assert all(0 < x["latency_s"] < 120 for r in rel for x in r["loads"].values())


def test_distribution_inputs(p7):
    rounds = p7["page"]["islands"]["0"]["rounds"]
    assert all(r["dur"].get("R") for r in rounds)
    assert all(r.get("generate_s") for r in p7["page"]["async"]["samplers"]["2"]["rounds"])


def test_gpu_memory_note(p7):
    assert "--rl-resource-sample-interval" in p7["page"]["islands"]["0"]["gpu_mem_note"]


def test_page_renders(p7):
    html = render_page(p7)
    assert 'id="async"' in html and 'id="nojs"' in html


def test_nc_sampler_without_island_id():
    r = Reducer()
    load_all(r, [str(FX / "nc")])
    view = r.page_view()
    (card,) = view["overview"]["islands"]
    assert card["id"] == "nc-s1" and card["role"] == "sampler"
    assert card["status"] == "waiting" and card["waiting_reason"].startswith("backlog full")
    s = view["async"]["samplers"]["nc-s1"]
    assert s["n_rounds"] == 5 and s["zero_variance_dropped"] == 5 and s["truncated"] == 2
    assert "pynvml" in view["islands"]["nc-s1"]["gpu_mem_note"]
