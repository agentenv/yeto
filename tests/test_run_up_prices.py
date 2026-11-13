"""``yeto run up`` preflight: USD-priced pools need a real price under a spend cap,
and ``budget.usd: null`` means no cap."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from yeto.rl import up, up_consistency

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "KIMI_K31_ASYNC_RL_GUIDE.md"


@pytest.fixture
def run_with(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    nc_cfg = tmp_path / "pool.json"
    nc_cfg.write_text(json.dumps({"shard_base": "nc"}))

    def build(samplers, budget):
        cfg = {"run": "r1", "store": "s3://b/runs/r1", "samplers": samplers,
               "trainer": {"argv": ["--gpu", "modal:2x8xb300"], "usd_per_h": 100}, "budget": budget}
        return up.Run(cfg)

    build.nc = {"name": "nc", "provider": "nc", "config": str(nc_cfg)}
    return build


def _modal(price=None, **entry):
    section = {"kind": "modal", "image": "img", "gpu": "B300", "gpus": 8}
    if price is not None:
        section["price_usd_h"] = price
    return {"name": "modal", "provider": section, "shard_base": "m", "slots": [1], **entry}


def _daytona(price=None):
    section = {"kind": "daytona", "image": "img", "gpus": 8}
    if price is not None:
        section["price_usd_h"] = price
    return {"name": "daytona", "provider": section, "shard_base": "d", "slots": [1]}


@pytest.mark.parametrize("price", [None, 0, -1, "<USD per island hour: required>"])
def test_a_capped_run_refuses_an_unpriced_modal_pool(run_with, price):
    problem = run_with([run_with.nc, _modal(price)], {"usd": 450}).budget_problem()
    assert problem is not None
    assert "pool modal (modal" in problem and "provider.price_usd_h" in problem
    assert "modal.com/pricing" in problem


def test_a_capped_run_refuses_an_unpriced_daytona_pool(run_with):
    problem = run_with([_daytona()], {"usd": 450}).budget_problem()
    assert problem is not None and "pool daytona (daytona" in problem and "Daytona" in problem


def test_an_entry_price_of_zero_overrides_the_provider_price_and_is_refused(run_with):
    problem = run_with([_modal(70.0, usd_per_h=0)], {"usd": 450}).budget_problem()
    assert problem is not None and "samplers[modal].usd_per_h" in problem


@pytest.mark.parametrize("pools", [
    [_modal(70.0)],
    [_modal(None, usd_per_h=70)],
    [_daytona(40.0)],
])
def test_a_priced_pool_passes(run_with, pools):
    assert run_with(pools, {"usd": 450}).budget_problem() is None


def test_nc_is_not_affected(run_with):
    assert run_with([run_with.nc], {"usd": 450}).budget_problem() is None


@pytest.mark.parametrize("budget", [{}, {"usd": None}])
def test_without_a_cap_an_unpriced_pool_passes(run_with, budget):
    assert run_with([_modal()], budget).budget_problem() is None


def test_budget_usd_null_is_no_cap_everywhere(run_with, capsys):
    run = run_with([run_with.nc], {"usd": None, "wall_h": 3})
    assert run.budget_problem() is None  # no TypeError
    assert "budget.usd is null: no spend cap" in capsys.readouterr().out
    assert up_consistency.budget_usd(run.cfg["budget"]) == up_consistency.NO_CAP_USD
    assert run.breach(trainer=False) is None


def test_budget_usd_not_a_number_is_a_clear_problem(run_with):
    problem = run_with([run_with.nc], {"usd": "lots"}).budget_problem()
    assert problem is not None and "budget.usd must be a number" in problem


def _guide_samplers(tmp_path: Path) -> list[dict]:
    text = GUIDE.read_text()
    block = re.search(r'```json\n("samplers": \[.*?\n\])\n```', text, re.S).group(1)
    nc_cfg = tmp_path / "guide-pool.json"
    nc_cfg.write_text(json.dumps({"shard_base": "nc"}))
    return json.loads("{" + block.replace("<pool.json>", str(nc_cfg)) + "}")["samplers"]


def test_the_guide_modal_example_is_refused_until_a_real_price_is_filled(run_with, tmp_path):
    samplers = _guide_samplers(tmp_path)
    modal = next(s for s in samplers if s["name"] == "modal")
    assert modal["provider"]["price_usd_h"] == "<USD per island hour: required>"
    assert "usd_per_h" not in modal  # an entry price would override the provider's
    assert run_with(samplers, {"usd": 450}).budget_problem() is not None
    modal["provider"]["price_usd_h"] = 70.0
    assert run_with(samplers, {"usd": 450}).budget_problem() is None
