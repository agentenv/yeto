"""``yeto run up`` preflight: USD-priced pools need a real price under a spend cap,
and ``budget.usd: null`` means no cap."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.test_own_cluster_templates import kit  # noqa: F401 - the own-cluster kit fixture
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


def test_a_modal_pool_without_a_price_is_budgeted_at_the_unified_price_table(run_with):
    """No price_usd_h: the pool's B300 x 8 and its function's default 64 cores / 1024 GiB at the
    Modal list prices of the unified price table (yeto.cost.prices: what a Modal trainer of the
    same request is billed at), not $0 and not the GPU line alone."""
    from yeto.cost import prices

    run = run_with([run_with.nc, _modal(None)], {"usd": 450})
    assert run.budget_problem() is None
    assert run.node_usd_per_h(1) == pytest.approx(
        prices.modal_container_usd_h("B300", 8, cpu=64, memory_gib=1024)) == pytest.approx(68.0 + 3.01824 + 8.183808)


@pytest.mark.parametrize("price", [0, -1, "<USD per island hour: required>"])
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


# -- the monitor's spend never turns NaN or drops on a bad price (run.budget stays armed) -----

def _billed_run(run_with, monkeypatch, tmp_path, pools, budget=None, members=None):
    """A run (no fleet: the customer's spend_usd path) whose every pool has one member billed
    for the last hour, and no trainer hours."""
    import time  # noqa: PLC0415

    state = tmp_path / "nc"
    state.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(state))
    run = run_with(pools, budget or {"usd": 10_000})
    now = time.time()
    run.state.update({"run_t0": now - 3600, "ups": [{"pid": 1, "start": now - 3600, "seen": now, "end": None}]})
    for ref in run.pools:
        doc = {"members": members if members is not None else [{"state": "ready", "granted": now - 3600}]}
        (state / f"pool-{ref.load()['shard_base']}.json").write_text(json.dumps(doc))
    assert run.fleet_plan() is None
    return run


def _warnings(capsys, text):
    return sum(text in line for line in capsys.readouterr().out.splitlines())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0])
def test_a_bad_pool_usd_per_h_is_priced_from_its_config(run_with, monkeypatch, tmp_path, capsys, bad):
    """A usd_per_h that is NaN, inf or negative is not a price (the run config refuses one; a
    pool ref built past that check, e.g. by a reconcile): the pool counts at its provider
    config's price (as without the override), the spend stays finite, one WARNING per up."""
    import dataclasses  # noqa: PLC0415
    import math  # noqa: PLC0415

    import time  # noqa: PLC0415

    from yeto.launch.pool.money import to_usd  # noqa: PLC0415

    now = time.time()
    run = _billed_run(run_with, monkeypatch, tmp_path, [_modal(70.0)],
                      members=[{"state": "ready", "granted": now - 3600},
                               {"state": "ready", "granted": now - 3600, "hourly_ucr": 10.0}])
    run.pools = [dataclasses.replace(run.pools[0], usd_per_h=bad)]
    assert run.node_usd_per_h(0) == 70.0
    for _ in range(2):  # each member at its own offer, as a pool without the override
        spend = run.spend_usd()
        assert math.isfinite(spend) and spend == pytest.approx(70.0 + to_usd(10.0, "ucr"), rel=1e-3)
    assert _warnings(capsys, "usd_per_h") == 1
    run.cfg["budget"]["usd"] = 50  # run.budget still trips
    assert run.breach(trainer=False) is not None


def test_an_unset_pool_usd_per_h_is_priced_from_its_config_as_before(run_with, monkeypatch, tmp_path, capsys):
    """None: the provider config's price, each member at its own offer (a member without one
    at the pool's price); no WARNING. The number is the plain sum of the pools' spends."""
    import time  # noqa: PLC0415

    now = time.time()
    run = _billed_run(run_with, monkeypatch, tmp_path, [_modal(70.0)],
                      members=[{"state": "ready", "granted": now - 3600},
                               {"state": "ready", "granted": now - 3600, "hourly_ucr": None}])
    assert run.pools[0].usd_per_h is None and run.node_usd_per_h(0) == 70.0
    assert run.spend_usd() == pytest.approx(140.0, rel=1e-3)
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, "x", True])
def test_a_members_bad_offer_rate_counts_at_the_pools_price(run_with, monkeypatch, tmp_path, capsys, bad):
    import math  # noqa: PLC0415
    import time  # noqa: PLC0415

    now = time.time()
    run = _billed_run(run_with, monkeypatch, tmp_path, [_modal(70.0)],
                      members=[{"state": "ready", "granted": now - 3600, "hourly_ucr": bad}])
    for _ in range(2):
        spend = run.spend_usd()
        assert math.isfinite(spend) and spend == pytest.approx(70.0, rel=1e-3)
    assert _warnings(capsys, "hourly_ucr") == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_a_bad_provider_price_is_priced_as_a_pool_that_states_none(run_with, monkeypatch, tmp_path, capsys, bad):
    """The provider config's price_usd_h NaN, inf or negative: the price of the same pool
    without one (a Modal pool's gpu x gpus at the unified price table), one WARNING; a pool
    the table cannot price either (sky without offers) counts at budget.nc_usd_per_h."""
    table = run_with([_modal(None)], {"usd": 10_000}).node_usd_per_h(0)
    run = _billed_run(run_with, monkeypatch, tmp_path, [_modal(bad)], budget={"usd": 10_000, "nc_usd_per_h": 9.0})
    for _ in range(2):
        assert run.node_usd_per_h(0) == table and run.spend_usd() == pytest.approx(table, rel=1e-3)
    assert _warnings(capsys, "its provider config's price") == 1
    sky = {"name": "sky", "provider": {"kind": "sky", "price_usd_h": bad}, "shard_base": "s", "slots": [1]}
    run = _billed_run(run_with, monkeypatch, tmp_path / "sky", [sky], budget={"usd": 10_000, "nc_usd_per_h": 9.0})
    assert run.node_usd_per_h(0) == 9.0 and run.spend_usd() == pytest.approx(9.0, rel=1e-3)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, "x"])
def test_with_no_price_at_all_a_node_counts_at_the_mi355x_rate_not_zero(run_with, monkeypatch, tmp_path, capsys,
                                                                         bad):
    """No usd_per_h, a National Compute pool and a budget.nc_usd_per_h that is not a price:
    the unified table's MI355X node rate (an unpriced node's rate), never NaN and never 0."""
    run = _billed_run(run_with, monkeypatch, tmp_path, [run_with.nc], budget={"usd": 10_000, "nc_usd_per_h": bad})
    rate = up._nc_node_usd_h()
    assert rate > 0 and run.node_usd_per_h(0) == rate
    for _ in range(2):
        assert run.spend_usd() == pytest.approx(rate, rel=1e-3)
    assert _warnings(capsys, "budget.nc_usd_per_h") == 1


def test_the_spend_of_good_prices_is_the_plain_sum_of_the_pools(run_with, monkeypatch, tmp_path):
    """The clamp changes nothing for real prices: the same float as the sum of each pool's spend."""
    run = _billed_run(run_with, monkeypatch, tmp_path, [run_with.nc, _modal(70.0, usd_per_h=12.5)],
                      budget={"usd": 10_000, "nc_usd_per_h": 3.25})
    frozen = up.time.time()
    monkeypatch.setattr(up.time, "time", lambda: frozen)
    assert run.spend_usd() == sum(run.node_spend_usd(i, frozen) for i in range(2)) + run.trainer_hours(frozen) * 100


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0, None])
def test_a_pool_spend_that_is_not_an_amount_counts_its_hours_at_the_mi355x_rate(run_with, monkeypatch, tmp_path,
                                                                                 capsys, bad):
    """Past the price checks (a bug): the pools' sum is clamped as the fleet's, and a pool whose
    spend is not an amount counts its billed hours at the MI355X rate, never NaN, inf or 0."""
    import math  # noqa: PLC0415

    run = _billed_run(run_with, monkeypatch, tmp_path, [run_with.nc, _modal(70.0)], budget={"usd": 10_000})
    real = up.Run.node_spend_usd
    monkeypatch.setattr(up.Run, "node_spend_usd", lambda self, i=0, now=None: bad if i == 1 else real(self, i, now))
    rate = up._nc_node_usd_h()
    for _ in range(2):
        spend = run.spend_usd()
        assert math.isfinite(spend) and spend == pytest.approx(2 * rate, rel=1e-3)
    assert _warnings(capsys, "is not an amount") == 1


# -- budget.usd 0: refused only when something is priced (REV-903 L1) -----------------

def _ssh(**entry):
    section = {"kind": "ssh", "inventory": "inventory.yaml", "nodes": ["own-smp-0"]}
    return {"name": "own", "provider": section, "shard_base": "o", "slots": [1], **entry}


def _own(run_with, samplers, budget, trainer_usd=0):
    run = run_with(samplers, budget)
    run.cfg["trainer"] = {"argv": ["--gpu", "ssh:1x8xb300"], "usd_per_h": trainer_usd}
    return run


@pytest.mark.parametrize("budget", [{"usd": 0}, {"usd": 0, "nc_usd_per_h": 24}])
def test_a_run_all_on_own_nodes_may_set_budget_usd_zero(run_with, budget):
    """An ssh trainer and an ssh pool bill nothing: no NC pool, so the NC node rate (the
    price table's, or a budget.nc_usd_per_h left in the template) is not a price here."""
    run = _own(run_with, [_ssh()], budget)
    assert run.pools[0].kind() == "ssh" and run.node_usd_per_h(0) == 0
    assert run.budget_problem() is None


def test_an_own_run_with_a_priced_trainer_or_pool_keeps_the_refusal(run_with):
    problem = _own(run_with, [_ssh()], {"usd": 0}, trainer_usd=8).budget_problem()
    assert problem is not None and "budget.usd is 0 while prices are set" in problem
    problem = _own(run_with, [_ssh(), _modal(70.0)], {"usd": 0}).budget_problem()
    assert problem is not None and "budget.usd is 0 while prices are set" in problem
    problem = _own(run_with, [_ssh(usd_per_h=3)], {"usd": 0}).budget_problem()  # an own node with a declared price
    assert problem is not None and "budget.usd is 0 while prices are set" in problem


def test_with_an_nc_pool_budget_usd_zero_is_judged_as_before(run_with):
    """An NC pool: its node rate counts (the price table's when budget.nc_usd_per_h is
    unset), as on main; budget.nc_usd_per_h 0 with an unpriced trainer passes, as before."""
    problem = _own(run_with, [run_with.nc], {"usd": 0}).budget_problem()
    assert problem is not None and "budget.usd is 0 while prices are set" in problem
    problem = _own(run_with, [run_with.nc, _ssh()], {"usd": 0}).budget_problem()
    assert problem is not None and "budget.usd is 0 while prices are set" in problem
    assert _own(run_with, [run_with.nc], {"usd": 0, "nc_usd_per_h": 0}).budget_problem() is None
    problem = run_with([run_with.nc], {"usd": 0, "nc_usd_per_h": 0}).budget_problem()  # trainer $100/h
    assert problem is not None and "budget.usd is 0 while prices are set" in problem
    assert run_with([run_with.nc], {"usd": 450}).budget_problem() is None
    # an NC pool's entry price does not hide the NC node rate (as before)
    problem = _own(run_with, [{**run_with.nc, "usd_per_h": 0}], {"usd": 0}).budget_problem()
    assert problem is not None and "budget.usd is 0 while prices are set" in problem


def test_a_fleet_counts_the_nc_rate_as_before(run_with, monkeypatch):
    run = _own(run_with, [_ssh()], {"usd": 0})
    monkeypatch.setattr(up.Run, "fleet_plan", lambda self: object())  # a fleet: its islands price themselves
    problem = run.budget_problem()
    assert problem is not None and "budget.usd is 0 while prices are set" in problem


def test_an_unreadable_pool_config_counts_the_nc_rate_as_before(run_with, tmp_path):
    gone = tmp_path / "gone.json"
    gone.write_text(json.dumps({"shard_base": "g"}))
    run = _own(run_with, [{"name": "gone", "provider": "nc", "config": str(gone)}], {"usd": 0})
    Path(run.pools[0].config).unlink()  # read at run up, gone by the preflight
    assert run.pools[0].kind() == "?"
    problem = run.budget_problem()
    assert problem is not None and "budget.usd is 0 while prices are set" in problem


def test_the_own_cluster_kit_all_on_own_nodes_passes_with_budget_usd_zero(kit, capsys):
    """The own-cluster templates' ssh trainer with their ssh sampler pool, budget.usd 0:
    the dry run passes (it was refused for the NC node rate of a run without an NC pool)."""
    from tests.test_own_cluster_templates import _dry_run

    own = json.loads(kit.files["run.own-trainer-sky-sampler.json"].read_text())["trainer"]
    path = kit.files["run.own-sampler-pool.json"]
    cfg = json.loads(path.read_text())
    cfg["trainer"] = {**cfg["trainer"], "argv": own["argv"], "usd_per_h": 0}
    cfg["budget"] = {**cfg.get("budget", {}), "usd": 0}
    path.write_text(json.dumps(cfg))
    rc, out = _dry_run(kit, "run.own-sampler-pool.json", capsys, check_nodes=False)
    assert "while prices are set" not in out
    assert rc == 0, out.why
    assert "dry run: spend estimate $0.00/h = pools $0.00/h" in out
