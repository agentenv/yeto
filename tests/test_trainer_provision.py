"""Trainer first launch through source adapters (yeto.launch.trainer_provision).

The launcher's render of every source is held byte for byte by
tests/test_trainer_launch_render.py; these tests cover the adapter registry and the
prices a trainer's machines are recorded with."""

from __future__ import annotations

import types

import pytest

from yeto.cost import billing, prices
from yeto.cost.ledger import Resource
from yeto.gpu_spec import ClusterSpec, parse_gpu_spec
from yeto.launch import trainer_provision as tp
from yeto.launch.pool.sky_provider import SkyProvider
from yeto.launch.rollout.offers import offers_from_config


def _prov(specs, names, **kw):
    args = types.SimpleNamespace(cluster_prefix="run-t", training_mode="rl", spot=False)
    return tp.TrainerProvision(args, sky=None, specs=specs, names=names, no_sync=True, **kw)


def test_each_island_goes_to_the_adapter_of_its_source():
    specs = [ClusterSpec("aws", "us-east-1", 1, 8, "H100"), ClusterSpec("verda", None, 1, 8, "B300"),
             ClusterSpec("modal", None, 1, 8, "H100"), ClusterSpec("ssh", None, 2, 8, "H100"),
             ClusterSpec("ssh", None, 1, 8, "H100")]
    names = [f"run-t-l{m}" for m in range(len(specs))]
    prov = _prov(specs, names, ssh_names={"run-t-l3"})
    kinds = [prov.source(n, s).kind for n, s in zip(names, specs)]
    # a resident ssh: island is the inventory's; any other ssh: island (a sky ssh node pool) is sky's
    assert kinds == ["sky", "verda", "modal", "ssh", "sky"]
    assert [prov.source(n, s).ordered for n, s in zip(names, specs)] == [True, True, False, False, True]


def test_a_new_source_is_one_adapter_and_no_launcher_branch():
    calls = []

    class Acme(tp.Source):
        kind = "acme"
        ordered = True

        def order(self, prov, island):
            calls.append(("order", island.name))
            return "ticket-1"

        def wait(self, prov, island, handle):
            calls.append(("wait", handle))
            return (1, types.SimpleNamespace(head_ip="10.0.0.9"))

        def ledger_attrs(self, prov, island):
            return {"cloud": "acme"}

    tp.register(Acme())
    try:
        spec = ClusterSpec("acme", None, 1, 8, "H100")
        prov = _prov([spec], ["run-t-l0"])
        island = prov.island("run-t-l0", 0, spec, None, task="the task")
        prov.add(island)
        assert island.source == "acme" and prov.ordered(island)
        handle = prov.order(island)
        assert prov.wait(island, handle)[0] == 1 and calls == [("order", "run-t-l0"), ("wait", "ticket-1")]
        assert prov.ledger_attrs(island) == {"cloud": "acme"}
        assert [a.kind for a in tp._SOURCES][-1] == "sky"  # still the default, last
    finally:
        tp._SOURCES[:] = [a for a in tp._SOURCES if a.kind != "acme"]


@pytest.mark.parametrize("spot", [True, False])
@pytest.mark.parametrize("nodes", [1, 2])
def test_a_sky_trainer_and_a_sky_sampler_on_the_same_card_cost_the_same(spot, nodes, tmp_path):
    """The trainer's sky cluster is recorded with what prices it (cloud, card, region,
    spotness); the run's cost reads the unified price table row a sky sampler's offer
    reads, per node."""
    spec = ClusterSpec("nebius", "eu-north1", nodes, 8, "H100")
    task = types.SimpleNamespace(resources={"use_spot": spot}, envs={})
    prov = _prov([spec], ["run-t-l0"])
    island = prov.island("run-t-l0", 0, spec, None, task)
    attrs = prov.ledger_attrs(island)
    assert attrs == {"role": "trainer", "cloud": "nebius", "gpu": "H100", "gpus": 8 * nodes, "num_nodes": nodes,
                     "region": "eu-north1", "use_spot": spot}
    trainer = billing.resource_costs([Resource("sky_cluster", "run-t-l0", "run-t-l0", 0.0, attrs)], 3600.0)
    (item,) = trainer.items
    assert item.priced and item.note == "unified price table (list price)"

    offer = offers_from_config({"resources": {"infra": "nebius/eu-north1", "accelerators": "H100:8",
                                              "use_spot": spot}})
    sampler = SkyProvider("run-s", offers=offer, slots=[1], sky=object(), record=tmp_path / "sky.json")
    row = prices.lookup("nebius", "H100", spot=spot, region="eu-north1")
    assert sampler.hourly_cost() == pytest.approx(row.usd_h)
    assert item.rate_usd_per_hour == pytest.approx(nodes * sampler.hourly_cost())


def test_a_trainer_island_override_of_spot_is_what_its_cluster_is_priced_at():
    spec = ClusterSpec("nebius", "eu-north1", 1, 8, "H100")
    prov = _prov([spec], ["run-t-l0"])  # --spot off for the run
    task = types.SimpleNamespace(resources=[types.SimpleNamespace(use_spot=True)], envs={})
    assert prov.ledger_attrs(prov.island("run-t-l0", 0, spec, None, task))["use_spot"] is True
    unreadable = types.SimpleNamespace(envs={})
    assert prov.ledger_attrs(prov.island("run-t-l0", 0, spec, None, unreadable))["use_spot"] is False


# -- a budget needs a price (as a capped run refuses an unpriced sampler offer) ----------


def _launch_args(gpu, *extra):
    from test_rl_algorithm_provenance import _launcher_args

    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-single-island-no-sync", "--controller", "local", *extra), gpu=gpu)
    _prepare_rl_args(args)
    return args


UNPRICED = "aws:1xh100@us-east-1"  # no billing row in the unified price table
PRICED = "nebius:1x8xh100@eu-north1"  # the table's nebius H100:8 eu-north1 rows


@pytest.fixture
def _offline(monkeypatch, tmp_path):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")


@pytest.mark.usefixtures("_offline")
def test_a_budget_refuses_a_sky_trainer_no_price_covers_before_any_order():
    import yeto.launch_preflight as lp
    import yeto.launcher as launcher
    from yeto.config_errors import ConfigError

    capped = _launch_args(UNPRICED, "--budget-usd", "50")
    with pytest.raises(ConfigError, match="--budget-usd 50 cannot see .*aws.*--trainer-usd-per-hour"):
        lp.pre_cloud_checks(capped, threads=False)  # the launch path, before any order
    with pytest.raises(ConfigError, match="--trainer-usd-per-hour"):
        launcher.dry_run_plan(capped)
    lp.pre_cloud_checks(_launch_args(UNPRICED), threads=False)  # no budget: unchanged
    lp.pre_cloud_checks(_launch_args(PRICED, "--budget-usd", "50"), threads=False)  # the table prices it


@pytest.mark.usefixtures("_offline")
def test_a_stated_trainer_price_lifts_the_refusal_and_prices_the_cluster():
    import yeto.launch_preflight as lp

    stated = _launch_args("aws:2x1xh100@us-east-1", "--budget-usd", "50", "--trainer-usd-per-hour", "4.5")
    lp.pre_cloud_checks(stated, threads=False)
    spec = parse_gpu_spec(stated.gpu)[0]
    prov = tp.TrainerProvision(stated, sky=None, specs=[spec], names=["run-t-l0"], no_sync=True)
    attrs = prov.ledger_attrs(prov.island("run-t-l0", 0, spec, None, types.SimpleNamespace(envs={})))
    assert attrs["usd_per_hour"] == pytest.approx(9.0)  # 4.5 per node-hour x 2 nodes
    (item,) = billing.resource_costs([Resource("sky_cluster", "run-t-l0", "run-t-l0", 0.0, attrs)], 3600.0).items
    assert item.priced and item.rate_usd_per_hour == pytest.approx(9.0)


def test_only_the_sky_islands_the_launcher_orders_are_checked():
    args = types.SimpleNamespace(training_mode="sft", spot=False, budget_usd=10.0)
    specs = parse_gpu_spec("modal:1xh100,verda:1xb300,ssh:1xh100,kubernetes:1xh100,aws:1xh100@us-east-1")
    (problem,) = tp.sky_price_problems(args, specs)
    assert problem.startswith("--gpu island 4 (aws")


def test_the_record_and_lock_hooks_run_before_each_order(monkeypatch):
    """Behaviour, not source text: on_order before every sky / Verda / Modal order,
    guard_order before the Modal deploy and the resident ssh start, on_call wired to the
    Modal island ops, adopted islands not ordered again."""
    import yeto.launch.sky_ops as sky_ops
    import yeto.modal_runner as modal_runner

    log = []

    class Ops:
        def __init__(self, name):
            self.app_name, self.ledger = name, None

        def define(self, cfg):
            log.append(("define", cfg.learner_id))

        def deploy(self):
            log.append(("deploy", self.app_name))

    class IslandOps:
        def __init__(self, ops):
            self.calls, self.on_call = {}, None

        def relaunch(self, cfg, name):
            log.append(("spawn", name))
            return f"call-{cfg.learner_id}"

    monkeypatch.setattr(sky_ops, "submit_island_launch", lambda sky, args, spec, task, name: log.append(
        ("submit", name)) or f"rid-{name}")
    monkeypatch.setattr(modal_runner, "ModalOps", Ops)
    monkeypatch.setattr(modal_runner, "ModalIslandOps", IslandOps)
    ssh_kw = {}
    monkeypatch.setattr(tp, "start_resident_ssh", lambda args, tasks, names, *rest, **kw: ssh_kw.update(kw) or log.append(
        ("ssh start", tuple(sorted(names)))) or "ssh-ops")
    on_call = lambda name, call_id: None  # noqa: E731
    specs = [ClusterSpec("aws", "us-east-1", 1, 8, "H100"), ClusterSpec("verda", None, 1, 8, "B300"),
             ClusterSpec("modal", None, 1, 8, "H100"), ClusterSpec("modal", None, 1, 8, "H100"),
             ClusterSpec("ssh", None, 1, 8, "H100")]
    names = ["r-l0", "r-l1", "r-l2", "r-l3", "r-l4"]
    prov = _prov(specs, names, ssh_names={"r-l4"}, on_order=lambda n: log.append(("on_order", n)),
                 guard_order=lambda n: log.append(("guard", n)), on_call=on_call,
                 adopted=lambda n: ("call-kept",) if n == "r-l3" else None)
    for m, (name, spec) in enumerate(zip(names, specs)):
        task = types.SimpleNamespace(learner_id=m, envs={})
        island = tp.Island(name, m, spec, None, task, prov.source(name, spec).kind)
        prov.add(island)
        if prov.ordered(island):
            prov.order(island)
    ops = prov.modal_app()
    island_ops = prov.spawn_modal(ops)
    prov.start_ssh()
    assert log == [("on_order", "r-l0"), ("submit", "r-l0"), ("on_order", "r-l1"),
                   ("define", 2), ("define", 3), ("guard", ops.app_name), ("deploy", ops.app_name),
                   ("on_order", "r-l2"), ("spawn", "r-l2"),  # r-l3 is adopted: kept, never spawned
                   ("guard", "r-l4"), ("ssh start", ("r-l4",))]
    assert island_ops.on_call is on_call and island_ops.calls == {"r-l3": "call-kept"}
    assert ssh_kw == {"island_index": {"r-l4": 4}, "num_islands": 5}  # its own checkpoint keys (spare nodes)
    assert prov.results["r-l2"] == ("call-2", None) and prov.results["r-l3"] == ("call-kept", None)


@pytest.mark.parametrize("bad", ["-5", "nan", "inf", "-inf", "four"])
def test_a_stated_trainer_price_is_a_finite_number_not_below_zero(bad, capsys):
    """A negative or NaN rate would keep the run's spend below --budget-usd forever."""
    from test_rl_algorithm_provenance import _launcher_args

    with pytest.raises(SystemExit):
        _launcher_args("ports", ("--trainer-usd-per-hour", bad), gpu=UNPRICED)
    assert "--trainer-usd-per-hour" in capsys.readouterr().err
    for ok in ("0", "4.5"):
        assert _launcher_args("ports", ("--trainer-usd-per-hour", ok), gpu=UNPRICED).trainer_usd_per_hour == float(ok)
