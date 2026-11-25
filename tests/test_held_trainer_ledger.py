"""The trainer scheduler's held sky island in the run's cost ledger.

Each incarnation of a held sky trainer is a new cluster ``<island>-i<n>``
(:class:`~yeto.launch.trainer_island.SkyTrainerIslands`). It is a ``sky_cluster`` row
of the run's ledger with what prices it (the launcher's ``SkySource.ledger_attrs``):
created once its launch is submitted, released once its teardown is confirmed. The
cost guard (``--budget-usd``, the ``launch.budget`` breaker) and the dashboard read
that ledger, so they count the held trainer like any other island. (A held Modal
trainer is the run's Modal app and its calls; a held SSH trainer is an
``ssh_trainer`` row: both were in the ledger already.) No cloud resource is started."""

from __future__ import annotations

import json
import sys
import threading
import types

import pytest
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401  (autouse fixtures)
from test_rl_trainer_sky_island import _SkyOps, _Task

from yeto.cost import prices
from yeto.cost.billing import resource_costs
from yeto.cost.guard import EVENTS_FILE, CostGuard, GuardConfig
from yeto.cost.ledger import ResourceLedger
from yeto.dashboard.reducer import Reducer
from yeto.gpu_spec import ClusterSpec
from yeto.launch import trainer_provision as tp
from yeto.launch.trainer_island import SkyTrainerIslands

SPEC = ClusterSpec("nebius", "eu-north1", 1, 8, "H100")  # a unified price table row (billing grade)
T0 = 1_800_000_000.0


class _Clock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> float:
        return self.t


def _attrs(spec=SPEC):
    """What the launcher hands the held island: ``SkySource.ledger_attrs`` of it."""
    args = types.SimpleNamespace(cluster_prefix="r", training_mode="rl", spot=False)
    prov = tp.TrainerProvision(args, sky=None, specs=[spec], names=["r-l1"], no_sync=True)
    return prov.ledger_attrs(prov.island("r-l1", 0, spec, None, types.SimpleNamespace(envs={})))


def _islands(tmp_path, ops, clock, **kw):
    ledger = ResourceLedger(tmp_path / "resources.jsonl", clock=clock)
    islands = SkyTrainerIslands(_Task(), ops, name="r-l1", jobs_path=tmp_path / "jobs.json",
                                ledger=ledger, ledger_attrs=_attrs(), clock=clock, **kw)
    return islands, ledger


def _ended(job):
    job.provisioning.join(10)
    assert not job.provisioning.is_alive()


def test_each_incarnation_is_a_priced_row_from_its_submit_to_its_confirmed_teardown(tmp_path):
    clock = _Clock()
    ops = _SkyOps()
    ops.provision_error = RuntimeError("fake: no job")  # the provisioning thread ends at once
    islands, ledger = _islands(tmp_path, ops, clock)
    rate = prices.node_usd_h("nebius", "H100", 8, spot=False, region="eu-north1")
    assert rate is not None

    job0 = islands.launch(0, None)
    _ended(job0)
    (row,) = ledger.resources()
    assert (row.kind, row.id, row.open) == ("sky_cluster", "r-l1-i0", True)
    assert row.attrs == {**_attrs(), "island": "r-l1", "incarnation": 0}
    clock.t += 3600
    assert islands.release(job0) is True and ops.downed == ["r-l1-i0"]
    job1 = islands.launch(1, None)
    _ended(job1)
    clock.t += 1800

    summary = resource_costs(ledger.resources(), clock())
    by_id = {i.key.split(":", 1)[1]: i for i in summary.items}
    assert sorted(by_id) == ["r-l1-i0", "r-l1-i1"] and summary.unpriced == []
    assert not by_id["r-l1-i0"].open and by_id["r-l1-i1"].open
    assert by_id["r-l1-i0"].note == "unified price table (list price)"
    assert by_id["r-l1-i0"].usd == pytest.approx(rate)  # one hour, released
    assert by_id["r-l1-i1"].usd == pytest.approx(rate / 2)  # half an hour so far
    assert summary.total_usd == pytest.approx(1.5 * rate)


def test_the_cost_guard_and_the_dashboard_count_the_held_trainer(tmp_path):
    """``launch.budget`` (the guard's ``cost_tick``) and the dashboard's Breakers panel."""
    clock = _Clock()
    ops = _SkyOps()
    ops.provision_error = RuntimeError("fake: no job")
    islands, ledger = _islands(tmp_path, ops, clock)
    rate = prices.node_usd_h("nebius", "H100", 8, spot=False, region="eu-north1")
    events = tmp_path / EVENTS_FILE
    guard = CostGuard("r", GuardConfig(budget_usd=10 * rate), ledger=ledger, events_path=events, clock=clock,
                      log=lambda m: None, start=T0)
    _ended(islands.launch(0, None))
    clock.t += 7200
    summary = guard.tick()
    assert summary.total_usd == pytest.approx(2 * rate)
    (tick,) = [json.loads(line) for line in events.read_text().splitlines()
               if json.loads(line)["event"] == "cost_tick"]
    assert tick["open"] == ["sky_cluster:r-l1-i0"] and tick["unpriced"] == []
    assert tick["total_usd"] == pytest.approx(2 * rate, abs=1e-3)

    r = Reducer()
    assert r.feed_breaker(tick, source=str(events))
    gauge = r.breakers.status[("launch.budget", "")]
    assert gauge["value"] == pytest.approx(2 * rate, abs=1e-3) and gauge["ratio"] == pytest.approx(0.2, abs=1e-3)


def test_an_unconfirmed_teardown_keeps_the_row_open(tmp_path):
    """The cancel did not end the launch in time: the cluster may still come up, so the
    row stays open (the guard keeps counting it, ``yeto cleanup`` retries it)."""
    clock = _Clock()
    ops = _SkyOps()
    ops.provision_gate = threading.Event()
    islands, ledger = _islands(tmp_path, ops, clock, cancel_wait_s=0.01)
    job = islands.launch(0, None)
    try:
        assert islands.release(job) is False
        assert [r.open for r in ledger.resources()] == [True]
    finally:
        ops.provision_gate.set()
        job.provisioning.join(10)
    ops.down = lambda cluster: False  # the cloud does not confirm the teardown
    _ended(job)
    assert islands.release(job) is False and [r.open for r in ledger.resources()] == [True]


def test_a_dead_heads_incarnation_closes_when_the_next_head_releases_it(tmp_path):
    clock = _Clock()
    ops = _SkyOps()
    ops.provision_error = RuntimeError("fake: no job")
    islands, ledger = _islands(tmp_path, ops, clock)
    _ended(islands.launch(4, None))  # the head dies here: the row is open
    head2, _ = _islands(tmp_path, ops, clock)
    assert head2.release_incarnation(4) is True
    (row,) = ledger.resources()
    assert row.id == "r-l1-i4" and not row.open


def test_a_submit_that_raised_leaves_no_row(tmp_path):
    clock = _Clock()
    ops = _SkyOps()

    def submit(task, cluster):
        raise RuntimeError("sky API server unreachable")

    ops.submit = submit
    islands, ledger = _islands(tmp_path, ops, clock)
    with pytest.raises(RuntimeError, match="unreachable"):
        islands.launch(0, None)
    assert ledger.resources() == []  # nothing acquired (the lifecycle releases nothing either)


# -- the launcher hands the held island its ledger and price ------------------------------


def test_the_launcher_hands_a_held_sky_trainer_the_run_ledger_and_its_price(tmp_path, monkeypatch):
    """``launcher.run`` with a held sky trainer island: its binding records each
    incarnation on the run's ledger, priced from the unified table."""
    from test_rl_sampler_trainer_pair_launch import PAIR, STORE, _args
    from test_trainer_launch_render import _Provisioned, _setup

    import yeto.launch.trainer_island as ti
    import yeto.launcher as launcher
    from yeto.cloud import s3_store
    from yeto.cost.ledger import LEDGER_ENV
    from yeto.rl.engine.object_store import DirectoryObjectStore

    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    monkeypatch.setenv(s3_store.KEY_ID_ENV, "AKIDSTORE")
    monkeypatch.setenv(s3_store.SECRET_ENV, "store-secret")
    monkeypatch.delenv(LEDGER_ENV, raising=False)  # the launcher's own ledger reaches the binding, no fallback
    _setup(monkeypatch, tmp_path, [])
    del sys.modules["sky"].launch  # nothing is launched by the fleet: the held island is the scheduler's

    class Controller:
        downed_clusters: set = set()
        pending_downs: dict = {}

        def __init__(self, *a, **k):
            pass

        def run(self):
            raise AssertionError("not reached")

        def abort(self, error):
            pass

    held = {}

    def run_with_trainer_loop(args, run, *, island=None, abort=None, factory=None):
        held["args"], held["island"] = args, island
        raise _Provisioned

    monkeypatch.setattr(launcher, "FleetController", Controller)
    monkeypatch.setattr("yeto.launch.trainer_loop.run_with_trainer_loop", run_with_trainer_loop)
    base = tuple("s3://yeto-ckpt/run-a" if t == STORE else t for t in PAIR)
    args = _args(("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1"),
                 gpu="modal:2xh100,nebius:1x2xh100@eu-north1", base=base)
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    args.no_cost_guard = True  # the run's ledger, no guard thread
    with pytest.raises(_Provisioned):
        launcher.run(args)
    path = ResourceLedger.for_run(args.cluster_prefix).path

    island = held["island"]
    assert island.cloud == "nebius" and island.ledger.path == path
    spot = held["args"].spot  # an RL launch's islands are spot unless --no-spot; the price check reads the same
    assert island.ledger_attrs == {"role": "trainer", "cloud": "nebius", "gpu": "H100", "gpus": 2, "num_nodes": 1,
                                   "region": "eu-north1", "use_spot": spot}
    ops = _SkyOps()
    ops.provision_error = RuntimeError("fake: no job")
    monkeypatch.setattr(ti, "SkyIslandOps", lambda args, spec: ops)
    task = island.sky_task  # the fake sky Task: a real one has update_envs
    task.update_envs = lambda envs: task.envs.update(envs)
    binding = ti.sky_binding(held["args"], island, journal_dir=tmp_path / "journal",
                             files=DirectoryObjectStore(tmp_path / "store"))
    islands = binding.lifecycle(0).islands
    _ended(islands.launch(0, None))
    rows = [r for r in ResourceLedger(path).resources() if r.kind == "sky_cluster"]
    assert [(r.id, r.open, r.attrs["incarnation"]) for r in rows] == [(islands.cluster_name(0), True, 0)]
    (item,) = resource_costs(rows, rows[0].created_at + 3600).items
    assert item.usd == pytest.approx(prices.node_usd_h("nebius", "H100", 2, spot=spot, region="eu-north1"))


def test_a_budget_refuses_a_held_sky_trainer_no_price_covers(monkeypatch):
    """The held island's incarnations are priced like an ordered sky island's: with
    ``--budget-usd``, one no price covers is refused before any launch."""
    from test_rl_sampler_trainer_pair_launch import PAIR, STORE, _args

    from yeto.cloud import s3_store
    from yeto.config_errors import ConfigError
    from yeto.gpu_spec import parse_gpu_spec

    monkeypatch.setenv(s3_store.KEY_ID_ENV, "AKIDSTORE")  # the islands' store credential
    monkeypatch.setenv(s3_store.SECRET_ENV, "store-secret")
    base = tuple("s3://yeto-ckpt/run-a" if t == STORE else t for t in PAIR)
    extra = ("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1")
    unpriced = _args(extra + ("--budget-usd", "50"), gpu="modal:2xh100,aws:1x2xh100@us-east-1", base=base)
    assert unpriced.rl_trainer_island == 1
    with pytest.raises(ConfigError, match=r"--gpu island 1 \(aws.*--trainer-usd-per-hour"):
        tp.check_sky_prices(unpriced, parse_gpu_spec(unpriced.gpu))
    unpriced.budget_usd = None
    tp.check_sky_prices(unpriced, parse_gpu_spec(unpriced.gpu))  # no budget: unchanged
    priced = _args(extra + ("--budget-usd", "50"), gpu="modal:2xh100,nebius:1x2xh100@eu-north1", base=base)
    tp.check_sky_prices(priced, parse_gpu_spec(priced.gpu))
    on_modal = _args(extra + ("--budget-usd", "50"), gpu="aws:1x2xh100@us-east-1,modal:2xh100", base=base)
    assert tp.sky_price_problems(on_modal, parse_gpu_spec(on_modal.gpu)) == [
        "--gpu island 0 (" + str(parse_gpu_spec(on_modal.gpu)[0]) + ", spot): no unified price table row for "
        "aws H100 in us-east-1"]  # the held Modal trainer is its app's billing; the aws sampler is checked
