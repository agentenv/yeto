"""SkyProvider with the spot watchdog's machine management (offer rotation, start
timeout, adoption, price) over a fake sky module; the watchdog's candidate list
is the shared one (yeto.launch.rollout.offers)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.launch.nc import cli, provider as prov
from yeto.launch.nc.sky_provider import SkyProvider, from_config, offline_problems
from yeto.launch.rollout import offers
from yeto.rl.spot import watchdog as wd


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("YETO_RESOURCE_LEDGER", raising=False)


class FakeSky:
    def __init__(self):
        self.clusters, self.calls, self.fail_launch = {}, [], set()

    class Task:
        def __init__(self, name=None, run=None):
            self.name, self.run, self.resources, self.envs, self.secrets = name, run, None, {}, {}

        @classmethod
        def from_yaml(cls, path):
            task = cls(name=f"yaml:{path}")
            return task

        def set_resources(self, r):
            self.resources = r

        def update_envs(self, envs):
            self.envs.update(envs)

        def update_secrets(self, secrets):
            self.secrets.update(secrets)

    @staticmethod
    def Resources(**kw):  # noqa: N802 - sky's name
        return dict(kw)

    def launch(self, task, cluster_name, **kw):
        self.calls.append(("launch", cluster_name, task))
        if task.resources.get("infra") in self.fail_launch:
            raise RuntimeError("ResourcesUnavailableError: no capacity")
        self.clusters[cluster_name] = {"name": cluster_name, "status": "INIT",
                                       "handle": SimpleNamespace(head_ip=None, ssh_user="ubuntu")}

    def status(self, names):
        if names is None:
            return list(self.clusters.values())
        return [self.clusters[n] for n in names if n in self.clusters]

    def down(self, name):
        self.calls.append(("down", name))
        self.clusters.pop(name, None)

    @staticmethod
    def get(value):
        return value


def _cfg(**provider):
    return {"target": 1, "accounts": [1, 2, 3], "shard_base": "s", "run": "run-k",
            "provider": {"kind": "sky", **provider}}


def test_watchdog_reexports_the_shared_candidate_list():
    assert wd.Candidate is offers.Candidate and wd.candidates is offers.candidates
    assert wd.PRICES_8GPU is offers.PRICES_8GPU and wd.load_prices is offers.load_prices


def test_candidates_become_offers_in_the_watchdog_order():
    got = offers.offers_from_config({"candidates": {"gpus": ["H100", "H200"], "infra": "nebius"}})
    assert [o.key for o in got] == [c.key for c in offers.candidates()]
    first = got[0]
    assert first.resources == {"infra": "nebius/eu-north1", "accelerators": "H100:8", "use_spot": False}
    assert first.price_usd_h == 36.0
    spot = offers.offers_from_config({"candidates": {"spot_only": True}})
    assert all(o.resources["use_spot"] for o in spot) and spot[0].price_usd_h == 6.32
    with pytest.raises(ValueError, match="not both"):
        offers.offers_from_config({"offers": [], "candidates": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        offers.offers_from_config({"candidates": {"zones": ["x"]}})
    with pytest.raises(ValueError, match="resources"):
        offers.offers_from_config({"offers": [{"price_usd_h": 1}]})


def test_a_failed_launch_moves_to_the_next_offer_and_a_success_resets():
    sky = FakeSky()
    p = from_config(_cfg(offers=[{"resources": {"infra": "nebius/eu-north1", "accelerators": "H100:8",
                                                "use_spot": True}, "price_usd_h": 6.32},
                                 {"resources": {"infra": "nebius/eu-west1", "accelerators": "H200:8",
                                                "use_spot": True}, "price_usd_h": 11.2}]), "run-k", sky=sky)
    sky.fail_launch.add("nebius/eu-north1")
    with pytest.raises(RuntimeError, match="capacity"):
        p.order(1)  # the pool's breaker counts it
    assert p.cursor == 1
    name = p.order(1)
    assert sky.calls[-1][2].resources["infra"] == "nebius/eu-west1" and p.hourly_cost(name) == 11.2
    assert p.cursor == 0  # the next order starts at the cheapest offer again
    first = sky.calls[0][1]
    assert (1, first) not in p.run_tickets()  # it left no cluster: nothing bills, nothing to release


def test_a_failed_launch_that_left_a_cluster_stays_open_for_release():
    sky = FakeSky()
    p = from_config(_cfg(resources={"infra": "nebius/eu-north1"}), "run-k", sky=sky)

    def half_made(task, cluster_name, **kw):
        sky.clusters[cluster_name] = {"name": cluster_name, "status": "INIT"}
        raise RuntimeError("provisioning interrupted")

    sky.launch = half_made
    with pytest.raises(RuntimeError):
        p.order(1)
    assert p.run_tickets() == [(1, "yeto-smp-run-k-1")] and not p.account_free(1)
    assert p.release(1, "yeto-smp-run-k-1")["state"] == "ended" and not sky.clusters


def test_a_cluster_stuck_starting_reads_lost_past_the_start_timeout():
    t = [0.0]
    sky = FakeSky()
    p = SkyProvider("run-k", slots=[1], resources={"infra": "aws", "accelerators": "H100:8"}, sky=sky,
                    start_timeout_s=600, clock=lambda: t[0])
    name = p.order(1)
    assert p.status(1, name).state == "waiting"
    t[0] = 601
    status = p.status(1, name)
    assert status.state == "ended" and status.end_reason == "lost"
    sky.clusters[name]["status"] = "UP"
    assert p.status(1, name).state == "active"


def test_a_listed_cluster_of_this_run_is_adopted_and_others_are_not():
    sky = FakeSky()
    p = from_config(_cfg(resources={"cloud": "aws", "accelerators": "H100:8"}), "run-k", sky=sky)
    sky.clusters["yeto-smp-run-k-7"] = {"name": "yeto-smp-run-k-7", "status": "UP"}  # launch never recorded
    sky.clusters["yeto-smp-other-1"] = {"name": "yeto-smp-other-1", "status": "UP"}
    sky.clusters["prod-trainer"] = {"name": "prod-trainer", "status": "UP"}
    tickets = p.run_tickets()
    assert [t for _, t in tickets] == ["yeto-smp-run-k-7"]
    assert p.release(tickets[0][0], "yeto-smp-run-k-7")["state"] == "ended"
    assert "yeto-smp-other-1" in sky.clusters and "prod-trainer" in sky.clusters
    with pytest.raises(PermissionError):
        p.release(1, "yeto-smp-other-1")
    assert p.order(1) == "yeto-smp-run-k-8"  # numbering continues past the adopted cluster


def test_task_yaml_envs_and_secrets_go_on_the_launched_task(tmp_path, monkeypatch):
    sky = FakeSky()
    yaml = tmp_path / "island.yaml"
    yaml.write_text("run: echo\n")
    monkeypatch.setenv("LOCAL_TOKEN", "value-not-printed")
    p = from_config(_cfg(resources={"cloud": "aws"}, task_yaml=str(yaml), envs={"TP": "8"},
                         secrets={"ISLAND_TOKEN": "LOCAL_TOKEN"}), "run-k", sky=sky)
    p.order(1)
    task = sky.calls[-1][2]
    assert task.name == f"yaml:{yaml}" and task.envs == {"TP": "8"}
    assert task.secrets == {"ISLAND_TOKEN": "value-not-printed"}


def test_the_skeleton_record_format_is_still_read(tmp_path):
    sky = FakeSky()
    p = SkyProvider("run-k", slots=[1], resources={"cloud": "aws"}, sky=sky)
    p.record.parent.mkdir(parents=True, exist_ok=True)
    p.record.write_text(json.dumps({"yeto-smp-run-k-3": {"slot": 1, "n": 3, "open": True}}))
    assert p.run_tickets() == [(1, "yeto-smp-run-k-3")] and not p.account_free(1)
    assert p.order(2) == "yeto-smp-run-k-4"


def test_kind_sky_is_built_in_and_the_dry_run_checks_it_offline(tmp_path):
    cfg = _cfg(candidates={"spot_only": True}, ssh_key=str(tmp_path / "missing-key"))
    assert prov.provider_kind(cfg) == "sky" and not prov.is_national_compute(cfg)
    assert prov.provider_kind({"provider": {"factory": "yeto.launch.nc.sky_provider:from_config"}}) == "sky"
    ops = cli.pool_ops(cfg)
    assert isinstance(ops.provider, SkyProvider) and len(ops.provider.offers) == 4
    problems = offline_problems(cfg)
    assert any("ssh_key" in p for p in problems)
    assert offline_problems(_cfg(zone="x")) == ["sky provider: unknown keys ['zone']"]


def test_sky_orders_and_releases_write_the_resource_ledger(tmp_path, monkeypatch):
    ledger = tmp_path / "resources.jsonl"
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(ledger))
    sky = FakeSky()
    p = from_config(_cfg(resources={"cloud": "aws"}, price_usd_h=9.5), "run-k", sky=sky)
    name = p.order(1)
    p.release(1, name)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    kinds = [(r.get("event") or r.get("op"), r.get("kind")) for r in rows]
    assert all(k == "sky_cluster" for _, k in kinds) and len(rows) == 3


def test_the_watchdog_runs_its_loop_over_the_sky_provider(tmp_path):
    """--via-provider: the watchdog's candidate rotation, preemption and budget, with the
    machine calls going through SkyProvider (record, own-name guard)."""
    t = [1_000_000.0]
    clock = lambda: t[0]  # noqa: E731
    sky = FakeSky()
    sky.fail_launch.add("nebius/eu-north1")  # no on-demand / spot H100 capacity
    args = SimpleNamespace(run_id="runw", cluster_prefix="spot-w", yaml=None, secret=[], starting_timeout_s=600,
                           dir=str(tmp_path), own_name_re="spot")
    cands = wd.candidates(spot_only=True)
    ops = wd.provider_ops(args, cands, ["RUN_ID=runw"], sky=sky)
    original = sky.launch

    def launch_up(task, cluster_name, **kw):
        original(task, cluster_name, **kw)
        sky.clusters[cluster_name]["status"] = "UP"

    sky.launch = launch_up
    cfg = wd.Config(cluster_prefix="spot-w", run_id="runw", island="spot-0", budget_usd=1000.0,
                    log=str(tmp_path / "wd.jsonl"), state=str(tmp_path / "wd.state.json"),
                    stop_file=str(tmp_path / "wd.stop"), poll_s=60.0, round_sleep_s=300.0, max_rounds=1)
    w = wd.Watchdog(cfg, cands, ops, clock=clock, sleep=lambda s: t.__setitem__(0, t[0] + s))
    assert w.acquire() is None and w.cluster == "spot-w-2"  # the H100 launch failed, H200 eu-west1 is up
    assert sky.calls[-1][2].envs["CARD"] == "H200" and sky.calls[-1][2].envs["RUN_ID"] == "runw"
    assert ops.state("spot-w-2", "eu-west1") == "RUNNING"
    del sky.clusters["spot-w-2"]  # reclaimed
    w.watch_once()
    assert w.cluster is None and any(e["event"] == "preempted" for e in w.history)
    with pytest.raises(AssertionError):
        ops.down("prod-trainer")  # the own-name guard stays


def test_a_failing_offer_cools_down_alone_and_the_pool_breaker_counts_only_without_a_fallback():
    from yeto.launch.pool.provider import ProviderPoolOps

    t = [1000.0]
    sky = FakeSky()
    two = [{"resources": {"infra": "nebius/eu-north1"}, "price_usd_h": 6.32},
           {"resources": {"infra": "nebius/eu-west1"}, "price_usd_h": 11.2}]
    p = SkyProvider("run-k", slots=[1, 2], offers=offers.offers_from_config({"offers": two}), sky=sky,
                    clock=lambda: t[0])
    sky.fail_launch.add("nebius/eu-north1")
    with pytest.raises(RuntimeError):
        p.order(1)
    assert p.has_fallback_offer() and ProviderPoolOps(p).has_fallback_offer()
    a = p.order(1)
    b = p.order(2)  # the cursor is back at the cheapest, but it is cooling: the next offer again
    assert p.offer_of(a) == p.offer_of(b) == p.offers[1].key
    assert p.offer_breakers()[p.offers[0].key]["fails"] == 1
    sky.fail_launch.add("nebius/eu-west1")
    with pytest.raises(RuntimeError):
        p.order(1)
    assert not p.has_fallback_offer()  # every offer cools down: now the pool's breaker counts
    t[0] += 601
    sky.fail_launch.clear()
    c = p.order(1)
    assert p.offer_of(c) == p.offers[0].key and p.offers[0].key not in p.offer_breakers()
