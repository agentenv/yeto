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
from tests.test_nc_sky_provider import FakeSky, ResourcesUnavailableError, marked


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("YETO_RESOURCE_LEDGER", raising=False)


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
    first = p.order(1)
    status = p.status(1, first)  # the launch request failed: the pool releases it (offer_failed)
    assert status.state == "ended" and "capacity" in status.detail
    assert p.cursor == 1
    name = p.order(1)
    assert [c for c in sky.calls if c[0] == "launch"][-1][2].resources["infra"] == "nebius/eu-west1"
    assert p.hourly_cost(name) == 11.2
    sky.up(name)
    assert p.status(1, name).state == "active"
    assert p.cursor == 0  # the next order starts at the cheapest offer again
    assert (1, first) not in p.run_tickets()  # it left no cluster: nothing bills, nothing to release


def test_a_failed_launch_that_left_a_cluster_stays_open_for_release():
    sky = FakeSky()
    p = from_config(_cfg(resources={"infra": "nebius/eu-north1"}), "run-k", sky=sky)
    original = sky.launch

    def half_made(task, cluster_name, **kw):
        rid = original(task, cluster_name, **kw)
        sky.requests[rid].update(status="FAILED", error=RuntimeError("provisioning interrupted"))
        return rid

    sky.launch = half_made
    name = p.order(1)
    assert p.status(1, name).end_reason == "lost"
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
    sky.clusters["yeto-smp-run-k-7"] = marked("yeto-smp-run-k-7", "run-k")  # launch never recorded
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
    task = [c for c in sky.calls if c[0] == "launch"][-1][2]
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
    sky.up(name)
    assert p.status(1, name).state == "active"  # the launch is confirmed: "created"
    p.release(1, name)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    kinds = [(r.get("event") or r.get("op"), r.get("kind")) for r in rows]
    assert all(k == "sky_cluster" for _, k in kinds) and len(rows) == 3


def test_the_watchdog_runs_its_loop_over_the_sky_provider(tmp_path):
    """The watchdog's candidate rotation, preemption and budget, with the
    machine calls going through SkyProvider (record, own-name guard)."""
    t = [1_000_000.0]
    clock = lambda: t[0]  # noqa: E731
    sky = FakeSky()
    sky.fail_launch.add("nebius/eu-north1")  # no on-demand / spot H100 capacity
    args = SimpleNamespace(run_id="runw", cluster_prefix="spot-w", yaml=None, secret=[], starting_timeout_s=600,
                           launch_timeout_s=1800, dir=str(tmp_path), own_name_re="spot")
    cands = wd.candidates(spot_only=True)
    advance = lambda s: t.__setitem__(0, t[0] + s)  # noqa: E731
    ops = wd.provider_ops(args, cands, ["RUN_ID=runw"], sky=sky, clock=clock, sleep=advance)
    assert ops.provider.start_timeout_s == 1800  # the old `sky launch` bound, not the STARTING one
    original = sky.launch

    def launch_up(task, cluster_name, **kw):
        rid = original(task, cluster_name, **kw)
        if cluster_name in sky.clusters:  # not a refused launch
            sky.up(cluster_name)
        return rid

    sky.launch = launch_up
    cfg = wd.Config(cluster_prefix="spot-w", run_id="runw", island="spot-0", budget_usd=1000.0,
                    log=str(tmp_path / "wd.jsonl"), state=str(tmp_path / "wd.state.json"),
                    stop_file=str(tmp_path / "wd.stop"), poll_s=60.0, round_sleep_s=300.0, max_rounds=1)
    w = wd.Watchdog(cfg, cands, ops, clock=clock, sleep=advance)
    assert w.acquire() is None and w.cluster == "spot-w-2"  # the H100 launch failed, H200 eu-west1 is up
    last = [c for c in sky.calls if c[0] == "launch"][-1]
    assert last[2].envs["CARD"] == "H200" and last[2].envs["RUN_ID"] == "runw"
    assert ops.state("spot-w-2", "eu-west1") == "RUNNING"
    assert sky.calls[-1] == ("status", ("spot-w-2",), "AUTO")  # refreshed: a reclaim is seen at once
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
    assert p.status(1, p.order(1)).end_reason == "lost"
    assert p.has_fallback_offer() and ProviderPoolOps(p).has_fallback_offer()
    assert cli.NcSshOps({"run": "run-k"}, provider=p).has_fallback_offer()  # the pool route's ops ask it too
    a = p.order(1)
    b = p.order(2)  # the cursor is back at the cheapest, but it is cooling: the next offer again
    assert p.offer_of(a) == p.offer_of(b) == p.offers[1].key
    assert p.offer_breakers()[p.offers[0].key]["fails"] == 1
    sky.fail_launch.add("nebius/eu-west1")
    assert p.status(1, p.order(1)).end_reason == "lost"
    assert not p.has_fallback_offer()  # every offer cools down: now the pool's breaker counts
    t[0] += 601
    sky.fail_launch.clear()
    c = p.order(1)
    sky.up(c)
    assert p.status(1, c).state == "active"
    assert p.offer_of(c) == p.offers[0].key and p.offers[0].key not in p.offer_breakers()


def test_a_watchdog_launch_that_never_finishes_times_out_and_is_cancelled(tmp_path):
    """The old CLI path bounded `sky launch` with --launch-timeout-s; the provider path polls
    the confirmed launch the same way, and the delete cancels the request still in flight."""
    t = [1_000_000.0]
    clock = lambda: t[0]  # noqa: E731
    sky = FakeSky()  # a launch request stays RUNNING until sky.up: here it never comes up
    args = SimpleNamespace(run_id="runt", cluster_prefix="spot-t", yaml=None, secret=[], starting_timeout_s=600,
                           launch_timeout_s=1800, dir=str(tmp_path), own_name_re="spot")
    cands = wd.candidates(spot_only=True)[:1]
    ops = wd.provider_ops(args, cands, ["RUN_ID=runt"], sky=sky, clock=clock,
                          sleep=lambda s: t.__setitem__(0, t[0] + s))
    ok, msg = ops.launch("spot-t-1", cands[0], 1800)
    # the provider's start timeout (= --launch-timeout-s) or the watchdog's own bound, whichever reads first
    assert not ok and msg in ("lost", "launch timeout after 1800s") and t[0] - 1_000_000.0 <= 1800 + 2 * ops.poll_s
    assert ops.down("spot-t-1") is True
    assert any(c[0] == "cancel" for c in sky.calls) and "spot-t-1" not in sky.clusters


def _legacy_cli_launch(cluster, cand, yaml, envs, secrets):
    """The argv of the removed CLI path (``watchdog.Ops.launch`` with ``main``'s CARD
    override, verbatim up to the binary name), kept here as the shadow reference."""
    envs = [e for e in envs if not e.startswith("CARD=")] + [f"CARD={cand.gpu}"]
    cmd = ["sky", "launch", "-c", cluster, yaml, "--infra", f"nebius/{cand.region}",
           "--gpus", f"{cand.gpu}:{cand.count}", "--use-spot" if cand.spot else "--no-use-spot", "-d", "-y"]
    for e in envs:
        cmd += ["--env", e]
    for s in secrets:
        cmd += ["--secret", s]
    return cmd


def _parse_legacy(cmd):
    out = {"cluster": cmd[3], "yaml": cmd[4], "envs": {}, "secrets": set()}
    rest = cmd[5:]
    for flag, value in zip(rest[::1], rest[1:] + [None]):
        if flag == "--infra":
            out["infra"] = value
        elif flag == "--gpus":
            out["accelerators"] = value
        elif flag == "--env":
            key, _, val = value.partition("=")
            out["envs"][key] = val
        elif flag == "--secret":
            out["secrets"].add(value)
    out["use_spot"] = "--use-spot" in rest
    out["detached_no_down"] = "-d" in rest and "--down" not in rest
    return out


def test_shadow_the_provider_launches_what_the_removed_cli_launched(tmp_path, monkeypatch):
    """Shadow comparison of 7.2: for every candidate of the default list, the cluster the
    watchdog asks SkyProvider for has the same name, task file, infra, GPUs, spot flag,
    envs (CARD per candidate, RUN_ID, the operator's) and secrets as the `sky launch`
    argv of the removed CLI path. The one intended difference: the task (it has a run:
    the sampler is the sky job) is launched with down=True, so the cluster goes when the
    job ends; the CLI path left an idle VM billing until the watchdog noticed."""
    monkeypatch.setenv("MODAL_TOKEN_ID", "id-x")
    yaml = tmp_path / "island.yaml"
    yaml.write_text("run: python -m yeto.rl.spot_rollout.sampler --spec spec.json\n")
    user_envs = ["TP=8", "BOAT_STARTS_PER_HOUR=60", "CARD=stale", "RUN_ID=runs"]
    secrets = ["MODAL_TOKEN_ID"]
    cands = wd.candidates()
    t = [1_000_000.0]
    sky = FakeSky()
    original = sky.launch

    def launch_up(task, cluster_name, **kw):
        rid = original(task, cluster_name, **kw)
        sky.up(cluster_name)
        return rid

    sky.launch = launch_up
    args = SimpleNamespace(run_id="runs", cluster_prefix="spot-s", yaml=str(yaml), secret=secrets,
                           starting_timeout_s=600, launch_timeout_s=1800, dir=str(tmp_path), own_name_re="spot")
    envs = [e for e in user_envs if not e.startswith("RUN_ID=")] + ["RUN_ID=runs"]  # as main() builds them
    ops = wd.provider_ops(args, cands, envs, sky=sky, clock=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s))
    for i, cand in enumerate(cands, 1):
        name = f"spot-s-{i}"
        assert ops.launch(name, cand, 1800) == (True, "")
        _, cluster, task, kw = [c for c in sky.calls if c[0] == "launch"][-1]
        old = _parse_legacy(_legacy_cli_launch(name, cand, str(yaml), envs, secrets))
        assert cluster == old["cluster"]
        assert task.name == f"yaml:{old['yaml']}"
        assert task.resources == {"infra": old["infra"], "accelerators": old["accelerators"],
                                  "use_spot": old["use_spot"]}
        assert task.envs == old["envs"], cand.key
        assert set(task.secrets) == old["secrets"] and task.secrets["MODAL_TOKEN_ID"] == "id-x"
        assert old["detached_no_down"] and kw == {"down": True}  # the intended difference (docstring)
        assert ops.down(name) is True


def test_the_watchdog_releases_only_clusters_of_its_own_run(tmp_path):
    """Ownership on the watchdog's exact_prefix path (REV on #765): a cluster an earlier
    watchdog process of this run launched is adopted and can be released; a cluster of
    another run under the same name pattern, or one opened by hand, is never adopted,
    released or downed."""
    t = [1_000_000.0]
    clock = lambda: t[0]  # noqa: E731
    advance = lambda s: t.__setitem__(0, t[0] + s)  # noqa: E731
    sky = FakeSky()
    original = sky.launch

    def launch_up(task, cluster_name, **kw):
        rid = original(task, cluster_name, **kw)
        sky.up(cluster_name)
        return rid

    sky.launch = launch_up
    cands = wd.candidates(spot_only=True)[:1]

    def ops_for(run, where):
        args = SimpleNamespace(run_id=run, cluster_prefix="spot-w", yaml=None, secret=[], starting_timeout_s=600,
                               launch_timeout_s=1800, dir=str(tmp_path / where), own_name_re="spot")
        (tmp_path / where).mkdir(exist_ok=True)
        return wd.provider_ops(args, cands, [f"RUN_ID={run}"], sky=sky, clock=clock, sleep=advance)

    earlier = ops_for("runw", "earlier")  # an earlier watchdog process of this run (its record is lost)
    assert earlier.launch("spot-w-1", cands[0], 1800) == (True, "")
    other = ops_for("runx", "other")  # another run's watchdog with the same cluster prefix
    assert other.launch("spot-w-7", cands[0], 1800) == (True, "")
    sky.clusters["spot-w-debug"] = {"name": "spot-w-debug", "status": "UP", "kw": {},
                                    "handle": SimpleNamespace(head_ip="198.51.100.9", ssh_user="ubuntu")}
    mine = ops_for("runw", "mine")
    owned = {name for _, name in mine.provider.run_tickets()}
    assert "spot-w-1" in owned and not owned & {"spot-w-7", "spot-w-debug"}
    assert mine.down("spot-w-1") is True and "spot-w-1" not in sky.clusters
    for foreign in ("spot-w-7", "spot-w-debug"):
        assert mine.down(foreign) is False  # refused: never this run's
        assert foreign in sky.clusters
    assert not [c for c in sky.calls if c[0] == "down" and c[1] in ("spot-w-7", "spot-w-debug")]
