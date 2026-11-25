"""The buckets sky creates for a sky island's volume are in the run's resource ledger and
released at the launcher's teardown, by ``yeto cleanup`` / ``run down``, also when the launch
failed for capacity (VC r4: an empty S3 bucket per failed AWS move). Only the exact bucket
name yeto derives from the run's prefix is ever deleted. Fake sky, fake boto: no cloud."""

from __future__ import annotations

import sys
import types

import pytest
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401  (fixtures)

from yeto.cost import cleanup
from yeto.cost.ledger import ResourceLedger
from yeto.launch import sky_storage_ledger as ssl
from yeto.launch.storage import _rl_checkpoint_storage_name

PREFIX = "verl-f8-r4-i0-m1"
NAME = _rl_checkpoint_storage_name(PREFIX, 0)


class FakeS3Error(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """The S3 calls :func:`ssl.delete_empty_s3_bucket` makes, on an in-memory bucket table."""

    def __init__(self, buckets=None, head_error=None, endpoint="https://s3.us-west-2.amazonaws.com"):
        self.buckets = {k: list(v) for k, v in (buckets or {}).items()}
        self.head_error = head_error
        self.calls = []
        self.meta = types.SimpleNamespace(endpoint_url=endpoint)

    def head_bucket(self, Bucket):
        self.calls.append(("head", Bucket))
        if self.head_error:
            raise FakeS3Error(self.head_error)
        if Bucket not in self.buckets:
            raise FakeS3Error("404")

    def list_objects_v2(self, Bucket, MaxKeys):
        self.calls.append(("list", Bucket))
        keys = self.buckets[Bucket][:MaxKeys]
        return {"KeyCount": len(keys), **({"Contents": [{"Key": k} for k in keys]} if keys else {})}

    def delete_bucket(self, Bucket):
        self.calls.append(("delete_bucket", Bucket))
        assert not self.buckets[Bucket], "only an empty bucket is ever deleted"
        del self.buckets[Bucket]


class FakeSky:
    def __init__(self, known=(), error=None):
        self.known, self.error, self.deleted = set(known), error, []

    def storage_delete(self, name):
        return ("storage_delete", name)

    def stream_and_get(self, rid):
        _op, name = rid
        if self.error:
            raise self.error
        if name not in self.known:
            raise ValueError(f"Storage name {name!r} not found.")
        self.known.discard(name)
        self.deleted.append(name)


def _attrs(cloud="aws", prefix=PREFIX):
    return {"cluster": f"{prefix}-l0-us-west-2", "cluster_prefix": prefix, "cloud": cloud, "region": "us-west-2"}


def test_only_the_bucket_name_of_the_run_prefix_is_owned():
    assert ssl.yeto_owned(NAME, PREFIX)
    assert not ssl.yeto_owned(NAME, "verl-f8-r4-i0-m3")  # another island's prefix: another hash
    assert not ssl.yeto_owned("store-bucket", PREFIX)
    assert not ssl.yeto_owned(NAME.replace("-rl-0", "-rl-1"), "other")
    assert not ssl.yeto_owned(NAME, None) and not ssl.yeto_owned(None, PREFIX)


def test_a_sky_owned_storage_is_deleted_through_sky():
    sky = FakeSky(known={NAME})
    s3 = FakeS3({NAME: []})
    status, what = ssl.delete(NAME, _attrs(), sky=sky, s3_client=lambda region: s3)
    assert status == ssl.RELEASED and sky.deleted == [NAME] and s3.calls == [] and "sky storage delete" in what


def test_a_persistent_storage_is_never_deleted():
    sky, s3 = FakeSky(known={NAME}), FakeS3({NAME: []})
    status, what = ssl.delete(NAME, {**_attrs(), "persistent": True}, sky=sky, s3_client=lambda r: s3)
    assert status == ssl.SKIPPED and "persistent" in what and sky.deleted == [] and s3.calls == []


def test_a_bucket_sky_does_not_know_is_deleted_on_s3_only_when_empty_and_never_emptied():
    s3 = FakeS3({NAME: [], "store-bucket": ["keep"], "other-bucket": ["keep"]})
    regions = []
    status, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: regions.append(r) or s3)
    assert status == ssl.RELEASED and "deleted" in what and regions == ["us-west-2"]
    assert NAME not in s3.buckets and s3.buckets["store-bucket"] == ["keep"] and s3.buckets["other-bucket"] == ["keep"]
    assert {b for _op, b in s3.calls} == {NAME}  # no other bucket is even looked at
    status, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: s3)  # already gone
    assert status == ssl.RELEASED and "absent" in what
    # it holds the island's completed groups: left as it is, for a person
    s3 = FakeS3({NAME: ["groups/0.json", "groups/1.json"]})
    status, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: s3)
    assert status == ssl.SKIPPED and "holds objects" in what and "aws s3 rb" in what
    assert s3.buckets[NAME] == ["groups/0.json", "groups/1.json"] and ("delete_bucket", NAME) not in s3.calls


def test_failures_are_not_releases_and_a_foreign_name_is_refused_untouched():
    s3 = FakeS3({NAME: []}, head_error="403")
    status, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: s3)
    assert status == ssl.FAILED and "aws s3 rb" in what and NAME in s3.buckets
    status, what = ssl.delete(NAME, _attrs(), sky=FakeSky(error=RuntimeError("server down")), s3_client=lambda r: s3)
    assert status == ssl.FAILED and "server down" in what
    sky, s3 = FakeSky(known={"store-bucket"}), FakeS3({"store-bucket": ["x"]})
    status, what = ssl.delete("store-bucket", _attrs(), sky=sky, s3_client=lambda r: s3)
    assert status == ssl.FAILED and what.startswith("refused") and sky.deleted == [] and s3.calls == []
    # another cloud: sky down of its cluster deleted the non-persistent storage
    status, what = ssl.delete(NAME, _attrs(cloud="nebius"), sky=FakeSky(), s3_client=lambda r: 1 / 0)
    assert status == ssl.RELEASED and "deleted with its cluster" in what


def test_the_s3_fallback_never_reaches_another_endpoint_or_runs_without_a_region(monkeypatch):
    """The guard against emptying a bucket on the run's R2 / MinIO store: the client is
    pinned to the regional AWS endpoint; no region, or a client on another endpoint, and
    nothing is looked up at all."""
    made = []
    status, what = ssl.delete(NAME, {**_attrs(), "region": "not a region"}, sky=FakeSky(),
                              s3_client=lambda r: made.append(r))
    assert status == ssl.FAILED and "no AWS region" in what and made == []
    # no region recorded: the lookup client must be AWS's global endpoint too
    r2_lookup = FakeS3({NAME: []}, endpoint="https://0123abcd.r2.cloudflarestorage.com")
    status, what = ssl.delete(NAME, {**_attrs(), "region": None}, sky=FakeSky(), s3_client=lambda r: r2_lookup)
    assert status == ssl.FAILED and "not AWS S3" in what and r2_lookup.calls == []
    r2 = FakeS3({NAME: []}, endpoint="https://0123abcd.r2.cloudflarestorage.com")
    status, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: r2)
    assert status == ssl.FAILED and "not AWS S3" in what and r2.calls == [] and NAME in r2.buckets
    # the real client: AWS_ENDPOINT_URL (the run's R2 store) does not move it off AWS
    pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://0123abcd.r2.cloudflarestorage.com")
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "https://0123abcd.r2.cloudflarestorage.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    assert ssl._s3_client("us-west-2").meta.endpoint_url == "https://s3.us-west-2.amazonaws.com"
    assert ssl._s3_client(None).meta.endpoint_url == "https://s3.amazonaws.com"  # the lookup: global AWS
    with pytest.raises(ValueError, match="no AWS region"):
        ssl._s3_client("not a region")


class FakeAWS:
    """S3 as seen from AWS's global endpoint (region None) and the regional ones: a
    HeadBucket on the global endpoint of a bucket in another region answers 301 with the
    bucket's region in ``x-amz-bucket-region``."""

    def __init__(self, buckets):
        self.buckets = buckets  # name -> (region, keys)
        self.clients = []

    def __call__(self, region):
        aws = self
        endpoint = "https://s3.amazonaws.com" if region is None else f"https://s3.{region}.amazonaws.com"

        class Client(FakeS3):
            def head_bucket(self, Bucket):
                self.calls.append(("head", Bucket))
                if Bucket not in aws.buckets:
                    raise FakeS3Error("404")
                where = aws.buckets[Bucket][0]
                header = {"ResponseMetadata": {"HTTPHeaders": {"x-amz-bucket-region": where}}}
                if (region or "us-east-1") != where:
                    error = FakeS3Error("301")
                    error.response.update(header)
                    raise error
                return header

            def list_objects_v2(self, Bucket, MaxKeys):
                assert region is not None, "the global endpoint is only used to look the bucket up"
                self.calls.append(("list", Bucket))
                keys = aws.buckets[Bucket][1][:MaxKeys]
                return {"KeyCount": len(keys)}

            def delete_bucket(self, Bucket):
                assert region == aws.buckets[Bucket][0], "a delete goes to the bucket's own regional endpoint"
                self.calls.append(("delete_bucket", Bucket))
                del aws.buckets[Bucket]

        client = Client(endpoint=endpoint)
        self.clients.append((region, client))
        return client


def test_a_row_without_a_region_is_located_on_the_global_endpoint_then_handled_by_the_rules():
    """sky chose the region (the island named none) and the row has no region: one read-only
    HeadBucket on AWS's global endpoint. Gone (sky down deleted it): released as absent,
    not a release_failed forever. Still there: its region from the answer, then the regional
    rules (an empty bucket deleted there; one that holds objects left)."""
    no_region = {**_attrs(), "region": None}
    aws = FakeAWS({})
    status, what = ssl.delete(NAME, no_region, sky=FakeSky(), s3_client=aws)
    assert status == ssl.RELEASED and "absent" in what and [r for r, _c in aws.clients] == [None]
    aws = FakeAWS({NAME: ("us-east-2", [])})
    status, what = ssl.delete(NAME, no_region, sky=FakeSky(), s3_client=aws)
    assert status == ssl.RELEASED and "deleted" in what and NAME not in aws.buckets
    assert [r for r, _c in aws.clients] == [None, "us-east-2"]
    aws = FakeAWS({NAME: ("eu-west-1", ["groups/0.json"])})
    status, what = ssl.delete(NAME, no_region, sky=FakeSky(), s3_client=aws)
    assert status == ssl.SKIPPED and "holds objects" in what and NAME in aws.buckets
    aws = FakeAWS({NAME: ("us-east-1", [])})  # in us-east-1 the global endpoint answers 200
    status, what = ssl.delete(NAME, no_region, sky=FakeSky(), s3_client=aws)
    assert status == ssl.RELEASED and [r for r, _c in aws.clients] == [None, "us-east-1"]


def test_the_region_sky_chose_is_added_to_the_bucket_row_once_the_cluster_is_up(tmp_path):
    from yeto import launcher

    led = ResourceLedger(tmp_path / "resources.jsonl")
    led.created("sky_storage", NAME, NAME, **{**_attrs(), "region": None})
    handle = types.SimpleNamespace(launched_resources=types.SimpleNamespace(region="us-east-2"))
    launcher._record_sky_storage_regions(led, {NAME: "c0"}, {"c0": (1, handle)})
    (row,) = led.open_resources()
    assert row.attrs["region"] == "us-east-2" and row.attrs["cluster_prefix"] == PREFIX  # fields only added
    # a recorded region is kept; a failed launch (no result) adds nothing; never raises
    launcher._record_sky_storage_regions(led, {NAME: "c0"}, {"c0": (1, types.SimpleNamespace(
        launched_resources=types.SimpleNamespace(region="eu-west-1")))})
    launcher._record_sky_storage_regions(led, {NAME: "c0"}, {})
    launcher._record_sky_storage_regions(object(), {NAME: "c0"}, {})
    assert led.open_resources()[0].attrs["region"] == "us-east-2"


def test_record_takes_only_named_storages_sky_creates_for_this_prefix(tmp_path):
    led = ResourceLedger(tmp_path / "resources.jsonl")
    task = types.SimpleNamespace(storage_mounts={
        "~/yeto-rl/groups": types.SimpleNamespace(name=NAME, source=None, persistent=False),
        "/mnt/store": types.SimpleNamespace(name=None, source="s3://store-bucket", persistent=True),
        # a named mount with a source is an existing bucket: sky did not create it, even under our name
        "/mnt/existing": types.SimpleNamespace(name=_rl_checkpoint_storage_name(PREFIX, 1),
                                               source="s3://" + _rl_checkpoint_storage_name(PREFIX, 1),
                                               persistent=True),
        "/mnt/other": types.SimpleNamespace(name="someone-elses-bucket", source=None, persistent=False),
    })
    assert ssl.record(led, "c0", task, cluster_prefix=PREFIX, cloud="aws", region="us-west-2") == [NAME]
    (res,) = led.open_resources()
    assert (res.kind, res.id, res.attrs["cluster"], res.attrs["cloud"]) == ("sky_storage", NAME, "c0", "aws")


def test_cleanup_releases_the_bucket_only_after_every_island_cluster_is_gone(tmp_path):
    led = ResourceLedger(tmp_path / "resources.jsonl")
    led.created("sky_cluster", "c0", "c0")
    led.created("sky_cluster", "pool-node", "pool-node", slot=0)  # a sampler pool's: never mounts it
    led.created("sky_storage", NAME, NAME, **_attrs())
    order = []

    class Ops:
        down_ok = True

        def sky_down(self, cluster):
            order.append(("down", cluster))
            return self.down_ok

        def sky_storage_delete(self, name, attrs):
            order.append(("bucket", name))
            return ssl.delete(name, attrs, sky=FakeSky(), s3_client=lambda r: FakeS3({NAME: []}))

    keys = tuple(r.key for r in led.open_resources() if r.id != "pool-node")
    assert all(a.ok for a in cleanup.cleanup(led, Ops(), dry_run=True, keys=keys)) and order == []
    actions = cleanup.cleanup(led, Ops(), keys=keys)
    assert all(a.ok for a in actions) and order == [("down", "c0"), ("bucket", NAME)]
    assert [r.id for r in led.open_resources()] == ["pool-node"]  # ledger_check: no island resource open

    # the cluster's down is not confirmed: its bucket is not touched (release_skipped, stays open)
    led2 = ResourceLedger(tmp_path / "r2.jsonl")
    led2.created("sky_cluster", "c0", "c0")
    led2.created("sky_storage", NAME, NAME, **_attrs())
    ops = Ops()
    ops.down_ok, order[:] = False, []
    actions = cleanup.cleanup(led2, ops)
    assert order == [("down", "c0")] and not any(a.ok for a in actions)
    bucket = next(r for r in led2.open_resources() if r.kind == "sky_storage")
    assert bucket.release_error.startswith("release_skipped: kept: sky cluster(s) c0 not confirmed terminated")

    # a bucket that holds data: release_skipped, never emptied; a failure: release_failed
    for verdict, prefix in (((ssl.SKIPPED, "holds objects"), "release_skipped: holds objects"),
                            ((ssl.FAILED, "S3 bucket: AccessDenied"), "S3 bucket: AccessDenied")):
        led3 = ResourceLedger(tmp_path / f"r3-{verdict[0]}.jsonl")
        led3.created("sky_storage", NAME, NAME, **_attrs())

        class Verdict(Ops):
            def sky_storage_delete(self, name, attrs, v=verdict):
                return v

        (action,) = cleanup.cleanup(led3, Verdict())
        assert not action.ok and led3.open_resources()[0].release_error == prefix


def test_teardown_release_waits_for_every_island_cluster(tmp_path, monkeypatch):
    from yeto import launcher

    other = _rl_checkpoint_storage_name(PREFIX, 1)
    deleted = []
    monkeypatch.setattr(ssl, "delete", lambda name, attrs, **kw: (deleted.append(name) or ssl.RELEASED, "ok"))

    def ledger(path, open_cluster=None):
        led = ResourceLedger(tmp_path / path)
        for name, cluster in ((NAME, "c0"), (other, "c1")):
            led.created("sky_storage", name, name, **{**_attrs(), "cluster": cluster})
        if open_cluster:
            led.created("sky_cluster", open_cluster, open_cluster)
        return led

    led = ledger("a.jsonl")
    launcher._release_sky_storages(led, {NAME: "c0", other: "c1"}, ["c1-relaunched"])  # unconfirmed: keep all
    assert deleted == [] and len(led.open_resources()) == 2
    led = ledger("b.jsonl", open_cluster="c0")  # a failed launch's cluster still in the ledger
    launcher._release_sky_storages(led, {NAME: "c0", other: "c1"}, [])
    assert deleted == [] and len(led.open_resources()) == 3
    led = ledger("c.jsonl")
    launcher._release_sky_storages(led, {NAME: "c0", other: "c1"}, [])
    assert sorted(deleted) == sorted([NAME, other]) and led.open_resources() == []
    launcher._release_sky_storages(object(), {NAME: "c0"}, [])  # no ledger to read: never raises


def _capacity_failure(tmp_path, monkeypatch, *, keep):
    """A --spot AWS island (its volume is a bucket sky creates) whose sky.launch fails for
    capacity (the VC r4 case); returns (ledger, the bucket name, buckets sky still has,
    buckets deleted through sky)."""
    from test_trainer_launch_render import _setup

    from yeto import launcher

    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    ledger_path = tmp_path / "resources.jsonl"
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(ledger_path))
    _setup(monkeypatch, tmp_path, [])
    sky = sys.modules["sky"]
    created, deleted = set(), []

    def launch(task, **kw):
        for _mount, storage in (getattr(task, "storage_mounts", None) or {}).items():
            if getattr(storage, "name", None):
                created.add(storage.name)  # sky creates the bucket, then fails to get an instance
        return ("launch", kw.get("cluster_name"))

    def stream_and_get(rid):
        if rid[0] == "launch":
            raise RuntimeError("ResourcesUnavailableError: Max spot instance count exceeded")
        name = rid[1]
        if name not in created:
            raise ValueError(f"Storage name {name!r} not found.")
        created.discard(name)
        deleted.append(name)

    sky.launch, sky.stream_and_get = launch, stream_and_get
    sky.storage_delete = lambda name: ("storage_delete", name)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync", "--spot"),
                          gpu="aws:1xh100@us-west-2")
    args.keep, args.recover_timeout, args.controller_poll = keep, 0.0, 30.0
    with pytest.raises(RuntimeError, match="failed to provision"):
        launcher.run(args)
    return ResourceLedger(ledger_path), _rl_checkpoint_storage_name(args.cluster_prefix, 0), created, deleted


def test_a_launch_that_fails_for_capacity_releases_the_bucket_sky_created(tmp_path, monkeypatch, capsys):
    """End to end: the bucket is recorded before the order; the teardown keeps it while the
    failed launch's cluster row is open (the cluster may exist), and the backstop cleanup
    downs that cluster, then releases the bucket through sky: nothing open in the ledger."""
    led, expected, created, deleted = _capacity_failure(tmp_path, monkeypatch, keep=False)
    assert deleted == [] and "keeping the island buckets" in capsys.readouterr().err
    buckets = [r for r in led.resources() if r.kind == "sky_storage"]
    assert [r.id for r in buckets] == [expected]

    class Ops:
        def sky_down(self, cluster):
            return True

        def sky_storage_delete(self, name, attrs):
            return ssl.delete(name, attrs)  # the fake sky of the launch

    cleanup.cleanup(led, Ops())  # the backstop cleanup after the launcher
    assert deleted == [expected] and not created
    assert not [r for r in led.open_resources() if r.kind in ("sky_storage", "sky_cluster")]


def test_a_kept_launch_never_touches_the_bucket(tmp_path, monkeypatch):
    """--keep (the run goes on: a relaunch reuses the cluster and its volume): no release."""
    led, expected, created, deleted = _capacity_failure(tmp_path, monkeypatch, keep=True)
    assert deleted == [] and expected in created
    assert [r.id for r in led.open_resources() if r.kind == "sky_storage"] == [expected]


def test_a_launched_island_without_a_region_gets_sky_s_region_on_its_bucket_row(tmp_path, monkeypatch):
    """End to end: an AWS --spot island that names no region; once sky placed it, its bucket
    row carries the region sky chose (so the release needs no lookup)."""
    from test_trainer_launch_render import _Provisioned, _setup

    from yeto import launcher

    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    monkeypatch.setenv("YETO_SKY_SERVER_CHECK", "off")
    ledger_path = tmp_path / "resources.jsonl"
    monkeypatch.setenv("YETO_RESOURCE_LEDGER", str(ledger_path))
    _setup(monkeypatch, tmp_path, [])
    sky = sys.modules["sky"]
    handle = types.SimpleNamespace(head_ip="10.0.0.1",
                                   launched_resources=types.SimpleNamespace(region="us-east-2"))
    sky.launch = lambda task, **kw: ("launch", kw.get("cluster_name"))
    sky.stream_and_get = lambda rid: (1, handle)
    args = _launcher_args("ports", ("--controller", "local", "--rl-single-island-no-sync", "--spot"),
                          gpu="aws:1xh100")
    args.keep, args.recover_timeout, args.controller_poll = True, 0.0, 30.0
    try:
        launcher.run(args)
    except _Provisioned:
        pass
    (row,) = [r for r in ResourceLedger(ledger_path).resources() if r.kind == "sky_storage"]
    assert row.id == _rl_checkpoint_storage_name(args.cluster_prefix, 0) and row.attrs["region"] == "us-east-2"
