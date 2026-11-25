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
    """The S3 calls :func:`ssl.delete_s3_bucket` makes, on an in-memory bucket table."""

    def __init__(self, buckets=None, head_error=None):
        self.buckets = {k: list(v) for k, v in (buckets or {}).items()}
        self.head_error = head_error
        self.calls = []

    def head_bucket(self, Bucket):
        self.calls.append(("head", Bucket))
        if self.head_error:
            raise FakeS3Error(self.head_error)
        if Bucket not in self.buckets:
            raise FakeS3Error("404")

    def get_paginator(self, op):
        assert op == "list_objects_v2"
        s3 = self

        class Pages:
            def paginate(self, Bucket):
                s3.calls.append(("list", Bucket))
                return [{"Contents": [{"Key": k} for k in s3.buckets[Bucket]]}]
        return Pages()

    def delete_objects(self, Bucket, Delete):
        self.calls.append(("delete_objects", Bucket))
        gone = {o["Key"] for o in Delete["Objects"]}
        self.buckets[Bucket] = [k for k in self.buckets[Bucket] if k not in gone]

    def delete_bucket(self, Bucket):
        self.calls.append(("delete_bucket", Bucket))
        assert not self.buckets[Bucket], "a bucket is emptied first"
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
    ok, what = ssl.delete(NAME, _attrs(), sky=sky, s3_client=lambda region: s3)
    assert ok and sky.deleted == [NAME] and s3.calls == [] and "sky storage delete" in what


def test_a_bucket_sky_does_not_know_is_checked_and_deleted_on_s3_by_its_exact_name():
    s3 = FakeS3({NAME: ["groups/0.json"], "store-bucket": ["keep"], "other-bucket": ["keep"]})
    regions = []
    ok, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: regions.append(r) or s3)
    assert ok and "deleted" in what and regions == ["us-west-2"]
    assert NAME not in s3.buckets and s3.buckets["store-bucket"] == ["keep"] and s3.buckets["other-bucket"] == ["keep"]
    assert {b for _op, b in s3.calls} == {NAME}  # no other bucket is even looked at
    # already gone: released, nothing deleted
    ok, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: s3)
    assert ok and "absent" in what


def test_failures_are_not_releases_and_a_foreign_name_is_refused_untouched():
    s3 = FakeS3({NAME: []}, head_error="403")
    ok, what = ssl.delete(NAME, _attrs(), sky=FakeSky(), s3_client=lambda r: s3)
    assert not ok and "aws s3 rb" in what and NAME in s3.buckets
    ok, what = ssl.delete(NAME, _attrs(), sky=FakeSky(error=RuntimeError("server down")), s3_client=lambda r: s3)
    assert not ok and "server down" in what
    sky, s3 = FakeSky(known={"store-bucket"}), FakeS3({"store-bucket": ["x"]})
    ok, what = ssl.delete("store-bucket", _attrs(), sky=sky, s3_client=lambda r: s3)
    assert not ok and what.startswith("refused") and sky.deleted == [] and s3.calls == []
    # another cloud: sky down of its cluster deleted the non-persistent storage
    ok, what = ssl.delete(NAME, _attrs(cloud="nebius"), sky=FakeSky(), s3_client=lambda r: 1 / 0)
    assert ok and "deleted with its cluster" in what


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


def test_cleanup_releases_the_bucket_after_the_clusters(tmp_path):
    led = ResourceLedger(tmp_path / "resources.jsonl")
    led.created("sky_cluster", "c0", "c0")
    led.created("sky_storage", NAME, NAME, **_attrs())
    order = []

    class Ops:
        def sky_down(self, cluster):
            order.append(("down", cluster))
            return True

        def sky_storage_delete(self, name, attrs):
            order.append(("bucket", name))
            return ssl.delete(name, attrs, sky=FakeSky(), s3_client=lambda r: FakeS3({NAME: []}))

    assert all(a.ok for a in cleanup.cleanup(led, Ops(), dry_run=True)) and led.open_resources()
    actions = cleanup.cleanup(led, Ops())
    assert all(a.ok for a in actions) and order == [("down", "c0"), ("bucket", NAME)]
    assert led.open_resources() == []  # ledger_check: nothing open
    # a failed delete stays open with its error (a rerun retries it)
    led2 = ResourceLedger(tmp_path / "r2.jsonl")
    led2.created("sky_storage", NAME, NAME, **_attrs())

    class Failing(Ops):
        def sky_storage_delete(self, name, attrs):
            return False, "S3 bucket: AccessDenied"

    (action,) = cleanup.cleanup(led2, Failing())
    assert not action.ok and led2.open_resources()[0].release_error == "S3 bucket: AccessDenied"


def test_teardown_release_skips_a_cluster_whose_termination_is_unconfirmed(tmp_path, monkeypatch):
    from yeto import launcher

    other = _rl_checkpoint_storage_name(PREFIX, 1)
    led = ResourceLedger(tmp_path / "resources.jsonl")
    for name, cluster in ((NAME, "c0"), (other, "c1")):
        led.created("sky_storage", name, name, **{**_attrs(), "cluster": cluster})
    deleted = []
    monkeypatch.setattr(ssl, "delete", lambda name, attrs, **kw: (deleted.append(name) or True, "ok"))
    launcher._release_sky_storages(led, {NAME: "c0", other: "c1"}, ["c1"])
    assert deleted == [NAME] and [r.id for r in led.open_resources()] == [other]
    launcher._release_sky_storages(object(), {NAME: "c0"}, [])  # no ledger to read: never raises


def test_a_launch_that_fails_for_capacity_releases_the_bucket_sky_created(tmp_path, monkeypatch):
    """The VC r4 case end to end: a --spot AWS island (its volume is a bucket sky creates),
    sky.launch fails with no capacity; the launcher records the bucket before the order and
    releases it at its teardown, through sky, so the ledger has nothing open."""
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
    args.keep, args.recover_timeout, args.controller_poll = False, 0.0, 30.0
    with pytest.raises(RuntimeError, match="failed to provision"):
        launcher.run(args)
    expected = _rl_checkpoint_storage_name(args.cluster_prefix, 0)
    assert deleted == [expected] and not created
    led = ResourceLedger(ledger_path)
    buckets = [r for r in led.resources() if r.kind == "sky_storage"]
    assert [r.id for r in buckets] == [expected] and not buckets[0].open
    assert not [r for r in led.open_resources() if r.kind == "sky_storage"]
