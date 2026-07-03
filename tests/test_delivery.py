"""--output classification, fetch/upload plumbing, self-termination gating."""

from __future__ import annotations

import pytest

from yeto import delivery


def test_kind_classification():
    assert delivery.kind(None) == "none"
    # Every sky-supported object store URI classifies as "store" — the
    # scheme list is sky's own registry, not a hand-rolled tuple.
    for uri in ("s3://bucket/models/run1", "gs://b/p", "r2://b", "oci://b/x"):
        assert delivery.kind(uri) == "store", uri
    assert delivery.kind("hf://org/repo") == "hf"
    assert delivery.kind("./out") == "local"
    assert delivery.kind("/abs/out") == "local"
    assert not delivery.is_remote(None) and not delivery.is_remote("./out")
    assert delivery.is_remote("s3://b") and delivery.is_remote("hf://o/r")


def test_fetch_cmd_uses_sky_ssh_alias():
    assert delivery.fetch_cmd("run-l0-us-east-2", "/tmp/out") == [
        "rsync", "-az", "run-l0-us-east-2:yeto-output/", "/tmp/out/",
    ]


def test_deliver_object_store_goes_through_sky_storage(monkeypatch):
    calls = []
    monkeypatch.setattr(delivery, "_upload_sky", lambda output, src: calls.append((output, src)))
    delivery.deliver("s3://bucket/prefix/x", "/tmp/model")
    delivery.deliver("gs://b", "/tmp/model")
    assert calls == [("s3://bucket/prefix/x", "/tmp/model"), ("gs://b", "/tmp/model")]


def test_upload_sky_resolves_store_type_from_uri(monkeypatch):
    import sky.data as sky_data

    created = {}

    class FakeStorage:
        def __init__(self, name, source, _bucket_sub_path=None):
            created.update(name=name, source=source, sub=_bucket_sub_path)

        def add_store(self, store_type):
            created["store"] = store_type

        def sync_all_stores(self):
            created["synced"] = True

    monkeypatch.setattr(sky_data, "Storage", FakeStorage)
    delivery._upload_sky("s3://bucket/models/run1", "/tmp/model")
    assert created["name"] == "bucket" and created["sub"] == "models/run1"
    assert created["store"].name == "S3" and created["synced"]


def test_deliver_rejects_local(monkeypatch):
    with pytest.raises(ValueError, match="remote outputs"):
        delivery.deliver("./out", "/tmp/model")


def test_deliver_hf_uploads_folder(monkeypatch):
    import sys
    import types

    events = []

    class FakeApi:
        def create_repo(self, repo_id, exist_ok, private):
            events.append(("create", repo_id, exist_ok, private))

        def upload_folder(self, repo_id, folder_path):
            events.append(("upload", repo_id, folder_path))

    fake_hub = types.SimpleNamespace(HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    delivery.deliver("hf://org/adapter", "/tmp/model")
    assert events == [
        ("create", "org/adapter", True, True),
        ("upload", "org/adapter", "/tmp/model"),
    ]


def test_self_terminate_detaches_sky_down(monkeypatch):
    import subprocess

    popens = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda cmd, **kw: popens.append((cmd, kw.get("start_new_session")))
    )
    delivery.self_terminate("run-head")
    assert popens == [(["sky", "down", "-y", "run-head"], True)]
