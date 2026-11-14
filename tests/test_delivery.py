"""--output classification, fetch/upload plumbing, self-termination gating."""

from __future__ import annotations

import http.server
import threading

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


def test_deliver_s3_goes_through_the_s3_upload(monkeypatch):
    calls = []
    monkeypatch.setattr(delivery, "_upload_s3", lambda output, src: calls.append((output, src)))
    monkeypatch.setattr(
        delivery,
        "_upload_sky",
        lambda output, src: pytest.fail(f"unexpected sky upload: {output}"),
    )
    delivery.deliver("s3://bucket/prefix/x", "/tmp/model")
    assert calls == [("s3://bucket/prefix/x", "/tmp/model")]


def test_deliver_non_s3_object_store_goes_through_sky_storage(monkeypatch):
    calls = []
    monkeypatch.setattr(
        delivery,
        "_upload_s3",
        lambda output, src: pytest.fail(f"unexpected s3 upload: {output}"),
    )
    monkeypatch.setattr(delivery, "_upload_sky", lambda output, src: calls.append((output, src)))
    delivery.deliver("gs://b", "/tmp/model")
    assert calls == [("gs://b", "/tmp/model")]


class _PutRecorder(http.server.BaseHTTPRequestHandler):
    puts: list = []

    def do_PUT(self):  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).puts.append((self.path, body))
        self.send_response(200)
        self.send_header("ETag", '"e"')
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def test_upload_s3_goes_through_the_store_client_and_its_endpoint(tmp_path, monkeypatch):
    """An s3:// output of a run whose store is R2/MinIO (YETO_RL_STORE_S3_ENDPOINT_URL)
    goes to that endpoint with the store credentials, not to AWS via the aws CLI."""
    import boto3

    from yeto.cloud import s3_store

    (tmp_path / "sub").mkdir()
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "sub" / "w.bin").write_bytes(b"weights")
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("bare boto3.client: default chain, AWS"))
    monkeypatch.setattr(delivery.subprocess, "run", lambda *a, **k: pytest.fail(f"subprocess {a}"))
    _PutRecorder.puts = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PutRecorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for name in ("AWS_PROFILE", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", s3_store.PROFILE_ENV):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(s3_store.ENDPOINT_ENV, f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv(s3_store.KEY_ID_ENV, "k")
        monkeypatch.setenv(s3_store.SECRET_ENV, "s")
        monkeypatch.setenv(s3_store.REGION_ENV, "auto")
        delivery._upload_s3("s3://bucket/models/run1/", str(tmp_path))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    got = {path.split("?")[0]: body for path, body in _PutRecorder.puts}
    assert got == {"/bucket/models/run1/config.json": b"{}", "/bucket/models/run1/sub/w.bin": b"weights"}


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
    delivery._upload_sky("gs://bucket/models/run1", "/tmp/model")
    assert created["name"] == "bucket" and created["sub"] == "models/run1"
    assert created["store"].name == "GCS" and created["synced"]


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
