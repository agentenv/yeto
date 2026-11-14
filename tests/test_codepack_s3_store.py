"""The ``s3://`` code pack store reaches the run's object store: the store's
endpoint, credentials, timeouts and retries (``yeto.cloud.s3_store``), not a bare
``boto3.client("s3")`` that knows only the AWS environment."""

from __future__ import annotations

import http.server
import threading

import boto3
import pytest

from yeto import codepack
from yeto.cloud import s3_store


class _Bucket(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    objects: dict[str, bytes] = {}
    auth: list[str] = []
    refuse_next_put = 0

    def _answer(self, status: int, body: bytes = b"") -> None:
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self):  # noqa: N802
        type(self).auth.append(self.headers.get("Authorization", ""))
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if type(self).refuse_next_put:
            type(self).refuse_next_put -= 1
            return self._answer(503, b"<Error><Code>ServiceUnavailable</Code></Error>")
        type(self).objects[self.path] = body
        self._answer(200)

    def do_GET(self):  # noqa: N802
        body = type(self).objects.get(self.path)
        if body is None:
            return self._answer(404, b"<Error><Code>NoSuchKey</Code></Error>")
        self._answer(200, body)

    def log_message(self, *args):
        pass


@pytest.fixture
def bucket(monkeypatch):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
                 "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", s3_store.PROFILE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    def bare(*args, **kwargs):  # would talk to AWS with the AWS environment only
        raise AssertionError("bare boto3.client ignores the store's endpoint and credentials")

    monkeypatch.setattr(boto3, "client", bare)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Bucket)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _Bucket.objects, _Bucket.auth, _Bucket.refuse_next_put = {}, [], 0
    monkeypatch.setenv(s3_store.ENDPOINT_ENV, f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv(s3_store.REGION_ENV, "auto")
    monkeypatch.setenv(s3_store.KEY_ID_ENV, "AKIDSTORE")
    monkeypatch.setenv(s3_store.SECRET_ENV, "store-secret")
    monkeypatch.setenv(s3_store.MAX_ATTEMPTS_ENV, "3")
    try:
        yield _Bucket
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_s3_code_pack_goes_to_the_store_endpoint_with_the_store_credentials(bucket):
    store = codepack.pack_store_for("s3://bkt/run")
    bucket.refuse_next_put = 1  # a 503 is retried, like every other store request
    uri = store.put("abc123", b"pack bytes")
    assert uri == f"s3://bkt/run/{codepack.PACK_DIR}/abc123.tar"
    assert bucket.objects == {f"/bkt/run/{codepack.PACK_DIR}/abc123.tar": b"pack bytes"}
    assert len(bucket.auth) == 2 and all("AKIDSTORE/" in a for a in bucket.auth)
    assert store.get(uri) == b"pack bytes"


def test_a_missing_s3_code_pack_is_a_code_pack_error(bucket):
    with pytest.raises(codepack.CodePackError, match="missing"):
        codepack.pack_store_for("s3://bkt/run").get(f"s3://bkt/run/{codepack.PACK_DIR}/gone.tar")
