"""The S3 store client's retries, timeouts and range listings
(``yeto.cloud.s3_store.client_config`` / ``S3ObjectStore.list_meta``)."""

from __future__ import annotations

import datetime
import http.server
import threading

import pytest

from tests.test_s3_store_throttled_download import record_sleeps
from yeto.cloud import s3_store

CREDS = {s3_store.KEY_ID_ENV: "AKIDSTORE", s3_store.SECRET_ENV: "store-secret"}


def test_default_client_config_is_standard_retries_nine_requests_and_a_short_connect_timeout():
    config = s3_store.client_config({})
    assert config == {"max_pool_connections": 128, "connect_timeout": 10.0, "read_timeout": 60.0,
                      "retries": {"mode": "standard", "total_max_attempts": 9}}
    client = s3_store._client({**CREDS, s3_store.ENDPOINT_ENV: "http://127.0.0.1:9"}, require_credentials=True)
    assert client.meta.config.retries == {"mode": "standard", "total_max_attempts": 9}
    assert client.meta.config.connect_timeout == 10.0 and client.meta.config.read_timeout == 60.0


def test_retry_mode_and_attempts_are_overridden_one_by_one():
    """main dropped its own retry config as soon as either AWS variable was set, so
    AWS_MAX_ATTEMPTS alone fell back to botocore's default mode ("legacy")."""
    retries = lambda env: s3_store.client_config(env)["retries"]  # noqa: E731
    assert retries({"AWS_MAX_ATTEMPTS": "4"}) == {"mode": "standard", "total_max_attempts": 4}
    assert retries({"AWS_RETRY_MODE": "adaptive"}) == {"mode": "adaptive", "total_max_attempts": 9}
    assert retries({"AWS_RETRY_MODE": "adaptive", "AWS_MAX_ATTEMPTS": "4", s3_store.RETRY_MODE_ENV: "standard",
                    s3_store.MAX_ATTEMPTS_ENV: "12"}) == {"mode": "standard", "total_max_attempts": 12}
    client = s3_store._client({**CREDS, "AWS_MAX_ATTEMPTS": "4"}, require_credentials=True)
    assert client.meta.config.retries == {"mode": "standard", "total_max_attempts": 4}
    for bad in ({s3_store.RETRY_MODE_ENV: "fast"}, {s3_store.MAX_ATTEMPTS_ENV: "0"},
                {s3_store.CONNECT_TIMEOUT_ENV: "0"}, {s3_store.READ_TIMEOUT_ENV: "-1"}):
        with pytest.raises(ValueError):
            s3_store.client_config(bad)
    timeouts = s3_store.client_config({s3_store.CONNECT_TIMEOUT_ENV: "3", s3_store.READ_TIMEOUT_ENV: "120"})
    assert (timeouts["connect_timeout"], timeouts["read_timeout"]) == (3.0, 120.0)


def test_the_client_settings_reach_an_island():
    for name in (s3_store.CONNECT_TIMEOUT_ENV, s3_store.READ_TIMEOUT_ENV, s3_store.RETRY_MODE_ENV,
                 s3_store.MAX_ATTEMPTS_ENV):
        assert name in s3_store.SETTING_ENVS
    envs = s3_store.island_envs("s3://bkt/run", {**CREDS, s3_store.MAX_ATTEMPTS_ENV: "12"})
    assert envs[s3_store.MAX_ATTEMPTS_ENV] == "12"


class _Refusing(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits: list[str] = []

    def do_GET(self):  # noqa: N802
        type(self).hits.append(self.path)
        body = (b'<?xml version="1.0" encoding="UTF-8"?><Error><Code>ServiceUnavailable</Code>'
                b"<Message>Reduce your concurrent request rate for the same object.</Message></Error>")
        self.send_response(503)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_a_listing_refused_with_503_is_sent_nine_times_with_jittered_waits(monkeypatch):
    """What a sampler does when R2 answers ``503 ServiceUnavailable: Reduce your
    concurrent request rate`` to every try: 9 requests, 8 waits of
    ``random() * min(2 ** n, 20)`` s (91 s at most), then the error."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Refusing)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    waits: list[float] = []
    record_sleeps(monkeypatch, waits.append, callers=("botocore",))  # its retry waits only
    _Refusing.hits = []
    try:
        client = s3_store._client({**CREDS, s3_store.ENDPOINT_ENV: f"http://127.0.0.1:{server.server_address[1]}",
                                   s3_store.REGION_ENV: "auto"}, require_credentials=True)
        objects = s3_store.S3ObjectStore("s3://bkt/run", client=client)
        with pytest.raises(Exception, match="ServiceUnavailable") as caught:
            objects.list_meta("task-dispense/")
    finally:
        server.shutdown()
        server.server_close()
    assert len(_Refusing.hits) == 9 and len(waits) == 8
    assert all(0 <= wait <= min(2 ** n, 20) for n, wait in enumerate(waits)) and sum(waits) <= 91
    assert caught.value.response["ResponseMetadata"]["RetryAttempts"] == 8


class _PagedS3:
    """``list_objects_v2`` over sorted keys, ``page`` keys an answer; ``requests``
    counts the answers."""

    def __init__(self, keys, page=3):
        self.keys, self.page, self.requests = sorted(keys), page, 0
        self.asked: list[dict] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        fake = self

        class Paginator:
            def paginate(self, Bucket, Prefix, StartAfter=""):  # noqa: N803
                fake.asked.append({"Prefix": Prefix, "StartAfter": StartAfter})
                keys = [k for k in fake.keys if k.startswith(Prefix) and k > StartAfter]
                when = datetime.datetime(2026, 11, 12, tzinfo=datetime.timezone.utc)
                for i in range(0, max(len(keys), 1), fake.page):
                    fake.requests += 1
                    yield {"Contents": [{"Key": k, "LastModified": when} for k in keys[i:i + fake.page]],
                           "ResponseMetadata": {"HTTPHeaders": {"date": "Thu, 12 Nov 2026 00:00:10 GMT"}}}

        return Paginator()


def test_a_range_listing_starts_after_a_key_and_stops_at_the_first_page_past_its_end():
    client = _PagedS3([f"run/ns/done/{i:04d}.trained" for i in range(30)] + ["run/ns/fail/0001/0000.infra"])
    objects = s3_store.S3ObjectStore("s3://bkt/run", client=client)
    keys, now = objects.list_meta("ns/done/")
    assert len(keys) == 30 and client.requests == 10
    assert now == datetime.datetime(2026, 11, 12, 0, 0, 10, tzinfo=datetime.timezone.utc).timestamp()
    client.requests = 0
    keys, _now = objects.list_meta("ns/done/", start_after="ns/done/0010", end="ns/done/0014")
    assert [k for k, _t in keys] == [f"ns/done/{i:04d}.trained" for i in range(10, 14)]
    assert client.requests == 2  # 0010-0012, then the page that reaches 0014: not the 5 more to the end
    assert client.asked[-1] == {"Prefix": "run/ns/done/", "StartAfter": "run/ns/done/0010"}
    client.requests = 0
    keys, _now = objects.list_meta("ns/done/", start_after="ns/done/0027")
    assert [k for k, _t in keys] == [f"ns/done/{i:04d}.trained" for i in (27, 28, 29)] and client.requests == 1
    assert objects.list_meta("ns/done/", start_after="ns/done/0040", end="ns/done/0050")[0] == []
