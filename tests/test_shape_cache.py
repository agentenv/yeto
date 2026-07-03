"""Tests for the shape disk TTL cache (no network, no cloud SDKs)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from yeto.shape import cache as cache_mod
from yeto.shape.cache import TTLCache


def _counting_fetch(value="v"):
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return value

    return fetch, calls


def test_get_or_fetches_then_caches(tmp_path):
    c = TTLCache(path=tmp_path / "c.json", ttl=100.0)
    fetch, calls = _counting_fetch("hello")
    assert c.get_or("k", fetch) == "hello"
    assert c.get_or("k", fetch) == "hello"
    assert calls["n"] == 1


def test_ttl_expiry_refetches(tmp_path, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod, "_now", lambda: now["t"])
    c = TTLCache(path=tmp_path / "c.json", ttl=10.0)
    fetch, calls = _counting_fetch()
    c.get_or("k", fetch)
    now["t"] += 5.0
    c.get_or("k", fetch)
    assert calls["n"] == 1  # still fresh
    now["t"] += 6.0  # 11s since fetch, past ttl
    c.get_or("k", fetch)
    assert calls["n"] == 2


def test_fetch_exception_propagates_and_is_not_cached(tmp_path):
    c = TTLCache(path=tmp_path / "c.json")
    calls = {"n": 0}

    def bad_fetch():
        calls["n"] += 1
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        c.get_or("k", bad_fetch)
    with pytest.raises(RuntimeError):
        c.get_or("k", bad_fetch)
    assert calls["n"] == 2
    assert not (tmp_path / "c.json").exists()


def test_disabled_bypasses_persistence(tmp_path):
    path = tmp_path / "c.json"
    c = TTLCache(path=path, enabled=False)
    fetch, calls = _counting_fetch()
    c.get_or("k", fetch)
    c.get_or("k", fetch)
    assert calls["n"] == 2
    assert not path.exists()


def test_persistence_across_instances(tmp_path):
    path = tmp_path / "c.json"
    TTLCache(path=path).get_or("k", lambda: 42)
    fetch, calls = _counting_fetch()
    assert TTLCache(path=path).get_or("k", fetch) == 42
    assert calls["n"] == 0


def test_corrupt_file_starts_empty(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json!!")
    c = TTLCache(path=path)
    assert c.get_or("k", lambda: "fresh") == "fresh"


def test_in_flight_dedup(tmp_path):
    c = TTLCache(path=tmp_path / "c.json")
    calls = {"n": 0}

    def slow_fetch():
        calls["n"] += 1
        time.sleep(0.05)
        return "shared"

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(c.get_or("k", slow_fetch)))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1
    assert results == ["shared"] * 8


def test_per_call_ttl_overrides_default(tmp_path, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod, "_now", lambda: now["t"])
    c = TTLCache(path=tmp_path / "c.json", ttl=100.0)
    fresh_fetch, fresh_calls = _counting_fetch("slow-moving")
    stale_fetch, stale_calls = _counting_fetch("fast-moving")
    c.get_or("limits", fresh_fetch)  # instance default: 100s
    c.get_or("usage", stale_fetch, ttl=5.0)
    now["t"] += 6.0  # past the override, well within the default
    c.get_or("limits", fresh_fetch)
    c.get_or("usage", stale_fetch, ttl=5.0)
    assert fresh_calls["n"] == 1  # default ttl: still cached
    assert stale_calls["n"] == 2  # 5s override: expired independently
    # ttl=None means the instance default, not "expired".
    c.get_or("usage", stale_fetch, ttl=None)
    assert stale_calls["n"] == 2


def test_merge_on_save_across_instances(tmp_path):
    path = tmp_path / "c.json"
    TTLCache(path=path).get_or("key1", lambda: 1)
    TTLCache(path=path).get_or("key2", lambda: 2)
    third = TTLCache(path=path)
    fetch, calls = _counting_fetch()
    assert third.get_or("key1", fetch) == 1
    assert third.get_or("key2", fetch) == 2
    assert calls["n"] == 0


def test_save_creates_lock_file(tmp_path):
    path = tmp_path / "c.json"
    TTLCache(path=path).get_or("k", lambda: 1)
    assert (tmp_path / "c.json.lock").exists()


def test_interleaved_instance_writes_preserve_both_keys(tmp_path):
    """Two live instances writing in turn must not clobber each other.

    Both instances load the (empty) file before either saves, so without
    merge-on-save the second write would drop the first instance's entry.
    """
    path = tmp_path / "c.json"
    a = TTLCache(path=path)
    b = TTLCache(path=path)
    a.get_or("k1", lambda: "from-a")
    b.get_or("k2", lambda: "from-b")
    on_disk = json.loads(path.read_text())
    assert set(on_disk) == {"k1", "k2"}
    fetch, calls = _counting_fetch()
    fresh = TTLCache(path=path)
    assert fresh.get_or("k1", fetch) == "from-a"
    assert fresh.get_or("k2", fetch) == "from-b"
    assert calls["n"] == 0


def test_save_drops_expired_entries(tmp_path, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(cache_mod, "_now", lambda: now["t"])
    path = tmp_path / "c.json"
    c = TTLCache(path=path, ttl=10.0)
    c.get_or("old", lambda: 1)
    now["t"] += 20.0
    c.get_or("new", lambda: 2)
    on_disk = json.loads(path.read_text())
    assert "new" in on_disk and "old" not in on_disk
