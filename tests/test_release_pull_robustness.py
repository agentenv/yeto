"""A sampler that cannot pull a release must not serve the old weights for ever.

CPU, fakes (a directory object store; no cloud, no engine)."""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError

from tests.test_rl_object_store import _slow_release_world
from yeto.rl.engine.policy_release import PolicyFollower, ReleaseNotLoaded, SchemeMismatch
from yeto.rl.engine.sampler_loop import SamplerLoop, StoreReadOutage


def access_denied() -> ClientError:
    return ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"},
                        "ResponseMetadata": {"HTTPStatusCode": 403}}, "GetObject")


def _failing_download(monkeypatch, error_factory):
    from yeto.rl.engine import policy_release

    def download(objects, key, path):
        raise error_factory()

    monkeypatch.setattr(policy_release, "download_file", download)


def _loop(follower, now):
    events = []
    loop = SamplerLoop(follower=follower, buffer=None, generate=lambda rid: None, num_rollouts=None,
                       max_backlog_groups=1, emit=lambda name, **f: events.append((name, f)),
                       clock=lambda: now[0])
    return loop, events


def _download_until_reported(follower):
    for _ in range(200):
        try:
            follower.refresh()
        except ReleaseNotLoaded as error:
            return error
        if follower._fetch is not None:
            follower._fetch.done.wait(5)
    raise AssertionError("the failed download was never reported")


@pytest.mark.parametrize("background", [True, False])
def test_a_download_refused_by_the_store_is_raised_not_retried(tmp_path, monkeypatch, background):
    trainer, sampler, gate, _, weights = _slow_release_world(tmp_path, monkeypatch)
    trainer.release(1, "h1", weights)
    _failing_download(monkeypatch, access_denied)
    follower = PolicyFollower(sampler, lambda r, d: None, background_download=background)
    with pytest.raises(ClientError):
        if background:
            _download_until_reported(follower)
        else:
            follower.refresh()
    assert follower.version == 0


def test_a_download_that_keeps_failing_ends_the_island_after_the_outage_bound(tmp_path, monkeypatch):
    trainer, sampler, gate, _, weights = _slow_release_world(tmp_path, monkeypatch)
    trainer.release(1, "h1", weights)
    _failing_download(monkeypatch, lambda: ConnectionResetError("reset by peer"))
    now = [0.0]
    follower = PolicyFollower(sampler, lambda r, d: None, background_download=True,
                              download_retry_s=0.0)
    loop, events = _loop(follower, now)
    loop.max_read_outage_s = 100.0
    with pytest.raises(StoreReadOutage):
        for step in range(400):
            now[0] = float(step)
            loop._load_newest()
            if follower._fetch is not None:
                follower._fetch.done.wait(5)
    assert now[0] > 100.0
    assert [f["reason"] for n, f in events if n == "rl_sampler_release_not_loaded"][0] == "download_failed"


def test_a_load_ends_the_download_outage(tmp_path, monkeypatch):
    trainer, sampler, gate, fail, weights = _slow_release_world(tmp_path, monkeypatch)
    gate.set()
    trainer.release(1, "h1", weights)
    fail["left"] = 1
    now = [0.0]
    follower = PolicyFollower(sampler, lambda r, d: None, background_download=True, download_retry_s=0.0)
    loop, _ = _loop(follower, now)
    loop.max_read_outage_s = 100.0
    for step in range(50):
        now[0] = float(step)
        loop._load_newest()
        if follower.version == 1:
            break
        if follower._fetch is not None:
            follower._fetch.done.wait(5)
    assert follower.version == 1
    trainer.release(2, "h2", weights)
    fail["left"] = 1
    for step in range(50):  # a new failure starts a new outage: not over the bound
        now[0] = 10_000.0 + step
        loop._load_newest()
        if follower.version == 2:
            break
        if follower._fetch is not None:
            follower._fetch.done.wait(5)
    assert follower.version == 2


def test_a_release_of_another_scheme_is_raised_by_refresh_not_skipped(tmp_path, monkeypatch):
    """prefetch() reads False for it; the next refresh raises, also while trials hold the weights."""
    trainer, sampler, gate, _, weights = _slow_release_world(tmp_path, monkeypatch)
    trainer.release(1, "h1", weights)
    follower = PolicyFollower(sampler, lambda r, d: None, background_download=True, scheme="w4a8-int4-channel")
    follower.hold = lambda: True
    assert follower.prefetch() is False
    with pytest.raises(SchemeMismatch):
        follower.refresh()
