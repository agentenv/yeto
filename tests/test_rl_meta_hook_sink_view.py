"""E1: one metadata-hook run reads each sink value once and looks the sink actor up once."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import rollout_meta_hook as hook


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _FakeRay:
    def __init__(self):
        self.values = {"get_token": "yeto:4:" + "a" * 64, "get_limit": 1}
        self.lookups = 0
        self.calls = []

    def get_actor(self, name):
        self.lookups += 1
        ray = self

        class Actor:
            def __getattr__(self, method):
                def call(*args):
                    ray.calls.append(method)
                    if method.startswith("set_"):
                        ray.values["get_" + method[4:]] = args[0]
                        return None
                    return ray.values.get(method)
                return _Remote(call)

        return Actor()

    def get(self, value):
        return value


@pytest.fixture()
def fake_ray(monkeypatch):
    ray = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get_actor=ray.get_actor, get=ray.get))
    monkeypatch.setenv(hook.META_SINK_ENV, "ray:yeto_rollout_meta")
    return ray


def test_reads_inside_one_view_hit_the_actor_once(fake_ray):
    with hook.one_sink_view():
        for _ in range(5):
            assert hook.current_policy_token() == "yeto:4:" + "a" * 64
            assert hook.max_policy_age_override() == 1
        with hook.one_sink_view():  # nested: the same view
            hook.current_policy_token()
    assert fake_ray.calls == ["get_token", "get_limit"] and fake_ray.lookups == 1


def test_outside_a_view_every_read_goes_to_the_actor(fake_ray):
    hook.current_policy_token()
    hook.current_policy_token()
    assert fake_ray.calls == ["get_token", "get_token"] and fake_ray.lookups == 2


def test_a_write_inside_the_view_is_read_back(fake_ray):
    with hook.one_sink_view():
        assert hook.max_policy_age_override() == 1
        hook.put_max_policy_age(0)
        assert hook.max_policy_age_override() == 0
    assert hook.max_policy_age_override() == 0


def test_the_hook_limit_is_read_once_per_run(fake_ray):
    args = SimpleNamespace(yeto_rl_max_policy_age=2)
    with hook.one_sink_view():
        assert [hook._carry.max_policy_age(args) for _ in range(64)] == [1] * 64
    assert fake_ray.calls.count("get_limit") == 1
