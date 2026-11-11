"""``start_on_latest``: a (re)started engine sampler starts its engine on the store's
newest release instead of generating a first round on the base weights."""
from __future__ import annotations

import json

import pytest

from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine.role_switch import RoleSwitchError

from test_rl_k31_e2e_cpu import _release_dir
from test_stage1_gaps import _sampler_spec

pytestmark = pytest.mark.usefixtures("ports_v1_pin")


def _events(tmp_path):
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    return [json.loads(line) for line in tape.read_text().splitlines()]


def _store_with_release(spec, manifest):
    releases, _ = entry.build_stores(spec)
    releases.release(3, "h3", lambda d: _release_dir(d), manifest=manifest)
    return releases


def test_start_on_latest_starts_the_engine_on_the_newest_release(tmp_path):
    spec = _sampler_spec(tmp_path, num_rollouts=1, start_on_latest=True)
    _store_with_release(spec, {})
    assert entry.run(spec, resource_sampler=lambda emit: None) == 1
    events = _events(tmp_path)
    (start,) = [e for e in events if e["event"] == "rl_sampler_start_on_latest"]
    assert start["base_version"] == 3 and "policy-releases" in start["base_dir"]
    loads = [e for e in events if e["event"] == "rl_sampler_load_checked"]
    assert [e["policy_version"] for e in loads] == [3]  # started on it, never reloaded
    rounds = [e for e in events if e["event"] == "rl_sampler_round"]
    assert rounds and all(e["policy_version"] == 3 for e in rounds)


def test_default_off_starts_on_the_base_and_loads_the_release_later(tmp_path):
    spec = _sampler_spec(tmp_path, num_rollouts=1)
    _store_with_release(spec, {})
    entry.run(spec, resource_sampler=lambda emit: None)
    events = _events(tmp_path)
    assert not [e for e in events if e["event"] == "rl_sampler_start_on_latest"]
    assert [e for e in events if e["event"] == "rl_sampler_load_checked"][0]["policy_version"] == 0


def test_start_on_latest_without_a_release_keeps_the_base(tmp_path):
    spec = _sampler_spec(tmp_path, num_rollouts=1, start_on_latest=True)
    entry.run(spec, resource_sampler=lambda emit: None)
    (start,) = [e for e in _events(tmp_path) if e["event"] == "rl_sampler_start_on_latest"]
    assert start["base_version"] == 0 and start["base_dir"] is None


def test_start_on_latest_refuses_a_release_of_another_expert_selection(tmp_path):
    spec = _sampler_spec(tmp_path, num_rollouts=1, start_on_latest=True, expert_selection_sha256="a" * 64)
    _store_with_release(spec, {"expert_selection_sha256": "b" * 64})
    with pytest.raises(RoleSwitchError, match="carries selection"):
        entry.run(spec, resource_sampler=lambda emit: None)
