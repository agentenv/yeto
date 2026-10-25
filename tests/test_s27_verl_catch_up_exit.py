"""S27 G-EC: the verl test exit hook (``driver_extras.TestExitPublisher``) ends only
the island's first process, so a relaunched full-parameter island (which starts
its publication count at 0 again) rejoins and finishes instead of exiting again.

CPU only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import driver_extras
from yeto.rl.engine.driver import MODAL_INCARNATION_ENV


class _Inner:
    def publish(self, state):
        return ("published", state.policy_version)


def _publish(publisher, versions, monkeypatch):
    exits = []
    monkeypatch.setattr(driver_extras.os, "_exit", lambda code: exits.append(code) or (_ for _ in ()).throw(
        SystemExit(code)))
    for version in versions:
        try:
            publisher.publish(SimpleNamespace(policy_version=version))
        except SystemExit:
            break
    return exits


@pytest.mark.parametrize("incarnation, rejoined, first, exits", [
    (None, False, 0, [17]),   # first process, first container
    ("0", False, 0, [17]),
    ("1", False, 0, []),      # a Modal re-run in a new container
    ("0", True, 0, []),       # a full-parameter relaunch that joined the running pool
    ("0", False, 3, []),      # a LoRA relaunch starts from the syncer's version
])
def test_the_exit_fires_in_the_first_process_only(monkeypatch, incarnation, rejoined, first, exits):
    environ = {} if incarnation is None else {MODAL_INCARNATION_ENV: incarnation}
    tape = []
    publisher = driver_extras.TestExitPublisher(_Inner(), 4, SimpleNamespace(append=tape.append),
                                                rejoined=lambda: rejoined, environ=environ)
    assert _publish(publisher, range(first, first + 8), monkeypatch) == exits
    armed = [record for record in tape if record["event"] == "rl_test_exit_armed"]
    assert len(armed) == 1 and armed[0]["armed"] is bool(exits)
    assert armed[0]["rejoined"] is rejoined and armed[0]["first_published_version"] == first
    assert [r["policy_version"] for r in tape if r["event"] == "rl_test_island_exit"] == ([4] if exits else [])


def test_without_a_version_nothing_is_armed(monkeypatch):
    tape = []
    publisher = driver_extras.TestExitPublisher(_Inner(), None, SimpleNamespace(append=tape.append), environ={})
    assert _publish(publisher, range(6), monkeypatch) == [] and tape == []
