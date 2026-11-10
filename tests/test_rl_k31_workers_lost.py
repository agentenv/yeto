"""Miles k31/b4-lost efc83590 (fast-forwarded onto k31/b8): a cell whose workers died
is described ``workers_lost`` with the dead workers in ``lost_workers``.

* the pin and the feature: ``workers_lost`` is registered on the k31 head only;
* ``MilesRolloutPool.member_states`` passes the state and ``lost_workers`` through
  (None in every other state and on a fork that does not report the field);
* the watchdog's classification on the efc83590 ``describe_cells`` shape is
  ``workers_lost``; the explicit stop that acknowledges the loss reads ``stopped``;
* the runtime manifest probes the cell state property, and no capability needs it.
"""

from __future__ import annotations

import pytest

from yeto.rl.adapters.miles import pins
from yeto.rl.adapters.miles import runtime_manifest as rm

LOST = "efc83590dcb8daf9b230c71b45ffb4f1e6b54cb3"


def _pool(described):
    from yeto.rl.adapters.miles.rollout import MilesRolloutPool

    class Ctl:
        async def describe_cells(self):
            return described

    return MilesRolloutPool(inference_controller=Ctl(), rollout_executor=object(), metadata=None,
                            expected_policy=lambda: (1, "h"), declared_cells=sorted(described))


def _entry(state, lost=None, **kw):
    """One InferenceController.describe_cells entry of efc83590: the worker manager's
    CellInfo-like fields plus the controller's tracked / serving / awaiting_admission."""
    base = {"cell_id": "c", "alias": None, "pool_id": "engine", "deferred": False, "state": state,
            "lost_workers": lost, "generation": 1, "pg_name": "rollout", "pg_slot_offset": 0,
            "bundles": [0], "gpu_ids": [0], "tracked": False, "serving": False, "awaiting_admission": False}
    base.update(kw)
    return base


def test_the_k31_head_carries_workers_lost():
    assert pins.MILES_K31_B8_COMMIT == LOST and pins.pin_has("workers_lost", pins.MILES_NEXT_COMMIT)
    assert pins.K31_FEATURES["workers_lost"] == LOST
    assert pins.pin_has("workers_lost") and pins.pin_has("workers_lost", LOST)
    assert not pins.pin_has("workers_lost", pins.MILES_K31_B4_COMMIT)


def test_member_states_pass_the_worker_loss_through():
    pool = _pool({
        "c0": _entry("running", tracked=True, serving=True),
        "c1": _entry("workers_lost", ["engine-1-0", "engine-1-1"]),
        "c2": _entry("stopped"),
        "c3": _entry("unbound", deferred=True),
    })
    states = pool.member_states()
    assert {m: (s["state"], s["lost_workers"]) for m, s in states.items()} == {
        "engine:c0": ("running", None),
        "engine:c1": ("workers_lost", ["engine-1-0", "engine-1-1"]),
        "engine:c2": ("stopped", None),
        "engine:c3": ("unbound", None),
    }
    assert states["engine:c0"] == {"state": "running", "lost_workers": None, "tracked": True,
                                   "serving": True, "awaiting_admission": False}
    assert states["engine:c1"]["tracked"] is False and states["engine:c1"]["serving"] is False


def test_member_states_without_the_field_or_outside_the_state():
    # k31 before efc83590: no lost_workers key, a dead engine reads "stopped"
    old = {"state": "stopped", "tracked": False, "serving": False, "awaiting_admission": False}
    assert _pool({"c0": old}).member_states()["engine:c0"]["lost_workers"] is None
    # a stray list outside workers_lost is not reported as a loss
    stray = _entry("running", ["engine-0-0"], tracked=True, serving=True)
    assert _pool({"c0": stray}).member_states()["engine:c0"]["lost_workers"] is None
    # workers_lost with an empty list stays a (named-nobody) loss, not None
    assert _pool({"c0": _entry("workers_lost", [])}).member_states()["engine:c0"]["lost_workers"] == []


def test_recovery_reads_a_lost_cell_as_bound():
    """The restart recovery refuses only ``unbound`` targets: a cell in ``workers_lost``
    keeps its binding (only its actors are gone), so it is restartable like ``stopped``."""
    states = _pool({"c0": _entry("workers_lost", ["engine-0-0"])}).member_states()
    assert states["engine:c0"]["state"] != "unbound"


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _Ray:
    def get(self, ref, timeout=None):
        return ref


def test_watchdog_classifies_the_efc83590_shape():
    from yeto.rl.adapters.miles.elastic_wiring import _lost_workers_of

    described = {"c1": _entry("workers_lost", ["engine-1-0"]), "c2": _entry("stopped")}
    manager = type("M", (), {"describe_cells": _Remote(lambda pool_ids=None: described)})()
    assert _lost_workers_of(manager, _Ray(), "c1", 1.0) == ["engine-1-0"]
    # stop_cells on the lost cell acknowledges the loss: it reads "stopped" -> no_workers
    assert _lost_workers_of(manager, _Ray(), "c2", 1.0) is None


def test_runtime_manifest_probes_the_cell_state():
    assert rm.INTERFACES["fork_b4_workers_lost"] == (
        "miles.utils.workers.ray_worker_manager", "_CellManager.state")
    assert not [c for c, need in rm.CAPABILITY_REQUIREMENTS.items() if "fork_b4_workers_lost" in need]


def test_the_probe_resolves_in_an_efc83590_checkout():
    pytest.importorskip("ray")
    mod = pytest.importorskip("miles.utils.workers.ray_worker_manager")
    if not isinstance(getattr(getattr(mod, "_CellManager", None), "state", None), property):
        pytest.skip("Miles checkout before k31/b4-lost efc83590")
    from yeto.rl.engine.runtime_manifest import _probe

    assert _probe(*rm.INTERFACES["fork_b4_workers_lost"])
