"""Driver features of a verl full-parameter island beyond the sync (S27 G-DS).

Kept out of :mod:`.trainer` (which imports verl) so the wiring is testable on
the CPU without verl.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable

# Set by the Modal runner: 0 for an island's first container, k for its k-th
# re-run (the engine driver's fault injection reads the same variable).
from yeto.rl.engine.driver import MODAL_INCARNATION_ENV


def wire_full_parameter_extras(driver, island, plan: dict) -> None:
    """S27 G-DS: the driver's data position log and HF exports of a full-parameter
    island (``full_parameter_stream.full_parameter_extras``; nothing when the plan has neither).

    * data positions (``yeto.rl.engine.data_position``): the driver records the
      verl data cursor (:class:`.data_cursor.VerlDataCursor`) after every sync
      boundary and a relaunched process that joined the running pool seeks to
      the last record, as on a Miles island (``miles.entry.data_position_log_for``);
    * HF exports (``yeto.rl.engine.hf_export``): the driver exports the due
      rounds at their boundary with :class:`.hf_export.VerlHfWriter`;
    * observation (S27 G-EC, :func:`.train_options.wire_observation`): the
      driver's timeline, heartbeat and resource sampler, as on a Miles island."""
    store = plan.get("data_position_store")
    if store:
        from yeto.rl.engine.data_position import data_position_log
        from yeto.rl.engine.data_split import DataSlot

        slot = DataSlot.parse(plan.get("data_slot"))
        driver.data_positions = data_position_log(str(store), learner_id=int(plan["learner_id"]),
                                                  slot=None if slot.whole else slot.env())
    exports = plan.get("hf_exports")
    if exports:
        from .full_parameter_stream import _LoopRunner
        from .hf_export import hf_round_exports

        driver.hf_exports = hf_round_exports(exports, group=island.full_group, model_path=plan["model_path"],
                                             run=_LoopRunner().run)
    from .train_options import wire_observation  # S27 G-EC: timeline and samplers

    wire_observation(driver, plan)


class TestExitPublisher:
    """TEST ONLY (V2 leave/rejoin): end this island's process right after it
    published ``version``, in its first process only, so a relaunched island
    rejoins and finishes.

    A process is the first one when all hold (S27 G-EC; before, only the first):
    1. its first publication is policy version 0 (a relaunched LoRA island
       starts from the syncer's committed version > 0);
    2. it runs in the island's first Modal container (``YETO_MODAL_INCARNATION``
       unset or 0; the rule of the engine driver's ``exit_at_round``);
    3. ``rejoined()`` is False: a full-parameter island's process starts every
       publication count at 0, and one that joined a running pool (catch-up)
       is a relaunch."""

    __test__ = False  # not a pytest class

    def __init__(self, inner, version: int | None, tape, *, rejoined: Callable[[], bool] | None = None,
                 environ=None) -> None:
        self.inner, self.version, self.tape = inner, version, tape
        self.rejoined = rejoined or (lambda: False)
        self.environ = os.environ if environ is None else environ
        self.armed: bool | None = None if version is not None else False

    def publish(self, state):
        result = self.inner.publish(state)
        published = int(state.policy_version)
        if self.armed is None:
            incarnation = str(self.environ.get(MODAL_INCARNATION_ENV, "0") or "0")
            rejoined = bool(self.rejoined())
            self.armed = published == 0 and incarnation == "0" and not rejoined
            self.tape.append({"event": "rl_test_exit_armed", "armed": self.armed,
                              "first_published_version": published, "exit_after_version": self.version,
                              "modal_incarnation": incarnation, "rejoined": rejoined})
        if self.armed and published == self.version:
            self.tape.append({"event": "rl_test_island_exit", "policy_version": self.version,
                              "time_unix": time.time()})
            print(f"[yeto-verl] TEST exit after publishing v{self.version}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(17)
        return result
