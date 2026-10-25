"""Driver features of a verl full-parameter island beyond the sync (S27 G-DS).

Kept out of :mod:`.trainer` (which imports verl) so the wiring is testable on
the CPU without verl.
"""

from __future__ import annotations


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
