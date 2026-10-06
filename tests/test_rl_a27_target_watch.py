"""A27 (GPU 6r1/6r2 d2): a target engine killed during the member ``update_weights``.

The fork's call waits for the NCCL rendezvous of the dead engine with the controller lock
held; the publisher watches the target cells' workers meanwhile and fails the publication
(UPDATE_FAILED naming the dead member) so the transaction goes to REBUILD_OLD instead of
stalling. The fork's ``placement_group.update_weights`` aborts the update lock on any
BaseException (CancelledError included); the fake here mirrors that contract.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from yeto.rl.engine.miles_adapter.publish import (
    MilesPublisher,
    PublicationError,
    RayTargetLiveness,
    TargetWorkersLostError,
)
from yeto.rl.engine.ports import PublicationCause


class _Controller:
    def __init__(self) -> None:
        self.aborted = 0
        self.ended = 0

    async def wait_cells_tracked(self, cells, timeout_seconds):
        pass

    async def start_update_weights(self, members, expected_epoch):
        return SimpleNamespace(rollout_engines=[], snapshot_cell_id_to_hashes={c: "h" for c in members})

    async def abort_update_weights(self):
        self.aborted += 1

    async def end_update_weights(self, **_):
        self.ended += 1


class _Probe:
    """RayTargetLiveness-like: alive until ``die(cell)``."""

    def __init__(self) -> None:
        self.dead_cell = None
        self.snapshots = []
        self.polls = 0

    async def snapshot(self, cells):
        self.snapshots.append(list(cells))

    def die(self, cell):
        self.dead_cell = cell

    async def status(self):
        self.polls += 1
        return "alive" if self.dead_cell is None else f"dead: cell {self.dead_cell} has no workers (stopped)"


def _publisher(controller, update_weights, probe):
    pub = MilesPublisher(args=SimpleNamespace(), actor_model=None, rollout_executor=None,
                         inference_controller=controller, update_weights=update_weights)
    pub.liveness_probe = probe
    pub.target_watch_poll_s = 0.02
    pub.dead_target_grace_s = 0.1
    records = []
    pub.event_sink = lambda event, **fields: records.append((event, fields))
    return pub, records


def _blocked_update(controller, started: asyncio.Event):
    """The fork's update_weights blocked in the rendezvous; aborts the update lock when cancelled."""
    async def update_weights(*a, **k):
        started.set()
        try:
            await asyncio.Event().wait()  # never
        except BaseException:
            await controller.abort_update_weights()  # placement_group.update_weights: except BaseException
            raise
    return update_weights


def test_a_dead_target_fails_the_member_update_without_waiting_for_the_rendezvous():
    async def scenario():
        controller, probe, started = _Controller(), _Probe(), asyncio.Event()
        pub, records = _publisher(controller, _blocked_update(controller, started), probe)

        async def kill_later():
            await started.wait()
            await asyncio.sleep(0.05)
            probe.die("c2")

        t0 = time.monotonic()
        killer = asyncio.ensure_future(kill_later())
        with pytest.raises(PublicationError) as info:
            await pub._publish_members("tok", ["c2", "c3"], epoch=1)
        await killer
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0, f"waited {elapsed:.2f}s"
        assert info.value.cause == PublicationCause.UPDATE_FAILED
        assert sorted(info.value.engine_ids) == ["engine:c2"]  # the dead member, not both
        assert controller.aborted == 1, "the cancelled fork call released the update lock"
        assert controller.ended == 0
        assert probe.snapshots == [["c2", "c3"]]
        assert [e for e, _ in records] == ["target_workers_lost"]
        assert records[0][1]["lost_members"] == ["engine:c2"]
        assert "cancelled" in str(info.value.__cause__)

    asyncio.run(scenario())


def test_a_call_that_fails_on_its_own_after_the_loss_is_not_cancelled():
    """With the fork's bounded connect the call fails first; the grace period lets it."""
    async def scenario():
        controller, probe = _Controller(), _Probe()
        release = asyncio.Event()

        async def update_weights(*a, **k):
            await release.wait()
            await controller.abort_update_weights()
            raise RuntimeError("engine 0 failed to join weight update group")

        pub, _ = _publisher(controller, update_weights, probe)
        pub.dead_target_grace_s = 5.0

        async def die_then_fail():
            await asyncio.sleep(0.05)
            probe.die("c3")
            await asyncio.sleep(0.05)
            release.set()

        side = asyncio.ensure_future(die_then_fail())
        with pytest.raises(PublicationError) as info:
            await pub._publish_members("tok", ["c3"], epoch=1)
        await side
        assert isinstance(info.value.__cause__, TargetWorkersLostError)
        assert "failed on its own" in str(info.value.__cause__)
        assert controller.aborted == 1

    asyncio.run(scenario())


def test_a_healthy_member_update_runs_to_completion_under_the_watch():
    async def scenario():
        controller, probe = _Controller(), _Probe()
        calls = []

        async def update_weights(*a, members, **k):
            await asyncio.sleep(0.06)  # a few polls
            calls.append(list(members))
            await controller.end_update_weights()

        pub, records = _publisher(controller, update_weights, probe)
        # no engines in the fake controller -> the token verification step is what fails, not the watch
        with pytest.raises(PublicationError, match="update lock covers"):
            await pub._publish_members("tok", ["c2"], epoch=1)
        assert calls == [["c2"]]
        assert probe.polls >= 1
        assert records == []  # the watch saw nothing (the abort below comes from the token step's own cleanup)

    asyncio.run(scenario())


def test_an_unavailable_probe_leaves_the_update_unwatched():
    async def scenario():
        controller = _Controller()
        calls = []

        class Broken:
            async def snapshot(self, cells):
                raise RuntimeError("no RayWorkerManager here")

        async def update_weights(*a, members, **k):
            calls.append(list(members))
            await controller.end_update_weights()

        pub, _ = _publisher(controller, update_weights, Broken())
        with pytest.raises(PublicationError, match="update lock covers"):
            await pub._publish_members("tok", ["c2"], epoch=1)
        assert calls == [["c2"]]

    asyncio.run(scenario())


def test_the_watch_can_be_switched_off():
    async def scenario():
        controller, probe = _Controller(), _Probe()

        async def update_weights(*a, **k):
            await controller.end_update_weights()

        pub, _ = _publisher(controller, update_weights, probe)
        pub.watch_targets = False
        with pytest.raises(PublicationError, match="update lock covers"):
            await pub._publish_members("tok", ["c2"], epoch=1)
        assert probe.snapshots == []

    asyncio.run(scenario())


def test_the_liveness_probe_names_the_dead_cell():
    async def scenario():
        async def _ready(v):
            return v

        infos = {"c2": [SimpleNamespace(name="c2/w0", generation=1)],
                 "c3": [SimpleNamespace(name="c3/w0", generation=1)]}

        class _Remote:
            def __init__(self, fn):
                self.remote = fn

        class _Handle:
            __ray_ready__ = _Remote(lambda: _ready(None))

        class _Mgr:
            get_worker_infos = _Remote(lambda cell: _ready(infos[cell]))
            get_actor_handle = _Remote(lambda name, expected_generation: _ready(_Handle()))

        probe = RayTargetLiveness(manager=_Mgr(), ray_module=SimpleNamespace(
            exceptions=SimpleNamespace(RayActorError=KeyError)))
        await probe.snapshot(["c2", "c3"])
        assert await probe.status() == "alive" and probe.dead_cell is None
        infos["c3"] = []
        assert (await probe.status()).startswith("dead: cell c3")
        assert probe.dead_cell == "c3"

    asyncio.run(scenario())
