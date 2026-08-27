import threading

import pytest
import torch

from yeto.diloco_sync import DiLoCoSyncState, sync_diloco_boundary
from yeto.fragments import Fragment, FragmentLayout, MERGE_AVG
from yeto.protocol import BcastFragment, DTYPE_F32, FinalManifest, PullRequest
from yeto.tensor_io import apply_fragment, pack_tensor, unpack_fragment


class _Client:
    def __init__(self, *, updates=(), pulls=()):
        self.dtype = DTYPE_F32
        self.finalizing = threading.Event()
        self.finalized = threading.Event()
        self.shutdown = threading.Event()
        self.updates = list(updates)
        self.pulls = list(pulls)
        self.pushes = []
        self.events = []
        self.shutdown_on_drain_pulls = False

    def check_health(self):
        self.events.append("health")

    def drain_updates(self):
        self.events.append("updates")
        updates, self.updates = self.updates, []
        return updates

    def drain_pulls(self):
        self.events.append("pulls")
        if self.shutdown_on_drain_pulls:
            self.shutdown.set()
        pulls, self.pulls = self.pulls, []
        return pulls

    def push_fragment(self, *args):
        self.events.append("push")
        self.pushes.append(args)


def _layout(*names):
    return FragmentLayout(
        [Fragment(MERGE_AVG, [(name, 2)]) for name in names]
    )


def test_update_resets_self_clock_before_pending_pull_is_answered():
    layout = _layout("weight")
    raw_global = torch.tensor([4.0, 6.0])
    client = _Client(
        updates=[BcastFragment(0, 7, pack_tensor(raw_global, DTYPE_F32))],
        pulls=[PullRequest(0, 5, 2)],
    )
    params = {"weight": torch.tensor([10.0, 14.0])}
    state = DiLoCoSyncState.create(1, track_anchors=True)

    finalized = sync_diloco_boundary(
        client,
        layout,
        state,
        steps_total=3,
        units_total=96,
        merge_alpha=0.25,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
        finalize=None,
        device=torch.device("cpu"),
    )

    assert not finalized
    assert client.events == ["health", "updates", "pulls"]
    assert client.pushes == []
    assert len(state.pending_pulls) == 1
    assert torch.equal(state.anchors[0], raw_global)
    assert torch.equal(params["weight"], torch.tensor([5.5, 8.0]))
    assert state.steps_at_reset == [3]
    assert state.units_at_reset == [96]
    assert state.fragment_versions == [7]
    assert state.global_step == 7

    params["weight"].copy_(torch.tensor([7.5, 11.0]))
    sync_diloco_boundary(
        client,
        layout,
        state,
        steps_total=4,
        units_total=128,
        merge_alpha=0.25,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
        finalize=None,
        device=torch.device("cpu"),
    )

    assert client.events[-3:] == ["updates", "pulls", "push"]
    assert state.pending_pulls == []
    push = client.pushes[0]
    assert push[:7] == (0, 5, 2, 7, 4, 1, 32)
    delta = unpack_fragment(layout.fragments[0], push[7], DTYPE_F32)
    assert torch.equal(delta, torch.tensor([3.5, 5.0]))


def test_merge_and_push_use_separate_lazy_backend_snapshots():
    layout = _layout("updated", "pushed")
    client = _Client(
        updates=[
            BcastFragment(
                0,
                4,
                pack_tensor(torch.tensor([2.0, 4.0]), DTYPE_F32),
            )
        ],
        pulls=[PullRequest(1, 3, 1)],
    )
    params = {
        "updated": torch.tensor([6.0, 8.0]),
        "pushed": torch.tensor([10.0, 12.0]),
    }
    state = DiLoCoSyncState.create(2, track_anchors=True)
    state.anchors[1] = torch.tensor([7.0, 9.0])
    snapshots = []

    def snapshot_params():
        snapshot = {name: value.clone() for name, value in params.items()}
        snapshots.append(snapshot)
        return snapshot

    sync_diloco_boundary(
        client,
        layout,
        state,
        steps_total=2,
        units_total=20,
        merge_alpha=0.5,
        snapshot_params=snapshot_params,
        apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
        finalize=None,
    )

    assert len(snapshots) == 2
    assert torch.equal(params["updated"], torch.tensor([4.0, 6.0]))
    pushed = unpack_fragment(layout.fragments[1], client.pushes[0][7], DTYPE_F32)
    assert torch.equal(pushed, torch.tensor([3.0, 3.0]))


def test_finalization_and_shutdown_poll_timing_are_parameterized():
    layout = _layout("weight")
    params = {"weight": torch.zeros(2)}
    state = DiLoCoSyncState.create(1, track_anchors=True)
    state.global_step = 3
    client = _Client()
    client.finalizing.set()
    calls = []

    finalized = sync_diloco_boundary(
        client,
        layout,
        state,
        steps_total=5,
        units_total=50,
        merge_alpha=0.0,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
        finalize=lambda: calls.append("finalize") or FinalManifest(9, (4,)),
    )

    assert finalized
    assert calls == ["finalize"]
    assert client.events == ["health"]
    assert state.global_step == 9
    assert state.shutdown

    before_client = _Client()
    before_client.shutdown_on_drain_pulls = True
    before_state = DiLoCoSyncState.create(1, track_anchors=True)
    sync_diloco_boundary(
        before_client,
        layout,
        before_state,
        steps_total=1,
        units_total=1,
        merge_alpha=0.0,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: None,
        finalize=None,
    )
    assert not before_state.shutdown

    after_client = _Client()
    after_client.shutdown_on_drain_pulls = True
    after_state = DiLoCoSyncState.create(1, track_anchors=True)
    sync_diloco_boundary(
        after_client,
        layout,
        after_state,
        steps_total=1,
        units_total=1,
        merge_alpha=0.0,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: None,
        finalize=None,
        shutdown_after_pulls=True,
    )
    assert after_state.shutdown


def test_observer_reports_merge_and_push_activity():
    layout = _layout("weight")
    client = _Client(
        updates=[BcastFragment(0, 7, pack_tensor(torch.tensor([4.0, 6.0]), DTYPE_F32))],
        pulls=[PullRequest(0, 5, 2)],
    )
    params = {"weight": torch.tensor([10.0, 14.0])}
    state = DiLoCoSyncState.create(1, track_anchors=True)
    seen = []

    def boundary(**kwargs):
        return sync_diloco_boundary(
            client,
            layout,
            state,
            merge_alpha=0.0,
            snapshot_params=lambda: params,
            apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
            finalize=None,
            device=torch.device("cpu"),
            observer=seen.append,
            **kwargs,
        )

    # A boundary that applies a broadcast reports the merge, no push yet
    # (the pull arrived in the same drain and c_steps is still 0).
    boundary(steps_total=3, units_total=96)
    assert seen[0]["sync/merges_applied"] == 1
    assert seen[0]["sync/pushes"] == 0
    assert seen[0]["sync/pending_pulls"] == 1
    assert seen[0]["global_step"] == 7
    assert seen[0]["local_step"] == 3
    assert seen[0]["sync/staleness_max"] == 0  # fragment is at the newest version

    # The next boundary answers the pull: one push, its exact wire size, and
    # the delta's norm.
    params["weight"].copy_(torch.tensor([7.0, 10.0]))
    boundary(steps_total=4, units_total=128)
    assert seen[1]["sync/pushes"] == 1
    assert seen[1]["sync/pending_pulls"] == 0
    assert seen[1]["sync/push_bytes"] == len(client.pushes[0][7])
    assert seen[1]["sync/push_delta_norm"] == pytest.approx(5.0)  # |(3, 4)|


def test_observer_is_silent_on_a_boundary_where_nothing_moved():
    layout = _layout("weight")
    client = _Client()
    params = {"weight": torch.tensor([1.0, 2.0])}
    state = DiLoCoSyncState.create(1, track_anchors=True)
    seen = []

    sync_diloco_boundary(
        client,
        layout,
        state,
        steps_total=1,
        units_total=32,
        merge_alpha=0.0,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
        finalize=None,
        device=torch.device("cpu"),
        observer=seen.append,
    )
    # Most step boundaries merge nothing; logging them would bury the ones
    # that did under thousands of empty points.
    assert seen == []


def test_observer_reports_the_island_falling_behind():
    layout = _layout("a", "b")
    # Fragment 0 advances to global version 9, fragment 1 stays at 4: the
    # island is 5 versions stale on half its parameters.
    client = _Client(
        updates=[
            BcastFragment(1, 4, pack_tensor(torch.tensor([0.0, 0.0]), DTYPE_F32)),
            BcastFragment(0, 9, pack_tensor(torch.tensor([0.0, 0.0]), DTYPE_F32)),
        ]
    )
    params = {"a": torch.tensor([1.0, 1.0]), "b": torch.tensor([1.0, 1.0])}
    state = DiLoCoSyncState.create(2, track_anchors=True)
    seen = []

    sync_diloco_boundary(
        client,
        layout,
        state,
        steps_total=10,
        units_total=320,
        merge_alpha=0.0,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
        finalize=None,
        device=torch.device("cpu"),
        observer=seen.append,
    )
    assert seen[0]["sync/staleness_max"] == 5
    assert seen[0]["sync/staleness_mean"] == 2.5


def test_replicas_never_double_report():
    layout = _layout("weight")
    params = {"weight": torch.tensor([1.0, 2.0])}
    state = DiLoCoSyncState.create(1, track_anchors=False)
    seen = []

    sync_diloco_boundary(
        None,
        layout,
        state,
        steps_total=1,
        units_total=32,
        merge_alpha=0.0,
        snapshot_params=lambda: params,
        apply_flat=lambda fragment, flat: apply_fragment(fragment, flat, params),
        finalize=None,
        rank=1,
        world=1,
        device=torch.device("cpu"),
        observer=seen.append,
    )
    assert seen == []
