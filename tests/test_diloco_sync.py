import threading

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
