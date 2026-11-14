"""PoolIslandClient answers a PULL the syncer replays on a new link."""

from __future__ import annotations

import threading

import torch

from yeto.protocol import PullRequest
from yeto.rl.core import CanonicalTensorSpec, build_avg_layout
from yeto.rl.pool_client import PoolClientConfig, PoolIslandClient


class _Link:
    """The SyncerClient surface PoolIslandClient reads, with a settable link."""

    def __init__(self) -> None:
        self.connection_generation = 1
        self.lease_expiries = 0
        self.finalizing = threading.Event()
        self.shutdown = threading.Event()
        self.pulls: list[PullRequest] = []
        self.pushes: list[tuple[int, int, int]] = []

    def check_health(self) -> None:
        pass

    def drain_updates(self):
        return []

    def drain_pulls(self):
        pulls, self.pulls = self.pulls, []
        return pulls

    def push_fragment(self, fragment_id, step, attempt, base, local_step, c_steps, c_tokens, data):
        self.pushes.append((self.connection_generation, step, attempt))


def _client(link: _Link) -> PoolIslandClient:
    layout = build_avg_layout((CanonicalTensorSpec("base_model.model.layer.lora_A.weight", (2, 2), "float32", 4),))
    return PoolIslandClient(PoolClientConfig(("127.0.0.1", 1), 0), layout,
                            client_factory=lambda *args, **kwargs: link)


def test_a_pull_replayed_on_a_new_link_is_answered_again():
    # The push answering PULL (1, attempt 1) was queued on link 1, which
    # dropped before the bytes left; the syncer replays the same PULL on
    # link 2.  The island must answer it, not treat it as already answered.
    link = _Link()
    client = _client(link)
    link.pulls = [PullRequest(0, 1, 1)]
    update = torch.zeros(4)
    assert client.delta_tensor(base_version=0, c_tokens=1, c_steps=1, update=update, timeout_s=1.0) == 1

    link.connection_generation = 2
    link.pulls = [PullRequest(0, 1, 1)]
    assert client.delta_tensor(base_version=0, c_tokens=1, c_steps=1, update=update, timeout_s=1.0) == 1
    assert link.pushes == [(1, 1, 1), (2, 1, 1)]


def test_a_pull_repeated_on_the_same_link_is_not_answered_twice():
    link = _Link()
    client = _client(link)
    link.pulls = [PullRequest(0, 1, 1)]
    update = torch.zeros(4)
    client.delta_tensor(base_version=0, c_tokens=1, c_steps=1, update=update, timeout_s=1.0)
    link.pulls = [PullRequest(0, 1, 1)]
    client._drain()
    assert client._permits == {}
