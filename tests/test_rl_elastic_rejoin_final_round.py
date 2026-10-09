"""S17 M1: a re-JOINed elastic island ends on the syncer's final outer version, not on its
local round count (s17-m1-20261008a: island 1's local count ran one ahead after the rejoin,
so it left after v5 and the last outer step merged island 0 alone). CPU, no Ray."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from test_rl_engine_driver import _driver, _engine, _strict_config
from yeto.rl.engine.bridges import ElasticAvgSync


class _Client:
    """Scripted elastic client: the outer version returned after the n-th delta."""

    def __init__(self, versions_after_delta):
        self.versions = list(versions_after_delta)
        self.params = None
        self.deltas = []
        self.inner_step = 0
        self.final_outer_version = None
        self.latest_base = None
        self.left = False
        self._on_event = None

    def join(self):
        return SimpleNamespace(catch_up=False, base_version=0)

    def _base(self, version):
        self.latest_base = SimpleNamespace(params=tuple(self.params), outer_version=version)
        return self.latest_base

    def wait_base(self, *, newer_than, timeout_s):
        if self.params is None:
            return None
        if not self.deltas:
            return self._base(0)
        return self._base(self.versions[len(self.deltas) - 1])

    def elastic_init(self, params):
        self.params = list(params)

    def raise_if_finished(self):
        pass

    def delta_tensor(self, *, base_version, c_tokens, c_steps, update):
        self.deltas.append(base_version)

    def leave(self):
        self.left = True

    def close(self):
        pass


def _run(tmp_path, versions, rounds=3):
    engine = _engine(torch.tensor([1.0, 3.0]))
    client = _Client(versions)
    sync = ElasticAvgSync(_strict_config(tmp_path, engine, learner_id=1, rounds=rounds), client=client,
                          base_wait_s=1.0)
    _driver(engine, sync, tmp_path, learner_id=1).run()
    ev = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    return client, sync, ev


def test_rejoined_island_keeps_training_until_the_final_outer_version(tmp_path):
    # v1, then the catch-up re-JOIN hands back v1 again (delta refused), then v2, v3 (= final).
    client, sync, ev = _run(tmp_path, [1, 1, 2, 3])
    assert client.deltas == [0, 1, 1, 2]  # four local rounds: one more than global_rounds
    assert sync.base_version == 3 and client.left
    assert sum(e.get("event") == "rl_local_round" for e in ev) == 4
    extra = [e for e in ev if e.get("phase") == "elastic_extra_round"]
    assert [e["rollout_id"] for e in extra] == [3]


def test_no_rejoin_keeps_the_old_round_count(tmp_path):
    client, sync, ev = _run(tmp_path, [1, 2, 3])
    assert client.deltas == [0, 1, 2] and sync.base_version == 3
    assert not [e for e in ev if e.get("phase") == "elastic_extra_round"]


def test_base_jumping_to_final_stops_early(tmp_path):
    client, sync, _ = _run(tmp_path, [3])  # a late island whose first delta lands after the final step
    assert client.deltas == [0] and sync.base_version == 3


def test_events_carry_the_syncer_outer_version_next_to_the_local_counter(tmp_path):
    """Cross-island comparisons (hash checks, lateness, dashboard) must key on the syncer's
    version: after the rejoin the local policy_version runs one ahead (s17-g1-island, M1 run a)."""
    _, _, ev = _run(tmp_path, [1, 1, 2, 3])
    applies = [e for e in ev if e.get("event") == "rl_policy_apply"]
    assert [e["policy_version"] for e in applies] == [0, 1, 2, 3, 4]  # local counter (publication/ledger)
    assert [e["sync/outer_version"] for e in applies] == [0, 1, 1, 2, 3]  # syncer version
    pubs = [e for e in ev if e.get("event") == "rl_publication"]
    assert [e["sync/outer_version"] for e in pubs] == [0, 1, 1, 2, 3]
    rounds = [e for e in ev if e.get("event") == "rl_local_round"]
    assert [e["sync/base_outer_version"] for e in rounds] == [0, 1, 1, 2]


def test_dashboard_keys_island_applies_by_outer_version():
    from yeto.dashboard.reducer import Reducer

    red = Reducer()
    for iid, versions in ((0, [(1, 1), (2, 2)]), (1, [(1, 1), (2, 1), (3, 2)])):
        for local, outer in versions:
            red.feed({"event": "rl_policy_apply", "island_id": iid, "policy_version": local,
                        "sync/outer_version": outer, "time_unix": 1.0})
    assert red.applies[1] == {"0", "1"} or red.applies[1] == {0, 1}
    assert red.applies[2] == {"0", "1"} or red.applies[2] == {0, 1}
    assert 3 not in red.applies
