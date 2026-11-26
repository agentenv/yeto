"""The local-round lead cap of the asynchronous full-parameter rounds.

``--rl-sync-max-local-rounds-ahead K`` (island ``--sweep-max-local-rounds-ahead``)
makes an island wait at a boundary instead of training a K-th local round on
top of an unmerged push of its own.  K = 1 cuts the global steps into windows
of P (the pipeline depth) steps: each local round pushes one window from its
cut and the next round starts once the window merged.  With a fixed roster
(quorum 2 of 2) every step needs every island, so a fast island stays at most
one local round ahead of a slow one instead of training rounds whose progress
the next broadcast overwrites and whose c_steps inflate its tokens^2/steps
merge weight.

Covered here: the pairing rule on fakes (who blocks, the step window: late and
crossed PULLs of the window, the next window held, steps closed without the
island, a caught-up island, the last window, replays, deferred capture; the
half-timeout escape, the terminal phase, peer snapshots during the wait), the
profile and the launcher/island wiring (default off, hashes unchanged), and
two islands of the real pairing and session (:class:`FullParameterSync`)
against the real Rust syncer: a fast island and a slow island, with and
without the cap.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import test_rl_full_parameter_async as harness
from test_rl_full_parameter_async import (  # noqa: F401 - pytest fixtures
    ASYNC,
    _flag,
    _hmac_key,
    _plan,
    needs_cargo,
    syncer_binary,
)
from test_rl_full_parameter_sync import _PairedRole

from yeto.protocol import PullRequest
from yeto.rl.adapters.miles.full_parameter.role_stream import RoleSubmissionCandidate
from yeto.rl.engine import full_parameter_sync as pairing_module
from yeto.rl.engine.full_parameter_sync import (
    LEAD_WAIT_POLL_S,
    FullParameterSync,
    FullParameterSyncProfile,
    RolePairing,
)

# -- the pairing rule on fakes ----------------------------------------------------


def _profile(**overrides):
    values = {"roles": ("actor",), "total_fragment_steps": 40, "fragments": 2, "local_horizon": 1,
              "pipeline_depth": 2, "wait_timeout": 900.0, "poll_seconds": 0.001, "partial_quorum": True}
    values.update(overrides)
    return FullParameterSyncProfile(**values)


def _candidate(global_step: int, local_step: int) -> RoleSubmissionCandidate:
    return RoleSubmissionCandidate(
        role="actor", permit=PullRequest((global_step - 1) % 2, global_step, 1),
        base_version=max(0, global_step - 2), local_step=local_step, c_steps=1, c_tokens=11,
        payload_bytes=8, submission_cut=object(), anchor=object())


class _ModuleShim:
    """``module`` with some attributes replaced."""

    def __init__(self, module, **replaced):
        self._module = module
        self.__dict__.update(replaced)

    def __getattr__(self, name):
        return getattr(self._module, name)


@pytest.fixture
def lead_clock(monkeypatch):
    """The pairing module's clock, advanced only by the module's own sleeps (the
    lead wait's polls): give-up and timeout arithmetic without a wall-clock bound,
    which a loaded host breaks by stretching real sleeps. Returns the clock."""
    now = [1024.0]
    real_sleep = asyncio.sleep

    async def sleep(delay, result=None):
        now[0] += delay
        await real_sleep(0)
        return result

    monkeypatch.setattr(pairing_module, "time", _ModuleShim(time, monotonic=lambda: now[0]))
    monkeypatch.setattr(pairing_module, "asyncio", _ModuleShim(asyncio, sleep=sleep))
    return lambda: now[0]


# One poll of float error on either side of a computed give-up time.
_EPS = 1e-9


def _pairing(role, **profile):
    role.fragment_versions = (0, 0)
    return RolePairing({"actor": role}, _profile(**profile))


def test_the_lead_cap_is_off_by_default_and_needs_the_streaming_profile():
    assert _profile().max_local_rounds_ahead is None
    assert _profile(max_local_rounds_ahead=1).max_local_rounds_ahead == 1
    with pytest.raises(ValueError, match="integer >= 1"):
        _profile(max_local_rounds_ahead=0)
    with pytest.raises(ValueError, match="streaming profile"):
        _profile(sweep=True, partial_quorum=False, fragments=2, pipeline_depth=2,
                  local_horizon=1, max_local_rounds_ahead=1)


def test_k_minus_one_rounds_may_train_on_top_of_an_unmerged_push():
    role = _PairedRole("actor", [_candidate(1, 1), _candidate(2, 1)], [], release_broadcasts=False)
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    assert sorted(pair.local_round for pair in pairing.inflight.values()) == [1, 1]
    assert len(pairing.lead_blockers()) == 2  # K = 1: no round on top of an unmerged push
    pairing.profile = replace(pairing.profile, max_local_rounds_ahead=2)
    assert pairing.lead_blockers() == []  # K = 2: one round on top is allowed
    pairing.local_round = 2
    assert len(pairing.lead_blockers()) == 2  # ... but not a second one
    off = _pairing(_PairedRole("actor", [_candidate(1, 1)], [], release_broadcasts=False))
    off.local_round = 1
    off.submit_ready()
    off.local_round = 9
    assert off.lead_blockers() == []


@pytest.mark.asyncio
async def test_the_wait_ends_when_the_blocking_round_merges_and_serves_peers_meanwhile():
    role = _PairedRole("actor", [_candidate(1, 1), _candidate(2, 1)], [], release_broadcasts=False)
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    served = []

    async def serve():
        served.append(time.monotonic())
        if len(served) == 3:
            role.release_broadcasts = True  # the peer pushed: both rounds merge
        return 0

    waited = await pairing.wait_for_lead(serve)
    assert waited > 0 and pairing.inflight == {}
    assert role.fragment_versions == (1, 2)
    assert len(served) >= 3
    assert (pairing.lead_waits, pairing.lead_wait_escapes) == (1, 0)


@pytest.mark.asyncio
async def test_the_wait_gives_up_at_half_the_wait_timeout_and_never_times_out_itself(lead_clock):
    role = _PairedRole("actor", [_candidate(1, 1)], [], release_broadcasts=False)
    pairing = _pairing(role, max_local_rounds_ahead=1, wait_timeout=0.4)
    pairing.local_round = 1
    pairing.submit_ready()
    started = lead_clock()
    waited = await pairing.wait_for_lead()
    # The first poll at or past half of 0.4 s gives up, long before the pair's 0.4 s timeout.
    assert 0.2 - _EPS <= lead_clock() - started < 0.2 + LEAD_WAIT_POLL_S + _EPS
    assert waited == pytest.approx(lead_clock() - started)
    assert (pairing.lead_waits, pairing.lead_wait_escapes) == (1, 1)
    assert len(pairing.inflight) == 1 and pairing.failure is None


@pytest.mark.asyncio
async def test_the_wait_also_covers_rounds_trained_before_the_next_pull_arrives(lead_clock):
    """After a merge and before the syncer's next PULLs nothing is in flight,
    but a round trained then lands in the next push all the same: with K = 1
    the island waits until the next window, pushed from its current cut, merged."""
    role = _PairedRole("actor", [_candidate(1, 1), _candidate(2, 1)], [])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    await pairing.commit_staged_broadcasts()
    assert pairing.inflight == {} and role.fragment_versions == (1, 2)
    assert await pairing.wait_for_lead() == 0.0  # window 1..2 pushed from this round's cut and merged
    pairing.local_round = 2  # trained one more round; the PULLs of window 3..4 are late
    assert pairing.lead_window() == (3, 4) and pairing.lead_window_open()
    served = []

    async def serve():
        served.append(lead_clock())
        if len(served) == 3:
            role.candidates.extend([_candidate(3, 2), _candidate(4, 2)])  # the next PULLs arrive
        return 0

    assert await pairing.wait_for_lead(serve) > 0
    assert len(served) >= 4 and ("actor", 3, True) in role.events and ("actor", 4, True) in role.events
    assert pairing.inflight == {} and role.fragment_versions == (3, 4)
    assert (pairing.lead_waits, pairing.lead_wait_escapes) == (1, 0)


@pytest.mark.asyncio
async def test_the_wait_for_a_late_pull_gives_up_at_half_the_wait_timeout(lead_clock):
    role = _PairedRole("actor", [_candidate(1, 1)], [])
    pairing = _pairing(role, max_local_rounds_ahead=1, wait_timeout=0.4)
    pairing.local_round = 1
    pairing.submit_ready()
    await pairing.commit_staged_broadcasts()
    pairing.local_round = 2
    started = lead_clock()  # the push of round 1 was retained at this time
    await pairing.wait_for_lead()
    assert 0.2 - _EPS <= lead_clock() - started < 0.2 + LEAD_WAIT_POLL_S + _EPS
    assert (pairing.lead_waits, pairing.lead_wait_escapes) == (1, 1)
    pairing.local_round = 3
    # Past the give-up time: one poll, then trains on.
    assert await pairing.wait_for_lead() == pytest.approx(LEAD_WAIT_POLL_S)
    assert pairing.lead_wait_escapes == 2


class _PeerPacedRole(_PairedRole):
    """A role whose broadcast of a step arrives only once the peer island
    pushed that step too (``peer_pushed``): a fixed roster's merge.
    ``close_without_me`` stages a step this island never pushed (a quorum
    below the island count, a soft deadline)."""

    def __init__(self, candidates, *, roster=None, learner=0):
        super().__init__("actor", candidates, [], release_broadcasts=False)
        self.peer_pushed: set[int] = set()
        self.roster = roster  # a shared _Roster: a step merges once every island pushed it
        self.learner = learner

    def push_candidate(self, candidate):
        submission = super().push_candidate(candidate)
        if self.roster is not None and submission is not None:
            self.roster.pushed.setdefault(candidate.permit.global_step, set()).add(self.learner)
        return submission

    def close_without_me(self, step):
        self._staged_steps.append(step)

    async def stage_available_broadcasts(self):
        def merged(step):
            if self.roster is not None:
                return self.roster.pushed.get(step, set()) >= self.roster.islands
            return step in self.peer_pushed

        for step in [step for step in self._queued_steps if merged(step)]:
            self._queued_steps.remove(step)
            self._staged_steps.append(step)
        return await super().stage_available_broadcasts()


class _Roster:
    """A fixed roster on fakes: every island must push a step before it merges."""

    def __init__(self, islands):
        self.islands = set(islands)
        self.pushed: dict[int, set[int]] = {}


def _pushed(role):
    return [step for _, step, _ in role.events]


@pytest.mark.asyncio
async def test_with_k_1_the_wait_answers_its_window_late_and_holds_the_next_window(lead_clock):
    """The window of round 1 is steps 1..2 (P = 2). Step 2's PULL arrives during
    the wait: it is still pushed from cut 1 (a fast island absorbs a late PULL of
    its window instead of pushing it from its next cut, where it would merge at
    once with the slow island's push from an older cut). Step 3's PULL (the next
    window) is held: pushing it from cut 1 would hold the wait one more merge, and
    a slow island would sit at its boundary while a fast island trained a round
    per merge. The held PULL goes out at the next boundary, from cut 2."""
    role = _PeerPacedRole([_candidate(1, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    assert pairing.lead_window() == (1, 2)
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 1:
            role.candidates.extend([_candidate(2, 1), _candidate(3, 1)])  # late, and the next window
        if len(served) == 3:
            role.peer_pushed.update({1, 2})  # the peer pushed window 1..2: it merges
        return 0

    assert await pairing.wait_for_lead(serve) > 0
    assert (pairing.lead_waits, pairing.lead_wait_escapes) == (1, 0)
    assert pairing.inflight == {} and role.fragment_versions == (1, 2)
    assert _pushed(role) == [1, 2]  # step 3 was not pushed from cut 1
    pairing.local_round = 2
    pairing.submit_ready()
    assert pairing.lead_window() == (3, 4)
    assert [(key[0], pair.local_round) for key, pair in pairing.inflight.items()] == [(3, 2)]


@pytest.mark.asyncio
async def test_with_k_1_the_next_round_waits_until_every_step_of_its_window_merged(lead_clock):
    """Step 1 merged and nothing is in flight, but step 2 of the window has no
    PULL yet: the island trains on only once step 2, pushed from this cut, merged.
    The K >= 2 rule (no blocker, no unpushed round) would train on here and push
    step 2 from the next cut, one round late: the lead then grows with the run."""
    role = _PeerPacedRole([_candidate(1, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    role.peer_pushed.add(1)  # step 1 merges first
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 4:  # step 2's PULL arrives late; the peer pushes it too
            role.candidates.append(_candidate(2, 1))
            role.peer_pushed.add(2)
        return 0

    assert await pairing.wait_for_lead(serve) > 0
    assert len(served) >= 4 and _pushed(role) == [1, 2] and role.fragment_versions == (1, 2)
    assert pairing.lead_wait_escapes == 0


@pytest.mark.asyncio
async def test_with_k_1_a_wait_entered_before_any_pull_answers_each_pull_of_its_window(lead_clock):
    """The window is decided once per round: a wait entered with nothing pushed
    pushes the first PULL of its window, keeps answering the window's PULLs as
    they come, and still holds the next window's."""
    role = _PeerPacedRole([])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 1:
            role.candidates.append(_candidate(1, 1))
        if len(served) == 2:
            role.candidates.append(_candidate(3, 1))  # next window: held
        if len(served) == 3:
            role.candidates.append(_candidate(2, 1))  # still this window: pushed
        if len(served) == 4:
            cuts.extend(sorted((key[0], pair.local_round) for key, pair in pairing.inflight.items()))
            role.peer_pushed.update({1, 2})
        return 0

    cuts: list[tuple[int, int]] = []
    assert await pairing.wait_for_lead(serve) > 0
    assert _pushed(role) == [1, 2] and pairing.lead_wait_escapes == 0
    assert cuts == [(1, 1), (2, 1)]  # both from cut 1


@pytest.mark.asyncio
async def test_with_k_1_crossed_pull_orders_do_not_lock_two_islands(lead_clock):
    """Island A sees only step 2 at its boundary, island B only step 1; each one's
    missing PULL arrives during its wait. Both are of the window, so each island
    pushes it and both windows merge: no island waits for a step the other holds
    (a sharded head merges PULLs of several shards in no fixed order)."""
    roster = _Roster({0, 1})
    a = _PeerPacedRole([_candidate(2, 1)], roster=roster, learner=0)
    b = _PeerPacedRole([_candidate(1, 1)], roster=roster, learner=1)
    pa, pb = _pairing(a, max_local_rounds_ahead=1), _pairing(b, max_local_rounds_ahead=1)
    for pairing in (pa, pb):
        pairing.local_round = 1
        pairing.submit_ready()
    assert (_pushed(a), _pushed(b)) == ([2], [1])

    def late(role, step):
        served = []

        async def serve():
            served.append(len(served))
            if len(served) == 1:
                role.candidates.append(_candidate(step, 1))
            return 0
        return serve

    await asyncio.gather(pa.wait_for_lead(late(a, 1)), pb.wait_for_lead(late(b, 2)))
    assert (pa.lead_wait_escapes, pb.lead_wait_escapes) == (0, 0)
    assert sorted(_pushed(a)) == sorted(_pushed(b)) == [1, 2]
    assert a.fragment_versions == b.fragment_versions == (1, 2)


@pytest.mark.asyncio
async def test_with_k_1_a_step_closed_without_the_island_closes_its_window(lead_clock):
    """A quorum below the island count or a soft deadline closes a step without
    this island: the window closes all the same (no wait, no escape), and the
    next round takes the window of the smallest open step."""
    role = _PeerPacedRole([_candidate(1, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 2:
            role.peer_pushed.add(1)
            role.close_without_me(2)  # step 2 closed before its PULL reached this island
        return 0

    assert await pairing.wait_for_lead(serve) > 0
    assert pairing.lead_wait_escapes == 0 and _pushed(role) == [1]
    assert role.fragment_versions == (1, 2)
    pairing.local_round = 2
    assert pairing.lead_window() == (3, 4)


def test_with_k_1_a_caught_up_island_takes_the_window_of_its_first_open_step():
    """A joining or relaunched process starts from the syncer's fragments: its
    first window is the one of the smallest step they leave open."""
    role = _PeerPacedRole([_candidate(7, 1), _candidate(9, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    role.fragment_versions = (5, 8)  # steps 5 and 8 merged last: step 7 (fragment 0) is open
    pairing.local_round = 1
    assert pairing.lead_window() == (7, 8)
    pairing.submit_ready()
    assert _pushed(role) == [7]  # step 9 belongs to the next window
    role.fragment_versions = (7, 8)
    pairing.local_round = 2
    assert pairing.lead_window() == (9, 10)


def test_with_k_1_windows_stay_aligned_to_multiples_of_the_pipeline_depth():
    """Versions (7, 6): step 7 merged, step 8 is the first open step. The window
    is the fixed window 7..8 that holds it, not 8..9: every island cuts the same
    windows, whichever of its steps closed first."""
    role = _PeerPacedRole([_candidate(8, 1), _candidate(9, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    role.fragment_versions = (7, 6)
    pairing.local_round = 1
    assert pairing.lead_window() == (7, 8)
    pairing.submit_ready()
    assert _pushed(role) == [8]  # step 9 belongs to window 9..10


@pytest.mark.asyncio
async def test_with_k_1_the_last_window_ends_at_the_last_step_and_the_terminal_phase_ends_a_wait(lead_clock):
    role = _PeerPacedRole([_candidate(1, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=1, total_fragment_steps=5)
    role.fragment_versions = (3, 4)
    pairing.local_round = 1
    assert pairing.lead_window() == (5, 5)  # 5 steps, P = 2: the last window has one step
    role.candidates[:] = [_candidate(5, 1)]
    pairing.submit_ready()
    assert pairing.lead_window_open()
    pairing.terminal_seen = True  # the data ended or the last rounds are ready mid-window
    assert await pairing.wait_for_lead() == 0.0
    pairing.terminal_seen = False

    async def serve():
        pairing.terminal_seen = True  # the terminal phase starts during the wait
        return 0

    assert await pairing.wait_for_lead(serve) == pytest.approx(LEAD_WAIT_POLL_S)
    assert pairing.lead_wait_escapes == 0


@pytest.mark.asyncio
async def test_with_k_1_the_terminal_phase_lifts_the_window_so_the_final_cut_answers_every_pull(lead_clock):
    """The last rounds' PULLs (steps 5 and 6 of 6) arrive while a wait holds
    window 3..4. They start the terminal phase, so no round trains after this
    cut: the wait ends and the final cut answers them although they are past its
    window (a window filter kept here leaves both islands waiting on each other)."""
    role = _PeerPacedRole([_candidate(3, 2), _candidate(4, 2)])
    pairing = _pairing(role, max_local_rounds_ahead=1, total_fragment_steps=6)
    role.fragment_versions = (1, 2)
    pairing.local_round = 2
    pairing.submit_ready()
    assert pairing.lead_window() == (3, 4) and _pushed(role) == [3, 4]
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 1:
            role.peer_pushed.update({3, 4})
            role.candidates.extend([_candidate(5, 2), _candidate(6, 2)])
        return 0

    await pairing.wait_for_lead(serve)
    assert pairing.any_finalizing and pairing.lead_wait_escapes == 0
    assert pairing.lead_window() == (3, 4)  # still this round's window ...
    pairing.submit_ready()
    assert _pushed(role) == [3, 4, 5, 6]  # ... but the final cut answers every PULL


@pytest.mark.asyncio
async def test_with_k_1_a_replayed_pull_of_a_pair_in_flight_is_answered_during_the_wait(lead_clock):
    """After a redial (or a restarted syncer) the syncer re-sends the PULL of a
    round the island already pushed: the retained pair replays its bytes at
    once, the window does not hold it."""
    role = _PeerPacedRole([_candidate(1, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    pairing.submit_ready()
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 1:
            again = _candidate(1, 1)
            role.candidates.append(replace(again, permit=PullRequest(0, 1, 2)))  # attempt 2
            role.candidates.append(_candidate(2, 1))
        if len(served) == 3:
            role.peer_pushed.update({1, 2})
        return 0

    assert await pairing.wait_for_lead(serve) > 0
    assert _pushed(role) == [1, 1, 2] and pairing.lead_wait_escapes == 0
    assert role.fragment_versions == (1, 2)


class _DeferredRole(_PeerPacedRole):
    """Deferred capture: a fragment is materialized only when planned for a push."""

    def __init__(self, candidates, pairing_round):
        super().__init__(candidates)
        self.materialized: list[tuple[int, tuple[int, ...]]] = []
        self._round = pairing_round

    def ready_submission_candidates(self, include_deferred=False):
        return super().ready_submission_candidates()

    async def materialize_submission(self, fragment_ids):
        self.materialized.append((self._round(), tuple(fragment_ids)))


@pytest.mark.asyncio
async def test_with_k_1_a_held_pull_is_not_materialized_ahead(lead_clock):
    holder = {}
    role = _DeferredRole([_candidate(1, 1)], lambda: holder["pairing"].local_round)
    pairing = holder["pairing"] = _pairing(role, max_local_rounds_ahead=1)
    pairing.local_round = 1
    await pairing.prepare_submissions()
    pairing.submit_ready()
    role.peer_pushed.add(1)  # step 1 merges at the first poll: two pipeline slots are free
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 1:
            assert pairing.inflight == {}
            role.candidates.extend([_candidate(3, 1), _candidate(2, 1)])  # next window, this window
        if len(served) == 3:
            role.peer_pushed.add(2)
        return 0

    await pairing.wait_for_lead(serve)
    # Cut 1 materialized fragment 0 (step 1) and fragment 1 (step 2), never step 3's,
    # although a slot was free for it.
    assert role.materialized == [(1, (0,)), (1, (1,))]
    pairing.local_round = 2
    await pairing.prepare_submissions()
    assert role.materialized[-1] == (2, (0,))


def test_a_pair_reads_the_injected_clock_when_it_is_retained(lead_clock):
    from yeto.rl.engine.full_parameter_sync import PairedSubmission

    before = lead_clock()
    pair = PairedSubmission({"actor": _candidate(1, 1)})
    assert pair.retained_at == before  # read at construction, before any push


@pytest.mark.asyncio
async def test_with_k_2_rounds_trained_after_the_last_retained_push_still_count():
    """K >= 2 keeps the blocker rule: after a merge and before the next PULL
    nothing is in flight, yet the rounds trained since the island last retained
    a pair land in its next push, so K of them stop the next round."""
    role = _PairedRole("actor", [_candidate(1, 1)], [])
    pairing = _pairing(role, max_local_rounds_ahead=2)
    pairing.local_round = 1
    pairing.submit_ready()
    await pairing.commit_staged_broadcasts()
    assert pairing.inflight == {} and role.fragment_versions == (1, 0)
    pairing.local_round = 2  # one round on top of the push: allowed
    assert pairing.lead_blockers() == [] and not pairing.lead_unpushed()
    assert await pairing.wait_for_lead() == 0.0
    pairing.local_round = 3  # a second one: the next round waits for a push
    assert pairing.lead_blockers() == [] and pairing.lead_unpushed()
    assert pairing.lead_window() is None


@pytest.mark.asyncio
async def test_with_k_2_the_wait_still_pushes_a_pull_that_cannot_hold_it(lead_clock):
    """With K = 2 a pair from the current cut does not block this wait (one
    round may train on top of it), so the wait keeps answering PULLs."""
    role = _PeerPacedRole([_candidate(1, 1)])
    pairing = _pairing(role, max_local_rounds_ahead=2, wait_timeout=2.0)
    pairing.local_round = 1
    pairing.submit_ready()
    pairing.local_round = 2  # a round on top of the unmerged step 1: the next one waits
    assert len(pairing.lead_blockers()) == 1
    served = []

    async def serve():
        served.append(len(served))
        if len(served) == 1:
            role.candidates.append(_candidate(2, 2))
        if len(served) == 3:
            role.peer_pushed.add(1)
        return 0

    assert await pairing.wait_for_lead(serve) > 0
    assert (pairing.lead_waits, pairing.lead_wait_escapes) == (1, 0)
    assert [step for _, step, _ in role.events] == [1, 2]
    assert [(key[0], pair.local_round) for key, pair in pairing.inflight.items()] == [(2, 2)]


@pytest.mark.asyncio
async def test_no_wait_in_the_terminal_phase_or_without_blockers():
    role = _PairedRole("actor", [_candidate(1, 1)], [], release_broadcasts=False)
    pairing = _pairing(role, max_local_rounds_ahead=2)
    pairing.local_round = 1
    pairing.submit_ready()
    assert await pairing.wait_for_lead() == 0.0  # K = 2: one round on top is allowed
    pairing.local_round = 2
    pairing.terminal_seen = True  # the terminal barrier waits instead
    assert await pairing.wait_for_lead() == 0.0
    assert pairing.lead_waits == 0


# -- launcher and island wiring ------------------------------------------------------


def test_the_launcher_forwards_the_cap_to_every_island_and_no_hash_moves():
    base = ASYNC + ("--rl-sync-fragment-pipeline", "2")
    _, plain = _plan(base)
    _, capped = _plan(base + ("--rl-sync-max-local-rounds-ahead", "1"))
    for before, after in zip(plain["island_requests"], capped["island_requests"], strict=True):
        assert "--sweep-max-local-rounds-ahead" not in before["learner_command"]
        assert _flag(after["learner_command"], "--sweep-max-local-rounds-ahead") == "1"
        for flag in ("--syncer-profile-sha256", "--parameter-layout-sha256"):
            assert _flag(after["learner_command"], flag) == _flag(before["learner_command"], flag)
    assert [s["command"] for s in plain["role_syncers"]] == [s["command"] for s in capped["role_syncers"]]


@pytest.mark.parametrize(("extra", "message"), [
    (("--rl-sync-max-local-rounds-ahead", "1"), "async"),  # the strict sweep
    (("--rl-sync-full-parameter-rounds", "async", "--rl-sync-max-local-rounds-ahead", "0"), "1..1000"),
])
def test_the_launcher_refuses_a_cap_outside_the_async_rounds(extra, message):
    with pytest.raises((ValueError, SystemExit), match=message):
        _plan(harness.FULL + extra)


def test_the_island_builds_the_profile_with_the_cap():
    from yeto.rl.adapters.miles.island_entry import _max_local_rounds_ahead

    assert _max_local_rounds_ahead(SimpleNamespace(), True) is None
    assert _max_local_rounds_ahead(SimpleNamespace(sweep_max_local_rounds_ahead=1), True) == 1
    with pytest.raises(ValueError, match="async"):
        _max_local_rounds_ahead(SimpleNamespace(sweep_max_local_rounds_ahead=1), False)
    with pytest.raises(ValueError, match="1..1000"):
        _max_local_rounds_ahead(SimpleNamespace(sweep_max_local_rounds_ahead=0), True)


def test_the_cap_stays_out_of_the_session_contract():
    """The cap lives on FullParameterSyncConfig, not on RoleStreamConfig, so
    the session contract hash (HELLO) of a capped island is unchanged."""
    from test_rl_full_parameter_sync import _role_stream_config

    from yeto.rl.adapters.miles.full_parameter.session import FullParameterSyncConfig

    config = _role_stream_config("actor", 30100)
    config = replace(config, partial_quorum=True)
    plain = FullParameterSyncConfig((config,))
    capped = FullParameterSyncConfig((config,), max_local_rounds_ahead=1)
    assert plain.roles == capped.roles
    assert capped.profile().max_local_rounds_ahead == 1 and plain.profile().max_local_rounds_ahead is None


# -- two islands on the real Rust syncer ---------------------------------------------

FRAGMENTS = 4
PIPELINE = 2
OUTER_ROUNDS = 3  # every fragment merges 3 times
NUMEL = 3
TOKENS = 5  # tokens per local round: tokens^2/steps = c_steps * 25


def _initial(fid: int) -> np.ndarray:
    return np.asarray([float(fid) + 0.25 * i for i in range(NUMEL)], dtype=np.float32)


def _step(learner: int, local_round: int, fid: int) -> np.ndarray:
    """The local training update of one round (deterministic, learner-specific)."""
    return np.asarray([0.01 * (learner + 1) + 0.001 * local_round + 0.0001 * fid + 0.00001 * i
                       for i in range(NUMEL)], dtype=np.float32)


class _WireRole:
    """A RoleStream over a real SyncerClient: f32 pushes of ``cut - anchor``,
    broadcasts overwrite the fragment (alpha = 0), FINAL ends the run."""

    monotonic_broadcasts = True
    role = "actor"

    def __init__(self, learner: int, client, record) -> None:
        self.learner = learner
        self.client = client
        self.record = record
        self.params = {f: _initial(f) for f in range(FRAGMENTS)}
        self.anchor = {f: _initial(f) for f in range(FRAGMENTS)}
        self.versions = [0] * FRAGMENTS
        self.rounds = 0
        self.rounds_at_anchor = [0] * FRAGMENTS
        self.cut: dict | None = None
        self.cut_versions: tuple[int, ...] | None = None
        self.cut_rounds = 0
        self.pending: dict[int, PullRequest] = {}
        self.staged: dict[tuple[int, int], np.ndarray] = {}
        self.terminal_submitted = False
        self.final_acknowledged = False
        self.manifest = None
        self.final: dict[int, np.ndarray] | None = None

    # -- RoleStream ---------------------------------------------------------
    @property
    def fragment_versions(self):
        return tuple(self.versions)

    @property
    def content_hash(self):
        return "x"

    @property
    def finalizing(self):
        return self.client.finalizing.is_set()

    async def initialize(self, group):
        """Like the Miles stream, which waits for the syncer's initial cut:
        training starts once the syncer opened its first rounds."""
        del group
        deadline = time.monotonic() + 20
        while not self.pending:
            assert time.monotonic() < deadline, "the syncer opened no round"
            self._drain()
            await asyncio.sleep(0.005)

    async def complete_local_round(self, *, rollout_id, trajectories):
        del trajectories
        self.rounds += 1
        for f in range(FRAGMENTS):
            self.params[f] = self.params[f] + _step(self.learner, self.rounds, f)
        self.cut = {f: value.copy() for f, value in self.params.items()}
        self.cut_versions = tuple(self.versions)
        self.cut_rounds = self.rounds
        self.record("round", self.learner, rollout_id + 1)

    def resident_state(self, policy_version):
        return SimpleNamespace(policy_version=policy_version, policy_tensor_hash=lambda: "h")

    def _drain(self):
        self.client.check_health()
        for permit in self.client.drain_pulls():
            known = self.pending.get(permit.fragment_id)
            if known is None or (permit.global_step, permit.round_attempt) > (
                    known.global_step, known.round_attempt):
                self.pending[permit.fragment_id] = permit

    def ready_submission_candidates(self, include_deferred=False):
        del include_deferred
        if self.cut is None or self.final is not None:
            return ()
        self._drain()
        out = []
        for f, permit in sorted(self.pending.items()):
            base = max(0, permit.global_step - FRAGMENTS)
            c_steps = self.cut_rounds - self.rounds_at_anchor[f]
            if self.versions[f] != base or self.cut_versions[f] != base or c_steps < 1:
                continue
            out.append(RoleSubmissionCandidate(
                role="actor", permit=permit, base_version=base, local_step=self.cut_rounds,
                c_steps=c_steps, c_tokens=c_steps * TOKENS, payload_bytes=NUMEL * 4,
                submission_cut=self.cut, anchor=self.anchor[f]))
        return tuple(out)

    def retain_candidate(self, candidate):
        del candidate

    def release_candidate(self, candidate):
        del candidate

    def push_candidate(self, candidate):
        f = candidate.permit.fragment_id
        if self.pending.get(f) != candidate.permit:
            return None
        delta = candidate.submission_cut[f] - candidate.anchor
        if self.client.push_fragment_parts(
                f, candidate.permit.global_step, candidate.permit.round_attempt, candidate.base_version,
                candidate.local_step, candidate.c_steps, candidate.c_tokens,
                [delta.astype(np.float32).tobytes()]) is not True:
            return None
        del self.pending[f]
        self.terminal_submitted |= candidate.permit.global_step == FRAGMENTS * OUTER_ROUNDS
        self.record("push", self.learner, (candidate.permit.global_step, candidate.c_steps,
                                           candidate.local_step, f))
        return SimpleNamespace(role="actor", fragment_id=f, global_step=candidate.permit.global_step,
                               round_attempt=candidate.permit.round_attempt,
                               base_version=candidate.base_version, c_steps=candidate.c_steps,
                               c_tokens=candidate.c_tokens, payload_bytes=NUMEL * 4,
                               pull_to_push_seconds=0.0)

    async def stage_available_broadcasts(self):
        self.client.check_health()
        for update in self.client.drain_updates():
            if update.version > self.versions[update.fragment_id]:
                self.staged[(update.fragment_id, update.version)] = np.frombuffer(
                    bytes(update.data), dtype=np.float32)[-NUMEL:].copy()
        return tuple(self.staged)

    @property
    def staged_fragment_versions(self):
        return tuple(self.staged)

    def has_staged_broadcast(self, fragment_id, version):
        return (fragment_id, version) in self.staged

    async def apply_staged_broadcast(self, fragment_id, version, *, catch_up=False):
        del catch_up
        value = self.staged.pop((fragment_id, version))
        self.params[fragment_id] = value.copy()
        self.anchor[fragment_id] = value.copy()
        self.versions[fragment_id] = version
        self.rounds_at_anchor[fragment_id] = self.rounds
        permit = self.pending.get(fragment_id)
        if permit is not None and permit.global_step <= version:
            del self.pending[fragment_id]
        return (fragment_id,)

    async def wait_and_apply_final(self):
        manifest, fragments = await asyncio.to_thread(self.client.wait_for_final_fragments, 30)
        self.final = {f.fragment_id: np.frombuffer(bytes(f.data), dtype=np.float32)[-NUMEL:].copy()
                      for f in fragments}
        self.params = {f: value.copy() for f, value in self.final.items()}
        self.versions = list(manifest.versions)
        self.manifest = manifest

    def acknowledge_final(self):
        self.client.acknowledge_finalization(self.manifest)
        self.final_acknowledged = True

    def discard_detached_submission_cut(self):
        pass

    def close(self):
        self.client.close()


def _island(learner, client, profile, train_s, record, errors):
    loop = asyncio.new_event_loop()
    role = _WireRole(learner, client, record)
    sync = FullParameterSync(RolePairing({"actor": role}, profile), groups={"actor": None},
                             run=loop.run_until_complete, trajectories=lambda rollout_id: None)
    driver = SimpleNamespace(emit=lambda name, **fields: record("event", learner, (name, fields)),
                             outer_version=None)
    try:
        sync.start(driver)
        for rollout_id in range(1000):
            record("start", learner, rollout_id + 1)
            time.sleep(train_s)
            if sync.boundary(driver, rollout_id=rollout_id, stats=None).stop:
                sync.published(driver, rollout_id=rollout_id, policy_hash="h")
                sync.finish(driver)
                break
    except BaseException as error:  # noqa: BLE001 - re-raised by the test
        errors.append(error)
    finally:
        sync.close()
        loop.close()
    return role


def _run_two_islands(binary, tmp_path, monkeypatch, *, cap, fast_s=0.005, slow_s=0.25):
    monkeypatch.setattr(harness, "P", FRAGMENTS)
    monkeypatch.setattr(harness, "NUMEL", NUMEL)
    extra = ("--rl-sync-fragment-pipeline", str(PIPELINE), "--rl-sync-quorum", "2",
             "--rl-sync-grace-ms", "0", "--rl-sync-max-base-lag", "0",
             "--rl-sync-learner-weight", "tokens2-over-steps",
             "--rl-sync-outer-lr", "1.0", "--rl-sync-outer-momentum", "0")
    tmp_path.mkdir(parents=True, exist_ok=True)
    log_file = tmp_path / "syncer.log"
    process, port, profile_hash = harness._start_async_syncer(
        binary, tmp_path, *extra, total_steps=OUTER_ROUNDS, log_file=log_file)
    profile = _profile(total_fragment_steps=FRAGMENTS * OUTER_ROUNDS, fragments=FRAGMENTS,
                       pipeline_depth=PIPELINE, wait_timeout=60.0, poll_seconds=0.002,
                       max_local_rounds_ahead=cap)
    lock = threading.Lock()
    log: list[tuple[float, str, int, object]] = []

    def record(kind, learner, value):
        with lock:
            log.append((time.monotonic(), kind, learner, value))

    # The clients dial for 10 s; a loaded machine can take longer to start the syncer.
    deadline = time.monotonic() + 60
    while "syncer listening" not in log_file.read_text():
        assert process.poll() is None and time.monotonic() < deadline, "the syncer did not start"
        time.sleep(0.05)
    clients = [harness._client(learner, port, profile_hash) for learner in (0, 1)]
    errors: list[BaseException] = []
    roles: dict[int, _WireRole] = {}
    try:
        for client in clients:
            client.start()
        for fid in range(FRAGMENTS):
            clients[0].send_init(fid, _initial(fid).tobytes())
        threads = [threading.Thread(
            target=lambda learner=learner, train_s=train_s: roles.__setitem__(
                learner, _island(learner, clients[learner], profile, train_s, record, errors)))
            for learner, train_s in ((0, fast_s), (1, slow_s))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not any(thread.is_alive() for thread in threads), "islands did not finish"
        if errors:
            raise errors[0]
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients:
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    return SimpleNamespace(roles=roles, log=log, syncer_log=log_file.read_text())


def _max_lead(log, fast=0, slow=1) -> int:
    """Largest (rounds the fast island completed) - (rounds the slow island
    completed) at the moment the fast island starts training a round."""
    completed = {fast: 0, slow: 0}
    lead = 0
    for _, kind, learner, value in sorted(log, key=lambda item: item[0]):
        if kind == "round":
            completed[learner] = value
        elif kind == "start" and learner == fast:
            lead = max(lead, (value - 1) - completed[slow])
    return lead


def _pushes(log, learner):
    return {step: (c_steps, local) for _, kind, who, value in log if kind == "push" and who == learner
            for step, c_steps, local, _ in [value]}


def _replayed_finals(log) -> dict[int, np.ndarray]:
    """The final fragments recomputed in float64 from the pushes in ``log``.

    A push of fragment f at local round r with c_steps c carries the island's
    progress since its anchor: the sum of ``_step(learner, k, f)`` for
    k = r - c + 1 .. r (the anchor is the fragment at the push's base
    version).  Each merge adds the tokens^2/steps-weighted mean of the
    pushes to that base (outer lr 1, momentum 0)."""
    merges: dict[int, dict[int, tuple[int, int, int]]] = {}
    for _, kind, who, value in log:
        if kind == "push":
            step, c_steps, local, fid = value
            merges.setdefault(step, {})[who] = (c_steps, local, fid)
    params = {fid: _initial(fid).astype(np.float64) for fid in range(FRAGMENTS)}
    for step in sorted(merges):
        pushes = merges[step]
        (fid,) = {fid for _, _, fid in pushes.values()}
        assert fid == (step - 1) % FRAGMENTS, (step, fid)
        weights = {who: (c_steps * TOKENS) ** 2 / c_steps for who, (c_steps, _, _) in pushes.items()}
        total = sum(weights.values())
        merged = np.zeros(NUMEL, dtype=np.float64)
        for who, (c_steps, local, _) in sorted(pushes.items()):
            progress = sum(_step(who, k, fid).astype(np.float64) for k in range(local - c_steps + 1, local + 1))
            merged += weights[who] / total * progress
        params[fid] = params[fid] + merged
    return params


# The float32 path (island parameters, wire deltas, Rust merge and outer
# step) differs from the float64 replay by float32 rounding only: at most
# 2.7e-7 relative in 12 local runs of both tests.  A push merged
# with a wrong round, weight or fragment moves the result by 1e-3 relative or
# more (one local round adds 0.001 per element).
REPLAY_RTOL = 1e-5


def _check_merges(run):
    steps = range(1, FRAGMENTS * OUTER_ROUNDS + 1)
    for learner in (0, 1):
        assert sorted(_pushes(run.log, learner)) == list(steps), learner  # every merge has both islands
    assert "push rejected" not in run.syncer_log and "OutOfRound" not in run.syncer_log
    finals = [run.roles[learner].final for learner in (0, 1)]
    for fid in range(FRAGMENTS):
        assert np.array_equal(finals[0][fid], finals[1][fid])
        assert np.all(np.isfinite(finals[0][fid]))
    # The syncer's result is the weighted merge of exactly the pushes the
    # islands made: no push lost, double-counted or merged with a wrong weight.
    replayed = _replayed_finals(run.log)
    for fid in range(FRAGMENTS):
        np.testing.assert_allclose(finals[0][fid], replayed[fid], rtol=REPLAY_RTOL, atol=0,
                                   err_msg=f"fragment {fid}")
    return finals[0]


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_real_syncer_fast_island_stays_within_the_cap_and_both_islands_join_every_merge(
        syncer_binary, tmp_path, monkeypatch):  # noqa: F811 - the imported fixture
    capped = _run_two_islands(syncer_binary, tmp_path / "capped", monkeypatch, cap=1)
    _check_merges(capped)
    # With K = 1 a fast island starts round r + 1 only once every push it
    # made from a cut <= r was merged, so the slow island pushed the same
    # step (both islands join every merge).
    fast_pushes: dict[int, int] = {}
    slow_steps: set[int] = set()
    for _, kind, who, value in sorted(capped.log, key=lambda item: item[0]):
        if kind == "push" and who == 0:
            fast_pushes[value[0]] = value[2]
        elif kind == "push":
            slow_steps.add(value[0])
        elif kind == "start" and who == 0:
            unmerged = [step for step, cut in fast_pushes.items() if cut < value and step not in slow_steps]
            assert not unmerged, (value, unmerged)
            # ... and only after it pushed from its newest cut (round r).
            assert value == 1 or value - 1 in fast_pushes.values(), (value, fast_pushes)
    # So it never starts a round before the slow island completed as many:
    # every island pushes window b of P steps from a cut of round b or later
    # (RolePairing.lead_window), and the fast island starts round b + 1 only
    # once window b merged.
    assert _max_lead(capped.log) <= 0
    waits = [fields for _, kind, who, (name, fields) in
             ((t, k, w, v) for t, k, w, v in capped.log if k == "event")
             if who == 0 and name == "rl_full_parameter_round"]
    assert waits[-1]["sync/max_local_rounds_ahead"] == 1
    assert waits[-1]["sync/lead_waits"] >= 1 and waits[-1]["sync/lead_wait_escapes"] == 0
    # Not checked: that a second run with the same inputs gives the same bits.
    # Which local round a push carries depends on wall-clock order.  One slow
    # boundary closes two rounds a few ms apart (the syncer writes a checkpoint
    # before each broadcast), and the syncer paces the PULLs of new rounds, so
    # whether a PULL of the next window is visible at the fast island's
    # boundary or arrives during its lead wait (then it is pushed at the next
    # boundary, from the newer cut: a K = 1 wait pushes only its own window)
    # depends on wall-clock order.  Both schedules keep the cap.  The syncer orders a
    # round's pushes by learner id (server.rs), so arrival order inside a
    # round does not change the merge.  _check_merges instead checks that the
    # result equals, up to float32 rounding, the merge of this run's pushes.


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_real_syncer_without_the_cap_the_fast_island_runs_ahead(syncer_binary, tmp_path, monkeypatch):  # noqa: F811
    """The control (old behaviour): same islands, no cap.  Every merge still
    has both islands and the result is finite, but the fast island runs many
    rounds ahead and pushes inflated c_steps."""
    free = _run_two_islands(syncer_binary, tmp_path, monkeypatch, cap=None)
    _check_merges(free)
    assert _max_lead(free.log) > 1
    assert max(c for c, _ in _pushes(free.log, 0).values()) > 2
    assert not any(kind == "event" and "sync/lead_waits" in value[1] for _, kind, _, value in free.log)
