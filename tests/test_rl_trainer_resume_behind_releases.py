"""CPU: a single buffer trainer relaunched from a round cut that is behind the
store's policy releases (the K3.1 trainer: round cuts and policy releases are two
commit chains, and a cut is far rarer than a release).

Every launch here is a real ``IslandDriver`` on ``LocalOnlySync`` with the real
round-cut controller (``ResumeController`` + ``RoundCutCheckpoint``, a torch
model behind ``VerlCutTrainer``), the real batch ledger, the real policy release
store (it keeps the newest two releases), the real ``ReleasingPublisher`` and
``_resume_single_trainer`` of the Miles trainer island, and the real
``BatchReader`` over a ``FileSampleBuffer``. Only the rollout pool is a stand-in:
it hands the reader's groups over as ``MilesRolloutPool`` does for a buffer
trainer (the plain behavior version as the group's policy token,
``rollout_meta_hook.group_record``). A launch on a new state dir is a new
container; a launch on the state dir of an earlier one is an in-place restart.

Regression tests: groups of a version released after the cut and pruned since,
the cut at the first safe point after such a relaunch, a relaunch without a cut
over the earlier ledger, and one in a new container.
"""

from __future__ import annotations

import dataclasses
import json
import time
from types import SimpleNamespace

import pytest
import torch
from test_rl_verl_cut import _handles, _model
from test_rl_verl_round_cuts import NAME, _plan, _TorchTrainer

from tests.buffer_rollout_procs import sample
from yeto.rl.adapters.miles import sampler_island as si
from yeto.rl.adapters.miles.buffer_rollout import BatchReader, GroupWriter
from yeto.rl.adapters.verl import round_cuts
from yeto.rl.engine import resume as rs
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.buffer_payload import group_entry_fields
from yeto.rl.engine.driver import EventTape, IslandDriver, PolicyIdentityError
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.fake import FakeEngine, FakeRolloutPool, fake_capabilities
from yeto.rl.engine.ledger import BatchLedger, LedgerError
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.ports import GroupMetadata, RolloutBatchHandle
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

GROUPS = 2  # groups per round


class _Killed(BaseException):
    """The trainer process is gone (SIGKILL, a reclaimed container): nothing after
    this point of the launch runs, no final cut is taken."""


class _Samplers:
    """The sampler islands of the run: before every trainer round they write one
    round of groups generated with the newest release in the store (v0, the base,
    while there is none). They outlive the trainer's launches."""

    def __init__(self, store):
        self.releases = PolicyReleaseStore(store)
        self.buffer = FileSampleBuffer(store, batch_size=1)
        self.writer = GroupWriter(self.buffer, "s0")
        self.written = 0
        self.versions: dict[str, int] = {}  # group id -> generating version

    def write_round(self):
        latest = self.releases.latest()
        version = 0 if latest is None else int(latest.version)
        for _ in range(GROUPS):
            g = self.written
            self.writer.write([sample(2 * g, g, version), sample(2 * g + 1, g, version)], group_key=f"g{g}")
            self.versions[f"g{g}"] = version
            self.written += 1


class _BufferPool(FakeRolloutPool):
    """The rollout port of a buffer trainer: a round is the next ``GROUPS`` groups
    of the reader, each named by its plain behavior version."""

    def __init__(self, engine, reader, samplers, kill_in_round):
        super().__init__(engine)
        self.reader = reader
        self.samplers = samplers
        self.kill_in_round = kill_in_round

    def generate(self, rollout_id, *, expected_policy_version=None):
        version, digest = self.engine.published
        self.samplers.write_round()
        if rollout_id == self.kill_in_round:
            raise _Killed(f"in round {rollout_id}")
        taken, report = self.reader.take(rollout_id, GROUPS, current_version=version)
        groups = []
        for _position, samples in taken:
            behavior, tokens = group_entry_fields(samples)
            groups.append(GroupMetadata(
                f"g{samples[0]['group_index']}", tuple(f"s{s['index']}" for s in samples),
                str(behavior), 0.5, 0.5, int(tokens), policy_versions=(int(behavior),)))
        return RolloutBatchHandle(
            rollout_id, version, digest, tuple(groups), len(groups), 0, payload=object(),
            buffer_stale_groups=report.stale_groups, data_cursor=self.data_cursor(),
            # buffer_rollout_miles._unidentified_metadata -> MilesRolloutPool._batch_of
            buffer_unidentified_groups=report.unidentified_policy_groups or None,
            buffer_unidentified_versions=[[v, n] for v, n in report.unidentified_policy_versions.items()] or None)

    def set_unidentified_policy_gate(self, start_version, known_versions):  # MilesRolloutPool: in the executor
        self.reader.set_unidentified_gate(start_version, known_versions)

    # MilesRolloutPool reports and seeks the executor's data source; a buffer trainer
    # draws no prompts, so the position never moves
    def data_cursor(self):
        return {"sample_offset": 0}

    def seek_data_cursor(self, cursor):
        return {"sample_offset": int(cursor["sample_offset"])}

    # a buffer trainer holds no unfinished trajectory (the samplers do)
    def export_in_flight(self):
        return {"entries": [], "buffer_groups": 0}


class _UploadDuringNextRound:
    """A ``BackgroundReleaser`` whose upload ends while the next round runs (a staged
    release of hundreds of GB takes minutes): ``run_pending`` is that moment."""

    busy = False

    def __init__(self):
        self.job = None

    def wait(self):
        self.run_pending()
        return 0.0

    def submit(self, job):
        self.run_pending()
        self.job = job
        return 0.0

    def run_pending(self):
        job, self.job = self.job, None
        if job is not None:
            job()

    drain = wait


def _launch(tmp_path, name, samplers, *, age, every, rounds=64, state_dir=None,
            kill_in_round=None, kill_before_safe_point=None, set_aside_before=None,
            store=None, staged=False, init_release=None):
    """One trainer launch over the run's store. ``kill_in_round`` /
    ``kill_before_safe_point``: where this process dies. ``store``: this host's
    store (default: the shared directory ``tmp_path / "store"``). ``staged``: the
    releases are staged and uploaded in the background, as a trainer island's full
    releases are; each upload ends after the next round selected its groups."""
    store = CheckpointStore(tmp_path / "store") if store is None else store
    store_dir = store.root
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0, groups=GROUPS)
    reader = BatchReader(FileSampleBuffer(store, batch_size=1, create_index=False), max_policy_age=age,
                         wait_s=0.0, poll_s=0.0, read_workers=1, prefetch=False, incarnation_gate=None)
    engine.rollout = _BufferPool(engine, reader, samplers, kill_in_round)
    engine.trainer = _TorchTrainer(engine, _handles(_model()))
    algorithm = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=age),
                              correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    caps = fake_capabilities("miles-like")
    caps = dataclasses.replace(caps, execution=dataclasses.replace(caps.execution, max_policy_staleness=age))
    profile = ExecutionProfile("p", "colocated-serial", "strict-avg", max_policy_age=age,
                               algorithm_contract="bounded-staleness", groups_per_batch=GROUPS,
                               samples_per_group=engine.samples_per_group,
                               algorithm_spec_sha256=algorithm.sha256())
    plan = _plan(store_dir, state_dir or tmp_path / name, every=every)
    controller = rs.ResumeController(state_dir=plan["resume"]["state_dir"], store=store,
                                     runtime_fingerprint="verl@test", keep=2)  # round_cuts.resume_wiring_for
    resume = round_cuts.ResumeWiring(controller=controller, ledger=BatchLedger(controller.state_dir),
                                     config=dict(plan["resume"]))
    tape = tmp_path / f"{name}.jsonl"
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=engine.publisher, placement=engine.placement, algorithm=algorithm, capabilities=caps,
        sync=LocalOnlySync(rounds), events=EventTape(tape, 0), ledger=resume.ledger, profile=profile)
    driver.ledger_set_aside_before = set_aside_before
    driver.data_cursor_optional = "sample_buffer"  # wire_policy_releases, role trainer
    # a Miles single trainer wires its round cuts before its releases (sampler_island module doc)
    checkpoint = round_cuts.wire_round_cuts(driver, resume, plan=plan, algorithm=algorithm,
                                            install_signals=False)
    checkpoint.buffer_watermark = reader.watermark_before  # buffer_rollout.wire_sample_buffer
    checkpoint.buffer_resume = reader.reset
    releases = PolicyReleaseStore(store)
    resumed = si._resume_single_trainer(driver, releases, init_release=init_release)
    # k31/resume-pruned-release-groups: wire_policy_releases hands the publisher the
    # driver's published policies for the release records (absent on main)
    published_of = getattr(si, "_published_of", None)
    if published_of is not None:
        resumed = {**resumed, "published": published_of(driver)}
    releaser = _UploadDuringNextRound() if staged else None
    driver.publisher = si.ReleasingPublisher(
        driver.publisher, releases, lambda version, directory: (directory / "w.bin").write_text(str(version)),
        read_position=lambda: reader.buffer.read_cursor, read_trained=reader.trained_before,
        emit=driver.emit, skip_base=True, **resumed,
        **({"staging": tmp_path / f"staging-{name}", "releaser": releaser} if staged else {}))
    if staged:
        generate = engine.rollout.generate

        def generate_then_the_upload_ends(rollout_id, **kwargs):
            batch = generate(rollout_id, **kwargs)  # the round selected its groups ...
            releaser.run_pending()  # ... and the upload of the previous version ends now
            return batch

        engine.rollout.generate = generate_then_the_upload_ends
    if kill_before_safe_point is not None:
        safe_point = driver.safe_point

        def killed_before(rollout_id):
            if rollout_id == kill_before_safe_point:
                raise _Killed(f"before the safe point of rollout {rollout_id}")
            return safe_point(rollout_id)

        driver.safe_point = killed_before
    return SimpleNamespace(driver=driver, checkpoint=checkpoint, tape=tape, releases=releases,
                           state_dir=resume.controller.state_dir, store_dir=store_dir, store=store,
                           controller=controller)


def _run(launch):
    """Run the launch to its end: ``"finished"``, ``"killed"``, or the error it died with."""
    try:
        launch.driver.run()
        outcome = "finished"
    except _Killed:
        outcome = "killed"
    except Exception as error:  # noqa: BLE001 - the outcome under test
        outcome = error
    commit = launch.checkpoint._commit  # a cut commit that ran in the background ends
    if commit is not None:
        commit[0].join()
    launch.driver.ledger.close()  # the process is gone: its journal lock with it
    return outcome


def _events(launch, kind):
    return [e for e in map(json.loads, launch.tape.read_text().splitlines()) if e.get("event") == kind]


def _trained(launch):
    """rollout id -> the generating versions of the groups it trained."""
    return {e["rollout_id"]: sorted(int(g[1].split(":")[1]) for g in e["groups"])
            for e in _events(launch, "rl_group_policies")}


def _release_versions(tmp_path):
    return [r.version for r in PolicyReleaseStore(CheckpointStore(tmp_path / "store")).records()]


def _latest_cut(tmp_path):
    return rs.read_latest(tmp_path / "store")


def _first_launch(tmp_path, samplers, *, age, cut_every, dies_in_round):
    """The run's first trainer: cuts every ``cut_every`` rounds, dies in round K."""
    first = _launch(tmp_path, "launch1", samplers, age=age, every=cut_every, kill_in_round=dies_in_round)
    assert _run(first) == "killed"
    return first


# ------------------------------------------------------------------------------------------
# versions released after the cut and pruned since: nothing durable names them
# ------------------------------------------------------------------------------------------
def _pruned_release_run(tmp_path, *, cut_at, dies_at, age):
    """Cut at M, releases v1..vK (the store keeps vK-1, vK), killed in round K. The
    samplers wrote round r's groups with v_r, so rounds M..K-1 trained groups of vM..vK-1."""
    samplers = _Samplers(CheckpointStore(tmp_path / "store"))
    _first_launch(tmp_path, samplers, age=age, cut_every=cut_at, dies_in_round=dies_at)
    assert _latest_cut(tmp_path)["next_rollout_id"] == cut_at
    assert _release_versions(tmp_path) == [dies_at - 1, dies_at]
    return samplers


def _unidentified(launch):
    """version -> groups the launch passed over as of a version no record names."""
    out: dict[int, int] = {}
    for event in _events(launch, "rl_buffer_unidentified_policy_skipped"):
        for version, groups in event["versions"]:
            out[version] = out.get(version, 0) + groups
    return out


@pytest.mark.parametrize("cut_at, dies_at, age, unidentified", [
    pytest.param(4, 7, 8, {4: GROUPS, 5: GROUPS}, id="K-M=3,age=8"),
    pytest.param(4, 7, 2, {5: GROUPS}, id="K-M=3,age=2"),  # the K3.1 example limit; v4 is over-age
    pytest.param(4, 6, 2, {4: GROUPS}, id="K-M=2,age=2"),  # v4 = K - 2: the cut's own version
    pytest.param(4, 5, 2, {}, id="K-M=1,age=2"),           # v4 = K - 1: still in the store
    pytest.param(4, 7, 1, {}, id="K-M=3,age=1"),           # window = {K-1}: older groups are over-age
])
def test_a_relaunch_behind_the_releases_passes_over_the_groups_of_pruned_releases(
        tmp_path, cut_at, dies_at, age, unidentified):
    """Cut M, releases up to vK, killed. The relaunch restores cut M, continues the
    account at vK and replays the buffer from cut M's watermark; the store names
    vK-1 only. A replayed group of a version in [K - max_policy_age, K - 2] has no
    record left. It used to stop the trainer (PolicyIdentityError "not a published
    policy"), and every relaunch after it on the same group. It is now consumed
    untrained and counted; the rounds go on with the groups that have an identity."""
    samplers = _pruned_release_run(tmp_path, cut_at=cut_at, dies_at=dies_at, age=age)
    launch = _launch(tmp_path, "launch2", samplers, age=age, every=1000, rounds=3)
    assert _run_or_raise(launch) == "finished"
    [gate] = _events(launch, "rl_buffer_unidentified_policy_gate")
    # v0 is named by the restored ledger while it is inside the window (age 8)
    known = ([0] if dies_at - age <= 0 else []) + [dies_at - 1]
    assert (gate["start_version"], gate["known_versions"]) == (dies_at, known)
    assert _unidentified(launch) == unidentified
    trained = _trained(launch)
    assert sorted(trained) == [dies_at, dies_at + 1, dies_at + 2]
    assert all(dies_at - 1 <= v and rid - v <= age for rid, versions in trained.items() for v in versions)
    passed_over = {g for g, v in samplers.versions.items() if v in unidentified}
    assert not passed_over & {g for groups in _trained_groups(launch).values() for g in groups}
    assert bool(passed_over) == bool(unidentified)
    # The relaunch ended with a cut at its newest release, and a backlog behind it.
    # The next launch resumes from that cut; the backlog holds groups of versions
    # its store pruned meanwhile, which stopped it the same way.
    again = _launch(tmp_path, "launch3", samplers, age=age, every=1000, rounds=1)
    assert _run_or_raise(again) == "finished"
    assert _events(again, "rl_resume")[0]["rollout_id"] == dies_at + 3


def test_an_unpublished_version_is_still_an_identity_error(tmp_path):
    """The gate is for versions below the start version of a trainer that resumed
    behind its releases. A group that names a version nothing published stops the
    trainer as before: in a run that starts here, and in such a relaunch for a
    version from its start version on."""
    def forged(samplers, version):
        g = samplers.written
        samplers.writer.write([sample(2 * g, g, version), sample(2 * g + 1, g, version)], group_key=f"g{g}")
        samplers.written += 1

    (tmp_path / "a").mkdir()
    fresh = _Samplers(CheckpointStore(tmp_path / "a" / "store"))
    forged(fresh, 1)  # the run starts at v0: nothing published v1
    launch = _launch(tmp_path / "a", "launch1", fresh, age=2, every=1000, rounds=1)
    with pytest.raises(PolicyIdentityError):
        _run_or_raise(launch)
    assert _events(launch, "rl_buffer_unidentified_policy_gate") == []

    (tmp_path / "b").mkdir()
    samplers = _pruned_release_run(tmp_path / "b", cut_at=4, dies_at=7, age=8)
    forged(samplers, 99)  # above the relaunch's start version v7
    relaunch = _launch(tmp_path / "b", "launch2", samplers, age=8, every=1000, rounds=6)
    with pytest.raises(PolicyIdentityError, match="=99 "):
        _run_or_raise(relaunch)


def _early_run(tmp_path, *, age, cut_every, dies_in_round, backlog=3):
    """A run that dies early: the samplers filled the buffer with ``backlog`` rounds of
    base-weight (v0) groups before the trainer started, as at the start of every run."""
    samplers = _Samplers(CheckpointStore(tmp_path / "store"))
    for _ in range(backlog):
        samplers.write_round()
    first = _launch(tmp_path, "launch1", samplers, age=age, every=cut_every, kill_in_round=dies_in_round)
    assert _run(first) == "killed"
    return samplers, first


def test_the_base_policy_is_named_by_the_start_state_of_a_relaunch_without_a_cut(tmp_path):
    """max_policy_age 2. The trainer released v1 and died in round 1, before any cut;
    the buffer still holds groups of the base weights v0, which have no release (the
    base policy is never released). The relaunch starts at v1 from the base
    checkpoint: its start state is v0, so the groups of v0 inside the window are
    trained, not passed over as unidentified."""
    samplers, _first = _early_run(tmp_path, age=2, cut_every=1000, dies_in_round=1)
    assert _release_versions(tmp_path) == [1] and _latest_cut(tmp_path) is None
    launch = _launch(tmp_path, "launch2", samplers, age=2, every=1000, rounds=2)
    assert _run_or_raise(launch) == "finished"
    [base] = _events(launch, "rl_base_policy_restored")
    assert base["source"] == "start_state"
    [gate] = _events(launch, "rl_buffer_unidentified_policy_gate")
    assert (gate["start_version"], gate["known_versions"], gate["base_version_named"]) == (1, [0], True)
    assert _unidentified(launch) == {}
    assert _trained(launch)[1] == [0, 0]  # round 1 trains groups of v0


def test_a_relaunch_that_loaded_a_release_does_not_take_its_start_state_for_the_base_policy(tmp_path):
    """``--rl-init-from-release``: the start state is the release's weights, not the
    base checkpoint's. v0 stays unnamed and its groups are passed over and counted."""
    samplers, _first = _early_run(tmp_path, age=2, cut_every=1000, dies_in_round=1)
    record = PolicyReleaseStore(CheckpointStore(tmp_path / "store")).latest()
    launch = _launch(tmp_path, "launch2", samplers, age=2, every=1000, rounds=1,
                     init_release={"path": record.path, "version": record.version,
                                   "policy_hash": record.policy_hash})
    assert _run_or_raise(launch) == "finished"
    assert _events(launch, "rl_base_policy_restored") == []
    [gate] = _events(launch, "rl_buffer_unidentified_policy_gate")
    assert (gate["known_versions"], gate["base_version_named"]) == ([], False)
    assert set(_unidentified(launch)) == {0} and 0 not in _trained(launch)[1]


def test_the_base_policy_is_named_by_the_ledger_of_a_restored_cut(tmp_path):
    """A cut at 1, releases v1 and v2, killed in round 2. The relaunch restores the
    cut (its weights are v1's, not the base's) and starts at v2 with v0 still inside
    the window. The ledger restored with the cut recorded round 0's policy: v0."""
    samplers, first = _early_run(tmp_path, age=2, cut_every=1, dies_in_round=2)
    assert _latest_cut(tmp_path)["next_rollout_id"] == 2 and _release_versions(tmp_path) == [1, 2]
    v0 = [e for e in _events(first, "rl_publication") if e["policy_version"] == 0][0]["rl/policy_token"]
    launch = _launch(tmp_path, "launch2", samplers, age=2, every=1000, rounds=2)
    assert _run_or_raise(launch) == "finished"
    [base] = _events(launch, "rl_base_policy_restored")
    assert (base["source"], f"yeto:0:{base['policy_hash']}") == ("ledger", v0)
    assert _events(launch, "rl_buffer_unidentified_policy_gate")[0]["known_versions"] == [0, 1]
    assert _unidentified(launch) == {} and 0 in _trained(launch)[2]


def test_the_base_policy_stays_unnamed_after_a_cut_whose_ledger_lost_round_zero(tmp_path):
    """The ledger of an earlier launch was set aside (a relaunch without a cut), then
    a cut was taken: its ledger starts at that relaunch. A launch that restores this
    cut has no record of v0; its groups are passed over, and the gate event says so."""
    samplers, first = _early_run(tmp_path, age=3, cut_every=1000, dies_in_round=1, backlog=6)
    second = _launch(tmp_path, "launch2", samplers, age=3, every=1, kill_in_round=2,
                     state_dir=first.state_dir, set_aside_before=time.time() + 60)
    assert _run(second) == "killed"
    assert _events(second, "rl_ledger_set_aside") and _latest_cut(tmp_path)["next_rollout_id"] == 2
    third = _launch(tmp_path, "launch3", samplers, age=3, every=1000, rounds=1)
    assert _run_or_raise(third) == "finished"
    assert _events(third, "rl_base_policy_restored") == []
    [gate] = _events(third, "rl_buffer_unidentified_policy_gate")
    assert gate["base_version_named"] is False and 0 not in gate["known_versions"]
    assert set(_unidentified(third)) == {0}


def test_a_resume_from_the_newest_cut_passes_nothing_over_when_every_version_is_named(tmp_path):
    samplers = _Samplers(CheckpointStore(tmp_path / "store"))
    first = _launch(tmp_path, "launch1", samplers, age=1, every=1000, rounds=3)
    assert _run_or_raise(first) == "finished"  # the final cut at 3 = the newest release
    second = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1)
    assert _run_or_raise(second) == "finished"
    assert _events(second, "rl_buffer_unidentified_policy_gate")[0]["known_versions"] == [2]
    assert _unidentified(second) == {}


# ------------------------------------------------------------------------------------------
# a launch after a relaunch that was behind the releases
# ------------------------------------------------------------------------------------------
def test_a_launch_after_one_round_of_a_relaunch_behind_the_releases_starts(tmp_path):
    """Cut 2, releases up to v3 (the store keeps v2, v3), killed. The relaunch
    continues at v3 with the cut's weights (v3 of this process is the cut's hash; the
    store's v3 is the dead trainer's), trains one round, releases v4, cuts at 4 and
    stops. The store still holds v3 next to v4: the next launch must start.

    Green on main. A resume that refuses a version the cut and a release record
    name with different hashes (PolicyVersionConflict) fails here at every start
    until v3 leaves the store: the cut at 4 recorded v3 with the cut-2 hash, the
    store's v3 record holds the dead trainer's."""
    samplers = _Samplers(CheckpointStore(tmp_path / "store"))
    _first_launch(tmp_path, samplers, age=1, cut_every=2, dies_in_round=3)
    assert _latest_cut(tmp_path)["next_rollout_id"] == 2 and _release_versions(tmp_path) == [2, 3]
    store_v3 = {r.version: r.policy_hash for r in PolicyReleaseStore(
        CheckpointStore(tmp_path / "store")).records()}[3]

    second = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1)
    assert _run(second) == "finished"  # round 3, then the final cut at 4
    assert _events(second, "rl_round_cut_behind_releases")[0]["latest_release_version"] == 3
    assert _latest_cut(tmp_path)["next_rollout_id"] == 4 and _release_versions(tmp_path) == [3, 4]
    # v3 names two sets of weights: the store's release and this process's publication
    published_v3 = [e for e in _events(second, "rl_publication") if e["policy_version"] == 3][0]
    assert published_v3["rl/policy_token"] != f"yeto:3:{store_v3}"

    third = _launch(tmp_path, "launch3", samplers, age=1, every=1000, rounds=1)
    outcome = _run(third)
    if isinstance(outcome, Exception):
        raise outcome
    assert outcome == "finished" and sorted(_trained(third)) == [4]


# ------------------------------------------------------------------------------------------
# the cut at the first safe point of a relaunch that is behind the releases
# ------------------------------------------------------------------------------------------
def _run_or_raise(launch):
    outcome = _run(launch)
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


def _recorded_rounds(launch):
    journal = (launch.state_dir / "ledger" / "journal.jsonl").read_text().splitlines()
    return sorted({r["rollout_id"] for r in map(json.loads, journal) if r.get("kind") == "outer_recorded"})


def test_a_relaunch_behind_the_releases_takes_no_cut_before_it_trained_a_round(tmp_path):
    """``--rl-cut-every 2``. The first trainer cut at 2, released v3 and v4 and died
    before its cut at 4 was committed. The relaunch restores cut 2 and continues at
    v4, where a cut is due at its first safe point. Its ledger ends at round 1 (2 and
    3 were the dead trainer's): a cut there named rollout 4 over a ledger without
    rollout 3, ``resume`` refused it (RoundCutError "no outer-recorded rollout 3"),
    and as LATEST it stopped every later launch. No cut is taken there now; the
    relaunch dies in round 5 and the next launch resumes from cut 2 again."""
    samplers = _Samplers(CheckpointStore(tmp_path / "store"))
    first = _launch(tmp_path, "launch1", samplers, age=1, every=2, kill_before_safe_point=4)
    assert _run(first) == "killed"
    assert _latest_cut(tmp_path)["next_rollout_id"] == 2 and _release_versions(tmp_path) == [3, 4]

    second = _launch(tmp_path, "launch2", samplers, age=1, every=2, kill_in_round=5)
    assert _run(second) == "killed"
    assert _events(second, "rl_cut_saved") == []
    [skipped] = _events(second, "rl_round_cut_skipped")
    assert (skipped["rollout_id"], skipped["reason"]) == (4, "no_round_trained_since_start")
    assert sorted(_trained(second)) == [4] and _recorded_rounds(second) == [0, 1, 4]
    assert _latest_cut(tmp_path)["next_rollout_id"] == 2

    third = _launch(tmp_path, "launch3", samplers, age=1, every=2, rounds=2)
    assert _run_or_raise(third) == "finished"
    assert _events(third, "rl_resume")[0]["rollout_id"] == 2
    assert _events(third, "rl_round_cut_behind_releases")[0]["latest_release_version"] == 5
    # rounds 5 and 6: the periodic cut at 6 (after a trained round) and the final one at 7
    assert [e["rollout_id"] for e in _events(third, "rl_cut_saved")] == [6, 7]

    fourth = _launch(tmp_path, "launch4", samplers, age=1, every=2, rounds=1)
    assert _run_or_raise(fourth) == "finished"
    assert _events(fourth, "rl_resume")[0]["rollout_id"] == 7  # a cut of the relaunch resumes


def test_the_default_timed_interval_takes_no_cut_at_the_first_safe_point():
    """Which interval asked for that cut. ``--rl-cut-every N``: when K % N == 0
    (above). ``--rl-cut-auto-mtbf-s``: always, the first safe point of every process
    is due. Neither flag (the default, the K3.1 examples): the first cut waits
    DEFAULT_FIRST_CUT_S, so the first safe point is never due."""
    from yeto.rl.engine.cut_interval import DEFAULT_FIRST_CUT_S, cut_interval_for

    now = [0.0]
    auto = cut_interval_for(every=None, auto_mtbf_s=3600.0, max_interval_s=None)
    assert auto.due(7) is True
    default = cut_interval_for(every=None, auto_mtbf_s=None, max_interval_s=None)
    default.clock = lambda: now[0]
    assert default.first_cut_s == DEFAULT_FIRST_CUT_S and default.due(7) is False
    now[0] = DEFAULT_FIRST_CUT_S
    assert default.due(8) is True  # a later safe point: the ledger then holds round 7
    assert cut_interval_for(every=4, auto_mtbf_s=None, max_interval_s=None).due(8) is True


# ------------------------------------------------------------------------------------------
# releases but no committed cut
# ------------------------------------------------------------------------------------------
def _no_cut_run(tmp_path, *, dies, cut_every=1000, staged=False, store=None, samplers=None):
    """No cut was ever committed. The first trainer released v1..v3 and died: in
    round 3 (``"in_round"``), or after it recorded round 3 and before v4 reached the
    store (``"before_release"``: a staged release takes minutes to upload)."""
    samplers = samplers or _Samplers(CheckpointStore(tmp_path / "store"))
    first = _launch(tmp_path, "launch1", samplers, age=1, every=cut_every, staged=staged, store=store,
                    **({"kill_in_round": 3} if dies == "in_round" else {}))
    if dies == "before_release":
        publish = first.driver.publisher.publish

        def killed_releasing(state, **kwargs):
            if int(state.policy_version) == 4:
                raise _Killed("releasing v4")
            return publish(state, **kwargs)

        first.driver.publisher.publish = killed_releasing
    assert _run(first) == "killed"
    if store is None:
        assert _latest_cut(tmp_path) is None and _release_versions(tmp_path) == [2, 3]
    return samplers, first


def _trained_groups(launch):
    return {e["rollout_id"]: sorted(g[0] for g in e["groups"]) for e in _events(launch, "rl_group_policies")}


@pytest.mark.parametrize("dies", ["in_round", "before_release"])
def test_a_new_container_without_a_cut_continues_above_the_releases(tmp_path, dies):
    """The store holds releases up to v3 and no cut (the trainer died before its
    first cut was committed), and the relaunch runs in a new container: an empty
    ledger. Its account starts at v3. Before: ``_restore_data_cursor`` refused the
    start (DriverError "the ledger holds no data cursor for rollout 2") although a
    buffer trainer draws no prompts, and the buffer was read from position 0 again.

    Now it starts, and reads the buffer from the position the release of v3 is
    trained up to (``buffer_trained``: the 6 groups rounds 0..2 selected). None of
    those is selected again; the groups of round 3, whose update never reached the
    store, are trained."""
    samplers, first = _no_cut_run(tmp_path, dies=dies)
    before = _trained_groups(first)
    assert sorted(before)[:3] == [0, 1, 2] and before[2] == ["g4", "g5"]
    launch = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1)
    assert _run_or_raise(launch) == "finished"
    [versions] = _events(launch, "rl_buffer_resume_versions")
    assert (versions["start_version"], versions["buffer_trained"]) == (3, 6)
    [cursor] = _events(launch, "rl_data_cursor_restored")
    assert (cursor["rollout_id"], cursor["source"], cursor["data_cursor"]) == (3, "sample_buffer", None)
    assert _trained_groups(launch) == {3: ["g6", "g7"]}
    [stale] = _events(launch, "rl_buffer_stale_skipped")
    assert stale["stale_groups"] == 0  # nothing before the position was read, not even to skip it


def test_a_relaunch_trains_the_groups_selected_while_the_newest_release_was_uploading(tmp_path):
    """Staged releases: the record of vK takes the read position (``buffer_selected``)
    on the upload thread, when the trainer has already selected round K. The trainer
    dies before vK+1 reaches the store. Reading on from ``buffer_selected`` skipped
    round K's groups for good (their tasks are settled in the dispenser, and they were
    trained only into the lost vK+1); ``buffer_trained`` is read at the publication,
    so the relaunch trains them."""
    samplers, first = _no_cut_run(tmp_path, dies="before_release", staged=True)
    assert _trained_groups(first)[3] == ["g6", "g7"]  # trained into v4, which is lost
    newest = PolicyReleaseStore(CheckpointStore(tmp_path / "store")).latest()
    assert (newest.version, newest.buffer_selected, newest.buffer_trained) == (3, 8, 6)
    launch = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1)
    assert _run_or_raise(launch) == "finished"
    assert _events(launch, "rl_buffer_resume_versions")[0]["buffer_trained"] == 6
    assert _trained_groups(launch) == {3: ["g6", "g7"]}


def test_a_release_record_of_an_earlier_yeto_is_read_from_zero_not_from_its_read_position(tmp_path):
    """A record without ``buffer_trained``: its ``buffer_selected`` may lie past
    groups its weights never trained, so it is not used. The relaunch reads from 0:
    the policy-age gate passes over what is too old, and what is inside the window
    is trained again (here the 2 groups of v2, already trained in round 2) rather
    than anything skipped."""
    samplers, _first = _no_cut_run(tmp_path, dies="in_round")
    for record in (tmp_path / "store" / "policy-releases").rglob("*.json"):
        data = json.loads(record.read_text())
        assert data.pop("buffer_trained") is not None and data["buffer_selected"] is not None
        record.write_text(json.dumps(data, sort_keys=True))
    launch = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1)
    assert _run_or_raise(launch) == "finished"
    assert _events(launch, "rl_buffer_resume_versions")[0]["buffer_trained"] is None
    assert _events(launch, "rl_buffer_stale_skipped")[0]["stale_groups"] == 4  # v0, v1
    assert _trained_groups(launch) == {3: ["g4", "g5"]}  # v2: inside the window, read again


def test_a_release_position_beyond_the_buffer_index_falls_back_to_zero(tmp_path):
    """The position is a hint of where to read on. One the buffer refuses (a store
    whose index holds fewer entries than the record names) is reported and the
    buffer is read from 0; the start is not refused."""
    samplers, _first = _no_cut_run(tmp_path, dies="in_round")
    for record in (tmp_path / "store" / "policy-releases").rglob("*.json"):
        data = json.loads(record.read_text())
        data["buffer_trained"] = 10_000
        record.write_text(json.dumps(data, sort_keys=True))
    launch = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1)
    assert _run_or_raise(launch) == "finished"
    [refused] = _events(launch, "rl_buffer_start_position_refused")
    assert refused["position"] == 10_000 and "outside the index" in refused["error"]
    assert _trained_groups(launch) == {3: ["g4", "g5"]}


def test_a_new_container_without_a_cut_takes_no_cut_at_its_first_safe_point(tmp_path):
    """``--rl-cut-every 3`` and a first cut that was never committed: the relaunch
    starts at v3 with an empty ledger and a cut due at its first safe point. A cut
    there could not be resumed from (no rollout 2 in the ledger); it is skipped and
    the cut after the first trained rounds resumes."""
    samplers, _first = _no_cut_run(tmp_path, dies="in_round", cut_every=1000)
    launch = _launch(tmp_path, "launch2", samplers, age=1, every=3, rounds=3)
    assert _run_or_raise(launch) == "finished"
    assert [e["rollout_id"] for e in _events(launch, "rl_round_cut_skipped")] == [3]
    assert [e["rollout_id"] for e in _events(launch, "rl_cut_saved")] == [6]
    third = _launch(tmp_path, "launch3", samplers, age=1, every=3, rounds=1)
    assert _run_or_raise(third) == "finished"
    assert _events(third, "rl_resume")[0]["rollout_id"] == 6


@pytest.mark.parametrize("dies", ["in_round", "before_release"])
def test_a_restart_without_a_cut_sets_the_ledger_of_the_earlier_launch_aside(tmp_path, dies):
    """The same relaunch on the state dir of the dead trainer (the island's volume;
    the store holds no cut, so nothing restores over it), with
    ``--rl-ledger-set-aside-before`` as ``yeto run up`` passes it on a restart.
    Before, the ledger was set aside only for a start at rollout 0; a start at v3
    kept it and ``rebase`` refused the restart (LedgerError "restart at rollout 3 is
    behind outer-recorded rollouts [3]") when the dead trainer had recorded round 3.
    The earlier ledger now moves aside and round 3 is trained."""
    samplers, first = _no_cut_run(tmp_path, dies=dies)
    recorded = [0, 1, 2] + ([3] if dies == "before_release" else [])
    assert _recorded_rounds(first) == recorded
    launch = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1,
                     state_dir=first.state_dir, set_aside_before=time.time() + 60)
    assert _run_or_raise(launch) == "finished"
    [aside] = _events(launch, "rl_ledger_set_aside")
    assert (aside["rollout_id"], aside["recorded_rollouts"]) == (3, recorded)
    assert list(first.state_dir.glob("ledger.before-*")) and _recorded_rounds(launch) == [3]
    assert _trained_groups(launch) == {3: ["g6", "g7"]}


def test_a_restart_without_a_cut_keeps_refusing_the_rounds_of_its_own_launch(tmp_path):
    """The protection of ``--rl-ledger-set-aside-before`` is unchanged: rounds
    recorded after the cutoff (this launch's own: a container the platform retried)
    or no cutoff at all leave the ledger where it is. A dead trainer that recorded
    round 3 is then still refused; one that died inside round 3 starts (the buffer
    is read from the release's position, so the ledger sees no group twice)."""
    samplers, first = _no_cut_run(tmp_path, dies="before_release")
    for n, cutoff in ((2, None), (3, time.time() - 3600)):
        launch = _launch(tmp_path, f"launch{n}", samplers, age=1, every=1000, rounds=1,
                         state_dir=first.state_dir, set_aside_before=cutoff)
        with pytest.raises(LedgerError, match=r"restart at rollout 3 is behind outer-recorded rollouts \[3\]"):
            _run_or_raise(launch)
        assert _events(launch, "rl_ledger_set_aside") == []


def test_a_retried_container_that_died_inside_a_round_continues_on_its_ledger(tmp_path):
    samplers, first = _no_cut_run(tmp_path, dies="in_round")
    launch = _launch(tmp_path, "launch2", samplers, age=1, every=1000, rounds=1, state_dir=first.state_dir)
    assert _run_or_raise(launch) == "finished"
    assert _events(launch, "rl_ledger_set_aside") == [] and _recorded_rounds(launch) == [0, 1, 2, 3]
    assert _trained_groups(launch) == {3: ["g6", "g7"]}


# ------------------------------------------------------------------------------------------
# the Kimi-K3.1 form: a staged s3 store, releases uploaded in the background
# ------------------------------------------------------------------------------------------
def test_a_new_container_after_a_dead_first_cut_on_a_staged_s3_store_continues_and_cuts(tmp_path):
    """Every container has its own empty stage over one bucket. The first trainer
    releases v1..v3 (each upload ends a round later), trains round 3 and dies before
    v4 reaches the bucket, while its first round cut is still uploading (marked,
    shards in the bucket, no LATEST). The second container drops that cut, continues
    at v3 with the base weights and reads the buffer from where v3 is trained up to:
    round 3's groups are trained again, nothing before them is. It cuts; a third
    container resumes from that cut."""
    import inspect

    from test_rl_object_store import URI, FakeS3

    from yeto.cloud import s3_store

    s3 = FakeS3()
    env = {rs.STORE_IMPL_ENV: s3_store.STORE_IMPL, s3_store.URI_ENV: URI}

    def host(name):
        return rs.store_for(tmp_path / "mount", environ={**env, s3_store.STAGE_ENV: str(tmp_path / name)},
                            client=s3)

    samplers = _Samplers(host("sampler"))
    _samplers, first = _no_cut_run(tmp_path, dies="before_release", staged=True, store=host("trainer-a"),
                                   samplers=samplers)
    assert _trained_groups(first) == {0: ["g0", "g1"], 1: ["g2", "g3"], 2: ["g4", "g5"], 3: ["g6", "g7"]}
    newest = PolicyReleaseStore(host("probe")).latest()
    assert (newest.version, newest.buffer_selected, newest.buffer_trained) == (3, 8, 6)
    assert rs.read_latest(host("probe-cuts").root) is None
    dead = None
    if "store" in inspect.signature(rs.mark_first_cut).parameters:  # the mark reaches the bucket at once
        dead = "r000004-dead"
        rs.mark_first_cut(first.store.root, dead, store=first.store)
        shard = first.store.root / rs.ROUND_CUTS / dead / "rank-0.pt"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"half a cut")
        first.store.push_local_file(shard.relative_to(first.store.root).as_posix())
        assert any(key.endswith(f"{dead}/rank-0.pt") for _, key in s3.objects)

    second = _launch(tmp_path, "launch2", samplers, age=1, every=2, rounds=3, staged=True,
                     store=host("trainer-b"))
    if dead is not None:
        assert second.controller.restored is None and second.controller.discarded_first_cut
        assert not any(dead in key for _, key in s3.objects)
    assert _run_or_raise(second) == "finished"
    [versions] = _events(second, "rl_buffer_resume_versions")
    assert (versions["start_version"], versions["buffer_trained"]) == (3, 6)
    trained = _trained_groups(second)
    assert sorted(trained) == [3, 4, 5] and trained[3] == ["g6", "g7"]  # round 3 again; g0..g5 never
    assert not {"g0", "g1", "g2", "g3", "g4", "g5"} & {g for groups in trained.values() for g in groups}
    assert [e["rollout_id"] for e in _events(second, "rl_cut_saved")] == [4, 6]
    assert _events(second, "rl_round_cut_skipped") == []  # 3 is not due with --rl-cut-every 2

    third = _launch(tmp_path, "launch3", samplers, age=1, every=2, rounds=1, store=host("trainer-c"))
    assert _run_or_raise(third) == "finished"
    assert _events(third, "rl_resume")[0]["rollout_id"] == 6 and sorted(_trained_groups(third)) == [6]
