"""S27 P2: full-parameter asynchronous sharded rounds. No Ray.

``--rl-sync-full-parameter-rounds async`` replaces the strict policy sweep
(fixed roster, equal weights, no momentum) with ordinary syncer fragment
rounds: P fragments, at most two rounds in flight, quorum K plus adaptive
grace, tokens^2/steps weights, outer Nesterov, optional RDA, and a broadcast
that overwrites (alpha = 0) the fragment of every island, including islands
the round closed without. The sweep stays the default and the control.

Covered here: the island pairing rule under a partial quorum (absent island,
late island, broadcasts out of order, closed permits), the role stream's
permit bookkeeping, the launcher wiring and refusals, and the real Rust
syncer running the launcher's async command (absent island, late push, base
lag, merge weights and Nesterov against a numpy reference).
"""

from __future__ import annotations

import shlex
import shutil
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_sync import _candidate, _PairedRole, _role_stream_config
from yeto.protocol import PullRequest
from yeto.rl.engine.full_parameter_sync import FullParameterSyncProfile, RolePairing

LAYOUT = "e" * 64
TWO_ISLANDS = "modal:2xh100,modal:2xh100"
FULL = ("--rl-sync-preset", "full-parameter", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-parameter-layout-sha256", LAYOUT, "--fragments", "4")
ASYNC = FULL + ("--rl-sync-full-parameter-rounds", "async")


# -- island pairing under a partial quorum -----------------------------------


def _async_pairing(*streams, partial_quorum=True, pipeline_depth=2):
    profile = FullParameterSyncProfile(
        roles=tuple(stream.role for stream in streams),
        total_fragment_steps=6,
        fragments=2,
        local_horizon=2,
        pipeline_depth=pipeline_depth,
        wait_timeout=900.0,
        poll_seconds=0.001,
        partial_quorum=partial_quorum,
    )
    return RolePairing({stream.role: stream for stream in streams}, profile)


def _submitted(progress, role="actor"):
    return [value.global_step for value in progress[role].submissions]


@pytest.mark.asyncio
async def test_absent_island_overwrites_a_round_it_never_joined():
    actor = _PairedRole("actor", [], [])
    pairing = _async_pairing(actor)
    actor._staged_steps.append(1)  # K islands closed round 1 without this one

    committed = await pairing.commit_staged_broadcasts()

    assert committed["actor"].applied_fragments == (0,)
    assert actor.fragment_versions == (1, 0)
    assert pairing.overwritten_rounds == 1 and pairing.journal == []


@pytest.mark.asyncio
async def test_strict_profile_still_refuses_an_unretained_broadcast():
    actor = _PairedRole("actor", [], [])
    pairing = _async_pairing(actor, partial_quorum=False)
    actor._staged_steps.append(1)

    with pytest.raises(RuntimeError, match="no retained paired submission"):
        await pairing.commit_staged_broadcasts()
    assert actor.closed


@pytest.mark.asyncio
async def test_late_island_applies_the_round_its_push_may_have_missed():
    actor = _PairedRole("actor", [_candidate("actor", 1)], [], release_broadcasts=False)
    pairing = _async_pairing(actor)
    assert _submitted(pairing.submit_ready()) == [1]
    # The round closed (the syncer kept or dropped this push); its BCAST is
    # the normal paired apply either way.
    actor._staged_steps.append(1)
    actor._queued_steps.clear()

    committed = await pairing.commit_staged_broadcasts()

    assert committed["actor"].applied_fragments == (0,)
    assert pairing.inflight == {} and len(pairing.journal) == 1
    assert pairing.overwritten_rounds == 0


@pytest.mark.asyncio
async def test_an_unsent_retained_push_is_released_when_its_round_closes():
    actor = _PairedRole("actor", [_candidate("actor", 1)], [], false_pushes=1)
    pairing = _async_pairing(actor)
    assert _submitted(pairing.submit_ready()) == []  # zero-chunk outage: retained, not sent
    assert len(pairing.inflight) == 1
    actor._staged_steps.append(1)

    await pairing.commit_staged_broadcasts()

    assert pairing.inflight == {} and pairing.journal == []
    assert pairing.overwritten_rounds == 1 and actor.fragment_versions == (1, 0)


@pytest.mark.asyncio
async def test_closed_round_is_not_uploaded():
    actor = _PairedRole("actor", [_candidate("actor", 1)], [])
    pairing = _async_pairing(actor)
    actor._staged_steps.append(1)  # broadcast 1 landed before this island pushed

    assert _submitted(pairing.submit_ready()) == []
    assert pairing.inflight == {}


@pytest.mark.asyncio
async def test_out_of_order_broadcasts_apply_in_version_order():
    actor = _PairedRole("actor", [], [])
    pairing = _async_pairing(actor)
    actor._staged_steps.append(3)  # fragment 0, version 3: its base 1 is still on the wire

    committed = await pairing.commit_staged_broadcasts()
    assert committed["actor"].applied_fragments == ()
    assert actor.fragment_versions == (0, 0)

    actor._staged_steps.extend([2, 1])
    committed = await pairing.commit_staged_broadcasts()
    assert actor.fragment_versions == (3, 2)
    assert set(committed["actor"].applied_fragments) == {0, 1}
    assert pairing.overwritten_rounds == 3


@pytest.mark.asyncio
async def test_terminal_broadcast_waits_for_the_final_cut():
    actor = _PairedRole("actor", [], [])
    actor.fragment_versions = (5, 4)
    pairing = _async_pairing(actor)
    actor._staged_steps.append(6)

    committed = await pairing.commit_staged_broadcasts()

    assert committed["actor"].applied_fragments == () and actor.fragment_versions == (5, 4)


@pytest.mark.asyncio
async def test_two_roles_overwrite_together_or_not_at_all():
    actor, critic = _PairedRole("actor", [], []), _PairedRole("critic", [], [])
    pairing = _async_pairing(actor, critic)
    actor._staged_steps.append(1)  # the critic syncer's BCAST 1 has not arrived

    await pairing.commit_staged_broadcasts()
    assert actor.fragment_versions == critic.fragment_versions == (0, 0)

    critic._staged_steps.append(1)
    await pairing.commit_staged_broadcasts()
    assert actor.fragment_versions == critic.fragment_versions == (1, 0)


@pytest.mark.asyncio
async def test_final_cut_releases_an_unsent_retained_push():
    actor = _PairedRole("actor", [_candidate("actor", 1)], [], false_pushes=1)
    pairing = _async_pairing(actor)
    pairing.submit_ready()
    actor.finalizing = True

    await pairing.finish_without_more_training()

    assert pairing.inflight == {}


def test_boundary_applies_closed_rounds_before_it_submits():
    import asyncio

    from yeto.rl.engine.full_parameter_sync import FullParameterSync
    from test_rl_full_parameter_sync import _Driver

    events = []
    actor = _PairedRole("actor", [_candidate("actor", 1)], events)
    pairing = _async_pairing(actor)
    session = FullParameterSync(pairing, groups={"actor": "g"}, run=asyncio.run,
                                trajectories=lambda rollout_id: rollout_id)
    driver = _Driver()
    session.start(driver)
    actor._staged_steps.append(1)  # round 1 closed while this island trained

    session.boundary(driver, rollout_id=0, stats=None)

    assert ("actor", 1, True) not in events  # no refused upload
    name, fields = driver.events[-1]
    assert name == "rl_full_parameter_round"
    assert fields["sync/fragment_versions"] == [1, 0]
    assert fields["sync/overwritten_rounds"] == 1


def test_partial_quorum_is_a_streaming_profile():
    with pytest.raises(ValueError, match="partial quorum"):
        FullParameterSyncProfile(roles=("actor",), total_fragment_steps=4, fragments=2,
                                 local_horizon=1, pipeline_depth=1, wait_timeout=1.0,
                                 poll_seconds=0.01, sweep=True, partial_quorum=True)


# -- role stream permits ---------------------------------------------------------


class _PullClient:
    connection_generation = 0

    def __init__(self):
        self.pulls: list[PullRequest] = []

    def drain_pulls(self):
        pulls, self.pulls = self.pulls, []
        return pulls


def _stream(partial_quorum: bool):
    from yeto.rl.adapters.miles.full_parameter.role_stream import MilesFullParameterRoleStream

    stream = MilesFullParameterRoleStream(
        replace(_role_stream_config("actor", 30100), partial_quorum=partial_quorum))
    stream.client = _PullClient()
    return stream


def test_newer_pull_supersedes_a_closed_round_only_with_a_partial_quorum():
    stream = _stream(partial_quorum=True)
    stream.client.pulls = [PullRequest(0, 1, 1)]
    stream._queue_pulls()
    # PULL 3 (fragment 0) overtook BCAST 1 on another WAN stream.
    stream.client.pulls = [PullRequest(0, 3, 1)]
    stream._queue_pulls()
    assert stream._pending == {0: PullRequest(0, 3, 1)}
    stream.client.pulls = [PullRequest(0, 1, 1)]  # a stale duplicate never wins
    stream._queue_pulls()
    assert stream._pending == {0: PullRequest(0, 3, 1)}

    strict = _stream(partial_quorum=False)
    strict.client.pulls = [PullRequest(0, 1, 1), PullRequest(0, 3, 1)]
    with pytest.raises(RuntimeError, match="conflicting PULLs"):
        strict._queue_pulls()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_quorum", [True, False])
async def test_applying_a_round_drops_its_unanswered_pull(partial_quorum):
    stream = _stream(partial_quorum)
    stream._fragment_versions = [0, 0]
    stream._pending = {0: PullRequest(0, 1, 1)}
    stream._staged = {(0, 1): object()}

    async def apply(selected, **_):
        stream._fragment_versions[0] = 1

    stream._apply_authoritative = apply
    await stream.apply_staged_broadcast(0, 1)

    assert (stream._pending == {}) is partial_quorum


def test_partial_quorum_is_bound_into_the_session_contract(monkeypatch):
    from yeto.rl.adapters.miles.full_parameter import role_stream

    monkeypatch.setattr("yeto.rl.adapters.miles.identity.backend_identity",
                        lambda: type("I", (), {"sha256": lambda self: "0" * 64})())
    strict = _role_stream_config("actor", 30100)
    digest = role_stream.role_stream_session_contract_hash("1" * 64, strict)
    partial = role_stream.role_stream_session_contract_hash(
        "1" * 64, replace(strict, partial_quorum=True))
    assert digest != partial
    with pytest.raises(ValueError, match="matrix merge"):
        replace(strict, matrix_merge="iso")


def test_rda_layout_merges_only_non_embedding_fragments_and_changes_the_hash():
    from yeto.fragments import MERGE_AVG, MERGE_RDA
    from yeto.rl.local_learner import ComponentIdentity, ParameterLayout, ParameterSpec

    specs = (ParameterSpec("actor", "embedding.word_embeddings.weight", (4, 2), "float32", 8, "tp0"),
             ParameterSpec("actor", "decoder.layers.0.mlp.linear_fc1.weight", (2, 2), "float32", 4, "tp0"))
    component = ComponentIdentity("actor", "a" * 40, "c" * 64)
    kwargs = dict(algorithm="grpo", components=(component,), specs=specs, num_fragments=2,
                  fragment_strategy="owner_affine")
    avg = ParameterLayout.create(**kwargs)
    rda = ParameterLayout.create(**kwargs, matrix_merge="rda")
    assert [f.merge_mode for f in avg.fragments.fragments] == [MERGE_AVG, MERGE_AVG]
    modes = {f.tensors[0][0].rsplit("::", 1)[1]: f.merge_mode for f in rda.fragments.fragments}
    assert modes == {"embedding.word_embeddings.weight": MERGE_AVG,
                     "decoder.layers.0.mlp.linear_fc1.weight": MERGE_RDA}
    assert avg.layout_hash != rda.layout_hash
    assert avg.layout_hash == ParameterLayout.create(**kwargs, matrix_merge="avg").layout_hash


# -- launcher ---------------------------------------------------------------------


def _plan(extra=()):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", tuple(extra), gpu=TWO_ISLANDS)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _flag(command: str, flag: str) -> str:
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1]


def test_explicit_sweep_is_the_default_control():
    _, default = _plan(FULL)
    _, sweep = _plan(FULL + ("--rl-sync-full-parameter-rounds", "sweep"))
    assert default == sweep
    assert "--full-parameter-rounds" not in default["island_requests"][0]["learner_command"]


def test_async_syncer_runs_ordinary_fragment_rounds_with_the_decoupled_outer_step():
    from yeto import launcher

    args, plan = _plan(ASYNC + ("--local-rl-rounds-per-sync", "3",
                                "--rl-sync-fragment-pipeline", "2"))
    [actor] = plan["role_syncers"]
    command = actor["command"]
    assert "--policy-sweep-fragments" not in command and "--require-profile-binding" in command
    expected = {"--learners": "2", "--quorum": "2", "--grace-ms": "0", "--pipeline": "2",
                "--sync-interval-steps": "3.0", "--delta-correction": "none",
                "--total-steps": "12", "--outer-lr": "0.7", "--outer-momentum": "0.9",
                "--learner-weight": "equal", "--max-base-lag": "0", "--checkpoint-every": "1"}
    assert {flag: _flag(command, flag) for flag in expected} == expected
    profile = launcher.rl_syncer_profile(args, 2)
    assert profile.policy_sweep_fragments is None and profile.require_profile_binding
    sweep_args, _ = _plan(FULL)
    assert profile.sha256 != launcher.rl_syncer_profile(sweep_args, 2).sha256
    for island in plan["island_requests"]:
        island_command = island["learner_command"]
        assert _flag(island_command, "--full-parameter-rounds") == "async"
        assert _flag(island_command, "--local-horizon") == "3"
        assert _flag(island_command, "--pipeline") == "2"
        assert _flag(island_command, "--syncer-profile-sha256") == profile.sha256


def test_async_rounds_take_the_outer_merge_options():
    args, plan = _plan(ASYNC + (
        "--rl-sync-quorum", "1", "--rl-sync-grace-ms", "250",
        "--rl-sync-learner-weight", "tokens2-over-steps", "--rl-sync-max-base-lag", "1",
        "--rl-sync-outer-lr", "0.5", "--rl-sync-outer-momentum", "0.8",
        "--rl-sync-matrix-merge", "rda"))
    command = plan["role_syncers"][0]["command"]
    assert {flag: _flag(command, flag) for flag in (
        "--quorum", "--grace-ms", "--learner-weight", "--max-base-lag",
        "--outer-lr", "--outer-momentum")} == {
        "--quorum": "1", "--grace-ms": "250", "--learner-weight": "tokens2-over-steps",
        # one round of lag is P versions for the syncer
        "--max-base-lag": str(args.fragments), "--outer-lr": "0.5", "--outer-momentum": "0.8"}
    assert args.fragments > 1
    assert all(_flag(island["learner_command"], "--rl-sync-matrix-merge") == "rda"
               for island in plan["island_requests"])


def test_async_checkpoint_interval_reaches_the_syncer(monkeypatch):
    from yeto.launcher import _prepare_rl_args, syncer_command

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", ASYNC + ("--rl-sync-checkpoint-every", "4000"), gpu=TWO_ISLANDS)
    _prepare_rl_args(args)
    assert _flag(syncer_command(args, 2), "--checkpoint-every") == "4000"
    assert "--async-checkpoint" not in syncer_command(args, 2).split()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (ASYNC + ("--rl-sync-fragment-pipeline", "3"), r"between 1 and min\(2, --fragments\)"),
        (ASYNC + ("--rl-sync-checkpoint-every", "0"), "checkpoint-every >= 1"),
        (ASYNC + ("--rl-sync-async-checkpoint",), "no --rl-sync-async-checkpoint"),
        (ASYNC + ("--rl-sync-overlap", "--rl-sync-fragment-pipeline", "2"), "no --rl-sync-overlap"),
        (FULL + ("--local-rl-rounds-per-sync", "2"), "local-rl-rounds-per-sync 1"),
        (FULL + ("--rl-sync-quorum", "1"), "full-parameter-rounds async"),
        (FULL + ("--rl-sync-matrix-merge", "rda"), "full-parameter-rounds async"),
        (("--rl-sync-full-parameter-rounds", "async"), "only applies to --rl-sync-preset full-parameter"),
    ],
)
def test_async_refusals(extra, message, monkeypatch):
    from yeto.launcher import _prepare_rl_args

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", extra, gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


def _island_args(**overrides):
    from types import SimpleNamespace

    values = dict(optimizer_steps=1, local_horizon=2, rl_max_policy_age=0, global_rounds=3,
                  fragments=2, total_fragment_steps=6, parameter_layout_sha256="e" * 64,
                  syncer_profile_sha256="f" * 64, full_parameter_rounds="async", pipeline=2,
                  sweep_overlap=False, rl_sync_matrix_merge="avg", eval_only=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_island_checks_the_async_profile():
    from yeto.rl.adapters.miles.island_entry import _check_dense_full_sweep

    _check_dense_full_sweep(_island_args(), yeto_policy_sync=True)
    _check_dense_full_sweep(_island_args(rl_sync_matrix_merge="rda"), yeto_policy_sync=True)
    with pytest.raises(ValueError, match="--pipeline 1..min"):
        _check_dense_full_sweep(_island_args(pipeline=3, fragments=4, total_fragment_steps=12),
                                yeto_policy_sync=True)
    with pytest.raises(ValueError, match="strict sweep requires H=1"):
        _check_dense_full_sweep(_island_args(full_parameter_rounds="sweep"), yeto_policy_sync=True)
    with pytest.raises(ValueError, match="plain averaging"):
        _check_dense_full_sweep(_island_args(full_parameter_rounds="sweep", local_horizon=1,
                                             rl_sync_matrix_merge="rda"), yeto_policy_sync=True)


# -- the real Rust syncer -----------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
NUMEL = 3
P = 2
needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


@pytest.fixture
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "async-rounds-test-key")


def _start_async_syncer(binary: Path, tmp_path: Path, *extra: str, total_steps: int,
                        syncer_args: tuple[str, ...] = (), log_file: Path | None = None):
    """Start the launcher's async syncer; its output goes to a pipe, or to
    ``log_file`` when a test must read the log while the syncer runs."""
    from yeto import launcher

    args, _ = _plan(FULL[:-1] + (str(P), "--rl-sync-full-parameter-rounds", "async",
                                 "--total-steps", str(total_steps), *extra))
    command = launcher.role_syncer_commands(args, 2, binary=str(binary))["actor"]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = (command
               .replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    command = " ".join((command, *map(shlex.quote, syncer_args)))
    assert "--policy-sweep-fragments" not in command
    if log_file is None:
        process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
    else:
        with log_file.open("w") as out:
            process = subprocess.Popen(["bash", "-c", command], stdout=out,
                                       stderr=subprocess.STDOUT, text=True)
    return process, port, launcher.rl_syncer_profile(args, 2).sha256


def _layout():
    from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout

    return FragmentLayout([Fragment(MERGE_AVG, [(f"actor.w{i}", NUMEL)]) for i in range(P)])


def _client(learner_id: int, port: int, profile_hash: str):
    from yeto.protocol import DTYPE_F32, SyncerClient

    return SyncerClient(("127.0.0.1", port), learner_id, _layout(), dtype=DTYPE_F32,
                        num_streams=1, connect_timeout=10, max_reconnects=0,
                        finalization_timeout=30, syncer_profile_hash=bytes.fromhex(profile_hash))


def _f32(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _wait(predicate, what: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise TimeoutError(what)


def _initial(fid: int) -> np.ndarray:
    return np.asarray([float(fid) + 0.25 * i for i in range(NUMEL)], dtype=np.float32)


def _delta(learner: int, step: int) -> np.ndarray:
    return np.asarray([0.5 + 0.125 * learner + 0.03125 * step + 0.001 * i for i in range(NUMEL)],
                      dtype=np.float32)


def _nesterov_reference(rounds, *, lr: float, mu: float) -> dict[int, np.ndarray]:
    """The syncer's outer step in f32, reduced in learner order.

    ``rounds`` maps a global step to ``{learner: (weight, delta)}``;
    g = -(weighted mean delta); buf <- mu*buf + g; theta <- theta - lr*(g + mu*buf)."""

    lr32, mu32 = np.float32(lr), np.float32(mu)
    params = {fid: _initial(fid) for fid in range(P)}
    buffers = {fid: np.zeros(NUMEL, dtype=np.float32) for fid in range(P)}
    for step in sorted(rounds):
        fid = (step - 1) % P
        contributions = [rounds[step][learner] for learner in sorted(rounds[step])]
        total = sum(weight for weight, _ in contributions)
        merged = np.zeros(NUMEL, dtype=np.float32)
        for weight, delta in contributions:
            merged += np.float32(weight / total) * -delta
        buffers[fid] = mu32 * buffers[fid] + merged
        params[fid] = params[fid] - lr32 * (merged + mu32 * buffers[fid])
    return params


def _drive(binary, tmp_path, extra, *, total_steps, push, late=None):
    """Run the launcher's async syncer with two learners; return (finals, log).

    ``push(learner, permit)`` is called once per PULL a learner sees and returns
    ``(base, c_steps, c_tokens, delta)``, or None to stay silent.
    ``late(versions)`` is called every poll with learner 0's committed fragment
    versions and returns ``(learner, permit, answer)`` pushes to send now."""

    log_file = tmp_path / "syncer.log"
    process, port, profile_hash = _start_async_syncer(binary, tmp_path, *extra,
                                                      total_steps=total_steps, log_file=log_file)
    clients = [_client(learner, port, profile_hash) for learner in (0, 1)]
    seen: list[set] = [set(), set()]
    versions = [0] * P

    def send(learner, permit, answer):
        base, c_steps, c_tokens, delta = answer
        clients[learner].push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                       base, permit.global_step, c_steps, c_tokens, _f32(delta))

    def poll():
        for learner, client in enumerate(clients):
            client.check_health()
            for update in client.drain_updates():
                if learner == 0:
                    versions[update.fragment_id] = max(versions[update.fragment_id], update.version)
            for permit in client.drain_pulls():
                if (permit.global_step, permit.round_attempt) in seen[learner]:
                    continue
                seen[learner].add((permit.global_step, permit.round_attempt))
                answer = push(learner, permit)
                if answer is not None:
                    send(learner, permit, answer)
        for learner, permit, answer in (late(tuple(versions)) if late else ()):
            send(learner, permit, answer)
        return all(client.finalizing.is_set() for client in clients)

    try:
        for client in clients:
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=60)
        finals = []
        for client in clients:
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals.append({f.fragment_id: np.frombuffer(bytes(f.data), dtype=np.float32)[-NUMEL:]
                           for f in fragments})
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        for client in clients:
            client.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    log = log_file.read_text()
    assert finals[0].keys() == finals[1].keys() == set(range(P))
    for fid in range(P):
        assert np.array_equal(finals[0][fid], finals[1][fid])  # one global cut
    return finals[0], log


def _base(step: int) -> int:
    return max(0, step - P)


def _assert_matches(finals, expected):
    for fid in range(P):
        np.testing.assert_allclose(finals[fid], expected[fid], rtol=0, atol=1e-6)


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_quorum_one_closes_rounds_without_an_absent_island(syncer_binary, tmp_path):
    rounds: dict[int, dict] = {}

    def push(learner, permit):
        if learner == 1:
            return None  # connected, never contributes: absent from every round
        delta = _delta(0, permit.global_step)
        rounds[permit.global_step] = {0: (1.0, delta)}
        return _base(permit.global_step), 1, 7, delta

    # Two outer rounds per fragment, so the momentum buffer carries over.
    finals, log = _drive(syncer_binary, tmp_path, ("--rl-sync-quorum", "1"),
                         total_steps=2, push=push)
    assert sorted(rounds) == list(range(1, 2 * P + 1))
    _assert_matches(finals, _nesterov_reference(rounds, lr=0.7, mu=0.9))
    assert "push rejected" not in log


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_tokens_squared_over_steps_weights_match_the_reference(syncer_binary, tmp_path):
    counters = {0: (1, 10), 1: (3, 40)}  # (c_steps, c_tokens): weights 100 and 533.3
    rounds: dict[int, dict] = {}

    def push(learner, permit):
        c_steps, c_tokens = counters[learner]
        delta = _delta(learner, permit.global_step)
        rounds.setdefault(permit.global_step, {})[learner] = (c_tokens ** 2 / c_steps, delta)
        return _base(permit.global_step), c_steps, c_tokens, delta

    finals, _ = _drive(syncer_binary, tmp_path, (
        "--rl-sync-learner-weight", "tokens2-over-steps",
        "--rl-sync-outer-lr", "1.0", "--rl-sync-outer-momentum", "0.0"),
        total_steps=1, push=push)
    _assert_matches(finals, _nesterov_reference(rounds, lr=1.0, mu=0.0))
    for fid in range(P):
        equal = _initial(fid) + (_delta(0, fid + 1) + _delta(1, fid + 1)) / 2
        assert not np.allclose(finals[fid], equal, atol=1e-4)  # the weights mattered


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
def test_late_push_after_its_round_closed_is_refused_and_not_merged(syncer_binary, tmp_path):
    rounds: dict[int, dict] = {}
    held: dict[int, PullRequest] = {}
    last = 2 * P  # the run's final outer step
    held_last: list = []  # learner 0's answer for the final step
    log_file = tmp_path / "syncer.log"

    def push(learner, permit):
        if learner == 1:
            held[permit.global_step] = permit  # this island is slow
            return None
        delta = _delta(0, permit.global_step)
        rounds[permit.global_step] = {0: (1.0, delta)}
        answer = _base(permit.global_step), 1, 7, delta
        if permit.global_step == last:
            # Keep the run open until the syncer has handled the late push;
            # otherwise it can finish first and never log the refusal.
            held_last.append((0, permit, answer))
            return None
        return answer

    def late(versions):
        # Learner 1 answers PULL 1 only after round 1 committed without it.
        if versions[0] >= 1 and 1 in held:
            return [(1, held.pop(1), (0, 1, 7, _delta(1, 1)))]
        if held_last and 1 not in held and "push rejected" in log_file.read_text():
            return [held_last.pop()]
        return []

    finals, log = _drive(syncer_binary, tmp_path, ("--rl-sync-quorum", "1"),
                         total_steps=2, push=push, late=late)
    assert 1 not in held  # the late push was sent
    _assert_matches(finals, _nesterov_reference(rounds, lr=0.7, mu=0.9))
    assert "push rejected" in log


@needs_cargo
@pytest.mark.usefixtures("_hmac_key")
@pytest.mark.parametrize("max_base_lag", [0, 1])
def test_max_base_lag_bounds_a_stale_base(syncer_binary, tmp_path, max_base_lag):
    """Learner 1 answers fragment 0's second round against the initial
    version (one round stale). Lag 1 merges it; lag 0 is a strict failure."""

    rounds: dict[int, dict] = {}

    def push(learner, permit):
        step = permit.global_step
        delta = _delta(learner, step)
        base = 0 if (learner == 1 and step == P + 1) else _base(step)
        rounds.setdefault(step, {})[learner] = (1.0, delta)
        return base, 1, 7, delta

    extra = ("--rl-sync-max-base-lag", str(max_base_lag),
             "--rl-sync-outer-lr", "1.0", "--rl-sync-outer-momentum", "0.0")
    if max_base_lag:
        finals, _ = _drive(syncer_binary, tmp_path, extra, total_steps=2, push=push)
        _assert_matches(finals, _nesterov_reference(rounds, lr=1.0, mu=0.0))
        return
    with pytest.raises(Exception):
        _drive(syncer_binary, tmp_path, extra, total_steps=2, push=push)
    tape = (tmp_path / "tape.jsonl").read_text()
    assert "rejected_stale_updates" in tape


def test_async_streaming_weighted_fold_reaches_the_rl_syncer(monkeypatch):
    from yeto.launcher import _prepare_rl_args, syncer_command

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")
    args = _launcher_args("ports", ASYNC + ("--rl-sync-quorum", "1", "--syncer-weighted-fold", "streaming"),
                          gpu=TWO_ISLANDS)
    _prepare_rl_args(args)
    assert _flag(syncer_command(args, 2), "--weighted-fold") == "streaming"
    default = _launcher_args("ports", ASYNC, gpu=TWO_ISLANDS)
    _prepare_rl_args(default)
    assert "--weighted-fold" not in syncer_command(default, 2).split()
