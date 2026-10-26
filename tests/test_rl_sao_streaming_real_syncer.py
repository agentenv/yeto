"""SAO streaming role syncers against the real Rust syncer (P0-1.7b.3).

The launcher starts one syncer per role (actor, critic) for
``--rl-sync-preset sao-streaming-full``, each serving the contract bundle's
frozen semantic profile.  Both role syncers run the exact launcher commands
at the same time; one learner streams every fragment of both roles to the
final cut.  A HELLO with any other profile hash must be refused.
"""

from __future__ import annotations

import socket
import subprocess
import threading
from pathlib import Path

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_dense_sweep_real_syncer import (  # noqa: F401
    CARGO,
    FRAGMENTS,
    NUMEL,
    _client,
    _f32,
    _hmac_key,
    _stop,
    _wait,
    syncer_binary,
)
from test_rl_launcher_sao_streaming import _manifest, _sao_args
from test_rl_sao_streaming_runtime import _syncer_profile

pytestmark = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")

ROLES = ("actor", "critic")
# One learner, serial rounds, a plain outer step: the final cut is init + delta.
PROFILE = {**_syncer_profile(), "learners": 1, "quorum": 1, "pipeline": 1,
           "total_steps": FRAGMENTS, "outer_lr": 1.0, "outer_momentum": 0.0}


def _prepared_args(tmp_path):
    from yeto import launcher

    args = _sao_args(tmp_path, _manifest(tmp_path, PROFILE, islands=1))
    args.gpu = "modal:2xh100"
    args.controller = "local"
    launcher._prepare_rl_args(args)
    return args


def _start_role_syncers(binary, tmp_path):
    """Both launcher role syncers, relocated to free ports and tmp_path."""
    from yeto import launcher

    args = _prepared_args(tmp_path)
    commands = launcher.role_syncer_commands(args, 1, binary=str(binary))
    assert tuple(commands) == ROLES
    started = {}
    for role, command in commands.items():
        syncer = launcher.ROLE_SYNCERS[role]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        command = (command
                   .replace(f"--port {syncer.port}", f"--port {port}")
                   .replace(syncer.checkpoint, str(tmp_path / f"{role}.ckpt"))
                   .replace(syncer.event_tape, str(tmp_path / f"{role}.jsonl")))
        assert "--mark-final-checkpoint" in command and "--require-profile-binding" in command
        process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        started[role] = (process, port)
    return started, args.rl_sao_manifest.profile.sha256


def _stream_to_final(port: int, profile_hash: str, offset: float, results: dict, role: str):
    client = _client(port, profile_hash)
    initial = {fid: [offset + fid] * NUMEL for fid in range(FRAGMENTS)}
    delta = {fid: [0.25 + fid] * NUMEL for fid in range(FRAGMENTS)}
    try:
        client.start()
        for fid in range(FRAGMENTS):
            client.send_init(fid, _f32(initial[fid]))

        def next_pull():
            client.check_health()
            return next(iter(client.drain_pulls()), None)

        pushed = []
        while len(pushed) < FRAGMENTS:
            permit = _wait(next_pull, f"{role} syncer did not permit fragment {len(pushed)}")
            client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                 0, 1, 1, 1, _f32(delta[permit.fragment_id]))
            pushed.append(permit.fragment_id)
        manifest, finals = client.wait_for_final_fragments(timeout=20)
        merged = {}
        for fragment in finals:
            values = torch.frombuffer(bytearray(fragment.data), dtype=torch.float32)[-NUMEL:]
            expected = torch.tensor(initial[fragment.fragment_id]) + torch.tensor(
                delta[fragment.fragment_id])
            merged[fragment.fragment_id] = torch.equal(values, expected)
        client.acknowledge_finalization(manifest)
        results[role] = (sorted(pushed), merged)
    except BaseException as error:  # noqa: BLE001 - reported by the test thread
        results[role] = error
    finally:
        client.close()


def test_both_role_syncers_stream_to_their_final_cut(syncer_binary, tmp_path):
    started, profile_hash = _start_role_syncers(syncer_binary, tmp_path)
    results: dict = {}
    threads = [
        threading.Thread(target=_stream_to_final,
                         args=(port, profile_hash, 10.0 * index, results, role))
        for index, (role, (_process, port)) in enumerate(started.items())
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        for role, (process, _port) in started.items():
            assert process.wait(timeout=15) == 0, role
    finally:
        logs = {role: _stop(process) for role, (process, _port) in started.items()}
    for role in ROLES:
        outcome = results[role]
        assert not isinstance(outcome, BaseException), (role, outcome, logs[role])
        pushed, merged = outcome
        assert pushed == list(range(FRAGMENTS))
        assert merged == {fid: True for fid in range(FRAGMENTS)}, role
        assert "syncer semantic profile" not in logs[role]


def test_role_syncer_refuses_another_profile_hash(syncer_binary, tmp_path):
    started, profile_hash = _start_role_syncers(syncer_binary, tmp_path)
    wrong = ("0" if profile_hash[0] != "0" else "1") + profile_hash[1:]
    process, port = started["critic"]
    client = _client(port, wrong)
    try:
        with pytest.raises(Exception):
            client.start()
            _wait(lambda: client.check_health(), "client stayed healthy", timeout=10)
    finally:
        client.close()
        logs = {role: _stop(p) for role, (p, _port) in started.items()}
    assert "syncer semantic profile does not match" in logs["critic"]
