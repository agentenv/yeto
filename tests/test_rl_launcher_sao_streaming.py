"""Launcher end-to-end wiring of --rl-sync-preset sao-streaming-full (P0-1.7b.3). No Ray.

The final contract bundle freezes the syncer semantic profile.  The launcher
starts one syncer per role (actor, critic) with exactly that profile and
ships every island its own runtime contract and layout attestation.
"""

from __future__ import annotations

import hashlib
import json
import shlex

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_sao_streaming_runtime import _syncer_profile

from yeto.syncer_profile import SyncerSemanticProfile

TWO_ISLANDS = "modal:2xh100,modal:2xh100"
UNVERIFIED = ("execution:critic", "advantage_estimators:ppo", "features:sao_dis",
              "features:gae_decoupled", "features:gae_length_adaptive",
              "features:value_hl_gauss", "features:critic_multi_update")


def _write(path, payload) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _manifest(tmp_path, profile=None, islands=2, ports=None, horizon=1):
    profile = profile or _syncer_profile()
    ports = ports or {"actor": 29400, "critic": 29401}
    records = []
    for island_id in range(islands):
        shipped = {}
        for name, file in (("streaming_context", "sao-streaming-context.json"),
                           ("layout_attestation", "layout-attestation.json")):
            host = tmp_path / f"island-{island_id}" / file
            content = {"island": island_id, "file": file,
                       **{role: {"syncer": {"port": port}, "local_horizon": horizon}
                          for role, port in ports.items()}}
            shipped[name] = {"host_path": str(host),
                             "container_path": f"/contracts/island-{island_id}/{file}",
                             "sha256": _write(host, content)}
        records.append({"island_id": island_id,
                        "sao_context": {"sha256": f"{island_id + 1}" * 64}, **shipped})
    manifest = tmp_path / "manifest.json"
    _write(manifest, {
        "schema": "yeto.tbench21-sao-streaming-contracts.v1",
        "phase": "final",
        "expected_fragments": profile["total_steps"],
        "syncer_profile": {"profile": profile,
                           "semantic_sha256": SyncerSemanticProfile.from_mapping(profile).sha256},
        "islands": records,
    })
    return manifest


def _sao_args(tmp_path, manifest, extra=()):
    from yeto.rl.algos.sao import sao_algorithm_spec

    spec = tmp_path / "sao-spec.json"
    spec.write_text(json.dumps(sao_algorithm_spec("coding").to_dict()))
    allow = [token for mechanism in UNVERIFIED
             for token in ("--rl-allow-unverified-mechanism", mechanism)]
    return _launcher_args("ports", (
        "--rl-sync-preset", "sao-streaming-full", "--rl-placement", "fixed-partition",
        "--rl-rollout-gpus", "1", "--rl-sao-contracts", str(manifest),
        "--rl-algorithm-spec", str(spec), "--rl-lr-schedule", "constant", *allow, *extra,
    ), gpu=TWO_ISLANDS)


def _plan(tmp_path, extra=()):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _sao_args(tmp_path, _manifest(tmp_path), extra)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _flag(command: str, flag: str) -> str:
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1]


def test_head_controller_reloads_the_contract_bundle(tmp_path, monkeypatch):
    """The head replays the launch args as JSON, which drops the loaded
    bundle; the head must reload it to render the role syncers
    (s23-g3-sao-probe-20261014a: AttributeError rl_sao_manifest)."""
    from yeto import cli, launcher

    args, plan = _plan(tmp_path)
    seen = {}

    class Syncer:
        def __init__(self, head_args, num_learners):
            seen["command"] = launcher.syncer_command(head_args, num_learners)

        def start(self):
            pass

        def start_log_forwarder(self):
            pass

        def start_tape_forwarder(self, _args):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(launcher, "LocalSyncer", Syncer)
    monkeypatch.setattr(launcher, "run", lambda *_a, **_k: 0)
    assert cli.cmd_head(json.dumps(cli._serializable_args(args))) == 0
    assert seen["command"] == launcher.syncer_command(args, 2)
    assert "--require-profile-binding" in str(seen["command"])


def test_role_syncers_serve_exactly_the_frozen_profile(tmp_path):
    from yeto import launcher

    args, plan = _plan(tmp_path)
    actor, critic = plan["role_syncers"]
    assert (actor["role"], critic["role"]) == ("actor", "critic")
    frozen = _syncer_profile()
    for syncer in (actor, critic):
        command = syncer["command"]
        expected = {"--learners": "2", "--quorum": "2", "--grace-ms": "0", "--pipeline": "2",
                    "--sync-interval-steps": "24.0", "--delta-correction": "none",
                    "--total-steps": "8", "--outer-lr": "0.7", "--outer-momentum": "0.9",
                    "--quorum-timeout-s": "900", "--final-ack-timeout-s": "900",
                    "--learner-weight": "equal", "--max-base-lag": "0",
                    "--checkpoint-every": "1"}
        assert {flag: _flag(command, flag) for flag in expected} == expected
        assert "--mark-final-checkpoint" in command and "--require-profile-binding" in command
        assert "--policy-sweep-fragments" not in command
    assert _flag(actor["command"], "--port") != _flag(critic["command"], "--port")
    assert launcher.rl_syncer_profile(args, 2).sha256 == (
        SyncerSemanticProfile.from_mapping(frozen).sha256)


def test_each_island_gets_its_own_contract(tmp_path):
    from yeto import launcher

    args, plan = _plan(tmp_path)
    for learner_id, island in enumerate(plan["island_requests"]):
        command = island["learner_command"]
        assert _flag(command, "--sync-preset") == "sao-streaming-full"
        assert _flag(command, "--parameter-mode") == "full"
        assert _flag(command, "--sao-streaming-runtime") == (
            f"/contracts/island-{learner_id}/sao-streaming-context.json")
        assert _flag(command, "--sao-context-sha256") == f"{learner_id + 1}" * 64
        assert "--lora-r" not in command
        mounts = launcher.sao_contract_mounts(args, learner_id)
        assert set(mounts) == {f"/contracts/island-{learner_id}/sao-streaming-context.json",
                               f"/contracts/island-{learner_id}/layout-attestation.json"}
        runtime = args.rl_sao_manifest.island(learner_id).runtime
        assert _flag(command, "--sao-streaming-runtime-sha256") == runtime.sha256


def test_island_entry_accepts_the_sao_island_command(tmp_path):
    from yeto.rl.adapters.miles import island_entry

    _, plan = _plan(tmp_path)
    command = plan["island_requests"][1]["learner_command"]
    tokens = shlex.split(command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1])
    env = {"$SYNCER_ADDR": "10.0.0.1:29400", "$CRITIC_SYNCER_ADDR": "10.0.0.1:29401",
           "$LEARNER_ID": "1"}
    parsed = island_entry.parse_args([env.get(token, token) for token in tokens])
    assert (parsed.parameter_mode, parsed.sync_preset) == ("full", "sao-streaming-full")
    assert parsed.critic_syncer == "10.0.0.1:29401"


@pytest.mark.parametrize(
    ("extra", "message"),
    (
        (("--rl-sync-island-lease-s", "30"), "island pool"),
        (("--experimental-rl-sync",), "experimental"),
    ),
)
def test_launcher_rejects_combinations_outside_the_sao_profile(tmp_path, extra, message):
    from yeto.launcher import _prepare_rl_args

    args = _sao_args(tmp_path, _manifest(tmp_path), extra)
    with pytest.raises(ValueError, match=message):
        _prepare_rl_args(args)


def test_launcher_rejects_a_bundle_for_another_island_count(tmp_path):
    from yeto.launcher import _prepare_rl_args

    profile = {**_syncer_profile(), "learners": 3, "quorum": 3}
    args = _sao_args(tmp_path, _manifest(tmp_path, profile, islands=3))
    with pytest.raises(ValueError, match="3 islands"):
        _prepare_rl_args(args)


def test_launcher_rejects_a_changed_contract_file(tmp_path):
    from yeto.launcher import _prepare_rl_args

    manifest = _manifest(tmp_path)
    (tmp_path / "island-1" / "layout-attestation.json").write_text("{}")
    with pytest.raises(ValueError, match="missing or changed"):
        _prepare_rl_args(_sao_args(tmp_path, manifest))


def test_launcher_rejects_contracts_for_other_syncer_ports(tmp_path):
    from yeto.launcher import _prepare_rl_args

    manifest = _manifest(tmp_path, ports={"actor": 29400, "critic": 29500})
    with pytest.raises(ValueError, match="role syncer ports"):
        _prepare_rl_args(_sao_args(tmp_path, manifest))


@pytest.mark.parametrize(
    ("extra", "conflict"),
    (
        (("--outer-lr", "0.5"), "--outer-lr 0.5 (contract: 0.7)"),
        (("--quorum", "3"), "--quorum 3 (contract: 2)"),
        (("--pipeline", "1"), "--pipeline 1 (contract: 2)"),
        (("--fragments", "4"), "--fragments 4 (contract: 8)"),
    ),
)
def test_launcher_rejects_explicit_settings_that_conflict_with_the_contract(
        tmp_path, extra, conflict):
    from yeto.launcher import _prepare_rl_args

    with pytest.raises(ValueError, match="conflict with the SAO contracts") as raised:
        _prepare_rl_args(_sao_args(tmp_path, _manifest(tmp_path), extra))
    assert conflict in str(raised.value)


def test_explicit_settings_equal_to_the_contract_are_accepted(tmp_path):
    from yeto.launcher import _prepare_rl_args

    args = _sao_args(tmp_path, _manifest(tmp_path), ("--outer-lr", "0.7", "--quorum", "2"))
    _prepare_rl_args(args)
    assert (args.outer_lr, args.quorum) == (0.7, 2)


def test_local_horizon_comes_from_the_contract(tmp_path):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _sao_args(tmp_path, _manifest(tmp_path, horizon=2))
    args.controller = "local"
    _prepare_rl_args(args)
    command = dry_run_plan(args)["island_requests"][0]["learner_command"]
    assert _flag(command, "--local-horizon") == "2"


def test_local_horizon_conflict_is_rejected(tmp_path):
    from yeto.launcher import _prepare_rl_args

    with pytest.raises(ValueError, match=r"--local-rl-rounds-per-sync 3 \(contract: 1\)"):
        _prepare_rl_args(_sao_args(tmp_path, _manifest(tmp_path),
                                   ("--local-rl-rounds-per-sync", "3")))


def test_contracts_only_apply_to_the_sao_preset(tmp_path):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", ("--rl-sao-contracts", str(_manifest(tmp_path))),
                          gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match="only applies"):
        _prepare_rl_args(args)


def _locked_syncer(port: int, state: str, tape: str) -> str:
    return (
        f"~/yeto-syncer --port {port} --learners 2 --quorum 2 --grace-ms 0"
        " --grace-gamma 0.8 --grace-tau 2.0 --pipeline 2 --sync-interval-steps 24.0"
        " --delta-correction none --total-steps 8 --quorum-timeout-s 900"
        " --final-ack-timeout-s 900 --mark-final-checkpoint --require-profile-binding"
        " --outer-lr 0.7 --outer-momentum 0.9 --max-base-lag 0 --learner-weight equal"
        f" --checkpoint-path ~/yeto-output/{state} --checkpoint-every 1"
        f" $(test -f ~/yeto-output/{state} && echo --resume)"
        f" --event-tape ~/yeto-output/{tape}"
    )


def test_dry_run_locks_the_complete_sao_commands(tmp_path):
    """The whole role-syncer commands and the SAO part of the island command."""
    args, plan = _plan(tmp_path)
    assert [(s["role"], s["command"]) for s in plan["role_syncers"]] == [
        ("actor", _locked_syncer(29400, "yeto-state.ckpt", "yeto-tape.jsonl")),
        ("critic", _locked_syncer(29401, "yeto-critic-state.ckpt", "yeto-critic-tape.jsonl")),
    ]
    runtime = args.rl_sao_manifest.island(1).runtime
    command = plan["island_requests"][1]["learner_command"]
    assert " --syncer $SYNCER_ADDR --critic-syncer $CRITIC_SYNCER_ADDR " in command
    assert (
        " --sync-preset sao-streaming-full --fragments 8 --pipeline 2 --local-horizon 1"
        " --total-fragment-steps 8 "
    ) in command
    assert (
        " --parameter-mode full"
        " --sao-streaming-runtime /contracts/island-1/sao-streaming-context.json"
        f" --sao-streaming-runtime-sha256 {runtime.sha256}"
        f" --sao-context-sha256 {'2' * 64}"
        " --inner-lr "
    ) in command
    for absent in ("--lora-r", "--lora-targets", "--parameter-layout-sha256",
                   "--syncer-profile-sha256", "--policy-sweep-fragments"):
        assert absent not in command


def test_run_d_lora_sao_flags_still_launch(tmp_path):
    """P0-1.2上卡 run d (s23-g1-ports-sao-20261013d): single-island ports SAO
    with LoRA, the critic spec and --rl-lr-schedule constant must keep the
    LoRA island command; the streaming preset must not leak into it."""
    from yeto.launcher import _prepare_rl_args, dry_run_plan
    from yeto.rl.algos.sao import sao_algorithm_spec

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(sao_algorithm_spec("reasoning").to_dict()))
    allow = [token for mechanism in UNVERIFIED
             for token in ("--rl-allow-unverified-mechanism", mechanism)]
    args = _launcher_args("ports", (
        "--rl-lr-schedule", "constant", "--rl-single-island-no-sync",
        "--tuning", "lora", "--lora-r", "16", "--lora-targets", "attention",
        "--fragments", "1", "--pipeline", "1", "--rl-algorithm-spec", str(spec), *allow,
    ), gpu="modal:1xh100")
    args.controller = "local"
    _prepare_rl_args(args)
    (island,) = dry_run_plan(args)["island_requests"]
    command = island["learner_command"]
    assert _flag(command, "--lora-r") == "16"
    assert _flag(command, "--lora-targets") == "attention"
    assert _flag(command, "--rl-lr-schedule") == "constant"
    assert _flag(command, "--sync-preset") == "strict-avg"
    for absent in ("--parameter-mode", "--sao-streaming-runtime", "--critic-syncer"):
        assert absent not in command
