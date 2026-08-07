import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yeto.rl import (
    MILES_COMMIT,
    MILES_IMAGE,
    MILES_PEFT_VERSION,
    MILES_REPOSITORY,
    SGLANG_COMMIT,
    SGLANG_REPOSITORY,
    ssh_harness,
)
from yeto.rl.bridge import _write_round_audit
from yeto.rl.core import (
    CanonicalTensorSpec,
    canonical_layout_hash,
    canonical_state,
    policy_hash,
    tensors_from_flat,
)
from yeto.rl.learner import parse_args as parse_learner_args
from yeto.rl.ssh_harness import (
    HarnessError,
    _container_name,
    _container_succeeded,
    _docker_ref,
    _host_setup_script,
    _learner_argv,
    _node_start_script,
    _syncer_argv,
    _target_host,
    _verify_oracle,
    _write_plan,
    load_plan,
)

MODEL_REVISION = "a" * 40
LORA_CONFIG_HASH = "b" * 64
IMAGE = "radixark/miles@sha256:" + "d" * 64
SPECS = (
    CanonicalTensorSpec(
        "base_model.model.layer.lora_A.weight", (2,), "float32", 2
    ),
)
LAYOUT_HASH = canonical_layout_hash(SPECS)


def _plan():
    return {
        "schema": 1,
        "run_id": "acceptance",
        "remote_run": ".cache/yeto-rl-ssh/acceptance",
        "remote_env_file": None,
        "ssh_options": [],
        "islands": [
            {"hosts": ["alice@a0", "alice@a1"], "gpus_per_node": 4},
            {"hosts": ["alice@b0", "alice@b1"], "gpus_per_node": 4},
        ],
        "syncer_address": "a0:29400",
        "syncer_port": 29400,
        "docker_image": IMAGE,
        "miles": {
            "repository": MILES_REPOSITORY,
            "commit": MILES_COMMIT,
            "peft_version": MILES_PEFT_VERSION,
        },
        "sglang": {
            "repository": SGLANG_REPOSITORY,
            "commit": SGLANG_COMMIT,
        },
        "source_sha256": "c" * 64,
        "reward_sha256": "e" * 64,
        "syncer_source_sha256": "1" * 64,
        "learner": {
            "model": "org/model",
            "model_revision": MODEL_REVISION,
            "data": "org/data",
            "data_revision": "f" * 40,
            "data_local_path": None,
            "data_sha256": None,
            "reward_function": "pkg.reward:score",
            "global_rounds": 2,
            "groups_per_round": 4,
            "samples_per_group": 2,
            "over_sampling_batch_size": 4,
            "optimizer_steps": 1,
            "rollout_max_response_len": 128,
            "custom_generate_function_path": None,
            "use_session_server": False,
            "session_server_ip": None,
            "session_server_port": None,
            "tito_model": None,
            "expert_parallel": 1,
            "lora_r": 8,
            "lora_targets": "attention",
            "inner_lr": 1e-5,
            "seq_len": 512,
            "seed": 7,
            "wan_streams": 4,
            "trust_remote_code": True,
            "cybergym_url": "http://10.0.0.8:8666",
            "cybergym_agent_id": "benchmark-agent",
            "cybergym_timeout": 90.0,
        },
    }


def _local_data_plan(path: Path):
    plan = _plan()
    plan["learner"].update(
        {
            "data": "/workspace/data/dataset.jsonl",
            "data_revision": None,
            "data_local_path": str(path.resolve()),
            "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    return plan


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _state(version: int, values: list[float]):
    return canonical_state(
        version,
        tensors_from_flat(torch.tensor(values, dtype=torch.float32), SPECS),
        base_model_revision=MODEL_REVISION,
        lora_config_hash=LORA_CONFIG_HASH,
        layout_hash=LAYOUT_HASH,
    )


def _oracle_fixture(tmp_path: Path):
    plan = _plan()
    bases = [_state(0, [0, 0]), _state(1, [2, 2])]
    locals_by_round = [
        [_state(0, [1, 3]), _state(0, [3, 1])],
        [_state(1, [4, 2]), _state(1, [2, 4])],
    ]
    commits = [_state(1, [2, 2]), _state(2, [3, 3])]

    for round_index, base in enumerate(bases, start=1):
        for learner_id in range(2):
            local = locals_by_round[round_index - 1][learner_id]
            delta = torch.tensor(
                [
                    local.tensors[SPECS[0].name][index]
                    - base.tensors[SPECS[0].name][index]
                    for index in range(2)
                ]
            )
            _write_round_audit(
                tmp_path / f"island-{learner_id}" / "audit",
                learner_id=learner_id,
                target_step=round_index,
                base=base,
                delta=delta,
            )

    syncer_events = []
    for step in (1, 2):
        syncer_events.append(
            {
                "step": step,
                "fragment": 0,
                "launch_base_version": step - 1,
                "expected": [0, 1],
                "responded": [0, 1],
                "sync/layout_hash": LAYOUT_HASH,
                "responders": [
                    {
                        "id": learner_id,
                        "base_version": step - 1,
                        "c_steps": 1,
                        "c_tokens": 1,
                        "contribution": 0.5,
                    }
                    for learner_id in (0, 1)
                ],
            }
        )
    _write_jsonl(tmp_path / "syncer" / "events.jsonl", syncer_events)

    for learner_id in range(2):
        events = [
            {
                "event": "rl_policy_apply",
                "policy_version": state.policy_version,
                "sync/global_policy_hash": policy_hash(state),
            }
            for state in [_state(0, [0, 0]), *commits]
        ]
        events.extend(
            {
                "event": "rl_local_round",
                "local_round_id": step,
                "base_policy_version": step - 1,
            }
            for step in (1, 2)
        )
        _write_jsonl(tmp_path / f"island-{learner_id}" / "events.jsonl", events)

    checkpoint = SimpleNamespace(
        global_step=2,
        fragments=[(2, torch.tensor([3.0, 3.0]), torch.zeros(2))],
        ledger={0: (2, 2, 2), 1: (2, 2, 2)},
        layout_hash=LAYOUT_HASH,
    )
    return plan, checkpoint


def _decoupled_oracle_fixture(tmp_path: Path):
    plan = _plan()
    plan["learner"].update(
        {
            "sync_preset": "decoupled",
            "fragments": 2,
            "pipeline": 2,
            "local_horizon": 2,
            "total_fragment_steps": 4,
        }
    )
    final_hash = "9" * 64
    terminal_versions = [3, 4]
    syncer_events = []
    token_totals = {0: 0, 1: 0}
    for step in range(1, 5):
        base_version = max(0, step - 2)
        responders = []
        for learner_id, tokens in ((0, 10), (1, 20)):
            token_totals[learner_id] += tokens
            responders.append(
                {
                    "id": learner_id,
                    "base_version": base_version,
                    "c_steps": 2,
                    "c_tokens": tokens,
                }
            )
        syncer_events.append(
            {
                "step": step,
                "fragment": (step - 1) % 2,
                "launch_base_version": base_version,
                "expected": [0, 1],
                "responded": [0, 1],
                "sync/layout_hash": LAYOUT_HASH,
                "responders": responders,
            }
        )
    _write_jsonl(tmp_path / "syncer" / "events.jsonl", reversed(syncer_events))

    for learner_id in range(2):
        _write_jsonl(
            tmp_path / f"island-{learner_id}" / "events.jsonl",
            [
                {
                    "event": "rl_policy_apply",
                    "sync/global_policy_hash": final_hash,
                },
                {
                    "event": "rl_policy_snapshot",
                    "rl/fragment_versions": terminal_versions,
                    "rl/policy_hash": final_hash,
                },
                {"event": "rl_final_cut"},
            ],
        )
    checkpoint = SimpleNamespace(
        global_step=4,
        fragments=[
            (3, torch.tensor([1.0]), torch.tensor([0.0])),
            (4, torch.tensor([2.0]), torch.tensor([0.0])),
        ],
        ledger={
            learner_id: (4, 8, token_totals[learner_id])
            for learner_id in range(2)
        },
        layout_hash=LAYOUT_HASH,
    )
    return plan, checkpoint, final_hash


def test_oracle_matches_each_ordered_f32_average_and_final_apply(tmp_path):
    plan, checkpoint = _oracle_fixture(tmp_path)
    _verify_oracle(plan, checkpoint, tmp_path)

    delta = tmp_path / "island-1/audit/round-00000002.delta.f32"
    delta.write_bytes(torch.tensor([9.0, 9.0]).numpy().astype("<f4").tobytes())
    with pytest.raises(HarnessError, match="audit identity mismatch"):
        _verify_oracle(plan, checkpoint, tmp_path)


def test_decoupled_oracle_checks_terminal_fragment_cut_and_fixed_roster(tmp_path):
    plan, checkpoint, final_hash = _decoupled_oracle_fixture(tmp_path)

    assert ssh_harness._verify_decoupled(plan, checkpoint, tmp_path) == final_hash

    events = tmp_path / "syncer" / "events.jsonl"
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    rows[0]["fragment"] = 0
    _write_jsonl(events, rows)
    with pytest.raises(HarnessError, match="fragment schedule"):
        ssh_harness._verify_decoupled(plan, checkpoint, tmp_path)


def test_verify_dispatches_and_exports_with_decoupled_plan(tmp_path, monkeypatch):
    plan, checkpoint, final_hash = _decoupled_oracle_fixture(tmp_path / "fixture")
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    artifacts = tmp_path / "artifacts"
    for learner_id, island in enumerate(plan["islands"]):
        for node_id in range(len(island["hosts"])):
            path = artifacts / f"island-{learner_id}" / f"node-{node_id}.inspect.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps([{"State": {"Status": "exited", "ExitCode": 0}}])
            )

    import yeto.export as checkpoint_export
    from yeto.rl import export as rl_export

    monkeypatch.setattr(checkpoint_export, "parse_checkpoint", lambda _path: checkpoint)
    monkeypatch.setattr(
        ssh_harness,
        "_verify_oracle",
        lambda *_args: pytest.fail("strict oracle used for a decoupled plan"),
    )
    monkeypatch.setattr(
        ssh_harness,
        "_verify_decoupled",
        lambda *_args: final_hash,
        raising=False,
    )
    exported = {}
    monkeypatch.setattr(
        rl_export,
        "export_rl_checkpoint",
        lambda *_args, **kwargs: exported.update(kwargs),
    )

    ssh_harness.verify(plan_path, str(tmp_path / "adapter"))

    assert exported["sync_preset"] == "decoupled"
    assert exported["fragments"] == 2
    assert exported["pipeline"] == 2
    assert exported["local_horizon"] == 2


def test_plan_digest_and_current_miles_pin_are_validated(tmp_path):
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _plan())
    _, loaded = load_plan(plan_path)
    assert loaded["miles"]["commit"] == MILES_COMMIT

    payload = json.loads(plan_path.read_text())
    payload["plan"]["miles"]["commit"] = "0" * 40
    plan_path.write_text(json.dumps(payload))
    with pytest.raises(HarnessError, match="plan digest"):
        load_plan(plan_path)

    mismatched = _plan()
    mismatched["islands"][1]["gpus_per_node"] = 2
    _write_plan(plan_path, mismatched)
    with pytest.raises(HarnessError, match="same GPU count"):
        load_plan(plan_path)


def test_miles_and_sglang_pins_include_the_compatible_builds():
    assert MILES_COMMIT == "a5568756b64da5d4c3cdaa5fb80f6bf322308c5e"
    assert SGLANG_COMMIT == "b34df47444271ebda0673d68fe000399804c181b"
    assert MILES_IMAGE == (
        "docker:ghcr.io/alexeisie/miles@sha256:"
        "5be3e0722c7b0174c3c1a5526064872987c7bc367af700117a3589efbd6b19bd"
    )


def test_plan_requires_the_patched_sglang_pin():
    plan = _plan()
    plan.pop("sglang")

    with pytest.raises(HarnessError, match="SGLang"):
        ssh_harness._validate_plan(plan)


def test_local_prompt_plan_is_content_bound_without_a_hub_revision(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _local_data_plan(prompts))

    _, plan = load_plan(plan_path)

    assert plan["learner"]["data"] == "/workspace/data/dataset.jsonl"
    assert plan["learner"]["data_revision"] is None
    assert plan["learner"]["data_sha256"] == hashlib.sha256(
        prompts.read_bytes()
    ).hexdigest()


def test_prepare_maps_a_local_prompt_file_into_the_remote_plan(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    learner = _plan()["learner"]
    args = SimpleNamespace(
        **{
            **learner,
            "data": str(prompts),
            "data_revision": None,
            "total_steps": learner["global_rounds"],
            "rollout_batch_size": learner["groups_per_round"],
            "n_samples_per_prompt": learner["samples_per_group"],
            "rl_image": "docker:" + IMAGE,
            "source_sha256": "c" * 64,
            "reward_sha256": "e" * 64,
            "_provenance": {"dataset": {"source": "local"}},
        }
    )
    namespace = SimpleNamespace(
        run_id="local-prompts",
        host=["alice@a0", "alice@b0"],
        gpus_per_node=1,
        remote_root=".cache/yeto-rl-ssh",
        remote_env_file=None,
        syncer_address=None,
        output_dir=tmp_path / "run",
        ssh_option=[],
    )
    monkeypatch.setattr(ssh_harness, "_resolved_launch_args", lambda *unused: args)
    monkeypatch.setattr(ssh_harness, "_syncer_source_sha256", lambda: "1" * 64)

    plan_path = ssh_harness.prepare(namespace)
    _, plan = load_plan(plan_path)

    assert plan["learner"]["data"] == "/workspace/data/dataset.jsonl"
    assert plan["learner"]["data_local_path"] == str(prompts.resolve())
    assert plan["learner"]["data_revision"] is None


def test_local_prompt_directory_hash_is_stable_and_rejects_symlinks(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
    (first / "nested/b.jsonl").write_text('{"prompt": "b"}\n')
    (first / "a.jsonl").write_text('{"prompt": "a"}\n')
    (second / "a.jsonl").write_text('{"prompt": "a"}\n')
    (second / "nested/b.jsonl").write_text('{"prompt": "b"}\n')

    assert ssh_harness._local_data_sha256(first) == ssh_harness._local_data_sha256(
        second
    )

    (second / "linked.jsonl").symlink_to(second / "a.jsonl")
    with pytest.raises(HarnessError, match="symlink"):
        ssh_harness._local_data_sha256(second)


def test_local_prompt_change_is_rejected_before_deploy(tmp_path, monkeypatch):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _local_data_plan(prompts)
    monkeypatch.setattr("yeto.provenance.verify_source_tree_sha256", lambda value: value)
    monkeypatch.setattr("yeto.provenance.python_spec_sha256", lambda *a, **k: "e" * 64)
    monkeypatch.setattr(ssh_harness, "_syncer_source_sha256", lambda: "1" * 64)
    prompts.write_text('{"messages": [{"role": "user"}]}\n', encoding="utf-8")

    with pytest.raises(HarnessError, match="local dataset changed"):
        ssh_harness._attest_local(plan)


def test_deploy_marks_partial_copy_for_idempotent_retry(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _plan())
    scripts = []
    commands = []
    monkeypatch.setattr(ssh_harness, "_require_program", lambda name: None)
    monkeypatch.setattr(ssh_harness, "_attest_local", lambda plan: None)
    monkeypatch.setattr(
        ssh_harness, "_run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: scripts.append(script),
    )

    ssh_harness.deploy(plan_path)

    assert any("control/deploying.sha256" in script for script in scripts)
    assert any(
        'mv "$RUN/control/deploying.sha256" "$RUN/control/plan.sha256"'
        in script
        for script in scripts
    )
    source_rsync = next(command for command in commands if "--delete" in command)
    assert ".env" in source_rsync and ".env.*" in source_rsync
    assert "compare-report/" in source_rsync


def test_deploy_copies_local_prompts_to_every_host_and_mounts_read_only(
    tmp_path, monkeypatch
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _local_data_plan(prompts)
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    commands = []
    monkeypatch.setattr(ssh_harness, "_require_program", lambda name: None)
    monkeypatch.setattr(ssh_harness, "_attest_local", lambda value: None)
    monkeypatch.setattr(
        ssh_harness, "_run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(ssh_harness, "_ssh", lambda *args, **kwargs: None)

    ssh_harness.deploy(plan_path)

    copies = [command for command in commands if str(prompts.resolve()) in command]
    assert len(copies) == 4
    assert all(command[-1].endswith("/data/dataset.jsonl") for command in copies)


def test_local_prompt_learner_command_omits_hub_revision_and_mounts_read_only(
    tmp_path,
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _local_data_plan(prompts)
    argv = _learner_argv(plan, 0)
    assert "--data-revision" not in argv
    assert parse_learner_args(argv[3:]).data_revision is None
    node = _node_start_script(plan, 0, 0)
    assert '--volume "$RUN/data:/workspace/data:ro"' in node


def test_restart_removes_every_old_island_node_before_starting_ray(
    tmp_path, monkeypatch
):
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _plan())
    calls = []
    monkeypatch.setattr(ssh_harness, "_wait_for_syncer", lambda plan: None)
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: calls.append(("remove", target)),
    )
    monkeypatch.setattr(
        ssh_harness,
        "_start_island",
        lambda plan, learner_id: calls.append(("start", learner_id)),
    )

    ssh_harness.restart_learner(plan_path, 0)

    assert calls == [("remove", "alice@a0"), ("remove", "alice@a1"), ("start", 0)]


def test_kill_learner_stops_workers_before_the_ray_head(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _plan())
    calls = []
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: calls.append((target, script)),
    )

    ssh_harness.kill_learner(plan_path, 0)

    assert [target for target, _ in calls] == ["alice@a1", "alice@a0"]
    assert all("|| true" in script for _, script in calls)


def test_kill_syncer_waits_for_the_old_process_before_restart(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _plan())
    scripts = []
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: scripts.append(script),
    )

    ssh_harness.kill_syncer(plan_path)

    assert "kill -KILL -- -\"$PID\"" in scripts[0]
    assert "kill -0 \"$PID\"" in scripts[0]
    assert "syncer did not exit after SIGKILL" in scripts[0]


def test_syncer_source_is_rehashed_remotely_before_build(monkeypatch):
    scripts = []
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: scripts.append(script),
    )

    ssh_harness._build_syncer(_plan())

    script = scripts[0]
    assert "remote syncer source identity mismatch" in script
    assert "1" * 64 in script
    assert script.index('ACTUAL="$(docker run') < script.index("build --release")
    assert script.index('$HOME/.cargo/bin/cargo') < script.index("command -v cargo")


def test_harness_requires_one_digest_pinned_docker_image():
    assert _docker_ref("docker:" + IMAGE) == IMAGE
    with pytest.raises(HarnessError, match="sha256:DIGEST"):
        _docker_ref("docker:radixark/miles:latest")


def test_target_host_strips_only_the_ssh_user():
    assert _target_host("alice@h200-a.tailnet.ts.net") == "h200-a.tailnet.ts.net"
    assert _target_host("100.64.1.2") == "100.64.1.2"


def test_learner_command_uses_current_multinode_miles_contract():
    plan = _plan()
    argv = _learner_argv(plan, 1)
    assert argv[argv.index("--learner-id") + 1] == "1"
    assert argv[argv.index("--actor-num-nodes") + 1] == "2"
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "4"
    assert argv[argv.index("--completed-groups-path") + 1] == (
        "/workspace/state/island-checkpoint.pt"
    )
    assert argv[argv.index("--audit-dir") + 1] == "/workspace/audit"
    assert "--num-learners" not in argv
    assert "--manifest-sha256" not in argv
    assert "--trust-remote-code" in argv
    assert _container_name(plan, 1, 0) == "yeto-rl-acceptance-i1-n0"


def test_harness_forwards_chat_template_kwargs_to_the_learner():
    plan = _plan()
    plan["learner"]["apply_chat_template_kwargs"] = {
        "enable_thinking": False
    }

    ssh_harness._validate_plan(plan)
    argv = _learner_argv(plan, 0)

    assert json.loads(argv[argv.index("--apply-chat-template-kwargs") + 1]) == {
        "enable_thinking": False
    }
    assert parse_learner_args(argv[3:]).apply_chat_template_kwargs == {
        "enable_thinking": False
    }


def test_syncer_and_node_scripts_use_fixed_roster_and_ray_topology():
    plan = _plan()
    plan["learner"].update(
        {"cybergym_reward_scheme": "shaped_v1", "cybergym_reward_view": "train"}
    )
    syncer = _syncer_argv(plan)
    assert syncer[syncer.index("--learners") + 1] == "2"
    assert syncer[syncer.index("--quorum") + 1] == "2"
    assert syncer[syncer.index("--learner-weight") + 1] == "equal"
    assert "--mark-final-checkpoint" not in syncer

    head = _node_start_script(plan, 0, 0)
    worker = _node_start_script(plan, 0, 1)
    assert "ray start --head" in head
    assert "--include-dashboard=true" in head
    assert "HEAD_IP=$(getent ahostsv4 a0" in head
    assert '--node-ip-address="$HEAD_IP"' in head
    assert "python3 -m yeto.rl.learner" in head
    assert "--gpus '\"device=0,1,2,3\"'" in head
    assert "ray start --address=a0:6379" in worker
    assert "python3 -m yeto.rl.learner" not in worker
    setup = _host_setup_script(plan, 4)
    assert SGLANG_REPOSITORY in setup
    assert SGLANG_COMMIT in setup
    assert '"$RUN/sglang"' in setup
    for script in (head, worker):
        assert "--env CYBERGYM_REWARD_SCHEME=shaped_v1" in script
        assert "--env CYBERGYM_REWARD_VIEW=train" in script
        assert '--volume "$RUN/sglang:/workspace/sglang:ro"' in script
        assert "export PYTHONPATH=/workspace/sglang/python:/workspace/yeto:/workspace/miles" in script
        assert script.index("export PYTHONPATH=") < script.index("ray start")

    plan["remote_env_file"] = ".config/yeto/rl.env"
    with_secrets = _node_start_script(plan, 0, 0)
    assert with_secrets.index("--env-file") < with_secrets.index(
        "--env CYBERGYM_URL"
    )


def test_decoupled_plan_propagates_fragment_and_variance_filter_settings():
    plan = _plan()
    plan["learner"].update(
        {
            "global_rounds": 55,
            "sync_preset": "decoupled",
            "fragments": 8,
            "pipeline": 2,
            "local_horizon": 4,
            "total_fragment_steps": 440,
            "groups_per_round": 4,
            "over_sampling_batch_size": 16,
            "dynamic_sampling_filter_path": (
                "miles.rollout.filter_hub.dynamic_sampling_filters."
                "check_reward_nonzero_std"
            ),
            "dynamic_sampling_max_replacements": 8,
            "rl_offload_train": True,
            "rl_distributed_timeout_minutes": 7,
        }
    )
    ssh_harness._validate_plan(plan)

    syncer = _syncer_argv(plan)
    assert syncer[syncer.index("--pipeline") + 1] == "2"
    assert syncer[syncer.index("--sync-interval-steps") + 1] == "4"
    assert syncer[syncer.index("--total-steps") + 1] == "440"
    assert syncer[syncer.index("--outer-lr") + 1] == "0.7"
    assert syncer[syncer.index("--outer-momentum") + 1] == "0.9"

    learner = _learner_argv(plan, 0)
    assert learner[learner.index("--sync-preset") + 1] == "decoupled"
    assert learner[learner.index("--fragments") + 1] == "8"
    assert learner[learner.index("--pipeline") + 1] == "2"
    assert learner[learner.index("--local-horizon") + 1] == "4"
    assert learner[learner.index("--total-fragment-steps") + 1] == "440"
    assert learner[learner.index("--dynamic-sampling-filter-path") + 1].endswith(
        "check_reward_nonzero_std"
    )
    assert learner[learner.index("--dynamic-sampling-max-replacements") + 1] == "8"
    assert "--rl-offload-train" in learner
    assert learner[learner.index("--rl-distributed-timeout-minutes") + 1] == "7"


def test_verification_requires_an_exited_zero_status_container():
    assert _container_succeeded([{"State": {"Status": "exited", "ExitCode": 0}}])
    assert not _container_succeeded(
        [{"State": {"Status": "running", "ExitCode": 0}}]
    )
    assert not _container_succeeded(
        [{"State": {"Status": "exited", "ExitCode": 1}}]
    )
