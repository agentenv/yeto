import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import yeto.rl as rl_config
from yeto.rl import (
    MILES_COMMIT,
    MILES_IMAGE,
    MILES_PEFT_VERSION,
    MILES_REPOSITORY,
    SGLANG_BUNDLE_PATH,
    SGLANG_BUNDLE_SHA256,
    SGLANG_COMMIT,
    SGLANG_REPOSITORY,
    SGLANG_UPSTREAM_COMMIT,
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
    _parse_islands,
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
    CanonicalTensorSpec("base_model.model.layer.lora_A.weight", (2,), "float32", 2),
)
LAYOUT_HASH = canonical_layout_hash(SPECS)


def _plan():
    return {
        "schema": 2,
        "run_id": "acceptance",
        "remote_run": ".cache/yeto-rl-ssh/acceptance",
        "remote_env_file": None,
        "ssh_options": [],
        "islands": [
            {
                "hosts": ["alice@a0", "alice@a1"],
                "gpus_per_node": 4,
                "accelerator": "H200",
            },
            {
                "hosts": ["alice@b0", "alice@b1"],
                "gpus_per_node": 4,
                "accelerator": "H200",
            },
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
            "upstream_commit": SGLANG_UPSTREAM_COMMIT,
            "commit": SGLANG_COMMIT,
            "bundle_path": SGLANG_BUNDLE_PATH,
            "bundle_sha256": SGLANG_BUNDLE_SHA256,
        },
        "source_sha256": "c" * 64,
        "reward_sha256": "e" * 64,
        "syncer_source_sha256": "1" * 64,
        "learner": {
            "model": "org/model",
            "rollout_model": None,
            "rollout_model_revision": None,
            "rl_model_recipe": "generic",
            "model_mounts": [],
            "model_manifest_sha256": None,
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
            "custom_agent_function_path": None,
            "agent_max_seq_len": None,
            "use_session_server": False,
            "session_server_ip": None,
            "session_server_port": None,
            "tito_model": None,
            "tito_allowed_append_roles": None,
            "tensor_parallel": 1,
            "pipeline_parallel": 1,
            "expert_parallel": 1,
            "rollout_num_gpus_per_engine": 1,
            "sglang_tp_size": None,
            "sglang_dp_size": None,
            "sglang_ep_size": None,
            "sglang_mem_fraction_static": 0.4,
            "sglang_attention_backend": None,
            "sglang_deterministic_inference": True,
            "sglang_page_size": None,
            "sglang_max_running_requests": None,
            "sglang_chunked_prefill_size": None,
            "use_rollout_routing_replay": False,
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


def _secrlenv_daemon_contract(run_id="acceptance"):
    return {
        "source_root": "/data/yeto-rl/src/secrlenv-v24",
        "source_sha256": "2" * 64,
        "task_pack": "/data/yeto-rl/taskpacks/" + "3" * 64,
        "task_pack_sha256": "3" * 64,
        "state_root": f"/data/yeto-rl/secrlenv-runs/{run_id}",
        "bind": "127.0.0.1",
        "port": 28765,
        "operator_image": "secrlenv-operator:pinned",
        "operator_image_id": "sha256:" + "4" * 64,
        "max_active_episodes": 16,
    }


def _enable_secrlenv_agent(plan):
    plan["remote_env_file"] = ".config/yeto/rl.env"
    plan["learner"].update(
        custom_agent_function_path="yeto_miles_secrlenv.agent.run",
        custom_generate_function_path="miles.rollout.generate",
        use_session_server=True,
        tito_model="org/model",
    )


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


def _tms_patch_contract():
    return {
        "repository": ssh_harness.TMS_PRELOAD_PATCH_REPOSITORY,
        "base_commit": ssh_harness.TMS_PRELOAD_PATCH_BASE_COMMIT,
        "patch_commit": ssh_harness.TMS_PRELOAD_PATCH_COMMIT,
        "source_path": (
            "yeto/rl/vendor/torch_memory_saver/"
            "torch_memory_saver_hook_mode_preload_cu13.abi3.so"
        ),
        "container_path": ssh_harness.TMS_PRELOAD_PATCH_CONTAINER_PATH,
        "base_binary_sha256": ssh_harness.TMS_PRELOAD_BASE_BINARY_SHA256,
        "binary_sha256": "2" * 64,
    }


def _tms_disk_backup_contract(plan):
    return ssh_harness._tms_train_disk_backup_contract(
        "/data/yeto-rl/tms-disk-backup",
        plan["run_id"],
        plan["islands"],
        256,
    )


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_round_audit_streams_f32_files_without_path_write_bytes(tmp_path, monkeypatch):
    state = _state(0, [1.0, 2.0])
    original = Path.write_bytes

    def reject_large_write_bytes(path, payload):
        if path.suffix == ".f32":
            raise AssertionError("round audit tensors must be streamed")
        return original(path, payload)

    monkeypatch.setattr(Path, "write_bytes", reject_large_write_bytes)
    _write_round_audit(
        tmp_path,
        learner_id=0,
        target_step=1,
        base=state,
        delta=torch.tensor([0.25, -0.5], dtype=torch.float32),
    )

    assert (tmp_path / "round-00000001.base.f32").read_bytes() == (
        torch.tensor([1.0, 2.0], dtype=torch.float32).numpy().tobytes()
    )
    assert (tmp_path / "round-00000001.delta.f32").read_bytes() == (
        torch.tensor([0.25, -0.5], dtype=torch.float32).numpy().tobytes()
    )


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
            learner_id: (4, 8, token_totals[learner_id]) for learner_id in range(2)
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


def test_oracle_hashes_authoritative_checkpoint_signed_zero_bits(tmp_path):
    plan, checkpoint = _oracle_fixture(tmp_path)
    round_inputs = (
        (1, _state(0, [0.0, 0.0]), ((-0.0, 3.0), (-0.0, 1.0))),
        (2, _state(1, [0.0, 2.0]), ((-0.0, 0.0), (-0.0, 2.0))),
    )
    for step, base, learner_deltas in round_inputs:
        for learner_id, values in enumerate(learner_deltas):
            _write_round_audit(
                tmp_path / f"island-{learner_id}" / "audit",
                learner_id=learner_id,
                target_step=step,
                base=base,
                delta=torch.tensor(values, dtype=torch.float32),
            )

    checkpoint_params = torch.tensor([-0.0, 3.0], dtype=torch.float32)
    checkpoint.fragments[0] = (2, checkpoint_params, torch.zeros(2))
    final = _state(2, [-0.0, 3.0])
    for learner_id in range(2):
        path = tmp_path / f"island-{learner_id}" / "events.jsonl"
        events = [
            json.loads(line) for line in path.read_text().splitlines() if line
        ]
        events.append(
            {
                "event": "rl_policy_apply",
                "policy_version": 2,
                "sync/global_policy_hash": policy_hash(final),
            }
        )
        _write_jsonl(path, events)

    independent = torch.tensor([0.0, 3.0], dtype=torch.float32)
    assert torch.equal(checkpoint_params, independent)
    assert torch.signbit(checkpoint_params[0])
    assert not torch.signbit(independent[0])
    assert _verify_oracle(plan, checkpoint, tmp_path) == policy_hash(final)


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


def test_plan_rejects_previous_pipeline_identity_schema():
    plan = _plan()
    plan["schema"] = 1

    with pytest.raises(HarnessError, match="plan schema"):
        ssh_harness._validate_plan(plan)


def test_miles_and_sglang_pins_include_the_compatible_builds():
    assert MILES_COMMIT == "c252d87f12a2b3b11aa953e4d514a6aceb1a91b5"
    assert not hasattr(rl_config, "MILES_UPSTREAM_COMMIT")
    assert not hasattr(rl_config, "MILES_BUNDLE_PATH")
    assert not hasattr(rl_config, "MILES_BUNDLE_SHA256")
    assert SGLANG_UPSTREAM_COMMIT == "95d4d69665f1712bc6fd3f503af2655b9b301e13"
    assert SGLANG_COMMIT == "e1b57eb8e7749235c987cc6b1b2824ce3265369b"
    assert SGLANG_BUNDLE_SHA256 == (
        "fdb6bd844507a33d870fb28857011c62ab6ec97d96425a020f42aeb0115582f9"
    )
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
    assert (
        plan["learner"]["data_sha256"]
        == hashlib.sha256(prompts.read_bytes()).hexdigest()
    )


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
        jit_cache_root="/data/yeto-rl/jit-cache",
        model_manifest_sha256=None,
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
    assert plan["jit_cache"] == ssh_harness._jit_cache_contract(
        "/data/yeto-rl/jit-cache", plan
    )


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
    monkeypatch.setattr(
        "yeto.provenance.verify_source_tree_sha256", lambda value: value
    )
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
        'mv "$RUN/control/deploying.sha256" "$RUN/control/plan.sha256"' in script
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

    assert 'kill -KILL -- -"$PID"' in scripts[0]
    assert 'kill -0 "$PID"' in scripts[0]
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
    assert script.index("$HOME/.cargo/bin/cargo") < script.index("command -v cargo")


def test_harness_requires_one_digest_pinned_docker_image():
    assert _docker_ref("docker:" + IMAGE) == IMAGE
    with pytest.raises(HarnessError, match="sha256:DIGEST"):
        _docker_ref("docker:radixark/miles:latest")


def test_target_host_strips_only_the_ssh_user():
    assert _target_host("alice@h200-a.tailnet.ts.net") == "h200-a.tailnet.ts.net"
    assert _target_host("100.64.1.2") == "100.64.1.2"


def test_harness_accepts_runtime_sized_h200_rosters():
    islands = _parse_islands(
        ["root@h200-n1,root@h200-n2", "root@h200-n4,root@h200-n5"],
        8,
    )
    assert [island["accelerator"] for island in islands] == ["H200", "H200"]

    one = _plan()
    one["islands"] = [one["islands"][0]]
    ssh_harness._validate_plan(one)
    syncer = _syncer_argv(one)
    assert syncer[syncer.index("--learners") + 1] == "1"
    assert syncer[syncer.index("--quorum") + 1] == "1"
    assert _learner_argv(one, 0)
    with pytest.raises(HarnessError, match="outside the fixed island roster"):
        _learner_argv(one, 1)

    one["learner"].update(
        {
            "sglang_attention_backend": "dsv4",
            "sglang_deterministic_inference": False,
        }
    )
    ssh_harness._validate_plan(one)
    argv = _learner_argv(one, 0)
    assert "--no-sglang-deterministic-inference" in argv

    three = _plan()
    three["islands"].append(
        {"hosts": ["alice@c0", "alice@c1"], "gpus_per_node": 4, "accelerator": "H200"}
    )
    ssh_harness._validate_plan(three)
    syncer = _syncer_argv(three)
    assert syncer[syncer.index("--learners") + 1] == "3"
    assert syncer[syncer.index("--quorum") + 1] == "3"


def test_learner_command_uses_current_multinode_miles_contract():
    plan = _plan()
    plan["learner"]["rollout_model"] = "/data/models/deepseek-v4-flash-fp8"
    plan["learner"]["rollout_model_revision"] = "7" * 40
    argv = _learner_argv(plan, 1)
    assert argv[argv.index("--learner-id") + 1] == "1"
    assert argv[argv.index("--actor-num-nodes") + 1] == "2"
    assert argv[argv.index("--actor-num-gpus-per-node") + 1] == "4"
    assert argv[argv.index("--rollout-model") + 1] == (
        "/data/models/deepseek-v4-flash-fp8"
    )
    assert parse_learner_args(argv[3:]).rollout_model == (
        "/data/models/deepseek-v4-flash-fp8"
    )
    assert parse_learner_args(argv[3:]).rollout_model_revision == "7" * 40
    assert argv[argv.index("--completed-groups-path") + 1] == (
        "/workspace/state/island-checkpoint.pt"
    )
    assert argv[argv.index("--audit-dir") + 1] == "/workspace/audit"
    assert "--num-learners" not in argv
    assert "--manifest-sha256" not in argv
    assert "--trust-remote-code" in argv
    assert _container_name(plan, 1, 0) == "yeto-rl-acceptance-i1-n0"


def test_node_mounts_only_exact_remote_model_roots():
    plan = _plan()
    plan["learner"].update(
        {
            "model": "/data/yeto-rl/models/deepseek-v4-flash-bf16",
            "rollout_model": (
                "/data/hf/models--AtlasCloud--DeepSeek-V4-Flash-0731-FP8-DSpark/"
                "snapshots/7eb21d27aee405755da5251f4458e9fff87c047b"
            ),
            "rollout_model_revision": ("7eb21d27aee405755da5251f4458e9fff87c047b"),
            "model_mounts": [
                "/data/hf/models--AtlasCloud--DeepSeek-V4-Flash-0731-FP8-DSpark",
                "/data/yeto-rl/models/deepseek-v4-flash-bf16",
            ],
            "model_manifest_sha256": "9" * 64,
        }
    )

    ssh_harness._validate_plan(plan)
    script = _node_start_script(plan, 0, 0)

    for mount in plan["learner"]["model_mounts"]:
        assert f"--volume {mount}:{mount}:ro" in script
    assert "9" * 64 in _host_setup_script(plan, 8)
    assert "/conversion_manifest.json" in _host_setup_script(plan, 8)
    assert "--volume /data:/data" not in script
    assert "--volume /data/hf:/data/hf" not in script


def test_offloaded_training_requires_attests_and_mounts_the_tms_fix():
    plan = _plan()
    plan["learner"]["rl_offload_train"] = True

    with pytest.raises(HarnessError, match="TMS preload patch"):
        ssh_harness._validate_plan(plan)

    plan["tms_preload_patch"] = _tms_patch_contract()
    with pytest.raises(HarnessError, match="disk-backup contract"):
        ssh_harness._validate_plan(plan)

    plan["tms_train_disk_backup"] = _tms_disk_backup_contract(plan)
    ssh_harness._validate_plan(plan)
    setup = _host_setup_script(plan, 4)
    start = _node_start_script(plan, 0, 0)
    second = _node_start_script(plan, 0, 1)

    assert plan["tms_preload_patch"]["binary_sha256"] in setup
    assert ssh_harness.TMS_PRELOAD_BASE_BINARY_SHA256 in setup
    assert "docker run --rm --entrypoint sha256sum" in setup
    assert "MILES_TMS_TRAIN_DISK_BACKUP_DIR=/workspace/tms-disk-backup" in start
    assert "MILES_TMS_TRAIN_DISK_BACKUP_CHUNK_MB=256" in start
    assert "acceptance/island-0-node-0" in start
    assert "acceptance/island-0-node-1" in second
    assert '--volume "$TMS_DISK_BACKUP_HOST:/workspace/tms-disk-backup"' in start
    assert "--volume /data:/data" not in start
    assert (
        '"$RUN/source/yeto/rl/vendor/torch_memory_saver/'
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so:"
        "/usr/local/lib/python3.12/dist-packages/"
        'torch_memory_saver_hook_mode_preload_cu13.abi3.so:ro"'
    ) in start

    with pytest.raises(HarnessError, match="dedicated"):
        ssh_harness._tms_train_disk_backup_contract(
            "/data",
            plan["run_id"],
            plan["islands"],
            256,
        )

    plan["tms_preload_patch"]["patch_commit"] = "3" * 40
    with pytest.raises(HarnessError, match="pinned TMS"):
        ssh_harness._validate_plan(plan)


def test_tms_disk_backup_cli_and_non_offload_rejection():
    parsed = ssh_harness.build_parser().parse_args(
        [
            "prepare",
            "--host",
            "alice@a0,alice@a1",
            "--run-id",
            "disk-test",
            "--tms-train-disk-backup-root",
            "/data/yeto-rl/tms-disk-backup",
            "--tms-train-disk-backup-chunk-mb",
            "128",
            "--jit-cache-root",
            "/data/yeto-rl/jit-cache",
        ]
    )
    assert parsed.tms_train_disk_backup_root == (
        "/data/yeto-rl/tms-disk-backup"
    )
    assert parsed.tms_train_disk_backup_chunk_mb == 128
    assert parsed.jit_cache_root == "/data/yeto-rl/jit-cache"

    plan = _plan()
    plan["tms_preload_patch"] = _tms_patch_contract()
    plan["tms_train_disk_backup"] = _tms_disk_backup_contract(plan)
    with pytest.raises(HarnessError, match="requires --rl-offload-train"):
        ssh_harness._validate_plan(plan)


def test_deepseek_v4_nodes_use_the_verified_sglang_environment():
    plan = _plan()
    plan["learner"]["rl_model_recipe"] = "deepseek-v4-flash"

    script = _node_start_script(plan, 0, 0)

    for value in (
        "YETO_DSV4_EXPERT_CLONE=1",
        "YETO_DSV4_CLONE_ONLY_LORA=1",
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK=1",
        "SGLANG_DSV4_FP4_EXPERTS=0",
        "SGLANG_HEALTH_CHECK_TIMEOUT=120",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS=1",
        "SGLANG_OPT_FP8_WO_A_GEMM=0",
        "SGLANG_OPT_FUSE_WQA_WKV=0",
        "NCCL_IB_DISABLE=1",
        "NCCL_SOCKET_IFNAME=eno3",
        "GLOO_SOCKET_IFNAME=eno3",
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "--ulimit nofile=1048576:1048576",
        "--ulimit core=-1",
        "PYTHONFAULTHANDLER=1",
        "TORCH_SHOW_CPP_STACKTRACES=1",
        "TORCH_NCCL_DUMP_ON_TIMEOUT=1",
        "TORCH_NCCL_TRACE_BUFFER_SIZE=1048576",
        "TORCH_FR_BUFFER_SIZE=1048576",
        "NCCL_DEBUG=INFO",
        "NCCL_DEBUG_SUBSYS=INIT,NET",
        "YETO_TMS_POST_PAUSE_IDLE_S=30",
        "cores:/var/lib/vastai_kaalia/data",
    ):
        assert value in script
    assert "NVTE_FLASH_ATTN" not in script


def test_jit_cache_is_future_only_attested_and_narrowly_mounted():
    plan = _plan()
    plan["learner"].update(
        rl_model_recipe="deepseek-v4-flash",
        tensor_parallel=8,
        expert_parallel=8,
        rollout_num_gpus_per_engine=8,
        sglang_tp_size=8,
        sglang_dp_size=1,
        sglang_ep_size=8,
        sglang_deterministic_inference=False,
        lora_targets="attention-routed-experts",
    )
    legacy_setup = _host_setup_script(plan, 4)
    legacy_start = _node_start_script(plan, 0, 0)
    assert "JIT_CACHE_HOST" not in legacy_setup
    assert "JIT_CACHE_HOST" not in legacy_start

    plan["jit_cache"] = ssh_harness._jit_cache_contract(
        "/data/yeto-rl/jit-cache", plan
    )
    ssh_harness._validate_plan(plan)
    setup = _host_setup_script(plan, 4)
    start = _node_start_script(plan, 0, 0)

    assert "--query-gpu=name,compute_cap,driver_version" in setup
    assert plan["jit_cache"]["compatibility_sha256"] in setup
    assert 'test ! -L "$JIT_CACHE_HOST"' in setup
    assert 'chmod 0700 "$JIT_CACHE_HOST"' in setup
    assert '"$RUN/control/jit-cache-host-path"' in setup
    for name, container_path in ssh_harness.JIT_CACHE_MOUNTS:
        assert f'"$JIT_CACHE_HOST/{name}:{container_path}"' in start
    assert '"$JIT_CACHE_HOST/deep-gemm:/tmp/sglang_deep_gemm"' in start
    assert '"$JIT_CACHE_HOST/tvm-ffi:/root/.cache/tvm-ffi"' in start
    assert "SGLANG_DG_CACHE_DIR_PER_PROCESS=1" in start
    assert "--volume /data:/data" not in start
    assert "--volume /root/.cache:/root/.cache" not in start


def test_jit_cache_identity_is_run_independent_and_strictly_validated():
    first = _plan()
    second = _plan()
    second["run_id"] = "next-run"
    second["remote_run"] = ".cache/yeto-rl-ssh/next-run"
    first_contract = ssh_harness._jit_cache_contract(
        "/data/yeto-rl/jit-cache", first
    )
    second_contract = ssh_harness._jit_cache_contract(
        "/data/yeto-rl/jit-cache", second
    )
    assert first_contract == second_contract

    changed_image = _plan()
    changed_image["docker_image"] = "radixark/miles@sha256:" + "a" * 64
    assert ssh_harness._jit_cache_contract(
        "/data/yeto-rl/jit-cache", changed_image
    ) != first_contract

    with pytest.raises(HarnessError, match="dedicated"):
        ssh_harness._jit_cache_contract("/data/yeto-rl/other", first)
    with pytest.raises(HarnessError, match="dedicated"):
        ssh_harness._jit_cache_contract(
            "/data/yeto-rl/jit-cache/../escape", first
        )

    first["jit_cache"] = dict(first_contract)
    first["jit_cache"]["compatibility_sha256"] = "0" * 64
    with pytest.raises(HarnessError, match="compatibility identity"):
        ssh_harness._validate_plan(first)

    first["jit_cache"] = {**first_contract, "unexpected": True}
    with pytest.raises(HarnessError, match="invalid JIT cache contract"):
        ssh_harness._validate_plan(first)


def test_node_network_interface_is_attested_and_propagated():
    plan = _plan()
    plan["network_interface"] = "tailscale0"

    ssh_harness._validate_plan(plan)
    script = _node_start_script(plan, 0, 0)

    assert "NCCL_SOCKET_IFNAME=tailscale0" in script
    assert "GLOO_SOCKET_IFNAME=tailscale0" in script
    assert "NETWORK_INTERFACE=tailscale0" in script
    assert "fcntl.ioctl" in script
    assert "sys.argv[1].encode()[:15]" in script
    assert '"$NETWORK_INTERFACE"' in script
    assert "ip -4 -o addr" not in script
    assert '--node-ip-address="$NODE_IP"' in script
    assert 'export MILES_HOST_IP="$NODE_IP"' in script

    plan["network_interface"] = "bad interface"
    with pytest.raises(HarnessError, match="invalid network interface"):
        ssh_harness._validate_plan(plan)


def test_expert_full_plan_forwards_attestation_and_runtime_environment():
    plan = _plan()
    plan["islands"] = [
        {
            "hosts": ["root@h200-n1", "root@h200-n2"],
            "gpus_per_node": 8,
            "accelerator": "H200",
        }
    ]
    plan["learner"].update(
        {
            "rl_model_recipe": "deepseek-v4-flash",
            "global_rounds": 1,
            "groups_per_round": 1,
            "samples_per_group": 2,
            "over_sampling_batch_size": 1,
            "lora_r": 8,
            "lora_targets": "attention",
            "inner_lr": 1e-5,
            "expert_full_count": 16,
            "expert_full_lr": 1e-6,
            "expert_selection_sha256": "a" * 64,
            "expert_selection_contract_sha256": "b" * 64,
            "tensor_parallel": 8,
            "pipeline_parallel": 2,
            "expert_parallel": 8,
            "rollout_num_gpus_per_engine": 8,
            "sglang_tp_size": 8,
            "sglang_dp_size": 1,
            "sglang_ep_size": 8,
            "sglang_attention_backend": "dsv4",
            "sglang_deterministic_inference": False,
            "sglang_page_size": 256,
        }
    )

    ssh_harness._validate_plan(plan)
    argv = _learner_argv(plan, 0)
    parsed = parse_learner_args(argv[3:])
    script = _node_start_script(plan, 0, 0)

    assert parsed.expert_full_count == 16
    assert parsed.expert_full_lr == 1e-6
    assert parsed.pipeline_parallel == 2
    assert parsed.expert_selection_sha256 == "a" * 64
    assert parsed.expert_selection_contract_sha256 == "b" * 64
    for value in (
        "YETO_DSV4_EXPERT_CLONE=1",
        "YETO_DSV4_EXPERT_FULL=1",
        "YETO_DSV4_EXPERT_FULL_COUNT=16",
        "YETO_DSV4_EXPERT_FULL_LR=1e-06",
        "NVTE_GROUPED_LINEAR_SINGLE_PARAM=0",
        "--shm-size 64g",
    ):
        assert value in script
    assert "YETO_DSV4_CLONE_ONLY_LORA" not in script


def test_harness_forwards_chat_template_kwargs_to_the_learner():
    plan = _plan()
    plan["learner"]["apply_chat_template_kwargs"] = {"enable_thinking": False}

    ssh_harness._validate_plan(plan)
    argv = _learner_argv(plan, 0)

    assert json.loads(argv[argv.index("--apply-chat-template-kwargs") + 1]) == {
        "enable_thinking": False
    }
    assert parse_learner_args(argv[3:]).apply_chat_template_kwargs == {
        "enable_thinking": False
    }


def test_harness_forwards_agentic_session_contract_to_the_learner():
    plan = _plan()
    plan["learner"].update(
        {
            "custom_generate_function_path": (
                "miles.rollout.generate_hub.agentic_tool_call.generate"
            ),
            "custom_agent_function_path": "secrlenv_miles.agent.run",
            "agent_max_seq_len": 384,
            "use_session_server": True,
            "session_server_port": [31000],
            "tito_model": "deepseekv4",
            "tito_allowed_append_roles": ["tool", "user"],
        }
    )

    ssh_harness._validate_plan(plan)
    argv = _learner_argv(plan, 0)
    parsed = parse_learner_args(argv[3:])

    assert parsed.custom_agent_function_path == "secrlenv_miles.agent.run"
    assert parsed.agent_max_seq_len == 384
    assert parsed.tito_model == "deepseekv4"
    assert parsed.tito_allowed_append_roles == ["tool", "user"]
    for node_id in (0, 1):
        script = _node_start_script(plan, 0, node_id)
        assert "--env MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1" in script


def test_harness_forwards_deepseek_v4_recipe_to_the_learner():
    plan = _plan()
    plan["learner"]["rl_model_recipe"] = "deepseek-v4-flash"

    argv = _learner_argv(plan, 0)

    assert parse_learner_args(argv[3:]).rl_model_recipe == "deepseek-v4-flash"


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
    assert 'HEAD_IP="$NODE_IP"' in head
    assert '--node-ip-address="$NODE_IP"' in head
    assert "python3 -m yeto.rl.learner" in head
    assert "--gpus '\"device=0,1,2,3\"'" in head
    assert "ray start --address=a0:6379" in worker
    assert '--node-ip-address="$NODE_IP"' in worker
    assert "python3 -m yeto.rl.learner" not in worker
    setup = _host_setup_script(plan, 4)
    assert SGLANG_REPOSITORY in setup
    assert SGLANG_COMMIT in setup
    assert f'git -C "$RUN/miles" remote set-url origin {MILES_REPOSITORY}' in setup
    assert (
        f'git -C "$RUN/miles" fetch --depth 1 origin {MILES_COMMIT}'
        in setup
    )
    assert "MILES_BUNDLE" not in setup
    assert f'git -C "$RUN/sglang" remote set-url origin {SGLANG_REPOSITORY}' in setup
    assert '"$RUN/sglang"' in setup
    assert SGLANG_BUNDLE_PATH in setup
    assert SGLANG_BUNDLE_SHA256 in setup
    assert SGLANG_UPSTREAM_COMMIT in setup
    assert 'git -C "$RUN/sglang" fetch "$SGLANG_BUNDLE" HEAD' in setup
    assert setup.count("docker image inspect") == 2
    assert setup.index("docker image inspect") < setup.index("docker pull")
    for script in (head, worker):
        assert "--env CYBERGYM_REWARD_SCHEME=shaped_v1" in script
        assert "--env CYBERGYM_REWARD_VIEW=train" in script
        assert '--volume "$RUN/sglang:/workspace/sglang:ro"' in script
        assert (
            "export PYTHONPATH=/workspace/sglang/python:/workspace/yeto:/workspace/miles"
            in script
        )
        assert script.index("export PYTHONPATH=") < script.index("ray start")

    plan["remote_env_file"] = ".config/yeto/rl.env"
    with_secrets = _node_start_script(plan, 0, 0)
    assert with_secrets.index("--env-file") < with_secrets.index("--env CYBERGYM_URL")


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
    plan["tms_preload_patch"] = _tms_patch_contract()
    plan["tms_train_disk_backup"] = _tms_disk_backup_contract(plan)
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


def test_secrlenv_daemon_contract_is_required_and_propagated_to_nodes():
    plan = _plan()
    _enable_secrlenv_agent(plan)

    with pytest.raises(HarnessError, match="pinned daemon contract"):
        ssh_harness._validate_plan(plan)

    plan["secrlenv_daemon"] = _secrlenv_daemon_contract()
    script = ssh_harness._secrlenv_daemon_script(plan)
    node = _node_start_script(plan, 0, 0)

    assert "python3 -m secrlenv_rl.server" in script
    assert "--enable-dind-debug" not in script
    assert "task_pack_sha256" in script
    assert "python3 -m yeto.rl.secrlenv_task_images" in script
    assert "secrlenv_task_images=ready" not in script
    assert script.index('test "$SOURCE_SHA"') < script.index(
        "python3 -m yeto.rl.secrlenv_task_images"
    ) < script.index("python3 -m secrlenv_rl.server")
    assert "secrlenv_daemon=ready" in script
    assert 'ENV_FILE="$HOME/.config/yeto/rl.env"' in script
    assert 'TOKEN_FILE="$STATE_ROOT/daemon.token"' in script
    assert "secrets.token_hex(32)" in script
    assert '--token-file "$TOKEN_FILE"' in script
    assert "--env SECRLENV_DAEMON_URL=http://127.0.0.1:28765" in node
    assert ("--env SECRLENV_TASK_PACK_SHA256=" + "3" * 64) in node
    assert (
        "--env SECRLENV_BEARER_TOKEN_FILE=/run/secrlenv/daemon.token" in node
    )
    assert (
        "/data/yeto-rl/secrlenv-runs/acceptance/daemon.token:"
        "/run/secrlenv/daemon.token:ro" in node
    )


def test_start_attests_secrlenv_daemons_before_host_gpu_setup(tmp_path, monkeypatch):
    plan = _plan()
    _enable_secrlenv_agent(plan)
    plan["secrlenv_daemon"] = _secrlenv_daemon_contract()
    path = _write_plan(tmp_path / "plan.json", plan)
    events = []

    monkeypatch.setattr(ssh_harness, "deploy", lambda _path: events.append("deploy"))
    monkeypatch.setattr(
        ssh_harness,
        "_start_secrlenv_daemons",
        lambda _plan: events.append("daemon"),
    )
    monkeypatch.setattr(ssh_harness, "_host_setup_script", lambda *_args: "setup")
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda *_args, **_kwargs: events.append("host_setup"),
    )
    monkeypatch.setattr(ssh_harness, "_build_syncer", lambda _plan: None)
    monkeypatch.setattr(ssh_harness, "_start_syncer", lambda _plan: None)
    monkeypatch.setattr(ssh_harness, "_wait_for_syncer", lambda _plan: None)
    monkeypatch.setattr(ssh_harness, "_start_island", lambda *_args: None)

    ssh_harness.start(path)

    assert events[:3] == ["deploy", "daemon", "host_setup"]


def test_verification_requires_an_exited_zero_status_container():
    assert _container_succeeded([{"State": {"Status": "exited", "ExitCode": 0}}])
    assert not _container_succeeded([{"State": {"Status": "running", "ExitCode": 0}}])
    assert not _container_succeeded([{"State": {"Status": "exited", "ExitCode": 1}}])
