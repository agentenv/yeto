import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import yeto.rl as rl_config
from yeto.provenance import python_spec_sha256
from yeto.rl import (
    MILES_BASE_COMMIT,
    MILES_BUNDLE_PATH,
    MILES_BUNDLE_SHA256,
    MILES_COMMIT,
    MILES_IMAGE,
    MILES_PEFT_VERSION,
    MILES_REPOSITORY,
    SGLANG_COMMIT,
    SGLANG_REPOSITORY,
    ssh_harness,
)
from yeto.rl.bridge import _write_round_audit
from yeto.rl.codex_backend import (
    QWEN35_08B_MODEL,
    QWEN35_08B_REVISION,
    QWEN38_MODEL,
    QWEN38_REVISION,
    stock_codex_backend_contract,
)
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
            "base_commit": MILES_BASE_COMMIT,
            "commit": MILES_COMMIT,
            "bundle_path": MILES_BUNDLE_PATH,
            "bundle_sha256": MILES_BUNDLE_SHA256,
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
        "enable_dind_debug": False,
        "dind_image": None,
        "dind_image_id": None,
        "dind_debug_script_sha256": None,
    }


def _enable_secrlenv_agent(plan):
    plan["remote_env_file"] = ".config/yeto/rl.env"
    plan["learner"].update(
        custom_agent_function_path="yeto_miles_secrlenv.agent.run",
        custom_generate_function_path=ssh_harness.SECRLENV_GENERATE,
        use_session_server=True,
        tito_model="org/model",
        reward_function=ssh_harness.SECRLENV_REWARD,
        dynamic_sampling_filter_path=ssh_harness.SECRLENV_GROUP_FILTER,
        dynamic_sampling_max_replacements=0,
        secrlenv_max_infrastructure_replacements=1,
        over_sampling_batch_size=plan["learner"]["groups_per_round"] + 1,
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


def test_secrlenv_plan_round_trip_requires_exact_replacement_contract():
    plan = _plan()
    _enable_secrlenv_agent(plan)
    plan["secrlenv_daemon"] = _secrlenv_daemon_contract(plan["run_id"])

    ssh_harness._validate_plan(plan)
    parsed = parse_learner_args(_learner_argv(plan, 0)[3:])

    assert parsed.dynamic_sampling_filter_path == ssh_harness.SECRLENV_GROUP_FILTER
    assert parsed.dynamic_sampling_max_replacements == 0
    assert parsed.secrlenv_max_infrastructure_replacements == 1
    assert parsed.over_sampling_batch_size == parsed.groups_per_round + 1

    mutations = [
        (
            "custom_generate_function_path",
            "miles.rollout.generate_hub.agentic_tool_call.generate",
        ),
        ("reward_function", "other.reward:score"),
        ("dynamic_sampling_filter_path", None),
        ("dynamic_sampling_filter_path", "other.filter"),
        ("dynamic_sampling_max_replacements", None),
        ("dynamic_sampling_max_replacements", False),
        ("dynamic_sampling_max_replacements", 1),
        ("secrlenv_max_infrastructure_replacements", None),
        ("secrlenv_max_infrastructure_replacements", True),
        ("secrlenv_max_infrastructure_replacements", 2),
        ("over_sampling_batch_size", plan["learner"]["groups_per_round"]),
    ]
    for name, value in mutations:
        invalid = copy.deepcopy(plan)
        invalid["learner"][name] = value
        with pytest.raises(ssh_harness.HarnessError):
            ssh_harness._validate_plan(invalid)


def _secrlenv_eval_plan(path: Path):
    plan = _local_data_plan(path)
    _enable_secrlenv_agent(plan)
    plan["learner"]["reward_function"] = ssh_harness.SECRLENV_REWARD
    plan["reward_sha256"] = python_spec_sha256(ssh_harness.SECRLENV_REWARD)
    plan["secrlenv_daemon"] = _secrlenv_daemon_contract(plan["run_id"])
    plan["final_ack_timeout_s"] = 21600
    plan["eval_checkpoint"] = ssh_harness._eval_checkpoint_contract(
        "root@h200-n6",
        "yeto-rl-ssh-data/train-run/state/state.ckpt",
        "f" * 64,
        138_632_000_000,
        plan["learner"]["global_rounds"],
    )
    plan["learner"]["evaluation"] = {
        "eval_only": True,
        "dataset_name": "flaky100",
        "data": plan["learner"]["data"],
        "data_sha256": plan["learner"]["data_sha256"],
        "interval": 1,
        "samples_per_prompt": 2,
        "prompt_count": 1,
        "skip_before_train": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_prompt_len": 256,
        "max_response_len": 256,
        "max_context_len": 512,
    }
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
        events = [json.loads(line) for line in path.read_text().splitlines() if line]
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


def test_decoupled_oracle_prefers_accounted_sweep_progress(tmp_path):
    plan, checkpoint, final_hash = _decoupled_oracle_fixture(tmp_path)
    event_path = tmp_path / "syncer" / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    accounted = {0: [0, 0], 1: [0, 0]}
    for event in events:
        complete = event["step"] % plan["learner"]["fragments"] == 0
        for responder in event["responders"]:
            responder["accounted_c_steps"] = responder["c_steps"] if complete else 0
            responder["accounted_c_tokens"] = responder["c_tokens"] if complete else 0
            accounted[responder["id"]][0] += responder["accounted_c_steps"]
            accounted[responder["id"]][1] += responder["accounted_c_tokens"]
    _write_jsonl(event_path, events)
    checkpoint.ledger = {
        learner_id: (4, steps, tokens)
        for learner_id, (steps, tokens) in accounted.items()
    }

    assert ssh_harness._verify_decoupled(plan, checkpoint, tmp_path) == final_hash


def test_decoupled_oracle_reconciles_missing_merge_from_final_ledger_snapshot(
    tmp_path,
):
    plan, checkpoint, final_hash = _decoupled_oracle_fixture(tmp_path)
    event_path = tmp_path / "syncer" / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    # Simulate the exact crash window: step 2 reached the durable checkpoint,
    # but its ordinary diagnostic append never landed.
    events = [event for event in events if event.get("step") != 2]
    events.append(
        {
            "event": "policy_sweep_ledger",
            "event_id": (f"policy-sweep-ledger:{checkpoint.layout_hash}:complete:4"),
            "phase": "complete",
            "protocol_version": 4,
            "sync/layout_hash": checkpoint.layout_hash,
            "global_step": 4,
            "policy_round": 2,
            "sweep_fragments": 2,
            "sweep_complete": True,
            "versions": [3, 4],
            "ledger": [
                {
                    "id": learner_id,
                    "merges": merges,
                    "steps": steps,
                    "tokens": tokens,
                }
                for learner_id, (merges, steps, tokens) in checkpoint.ledger.items()
            ],
        }
    )
    # A short write may leave an invalid diagnostic tail. The Rust syncer
    # terminates that line before appending the durable snapshot.
    ordinary, snapshot = events[:-1], events[-1]
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in ordinary)
        + '{"step":2,"responders":'
        + "\n"
        + json.dumps(snapshot)
        + "\n"
    )

    assert ssh_harness._verify_decoupled(plan, checkpoint, tmp_path) == final_hash


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
                json.dumps(
                    {
                        "Status": "exited",
                        "ExitCode": 0,
                        "OOMKilled": False,
                        "RestartCount": 0,
                    }
                )
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


def test_prepare_cli_exposes_final_secrlenv_eval_controls():
    args = ssh_harness.build_parser().parse_args(
        [
            "prepare",
            "--host",
            "root@h200-n1,root@h200-n2",
            "--run-id",
            "flaky100",
            "--final-ack-timeout-s",
            "21600",
            "--eval-checkpoint-host",
            "root@h200-n6",
            "--eval-checkpoint-path",
            "yeto-rl-ssh-data/train/state/state.ckpt",
            "--eval-checkpoint-sha256",
            "f" * 64,
            "--eval-checkpoint-size-bytes",
            "138632000000",
            "--eval-checkpoint-global-step",
            "8",
            "--eval-dataset-name",
            "flaky100",
            "--eval-interval",
            "1",
            "--eval-samples-per-prompt",
            "2",
            "--eval-temperature",
            "0.7",
            "--eval-top-p",
            "0.95",
            "--eval-max-prompt-len",
            "4096",
            "--eval-max-response-len",
            "32768",
            "--eval-max-context-len",
            "32768",
        ]
    )

    assert args.final_ack_timeout_s == 21600
    assert args.eval_checkpoint_host == "root@h200-n6"
    assert args.eval_checkpoint_path.endswith("train/state/state.ckpt")
    assert args.eval_checkpoint_sha256 == "f" * 64
    assert args.eval_checkpoint_size_bytes == 138_632_000_000
    assert args.eval_checkpoint_global_step == 8
    assert args.eval_dataset_name == "flaky100"
    assert args.eval_interval == 1
    assert args.eval_samples_per_prompt == 2
    assert args.eval_temperature == 0.7
    assert args.eval_top_p == 0.95
    assert args.eval_max_prompt_len == 4096
    assert args.eval_max_response_len == 32768
    assert args.eval_max_context_len == 32768


def test_miles_and_sglang_pins_include_the_compatible_builds():
    assert MILES_BASE_COMMIT == "6062afe0a9d5d6471e8395dedc81c78dd9f4a84f"
    assert MILES_COMMIT == "e2ad83d84a6b32a0f7d79ff196ad8c64fc67586a"
    assert not hasattr(rl_config, "MILES_UPSTREAM_COMMIT")
    assert MILES_BUNDLE_PATH == "yeto/rl/vendor/miles-qwen38.bundle"
    assert MILES_BUNDLE_SHA256 == (
        "ff0e2ed7de75e06926c4637545a8a918c2e3ae1d5ceeb2b49d6b87511c0598b2"
    )
    assert SGLANG_COMMIT == "e1b57eb8e7749235c987cc6b1b2824ce3265369b"
    assert not hasattr(rl_config, "SGLANG_UPSTREAM_COMMIT")
    assert not hasattr(rl_config, "SGLANG_BUNDLE_PATH")
    assert not hasattr(rl_config, "SGLANG_BUNDLE_SHA256")
    assert MILES_IMAGE == (
        "docker:ghcr.io/agentenv/miles@sha256:"
        "80c20538b63f76defde06ad5d4cfa564ae6f261110696eb1864470cb835e1590"
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


def test_secrlenv_eval_reuses_the_attested_training_dataset_and_extends_final_ack(
    tmp_path,
):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)

    ssh_harness._validate_plan(plan)
    evaluation = plan["learner"]["evaluation"]
    assert evaluation["data"] == plan["learner"]["data"]
    assert evaluation["data_sha256"] == plan["learner"]["data_sha256"]
    assert evaluation["eval_only"] is True
    assert evaluation["interval"] == 1
    assert evaluation["prompt_count"] == 1
    assert plan["learner"]["reward_function"] == ssh_harness.SECRLENV_REWARD
    assert plan["reward_sha256"] == python_spec_sha256(ssh_harness.SECRLENV_REWARD)
    assert plan["eval_checkpoint"]["global_step"] == plan["learner"]["global_rounds"]
    syncer = _syncer_argv(plan)
    assert syncer[syncer.index("--final-ack-timeout-s") + 1] == "21600"

    argv = _learner_argv(plan, 0)
    parsed = parse_learner_args(argv[3:])
    assert parsed.eval_only is True
    assert parsed.eval_dataset_name == "flaky100"
    assert parsed.eval_data_sha256 == plan["learner"]["data_sha256"]
    assert parsed.eval_interval == 1
    assert parsed.eval_samples_per_prompt == 2
    assert parsed.reward_function == ssh_harness.SECRLENV_REWARD
    assert parsed.eval_temperature == 0.7
    assert parsed.eval_top_p == 0.95
    assert parsed.eval_max_prompt_len == 256
    assert parsed.eval_max_response_len == 256
    assert parsed.eval_max_context_len == 512


@pytest.mark.parametrize("field", ["data", "data_sha256"])
def test_secrlenv_eval_rejects_a_distinct_training_dataset_identity(tmp_path, field):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)
    plan["learner"]["evaluation"][field] = (
        "/workspace/data/other.jsonl" if field == "data" else "0" * 64
    )

    with pytest.raises(HarnessError, match="exact training dataset"):
        ssh_harness._validate_plan(plan)


def test_secrlenv_eval_requires_an_explicitly_extended_final_ack_timeout(tmp_path):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)
    plan.pop("final_ack_timeout_s")

    with pytest.raises(HarnessError, match="extended final_ack_timeout"):
        ssh_harness._validate_plan(plan)


def test_secrlenv_eval_rejects_a_different_reward_contract(tmp_path):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)
    plan["learner"]["reward_function"] = "pkg.reward:score"

    with pytest.raises(HarnessError, match="signed secrlenv reward"):
        ssh_harness._validate_plan(plan)


def test_secrlenv_eval_checkpoint_step_must_match_terminal_training_step(tmp_path):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)
    plan["eval_checkpoint"]["global_step"] -= 1

    with pytest.raises(HarnessError, match="eval checkpoint identity"):
        ssh_harness._validate_plan(plan)


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
        syncer_address="100.64.0.6:29400",
        syncer_host="root@syncer0",
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
    assert plan["syncer_address"] == "100.64.0.6:29400"
    assert plan["syncer_host"] == "root@syncer0"
    assert "evaluation" not in plan["learner"]
    assert "eval_checkpoint" not in plan
    assert "final_ack_timeout_s" not in plan
    assert plan["jit_cache"] == ssh_harness._jit_cache_contract(
        "/data/yeto-rl/jit-cache", plan
    )


def test_prepare_wires_final_secrlenv_eval_and_same_dataset_sha(tmp_path, monkeypatch):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    learner = _plan()["learner"]
    learner.update(
        custom_agent_function_path="yeto_miles_secrlenv.agent.run",
        custom_generate_function_path=ssh_harness.SECRLENV_GENERATE,
        use_session_server=True,
        tito_model="org/model",
        reward_function=ssh_harness.SECRLENV_REWARD,
        dynamic_sampling_filter_path=ssh_harness.SECRLENV_GROUP_FILTER,
        dynamic_sampling_max_replacements=0,
        secrlenv_max_infrastructure_replacements=1,
        over_sampling_batch_size=learner["groups_per_round"] + 1,
    )
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
            "reward_sha256": python_spec_sha256(ssh_harness.SECRLENV_REWARD),
            "_provenance": {"dataset": {"source": "local"}},
        }
    )
    daemon = _secrlenv_daemon_contract("final-eval")
    checkpoint_sha256 = "f" * 64
    namespace = SimpleNamespace(
        run_id="final-eval",
        host=["alice@a0", "alice@b0"],
        gpus_per_node=1,
        remote_root=".cache/yeto-rl-ssh",
        remote_env_file=".config/yeto/rl.env",
        model_manifest_sha256=None,
        syncer_address="100.64.0.6:29400",
        syncer_host="root@syncer0",
        output_dir=tmp_path / "run",
        ssh_option=[],
        final_ack_timeout_s=21600,
        eval_checkpoint_host="root@h200-n6",
        eval_checkpoint_path="yeto-rl-ssh-data/train/state/state.ckpt",
        eval_checkpoint_sha256=checkpoint_sha256,
        eval_checkpoint_size_bytes=138_632_000_000,
        eval_checkpoint_global_step=learner["global_rounds"],
        eval_dataset_name="flaky100",
        eval_interval=1,
        eval_samples_per_prompt=2,
        eval_temperature=0.7,
        eval_top_p=0.95,
        eval_max_prompt_len=256,
        eval_max_response_len=256,
        eval_max_context_len=512,
        secrlenv_source_root=daemon["source_root"],
        secrlenv_source_sha256=daemon["source_sha256"],
        secrlenv_task_pack=daemon["task_pack"],
        secrlenv_task_pack_sha256=daemon["task_pack_sha256"],
        secrlenv_operator_image=daemon["operator_image"],
        secrlenv_operator_image_id=daemon["operator_image_id"],
        secrlenv_port=daemon["port"],
        secrlenv_max_active_episodes=daemon["max_active_episodes"],
        secrlenv_enable_dind_debug=False,
        secrlenv_dind_image=None,
        secrlenv_dind_image_id=None,
        secrlenv_dind_debug_script_sha256=None,
    )
    monkeypatch.setattr(ssh_harness, "_resolved_launch_args", lambda *unused: args)
    monkeypatch.setattr(ssh_harness, "_syncer_source_sha256", lambda: "1" * 64)

    plan_path = ssh_harness.prepare(namespace)
    _, plan = load_plan(plan_path)
    evaluation = plan["learner"]["evaluation"]

    assert evaluation["data"] == plan["learner"]["data"]
    assert evaluation["data_sha256"] == plan["learner"]["data_sha256"]
    assert evaluation["eval_only"] is True
    assert evaluation["interval"] == 1
    assert evaluation["prompt_count"] == 1
    assert evaluation["skip_before_train"] is True
    assert plan["final_ack_timeout_s"] == 21600
    assert plan["eval_checkpoint"]["source_host"] == "root@h200-n6"
    assert plan["eval_checkpoint"]["source_path"].endswith("train/state/state.ckpt")
    assert plan["eval_checkpoint"]["sha256"] == checkpoint_sha256
    assert plan["eval_checkpoint"]["global_step"] == learner["global_rounds"]
    assert plan["secrlenv_daemon"]["placement"] == "island-heads"


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


def test_deploy_includes_a_dedicated_syncer_host(tmp_path, monkeypatch):
    plan = _plan()
    plan["syncer_host"] = "root@syncer0"
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    targets = []
    commands = []
    monkeypatch.setattr(ssh_harness, "_require_program", lambda name: None)
    monkeypatch.setattr(ssh_harness, "_attest_local", lambda value: None)
    monkeypatch.setattr(
        ssh_harness, "_run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: targets.append(target),
    )

    ssh_harness.deploy(plan_path)

    assert set(targets) == {
        "alice@a0",
        "alice@a1",
        "alice@b0",
        "alice@b1",
        "root@syncer0",
    }
    source_copies = [command for command in commands if "--delete" in command]
    assert len(source_copies) == 5
    assert any(command[-1].startswith("root@syncer0:") for command in source_copies)


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


def test_eval_deploy_copies_terminal_checkpoint_only_to_fresh_syncer(
    tmp_path, monkeypatch
):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)
    plan["syncer_host"] = "root@syncer0"
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    commands = []
    scripts = []
    monkeypatch.setattr(ssh_harness, "_require_program", lambda name: None)
    monkeypatch.setattr(ssh_harness, "_attest_local", lambda value: None)
    monkeypatch.setattr(
        ssh_harness, "_run", lambda command, **kwargs: commands.append(command)
    )
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda _plan, target, script, **kwargs: scripts.append((target, script)),
    )

    ssh_harness.deploy(plan_path)

    checkpoint_copies = [
        command for command in commands if command[:2] == ["scp", "-3"]
    ]
    assert len(checkpoint_copies) == 1
    assert checkpoint_copies[0][-2] == (
        "root@h200-n6:yeto-rl-ssh-data/train-run/state/state.ckpt"
    )
    assert checkpoint_copies[0][-1].endswith(
        "root@syncer0:.cache/yeto-rl-ssh/acceptance/state/state.ckpt"
    )
    source_attestation = next(
        script for target, script in scripts if target == "root@h200-n6"
    )
    assert plan["eval_checkpoint"]["sha256"] in source_attestation
    assert str(plan["eval_checkpoint"]["size_bytes"]) in source_attestation
    assert "syncer checkpoint global step mismatch" in source_attestation
    syncer_script = ssh_harness._syncer_start_script(plan)
    assert plan["eval_checkpoint"]["sha256"] in syncer_script
    assert "sha256sum --check -" in syncer_script


def test_eval_collect_whitelists_small_evidence_and_excludes_checkpoint(
    tmp_path, monkeypatch
):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)
    plan["islands"] = [plan["islands"][0]]
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    commands = []
    ssh_scripts = []

    def ssh(_plan, _target, script, **_kwargs):
        ssh_scripts.append(script)
        if script.startswith("docker inspect"):
            stdout = json.dumps(
                {
                    "Status": "exited",
                    "ExitCode": 0,
                    "OOMKilled": False,
                    "RestartCount": 0,
                }
            )
        elif "final_ack_count" in script:
            stdout = json.dumps({"schema": 1, "resume_count": 1, "final_ack_count": 1})
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(ssh_harness, "_require_program", lambda _name: None)
    monkeypatch.setattr(ssh_harness, "_ssh", ssh)
    monkeypatch.setattr(
        ssh_harness, "_run", lambda command, **_kwargs: commands.append(command)
    )

    ssh_harness.collect(plan_path)

    remote_sources = [command[-2] for command in commands]
    assert not any("/island-0/state/" in source for source in remote_sources)
    assert not any("/island-0/audit/" in source for source in remote_sources)
    assert not any("/island-0/output/" in source for source in remote_sources)
    learner_filters = [script for script in ssh_scripts if "eval-learner" in script]
    assert len(learner_filters) == 1
    syncer = next(command for command in commands if command[-2].endswith("/state/"))
    assert "--include=/syncer.exit" in syncer
    assert "--include=/syncer.log" not in syncer
    assert "--include=/events.jsonl" not in syncer
    assert "--exclude=*" in syncer
    assert not any("state.ckpt" in value for value in syncer)
    assert not any("docker logs" in script for script in ssh_scripts)
    inspect_scripts = [
        script for script in ssh_scripts if script.startswith("docker inspect")
    ]
    assert inspect_scripts
    assert all("--format" in script for script in inspect_scripts)
    for inspection in (tmp_path / "artifacts").glob("island-*/node-*.inspect.json"):
        assert set(json.loads(inspection.read_text())) == {
            "Status",
            "ExitCode",
            "OOMKilled",
            "RestartCount",
        }
    lifecycle = json.loads(
        (tmp_path / "artifacts/syncer/lifecycle-evidence.json").read_text()
    )
    assert lifecycle == {"schema": 1, "resume_count": 1, "final_ack_count": 1}
    assert not list((tmp_path / "artifacts").glob("island-*/state"))


def test_eval_verify_uses_scalar_final_ack_evidence_without_checkpoint(
    tmp_path, capsys
):
    prompts = tmp_path / "flaky100.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _secrlenv_eval_plan(prompts)
    plan["islands"] = [plan["islands"][0]]
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    artifacts = tmp_path / "artifacts"
    island = artifacts / "island-0"
    output = island / "output"
    output.mkdir(parents=True)
    for node_id in range(len(plan["islands"][0]["hosts"])):
        (island / f"node-{node_id}.inspect.json").write_text(
            json.dumps(
                {
                    "Status": "exited",
                    "ExitCode": 0,
                    "OOMKilled": False,
                    "RestartCount": 0,
                }
            )
        )
    _write_jsonl(
        output / "events.jsonl",
        [
            {
                "event": "rl_policy_apply",
                "island_id": 0,
                "policy_version": plan["learner"]["global_rounds"],
                "sync/global_policy_hash": "a" * 64,
            },
            {
                "event": "rl_eval_result",
                "island_id": 0,
                "rollout_id": 0,
                "policy_version": plan["learner"]["global_rounds"],
                "dataset_name": "flaky100",
                "sample_count": 2,
                "rl/eval/result": 0.5,
                "rl/eval/pass_at_1": 0.5,
            },
        ],
    )
    syncer = artifacts / "syncer"
    syncer.mkdir()
    (syncer / "syncer.exit").write_text(
        "result=success\n"
        "exit_code=exited\n"
        "exit_status=0\n"
        "timestamp=2026-08-17T00:00:00Z\n"
    )
    (syncer / "lifecycle-evidence.json").write_text(
        json.dumps({"schema": 1, "resume_count": 1, "final_ack_count": 1})
    )

    ssh_harness.verify(plan_path)

    assert "verified terminal eval flaky100" in capsys.readouterr().out
    assert not (syncer / "state.ckpt").exists()


def test_local_prompt_learner_command_omits_hub_revision_and_mounts_read_only(
    tmp_path,
):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"messages": []}\n', encoding="utf-8")
    plan = _local_data_plan(prompts)
    argv = _learner_argv(plan, 0)
    assert "--data-revision" not in argv
    assert "--eval-only" not in argv
    assert "--eval-interval" not in argv
    assert parse_learner_args(argv[3:]).data_revision is None
    node = _node_start_script(plan, 0, 0)
    assert '--volume "$RUN/data:/workspace/data:ro"' in node
    assert "refusing to reuse a same-name learner container" in node
    assert 'if [ "$STATUS" = running ]; then exit 0; fi' not in node


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

    assert "systemctl kill --signal=SIGKILL --kill-who=all" in scripts[0]
    assert "exit_status=9" in scripts[0]
    assert 'kill -KILL -- -"$PID"' in scripts[0]
    assert 'kill -0 "$PID"' in scripts[0]
    assert "syncer did not exit after SIGKILL" in scripts[0]


def test_syncer_lifecycle_uses_the_dedicated_host(tmp_path, monkeypatch):
    plan = _plan()
    plan["syncer_host"] = "root@syncer0"
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    calls = []
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: calls.append((target, script)),
    )

    ssh_harness._build_syncer(plan)
    ssh_harness._start_syncer(plan)
    ssh_harness.kill_syncer(plan_path)

    assert [target for target, _ in calls] == ["root@syncer0"] * 3


def test_syncer_launch_uses_a_durable_systemd_service(monkeypatch):
    plan = _plan()
    plan["syncer_host"] = "root@syncer0"
    calls = []
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: calls.append((target, script)),
    )

    ssh_harness._start_syncer(plan)

    assert len(calls) == 1
    target, script = calls[0]
    assert target == "root@syncer0"
    assert "systemd-run --quiet" in script
    assert "--service-type=exec" in script
    assert "--property=StandardInput=null" in script
    assert "--property=KillMode=mixed" in script
    assert '"$WRAPPER" "$EXIT_FILE"' in script
    assert "yeto-rl-syncer-acceptance.service" in script
    assert 'EXIT_FILE="$RUN/state/syncer.exit"' in script
    assert "nohup" not in script
    assert "setsid" not in script
    assert '[ ! -e "$UNIT_FILE" ] || return 1' in script
    assert 'ACTUAL_EXE="$(readlink -f "/proc/$PID/exe"' in script
    assert 'grep -Fqx -- "$CHECKPOINT_PATH"' in script
    assert 'if [ -n "$LEGACY_PID" ] && [ ! -s "$EXIT_FILE" ]' in script
    assert 'rm -f "$PID_FILE"' in script
    assert 'FRAGMENT="$(systemctl show --property=FragmentPath' in script
    assert "--property=Restart --value" in script
    assert "--property=NRestarts --value" in script
    assert "mapfile -d '' -t ACTUAL_ARGV" in script
    assert "syncer_child_pid" in script
    assert "EXPECTED_SYNCER_ARGV" in script
    assert "syncer_listener_owned_by_unit" in script
    assert "/sys/fs/cgroup$CONTROL_GROUP/cgroup.procs" in script
    assert "/proc/net/tcp6" in script
    assert "stale or unrelated process already listens" in script
    syntax = subprocess.run(
        ["bash", "-n"], input=script, text=True, capture_output=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr


def test_syncer_readiness_attests_exact_unit_and_listener_before_tcp(
    monkeypatch,
):
    plan = _plan()
    plan["syncer_host"] = "alice@b1"
    calls = []

    def fake_ssh(plan, target, script, **kwargs):
        calls.append((target, script))
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(ssh_harness, "_ssh", fake_ssh)

    ssh_harness._wait_for_syncer(plan, timeout_s=3)

    assert calls[0][0] == "alice@b1"
    identity_script = calls[0][1]
    assert "syncer_unit_pid" in identity_script
    assert "syncer_listener_owned_by_unit" in identity_script
    assert "EXPECTED_UNIT_ARGV" in identity_script
    assert "syncer unit/listener identity did not become ready" in identity_script
    assert "/sys/fs/cgroup$CONTROL_GROUP/cgroup.procs" in identity_script
    assert "getent ahosts" in identity_script
    assert "ip -o addr show" in identity_script
    assert "syncer address does not resolve" in identity_script
    assert [target for target, _script in calls[1:]] == ssh_harness._all_hosts(plan)
    assert all("/dev/tcp/" in script for _target, script in calls[1:])


@pytest.mark.parametrize(
    ("child", "returncode", "exit_result", "exit_code", "exit_status"),
    (
        (("/bin/sh", "-c", "exit 23"), 23, "exit-code", "exited", "23"),
        (("/bin/sh", "-c", "kill -TERM $$"), 143, "signal", "killed", "15"),
    ),
)
def test_syncer_service_wrapper_records_a_failed_exit(
    tmp_path, child, returncode, exit_result, exit_code, exit_status
):
    script = ssh_harness._syncer_start_script(_plan())
    prefix = "cat > \"$WRAPPER\" <<'SH'\n"
    wrapper_source = script.split(prefix, 1)[1].split("\nSH\n", 1)[0]
    wrapper = tmp_path / "run-syncer"
    exit_file = tmp_path / "syncer.exit"
    wrapper.write_text(wrapper_source)
    wrapper.chmod(0o700)

    result = ssh_harness.subprocess.run(
        [str(wrapper), str(exit_file), *child],
        check=False,
    )

    assert result.returncode == returncode
    fields = dict(line.split("=", 1) for line in exit_file.read_text().splitlines())
    assert fields["result"] == exit_result
    assert fields["exit_code"] == exit_code
    assert fields["exit_status"] == exit_status
    assert fields["timestamp"].endswith("Z")
    assert not list(tmp_path.glob("syncer.exit.tmp.*"))


def test_syncer_status_reports_the_recorded_exit_cause(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "syncer.exit").write_text(
        "result=signal\n"
        "exit_code=killed\n"
        "exit_status=15\n"
        "timestamp=2026-08-13T18:26:08Z\n"
    )
    script = f"RUN={ssh_harness.shlex.quote(str(tmp_path))}\n"
    script += ssh_harness._syncer_status_script(_plan())

    result = ssh_harness.subprocess.run(
        ["/bin/bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == (
        "syncer=stopped unit=yeto-rl-syncer-acceptance.service "
        "result=signal exit_code=killed exit_status=15 "
        "timestamp=2026-08-13T18:26:08Z"
    )


def test_syncer_status_prefers_exit_record_over_a_live_stale_pid(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "syncer.pid").write_text(f"{os.getpid()}\n")
    (state / "syncer.exit").write_text(
        "result=exit-code\n"
        "exit_code=exited\n"
        "exit_status=23\n"
        "timestamp=2026-08-13T18:26:08Z\n"
    )
    script = f"RUN={ssh_harness.shlex.quote(str(tmp_path))}\n"
    script += ssh_harness._syncer_status_script(_plan())

    result = ssh_harness.subprocess.run(
        ["/bin/bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "syncer=stopped" in result.stdout
    assert "result=exit-code" in result.stdout
    assert "mode=legacy" not in result.stdout


def test_syncer_host_defaults_to_the_first_learner_and_is_validated():
    plan = _plan()
    assert ssh_harness._syncer_host(plan) == "alice@a0"

    plan["syncer_host"] = "root@syncer0"
    ssh_harness._validate_plan(plan)
    assert ssh_harness._syncer_host(plan) == "root@syncer0"

    plan["syncer_host"] = "root@bad host"
    with pytest.raises(HarnessError, match="invalid SSH target"):
        ssh_harness._validate_plan(plan)


def test_status_and_stop_use_the_dedicated_syncer_host(tmp_path, monkeypatch):
    plan = _plan()
    plan["syncer_host"] = "root@syncer0"
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    calls = []

    def fake_ssh(plan, target, script, **kwargs):
        calls.append((target, script))
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(ssh_harness, "_ssh", fake_ssh)

    ssh_harness.status(plan_path)
    ssh_harness.stop(plan_path)

    syncer_calls = [target for target, script in calls if "syncer.pid" in script]
    assert syncer_calls == ["root@syncer0", "root@syncer0"]
    assert "systemctl is-active" in calls[0][1]
    assert "syncer.exit" in calls[0][1]
    assert "systemctl stop" in calls[-1][1]
    assert 'systemctl stop "$UNIT" || true' not in calls[-1][1]
    assert "syncer unit did not become inactive" in calls[-1][1]
    assert 'kill -0 "$PID"' in calls[-1][1]
    assert "legacy syncer did not exit after SIGTERM" in calls[-1][1]
    assert not any("docker logs" in script for _target, script in calls)


def test_status_includes_a_syncer_colocated_on_a_nonfirst_learner(
    tmp_path, monkeypatch
):
    plan = _plan()
    plan["syncer_host"] = "alice@b1"
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    calls = []

    def fake_ssh(plan, target, script, **kwargs):
        calls.append((target, script))
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(ssh_harness, "_ssh", fake_ssh)

    ssh_harness.status(plan_path)

    syncer_calls = [target for target, script in calls if "syncer.pid" in script]
    assert syncer_calls == ["alice@b1"]


def test_syncer_stop_refuses_a_same_name_unit_without_exact_identity(
    tmp_path, monkeypatch
):
    plan = _plan()
    script = ssh_harness._syncer_stop_script(plan)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "systemctl").write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  show) printf 'loaded\\n' ;;\n"
        "  stop) exit 42 ;;\n"
        "esac\n"
    )
    (bin_dir / "systemctl").chmod(0o700)
    home = tmp_path / "home"
    (home / plan["remote_run"] / "state").mkdir(parents=True)

    result = ssh_harness.subprocess.run(
        ["/bin/bash", "-c", script],
        env={"HOME": str(home), "PATH": f"{bin_dir}:/usr/bin:/bin"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "drifted identity" in result.stderr


def test_collect_reads_syncer_state_from_the_dedicated_host(tmp_path, monkeypatch):
    plan = _plan()
    plan["syncer_host"] = "root@syncer0"
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    commands = []
    monkeypatch.setattr(ssh_harness, "_require_program", lambda name: None)
    monkeypatch.setattr(
        ssh_harness, "_collect_capacity_preflight", lambda _plan, _artifacts: None
    )

    def ssh(*args, **kwargs):
        script = args[2]
        stdout = (
            json.dumps(
                {
                    "Status": "exited",
                    "ExitCode": 0,
                    "OOMKilled": False,
                    "RestartCount": 0,
                }
            )
            if script.startswith("docker inspect")
            else ""
        )
        return SimpleNamespace(stdout=stdout, stderr="")

    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        ssh,
    )
    monkeypatch.setattr(
        ssh_harness, "_run", lambda command, **kwargs: commands.append(command)
    )

    ssh_harness.collect(plan_path)

    assert any(
        command[-2].startswith("root@syncer0:") and command[-2].endswith("/state/")
        for command in commands
    )


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

    plan["tms_preload_patch"]["base_binary_sha256"] = "3" * 64
    with pytest.raises(HarnessError, match="pinned TMS"):
        ssh_harness._validate_plan(plan)

    plan["tms_preload_patch"] = _tms_patch_contract()
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
    assert parsed.tms_train_disk_backup_root == ("/data/yeto-rl/tms-disk-backup")
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
        "TORCH_DISABLE_ADDR2LINE=1",
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

    plan["jit_cache"] = ssh_harness._jit_cache_contract("/data/yeto-rl/jit-cache", plan)
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
    first_contract = ssh_harness._jit_cache_contract("/data/yeto-rl/jit-cache", first)
    second_contract = ssh_harness._jit_cache_contract("/data/yeto-rl/jit-cache", second)
    assert first_contract == second_contract

    changed_image = _plan()
    changed_image["docker_image"] = "radixark/miles@sha256:" + "a" * 64
    assert (
        ssh_harness._jit_cache_contract("/data/yeto-rl/jit-cache", changed_image)
        != first_contract
    )

    with pytest.raises(HarnessError, match="dedicated"):
        ssh_harness._jit_cache_contract("/data/yeto-rl/other", first)
    with pytest.raises(HarnessError, match="dedicated"):
        ssh_harness._jit_cache_contract("/data/yeto-rl/jit-cache/../escape", first)

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


def test_harness_freezes_stock_codex_identity_and_binary_mount():
    plan = _plan()
    plan["remote_env_file"] = ".config/yeto/rl.env"
    plan["secrlenv_daemon"] = _secrlenv_daemon_contract()
    plan["learner"].update(
        {
            "rl_model_recipe": "deepseek-v4-flash",
            "custom_generate_function_path": rl_config.SECRLENV_GENERATE,
            "custom_agent_function_path": rl_config.CODEX_HARNESS_AGENT,
            "codex_reasoning_effort": "xhigh",
            "use_session_server": True,
            "tito_model": "deepseekv4",
            "tito_allowed_append_roles": ["tool", "user"],
            "tensor_parallel": 8,
            "expert_parallel": 8,
            "rollout_num_gpus_per_engine": 8,
            "sglang_tp_size": 8,
            "sglang_dp_size": 1,
            "sglang_ep_size": 8,
            "lora_targets": "attention-routed-experts",
            "apply_chat_template_kwargs": {
                "thinking_mode": "thinking",
                "reasoning_effort": "max",
                "drop_thinking": False,
            },
            "reward_function": ssh_harness.SECRLENV_REWARD,
            "dynamic_sampling_filter_path": ssh_harness.SECRLENV_GROUP_FILTER,
            "dynamic_sampling_max_replacements": 0,
            "secrlenv_max_infrastructure_replacements": 1,
            "over_sampling_batch_size": plan["learner"]["groups_per_round"] + 1,
        }
    )
    plan["codex_harness"] = {
        "agent_function_path": rl_config.CODEX_HARNESS_AGENT,
        "agent_source_sha256": rl_config.CODEX_HARNESS_AGENT_SHA256,
        "controller_binary_path": "/controller/codex",
        "controller_package_manifest_path": "/controller/codex-package.json",
        "controller_app_server_schema_path": "/controller/app-server-v2.json",
        "bundle_binary_path": ssh_harness.CODEX_REMOTE_BUNDLE_PATH,
        "bundle_package_manifest_path": ssh_harness.CODEX_REMOTE_MANIFEST_PATH,
        "bundle_app_server_schema_path": ssh_harness.CODEX_REMOTE_SCHEMA_PATH,
        "container_binary_path": rl_config.CODEX_CONTAINER_BINARY_PATH,
        "container_app_server_schema_path": (
            rl_config.CODEX_CONTAINER_APP_SERVER_SCHEMA_PATH
        ),
        "binary_sha256": rl_config.CODEX_LINUX_BINARY_SHA256,
        "binary_size_bytes": rl_config.CODEX_LINUX_BINARY_SIZE_BYTES,
        "cli_version": rl_config.CODEX_CLI_VERSION,
        "npm_package": rl_config.CODEX_NPM_PACKAGE,
        "target": rl_config.CODEX_LINUX_TARGET,
        "npm_tarball_sha256": rl_config.CODEX_NPM_TARBALL_SHA256,
        "package_manifest_sha256": rl_config.CODEX_PACKAGE_MANIFEST_SHA256,
        "app_server_protocol_revision": (rl_config.CODEX_APP_SERVER_PROTOCOL_REVISION),
        "app_server_schema_sha256": rl_config.CODEX_APP_SERVER_SCHEMA_SHA256,
        "base_instructions_sha256": rl_config.CODEX_BASE_INSTRUCTIONS_SHA256,
        "terminal_exec_tool_schema_sha256": (
            rl_config.CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256
        ),
        "submit_tool_schema_sha256": (rl_config.CODEX_SUBMIT_TOOL_SCHEMA_SHA256),
        "dynamic_tools_schema_sha256": (rl_config.CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256),
        "reasoning_effort": "xhigh",
        "backend": {
            "profile": "deepseekv4",
            "tito_model": "deepseekv4",
            "model": "deepseekv4",
            "max_tokens": plan["learner"]["rollout_max_response_len"],
            "reasoning_effort": "max",
            "thinking": {"type": "enabled"},
            "chat_template": "deepseekv4",
            "chat_template_kwargs": {
                "thinking_mode": "thinking",
                "reasoning_effort": "max",
                "drop_thinking": False,
            },
            "tito_allowed_append_roles": ["tool", "user"],
        },
    }

    ssh_harness._validate_plan(plan)
    argv = _learner_argv(plan, 0)
    parsed = parse_learner_args(argv[3:])

    assert parsed.codex_harness_contract == plan["codex_harness"]
    for node_id in (0, 1):
        script = _node_start_script(plan, 0, node_id)
        assert rl_config.CODEX_CONTAINER_BINARY_PATH in script
        assert "YETO_CODEX_REASONING_EFFORT=xhigh" in script
        assert "YETO_CODEX_APP_SERVER_SCHEMA_SHA256=" in script
        assert "YETO_CODEX_HARNESS_CONTRACT_SHA256=" in script
        syntax = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        assert syntax.returncode == 0, syntax.stderr

    drifted = json.loads(json.dumps(plan))
    drifted["codex_harness"]["backend"]["chat_template_kwargs"]["drop_thinking"] = True
    with pytest.raises(HarnessError, match="backend/TITO contract drifted"):
        ssh_harness._validate_plan(drifted)

    for field in (
        "agent_source_sha256",
        "base_instructions_sha256",
        "terminal_exec_tool_schema_sha256",
        "submit_tool_schema_sha256",
        "dynamic_tools_schema_sha256",
    ):
        drifted = json.loads(json.dumps(plan))
        drifted["codex_harness"][field] = "0" * 64
        with pytest.raises(HarnessError, match="pinned stock Codex runtime"):
            ssh_harness._validate_plan(drifted)

    qwen = copy.deepcopy(plan)
    qwen["learner"].update(
        {
            "model": QWEN38_MODEL,
            "model_revision": QWEN38_REVISION,
            "rollout_model": None,
            "rollout_model_revision": None,
            "rl_model_recipe": "generic",
            "tito_model": "qwen38",
            "apply_chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "reasoning_effort": "xhigh",
            },
            "lora_targets": "attention",
            "expert_full_count": 0,
            "expert_parallel": 1,
            "tensor_parallel": 4,
            "pipeline_parallel": 2,
            "rollout_num_gpus_per_engine": 1,
            "sglang_tp_size": 1,
            "sglang_dp_size": 1,
            "sglang_ep_size": 1,
            "sglang_attention_backend": None,
            "sglang_page_size": None,
        }
    )
    qwen["codex_harness"]["backend"] = stock_codex_backend_contract(
        "qwen38",
        qwen["learner"]["rollout_max_response_len"],
    )

    ssh_harness._validate_plan(qwen)
    qwen_args = parse_learner_args(_learner_argv(qwen, 0)[3:])
    assert qwen_args.model == QWEN38_MODEL
    assert qwen_args.model_revision == QWEN38_REVISION
    assert qwen_args.tito_model == "qwen38"
    assert qwen_args.apply_chat_template_kwargs == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    }

    openenv = copy.deepcopy(qwen)
    openenv.pop("secrlenv_daemon")
    openenv["learner"].update(
        {
            "model": QWEN35_08B_MODEL,
            "model_revision": QWEN35_08B_REVISION,
            "tito_model": "qwen35",
            "codex_backend_profile": "qwen35_08b",
            "apply_chat_template_kwargs": {"clear_thinking": False},
            "custom_generate_function_path": (
                "miles.rollout.generate_hub.agentic_tool_call.generate"
            ),
            "custom_agent_function_path": rl_config.CODEX_OPENENV_AGENT,
            "reward_function": "openenv_generate.reward_func",
            "dynamic_sampling_filter_path": (
                "openenv_generate.check_terminal_bench_episode"
            ),
            "secrlenv_max_infrastructure_replacements": None,
        }
    )
    openenv["codex_harness"]["backend"] = stock_codex_backend_contract(
        "qwen35_08b",
        openenv["learner"]["rollout_max_response_len"],
    )
    openenv["codex_harness"]["openenv_identity_env"] = dict(
        rl_config.CODEX_OPENENV_IDENTITY_ENV
    )

    ssh_harness._validate_plan(openenv)
    openenv_args = parse_learner_args(_learner_argv(openenv, 0)[3:])
    assert openenv_args.tito_model == "qwen35"
    assert openenv_args.codex_backend_profile == "qwen35_08b"
    openenv_script = _node_start_script(openenv, 0, 0)
    assert "YETO_CODEX_OPENENV_BACKEND_PROFILE=qwen35_08b" in openenv_script
    assert "YETO_CODEX_OPENENV_MODEL_ID=Qwen/Qwen3.5-0.8B" in openenv_script

    drifted_openenv = copy.deepcopy(openenv)
    drifted_openenv["codex_harness"]["openenv_identity_env"][
        "YETO_CODEX_OPENENV_MODEL_REVISION"
    ] = "0" * 40
    with pytest.raises(HarnessError, match="pinned Codex OpenEnv surface"):
        ssh_harness._validate_plan(drifted_openenv)

    qwen["learner"]["model_revision"] = "0" * 40
    with pytest.raises(HarnessError, match="model profile drifted"):
        ssh_harness._validate_plan(qwen)


def test_codex_controller_artifacts_are_attested_before_plan_write(
    tmp_path, monkeypatch
):
    binary = tmp_path / "codex"
    manifest = tmp_path / "codex-package.json"
    schema = tmp_path / "app-server-v2.json"
    binary.write_bytes(b"stock-linux-codex")
    binary.chmod(0o555)
    manifest.write_bytes(b"signed-package-manifest")
    schema.write_bytes(b"signed-v2-schema")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        ssh_harness, "CODEX_LINUX_BINARY_SIZE_BYTES", binary.stat().st_size
    )
    monkeypatch.setattr(ssh_harness, "CODEX_LINUX_BINARY_SHA256", digest(binary))
    monkeypatch.setattr(ssh_harness, "CODEX_PACKAGE_MANIFEST_SHA256", digest(manifest))
    monkeypatch.setattr(ssh_harness, "CODEX_APP_SERVER_SCHEMA_SHA256", digest(schema))
    identity = {
        "base_instructions_sha256": rl_config.CODEX_BASE_INSTRUCTIONS_SHA256,
        "terminal_exec_tool_schema_sha256": (
            rl_config.CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256
        ),
        "submit_tool_schema_sha256": (rl_config.CODEX_SUBMIT_TOOL_SCHEMA_SHA256),
        "dynamic_tools_schema_sha256": (rl_config.CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256),
    }
    monkeypatch.setattr(ssh_harness, "_codex_adapter_identity", lambda: identity)
    namespace = SimpleNamespace(
        codex_harness_binary=str(binary),
        codex_package_manifest=str(manifest),
        codex_app_server_schema=str(schema),
    )
    args = SimpleNamespace(
        codex_reasoning_effort="xhigh",
        tito_model="deepseekv4",
        rollout_max_response_len=4096,
        tito_allowed_append_roles=["tool", "user"],
        apply_chat_template_kwargs={
            "thinking_mode": "thinking",
            "reasoning_effort": "max",
            "drop_thinking": False,
        },
    )

    contract = ssh_harness._codex_harness_contract(namespace, args)

    assert contract["controller_binary_path"] == str(binary)
    assert contract["binary_sha256"] == digest(binary)
    assert contract["app_server_schema_sha256"] == digest(schema)
    assert contract["backend"]["reasoning_effort"] == "max"

    binary.chmod(0o755)
    binary.write_bytes(b"drifted")
    binary.chmod(0o555)
    with pytest.raises(HarnessError, match="binary size"):
        ssh_harness._codex_harness_contract(namespace, args)


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
    assert syncer[syncer.index("--quorum-timeout-s") + 1] == "900"
    assert syncer[syncer.index("--final-ack-timeout-s") + 1] == "3600"
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
    assert f'git -C "$RUN/miles" fetch --depth 1 origin {MILES_BASE_COMMIT}' in setup
    assert f'MILES_BUNDLE="$RUN/source/{MILES_BUNDLE_PATH}"' in setup
    assert MILES_BUNDLE_SHA256 in setup
    assert f'git -C "$RUN/miles" fetch "$MILES_BUNDLE" {MILES_COMMIT}' in setup
    assert f'git -C "$RUN/sglang" remote set-url origin {SGLANG_REPOSITORY}' in setup
    assert '"$RUN/sglang"' in setup
    assert "SGLANG_BUNDLE" not in setup
    assert f'git -C "$RUN/sglang" fetch --depth 1 origin {SGLANG_COMMIT}' in setup
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
    assert "\n      --enable-dind-debug" not in script
    assert "\n      --dind-image" not in script
    assert "task_pack_sha256" in script
    assert "python3 -m yeto.rl.secrlenv_task_images" in script
    assert "secrlenv_task_images=ready" not in script
    assert (
        script.index('test "$SOURCE_SHA"')
        < script.index("python3 -m yeto.rl.secrlenv_task_images")
        < script.index("python3 -m secrlenv_rl.server")
    )
    assert "secrlenv_daemon=ready" in script
    assert 'ENV_FILE="$HOME/.config/yeto/rl.env"' in script
    assert 'TOKEN_FILE="$STATE_ROOT/daemon.token"' in script
    assert "secrets.token_hex(32)" in script
    assert '--token-file "$TOKEN_FILE"' in script
    assert "--env SECRLENV_DAEMON_URL=http://127.0.0.1:28765" in node
    assert ("--env SECRLENV_TASK_PACK_SHA256=" + "3" * 64) in node
    assert "--env SECRLENV_BEARER_TOKEN_FILE=/run/secrlenv/daemon.token" in node
    assert (
        "/data/yeto-rl/secrlenv-runs/acceptance/daemon.token:"
        "/run/secrlenv/daemon.token:ro" in node
    )


def test_secrlenv_dind_debug_is_explicit_pinned_and_propagated():
    plan = _plan()
    _enable_secrlenv_agent(plan)
    plan["secrlenv_daemon"] = {
        **_secrlenv_daemon_contract(),
        "enable_dind_debug": True,
        "dind_image": "docker:24-dind",
        "dind_image_id": "sha256:" + "5" * 64,
        "dind_debug_script_sha256": "6" * 64,
    }

    ssh_harness._validate_plan(plan)
    script = ssh_harness._secrlenv_daemon_script(plan)

    assert "--enable-dind-debug" in script
    assert "--dind-image sha256:" + "5" * 64 in script
    assert "DIND_IMAGE_ID=" in script
    assert "sha256:" + "5" * 64 in script
    assert "DIND_DEBUG_SCRIPT=" in script
    assert "6" * 64 in script
    assert "existing secrlenv daemon has the wrong DinD debug identity" in script


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"enable_dind_debug": 1}, "explicit boolean"),
        (
            {"enable_dind_debug": True},
            "requires a pinned image and image ID",
        ),
        (
            {
                "enable_dind_debug": True,
                "dind_image": "docker:24-dind",
                "dind_image_id": "sha256:" + "5" * 64,
            },
            "requires a pinned debug script",
        ),
        (
            {"dind_image": "docker:24-dind"},
            "must not carry unused debug identities",
        ),
    ],
)
def test_secrlenv_dind_debug_contract_fails_closed(updates, message):
    plan = _plan()
    _enable_secrlenv_agent(plan)
    plan["secrlenv_daemon"] = {**_secrlenv_daemon_contract(), **updates}

    with pytest.raises(HarnessError, match=message):
        ssh_harness._validate_plan(plan)


def test_secrlenv_island_head_placement_is_explicit_and_fail_closed(
    tmp_path, monkeypatch
):
    plan = _plan()
    _enable_secrlenv_agent(plan)
    plan["secrlenv_daemon"] = {
        **_secrlenv_daemon_contract(),
        "placement": "island-heads",
    }
    ssh_harness._validate_plan(plan)

    assert ssh_harness._secrlenv_daemon_hosts(plan) == [
        "alice@a0",
        "alice@b0",
    ]
    for learner_id in range(2):
        head = _node_start_script(plan, learner_id, 0)
        worker = _node_start_script(plan, learner_id, 1)
        assert "SECRLENV_DAEMON_URL=" in head
        assert "/run/secrlenv/daemon.token:ro" in head
        assert "SECRLENV_DAEMON_URL=" not in worker
        assert "/run/secrlenv/daemon.token:ro" not in worker

    calls = []

    def fake_ssh(plan, target, script, **kwargs):
        calls.append((target, script))
        return SimpleNamespace(stdout="", stderr="")

    monkeypatch.setattr(ssh_harness, "_ssh", fake_ssh)
    ssh_harness._start_secrlenv_daemons(plan)
    assert [target for target, _script in calls] == ["alice@a0", "alice@b0"]
    assert all(
        "python3 -m yeto.rl.secrlenv_task_images" in script for _target, script in calls
    )

    calls.clear()
    plan_path = _write_plan(tmp_path / "plan.json", plan)
    ssh_harness.status(plan_path)
    daemon_status_targets = [
        target for target, script in calls if "secrlenv_daemon=running" in script
    ]
    assert daemon_status_targets == ["alice@a0", "alice@b0"]

    calls.clear()
    ssh_harness._stop_secrlenv_daemons(plan)
    assert [target for target, _script in calls] == ["alice@a0", "alice@b0"]

    plan["secrlenv_daemon"]["placement"] = "unknown"
    with pytest.raises(HarnessError, match="daemon contract"):
        ssh_harness._validate_plan(plan)
    with pytest.raises(HarnessError, match="invalid placement"):
        ssh_harness._secrlenv_daemon_hosts(plan)


def test_legacy_secrlenv_plan_without_placement_retains_all_host_semantics(
    monkeypatch,
):
    plan = _plan()
    _enable_secrlenv_agent(plan)
    plan["secrlenv_daemon"] = _secrlenv_daemon_contract()
    ssh_harness._validate_plan(plan)

    assert ssh_harness._secrlenv_daemon_hosts(plan) == ssh_harness._all_hosts(plan)
    assert "SECRLENV_DAEMON_URL=" in _node_start_script(plan, 0, 1)

    calls = []
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: calls.append((target, script)),
    )
    ssh_harness._start_secrlenv_daemons(plan)
    assert [target for target, _script in calls] == ssh_harness._all_hosts(plan)

    calls.clear()
    ssh_harness._stop_secrlenv_daemons(plan)
    assert [target for target, _script in calls] == ssh_harness._all_hosts(plan)


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


def test_start_sets_up_a_dedicated_syncer_without_gpu_requirements(
    tmp_path, monkeypatch
):
    plan = _plan()
    plan["syncer_host"] = "root@syncer0"
    path = _write_plan(tmp_path / "plan.json", plan)
    calls = []

    monkeypatch.setattr(ssh_harness, "deploy", lambda _path: None)
    monkeypatch.setattr(ssh_harness, "_start_secrlenv_daemons", lambda _plan: None)
    monkeypatch.setattr(
        ssh_harness, "_host_setup_script", lambda *_args: "learner-setup"
    )
    monkeypatch.setattr(
        ssh_harness,
        "_syncer_host_setup_script",
        lambda _plan: "syncer-setup",
        raising=False,
    )
    monkeypatch.setattr(
        ssh_harness,
        "_ssh",
        lambda plan, target, script, **kwargs: calls.append((target, script)),
    )
    monkeypatch.setattr(ssh_harness, "_build_syncer", lambda _plan: None)
    monkeypatch.setattr(ssh_harness, "_start_syncer", lambda _plan: None)
    monkeypatch.setattr(ssh_harness, "_wait_for_syncer", lambda _plan: None)
    monkeypatch.setattr(ssh_harness, "_start_island", lambda *_args: None)

    ssh_harness.start(path)

    assert calls.count(("root@syncer0", "syncer-setup")) == 1
    assert all(
        target != "root@syncer0"
        for target, script in calls
        if script == "learner-setup"
    )


def test_verification_requires_an_exited_zero_status_container():
    inspection = {
        "Status": "exited",
        "ExitCode": 0,
        "OOMKilled": False,
        "RestartCount": 0,
    }
    assert _container_succeeded(inspection)
    assert not _container_succeeded({**inspection, "Status": "running"})
    assert not _container_succeeded({**inspection, "ExitCode": 1})
    assert not _container_succeeded({**inspection, "OOMKilled": True})
    assert not _container_succeeded({**inspection, "RestartCount": 1})


def test_container_state_filter_rejects_secret_bearing_inspect_payload():
    raw = json.dumps(
        {
            "Status": "exited",
            "ExitCode": 0,
            "OOMKilled": False,
            "RestartCount": 0,
            "Config": {"Env": ["SECRET=do-not-collect"]},
        }
    )

    with pytest.raises(HarnessError, match="unapproved fields"):
        ssh_harness._parse_container_inspection(raw)


def test_remote_event_filters_never_emit_prompt_or_tool_payloads(tmp_path):
    events = tmp_path / "events.jsonl"
    secret = "SECRET prompt response and tool payload"
    events.write_text(
        json.dumps(
            {
                "event": "rl_policy_apply",
                "policy_version": 2,
                "sync/global_policy_hash": "a" * 64,
                "prompt": secret,
                "response": secret,
            }
        )
        + "\n"
        + json.dumps({"event": "unapproved", "payload": secret})
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ssh_harness._EVENT_EVIDENCE_FILTER,
            str(events),
            "strict-learner",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert secret not in result.stdout
    assert json.loads(result.stdout) == {
        "event": "rl_policy_apply",
        "policy_version": 2,
        "sync/global_policy_hash": "a" * 64,
    }


def test_training_collect_capacity_fails_before_large_transfer(tmp_path, monkeypatch):
    plan = _plan()
    calls = []

    def fake_ssh(_plan, target, script, **_kwargs):
        calls.append((target, script))
        return SimpleNamespace(stdout="10000000000\n", stderr="")

    monkeypatch.setattr(ssh_harness, "_ssh", fake_ssh)
    monkeypatch.setattr(
        ssh_harness.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1024),
    )

    with pytest.raises(HarnessError, match="lacks space"):
        ssh_harness._collect_capacity_preflight(plan, tmp_path)

    assert calls
