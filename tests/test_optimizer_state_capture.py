from __future__ import annotations

import copy
import json
import random
import threading
import time
from collections import OrderedDict

import numpy as np
import pytest
import torch
import yeto.optimizer_state_capture as capture_module

from yeto.bounded_background_writer import BackgroundWriterFailed, WriterState
from yeto.fragments import Fragment, FragmentLayout, MERGE_RDA
from yeto.optimizer_state_capture import (
    CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL,
    CaptureIntegrityError,
    OptimizerStateCapture,
    load_capture,
)
from yeto.protocol import DTYPE_F32
from yeto.tensor_io import pack_flat


def _fixture(
    tmp_path,
    *,
    max_bytes=10_000_000,
    every=1,
    max_hmc=8,
    max_midpoint=8,
    background_writer=False,
    writer_max_items=4,
    writer_max_bytes=10_000_000,
    capture_profile="full",
):
    params = OrderedDict(
        [
            ("a", torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))),
            ("b", torch.nn.Parameter(torch.tensor([-3.0], dtype=torch.float32))),
        ]
    )
    layout = FragmentLayout([Fragment(MERGE_RDA, [("a", 2), ("b", 1)])])
    optimizer = torch.optim.AdamW(
        params.values(), lr=0.125, betas=(0.7, 0.91), eps=1e-6, weight_decay=0.03
    )
    optimizer.state[params["a"]] = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.tensor([0.25, -0.5]),
        "exp_avg_sq": torch.tensor([1.25, 2.5]),
    }
    optimizer.state[params["b"]] = {
        "step": torch.tensor(11.0),
        "exp_avg": torch.tensor([0.75]),
        "exp_avg_sq": torch.tensor([3.5]),
    }
    capture = OptimizerStateCapture(
        tmp_path,
        params=params,
        layout=layout,
        optimizer=optimizer,
        learner_id=4,
        rank=0,
        capture_profile=capture_profile,
        every=every,
        max_hmc_events=max_hmc,
        max_midpoint_windows=max_midpoint,
        max_bytes=max_bytes,
        background_writer=background_writer,
        background_writer_max_items=writer_max_items,
        background_writer_max_bytes=writer_max_bytes,
    )
    return params, layout, optimizer, capture


def _finish_two_step_window(params, optimizer, capture):
    window_uuid = capture.note_broadcast(
        0, 17, local_step=30, tokens_total=3_000, window_steps=2
    )
    assert window_uuid is not None
    for local_step in (31, 32):
        for param in params.values():
            param.grad = torch.full_like(param, 0.25)
        capture.capture_first_post_broadcast_gradients(
            local_step_before_update=local_step - 1,
            tokens_total=3_000 + (local_step - 31) * 128,
            clip_total_norm=torch.tensor(0.5),
            clip_max_norm=1.0,
        )
        with torch.no_grad():
            params["a"].add_(1.0)
            params["b"].sub_(2.0)
            for param in params.values():
                optimizer.state[param]["step"].add_(1)
                optimizer.state[param]["exp_avg"].add_(0.1)
                optimizer.state[param]["exp_avg_sq"].add_(0.2)
        capture.after_optimizer_step(
            local_step=local_step,
            tokens_total=3_000 + (local_step - 30) * 128,
            current_window_steps=2,
        )
    endpoint = torch.cat([params["a"].detach(), params["b"].detach()])
    push = capture.note_push(
        window_uuid=window_uuid,
        fragment_id=0,
        pull_global_step=44,
        base_version=17,
        local_step=32,
        c_steps=2,
        c_tokens=256,
        wire_codec="f32",
        payload=pack_flat(endpoint, DTYPE_F32),
    )
    capture.note_push_enqueued(push["attempt_serial"])
    capture.close()


def _artifact(directory, kind):
    paths = list(directory.glob(f"*-{kind}-*.pt"))
    assert len(paths) == 1
    return load_capture(paths[0])


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for background capture state")
        time.sleep(0.001)


def _clone_live_state(params, optimizer):
    return {
        "params": {name: value.detach().clone() for name, value in params.items()},
        "grads": {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in params.items()
        },
        "optimizer": {
            name: {
                key: value.detach().clone()
                if isinstance(value, torch.Tensor)
                else copy.deepcopy(value)
                for key, value in optimizer.state[param].items()
            }
            for name, param in params.items()
        },
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state().clone(),
    }


def _assert_live_state_equal(before, params, optimizer):
    for name, value in params.items():
        assert torch.equal(value, before["params"][name])
        expected_grad = before["grads"][name]
        if expected_grad is None:
            assert value.grad is None
        else:
            assert torch.equal(value.grad, expected_grad)
        for key, expected in before["optimizer"][name].items():
            actual = optimizer.state[value][key]
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected
    assert random.getstate() == before["python_rng"]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before["numpy_rng"][0]
    assert np.array_equal(after_numpy[1], before["numpy_rng"][1])
    assert after_numpy[2:] == before["numpy_rng"][2:]
    assert torch.equal(torch.get_rng_state(), before["torch_rng"])


def test_exact_adamw_moments_steps_and_first_clipped_gradient_are_passive(tmp_path):
    params, _layout, optimizer, capture = _fixture(tmp_path)
    params["a"].grad = torch.tensor([0.4, -0.2])
    params["b"].grad = torch.tensor([0.1])

    before = _clone_live_state(params, optimizer)
    capture.note_broadcast(0, 23, local_step=100, tokens_total=4096, window_steps=4)
    capture.capture_first_post_broadcast_gradients(
        local_step_before_update=100, tokens_total=4096
    )
    _assert_live_state_equal(before, params, optimizer)

    record = _artifact(tmp_path, "adamw_first_gradient")
    assert record["metadata"]["fragment_version"] == 23
    assert record["metadata"]["steps_since_reset_before_update"] == 0
    assert record["metadata"]["gradient_boundary"] == (
        "post_allreduce_post_clip_pre_optimizer_step"
    )
    tensors = record["payload"]["tensors"]
    assert tensors["a"]["optimizer_step"] == 7
    assert tensors["b"]["optimizer_step"] == 11
    assert torch.equal(tensors["a"]["exp_avg"], torch.tensor([0.25, -0.5]))
    assert torch.equal(tensors["a"]["exp_avg_sq"], torch.tensor([1.25, 2.5]))
    assert torch.equal(
        tensors["a"]["first_clipped_gradient"], torch.tensor([0.4, -0.2])
    )
    assert tensors["a"]["optimizer_group_config"]["betas"] == [0.7, 0.91]
    assert tensors["a"]["optimizer_group_config"]["weight_decay"] == 0.03

    # Captured tensors are real copies, not aliases of subsequent live state.
    optimizer.state[params["a"]]["exp_avg"].add_(99)
    params["a"].grad.add_(99)
    assert torch.equal(tensors["a"]["exp_avg"], torch.tensor([0.25, -0.5]))
    assert torch.equal(
        tensors["a"]["first_clipped_gradient"], torch.tensor([0.4, -0.2])
    )


def test_crp_pti_directional_profile_is_closed_small_and_push_joined(tmp_path):
    full_dir = tmp_path / "full"
    reduced_dir = tmp_path / "reduced"
    full_params, _layout, full_optimizer, full = _fixture(full_dir)
    reduced_params, _layout, reduced_optimizer, reduced = _fixture(
        reduced_dir,
        max_hmc=0,
        capture_profile=CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL,
    )

    _finish_two_step_window(full_params, full_optimizer, full)
    _finish_two_step_window(reduced_params, reduced_optimizer, reduced)

    reduced_manifest = json.loads((reduced_dir / "manifest.json").read_text())
    assert reduced_manifest["config"]["capture_profile"] == (
        CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL
    )
    assert reduced_manifest["config"]["scientific_scope"] == (
        "crp_pti_direction_evidence_only"
    )
    assert reduced_manifest["config"]["capture_v2_restore_complete"] is False
    assert reduced_manifest["counters"]["hmc_events_admitted"] == 0
    assert {row["kind"] for row in reduced_manifest["artifacts"]} == {
        "richardson_window",
        "push_candidate",
    }
    assert not list(reduced_dir.glob("*-adamw_first_gradient-*.pt"))

    reduced_window = _artifact(reduced_dir, "richardson_window")
    full_window = _artifact(full_dir, "richardson_window")
    assert set(reduced_window["payload"]) == {
        "anchor",
        "midpoint",
        "endpoint",
        "step_history",
        "lr_mass_first_by_group",
        "lr_mass_second_by_group",
    }
    for boundary in ("anchor", "midpoint", "endpoint"):
        assert set(reduced_window["payload"][boundary]) == {
            "tensor_order",
            "parameters_f32",
        }
    parameter_bytes = sum(
        param.numel() * param.element_size() for param in reduced_params.values()
    )
    assert capture_module._tensor_storage_bytes(reduced_window["payload"]) == (
        3 * parameter_bytes + 4 * torch.tensor(0.5).element_size()
    )
    assert capture_module._tensor_storage_bytes(full_window["payload"]) > (
        3 * capture_module._tensor_storage_bytes(reduced_window["payload"])
    )


def test_crp_pti_directional_profile_rejects_hmc_and_unknown_profiles(tmp_path):
    with pytest.raises(ValueError, match="requires max_hmc_events=0"):
        _fixture(
            tmp_path / "hmc",
            capture_profile=CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL,
            max_hmc=1,
        )
    with pytest.raises(ValueError, match="unknown optimizer-state capture profile"):
        _fixture(tmp_path / "unknown", capture_profile="restore-ish")


def test_exact_anchor_midpoint_endpoint_and_identity(tmp_path):
    params, _layout, _optimizer, capture = _fixture(tmp_path)
    capture.note_window_reset(
        0, 9, local_step=20, tokens_total=2_000, window_steps=4, reason="lag_push"
    )

    expected = {}
    for local_step in range(21, 25):
        capture.capture_first_post_broadcast_gradients(
            local_step_before_update=local_step - 1,
            tokens_total=2_000 + (local_step - 21) * 128,
            clip_total_norm=torch.tensor(0.75),
        )
        with torch.no_grad():
            params["a"].add_(torch.tensor([1.0, 2.0]))
            params["b"].add_(3.0)
            for param in params.values():
                optimizer_state = _optimizer.state[param]
                optimizer_state["step"].add_(1)
                optimizer_state["exp_avg"].add_(0.1)
                optimizer_state["exp_avg_sq"].add_(0.2)
        if local_step in (22, 24):
            expected[local_step] = torch.cat(
                [params["a"].detach().clone(), params["b"].detach().clone()]
            )
        capture.after_optimizer_step(
            local_step=local_step,
            tokens_total=2_000 + (local_step - 20) * 128,
            current_window_steps=4,
        )

    record = _artifact(tmp_path, "richardson_window")
    assert record["metadata"]["reset_reason"] == "lag_push"
    assert record["metadata"]["fragment_version"] == 9
    assert record["metadata"]["reset_local_step"] == 20
    assert record["metadata"]["midpoint_local_step"] == 22
    assert record["metadata"]["endpoint_local_step"] == 24
    assert record["payload"]["anchor"]["tensor_order"] == ["a", "b"]
    assert record["payload"]["anchor"]["parameters_f32"].dtype == torch.float32
    assert torch.equal(
        record["payload"]["anchor"]["parameters_f32"], torch.tensor([1.0, 2.0, -3.0])
    )
    assert torch.equal(record["payload"]["midpoint"]["parameters_f32"], expected[22])
    assert torch.equal(record["payload"]["endpoint"]["parameters_f32"], expected[24])
    anchor_a = record["payload"]["anchor"]["optimizer"]["a"]
    midpoint_a = record["payload"]["midpoint"]["optimizer"]["a"]
    endpoint_a = record["payload"]["endpoint"]["optimizer"]["a"]
    assert anchor_a["optimizer_step"] == 7
    assert midpoint_a["optimizer_step"] == 9
    assert endpoint_a["optimizer_step"] == 11
    assert torch.equal(
        anchor_a["raw_optimizer_state"]["exp_avg"], torch.tensor([0.25, -0.5])
    )
    assert len(record["payload"]["step_history"]) == 4
    assert record["payload"]["lr_mass_first_by_group"] == [0.25]
    assert record["payload"]["lr_mass_second_by_group"] == [0.25]
    assert torch.allclose(
        record["payload"]["decoupled_decay_first_f32"],
        torch.tensor([3.0, 6.0, -3.0]) * (0.125 * 0.03),
        rtol=0,
        atol=1e-9,
    )
    assert torch.allclose(
        record["payload"]["decoupled_decay_second_f32"],
        torch.tensor([7.0, 14.0, 9.0]) * (0.125 * 0.03),
        rtol=0,
        atol=1e-9,
    )
    assert all(
        row["clip"]["max_norm"] == 1.0 for row in record["payload"]["step_history"]
    )
    assert all(
        torch.allclose(row["clip"]["coefficient"], torch.tensor(1.0))
        for row in record["payload"]["step_history"]
    )


def test_atomic_sidecar_tamper_rejection_and_no_temporary_files(tmp_path):
    params, _layout, _optimizer, capture = _fixture(tmp_path)
    params["a"].grad = torch.tensor([0.1, 0.2])
    params["b"].grad = torch.tensor([0.3])
    capture.note_broadcast(0, 1, local_step=0, tokens_total=0, window_steps=4)
    capture.capture_first_post_broadcast_gradients(
        local_step_before_update=0, tokens_total=0
    )
    path = next(tmp_path.glob("*-adamw_first_gradient-*.pt"))
    assert path.with_suffix(".pt.sha256").exists()
    assert not list(tmp_path.glob("*.tmp-*"))
    assert load_capture(path)["kind"] == "adamw_first_gradient"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "background_writer" not in manifest
    assert "background_writer" not in manifest["config"]

    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(CaptureIntegrityError, match="file checksum mismatch"):
        load_capture(path)


def test_background_writer_completes_live_fifo_artifacts_and_emits_stats(
    tmp_path, monkeypatch
):
    main_thread = threading.current_thread().name
    phase = {"value": "construct"}
    manifest_publications = []
    artifact_hashes = []
    artifact_sidecars = []
    fsync_calls = []
    admitted_headers = []

    original_manifest_publication = OptimizerStateCapture._write_manifest_raw
    original_artifact_hash = capture_module._background_artifact_sha256
    original_artifact_sidecar = capture_module._background_artifact_sidecar_bytes
    original_publish = OptimizerStateCapture._publish_background_item
    original_fsync = capture_module.os.fsync

    def traced_manifest_publication(self):
        manifest_publications.append((phase["value"], threading.current_thread().name))
        return original_manifest_publication(self)

    def traced_artifact_hash(raw):
        artifact_hashes.append((phase["value"], threading.current_thread().name))
        return original_artifact_hash(raw)

    def traced_artifact_sidecar(digest, name):
        artifact_sidecars.append((phase["value"], threading.current_thread().name))
        return original_artifact_sidecar(digest, name)

    def inspect_publish(self, item):
        raw = item.payload
        header_bytes = int.from_bytes(
            raw[-capture_module.BACKGROUND_TRAILER_BYTES :], "big"
        )
        header_start = len(raw) - capture_module.BACKGROUND_TRAILER_BYTES - header_bytes
        admitted_headers.append(
            json.loads(raw[header_start : -capture_module.BACKGROUND_TRAILER_BYTES])
        )
        return original_publish(self, item)

    def traced_fsync(descriptor):
        fsync_calls.append((phase["value"], threading.current_thread().name))
        return original_fsync(descriptor)

    monkeypatch.setattr(
        OptimizerStateCapture, "_write_manifest_raw", traced_manifest_publication
    )
    monkeypatch.setattr(
        capture_module, "_background_artifact_sha256", traced_artifact_hash
    )
    monkeypatch.setattr(
        capture_module,
        "_background_artifact_sidecar_bytes",
        traced_artifact_sidecar,
    )
    monkeypatch.setattr(
        OptimizerStateCapture, "_publish_background_item", inspect_publish
    )
    monkeypatch.setattr(capture_module.os, "fsync", traced_fsync)

    params, _layout, optimizer, capture = _fixture(
        tmp_path,
        background_writer=True,
        writer_max_items=2,
    )
    assert manifest_publications == [("construct", main_thread)]
    phase["value"] = "hot"
    params["a"].grad = torch.tensor([0.4, -0.2])
    params["b"].grad = torch.tensor([0.1])
    window_uuid = capture.note_broadcast(
        0, 17, local_step=30, tokens_total=3_000, window_steps=2
    )
    assert window_uuid is not None
    for local_step in (31, 32):
        capture.capture_first_post_broadcast_gradients(
            local_step_before_update=local_step - 1,
            tokens_total=3_000 + (local_step - 31) * 128,
            clip_total_norm=torch.tensor(0.5),
        )
        with torch.no_grad():
            params["a"].add_(1.0)
            params["b"].sub_(2.0)
            for param in params.values():
                optimizer.state[param]["step"].add_(1)
        capture.after_optimizer_step(
            local_step=local_step,
            tokens_total=3_000 + (local_step - 30) * 128,
            current_window_steps=2,
        )
    endpoint = torch.cat([params["a"].detach(), params["b"].detach()])
    candidate = capture.note_push(
        window_uuid=window_uuid,
        fragment_id=0,
        pull_global_step=44,
        base_version=17,
        local_step=32,
        c_steps=2,
        c_tokens=256,
        wire_codec="f32",
        payload=pack_flat(endpoint, DTYPE_F32),
    )
    capture.note_push_enqueued(candidate["attempt_serial"])
    assert capture._background_writer is not None
    writer_thread = capture._background_writer.thread_name
    _wait_until(lambda: capture._background_writer.snapshot().completed_items == 3)

    assert manifest_publications == [("construct", main_thread)]
    assert len(admitted_headers) == 3
    assert all("sha256" not in header for header in admitted_headers)
    assert artifact_hashes == [("hot", writer_thread)] * 3
    assert artifact_sidecars == [("hot", writer_thread)] * 3
    hot_fsyncs = [thread for event_phase, thread in fsync_calls if event_phase == "hot"]
    assert hot_fsyncs
    assert set(hot_fsyncs) == {writer_thread}

    phase["value"] = "final"
    capture.close()

    assert manifest_publications == [
        ("construct", main_thread),
        ("final", main_thread),
    ]
    assert not [event for event in manifest_publications if event[0] == "hot"]

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counters"]["closed"] is True
    assert manifest["counters"]["background_artifact_reserved_bytes"] == 0
    assert manifest["counters"]["drop_counts"] == {}
    assert [row["kind"] for row in manifest["artifacts"]] == [
        "adamw_first_gradient",
        "richardson_window",
        "push_candidate",
    ]
    assert [
        load_capture(tmp_path / row["path"])["kind"] for row in manifest["artifacts"]
    ] == [
        "adamw_first_gradient",
        "richardson_window",
        "push_candidate",
    ]
    stats = manifest["background_writer"]
    assert stats["state"] == "closed"
    assert stats["accepted_items"] == stats["completed_items"] == 3
    assert stats["accepted_bytes"] == stats["completed_bytes"]
    assert stats["abandoned_items"] == stats["reserved_items"] == 0
    assert stats["queued_items"] == stats["in_flight_items"] == 0
    assert stats["worker_alive"] is False
    assert not list(tmp_path.glob("*.tmp-*"))


def test_background_close_does_not_publish_completion_before_worker_drain(
    tmp_path, monkeypatch
):
    entered = threading.Event()
    release = threading.Event()
    original = OptimizerStateCapture._publish_background_item

    def blocked_publish(self, item):
        entered.set()
        assert release.wait(3.0)
        return original(self, item)

    monkeypatch.setattr(
        OptimizerStateCapture, "_publish_background_item", blocked_publish
    )
    params, _layout, _optimizer, capture = _fixture(tmp_path, background_writer=True)
    params["a"].grad = torch.ones_like(params["a"])
    params["b"].grad = torch.ones_like(params["b"])
    capture.note_broadcast(0, 1, local_step=0, tokens_total=0, window_steps=4)
    capture.capture_first_post_broadcast_gradients(
        local_step_before_update=0, tokens_total=0
    )
    assert entered.wait(3.0)

    close_errors = []

    def close_capture():
        try:
            capture.close()
        except BaseException as exc:
            close_errors.append(exc)

    closer = threading.Thread(target=close_capture, name="capture-close-test")
    closer.start()
    assert capture._background_writer is not None
    _wait_until(
        lambda: capture._background_writer.snapshot().state is WriterState.CLOSING
    )
    incomplete = json.loads((tmp_path / "manifest.json").read_text())
    assert incomplete["counters"]["closed"] is False
    assert not list(tmp_path.glob("*-adamw_first_gradient-*.pt"))
    assert closer.is_alive()

    release.set()
    closer.join(3.0)
    assert not closer.is_alive()
    assert close_errors == []
    complete = json.loads((tmp_path / "manifest.json").read_text())
    assert complete["counters"]["closed"] is True
    assert complete["background_writer"]["state"] == "closed"
    assert complete["background_writer"]["completed_items"] == 1
    assert len(list(tmp_path.glob("*-adamw_first_gradient-*.pt"))) == 1
    assert complete["counters"]["drop_counts"] == {
        "midpoint_incomplete_at_close": 1,
        "window_unpushed_at_close": 1,
    }


def test_background_publication_failure_aborts_at_next_capture_boundary(
    tmp_path, monkeypatch
):
    injected = OSError("injected artifact fsync failure")

    def fail_publish(_self, _item):
        raise injected

    monkeypatch.setattr(OptimizerStateCapture, "_publish_background_item", fail_publish)
    params, _layout, _optimizer, capture = _fixture(tmp_path, background_writer=True)
    params["a"].grad = torch.ones_like(params["a"])
    params["b"].grad = torch.ones_like(params["b"])
    capture.note_broadcast(0, 1, local_step=0, tokens_total=0, window_steps=4)
    capture.capture_first_post_broadcast_gradients(
        local_step_before_update=0, tokens_total=0
    )
    assert capture._background_writer is not None
    _wait_until(
        lambda: capture._background_writer.snapshot().state is WriterState.FAILED
    )

    with pytest.raises(BackgroundWriterFailed) as boundary_error:
        capture.after_optimizer_step(
            local_step=1, tokens_total=128, current_window_steps=4
        )
    assert boundary_error.value.cause is injected
    with pytest.raises(BackgroundWriterFailed) as close_error:
        capture.close()
    assert close_error.value.cause is injected

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counters"]["closed"] is False
    assert manifest["counters"]["background_artifact_reserved_bytes"] == 0
    assert manifest["counters"]["drop_counts"] == {}
    assert manifest["background_writer"]["state"] == "failed"
    assert manifest["background_writer"]["failure_type"] == "OSError"
    assert manifest["background_writer"]["worker_alive"] is False
    assert capture._background_writer.thread_alive is False


def test_background_queue_full_blocks_without_capture_drop(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    original = OptimizerStateCapture._publish_background_item

    def blocked_first_publish(self, item):
        if item.sequence == 1:
            entered.set()
            assert release.wait(3.0)
        return original(self, item)

    monkeypatch.setattr(
        OptimizerStateCapture, "_publish_background_item", blocked_first_publish
    )
    params, _layout, _optimizer, capture = _fixture(
        tmp_path,
        background_writer=True,
        writer_max_items=1,
    )
    params["a"].grad = torch.ones_like(params["a"])
    params["b"].grad = torch.ones_like(params["b"])
    capture.note_broadcast(0, 1, local_step=0, tokens_total=0, window_steps=4)
    capture.capture_first_post_broadcast_gradients(
        local_step_before_update=0, tokens_total=0
    )
    assert entered.wait(3.0)

    producer_errors = []

    def produce_second():
        try:
            assert capture._write_artifact(
                "push_candidate",
                {"fragment_id": 0, "fragment_version": 1},
                {},
            )
        except BaseException as exc:
            producer_errors.append(exc)

    producer = threading.Thread(target=produce_second, name="capture-producer-test")
    producer.start()
    assert capture._background_writer is not None
    _wait_until(
        lambda: capture._background_writer.snapshot().producer_block_events == 1
    )
    while_full = json.loads((tmp_path / "manifest.json").read_text())
    assert while_full["counters"]["drop_counts"] == {}
    assert producer.is_alive()

    release.set()
    producer.join(3.0)
    assert not producer.is_alive()
    assert producer_errors == []
    capture.close()
    final = json.loads((tmp_path / "manifest.json").read_text())
    assert final["counters"]["drop_counts"] == {
        "midpoint_incomplete_at_close": 1,
        "window_unpushed_at_close": 1,
    }
    assert final["background_writer"]["producer_block_events"] == 1
    assert final["background_writer"]["completed_items"] == 2


def test_cadence_limits_schedule_change_and_disk_budget_are_manifested(tmp_path):
    params, _layout, _optimizer, capture = _fixture(
        tmp_path, max_bytes=1, every=2, max_hmc=1, max_midpoint=2
    )
    params["a"].grad = torch.ones_like(params["a"])
    params["b"].grad = torch.ones_like(params["b"])

    # Broadcast 1 is selected. Its artifact cannot fit the one-byte budget;
    # its midpoint anchor cannot be admitted either.
    capture.note_broadcast(0, 1, local_step=0, tokens_total=0, window_steps=4)
    capture.capture_first_post_broadcast_gradients(
        local_step_before_update=0, tokens_total=0
    )
    # Broadcast 2 is skipped by cadence. Broadcast 3 is selected but the HMC
    # admission cap was already consumed by broadcast 1.
    capture.note_broadcast(0, 2, local_step=1, tokens_total=1, window_steps=4)
    capture.note_broadcast(0, 3, local_step=2, tokens_total=2, window_steps=4)
    capture.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    drops = manifest["counters"]["drop_counts"]
    assert drops["adamw_first_gradient_disk_byte_limit"] == 1
    assert drops["midpoint_pending_memory_limit"] >= 1
    assert drops["hmc_event_limit"] == 1
    assert manifest["counters"]["artifact_bytes"] == 0
    assert manifest["counters"]["closed"] is True


def test_window_schedule_change_invalidates_pending_exact_window(tmp_path):
    _params, _layout, _optimizer, capture = _fixture(tmp_path)
    capture.note_window_reset(
        0, 5, local_step=10, tokens_total=1_000, window_steps=4, reason="broadcast"
    )
    capture.after_optimizer_step(
        local_step=11, tokens_total=1_100, current_window_steps=6
    )
    capture.close()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["counters"]["drop_counts"]["midpoint_window_schedule_changed"] == 1
    assert not list(tmp_path.glob("*-richardson_window-*.pt"))


def test_non_fp32_parameters_fail_closed_instead_of_silent_cast(tmp_path):
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    layout = FragmentLayout([Fragment(MERGE_RDA, [("p", 2)])])
    optimizer = torch.optim.AdamW([param])
    with pytest.raises(TypeError, match="refusing a silent cast"):
        OptimizerStateCapture(
            tmp_path,
            params={"p": param},
            layout=layout,
            optimizer=optimizer,
            learner_id=0,
            rank=0,
        )


def test_window_uuid_push_join_and_retry_mapping(tmp_path):
    params, _layout, optimizer, capture = _fixture(tmp_path)
    window_uuid = capture.note_window_reset(
        0, 17, local_step=30, tokens_total=3_000, window_steps=2, reason="broadcast"
    )
    assert window_uuid is not None and len(window_uuid) == 36

    for local_step in (31, 32):
        capture.capture_first_post_broadcast_gradients(
            local_step_before_update=local_step - 1,
            tokens_total=3_000 + (local_step - 31) * 128,
            clip_total_norm=torch.tensor(0.5),
            clip_max_norm=1.0,
        )
        with torch.no_grad():
            params["a"].add_(1.0)
            params["b"].sub_(2.0)
            for param in params.values():
                optimizer.state[param]["step"].add_(1)
        capture.after_optimizer_step(
            local_step=local_step,
            tokens_total=3_000 + (local_step - 30) * 128,
            current_window_steps=2,
        )

    endpoint = torch.cat([params["a"].detach(), params["b"].detach()])
    payload = pack_flat(endpoint, DTYPE_F32)
    common = {
        "window_uuid": window_uuid,
        "fragment_id": 0,
        "pull_global_step": 44,
        "base_version": 17,
        "local_step": 32,
        "c_steps": 2,
        "c_tokens": 256,
        "wire_codec": "f32",
        "payload": payload,
    }
    first = capture.note_push(**common)
    capture.note_push_enqueued(first["attempt_serial"])
    second = capture.note_push(**common)
    capture.note_push_enqueued(second["attempt_serial"])
    assert first["retry_identity"] == second["retry_identity"]
    assert first["retry_ordinal"] == 1
    assert second["retry_ordinal"] == 2
    assert first["attempt_serial"] == 1
    assert second["attempt_serial"] == 2
    assert first["window_uuid"] == window_uuid

    candidates = sorted(tmp_path.glob("*-push_candidate-*.pt"))
    assert len(candidates) == 2
    assert [load_capture(path)["metadata"]["retry_ordinal"] for path in candidates] == [
        1,
        2,
    ]
    with pytest.raises(CaptureIntegrityError, match="immutable capture endpoint"):
        capture.note_push(**{**common, "payload": payload[:-4] + b"nope"})

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    lifecycle = manifest["window_lifecycles"][0]
    assert lifecycle["window_uuid"] == window_uuid
    assert lifecycle["status"] == "pushed"
    assert lifecycle["push_attempts"] == 2
    assert lifecycle["enqueued_pushes"] == 2
    retry = manifest["push_retries"][first["retry_identity"]]
    assert retry["attempts"] == 2
    assert retry["candidate"]["pull_global_step"] == 44
    with pytest.raises(CaptureIntegrityError, match="finalized twice"):
        capture.note_push_enqueued(first["attempt_serial"])


def test_superseded_and_closed_windows_are_explicitly_unpushed(tmp_path):
    _params, _layout, _optimizer, capture = _fixture(tmp_path)
    first_uuid = capture.note_window_reset(
        0, 1, local_step=0, tokens_total=0, window_steps=4, reason="broadcast"
    )
    second_uuid = capture.note_window_reset(
        0, 2, local_step=1, tokens_total=128, window_steps=4, reason="broadcast"
    )
    assert first_uuid != second_uuid
    capture.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    by_uuid = {row["window_uuid"]: row for row in manifest["window_lifecycles"]}
    assert by_uuid[first_uuid]["status"] == "superseded_unpushed"
    assert by_uuid[second_uuid]["status"] == "closed_unpushed"
    drops = manifest["counters"]["drop_counts"]
    assert drops["window_superseded_unpushed"] == 1
    assert drops["window_unpushed_at_close"] == 1


def test_scaler_configuration_fails_closed(tmp_path):
    params = {"p": torch.nn.Parameter(torch.ones(1))}
    layout = FragmentLayout([Fragment(MERGE_RDA, [("p", 1)])])
    optimizer = torch.optim.AdamW(params.values())
    with pytest.raises(TypeError, match="native no-scaler"):
        OptimizerStateCapture(
            tmp_path,
            params=params,
            layout=layout,
            optimizer=optimizer,
            scaler=object(),
            learner_id=0,
            rank=0,
        )


def test_non_adamw_optimizer_fails_closed(tmp_path):
    param = torch.nn.Parameter(torch.ones(1))
    layout = FragmentLayout([Fragment(MERGE_RDA, [("p", 1)])])
    optimizer = torch.optim.SGD([param], lr=0.1)
    with pytest.raises(TypeError, match="requires torch.optim.AdamW"):
        OptimizerStateCapture(
            tmp_path,
            params={"p": param},
            layout=layout,
            optimizer=optimizer,
            learner_id=0,
            rank=0,
        )


def _strict_capture_args(*extra):
    from yeto.learner import parse_args

    return parse_args(
        [
            "--model",
            "m",
            "--data",
            "d",
            "--syncer",
            "host:1",
            "--learner-id",
            "0",
            "--num-learners",
            "1",
            "--optimizer-state-capture-dir",
            "/tmp/capture",
            "--inner-optimizer",
            "adamw",
            "--tuning",
            "lora",
            "--wire-dtype",
            "f32",
            "--merge-alpha",
            "0",
            "--inner-control-variate",
            "none",
            "--debug-broadcast-lag-commits",
            "0",
            "--max-reconnects",
            "0",
            "--fixed-window-microsteps",
            "4",
        ]
        + list(extra)
    )


def test_strict_capture_configuration_accepts_only_unambiguous_native_path():
    from yeto.learner import validate_optimizer_state_capture_args

    validate_optimizer_state_capture_args(_strict_capture_args())


def test_directional_capture_cli_requires_explicit_zero_hmc_cap():
    from yeto.learner import validate_optimizer_state_capture_args

    args = _strict_capture_args(
        "--optimizer-state-capture-profile",
        "crp_pti_directional",
        "--optimizer-state-capture-max-hmc-events",
        "0",
    )
    validate_optimizer_state_capture_args(args)
    assert args.optimizer_state_capture_profile == CAPTURE_PROFILE_CRP_PTI_DIRECTIONAL

    with pytest.raises(SystemExit):
        _strict_capture_args("--optimizer-state-capture-profile", "crp_pti_directional")


def test_background_writer_cli_is_explicit_and_validates_positive_caps():
    from yeto.learner import validate_optimizer_state_capture_args

    args = _strict_capture_args(
        "--optimizer-state-capture-background-writer",
        "--optimizer-state-capture-writer-max-items",
        "3",
        "--optimizer-state-capture-writer-max-bytes",
        "123456",
    )
    validate_optimizer_state_capture_args(args)
    assert args.optimizer_state_capture_background_writer is True
    assert args.optimizer_state_capture_writer_max_items == 3
    assert args.optimizer_state_capture_writer_max_bytes == 123456


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("syncer", "none", "async --syncer"),
        ("tuning", "full", "--tuning lora"),
        ("inner_optimizer", "sgd", "--inner-optimizer adamw"),
        ("wire_dtype", "bf16", "--wire-dtype f32"),
        ("merge_alpha", 0.5, "--merge-alpha 0"),
        ("inner_control_variate", "scaffold_lite", "--inner-control-variate none"),
        ("debug_broadcast_lag_commits", 1, "--debug-broadcast-lag-commits 0"),
        ("max_reconnects", None, "--max-reconnects 0"),
        ("fixed_window_microsteps", 3, "fixed even --fixed-window-microsteps"),
        ("fixed_window_tokens", 1, "fixed even --fixed-window-microsteps"),
        ("fixed_window_schedule", [(0, 4)], "fixed even --fixed-window-microsteps"),
    ],
)
def test_ambiguous_capture_configuration_fails_closed(field, value, expected):
    from yeto.learner import validate_optimizer_state_capture_args

    args = _strict_capture_args()
    setattr(args, field, value)
    with pytest.raises(RuntimeError, match=expected):
        validate_optimizer_state_capture_args(args)


def test_fixed_window_snapshot_carries_immutable_uuid_and_endpoint_copy():
    from yeto.learner import make_fixed_window_snapshot

    params = {"p": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}
    fragment = Fragment(MERGE_RDA, [("p", 2)])
    anchor = torch.tensor([9.0, 8.0])
    window_uuid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    snapshot = make_fixed_window_snapshot(
        fragment,
        params,
        anchor=anchor,
        c_steps=4,
        c_tokens=512,
        local_step=12,
        base_version=7,
        window_uuid=window_uuid,
    )
    params["p"].data.add_(100)
    anchor.add_(100)
    assert snapshot["window_uuid"] == window_uuid
    assert torch.equal(snapshot["flat"], torch.tensor([1.0, 2.0]))
    assert torch.equal(snapshot["anchor"], torch.tensor([9.0, 8.0]))
    assert snapshot["c_steps"] == 4
    assert snapshot["c_tokens"] == 512


def test_candidate_identity_encodes_exactly_into_audited_wire_header(tmp_path):
    import uuid

    from yeto.learner import push_audit_from_candidate

    params, _layout, optimizer, capture = _fixture(tmp_path)
    window_uuid = capture.note_window_reset(
        0, 3, local_step=0, tokens_total=0, window_steps=2, reason="broadcast"
    )
    for local_step in (1, 2):
        capture.capture_first_post_broadcast_gradients(
            local_step_before_update=local_step - 1,
            tokens_total=(local_step - 1) * 16,
            clip_total_norm=torch.tensor(0.5),
        )
        with torch.no_grad():
            for param in params.values():
                param.add_(0.25)
                optimizer.state[param]["step"].add_(1)
        capture.after_optimizer_step(
            local_step=local_step,
            tokens_total=local_step * 16,
            current_window_steps=2,
        )
    endpoint = torch.cat([params["a"].detach(), params["b"].detach()])
    payload = pack_flat(endpoint, DTYPE_F32)
    candidate = capture.note_push(
        window_uuid=window_uuid,
        fragment_id=0,
        pull_global_step=5,
        base_version=3,
        local_step=2,
        c_steps=2,
        c_tokens=32,
        wire_codec="f32",
        payload=payload,
    )
    audit = push_audit_from_candidate(candidate)
    assert audit.window_uuid == uuid.UUID(window_uuid).bytes
    assert audit.attempt_serial == candidate["attempt_serial"]
    assert audit.payload_sha256.hex() == candidate["payload_sha256"]
    capture.close()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["window_lifecycles"][0]["status"] == "closed_unpushed"
    assert (
        manifest["push_attempts"][str(candidate["attempt_serial"])]["enqueued"] is False
    )
