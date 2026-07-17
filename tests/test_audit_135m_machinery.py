from __future__ import annotations

import argparse
import json
import struct
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import audit_135m_contract as audit
from scripts import audit_135m_campaign_controller as controller
from scripts import audit_135m_kernel_law as kernel
from scripts import audit_135m_kernel_capture as capture
from scripts import audit_135m_hidden_evaluator as hidden
from scripts import audit_135m_phase_manifest as promotion
from scripts import audit_135m_report as final_report
from scripts import audit_135m_replay as replay
from scripts import run_parallel_phase_map as parallel


def test_ceiling_amendment_covers_full_survivable_stage_paths() -> None:
    authority = audit.load_authority()
    price = controller.PRICE_PER_VM_HOUR["us-west4"]["a2-highgpu-1g"]
    a1 = authority["costs"]["blocks"]["A1"]
    assert a1["corrected_cheapest_complete_1g_path_usd_approx"] == 106.0
    assert a1["range_usd"] == [106.0, 132.5]
    assert a1["hard_ceiling_usd"] == 140.0
    assert a1["width_cap"] == 2
    assert a1["abort_burn_kill_usd"] == 40.0
    hours = {
        "A3": 10 * controller.CELL_HOURS[8]
        + 12 * controller.CELL_HOURS[512]
        + controller.CELL_HOURS[16]
        + controller.FINITE_KERNEL_EXTRA_HOURS[16]
        + controller.CELL_HOURS[64]
        + controller.FINITE_KERNEL_EXTRA_HOURS[64]
        + controller.CELL_HOURS[256]
        + controller.FINITE_KERNEL_EXTRA_HOURS[256],
        "A4": 80 * controller.CELL_HOURS[16]
        + 80 * controller.CELL_HOURS[256],
    }
    for stage, stage_hours in hours.items():
        block = authority["costs"]["blocks"][stage]
        lower_bound = stage_hours * price
        assert block["a100_hours_lower_bound"] == pytest.approx(stage_hours)
        assert block["lower_bound_usd"] == pytest.approx(lower_bound)
        assert block["range_usd"][1] == pytest.approx(
            lower_bound * controller.SPOT_PREEMPTION_RESERVE_FACTOR
        )
        assert block["hard_ceiling_usd"] >= lower_bound * 1.30
        assert block["hard_ceiling_usd"] - lower_bound * 1.30 < 0.01

    assert sum(
        float(block["hard_ceiling_usd"])
        for block in authority["costs"]["blocks"].values()
    ) == pytest.approx(2485.0)
    assert authority["costs"]["sum_of_block_hard_ceilings_usd"] == 2485.0
    assert authority["costs"]["program_hard_ceiling_usd"] == 2485.0

    assert controller.FUTURE_STAGE_CELL_COUNTS["a1d"] == {16: 36, 256: 36}
    assert controller.FUTURE_STAGE_CELL_COUNTS["a3k"] == {8: 10, 512: 12}
    assert controller.FUTURE_STAGE_CELL_COUNTS["a4d"] == {16: 56, 256: 56}


def test_abort_burn_guard_stops_only_pre_science_aborted_launch_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign-root"
    (campaign / "campaign").mkdir(parents=True)
    ledger = {
        "pre_science_aborted_launch_spend_usd": 39.5,
        "abort_burn_kill_usd": 40.0,
    }
    monkeypatch.setattr(
        controller,
        "_current_campaign_cost",
        lambda _executor, _campaign_root: (0.51, []),
    )
    with pytest.raises(controller.AbortBurnStop, match="exceeds"):
        controller._guard_abort_burn(
            executor=SimpleNamespace(),
            campaign_root=campaign,
            stage_ledger=ledger,
            phase="test",
        )
    evidence = json.loads((campaign / "campaign" / "abort-burn-guard.json").read_text())
    assert evidence["status"] == "STOP"
    assert evidence["projected_pre_science_aborted_launch_spend_usd"] == 40.01

    partial = campaign / "vms" / "v0" / "g1" / "manifests" / "vm-partial-manifest.json"
    partial.parent.mkdir(parents=True)
    partial.write_text(json.dumps({"attempts": [{"attempt_id": "science-started"}]}))
    assert controller._guard_abort_burn(
        executor=SimpleNamespace(),
        campaign_root=campaign,
        stage_ledger=ledger,
        phase="after-science",
    ) == 39.5


def test_deferred_evaluation_modes_are_exact() -> None:
    assert (
        controller._deferred_evaluation_role(
            {"cells": [{"evaluation_mode": "development_endpoint"}]}
        )
        is None
    )
    assert (
        controller._deferred_evaluation_role(
            {"cells": [{"evaluation_mode": "confirmation_audit_pending"}]}
        )
        == "confirmation_audit"
    )
    assert (
        controller._deferred_evaluation_role(
            {"cells": [{"evaluation_mode": "development_prediction_pending"}]}
        )
        == "development_prediction_endpoint"
    )
    with pytest.raises(controller.ControllerError, match="cannot mix"):
        controller._deferred_evaluation_role(
            {
                "cells": [
                    {"evaluation_mode": "confirmation_audit_pending"},
                    {"evaluation_mode": "development_prediction_pending"},
                ]
            }
        )


def test_reboot_restoration_binds_reviewed_backend_to_tracked_workspace() -> None:
    base = SimpleNamespace(WORKSPACE=Path("/private/tmp/missing-helper-tree"))
    controller._bind_reviewed_backend_workspace(base)
    assert base.WORKSPACE == controller.REPO_ROOT
    assert (base.WORKSPACE / "scripts" / "run_parallel_phase_map.py").is_file()


def test_hidden_remote_command_uses_guest_paths() -> None:
    command = controller._remote_hidden_command(
        identity=SimpleNamespace(run_id="run"),
        backend=object(),
        science_root="/tmp/audit-135m-science/example",
        evaluation_role="development_prediction_endpoint",
        first_seed=359,
        remote_authorization="/home/shou/auth.json",
        remote_preseal="/home/shou/preseal.json",
        remote_bound="/home/shou/bound.json",
        remote_output="/tmp/hidden/batch-attempt-1",
    )
    assert "/home/shou/venv/bin/python" in command
    assert "/tmp/yeto-best-paper/scripts/audit_135m_hidden_evaluator.py" in command
    assert "/phase-map/frozen-eval/seed-359/materialized/eval.jsonl" in command
    assert "/phase-map/frozen-eval/seed-359/parallel-eval-freeze.json" in command
    assert "--source-data" not in command


def test_kernel_chronology_parses_instants_not_lexical_text() -> None:
    earlier = kernel._parse_utc("2026-07-16T23:59:59.900000Z", "earlier")
    later = kernel._parse_utc("2026-07-17T00:00:00Z", "later")
    assert earlier < later
    with pytest.raises(kernel.KernelLawError, match="UTC Z"):
        kernel._parse_utc("2026-07-17T00:00:00+00:00", "bad")


def test_hidden_divergence_row_is_json_safe() -> None:
    expected = {
        "cell": {
            "cell_id": "cell",
            "evaluation_mode": "confirmation_audit_pending",
            "h": 16,
            "m": 4,
            "mu": 0.9,
            "eta": 0.01,
            "seed": 383,
            "training_seed": 383383,
            "audit_stage": "A1",
            "audit_phase": "confirmation",
            "analysis_role": "tuned_method",
            "pair_key": "pair",
            "paired_control_id": "control",
            "block_id": "block",
        }
    }
    campaign = {
        "attempts": [
            {
                "attempt_id": "cell-attempt-1",
                "cell_id": "cell",
                "attempt": 1,
                "status": "DIVERGED",
            }
        ],
        "analysis_rounds": {"cell": {"attempt_id": "cell-attempt-1"}},
    }
    authorization = {"evaluation_order": ["cell"]}
    bundle = {
        "evaluation_role": "confirmation_audit",
        "bundle_canonical_sha256": "a" * 64,
        "surface": {},
        "results": [
            {
                "cell_id": "cell",
                "training_status": "DIVERGED",
                "audit_status": "SCIENTIFIC_DIVERGENCE",
                "audit_loss": None,
                "analysis_loss_kind": "positive_infinity_scientific_divergence",
                "started_at_utc": "2026-07-17T00:00:00Z",
                "ended_at_utc": "2026-07-17T00:00:00Z",
            }
        ],
    }
    rows = promotion._hidden_rows(
        bundle=bundle,
        authorization=authorization,
        campaign=campaign,
        expected=expected,
    )
    assert rows[0]["analysis_loss"] is None
    json.dumps(rows, allow_nan=False)


def test_promotion_rejects_non_authorized_descriptor(tmp_path: Path) -> None:
    descriptor = tmp_path / "aggregation-descriptor.json"
    descriptor.write_text(json.dumps({"aggregation_authorized": False}))
    with pytest.raises(promotion.PromotionError, match="does not authorize"):
        promotion._verify_campaign(
            descriptor=descriptor,
            campaign_manifest_path=tmp_path / "missing-manifest.json",
            campaign_seal_path=tmp_path / "missing-seal.json",
        )


def test_cost_watchdog_exact_kill_trigger_is_loss_blind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller, "WATCHDOG_POLL_SECONDS", 0.01)
    monkeypatch.setattr(
        controller,
        "_current_campaign_cost",
        lambda executor, campaign_root: (9.8, []),
    )
    monkeypatch.setattr(controller, "_active_teardown_reserve", lambda executor: 0.3)
    monkeypatch.setattr(
        controller,
        "_global_a100_census",
        lambda backend: {"total_attached_a100_equivalent": 0},
    )
    deleted = [
        {
            "slot": "v0",
            "generation": 1,
            "instance_numeric_id": "123",
            "boot_disk_numeric_id": "456",
        }
    ]
    monkeypatch.setattr(
        controller,
        "_emergency_exact_delete_active",
        lambda **kwargs: deleted,
    )
    notes: list[str] = []
    backend = SimpleNamespace(note=notes.append)
    watchdog = controller.CostWatchdog(
        executor=object(),
        backend=backend,
        campaign_root=tmp_path,
        stage_ledger={"estimated_spend_usd": 0.0},
        ceiling=10.0,
        lifecycle_lock=threading.RLock(),
    )
    watchdog.start()
    watchdog.thread.join(timeout=2)
    assert not watchdog.thread.is_alive()
    with pytest.raises(controller.HardCeilingStop):
        watchdog.raise_if_triggered()
    evidence = json.loads(
        (tmp_path / "campaign" / "hard-ceiling-stop.json").read_text()
    )
    assert evidence["loss_inspected"] is False
    assert evidence["deleted_generations"] == deleted
    assert notes and "HARD CEILING STOP" in notes[0]


def test_active_teardown_reserve_sums_every_live_shape() -> None:
    executor = SimpleNamespace(
        active={
            "v0": SimpleNamespace(generation=1),
            "v1": SimpleNamespace(generation=2),
        },
        providers={
            ("v0", 1): {"region": "us-east1", "machine_type": "a2-highgpu-1g"},
            ("v1", 2): {"region": "us-west4", "machine_type": "a2-highgpu-4g"},
        },
    )
    expected = (
        controller.PRICE_PER_VM_HOUR["us-east1"]["a2-highgpu-1g"]
        + controller.PRICE_PER_VM_HOUR["us-west4"]["a2-highgpu-4g"]
    ) * controller.TEARDOWN_RESERVE_SECONDS / 3600.0
    expected += controller.TEARDOWN_RESERVE_FIXED_USD
    assert controller._active_teardown_reserve(executor) == pytest.approx(expected)


def test_shape_selection_keeps_only_cost_eligible_landed_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = SimpleNamespace(
        active={
            "v0": SimpleNamespace(generation=1),
            "v1": SimpleNamespace(generation=1),
        },
        providers={
            ("v0", 1): {"region": "us-east1", "machine_type": "a2-highgpu-1g"},
            ("v1", 1): {"region": "us-west4", "machine_type": "a2-highgpu-1g"},
        },
    )
    monkeypatch.setattr(
        controller,
        "_shape_stage_forecast",
        lambda **kwargs: 80.0 if kwargs["region"] == "us-east1" else 70.0,
    )
    shape, slots, forecast = controller._select_cost_eligible_shape(
        executor=executor,
        stage_code="a1d",
        ceiling=75.0,
        prior_spend=0.0,
        current_spend=1.0,
        plan={},
        scientific={},
        planned_index=0,
        target_width=3,
    )
    assert shape == "a2-highgpu-1g"
    assert slots == ("v1",)
    assert forecast == 70.0


def test_remaining_cost_preflight_rejects_before_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forecast(**kwargs) -> float:
        if (
            kwargs["region"] == "us-west4"
            and kwargs["machine_type"] == "a2-highgpu-1g"
        ):
            return 72.852615
        return 77.303160

    monkeypatch.setattr(controller, "_shape_stage_forecast", forecast)
    provisioned: list[str] = []
    notes: list[str] = []

    def provision(identity):
        provisioned.append(identity.run_id)
        raise AssertionError("provider mutation must not be reached")

    backend = SimpleNamespace(
        audit_transient_generations=set(),
        note=notes.append,
        provision=provision,
    )
    registry = SimpleNamespace(
        identities=[],
        _prebound_nonces={"v0": "1" * 32},
        stage_code="a1d",
        study_id="study",
        roster_hash="a" * 64,
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root="gs://bucket/campaign",
    )
    executor = SimpleNamespace(active={}, providers={})
    evidence_path = (
        tmp_path / "campaign-root" / "campaign" / "cost-feasibility-stop.json"
    )

    def pre_launch_guard() -> None:
        controller._guard_provider_mutation_cost(
            executor=executor,
            campaign_root=tmp_path / "campaign-root",
            stage_ledger={"estimated_spend_usd": 3.884564},
            stage_code="a1d",
            plan={},
            scientific={},
            planned_index=0,
            ceiling=75.0,
            phase="test_prelaunch",
        )

    errors: list[BaseException] = []
    controller._audit_provision_loop(
        base=SimpleNamespace(),
        p1=SimpleNamespace(ZONE_ROTATION=("us-west4-b",)),
        pexec=SimpleNamespace(GenerationIdentity=parallel.GenerationIdentity),
        executor=executor,
        registry=registry,
        backend=backend,
        slot="v0",
        stop_event=threading.Event(),
        science_started=threading.Event(),
        lock=threading.RLock(),
        errors=errors,
        required=False,
        pre_launch_guard=pre_launch_guard,
    )

    assert provisioned == []
    assert len(errors) == 1
    assert isinstance(errors[0], controller.HardCeilingStop)
    evidence = json.loads(evidence_path.read_text())
    assert evidence["provider_mutation_authorized"] is False
    assert evidence["estimated_stage_spend_if_continued_usd"] == pytest.approx(
        76.737179
    )
    assert evidence["cheapest_registered_remaining_forecast"] == {
        "forecast_usd": 72.852615,
        "machine_type": "a2-highgpu-1g",
        "region": "us-west4",
    }
    assert len(evidence["registered_remaining_forecasts"]) == 8
    assert notes and "before provider mutation" in notes[-1]


def test_cost_eligible_zone_rotation_omits_unaffordable_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller,
        "_shape_stage_forecast",
        lambda **kwargs: 70.0 if kwargs["region"] == "us-west4" else 80.0,
    )

    zones = controller._cost_eligible_zone_rotation(
        stage_code="a1d",
        machine_type="a2-highgpu-1g",
        plan={},
        scientific={},
        planned_index=0,
        prior_stage_spend=4.0,
        current_campaign_spend=0.0,
        ceiling=75.0,
    )

    assert zones == ("us-west4-b", "us-west4-a")


def test_global_a100_count_uses_accelerator_inventory_or_machine_shape() -> None:
    assert (
        controller._instance_a100_count(
            {
                "guestAccelerators": [
                    {
                        "acceleratorType": "projects/x/zones/y/acceleratorTypes/nvidia-tesla-a100",
                        "acceleratorCount": 4,
                    }
                ]
            }
        )
        == 4
    )
    assert (
        controller._instance_a100_count(
            {"machineType": "projects/x/zones/y/machineTypes/a2-highgpu-8g"}
        )
        == 8
    )
    assert controller._instance_a100_count({"machineType": "e2-standard-8"}) == 0


def test_launch_census_deduplicates_provider_visible_pending_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "preferred-1g.json"
    spec.write_text(
        json.dumps({"cloud": {"machine_type": "a2-highgpu-1g"}})
    )
    visible: dict[str, int] = {}
    release = threading.Event()
    visible_events = {name: threading.Event() for name in ("run-a", "run-b")}
    errors: list[BaseException] = []

    class Backend:
        def __init__(self) -> None:
            self.audit_direct_1g_authorized_zones: dict[str, dict] = {}

        @staticmethod
        def zone_for_name(run_id: str) -> str:
            return "us-central1-a"

        @staticmethod
        def _packet_paths(identity):
            return spec, None, None, None

        def provision(self, identity):
            if identity.run_id in visible_events:
                visible[identity.run_id] = 1
                visible_events[identity.run_id].set()
                assert release.wait(timeout=2)
            return {
                "run_id": identity.run_id,
                "machine_type": "a2-highgpu-1g",
                "zone": "us-central1-a",
                "region": "us-central1",
                "instance_numeric_id": identity.run_id,
            }

    backend = Backend()
    monkeypatch.setattr(
        controller,
        "_global_a100_census",
        lambda backend: {
            "total_attached_a100_equivalent": 12 + sum(visible.values()),
            "instances": [{"name": "foreign-a100", "a100_count": 12}]
            + [
                {"name": name, "a100_count": count}
                for name, count in visible.items()
            ],
        },
    )
    controller._install_launch_census_and_direct_fallback_patch(
        base=SimpleNamespace(), backend=backend, campaign_root=tmp_path
    )

    def launch(run_id: str) -> None:
        try:
            backend.provision(SimpleNamespace(run_id=run_id, slot="v0", generation=1))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=launch, args=("run-a",))
    first.start()
    assert visible_events["run-a"].wait(timeout=2)
    second = threading.Thread(target=launch, args=("run-b",))
    second.start()
    assert visible_events["run-b"].wait(timeout=2)

    third = backend.provision(
        SimpleNamespace(run_id="run-c", slot="v2", generation=1)
    )
    assert third["run_id"] == "run-c"
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert backend.audit_pending_launches == {}
    assert backend.audit_pending_launch_a100s == 0


def test_launch_census_still_counts_unseen_pending_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / "preferred-1g.json"
    spec.write_text(
        json.dumps({"cloud": {"machine_type": "a2-highgpu-1g"}})
    )
    provisioned: list[str] = []
    backend = SimpleNamespace(
        provision=lambda identity: provisioned.append(identity.run_id),
        zone_for_name=lambda run_id: "us-central1-a",
        _packet_paths=lambda identity: (spec, None, None, None),
    )
    monkeypatch.setattr(
        controller,
        "_global_a100_census",
        lambda backend: {
            "total_attached_a100_equivalent": 12,
            "instances": [{"name": "visible-foreign", "a100_count": 12}],
        },
    )
    controller._install_launch_census_and_direct_fallback_patch(
        base=SimpleNamespace(), backend=backend, campaign_root=tmp_path
    )
    backend.audit_pending_launches.update(
        {"unseen-a": 1, "unseen-b": 1, "unseen-c": 1, "unseen-d": 1}
    )
    backend.audit_pending_launch_a100s = 4

    with pytest.raises(controller.ControllerError, match="global 16-A100 ceiling"):
        backend.provision(
            SimpleNamespace(run_id="run-d", slot="v3", generation=1)
        )

    assert provisioned == []
    assert backend.audit_pending_launch_a100s == 4


def test_audit_packet_specs_prefer_one_gpu_spot_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_audit_135m_launch_packet as packet

    monkeypatch.setattr(packet, "_configure_base", lambda: None)
    monkeypatch.setattr(
        packet._base,
        "spec_value",
        lambda **kwargs: {
            "cloud": {
                "machine_type": "a2-highgpu-4g",
                "accelerator_count": 4,
                "provisioning_model": "SPOT",
                "labels": {},
            },
            "execution": {},
        },
    )
    monkeypatch.setitem(packet._ACTIVE, "stage_code", "a1d")

    value = packet.spec_value()

    assert packet.PREFERRED_PROBE_A100S == 1
    assert controller.PREFERRED_PROBE_A100S == 1
    assert value["cloud"]["machine_type"] == "a2-highgpu-1g"
    assert value["cloud"]["accelerator_count"] == 1


def test_transient_pending_create_is_exactly_proved_and_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controller, "TRANSIENT_PROVIDER_POLL_SECONDS", 0.001)
    nonce = "1" * 32
    run_id = "bp-a1d-test-c1-v1-g1"
    labels = {
        "campaign": "audit-135m",
        "campaign-tag": "a" * 16,
        "draft": "false",
        "logical-slot": "v1",
        "managed-by": "yeto-optimizer-harness",
        "physical-generation": "1",
        "run-id": run_id,
        "stage": "a1d",
    }
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "cloud": {
                    "instance_name": run_id,
                    "zone": "us-central1-a",
                    "project": "model-training-497007",
                    "machine_type": "a2-highgpu-4g",
                    "accelerator_count": 4,
                    "labels": labels,
                },
                "artifacts": {"uri": f"gs://bucket/prefix/{run_id}"},
            }
        )
    )

    class FakeBackend:
        def __init__(self) -> None:
            self.pexec = parallel
            self.live = False
            self.disk_seen = threading.Event()
            self.notes: list[str] = []

        def run(self, command, *, check=True, timeout=None):
            values = list(command)
            if "yeto.optimizer_harness" in values and "launch" in values:
                self.live = True
                assert self.disk_seen.wait(2)
                self.live = False
                raise RuntimeError("OPERATION_CANCELED_BY_USER")
            if values[:4] == ["gcloud", "compute", "instances", "list"]:
                return subprocess.CompletedProcess(values, 0, "[]", "")
            if values[:4] == ["gcloud", "compute", "disks", "describe"]:
                return subprocess.CompletedProcess(values, 1, "", "NOT_FOUND")
            if values[:4] == ["gcloud", "compute", "operations", "list"]:
                return subprocess.CompletedProcess(
                    values,
                    0,
                    json.dumps(
                        [
                            {
                                "name": "operation-1",
                                "operationType": "insert",
                                "status": "DONE",
                                "targetId": "123",
                                "targetLink": f"zones/us-central1-a/instances/{run_id}",
                                "error": {
                                    "errors": [
                                        {"code": "OPERATION_CANCELED_BY_USER"}
                                    ]
                                },
                            }
                        ]
                    ),
                    "",
                )
            if values[:3] == ["gcloud", "storage", "cp"]:
                return subprocess.CompletedProcess(values, 0, "", "")
            return subprocess.CompletedProcess(values, 0, "", "")

        def gcloud(self, *args, check=True, timeout=None):
            return self.run(
                ["gcloud", *args], check=check, timeout=timeout
            )

        def describe_instance(self, name, *, check=True):
            if not self.live:
                return None
            return {
                "id": "123",
                "name": name,
                "zone": "zones/us-central1-a",
                "machineType": "zones/us-central1-a/machineTypes/a2-highgpu-4g",
                "creationTimestamp": "2026-07-17T11:00:00-07:00",
                "labels": {**labels, "ownership-nonce": nonce},
                "scheduling": {"provisioningModel": "SPOT"},
                "guestAccelerators": [
                    {
                        "acceleratorType": "acceleratorTypes/nvidia-tesla-a100",
                        "acceleratorCount": 4,
                    }
                ],
                "disks": [
                    {
                        "source": f"zones/us-central1-a/disks/{run_id}",
                        "autoDelete": True,
                    }
                ],
            }

        def _disk_description(self, instance):
            self.disk_seen.set()
            return {
                "name": run_id,
                "id": "456",
                "sourceImageId": "789",
            }

        def note(self, value: str) -> None:
            self.notes.append(value)

    backend = FakeBackend()
    campaign_root = tmp_path / "campaign-root"
    controller._install_transient_provider_generation_monitor(
        backend=backend, campaign_root=campaign_root
    )
    command = [
        "/explicit/python",
        "-m",
        "yeto.optimizer_harness",
        "--state-dir",
        str(tmp_path / "state"),
        "launch",
        str(spec),
        "--ownership-nonce",
        nonce,
        "--yes",
    ]
    with pytest.raises(controller.TransientProviderGeneration) as captured:
        backend.run(command)
    lifecycle = captured.value.lifecycle
    assert lifecycle["instance_numeric_id"] == "123"
    assert lifecycle["boot_disk_numeric_id"] == "456"
    assert lifecycle["provider_spot_preempted"] is True
    assert lifecycle["scientific_attempt_started"] is False
    assert lifecycle["loss_inspected"] is False
    assert ("v1", 1) in backend.audit_transient_generations
    lifecycle_path = (
        campaign_root
        / "common"
        / "transient-provider-lifecycles"
        / f"{run_id}.json"
    )
    assert lifecycle_path.is_file()
    assert backend.notes and "may never be retried" in backend.notes[-1]


def test_transient_generation_forces_fresh_identity(tmp_path: Path) -> None:
    registry = SimpleNamespace(
        identities=(),
        _prebound_nonces={"v1": "1" * 32},
        stage_code="a1d",
        study_id="study",
        roster_hash="a" * 64,
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root="gs://bucket/campaign",
    )
    backend = SimpleNamespace(audit_transient_generations={('v1', 1)})
    identity = controller._audit_provisional_identity(
        pexec=parallel, registry=registry, backend=backend, slot="v1"
    )
    assert identity.generation == 2
    assert identity.run_id.endswith("-v1-g2")
    assert identity.ownership_nonce != registry._prebound_nonces["v1"]


def test_transient_provider_recovery_is_per_slot_and_preserves_ready_sibling(
    tmp_path: Path,
) -> None:
    roster_hash = "a" * 64
    sibling = parallel.GenerationIdentity(
        stage_code="a1d",
        study_id="study",
        roster_hash=roster_hash,
        campaign_attempt=1,
        slot="v0",
        generation=1,
        ownership_nonce="0" * 32,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root="gs://bucket/campaign",
    )
    registry = SimpleNamespace(
        identities=[sibling],
        _prebound_nonces={"v1": "1" * 32},
        stage_code="a1d",
        study_id="study",
        roster_hash=roster_hash,
        campaign_attempt=1,
        campaign_state_root=tmp_path / "state",
        campaign_artifact_root="gs://bucket/campaign",
    )
    sibling_provider = {
        "run_id": sibling.run_id,
        "instance_numeric_id": "100",
        "region": "us-central1",
        "zone": "us-central1-a",
        "machine_type": "a2-highgpu-4g",
    }
    finalized: list[tuple[str, bool]] = []
    executor = SimpleNamespace(
        active={"v0": sibling},
        providers={("v0", 1): sibling_provider},
        ready_at={},
        _finalize_identity=lambda identity, preempted: finalized.append(
            (identity.run_id, preempted)
        ),
    )

    class Backend:
        def __init__(self) -> None:
            self.audit_transient_generations: set[tuple[str, int]] = set()
            self.notes: list[str] = []

        def provision(self, identity):
            if identity.generation == 1:
                self.audit_transient_generations.add((identity.slot, 1))
                raise controller.TransientProviderGeneration(
                    {
                        "run_id": identity.run_id,
                        "slot": identity.slot,
                        "generation": identity.generation,
                        "ownership_nonce": identity.ownership_nonce,
                    }
                )
            return {
                "run_id": identity.run_id,
                "instance_numeric_id": "200",
                "region": "us-central1",
                "zone": "us-central1-b",
                "machine_type": "a2-highgpu-4g",
            }

        def note(self, value: str) -> None:
            self.notes.append(value)

        @staticmethod
        def describe_instance(name, *, check=True):
            return None

    backend = Backend()

    def initialize_registered_generation(**kwargs) -> None:
        identity = kwargs["identity"]
        provider = kwargs["provider"]
        registry.identities.append(identity)
        executor.providers[(identity.slot, identity.generation)] = provider

    def mark_ready(**kwargs) -> str:
        identity = kwargs["identity"]
        ready = "2026-07-17T18:00:00Z"
        executor.ready_at[(identity.slot, identity.generation)] = ready
        return ready

    p1 = SimpleNamespace(
        ZONE_ROTATION=("us-central1-a", "us-central1-b"),
        RETRY_SECONDS=0,
        cache_zone_render=lambda backend, identity, zone: None,
        initialize_registered_generation=initialize_registered_generation,
        mark_ready=mark_ready,
    )
    pexec = SimpleNamespace(
        GenerationIdentity=parallel.GenerationIdentity,
        region_for_zone=lambda zone: zone.rsplit("-", 1)[0],
        validate_provider_record=lambda provider, identity: None,
    )
    errors: list[BaseException] = []
    controller._audit_provision_loop(
        base=SimpleNamespace(HARNESS_STATE_ROOT=tmp_path / "harness"),
        p1=p1,
        pexec=pexec,
        executor=executor,
        registry=registry,
        backend=backend,
        slot="v1",
        stop_event=threading.Event(),
        science_started=threading.Event(),
        lock=threading.RLock(),
        errors=errors,
        required=False,
    )
    assert errors == []
    assert finalized == []
    assert executor.active["v0"] is sibling
    assert executor.providers[("v0", 1)] is sibling_provider
    assert executor.active["v1"].generation == 2
    assert executor.active["v1"].ownership_nonce != "1" * 32
    assert any("permanently consumed" in note for note in backend.notes)


def test_transient_generation_cost_is_included(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign-root"
    lifecycle_root = campaign_root / "common" / "transient-provider-lifecycles"
    lifecycle_root.mkdir(parents=True)
    lifecycle = {
        "schema": "audit_135m_transient_provider_lifecycle_v1",
        "status": "TRANSIENT_PROVIDER_PREEMPTED_AND_EXACT_IDS_ABSENT",
        "run_id": "run-v1-g1",
        "slot": "v1",
        "generation": 1,
        "instance_numeric_id": "123",
        "boot_disk_numeric_id": "456",
        "region": "us-central1",
        "zone": "us-central1-a",
        "machine_type": "a2-highgpu-4g",
        "creation_timestamp": "2026-07-17T18:00:00Z",
        "deletion_completed_at_utc": "2026-07-17T19:00:00Z",
        "scientific_attempt_started": False,
        "loss_inspected": False,
    }
    (lifecycle_root / "run-v1-g1.json").write_text(json.dumps(lifecycle))
    total, rows = controller._current_campaign_cost(
        SimpleNamespace(providers={}), campaign_root
    )
    assert total == pytest.approx(
        controller.PRICE_PER_VM_HOUR["us-central1"]["a2-highgpu-4g"]
    )
    assert rows == [
        {
            "slot": "v1",
            "generation": 1,
            "run_id": "run-v1-g1",
            "instance_numeric_id": "123",
            "region": "us-central1",
            "zone": "us-central1-a",
            "machine_type": "a2-highgpu-4g",
            "creation_timestamp": "2026-07-17T18:00:00Z",
            "ended_at_utc": "2026-07-17T19:00:00Z",
            "status": "TRANSIENT_PRE_READY_FINAL",
            "estimated_cost_usd": 7.712,
            "lifecycle_raw_sha256": controller._sha256_file(
                lifecycle_root / "run-v1-g1.json"
            ),
        }
    ]


def _write_capture_state(path: Path, step: int, values: list[float]) -> None:
    payload = bytearray()
    payload.extend(struct.pack("<I", capture.CKPT_MAGIC))
    payload.extend(struct.pack("<Q", step))
    payload.extend(struct.pack("<I", 1))
    payload.extend(struct.pack("<QQ", step, len(values)))
    payload.extend(struct.pack(f"<{len(values)}f", *values))
    payload.extend(struct.pack(f"<{len(values)}f", *([0.0] * len(values))))
    path.write_bytes(bytes(payload))


def test_tiny_finite_kernel_capture_is_exact_and_cleans_large_state(
    tmp_path: Path,
) -> None:
    capture_dir = tmp_path / "capture"
    states = capture_dir / "states"
    candidates = capture_dir / "candidates"
    states.mkdir(parents=True)
    candidates.mkdir()
    _write_capture_state(states / "state_before_step_1.ckpt", 0, [0.0, 0.0])
    _write_capture_state(states / "state_before_step_2.ckpt", 1, [1.0, 0.0])
    final = tmp_path / "final.ckpt"
    _write_capture_state(final, 2, [2.0, 1.0])
    (candidates / "unused.f32").write_bytes(b"unused")
    index = capture_dir / "index.jsonl"
    index.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema": "syncer_probe_capture_v1",
                    "step": step,
                    "learner_id": 0,
                    "fragment": 0,
                }
            )
            for step in (1, 2)
        )
        + "\n"
    )
    tape = tmp_path / "tape.jsonl"
    tape.write_text(
        json.dumps({"step": 1, "fragment": 0})
        + "\n"
        + json.dumps({"step": 2, "fragment": 0})
        + "\n"
    )
    done = tmp_path / "done"
    done.write_text("done\n")
    scratch = tmp_path / "scratch"
    output = tmp_path / "finite-kernel.json"
    status = tmp_path / "status.json"
    result = capture.capture(
        argparse.Namespace(
            capture_dir=capture_dir,
            final_checkpoint=final,
            event_tape=tape,
            done_file=done,
            scratch_dir=scratch,
            output=output,
            status=status,
            expected_outer_steps=2,
            fragment_count=1,
            learner_count=1,
            outer_eta=0.1,
            coordinate_chunk=1,
            poll_seconds=0.001,
        )
    )
    assert result["K_H"] == 2
    assert result["observed_outer_steps"] == 2
    assert result["state_transition_replay_exact"] is True
    assert result["all_registered_updates_covered"] is True
    assert result["V_H_psd"] > 0.0
    assert not capture_dir.exists()
    assert not scratch.exists()
    assert output.is_file()
    assert json.loads(status.read_text())["phase"] == "SEALED"


def _hidden_authorization_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    campaign_root = tmp_path / "campaign-root"
    attempt_root = (
        campaign_root
        / "vms"
        / "v0"
        / "g1"
        / "cells"
        / "cell"
        / "attempt-1"
    )
    export = attempt_root / "work" / "m4" / "export"
    export.mkdir(parents=True)
    model = export / "model.bin"
    model.write_bytes(b"checkpoint-export")
    checkpoint = attempt_root / "work" / "m4" / "state.ckpt"
    checkpoint.write_bytes(b"state")
    files = [
        {
            "path": checkpoint.relative_to(attempt_root).as_posix(),
            "sha256": promotion.sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        {
            "path": model.relative_to(attempt_root).as_posix(),
            "sha256": promotion.sha256_file(model),
            "size_bytes": model.stat().st_size,
        },
    ]
    inventory = {
        "schema": "audit_135m_checkpoint_inventory_v1",
        "cell_id": "cell",
        "attempt": 1,
        "loss_exposed": False,
        "files": files,
    }
    inventory["inventory_canonical_sha256"] = hidden.canonical_sha256(inventory)
    inventory_path = (
        attempt_root / "report" / "parallel-evidence" / "checkpoint-inventory.json"
    )
    inventory_path.parent.mkdir(parents=True)
    inventory_path.write_text(json.dumps(inventory))
    attempt = {
        "attempt_id": "cell-attempt-1",
        "cell_id": "cell",
        "status": "COMPLETED",
        "attempt_prefix": "gs://example/audit/vms/v0/g1/cells/cell/attempt-1/",
        "artifact_inventory": {
            "checkpoint_inventory": {
                "path": inventory_path.relative_to(campaign_root).as_posix(),
                "sha256": promotion.sha256_file(inventory_path),
                "size_bytes": inventory_path.stat().st_size,
            }
        },
    }
    checkpoint_rows = [
        {
            "cell_id": "cell",
            "attempt_id": "cell-attempt-1",
            "status": "COMPLETED",
            "evaluation_mode": "confirmation_audit_pending",
            "checkpoint_inventory_canonical_sha256": inventory[
                "inventory_canonical_sha256"
            ],
        }
    ]
    bound = {"study_id": "test", "expected_cells": [{"cell_id": "cell"}]}
    bound_path = tmp_path / "bound.json"
    bound_path.write_text(json.dumps(bound))
    preseal = {
        "schema": "audit_135m_checkpoint_preseal_v1",
        "status": "SEALED_TRAINING_AND_CHECKPOINT_REGISTRY",
        "stage_code": "a1c",
        "loss_exposed": False,
        "partial_outcomes_exposed": False,
        "provider_lifecycle_final_pending": True,
        "bound_manifest_canonical_sha256": hidden.canonical_sha256(bound),
        "attempts": [attempt],
        "attempts_canonical_sha256": hidden.canonical_sha256([attempt]),
        "audit_checkpoint_registry": {
            "schema": "audit_135m_checkpoint_registry_v1",
            "loss_exposed": False,
            "cells": checkpoint_rows,
            "checkpoint_registry_hash": hidden.canonical_sha256(checkpoint_rows),
        },
        "evaluation_required_cell_ids": ["cell"],
        "maximum_training_completion_utc": "2026-07-17T00:00:00Z",
        "sealed_at_utc": "2026-07-17T00:00:01Z",
    }
    preseal["preseal_canonical_sha256"] = hidden.canonical_sha256(preseal)
    preseal_path = tmp_path / "preseal.json"
    preseal_path.write_text(json.dumps(preseal))
    return campaign_root, bound_path, preseal_path, model


def test_hidden_authorization_binds_complete_checkpoint_registry(tmp_path: Path) -> None:
    campaign_root, bound, preseal, _model = _hidden_authorization_fixture(tmp_path)
    output = tmp_path / "authorization.json"
    result = hidden.authorize(
        argparse.Namespace(
            checkpoint_preseal=preseal,
            bound_manifest=bound,
            campaign_root=campaign_root,
            evaluation_role="confirmation_audit",
            prediction_freeze=None,
            output=output,
        )
    )
    authorization = json.loads(output.read_text())
    assert result["loss_exposed"] is False
    assert authorization["evaluation_order"] == ["cell"]
    assert authorization["partial_exposure_forbidden"] is True
    assert authorization["whole_batch_retry_only"] is True


def test_hidden_authorization_rejects_checkpoint_tamper(tmp_path: Path) -> None:
    campaign_root, bound, preseal, model = _hidden_authorization_fixture(tmp_path)
    model.write_bytes(b"tampered")
    with pytest.raises(hidden.HiddenEvaluationError, match="hash/size differs"):
        hidden.authorize(
            argparse.Namespace(
                checkpoint_preseal=preseal,
                bound_manifest=bound,
                campaign_root=campaign_root,
                evaluation_role="confirmation_audit",
                prediction_freeze=None,
                output=tmp_path / "authorization.json",
            )
        )


def test_final_report_renders_registered_a3_disposition(tmp_path: Path) -> None:
    phase = {
        "status": "sealed_results",
        "expected_cells": [{"cell_id": "cell"}],
        "results": [{"cell_id": "cell", "attempt": 1}],
    }
    analysis_value = {
        "schema": "audit_135m_a3_analysis_v1",
        "status": "SEALED",
        "rows": {
            "8": {
                "observed_selected_eta": 0.02,
                "predicted_eta": 0.021,
                "absolute_log2_eta_error": 0.07,
                "eta_error_pass": True,
                "observed_tuned_loss": 2.0,
                "predicted_frontier_loss": 2.001,
                "prediction_interval_95": [1.99, 2.01],
                "prediction_interval_coverage": True,
                "bracketed": True,
            },
            "512": {
                "observed_selected_eta": 0.04,
                "predicted_eta": 0.041,
                "absolute_log2_eta_error": 0.04,
                "eta_error_pass": True,
                "observed_tuned_loss": 2.2,
                "predicted_frontier_loss": 2.199,
                "prediction_interval_95": [2.18, 2.22],
                "prediction_interval_coverage": True,
                "bracketed": True,
            },
        },
        "five_point_frontier": {"8": 2.0, "16": 2.1, "64": 2.15, "256": 2.18, "512": 2.2},
        "extension_point_frontier_rmse": 0.001,
        "required_endpoint_ordering_pass": True,
        "gates": {
            "G5_A3_kernel_integrity": "PASS",
            "G6_A3_eta_prediction": "PASS",
            "G7_A3_frontier_prediction": "PASS",
            "A3_quantitative_law": "PASS",
        },
    }
    phase_path = tmp_path / "phase.json"
    analysis_path = tmp_path / "analysis.json"
    replay_path = tmp_path / "replay.json"
    ledger_path = tmp_path / "ledger.json"
    phase_path.write_text(json.dumps(phase))
    analysis_path.write_text(json.dumps(analysis_value))
    replay_value = {
        "schema": "audit_135m_replay_report_v1",
        "status": "PASS",
        "stage": "A3",
        "hostname": "isolated",
        "source": {
            "git_commit": "a" * 40,
            "tracked_sha256": {
                "experiment-specs/tuned-baseline-audit-prereg.json": "b" * 64
            },
        },
        "verified_file_count": 1,
        "campaigns": [
            {
                "attempt_count": 1,
                "launch_cell_count": 1,
                "generations": [{"slot": "v0"}],
            }
        ],
        "final_phase_manifest_canonical_sha256": promotion.canonical_sha256(phase),
    }
    replay_path.write_text(json.dumps(replay_value))
    ledger_path.write_text(
        json.dumps(
                {
                    "audit_stage": "A3",
                    "estimated_spend_usd": 12.5,
                    "hard_ceiling_usd": 31.18,
                    "pre_science_aborted_launch_spend_usd": 0.0,
                    "abort_burn_kill_usd": 40.0,
                }
        )
    )
    output = tmp_path / "AUDIT-135M-A3-FINAL.md"
    result = final_report.render(
        argparse.Namespace(
            stage="A3",
            final_phase_manifest=phase_path,
            analysis=analysis_path,
            replay_report=replay_path,
            stage_spend_ledger=ledger_path,
            output=output,
        )
    )
    text = output.read_text()
    assert result["status"] == "SEALED"
    assert "registered stage disposition `PASS`" in text
    assert "Isolated replay" in text
    assert "$12.500000" in text


def test_replay_builder_rewrites_descriptor_to_portable_relative_paths(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign-root"
    (campaign_root / "campaign").mkdir(parents=True)
    (campaign_root / "campaign" / "placeholder.json").write_text("{}\n")
    inputs = {}
    for field in (
        "parent_manifest",
        "bound_manifest",
        "scientific_plan",
        "parallel_roster",
        "parallel_plan",
        "vm_registry",
        "evaluation_registry",
        "final_provider_census",
        "runtime_authorization",
    ):
        path = tmp_path / f"{field}.json"
        path.write_text("{}\n")
        inputs[field] = str(path)
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text(
        json.dumps(
            {
                "stage_code": "a3k",
                "campaign_attempt": 1,
                "campaign_root": str(campaign_root),
                "aggregation_authorized": True,
                **inputs,
            }
        )
    )
    phase = tmp_path / "phase.json"
    attestation = tmp_path / "attestation.json"
    analysis_path = tmp_path / "analysis-source.json"
    ledger = tmp_path / "ledger-source.json"
    for path in (phase, attestation, analysis_path, ledger):
        path.write_text("{}\n")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    source = tmp_path / "source-index.json"
    source.write_text(
        json.dumps(
            {
                "schema": replay.SOURCE_SCHEMA,
                "stage": "A3",
                "source_commit": head,
                "final_phase_manifest": str(phase),
                "analysis": str(analysis_path),
                "stage_spend_ledger": str(ledger),
                "campaigns": [
                    {
                        "stage_code": "a3k",
                        "campaign_root": str(campaign_root),
                        "aggregation_descriptor": str(descriptor),
                        "phase_manifest": str(phase),
                        "phase_attestation": str(attestation),
                    }
                ],
            }
        )
    )
    output = tmp_path / "portable"
    result = replay.build(
        argparse.Namespace(
            source_index=source,
            output_dir=output,
            repo_root=Path(__file__).resolve().parents[1],
        )
    )
    portable_descriptor = json.loads(
        next((output / "campaigns").glob("*/inputs/aggregation-descriptor.json")).read_text()
    )
    assert result["status"] == "SEALED"
    assert portable_descriptor["campaign_root"] == "../campaign-root"
    assert portable_descriptor["parent_manifest"] == "parent_manifest.json"
    assert not Path(portable_descriptor["parent_manifest"]).is_absolute()
    index = json.loads((output / "input-index.json").read_text())
    assert index["file_registry_hash"] == replay.canonical_sha256(index["files"])
