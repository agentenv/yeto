"""The verl island tape records ``rl_engine_selected``, the record a Miles island writes.

Before, a verl tape had none, so ``yeto-rl-export --rl-event-tape`` found 0
algorithms on it (and the SSH harness export and the dashboard's outer-sync
topology had nothing to read). Every verl island path (synchronous FSDP2,
fully_async, Megatron) now appends it from the shared builder
(``yeto.rl.engine.selection.engine_selected_event``): the same keys as Miles', with
the verl pin in place of Miles' commit. CPU only.
"""
from __future__ import annotations

import inspect
import json

import pytest

from verl_island_fakes import drive_trainer, full_plan, stub_verl_and_ray
from yeto.rl.adapters.miles.entry import selection_event
from yeto.rl.adapters.verl import driver_extras
from yeto.rl.adapters.verl.pins import VERL_COMMIT
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.driver import EventTape
from yeto.rl.export import algorithm_from_event_tape

SPEC = {"advantage_estimator": "grpo", "loss": "policy_loss"}
MILES_KEYS = ["event", "rl_engine", "miles_commit", "rl/algorithm_spec_sha256", "rl/algorithm_spec",
              "rl/algorithm_absorbed_flags", "placement", "rl/outer_sync"]


def _spec() -> AlgorithmSpec:
    return AlgorithmSpec.from_dict(SPEC)


def test_the_miles_record_is_unchanged():
    from types import SimpleNamespace

    launch = SimpleNamespace(placement=SimpleNamespace(kind="colocated"), absorbed_flags={})
    event = selection_event(launch=launch, algorithm=_spec(), miles_commit="m" * 40)
    assert list(event) == MILES_KEYS and event["miles_commit"] == "m" * 40


@pytest.mark.parametrize("plan, placement, outer_sync", [
    ({"sync": "decoupled"}, "colocated", True),
    ({"sync": "strict", "placement": "fixed-partition"}, "fixed-partition", True),
    ({"sync": "none"}, "colocated", False),
])
def test_the_verl_record_has_the_miles_schema(plan, placement, outer_sync):
    event = driver_extras.engine_selected(plan, _spec())
    assert [k for k in event if k not in ("rl_backend", "verl_commit")] == [k for k in MILES_KEYS
                                                                            if k != "miles_commit"]
    assert (event["rl_backend"], event["verl_commit"]) == ("verl", VERL_COMMIT)
    assert event["rl/algorithm_spec"] == _spec().canonical_json()
    assert event["rl/algorithm_spec_sha256"] == _spec().sha256()
    assert (event["placement"], event["rl/outer_sync"]) == (placement, outer_sync)


def test_yeto_rl_export_reads_the_algorithm_from_a_verl_tape(tmp_path):
    path = tmp_path / "tape.jsonl"
    tape = EventTape(str(path), 0)
    tape.append({"event": "rl_island_hardware"})
    tape.append(driver_extras.engine_selected({"sync": "decoupled"}, _spec()))
    tape.append({"event": "rl_learner_finalized"})
    spec, mechanisms = algorithm_from_event_tape(path)  # strict: complete tape, hash checked
    assert spec == _spec().canonical_json() and mechanisms == ()


def _selected(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if json.loads(line).get("event") == "rl_engine_selected"]


@pytest.mark.parametrize("path, shared", [("sync", "full_parameter_stream.full_parameter_policy"),
                                          ("fully_async", "fully_async_full.full_parameter_ports")])
def test_each_fsdp_island_path_appends_it_before_the_driver_runs(monkeypatch, tmp_path, path, shared):
    from verl_island_fakes import modules

    module_name, function = shared.split(".")
    with stub_verl_and_ray():
        (module,) = modules(f"yeto.rl.adapters.verl.{module_name}")
        monkeypatch.setattr(module, function, lambda *a, **k: ("sync", "policy-state"))
        drive_trainer(monkeypatch, tmp_path, path, plan=full_plan(tmp_path, algorithm_spec=SPEC))
    (event,) = _selected(tmp_path / "tape.jsonl")
    assert event["rl/algorithm_spec_sha256"] == _spec().sha256() and event["rl_backend"] == "verl"
    assert event["rl/outer_sync"] is False


def test_the_megatron_island_appends_it_too():
    from verl_island_fakes import modules

    with stub_verl_and_ray():
        (megatron_ports,) = modules("yeto.rl.adapters.verl.megatron_ports")
        source = inspect.getsource(megatron_ports)
    assert "tape.append(engine_selected(plan, spec))" in source
