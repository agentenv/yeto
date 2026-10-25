"""S27 G-EN: training-time heldout eval on a verl island (VERL-PARITY #34).

``--rl-eval-interval`` and the other ``--rl-eval-*`` flags reach the verl
island in the Miles island spellings (``launch/rl_ports._ports_infra_flags``);
the island checks the heldout file with the engine's
``run_config.verify_eval_dataset_file``, writes it as verl's validation
parquet with the eval sampling (``eval_port``), and the engine driver calls
verl's ``_validate()`` through ``eval_port.VerlValidate`` on its own schedule
(rollout 0, every ``interval`` rollouts, the last one).  verl itself is not
imported: the trainer is a stand-in with verl's ``_validate`` contract.  CPU
only.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv, _prepared, _verl_args
from yeto.launch.rl_ports import EVAL_ISLAND_DATA_PATH
from yeto.rl.adapters.verl import eval_port, island_entry, launch_flags, run_config_rules
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.fake import FakeEngine, fake_capabilities

NAME = "base_model.model.layer.lora_A.weight"
DATASET = "gsm8k-heldout"
ROWS = [{"messages": [{"role": "user", "content": f"What is {i} + {i}?"}], "label": str(2 * i)}
        for i in range(5)]


def _heldout(tmp_path: Path, rows=ROWS) -> Path:
    path = tmp_path / "heldout.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _eval_flags(path: Path, *extra: str) -> tuple[str, ...]:
    return ("--rl-eval-interval", "2", "--rl-eval-data", str(path), "--rl-eval-dataset-name", DATASET,
            "--rl-eval-samples-per-prompt", "2", *extra)


def _island_args(args, heldout: Path):
    """The verl island's parsed argv, with the heldout file at the path the
    launcher's prelude writes it to replaced by ``heldout`` (same bytes)."""

    argv = [str(heldout) if token == EVAL_ISLAND_DATA_PATH else token for token in _island_argv(args, 0)]
    return island_entry.parse_args(argv)


def _section(island, tmp_path, **kwargs):
    written = {}
    section = eval_port.eval_section(
        island, work=tmp_path / "work", max_prompt_len=kwargs.pop("max_prompt_len", 512),
        max_response_len=kwargs.pop("max_response_len", island.rollout_max_response_len),
        temperature=1.0, top_p=1.0, write=lambda rows, path: written.update(rows=rows, path=path), **kwargs)
    return section, written


# ----------------------------------------------------------------------------- launcher -> island
def test_the_launcher_sends_the_eval_flags_and_the_verl_island_reads_every_one(tmp_path):
    heldout = _heldout(tmp_path)
    args = _prepared(_verl_args(*_eval_flags(heldout, "--rl-eval-temperature", "0.6", "--rl-eval-top-p", "0.95")))
    island, ignored = _island_args(args, heldout)
    assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
    assert (island.eval_interval, island.eval_dataset_name, island.eval_samples_per_prompt) == (2, DATASET, 2)
    assert (island.eval_temperature, island.eval_top_p) == (0.6, 0.95)
    assert island.eval_data_sha256 == hashlib.sha256(heldout.read_bytes()).hexdigest()

    section, written = _section(island, tmp_path)
    assert section == {
        "interval": 2, "dataset_name": DATASET, "data_source": "yeto-eval/" + DATASET,
        "data_sha256": island.eval_data_sha256, "val_file": str(tmp_path / "work" / "eval.parquet"),
        "rows": len(ROWS), "samples_per_prompt": 2, "temperature": 0.6, "top_p": 0.95}
    assert written["path"] == tmp_path / "work" / "eval.parquet"
    first = written["rows"][1]
    assert first["prompt"] == ROWS[1]["messages"] and first["reward_model"]["ground_truth"] == "2"
    assert first["data_source"] == "yeto-eval/" + DATASET and first["extra_info"]["index"] == 1

    overrides, asserted = eval_port.apply_eval_overrides(
        section, ["data.val_files=/old/test.parquet", "data.seed=1"], ("actor_rollout_ref.actor.strategy",))
    val = "actor_rollout_ref.rollout.val_kwargs"
    assert overrides == ["data.val_files=" + section["val_file"], "data.seed=1", f"{val}.n=2",
                         f"{val}.temperature=0.6", f"{val}.top_p=0.95", f"{val}.top_k=-1", f"{val}.do_sample=True"]
    assert asserted == ("actor_rollout_ref.actor.strategy", *eval_port.ASSERTED_KEYS)


def test_the_eval_parquet_holds_the_rows_verl_reads(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    heldout = _heldout(tmp_path)
    island, _ = _island_args(_prepared(_verl_args(*_eval_flags(heldout))), heldout)
    section = eval_port.eval_section(island, work=tmp_path / "work", max_prompt_len=512,
                                     max_response_len=island.rollout_max_response_len, temperature=1.0, top_p=1.0)
    frame = pd.read_parquet(section["val_file"])
    assert list(frame.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    assert len(frame) == len(ROWS) and frame["reward_model"][4]["ground_truth"] == "8"
    assert [dict(m) for m in frame["prompt"][0]] == ROWS[0]["messages"]


def test_without_eval_sampling_flags_the_eval_samples_as_the_training_rollout(tmp_path):
    heldout = _heldout(tmp_path)
    island, _ = _island_args(_prepared(_verl_args(*_eval_flags(heldout))), heldout)
    section, _ = _section(island, tmp_path)
    assert (section["temperature"], section["top_p"]) == (1.0, 1.0)  # Miles: eval_* else rollout_*
    greedy, _ = _section(SimpleNamespace(**{**vars(island), "eval_temperature": 0.0}), tmp_path)
    overrides, _ = eval_port.apply_eval_overrides(greedy, [], ())
    assert "actor_rollout_ref.rollout.val_kwargs.do_sample=False" in overrides
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=0.0" in overrides


def test_a_run_without_eval_flags_keeps_its_plan_and_overrides(tmp_path):
    args = _prepared(_verl_args())
    island, _ = island_entry.parse_args(_island_argv(args, 0))
    assert island.eval_interval is None and not eval_port.eval_requested(island)
    assert eval_port.eval_section(island, work=tmp_path, max_prompt_len=512, max_response_len=128,
                                  temperature=1.0, top_p=1.0) is None
    overrides, keys = ["data.val_files=/v.parquet"], ("a",)
    assert eval_port.apply_eval_overrides(None, overrides, keys) == (overrides, keys)
    assert eval_port.driver_options(object(), {"learner_id": 0}) == {}


@pytest.mark.parametrize("change, match", [
    (dict(eval_data_sha256="0" * 64), "SHA256 mismatch"),
    (dict(eval_max_response_len=64), "--eval-max-response-len 64: verl's validation shares"),
    (dict(eval_max_prompt_len=256), "--eval-max-prompt-len 256"),
    (dict(eval_max_context_len=4096), "--eval-max-context-len 4096"),
    (dict(eval_interval=0), "positive int"),
    (dict(eval_interval=None), "need --eval-interval"),
    (dict(eval_samples_per_prompt=0), "positive sample count"),
    (dict(eval_top_p=0.0), "eval sampling out of range"),
])
def test_eval_flags_the_verl_island_cannot_run_are_refused(tmp_path, change, match):
    heldout = _heldout(tmp_path)
    island, _ = _island_args(_prepared(_verl_args(*_eval_flags(heldout))), heldout)
    with pytest.raises(ValueError, match=match):
        _section(SimpleNamespace(**{**vars(island), **change}), tmp_path)


def test_matching_length_flags_are_accepted(tmp_path):
    heldout = _heldout(tmp_path)
    island, _ = _island_args(_prepared(_verl_args(*_eval_flags(heldout))), heldout)
    same = SimpleNamespace(**{**vars(island), "eval_max_prompt_len": 512, "eval_max_response_len": 128,
                              "eval_max_context_len": 640})
    assert _section(same, tmp_path, max_response_len=128)[0]["rows"] == len(ROWS)


@pytest.mark.parametrize("rows, match", [
    ([{"prompt": "1 + 1?", "label": "2"}], "chat 'messages'"),
    ([{"messages": [{"role": "user", "content": "1 + 1?"}]}], "no 'label'"),
])
def test_heldout_rows_need_the_training_row_schema(tmp_path, rows, match):
    with pytest.raises(eval_port.EvalPortError, match=match):
        eval_port.heldout_rows(_heldout(tmp_path, rows), data_source="yeto-eval/x")


def test_the_verl_paths_without_the_hook_refuse_the_eval():
    for args in (SimpleNamespace(rl_eval_interval=2, rl_max_policy_age=1),
                 SimpleNamespace(rl_eval_interval=2, rl_sample_role="trainer"),
                 SimpleNamespace(rl_eval_interval=2, rl_island_sample_role="1:sampler")):
        with pytest.raises(ValueError, match="--rl-eval-interval on verl"):
            run_config_rules.check_eval(args)
    run_config_rules.check_eval(SimpleNamespace(rl_eval_interval=2, rl_sample_role="both"))
    run_config_rules.check_eval(SimpleNamespace(rl_eval_interval=None, rl_max_policy_age=1))
    with pytest.raises(eval_port.EvalPortError, match="fully_async path has no evaluate hook"):
        eval_port.refuse_unwired(SimpleNamespace(eval_interval=2), "fully_async")
    eval_port.refuse_unwired(SimpleNamespace(eval_interval=None), "fully_async")


def test_a_verl_lora_launch_sends_the_eval_to_its_synchronous_island(tmp_path):
    heldout = _heldout(tmp_path)
    args = _prepared(_launcher_args("ports", ("--rl-backend", "verl", "--reward-function", "gsm8k_reward:score",
                                              *_eval_flags(heldout)), gpu="modal:1xh200,modal:1xh200"))
    island, _ = _island_args(args, heldout)
    assert island.parameter_mode == "lora" and island.eval_interval == 2
    assert _section(island, tmp_path)[0]["rows"] == len(ROWS)


def test_eval_interval_is_no_longer_refused_on_a_full_parameter_launch():
    refused = {option for _dest, _off, option in launch_flags.IGNORED_ON_VERL_FULL_PARAMETER}
    assert "--rl-eval-interval" not in refused and "--rl-eval-interval" not in launch_flags.NOT_MAPPED_REASONS


# ----------------------------------------------------------------------------- the evaluate hook
class _VerlTrainer:
    """verl's validation contract as ``VerlValidate`` uses it: ``val_dataset``,
    ``on_validate_begin`` / ``_validate`` / ``on_validate_end``.  ``_validate``
    reports the policy version the engine serves (the last publication)."""

    def __init__(self, rows: int, engine: FakeEngine | None = None, key: str | None = None):
        self.val_dataset = [None] * rows
        self.engine, self.key, self.calls = engine, key, []

    def on_validate_begin(self):
        self.calls.append("begin")

    def _validate(self):
        self.calls.append("validate")
        served = self.engine.published[0] if self.engine is not None else 3
        return {self.key or f"val-core/yeto-eval/{DATASET}/reward/mean@2": np.float64(served / 10),
                "val-aux/num_turns/mean": np.int64(2), "val-aux/note": "text"}

    def on_validate_end(self):
        self.calls.append("end")


def _plan(interval=2, rows=3):
    return {"eval": {"interval": interval, "dataset_name": DATASET, "data_source": "yeto-eval/" + DATASET,
                     "rows": rows, "samples_per_prompt": 2, "temperature": 1.0, "top_p": 1.0}}


def test_verl_validate_runs_verls_validation_and_names_the_miles_metric():
    trainer = _VerlTrainer(3)
    options = eval_port.driver_options(trainer, _plan())
    assert options["eval_interval"] == 2
    metrics = options["evaluate"](4)
    assert trainer.calls == ["begin", "validate", "end"]
    assert metrics[DATASET] == metrics[f"val-core/yeto-eval/{DATASET}/reward/mean@2"] == pytest.approx(0.3)
    assert metrics["val-aux/num_turns/mean"] == 2.0 and "val-aux/note" not in metrics
    assert all(type(value) is float for value in metrics.values()) and metrics["verl_validate_seconds"] >= 0


def test_verl_validate_refuses_a_smaller_eval_set_and_a_missing_mean():
    with pytest.raises(eval_port.EvalPortError, match="loaded 2 of the 3 eval rows"):
        eval_port.VerlValidate(_VerlTrainer(2), _plan()["eval"])
    with pytest.raises(RuntimeError, match="no val-core/yeto-eval/"):
        eval_port.VerlValidate(_VerlTrainer(3, key="val-core/other/reward/mean@2"), _plan()["eval"])(0)


def test_the_driver_evaluates_each_due_policy_right_after_publishing_it(tmp_path):
    engine = FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0)
    trainer = _VerlTrainer(3, engine)
    tape = tmp_path / "events.jsonl"
    driver = IslandDriver(
        learner_id=0, rollout=engine.rollout, trainer=engine.trainer, policy_state=engine.policy_state,
        publisher=engine.publisher, placement=engine.placement, capabilities=fake_capabilities(),
        algorithm=AlgorithmSpec(), sync=LocalOnlySync(5), events=EventTape(tape, 0),
        **eval_port.driver_options(trainer, _plan()))
    driver.run()
    evals = [e for e in map(json.loads, tape.read_text().splitlines()) if e["event"] == "rl_eval"]
    # rollout 0, every second rollout and the last one; each sees the policy just published
    assert [e["policy_version"] for e in evals] == [0, 2, 4, 5]
    assert [e[f"eval/{DATASET}"] for e in evals] == pytest.approx([0.0, 0.2, 0.4, 0.5])
    assert trainer.calls.count("validate") == 4


def test_the_synchronous_verl_trainer_wires_the_hook_and_the_fully_async_one_does_not():
    source = Path(island_entry.__file__).with_name("trainer.py").read_text()
    sync_part, async_part = source.split("def run_fully_async_island", 1)
    assert "**eval_port.driver_options(trainer, plan)" in sync_part.split("def run_island", 1)[1]
    assert "eval_port" not in async_part
    entry = inspect.getsource(island_entry.main)
    assert 'eval_port.refuse_unwired(args, "fully_async" if limit else "buffer trainer")' in entry
