"""The examples/k31 one-command deploy kits (``run.*.json`` -> ``pool.*.json`` -> sampler spec)
go through ``yeto run up``'s own reading path and stay mutually consistent: the run config through
``load_config``, every pool through ``pool_config``, the spec through ``build_engine``; then
``up_consistency.problems`` and the drain check are clean, and the trainer / pool / spec values that
must agree do agree. The 262k MI355X agentic kit also keeps its measured values locked."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from yeto.launch.nc import cli as nc_cli
from yeto.launch.rollout.pools import resolve
from yeto.rl import up, up_consistency
from yeto.rl.adapters.sampler_engines.vllm_server import K31_PRESETS
from yeto.rl.engine import agentic_round
from yeto.rl.engine import engine_sampler_entry as entry

from test_k31_guide_examples_run import _filled

ROOT = Path(__file__).resolve().parents[1]
K31 = ROOT / "examples" / "k31"
RETAIN = "--prefix-cache-retention-interval"
# Shared placeholders filled with one value in all three files, so that names derived from them
# (``<run>-0`` buffer shards, ``<run>`` shard base, the ``s3://<bucket>/runs/<run>`` store) still line up.
SHARED = {"<run>": "k31kit", "<bucket>": "k31bucket"}


def _ref(placeholder: str) -> str:
    """``<pool.x.json>`` -> ``pool.x.json``: a kit names its next file as a placeholder."""
    m = re.fullmatch(r"<([\w.-]+\.json)>", placeholder)
    assert m, f"not a kit file reference: {placeholder!r}"
    return m.group(1)


def kits() -> list[tuple[str, str, str]]:
    """(run, pool, spec) file names of every run.*.json, following its own references."""
    out = []
    for run_path in sorted(K31.glob("run.*.json")):
        pool_name = _ref(json.loads(run_path.read_text())["pool"]["config"])
        spec_name = _ref(json.loads((K31 / pool_name).read_text())["spec"])
        out.append((run_path.name, pool_name, spec_name))
    return out


KITS = kits()


def _raw(name: str) -> dict[str, Any]:
    text = (K31 / name).read_text()
    for key, value in SHARED.items():
        text = text.replace(key, value)
    return json.loads(text)


def _flag(argv: list[str], name: str) -> str:
    values = _flags(argv, name)
    assert len(values) == 1, f"{name}: {values}"
    return values[0]


def _flags(argv: list[str], name: str) -> list[str]:
    return [argv[i + 1] if a == name else a.split("=", 1)[1]
            for i, a in enumerate(argv) if a == name or a.startswith(name + "=")]


class Kit:
    """The three files filled and written as ``yeto run up`` reads them."""

    def __init__(self, names: tuple[str, str, str], tmp: Path):
        run_name, pool_name, spec_name = names
        self.spec = _filled(_raw(spec_name), tmp)
        self.spec.pop("_comment", None)
        spec_file = tmp / "spec.json"
        spec_file.write_text(json.dumps(self.spec))
        self.pool = _filled(_raw(pool_name), tmp)
        self.pool["spec"] = str(spec_file)
        pool_file = tmp / "pool.json"
        pool_file.write_text(json.dumps(self.pool))
        self.run = _filled(_raw(run_name), tmp)
        self.run["pool"]["config"] = str(pool_file)
        self.run_file = tmp / "run.json"
        self.run_file.write_text(json.dumps(self.run))
        self.argv = [str(a) for a in self.run["trainer"]["argv"]]
        # the placeholders the trainer parser types (an int, a spec it reads): without them the argv
        # does not parse and up_consistency.problems has nothing to compare
        for flag, value in (("--total-steps", "100"),
                            ("--rl-algorithm-spec", str(K31 / "algorithm-spec.async-age12.json"))):
            if flag in self.argv:
                self.argv[self.argv.index(flag) + 1] = value
        self.agentic = self.spec["agentic"]
        self.options = self.spec["engine"]["options"]
        self.server = self.spec["engine"]["server"]


@pytest.fixture(params=KITS, ids=lambda k: k[0])
def kit(request, tmp_path, monkeypatch):
    monkeypatch.setattr(nc_cli.tunnel, "state_dir", lambda: tmp_path / "nc-state", raising=False)
    return Kit(request.param, tmp_path)


def test_the_repository_ships_the_agentic_262k_kit():
    assert ("run.agentic-262k.json", "pool.nc-mi355x-agentic-262k.json",
            "sampler-spec.nc-mi355x-vllm-agentic-262k.json") in KITS


# ------------------------------------------------------------------ yeto run up's reading path
def test_run_up_reads_the_run_config_and_every_pool(kit, tmp_path):
    loaded = up.load_config(str(kit.run_file))
    refs = list(resolve(loaded, tmp_path / "state"))
    assert refs
    for ref in refs:
        assert nc_cli.pool_config(ref.load()) is not None


def test_the_sampler_spec_builds_its_engine(kit):
    assert entry.build_engine(kit.spec) is not None


def test_up_consistency_and_the_drain_check_are_clean(kit):
    assert up_consistency.trainer_args(kit.argv) is not None  # the trainer / spec pair is compared
    assert up_consistency.problems(kit.run["store"], kit.argv, kit.pool) == []
    assert agentic_round.drain_timeout_warning(kit.spec) is None


# ------------------------------------------------------------------ trainer / pool / spec agree
def test_sampling_and_rollout_fields_match_the_trainer(kit):
    assert kit.spec["sampling"]["n"] == int(_flag(kit.argv, "--n-samples-per-prompt"))
    assert kit.spec["rollout_fields"]["top_logprobs_num"] == int(_flag(kit.argv, "--rl-rollout-top-logprobs-num"))


def test_context_budget_matches_trainer_and_engine(kit):
    # a row is at most the engine context (a trial ends there): max_response_len is the engine
    # context, as context compaction needs, and the trainer's rows hold it
    assert kit.agentic["max_response_len"] == int(_flag(kit.argv, "--rollout-max-response-len")) \
        == int(_flag(kit.argv, "--seq-len")) == kit.server["max_model_len"]
    assert kit.agentic["max_prompt_len"] < kit.server["max_model_len"]


def test_the_documented_compaction_block_passes_the_run_up_checks(kit, tmp_path):
    """The compaction block the spec's comment documents (off by default) turns on with
    these rows and trainer: no up_consistency problem, the summary turn fits."""
    from yeto.rl import compaction

    assert "compaction" not in kit.agentic  # off by default
    raw = json.loads((K31 / "sampler-spec.nc-mi355x-vllm-agentic-262k.json").read_text())
    line = next(c for c in raw["_comment"] if '"compaction"' in c)
    block = json.loads("{" + line.strip() + "}")["compaction"]
    assert block == {"context_budget": 245760, "t_comp": 10240, "keep_recent_steps": 2, "max_compactions": 3}
    spec = {**kit.spec, "agentic": {**kit.agentic, "compaction": block}}
    cfg = compaction.config_from_spec(spec)
    headroom = cfg.check_summary_headroom(kit.server["max_model_len"], kit.agentic["turn_max_tokens"])
    assert headroom == 13704 and cfg.context_budget + headroom == 259464
    (tmp_path / "compacting.json").write_text(json.dumps(spec))
    pool = {**kit.pool, "spec": str(tmp_path / "compacting.json")}
    assert up_consistency.problems(kit.run["store"], kit.argv, pool) == []
    # CP2 with --rl-max-tokens-per-gpu = seq-len / CP: no micro batch wider than one row
    assert up_consistency.micro_batch_warning(up_consistency.trainer_args(kit.argv)) is None


def test_engine_slots_and_drain_bounds(kit):
    assert kit.options["max_num_seqs"] >= kit.agentic["max_in_flight"]
    assert kit.agentic["drain_timeout_s"] <= kit.spec["reload_max_defer_s"]
    defaults = kit.agentic["task_defaults"]
    assert defaults["wall_clock_s"] > defaults["agent_timeout_s"]


def test_pool_has_spare_accounts_and_the_trainer_reads_every_shard(kit):
    assert len(kit.pool["accounts"]) > kit.pool["target"]
    shards = _flag(kit.argv, "--rl-buffer-shards").split(",")
    assert len(shards) == kit.pool["shard_limit"]
    assert all(s.startswith(kit.pool["shard_base"]) for s in shards), shards


def test_policy_age_covers_a_release_interval_plus_margin(kit):
    assert int(_flag(kit.argv, "--rl-max-policy-age")) >= int(_flag(kit.argv, "--rl-release-every")) + 6


def test_the_named_algorithm_spec_allows_the_policy_age(kit):
    """The ``--rl-algorithm-spec`` placeholder names a shipped spec whose staleness
    covers ``--rl-max-policy-age`` (preflight refuses a smaller one)."""
    from yeto.rl.engine.algorithm import AlgorithmSpec  # noqa: PLC0415

    for path in K31.glob("run.*.json"):
        argv = json.loads(path.read_text())["trainer"]["argv"]
        name = re.sub(r"[<>]", "", argv[argv.index("--rl-algorithm-spec") + 1])
        spec = AlgorithmSpec.from_json_file(K31 / name)
        assert spec.execution.max_policy_staleness >= int(argv[argv.index("--rl-max-policy-age") + 1]), name


def test_rocm_preset_serves_with_the_r3_prefix_retention_interval(kit):
    preset = kit.options["k31_preset"]
    assert K31_PRESETS[preset]["platform"] == "rocm"
    engine = entry.build_engine(kit.spec)
    assert _flags(engine.argv(engine.config.base_dir), RETAIN) == ["9216"]


# ------------------------------------------------------------------ measured values (262k kit)
def test_agentic_262k_kit_keeps_the_measured_values(tmp_path, monkeypatch):
    monkeypatch.setattr(nc_cli.tunnel, "state_dir", lambda: tmp_path / "nc-state", raising=False)
    kit = Kit(next(k for k in KITS if k[0] == "run.agentic-262k.json"), tmp_path)
    # 2026-11-18 MI355X measurement: per-stream decode 50.6 / 42.2 / 36.8 tok/s at 8 / 16 / 24
    # streams; 16 in flight keeps a long trial inside agent_timeout_s.
    assert kit.agentic["max_in_flight"] == 16
    # 2026-11-18 MI355X measurement: 262k context with R3 (DCP 1) fits the KV cache (~2.6M tokens).
    assert kit.server["max_model_len"] == 262144
    # 2026-11-18 MI355X measurement: a 245760-token response at 42.2 tok/s (16 streams) decodes
    # in about 5800 s, so the agent timeout stays at 2 h or more.
    assert kit.agentic["task_defaults"]["agent_timeout_s"] >= 7200
