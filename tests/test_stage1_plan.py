"""Stage-1 plan expansion and consistency checks (S25 T-STAGE1). CPU only, no SSH."""

from __future__ import annotations

import copy
import shlex
from pathlib import Path

import pytest
import yaml

from yeto.launch import stage1

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "examples" / "stage1" / "glm53-flash-ppu.yaml"


@pytest.fixture(scope="module")
def shipped():
    return stage1.expand(stage1.load_plan(PLAN))


def _variant(tmp_path: Path, edit) -> stage1.Expanded:
    """Expand a copy of the shipped plan after ``edit(doc)`` changed it."""
    doc = yaml.safe_load(PLAN.read_text())
    edit(doc)
    (tmp_path / "ppu-inventory.yaml").write_text((PLAN.parent / "ppu-inventory.yaml").read_text())
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(doc))
    return stage1.expand(stage1.load_plan(path))


def test_shipped_plan_is_consistent(shipped):
    assert shipped.errors == []
    assert shipped.trainer["cards"] == 24
    assert shipped.trainer["miles"]["layout"] == {"tp": 4, "pp": 3, "cp": 1, "ep": 8, "etp": 1, "dp": 2,
                                                  "expert_dp": 1}
    assert [e["node"] for e in shipped.sampler["engines"]] == ["ppu4"]


def test_names_agree_across_roles(shipped):
    run = shipped.store["run"]
    assert run.endswith("/s25-stage1-glm53-ppu") and run.startswith("oss://")
    req = shipped.requant["argv"]
    assert req[req.index("--store") + 1] == run
    assert req[req.index("--target-dir") + 1] == "policy-releases-mxfp4-g32-rtn"
    island = shipped.trainer["miles"]["island_argv"]
    assert island[island.index("--rl-resume-store") + 1] == run
    for step in shipped.steps:
        if step.script.startswith("docker run"):
            assert f"YETO_RL_STORE={run}" in step.script
            assert "yeto-s25-stage1-glm53-ppu-" in step.script


def test_sampler_command_comes_from_the_adapter(shipped):
    engine = shipped.sampler["engines"][0]
    argv = engine["argv"]
    assert argv[argv.index("--tp") + 1] == "8"
    assert argv[argv.index("--model-path") + 1].endswith("mxfp4-experts-70e86a0")
    assert "--speculative-algorithm" not in argv  # no NEXTN in stage 1
    assert argv[argv.index("--moe-runner-backend") + 1] == "triton"
    assert engine["env"]["MX_PATCH"] == "1"
    assert engine["url"] == "http://10.0.0.77:30000"


def test_max_tokens_is_one_value_everywhere(shipped):
    island = shipped.trainer["miles"]["island_argv"]
    margs = shipped.trainer["miles"]["megatron_args"]
    assert island[island.index("--rollout-max-response-len") + 1] == "4096"
    assert margs[margs.index("--rollout-max-response-len") + 1] == "4096"
    assert "data.max_response_length=4096" in shipped.trainer["verl"]["overrides"]
    assert shipped.sampler["sampling"]["max_tokens"] == 4096


def test_trainer_is_bf16_with_host_optimizer(shipped):
    margs = " ".join(shipped.trainer["miles"]["megatron_args"])
    assert "--bf16" in margs and "fp8" not in margs
    assert "--optimizer-cpu-offload" in margs
    assert "--decoder-first-pipeline-num-layers 15 --decoder-last-pipeline-num-layers 15" in margs
    assert shipped.trainer["miles"]["drop_model_args"] == []  # TE stays the default (ruling 10-19 (3))


def test_island_argv_parses_with_the_island_parser(shipped):
    assert stage1._check_island_argv(shipped.trainer["miles"]["island_argv"]) is None


def test_steps_order_and_staging(shipped):
    names = [s.name for s in shipped.steps]
    assert names[:4] == [f"preflight-ppu{i}" for i in range(1, 5)]
    assert names.index("stage-sampler_initial-ppu1") < names.index("requant-sidecar")  # template on node 0
    assert names.index("sampler-ppu4") < names.index("trainer-island")
    assert names[-1] == "requant-sidecar"
    for step in shipped.steps:
        if step.script.startswith("docker run"):
            shlex.split(step.script)  # quotable as one command


def test_gaps_block_run_until_wired(shipped, capsys):
    assert any("N0.2" in g for g in shipped.gaps)
    assert any("OSS" in g for g in shipped.gaps)
    assert stage1.main(["run", str(PLAN), "--confirm-machines-free"]) == 2
    assert "run refused" in capsys.readouterr().err


def test_plan_command_prints_without_ssh(capsys):
    assert stage1.main(["plan", str(PLAN), "--json"]) == 0
    assert '"schema": "yeto.stage1/v1"' in capsys.readouterr().out


def test_run_needs_confirmation(monkeypatch, shipped, capsys):
    clean = copy.copy(shipped)
    clean.gaps = []
    monkeypatch.setattr(stage1, "expand", lambda plan, hf_config=None: clean)
    calls = []
    monkeypatch.setattr(stage1, "run_steps", lambda out: calls.append(out) or 0)
    assert stage1.main(["run", str(PLAN)]) == 2
    assert "--confirm-machines-free" in capsys.readouterr().err
    assert stage1.main(["run", str(PLAN), "--confirm-machines-free"]) == 0
    assert calls == [clean]


def test_run_steps_stops_at_first_failure(shipped):
    seen = []

    class Done:
        def __init__(self, rc):
            self.returncode = rc

    def runner(argv, input=None, text=True):
        seen.append(argv)
        return Done(1 if len(seen) == 2 else 0)

    assert stage1.run_steps(shipped, runner=runner) == 1
    assert len(seen) == 2 and seen[0][-3:] == ["ppu1", "bash", "-s"]


@pytest.mark.parametrize("edit, message", [
    (lambda d: d["trainer"]["miles"].update(pp_layers=[15, 15, 14]), "sum to 44"),
    (lambda d: d["trainer"]["miles"].update(pp_layers=[23, 22]), "2 stages"),
    (lambda d: d["sampling"].update(repetition_penalty=1.1), "repetition_penalty"),
    (lambda d: d["quant"].update(release_dir="policy-releases-w4a8"), "release_dir"),
    (lambda d: d["quant"].update(scheme="w4a8-int4-channel"), "is scheme mxfp4-g32-rtn"),
    (lambda d: d["sampler"].update(renames_weight_packed=False), "weight_packed"),
    (lambda d: d["sampler"].update(engine_version="0.5.18"), "engine version"),
    (lambda d: d["sampler"].update(max_model_len=4096), "max_model_len"),
    (lambda d: d["model"]["sampler_initial"].update(revision="aeb9adf"), "revision aeb9adf"),
    (lambda d: d["model"]["bf16_source"].update(local_dir="/data/models/bf16"), "bf16_source revision"),
    (lambda d: d["roles"].update(sampler=["ppu3", "ppu4"]), "both trainer and sampler"),
    (lambda d: d["rendezvous"].update(ppu2="ppu2"), "not an IP address"),
    (lambda d: d["trainer"]["miles"].update(optimizer_offload_fraction=0.0), "activations"),
    (lambda d: d.update(store="s3://bucket/x"), "oss://"),
])
def test_contradictions_are_errors(tmp_path, edit, message):
    out = _variant(tmp_path, edit)
    assert any(message in e for e in out.errors), out.errors


def test_layout_megatron_refuses(tmp_path):
    with pytest.raises(stage1.Stage1Error, match="inside one node"):
        _variant(tmp_path, lambda d: d["trainer"]["miles"].update(ep=12, pp=1, tp=1, pp_layers=[45]))
    with pytest.raises(stage1.Stage1Error, match="multiple"):
        _variant(tmp_path, lambda d: d["trainer"]["miles"].update(tp=5))


def test_te_fallback_drops_te_only_model_args(tmp_path):
    out = _variant(tmp_path, lambda d: d["trainer"]["miles"].update(transformer_impl="local"))
    assert out.errors == []
    assert out.trainer["miles"]["drop_model_args"] == ["--moe-grouped-gemm", "--moe-permute-fusion"]
    assert "--no-gradient-accumulation-fusion" in out.trainer["miles"]["megatron_args"]


def test_verl_backend_expands(tmp_path):
    out = _variant(tmp_path, lambda d: d["trainer"].update(backend="verl"))
    assert out.errors == []
    assert "trainer.nnodes=3" in out.trainer["verl"]["overrides"]
    assert any("verl" in g for g in out.gaps)
