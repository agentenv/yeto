"""``yeto image adopt`` step 5 and the command (own-cluster-fleet-stability 7.3, 7.5):
the trainer smoke (a cut model, one round, a cut and a resume) and the sampler smoke
(the node runtime's container start of yeto.node.smoke: start, generate, reload,
generate) on CPU fakes; the catalog is written only after a passing smoke; the dry
run prints the smoke's commands; the documented commands run in a dry run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_cluster_preflight import FakeCluster, b300
from tests.test_image_adopt import PINNED, TAGGED, FakeNodes, probe_out, registry
from yeto.cli import build_parser
from yeto.launch import image_adopt as ia
from yeto.launch import image_smoke as sm
from yeto.launch import site_images as si
from yeto.node import smoke as node_smoke
from yeto.rl.adapters.sampler_engines.fake import FakeEngine

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_CONFIGS = ROOT / "tests" / "data" / "model_registry"
QWEN = json.loads((REGISTRY_CONFIGS / "qwen3_30b_a3b.json").read_text())
K31 = json.loads((REGISTRY_CONFIGS / "kimi_k31.json").read_text())
FAKE_SPEC = {"sampler_id": "adopt", "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                               "adapter": "fake", "kwargs": {}}}
REV = "f" * 40


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("YETO_IMAGE_MANIFEST_CHECK", raising=False)


def node(i=0, **kw):
    return b300(i, **{"scratch": "/mnt/nvme/yeto", "image": None, **kw})


# -- the node side of the sampler smoke ----------------------------------------------


class ReloadFails(FakeEngine):
    def load(self, release_dir, version):
        raise RuntimeError("update_weights_from_disk answered 500")


def test_the_node_smoke_starts_generates_reloads_and_generates(capsys):
    engine = FakeEngine()
    record = node_smoke.run_smoke(engine, release=None)
    assert record["ok"] and record["step"] is None and record["reload_mode"] == "in_place"
    assert set(record["seconds"]) == set(node_smoke.STEPS)
    assert engine.loads == [(None, 0), (None, 1)] and engine.stopped
    lines = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [r["step"] for r in lines] == list(node_smoke.STEPS) and all(r["ok"] for r in lines)


def test_a_failed_reload_names_the_step_and_stops_the_engine():
    engine = ReloadFails()
    record = node_smoke.run_smoke(engine, release="/models/r1", out=open(os.devnull, "w"))
    assert (record["ok"], record["step"]) == (False, "reload")
    assert "update_weights_from_disk answered 500" in record["error"] and engine.stopped


def test_the_node_smoke_entry_reads_the_spec(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(FAKE_SPEC))
    assert node_smoke.main(["--spec", str(spec)]) == 0
    assert node_smoke.parse_result(capsys.readouterr().out)["ok"] is True
    spec.write_text(json.dumps({"engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                           "adapter": "nope"}}))
    assert node_smoke.main(["--spec", str(spec)]) == 1
    result = node_smoke.parse_result(capsys.readouterr().out)
    assert result["step"] == "start" and "unknown sampler engine 'nope'" in result["error"]


# -- the sampler smoke through the adoption -------------------------------------------


class SmokeNodes(FakeNodes):
    """The fake cluster, plus the sampler smoke: the upload is accepted, the engine
    step runs ``python -m yeto.node.smoke`` here on the spec it uploaded."""

    def __init__(self, nodes, tmp_path, *, pythonpath=(), **kw):
        super().__init__(nodes, probe=probe_out(packages={"vllm": "0.19.0", "torch": "2.9.1"}), **kw)
        self.tmp_path = tmp_path
        self.pythonpath = pythonpath

    def run(self, node, script, *, timeout=300.0):
        if "YETO_TREE_EOF" in script:
            self.scripts.append((node.name, "upload"))
            spec = re.search(r"<<'YETO_SPEC_EOF'\n(.*)\nYETO_SPEC_EOF", script, re.S).group(1)
            (self.tmp_path / "uploaded-spec.json").write_text(spec)
            return self.done(0, "")
        if "yeto.node.smoke" in script:
            self.scripts.append((node.name, "engine"))
            argv = shlex.split(script)
            assert PINNED in argv and "--rm" in argv and "--gpus" in argv
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), *map(str, self.pythonpath)])}
            done = subprocess.run([sys.executable, "-m", "yeto.node.smoke", "--spec",
                                   str(self.tmp_path / "uploaded-spec.json")], capture_output=True, text=True,
                                  env=env, timeout=120)
            return self.done(done.returncode, done.stdout + done.stderr)
        return super().run(node, script, timeout=timeout)


def _adopt(nodes, tmp_path, plan, ops, role):
    opener, _ = registry({"k99": 200})
    req = ia.AdoptRequest(ref=TAGGED, model="org/new-model", role=role, nodes=tuple(nodes),
                          catalog=str(tmp_path / "site-images.yaml"), engine=plan.data.get("engine"))

    def smoke(adopter, ref, facts, engine):
        return sm.run_plan(plan, ops=adopter.ops, ref=ref, python=facts.python_path, log=lambda m: None,
                           run=getattr(ops, "launch", subprocess.run), clock=lambda: 0.0)

    return ia.Adopter(req, ops=ops, smoke=smoke, urlopen=opener, login=None, log=lambda m: None,
                      clock=lambda: FakeCluster.NOW)


def test_a_sampler_image_whose_smoke_passes_is_recorded(tmp_path):
    nodes = [node(0), node(1)]
    plan = sm.sampler_plan(nodes[0], spec=FAKE_SPEC, run_id="adopt-t", code=("t000", b"tar"))
    report = _adopt(nodes, tmp_path, plan, SmokeNodes(nodes, tmp_path), "sampler").run()
    assert report.ok and report.steps[-1].detail.startswith("engine started, generated, reloaded")
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert (rec.role, rec.engine, rec.ref, rec.card_archs) == ("sampler", "vllm", si.parse_record(
        rec.to_doc()).ref, ("sm103",))
    assert rec.evidence["smoke"]["reload_mode"] == "in_place"
    assert set(rec.evidence["smoke"]["engine_steps"]) == set(node_smoke.STEPS)


def test_a_sampler_whose_reload_fails_is_not_recorded(tmp_path):
    (tmp_path / "adopt_fail_engine.py").write_text(
        "from tests.test_image_smoke import ReloadFails\n"
        "def build(section):\n    return ReloadFails()\n")
    spec = {"sampler_id": "adopt", "engine": {"factory": "adopt_fail_engine:build"}}
    nodes = [node(0)]
    plan = sm.sampler_plan(nodes[0], spec=spec, run_id="adopt-t", code=("t000", b"tar"))
    with pytest.raises(ia.AdoptError) as error:
        _adopt(nodes, tmp_path, plan, SmokeNodes(nodes, tmp_path, pythonpath=[tmp_path]), "sampler").run()
    assert error.value.step == "smoke"
    assert error.value.reason.startswith("engine: reload failed: RuntimeError: update_weights_from_disk answered 500")
    assert not (tmp_path / "site-images.yaml").exists()


def test_the_sampler_smoke_starts_like_the_node_runtime():
    from yeto.launch.node_inventory import profile_of
    from yeto.node import launch as node_launch

    n = node(0)
    plan = sm.sampler_plan(n, spec=FAKE_SPEC, run_id="adopt-t", code=("t000", b"tar"))
    engine = shlex.split(plan.steps[1].script)
    runtime = node_launch.container_argv(profile_of(n), name="yeto-adopt-adopt-t", image=sm.REF,
                                         env={"PYTHONPATH": "/work/image-adopt/yeto/t000", "PYTHONUNBUFFERED": "1",
                                              **profile_of(n).family_env()},
                                         python=sm.PYTHON, spec_path="/work/image-adopt/adopt-t/spec.json")
    assert engine[:-1] == runtime[:-1]  # the same flags, mounts and environment
    assert engine[-1].endswith("-m yeto.node.smoke --spec /work/image-adopt/adopt-t/spec.json")
    assert runtime[-1].endswith(f"-m {node_launch.SAMPLER_MODULE} --spec /work/image-adopt/adopt-t/spec.json")
    with pytest.raises(sm.SmokeError, match="needs a container runtime"):
        sm.sampler_plan(node(0, runtime="rootfs"), spec=FAKE_SPEC, run_id="x", code=("t", b""))


# -- the trainer smoke ----------------------------------------------------------------


def test_the_cut_is_the_most_layers_that_fit_one_node():
    layout = sm.smoke_layout(QWEN, node(0), engine=None)
    assert layout["engine"] == "miles" and layout["layers"] == 16 and layout["plan"]["world"] <= 8
    assert sm.smoke_layout(QWEN, node(0), engine=None, layers=4)["layers"] == 4
    small = node(0, card="L4", hbm_gib=24.0, cards_per_node=1)
    assert sm.smoke_layout(QWEN, small, engine="miles")["layers"] < 16
    with pytest.raises(sm.SmokeError, match="the model has 48 layers"):
        sm.smoke_layout(QWEN, node(0), engine=None, layers=99)


def _trainer_plan(tmp_path, **kw):
    target = kw.pop("node", node(0))
    args = {"model": "Qwen/Qwen3-30B-A3B", "config": QWEN, "engine": None, "store": "s3://bucket/smoke",
            "run_id": "adopt-t", "workdir": tmp_path / "work", "python": "py", **kw}
    return sm.trainer_plan(target, **args)


def test_the_trainer_plan_cuts_trains_and_resumes_on_one_node(tmp_path):
    plan = _trainer_plan(tmp_path)
    assert [s.name for s in plan.steps] == ["prune", "train", "resume"]
    prune, train, resume = plan.steps
    assert "--layers 16" in prune.script and "hf://Qwen/Qwen3-30B-A3B" in prune.script
    assert "base64 -d" in prune.script and sm.REVISION_MARK in prune.script
    argv = list(train.argv)
    assert argv[:4] == ["py", "-m", "yeto.cli", "launch"]
    for flag, value in (("--rl-backend", "miles"), ("--gpu", "ssh:1x8xB300"), ("--rl-image", f"docker:{sm.REF}"),
                        ("--model", "yeto-local/Qwen3-30B-A3B-16layer"), ("--model-revision", sm.REVISION),
                        ("--model-store", f"local://{sm.HUB_IN_CONTAINER}"),
                        ("--rl-checkpoint-store", "s3://bucket/smoke/adopt-t"), ("--rl-stop-after-rounds", "1")):
        assert argv[argv.index(flag) + 1] == value
    assert argv.count("--model") == 1
    assert list(resume.argv)[-2:] == ["--rl-stop-after-rounds", "2"] and resume.expect
    synth = json.loads(next(a for a in argv if a.startswith(sm.SYNTH_ENV + "=")).split("=", 1)[1])
    assert synth["layers"] == 16 and synth["experts"] == 128 and synth["router_topk"] == 8
    (inventory, text), = plan.files.items()
    (entry,) = yaml.safe_load(text)["nodes"]
    assert entry["mounts"]["/mnt/nvme/yeto/image-adopt/hub"] == sm.HUB_IN_CONTAINER and "image" not in entry
    assert entry["mounts"]["/mnt/nvme/models"] == "/data/models"  # the node's own mounts stay
    assert entry["roles"] == ["trainer"] and entry["scratch"] == "/mnt/nvme/yeto"
    assert plan.data["layers"] == 16 and plan.data["engine"] == "miles"
    assert any("16 of 48 layers" in n for n in plan.notes)


def test_a_node_path_takes_the_checkpoint_from_the_node(tmp_path):
    prune = _trainer_plan(tmp_path, model_path="/mnt/models/qwen3", model_revision="abc").steps[0]
    assert "/mnt/models/qwen3:/mnt/models/qwen3:ro" in prune.script and "hf://" not in prune.script
    assert "--revision" not in prune.script
    assert "--revision abc" in _trainer_plan(tmp_path, model_revision="abc").steps[0].script


@pytest.mark.parametrize("kw, message", [
    ({"node": node(0, scratch=None)}, "has no scratch"),
    ({"store": None}, "pass --smoke-store"),
    ({"node": node(0, runtime="rootfs")}, "runs rootfs"),
    ({"engine": "verl"}, "no trainer smoke for backend verl"),
])
def test_the_trainer_plan_refuses_what_it_cannot_run(tmp_path, kw, message):
    with pytest.raises(sm.SmokeError, match=message):
        _trainer_plan(tmp_path, **kw)


def test_a_registry_placeholder_needs_a_smoke_arg(tmp_path):
    with pytest.raises(sm.SmokeError, match=r"need \['<expert_selection.json>'\]"):
        _trainer_plan(tmp_path, model="moonshotai/Kimi-K3.1-Pro", config=K31)
    plan = _trainer_plan(tmp_path, model="moonshotai/Kimi-K3.1-Pro", config=K31,
                         extra_args=("--rl-expert-selection=/sel.json",))
    argv = list(plan.steps[1].argv)
    assert "--rl-expert-selection=/sel.json" in argv and "--rl-expert-selection" not in argv
    assert "<expert_selection.json>" not in argv


class Launches:
    """The launcher's subprocess for the train and resume steps."""

    def __init__(self, resume_out="rl_resume: from cut round 1", rc=0):
        self.calls, self.resume_out, self.rc = [], resume_out, rc

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        resume = argv[-1] == "2"
        return subprocess.CompletedProcess(argv, self.rc, stdout=self.resume_out if resume else "round 1 cut",
                                           stderr="")


class TrainerNodes(FakeNodes):
    def __init__(self, nodes, launches, **kw):
        super().__init__(nodes, **kw)
        self.launch = launches

    def run(self, node, script, *, timeout=300.0):
        if "YETO_PRUNE_EOF" in script:
            self.scripts.append((node.name, "prune"))
            assert PINNED in script and "/usr/bin/python3 /tmp/yeto-prune_layers.py" in script
            return self.done(0, f"{sm.REVISION_MARK} {REV}\n")
        return super().run(node, script, timeout=timeout)


def test_a_trainer_image_whose_smoke_passes_is_recorded(tmp_path):
    nodes = [node(0), node(1)]
    plan = _trainer_plan(tmp_path)
    launches = Launches()
    report = _adopt(nodes, tmp_path, plan, TrainerNodes(nodes, launches), "trainer").run()
    assert report.ok
    train, resume = launches.calls
    assert train[train.index("--model-revision") + 1] == REV and resume[resume.index("--rl-image") + 1] == \
        f"docker:{PINNED}"
    assert Path(train[train.index("--rl-node-inventory") + 1]).is_file()
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert (rec.role, rec.engine, rec.model) == ("trainer", "miles", "org/new-model")
    assert rec.evidence["smoke"]["layers"] == 16 and rec.evidence["smoke"]["revision"] == REV
    assert set(rec.evidence["smoke"]["steps"]) == {"prune", "train", "resume"}


@pytest.mark.parametrize("launches, message", [
    (Launches(resume_out="round 2 done"), "resume: the output has no"),
    (Launches(rc=1), "train: rc 1"),
])
def test_a_trainer_smoke_that_does_not_resume_or_train_is_not_recorded(tmp_path, launches, message):
    nodes = [node(0)]
    with pytest.raises(ia.AdoptError, match=message):
        _adopt(nodes, tmp_path, _trainer_plan(tmp_path), TrainerNodes(nodes, launches), "trainer").run()
    assert not (tmp_path / "site-images.yaml").exists()


def test_a_prune_without_a_revision_stops_the_smoke(tmp_path):
    class NoRev(TrainerNodes):
        def run(self, node, script, *, timeout=300.0):
            if "YETO_PRUNE_EOF" in script:
                return self.done(0, "cut done\n")
            return super().run(node, script, timeout=timeout)
    nodes = [node(0)]
    with pytest.raises(ia.AdoptError, match="prune: the output has no 'YETO_SMOKE_REVISION'"):
        _adopt(nodes, tmp_path, _trainer_plan(tmp_path), NoRev(nodes, Launches()), "trainer").run()


DOCKER_SHIM = r"""#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
assert args[:2] == ["run", "--rm"], args
mounts, i = [], 2
while args[i] != "--entrypoint":
    if args[i] == "-v":
        host, inside = args[i + 1].split(":")[:2]
        mounts.append((inside, host))
        i += 2
    else:
        i += 1
inner = args[args.index("-c") + 1]
for inside, host in sorted(mounts, key=lambda m: -len(m[0])):
    inner = inner.replace(inside, host)
sys.exit(subprocess.run(["bash", "-c", inner]).returncode)
"""


def _safetensors(path: Path, names):
    import struct

    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(names)))


def test_the_prune_script_on_the_node_writes_a_hub_snapshot(tmp_path):
    """The prune step's node script, with ``docker`` a shim that runs the container's
    bash here (the container paths mapped to their host directories)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({**QWEN, "num_hidden_layers": 48}))
    _safetensors(src / "model.safetensors", ["model.embed_tokens.weight", *[f"model.layers.{i}.w" for i in range(6)],
                                             "lm_head.weight"])
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    shim.write_text(DOCKER_SHIM)
    shim.chmod(0o755)
    n = node(0, scratch=str(tmp_path / "scratch"))
    plan = _trainer_plan(tmp_path, node=n, model_path=str(src), layers=4)
    prune = plan.steps[0].render({sm.REF: PINNED, sm.PYTHON: sys.executable})
    done = subprocess.run(["bash", "-c", prune.script], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"})
    assert done.returncode == 0, done.stdout + done.stderr
    rev = re.search(rf"{sm.REVISION_MARK} ([0-9a-f]{{40}})", done.stdout).group(1)
    hub = tmp_path / "scratch" / "image-adopt" / "hub" / "models--yeto-local--Qwen3-30B-A3B-4layer"
    snapshot = hub / "snapshots" / rev
    assert (hub / "refs" / "main").read_text() == rev
    assert json.loads((snapshot / "config.json").read_text())["num_hidden_layers"] == 4
    import hashlib

    assert hashlib.sha1((snapshot / "config.json").read_bytes()).hexdigest() == rev
    kept = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    assert sorted(kept) == sorted(["model.embed_tokens.weight", "lm_head.weight",
                                   *[f"model.layers.{i}.w" for i in range(4)]])
    assert not list((tmp_path / "scratch" / "image-adopt" / "hub").glob(".cut-*"))  # the temporary cut moved


# -- yeto image adopt -------------------------------------------------------------------


def _inventory(tmp_path, **node_kw):
    entry = {"name": "b300-0", "host": "ops@b300-0", "device_family": "cuda", "cpu_arch": "x86_64",
             "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, "runtime": "docker",
             "docker_args": ["--gpus", "all"], "scratch": "/mnt/nvme/yeto", **node_kw}
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump({"nodes": [entry]}))
    return path


def _cli(*argv):
    return build_parser().parse_args(["image", "adopt", *map(str, argv)])


def test_the_dry_run_prints_every_step_and_the_smoke_commands_and_writes_nothing(tmp_path, capsys):
    inv = _inventory(tmp_path)
    args = _cli(TAGGED, "--model", "Qwen/Qwen3-30B-A3B", "--role", "trainer", "--inventory", inv,
                "--model-config", REGISTRY_CONFIGS / "qwen3_30b_a3b.json", "--smoke-store", "s3://b/s", "--dry-run")
    assert ia.main(args, ops=SimpleNamespace(run=None)) == 0
    out = capsys.readouterr().out
    for want in ("1. manifest:", "2. probe on b300-0: docker run --rm", "3. fit:", "4. pull:",
                 "smoke (trainer) on node b300-0", "1. prune [node]", "2. train [launcher]", "3. resume [launcher]",
                 "--rl-stop-after-rounds 1", f"then: a verified-smoke record in {tmp_path / 'home'}"):
        assert want in out
    assert not (tmp_path / "home" / ".yeto").exists()  # no catalog, no smoke inventory
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(FAKE_SPEC))
    args = _cli(TAGGED, "--model", "org/m", "--role", "sampler", "--inventory", inv, "--sampler-spec", spec,
                "--dry-run")
    assert ia.main(args) == 0
    out = capsys.readouterr().out
    assert "1. upload [node]" in out and "-m yeto.node.smoke --spec" in out


@pytest.mark.parametrize("extra, message", [
    (["--role", "sampler"], "pass --sampler-spec"),
    (["--role", "trainer", "--model-config", REGISTRY_CONFIGS / "qwen3_30b_a3b.json"], "pass --smoke-store"),
    (["--role", "trainer", "--node", "b300-9", "--smoke-store", "s3://b/s"], "--node b300-9 is not a trainer node"),
])
def test_a_bad_request_is_refused_before_anything(tmp_path, capsys, extra, message):
    args = _cli(TAGGED, "--model", "Qwen/Qwen3-30B-A3B", "--inventory", _inventory(tmp_path), "--dry-run", *extra)
    assert ia.main(args) == 2
    assert message in capsys.readouterr().err


def test_a_failed_step_exits_1_and_says_the_catalog_is_unchanged(tmp_path, capsys, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(FAKE_SPEC))
    monkeypatch.setattr(ia, "registry_login", lambda ref: None)
    monkeypatch.setattr(ia.image_manifest, "_OPENER", SimpleNamespace(open=registry({})[0]))
    args = _cli(TAGGED, "--model", "org/m", "--role", "sampler", "--inventory", _inventory(tmp_path),
                "--sampler-spec", spec)
    assert ia.main(args, ops=FakeNodes([node(0)])) == 1
    err = capsys.readouterr().err
    assert "step manifest failed: the registry has no manifest" in err and "is unchanged" in err


def test_the_documented_commands_run_in_a_dry_run(tmp_path, capsys):
    doc = (ROOT / "docs" / "OWN_CLUSTERS.md").read_text()
    section = doc.split("## Onboarding a model image", 1)[1].split("\n## ", 1)[0]
    commands = [" ".join(c.replace("\\\n", " ").split()) for c in re.findall(r"```\n(yeto image adopt.*?)```", section,
                                                                            re.S)[0].strip().split("\n\n")]
    assert len(commands) == 2
    inv = _inventory(tmp_path)
    spec = tmp_path / "sampler-spec.json"
    spec.write_text(json.dumps(FAKE_SPEC))
    for command in commands:
        argv = shlex.split(command)[1:]
        argv = [str(inv) if a == "cluster.yaml" else str(spec) if a == "sampler-spec.json" else a for a in argv]
        if "trainer" in argv:
            argv += ["--model-config", str(REGISTRY_CONFIGS / "qwen3_30b_a3b.json")]
        args = build_parser().parse_args(argv)
        assert args.dry_run and ia.main(args) == 0, capsys.readouterr().err


class StaysAtZero(FakeEngine):
    def load(self, release_dir, version):
        return super().load(release_dir, 0)


class Silent(FakeEngine):
    def generate(self, prompts, sampling):
        import dataclasses

        return [[dataclasses.replace(c, text="") for c in group] for group in super().generate(prompts, sampling)]


def test_a_reload_that_serves_the_old_version_or_an_empty_answer_fails():
    devnull = open(os.devnull, "w")
    record = node_smoke.run_smoke(StaysAtZero(), release=None, out=devnull)
    assert (record["ok"], record["step"]) == (False, "reload") and "serves version 0" in record["error"]
    record = node_smoke.run_smoke(Silent(), release=None, out=devnull)
    assert (record["ok"], record["step"]) == (False, "generate") and "empty completion" in record["error"]


def test_the_node_smoke_entry_exits_1_on_a_failed_step(tmp_path, monkeypatch, capsys):
    (tmp_path / "adopt_fail_engine2.py").write_text(
        "from tests.test_image_smoke import ReloadFails\n"
        "def build(section):\n    return ReloadFails()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"engine": {"factory": "adopt_fail_engine2:build"}}))
    assert node_smoke.main(["--spec", str(spec)]) == 1
    assert node_smoke.parse_result(capsys.readouterr().out)["step"] == "reload"


def test_the_sampler_smoke_reloads_the_release_given():
    plan = sm.sampler_plan(node(0), spec=FAKE_SPEC, run_id="adopt-t", release="/models/r1", code=("t000", b"tar"))
    assert shlex.split(plan.steps[1].script)[-1].endswith(
        "-m yeto.node.smoke --spec /work/image-adopt/adopt-t/spec.json --release /models/r1")
