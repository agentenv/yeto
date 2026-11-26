"""The trainer image ``yeto run up``'s cluster preflight pulls and probes: the image the
trainer's launch runs. A trainer argv without ``--rl-image`` gets the image the launch
fills in (the site image catalog's record for the model on these nodes, else the pinned
default; :func:`yeto.launch.rl_backend.rl_image_choice`), passed as ``--image``; the
preflight subcommand runs here in this process over a fake ssh (not short-circuited), so
the pull command it runs is checked. A preflight that has no image FAILs ("no trainer
image resolved") and never pulls an image named "None"."""

from __future__ import annotations

import collections
import json
import time
import types

import pytest
import yaml

from tests.test_cluster_preflight import Behavior, FakeCluster
from tests.test_own_cluster_templates import FakeSsh, _dry_run, kit  # noqa: F401 - the kit fixture
from tests.test_site_images import record, write_catalog
from yeto import cli, runs
from yeto.launch.preflight import cluster as cp
from yeto.launch.preflight import gate

TEMPLATE = "run.own-trainer-sky-sampler.json"
ARGV_REF = "ghcr.io/example/own-trainer@sha256:" + "1" * 64
SITE_REF = "registry.example/site/qwen3-miles@sha256:" + "2" * 64


class OwnNodes(FakeCluster):
    """The ssh of ``yeto preflight`` and of the egress check on the template's own nodes:
    every preflight script answered as a healthy B300 node would, the egress probe as
    :class:`FakeSsh` (every endpoint reached)."""

    scripts: list[tuple[str, str]] = []

    def __init__(self):  # SshOps(): built by the preflight with no arguments
        self.nodes, self.by_addr = {}, {}
        self.b = collections.defaultdict(Behavior)
        self.scripts = OwnNodes.scripts
        self.NOW = time.time()

    def run(self, node, script, *, timeout=300.0):
        if "@@yeto-section reach" in script:
            return FakeSsh().run(node, script, timeout=timeout)
        self.nodes.setdefault(node.name, node)
        self.by_addr.setdefault(node.address, node.name)
        return super().run(node, script, timeout=timeout)


@pytest.fixture
def real_up(kit, monkeypatch, tmp_path):  # noqa: F811 - the kit fixture
    """The kit with the preflight subcommand run in this process (``yeto preflight``
    through ``yeto.cli.main``) over :class:`OwnNodes`; run up stops before its first order."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cp, "SshOps", OwnNodes)
    OwnNodes.scripts = []
    dry = kit.up.Run._cmd
    preflights = []

    def cmd(self, argv, *, log, timeout=None):
        argv = [str(a) for a in argv]
        if argv[3:4] == ["preflight"]:
            kit.commands.append(argv)
            preflights.append(argv[3:])
            try:
                return int(cli.main(argv[3:]) or 0)
            except SystemExit as stop:
                return int(stop.code or 0)
        return dry(self, argv, log=log, timeout=timeout)

    monkeypatch.setattr(kit.up.Run, "_cmd", cmd)
    monkeypatch.setattr(kit.up.Run, "write_fleet_plan", lambda self: False)  # stop before the first order
    kit.preflights = preflights
    return kit


def _edit_trainer(kit, fn):
    path = kit.files[TEMPLATE]
    cfg = json.loads(path.read_text())
    fn(cfg["trainer"]["argv"])
    path.write_text(json.dumps(cfg))


def _edit_inventory(kit, **top):
    path = kit.files["inventory.own-nodes.yaml"]
    doc = yaml.safe_load(path.read_text())
    doc.update(top)
    path.write_text(yaml.safe_dump(doc))


def _pulls() -> list[str]:
    return [s for _, s in OwnNodes.scripts if s.startswith("docker pull")]


def _run(kit, capsys):
    rc, out = _dry_run(kit, TEMPLATE, capsys, dry_run=False)
    (argv,) = kit.preflights
    image = argv[argv.index("--image") + 1] if "--image" in argv else None
    return rc, out, image


def _launch_image(kit) -> str:
    """The ``--rl-image`` the trainer's own launch resolves (``yeto launch`` parses the
    argv, ``resolve_default_rl_image`` fills it in)."""
    from yeto.launch.rl_backend import resolve_default_rl_image
    from yeto.rl.up_consistency import trainer_args

    argv = [str(a) for a in json.loads(kit.files[TEMPLATE].read_text())["trainer"]["argv"]]
    args = trainer_args(argv)
    resolve_default_rl_image(args)
    return args.rl_image


def _passed(rc, out):
    assert rc == 2 and "fleet plan not written: nothing ordered" in out, out.why
    assert "cluster preflight failed" not in out


def test_an_argv_rl_image_is_the_image_pulled(real_up, capsys):
    _edit_trainer(real_up, lambda argv: argv.extend(["--rl-image", "docker:" + ARGV_REF]))
    rc, out, image = _run(real_up, capsys)
    _passed(rc, out)
    assert image == "docker:" + ARGV_REF == _launch_image(real_up)
    assert _pulls() == [f"docker pull -q {ARGV_REF}"]


def test_a_site_catalog_record_is_the_image_pulled(real_up, capsys, tmp_path):
    catalog = write_catalog(tmp_path / "site-images.yaml", record(model="Qwen/Qwen3-4B", ref=SITE_REF))
    _edit_inventory(real_up, site_images=str(catalog))
    rc, out, image = _run(real_up, capsys)
    _passed(rc, out)
    assert image == "docker:" + SITE_REF == _launch_image(real_up)
    assert _pulls() == [f"docker pull -q {SITE_REF}"]
    assert f"--image docker:{SITE_REF}" in out  # the command run up logs


def test_without_any_image_the_pinned_default_is_pulled(real_up, capsys):
    from yeto.launch.rl_backend import _builtin_rl_image

    rc, out, image = _run(real_up, capsys)
    _passed(rc, out)
    from yeto.rl.adapters.miles import pins

    default = _launch_image(real_up)
    assert image == default and default == _builtin_rl_image(types.SimpleNamespace(rl_backend=None,
                                                                                    rl_model_recipe=None,
                                                                                    tuning="full"))
    ref = default.removeprefix("docker:")
    ran = pins.overlay_safe_image(ref) or ref  # the image the trainer runs (ssh_trainer_island.run_image)
    assert "@sha256:" in ref and _pulls() == [f"docker pull -q {ran}"]
    assert not any("None" in s for s in _pulls())


def test_an_inventory_image_is_checked_against_the_run_image_as_the_launch_does(real_up, capsys):
    """An inventory ``images:`` entry turns the catalog off; the run's --rl-image (here the
    pinned default) decides, as in the launch (ssh_trainer_island.run_image): the same
    image is pulled, another one is refused by the preflight before any pull."""
    from yeto.rl.adapters.miles import pins

    default = _launch_image(real_up).removeprefix("docker:")
    _edit_inventory(real_up, images={"x86_64": default})
    rc, out, image = _run(real_up, capsys)
    _passed(rc, out)
    ran = pins.overlay_safe_image(default) or default
    assert image == _launch_image(real_up) and _pulls() == [f"docker pull -q {ran}"]
    real_up.preflights.clear()
    OwnNodes.scripts.clear()
    _edit_inventory(real_up, images={"x86_64": ARGV_REF})
    rc, out = _dry_run(real_up, TEMPLATE, capsys, dry_run=False)
    assert rc == 2 and f"inventory image ['{ARGV_REF}'] of nodes ['own-b300-0'] is not the run's --rl-image" in out
    assert _pulls() == []  # refused before any node pulls (the launch's dry run, or the preflight itself)


def test_the_dry_run_prints_the_resolved_image(kit, capsys, tmp_path):  # noqa: F811
    catalog = write_catalog(tmp_path / "site-images.yaml", record(model="Qwen/Qwen3-4B", ref=SITE_REF))
    _edit_inventory(kit, site_images=str(catalog))
    rc, out = _dry_run(kit, TEMPLATE, capsys, check_nodes=False)
    assert rc == 0, out.why
    line = next(line for line in out.splitlines() if "run up would run the trainer's cluster preflight" in line)
    assert f"--image docker:{SITE_REF} --no-bench" in line


# -- no image: FAIL, never "None" ------------------------------------------------------


def _bare_inventory(tmp_path, image=None):
    node = {"name": "b300-0", "host": "b300-0", "address": "192.0.2.10", "device_family": "cuda", "cpu_arch": "x86_64",
            "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, "fabric": "ib",
            "runtime": "docker", "docker_args": ["--gpus", "all"]}
    if image:
        node["image"] = image
    path = tmp_path / "inv.yaml"
    path.write_text(yaml.safe_dump({"nodes": [node]}))
    return path


def test_the_preflight_cli_refuses_docker_nodes_without_an_image(tmp_path, monkeypatch, capsys):
    """``yeto preflight`` with no --image and an inventory that names none: exit 2 before
    any ssh, naming the fix; host checks only (--no-pull --no-container) need no image."""
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cp, "SshOps", OwnNodes)
    OwnNodes.scripts = []
    inv = _bare_inventory(tmp_path)
    base = ["preflight", "--gpu", "ssh:1x8xb300", "--rl-node-inventory", str(inv), "--cluster-prefix", "oc"]
    assert cli.main([*base, "--no-bench", "--run-dir", str(tmp_path / "pf")]) == 2
    err = capsys.readouterr().err
    assert f"{cp.NO_IMAGE} for the docker nodes: {cp.NO_IMAGE_FIX}" in err
    assert OwnNodes.scripts == [] and not (tmp_path / "pf").exists()
    cli.main([*base, "--no-pull", "--no-container", "--run-dir", str(tmp_path / "host")])
    assert OwnNodes.scripts and not any("docker pull" in s or "docker run" in s for _, s in OwnNodes.scripts)
    assert (tmp_path / "host" / "preflight.json").exists()


def test_the_pull_fails_per_node_without_an_image_and_the_report_is_written(tmp_path):
    """ClusterPreflight itself (every caller, not only the CLI): a docker node with no image
    FAILs its pull ("no trainer image resolved") and is not probed; a node with its own image
    is pulled; nothing named "None" is pulled or run; the report is written."""
    from tests.test_cluster_preflight import IMAGE, b300

    OwnNodes.scripts = []
    nodes = [b300(0, image=None), b300(1, image=IMAGE)]
    report = cp.ClusterPreflight(nodes, image=None, ops=OwnNodes(), steps=cp.Steps(ib=False, nccl=False),
                                 out_dir=tmp_path / "pf", log=lambda m: None).run()
    pulls = {c.scope: c for c in report.checks if c.name == "image_pull"}
    assert pulls["b300-0"].status == cp.FAIL and pulls["b300-0"].detail == cp.NO_IMAGE
    assert pulls["b300-0"].fix == cp.NO_IMAGE_FIX and pulls["b300-1"].status == cp.PASS
    assert _pulls() == [f"docker pull -q {IMAGE}"]
    runs_on = [n for n, s in OwnNodes.scripts if "docker run" in s]
    assert runs_on and set(runs_on) == {"b300-1"}  # the node without an image is not probed
    assert not any("None" in s for _, s in OwnNodes.scripts)
    saved = json.loads((tmp_path / "pf" / "preflight.json").read_text())
    assert any(c["name"] == "image_pull" and c["detail"] == cp.NO_IMAGE for c in saved["checks"])


@pytest.mark.parametrize("image", [None, ""])
def test_the_container_probe_fails_without_an_image(tmp_path, image):
    from tests.test_cluster_preflight import b300

    OwnNodes.scripts = []
    node = b300(0, image=None)
    pf = cp.ClusterPreflight([node], image=image, ops=OwnNodes(), steps=cp.Steps(pull=False, ib=False, nccl=False),
                             log=lambda m: None)
    report = pf.run()
    (check,) = [c for c in report.checks if c.name == "container"]
    assert check.status == cp.FAIL and check.detail == cp.NO_IMAGE and check.fix == cp.NO_IMAGE_FIX
    assert not any("docker run" in s for _, s in OwnNodes.scripts)


def test_a_node_image_is_pulled_when_no_image_is_given(tmp_path):
    from tests.test_cluster_preflight import IMAGE, b300

    OwnNodes.scripts = []
    pf = cp.ClusterPreflight([b300(0, image=IMAGE)], image=None, ops=OwnNodes(), steps=cp.Steps(ib=False, nccl=False),
                             log=lambda m: None)
    report = pf.run()
    assert [c.status for c in report.checks if c.name == "image_pull"] == [cp.PASS]
    assert _pulls() == [f"docker pull -q {IMAGE}"]


# -- the gate's commands ---------------------------------------------------------------


SSH_ARGV = ["--training-mode", "rl", "--gpu", "ssh:1x8xb300", "--cluster-prefix", "oc", "--model", "Qwen/Qwen3-4B",
            "--data", "prompts.jsonl"]


def test_commands_carry_the_image_the_launch_fills_in(tmp_path, monkeypatch):
    from yeto.launch.rl_backend import _builtin_rl_image

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    inv = _bare_inventory(tmp_path)
    argv = [*SSH_ARGV, "--rl-node-inventory", str(inv)]
    (cmd,) = gate.commands_for_launch_argv(argv)
    default = _builtin_rl_image(types.SimpleNamespace(rl_backend=None, rl_model_recipe=None, tuning="lora"))
    assert cmd[cmd.index("--image") + 1] == default
    catalog = write_catalog(tmp_path / "c.yaml", record(model="Qwen/Qwen3-4B", ref=SITE_REF))
    inv.write_text(inv.read_text() + f"site_images: {catalog}\n")
    (cmd,) = gate.commands_for_launch_argv(argv)
    assert cmd[cmd.index("--image") + 1] == "docker:" + SITE_REF
    (cmd,) = gate.commands_for_launch_argv([*argv, "--rl-image", "docker:" + ARGV_REF])
    assert cmd[cmd.index("--image") + 1] == "docker:" + ARGV_REF
    # the egress check probes the registry of the same image
    (registry,) = gate.launch_endpoints(argv, {})
    assert registry.url == "https://registry.example/" and registry.source == SITE_REF


def test_a_catalog_the_launch_refuses_gives_no_image(tmp_path, monkeypatch):
    """A broken site catalog refuses the launch (it names why); the preflight command
    then carries no --image and FAILs on its own instead of pulling a guess."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [")
    inv = _bare_inventory(tmp_path)
    inv.write_text(inv.read_text() + f"site_images: {bad}\n")
    (cmd,) = gate.commands_for_launch_argv([*SSH_ARGV, "--rl-node-inventory", str(inv)])
    assert "--image" not in cmd
    with pytest.raises(ValueError, match="not valid YAML"):
        gate.launch_rl_image([*SSH_ARGV, "--rl-node-inventory", str(inv)])


def test_only_a_refusal_is_turned_into_no_image(monkeypatch):
    """A ValueError of the resolution (the launch's own refusal) gives no --image; any other
    error is a bug and is raised, not hidden behind a preflight FAIL."""
    from yeto.launch import rl_backend

    argv = [*SSH_ARGV, "--rl-node-inventory", "inv.yaml"]

    def boom(args):
        raise RuntimeError("a bug")

    monkeypatch.setattr(rl_backend, "rl_image_choice", boom)
    with pytest.raises(RuntimeError, match="a bug"):
        gate.commands_for_launch_argv(argv)
    with pytest.raises(RuntimeError, match="a bug"):
        gate.launch_endpoints(argv, {})


def test_the_customer_preflight_command_is_the_one_of_its_rl_image():
    """The customer K3.1 baremetal config names --rl-image: its preflight command is exactly
    what it was (the argv's own --rl-image; nothing resolved, the argv not even parsed)."""
    from pathlib import Path

    cfg = json.loads((Path(__file__).resolve().parents[1] / "examples/k31/run.baremetal-b300-262k.json").read_text())
    argv = [str(a) for a in cfg["trainer"]["argv"]]
    before = gate.preflight_commands(gate.argv_value(argv, "--gpu"), gate.argv_value(argv, "--rl-node-inventory"),
                                     gate.argv_value(argv, "--cluster-prefix"), gate.argv_value(argv, "--rl-image"))
    assert before and gate.argv_value(argv, "--rl-image").startswith("docker:")
    assert gate.commands_for_launch_argv(argv) == before
    assert gate.launch_rl_image(argv) == gate.argv_value(argv, "--rl-image")


def test_an_argv_the_launch_parser_refuses_keeps_its_own_image():
    assert "--image" not in gate.commands_for_launch_argv(["--gpu", "ssh:1x8xb300", "--bogus-flag"])[0]
    assert gate.commands_for_launch_argv(["--gpu", "modal:1x8xb300"]) == []
