"""Rendered argv under production defaults.

tests/conftest.py relaxes production defaults for every test (tests/prod_defaults.py
lists them). With them, an argv that yeto renders for another process passes the suite
even when that process refuses it: an image smoke's trainer argv without ``--inner-lr``
and without ``--preflight-model off`` was green here and refused on the node. Each test
here puts the production defaults back, renders the argv the way the product does and
hands it to the real parser of the process that receives it:

* ``run up`` -> ``yeto launch --dry-run`` and ``python -m yeto.launch.nc pool --dry-run``
  (the own-cluster templates, the K3.1 customer configs);
* a fleet's compiled trainer argv and each island's own launch (``island_launches.launch_argv``,
  what ``trainer_worker`` runs as ``python -m yeto.cli launch``);
* the island entry command of a Modal, a sky and an ``ssh:`` trainer island, parsed by the
  island's own parser with only the environment that island is given;
* the sampler spec an NC pool renders for a node, read the way ``engine_sampler`` reads it.

The image smoke's own trainer plan is not on main yet (image adopt); its argv goes through
the same :func:`prod_defaults.launch_dry_run` there."""

from __future__ import annotations

import json
import shlex
import sys
import types
from pathlib import Path

import prod_defaults
import pytest
from test_own_cluster_templates import TEMPLATES, kit  # noqa: F401 - the fixture

ROOT = Path(__file__).resolve().parents[1]
MILES_ENTRY = "yeto.rl.adapters.miles.island_entry"
SHELL_OPERATORS = ("&&", ";", "|", "||", ")")


@pytest.fixture
def fake_sky(monkeypatch):
    """The sky module of tests/test_rl_launcher.py: tasks are built in memory, nothing is asked."""
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    monkeypatch.setattr("yeto.launch.modal_islands._list_modal_apps", lambda: [])


def _launches(commands):
    return [c[4:] for c in commands if c[2] == "yeto.cli" and c[3] == "launch"]


def _without_inner_lr(argv):
    argv = [str(a) for a in argv]
    if "--inner-lr" in argv:
        k = argv.index("--inner-lr")
        del argv[k:k + 2]
    return [a for a in argv if not a.startswith("--inner-lr=")]


# -- run up -> yeto launch / nc pool: the own-cluster templates ----------------------------------

@pytest.mark.parametrize("name", TEMPLATES)
def test_run_up_dry_run_of_each_own_cluster_template_passes_with_production_defaults(
        kit, name, production_defaults, capsys):  # noqa: F811
    from test_own_cluster_templates import _dry_run

    asked = production_defaults
    rc, out = _dry_run(kit, name, capsys)
    assert rc == 0, out.why
    assert "dry run: preflight ok, nothing ordered" in out
    assert _launches(kit.commands), kit.commands
    assert asked["hub"] or asked["store"], "the trainer's model preflight did not run"


def test_a_run_config_without_inner_lr_is_refused_only_with_production_defaults(kit, monkeypatch, capsys):  # noqa: F811
    """The check above can fail: the same template without --inner-lr passes under the
    conftest defaults (what hid the image smoke's argv) and is refused under production's."""
    from test_own_cluster_templates import _dry_run

    name = "run.own-trainer-modal-sampler.json"
    path = kit.files[name]
    doc = json.loads(path.read_text())
    doc["trainer"]["argv"] = _without_inner_lr(doc["trainer"]["argv"])
    path.write_text(json.dumps(doc))
    rc, out = _dry_run(kit, name, capsys)
    assert rc == 0, out.why  # conftest: DEFAULT_INNER_LR is a plain float
    prod_defaults.apply(monkeypatch)
    rc, out = _dry_run(kit, name, capsys)
    assert rc != 0 and "pass --inner-lr explicitly" in out


# -- the K3.1 customer: run up, its trainer launch, its island and its pool ------------------------

def test_the_customer_bare_metal_run_up_dry_run_passes_with_production_defaults(tmp_path, monkeypatch, capsys):
    """examples/k31/run.baremetal-b300-262k.json through ``run up --dry-run`` as in
    tests/test_k31_baremetal_run_up_dry_run.py, but with the production defaults put back
    after that harness sets ``YETO_PREFLIGHT_MODEL=off``."""
    from test_k31_baremetal_run_up_dry_run import _run_up
    from yeto.rl import up

    real_main = up.main
    asked = {}

    def main(args):
        asked.update(prod_defaults.apply(monkeypatch))
        return real_main(args)

    monkeypatch.setattr(up, "main", main)
    lines = _run_up(tmp_path, monkeypatch, capsys)
    assert "dry run: preflight ok, nothing ordered" in lines
    assert asked["hub"] or asked["store"], "the trainer's model preflight did not run"


def _customer_launch(case, tmp_path, monkeypatch):
    """The customer's trainer argv (tests/test_k31_customer_plan_golden.py) through
    ``yeto launch --dry-run`` with production defaults; returns (rc, output, prepared args)."""
    import yeto.cli as cli
    import yeto.launcher as launcher
    import yeto.modal_runner as modal_runner
    from test_k31_customer_plan_golden import CASES, _argv, _hermetic_env
    from yeto.cloud import s3_store

    _hermetic_env(tmp_path, monkeypatch)
    for key, value in (("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test"), ("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test"),
                       ("HF_HUB_OFFLINE", "1"), ("HF_HUB_CACHE", str(tmp_path / "hf")),
                       ("HF_HOME", str(tmp_path / "hf-home"))):
        monkeypatch.setenv(key, value)
    for key in (s3_store.ENDPOINT_ENV, s3_store.REGION_ENV, s3_store.PROFILE_ENV, "AWS_ENDPOINT_URL", "AWS_PROFILE"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)
    ordered = []
    monkeypatch.setattr(launcher, "run", lambda *a, **k: ordered.append("run"))
    monkeypatch.setattr(cli, "_spawn_worker", lambda *a, **k: ordered.append("spawn"))
    monkeypatch.setattr(cli.runs, "create_run", lambda *a, **k: ordered.append("create"))
    prepared = []
    real_plan = launcher.dry_run_plan
    monkeypatch.setattr(launcher, "dry_run_plan", lambda args: prepared.append(args) or real_plan(args))
    asked = prod_defaults.apply(monkeypatch)
    rc, out = prod_defaults.launch_dry_run(_argv(CASES[case], tmp_path))
    assert ordered == []
    return rc, out, (prepared[0] if prepared else None), asked


@pytest.mark.parametrize("case", ["examples/run.agentic-262k.json", "examples/run.baremetal-b300-262k.json",
                                  "kit/run.json"])
def test_the_customer_trainer_launch_passes_with_production_defaults(case, tmp_path, monkeypatch, fake_sky):
    rc, out, args, asked = _customer_launch(case, tmp_path, monkeypatch)
    assert rc == 0, prod_defaults.refusal(out)
    assert args is not None and (asked["hub"] or asked["store"])  # the model preflight ran, in check mode


def _entry_argv(run: str, env: dict, module: str = MILES_ENTRY) -> list[str]:
    """The island entry argv of a run script, with the shell variables the island's own
    environment gives (``$LEARNER_ID``, ``$SYNCER_ADDR``) put in."""
    line = next(line for line in run.splitlines() if f"-m {module}" in line)
    tokens = shlex.split(line.split(f"-m {module}", 1)[1])
    tokens = tokens[:next((i for i, t in enumerate(tokens) if t in SHELL_OPERATORS), len(tokens))]
    return [env.get(t[1:], t) if t.startswith("$") and t[1:].isidentifier() else t for t in tokens]


def _island_parses(run: str, env: dict, monkeypatch, capsys) -> types.SimpleNamespace:
    """The island's parser and its compat group, with only ``env`` of what this process
    has from conftest (the island runs elsewhere)."""
    from yeto.hw.catalog import COMPAT_GROUP_ENV
    from yeto.rl.adapters.miles import island_entry

    for name in prod_defaults.CONFTEST_ENV:
        monkeypatch.delenv(name, raising=False)
    if env.get(COMPAT_GROUP_ENV):
        monkeypatch.setenv(COMPAT_GROUP_ENV, str(env[COMPAT_GROUP_ENV]))
    argv = _entry_argv(run, env)
    try:
        args = island_entry.parse_args(argv)
    except SystemExit as stop:
        raise AssertionError(f"the island entry refused its argv: {capsys.readouterr().err[-2000:]}") from stop
    group = island_entry._island_compat_group(args)
    return types.SimpleNamespace(args=args, group=group, argv=argv)


def _island_task(args):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task
    from yeto.launcher import _island_overrides, _island_task as island_task
    from yeto.rl.engine import sample_roles

    specs = parse_gpu_spec(args.gpu)
    return specs[0], island_task(make_miles_island_task, args, specs[0], 0,
                                 sample_roles.syncer_learner_count(args, len(specs)), "10.0.0.1:29400",
                                 _island_overrides(args, specs))


def test_the_customer_modal_island_entry_parses_with_only_its_own_environment(tmp_path, monkeypatch, fake_sky,
                                                                             capsys):
    from yeto.launch.modal_islands import build_modal_island_config

    rc, out, args, _ = _customer_launch("examples/run.agentic-262k.json", tmp_path, monkeypatch)
    assert rc == 0, prod_defaults.refusal(out)
    spec, task = _island_task(args)
    assert spec.cloud == "modal"
    cfg = build_modal_island_config(args, spec, 0, task, "10.0.0.1:29400")
    island = _island_parses(cfg.run_script, {**cfg.envs, "LEARNER_ID": "0"}, monkeypatch, capsys)
    assert island.group == "nvidia-b300"
    assert island.args.rl_single_island_no_sync and island.args.inner_lr == float(args.inner_lr)


@pytest.mark.parametrize("case", ["examples/run.baremetal-b300-262k.json", "kit/run.json"])
def test_the_customer_ssh_island_entry_parses_with_only_the_node_environment(case, tmp_path, monkeypatch, fake_sky,
                                                                            capsys):
    from yeto.launch import ssh_trainer_island as sti
    from yeto.launch.trainer_provision import ssh_trainer_spec

    rc, out, args, _ = _customer_launch(case, tmp_path, monkeypatch)
    assert rc == 0, prod_defaults.refusal(out)
    spec, task = _island_task(args)
    assert spec.cloud == "ssh"
    ssh_spec = ssh_trainer_spec(args, spec, task, name="island-0")
    env = sti.node_env(ssh_spec, 0, {})
    script = sti.start_script(ssh_spec, 0, env, sti.remote_paths("r", "island-0", 0), container="c0")
    assert "export YETO_RL_COMPAT_GROUP=nvidia-b300" in script  # the node's job gets it
    island = _island_parses(ssh_spec.run_script, {**env, "LEARNER_ID": "0"}, monkeypatch, capsys)
    assert island.group == "nvidia-b300"


def test_the_customer_pool_dry_run_and_the_node_spec_pass_with_production_defaults(tmp_path, monkeypatch, capsys):
    """``python -m yeto.launch.nc pool --dry-run`` of examples/k31's pool (what run up runs),
    then the spec it renders for a node read as ``engine_sampler`` reads it on the node."""
    from test_k31_baremetal_run_up_dry_run import _fill
    from yeto.launch.nc import cli as nc_cli
    from yeto.rl.engine import engine_sampler_entry as entry
    from yeto.rl.engine.rollout_data_path import normalize_spec

    _fill(tmp_path)
    config = tmp_path / "pool.nc-mi355x-agentic-262k.json"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("YETO_NC_ACCOUNTS", "101,102,103,104")
    monkeypatch.setenv("YETO_NC_STATE_DIR", str(tmp_path / "nc-state"))
    asked = prod_defaults.apply(monkeypatch)
    rc = nc_cli.main(["pool", "--config", str(config), "--dry-run"])
    out = capsys.readouterr()
    assert rc in (0, None), prod_defaults.refusal(out.out + out.err)
    assert asked["store"], "the pool's model check did not run"
    from yeto.launch.pool.island_pool import Member

    raw = json.loads(config.read_text())
    pc = nc_cli.pool_config(raw, allowed=None)
    member = Member(pc.accounts[0], "<ticket>", f"{pc.shard_base}-0",
                    data_slot=f"0/{pc.data_slots}" if pc.data_slots else None)
    spec_file = tmp_path / "node-spec.json"  # the file the pool uploads; the node reads it back
    spec_file.write_text(json.dumps(nc_cli.render_spec(raw, member, [f"{pc.shard_base}-{i}"
                                                                     for i in range(pc.target)])))
    spec = entry.apply_rollout_quant(normalize_spec(entry.load_spec(spec_file)))
    entry.island_config(spec)  # the node builds its island config from it before the engine starts


# -- a fleet: the compiled trainer argv and each island's own launch ------------------------------

FLEETS = ("examples/fleet/run.two-islands.json", "examples/fleet/run.mixed-miles.json")


def _fleet(tmp_path, example: str, *, per_island: bool) -> dict:
    from test_fleet_island_launches import example_doc
    from yeto.rl import up
    import test_fleet_island_launches as fleet_tests

    monkeypatched = fleet_tests.EXAMPLE
    try:
        fleet_tests.EXAMPLE = ROOT / example
        doc = example_doc(tmp_path, per_island=per_island)
    finally:
        fleet_tests.EXAMPLE = monkeypatched
    text = json.dumps(doc)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"q": f"{i}+1?", "a": str(i + 1)}) + "\n" for i in range(64)))
    text = text.replace("<prompts.jsonl>", str(prompts))
    path = tmp_path / "run.json"
    path.write_text(text)
    return up.load_config(str(path))


@pytest.fixture
def fleet_env(monkeypatch):
    import yeto.launcher as launcher
    import yeto.modal_runner as modal_runner
    import yeto.cli as cli

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "fleet-prod-defaults-key")
    monkeypatch.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test-key-id")
    monkeypatch.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)
    monkeypatch.setattr(launcher, "run", lambda *a, **k: pytest.fail("a dry run ordered"))
    monkeypatch.setattr(cli, "_spawn_worker", lambda *a, **k: pytest.fail("a dry run spawned a worker"))


@pytest.mark.parametrize("example", FLEETS)
def test_a_fleet_s_compiled_trainer_argv_passes_with_production_defaults(example, tmp_path, fake_sky, fleet_env,
                                                                        production_defaults):
    cfg = _fleet(tmp_path, example, per_island=False)
    asked = production_defaults
    rc, out = prod_defaults.launch_dry_run([*cfg["trainer"]["argv"], "--controller", "local"])
    assert rc == 0, prod_defaults.refusal(out)
    assert asked["hub"]


@pytest.mark.parametrize("example", FLEETS[:1])  # a Miles-sampler island is one launch with its trainer
def test_each_fleet_island_launch_passes_with_production_defaults(example, tmp_path, monkeypatch, fake_sky,
                                                                 fleet_env, capsys):
    """What ``trainer_worker`` runs per island (``python -m yeto.cli launch`` of
    ``island_launches.launch_argv``), then that island's entry command parsed with only the
    environment its Modal island config gives it."""
    import yeto.launcher as launcher
    from yeto.launch.modal_islands import build_modal_island_config
    from yeto.rl import fleet, island_launches

    cfg = _fleet(tmp_path, example, per_island=True)
    plan = fleet.plan_of(cfg)
    asked = prod_defaults.apply(monkeypatch)
    rc, out = prod_defaults.launch_dry_run([*cfg["trainer"]["argv"], "--controller", "local"])  # the syncer launch
    assert rc == 0, prod_defaults.refusal(out)
    prepared = []
    real_plan = launcher.dry_run_plan
    monkeypatch.setattr(launcher, "dry_run_plan", lambda args: prepared.append(args) or real_plan(args))
    members = island_launches.trainer_members(cfg, plan)
    assert len(members) == len(plan.islands) >= 2
    for member, island in zip(members, plan.islands):
        argv = island_launches.launch_argv(member, syncer={"addr": "10.0.0.1:29400", "syncer_epoch": 3})
        rc, out = prod_defaults.launch_dry_run(argv)
        assert rc == 0, f"island {island.learner}: {prod_defaults.refusal(out)}"
        args = prepared[-1]
        spec, task = _island_task(args)
        modal = build_modal_island_config(args, spec, 0, task, "10.0.0.1:29400")
        parsed = _island_parses(modal.run_script, {**modal.envs, "LEARNER_ID": str(island.learner)},
                                monkeypatch, capsys)
        assert int(parsed.args.learner_id) == island.learner
        assert parsed.group == "nvidia-h100"
    assert asked["hub"]


def test_a_sky_island_entry_parses_with_only_the_task_environment(tmp_path, monkeypatch, fake_sky, fleet_env,
                                                                  capsys):
    """A sky trainer island (island 0 of the per-island two-island fleet moved to an AWS
    cluster): its launch dry run, then the task's run script parsed with only the task's envs."""
    import yeto.launcher as launcher
    from yeto.rl import fleet, island_launches

    cfg = _fleet(tmp_path, FLEETS[0], per_island=True)
    member = dict(island_launches.trainer_members(cfg, fleet.plan_of(cfg))[0], gpu="aws:1x2xh100@us-east-2")
    prepared = []
    real_plan = launcher.dry_run_plan
    monkeypatch.setattr(launcher, "dry_run_plan", lambda args: prepared.append(args) or real_plan(args))
    prod_defaults.apply(monkeypatch)
    rc, out = prod_defaults.launch_dry_run(island_launches.launch_argv(member, syncer={"addr": "10.0.0.1:29400"}))
    assert rc == 0, prod_defaults.refusal(out)
    spec, task = _island_task(prepared[-1])
    assert spec.cloud == "aws"
    envs = {**task.envs, **{k: str(v) for k, v in (getattr(task, "secrets", None) or {}).items()}}
    parsed = _island_parses(task.run, {**envs, "LEARNER_ID": "0", "SYNCER_ADDR": "10.0.0.1:29400"},
                            monkeypatch, capsys)
    assert parsed.group == "nvidia-h100"


def test_a_missing_model_passes_under_conftest_and_is_refused_with_production_defaults(tmp_path, monkeypatch,
                                                                                     fake_sky, fleet_env):
    """The model preflight half of the image smoke's miss: conftest's YETO_PREFLIGHT_MODEL=off
    lets a launch whose model the Hub does not have pass; production refuses it before any
    order, so a rendered argv that must skip the check has to carry --preflight-model off."""
    from yeto.launch import model_check

    argv = [*_fleet(tmp_path, FLEETS[0], per_island=False)["trainer"]["argv"], "--controller", "local"]
    prod_defaults.apply(monkeypatch)
    monkeypatch.setattr(model_check, "hub_files", lambda repo, revision: ([], lambda name: None))  # no such files
    rc, out = prod_defaults.launch_dry_run(argv)
    assert rc != 0 and "trainer model" in out, prod_defaults.refusal(out)
    rc, out = prod_defaults.launch_dry_run([*argv, "--preflight-model", "off"])
    assert rc == 0, prod_defaults.refusal(out)
    monkeypatch.setenv(model_check.MODE_ENV, "off")  # what conftest sets for every test
    rc, out = prod_defaults.launch_dry_run(argv)
    assert rc == 0, prod_defaults.refusal(out)
