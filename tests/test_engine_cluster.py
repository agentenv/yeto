"""S27 G-DI two-cluster island (yeto.launch.engine_cluster): plan, tasks, launch, ops.

CPU only: no sky, no Ray. A fake sky module records launches, execs and cancels."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import engine_cluster as ec
from yeto.rl.engine.multinode import ISLAND_NODE_WAIT_ENV, wait_for_island_nodes

from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

ISLAND = "nebius:2x8xh200@eu-north1"
ENGINE = "nebius:8xh100@eu-north1"


def _fake_sky(monkeypatch):
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    return sys.modules["sky"]


def _rl_args(gpu=ISLAND, engine=ENGINE, extra=()):
    argv = ["--gpu", gpu, "--rl-engine", "ports", "--rl-placement", "fixed-partition",
            "--rl-rollout-gpus", "8", "--rollout-num-gpus-per-engine", "2",
            "--tensor-parallel", "1", *extra]
    if engine is not None:
        argv += ["--rl-engine-cluster", engine]
    args = _args(tuple(argv))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    return args


def _plan(args):
    specs = parse_gpu_spec(args.gpu)
    return ec.plan(args, specs, launcher.learner_cluster_names("run1", specs))


def _make(args):
    spec = parse_gpu_spec(args.gpu)[0]
    return lambda: launcher.make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400")


# ---- plan

def test_flag_unset_plans_nothing(monkeypatch):
    _fake_sky(monkeypatch)
    assert _plan(_rl_args(engine=None)) == {}
    assert not ec.requested(_rl_args(engine=None))


def test_plan_splits_the_last_node(monkeypatch):
    _fake_sky(monkeypatch)
    (split,) = _plan(_rl_args()).values()
    assert split.name == "run1-l0-eu-north1" and split.engine_name == "run1-l0-eu-north1-eng"
    assert str(split.trainer) == "nebius:1x8xH200@eu-north1"
    assert str(split.engine) == "nebius:1x8xH100@eu-north1"
    assert split.rank_offset == 1
    rec = split.record()
    assert rec["engine_island_ranks"] == [1] and rec["nccl"]["NCCL_IB_DISABLE"] == "1"
    assert ec.nodes_by_cluster({split.name: split}) == {split.name: 1, split.engine_name: 1}


def test_plan_per_island_entries(monkeypatch):
    _fake_sky(monkeypatch)
    args = _rl_args(gpu=f"{ISLAND},{ISLAND}", engine=f"none,{ENGINE}")
    splits = _plan(args)
    assert [s.learner_id for s in splits.values()] == [1]
    with pytest.raises(ValueError, match="give one, or one per"):
        _plan(_rl_args(gpu=f"{ISLAND},{ISLAND}", engine=f"{ENGINE},{ENGINE},{ENGINE}"))


@pytest.mark.parametrize("gpu, engine, extra, match", [
    ("nebius:2x8xh200@eu-north1", "nebius:8xh100@eu-west1", (), "same explicit region"),
    ("nebius:2x8xh200", "nebius:8xh100", (), "same explicit region"),
    ("nebius:2x8xh200@eu-north1", "aws:8xh100@eu-north1", (), "both clusters must be on"),
    ("nebius:2x8xh200@eu-north1", "nebius:4xh100@eu-north1", (), "GPUs per engine node"),
    ("nebius:3x8xh200@eu-north1", "nebius:8xh100@eu-north1", (), "exactly 1 node"),
    ("nebius:2x8xh200@eu-north1", "nebius:2x8xh100@eu-north1", (), "exactly 1 node"),
])
def test_plan_refusals(monkeypatch, gpu, engine, extra, match):
    _fake_sky(monkeypatch)
    try:
        args = _rl_args(gpu=gpu, engine=engine, extra=extra)
    except (SystemExit, ValueError) as exc:  # a launch rule refused it first: still refused
        pytest.skip(f"refused before the plan: {exc}")
    with pytest.raises(ValueError, match=match):
        _plan(args)


@pytest.mark.parametrize("attr, value, match", [
    ("rl_elastic", True, "--rl-elastic"),
    ("learner_instance_type", "gpu-h200-sxm", "--learner-instance-type"),
    ("rl_island_use_gpus_per_node", 4, "--rl-island-use-gpus-per-node"),
    ("rl_standby_gpus", 8, "standby"),
])
def test_plan_refuses_options(monkeypatch, attr, value, match):
    _fake_sky(monkeypatch)
    args = _rl_args()
    setattr(args, attr, value)
    with pytest.raises(ValueError, match=match):
        _plan(args)


def test_plan_refuses_an_engine_cluster_the_engines_do_not_fill(monkeypatch):
    _fake_sky(monkeypatch)
    args = _rl_args(gpu="nebius:3x8xh200@eu-north1", engine="nebius:2x8xh100@eu-north1",
                    extra=("--rl-rollout-gpus", "16"))
    (split,) = _plan(args).values()  # 8 trainer GPUs + 16 engine GPUs on 2 engine nodes
    assert split.engine.num_nodes == 2 and split.record()["engine_island_ranks"] == [1, 2]
    args.rollout_num_gpus = 8  # 16 trainer GPUs over two nodes: the trainer leaves its node
    with pytest.raises(ValueError, match="trainer must fill"):
        _plan(args)


def test_plan_refuses_colocated(monkeypatch):
    _fake_sky(monkeypatch)
    args = _rl_args()
    args.rl_placement = "colocated"
    with pytest.raises(ValueError, match="fixed-partition"):
        _plan(args)


def test_plan_refuses_modal_island(monkeypatch):
    _fake_sky(monkeypatch)
    args = _rl_args()
    args.gpu = "modal:2x8xh200"
    with pytest.raises(ValueError, match="both clusters must be on"):
        _plan(args)


# ---- run scripts and tasks

def test_run_scripts(monkeypatch):
    _fake_sky(monkeypatch)
    base = _make(_rl_args())().run
    trainer = ec.run_for(base, engine=False)
    engine = ec.run_for(base, engine=True, rank_offset=1)
    for run in (trainer, engine):
        assert run.count(ec.RUN_ANCHOR) == 1
        assert "/proc/net/route" in run
        # the interface default precedes the multinode prelude's detection
        assert run.index("/proc/net/route") < run.index("/sys/class/net")
    assert "SKYPILOT_NODE_RANK" not in trainer.split(ec.RUN_ANCHOR)[1].split("MASTER_ADDR=")[0]
    head = engine.split(ec.RUN_ANCHOR)[1].split("MASTER_ADDR=")[0]
    assert "SKYPILOT_NODE_RANK=$((SKYPILOT_NODE_RANK + 1))" in head
    assert '"$YETO_ISLAND_HEAD_ADDR" "$SKYPILOT_NODE_IPS"' in head
    assert "YETO_ISLAND_HEAD_ADDR is unset" in head and "exit 1" in head
    # removing the added lines gives the logical run back, byte for byte
    added = [line for line in engine.splitlines(keepends=True) if line not in base.splitlines(keepends=True)]
    assert "".join(line for line in engine.splitlines(keepends=True) if line not in added) == base
    with pytest.raises(ValueError, match="anchor"):
        ec.run_for("echo no anchor\n", engine=True, rank_offset=1)


def test_engine_rank_shift_in_shell(tmp_path):
    """The engine prelude, run by bash: rank 0 of the engine cluster becomes island rank
    1, and the head address is the first node ip (MASTER_ADDR)."""
    import subprocess

    script = tmp_path / "s.sh"
    run = ec.run_for(ec.RUN_ANCHOR + 'MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)\n'
                     'echo "rank=$SKYPILOT_NODE_RANK master=$MASTER_ADDR ifname=$NCCL_SOCKET_IFNAME"\n',
                     engine=True, rank_offset=1).replace("cd ~/sky_workdir\n", "")
    script.write_text(run)
    env = {"PATH": "/usr/bin:/bin", "SKYPILOT_NODE_RANK": "0", "SKYPILOT_NODE_IPS": "10.0.0.9",
           "YETO_ISLAND_HEAD_ADDR": "10.0.0.5", "NCCL_SOCKET_IFNAME": "eth7"}
    out = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=True).stdout
    assert "rank=1 master=10.0.0.5 ifname=eth7" in out
    env.pop("YETO_ISLAND_HEAD_ADDR")
    bad = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
    assert bad.returncode == 1 and "YETO_ISLAND_HEAD_ADDR is unset" in bad.stderr


def test_build_tasks(monkeypatch):
    sky = _fake_sky(monkeypatch)
    args = _rl_args()
    (split,) = _plan(args).values()
    logical = _make(args)()
    assert logical.num_nodes == 2 and logical.resources.network_tier == "best"
    tasks = ec.build_tasks(sky, args, split, _make(args))
    t, setup, run = tasks.trainer, tasks.engine_setup, tasks.engine_run
    assert t.num_nodes == 1 and t.resources.accelerators == "H200:8"
    assert not hasattr(t.resources, "network_tier")  # one node per cluster: plain VMs
    assert t.resources.infra == "nebius/eu-north1" and t.resources.use_spot == args.spot
    assert setup.num_nodes == 1 and setup.resources.accelerators == "H100:8" and setup.run is None
    assert setup.setup == logical.setup  # same setup on the engine cluster
    assert run.resources.accelerators == "H100:8" and "SKYPILOT_NODE_RANK + 1" in run.run
    assert t.envs[ISLAND_NODE_WAIT_ENV] == str(ec.NODE_WAIT_S)
    assert t.envs["NCCL_SOCKET_NTHREADS"] == run.envs["NCCL_SOCKET_NTHREADS"] == "4"
    assert ISLAND_NODE_WAIT_ENV not in run.envs
    # the learner command (island flags) is the logical island's on both clusters
    flags = [line for line in logical.run.splitlines() if "--actor-num-nodes" in line]
    assert flags and all(line in t.run.splitlines() and line in run.run.splitlines() for line in flags)
    assert "--rl-island-gpus-per-node 8" in t.run


# ---- launch, relaunch, ops

class _Handle(SimpleNamespace):
    pass


class _Status:
    def __init__(self, name):
        self.name = name

    def is_terminal(self):
        return self.name in ("SUCCEEDED", "FAILED", "FAILED_SETUP", "CANCELLED", "FAILED_DRIVER")


class _LaunchSky:
    def __init__(self, fail_engine=False, fail_exec=False, setup=("SETTING_UP", "SUCCEEDED")):
        self.calls, self.fail_engine, self.fail_exec = [], fail_engine, fail_exec
        self.next_job = 7
        self.setup = list(setup)

    def job_status(self, cluster, job_ids):
        self.calls.append(("job_status", cluster, tuple(job_ids)))
        state = self.setup.pop(0) if len(self.setup) > 1 else self.setup[0]
        return ("status", {job_ids[0]: _Status(state)})

    def stream_and_get(self, rid):
        kind, name = rid
        self.calls.append(("get", kind, name))
        if kind == "launch" and name.endswith("-eng"):
            if self.fail_engine:
                raise RuntimeError("no capacity")
            return 2, _Handle(cluster_name_on_cloud=f"{name}-abcd", cached_internal_ips=["10.0.0.9"])
        if kind == "launch":
            self.next_job += 1
            return self.next_job, _Handle(cluster_name_on_cloud=f"{name}-abcd",
                                          cached_internal_ips=["10.0.0.5"])
        if kind == "exec":
            if self.fail_exec:
                raise RuntimeError("exec failed")
            return 3, None
        raise AssertionError(rid)

    def exec(self, task, cluster_name):
        self.calls.append(("exec", cluster_name, dict(task.envs)))
        return ("exec", cluster_name)

    def cancel(self, cluster, job_ids):
        self.calls.append(("cancel", cluster, tuple(job_ids)))
        return ("cancel", cluster)

    def get(self, rid):
        return rid[1] if rid and rid[0] == "status" else None


def _island(monkeypatch, sky_rec, links):
    sky = _fake_sky(monkeypatch)
    args = _rl_args()
    (split,) = _plan(args).values()
    tasks = ec.build_tasks(sky, args, split, _make(args))
    submits = []

    def submit(spec, task, name):
        submits.append((str(spec), name, task.run is None))
        return ("launch", name)

    island = ec.MixedIsland(split, tasks, submit=submit,
                            link=lambda region, a, b: links.append(("link", region, a, b)),
                            unlink=lambda region, a, b: links.append(("unlink", region, a, b)),
                            log=lambda line: None, sleep=lambda s: None)
    return island, submits


def test_launch_links_then_execs_the_engine_run(monkeypatch):
    sky, links = _LaunchSky(), []
    island, submits = _island(monkeypatch, sky, links)
    job, handle = island.finish(sky, island.submit())
    assert submits == [("nebius:1x8xH200@eu-north1", "run1-l0-eu-north1", False),
                       ("nebius:1x8xH100@eu-north1", "run1-l0-eu-north1-eng", True)]
    assert job == 8 and island.engine_job == 3
    assert links == [("link", "eu-north1", "run1-l0-eu-north1-abcd", "run1-l0-eu-north1-eng-abcd")]
    (exec_call,) = [c for c in sky.calls if c[0] == "exec"]
    assert exec_call[1] == "run1-l0-eu-north1-eng" and exec_call[2][ec.HEAD_ADDR_ENV] == "10.0.0.5"
    # the engine run starts only after the engine cluster's detached setup job SUCCEEDED
    polls = [i for i, c in enumerate(sky.calls) if c[0] == "job_status"]
    assert len(polls) == 2 and sky.calls[polls[0]][1:] == ("run1-l0-eu-north1-eng", (2,))
    assert polls[-1] < sky.calls.index(exec_call)
    # the link precedes the exec (Ray join needs the rule)
    island.unlink()
    assert links[-1][0] == "unlink"


@pytest.mark.parametrize("fail", ["engine", "exec"])
def test_engine_failure_cancels_the_trainer_job(monkeypatch, fail):
    sky = _LaunchSky(fail_engine=fail == "engine", fail_exec=fail == "exec")
    island, _ = _island(monkeypatch, sky, [])
    with pytest.raises(RuntimeError):
        island.finish(sky, island.submit())
    assert ("cancel", "run1-l0-eu-north1", (8,)) in sky.calls
    assert island.relaunch(sky) is None  # relaunch reports failure, the fleet retries


def test_failed_engine_setup_cancels_the_trainer_job(monkeypatch):
    sky = _LaunchSky(setup=("SETTING_UP", "FAILED_SETUP"))
    island, _ = _island(monkeypatch, sky, [])
    with pytest.raises(RuntimeError, match="setup job 2 ended FAILED_SETUP"):
        island.finish(sky, island.submit())
    assert ("cancel", "run1-l0-eu-north1", (8,)) in sky.calls
    assert not [c for c in sky.calls if c[0] == "exec"]


def test_trainer_failure_waits_for_the_engine_launch(monkeypatch):
    sky = _LaunchSky()
    island, _ = _island(monkeypatch, sky, [])
    real = sky.stream_and_get

    def fail_trainer(rid):
        if rid == ("launch", "run1-l0-eu-north1"):
            raise RuntimeError("no H200 capacity")
        return real(rid)

    sky.stream_and_get = fail_trainer
    with pytest.raises(RuntimeError, match="no H200"):
        island.finish(sky, island.submit())
    assert ("get", "launch", "run1-l0-eu-north1-eng") in sky.calls  # the engine launch settled
    assert not [c for c in sky.calls if c[0] in ("exec", "cancel")]


def test_relaunch_relaunches_both_clusters(monkeypatch):
    sky, links = _LaunchSky(), []
    island, submits = _island(monkeypatch, sky, links)
    island.finish(sky, island.submit())
    assert island.relaunch(sky) == 9
    assert ("cancel", "run1-l0-eu-north1-eng", (3,)) in sky.calls  # old engine run first
    assert [s[1] for s in submits] == ["run1-l0-eu-north1", "run1-l0-eu-north1-eng"] * 2
    assert [c for c in sky.calls if c[0] == "exec"][-1][2][ec.HEAD_ADDR_ENV] == "10.0.0.5"


def test_ops_route_relaunch_and_down(monkeypatch):
    sky, links = _LaunchSky(), []
    island, _ = _island(monkeypatch, sky, links)
    island.finish(sky, island.submit())

    class Base:
        def __init__(self):
            self.downs, self.relaunches = [], []
            self.nodes_by_cluster = {}

        def relaunch(self, task, cluster):
            self.relaunches.append(cluster)
            return 1

        def down(self, cluster):
            self.downs.append(cluster)

        def job_status(self, cluster, job_id):
            return "RUNNING"

    base = Base()
    ops = ec.EngineClusterOps(base, {island.split.name: island}, sky)
    assert ops.relaunch(None, "other") == 1 and base.relaunches == ["other"]
    assert ops.relaunch(None, island.split.name) == 9 and base.relaunches == ["other"]
    ops.down(island.split.name)
    assert base.downs == ["run1-l0-eu-north1-eng", "run1-l0-eu-north1"]
    assert links[-1][0] == "unlink"
    ops.down("other")
    assert base.downs[-1] == "other"
    assert ops.job_status("x", 1) == "RUNNING" and ops.nodes_by_cluster == {}


def test_head_address():
    assert ec.head_address(_Handle(cached_internal_ips=["10.1.2.3", "10.1.2.4"])) == "10.1.2.3"
    assert ec.head_address(_Handle(cached_internal_ips=None, internal_ips=lambda: ["10.9.9.9"])) == "10.9.9.9"
    with pytest.raises(RuntimeError, match="no internal IP"):
        ec.head_address(_Handle(cached_internal_ips=[]))


# ---- driver-side node wait

def test_node_wait():
    assert wait_for_island_nodes(2, environ={}, ray_nodes=None, clock=None, sleep=None) is None
    t = [0.0]
    seen = [[{"Alive": True, "Resources": {"GPU": 8.0}}],
            [{"Alive": True, "Resources": {"GPU": 8.0}}, {"Alive": False, "Resources": {"GPU": 8.0}}],
            [{"Alive": True, "Resources": {"GPU": 8.0}}, {"Alive": True, "Resources": {"GPU": 8.0}},
             {"Alive": True, "Resources": {"CPU": 4.0}}]]

    def nodes():
        return seen.pop(0) if len(seen) > 1 else seen[0]

    def sleep(s):
        t[0] += s

    out = wait_for_island_nodes(2, environ={ISLAND_NODE_WAIT_ENV: "100"}, ray_nodes=nodes,
                                clock=lambda: t[0], sleep=sleep, log=lambda line: None)
    assert out == {"nodes": 2, "waited_s": 20.0}
    with pytest.raises(RuntimeError, match="never joined"):
        wait_for_island_nodes(3, environ={ISLAND_NODE_WAIT_ENV: "25"}, ray_nodes=nodes,
                              clock=lambda: t[0], sleep=sleep, log=lambda line: None)


# ---- security-group link (fake Nebius utils)

class _Vpc:
    class RuleAccessAction:
        ALLOW = 1

    class RuleProtocol:
        ANY = 0

    @staticmethod
    def RuleIngress(**kw):
        return SimpleNamespace(source_cidrs=[], destination_ports=[], **kw)

    @staticmethod
    def SecurityRuleSpec(**kw):
        return SimpleNamespace(**kw)


class _Utils:
    def __init__(self):
        self.groups = {"sky-sg-a1": "sg-a", "sky-sg-b1": "sg-b"}
        self.rules = {"sg-a": [], "sg-b": []}
        self.nebius = SimpleNamespace(vpc=lambda: _Vpc)
        self.n = 0

    def get_project_by_region(self, region):
        return "proj"

    def get_security_group_by_name(self, project, name):
        return self.groups.get(name)

    def list_security_rules(self, sg):
        return list(self.rules[sg])

    @staticmethod
    def _rule_signature(spec):
        return (spec.access, spec.protocol, spec.ingress.source_security_group_id)

    def _create_security_rule(self, sg, spec, prefix):
        self.n += 1
        self.rules[sg].append(SimpleNamespace(spec=spec, metadata=SimpleNamespace(id=f"r{self.n}")))

    def _delete_security_rule(self, rule_id):
        for sg in self.rules:
            self.rules[sg] = [r for r in self.rules[sg] if r.metadata.id != rule_id]


def test_sg_link_and_unlink():
    from yeto.cloud import nebius_sg_link as link

    utils = _Utils()
    assert link.link("eu-north1", "a1", "b1", utils=utils, log=lambda line: None) == {"a1": "sg-a", "b1": "sg-b"}
    assert [r.spec.ingress.source_security_group_id for r in utils.rules["sg-a"]] == ["sg-b"]
    assert [r.spec.ingress.source_security_group_id for r in utils.rules["sg-b"]] == ["sg-a"]
    link.link("eu-north1", "a1", "b1", utils=utils, log=lambda line: None)  # idempotent
    assert len(utils.rules["sg-a"]) == len(utils.rules["sg-b"]) == 1
    assert link.unlink("eu-north1", "a1", "b1", utils=utils, log=lambda line: None) == 2
    assert utils.rules == {"sg-a": [], "sg-b": []}
    with pytest.raises(RuntimeError, match="not found"):
        link.link("eu-north1", "a1", "zz", utils=utils, log=lambda line: None)
    assert link.unlink("eu-north1", "a1", "zz", utils=utils, log=lambda line: None) == 0
