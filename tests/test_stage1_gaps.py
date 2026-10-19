"""S25 stage 1, gaps 3-6 (T-GAPS): the engine sampler process entry, the
glm5_next Miles recipe, SSH trainer islands on docker nodes from the inventory,
and the head's OSS access through a relay node. CPU only, no SSH."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.cloud.oss_relay import RELAY_ENV, Relay, RelayOssObjectStore, relay_from_env
from yeto.gpu_spec import ClusterSpec
from yeto.launch.node_inventory import InventoryError, Node, parse_inventory
from yeto.launch.ssh_trainer_island import (
    SshIslandError,
    SshTrainerSpec,
    container_name,
    inventory_trainer_nodes,
    island_spec_from_inventory,
    kill_script,
    remote_paths,
    start_script,
)
from yeto.launch.trainer_island import head_store_objects
from yeto.rl.adapters.miles.models import glm5_next
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangServerEngine
from yeto.rl.engine import engine_sampler, engine_sampler_entry as entry
from yeto.rl.engine.object_store import ObjectStoreError
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.run_config import full_parameter_layout_problem
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_checks import GSM8K_FIXED_PROMPTS
from yeto.rl import stage1

PLAN = Path(__file__).resolve().parents[1] / "examples" / "stage1" / "glm53-flash-ppu.yaml"


# ------------------------------------------------------------ gap 3: sampler entry
def _sampler_spec(tmp_path: Path, **over) -> dict:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"q": f"question {i}", "a": str(i)}) + "\n" for i in range(3)))
    spec = {
        "sampler_id": "ppu4",
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "fake", "kwargs": {
            "answers": {p.prompt: f"The answer is {p.answer}." for p in GSM8K_FIXED_PROMPTS},
            "max_model_len": 64}},
        "sampling": {"max_tokens": 6, "n": 2},
        "num_rollouts": 2, "max_backlog_groups": 100, "max_prompt_tokens": 32,
        "store": str(tmp_path / "store"),
        "prompts": {"path": str(prompts), "input_key": "q", "label_key": "a", "groups_per_round": 2},
        "reward": "tests.test_stage1_gaps:half_reward",
        "event_tape": str(tmp_path / "tape" / "events.jsonl"),
    }
    return {**spec, **over}


def half_reward(item, completion) -> float:
    return 0.5


def test_engine_sampler_entry_runs_the_island_from_a_spec(tmp_path):
    assert entry.run(_sampler_spec(tmp_path), resource_sampler=lambda emit: None) == 2
    reader = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1)
    reader.refresh()
    assert len(reader.select("t", 10)) == 4  # 2 rounds x 2 prompts
    events = [json.loads(line) for line in (tmp_path / "tape" / "events.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    assert names[0] == "rl_sampler_start" and names.count("rl_sampler_engine_round") == 2
    assert all(e["sampler_id"] == "ppu4" for e in events)


def test_prompts_wrap_and_carry_labels(tmp_path):
    prompts = entry.build_prompts(_sampler_spec(tmp_path))
    assert [p.prompt for p in prompts(1)] == ["question 2", "question 0"]
    assert prompts(0)[1].label == "1"


def test_main_rejects_a_bad_spec(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(_sampler_spec(
        tmp_path, engine={"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "nope"})))
    assert engine_sampler.main(["--spec", str(spec)]) == 2
    assert "unknown sampler engine" in capsys.readouterr().err


def test_miles_reward_wraps_an_async_score():
    async def score(args, sample):
        return 1.0 if sample.response.endswith(sample.label) else 0.0

    reward = entry.miles_reward(score)
    item = engine_sampler.PromptItem("p", "q", label="7")
    assert reward(item, SimpleNamespace(text="it is 7")) == 1.0


def test_stage1_sampler_spec_builds_the_same_sglang_command():
    out = stage1.expand(stage1.load_plan(PLAN))
    run = out.sampler["engines"][0]
    engine = entry.build_engine(run["spec"])
    assert isinstance(engine, SGLangServerEngine)
    assert engine.argv(engine.config.base_dir) == run["engine_argv"]
    assert entry.island_config(run["spec"]).sampling.max_tokens == 4096
    assert run["spec"]["release_dir"] == "policy-releases-mxfp4-g32-rtn"


# ------------------------------------------------------------ gap 4: glm5_next recipe
def test_glm5_next_model_args_follow_transformer_impl():
    te = glm5_next.model_argv()
    local = glm5_next.model_argv(transformer_impl="local")
    assert "--moe-grouped-gemm" in te and "--transformer-impl" not in te
    assert "--moe-grouped-gemm" not in local and "--no-gradient-accumulation-fusion" in local
    assert te[te.index("--moe-layer-freq") + 1].count("1") == 42
    with pytest.raises(ValueError, match="transformer_impl"):
        glm5_next.model_argv(transformer_impl="te")


def test_glm5_next_recipe_replaces_the_bridge_geometry():
    generic = ["train.py", "--num-layers", "4", "--num-query-groups", "8", "--hidden-size", "64",
               "--no-gradient-accumulation-fusion", "--bf16"]
    te = glm5_next.apply_recipe(generic, glm5_next.model_argv())
    assert te[0] == "train.py" and "--num-query-groups" not in te
    assert te.count("--num-layers") == 1 and te[te.index("--num-layers") + 1] == "45"
    assert "--no-gradient-accumulation-fusion" in te and "--bf16" in te
    local = glm5_next.apply_recipe(generic, glm5_next.model_argv(transformer_impl="local"))
    assert local.count("--no-gradient-accumulation-fusion") == 1


def test_island_parser_takes_the_recipe_and_transformer_impl():
    from yeto.rl.adapters.miles.island_entry import build_parser

    argv = list(stage1.expand(stage1.load_plan(PLAN)).trainer["miles"]["island_argv"][3:])
    argv[argv.index("--rl-transformer-impl") + 1] = "local"
    args, unknown = build_parser().parse_known_args(argv)
    assert unknown == []
    assert (args.rl_model_recipe, args.rl_transformer_impl) == ("glm5_next", "local")


def test_full_parameter_layout_rule_is_shared():
    assert full_parameter_layout_problem(data_parallel=2, actor_num_nodes=3, rollout_gpus=0,
                                         rollout_gpus_per_engine=1) == "dense full-parameter GRPO requires DP=1"
    assert full_parameter_layout_problem(data_parallel=1, actor_num_nodes=1, rollout_gpus=8,
                                         rollout_gpus_per_engine=8) is None


# ------------------------------------------------------------ gap 5: SSH trainer island
def _node(name="ppu1", **kw) -> Node:
    base = dict(host=name, device_family="ppu", card="M890", cards_per_node=8, hbm_gib=144.0,
                host_memory_gib=2266.0, fabric="roce", interface="eth0", runtime="docker",
                image="img:1", mounts={"/data": "/data"}, docker_args=("--device", "/dev/x"))
    return Node(name=name, **(base | kw))


def test_inventory_reads_address_and_docker_args():
    raw = {"name": "n", "host": "ppu1", "device_family": "ppu", "card": "M890", "cards_per_node": 8,
           "hbm_gib": 144, "host_memory_gib": 2266, "image": "i", "address": "10.0.0.74",
           "docker_args": ["--device", "/dev/a"]}
    node = parse_inventory({"nodes": [raw]})[0]
    assert (node.address, node.docker_args) == ("10.0.0.74", ("--device", "/dev/a"))
    with pytest.raises(InventoryError, match="IP address"):
        parse_inventory({"nodes": [{**raw, "address": "ppu1"}]})


def test_docker_island_needs_device_args_and_alias_nodes_need_an_address():
    with pytest.raises(SshIslandError, match="docker_args"):
        SshTrainerSpec(nodes=(_node(docker_args=()),), run_id="r", workdir="/w", command=("x",))
    with pytest.raises(SshIslandError, match="rendezvous address"):
        SshTrainerSpec(nodes=(_node(), _node("ppu2")), run_id="r", workdir="/w", command=("x",))
    with pytest.raises(SshIslandError, match="one trainer command or one run script"):
        SshTrainerSpec(nodes=(_node(),), run_id="r", workdir="/w")


def test_docker_run_script_uses_inventory_addresses_and_the_container_is_removed():
    nodes = (_node(address="10.0.0.74"), _node("ppu2", address="10.0.0.75"))
    spec = SshTrainerSpec(nodes=nodes, run_id="r", workdir="/w", run_script='echo "$SKYPILOT_NODE_RANK"',
                          env={"SECRET": "s3cr3t"}, image="trainer:2")
    paths = remote_paths("r", "trainer", 1)
    name = container_name("r", "trainer", 1, 1)
    script = start_script(spec, 1, {}, paths, container=name)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "export SKYPILOT_NODE_IPS='10.0.0.74\n10.0.0.75'" in script and "SKYPILOT_NODE_RANK=1" in script
    assert f"--name {name}" in script and "--device /dev/x" in script and "trainer:2" in script
    assert "-e SECRET" in script and "SECRET=s3cr3t -" not in script.split("setsid", 1)[1]
    assert "ppu1" not in script.split("setsid", 1)[1]  # no SSH alias as rendezvous
    assert f"docker rm -f {name}" in kill_script(paths, container=name)
    with pytest.raises(SshIslandError, match="container name"):
        start_script(spec, 0, {}, paths)


def test_run_script_mode_runs_locally_with_the_sky_variables(tmp_path):
    spec = SshTrainerSpec(nodes=(_node(runtime="rootfs", docker_args=(), host="127.0.0.1"),), run_id="r",
                          workdir=str(tmp_path), run_script=f'echo "$SKYPILOT_NUM_NODES $SKYPILOT_NODE_IPS" > {tmp_path}/o\nexit 4')
    paths = {k: v.replace("$HOME", str(tmp_path)) for k, v in remote_paths("r", "t", 0).items()}
    script = start_script(spec, 0, {}, paths)
    subprocess.run(["bash", "-c", script], check=False, timeout=30)
    assert (tmp_path / "o").read_text().strip() == "1 127.0.0.1"
    assert Path(paths["rc"]).read_text().strip() == "4"


def test_launcher_builds_the_ssh_island_from_the_inventory():
    nodes = [_node("ppu1", address="10.0.0.74", roles=("trainer",)), _node("ppu2", address="10.0.0.75",
             roles=("trainer",)), _node("ppu4", roles=("sampler",))]
    args = SimpleNamespace(rl_node_inventory_nodes=nodes, rl_ssh_trainer_workdir="/data/yeto/code",
                           cluster_prefix="run7")
    island = ClusterSpec(cloud="ssh", region=None, num_nodes=2, gpus_per_node=8, gpu="M890")
    task = SimpleNamespace(setup="pip check", run="echo run", envs={"A": "1"}, secrets={"B": "2"})
    spec = island_spec_from_inventory(args, island, task)
    assert [n.name for n in spec.nodes] == ["ppu1", "ppu2"] and spec.addresses() == ["10.0.0.74", "10.0.0.75"]
    assert spec.env == {"A": "1", "B": "2"} and spec.run_script.endswith("echo run")
    with pytest.raises(SshIslandError, match="3 node"):
        inventory_trainer_nodes(args, ClusterSpec(cloud="ssh", region=None, num_nodes=3, gpus_per_node=8, gpu="M890"))
    with pytest.raises(SshIslandError, match="workdir"):
        island_spec_from_inventory(SimpleNamespace(**{**vars(args), "rl_ssh_trainer_workdir": None}), island, task)


# ------------------------------------------------------------ gap 6: OSS through a relay
class FakeRelay:
    """``ssh RELAY bash -c SCRIPT`` against a dict, by the ossutil verb."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.calls: list[str] = []

    def __call__(self, argv, data):
        script = argv[-1]
        self.calls.append(script)
        url = next(t for t in script.replace("'", " ").split() if t.startswith("oss://"))
        if " cat " in script:
            return (0, self.objects[url], b"") if url in self.objects else (1, b"", b"Error: NoSuchKey")
        if " cp -f " in script:
            self.objects[url] = data
            return 0, b"", b""
        if " stat " in script:
            return (0, b"", b"") if url in self.objects else (1, b"", b"StatusCode=404")
        if " ls -s " in script:
            listing = "".join(f"{k}\n" for k in sorted(self.objects) if k.startswith(url))
            return 0, (listing + "\nObject Number is: 2\n").encode(), b""
        if " rm " in script:
            self.objects.pop(url, None)
            return 0, b"", b""
        return 1, b"", b"permission denied"


def test_relay_store_reads_and_writes_through_ossutil():
    fake = FakeRelay()
    store = RelayOssObjectStore("oss://bkt/yeto/run", Relay("ppu1"), runner=fake)
    assert store.read("cuts/a.json") is None and not store.exists("cuts/a.json")
    store.write("cuts/a.json", b"{}")
    assert store.create("cuts/b.json", b"1") and not store.create("cuts/b.json", b"2")
    assert store.read("cuts/a.json") == b"{}" and store.read("cuts/b.json") == b"1"
    assert store.list("cuts/") == ["cuts/a.json", "cuts/b.json"]
    assert store.list("cuts/", start_after="cuts/a.json") == ["cuts/b.json"]
    store.delete("cuts/a.json")
    assert store.list("cuts/") == ["cuts/b.json"]
    assert all("ossutil" in c for c in fake.calls)


def test_relay_store_reports_other_failures():
    store = RelayOssObjectStore("oss://bkt", Relay("ppu1"), runner=lambda argv, data: (1, b"", b"AccessDenied"))
    with pytest.raises(ObjectStoreError, match="AccessDenied"):
        store.read("x")


def test_head_store_switches_between_relay_and_direct():
    assert relay_from_env({}) is None
    store = head_store_objects("oss://bkt/p", environ={RELAY_ENV: "ppu1"})
    assert isinstance(store, RelayOssObjectStore) and store.relay.host == "ppu1"


def test_stage1_plan_has_no_code_gaps_left_and_runs_the_relay_check():
    out = stage1.expand(stage1.load_plan(PLAN))
    assert out.store["head_access"]["env"] == {RELAY_ENV: "ppu1"}
    assert not any("OSS" in g or "engine_sampler" in g or "recipe" in g or "yeto launch" in g for g in out.gaps)
    relay = next(s for s in out.steps if s.name == "head-store-relay")
    assert relay.node is None and "ossutil ls -s" in relay.script
