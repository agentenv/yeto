"""rl-ssh-node-pool (S25 N1), CPU half: node inventory, device-family card
preflight, jump-host SSH options and the collective-communication variables.

Every card tool output here is a recorded or stub sample; no test reaches a
real machine. The NPU samples carry their driver version in the constant name
(design D3: the format changes with the driver).
"""

from __future__ import annotations

import json

import pytest

from yeto.gpu_spec import UnknownCard, canonical_card, device_family_of, device_type_of
from yeto.hw import card_tools
from yeto.hw.card_tools import CardMismatch, CardToolError, check_cards, parse_card_tool
from yeto.launch import node_inventory, topology
from yeto.launch.node_inventory import InventoryError, Node, load_inventory, parse_inventory
from yeto.rl import ssh_harness
from yeto.rl.ssh_harness import HarnessError

# --- recorded card tool output ----------------------------------------------

NVIDIA_SMI_H200 = """0, NVIDIA H200, 143771
1, NVIDIA H200, 143771
"""

# npu-smi info of a 910B4 64 GiB node, two cards shown. The shape follows the
# sample in yeto/gpu_spec.py; the driver version of that sample is unknown, so
# a real node must confirm it (task N1.2, GPU half).
NPU_SMI_INFO_64G = """+---------------------------+---------------+----------------------------------+
| NPU   Name                | Health        | Power(W)  Temp(C)  Hugepages      |
| Chip                      | Bus-Id        | AICore(%) Memory-Usage(MB)        |
+===========================+===============+==================================+
| 0     910B4               | OK            | 92.5      47       0 / 0         |
| 0                         | 0000:C1:00.0  | 0         3161 / 65536           |
+===========================+===============+==================================+
| 1     910B4               | OK            | 91.3      46       0 / 0         |
| 0                         | 0000:C2:00.0  | 0         3160 / 65536           |
+===========================+===============+==================================+
"""

# The same node with the 32 GiB sub-model (the ModelArts machine of S21).
NPU_SMI_INFO_32G = NPU_SMI_INFO_64G.replace("65536", "32768")

NPU_SMI_MEMORY_32G = """NPU ID                         : 0
HBM Capacity(MB)               : 32768
"""

def npu_smi_info(count: int, hbm_mb: int = 65536, card: str = "910B4") -> str:
    """``npu-smi info`` output for ``count`` cards (same shape as the sample)."""
    rows = [NPU_SMI_INFO_64G.splitlines()[index] for index in range(3)]
    for index in range(count):
        rows += [f"| {index}     {card}               | OK            | 92.5      47       0 / 0         |",
                 f"| 0                         | 0000:C{index}:00.0  | 0         3161 / {hbm_mb}           |",
                 "+===========================+===============+==================================+"]
    return "\n".join(rows) + "\n"


PPU_SMI_CSV = """0, PPU 810E, 65536
1, PPU 810E, 65536
"""

PPU_SMI_KEY_VALUE = """Device                         : 0
Product Name                   : 810E
HBM Capacity(MB)               : 65536
Device                         : 1
Product Name                   : 810E
HBM Capacity(MB)               : 65536
"""


def _inventory_node(**overrides) -> dict:
    node = {"name": "ascend00", "host": "user@172.28.11.10", "device_family": "npu",
            "cpu_arch": "aarch64", "card": "910B4", "cards_per_node": 8, "hbm_gib": 64,
            "host_memory_gib": 1024, "fabric": "roce", "interface": "eth0",
            "image": "registry.example/yeto-ascend:1"}
    node.update(overrides)
    return {key: value for key, value in node.items() if value is not None}


# --- N1.1 inventory ---------------------------------------------------------


def test_inventory_reads_every_declared_field(tmp_path):
    path = tmp_path / "nodes.yaml"
    path.write_text(json.dumps({"nodes": [
        _inventory_node(proxy_jump="user@bastion", mounts={"/mnt/shared": "/mnt/shared"},
                        price_per_card_hour=1.5, roles=["trainer"]),
        _inventory_node(name="ppu00", host="user@ppu00", device_family="ppu", cpu_arch="x86_64", card="810E",
                        cards_per_node=16, hbm_gib=64, host_memory_gib=2048,
                        fabric="roce", interface="hpn0",
                        image="registry.example/yeto-ppu:1"),
    ]}))
    nodes = load_inventory(path)
    assert [node.name for node in nodes] == ["ascend00", "ppu00"]
    first = nodes[0]
    assert (first.device_family, first.card, first.cards_per_node) == ("npu", "910B4", 8)
    assert (first.hbm_gib, first.host_memory_gib) == (64.0, 1024.0)
    assert first.hbm_gib_per_node == 512.0
    assert first.ssh_options() == ["-o", "ProxyJump=user@bastion"]
    assert first.roles == ("trainer",)
    assert nodes[1].device_family == "ppu"


def test_inventory_refuses_a_node_without_a_device_family():
    with pytest.raises(InventoryError, match="ascend00: device_family is required"):
        parse_inventory({"nodes": [_inventory_node(device_family=None)]})


def test_inventory_refuses_an_unknown_card_and_never_assumes_cuda():
    with pytest.raises(InventoryError, match="910B9.*not in the yeto card catalog"):
        parse_inventory({"nodes": [_inventory_node(card="910B9")]})


def test_inventory_refuses_a_card_of_another_device_family():
    with pytest.raises(InventoryError, match="H200 is a cuda card"):
        parse_inventory({"nodes": [_inventory_node(card="H200")]})


@pytest.mark.parametrize("overrides, message", [
    ({"host_memory_gib": None}, "host_memory_gib is required"),
    ({"hbm_gib": 0}, "hbm_gib must be a positive number"),
    ({"cards_per_node": 1.5}, "cards_per_node must be a positive int"),
    ({"fabric": "myrinet"}, "fabric 'myrinet' is unknown"),
    ({"runtime": "podman"}, "runtime 'podman' is unknown"),
    ({"image": None}, "runtime docker needs an image"),
    ({"price_per_card_hour": -1}, "price_per_card_hour must be a non-negative"),
    ({"mounts": ["/mnt/shared"]}, "mounts must be a map"),
    ({"gpus": 8}, "unknown inventory field"),
])
def test_inventory_refuses_a_bad_field(overrides, message):
    with pytest.raises(InventoryError, match=message):
        parse_inventory({"nodes": [_inventory_node(**overrides)]})


# --- N1.6 cpu_arch and image selection -------------------------------------


def test_inventory_requires_cpu_arch_and_normalizes_docker_spellings():
    with pytest.raises(InventoryError, match="ascend00: cpu_arch is required"):
        parse_inventory({"nodes": [_inventory_node(cpu_arch=None)]})
    with pytest.raises(InventoryError, match="cpu_arch 'riscv64' is unknown"):
        parse_inventory({"nodes": [_inventory_node(cpu_arch="riscv64")]})
    node = parse_inventory({"nodes": [_inventory_node(cpu_arch="arm64")]})[0]
    assert node.cpu_arch == "aarch64" and node.record()["cpu_arch"] == "aarch64"


def test_inventory_selects_the_image_of_the_node_arch():
    images = {"arm64": "reg.example/yeto-npu-sampler:1", "x86_64": "reg.example/yeto-ppu-train:1"}
    nodes = parse_inventory({"images": images, "nodes": [
        _inventory_node(image=None),
        _inventory_node(name="ppu00", host="user@ppu00", device_family="ppu", cpu_arch="x86_64",
                        card="810E", image=None),
        _inventory_node(name="ascend01", host="user@h1", image="site/own-ascend:2"),
    ]})
    assert [n.image for n in nodes] == ["reg.example/yeto-npu-sampler:1",
                                        "reg.example/yeto-ppu-train:1", "site/own-ascend:2"]


@pytest.mark.parametrize("overrides, message", [
    ({"image": "reg.example/yeto-ppu-train:1@sha256:ab"}, "x86_64 image ppu-train; the node is aarch64"),
    ({"cpu_arch": "x86_64", "image": "reg.example/yeto-npu-sampler:1"},
     "aarch64 image npu-sampler; the node is x86_64"),
    ({"image": None}, "set image, or images.aarch64"),
])
def test_inventory_refuses_an_image_of_another_arch(overrides, message):
    document = {"images": {"x86_64": "reg.example/yeto-ppu-train:1"},
                "nodes": [_inventory_node(**overrides)]}
    with pytest.raises(InventoryError, match=message):
        parse_inventory(document)


def test_inventory_refuses_a_catalog_image_of_another_device_family():
    with pytest.raises(InventoryError, match="ppu image ppu-train; the node is device_family cuda"):
        parse_inventory({"nodes": [_inventory_node(device_family="cuda", cpu_arch="x86_64",
                                                   card="H200", image="r/yeto-ppu-train:1")]})


@pytest.mark.parametrize("document, message", [
    ({"images": {"sparc": "r/x:1"}, "nodes": []}, "images: cpu_arch 'sparc' is unknown"),
    ({"images": {"arm64": "r/a:1", "aarch64": "r/b:1"}, "nodes": []}, "aarch64 is listed twice"),
    ({"images": ["r/x:1"], "nodes": []}, "images must be a map"),
    ({"image": "r/x:1", "nodes": []}, "unknown top-level inventory field"),
])
def test_inventory_refuses_a_bad_images_map(document, message):
    with pytest.raises(InventoryError, match=message):
        parse_inventory(document)


def test_node_checks_uname_against_cpu_arch():
    node = parse_inventory({"nodes": [_inventory_node()]})[0]
    node.check_cpu_arch("aarch64\n")
    node.check_cpu_arch("arm64")
    with pytest.raises(InventoryError, match="uname -m reports x86_64, the inventory declares cpu_arch aarch64"):
        node.check_cpu_arch("x86_64\n")


def test_inventory_accepts_a_rootfs_node_without_an_image():
    node = parse_inventory({"nodes": [_inventory_node(runtime="rootfs", image=None)]})[0]
    assert (node.runtime, node.image) == ("rootfs", None)


def test_inventory_refuses_a_repeated_node_or_ssh_target():
    with pytest.raises(InventoryError, match="name 'ascend00' twice"):
        parse_inventory({"nodes": [_inventory_node(), _inventory_node()]})
    with pytest.raises(InventoryError, match="SSH target .* twice"):
        parse_inventory({"nodes": [_inventory_node(), _inventory_node(name="ascend01")]})


def test_inventory_refuses_an_empty_file(tmp_path):
    path = tmp_path / "nodes.yaml"
    path.write_text("")
    with pytest.raises(InventoryError, match="is empty"):
        load_inventory(path)
    path.write_text("nodes: []")
    with pytest.raises(InventoryError, match="at least one node"):
        load_inventory(path)


def test_node_record_hides_the_ssh_user():
    node = parse_inventory({"nodes": [_inventory_node(proxy_jump="user@bastion")]})[0]
    record = node.record()
    assert record["behind_jump_host"] is True
    assert "user@" not in json.dumps(record)


def test_island_nodes_selects_by_role():
    nodes = parse_inventory({"nodes": [
        _inventory_node(roles=["trainer"]),
        _inventory_node(name="ascend01", host="user@h1", roles=["sampler"]),
        _inventory_node(name="ascend02", host="user@h2"),
    ]})
    assert [n.name for n in node_inventory.island_nodes(nodes, "trainer")] == \
        ["ascend00", "ascend02"]


# --- N1.1/N1.2 card catalog -------------------------------------------------


def test_an_unknown_card_name_is_refused_instead_of_defaulting_to_cuda():
    with pytest.raises(UnknownCard, match="not in the yeto card catalog"):
        device_type_of("910C9")
    with pytest.raises(UnknownCard):
        device_family_of("MI300X")


def test_device_family_and_torch_device_type_differ_on_a_ppu():
    assert (device_family_of("810E"), device_type_of("810E")) == ("ppu", "cuda")
    assert (device_family_of("910B4"), device_type_of("910B4")) == ("npu", "npu")
    assert (device_family_of("H200"), device_type_of("H200")) == ("cuda", "cuda")
    assert canonical_card("910b3") == "910B3"


# --- N1.2 card tool parsers -------------------------------------------------


def test_nvidia_smi_csv_gives_index_card_and_hbm():
    records = parse_card_tool("cuda", NVIDIA_SMI_H200)
    assert [(r.index, r.card) for r in records] == [(0, "H200"), (1, "H200")]
    assert round(records[0].hbm_gib, 1) == 140.4


def test_npu_smi_info_gives_card_hbm_and_health():
    records = parse_card_tool("npu", NPU_SMI_INFO_64G)
    assert [(r.index, r.card, r.hbm_gib, r.health) for r in records] == [
        (0, "910B4", 64.0, "OK"), (1, "910B4", 64.0, "OK")]


def test_npu_smi_memory_capacity_wins_over_the_usage_row():
    records = parse_card_tool("npu", NPU_SMI_INFO_64G + NPU_SMI_MEMORY_32G)
    assert records[0].hbm_gib == 32.0  # -t memory is the capacity, not a usage total
    assert records[1].hbm_gib == 64.0


def test_npu_smi_without_a_card_row_is_an_error():
    with pytest.raises(CardToolError, match="no card row"):
        parse_card_tool("npu", "npu-smi: command not found")


@pytest.mark.parametrize("text", [PPU_SMI_CSV, PPU_SMI_KEY_VALUE])
def test_ppu_smi_is_parsed_in_both_recorded_forms(text):
    records = parse_card_tool("ppu", text)
    assert [(r.index, r.card, r.hbm_gib) for r in records] == [
        (0, "810E", 64.0), (1, "810E", 64.0)]


def test_preflight_commands_per_device_family():
    assert card_tools.preflight_commands("npu")[0] == ("npu-smi", "info")
    assert card_tools.preflight_commands("ppu")[0][0] == "ppu-smi"
    assert card_tools.preflight_commands("cuda")[0][0] == "nvidia-smi"
    with pytest.raises(ValueError, match="unknown device family"):
        card_tools.preflight_commands("tpu")


# --- N1.2 preflight refusals ------------------------------------------------


def test_preflight_passes_when_the_node_matches_the_inventory():
    node = parse_inventory({"nodes": [_inventory_node()]})[0]
    records = node.check_cards(npu_smi_info(8))
    assert len(records) == 8


def test_preflight_refuses_an_hbm_mismatch_and_shows_both_values():
    node = parse_inventory({"nodes": [_inventory_node(cards_per_node=2)]})[0]
    with pytest.raises(CardMismatch) as exc:
        node.check_cards(NPU_SMI_INFO_32G)
    message = str(exc.value)
    assert "64 GiB" in message and "32.0 GiB" in message and "ascend00" in message


def test_preflight_refuses_a_card_count_mismatch():
    node = parse_inventory({"nodes": [_inventory_node()]})[0]
    with pytest.raises(CardMismatch, match="reports 2 x 910B4"):
        node.check_cards(NPU_SMI_INFO_64G)


def test_preflight_refuses_an_unknown_card_model():
    node = parse_inventory({"nodes": [_inventory_node(cards_per_node=2)]})[0]
    with pytest.raises(CardMismatch, match="not in the yeto card catalog"):
        node.check_cards(NPU_SMI_INFO_64G.replace("910B4", "910Z9"))


def test_preflight_refuses_another_known_card_of_the_same_family():
    node = parse_inventory({"nodes": [_inventory_node(cards_per_node=2)]})[0]
    with pytest.raises(CardMismatch, match=r"card\(s\) \['910B3'\].*declares 910B4"):
        node.check_cards(NPU_SMI_INFO_64G.replace("910B4", "910B3"))


def test_preflight_accepts_the_driver_reservation_on_an_h200():
    records = check_cards("cuda", NVIDIA_SMI_H200, card="H200", count=2, hbm_gib=141)
    assert len(records) == 2
    with pytest.raises(CardMismatch, match="141 GiB"):
        check_cards("cuda", NVIDIA_SMI_H200.replace("143771", "81559"),
                    card="H200", count=2, hbm_gib=141)


def test_preflight_refuses_when_the_tool_reports_no_hbm():
    with pytest.raises(CardMismatch, match="no HBM capacity"):
        check_cards("cuda", "0, NVIDIA H200, [N/A]", card="H200", count=1, hbm_gib=141)


# --- N1.4 collective communication variables --------------------------------


def test_an_ascend_island_gets_hccl_variables_and_no_nccl_variables():
    env = topology.island_comm_env("npu", "roce", "eth0", num_nodes=4)
    assert env["HCCL_SOCKET_IFNAME"] == "eth0"
    assert env["GLOO_SOCKET_IFNAME"] == "eth0"
    assert env["HCCL_CONNECT_TIMEOUT"] == str(topology.HCCL_CONNECT_TIMEOUT_S)
    assert not [name for name in env if name.startswith("NCCL_")]


def test_a_ppu_island_keeps_the_nccl_names_of_accl_p():
    env = topology.island_comm_env("ppu", "roce", "hpn0", num_nodes=2)
    assert env["NCCL_SOCKET_IFNAME"] == "hpn0"
    assert env["NCCL_IB_DISABLE"] == "0"  # the inventory declares RoCE
    assert not [name for name in env if name.startswith("HCCL_")]


def test_a_declared_fabric_is_never_disabled():
    for fabric in ("roce", "ib"):
        assert topology.island_comm_env("cuda", fabric, num_nodes=2)["NCCL_IB_DISABLE"] == "0"
    assert topology.island_comm_env("cuda", "tcp", num_nodes=2)["NCCL_IB_DISABLE"] == "1"


def test_a_single_node_island_sets_nothing():
    assert topology.island_comm_env("npu", "roce", "eth0", num_nodes=1) == {}
    assert topology.island_comm_env_prelude("npu", "roce", "eth0", num_nodes=1) == ""


def test_the_prelude_lets_an_existing_value_win():
    prelude = topology.island_comm_env_prelude("npu", "hccs", "eth0", num_nodes=2)
    assert prelude.splitlines() == [
        "export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}",
        "export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}",
        "export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}",
    ]


def test_an_unknown_device_family_has_no_network_environment():
    with pytest.raises(ValueError, match="unknown device family"):
        topology.island_comm_env("tpu", "tcp", num_nodes=2)


def test_the_cloud_prelude_is_unchanged():
    assert topology.multinode_env_prelude("modal", 2).splitlines() == [
        "export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}",
        "export NCCL_DEBUG=${NCCL_DEBUG:-WARN}",
    ]
    assert topology.multinode_env_prelude("aws", 1) == ""


# --- N1.3 (CPU half): the harness is no longer pinned to H200 ---------------


def _npu_plan() -> dict:
    from test_rl_ssh_harness import _plan

    plan = _plan()
    for island in plan["islands"]:
        island["accelerator"] = "910B4"
    plan["hbm_gib_per_card"] = 64.0
    return plan


def test_the_harness_accepts_an_ascend_plan():
    plan = _npu_plan()
    ssh_harness._validate_plan(plan)
    assert ssh_harness._plan_accelerator(plan) == "910B4"
    assert ssh_harness._plan_device_family(plan) == "npu"


def test_the_harness_refuses_an_unknown_accelerator():
    plan = _npu_plan()
    plan["islands"][0]["accelerator"] = "910B9"
    with pytest.raises(HarnessError, match="not in the yeto card catalog"):
        ssh_harness._validate_plan(plan)


def test_the_harness_refuses_islands_of_different_cards():
    plan = _npu_plan()
    plan["islands"][0]["accelerator"] = "910B3"
    with pytest.raises(HarnessError, match="same accelerator"):
        ssh_harness._validate_plan(plan)


def test_parse_islands_records_the_requested_card():
    islands = ssh_harness._parse_islands(["root@n0,root@n1"], 8, "910B4")
    assert islands[0]["accelerator"] == "910B4"
    assert ssh_harness._parse_islands(["root@n0"], 8)[0]["accelerator"] == "H200"
    with pytest.raises(HarnessError, match="not in the yeto card catalog"):
        ssh_harness._parse_islands(["root@n0"], 8, "mi300x")


def test_the_host_setup_script_runs_the_card_tool_of_the_family():
    cuda_plan = _npu_plan()
    for island in cuda_plan["islands"]:
        island["accelerator"] = "H200"
    cuda_lines = ssh_harness._card_tool_check_script(cuda_plan, 8)
    assert "nvidia-smi" in cuda_lines and "npu-smi" not in cuda_lines
    npu_lines = ssh_harness._card_tool_check_script(_npu_plan(), 8)
    assert "npu-smi" in npu_lines and "nvidia-smi" not in npu_lines
    ppu_plan = _npu_plan()
    for island in ppu_plan["islands"]:
        island["accelerator"] = "810E"
    assert "ppu-smi" in ssh_harness._card_tool_check_script(ppu_plan, 8)


def test_the_jit_cache_identity_of_an_h200_plan_is_unchanged():
    from test_rl_ssh_harness import _plan

    h200 = _plan()
    with_card = _plan()
    for island in with_card["islands"]:
        island.pop("accelerator")
    assert ssh_harness._jit_cache_compatibility_sha256(h200) == \
        ssh_harness._jit_cache_compatibility_sha256(with_card)


def test_the_jit_cache_is_refused_on_a_non_cuda_island():
    plan = _npu_plan()
    with pytest.raises(HarnessError, match="cuda islands only"):
        ssh_harness._require_cuda_jit_cache(plan)


# --- N1.3 (CPU half): fake SSH, jump host, preflight refusal ----------------


class _FakeSsh:
    """Records every ssh command and replays canned card tool output."""

    def __init__(self, outputs: dict[str, str], returncode: int = 0):
        self.outputs = outputs
        self.returncode = returncode
        self.commands: list[list[str]] = []

    def __call__(self, command, *, capture=False, check=True):
        import subprocess

        self.commands.append(list(command))
        target = next((item for item in command if "@" in item or item.startswith("n")), "")
        return subprocess.CompletedProcess(
            list(command), self.returncode,
            stdout=self.outputs.get(target, ""), stderr="npu-smi: not found")


def _patch_ssh(monkeypatch, fake):
    monkeypatch.setattr(ssh_harness, "_run", fake)


def test_the_card_preflight_passes_with_fake_smi_output(monkeypatch):
    plan = _npu_plan()
    hosts = [host for island in plan["islands"] for host in island["hosts"]]
    fake = _FakeSsh({host: npu_smi_info(4) for host in hosts})
    _patch_ssh(monkeypatch, fake)
    ssh_harness._check_island_cards(plan)
    assert len(fake.commands) == len(hosts)
    assert all("npu-smi info" in command[-1] for command in fake.commands)


def test_the_card_preflight_refuses_the_launch_on_an_hbm_mismatch(monkeypatch):
    plan = _npu_plan()
    hosts = [host for island in plan["islands"] for host in island["hosts"]]
    _patch_ssh(monkeypatch, _FakeSsh({host: npu_smi_info(4, hbm_mb=32768) for host in hosts}))
    with pytest.raises(HarnessError, match="64 GiB"):
        ssh_harness._check_island_cards(plan)


def test_the_card_preflight_refuses_the_launch_when_the_tool_is_missing(monkeypatch):
    plan = _npu_plan()
    _patch_ssh(monkeypatch, _FakeSsh({}, returncode=127))
    with pytest.raises(HarnessError, match="card tool failed"):
        ssh_harness._check_island_cards(plan)


def test_every_ssh_and_copy_to_a_jump_host_node_goes_through_the_bastion():
    node = parse_inventory({"nodes": [_inventory_node(proxy_jump="user@bastion")]})[0]
    options = node.ssh_options()
    assert options == ["-o", "ProxyJump=user@bastion"]
    plan = {"ssh_options": options}
    assert ssh_harness._ssh_command(plan, node.host, "true")[:4] == [
        "ssh", "-o", "ProxyJump=user@bastion", node.host]
    assert ssh_harness._rsync_shell(plan) == "ssh -o ProxyJump=user@bastion"
    assert parse_inventory({"nodes": [_inventory_node()]})[0].ssh_options() == []


def test_a_node_without_a_jump_host_uses_plain_ssh():
    node = Node(name="n0", host="user@n0", device_family="cuda", cpu_arch="x86_64", card="H200",
                cards_per_node=8, hbm_gib=141, host_memory_gib=2048)
    assert node.ssh_options() == []
    assert node.preflight_commands()[0][0] == "nvidia-smi"


# --- N1.1: the launcher validates the inventory before any cloud work --------


def test_the_launch_flag_validates_the_inventory_file(tmp_path):
    from types import SimpleNamespace

    from yeto.launch.rl_args import _prepare_node_inventory

    path = tmp_path / "nodes.yaml"
    path.write_text(json.dumps({"nodes": [_inventory_node()]}))
    args = SimpleNamespace(rl_node_inventory=str(path))
    _prepare_node_inventory(args)
    assert [node.name for node in args.rl_node_inventory_nodes] == ["ascend00"]

    path.write_text(json.dumps({"nodes": [_inventory_node(device_family=None)]}))
    with pytest.raises(InventoryError, match="device_family is required"):
        _prepare_node_inventory(SimpleNamespace(rl_node_inventory=str(path)))

    unset = SimpleNamespace(rl_node_inventory=None)
    _prepare_node_inventory(unset)
    assert unset.rl_node_inventory_nodes is None
