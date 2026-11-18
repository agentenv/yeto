"""Bare-metal cluster preflight (``yeto preflight``): the tool-output parsers, the
per-node and cross-node checks over a fake SSH, the report, and the CLI."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from yeto.launch.preflight import cluster as cp
from yeto.launch.node_inventory import Node
from yeto.launch.ssh_trainer_island import CUDA_RDMA_DOCKER_ARGS, SshOps

IMAGE = "ghcr.io/owner/yeto-miles-k31@sha256:" + "ab" * 32
BAD_IMAGE = "ghcr.io/owner/yeto-miles-k31@sha256:" + cp.OVERLAY_UNSAFE_DIGESTS[1]
KB_PER_GIB = 1 << 20


# ----------------------------------------------------------------- tool outputs
def topo_text(gpus=8, nics=10, sys_pair=None, far_nics=False) -> str:
    cols = [f"GPU{i}" for i in range(gpus)] + [f"NIC{j}" for j in range(nics)]
    lines = ["\t" + "\t".join(cols) + "\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID"]

    def cell(a, b):
        if a == b:
            return " X "
        if a.startswith("GPU") and b.startswith("GPU"):
            pair = {a, b}
            return "SYS" if sys_pair and pair == {f"GPU{sys_pair[0]}", f"GPU{sys_pair[1]}"} else "NV18"
        ia, ib = int(a[3:]), int(b[3:])
        if a.startswith("GPU") != b.startswith("GPU"):
            g, n = (ia, ib) if a.startswith("GPU") else (ib, ia)
            if far_nics:
                return "SYS"
            return "PIX" if g == n else ("NODE" if (g < 4) == (n < 4) else "SYS")
        return "NODE"

    for r in cols:
        lines.append(r + "\t" + "\t".join(cell(r, c) for c in cols) + "\t0-55,112-167\t0\t\tN/A")
    lines += ["", "Legend:", "", "  X    = Self", "  NV#  = Connection traversing a bonded set of # NVLinks",
              "", "NIC Legend:", ""]
    lines += [f"  NIC{j}: mlx5_{j}" for j in range(nics)]
    return "\n".join(lines)


def ibstat_text(n=8, layer="InfiniBand", rate=400, down=(), storage=True, fw="28.43.1014") -> str:
    out = []
    for i in range(n):
        out += [f"CA 'mlx5_{i}'", "\tCA type: MT4129", "\tNumber of ports: 1", f"\tFirmware version: {fw}",
                "\tHardware version: 0", "\tNode GUID: 0x0001", "\tPort 1:",
                f"\t\tState: {'Down' if i in down else 'Active'}",
                f"\t\tPhysical state: {'Polling' if i in down else 'LinkUp'}", f"\t\tRate: {rate}",
                "\t\tBase lid: 12", "\t\tLMC: 0", "\t\tSM lid: 1", "\t\tCapability mask: 0xa751e848",
                "\t\tPort GUID: 0x0001", f"\t\tLink layer: {layer}"]
    if storage:
        out += ["CA 'mlx5_8'", "\tCA type: MT4125", "\tFirmware version: 22.41.1000", "\tPort 1:",
                "\t\tState: Active", "\t\tPhysical state: LinkUp", "\t\tRate: 200", "\t\tLink layer: Ethernet"]
    return "\n".join(out)


def devinfo_text(n=8, mtu=4096, layer="InfiniBand") -> str:
    out = []
    for i in range(n):
        out += [f"hca_id:\tmlx5_{i}", "\ttransport:\t\t\tInfiniBand (0)", "\tfw_ver:\t\t\t\t28.43.1014",
                "\tphys_port_cnt:\t\t\t1", "\t\tport:\t1", "\t\t\tstate:\t\t\tPORT_ACTIVE (4)",
                "\t\t\tmax_mtu:\t\t4096 (5)", f"\t\t\tactive_mtu:\t\t{mtu} (5)", "\t\t\tsm_lid:\t\t\t1",
                "\t\t\tactive_width:\t\t4X (2)", "\t\t\tactive_speed:\t\t100.0 Gbps (128)",
                f"\t\t\tlink_layer:\t\t{layer}"]
    return "\n".join(out)


def gid_text(n=8, v2_index=3, ip_base=(198, 51, 100)) -> str:
    out = []
    for i in range(n):
        out.append(f"mlx5_{i} 1 0 fe80:0000:0000:0000:0a00:00ff:fe00:000{i} eth{i} IB/RoCE v1")
        out.append(f"mlx5_{i} 1 1 fe80:0000:0000:0000:0a00:00ff:fe00:000{i} eth{i} RoCE v2")
        a, b, c = ip_base
        out.append(f"mlx5_{i} 1 2 0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{i + 1:02x} eth{i} IB/RoCE v1")
        out.append(f"mlx5_{i} 1 {v2_index} 0000:0000:0000:0000:0000:ffff:{a:02x}{b:02x}:{c:02x}{i + 1:02x} "
                   f"eth{i} RoCE v2")
    return "\n".join(out)


def ib_bw_block(gbps: float) -> str:
    return "\n".join([
        "---------------------------------------------------------------------------------------",
        "                    RDMA_Write BW Test",
        " Dual-port       : OFF\t\tDevice         : mlx5_0",
        " Number of qps   : 4\t\tTransport type : IB",
        "---------------------------------------------------------------------------------------",
        " #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]",
        f" 1048576    140000           0.00               {gbps:.2f}  \t\t   0.046795",
        "---------------------------------------------------------------------------------------"])


DOCKER_INFO = {"ServerVersion": "29.9.2", "Driver": "overlayfs",
               "DriverStatus": [["driver-type", "io.containerd.snapshotter.v1"]],
               "DockerRootDir": "/mnt/nvme/docker", "Runtimes": {"nvidia": {}, "runc": {}}}


def disk_line(path, free_gib, size_gib=8000, fstype="ext4", dev="/dev/nvme0n1p1", mount=None):
    kb_free, kb_size = int(free_gib * KB_PER_GIB), int(size_gib * KB_PER_GIB)
    return f"{path}\t{fstype}\t{dev} {kb_size} {kb_size - kb_free} {kb_free} 10% {mount or path}"


def host_sections(node: Node, **over) -> dict[str, tuple[int, str]]:
    smi = "\n".join(f"{i}, NVIDIA B300 SXM6 AC, 275040" for i in range(8))
    sec = {
        "uname": (0, "x86_64\n6.8.0-60-generic"),
        "gpus": (0, smi),
        "driver": (0, "\n".join(["580.95.05"] * 8)),
        "fabric": (0, "\n".join(["    Fabric", "        State                             : Completed",
                                 "        Status                            : Success",
                                 "        CliqueId                          : 32766"] * 8)),
        "fabricmanager": (0, "active"),
        "topo": (0, topo_text()),
        "ibstat": (0, ibstat_text(layer="Ethernet" if node.fabric == "roce" else "InfiniBand")),
        "ibv_devinfo": (0, devinfo_text()),
        "gids": (0, gid_text() if node.fabric == "roce" else ""),
        "devib": (0, "rdma_cm\n" + "\n".join(f"uverbs{i}" for i in range(9))),
        "memlock": (0, "unlimited"),
        "meminfo": (0, f"MemTotal:       {2048 * KB_PER_GIB} kB\nMemAvailable:   {2000 * KB_PER_GIB} kB"),
        "addr": (0, f"2: {node.interface}    inet {node.address}/24 brd 192.0.2.255 scope global {node.interface}\\"
                    "       valid_lft forever preferred_lft forever\n1: lo    inet 127.0.0.1/8 scope host lo"),
        "peermem": (0, "6.8.0-60-generic\nnvidia_peermem         16384  0"),
        "docker": (0, json.dumps(DOCKER_INFO)),
        "clock": (0, "ntp=yes\nSystem time     : 0.000012345 seconds fast of NTP time\nLeap status     : Normal"),
        "disk": (0, "\n".join([disk_line("/", 80, 200, dev="/dev/sda2", mount="/"),
                               disk_line("/var/lib/containerd", 3000, mount="/mnt/nvme"),
                               disk_line("/mnt/nvme/docker", 3000, mount="/mnt/nvme"),
                               *[disk_line(p, 7000, mount="/mnt/nvme") for p in sorted(node.mounts)]])),
    }
    sec.update(over)
    return sec


def container_sections(node: Node, **over) -> dict[str, tuple[int, str]]:
    sec = {
        "memlock": (0, "unlimited"),
        "devib": (0, "rdma_cm\n" + "\n".join(f"uverbs{i}" for i in range(9))),
        "gpus": (0, "\n".join(f"GPU {i}: NVIDIA B300 SXM6 AC (UUID: GPU-{i})" for i in range(8))),
        "ibstat": (0, ibstat_text(layer="Ethernet" if node.fabric == "roce" else "InfiniBand")),
        "ibv_devinfo": (0, devinfo_text()),
        "tools": (0, "ib_write_bw=/usr/bin/ib_write_bw\ntorchrun=/usr/local/bin/torchrun\npython3=/usr/bin/python3\n"
                     "python=/usr/bin/python"),
        "torch": (0, "torch 2.9.1+cu130 cuda 13.0 available True devices 8 nccl 2.30.7"),
    }
    sec.update(over)
    return sec


# --------------------------------------------------------------------- fake SSH
@dataclass
class Behavior:
    host: dict = field(default_factory=dict)
    container: dict = field(default_factory=dict)
    ssh_rc: int = 0
    skew: float = 0.0
    pull: tuple[int, str] = (0, IMAGE)
    container_rc: int = 0
    blocked: set = field(default_factory=set)  # (node name, peer address, port)
    busy: tuple = ()
    rail_gbps: dict = field(default_factory=dict)  # hca -> Gb/s of this node as a client
    rail_error: str | None = None
    nccl: str = "ib"  # ib | ib-nogdr | socket | crash
    wrong: bool = False


def _sections_out(script: str, sections: dict[str, tuple[int, str]]) -> str:
    out = []
    for name in re.findall(rf"{cp.SECTION} ([A-Za-z_]+)", script):
        rc, text = sections.get(name, (127, f"bash: {name}: command not found"))
        out.append(f"{cp.SECTION} {name}\n{text}\n{cp.RC} {rc}")
    return "\n".join(out) + "\n"


class FakeCluster(SshOps):
    """Answers every script of the preflight like a node of a B300 cluster would."""

    NOW = 1_000_000.0

    def __init__(self, nodes, **per_node):
        self.nodes = {n.name: n for n in nodes}
        self.by_addr = {n.address: n.name for n in nodes}
        self.b = {n.name: per_node.get(n.name, Behavior()) for n in nodes}
        self.scripts: list[tuple[str, str]] = []

    def done(self, rc, out=""):
        return subprocess.CompletedProcess([], rc, stdout=out, stderr="")

    def run(self, node, script, *, timeout=300.0):
        self.scripts.append((node.name, script))
        b = self.b[node.name]
        if b.ssh_rc:
            return self.done(b.ssh_rc, "ssh: connect to host port 22: Connection timed out")
        if script == "date +%s.%N":
            return self.done(0, f"{self.NOW + b.skew:.6f}\n")
        if script.startswith("docker pull"):
            return self.done(*b.pull)
        if "docker run" in script and cp.SECTION in script:
            if b.container_rc:
                return self.done(b.container_rc, "docker: Error response from daemon: could not select device "
                                                 "driver \"\" with capabilities: [[gpu]].")
            return self.done(0, _sections_out(script, container_sections(node, **b.container)))
        if cp.SECTION in script:
            return self.done(0, _sections_out(script, host_sections(node, **b.host)))
        if " ports --ports " in script:
            return self.done(0, self._ports(node, script))
        if "ib_write_bw" in script:
            return self.done(0, self._ib(node, script))
        if "torch.distributed.run" in script:
            return self._nccl(node, script)
        raise AssertionError(f"unexpected script on {node.name}: {script[:200]}")

    def _ports(self, node, script):
        b = self.b[node.name]
        ports = re.search(r"--ports (\S+)", script).group(1).split(",")
        peers = re.search(r"--peers (\S+)", script).group(1).split(",")
        rec = {"mode": "ports", "node": node.name,
               "listen": {p: ("busy: Address already in use" if int(p) in b.busy else "listening") for p in ports},
               "dial": {a: {p: ("timeout" if (node.name, a, int(p)) in b.blocked else "ok") for p in ports}
                        for a in peers}}
        return f"{cp.BENCH} {json.dumps(rec)}\n"

    def _ib(self, node, script):
        b = self.b[node.name]
        client = re.search(r"--report_gbits(?: -x \d+)? (\d+\.\d+\.\d+\.\d+)", script)
        out = []
        for rail, hca in re.findall(rf"{cp.IB_MARK} (\d+) (mlx5_\d+)", script):
            out.append(f"{cp.IB_MARK} {rail} {hca}")
            if client and b.rail_error:
                out.append(b.rail_error)
            else:
                out.append(ib_bw_block(b.rail_gbps.get(hca, 392.5)))
        return "\n".join(out) + "\n"

    def _nccl(self, node, script):
        b = self.b[node.name]
        rank = int(re.search(r"--node-rank=(\d+)", script).group(1))
        nnodes = int(re.search(r"--nnodes=(\d+)", script).group(1))
        if b.nccl == "crash":
            return self.done(1, "NCCL WARN NET/IB : mlx5_3:1 Got async event : port error\n"
                                "torch.distributed.DistBackendError: NCCL error")
        if b.nccl == "socket":
            log = ["NCCL INFO NET/IB : No device found.", "NCCL INFO Using network Socket",
                   "NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/Socket/0"]
        else:
            gdr = "/GDRDMA" if b.nccl == "ib" else ""
            log = ["NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [2]mlx5_2:1/IB [3]mlx5_3:1/IB "
                   "[4]mlx5_4:1/IB [5]mlx5_5:1/IB [6]mlx5_6:1/IB [7]mlx5_7:1/IB [RO]; OOB bond0:192.0.2.10<0>",
                   "NCCL INFO Using network IB",
                   f"NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0{gdr}"]
        out = [f"[rank{rank * 8}]: {line}" for line in log]
        if rank == 0:
            world, local = 8 * nnodes, 8
            socket = any(self.b[n].nccl == "socket" for n in self.b)
            scale = 0.05 if socket else 1.0
            a2a_bound = 400 * (world - 1) / (local * (world - local))
            rows = {"all_reduce": 360 * scale, "all_to_all": 0.8 * a2a_bound * scale, "sendrecv": 45 * scale}
            out.append(f"{cp.BENCH} " + json.dumps({"mode": "collectives", "event": "init", "world": world,
                                                     "nodes": nnodes, "nccl": "2.30.7", "init_s": 4.2}))
            for op, busbw in rows.items():
                for size in (16 << 20, 1 << 30):
                    out.append(f"{cp.BENCH} " + json.dumps({
                        "mode": "collectives", "event": "result", "op": op, "bytes": size, "world": world,
                        "nodes": nnodes, "time_us": 100.0, "algbw_gbps": busbw, "busbw_gbps": busbw,
                        "correct": not b.wrong}))
            out.append(f"{cp.BENCH} " + json.dumps({"mode": "collectives", "event": "done",
                                                     "failed_ops": 3 if b.wrong else 0}))
        return self.done(1 if b.wrong and rank == 0 else 0, "\n".join(out) + "\n")


def b300(i: int, **kw) -> Node:
    base = dict(name=f"b300-{i}", host=f"ops@b300-{i}", device_family="cuda", cpu_arch="x86_64", card="B300",
                cards_per_node=8, hbm_gib=288.0, host_memory_gib=2048.0, fabric="ib", interface="bond0",
                address=f"192.0.2.{10 + i}", runtime="docker", docker_args=CUDA_RDMA_DOCKER_ARGS,
                mounts={"/mnt/nvme/yeto": "/root/.cache", "/mnt/nvme/models": "/data/models"}, image=IMAGE,
                env={"NCCL_IB_HCA": "=" + ",".join(f"mlx5_{j}" for j in range(8))})
    return Node(**(base | kw))


def run_pf(nodes, ops, tmp_path=None, **kw):
    pf = cp.ClusterPreflight(nodes, ops=ops, out_dir=tmp_path, log=lambda m: None, clock=lambda: FakeCluster.NOW,
                             **kw)
    return pf.run()


def find(report, name, scope=None):
    return [c for c in report.checks if c.name == name and (scope is None or c.scope == scope)]


def status(report, name, scope=None):
    return {c.status for c in find(report, name, scope)}


# ---------------------------------------------------------------------- parsers
def test_parse_ibstat_reads_every_port_and_link_layer():
    ports = cp.parse_ibstat(ibstat_text(down=(3,)))
    assert len(ports) == 9
    assert ports[0].ca == "mlx5_0" and ports[0].active and ports[0].rate_gbps == 400
    assert ports[0].link_layer == "InfiniBand" and ports[0].firmware == "28.43.1014"
    assert not ports[3].active and ports[3].physical == "Polling"
    assert ports[8].link_layer == "Ethernet" and ports[8].rate_gbps == 200


def test_parse_ibv_devinfo_derives_rate_and_mtu():
    ports = cp.parse_ibv_devinfo(devinfo_text(n=2, mtu=1024))
    assert [(p.ca, p.state, p.rate_gbps, p.active_mtu) for p in ports] == [
        ("mlx5_0", "Active", 400.0, 1024), ("mlx5_1", "Active", 400.0, 1024)]


def test_parse_topo_matrix_and_nic_legend():
    topo = cp.parse_topo(topo_text(sys_pair=(0, 5)))
    assert topo.gpus == [f"GPU{i}" for i in range(8)] and len(topo.nics) == 10
    assert topo.link[("GPU0", "GPU1")] == "NV18" and topo.link[("GPU0", "GPU5")] == "SYS"
    assert topo.link[("GPU3", "NIC3")] == "PIX" and topo.nic_names["NIC9"] == "mlx5_9"


def test_parse_fabric_and_driver():
    text = "    Fabric\n        State : In Progress\n        Status : N/A\n    Fabric\n        State : N/A\n"
    assert cp.parse_fabric(text) == [("In Progress", "N/A"), ("N/A", "")]
    assert cp.driver_versions("580.95.05\n580.95.05\nNVIDIA-SMI has failed") == ["580.95.05"] * 2
    assert cp.driver_major("570.86.15") == 570


def test_parse_gids_finds_roce_v2_ipv4_index():
    gids = cp.parse_gids(gid_text(n=2))
    v2 = [g for g in gids if g.index == 3]
    assert v2[0].ipv4 == "198.51.100.1" and v2[0].type == "RoCE v2"
    assert cp.roce_v2_index(gids, "mlx5_1") == 3 and cp.roce_v2_index(gids, "mlx5_9") is None


def test_parse_meminfo_memlock_disks_docker_clock():
    assert cp.parse_meminfo(f"MemTotal: {2 * KB_PER_GIB} kB\nMemAvailable: {KB_PER_GIB} kB") == (2.0, 1.0)
    assert cp.memlock_unlimited("unlimited") and cp.memlock_unlimited("8192") is False
    assert cp.memlock_unlimited("") is None
    disks, missing = cp.parse_disks(disk_line("/mnt/a", 100) + "\n/mnt/b\t-\tmissing")
    assert missing == ["/mnt/b"] and round(disks[0].free_gib) == 100 and disks[0].fstype == "ext4"
    info = cp.parse_docker_info("WARNING: bridge-nf-call\n" + json.dumps(DOCKER_INFO))
    assert cp.docker_snapshotter(info) and not cp.docker_snapshotter({"Driver": "overlay2", "DriverStatus": []})
    clock = cp.parse_clock("ntp=no\nSystem time     : 0.250000 seconds slow of NTP time\n")
    assert clock == {"ntp_synced": False, "chrony_offset_s": -0.25}


def test_parse_ib_write_bw_units_and_errors():
    assert cp.parse_ib_write_bw(ib_bw_block(391.2)) == (391.2, "")
    mb = ib_bw_block(1.0).replace("[Gb/sec]", "[MB/sec]").replace("1.00  ", "46000.00  ")
    gbps, _ = cp.parse_ib_write_bw(mb)
    assert gbps == pytest.approx(46000 * 8 * 1.048576 / 1000)
    gbps, err = cp.parse_ib_write_bw(" Couldn't connect to 192.0.2.10:18515\n Unable to open file descriptor")
    assert gbps is None and "Couldn't connect" in err
    assert cp.parse_ib_write_bw("")[0] is None


def test_parse_nccl_log_ib_gdr_socket_and_warnings():
    t = cp.parse_nccl_log("x NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB [RO]; OOB bond0:192.0.2.10<0>\n"
                          "x NCCL INFO Channel 00/0 : 0[0] -> 8[0] [send] via NET/IB/0/GDRDMA\n")
    assert t.ib and t.gdr and not t.socket and t.devices == ["mlx5_0:1", "mlx5_1:1"]
    t = cp.parse_nccl_log("NCCL INFO NET/IB : No device found.\nNCCL INFO Using network Socket\n"
                          "host:1:1 [0] NCCL WARN Call to ibv_reg_mr failed: Cannot allocate memory\n")
    assert t.socket and t.no_ib_device and not t.ib and t.warnings and "ibv_reg_mr" in t.warnings[0]


@pytest.mark.parametrize(("spec", "want", "missing"), [
    ("mlx5", [f"mlx5_{i}" for i in range(9)], []),
    ("=mlx5_0,mlx5_1", ["mlx5_0", "mlx5_1"], []),
    ("=mlx5_0:1,mlx5_7:1", ["mlx5_0", "mlx5_7"], []),
    ("^mlx5_8", [f"mlx5_{i}" for i in range(8)], []),
    ("^=mlx5_8", [f"mlx5_{i}" for i in range(8)], []),
    ("=mlx5_0,mlx5_12", ["mlx5_0"], ["mlx5_12"]),
])
def test_hca_filter_follows_nccl_ib_hca_syntax(spec, want, missing):
    selected, miss = cp.hca_filter(spec, cp.parse_ibstat(ibstat_text()))
    assert sorted(p.ca for p in selected) == sorted(want) and miss == missing


def test_roce_without_nccl_ib_hca_warns_about_extra_nics():
    nodes = [b300(i, fabric="roce", env={"NCCL_IB_GID_INDEX": "3"}) for i in range(2)]
    report = run_pf(nodes, FakeCluster(nodes))
    (c,) = find(report, "ib_ports", "b300-0")
    assert c.status == cp.WARN and "mixed link rates" in c.detail and "NCCL_IB_HCA" in c.fix
    beh = Behavior(host={"ibstat": (0, ibstat_text(n=10, layer="Ethernet", storage=False))})
    report = run_pf(nodes, FakeCluster(nodes, **{n.name: beh for n in nodes}))
    (c,) = find(report, "ib_ports", "b300-0")
    assert c.status == cp.WARN and "mlx5_8, mlx5_9" in c.detail


def test_compute_ports_default_drops_the_storage_nic_by_link_layer():
    ports = cp.parse_ibstat(ibstat_text(n=10, storage=False)) + cp.parse_ibstat(
        "CA 'mlx5_bond_0'\n\tPort 1:\n\t\tState: Active\n\t\tRate: 100\n\t\tLink layer: Ethernet")
    selected, _, how = cp.compute_ports(b300(0, env={}), ports)
    assert [p.ca for p in selected][:3] == ["mlx5_0", "mlx5_1", "mlx5_2"] and len(selected) == 10
    assert selected[-1].ca == "mlx5_9" and "infiniband" in how


@pytest.mark.parametrize("n", range(2, 8))
def test_round_robin_covers_every_pair_once_with_disjoint_rounds(n):
    rounds = cp.round_robin(n)
    pairs = [p for r in rounds for p in r]
    assert sorted(pairs) == sorted((i, j) for i in range(n) for j in range(i + 1, n))
    for r in rounds:
        flat = [x for p in r for x in p]
        assert len(flat) == len(set(flat))
    assert cp.disjoint_pairs(6) == [[(0, 1), (2, 3), (4, 5)]]
    assert cp.disjoint_pairs(3) == [[(0, 1)], [(0, 2)]]


def test_parse_sections_keeps_output_and_rc():
    script = cp.probe_script([("a", "echo hi"), ("b", "false")])
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    secs = cp.parse_sections(done.stdout)
    assert secs["a"] == cp.Section(0, "hi") and secs["b"].rc == 1


def test_host_probe_script_runs_under_bash():
    """Every host section is valid bash: on a machine without the tools each
    section still ends with its rc marker."""
    node = b300(0)
    done = subprocess.run(["bash", "-c", cp.probe_script(cp.host_sections(node))], capture_output=True, text=True,
                          timeout=120, check=False)
    secs = cp.parse_sections(done.stdout)
    assert set(secs) == {name for name, _ in cp.host_sections(node)}
    assert all(s.rc is not None for s in secs.values())
    assert secs["memlock"].rc == 0 and cp.memlock_unlimited(secs["memlock"].text) is not None


# ------------------------------------------------------------- the full cluster
def test_healthy_six_node_b300_cluster_passes_and_writes_the_report(tmp_path):
    nodes = [b300(i) for i in range(6)]
    ops = FakeCluster(nodes)
    report = run_pf(nodes, ops, tmp_path)
    bad = [c for c in report.checks if c.status in (cp.FAIL, cp.WARN)]
    assert report.ok and not bad, bad
    for name in ("driver", "gpus", "nvlink", "nvlink_fabric", "gpu_nic_affinity", "ib_ports", "dev_infiniband",
                 "memlock_container", "dev_infiniband_container", "host_memory", "disk", "disk_images", "docker",
                 "image_pull", "clock_sync", "address", "torch_cuda", "gpudirect"):
        assert status(report, name, "b300-0") == {cp.PASS}, name
    ib = find(report, "ib_write_bw")
    assert len(ib) == 15 and all("8 rails, 392.5 GB/s total" in c.detail for c in ib)
    assert "49.1" in ib[0].detail  # 392.5 Gb/s per rail in GB/s
    scopes = {c.scope for c in find(report, "nccl_all_reduce")}
    assert scopes == {"b300-0+b300-1", "b300-2+b300-3", "b300-4+b300-5", "all 6 nodes"}
    assert status(report, "nccl_transport") == {cp.PASS} and status(report, "tcp_ports") == {cp.PASS}
    data = json.loads((tmp_path / "preflight.json").read_text())
    assert data["ok"] and data["counts"]["FAIL"] == 0
    table = (tmp_path / "preflight.txt").read_text()
    assert "RESULT: PASS" in table and "nccl_all_to_all" in table
    assert (tmp_path / "raw" / "host" / "b300-3.log").exists()
    # the benchmarks ran in the trainer image with the inventory's docker_args and NCCL env
    nccl = [s for n, s in ops.scripts if "torch.distributed.run" in s]
    assert all("--pull never" in s and "--device /dev/infiniband" in s and "NCCL_IB_HCA==mlx5_0" in s
               and "NCCL_DEBUG=INFO" in s and "NCCL_IB_DISABLE=0" in s for s in nccl)
    assert len(nccl) == 2 * 3 + 6


def test_report_table_lists_fixes_for_findings():
    report = cp.Report([cp.Check("n0", "driver", cp.FAIL, "driver 570", "install >= 580"),
                        cp.Check("n0", "gpus", cp.PASS, "8 x B300")])
    table = report.table()
    assert "FIXES" in table and "[FAIL] n0 driver: install >= 580" in table and "RESULT: FAIL" in table


@pytest.mark.parametrize(("behavior", "check", "want", "fix"), [
    (Behavior(host={"driver": (0, "\n".join(["570.86.15"] * 8))}), "driver", cp.FAIL, ">= 580"),
    (Behavior(host={"gpus": (0, "\n".join(f"{i}, NVIDIA B300 SXM6 AC, 275040" for i in range(7)))}),
     "gpus", cp.FAIL, "cards_per_node"),
    (Behavior(host={"gpus": (9, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver")}),
     "gpus", cp.FAIL, "install the NVIDIA driver"),
    (Behavior(host={"fabric": (0, "    Fabric\n        State : In Progress\n        Status : N/A\n" * 8),
                    "fabricmanager": (3, "inactive")}), "nvlink_fabric", cp.FAIL, "fabricmanager"),
    (Behavior(host={"topo": (0, topo_text(sys_pair=(2, 6)))}), "nvlink", cp.FAIL, "NVLink"),
    (Behavior(host={"topo": (0, topo_text(far_nics=True))}), "gpu_nic_affinity", cp.WARN, "rail-aligned"),
    (Behavior(host={"ibstat": (0, ibstat_text(down=(5,)))}), "ib_ports", cp.FAIL, "subnet manager"),
    (Behavior(host={"ibstat": (0, ibstat_text(n=4))}), "ib_hca", cp.FAIL, "NCCL_IB_HCA"),
    (Behavior(host={"ibv_devinfo": (0, devinfo_text(mtu=2048))}), "ib_mtu", cp.WARN, "MTU"),
    (Behavior(host={"devib": (2, "ls: cannot access '/dev/infiniband': No such file or directory")}),
     "dev_infiniband", cp.FAIL, "RDMA stack"),
    (Behavior(host={"meminfo": (0, f"MemTotal: {1024 * KB_PER_GIB} kB\nMemAvailable: {1000 * KB_PER_GIB} kB")}),
     "host_memory", cp.FAIL, "1.6 TB"),
    (Behavior(host={"meminfo": (0, f"MemTotal: {2048 * KB_PER_GIB} kB\nMemAvailable: {900 * KB_PER_GIB} kB")}),
     "host_memory", cp.WARN, "leftover"),
    (Behavior(host={"disk": (0, "\n".join([disk_line("/", 50, 200, mount="/"),
                                           disk_line("/var/lib/containerd", 40, 200, mount="/"),
                                           disk_line("/mnt/nvme/docker", 3000, mount="/mnt/nvme"),
                                           disk_line("/mnt/nvme/yeto", 1500, mount="/mnt/nvme"),
                                           disk_line("/mnt/nvme/models", 1500, mount="/mnt/nvme")]))}),
     "disk_images", cp.FAIL, "does not follow data-root"),
    (Behavior(host={"disk": (0, "\n".join([disk_line("/var/lib/containerd", 3000, mount="/mnt/nvme"),
                                           disk_line("/mnt/nvme/yeto", 1500, mount="/mnt/nvme"),
                                           "/mnt/nvme/models\t-\tmissing"]))}), "disk", cp.FAIL, "mkdir"),
    (Behavior(host={"docker": (1, "permission denied while trying to connect to the Docker daemon socket")}),
     "docker", cp.FAIL, "usermod -aG docker"),
    (Behavior(host={"clock": (0, "ntp=no")}), "clock_sync", cp.WARN, "chrony"),
    (Behavior(skew=4.0), "ssh", cp.WARN, "chrony"),
    (Behavior(skew=-900.0), "ssh", cp.FAIL, "chrony"),
    (Behavior(host={"addr": (0, "2: eth0    inet 192.0.2.5/24 scope global eth0")}), "address", cp.FAIL,
     "cluster network"),
    (Behavior(host={"peermem": (0, "5.4.0-200-generic\npeermem: not loaded")}), "gpudirect", cp.WARN,
     "nvidia_peermem"),
    (Behavior(container={"memlock": (0, "8192")}), "memlock_container", cp.FAIL, "memlock=-1:-1"),
    (Behavior(container={"devib": (2, "ls: cannot access '/dev/infiniband'")}), "dev_infiniband_container",
     cp.FAIL, "--device /dev/infiniband"),
    (Behavior(container={"gpus": (0, "GPU 0: NVIDIA B300")}), "gpus_container", cp.FAIL, "--gpus all"),
    (Behavior(container={"torch": (1, "RuntimeError: The NVIDIA driver on your system is too old (found version "
                                      "12080)")}), "torch_cuda", cp.FAIL, "driver >= 580"),
    (Behavior(container_rc=125), "container", cp.FAIL, "nvidia-container-toolkit"),
    (Behavior(pull=(1, "failed to register layer: openat opt/sglang/x.dist-info/INSTALLER: not a directory")),
     "image_pull", cp.FAIL, "make_layer.py"),
    (Behavior(pull=(1, "write /var/lib/containerd/x: no space left on device")), "image_pull", cp.FAIL, "NVMe"),
    (Behavior(ssh_rc=255), "ssh", cp.FAIL, "BatchMode"),
    (Behavior(busy=(6379,)), "tcp_ports", cp.WARN, "ray stop"),
])
def test_node_finding_has_status_and_fix(behavior, check, want, fix):
    nodes = [b300(0), b300(1)]
    report = run_pf(nodes, FakeCluster(nodes, **{"b300-1": behavior}))
    found = find(report, check, "b300-1")
    assert found and want in {c.status for c in found}, report.table()
    assert any(fix in c.fix for c in found if c.status == want), [c.fix for c in found]
    if want == cp.FAIL:
        assert not report.ok
    assert status(report, check, "b300-0") <= {cp.PASS}


def test_unreachable_node_is_left_out_of_every_later_step():
    nodes = [b300(0), b300(1), b300(2)]
    ops = FakeCluster(nodes, **{"b300-2": Behavior(ssh_rc=255)})
    report = run_pf(nodes, ops)
    assert [n for n, s in ops.scripts if n == "b300-2"] == ["b300-2"]
    assert status(report, "ssh", "b300-2") == {cp.FAIL}
    assert {c.scope for c in find(report, "ib_write_bw")} == {"b300-0<->b300-1"}


def test_failed_pull_skips_that_nodes_container_steps():
    nodes = [b300(0), b300(1)]
    ops = FakeCluster(nodes, **{"b300-1": Behavior(pull=(1, "manifest unknown"))})
    report = run_pf(nodes, ops)
    assert not any("docker run" in s for n, s in ops.scripts if n == "b300-1")
    assert status(report, "nccl") == {cp.SKIP} and status(report, "tcp_ports") == {cp.SKIP}


def test_blocked_port_between_two_nodes_fails_with_the_pair():
    nodes = [b300(0), b300(1)]
    ops = FakeCluster(nodes, **{"b300-0": Behavior(blocked={("b300-0", "192.0.2.11", 29500)})})
    report = run_pf(nodes, ops)
    (c,) = find(report, "tcp_ports", "cluster")
    assert c.status == cp.FAIL and "b300-0->b300-1:29500 timeout" in c.detail and "random ports" in c.fix


def test_slow_and_broken_ib_rails():
    nodes = [b300(0), b300(1), b300(2)]
    ops = FakeCluster(nodes, **{"b300-1": Behavior(rail_gbps={"mlx5_6": 150.0}),
                                "b300-2": Behavior(rail_error=" Couldn't connect to 192.0.2.10:18521")})
    report = run_pf(nodes, ops)
    by_scope = {c.scope: c for c in find(report, "ib_write_bw")}
    assert by_scope["b300-0<->b300-1"].status == cp.FAIL  # 150/400 < 50 %
    assert "worst rail 38% of link" in by_scope["b300-0<->b300-1"].detail
    assert by_scope["b300-0<->b300-2"].status == cp.FAIL
    assert "Couldn't connect" in by_scope["b300-0<->b300-2"].detail
    ops = FakeCluster(nodes[:2], **{"b300-1": Behavior(rail_gbps={"mlx5_6": 260.0})})
    (c,) = find(run_pf(nodes[:2], ops), "ib_write_bw")
    assert c.status == cp.WARN


def test_nccl_socket_fallback_is_a_fail_and_bandwidth_is_flagged():
    nodes = [b300(0), b300(1)]
    report = run_pf(nodes, FakeCluster(nodes, **{"b300-1": Behavior(nccl="socket")}))
    (t,) = find(report, "nccl_transport")
    assert t.status == cp.FAIL and "NET/Socket" in t.detail and "/dev/infiniband" in t.fix
    assert status(report, "nccl_all_reduce") == {cp.WARN} and not report.ok


def test_nccl_without_gdr_warns_and_a_crash_or_wrong_result_fails():
    nodes = [b300(0), b300(1)]
    report = run_pf(nodes, FakeCluster(nodes, **{n.name: Behavior(nccl="ib-nogdr") for n in nodes}))
    assert status(report, "nccl_transport") == {cp.WARN}
    report = run_pf(nodes, FakeCluster(nodes, **{"b300-1": Behavior(nccl="crash")}))
    (c,) = find(report, "nccl")
    assert c.status == cp.FAIL and "b300-1" in c.detail and "port error" in c.detail
    report = run_pf(nodes, FakeCluster(nodes, **{"b300-0": Behavior(wrong=True)}))
    assert status(report, "nccl_all_reduce") == {cp.FAIL}


def test_roce_gid_index_checks():
    roce = dict(fabric="roce", env={"NCCL_IB_HCA": "=" + ",".join(f"mlx5_{j}" for j in range(8))})
    nodes = [b300(0, **roce), b300(1, **roce)]
    report = run_pf(nodes, FakeCluster(nodes))
    (g,) = find(report, "gid", "b300-0")
    assert g.status == cp.WARN and "NCCL_IB_GID_INDEX: 3" in g.fix
    assert status(report, "env", "b300-0") == {cp.WARN}
    good = {**roce, "env": {**roce["env"], "NCCL_IB_GID_INDEX": "3"}}
    nodes = [b300(0, **good), b300(1, **good)]
    ops = FakeCluster(nodes)
    report = run_pf(nodes, ops)
    assert status(report, "gid", "b300-0") == {cp.PASS}
    ib = [s for _, s in ops.scripts if "--report_gbits" in s]
    assert ib and all(" -x 3" in s for s in ib)
    bad = {**roce, "env": {**roce["env"], "NCCL_IB_GID_INDEX": "1"}}
    nodes = [b300(0, **bad), b300(1, **bad)]
    report = run_pf(nodes, FakeCluster(nodes, **{n.name: Behavior(host={"gids": (0, gid_text())}) for n in nodes}))
    (g,) = find(report, "gid", "b300-0")
    assert g.status == cp.FAIL and "[3]" in g.fix


def test_host_without_ibstat_reads_the_hcas_in_the_container():
    nodes = [b300(0), b300(1)]
    gone = {"ibstat": (127, "bash: ibstat: command not found"), "ibv_devinfo": (127, "bash: ibv_devinfo: not found")}
    report = run_pf(nodes, FakeCluster(nodes, **{n.name: Behavior(host=gone) for n in nodes}))
    (c,) = find(report, "ib_ports", "b300-0")
    assert c.status == cp.PASS and "(read in the container)" in c.detail
    assert len(find(report, "ib_write_bw")) == 1


def test_inventory_findings_before_any_ssh():
    nodes = [b300(0, docker_args=("--gpus", "all"), interface=None),
             b300(1, address=None, host="ops@b300-1", env={"NCCL_IB_DISABLE": "1"}, image=BAD_IMAGE)]
    checks = cp.inventory_checks(nodes, None)
    got = {(c.scope, c.name): c for c in checks}
    assert got[("b300-0", "docker_args")].status == cp.FAIL and "memlock" in got[("b300-0", "docker_args")].detail
    assert got[("b300-0", "interface")].status == cp.WARN
    assert got[("b300-1", "address")].status == cp.FAIL
    assert got[("b300-1", "env")].status == cp.FAIL
    assert got[("b300-1", "image")].status == cp.WARN and "make_layer" in got[("b300-1", "image")].fix


def test_overlay_unsafe_image_on_overlay_storage_fails_the_docker_check():
    nodes = [b300(0, image=BAD_IMAGE), b300(1, image=BAD_IMAGE)]
    report = run_pf(nodes, FakeCluster(nodes))
    assert status(report, "docker", "b300-0") == {cp.FAIL}


def test_single_node_runs_no_cross_node_step():
    nodes = [b300(0)]
    ops = FakeCluster(nodes)
    report = run_pf(nodes, ops)
    assert report.ok and not find(report, "tcp_ports") and not find(report, "nccl_transport")
    assert not any("--report_gbits" in s or "torch.distributed" in s or " ports " in s for _, s in ops.scripts)


def test_steps_can_be_skipped():
    nodes = [b300(0), b300(1)]
    ops = FakeCluster(nodes)
    run_pf(nodes, ops, steps=cp.Steps(pull=False, container=False))
    assert not any(s.startswith("docker pull") or "docker run" in s for _, s in ops.scripts)


# ------------------------------------------------------------------- the jobs
def test_container_job_runs_the_image_with_the_inventory_flags():
    node = b300(0)
    job = cp.container_job(node, image=None, inner="echo hi", env={"NCCL_DEBUG": "INFO"}, tag="t", timeout_s=60,
                           files={"x.py": "print('x')\n"})
    assert "docker run --rm --pull never --name yeto-preflight-t-b300-0 --network host --ipc host --gpus all" in job
    assert "--ulimit memlock=-1:-1" in job and "-e NCCL_DEBUG=INFO" in job and f"-e YETO_PF_DIR={cp.CONTAINER_DIR}"
    assert f"{IMAGE} timeout -k 10 60 bash -c 'echo hi'" in job and "print('x')" in job
    with pytest.raises(ValueError, match="delimiter"):
        cp.container_job(node, image=None, inner="", env={}, tag="t", timeout_s=1, files={"a": cp._DELIM})


def _timeout_shim(tmp_path: Path) -> dict:
    """``timeout`` for hosts without coreutils' (macOS): drop ``-k K SECONDS`` and run."""
    import os

    if shutil.which("timeout"):
        return dict(os.environ)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "timeout").write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
    (bin_dir / "timeout").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def test_rootfs_job_runs_the_bench_ports_mode_locally(tmp_path):
    """The job script and the shipped bench run for real (one node, on this host)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    node = b300(0, runtime="rootfs", docker_args=(), image=None)
    inner = cp.PY + (f"$PY \"$YETO_PF_DIR/{cp.BENCH_FILE}\" ports --ports {port} --peers 127.0.0.1 --wait 0.2 "
                     "--hold 0.5 --me n0")
    job = cp.container_job(node, image=None, inner=inner, env={}, tag="p", timeout_s=30,
                           files={cp.BENCH_FILE: cp.bench_source()})
    done = subprocess.run(["bash", "-c", job], capture_output=True, text=True, timeout=60, check=False,
                          env=_timeout_shim(tmp_path))
    (rec,) = cp.bench_records(done.stdout)
    assert rec == {"mode": "ports", "node": "n0", "listen": {str(port): "listening"},
                   "dial": {"127.0.0.1": {str(port): "ok"}}}, done.stderr


def test_rootfs_job_shape():
    node = b300(0, runtime="rootfs", docker_args=(), image=None)
    job = cp.container_job(node, image=None, inner="echo hi", env={"A": "b c"}, tag="p", timeout_s=5)
    assert "docker" not in job and "export A='b c'" in job and 'export YETO_PF_DIR="$d"' in job


# ----------------------------------------------------------------------- CLI
def _inventory(tmp_path: Path, n=2) -> Path:
    import yaml

    nodes = []
    for i in range(n):
        node = b300(i)
        nodes.append({"name": node.name, "host": node.host, "address": node.address, "interface": node.interface,
                      "device_family": "cuda", "cpu_arch": "x86_64", "card": "B300", "cards_per_node": 8,
                      "hbm_gib": 288, "host_memory_gib": 2048, "fabric": "ib", "runtime": "docker",
                      "docker_args": list(CUDA_RDMA_DOCKER_ARGS), "mounts": dict(node.mounts),
                      "env": dict(node.env)})
    path = tmp_path / "inventory.yaml"
    path.write_text(yaml.safe_dump({"images": {"x86_64": IMAGE}, "nodes": nodes}))
    return path


def test_resolve_nodes_from_gpu_spec_or_inventory_path(tmp_path):
    inv = _inventory(tmp_path)
    assert [n.name for n in cp.resolve_nodes(f"ssh:{inv}", None)] == ["b300-0", "b300-1"]
    assert [n.name for n in cp.resolve_nodes("ssh:2x8xb300", str(inv))] == ["b300-0", "b300-1"]
    with pytest.raises(cp.InventoryError, match="trainer node"):
        cp.resolve_nodes("ssh:6x8xb300", str(inv))
    with pytest.raises(cp.InventoryError, match="needs --rl-node-inventory"):
        cp.resolve_nodes("ssh:2x8xb300", None)
    with pytest.raises(cp.InventoryError, match="ssh:"):
        cp.resolve_nodes("modal:2x8xb200", str(inv))


def test_cli_preflight_writes_the_run_dir_and_exits_by_result(tmp_path, monkeypatch, capsys):
    from yeto import cli, runs

    inv = _inventory(tmp_path)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    args = cli.build_parser().parse_args(["preflight", "--gpu", "ssh:2x8xb300", "--rl-node-inventory", str(inv),
                                          "--cluster-prefix", "k31-bm", "--nccl-sizes", "16M,1G"])
    nodes = cp.resolve_nodes(args.gpu, args.rl_node_inventory)
    ops = FakeCluster(nodes)
    ops.NOW = time.time()  # main() reads the head clock
    rc = cp.main(args, ops=ops)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "RESULT: PASS" in out
    (run,) = (tmp_path / "runs" / "k31-bm" / "preflight").iterdir()
    assert json.loads((run / "preflight.json").read_text())["ok"]
    bad = FakeCluster(nodes, **{"b300-1": Behavior(host={"driver": (0, "550.54.15")})})
    bad.NOW = time.time()
    args = cli.build_parser().parse_args(["preflight", "--gpu", f"ssh:{inv}", "--run-dir", str(tmp_path / "r2"),
                                          "--json", "--no-bench"])
    assert cp.main(args, ops=bad) == 1
    assert json.loads(capsys.readouterr().out)["counts"]["FAIL"] >= 1
    assert cli.main(["preflight", "--gpu", "ssh:2x8xb300"]) == 2


def test_requirements_can_be_lowered_for_a_smaller_plan():
    nodes = [b300(0, host_memory_gib=1024.0)]
    beh = Behavior(host={"meminfo": (0, f"MemTotal: {1024 * KB_PER_GIB} kB\nMemAvailable: {1000 * KB_PER_GIB} kB")})
    req = replace(cp.Requirements(), min_host_memory_gib=900, rec_host_memory_gib=1000)
    report = run_pf(nodes, FakeCluster(nodes, **{"b300-0": beh}), req=req)
    assert status(report, "host_memory", "b300-0") == {cp.PASS}


def test_node_without_rendezvous_address_is_left_out_of_the_cross_node_steps():
    nodes = [b300(0), b300(1), b300(2, address=None)]
    report = run_pf(nodes, FakeCluster(nodes))
    assert status(report, "address", "b300-2") == {cp.FAIL}
    assert {c.scope for c in find(report, "ib_write_bw")} == {"b300-0<->b300-1"}
    assert {c.scope for c in find(report, "nccl_all_reduce")} == {"b300-0+b300-1"}


def test_preflight_checks_the_image_the_launcher_runs(monkeypatch):
    """An inventory that names the pinned K3.1 image that overlay appliers refuse is
    probed with its overlay-safe repack, as ssh_trainer_island.run_image runs it."""
    from yeto.rl.adapters.miles import pins

    old, new = next(iter(pins.OVERLAY_SAFE_DIGESTS.items()))
    rejected, repack = f"ghcr.io/o/yeto-miles-k31@sha256:{old}", f"ghcr.io/o/yeto-miles-k31@sha256:{new}"
    nodes = [b300(i, image=rejected) for i in range(2)]
    assert cp.trainer_image(None, nodes) == repack
    assert cp.trainer_image(rejected, nodes) == repack
    assert cp.trainer_image("ghcr.io/o/i:1", [b300(0, image=None)]) == "ghcr.io/o/i:1"
    with pytest.raises(cp.SshIslandError, match="is not the run's --rl-image"):
        cp.trainer_image("ghcr.io/o/other@sha256:" + "b" * 64, nodes)
    seen = {}

    class Stop(Exception):
        pass

    def capture(nodes, *, image, **kw):
        seen["image"] = image
        raise Stop

    monkeypatch.setattr(cp, "ClusterPreflight", capture)
    monkeypatch.setattr(cp, "resolve_nodes", lambda gpu, inv: nodes)
    from types import SimpleNamespace
    args = SimpleNamespace(gpu="ssh:2x8xb300", rl_node_inventory="x", image=None, run_dir="/tmp/unused",
                           no_bench=True, no_container=False, no_pull=False, no_ib=True, no_nccl=True,
                           ib_pairs="all", nccl_sizes="16M", nccl_iters=1, nccl_pairs_only=True,
                           min_driver=580, min_host_memory_gib=1.0, min_disk_free_gib=1.0, json=False)
    with pytest.raises(Stop):
        cp.main(args, ops=object())
    assert seen["image"] == repack


def test_the_port_probe_covers_the_ray_agent_ports_ssh_nodes_pin():
    """The agent ports the SSH island pins (rl_island_task.RAY_AGENT_PORTS) and Ray's
    fixed dashboard agent port are probed between nodes, as the head's ports are."""
    from yeto.launch.rl_island_task import RAY_AGENT_PORTS

    ports = set(cp.Requirements().ports)
    assert set(RAY_AGENT_PORTS.values()) | {52365, 6379, 8265, 10001, 29500} <= ports
