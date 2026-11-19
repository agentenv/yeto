"""bm-sim cluster: simulated bare-metal nodes as containers on one docker host.

``up`` builds the node image (``node/Dockerfile``), then starts on a private
bridge network:

* ``bm-sim-n<k>`` (k = 0..N-1): one node each, sshd for user ``yeto`` (key only),
  Ray and the yeto CPU runtime, the yeto checkout mounted read-only at the
  trainer workdir, a fake ``nvidia-smi`` reporting the inventory's cards and a
  ``docker`` shim for ``runtime: docker`` nodes (``node/docker-shim``);
* ``bm-sim-store``: an S3-compatible store (the Versity S3 Gateway over a POSIX
  directory) for the run's checkpoint store (``--store local``, default), or none (``--store r2``: the store credentials of
  the environment, a test prefix of a real bucket);
* ``bm-sim-head``: the launcher host, which reaches the nodes by ssh (known_hosts
  filled by ``ssh-keyscan``) and holds the docker socket to power nodes off and on.

It writes into the work dir: ``inventory.yaml`` (rootfs nodes),
``inventory-docker.yaml`` (the same nodes as ``runtime: docker``),
``store.env`` (the head's store settings), ``cluster.json`` and ``ssh/``.
``down`` removes every ``bm-sim-*`` container and the network.

The docker host needs only docker (Linux; a Modal sandbox with docker enabled
works, see ``run.py --modal``). The containers need no internet: the image
build does (``docker build --network host``).
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
IMAGE = "yeto-bm-sim-node:1"
# Versity S3 Gateway over a POSIX directory (MinIO publishes no images any more)
STORE_IMAGE = os.environ.get(
    "BM_SIM_STORE_IMAGE",
    "versity/versitygw@sha256:8319a530d08d537426e2f855bae5280709795dc73a3f0c347973ef915d6343c5")
NETWORK = "bm-sim"
SUBNET = os.environ.get("BM_SIM_SUBNET", "198.18.0.0/24")
CARDS = 2  # simulated cards per node (the inventory and the fake nvidia-smi agree)
CARD_MIB = 275040  # what nvidia-smi reports for a B300 (inside the 8% HBM tolerance of 288 GiB)
WORKDIR = "/opt/yeto"


def sh(*argv: str, check: bool = True, capture: bool = False, timeout: float | None = None) -> str:
    done = subprocess.run(list(argv), check=False, text=True, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)
    if check and done.returncode != 0:
        raise SystemExit(f"bm-sim: {' '.join(argv[:4])} ... failed (rc {done.returncode})"
                         + (f": {done.stdout[-2000:]}" if capture else ""))
    return done.stdout or ""


def addresses(nodes: int, subnet: str = SUBNET) -> dict[str, str]:
    """Fixed addresses on the bridge: head .2, store .5, node k .10+k."""
    net = ipaddress.ip_network(subnet)
    base = int(net.network_address)
    if nodes < 1 or 10 + nodes >= net.num_addresses - 1:
        raise ValueError(f"{nodes} node(s) do not fit {subnet}")
    out = {"head": str(ipaddress.ip_address(base + 2)), "store": str(ipaddress.ip_address(base + 5))}
    out.update({f"n{k}": str(ipaddress.ip_address(base + 10 + k)) for k in range(nodes)})
    return out


def inventory(nodes: int, ips: dict[str, str], *, runtime: str = "rootfs") -> dict:
    """The node inventory of the simulated cluster (``node_inventory`` format)."""
    rows = []
    for k in range(nodes):
        ip = ips[f"n{k}"]
        row = {"name": f"n{k}", "host": f"yeto@{ip}", "address": ip, "interface": "eth0",
               "device_family": "cuda", "cpu_arch": "x86_64", "card": "B300", "cards_per_node": CARDS,
               "hbm_gib": 288, "host_memory_gib": 64, "fabric": "tcp", "runtime": runtime,
               "roles": ["trainer"], "env": {"BM_SIM_NODE": f"n{k}"}}
        if runtime == "docker":
            # the customer form: an InfiniBand fabric, the RDMA docker flags, the HCA list
            # (no image: the run's --rl-image decides)
            row.update(fabric="ib",
                       docker_args=["--gpus", "all", "--device", "/dev/infiniband", "--ulimit", "memlock=-1:-1",
                                    "--ulimit", "stack=67108864", "--cap-add", "IPC_LOCK"],
                       env={"BM_SIM_NODE": f"n{k}", "NCCL_IB_HCA": "mlx5_0,mlx5_1"})
        rows.append(row)
    return {"nodes": rows}


def _yaml(data: dict) -> str:
    return json.dumps(data, indent=2)  # JSON is YAML; the inventory loader reads either


def up(work: Path, nodes: int, *, store: str = "local") -> dict:
    work = work.resolve()
    (work / "ssh").mkdir(parents=True, exist_ok=True)
    down(quiet=True)
    sh("docker", "build", "--network", "host", "-t", IMAGE, str(HERE / "node"))
    sh("docker", "network", "create", "--subnet", SUBNET, NETWORK)
    ips = addresses(nodes)
    key = work / "ssh" / "id_ed25519"
    if not key.exists():
        sh("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "bm-sim-head", "-f", str(key))
    env_lines = []
    if store == "local":
        user, password = "bmsim", secrets.token_hex(16)
        sh("docker", "run", "-d", "--name", "bm-sim-store", "--network", NETWORK, "--ip", ips["store"],
           "-e", f"ROOT_ACCESS_KEY={user}", "-e", f"ROOT_SECRET_KEY={password}", "--entrypoint", "sh",
           STORE_IMAGE, "-c", "mkdir -p /tmp/gw /tmp/meta && exec versitygw --port :9000 posix "
           "--sidecar /tmp/meta /tmp/gw")
        env_lines += [f"YETO_RL_STORE_S3_ACCESS_KEY_ID={user}", f"YETO_RL_STORE_S3_SECRET_ACCESS_KEY={password}",
                      f"YETO_RL_STORE_S3_ENDPOINT_URL=http://{ips['store']}:9000",
                      "YETO_RL_STORE_S3_REGION=us-east-1"]
        store_root = "s3://bm-sim"
    else:
        for name in ("YETO_RL_STORE_S3_ACCESS_KEY_ID", "YETO_RL_STORE_S3_SECRET_ACCESS_KEY",
                     "YETO_RL_STORE_S3_ENDPOINT_URL", "BM_SIM_R2_BUCKET"):
            if not os.environ.get(name):
                raise SystemExit(f"bm-sim --store r2 needs {name} in the environment")
        env_lines += [f"{n}={os.environ[n]}" for n in ("YETO_RL_STORE_S3_ACCESS_KEY_ID",
                                                        "YETO_RL_STORE_S3_SECRET_ACCESS_KEY",
                                                        "YETO_RL_STORE_S3_ENDPOINT_URL")]
        env_lines.append("YETO_RL_STORE_S3_REGION=auto")
        store_root = f"s3://{os.environ['BM_SIM_R2_BUCKET']}/tests/bm-sim"
    (work / "store.env").write_text("\n".join(env_lines) + "\n")
    os.chmod(work / "store.env", 0o600)
    for k in range(nodes):
        sh("docker", "run", "-d", "--name", f"bm-sim-n{k}", "--hostname", f"bm-sim-n{k}",
           "--network", NETWORK, "--ip", ips[f"n{k}"], "--shm-size", "1g",
           "-e", "BM_SIM_DOCKER=1", "-e", f"BM_SIM_CARDS={CARDS}", "-e", f"BM_SIM_CARD_MIB={CARD_MIB}",
           "-v", f"{REPO}:{WORKDIR}:ro", "-v", f"{key}.pub:/etc/bm-sim/authorized_keys:ro", IMAGE)
    sh("docker", "run", "-d", "--name", "bm-sim-head", "--hostname", "bm-sim-head", "--network", NETWORK,
       "--ip", ips["head"], "-v", f"{REPO}:{WORKDIR}:ro", "-v", f"{work}:/work",
       "-v", "/var/run/docker.sock:/var/run/docker.sock", "-w", WORKDIR,
       "-e", f"PYTHONPATH={WORKDIR}", "-e", "YETO_RUNS_DIR=/work/runs",
       "--entrypoint", "sleep", IMAGE, "infinity")
    node_ips = " ".join(ips[f"n{k}"] for k in range(nodes))
    head_ssh = (
        "set -eu; install -d -m 700 /root/.ssh; install -m 600 /work/ssh/id_ed25519 /root/.ssh/id_ed25519; "
        f"for ip in {node_ips}; do for _ in $(seq 60); do "
        "ssh-keyscan -T 2 -t ed25519 \"$ip\" 2>/dev/null >> /root/.ssh/known_hosts && break; sleep 1; done; done; "
        f"for ip in {node_ips}; do ssh -o BatchMode=yes yeto@$ip true; done")
    sh("docker", "exec", "bm-sim-head", "bash", "-c", head_ssh, timeout=300)
    (work / "inventory.yaml").write_text(_yaml(inventory(nodes, ips)))
    (work / "inventory-docker.yaml").write_text(_yaml(inventory(nodes, ips, runtime="docker")))
    if store == "local":
        wait_store(ips["store"])
    cluster = {"nodes": nodes, "cards": CARDS, "ips": ips, "store": store, "store_root": store_root,
               "image": IMAGE, "store_image": STORE_IMAGE, "workdir": WORKDIR}
    (work / "cluster.json").write_text(json.dumps(cluster, indent=2))
    return cluster


def wait_store(ip: str, timeout_s: float = 120.0) -> None:
    probe = f"import socket; socket.create_connection(('{ip}', 9000), timeout=2).close()"
    deadline = time.time() + timeout_s
    while subprocess.run(["docker", "exec", "bm-sim-head", "python3", "-c", probe]).returncode != 0:
        if time.time() > deadline:
            raise SystemExit("bm-sim: the S3 store did not come up")
        time.sleep(2)


def down(*, quiet: bool = False) -> None:
    names = sh("docker", "ps", "-aq", "--filter", "name=^bm-sim-", capture=True, check=False).split()
    if names:
        sh("docker", "rm", "-f", *names, capture=quiet, check=False)
    sh("docker", "network", "rm", NETWORK, capture=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bm-sim cluster on a docker host")
    parser.add_argument("action", choices=["up", "down"])
    parser.add_argument("--work", default=os.environ.get("BM_SIM_WORK", "/tmp/bm-sim"))
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--store", choices=["local", "r2"], default="local")
    opts = parser.parse_args(argv)
    if opts.action == "up":
        print(json.dumps(up(Path(opts.work), opts.nodes, store=opts.store), indent=2))
    else:
        down()
    return 0


if __name__ == "__main__":
    sys.exit(main())
