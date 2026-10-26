"""RL island topology: island specs, placement, trainer shape and node counts."""

from __future__ import annotations

import dataclasses

from ..gpu_spec import DEVICE_FAMILIES, parse_gpu_spec
from .rl_backend import _rl_backend_module


# rl-multinode-island D6: per-cloud NCCL/GLOO socket settings for a multi-node
# island (values are the CPU-side best guess; GPU task G1 confirms them).
# "auto": the first non-virtual interface that is up, resolved on the node
# (G1 2026-10-03: a Nebius L40S node's NIC is "network-interface-0", shown
# truncated as "network-interfa"; "eth0" made gloo fail with "Unable to find
# address for: eth0" in the SGLang scheduler).  "" leaves NCCL/gloo to their
# own detection.  NCCL_SOCKET_IFNAME in the environment always wins.
MULTINODE_IB_CLOUDS = frozenset({"modal"})
MULTINODE_SOCKET_IFNAME = {"nebius": "auto", "aws": "", "gcp": "", "ssh": ""}
_DETECT_IFACE = (
    "YETO_IFACE=${NCCL_SOCKET_IFNAME:-$(for d in /sys/class/net/*; do n=$(basename \"$d\"); "
    "case \"$n\" in lo|docker*|veth*|br-*|virbr*) continue;; esac; "
    "[ \"$(cat \"$d/operstate\" 2>/dev/null)\" = up ] && { echo \"$n\"; break; }; done)}"
)



# rl-ssh-node-pool D5: collective-communication variables of an SSH-pool
# island, from the inventory's device family and fabric instead of a cloud
# name. The cloud path above is unchanged.
#
# Variable names per family:
#  - cuda: NCCL/GLOO, as the cloud path already writes them.
#  - npu: HCCL. HCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME and HCCL_CONNECT_TIMEOUT
#    are confirmed (task N1.4): the Ascend deployment guides in
#    sglang/docs/hardware-platforms/ascend-npus export HCCL_SOCKET_IFNAME next
#    to GLOO_SOCKET_IFNAME, and HCCL_CONNECT_TIMEOUT / HCCL_EXEC_TIMEOUT appear
#    in the CANN collective-communication variable reference. ``HCCL_IF_IP``
#    (named in design D5) is NOT confirmed, so this module does not write it.
#  - ppu: ACCL-P keeps the NCCL names (T-PPU-gaps.md 2.3), so a PPU island
#    exports NCCL/GLOO variables; a 16-card node uses interface hpn0.
HCCL_CONNECT_TIMEOUT_S = 1800
RDMA_FABRICS = frozenset({"roce", "ib"})


def island_comm_env(device_family: str, fabric: str = "tcp", interface: str | None = None,
                    *, num_nodes: int = 1) -> dict[str, str]:
    """Collective-communication variables for one island node (pure function).

    ``{}`` for a single-node island, as on the cloud path. A fabric the
    inventory declares is never disabled: ``roce`` and ``ib`` keep NCCL's IB
    transport on.
    """
    if device_family not in DEVICE_FAMILIES:
        raise ValueError(f"unknown device family {device_family!r} "
                         f"(known: {DEVICE_FAMILIES})")
    if num_nodes <= 1:
        return {}
    if device_family == "npu":
        env = {"HCCL_CONNECT_TIMEOUT": str(HCCL_CONNECT_TIMEOUT_S)}
        if interface:
            env["HCCL_SOCKET_IFNAME"] = interface
            env["GLOO_SOCKET_IFNAME"] = interface
        return env
    env = {"NCCL_IB_DISABLE": "0" if fabric in RDMA_FABRICS else "1",
           "NCCL_DEBUG": "WARN"}
    if interface:
        env["NCCL_SOCKET_IFNAME"] = interface
        env["GLOO_SOCKET_IFNAME"] = interface
    return env


def island_comm_env_prelude(device_family: str, fabric: str = "tcp",
                            interface: str | None = None, *, num_nodes: int = 1) -> str:
    """:func:`island_comm_env` as shell exports; "" on a single node.

    An existing value in the environment wins, as in the cloud prelude.
    """
    env = island_comm_env(device_family, fabric, interface, num_nodes=num_nodes)
    if not env:
        return ""
    lines = [f"export {name}=${{{name}:-{value}}}" for name, value in sorted(env.items())]
    return "\n".join(lines) + "\n"


def multinode_env_prelude(cloud: str, num_nodes: int) -> str:
    """Shell exports every island node runs before Ray starts; "" on one node."""
    if num_nodes <= 1:
        return ""
    # Modal clustered islands request RDMA (ModalIslandConfig.rdma): leave
    # NCCL's IB transport on there; other clouds run NCCL over TCP.
    ib_default = 0 if cloud in MULTINODE_IB_CLOUDS else 1
    lines = [f"export NCCL_IB_DISABLE=${{NCCL_IB_DISABLE:-{ib_default}}}",
             "export NCCL_DEBUG=${NCCL_DEBUG:-WARN}"]
    iface = MULTINODE_SOCKET_IFNAME.get(cloud, "")
    if iface == "auto":
        lines += [_DETECT_IFACE,
                  'echo "[yeto-island] socket interface: ${YETO_IFACE:-<none>}"',
                  'export NCCL_SOCKET_IFNAME="$YETO_IFACE" GLOO_SOCKET_IFNAME="$YETO_IFACE"']
    elif iface:
        lines += [f"export NCCL_SOCKET_IFNAME={iface}", f"export GLOO_SOCKET_IFNAME={iface}"]
    return "\n".join(lines) + "\n"


# Clouds that accept sky's ``network_tier="best"`` only for some shapes: Nebius
# offers the InfiniBand tier for H100:8 / H200:8 nodes only and REJECTS the
# request for anything else ("Catalog does not contain any instances"), so a
# 2x1xL40S island must not ask for it (G1 of rl-multinode-island: the D6
# prelude already runs NCCL over TCP with NCCL_IB_DISABLE=1 there).
NETWORK_TIER_BEST_SHAPES = {"nebius": {("H100", 8), ("H200", 8)}}


def multinode_network_tier(cloud: str, gpu: str, gpus_per_node: int) -> str | None:
    """``"best"`` for a multi-node island when the cloud can honor it, else None."""
    shapes = NETWORK_TIER_BEST_SHAPES.get(cloud)
    if shapes is not None and (gpu.upper(), int(gpus_per_node)) not in shapes:
        return None
    return "best"


def rl_island_spec(args, spec):
    """The island view of ``spec`` under ``--rl-island-use-gpus-per-node M`` (m5 on
    8-GPU machines: provision N physical GPUs per node, give the island only local
    GPUs ``0..M-1``). Every island-side rule (trainer shape, layout, colocated
    engine check, gpu_pool, learner ``--rl-island-gpus-per-node``) sees M; the sky
    resources keep the physical ``spec``. Unset -> ``spec`` itself (no change)."""
    m = getattr(args, "rl_island_use_gpus_per_node", None)
    if m is None:
        return spec
    m = int(m)
    if m < 1 or m > spec.gpus_per_node:
        raise ValueError(f"--rl-island-use-gpus-per-node {m} must be in 1..{spec.gpus_per_node} "
                         f"(physical GPUs per node of {spec})")
    if spec.cloud == "modal":
        raise ValueError("--rl-island-use-gpus-per-node is not supported on Modal islands")
    if m == spec.gpus_per_node:
        return spec
    return dataclasses.replace(spec, gpus_per_node=m)


def rl_island_count(args) -> int:
    """The number of learner islands of the launch (one per ``--gpu`` entry)."""
    return len(parse_gpu_spec(args.gpu)) if getattr(args, "gpu", None) else 1


def rl_island_args(args, learner_id: int):
    """``args`` as island ``learner_id`` sees it (rl-mixed-island-placement): a
    copy with its per-island placement values, or ``args`` itself without any.
    Every island-side rule below (trainer shape, layout, minimum nodes, learner
    command) reads the placement from these args, never from the run-wide ones."""
    from ..rl.engine import island_placement

    return island_placement.island_args(args, learner_id, rl_island_count(args))


def rl_island_placement_args(args) -> list:
    """One island args copy per distinct placement value set of the launch."""
    seen, out = set(), []
    for m in range(rl_island_count(args)):
        iargs = rl_island_args(args, m)
        # rollout_num_gpus is included because a colocated island of a mixed
        # run drops it (island_placement.island_args); the other fields follow.
        key = (getattr(iargs, "rl_placement", "colocated"),
               getattr(iargs, "rl_standby_gpus", 0), getattr(iargs, "rollout_num_gpus", None),
               getattr(iargs, "rl_island_use_gpus_per_node", None))
        if key not in seen:
            seen.add(key)
            out.append(iargs)
    return out


def rl_island_placement_records(args) -> list[dict]:
    """Run manifest / dry-run rows: every island's placement values and the
    AlgorithmSpec hash it receives (the same on every island, P4-Q4)."""
    from ..rl.engine import island_placement

    algorithm = getattr(args, "rl_expected_algorithm_sha256", None)
    return [{**p.record(), "algorithm_sha256": algorithm}
            for p in island_placement.island_placements(args, rl_island_count(args))]


def rl_island_gpu_env(args, spec) -> str:
    """Shell exports pinning the island to local GPUs ``0..M-1`` (exported before
    ``ray start`` so Ray, the learner, Miles actors and the sglang subprocesses all
    inherit them); "" without ``--rl-island-use-gpus-per-node`` or when M equals the
    physical count."""
    island = rl_island_spec(args, spec)
    if island is spec:
        return ""
    m = island.gpus_per_node
    return (f"export CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(m))} "
            f"YETO_ISLAND_USE_GPUS_PER_NODE={m}\n"
            f'echo "[yeto-island] GPUs: physical {spec.gpus_per_node}/node, allocated {m}/node '
            f'(CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"\n')


def rl_island_ray_gpus(args, spec) -> str:
    island = rl_island_spec(args, spec)
    return "" if island is spec else f" --num-gpus={island.gpus_per_node}"


def rl_actor_gpus_per_node(args, spec) -> int:
    """Trainer GPUs per node: all of them when colocated; under
    ``--rl-placement fixed-partition`` the rest after rollout and standby."""
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return spec.gpus_per_node
    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    standby = int(getattr(args, "rl_standby_gpus", 0) or 0)
    if rollout < 1:
        raise ValueError("--rl-placement fixed-partition needs --rollout-num-gpus >= 1")
    if spec.num_nodes == 1:
        return spec.gpus_per_node - rollout - standby
    return rl_trainer_shape(args, spec)[1]


def rl_trainer_shape(args, spec) -> tuple[int, int]:
    """``(actor_num_nodes, actor_num_gpus_per_node)`` of the island's trainer
    (rl-multinode-island D5). Colocated: every GPU. Fixed partition: the island
    total minus ``--rl-rollout-gpus`` and ``--rl-standby-gpus`` (island totals),
    laid out as the leading logical bundles; it must occupy the same number of
    GPUs on every node it uses (Miles' rectangle) and keep every TP*PP group
    on one node."""
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return spec.num_nodes, spec.gpus_per_node
    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    standby = int(getattr(args, "rl_standby_gpus", 0) or 0)
    if rollout < 1:
        raise ValueError("--rl-placement fixed-partition needs --rollout-num-gpus >= 1")
    trainer = spec.total_gpus - rollout - standby
    if trainer < 1:
        raise ValueError(
            "--rl-placement fixed-partition needs --rl-rollout-gpus + --rl-standby-gpus "
            f"< island GPUs ({spec.total_gpus})"
        )
    layout = rl_island_layout(args, spec)
    if layout is not None:
        return layout[0], layout[1]
    from ..rl.engine.multinode import Topology, TopologyError, rectangular_trainer

    topology = Topology(spec.num_nodes, spec.gpus_per_node)
    try:
        return rectangular_trainer([topology.slot_of(b) for b in range(trainer)])
    except TopologyError as exc:
        raise ValueError(f"--rl-placement fixed-partition on {spec}: {exc}") from None


def rl_cross_node_switches(args) -> tuple[bool, bool]:
    """``(allow_cross_node_tp, allow_cross_node_engine_tp)`` (ruling 2026-10-04 v2):
    the CLI flags ``--rl-allow-cross-node-tp`` / ``--rl-allow-cross-node-engine-tp`` OR
    the initial elastic cfg's ``parallel.allow_cross_node_*`` (set by
    ``_check_ports_infra_switches``); either spelling lifts the default preference and
    both are forwarded to the learner."""
    cfg = getattr(args, "rl_elastic_initial_cfg_switches", None) or {}
    return (bool(getattr(args, "rl_allow_cross_node_tp", False) or cfg.get("allow_cross_node_tp")),
            bool(getattr(args, "rl_allow_cross_node_engine_tp", False) or cfg.get("allow_cross_node_engine_tp")))


def rl_colocated_engine_check(args, spec) -> list[dict[str, int]]:
    """m5: node rules of a colocated multi-node island's rollout engines (ruling
    2026-10-04 v2): engine i = logical bundles ``i*g .. i*g+g-1`` of the island; one
    node by default, a cross-node engine only with ``--rl-allow-cross-node-engine-tp``
    and as whole nodes. Returns the expected sglang TP rank map of engine 0 (the
    m5 judge's rank -> GPU reconciliation reference); raises ValueError otherwise."""
    from ..rl.engine.multinode import (Topology, TopologyError, colocated_engine_slots,
                                      node_placement_rejection, sglang_tp_rank_map)

    g = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    topology = Topology(spec.num_nodes, spec.gpus_per_node)
    _tp, cross_engine = rl_cross_node_switches(args)
    try:
        engines = colocated_engine_slots(topology.total, g, topology)
    except TopologyError as exc:
        raise ValueError(f"colocated rollout engines on {spec}: {exc}") from None
    why = node_placement_rejection({"trainer": [], "rollout": engines, "standby": []},
                                   gpus_per_engine=g, allow_cross_node_engine=cross_engine,
                                   gpus_per_node=spec.gpus_per_node)
    if why:
        raise ValueError(f"colocated rollout engines on {spec}: {why}")
    if cross_engine and g > spec.gpus_per_node:
        print("[launcher] WARNING: --rl-allow-cross-node-engine-tp: a colocated rollout engine of "
              f"{g} GPUs spans {g // spec.gpus_per_node} whole nodes (sglang nnodes>1; scales "
              "as one replica)")
    return sglang_tp_rank_map(engines[0])


def rl_cross_node_flags(args) -> str:
    tp, engine = rl_cross_node_switches(args)
    return _rl_backend_module(args, "launch_flags").cross_node_flags(tp, engine)


def rl_island_layout(args, spec) -> tuple[int, int, dict[str, tuple[int, ...]]] | None:
    """Q2 (2026-10-04 ruling, mixed rollout/trainer nodes): the trainer shape and
    role -> logical bundle map derived from the ``--rl-elastic-initial-config``'s
    explicit ``placement`` (``args.rl_elastic_initial_placement_slots``, set by
    ``_check_ports_infra_switches``), or None when the manifest declares no
    placement (the leading-bundle layout of :func:`rl_trainer_shape` stays).
    The placement must match ``--gpu`` / ``--rl-rollout-gpus`` / ``--rl-standby-gpus``
    (island totals), be a rectangle of contiguous per-node runs
    (``multinode.trainer_layout``) and agree with ``PlacementRequest.trainer_shape``
    (the learner-side rule set); any disagreement fails here, before any cloud work."""
    slots = getattr(args, "rl_elastic_initial_placement_slots", None)
    if slots is None or getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return None
    PlacementRequest = _rl_backend_module(args, "placement").PlacementRequest
    from ..rl.engine.multinode import Topology, TopologyError, trainer_layout

    rollout = int(getattr(args, "rollout_num_gpus", 0) or 0)
    standby = int(getattr(args, "rl_standby_gpus", 0) or 0)
    name = getattr(args, "rl_elastic_initial_config", None)
    counts = (len(slots["trainer"]), sum(len(e) for e in slots["rollout"]), len(slots["standby"]))
    if counts[1] != rollout or counts[2] != standby or sum(counts) != spec.total_gpus:
        raise ValueError(
            f"--rl-elastic-initial-config {name!r} placement is T{counts[0]} R{counts[1]} "
            f"S{counts[2]} but the launch asks for --rl-rollout-gpus {rollout} "
            f"--rl-standby-gpus {standby} on {spec} ({spec.total_gpus} GPUs)")
    topology = Topology(spec.num_nodes, spec.gpus_per_node)
    try:
        nodes, per_node, bundle_map = trainer_layout(slots, topology)
    except TopologyError as exc:
        raise ValueError(f"--rl-elastic-initial-config {name!r} placement on {spec}: {exc}") from None
    engine = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    tp = int(getattr(args, "tensor_parallel", 1) or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    cp = int(getattr(args, "context_parallel", 1) or 1)
    ep = int(getattr(args, "expert_parallel", 1) or 1)
    try:
        # Q1/Q3: only the in-node group tp*cp must stay on one node; PP and EP
        # groups may span nodes (M1: T2 = n0:0 + n1:0 with --pipeline-parallel 2).
        request = PlacementRequest(
            kind="fixed-partition", trainer_gpus=counts[0], rollout_gpus=rollout,
            gpus_per_engine=engine, standby_gpus=standby,
            gpus_per_node=spec.gpus_per_node if spec.num_nodes > 1 else None,
            model_parallel=tp * pp * cp, node_parallel=tp * cp, expert_parallel=ep,
            bundle_map=bundle_map, allow_cross_node_tp=rl_cross_node_switches(args)[0],
            allow_cross_node_engine_tp=rl_cross_node_switches(args)[1])
        shape = request.trainer_shape()
    except ValueError as exc:
        raise ValueError(f"--rl-elastic-initial-config {name!r} placement on {spec}: {exc}") from None
    if shape != (nodes, per_node):
        raise ValueError(
            f"--rl-elastic-initial-config {name!r}: launcher trainer shape {(nodes, per_node)} "
            f"disagrees with PlacementRequest.trainer_shape {shape}")
    return nodes, per_node, bundle_map


def rl_island_bundle_map_flag(args, spec) -> str:
    """`` --rl-island-bundle-map JSON`` for the learner when the cfg placement differs
    from the leading-bundle layout; "" otherwise (single node and whole-node
    trainer layouts are byte-identical to the pre-Q2 launcher)."""
    layout = rl_island_layout(args, spec)
    if layout is None:
        return ""
    from ..rl.engine.multinode import leading_bundle_map

    bundle_map = layout[2]
    counts = tuple(len(bundle_map[r]) for r in ("trainer", "rollout", "standby"))
    if bundle_map == leading_bundle_map(*counts):
        return ""
    return _rl_backend_module(args, "launch_flags").bundle_map_flag(bundle_map)


def rl_min_nodes(args, spec) -> int:
    """Minimum learner nodes (rl-multinode-island D8): one trainer model-parallel
    replica + one rollout engine + standby, or ``--rl-min-nodes-per-learner``
    when that is larger."""
    from ..rl.engine.multinode import TopologyError, min_nodes, trainer_replica_gpus

    explicit = int(getattr(args, "rl_min_nodes_per_learner", 0) or 0)
    if getattr(args, "rl_placement", "colocated") != "fixed-partition":
        return max(1, explicit)
    tp = int(getattr(args, "tensor_parallel", 1) or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    cp = int(getattr(args, "context_parallel", 1) or 1)
    ep = int(getattr(args, "expert_parallel", 1) or 1)
    etp = int(getattr(args, "expert_tensor_parallel", 1) or 1)  # the learner pins --expert-tensor-parallel-size 1
    engine = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    try:
        # Q1/Q3 ruling 2026-10-04: the smallest replica may span nodes (EP/PP
        # cross nodes); only tp*cp must stay inside a node. EP shares ranks with
        # TP x DP (trainer_replica_gpus), it is not a further multiplier.
        replica = trainer_replica_gpus({"tp": tp, "cp": cp, "pp": pp, "ep": ep, "etp": etp})
        cross_tp, cross_engine = rl_cross_node_switches(args)
        derived = min_nodes(trainer_min_gpus=replica, rollout_min_gpus=engine,
                            standby_gpus=int(getattr(args, "rl_standby_gpus", 0) or 0),
                            gpus_per_node=spec.gpus_per_node, node_parallel=tp * cp,
                            allow_cross_node_tp=cross_tp, allow_cross_node_engine=cross_engine)
    except TopologyError as exc:
        raise ValueError(f"recipe parallelism does not fit {spec}: {exc}") from None
    return max(derived, explicit)


# Per-node vCPU floor for RL (miles/ports) islands. Ray must fit every CPU
# actor next to the GPU workers; on a 1-GPU island miles requests roughly:
# InferenceController 1 + TrainerController 1 + RolloutExecutor 1 +
# registration reporter 1 (inference deploys) + MegatronTrainRayActor 0.4/GPU
# + RayWorkerManager helpers 0.2 each + CommandActor/router 0.001 each, i.e.
# ~4+ CPUs before sglang's own processes. A 4-vCPU node (AWS g5.xlarge /
# g6e.xlarge) left Ray with "Pending Demands: {'CPU': 1.0}" and the job hung
# (aws-g0-20261004c). 8+ keeps headroom; Nebius' smallest L40S (8 vCPU) and
# H100/H200 shapes (16/128 vCPU) already satisfy it, so they are unchanged.
# Applies per node, so multi-node islands get the same floor. An explicit
# --learner-cpus or --learner-instance-type wins. Pre-provisioned pools
# (ssh/kubernetes) and Modal size themselves and are left untouched.
RL_ISLAND_MIN_CPUS = "8+"

# Island roles (rl-eval-difficulty-buckets 5.1, D11.1). A "train" island trains,
# publishes and joins the merge pool; an "eval" island only serves inference for
# the eval store (yeto.cloud.modal_eval_island) and never joins the pool or
# submits a delta (yeto.rl.engine.island_ledger refuses it).
ISLAND_ROLE_ENV = "YETO_RL_ISLAND_ROLE"
ISLAND_ROLE_TRAIN = "train"
ISLAND_ROLE_EVAL = "eval"
ISLAND_ROLES = (ISLAND_ROLE_TRAIN, ISLAND_ROLE_EVAL)


def island_role_env(role: str) -> dict[str, str]:
    """Container env naming the island's role; unknown roles fail before any cloud call."""
    if role not in ISLAND_ROLES:
        raise ValueError(f"unknown island role {role!r} (known: {ISLAND_ROLES})")
    return {ISLAND_ROLE_ENV: role}


def rl_island_cpus(args, cloud: str | None = None):
    if getattr(args, "learner_cpus", None):
        return args.learner_cpus
    if getattr(args, "learner_instance_type", None):
        return None
    if cloud in ("ssh", "kubernetes", "modal"):
        return None
    return RL_ISLAND_MIN_CPUS
