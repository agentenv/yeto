"""Node profiles (unified-sampler-pool 2.1): unified device families, the profile
keys of an inventory entry, and the device flags / visible-card variable each family
derives."""

from __future__ import annotations

import pytest

from yeto import gpu_spec
from yeto.launch import node_inventory as inv
from yeto.launch.node_inventory import InventoryError, NodeProfile, parse_node, parse_profile


def raw(**kw):
    base = {"name": "n0", "host": "ops@n0", "device_family": "cuda", "cpu_arch": "x86_64", "card": "H200",
            "cards_per_node": 8, "hbm_gib": 141, "host_memory_gib": 1024}
    return {**base, **kw}


ASCEND = dict(device_family="npu", cpu_arch="aarch64", card="910B4", hbm_gib=64, runtime="rootfs")
ROCM = dict(device_family="rocm", card="MI355X", hbm_gib=288)


# -- family names


@pytest.mark.parametrize("name, unified, launcher", [
    ("cuda", "cuda", "cuda"), ("nvidia", "cuda", "cuda"), ("GPU", "cuda", "cuda"),
    ("rocm", "rocm", "rocm"), ("amd", "rocm", "rocm"),
    ("npu", "ascend", "npu"), ("ascend", "ascend", "npu"),
    ("ppu", "ppu", "ppu"), ("thead", "ppu", "ppu"),
])
def test_both_old_spellings_resolve_to_one_family(name, unified, launcher):
    assert gpu_spec.unified_family(name) == unified
    assert gpu_spec.launcher_family(name) == launcher


def test_the_aliases_change_no_launcher_value():
    assert gpu_spec.DEVICE_FAMILIES == ("cuda", "npu", "ppu", "rocm")
    assert gpu_spec.hw_family("ascend").family == "npu"  # as before
    with pytest.raises(gpu_spec.UnknownFamily):
        gpu_spec.hw_family("gpu")  # the extra alias lives in unified_family only
    for bad in ("cpu", "tpu", ""):
        with pytest.raises(gpu_spec.UnknownFamily):
            gpu_spec.unified_family(bad)


def test_an_inventory_may_spell_ascend_and_keeps_the_launcher_name():
    node = parse_node(raw(**{**ASCEND, "device_family": "ascend"}), index=0)
    assert node.device_family == "npu"
    with pytest.raises(InventoryError, match="device_family 'tpu' is unknown"):
        parse_node(raw(device_family="tpu"), index=0)


# -- the profile keys


def test_parse_node_accepts_and_ignores_the_profile_keys():
    entry = raw(card_arch="sm90", container="podman", paths={"work": "/srv/w", "models": "/srv/m"},
                model_source="store", reach="direct")
    with pytest.warns(UserWarning, match="are node profile keys"):
        node = parse_node(entry, index=0)  # the trainer island's reader says it ignores them
    assert type(node) is inv.Node
    profile = parse_profile(entry, index=0)
    assert isinstance(profile, inv.Node) and profile.container == "podman"
    assert profile.paths == inv.NodePaths(work="/srv/w", models="/srv/m")
    assert profile.model_source == "store" and profile.reach == inv.Reach("direct")
    with pytest.raises(InventoryError, match="container 'lxc' is unknown"):
        parse_profile(raw(container="lxc"), index=0)
    with pytest.raises(InventoryError, match="model_source"):
        parse_profile(raw(model_source="ftp"), index=0)
    with pytest.raises(InventoryError, match="paths"):
        parse_profile(raw(paths={"tmp": "/x"}), index=0)
    with pytest.raises(InventoryError, match="relay needs via"):
        parse_profile(raw(reach={"mode": "relay"}), index=0)


def test_profile_defaults_are_derived_from_the_node():
    p = parse_profile(raw(proxy_jump="ops@bastion", scratch="/mnt/nvme/yeto"), index=0)
    assert (p.family, p.card_arch, p.container) == ("cuda", "sm90", "docker")
    assert p.reach == inv.Reach("jump", "ops@bastion") and not inv.Reach("outbound").inbound
    assert p.paths.work == "/mnt/nvme/yeto"
    a = parse_profile(raw(**ASCEND), index=0)
    assert (a.family, a.device_family, a.card_arch, a.container) == ("ascend", "npu", "ascend910b", "process")
    r = parse_profile(raw(**ROCM), index=0)
    assert (r.family, r.card_arch) == ("rocm", "gfx950")
    assert parse_profile(raw(card="B300", hbm_gib=288), index=0).card_arch == "sm103"
    assert parse_profile(raw(card="B300", hbm_gib=288, card_arch="sm100"), index=0).card_arch == "sm100"


def test_a_profile_file_loads_and_the_trainer_reader_still_takes_it(tmp_path):
    path = tmp_path / "inv.yaml"
    path.write_text("nodes:\n  - {name: a, host: a, device_family: rocm, cpu_arch: x86_64, card: MI355X,\n"
                    "     cards_per_node: 8, hbm_gib: 288, host_memory_gib: 2048, container: podman,\n"
                    "     reach: {mode: relay, via: relay.example:2222}, model_source: preset}\n")
    (p,) = inv.load_profiles(path)
    assert p.container == "podman" and p.reach == inv.Reach("relay", "relay.example:2222") and p.reach.inbound
    with pytest.warns(UserWarning, match="are node profile keys"):
        (node,) = inv.load_inventory(path)  # the trainer's reader: a Node, the profile keys ignored
    assert type(node) is inv.Node and node.card == "MI355X"


def test_for_card_builds_a_rented_node_without_ssh():
    p = NodeProfile.for_card("modal-0", "l4", 1, hbm_gib=24, container="modal-fn", reach=inv.Reach("function"))
    assert (p.device_family, p.card, p.card_arch, p.host) == ("cuda", "L4", "sm89", "")
    assert p.devices().args == () and p.devices().env == {}  # the platform hands the function its card
    assert p.summary() == {"family": "cuda", "card": "L4", "card_arch": "sm89", "cards": 1, "hbm_gib": 24.0,
                           "cpu_arch": "x86_64", "container": "modal-fn", "reach": "function", "model_source": None}
    assert NodeProfile.from_node(p) is p


# -- devices per family


def test_the_visible_variable_is_the_hw_family_table():
    for family, card, hbm in (("cuda", "H200", 141), ("rocm", "MI355X", 288), ("npu", "910B4", 64),
                              ("ppu", "810E", 96)):
        p = NodeProfile.for_card("n", card, 8, hbm_gib=hbm)
        assert p.visible_devices_env == gpu_spec.hw_family(family).visible_devices_env
        assert p.family_env() == {gpu_spec.DEVICE_FAMILY_ENV: family}


def test_cuda_gets_gpus_or_cdi_and_never_rocm_devices():
    p = NodeProfile.for_card("h", "H200", 8, hbm_gib=141)
    whole = p.devices()
    assert whole.args == ("--gpus", "all") and whole.env == {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}
    part = p.devices([2, 3])
    assert part.args == ("--gpus", '"device=2,3"')  # docker's CSV value needs the inner quotes
    assert part.env == {"CUDA_VISIBLE_DEVICES": "0,1"}  # the container sees only those, numbered from 0
    cdi = p.devices([5], engine="podman")
    assert cdi.args == ("--device", "nvidia.com/gpu=5") and cdi.env == {"CUDA_VISIBLE_DEVICES": "0"}
    assert p.devices(engine="podman").args == ("--device", "nvidia.com/gpu=all")
    flat = " ".join(whole.args + part.args + cdi.args)
    assert "/dev/kfd" not in flat and "/dev/dri" not in flat and "HIP_VISIBLE_DEVICES" not in str(part.env)
    assert p.devices([2, 3], engine="process") == inv.DeviceSetup((), {"CUDA_VISIBLE_DEVICES": "2,3"})
    assert p.host_paths() == () and p.required_host_paths() == ()


def test_rocm_gets_the_nc_routes_flags():
    from yeto.launch.nc.ssh_sampler import ROCM_DOCKER_ARGS

    p = NodeProfile.for_card("r", "MI355X", 8, hbm_gib=288)
    s = p.devices([1, 6])
    assert s.args == ROCM_DOCKER_ARGS  # the very tuple: kfd, dri, video, seccomp, SYS_PTRACE, memlock, shm
    for flag in ("/dev/kfd", "/dev/dri", "video", "seccomp=unconfined", "SYS_PTRACE", "memlock=-1", "64g"):
        assert flag in s.args
    assert s.env == {"HIP_VISIBLE_DEVICES": "1,6"}  # every render node is passed: host numbers
    assert p.required_host_paths() == ("/dev/kfd", "/dev/dri")
    assert NodeProfile.for_card("r", "MI355X", 8, hbm_gib=288, container="process").required_host_paths() == ()


def test_the_customer_nc_mi355x_docker_command_is_unchanged_through_a_profile():
    """Shadow: an NC lease's node as a NodeProfile renders the NC route's docker run word for word,
    and the derived rocm flags of a bare MI355X profile are the ones that node carries."""
    from yeto.launch.nc import ssh_sampler
    from yeto.launch.nc.client import NodeLease

    lease = NodeLease(account=1, ticket="t-1", ip="192.0.2.7", user="ubuntu", host_key="k", granted_at=None,
                      lease_until=None, gpus=8, rate_ucr_per_gpu_hour=None)
    node = ssh_sampler.node_of(lease)
    profile = inv.profile_of(node)
    kw = dict(name="yeto-sampler-k31-0", command=["python", "-m", "yeto.rl.engine.engine_sampler"],
              env={"HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "PYTHONPATH": "/yeto"}, yeto_dir="/nvme/yeto/abc",
              env_file="/nvme/work/.store.env", detach=True)
    assert ssh_sampler.docker_run_argv(profile, **kw) == ssh_sampler.docker_run_argv(node, **kw)
    assert profile.devices().args == tuple(node.docker_args) == ssh_sampler.ROCM_DOCKER_ARGS
    bare = NodeProfile.for_card("x", "MI355X", 8, hbm_gib=288)
    assert bare.devices().args == tuple(node.docker_args)


def test_ascend_gets_its_cards_and_only_the_host_paths_the_node_has():
    p = parse_profile(raw(**ASCEND), index=0)
    s = p.devices([0], engine="podman", host_paths=["/dev/davinci_manager", "/usr/local/Ascend/driver"])
    assert s.args == ("--device", "/dev/davinci0", "--device", "/dev/davinci_manager",
                      "-v", "/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro")
    assert s.env == {"ASCEND_RT_VISIBLE_DEVICES": "0"}
    assert set(p.host_paths()) == set(inv.NPU_HOST_DEVICES + inv.NPU_HOST_MOUNTS)


def test_inventory_docker_args_replace_the_derived_flags():
    p = parse_profile(raw(runtime="docker", docker_args=["--gpus", "all", "--device", "/dev/infiniband"]), index=0)
    s = p.devices([2, 3])
    assert s.args == ("--gpus", "all", "--device", "/dev/infiniband")
    assert s.env == {"CUDA_VISIBLE_DEVICES": "2,3"}  # the site's flags pass every card: host numbers


def test_a_profile_is_a_node_for_every_node_consumer():
    p = parse_profile(raw(proxy_jump="ops@b"), index=0)
    assert p.ssh_options() == ["-o", "ProxyJump=ops@b"]
    assert p.compat_group == parse_node(raw(), index=0).compat_group
    row = p.record()
    assert row["name"] == "n0" and row["family"] == "cuda" and row["card_arch"] == "sm90"


def test_profile_readers_do_not_warn_and_reach_jump_dials_through_its_host(recwarn):
    p = parse_profile(raw(reach={"mode": "jump", "via": "ops@bastion"}), index=0)
    assert p.proxy_jump == "ops@bastion" and p.ssh_options() == ["-o", "ProxyJump=ops@bastion"]
    assert p.reach == inv.Reach("jump", "ops@bastion")
    assert parse_profile(raw(proxy_jump="ops@b", reach="jump"), index=0).reach == inv.Reach("jump", "ops@b")
    assert not [w for w in recwarn if "profile keys" in str(w.message)]
    with pytest.raises(InventoryError, match="two jump hosts"):
        parse_profile(raw(proxy_jump="ops@a", reach={"mode": "jump", "via": "ops@b"}), index=0)


def test_ascend_profile_preflights_with_npu_smi():
    a = parse_profile(raw(**ASCEND), index=0)
    assert a.preflight_commands()[0] == ("npu-smi", "info")
    assert a.hw.smi == "npu-smi" and a.visible_devices_env == "ASCEND_RT_VISIBLE_DEVICES"
    r = NodeProfile.for_card("r", "MI355X", 8, hbm_gib=288)
    assert r.preflight_commands()[0][0] == "amd-smi" and r.preflight_commands()[-1][0] == "rocm-smi"
