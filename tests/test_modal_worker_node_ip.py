"""Modal and SSH workers pin `ray start --node-ip-address` to their rank's address;
every other cloud keeps the worker argv byte-identical."""
from yeto import launcher


def test_modal_worker_pins_rank_ipv4():
    for cloud in ("modal", "ssh"):  # ssh: the inventory address of the rank, not Ray's route guess
        flag = launcher._worker_node_ip_flag(cloud)
        assert flag.startswith(' --node-ip-address="$(echo "$SKYPILOT_NODE_IPS"')
        assert '$((SKYPILOT_NODE_RANK + 1))p' in flag


def test_other_clouds_unchanged():
    for cloud in ("nebius", "aws", "verda", "runpod", "kubernetes"):
        assert launcher._worker_node_ip_flag(cloud) == ""


def test_flag_selects_this_rank_line():
    import subprocess
    out = subprocess.run(
        ["bash", "-c", 'SKYPILOT_NODE_IPS=$(printf "10.0.0.1\\n10.0.0.2"); SKYPILOT_NODE_RANK=1; echo'
         + launcher._worker_node_ip_flag("modal")], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "--node-ip-address=10.0.0.2"
