## Context

Facts used (sources in `infra-drafts/SESSION25/GOAL.md` §2):
- Local cluster: 4 nodes x 8 Ascend 910 cards; 64 GiB HBM version arrived 2026-10-16; sub-model and host memory not verified. Access: VPN (UniVPN on the submitting server) then a jump host. The user confirmed that the VPN routes only 172.28.11.0/24, and Modal and AWS stay reachable from the server.
- PPU machines: reached by SSH; count, model, and host memory not known yet.
- `yeto/gpu_spec.py` already parses `ssh:1x8x910b4`; `yeto/gpu_spec.py` has `assert_npu_cards` for `npu-smi info`.
- `yeto/rl/ssh_harness.py` refuses any accelerator other than H200 (around line 1077).
- `yeto/launch/topology.py` writes only NCCL/GLOO variables and forces `NCCL_IB_DISABLE=1` outside Modal.

## Decisions

### D1. One inventory for local and hand-opened cloud nodes
A node is a node: the inventory does not know if it is on-premises or a rented VM. Price, spot flag, and owner are inventory fields. This keeps the core neutral; provider differences stay in the file.

### D2. Device family is explicit and checked
The inventory states `device_family`. Preflight reads the card tool for that family and compares model, count, and HBM. There is no default family. Unknown models fail with the model string in the error.

### D3. Card-tool parsers are adapters
Each family has a parser that returns the same record (index, model, HBM GiB, health). Parsers are tested with recorded output. The output formats change with driver versions (`npu-smi` with driver 25.x); the recorded samples carry the driver version.

### D4. SSH executor reuses the harness
The executor wraps `ssh_harness` operations and adds `ProxyJump` from the inventory. It does not write to `~/.ssh/config`. Commands run inside the node's container when the inventory gives an image; nodes without docker or root (as on ModelArts) run in a prepared rootfs, recorded as `runtime: rootfs`.

### D5. Network environment from fabric
`topology.py` gains a function from `(device_family, fabric, interface)` to environment variables. The HCCL variable names (`HCCL_SOCKET_IFNAME`, `HCCL_IF_IP`, `HCCL_CONNECT_TIMEOUT`) are not verified against CANN documentation; task N1.4 checks them before use.

### Non-goals and rejected alternatives
- Rejected: SkyPilot ssh pool (k3s), because it needs root and changes the nodes.
- Rejected: a Huawei Cloud API provider in this change; ModelArts notebooks stay manual.

## Risks
- The VPN routes only 172.28.11.0/24 (user check, 2026-10-16). Preflight still checks that the nodes and the configured store (OSS) are reachable from the head, because the route can change when the VPN reconnects.
- A jump host that closes long SSH sessions would stop log streaming; the executor polls instead of holding a session.
