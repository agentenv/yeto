# Tasks

Codes follow `infra-drafts/SESSION25/GOAL.md` §3 (v2). CPU-first tasks need no NPU or PPU.(v2). CPU-first tasks need no NPU or PPU. Effort is in sub-agent person-days and is an estimate.

## N1. SSH node pool

- [ ] N1.1 Inventory schema, parser, and validation; device family field; host memory field; unknown model refused (design D1, D2). Acceptance: CPU tests. GPU: no. Estimate: 1–1.5 d
- [ ] N1.2 Card-tool parsers for `nvidia-smi`, `npu-smi`, `ppu-smi` with HBM (design D3). Acceptance: CPU tests on recorded output; one check on a real local node. GPU: parsers no; check yes (one node). Estimate: 2 d
- [ ] N1.3 SSH executor with `ProxyJump`; remove the H200 pin in `ssh_harness.py`; rootfs runtime (design D4). Acceptance: CPU tests; smoke on one local node (start, status, stop, collect). GPU: yes (one local node). Estimate: 2–3 d
- [ ] N1.4 Network environment by device family and fabric in `topology.py`; check HCCL variable names against CANN documentation first (design D5). Acceptance: CPU tests; existing NVIDIA argv unchanged byte for byte. GPU: no. Estimate: 1 d
- [ ] N1.5 Head reachability preflight: from the head, check every node and the configured store endpoint (design Risks). Acceptance: CPU test with fakes; one run with the VPN up. GPU: no. Estimate: 0.5 d
