# Kimi K3.1 examples

Configs and scripts for Kimi K3.1 async GRPO: a trainer island on Modal B300,
sampler islands on National Compute, and the sample buffer in an object store.

Start with the end-to-end guide:
[docs/KIMI_K31_ASYNC_RL_GUIDE.md](../../docs/KIMI_K31_ASYNC_RL_GUIDE.md).
It covers the run config, the pool config, the sampler spec, launch,
scaling, resume, the dashboard, and cost protection.

| File | Purpose |
|---|---|
| `launch-trainer-modal-b300.sh` | Trainer flags for the Modal B300 island |
| `sampler-spec.nc-mi355x-vllm.json` | Sampler spec for an NC MI355X node (vLLM) |
| `sampler-spec.nc-mi355x.json` | Sampler spec for an NC MI355X node |
| `sampler-spec.nc-mi355x-vllm-agentic-262k.json` | Agentic 262k sampler spec for an NC MI355X node (vLLM, `rl-precision`, 16 trials in flight); guide section 3.7 |
| `pool.nc-mi355x-agentic-262k.json` | NC pool for the agentic 262k sampler spec; guide section 3.7 |
| `run.agentic-262k.json` | Run config of the agentic 262k kit (6 x 8 B300 trainer, NC MI355X samplers): `yeto run up --config run.agentic-262k.json`; guide section 3.7 |
| `sampler-node-nc-mi355x.sh` | Sampler node script |
| `algorithm-spec.async-age2.json`, `algorithm-spec.async-age2-tis.json` | Algorithm specs for `--rl-max-policy-age 2` |
| `algorithm-spec.async-age12-tis.json` | Algorithm spec for `--rl-max-policy-age 12` (the 262k agentic run) |
| `sampler-spec.cpu-fake.json`, `convertible-spec.cpu-fake.json`, `launch-convertible-modal.sh` | CPU rehearsal and convertible-island examples |

## Health breakers

Run `yeto breaker` on a K3.1 run with `--sampler-tapes <sampler tape dir>` or
`--sampler-hardware amd`. Without either the sampler hardware is unknown, the
breaker fails closed to the `nvidia` limits (act 0.052), and the healthy MI355X
log-prob mismatch (0.076-0.099) trips. See [docs/BREAKER.md](../../docs/BREAKER.md).
