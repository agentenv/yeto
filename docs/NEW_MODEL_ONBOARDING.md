# New model onboarding: async GRPO checklist

The order a customer follows to bring a new model to an async GRPO run
(`yeto run up`): a buffer trainer island plus a rollout sampler pool, sharing
one object store. Every step below can be checked on a CPU host with dry runs;
nothing is ordered until step 6. The Kimi-K3.1 specifics are in
[KIMI_K31_ASYNC_RL_GUIDE.md](KIMI_K31_ASYNC_RL_GUIDE.md); the examples under
`examples/k31/` are the starting files for any model.

## 1. Classify the model

| Model | Trainer | Effort |
|---|---|---|
| Architecture Megatron-Bridge AutoBridge and vLLM / SGLang read (Qwen2.5/3, Qwen3-MoE, Llama 3, DeepSeek-V3 layout) | `--rl-model-recipe generic` (default) | flags only |
| A known family at a new size (Nemotron-H, GLM-5.x, DeepSeek-V4) | the family's recipe | check the provider fields, one smoke run |
| A `model_type` transformers does not know | a provider view, a Miles plugin spec, a release mode and tests | weeks; not a flag change |

## 2. Hardware and backend

| Cards | Trainer backend | Sampler engine |
|---|---|---|
| NVIDIA | `--rl-backend miles` (default) or `verl` | `vllm-server` / `sglang-server`, `"platform": "cuda"` |
| AMD MI3xx | `miles` | `vllm-server` (ROCm image), `"platform": "rocm"` |
| Ascend 910x | `--rl-backend verl` only: Miles has no NPU path (the launch warns) | `vllm-server` with `"platform": "npu"` (vllm-ascend) |

For your own SSH or Ascend nodes, use `provider.kind: ssh`
(`yeto/launch/ssh_sampler_pool.py`): the nodes of an SSH node inventory are the
pool's slots, nothing is ordered or billed, and each start preflights the cards
and the workdir before it runs. `samplers[].provider` is therefore `nc`,
`modal`, `daytona`, `sky`, `ssh`, or a provider section with a `factory`.
Starting the samplers by hand with
`python -m yeto.rl.engine.engine_sampler --spec <spec.json>` still works.

## 3. Model weights

- Trainer: a Hub id with `--model-revision <40-hex commit>`. A local directory
  or an `s3://` URI is refused for RL; push private weights to a private Hub
  repository (optionally `--modal-model-volume` for a Modal volume copy).
- Samplers: `engine.server.base_dir` is the checkpoint path on the sampler
  node. NC pools copy `model_pull.src_key` to the node; a Modal pool reads
  `provider.model_volume`. `pool --dry-run` does not check that a Modal island
  will find `base_dir`.
- Size: weights are about 2 bytes per parameter (bf16); full-parameter
  training adds about 16 bytes per parameter of gradients and optimizer state.

## 4. Sampler spec

Copy `examples/k31/sampler-spec.nc-mi355x-vllm.json` and drop the K3.1-only
keys (`rollout_fields`, `expert_selection_sha256`, `k31_preset`, the K3
parsers, `checks.expected_expert_matrices`). `engine.server` requires:

| Field | Example for a dense bf16 model |
|---|---|
| `base_dir` | `/models/<name>` |
| `card_model` | `H100`, `L4`, `MI355X`, `910B` |
| `device_family` | `cuda`, `rocm`, `npu` (same as `platform`) |
| `engine_version` | `vllm@0.29.0` (recorded in the sampler identity) |
| `precision` | `bf16` |
| `quant_scheme` | `bf16` (trained, unquantized weights) |
| `topology` | `tp1`, `tp8` |
| `max_model_len` | `4096` |

Set `start_timeout_s` from the checkpoint size: the default 3600 s covers a
small model; K3.1 (about 1.5 TB MXFP4) uses 7200 s.

`sampling.n` must equal the trainer's `--n-samples-per-prompt` (the preflight
refuses a mismatch).

## 5. Trainer flags (buffer trainer)

The flags an async GRPO buffer trainer needs beyond the model and data:

```
--training-mode rl --rl-sample-role trainer --rl-single-island-no-sync
--controller local --trust-remote-code
--rl-checkpoint-store s3://<bucket>/runs/<run> --rl-object-store-endpoint <url>
--rl-max-policy-age 2 --rl-algorithm-spec examples/k31/algorithm-spec.async-age2-tis.json
--n-samples-per-prompt <sampling.n> --rollout-batch-size <groups>
--seq-len <tokens> --rollout-max-response-len <tokens>
--reward-function <module under the yeto tree>:<function>
```

- `--trust-remote-code` is required by the pinned Miles even for models
  without remote code.
- The reward module must live inside the yeto tree (for example
  `examples/my_reward.py`); only that tree reaches the islands.
- Set `--seq-len` and `--rollout-max-response-len`: the defaults (32768)
  are sized for long-context runs and exceed small cards.
- Parallelism is not inferred: pass `--tensor-parallel`, `--pipeline-parallel`
  (and `--expert-parallel` for MoE). When the estimate is over the card, the
  dry run prints the smallest TP x PP shape that fits.
- Modal trainer: `--modal-store-secret <secret>`. SSH trainer: set
  `YETO_RL_STORE_S3_ACCESS_KEY_ID` / `YETO_RL_STORE_S3_SECRET_ACCESS_KEY`
  (or `YETO_RL_STORE_S3_PROFILE`) on the launcher.
- `--rl-backend verl`: `--rl-round-cuts auto` (not `off`), and a TIS
  correction with `correction.lower` 0.0 (verl clamps the upper side only).

## 6. Dry runs, then launch

1. Sampler pool checks (the sampler spec through the sampler's own checks):
   `python -m yeto.launch.nc pool --config <pool.json> --dry-run`.
2. `yeto launch <trainer argv> --dry-run`: read the `[preflight]` lines
   (GPU memory estimate, suggestions) and the `memory_estimate` of each island
   in the JSON. Ascend cards are not in the memory table; size them by hand.
3. `yeto run up --config <run.json> --dry-run`: the two checks above plus the
   store and group-size cross-checks.
4. `yeto run up --config <run.json>`.

## 7. Large models (100B and above)

- A 100B dense model at TP 1 estimates about 3 TB per GPU at 32k tokens;
  start from the TP x PP the dry run suggests.
- Fixed limits to raise for large checkpoints: `--rl-distributed-timeout-minutes`
  (default 10), `engine.server.start_timeout_s` (default 3600 s),
  `first_group.timeout_s` (default 5400 s), `model_pull.timeout_s`.
- The memory estimate sums every expert of a MoE and does not model MLA or
  expert parallelism; treat its numbers as an upper bound for MoE.
