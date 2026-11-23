# New model onboarding

Two run shapes, two checklists: [Part I](#part-i-async-grpo-checklist) brings a
model to an async GRPO run (`yeto run up`), [Part II](#part-ii-decoupled-diloco)
to a decoupled DiLoCo run (`yeto launch` with several islands and a syncer).
Both start from the same dry run, `yeto model check` (Part II, step 2).

## Part I: async GRPO checklist

The order a customer follows to bring a new model to an async GRPO run
(`yeto run up`): a buffer trainer island plus a rollout sampler pool, sharing
one object store. Every step below can be checked on a CPU host with dry runs;
nothing is ordered until step 6. The Kimi-K3.1 specifics are in
[KIMI_K31_ASYNC_RL_GUIDE.md](KIMI_K31_ASYNC_RL_GUIDE.md); the examples under
`examples/k31/` are the starting files for any model.

### 1. Classify the model

| Model | Trainer | Effort |
|---|---|---|
| Architecture Megatron-Bridge AutoBridge and vLLM / SGLang read (Qwen2.5/3, Qwen3-MoE, Llama 3, DeepSeek-V3 layout) | `--rl-model-recipe generic` (default) | flags only |
| A known family at a new size (Nemotron-H, GLM-5.x, DeepSeek-V4) | the family's recipe | check the provider fields, one smoke run |
| A `model_type` transformers does not know | a provider view, a Miles plugin spec, a release mode and tests | weeks; not a flag change |

### 2. Hardware and backend

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

### 3. Model weights

- Trainer: a Hub id with `--model-revision <40-hex commit>`. A local directory
  or an `s3://` URI is refused for RL; push private weights to a private Hub
  repository (optionally `--modal-model-volume` for a Modal volume copy).
- Samplers: `engine.server.base_dir` is the checkpoint path on the sampler
  node. NC pools copy `model_pull.src_key` to the node; a Modal pool reads
  `provider.model_volume`. `pool --dry-run` does not check that a Modal island
  will find `base_dir`.
- Size: weights are about 2 bytes per parameter (bf16); full-parameter
  training adds about 16 bytes per parameter of gradients and optimizer state.

### 4. Sampler spec

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

### 5. Trainer flags (buffer trainer)

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

### 6. Dry runs, then launch

1. Sampler pool checks (the sampler spec through the sampler's own checks):
   `python -m yeto.launch.nc pool --config <pool.json> --dry-run`.
2. `yeto launch <trainer argv> --dry-run`: read the `[preflight]` lines
   (GPU memory estimate, suggestions) and the `memory_estimate` of each island
   in the JSON. Ascend cards are not in the memory table; size them by hand.
3. `yeto run up --config <run.json> --dry-run`: the two checks above plus the
   store and group-size cross-checks.
4. `yeto run up --config <run.json>`.

### 7. Large models (100B and above)

- A 100B dense model at TP 1 estimates about 3 TB per GPU at 32k tokens;
  start from the TP x PP the dry run suggests.
- Fixed limits to raise for large checkpoints: `--rl-distributed-timeout-minutes`
  (default 10), `engine.server.start_timeout_s` (default 3600 s),
  `first_group.timeout_s` (default 5400 s), `model_pull.timeout_s`.
- The memory estimate sums every expert of a MoE and does not model MLA or
  expert parallelism; treat its numbers as an upper bound for MoE.

## Part II: decoupled DiLoCo

Several islands each run GRPO on their own; the syncer merges their weights in
fragments (`P`), each island keeps training `H` local rounds per sync, and up
to `tau` fragment rounds are in flight at once. `--rl-sync-preset decoupled`
names this for every trainable form; the form decides which sync stack runs it
(the launch prints `[sync-route] ...` when it reroutes):

| Form | Stack under `--rl-sync-preset decoupled` | What is synced |
|---|---|---|
| LoRA | LoRA fragments (`decoupled`; `strict-avg` is one fragment per round) | the canonical LoRA layout, bin-packed into `--fragments P` (2 <= P <= LoRA tensors) |
| Full parameters | full-parameter asynchronous rounds (the same as `--rl-sync-preset full-parameter --rl-sync-full-parameter-rounds async`) | every parameter, owner-affine fragments of at most 2 GiB in FP32 |
| Partial experts (MoE) | full-parameter asynchronous rounds, with the recipe's selection | the parameters the recipe trains (the selected routed experts and the trainable dense part) |

`--tuning full` with `strict-avg` is refused at launch: that preset merges
LoRA. Partial experts exist for two recipes today:
`--rl-model-recipe kimi_k31 --tuning full --rl-expert-selection <file>` and
`--rl-model-recipe deepseek-v4-flash --expert-full-count N`.

### 1. Checklist

- [ ] `config.json` reachable (Hub id, local directory or file); weights are not needed until launch.
- [ ] The model contract derives: `yeto model check` shows `contract ok`
      (layers, attention type, MoE layers and expert count). A `missing:` line
      names the field a recipe must supply.
- [ ] The trainer stack loads it: on Miles, Megatron-Bridge has a bridge for the
      architecture; full-parameter and partial-expert runs also need a Miles
      Megatron->HF converter (LoRA publishes through the bridge). On verl the
      FSDP2 trainer reads the HF model; TP/PP/EP > 1 moves to the verl Megatron
      island (full parameters only).
- [ ] The parallel shape divides the model: TP divides the attention heads and
      the KV heads (GQA), EP divides the routed experts and the trainer GPUs of
      an island, TP x PP divides the trainer GPUs.
- [ ] The LoRA targets exist and are adaptable: `attention` everywhere;
      `all-linear` on MoE adds per-expert LoRA only where the experts are
      separate modules (one adapter per expert); under EP > 1 the experts must
      be grouped-GEMM, BF16 and without capacity padding.
- [ ] The layout agrees across islands: LoRA layouts follow from the config;
      full-parameter layouts are bound by the syncer (`--rl-parameter-layout-sha256 auto`, Miles)
      or planned offline (verl).
- [ ] A small model of the same family runs two islands end to end (step 4).

### 2. `yeto model check`

`yeto model check <model> --form lora|full|partial-moe --preset <preset>
--backend miles|verl` reads `config.json` (fetched from the Hub when it is not
on disk; never weights) and prints, per item, `ok` / `warn` / `FAIL` with the
reason, a suggestion and where the rule lives: `launcher` (`yeto launch`
refuses it), `island` (the island refuses it at start), `upstream` (the pinned
Megatron-Bridge / Miles cannot run it) or `estimate`. It then prints the
recommended `P` / `tau` / `H`, a GPU-memory estimate (the launch preflight's
model for dense models, the model registry's smallest trainer for MoE), the
`yeto launch` flags of the cell and the templates a new model needs. The exit
code is 0 when the cell is runnable, 1 when not, 2 when the config cannot be
read.

| Option | Default | Meaning |
|---|---|---|
| `--form` | `lora` | `lora`, `full`, `partial-moe` |
| `--preset` | `decoupled` | `strict-avg`, `decoupled`, `full-parameter`, `single-island` |
| `--backend` | `miles` | `miles`, `verl` |
| `--recipe` | the registry family's | `--rl-model-recipe` to check against |
| `--tensor-parallel`, `--pipeline-parallel`, `--expert-parallel` | 1, 1, recommended | the trainer shape |
| `--lora-targets`, `--lora-rank` | `attention`, 16 | LoRA form only |
| `--islands`, `--gpus-per-island`, `--card` | 2, 8, `H100` | the fleet |
| `--seq-len` | 4096 | tokens per micro batch, for the memory estimate and the context check |
| `--index PATH\|auto` | none | `model.safetensors.index.json` to verify the routed-expert names (`auto` fetches it) |
| `--no-fetch`, `--json` | | disk only; one JSON object |

Three examples:

```text
$ yeto model check Qwen/Qwen3-0.6B --form full --preset full-parameter
verdict  RUNNABLE
  [ok  ] backend       miles loads Qwen3ForCausalLM (converter qwen2)  (upstream)
  [ok  ] layout        Miles full-parameter layout: --rl-parameter-layout-sha256 auto ...
sync     fragments=4, pipeline=2, local_rl_rounds_per_sync=4, fragment_bytes=596041728
flags    --rl-model-recipe generic --tuning full ... --rl-sync-preset full-parameter
         --rl-sync-full-parameter-rounds async --fragments 4 --local-rl-rounds-per-sync 4
         --rl-parameter-layout-sha256 auto --rl-placement fixed-partition --rl-rollout-gpus 4

$ yeto model check mistralai/Mixtral-8x7B-v0.1 --form full --preset full-parameter
verdict  NOT RUNNABLE
  [FAIL] backend       Megatron-Bridge a0c6d6e0 has no bridge for MixtralForCausalLM ...  (upstream)
         -> --rl-backend verl (FSDP2 reads the HF model), or a provider view and Miles recipe
  [FAIL] backend       no Miles Megatron->HF converter for MixtralConfig ...  (upstream)
template ...

$ yeto model check <kimi-k3.1 config> --form lora --preset decoupled
verdict  NOT RUNNABLE
  [FAIL] form          kimi_k31 LoRA behind the syncer runs strict-avg (got decoupled) ...  (launcher)
         -> --rl-sync-preset strict-avg
```

The `launcher` and run-config verdicts are tested against `yeto launch` and
the island's run-config resolution for every fixture model, form, preset and
backend (`tests/test_model_check_matrix.py`), so the check and the launcher
cannot drift apart. The upstream rules are read from the pinned sources
(`yeto/rl/model_registry/upstream.py`) and re-read when the Miles pin moves.

### 3. Recipe and mapping templates

`yeto model check` prints the template a failing item needs:

| Item | What to write |
|---|---|
| no Megatron-Bridge bridge | a provider view under `yeto/rl/adapters/miles/models/<family>/` (model it on `deepseek_v41/provider.py`) and a `--rl-model-recipe <family>` entry |
| no Miles converter (full / partial) | a Megatron->HF converter in the yeto Miles adapter, selected by the recipe's `--model-name` |
| partial experts on a new MoE | a `PartialExpertModelConfig` registration (in-place permuted experts) and a selection from `python -m yeto.rl.adapters.miles.models.partial_experts.selection` |
| checkpoint expert names | a name mapping in the recipe's provider view |
| unknown family | a `FamilyEntry` in `yeto/rl/model_registry/families.py`, so `yeto adapt` and the check name it |

Never patch the pinned upstream trees; the adapters carry the model.

### 4. Small-model validation

Before the target size, run the same family small (for example Qwen3-0.6B for
Qwen3, OLMoE-1B-7B for a generic MoE) with the flags the check printed:

1. `yeto launch <flags> --gpu modal:1x8xh100,modal:1x8xh100 --dry-run`: the
   preflight memory lines and the island plan; with
   `--rl-parameter-layout-sha256 auto` the dry run prints that the syncer binds
   the layout at the first HELLO.
2. A short run (a few global rounds). Check that every island's first HELLO is
   admitted, that full-parameter islands print `[yeto-layout-auto] {...}` with
   the same `parameter_layout_sha256`, and that fragments complete in the
   syncer log.
3. Pin the recorded hash (`--rl-parameter-layout-sha256 <hex>`) for the real
   run if you want the islands to refuse a changed layout before the syncer
   does.

### 5. Hardware

- LoRA on a model that fits one island: two or more 8-GPU islands, TP / PP from
  the memory line; MoE LoRA with routed experts under EP > 1 needs grouped-GEMM
  experts (the island refuses it otherwise).
- Full parameters: Miles islands run `--rl-placement fixed-partition` (half the
  GPUs train, half roll out, by default in the printed flags); the fragment
  count grows with the model (`P >= 1.25 x FP32 bytes / 2 GiB`).
- MoE above one island's memory: the check's `smallest trainer` line is the
  world size (and TP / PP / EP) that holds the model state; size each island to
  at least that, and treat the number as an upper bound.
- verl: TP / PP / EP > 1 runs only full parameters on the verl Megatron island,
  PP 1 for full-parameter sync.

