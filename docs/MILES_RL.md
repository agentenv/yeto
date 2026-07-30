# Miles RL v0

Miles RL runs reinforcement learning in several independent
[Miles](https://github.com/radixark/miles) islands and uses Yeto to turn their
complete local LoRA updates into one committed global policy. Miles owns the
rollout and GRPO loop inside each island; Yeto owns fleet lifecycle, the
cross-island synchronization boundary, recovery, and the final adapter.

This mode is not Decoupled DiLoCo. It is synchronous, fixed-roster LoRA
FedAvg: every global round waits for one exact-base result from every logical
island and averages them equally.

> **Status:** the core path is implemented and has passed real multi-GPU dense,
> MoE EP, two-island averaging, recovery, long-task, and 20-merge validation on
> eight A100 GPUs. The 24-hour soak and the operational gaps listed in
> [Validation status](#validation-status) remain outside that evidence.

## Why Miles is the RL runtime

An RL learner needs much more than a different loss function. It must keep a
rollout server and trainer coherent while it generates grouped trajectories,
scores responses, computes advantages, and updates the policy. Miles owns
those contracts around colocated SGLang and Megatron-Core. Yeto forwards
Miles' custom-generate and session-server entry points rather than implementing
a second agent or environment runtime.

Yeto therefore does not reproduce rollout, reward, or GRPO machinery in its
ordinary causal-LM learner. It adds one boundary around a pinned Miles local
round:

```text
                         committed global LoRA
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             Miles island 0              Miles island 1       … M islands
             SGLang rollout               SGLang rollout
             reward + GRPO                reward + GRPO
             Megatron step                Megatron step
                    │ complete local LoRA         │
                    └─────────────┬───────────────┘
                                  ▼
                         Yeto Rust syncer
                    fixed roster · f32 mean
                    checkpoint before broadcast
```

This division keeps Miles responsible for policy optimization and Yeto
responsible for distributed policy agreement.

## Supported contract

v0 deliberately supports one narrow model-parallel contract while allowing a
Miles island to use its full multi-node GPU allocation:

| dimension | supported value |
| --- | --- |
| model | causal language model |
| tuning | LoRA |
| local trainer | pinned Miles with Megatron-Core |
| rollout | colocated SGLang |
| island size | one or more nodes, one or more GPUs per node |
| parallelism | TP = PP = CP = 1; dense models use EP = 1; MoE EP is 1 or a divisor of the island world size; Miles supplies the DP layout |
| global state | one complete LoRA fragment |
| wire and merge math | f32, equal-weight AVG |
| membership | fixed logical island IDs `0..M-1`, `M >= 1` |

Model support is determined by the runtime tensor contract, not a Yeto model
family list. A model is usable when the pinned Miles/Megatron-Bridge stack can
load it, create LoRA, convert every trainable adapter tensor to standard PEFT
names and shapes, and apply the converted values without changing them.
Incompatibility fails during initialization instead of selecting a
model-specific fallback.

Every canonical policy carries the immutable base-model revision, a hash of
the effective LoRA configuration, the semantic tensor-layout hash, and its
policy version. Tensor specs include name, shape, fp32 dtype, and numel. The
runtime checks the complete identity between base, local, applied, and
broadcast states. The syncer checkpoint persists the layout hash, but not the
base revision or LoRA-config hash; export reconstructs those values from its
required explicit arguments and rejects a layout mismatch.
With EP>1, v0 therefore permits only replicated adapter targets (the normal
MoE `auto`/`attention` path), not LoRA on sharded expert weights.
The replicated path disables Megatron's distributed optimizer so every
trainer rank retains the complete fp32 LoRA masters and optimizer state needed
by the export/apply contract.
When `--expert-parallel` is omitted, dense models use EP=1 and MoE models use
the full island world size. An explicit MoE EP value must divide that world
size.

The RL path consumes `--lora-r` and `--lora-targets`. Its effective PEFT
configuration fixes alpha to the rank, dropout to zero, and bias to `none`;
the public RL launcher does not expose a separate alpha setting.

Not supported in v0:

- diffusion RL or full-parameter RL;
- stale, asynchronous, or partial-quorum global updates;
- multi-fragment synchronization, server momentum, RDA, HeLoCo, or blending;
- TP>1, PP>1, CP>1, or expert-sharded LoRA adapters;
- cross-island optimizer-state merging;
- Miles' experimental fault-tolerant trainer path;
- recovery of an unfinished trajectory or an unmerged local LoRA;
- RL-specific manifests, attestations, terminal markers, or wire messages.

The ordinary Yeto SFT and diffusion modes remain separate. Passing
`--training-mode rl` with a diffusion model is rejected.

## Current integration seams

The Miles checkout remains a clean detached checkout. At process startup the
current adapter installs Yeto export, apply, optimizer-step, and train-metric
methods on the pinned `MegatronTrainRayActor`, and invokes them through that
commit's non-FT `RayTrainGroup._broadcast` path. It also wraps the pinned
Megatron-Bridge provider/DDP construction, Miles train logging, and colocated
LoRA IPC completion in process memory. Yeto drives Miles rollout and training
primitives directly so the island-local result is not published to SGLang
before the global merge. There is currently no maintained Miles patch or
upstream train-loop synchronization hook. This private compatibility seam is
supported only for the pinned commit and must be revalidated if Miles is
changed.

The launcher selects strict RL behavior through the existing syncer's general
controls: `--max-base-lag 0`, `--learner-weight equal`, `--quorum M`, and
`--grace-ms 0`. It also supplies the one-fragment, unthrottled AVG settings
listed above and uses the existing `--checkpoint-every 1` control for a
durable cut before each broadcast. There is no RL-specific syncer mode or
wire message. Learners
send `c_steps=1` and `c_tokens=1`; `--learner-weight equal` independently
defines their global contribution. Protocol v4 is unchanged, and omitting the
two exact-base/equal-weight controls preserves SFT behavior.

## Global-round semantics

The RL flags describe the work at the Miles boundary:

- `M`: number of `--gpu` entries and therefore logical islands;
- `N`: Yeto's existing `--total-steps`;
- `G`: `--rollout-batch-size`, the complete GRPO groups per island round;
- `K`: `--n-samples-per-prompt`, the trajectories per group;
- local work: `--local-rl-rounds-per-sync 1` in v0.
- optimizer work: one Miles optimizer step per island round; v0 exposes no
  separate optimizer-step control.

`N`, `G`, and `K` must be positive. `G × K` must be divisible by every
island's Miles data-parallel size. `--over-sampling-batch-size` defaults to
`G` and may be raised to generate extra complete groups; it cannot be smaller
than `G`. Complete extras remain in the same-version queue after the selected
`G` groups are consumed.

The Miles argument mapping enables `--balance-data`. Multi-rank islands use
Miles' sequence-length partitioner to keep the total token count similar
across DP ranks while preserving equal sample counts. Yeto requires the
sample count to divide evenly across those ranks.

At committed version `v`, each island follows the same sequence:

1. Apply the complete global LoRA `theta_v` to the Megatron trainer and
   SGLang, then mark rollout policy version `v` active.
2. Generate exactly `G` groups of `K` terminal trajectories (completed or
   truncated at the configured limit). Every recorded rollout weight version
   must be `v`.
3. Run the one configured Miles GRPO training cycle without publishing the local
   result to SGLang.
4. Export the complete local LoRA `theta_i_v` and send
   `theta_i_v - theta_v` with base version `v`.
5. Wait for committed version `v + 1` before starting another local round.

The syncer accepts at most one result per logical island and waits for all
`M` results:

```text
theta_(v+1) = theta_v + mean(theta_i_v - theta_v)
            = mean(theta_i_v),  i = 0 .. M-1
```

All merge inputs have unit weight. Prompt counts, response lengths, and token
counts do not change an island's global weight.

Applying a new global policy removes the existing optimizer state for LoRA
parameters, including their moments and parameter step, then replaces its fp32
master parameters. It does not rebuild the optimizer or LR scheduler. Applying
committed policy `v` aligns the existing scheduler to
`v * num_steps_per_rollout * global_batch_size`: a replacement advances its
fresh scheduler to the committed progress, an in-process scheduler is already
there, and a scheduler ahead of the committed policy is rejected. This avoids
another warmup while preventing local Adam history from leaking across an
averaging boundary.

There is no fleet-wide "every island has applied" barrier. Each island has an
exact-base gate of its own, while the next merge still waits for the entire
fixed roster. A faster island may begin first, but it cannot commit without all
other islands at that same base version.

## Running a fleet

The launcher creates one Miles island for every `--gpu` entry. An entry may
describe multiple nodes and multiple GPUs per node. The launcher starts one
Ray head and joins the remaining island nodes as workers. A single entry is a
supported parity path; multiple entries enable fixed-roster averaging.
External learner slots are not supported.

The default production base image is pinned by digest:

```text
docker:radixark/miles@sha256:95b3afa9ee4313f5633e6ed3779c8276353cc8e24a2462e4f54ec0d5978fbae7
```

GPU validation used a local derivative of that exact base image with the
pinned Miles checkout and PEFT version preinstalled. The public launcher
performs those same source and PEFT setup steps on the digest-pinned base
image.

The Miles source itself is independently pinned to:

```text
https://github.com/radixark/miles
dfc66ff38752bfa2c5d325e0037ebc4b537c06de
```

The launcher checks out that commit as a detached HEAD and installs the
project's pinned PEFT version. At learner startup Yeto verifies the repository
origin, commit, clean worktree, and imported package path. Runtime
adaptation uses the compatibility seam described above without modifying the
checkout.

An illustrative two-island run is:

```bash
yeto launch \
  --training-mode rl \
  --gpu aws:8xa100@us-east-1,aws:8xa100@us-west-2 \
  --rl-runtime miles \
  --rl-image docker:radixark/miles@sha256:95b3afa9ee4313f5633e6ed3779c8276353cc8e24a2462e4f54ec0d5978fbae7 \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --model-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --data org/long-task-prompts \
  --data-revision 0123456789abcdef0123456789abcdef01234567 \
  --tuning lora \
  --lora-r 4 \
  --lora-targets attention \
  --total-steps 2 \
  --advantage-estimator grpo \
  --rollout-batch-size 16 \
  --over-sampling-batch-size 24 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len 512 \
  --local-rl-rounds-per-sync 1 \
  --rl-sync-preset strict-avg \
  --rl-policy-version strict \
  --rl-completed-groups-path ~/yeto-rl/island-checkpoint.pt \
  --reward-function project.rewards:score \
  --seq-len 2048 \
  --inner-lr 1e-4 \
  --seed 321 \
  --trust-remote-code \
  --controller local \
  --output ./rl-output
```

`--trust-remote-code` is required because the pinned Miles stack enables
remote-code loading in its internal model paths. Continue to pin model and
dataset revisions and enable it only for repositories you trust.

The example uses a local controller so `./rl-output` is fetched to the
submitting machine, which must remain available for the run. With the default
head controller, use a remote `--output` URI for automatic delivery or
retrieve the retained checkpoint from the head.

RL uses the public CLI above. `--total-steps` selects the global round count.
Generic DiLoCo controls such as `--fragments`, `--quorum`, `--pipeline`,
`--sync-interval-steps`, `--outer-lr`, `--outer-momentum`, and
`--wire-dtype` are not RL algorithm knobs; the launcher fixes their internal
values to the contract above.

`--rl-runtime`, `--advantage-estimator`, `--rl-sync-preset`, and
`--rl-policy-version` currently accept only `miles`, `grpo`, `strict-avg`, and
`strict`, respectively. Without `--experimental-rl-sync`, the strict preset
normalizes all generic sync values above. With the flag, the launcher preserves
and forwards working syncer controls such as quorum, grace, pacing, correction,
and outer LR/momentum. The bridge still requires `--fragments 1`,
`--pipeline 1`, `--merge-alpha 0`, and `--wire-dtype f32`; unsupported values
are rejected rather than accepted as ineffective overrides. Exact-base and
equal learner weighting remain explicit.

For RL, the launcher raises the effective `--seq-len` to at least
`--rollout-max-response-len`; this keeps the RL response default from being
silently clamped by the smaller generic SFT sequence default. Miles deducts the
tokenized prompt from that total rollout/trainer context, so the response value
is still a cap rather than a guaranteed length. At learner startup the actual
Megatron-Bridge provider must advertise a model context limit at least as large
as the effective sequence length.

### Prompt data

Each row must provide `messages`, or a string `prompt`/`input` that Yeto can
turn into a user message. `label`, `metadata`, and `tools` are preserved for
Miles and the reward implementation.

```json
{"messages":[{"role":"user","content":"Give a short proof."}],"label":"proof"}
```

RL v0 accepts revision-pinned Hugging Face dataset references. The prepared
prompt file is private to each island.

Without extra flags the learner uses Miles' default SGLang generation path, in
which one completion produces one trajectory. For environment-driven,
multi-turn or tool-use trajectories, pass an importable Miles generate callable
through `--custom-generate-function-path package.module.function`. Yeto also
forwards `--use-session-server`, optional `--session-server-ip`, and one port or
port range through `--session-server-port`. For a model that needs one of
Miles' model-specific incremental tokenizers, `--tito-model` is forwarded
unchanged and requires the session server. Yeto deliberately does not infer
this value from a model name. Miles continues to own session assembly,
tool/environment calls, terminal status, and weight-version records; Yeto does
not define a second trajectory format. Dataset `tools` and metadata are
preserved for that callable.

### Reward callable

`--reward-function` uses `package.module:function` syntax. The module must be
importable in the learner workdir. Its callable follows the pinned Miles
custom-reward API; a minimal implementation is:

```python
async def score(args, sample, **kwargs) -> float:
    del args, kwargs
    return 1.0 if "expected phrase" in sample.response else 0.0
```

GRPO needs reward variation within a group to produce a useful advantage.
A run can complete optimizer steps with constant rewards while learning
nothing, so inspect raw reward and advantage metrics during a shakedown.

Before provisioning, the launcher hashes the selected reward module source.
The learner verifies that digest before importing it. The source must be
inside the Yeto workdir synchronized by SkyPilot.

## Checkpoints, completion, and export

Version 0 and every successful global merge are atomically written to the
syncer's checkpoint before the new LoRA is broadcast. That file is the only
authoritative global RL state and includes the canonical layout hash.

Each Miles island also atomically stores its compatibility configuration,
local round, current policy version, rollout/reward statistics, and complete
unused group queue at `--rl-completed-groups-path`. The compatibility fields
cover model and dataset identifiers/revisions, topology, LoRA identity,
learning rate, sequence length, seed, group and oversampling sizes, optimizer
steps, reward digest, response limit, custom-generate/session mode, and the
selected TITO model. A process restart restores only complete groups whose
policy version and all persisted compatibility fields still match. For Spot RL tasks,
the launcher mounts that file's parent directory from a SkyPilot storage named
for `cluster_prefix + logical island ID`, with reconstruction sync enabled.
The stable per-island name survives a replacement VM without sharing queues
between islands; the non-persistent storage is removed when the task is finally
torn down. Spot paths must name a file inside an absolute or `~/` subdirectory.
On-demand tasks keep the ordinary local path. Groups selected for the current
training batch are removed from the queue; complete oversampling groups that
were not selected remain reusable only while that same global policy is
current. Unfinished groups, cross-version groups, and unmerged local LoRA
values are discarded.

With the local-controller example above, the launcher fetches the checkpoint
as:

```text
./rl-output/yeto-state.ckpt
```

The default head controller first retains the same file at
`~/yeto-output/yeto-state.ckpt` on the head, then delivers it when `--output`
is a remote URI.

When global version `N` is committed, the syncer sends its existing final
fragment protocol to every logical learner. Each learner applies the final
LoRA and acknowledges the cut before exiting normally.

Export the checkpoint with the same base model revision, rank, and target
selection used for training:

```bash
yeto-rl-export \
  --checkpoint ./rl-output/yeto-state.ckpt \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --model-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --lora-r 4 \
  --lora-targets attention \
  --output-dir ./adapter
```

The result contains standard `adapter_model.safetensors` and
`adapter_config.json` files and can be loaded with ordinary PEFT tooling. The
exporter reconstructs the expected tensor contract from the explicit model
arguments and requires its layout hash to match the checkpoint. It does not
infer configuration or roster membership from the syncer ledger.

## Failure and recovery

Recovery always starts from the most recent committed global checkpoint:

| failure | behavior |
| --- | --- |
| learner exits before its push is accepted | launcher restarts the same logical ID; it reapplies the committed policy and recomputes the round |
| learner exits after its push is accepted | syncer retains that logical learner's result and waits for the missing IDs |
| learner cannot recover | the run fails; the roster is never reduced |
| syncer process exits before checkpoint commit, with its disk retained | partial results are discarded; restart resumes the previous version and learners redo the round |
| syncer process exits after checkpoint commit but before broadcast, with its disk retained | restart loads and broadcasts the newly committed version |
| syncer VM or disk is lost | the current launcher has no durable mount for the global checkpoint, so automatic recovery is not available |
| learner exits while applying a global policy | replacement reapplies the complete committed policy |

A dead syncer connection makes an island exit at the next bridge health check.
If it drops while a synchronous Miles rollout/train call is in progress, that
local round may finish redundant computation, but it cannot begin another
round or become authoritative after restart. The launcher can restart the
syncer through its existing recovery path; each learner whose bridge exits is
restarted under the same logical ID rather than through a roster-wide restart
command. Duplicate computation is allowed, but duplicate merge is not.

Each island writes policy-apply, optimizer-reset, and `LocalRoundStats` JSONL;
the syncer writes roster, base-version, layout, merge, and responder metrics to
`~/yeto-output/yeto-tape.jsonl`. Miles group-task completion supplies peak
active groups, cancellations, and duration percentiles;
`Sample.non_generation_time` supplies tool wait; grouped raw rewards supply the
zero-variance ratio; and Miles' rank-zero train log supplies KL, ESS, and clip
fraction. The bridge records the actual protocol payload bytes, while each
canonical global apply records its policy hash.
The launcher does not currently enable a Miles or Yeto dashboard for these
records.

Mixed-version groups, rejected stale updates, layout mismatch, non-finite
deltas, or a policy hash mismatch after apply are deterministic strict
failures. The affected learner or syncer exits nonzero, and the fleet
controller terminates the whole run instead of relaunching the same
deterministic violation. Ordinary process, network, or spot failures remain
recoverable through the committed-checkpoint paths above.

## Runtime compatibility and diagnostics

The validated Miles/Transformer Engine stack cannot reliably use its
FlashAttention CUTE GQA kernel on A100. RL islands therefore select unfused
attention before Transformer Engine imports and propagate that choice into
the actual Megatron provider. This is a runtime-wide choice, not a model
family workaround.

Common startup and progress failures have distinct meanings:

- **PEFT import failure:** the learner did not run the current Miles setup,
  which installs the pinned `peft==0.20.0`.
- **`Operation creation failed` under `flash_attn/cute/pack_gqa.py`:** the
  process did not receive the RL launch environment or provider backend.
- **Miles revision/origin/dirty-tree error:** the runtime is not using the
  supported checkout; do not bypass this check or patch that tree in place.
- **PEFT/Megatron mapping mismatch:** the model is outside the currently
  supported tensor contract. Extend the generic upstream conversion path
  rather than adding a model-name branch in Yeto.
- **syncer waits with an incomplete roster:** a logical island is absent or
  still recovering. Fixed-roster mode never lowers quorum to make progress.
- **optimizer steps but negligible LoRA change:** first inspect reward
  variance, advantages, and the configured LR schedule.

## Intentional differences from INIT

| INIT plan | Current implementation | Assessment |
| --- | --- | --- |
| **Miles integration:** maintain a thin Miles branch with stable policy export, apply, and post-train synchronization hooks. | Keep the pinned upstream Miles checkout unchanged and adapt that exact version at runtime. | The required training and synchronization semantics are implemented and validated. A Miles upgrade still requires explicit compatibility revalidation. |
| **Global checkpoint recovery:** keep the authoritative policy recoverable across syncer failures, including replacement of its machine or disk. | Automatic recovery works while the syncer's disk is retained; losing that VM or disk also loses the checkpoint. | This does not change the RL algorithm, but it remains a production disaster-recovery gap. |
| **Monitoring:** connect the planned RL and synchronization metrics to a dashboard. | Emit the metrics to JSONL without enabling a dashboard. | The records are sufficient for validation and offline diagnosis, but not centralized live monitoring. |

## Validation status

The current source has automated coverage for multi-node task construction,
multi-rank actor results, DP rollout-shard collection, EP validation,
single-island and multi-island sync, canonical identity, completed-group
recovery, strict failures, provenance, checkpoint export, and the unchanged
SFT/diffusion defaults. The 2026-07-30 regression passed all 73 focused RL
tests, the full Python suite (`766 passed, 4 skipped`), `cargo fmt --check`, and
all 58 Rust tests.

Real validation ran on one GCP Spot VM with eight NVIDIA A100-SXM4-40GB GPUs.
It used the pinned Miles commit, PEFT 0.20.0, immutable model and dataset
revisions, real model generations, real rewards, and production Yeto learner
and Rust syncer paths. Observation hooks only captured tensors, tokens,
metrics, and fault windows.

- A one-island Qwen3-4B DP=8 run completed two global rounds. Each round
  trained on 16 trajectories and 8192 action tokens with nonconstant rewards.
  Local delta norms were `0.22248` and `0.10623`; both M=1 merges matched the
  saved local policy within `3.64e-12`.
- Two concurrent Qwen3-4B DP=4 islands used different prompt-token hashes.
  Their local delta norms were `0.223218` and `0.222651`; the committed f32
  policy exactly matched the offline mean, and both islands applied identical
  initial and final policies.
- `allenai/OLMoE-1B-7B-0125-Instruct` ran one EP=8 island with replicated
  attention LoRA, 16 trajectories, 4096 action tokens, nonconstant rewards,
  and a `0.10110` local delta. The merge error was zero. Standard PEFT loaded
  the exported adapter, produced finite logits with a nonzero adapter effect,
  and completed real generation.
- A direct pinned-Miles round and the production Yeto+Miles M=1 path produced
  identical sampled tokens and rewards; their LoRA tensors had max error
  `0.0`. The exact trainer-to-SGLang LoRA checksum checker passed throughout.
  Trainer-versus-rollout KL stayed around `6e-4` to `1.3e-3` before and after
  global applies rather than showing a material token-path mismatch.
- A 20-round Qwen run committed versions `1..20` exactly once, completing 160
  trajectories and 20,480 action tokens. Every round had a nonzero finite
  local delta, no mixed-version group was observed, and current-versus-rollout
  KL remained between `0.000329` and `0.001067`.
- Learners were killed during rollout, after local train, before push, after
  push, before broadcast, and after global apply. Every replacement completed,
  and each case produced one committed step. A separate syncer restart resumed
  the committed version and completed the next round.
- Completed-group recovery retained one real four-trajectory oversampling
  group across learner replacement and selected that group after restart. A
  separately replayed real trained delta was accepted once, while a stale
  exact-base update caused the strict connection to close.
- A Qwen3-4B session-server run completed two rounds and 16 real environment
  tasks. Ten trajectories made actual calculator calls and consumed their
  returned values; the second round had nonconstant rewards and a `0.10059`
  local update. Fourteen session traces had exact TITO reconstruction. The two
  remaining traces reached the 512-token response cap before a terminal token,
  were correctly marked truncated, and accounted for the reported 25% TITO
  structural mismatch in that round. Trainer-versus-rollout absolute logprob
  error remained below `0.01`.

The following boundaries remain unvalidated or intentionally excluded:

- the requested 24-hour soak was not run; the 20 consecutive merges are the
  bounded-duration stability evidence;
- no physical multi-node island or end-to-end SkyPilot provisioning and Spot
  VM replacement was exercised, although multi-node task construction is
  covered automatically;
- the syncer checkpoint still has no durable mount for syncer VM or disk loss;
- metrics remain JSONL-only and the launcher enables no dashboard.

The runtime-injection Miles seam described above remains an implementation
constraint, not a stable upstream interface.

## Extending v0 safely

The useful extension boundary is the observable contract, not a model-name
matrix. Preserve these invariants when changing the implementation:

1. The synchronized state is every trainable LoRA tensor, in standard PEFT
   naming and deterministic order, represented as contiguous CPU f32.
2. Every trajectory group records one actual global rollout version.
3. A local result is accepted only for its exact committed base and at most
   once per logical island.
4. A merge waits for all fixed logical IDs and uses equal weights.
5. The checkpoint is durable before any corresponding broadcast.
6. Applying global LoRA resets local optimizer history without rebuilding the
   optimizer or resetting scheduler progress.
7. Local trainer weights are not published to SGLang between global commits.

Adding a model should normally require no Yeto change: improve or select the
appropriate Miles/Megatron-Bridge/PEFT conversion and let the existing tensor
checks decide compatibility. Adding TP/PP, expert-sharded LoRA, multiple fragments, a
different aggregation algorithm, Miles FT actors, or Diffusion RL changes the
contract and requires a separate design rather than a compatibility branch.

For implementation navigation:

| area | responsibility |
| --- | --- |
| `yeto/rl/core.py` | canonical LoRA state and the single AVG layout |
| `yeto/rl/miles.py` | pinned Miles runtime, trainer/SGLang policy boundary |
| `yeto/rl/bridge.py` | exact-base island loop and protocol interaction |
| `yeto/rl/export.py` | committed checkpoint to PEFT adapter |
| `yeto/rl/learner.py` | island entry point and Miles argument mapping |
| `syncer/src/server.rs` | fixed roster and checkpoint-before-broadcast commit |

Run the focused and full regressions before a GPU shakedown:

```bash
python -m pytest -q \
  tests/test_rl_core.py tests/test_rl_export.py \
  tests/test_rl_integration.py tests/test_rl_launcher.py
python -m pytest -q
(cd syncer && cargo fmt --check && cargo test)
```
