# Clouds

Which clouds `yeto shape` can plan across and `yeto launch` can provision
on, where each one's credentials live, what "region" means to it, and
which behaviours have been verified on real machines. Yeto keeps no
credentials of its own: every cloud's own CLI writes them to a standard
place on the **submitting machine**, SkyPilot and Yeto read them there,
and in head controller mode the launcher copies only the files of the
clouds a fleet actually touches onto the head VM. Learner islands never
receive cloud credentials (only the Hugging Face token, and W&B /
CyberGym keys when those features are on).

On Windows, run everything from WSL: SkyPilot imports the POSIX-only
`resource` module and fails on native Windows. Credential files must be in
the WSL home directory (`/home/<user>/...`), not `C:\Users\...`.

## Credentials

| Cloud | Set up with | File Yeto/sky read | Env-var alternative |
|---|---|---|---|
| AWS | `aws configure` | `~/.aws/` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| RunPod | `runpod config` | `~/.runpod/config.toml` | `RUNPOD_API_KEY` |
| Nebius | `nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt` and `nebius --format json iam whoami \| jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt` | `~/.nebius/` | `NEBIUS_IAM_TOKEN` + `NEBIUS_TENANT_ID` |
| Verda | console → Credentials → Cloud API Credentials, saved as JSON | `~/.verda/config.json` (`client_id`, `client_secret`) | `VERDA_CLIENT_ID` + `VERDA_CLIENT_SECRET` |
| Modal | `modal token new` | `~/.modal.toml` | `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` |
| GCP (optional, `gs://` outputs only) | `gcloud auth application-default login` | `~/.config/gcloud/` | — |

`sky check` reports each sky cloud as enabled or explains what is missing.
Modal is not a sky cloud; `modal token` shows the active token.

**Nebius binds one project to one region.** Every Nebius region a fleet
touches needs `nebius.region_configs.<region>.project_id` in
`~/.sky/config.yaml`; `yeto launch` refuses a fleet that names a region
without one, before any VM is created. The same project ids let the
planner ask Nebius for live preemptible prices.

## What each cloud means by "region"

| Cloud | `--regions` entry | Meaning | Notes |
|---|---|---|---|
| AWS | `aws:us-east-1` (or bare `us-east-1`) | data-center region | quotas, placement scores and prices are per region; default `us-east-1,us-east-2,us-west-1,us-west-2` |
| Nebius | `nebius:eu-north1` | region | catalog (2026-09-23): eu-north1, eu-west1, me-west1, uk-south1, uk-south2, us-central1; H100×8 only in eu-north1, B200 only in me-west1/us-central1 |
| Verda | `verda:FIN-03` | location code | FIN-01/02/03 (Helsinki), ICL-01 (Iceland); the planner reads the live list, falling back to these |
| RunPod | `runpod:CA` | country code | one global pool; stock is not per country |
| Modal | `modal:us` / `modal:us-west` | placement hint with a surcharge | broad (`us`, `eu`, `ap`) ×1.15, narrow (e.g. `us-west`, `eu-north`) ×1.75; leave it out to run unpinned at the base price |

Clouds you do not name in `--regions` are unrestricted, except AWS, which
stays on its default list.

## What the planner knows per cloud

| Cloud | Catalog and price | Capacity signal | Spot |
|---|---|---|---|
| AWS | sky catalog (refreshed every few hours) | vCPU quota − usage, spot placement score (AWS APIs) | yes |
| RunPod | sky catalog (no H200 rows — the planner says so) | stock High/Medium/Low (RunPod GraphQL) | catalog spot = on-demand |
| Nebius | sky catalog with spot column; live preemptible price from the billing calculator when a project id is configured | Capacity API resource-advice, per region/platform/preset | yes; **dynamic spot pricing from 2026-10-08** — the live price overrides the catalog |
| Verda | Verda's public `/v1/instance-types` (sky's catalog is too thin) × locations | one bulk `/v1/instance-availability` call | flat 50% of on-demand |
| Modal | static table in `yeto/shape/providers.py` dated 2026-09-23 (GPU + reserved CPU + memory); the planner warns after 90 days | none; constant "will run, may queue" | none |

## Modal islands

Modal has no VMs and no SSH, so a `modal:` island is a Modal function call
(`yeto/modal_runner.py`), not a sky cluster. It runs the same `run`
script the sky task would run, with the `SKYPILOT_*` variables provided
from Modal's cluster info. Facts to keep in mind:

- The syncer must be reachable from Modal's network. Head controller mode
  (the default) puts it on a public head VM; under `--controller local`
  pass `--syncer-public-addr HOST:PORT` when the syncer address is private.
- Multi-container islands (`modal:2x8xh100`) must use whole nodes
  (`H100:8` per container, Modal rule since 2026-05-31) and get RDMA.
- RL islands need `--rl-image docker:<repo>@sha256:<digest>` — the same
  digest the sky islands use.
- Object-store data (`s3://`, `gs://`) cannot be mounted; use an HF dataset
  id or a local path.
- `yeto down <run>` stops the run's Modal app (`yeto-<prefix>`); every
  island's containers end with it, and the app must then list as
  `stopped` with 0 tasks or the run is reported not fully down.
- An all-Modal SFT fleet's model is not fetchable over ssh; recover it
  from the syncer checkpoint with `yeto-export`, or keep at least one sky
  island in the fleet.
- Manual join: launch with `--external-learners 1`, then
  `python -m yeto.modal_runner --learner-id N --num-learners M --syncer-addr H:P --run-script <file> ...`.

## Verified on real machines

Verification results decide runtime behaviour: the planner only plans RL
islands on clouds in `VERIFIED_DOCKER_IMAGE_CLOUDS`, prices RL spot only
on clouds in `VERIFIED_SPOT_STORAGE_CLOUDS`, and assumes an RDMA fabric
for multi-node islands only on `RDMA_CLOUDS` (all in
`yeto/shape/catalog.py`). Update the constants and this table together.

| Cloud | SFT island, 2 sync rounds | RL island via `docker:` image | Spot checkpoint store | Multi-node fabric | Notes |
|---|---|---|---|---|---|
| AWS | yes (pre-existing) | yes (pre-existing) | yes (`sky.Storage`) | EFA on p4/p5 | this account's G-and-VT vCPU quota is 0 in us-east-1, so no L4/g6 island; used as head/syncer in task 8.7 |
| RunPod | yes (pre-existing) | yes (pre-existing) | not verified | single node | |
| Nebius | yes (2026-09-23, run log below; also in the 8.7 mixed fleet) | pending | pending (S3-compatible store needs `aws configure --profile nebius`) | pending (expect InfiniBand on 8-GPU SXM) | head and learner both in eu-north1; also ran with an AWS head |
| Verda | pending | pending | not available in sky | single node | first run: FIN-03, 1×H100 |
| Modal | yes (2026-09-23, run log below; syncer on a Nebius head, and in the 8.7 mixed fleet with an AWS head) | yes, 8xH100 single container (2026-09-23, task 8.4, public `radixark/miles` digest; run log below) | pending (Modal Volume) | pending (RoCE, whole nodes) | WAN sync 123-338 ms/step after cold start; 299-452 ms/step alongside a Nebius island; RL 8xH100: sync blocks about 27% of a 38 s round |

Record each run's syncer-tape summary and the exact command here when a
row changes.

### Run log

**Nebius SFT island, 2026-09-23 (task 8.1).** Head controller on Nebius
(`cpu-d3_8vcpu-32gb`, eu-north1), one learner on
`gpu-h100-sxm_1gpu-16vcpu-200gb` in eu-north1, on-demand.

```bash
yeto launch --gpu nebius:1xh100@eu-north1 --syncer-region nebius/eu-north1 \
  --model qwen3-0.6b --data HuggingFaceH4/no_robots --assistant-mask-mode legacy \
  --max-rows 2000 --seq-len 1024 --total-steps 16 --fragments 8 --on-demand
```

Syncer tape summary: learner 0 connected (8 fragments, 4 streams), outer
steps 1-16 each answered by learner 0 (two full rounds over the 8
fragments), final checkpoint written at step 16, learner finalized, "all
learners acknowledged final cut", "training complete after 16 outer
steps". First two outer steps took about 6.3 s (cold start); steps 3-16
took 111-195 ms each. The head tore the learner down itself at the end.

Found while getting this to run (fixed in this change): the head needs
`~/.sky/config.yaml` for Nebius project ids, and the head must install
sky's `nebius`/`verda`/`runpod` extras (and `modal` when a fleet has a
Modal island). Also: Qwen3-0.6B's chat template has no `{% generation %}`
block, so it needs `--assistant-mask-mode legacy`; and a Nebius tenant's
default public-IPv4 quota is 3, i.e. at most three VMs across all runs.

**Modal SFT island, 2026-09-23.** Head controller (syncer) on Nebius
eu-north1; one learner as a Modal function on `H100:1`, unpinned region,
routed by the launcher (`yeto launch --gpu modal:...`).

```bash
yeto launch --gpu modal:1xh100 --syncer-region nebius/eu-north1 \
  --model qwen3-0.6b --data HuggingFaceH4/no_robots --assistant-mask-mode legacy \
  --max-rows 2000 --seq-len 1024 --total-steps 16 --fragments 8 --on-demand
```

Syncer tape summary: the Modal learner (learner 0) connected over the
public internet to the Nebius head, outer steps 1-16 each answered by it
(two full rounds), final checkpoint at step 16, "training complete after
16 outer steps"; the head tore the island down and `yeto down` stopped the
Modal app. Per-step gradient norms match the Nebius run step for step
(same data, seed and model). Sync time per outer step: about 9.8 s for
steps 1-2 (cold start), then 123-338 ms (Nebius-local learner: 111-195 ms).
At this model size the WAN adds well under a second per step, so no
Modal-specific `--pipeline` change is indicated yet; 8.4 still has to
measure it for an 8×H100 RL island.

Found while getting this to run (fixed in this change): the head must
install `modal`; a `modal:` island must not get sky resources; the Modal
image must add the repo directory last (`copy=False` mounts forbid later
build steps); and `modal app stop` needs `--yes` without a terminal.
An all-Modal SFT fleet's model stays in the syncer checkpoint on the head
(recover with `yeto-export`); tear down only after exporting if you need it.

**Mixed fleet, 2026-09-23 (task 8.7).** One launch, two learner islands on
two different clouds, syncer on a third: head controller on AWS
`m6i.2xlarge` in us-east-1 (CPU only), learner 0 on Nebius
`gpu-h100-sxm_1gpu-16vcpu-200gb` in eu-north1, learner 1 as a Modal
function on `H100:1` with an unpinned region.

```bash
yeto launch --gpu nebius:1xh100@eu-north1,modal:1xh100 --syncer-region us-east-1 \
  --model qwen3-0.6b --data HuggingFaceH4/no_robots --assistant-mask-mode legacy \
  --max-rows 2000 --seq-len 1024 --total-steps 16 --fragments 8 --quorum 2 --on-demand
```

`--quorum 2` matters here: at the default quorum of 1 the two islands
race and each answers a different outer step, so neither reaches a full
round with the other.

Syncer tape summary: syncer listening 14:27:43Z expecting 2 learners; the
Modal learner connected at 14:28:48Z and the Nebius learner at 14:32:28Z;
outer steps 1-16 each list **both** islands as responders
(`responders=[learner_id: 0, learner_id: 1]`, 16 of 16), i.e. two full
rounds over the 8 fragments with both clouds in every round; gradient
norms fall across the rounds (4.68 at step 1 / fragment 0 to 3.39 at step
9 / fragment 0); "all learners acknowledged final cut learners=2";
"training complete after 16 outer steps"; both learner jobs SUCCEEDED.

Timing: steps 1-2 took 4.6 s each (cold start), steps 3-16 took 299-452
ms each — the whole 16-step run took 6.8 s of wall clock after warm-up.
The slowest island sets the pace, and mixing a Nebius VM with a Modal
container costs nothing measurable over the single-cloud runs above.
Cold start to first step was dominated by Nebius VM provisioning
(4.8 min from syncer-up); the Modal island was ready in about 1 min
because its image layers were already cached from an earlier attempt.

Teardown: the head tore down both islands itself at the end, then
`yeto down yeto-mix87b` exited 0 with the head down. Verified clean on
all three clouds afterwards: `sky status` has no `yeto-mix87b-*` cluster,
`modal app list` shows `yeto-yeto-mix87b` as `stopped` with 0 tasks
(stopped 14:34:19Z), the Nebius eu-north1 project lists no `yeto-mix87b`
instance, and no EC2 instance is running in us-east-1. Full logs, the
syncer tape and the verification output are in the change's
`run-logs/` directory.

Metrics captured: this is an SFT run, so there is no reward or advantage
to record; the syncer tape carries the merged gradient norm per step and
the learners log `loss/token` and `target_tokens`. Only the Modal
island's per-step metrics survive in the saved log (2.56 down to 1.56
over 135 records, in `run-logs/task-8.7-modal-island-metrics.txt`): a sky
island's training log stays on its own cluster and is lost at teardown,
so run `yeto logs <prefix>` before `yeto down` if you need to compare two
islands step by step.

Three things found while running this, none fixed in this change:

- **The spec's literal fleet (`aws:1xl4@us-east-1` + `modal:1xh100`) is
  not runnable on this AWS account.** The "Running On-Demand G and VT
  instances" vCPU quota in us-east-1 is 0, and the G/VT spot quota is 0
  too, so no g4/g5/g6 instance — every L4 shape — can start; all five
  zones returned `VcpuLimitExceeded`. Only the P bucket has quota (96
  vCPU), where p3 is V100 (sm_70, no bf16) and p4d/p5 need capacity
  reservations. So the AWS island was replaced by a Nebius one and AWS
  was kept in the fleet as the head/syncer, which needs only standard
  on-demand vCPU. Anyone re-running the literal command needs a G-and-VT
  quota increase first. The abort path behaved correctly: when the AWS
  island failed to provision, the launcher tore the already-deployed
  Modal island back down and left nothing running.
- **A sky island's training logs never reach the head job log.** The head
  forwarded only four setup lines from the Nebius island while the Modal
  island's whole stdout came through, because Modal's SDK streams it back
  directly. The tape still proves the island trained (it responded on
  every step and finalized), but the per-step loss comparison between the
  two clouds is not recoverable after teardown.
- **`yeto down` reports "Modal app stop failed" when the app is already
  stopped.** After a clean run the head has already stopped the Modal
  app, so `yeto down` prints
  `Modal app stop failed: ... App is already stopped.` and then proceeds
  normally (exit code 0). Fixed with the head-run teardown change: an
  already-stopped app now prints as such, and the app's state is checked
  afterwards either way.

**Modal RL island, 8xH100, 2026-09-23 (task 8.4).** Head controller
(syncer) on Nebius eu-north1; one Miles RL island as a Modal function on
`H100:8`, unpinned region. The task text says CyberGym; the run used
MATH-500 with a rule-based math-answer reward instead, at the user's
direction. The default `ghcr.io/agentenv/miles` image is private
(anonymous pull returns 403 and this machine has no GitHub token), so the
run pinned the public upstream `radixark/miles:v0.1.0` by digest; Miles
and SGLang are still installed at their pinned commits by the island
setup, so the image swap does not change the code that runs.

```bash
yeto launch --training-mode rl --gpu modal:8xh100 --syncer-region nebius/eu-north1 \
  --cluster-prefix yeto-rl84f \
  --rl-image docker:radixark/miles@sha256:cd40db923225c4146e90fdf4aa04bc000b71c1e980cc42df6f368de7545eaa09 \
  --model Qwen/Qwen3-1.7B --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --data HuggingFaceH4/MATH-500 --data-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --reward-function yeto.rl.math_reward:reward_func \
  --apply-chat-template-kwargs '{"enable_thinking": false}' \
  --tuning lora --lora-r 8 --lora-targets attention --total-steps 3 \
  --rollout-batch-size 16 --n-samples-per-prompt 4 --rollout-max-response-len 1024 \
  --seq-len 2048 --inner-lr 1e-5 --seed 17 --trust-remote-code --on-demand
```

Syncer tape summary (strict-avg, so 1 fragment and every outer step is a
full round): outer steps 1, 2 and 3 each answered by `[learner 0]`,
`quorum=1/1`, `missed_grace` empty, `attempt=1`; the head tore the island
down and the job finished `SUCCEEDED`. Three full rounds, so the two the
task asks for. Rollouts were real: mean raw reward 0.609375, 15.6% of
responses truncated at 1024 tokens, mean response length 587 tokens.

Timing, reconstructed from island timestamps around each syncer commit
(steps 1/2/3): local train end to syncer commit 6.13/6.08/6.06 s (export,
delta, WAN upload, merge); syncer commit to the island starting
`update_weights` 4.83/4.58/4.50 s (WAN download, apply, hash check);
`update_weights` itself 1.03/1.02/1.02 s (the SGLang publish every Miles
round pays anyway). Total blocking 11.99/11.68/11.58 s out of a ~38.4 s
round, about 30%, or about 27% excluding the SGLang publish. The syncer's
own merge is only 48-60 ms of that: `sync_ms` in the tape is
`merge_seconds` and is NOT a WAN measurement, and the WAN time is folded
into `quorum_ms`, which the island's rollout and training dominate. Cold
start from spawn to the first closed outer step was about 17 min (image
pull 44 GB on first deploy only, then container setup, the Hub model
download without a token, Ray, Megatron and eight SGLang engines).

`--pipeline`: inapplicable to this run rather than merely unnecessary.
`strict-avg` forces `--fragments 1` and `--pipeline` is clamped to
`--fragments`, so it was 1 and changing it would have done nothing. The
27% blocking overhead is worth reclaiming, and the mechanism `--pipeline`
exists for is the right lever: switch to the decoupled preset and raise
`--fragments` so one fragment's WAN round trip overlaps another's
compute. The default `--pipeline 2` is a sound starting point; nothing
here argues for going above 2. Re-measure for a larger model or more
islands: the delta payload and the `quorum_ms` wait for peers both grow.

**The island trained nothing, and that is a separate bug.** The syncer
logged `gnorm="0.0000"` for all three steps, which is the `{gnorm:.4}`
format rounding the tape's real values (`4.161e-06`, `3.697e-06`,
`1.103e-06`) - read the tape, not the log line. But those are about 3000x
smaller than one Adam step at the logged `lr-pg_0=6.67e-06` should give
for 3.67M LoRA parameters (about 0.013), so they are fp32 round-trip
noise, and Megatron reports `train/grad_norm` as exactly `0.0` on every
step. A 1xH100 repro (about $1.50) found the cause: gradients are
computed correctly (224/224 adapter parameters have `requires_grad=True`
and non-zero `param.grad`, `lora_grad_l2=2.79e-02`) but never reach
Megatron DDP's gradient buffer, because
`miles/backends/megatron_utils/trainable_state.py:502` assigns an fp32
tensor to a bf16 `Parameter.data` and PyTorch resets the grad accumulator
on a dtype change, destroying the DDP hook registered on it. That file is
Yeto's policy-sync integration layer, so Miles' own LoRA tests never
exercise it. Tracked as the `fix-rl-lora-grad-hook` change; full evidence
in that change and in this change's `live-run-failures.md` (items 33 and 35). Note the
rollouts and advantages were healthy - this is not GRPO degeneracy.

This entry records the cross-cloud result the task asks for: an 8xH100
single-container Modal RL island, two or more full sync rounds with the
island in every round's responders, the WAN cost measured, and the
`--pipeline` conclusion. Whether to re-run it once the gradient bug is
fixed is an open decision recorded in `fix-rl-lora-grad-hook/design.md`.

**`yeto shape` with real credentials, 2026-09-23 (task 8.5).**
`yeto shape --clouds <cloud> --model qwen35-9b --budget 40`, spot. Checked
against the day's catalog: Nebius H100×1 live spot $2.15 = sky catalog
$2.150; Verda H100×1 spot $1.69 = half of the $3.382 on-demand price
(Verda's flat spot rule); Modal H100/B200 match the static table (GPU
$3.95 / $6.25 per hour plus reserved CPU and memory).

Nebius:

```
plan: 9 island(s), 6313.8 effective TFLOPs, est $34.15/hr — ≤ $39.21/hr with 15% spot-price margin (budget $40.00/hr)
  8x nebius:1xb200@me-west1  spot est $3.95/hr/island  score stock≈9  748.1 TFLOPs/island
  1x nebius:1xh100@eu-north1  spot est $2.15/hr/island (live)  score stock≈9  328.8 TFLOPs/island
  head: on-demand CPU VM  $0.40/hr
binding constraints: budget
warning: nebius: no project_id for region me-west1 in ~/.sky/config.yaml (nebius.region_configs); catalog spot price used there
warning: nebius: no project_id for region us-central1 in ~/.sky/config.yaml (nebius.region_configs); catalog spot price used there
warning: nebius: no project_id for region eu-west1 in ~/.sky/config.yaml (nebius.region_configs); catalog spot price used there
warning: nebius: Capacity API lists no preemptible entry for gpu-l40s-d_2gpu-64vcpu-384gb in eu-north1; capacity unknown
warning: nebius: Capacity API lists no preemptible entry for gpu-l40s-d_4gpu-128vcpu-768gb in eu-north1; capacity unknown
warning: nebius: 16 spot price(s) from the live pricing API override the catalog
warning: placement score unavailable for 2 shape(s) (e.g. nebius:4xl40s@eu-north1); assumed 10 — pass --strict-capacity-check to reject them instead
```

Verda:

```
plan: 16 island(s), 5261.5 effective TFLOPs, est $27.46/hr — ≤ $31.51/hr with 15% spot-price margin (budget $40.00/hr)
  16x verda:1xh100@FIN-02  spot est $1.69/hr/island  score stock≈9  328.8 TFLOPs/island
  head: on-demand CPU VM  $0.40/hr
binding constraints: max-islands
```

Modal:

```
plan: 2 island(s), 2992.5 effective TFLOPs, est $33.36/hr — ≤ $38.30/hr with 15% spot-price margin (budget $40.00/hr)
  1x modal:4xa10g  spot est $6.18/hr/island  score autoscale  157.5 TFLOPs/island
  1x modal:4xb200  spot est $26.78/hr/island  score autoscale  2835.0 TFLOPs/island
  head: on-demand CPU VM  $0.40/hr
binding constraints: budget
```

Two fixes came out of these runs: the Nebius Capacity API caps `pageSize`
at 200 (the planner asked for 1000 and every Nebius shape fell back to the
assumed score), and for non-AWS clouds a sold-out fat node no longer
hides its in-stock thinner siblings (Verda returned "no feasible plan"
while 1×H200 was in stock). Known limits: Verda's stock signal is binary,
so the planner may propose more 1-GPU islands than Verda actually has;
and the planner warns about, but still plans in, Nebius regions with no
project id, which `yeto launch` then refuses.

**Modal manual join, 2026-09-23 (task 5.3).** A fleet with one Nebius
island and one external seat, `--quorum 2` so every outer step needs both
islands (with the default `--quorum 1` the two islands raced and the
Modal island answered only steps 1-4):

```bash
yeto launch --gpu nebius:1xh100@eu-north1 --external-learners 1 --quorum 2 \
  --syncer-region nebius/eu-north1 --model qwen3-0.6b --data HuggingFaceH4/no_robots \
  --assistant-mask-mode legacy --max-rows 2000 --seq-len 1024 \
  --total-steps 16 --fragments 8 --on-demand
# then, with the seat printed by the launch log (learner 1 of 2):
python -m yeto.modal_runner --learner-id 1 --num-learners 2 --syncer-addr <head-ip>:29400 \
  --gpu h100 --gpus-per-node 1 --run-script <run script for seat 1> --follow
```

Syncer tape: learner 1 (Modal) and learner 0 (Nebius) connected; every
outer step 1-16 was answered by both (two full rounds), final checkpoint
at step 16, "all learners acknowledged final cut learners=2".

```
outer step step=1 fragment=0 responders=[L0, L1] gnorm="4.3737" ms=5740
outer step step=2 fragment=1 responders=[L0, L1] gnorm="4.2017" ms=5756
outer step step=3 fragment=2 responders=[L0, L1] gnorm="8.3074" ms=1153
outer step step=4 fragment=3 responders=[L0, L1] gnorm="8.1539" ms=1151
outer step step=5 fragment=4 responders=[L0, L1] gnorm="7.3872" ms=1155
outer step step=6 fragment=5 responders=[L0, L1] gnorm="6.8711" ms=1006
outer step step=7 fragment=6 responders=[L0, L1] gnorm="7.1109" ms=1137
outer step step=8 fragment=7 responders=[L0, L1] gnorm="6.7804" ms=1126
outer step step=9 fragment=0 responders=[L0, L1] gnorm="3.2665" ms=1097
outer step step=10 fragment=1 responders=[L0, L1] gnorm="3.2453" ms=1051
outer step step=11 fragment=2 responders=[L0, L1] gnorm="3.3925" ms=1021
outer step step=12 fragment=3 responders=[L0, L1] gnorm="3.4000" ms=1140
outer step step=13 fragment=4 responders=[L0, L1] gnorm="3.1929" ms=1022
outer step step=14 fragment=5 responders=[L0, L1] gnorm="3.1630" ms=1044
outer step step=15 fragment=6 responders=[L0, L1] gnorm="3.3730" ms=1153
outer step step=16 fragment=7 responders=[L0, L1] gnorm="3.2738" ms=1086
```

The run script must be built from the fleet's own arguments and source
(`launcher.make_learner_task` for a `modal:` spec at the seat's learner id):
the learner refuses to start when its source SHA256 differs from the
fleet's. The launch log still prints only an MLX join command, not a Modal
one; building the run script by hand is the gap left for manual Modal joins.

## Tearing a run down

`yeto down <prefix>` only says `run '<prefix>' is down` (exit 0) once every
cluster of the run is confirmed gone; anything less exits 1, leaves the run
in state `TEARDOWN_INCOMPLETE` with the unconfirmed clusters recorded, and
a rerun of `yeto down` continues from there. The order matters:

1. **Learners of a head run are torn down from the head.** Only the head's
   sky knows them: this machine's sky launched the head, the head's sky
   launched the learners, and a local `sky down <learner>` just answers
   "does not exist". `yeto down` cancels the controller job on the head
   (so it cannot relaunch what is being removed), then runs the downs over
   ssh on the head, where each learner is also checked at the cloud before
   it counts as down. Three attempts, 20 s apart, because the head's own
   sky answers 500 while it is busy.
2. **The head is deleted only after every learner is confirmed.** If any
   learner is still unconfirmed the head is kept, the command exits 1 and
   names the learners: rerun `yeto down`, or delete them in the cloud
   console. Deleting the head first is exactly how H100s were orphaned on
   2026-09-23 and twice on 2026-09-24 (`yeto-gh1`, `yeto-gh2`).
3. **Modal islands end with the run's app**, then the app must list as
   `stopped` with 0 tasks.
4. **Everything this machine's sky launched (the head, local-mode
   learners) is checked at the cloud after the down**, using sky's own
   per-cloud instance query; a surviving instance is retried and, if it
   outlives the retries, printed by id so it can be deleted by hand. A
   cloud that cannot be queried is reported as "not cloud-verifiable;
   trusting sky", and then only a clean down or "does not exist" counts.

Verified 2026-09-24 on `yeto-td2` (Nebius head, one Modal `1xh100` SFT
island, torn down while training at outer step 3): the app stopped and
listed as `stopped, 0 tasks`, the head was downed and confirmed at the
cloud, exit 0, nothing left on Nebius or Modal. `yeto-td1` (whose Nebius
learner never provisioned: the tenant's public-IPv4 quota of 3 was full)
exercised the head-side path: the head reported the learner as never
existing, then the head itself was downed. The head-side teardown of a
provisioned Nebius learner, and the "head already gone" failure path, are
still to be run once the IPv4 quota has room for head + learner.

If the head is already gone (deleted by hand, or by an older `yeto down`),
step 1 cannot run: the command exits 1 listing the learners, and the only
way to find them is the cloud's own listing, e.g.
`nebius compute instance list` — look for instances named after the
learner cluster.
