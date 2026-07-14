# Optimizer hunt: experiment log, harness, and formal audit

Last updated: 2026-07-14 (America/Los_Angeles)

This is the durable operator handoff for the isolated pushed branch
`experiment/optimizer-state-capture-round3`. Harness code, analyzers,
experiment specifications, tests, and Lean sources are tracked on that branch;
the gitignored `.optimizer-harness/` directory is deliberately local lifecycle
state, must not be committed, and is not experiment evidence. Historical
`exp2-53a` and `exp2-53a2` BC-MP execution remains pinned to pushed commit
`9f9edadf3e8a39c91ce5d64d5ac0a93ca22424c4` on
`origin/experiment/bcmp-shadow-round2`; later documentation and harness work
must not rewrite that provenance.

The handoff records the completed SCAFFOLD de-confound, timestamped cloud
state, ingress diagnosis, reusable GCP image work, optimizer proposals, local
formal results, experiment gates, and exact teardown rules. Cloud observations
are timestamped snapshots, not timeless inventory assertions.

The paper's current recommendation remains **memoryless outer SGD with
`outer_lr=0.28`**. The experiment below is a mechanism footnote, not a reason
to reopen that recommendation without fresh controlled evidence.

## Completed experiment

Experiment `exp2-52` began as a fresh two-arm H16 capture:

1. `scaffold_lite`
2. `scaffold_sgd`, the exact live control

Both arms use:

- outer optimizer: memoryless SGD, represented by Nesterov with momentum zero;
- outer learning rate: **0.28**;
- inner optimizer: plain SGD;
- inner learning rate: **0.001**;
- fixed window: H16;
- four learners, four fragments, strict quorum;
- LoRA rank 2, alpha 4;
- sequence length 128 and 700,000 training tokens per arm;
- f32 wire, overwrite broadcasts, version-matched anchors;
- `delta_correction=heloco` in both arms;
- seed 223223 and row-shuffle seed 223;
- a newly generated, common 64-row evaluation split.

The third-arm LR was deliberately unknown until the two captures completed. It
was fitted as

\[
s^\star =
\frac{\sum_t\langle u_{\mathrm{lite},t},u_{\mathrm{sgd},t}\rangle}
     {\sum_t\lVert u_{\mathrm{sgd},t}\rVert^2},
\qquad
\eta_{\mathrm{match}}=0.28s^\star.
\]

The complete 340-commit tapes gave `s*=2.676787578401296`, so the frozen third
arm used `eta_match=0.749500521952363`. It was not selected using final loss or
refitted after the confirmation result.

The live command contains `--baseline-loss 0.0`. In commit `f08563a`, this is
the historical way to suppress an irrelevant 5,469-step synchronous baseline.
It is report-only and has no effect on either training arm. It is not a valid
loss reference. The harness now adds `--skip-baseline` to the current branch so
future paired experiments can omit the baseline honestly instead of recording
a fake zero row.

### Final de-confound outcome

All three arms used the same model, commit, data/evaluation construction,
seeds, H16 schedule, token budget, inner SGD, and outer memoryless-SGD rule.

| arm | outer LR | eval loss/token |
|---|---:|---:|
| SCAFFOLD-lite | 0.28 | 1.480808494870277 |
| fresh SGD control | 0.28 | 1.543579193244204 |
| frozen matched SGD | 0.749500521952363 | 1.4723387220473634 |

The raw fresh-control gap was `0.0627706983739269`. The preregistered residual
was

\[
G_{\mathrm{res}} =
1.4723387220473634-1.480808494870277
=-0.008469772822913724.
\]

This is below the `0.018` kill threshold, and matched SGD actually beat
SCAFFOLD-lite by about `0.00847`. Therefore there is no practical residual
benefit beyond an LR/scale retune in this H16 protocol. The SCAFFOLD-lite
drift-correction claim is closed; full accumulating SCAFFOLD and identity
shuffle are not triggered.

The capture audit aligned all 340 commits in each arm, 85 per fragment, with
no missing keys. Its global diagnostics were `r_E=4.6691112772064995`,
`c_E=0.5732970193854111`, and `r_perp=0.8193476231513724`. Because
`c_E < 0.95`, the two independently evolved update tapes are not cleanly
collinear and the projection is not a same-state scalar causal decomposition.
That geometric caveat prevents a stronger mechanism claim; it does not rescue
SCAFFOLD-lite after the frozen matched control achieved lower final loss.

### GCP resource identity and final state

The following VM was the only source VM owned by this experiment:

- project: `model-training-497007`
- zone: `us-central1-b`
- name: `exp2-52c-deconfound`
- numeric instance ID: `6468323683998395950`
- type: `a2-highgpu-4g` (4 × A100 40 GB)
- provisioning: Spot
- preemption action: delete
- boot disk: `exp2-52c-deconfound`, numeric ID
  `6469542114680663598`, 250 GB
- artifact prefix: `gs://yeto-exp2-52-model-training-497007/exp2-52`
- pinned experiment commit: `f08563a9bf944062a51e1b85dc987cbc071ca7bd`

The instance was adopted into the new harness. The local, gitignored
state records the numeric ID, exact boot-disk self-link, and a random ownership
nonce. Every status, imaging, and deletion operation freshly verifies all of
those identities and the management labels. After successful imaging and
canary promotion, exact source instance ID `6468323683998395950` and exact
boot-disk ID `6469542114680663598` were deleted and independently verified
absent.

The retained GCP artifact is:

- image name: `yeto-optimizer-a100-20260714`
- numeric image ID: `7290368630472593484`
- family: `yeto-optimizer-a100`
- status: `READY`
- family resolution: exact numeric ID `7290368630472593484`
- canary status label: `passed`

The exact canary instance ID `7731932032036000111` and its exact auto-delete
boot-disk ID `4247364754860429679` were deleted after verification and are
also confirmed absent. No exp2-52 VM or disk remains.

The unrelated GCP VM `instance-20260526-guiflow` is out of scope and must
never be changed. The AWS p4de instance `i-037f7bb7977382df9` is the user's 27B
training node and is also strictly out of scope.

### Restart correction

The first launch accidentally omitted `--baseline-loss`, so the comparison
runner started the unnecessary synchronous baseline on one GPU. It was stopped
after 75 of 5,469 steps. Neither SCAFFOLD arm had started. Only the exact runner
PID and its exact learner child were terminated; the VM was preserved. The
aborted logs and command were archived below the run directory and synced to
GCS. The two-arm run was then restarted with the same `--baseline-loss 0.0`
convention used by exp2-51.

## What “two-arm SCAFFOLD de-confound” means

The original H16 observation was:

- SCAFFOLD-lite loss: `1.473220200939307`
- its live control loss: `1.5454717985155209`
- apparent gain: about `0.0722` loss/token

The concern is that SCAFFOLD-lite may mainly rescale the applied update rather
than correct a useful client-drift direction. Exp2-52 captured every applied
outer update for the lite and control trajectories, estimated the scalar
projection above, then ran ordinary SGD at `eta_match`.

The primary residual is

\[
G_{\mathrm{res}} =
L(\mathrm{matched\ SGD})-L(\mathrm{SCAFFOLD\mbox{-}lite}).
\]

The frozen interpretation is:

- `G_res < 0.018`: the practical gain is explained by scale; kill the
  drift-correction claim;
- `G_res > 0.036`: a meaningful non-scalar residual survives;
- `0.018 <= G_res <= 0.036`: run full accumulating SCAFFOLD and the
  identity-shuffled full control; real must beat shuffle by more than `0.009`.

The fit must additionally report:

\[
r_E = \sqrt{\frac{\sum_t\lVert u_{\mathrm{lite},t}\rVert^2}
                       {\sum_t\lVert u_{\mathrm{sgd},t}\rVert^2}},
\]

\[
c_E =
\frac{\sum_t\langle u_{\mathrm{lite},t},u_{\mathrm{sgd},t}\rangle}
{\sqrt{\sum_t\lVert u_{\mathrm{lite},t}\rVert^2
       \sum_t\lVert u_{\mathrm{sgd},t}\rVert^2}},
\qquad
r_\perp=\sqrt{\max(1-c_E^2,0)}.
\]

A scalar match is cleanly interpretable only when the sequences are highly
collinear; the preregistered diagnostic threshold is `c_E >= 0.95`. Capture
alignment should cover at least 98% of commits in each arm and must also be
reported per fragment.

There is an important causal limitation: the two tapes come from independently
evolving closed-loop states after the first update. Therefore `s*` is a useful
trajectory-level fit, not a same-state causal decomposition. If matched SGD
closes the loss gap, that is strong evidence for a scale confound. If it does
not, drift correction is not proved without identity shuffle, a heterogeneous
interaction, or a same-state shadow proposal.

“IID” also does not mean identical realized worker gradients. Yeto assigns
disjoint row shards and learner-specific RNG streams. Nonzero controls under
IID may shape sampling noise rather than persistent client heterogeneity.

## Reusable optimizer experiment harness

The harness lives in:

- `yeto/optimizer_harness.py`: validation, state, GCP provider, remote runner,
  artifact sync, analysis hooks, exact-ID lifecycle, and image creation;
- `scripts/optimizer_experiment.py`: thin CLI;
- `experiments/optimizer/exp2-52-scaffold-deconfound.json`: the exact adopted
  run and image recipe;
- `tests/test_optimizer_harness.py`: focused safety and validation tests.

Local state is written under the gitignored `.optimizer-harness/state/`. It is
local lifecycle evidence, must not be committed, and contains no cloud or
Hugging Face credential.

### CLI

From the repository root:

```bash
.venv/bin/python scripts/optimizer_experiment.py \
  validate experiments/optimizer/exp2-52-scaffold-deconfound.json

.venv/bin/python scripts/optimizer_experiment.py \
  doctor experiments/optimizer/exp2-52-scaffold-deconfound.json

.venv/bin/python scripts/optimizer_experiment.py \
  render experiments/optimizer/exp2-52-scaffold-deconfound.json

.venv/bin/python scripts/optimizer_experiment.py \
  status experiments/optimizer/exp2-52-scaffold-deconfound.json

.venv/bin/python scripts/optimizer_experiment.py \
  logs experiments/optimizer/exp2-52-scaffold-deconfound.json --lines 200

.venv/bin/python scripts/optimizer_experiment.py \
  sync experiments/optimizer/exp2-52-scaffold-deconfound.json

.venv/bin/python scripts/optimizer_experiment.py \
  analyze experiments/optimizer/exp2-52-scaffold-deconfound.json

.venv/bin/python scripts/optimizer_experiment.py render-matched \
  experiments/optimizer/exp2-52-scaffold-deconfound.json \
  /path/to/scaffold-scale-fit.json

.venv/bin/python scripts/optimizer_experiment.py start-matched \
  experiments/optimizer/exp2-52-scaffold-deconfound.json \
  /path/to/scaffold-scale-fit.json

.venv/bin/python scripts/optimizer_experiment.py create-image \
  experiments/optimizer/exp2-52-scaffold-deconfound.json \
  --instance-id EXACT_SOURCE_INSTANCE_ID --yes

.venv/bin/python scripts/optimizer_experiment.py create-canary \
  experiments/optimizer/exp2-52-scaffold-deconfound.json

.venv/bin/python scripts/optimizer_experiment.py test-canary \
  experiments/optimizer/exp2-52-scaffold-deconfound.json \
  --canary-id EXACT_CANARY_INSTANCE_ID

.venv/bin/python scripts/optimizer_experiment.py promote-image \
  experiments/optimizer/exp2-52-scaffold-deconfound.json \
  --canary-id EXACT_CANARY_INSTANCE_ID --yes

.venv/bin/python scripts/optimizer_experiment.py delete-canary \
  experiments/optimizer/exp2-52-scaffold-deconfound.json \
  --canary-id EXACT_CANARY_INSTANCE_ID --yes
```

For a new spec, `launch` creates a Spot VM and records its identity; `start`
checks the exact clean commit and required offline assets before starting the
runner and periodic GCS backup. `adopt` exists only to migrate an already-owned
VM and requires the exact numeric instance ID plus `--yes`.

`launch` itself also requires `--yes`. A successfully completed run is removed
with `delete`; an incomplete or failed run uses the separate `abandon` path:

```bash
.venv/bin/python scripts/optimizer_experiment.py abandon \
  experiments/optimizer/exp2-53a-bcmp-shadow-discovery.json \
  --instance-id EXACT_RECORDED_INSTANCE_ID \
  --reason "auditable nonempty reason" \
  --yes
```

`abandon` authenticates the recorded VM, labels, nonce, and auto-delete boot
disk, refuses a valid completed run, stops only PIDs whose command lines are
bound to the exact run directory, writes and round-trips a checksummed
`abandonment.json` through GCS, reauthenticates the resources, and accepts only
explicit provider 404s as proof that both VM and disk are gone. The local state
is retained as `ABANDONED` rather than erased.

The exp2-52 spec is marked `adopt_only`: its `cloud.image` records the real
DLVM source image, and it must not accidentally launch a second raw VM. Future
run templates should reference the immutable custom image ID recorded below,
not this adoption spec or a moving image family.

### Validation and safety properties

The implemented harness:

- requires a full 40-character Git commit;
- requires an explicit Spot model and delete-on-preemption policy;
- requires explicit confirmation for launch and destructive lifecycle actions;
- enforces a campaign-wide maximum accelerator count across every live VM in
  the project, including instances the harness does not own;
- supports a provider-enforced maximum run duration and verifies the live
  scheduling value before accepting ownership;
- supports a pinned `pd-standard`, `pd-balanced`, or `pd-ssd` boot-disk type;
- requires an immutable, run-specific GCS prefix;
- refuses launch if local state or the artifact prefix already exists;
- requires JSON argv arrays rather than an interpolated main-command string;
- checks expected arms and critical flags, including H, seeds, inner/outer LR,
  learner steps, and syncer steps;
- checks that an injected baseline is finite and explicitly report-only;
- supports an honest `--skip-baseline` mode and rejects a simultaneous injected
  baseline;
- rejects duplicate JSON keys before validating a spec;
- refuses to run a dirty or untracked remote worktree;
- records the spec, command, Git status, and binary diff in every run;
- verifies exact instance ID, project, zone, name, management labels, ownership
  nonce, recorded boot-disk self-link, and live attachment/auto-delete identity
  on owned lifecycle operations;
- when `cloud.expected_source_image_id` is set, launch/adoption additionally
  requires the exact canonical source-image path and numeric image ID, records
  the numeric boot-disk ID, and fails before accepting state on any mismatch;
- if post-create or post-adoption provenance fails, retains a quarantined exact
  instance/disk ownership state so the billable resource can be inspected and
  explicitly abandoned instead of becoming an untracked orphan;
- reserves `delete` for completed, checksummed runs and `abandon` for preserved,
  explicitly justified incomplete runs;
- performs a final GCS sync before exact-name deletion;
- creates images without a family, verifies exact source instance/disk IDs,
  and promotes a family only after a stored exact-image canary pass;
- records and verifies exact canary instance/disk/image IDs and confirms both
  auto-delete resources are absent after canary teardown;
- never discovers deletion targets by label, prefix, machine type, or broad
  list filtering;
- fails closed if live cloud verification fails or identity differs.

For strict-quorum exact-step runs, `checks.strict_quorum_step_budget` records
the fragment count and a required learner-cap surplus above the
barrier-synchronous lower bound. It can also pair an empirical shutdown upper
bound with required post-empirical headroom; when present, the preflight uses
the larger of the ideal-relative and empirical requirements. In the current
post-run hardened EXP2-53 metadata, those bounds are `1360+128=1488` and
`1471+129=1600`, so a 1,599-step cap fails validation and 1,600 passes. This is
still a preflight sanity check, not a proof of liveness under arbitrary
non-barrier asynchronous scheduling.

The current optimizer-harness suite passes 67 focused tests. Remaining
desirable work is a fully staged runner in which each arm has an immutable
completion manifest and can restart independently after Spot preemption. The
current `compare_diloco.py` is still monolithic and deletes its work directory
at startup, so the harness does not pretend it can resume a partially
completed arm.

## GCP image outcome

The validated VM environment contains:

- Ubuntu 22.04 DLVM;
- NVIDIA driver/CUDA stack for PyTorch 2.9.1+cu129;
- Python venv at `/home/shou/venv`;
- Rust 1.97 and the release syncer build;
- Qwen3.5-9B at `/home/shou/models/Qwen3.5-9B`;
- Capybara parquet at `/home/shou/data/Capybara-local/train.parquet`;
- the pinned candidate repo;
- 166 passing Rust tests, 20 passing SCAFFOLD Python tests, and a successful
  offline bf16 model load/forward on an A100.

The completed image stage created:

- custom image: `yeto-optimizer-a100-20260714`
- family: `yeto-optimizer-a100`
- storage location: `us-central1`
- numeric image ID: `7290368630472593484`

The image path is deliberately after experiment completion:

1. Require the exact runner to be dead and the result file to exist.
2. Perform a final GCS rsync.
3. Stop the exact backup PID after the final sync.
4. Remove the exact run directory and credential/history locations declared in
   the spec, including the Hugging Face/Xet cache and token, gcloud/AWS/GitHub
   CLI config, Docker config, package-registry credentials, netrc, git
   credentials, SSH material, and shell history.
5. Clean nested `/tmp` and `/var/tmp`, rotate/vacuum journals, and run
   `cloud-init clean --logs`.
6. Scan `/home/shou` and `/root` for credential-like filenames and fail closed
   if any remain.
7. Hash every model file and the parquet dataset into
   `/etc/yeto-model-files.sha256` and `/etc/yeto-data.sha256`; record exact
   Torch/Transformers/CUDA/Rust/driver/Git versions in
   `/etc/yeto-runtime.txt`; write `/etc/yeto-optimizer-image.json` with the
   source run and exact commit.
8. Clear machine identity, sync filesystems, and stop the exact Spot VM.
9. Freshly verify that the stopped VM still has the recorded numeric ID,
   ownership nonce, and boot disk.
10. Create an unpromoted candidate image from the stopped disk without
    `--force`; do not assign its image family yet.
11. Freshly describe and verify its numeric image ID, image nonce, source disk
    ID/self-link, source instance ID, labels, `READY` status, and absence of a
    family, then record those identities in local harness state.
12. Launch a one-A100 Spot canary from the exact image name. Record and verify
    the numeric canary instance and boot-disk IDs, ownership nonce,
    `sourceImageId`, auto-delete disk setting, and source image self-link.
13. Recheck model/data hashes, credential absence, regenerated machine ID,
    offline CUDA/model forward, focused Python tests, the Rust suite, and GCS
    read/write. A missing/preempted canary is inconclusive, never a pass.
14. Only after the stored pass, promote that exact image into
    `yeto-optimizer-a100` and prove `describe-from-family` resolves to the same
    numeric image ID.

`render-matched` refuses a scale file whose recorded source LR differs from
the exact control command, requires the analyzer's explicit non-causal
closed-loop label, bounds the fitted LR to the predeclared `(0, 2]` safety
range, freezes `--settings scaffold_sgd`, assigns a separate work/report
subdirectory, and drops expensive probe capture from the third arm.
`start-matched` additionally requires the base runner to be dead and its result
to exist, starts the frozen command as a separately logged/checksummed stage,
updates the active status/log pointer, and adds the matched result plus checksum
manifest to the exact-ID teardown/image gate.

After promotion, the harness deleted only the recorded canary and source
instance IDs and verified the canary and source auto-delete disks were gone.

The completed canary used one `NVIDIA A100-SXM4-40GB` with driver 580.159.03.
It verified every model/data checksum, regenerated machine identity and
credential absence, ran a full offline Qwen3.5-9B forward with PyTorch
2.9.1+cu129 (peak allocation 17,962,657,280 bytes), passed 66 focused Python
tests and 166 Rust tests, and completed an exact-prefix GCS write/read/remove.
Only after that pass was the family assigned.

Durable experiment artifacts include:

- `gs://yeto-exp2-52-model-training-497007/exp2-52/report/results.jsonl`
  (`sha256:299e82497700c7d985cb4f81966322c33804fa36f0d32f2efce7714a15e31f1a`)
- `gs://yeto-exp2-52-model-training-497007/exp2-52/matched-sgd/report/results.jsonl`
  (`sha256:2632f1d1a0d1d70b03af1946af273bba5105f22d08343cf5815b0040d8a594e1`)
- `gs://yeto-exp2-52-model-training-497007/exp2-52/analysis/scaffold-scale-fit.json`
  (`sha256:10f2602332a40dfa83fb6ae4a3b6ad5f8e6d53ca7ac4179a7ff6f58e06469904`)

Stopping before imaging is intentional. Google documents that Spot VMs can be
manually stopped and restarted; the delete termination action governs
preemption. Google also warns that `images create --force` is the path for an
image made from an attached running disk. This harness avoids that less-clean
path and images a stopped disk.

For the longer term, the stronger image pipeline is a separate disposable
builder from a pinned DLVM source, a checksummed model archive in GCS, an
allowlisted runtime recipe, a credential scan, and a fresh canary. Source code
should then be checked out at the experiment's exact commit rather than treated
as authoritative merely because a copy exists in the image.

The harness already supports that separation through
`execution.source_mode=checkout`: it clones the public repo into a new
run-specific path, fetches the exact 40-character commit, checks out detached,
and then applies the same clean-tree verification. The adopted exp2-52 run uses
`preinstalled_exact` because its training source was already present and
verified before the harness was introduced.

## Slow GCP ingress: diagnosis

The initial belief that all GCP ingress was slow was incorrect. Direct tests
on the chosen VM showed roughly:

- GCS: 96 MB/s;
- Cloudflare: 155 MB/s;
- a real Hugging Face shard: 332 MB/s when using a working signed endpoint.

The failure was endpoint-specific:

- on the GCP VM, Hugging Face resolved the affected transfer through
  `us.gcp.cdn.hf.co` and range/fallback requests returned `403 SignatureError`
  for an invalid key-pair ID;
- `hf-xet` 1.5.1 also stalled;
- resolving a short-lived signed public model URL from the Mac produced a
  working `cas-bridge.xethub.hf.co` URL;
- the bytes then flowed directly from the public endpoint to GCP. The Mac only
  relayed the short-lived URL, not the 19.3 GB model payload.

The final model cache has all 16 expected files and exactly 19,329,393,661
bytes. The local Capybara parquet has 15,806 rows and 37,162,331 bytes. Future
instances should avoid this ingress dependency entirely by launching from the
custom image or reading a checksummed private GCS model archive.

The custom image removed network ingress but exposed a different bottleneck:
the first exp2-53 smoke inherited a 250 GB `pd-standard` boot disk. During the
mandatory base-model evaluation, the Python process spent its time in kernel
`folio_wait_bit_common`; only about 3.24 GB had been read after roughly three
minutes, around 17--20 MB/s, while the A100 was effectively idle. That owned
smoke was terminated after partial logs were synced. Exact VM ID
`1744019134829053908` and its auto-delete disk are verified absent, but the
local harness record still says `RUNNING_EXPERIMENT`, and no
`abandonment.json`, result, or final manifest was observed at its GCS prefix.
Treat the prefix as incomplete negative provenance, do not reuse it, and
reconcile the stale local lifecycle record before citing this as a completed
harness `abandon` transaction.

The retry pinned `pd-ssd`. It read 6.18 GB in 25 seconds, about 247 MB/s and
roughly 12--15 times the observed `pd-standard` rate. Local parquet
materialization also fell from roughly 1.6 seconds to about 0.3 seconds. The
SSD image restore took about three and a half minutes at provisioning time, but
the successful smoke then completed end to end and left checksummed results in
`gs://yeto-exp2-52-model-training-497007/exp2-53-smoke2`. Exact retry VM ID
`7061704855101392328` and its auto-delete disk were verified absent after the
final sync. Future A100 experiment specs therefore pin `boot_disk_type` to
`pd-ssd`; “ingress” dashboards should separate network transfer, image restore,
local model reads, model initialization, and evaluation.

## Cloud inventory and earlier cleanup

### Verda

The Verda purge is complete: all 211 trash volumes were permanently deleted
and the trash listing reached zero with a valid token. The API behavior had
changed: permanent deletion required JSON `{"is_permanent": true}` and returned
202; the old query-string form was a no-op. No active Verda instance remains.

The desired `8L40S.160V` shape was no longer available after cleanup. The only
eight-GPU option observed was an expensive FIN03 8×B300 shape, so GCP Spot was
chosen instead.

### AWS

AWS is not currently viable for this four-A100 job. The us-east-1 P-Spot
request `06a577d8449947daa77c4d94f66eb413hv2KfCjg` did close successfully and
the applied quota is now 256. That removes the earlier quota blocker even with
the protected 96-vCPU p4de workload. A fresh read-only placement query on
2026-07-14, however, returned score `1` for `p4d.24xlarge` in every returned
us-east-1 availability-zone ID (`use1-az1`, `az2`, `az4`, and `az6`). AWS is
therefore quota-ready but presently capacity-poor. Other US p4d scores were
also poor in the prior survey, while eu-west-1 had capacity but zero P-Spot
quota.

No AWS resource was modified for exp2-52.

## Ten-agent optimizer search

Ten optimizer subagents were launched in waves because the workspace permits
only three subagents concurrently. Their assignments covered formal Lean
proofs, optimizer theory, tape replay, minimax impossibility, restart-sum audit,
literature/novelty, statistical gates, Adam boundary-state repair, harness
architecture, and final red-team synthesis.

The current ranking is:

| Candidate | Decision | Reason |
|---|---|---|
| SCAFFOLD-lite H16 de-confound | Completed; claim closed | Frozen scale-matched SGD beat SCAFFOLD-lite by about 0.00847, so the residual did not trigger full SCAFFOLD or identity shuffle. |
| BC-MP-AdamW boundary repair | Kill; discovery `NO_GO` | Ray/reset had positive but tiny next-gradient cosine near `0.00153`, far below the frozen `>0.02` gate. Slab was smaller and also failed action-size and positive-rate gates. No confirmation or CRN replay is triggered. |
| Product/quotient LoRA merge | Offline replay only | Most raw difference is product-space RDA; unique arithmetic residual is only about 1–3%. Gauge-aware LoRA is crowded prior art. |
| Lean transverse correction | Formal/oracle only | Exact local quadratic theorem is real, but requires true same-state gradient/HVP and positive curvature; the related CTTN oracle was null. |
| Restart-Sum/RS2 | Kill | Fails under actual persistent AdamW; a deterministic 1D convex quadratic made it 13.95% worse. |
| Generic scalar extrapolation | Low priority | FedExP/FedExProx prior art and prior Yeto scalar methods do not support another paid search. |

### BC-MP-AdamW mechanism and discovery outcome

At a broadcast, the learner applies
`merge_alpha * live_local + (1-merge_alpha) * received_global`. Thus
`merge_alpha=0` is a full received-global overwrite, `merge_alpha=1` retains
the live local fragment, and `exp2-53a`/`exp2-53a2` use a 0.5 blend. The learner
retains that fragment's local Adam `exp_avg`, `exp_avg_sq`, optimizer clock,
and scheduler. The retained adaptive state can therefore describe a
pre-broadcast trajectory at a changed parameter point.

The current implementation is shadow-only. At the first post-broadcast
optimizer boundary, after gradient all-reduce and global clipping and before
the factual AdamW step, it reads the fresh clipped gradient `g`, the
bias-corrected inherited first moment `m_hat`, and the diagonal preconditioner
`P` implied by AdamW's upcoming second-moment update. For fragment `f`, it
computes

\[
N_f=\sum_{j\in f}\langle g_j,P_j\widehat m_j\rangle,
\qquad
D_f=\sum_{j\in f}\langle g_j,P_jg_j\rangle,
\]

\[
a_f=\operatorname{clip}_{[0,1]}(N_f/D_f).
\]

It evaluates three counterfactual raw first moments: ray uses
`(1-beta1^t) a_f g`; slab uses
`(1-beta1^t) (m_hat + (a_f-N_f/D_f)g)`; reset uses zero. The slab is the minimum
gradient-parallel correction that clamps current preconditioned work while
retaining the `P`-work-transverse component. Ray deletes that component; reset
is the generic history-deletion control. All three preserve the factual second
moment/clock/scheduler and include ordinary AdamW weight decay in reconstructed
displacements.

None of these counterfactuals is applied in the live run. The logger retains
candidate-minus-stock directions and scores them against a later clipped
gradient while the factual learner continues with stock AdamW. The local
arithmetic is O(fragment size), adds no forward/backward or communication, but
host/GPU synchronization, copies, and JSON I/O can still perturb arrival timing
in an asynchronous system.

The first screen is an H16 stock-AdamW run with shadow instrumentation that
does not mutate local factual model or optimizer state. This is not a claim of
system-level timing equivalence: copies, scalar synchronization, and writes can
change when asynchronous broadcasts are observed.

Discovery analysis requires at least 128 resolved nonfallback events, at least
24 per fragment, at least 95% resolution, and coverage of all 16
learner-by-fragment cells. Candidate-specific intervention is `a<0.9` for ray,
`abs(a-a_raw)>1e-12` for slab, and raw reset moment-change L2 above `1e-30`
for reset. The frozen gates are at least 25% intervention, median
candidate-versus-stock displacement at least 5% of the stock norm, mean
next-gradient cosine above 0.02, positive rate at least 60%, at least three
positive fragment means, and a simultaneous 95% lower confidence bound above
zero. The confidence calculation uses 20,000 paired circular moving-block
bootstrap draws over global versions with block length 8 and a max-error
family-wise correction across ray, slab, and reset.

The logger's `shadow_wall_s/active_wall_s` is host-hook self-time, not a true
end-to-end slowdown. The analyzer uses the worse value when a matched
shadow-on/off timing is supplied; without that control, the hook ratio is only
a fallback gate. The frozen threshold is at most 2%.

The corrected discovery `exp2-53a2` completed all 340 outer commits and wrote
458 shadow events, 1,368 candidate resolutions, and eight explicit drops. Of
423 nonfallback shadows, 421 resolved completely, for a resolution fraction of
`0.9952718676122931`; every learner-by-fragment cell was nonempty and each
fragment had 103--110 resolved events. Aggregate tracker self-time was
`118.45827457100307 / 6984.481642412 = 0.016960209881816662`, below the frozen
2% ceiling.

The immutable discovery decision is **`NO_GO`**. Ray and reset cleared the
resolution, coverage, intervention, action, positive-rate, positive-fragment,
overhead, and multiplicity-adjusted-LCB gates, but their mean next-gradient
cosines were only `0.0015342625064343942` and `0.0015337018882351817`, far
below the preregistered `>0.02` requirement. Slab's mean was
`0.0004413715123634821`; it also missed the 5% median action and 60% positive
rate gates. No policy was selected, so the blinded confirmation was not
launched and the CRN micro-fork is not triggered by this family.

The first frozen analyzer failed closed before scoring because it incorrectly
required initialization broadcast version zero to have one committed fragment
owner. Version zero is the common initial broadcast and legitimately appeared
across all fragments; it was the only multi-fragment version. Before any
policy summary or bootstrap was computed, the integrity rule was amended only
to exempt version zero. Positive versions still fail if they map to multiple
fragments. The amendment, test, analyzer hashes, and chronology are preserved
in `docs/optimizer-reports/exp253-analyzer-amendment.md` and the GCS analysis
prefix. The resulting selection self-hash is
`cdd65b88eead63728ce8f237d84b8cd3dae1567d9ae16141638f71f3e79212e6`.

Discovery is analyzed through a `bcmp_analysis_run_v1` manifest that pins the
phase/completion status, source commit, image/model/data/family digests,
timestamps, seed, learner/fragment design, and overhead evidence. If discovery
selects one policy, its selection manifest must be written and hash-frozen
before opening the committed confirmation seed. Confirmation is a fresh frozen
single-policy shadow run, not another adaptive search.

The literature red-team narrows the claim. Stale or mismatched adaptive state
under federated synchronization is prior art in
[FedGaLore](https://arxiv.org/abs/2602.01746),
[DES-LOC](https://arxiv.org/abs/2505.22549), and
[FedAdamW](https://arxiv.org/abs/2510.27486). The reviewed sources did not show
the exact zero-communication per-fragment preconditioned-work slab used here,
so that narrow construction may still be new, but no novelty claim is made.
Ray is mechanistically weaker because it deletes the complete transverse first
moment, while slab makes the minimum parallel correction required to clamp
current preconditioned work. Both leave stale second moment `v` untouched.

A positive next-gradient shadow cosine is only a first-order, off-policy screen
on the stock trajectory. A one-dimensional quadratic counterexample has cosine
`+1` even though the finite candidate step overshoots and increases loss. A
passing discovery and confirmation therefore authorize only an offline
same-state common-random-number micro-fork.

The executable prototype captures the current group plus eight future groups:
the stored clipped gradient is update +1; future group 1 is evaluated
side-effect-free at +1 and then trained; future groups 1--7 produce updates +2
through +8; future group 8 is evaluated side-effect-free at +8 and is never
trained. It replays stock, slab, ray, and reset from identical model,
optimizer, scheduler, and RNG state without a syncer. A separate fixed held-out
probe is an alternative production design, not an implemented prototype
feature; choose and freeze one endpoint contract before inspecting outcomes.
The candidate must beat stock at horizon 8 and, for a projection-specific
claim, hard reset, without reversal or norm spikes.

The micro-fork is not implemented in the current repository or BC-MP source
commit. A model-generic PyTorch/AdamW prototype now exists under
`/tmp/bcmp_crn_prototype`, but it is not connected to Yeto's loader or learner
loop. Its four unit tests cover atomic/tamper-checked capture, exact restoration,
stochastic Dropout/BatchNorm replay, branch isolation, and order independence;
its ray/slab/reset raw moments match commit `9f9edad...` bit-for-bit. The Yeto
integration remains deliberately restricted to world-size 1, rank-0 LoRA/DDP
ownership, AdamW, standard cross-entropy, no control variate, and no
partial-event resume. Future CPU batches must be copied before `.to(device)`
without advancing or rewinding the live iterator.

For the actual four-learner, four-fragment run, 32 balanced events require two
events in every learner-by-fragment cell (`4*4*2=32`) or another explicit
allocation that truly totals 32. The prototype integration note's two-owner,
first-event rule yields only eight events when there are four fragments; do not
use its stale “16 fragments” arithmetic. Freeze the allocation outcome-blind.

For Qwen3.5-9B LoRA-r2, a content-addressed frozen bf16 base is about 16.678
GiB; each mutable event is about 82.56 MiB core or 85--90 MiB budgeted. Thirty-
two events therefore need about 2.66--2.81 GiB incrementally, or 19.33--19.49
GiB including the one frozen base. Naively duplicating the full model per event
would require about 608 GB and is prohibited. Each replay event entails 28
future training forward/backward steps across four arms, four stored-gradient
+1 steps, and held-out probes.

Only after that causal replay may a live frozen candidate be considered
against a fresh scale-matched stock control. Passing a shadow gate alone never
authorizes an active optimizer arm or changes the production SGD-0.28
recommendation.

### Round-two invention screens

Seven additional mechanisms were ranked before spending more GPU budget. Each
was audited against the exact production sign, responder order, RDA merge, and
outer f32 SGD-0.28 update. Seed 223 was development; every development decision
was written and hashed before seed 239 was opened.

| Candidate | Retained-tape result | Decision |
|---|---|---|
| JK-RDA delete-one jackknife | Production reconstructed bit-exact 76/76 on each seed. Mean held-out cosine gain was `-9.409e-5` on seed 223 and `-1.0297e-4` on seed 239; 0/4 worker folds and 0/4 fragments were positive on both. Median action was only about 1.67%. | Kill; no GPU arm. |
| HMC-AdamW endpoint proxy | True Adam metric state was not retained. A preregistered prior-only proxy had mean gain `1.236e-6` and `2.558e-6`, zero actions above 5%, only about 0.024 cross-worker log-metric dispersion, and made predicted endpoint dispersion about 10.3% worse. | Kill the proxy; factual HMC remains unidentified. |
| NG-TR midpoint Richardson | The archives contain no H/2 midpoint field or object. Two exact latent paths consistent with every stored endpoint can imply NG-TR actions about 53.13 degrees apart. Both seeds still reconstructed production 76/76. | Retained-tape replay is impossible; capture real midpoints before reconsidering. |
| CAMS-RDA nested merge switch | Exact production 76/76 on both seeds. Nested LOO gain was a stable `+0.00091903` and `+0.00092426`, with 4/4 workers and fragments positive and about 8.7% median action, but both missed the frozen `>0.002` gate by more than 2x. | Kill; reproducible but too small for GPU noise. |
| PC-RDA principal consensus | Exact production 76/76 and held-worker gain about `+0.105` on both seeds, but it rotated the update about 87 degrees, breached the 30-degree p99 cap, and had next-same-fragment gain `-0.21395` and `-0.21177`, with all fragments negative. | Kill; strong same-round consensus is temporally unstable and unsafe. |
| IQM-Merge robust order statistic | Exact production 76/76. Held-worker gain replicated at `+0.016378` and `+0.016510` with 4/4 workers/fragments positive, but sealed next-same-fragment gain was `-0.025497` and `-0.025255`, with 0/4 positive fragments and about 47.3% median action. | Kill; same-round held-worker prediction is not a safe optimizer objective. |
| CPER-SGD causal prequential router | Exact production 76/76 on both seeds. A ledger using only previously sealed next-fragment evidence routed baseline 76/76 times, with zero action and zero gain. The rejected barycenter and secant experts replicated strongly negative mean advantages: about `-0.0602` and `-0.0218`. | Kill; safe abstention proves the router rejected harmful temporal experts, but an optimizer that always returns SGD-0.28 has no effect to confirm. |

The checked-in tree retains this summary table but not the complete
preregistration/executable/output bundles for the earlier JK/HMC/NG-TR/CAMS/PC/
IQM screens. Treat those rows as summarized historical evidence until their
complete bundles are archived. The full CPER-SGD artifact set is durably
archived at
`gs://yeto-exp2-52-model-training-497007/optimizer-screens/cper-sgd-20260714`;
the earlier screens do not yet have equivalent durable object prefixes.

These results leave exact outer SGD-0.28 as the production optimizer. They also
sharpen capture requirements: HMC needs bias-corrected `exp_avg_sq`, clocks, and
first post-broadcast gradients; NG-TR needs exact full-f32 H/2 and H endpoints
with identity metadata. Neither quantity may be fabricated from endpoint norms
or adjacent rounds.

Earlier source worktrees also contained proof prototypes for Richardson,
harmonic scalar preconditioner consensus, and delete-one jackknife algebra.
`OptimizerRound2.lean` is not in this release branch, so those prototypes are
not durable evidence here and are deliberately not counted in the release
build. The replay results, not local algebraic identities, determine the
optimizer decisions.

## Formal results and limits

Lean 4.31 with Mathlib was used for small mechanism results. The release root
imports the checked-in modules and `lake build` succeeds with no `sorry`,
`admit`, or newly declared project axiom. Dedicated audit targets report only
Mathlib's conventional `propext`, `Classical.choice`, and `Quot.sound`
dependencies. None is a theorem about language-model loss.

### Transverse quadratic correction

For a quadratic with true gradient `g=H theta`, choose a displacement direction
`p` with `p^T g=0` and `p^T H p>0`. Write the corrected optimizer displacement
as `u_corr=eta g+alpha* p`, with parameter update `theta <- theta-u_corr`. The
exact line minimizer is

\[
\alpha^\star=-\eta\frac{g^THp}{p^THp}.
\]

Relative to the same-state SGD displacement, Lean gives the exact loss gap

\[
-\frac{\eta^2(g^THp)^2}{2p^THp}\le 0.
\]

There is an exact `eta=0.28` witness with
`H=[[2,1],[1,2]]`, `theta=(1,0)`, and `p=(1,-2)` whose quadratic gain is
`0.0588`. The theorem needs a true same-state gradient/HVP and positive
curvature; a DiLoCo pseudo-gradient does not automatically satisfy it.

### SCAFFOLD scale fit

The least-squares scale `s* = N/D` is the unique scalar projection when
`D>0`. Lean proves this algebraic mechanism identity. It is not a loss theorem,
especially when the two update streams came from different closed-loop states.

### Stylized BC-MP ray algebra

For a two-dimensional ray repair with a positive diagonal preconditioner and
positive fresh-gradient energy, Lean proves that repaired preconditioned work
lies between zero and one fresh-gradient unit, preserves in-range scalar work,
and makes the next bias-corrected first-moment contribution a coefficient
`k g` with `0<=k<=1`. Consequently that isolated contribution has a
nonpositive current-gradient directional derivative for nonnegative learning
rate.

This formalization includes reset as the endpoint `a=0`, but it does not model
the slab, stochastic gradients, decoupled weight decay, the possibly stale
second moment used to define `P`, scheduler/clock semantics, or the full AdamW
step and asynchronous system. It also proves an SPD two-dimensional quadratic
counterexample in which deleting a useful stale transverse component is
strictly worse in finite loss. The honest formal claim is bounded first-order
ray work, not uniform finite-step dominance or correctness of the complete
BC-MP implementation.

### Restart-Sum/RS2 no-go

The positive scalar SGD theorem says that under restricted nonoscillatory PSD
conditions a sum of restarted local displacements can dominate the sequential
gain at outer `eta <= 1/2`. That theorem does not instantiate Yeto's persistent
AdamW, scheduler, clipping, LoRA factorization, fragment coupling, or nonlinear
RDA merge.

An exact deterministic PyTorch counterexample uses `f(x)=x^2/2`, H16 split
8+8, inner LR 0.001, AdamW betas `(0.9,0.999)`, epsilon `1e-8`, weight decay
0.01, and outer eta 0.28. RS2's final quadratic loss is 13.95% worse than the
ordinary trajectory. This kills an immediate Qwen run.

### CRP residual accounting and non-identification

`CRPAccounting.lean` follows the executable sign convention exactly:
`r = Q - G` and a full residual is added to stock, so `G + r = Q`. It proves
the generic subtractive squared-norm identity and the additive identity used
by CRP,

\[
\lVert G + \lambda r\rVert^2 = \lVert G\rVert^2
  + 2\lambda\langle G,r\rangle + \lambda^2\lVert r\rVert^2.
\]

The finite-coordinate theorem exposes the missing stock/proposal cross term.
A one-dimensional constructive counterexample has equal stock and residual
norms but opposite cross terms and corrected squared energies `0` and `4`.
Therefore the retained BC-MP scalar norms do not identify a CRP correction.
The direct proof/audit target and 1,905-job module build pass; the final
aggregate job count is recorded with the release verification below. These
are accounting and impossibility results, not convergence or superiority
theorems.

## Statistical and provenance rules

All future candidate effects are positive-is-better paired losses:

\[
D_{s,w}=L_{\mathrm{live\ SGD\mbox{-}0.28},s,w}-L_{C,s,w}.
\]

Every new claim requires a fresh live SGD-0.28 control with the same code,
image, initial state, data/order, seed, H, tokens, LoRA shape, inner optimizer,
quorum, delta correction, and evaluation set. Historical absolute losses must
not be mixed across runs.

This rule matters because the historical vector repeatedly called
“SGD-0.28” (`1.351855`, `1.357837`, `1.380456`) came from H-sweep commands at
outer LR 0.175 for at least H16 and H256. Candidate-specific fresh controls are
the source of truth.

The product gate should define the core workloads as standard H16, H64, and
H256. Stress cells such as heterogeneous H64 or high inner LR cannot be chosen
post hoc as a second core win. A development candidate advances only if it has
at least one core gain over `0.018`, no core regression worse than `0.009`, and
a plausible second core workload. Final confirmation should use five fresh
paired seeds for one frozen candidate, not many adaptively selected candidates.

The practical `0.009` margin is a frozen decision margin, not an empirically
measured training-seed standard deviation.

## EXP2-54 exact-state optimizer substrate

EXP2-53 closed with `NO_GO`; exact outer SGD-0.28 remains the production
recommendation. The next campaign therefore starts with measurement, not a
new live optimizer. Its implemented purpose is narrower than the frozen replay
schema: collect exact per-learner AdamW midpoint/end evidence and join it to
committed wire use, without reconstructing unavailable state from endpoints.
The intended downstream candidates remain:

- **MTRF**, a metric/temporal Richardson force estimate using exact anchor,
  H/2, and H parameters plus bias-corrected first/second moments and optimizer
  clocks; and
- **MSTP**, a moment-space temporal proposal with a per-tensor turn-radius and
  a norm graft back to the production update.

The frozen preregistration, capture schema, and theory report are
`experiments/optimizer/exp2-54-exact-state-prereg.md`,
`experiments/optimizer/exp2-54-capture-schema.md`, and
`docs/optimizer-reports/exp2-54-theory-report.md`. Their source drafts were
hashed before implementation (`2bee9a...`, `518d2f...`, and `fdb45c...`);
materialized repository hashes and the executed source commit must be recorded
in the run manifest before any GCP launch.

### What is captured

The opt-in learner path accepts only native no-scaler `torch.optim.AdamW`, LoRA
parameters stored in fp32, an f32 wire, `merge_alpha=0`, no control variate,
no artificial lag, no reconnects, and one constant even H. At explicit
boundaries it records:

- the first post-broadcast, post-allreduce, post-clip gradient with exact raw
  AdamW moment state and per-parameter step clocks;
- exact fragment parameters, optimizer state, scheduler state, step history,
  LR mass, and decoupled-decay sums at anchor, H/2, and H; and
- the immutable f32 push bytes, a canonical window UUID, monotone learner
  attempt serial, retry identity, and SHA-256 digest at the real push site.

Every tensor artifact has a sidecar and is indexed by an atomic manifest. The
capture validator rejects missing, extra, temporary, malformed, non-finite,
or checksum-mismatched files and enforces the frozen caps and minimum complete
boundaries.

The transport join is also opt-in. Legacy push message type 4 is unchanged;
audited message type 12 adds the UUID, attempt serial, and SHA-256 of the exact
raw tensor-byte tail. When transcript mode is on, the Rust syncer refuses
legacy pushes, verifies the digest before decoding/quorum admission, and emits
a fresh fsynced JSONL transcript. A commit is evidence only when a
`syncer_round_commit_v1` responder points to its earlier accepted attempt with
the same learner, fragment, UUID, serial, versions, counters, weight bits, and
payload digest. Merely reaching `admitted_pending` is not production use.

This is still not full CRN state. It does not yet capture the whole model,
complete optimizer, RNG streams, data iterator/order, and sealed next-8-step
evaluation state needed for a causal k0/k8 loss claim. Until those are joined,
MTRF/MSTP output is a secondary mechanism screen and cannot establish that an
optimizer beats SGD-0.28. The recorder also writes per-learner `.pt` envelopes,
while the frozen replay skeleton consumes joined `state.npz`, `optimizer.json`,
and `crn.json` boundary bundles. A checksummed committed-responder materializer
from the envelopes/transcript to that schema is not implemented yet. Thus the
current acquisition makes the necessary local state observable, but does not
by itself make even direction replay executable; the materializer is required,
and causal k0/k8 evaluation additionally requires the missing full CRN layer.

### Capture-v2 and the joined replay boundary

The v1 artifacts are therefore a measurement substrate, not a causal optimizer
evaluation bundle. A fail-closed materializer may produce a **direction-only
joined bundle** for MTRF/MSTP only when it preserves the transcript's exact
commit/responder order and weight f64 bits, proves a common non-stale anchor,
preserves per-tensor clocks and optimizer-group semantics, removes or verifies
decay exactly, and reproduces the committed SGD-0.28 broadcast digest. An
ambiguous or stale anchor, unsupported group heterogeneity, unknown decay
contribution, missing responder, reordered responder, or byte mismatch must
produce `UNIDENTIFIABLE`; it must not produce a partial bundle.

V1 cannot support a causal k=0/k=8 loss claim. It does not preserve all
trainable fragments and model buffers, the complete optimizer state keyed by
parameter name, CPU/CUDA/Python/NumPy RNG, a restorable loader position, the
next eight actual update groups, or exact syncer pre/post boundary state. The
validator also proves the authoritative join but currently reduces it to
boundary keys instead of exporting a normalized
commit → responder → push → Richardson index. None of this missing state may be
reconstructed from endpoints or added after outcomes become visible.

| Candidate | Direction construction | Maximum honest use of v1 | Additional capability before a loss claim | Campaign |
| --- | --- | --- | --- | --- |
| MTRF | Exact responder join; anchor/H/2/H parameters; per-tensor Adam moments, metrics, clocks, LR mass, decay accounting; production RDA | Conditional direction-only replay after every fail-closed gate above | Full learner restore, syncer boundary state, next-eight groups, fixed evaluation object, RNG restore, isolated k0/k8 replay | EXP2-54; unscored until v2 |
| MSTP | The MTRF substrate plus the exact half-path and factual merged directions | Conditional direction-only replay under the same gates | The same full CRN restore/evaluation bundle | EXP2-54; unscored until v2 |
| PTI-SGD | Ordered same-fragment factual directions plus a hash-chained causal prequential ledger | At most a historical direction stream after exact materialization | Sealed shadow outcomes, full restore/evaluation, and a separately frozen preregistration | Separate hypothesis |
| CPLG-SGD | Two consecutive same-fragment directions, forward tangents, exact directional-sphere transport, and delayed causal scores | Executable full-vector shadow replay only; the action-capable Rust selector is implemented but no live experiment is authorized | Full policy-state restore, identical future-eight groups, fixed evaluation object, and isolated k0/k8 replay | Separate post-PTI hypothesis; unscored |
| CRP-SGD | Delayed resolution of individually tiny proposal residuals, then a bounded orthogonal pulse from a per-fragment FIFO bank | At most a historical causal-tape mechanism screen after exact materialization | Full restore/evaluation, a separately frozen proposal source, and isolated CRN replay | Separate hypothesis; unscored |
| CFLX-SGD | Exact global stock trial point, model autograd, and disjoint proposal/validation/audit streams | Not identifiable from v1 learner envelopes | Syncer global state, full model restore, restricted probe oracle, equal live-arm probe cost, and CRN replay | Separate campaign |

The shared replay layer will be capability-based. Every policy must emit the
same sealed outer action—fragment, pseudo-gradient bytes, outer-LR bits,
resulting fragment bytes, and configuration hash—while declaring requirements
such as `midpoint_adam`, `same_fragment_history`,
`global_boundary_state`, `model_autograd`, `proposal_stream`,
`worker_restore`, and `crn_train_k8`. Immutable inputs, actions sealed
before evaluation, and append-only outcomes are separate objects; losses are
never written into `state.npz`.

The bounded implementation order is:

1. export a checksummed committed-boundary index with exact event identity,
   responder order, attempt/window UUID, weight bits, and push/Richardson
   object hashes;
2. build the atomic direction-only materializer and require byte-exact
   production RDA/SGD-0.28 broadcast parity;
3. make the qualifier schedule deterministic with a true response barrier and
   move serialization, hashing, and fsync behind a bounded fail-closed writer;
4. add capture-v2 at the exact H endpoint, including all mutable learner state,
   buffers, named optimizer/scheduler/scaler state and RNG, then passively
   attach the next eight actual update groups as they are consumed;
5. add a syncer replay shard containing exact pre/post affected-fragment bytes
   and outer state, then a policy-agnostic isolated evaluator whose A/B and B/A
   executions have identical hashes and losses; and
6. run corruption, stale-anchor, responder-order, repeated-replay, missing
   capability, and branch-isolation tests, followed by a new matched GPU
   qualifier because v2 changes serialization and runtime cost.

The first part of that sequence is now implemented in commit
`47ad73594bbb9ae93d0d2220798b319e46fce6b6`. The existing full campaign
validator, rather than a second parser, exports a canonical checksummed
`optimizer_state_capture_committed_boundaries.json`. Each boundary preserves
authoritative transcript event order and exact responder merge order and binds
the admitted source attempt to exact push and Richardson object paths and
SHA-256 digests. The index also binds the source transcript, capture session,
common layout, validation expectations, and source-tree manifest. Validation
now rejects malformed/nonfinite/negative f64 weight bits, decimal/bit or Rust
token-weight-formula disagreement, malformed broadcast digests, Python
bool-as-int aliases, cross-learner UUID aliasing, same-payload retry aliasing,
responder reorder, and stale output after a failed rerun.

The capture-v2 storage foundation is commit
`ba16256e65586ef97ca3020ec5530e0443c62574`. It provides immutable SHA-256
objects and manifests, fsynced same-directory temporary writes, atomic
no-replace publication, exact verification, race-safe deduplication, canonical
strict JSON, logical/physical/deduplicated-byte accounting, and fail-closed
audits for corruption, missing objects, symlinks, temp files, unexpected tree
entries, false accounting, and orphans.

Commit `3c07d52fa0ebd3588a9e67a095208d38a4001083` adds the first exact
capture-v2 tensor pack over that CAS. Named fp32 trainables and supported exact
optimizer tensors are copied into canonical little-endian contiguous bytes,
ordered by category and ASCII name, and described by strict dtype, shape,
offset, size, and per-tensor SHA-256 metadata. Exact clocks are sorted
nonnegative int64 values. Tests cover insertion-order and stride independence,
all supported optimizer dtypes, signed-zero and NaN payload bits, scalar/empty
shapes, independent decoded storage, payload/manifest/per-tensor corruption,
wrong roles, reordered descriptors, bool-as-int aliases, and no-write input
failures. This remains a local POSIX storage codec: live learner/syncer hooks,
restore manifests, transactionally quiesced endpoint snapshots, bounded
asynchronous writers, CRN linkage, and cloud publication remain separate
gates.

Commit `af46f614c7cde0853d2ef99c6e6a042f8f4b86d9` added the first strict
learner-endpoint manifest over the packs. It binds contiguous all-fragment
pack identities and versions, mode, model buffers, scheduler/scaler metadata,
Python/NumPy/Torch CPU and indexed CUDA RNG objects, and an explicit
future-group union. `complete` requires exactly indices 0--7 and no reason;
`incomplete` requires fewer than eight refs plus a bounded nonempty reason.

Commit `cd90d61e39936efe0ff305ef4799767e10c90647` closes the endpoint identity gap and adds the paired syncer
boundary contract. Every endpoint now requires canonical capture-session and
window UUIDs, learner ID/rank/local step/active fragment, and an exact
provenance object bound to source commit, image ID, and model/data/config
digests. A syncer manifest binds session/commit/fragment and pre/post versions,
canonical merge and outer configurations, ordered endpoint responders and
their exact f64 weight bits/payload hashes, plus fixed-role pre-fragment,
post-fragment, outer-state, and broadcast CAS objects. Publication and load
cross-check endpoint session, active fragment, pre-version, identity, and
provenance. Reconstruction code receives verified endpoints, pre-fragment,
outer state, and configurations, but not the expected outputs; it passes only
if independently read captured post-fragment and broadcast bytes both match
exactly.

This remains a storage and reconstruction-verification layer. Opaque-state
codecs, transactionally quiesced live learner/syncer hooks, restore/apply,
policy actions, isolated CRN evaluation, and cloud publication are separate
gates and are not inferred from a manifest.

Commit `317464201d314491d0b79e9c22705a0801d780ee` adds the policy-agnostic action boundary. A frozen policy
manifest binds its source/config objects and declares a subset of the closed
capability vocabulary (`same_fragment_history`, `midpoint_adam`,
`global_boundary_state`, `model_autograd`, `proposal_stream`,
`worker_restore`, `crn_train_k8`). A sealed action binds one verified syncer
boundary, exact stock and selected pseudo-gradient objects, outer-LR bits,
resulting fragment, decision/config hashes, and its required capabilities.
Stock fallback must select the exact stock object; a non-stock action must
differ. No outcome fields are legal in an action. A separate immutable outcome
manifest binds k0/k8/evaluation objects and finite losses after the action has
already been sealed. Policy publication/load also decodes responder f64 bits
and rejects NaN, infinity, negative values, `+0.0`, and `-0.0` without
inventing a weight normalization rule.

Subsequent policy hardening binds capability claims to the actual boundary
evidence: global state must exist, worker restore requires all packs, buffers,
scheduler/scaler and RNG objects, and `crn_train_k8` requires exact future
groups 0--7. Boundary and action learning rates are finite positive f64 bits
and must agree exactly. A stock result must be the captured post-fragment; a
non-stock result remains formula-owned. The decision is an exact CAS object,
not a caller-supplied digest. Finally, sealed actions no longer accept free-form
reason prose. Their action/fallback fields use a closed code vocabulary with
kind/code compatibility, preventing loss values or outcome claims from being
smuggled through a reason string.

Commit `ecda3ee` adds a pre-outcome CRN authority. A campaign index admits one
exact plan per boundary. Each plan binds the boundary, stock/candidate actions,
exact endpoint/responder payload, evaluation object, ordered future groups
0--7, CUDA-stream count, horizons `[0,8]`, A/B then B/A schedule, evaluator
source/image/config provenance, and reserved attestation/outcome identities.
Production compatibility is exact SGD/Nesterov at f64 LR `0.28` with positive
zero momentum. The evaluator reloads that authority before any callback, runs
four fresh traces, retains restore/application/group/final state hashes and
exact finite loss bits, and publishes one atomic paired outcome only when the
two execution orders agree. Generic one-arm policy outcomes are explicitly
non-admissible. This closes post-hoc plan and one-arm-outcome substitution; it
does not supply the still-missing live capture-v2 producer/restore hooks.

Seed 223 stays locked through this implementation. Acquiring more v1 envelopes
now would expose development data without enabling the preregistered finite-loss
decision. It may open only after the v2 schema, code, commands, policy
interfaces, source/image/model/data hashes, and matched capture qualifier are
immutable.

### Qualifier and GCP sequence

The first full four-responder capture-v2 GPU qualifier is planned as a
behavior-preservation test, not a candidate score. One command runs matched
four-learner `capture_m4_off` and
`capture_m4_on` arms with strict quorum 4, four fragments, H=4, f32, no delta
correction, `merge_alpha=0`, native AdamW, and outer SGD-0.28. The gate requires
exact committed push/transcript joins, exact deterministic artifact parity
where the runtime format permits it, identical final state/evaluation, and no
more than 2% exact producer-side steady-state overhead. Each syncer records a
strict commit sequence and monotonic nanosecond timestamp at the commit point;
the metric is the interval from commit 1 to commit N, covering the same
commits 2..N in both arms. Startup is excluded because the fixed OFF-then-ON
order would otherwise charge only OFF for cold model/data/CUDA caches. Both
arms now use explicit AdamW and zero
reconnects; the ON-only generated-command difference is the exact capture
option allowlist. A one-A100 serialization canary is not enough to qualify the
four-responder join.

The parity verdict is not the only retained evidence. A successful gate emits
`optimizer_state_capture_parity.inputs.sha256`, a portable manifest covering
every probe index/payload, both event tapes, both final checkpoints and export
trees, the ON wire transcript, and the result rows consumed by the decision.
The gate recomputes the exact interval from those tapes and rejects a missing,
duplicate, reordered, or non-monotonic commit. OFF/ON ordered
`(commit_seq, step, fragment)` identities must match exactly, and their set
must equal the fully sampled probe/transcript commit set. The harness executes
that manifest as well as the verdict sidecar. Before the runner starts, it
also verifies the image's model and data manifests, copies those manifests plus
runtime/image metadata into `input-provenance/`, and seals that directory in
`input-provenance.sha256`.

The first live identity, `exp2-54-smoke`, instance ID
`5674137355695134252` and boot-disk ID `3761493731937326636`, reached that
input-integrity gate but did not start training. The runner renderer had
manually nested a single-quoted Bash body containing single-quoted checksum
diagnostics; the outer shell split the intended program and the remote log
failed with `checksum: line 4: is: command not found`. The verified model and
data manifests had already passed. The harness classified the run incomplete,
stopped only its recorded processes, synced the provenance and failure log,
wrote and round-tripped `abandonment.json` with SHA-256
`c137f2b3ae9621f7f41ca32681be253ce4378dce1aa1a05eae79f80062a39a2b`,
and deleted exactly that authenticated VM and auto-delete disk. Both provider
lookups now return not found. The failed run ID, local state, and GCS prefix are
permanently quarantined and will not be reused.

The repair constructs each complete inner runner/uploader program first and
quotes it exactly once with `shlex.join`. An executable regression traverses
the same outer-shell → `bash -c` boundary, forces the apostrophe-containing
missing-manifest branch, and requires exit 14 plus an atomic `runner.exit`.
At that repair commit, all 59 then-present harness tests and the then-current
783-test repository `tests/` target passed, and an independent renderer audit
verified the exact argv and exit-code protocol. The pushed fix commit is
`69cff38369041cef8d1bddc9c23a9ecb05843a90`.

The fresh identity `exp2-54-smoke-r2` passed both input manifests and completed
the OFF arm plus all 16 ON commits. It then failed before validation because
the absolute validator subprocess imported an older installed `yeto` package
instead of the pinned checkout and raised `ModuleNotFoundError` for
`yeto.optimizer_state_capture`. No candidate result was produced. The harness
preserved the incomplete tree under
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r2`, recorded exact VM
ID `5980313032922468927`, disk ID `3539552211744559679`, and abandonment
SHA-256
`4b9d3771aee6c63d5929500af0f70152c038a24e5590ca2a7dfc23f1a8728d77`,
then deleted only those authenticated resources; both now return 404. Commit
`afa0b07c379e8bce4140e4862da9861b7a2c8e74` fixes every child Python
subprocess by prepending the pinned checkout to `PYTHONPATH`. Its executable
regression places a deliberately stale fake `yeto` package first in the
parent environment and proves that the child still imports the checkout.

The next immutable identity, `exp2-54-smoke-r3`, pinned that fix and ran to the
actual scientific gate on four Spot A100s. Its capture validator passed: 16
authoritative joined boundaries, four per fragment, 64 primary attempts, 81
transcript events, 202 verified learner artifacts, no missing/unlisted/temp
artifact, and exact tree-manifest SHA-256
`7424d023e0abdc4b71fdea3a96991fbfef8b485faeaa0f2ce18878d7f50d3a7d`.
The four learner artifact-byte totals were `1,269,000,908`, `1,252,639,051`,
`1,252,639,051`, and `1,269,000,908`; the retained GCS run tree is
`7,821,538,975` bytes (`7.284 GiB`). This proves that the repaired transport,
capture, checksum, and validation paths execute end to end. It does not prove
behavior preservation.

The exact OFF/ON parity gate rejected r3. Responder arrival order differed
from the first commit, which the parity comparator intentionally
key-normalizes, but the first substantive metadata divergence was commit/step
6, fragment 1,
learner 2: OFF answered at local step 10 while ON answered at local step 11.
Capture therefore changed how much asynchronous local work completed before
the broadcast boundary. The final syncer checkpoint and exported adapter
bytes consequently differed. Descriptive eval losses were `1.3693659077`
OFF and `1.3677875564` ON, but that difference is invalid as an optimizer
comparison and is not promoted. The raw commit-1-to-commit-16 intervals were
`18.232708765 s` OFF and `32.728699397 s` ON, a descriptive `79.5054%`
increase; the parity program correctly refused to call this the threshold
metric after the earlier trajectory mismatch. Its signed FAIL artifact has
SHA-256
`515b800b25f2d9bef088748fe54c6c192cc3b54d195a6a5a8165dcd6cde02fab`.

The r3 evidence remains at
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r3`. Exact VM ID
`2042458077012922652`, boot-disk ID `6741716953892360476`, source-image ID
`7290368630472593484`, and ownership nonce `b0d2d40f3d61ac76` were recorded.
The harness synced and round-tripped the failure evidence, wrote abandonment
SHA-256
`c5f8e04b9de8768f8c85a1d2882bd4be1048ae1ba544fe30d9e85331e7fa804c`,
and deleted exactly that VM and auto-delete disk. Both return 404. The unrelated
`instance-20260526-guiflow` in `asia-east2-c` remains untouched.

This failure is not a reason to relax exact parity. The next qualifier must
use the already implemented `--barrier-sync` response boundary so OFF and ON
cannot take different numbers of inner steps while a merge is in flight. It
must also replace synchronous `torch.save`/hash/fsync on the learner path with
immutable CPU snapshots and a bounded writer queue that blocks rather than
drops evidence. No r4 GPU rerun is authorized until those mechanics and their
failure tests are frozen; another non-barrier v1 rerun would merely measure
the same scheduling race.

Only a passing qualifier can unlock seed-223 H16 **state acquisition**. Its
scientific minimum is 32 complete committed boundaries, eight per fragment,
but capture-v2 must run 36 commits: nine per fragment. The last four are a
predeclared drain tail so every one of the first eight selected endpoints per
fragment can acquire eight strictly future update groups. The drain endpoints
are not added to the scientific sample. Exactly 32 commits cannot guarantee
future-eight data for the final selected endpoints.
It does not unlock candidate scoring. Before MTRF/MSTP development can begin,
the joined-bundle materializer and full CRN capture/restore/evaluation layer
must be implemented, independently tested, and frozen against the existing
schema. Only then do the preregistered mean-k8, Holm-adjusted lower-bound, and
action/safety gates apply. Capture-v2 sizing is currently an engineering
projection, not a measurement: roughly `150--160 MiB` per committed responder,
`9.6--10.3 GiB` for the 16-commit H4 four-learner qualifier, and
`21.6--23.2 GiB` for the 36-commit H16 acquisition including its drain tail. A
one-A100 serialization/restore canary must measure hook latency, writer
throughput, queue high-water mark, RSS/pinned memory, physical unique bytes,
and restore parity before the four-A100 qualifier. Seed 239 remains locked
until one immutable winner, configuration, and analysis hash are selected.
There is no honest winner ETA while full CRN state is absent. No EXP2-54 VM is
currently running; GCP spend is paused after r5h passed correctness parity but
failed the frozen 2% capture-overhead gate at 7.964001%.
Future qualifiers retain a one-hour VM safety envelope so checkout, validation,
evaluation, upload, and teardown do not race provider auto-delete. Development
and a separately authorized confirmation retain two-hour envelopes.

The reusable image is
`projects/model-training-497007/global/images/yeto-optimizer-a100-20260714`,
exact image ID `7290368630472593484`, status `READY`. On 2026-07-14 the
`us-central1` Spot quota was 16 preemptible A100 GPUs with zero usage; the
standard A100 quota was only one. A separate `A2_CPUS` quota was only 12,
however, while `a2-highgpu-4g` requires 48. At `2026-07-14T11:58:08Z` the
least-privilege quota preference
`yeto-a2-cpus-us-central1-48-20260714` requested 48, with trace
`2b5bd8c8-2383-4de4-aa90-44b38f1dbe0c`; Google approved the request to 48 at
`2026-07-14T11:58:10Z`. The doctor now checks both A2 CPU and Spot A100 quota
and refuses launch if either is insufficient. The campaign uses at most four
A100s concurrently and at most eight only if two scientifically independent
jobs later become safe to run in parallel. Boot disks use `pd-ssd`: earlier
ingress profiling isolated the apparent download stall first to a Hugging Face
signed endpoint and then to `pd-standard` unpack/cache I/O, not a shortage of
network bandwidth.

The implementation is assembled in the isolated branch
`experiment/optimizer-state-capture-round3`. At the earlier r3 release
milestone, the frozen tree passed 783 Python tests, 171 Rust tests, Ruff
lint/format, the replay self-test, and all 8,567 then-present Lean build jobs.
Those counts are historical; the final r5h branch verification is recorded
below. The audited r3 capture implementation is commit
`452ebdea30503f10c1eda68e9fdcff704be2a792`; the remote-runner quoting repair
and its executable regression are commit
`69cff38369041cef8d1bddc9c23a9ecb05843a90`; the child-import repair is
`afa0b07c379e8bce4140e4862da9861b7a2c8e74`, which r3 pinned. Independent
launch audits found and closed the reconnect,
cold-start timing, portable parity-input, image-input provenance, and too-short
VM-envelope holes. The checked-in acquisition specs remain locked. The
quota-aware doctor is green and r5h established deterministic one-A100
correctness parity. Development and confirmation remain locked on the failed
end-to-end overhead gate and the still-missing full live capture-v2
producer/restore/evaluator path.

### EXP2-54 r3 parity failure: scheduling, not optimizer math

The completed `exp2-54-smoke-r3` matched run failed the parity gate. The first
substantive OFF/ON difference is syncer commit step 6, fragment 1: OFF records
all four responders at learner local step 10, while ON records learners 0--1
at step 10 and learners 2--3 at step 11. All responders still report the same
fixed `c_steps=4`, `c_tokens=512`, and pushed base version. The difference is
therefore not an oversized H window, token weighting, or an outer-optimizer
calculation. It is the exact local window selected for the push.

Capture was mathematically passive but not schedule-passive. The learner wrote
artifacts synchronously on its training thread, while broadcasts are drained
and applied only at step boundaries. Serialization therefore changed whether
a pending broadcast reset landed before or after the next optimizer step. The
change propagated to later candidates, the final checkpoint, the export, and
evaluation. OFF was not itself lockstep: by commit step 12 its fragment-3
responders were split across local steps 15 and 16. An OFF/ON equality check
alone could consequently accept a matched non-barrier race.

The cost evidence agrees with that mechanism. Learner 2 slowed from roughly
1.17 seconds per step in OFF to 1.81--2.00 seconds per step in ON. The sealed
commit interval rose from 18.2 to 32.7 seconds, about 79.7% overhead. ON
learner 2 wrote 50 artifacts totaling 1,252,639,051 bytes: 18 first-gradient
captures, 16 Richardson windows, and 16 push candidates. Its only recorded
drops were the declared incomplete shutdown-tail states, so the observed r3
failure is not evidence of a capacity drop.

The next qualifier must enable the existing true `--barrier-sync` learner
mode in both arms. Any parity qualification now requires the
`--require-barrier-schedule` contract, which proves the producer trace consists
of complete fragment waves: responders agree on local step/H/tokens/base in
each commit, every fragment in a wave is pushed at one local step, and
successive waves advance by exactly H. This checks observed behavior rather
than trusting an argv label. Parity qualification also requires the
`--strict-writer` contract. It rejects every non-terminal drop, including byte,
event, window, and pending-memory limits; only the three explicitly declared
incomplete-at-close tail reasons remain admissible. The compare driver exposes
these as `--optimizer-state-capture-parity-require-barrier` and
`--optimizer-state-capture-strict-writer`, and refuses invalid flag
combinations or nonpositive caps.

This barrier makes capture latency affect wall time rather than the
optimizer-step schedule. The strict writer contract remains separate from
capture-v2 CAS semantics, so parity qualification does not silently substitute
a new state schema.

### Bounded background capture publication

Commit `0335291` introduced the reusable `BoundedBackgroundWriter`. It accepts
only exact immutable `bytes` payloads with an explicit reservation equal to
their length. One non-daemon worker executes a fixed sink in FIFO admission
order. Both item and byte reservations remain charged while the item is queued
and throughout the sink call, so a slow write or `fsync` cannot free apparent
capacity while the worker still owns the payload. Producers block on full
capacity; no capacity drop path exists.

Close changes admission atomically from open to closing, wakes blocked
producers with a closed-admission error, drains every item accepted before the
transition, joins the worker, and is safe for concurrent callers. The first
sink exception changes the writer to failed, abandons and accounts for every
remaining accepted item, wakes blocked producers, stops the worker, and is
chained from every subsequent submit or close error. Statistics expose current
and high-water reservations and queues, accepted/completed/abandoned bytes and
items, producer block count and nanoseconds, queue-wait nanoseconds, worker
busy/idle nanoseconds, close-wait time, terminal error identity, and worker
liveness. Tests cover FIFO ordering, byte and item backpressure, injected
write- and fsync-like errors, six concurrent producers, concurrent close and
submission races, immutable/reservation rejection, idempotent close, and zero
remaining worker threads.

Commit `e625e0f` integrates that primitive behind the explicit
`--optimizer-state-capture-background-writer` flag without changing the
default synchronous path. The learner serializes an immutable artifact before
admission; the worker owns atomic `.pt` publication, checksum sidecars, file
and directory `fsync`, and first-error propagation. Queue caps block rather
than drop. `close()` drains and joins before `closed=true`; the strict validator
requires accepted/completed byte equality, no abandonment/reservation/queue or
in-flight residue, a stopped worker, and exact artifact-index completeness.

An audit then found that the first integration still hashed artifacts and
published a fsynced manifest on the learner hot path. Commit
`4cb2f18042eddeb7c297bc6f0555f66256a8c489` moves artifact SHA-256 and sidecar
construction entirely to the worker. In background mode it publishes one
initial manifest, keeps hot lifecycle/counter transitions in memory, and
publishes the final manifest only after drain; failure still forces a
diagnostic manifest. Instrumented tests prove that artifact hashes and all hot
phase `fsync` calls run on the named worker thread and that successful manifest
publication occurs exactly at construction and post-drain completion.

The corrected path was then exercised through three uniquely named one-A100
Spot attempts. The first two were execution-environment failures before
training: r5 invoked absent bare `python`, and r5b used the exact venv Python
but its detached non-login PATH omitted Cargo. Both failure records were synced
and their exact instances/disks were deleted. r5c therefore froze an explicit
PATH containing the venv, Cargo/Rust, `/snap/bin`, and system tools, listed the
interpreter/Cargo/Rust executables as required image paths, and passed a
detached `env -i` smoke covering Python imports, CUDA, Cargo, Rust, Git,
GCloud, SHA-256, model, and data before the full provenance scan.

The r5c runner completed both barrier arms and produced a scientifically valid
engineering failure at
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r5c-async-canary`.
OFF and ON had identical eval loss/token `1.4166344593414884`; all 16 ordered
commits, four fragment waves, syncer probe payloads, transcript joins, final
syncer checkpoint, and strict no-drop capture validation passed. ON retained
52 verified artifacts totaling `1,285,362,637` artifact bytes: 20 first
gradients, 16 Richardson windows, and 16 push records. The background worker
accepted and completed every byte with zero abandonment; it spent
`6.876044215` seconds in its sink, while producer capacity blocking totaled
only `0.625183101` seconds.

The exact commit-1-to-commit-16 interval nevertheless rose from
`18.482062741` to `26.342267777` seconds: **42.528830% overhead**, so the
frozen 2% gate failed. This localizes most remaining cost to producer-side
snapshot construction/serialization rather than queue backpressure or disk
publication. Export comparison also reported `adapter_config.json`, but the
only difference was permutation of PEFT's set-valued `target_modules` list;
model loss, checkpoint, and binary payload evidence were identical. The
failure was synced with abandonment record SHA-256
`e00c63b8ad0817e288e7a75b35f64cc6655133a004c49842c1a9e11a1a909535`.
Exact instance ID `3244200075441468763` and boot-disk ID
`1152111786048877915` now return 404. The checked-in four-A100 r6 draft was not
launched and is obsolete as an executable specification: it pins `e99179e`,
retains the failed r5e gate, and predates deterministic commit ordering,
endpoint reuse, and the SSH-safe harness. It must not be launched or edited in
place.

Commit `9c6ff53` closes both non-performance defects exposed around r5c. An
audit of PEFT 0.19.1 confirmed that `LoraConfig.__post_init__` converts a
list-valued `target_modules` to a set. Parity therefore sorts exactly the root
`adapter_config.json` top-level `list[str]` at that field while preserving
duplicate multiplicity. Membership changes, element/container-type changes,
nested fields, other JSON files, other lists, unknown metadata, and binary
payloads still compare exactly. The harness also accepts an optional closed
list of absolute `required_executables`, each already present in
`required_paths`, and exercises every one with `--version` under the exact
detached spec environment before `nohup`. This makes the venv/Cargo/Rust PATH
contract executable without weakening historical specs.

Commit `2d4c9f6` then moves the measured producer bottleneck across the
existing bounded-writer boundary. Background mode now transfers a one-shot,
exclusive, structurally frozen CPU snapshot before building the artifact
envelope. Exact-boundary tensor copies remain on the producer, because
deferring those copies would race live parameters and Adam state; structural
payload hashing, `torch.save`, artifact hashing, atomic publication, `fsync`,
and sidecar construction run on the worker. Queue ownership is conservatively
byte-reserved through sink completion, the worker verifies the exact serialized
size against that reservation, and final artifact accounting remains exact.
The synchronous default is unchanged. In a five-run 64 MiB isolated
post-snapshot benchmark, the producer section fell from median `118.723 ms` to
`0.0816 ms` while total drain remained `114.175 ms`: the work moved off the
hot path rather than disappearing.

Commit `5977360` adds the opt-in `crp_pti_directional` capture profile for the
next engineering canary. It retains exact contiguous f32 fragment vectors at
the anchor, accepted H/2 boundary, and accepted H boundary; exact tensor order;
contiguous accepted-step chronology and LR masses; and the existing
window-to-push-to-authoritative-commit join. It omits first post-broadcast
gradients, Adam moments/raw optimizer state, scheduler/scaler snapshots, and
accumulated decay vectors. On the observed r5c artifact mix this projects
roughly `1,285.3 MB` down to `261.8 MB`, a `79.6%` reduction. Its manifest is
explicitly scoped to `crp_pti_direction_evidence_only` and sets
`capture_v2_restore_complete=false`. It is not stock-boundary authority,
restore authority, a CRN-loss bundle, a policy action, or an optimizer result.
Commit `3111f39` additionally requires the closed full payload/optimizer schema
when validating legacy full captures, so deleting the directional markers
cannot relabel a reduced bundle as full evidence.

The uniquely named r5d one-A100 Spot canary then exercised that combined path.
It retained 32 artifacts totaling `259,962,886` artifact-plus-sidecar bytes,
versus r5c's 52 artifacts and `1,285,362,637` bytes. All 32 jobs completed with
zero abandonment, and final checkpoint, export, and eval loss/token
`1.4166344593414884` were exact across OFF/ON. The sealed parity gate still
failed: OFF's second fixed-H wave committed fragment steps `[5,7,6,8]`, while
ON committed `[5,6,7,8]`. At `(step=6, fragment=1, learner=0)`, the exact probe
metadata consequently had `syncer_global_step=7` OFF and `5` ON. The gate
correctly refused transcript/timing admission rather than canonicalizing that
schedule difference. Diagnostic, non-admitted tape intervals were
`17.708764577 s` OFF and `19.718775923 s` ON, or `11.350375896%` overhead.

The writer evidence localized the leading cause: a four-item reservation high
water and three-item queue high water saturated the configured item cap on
each eight-job H-wave burst. The producer blocked 16 times for
`1.334270645 s`, `66.38%` of the diagnostic interval delta, while structural
freeze took only `0.016468009 s`. All 32 owned jobs together used
`328,076,192` bytes, only `7.64%` of the unchanged 4 GiB byte cap. The failure
tree is retained at
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r5d-directional-canary`;
abandonment record SHA-256 is
`ae959086859a13343f55f0d6de523c73aec58f156bbd1ecf1a0f0a2ab58f5901`.
Exact instance ID `2261800274408238443` and disk ID `8196336138399247723`
now return 404.

Commit `e99179e` hardens strict validation before the conditional r5e retry:
reservation/queue item and byte high-water claims must not exceed their
configured caps, and queue high-water must not exceed reservation high-water.
r5e changes only the item cap from 4 to 32, enough to admit the complete
observed 32-job run while the 4 GiB ownership cap, payload, barrier, exact
parity, and 2% gate remain unchanged.

r5e proved that queue tuning was exhausted. All 32 items again completed with
zero abandonment, but producer capacity blocks fell from 16 to exactly zero;
reservation high water was seven items/`78,649,925` bytes. OFF nevertheless
committed wave 3 as `[9,11,10,12]`, while ON committed
`[9,10,11,12]`, so exact probe parity failed at step 10. Diagnostic intervals
were `18.255600152 s` OFF and `19.564660108 s` ON, or approximately `7.17%`
overhead. Final checkpoint, export, and eval remained exact. The retained tree
is
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r5e-directional-burst-canary`;
abandonment record SHA-256 is
`7a683f248c8b6fc196957a7b6ac9473765016c80e3b31a169123d170c1a6a15d`.
Exact instance ID `6150165584886360488` and disk ID `810480084346202536`
now return 404.

Two orthogonal repairs follow that result. Commit `0d8561b` preallocates the
final contiguous CPU f32 fragment buffer and synchronously copies each live
tensor directly into its final slice, removing per-parameter CPU intermediates
and the second full-fragment `torch.cat` copy. The next manifest measures exact
snapshot calls, bytes, source tensor count, CUDA calls, and total/max time.
Synthetic 16 MiB CPU fragments improved about `1.7--2.0x` for 8--128 tensors;
the A100 effect was measured in r5h below. Commit `2867ed0` adds the default-off
`--deterministic-commit-order` qualifier mode. Four-stream ingress, pipeline
launch, uploads, and quorum gathering remain concurrent, but the serialized
state mutation waits for the minimum in-flight request step. This prevents
ready later fragments from overtaking an older request without canonicalizing
evidence after the fact. It is a declared qualifier schedule, not a claim
about the production completion-order scheduler.

Commit `02ce0b1` then closes a duplicate endpoint-read defect. For one CUDA
fragment, capture now forms one f32 `torch.cat` in declared layout order and
performs one blocking fragment-sized device-to-host transfer. At the accepted
H boundary the original complete CPU endpoint is transferred exclusively to
the artifact worker and one distinct CPU clone is transferred to the learner's
fixed-window response. The learner therefore does not reread live CUDA
parameters to answer the same boundary. The exact f32 wire SHA-256 is computed
before worker handoff, and tests cover tensor-storage separation, two-way
mutation isolation while the worker is blocked, retry identity, layout drift,
noncontiguous source tensors, directional/full profiles, and synchronous versus
background equality. The manifest separately accounts snapshot calls, source
tensors, coalesced CUDA bytes, endpoint-identity hashing, and retained endpoint
cloning.

Two subsequent attempts produced no training evidence and are recorded only as
preflight failures. r5f reached its exact image and checkout but gcloud SSH's
extra shell parse consumed a nested quoting layer in the detached runner body.
Its failure tree is
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r5f-directional-endpoint-reuse-canary`;
abandonment SHA-256 is
`35d8ca7e9f40e9b55b9cbee480ac1ca10f12cce8dc3d7f431aae043d90489c6e`.
Exact instance ID `7370808347110138535` and disk ID
`5868512611846054567` return 404. Commit `e8c7d28` repairs the transport by
base64-encoding each nested Bash body and decoding it only inside the remote
`bash -c`; a real A100-host transport smoke passed with an apostrophe-bearing
payload. r5g was then invoked incorrectly from the operator shell without
binding `PYTHONPATH` to this worktree, so Python imported the original checkout
harness and reproduced the old wrapper failure before training. This was an
operator invocation error, not an image or experiment result. Its failure tree
is
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r5g-directional-ssh-safe-canary`;
abandonment SHA-256 is
`1dd34b46bc97b97e8304e97e920b1df68a15946615e050f76aa04be63f3c6f95`.
Exact instance ID `2748115238036168881` and disk ID
`107228130778538161` return 404.

The worktree-bound r5h retry finally produced an admissible parity comparison
and exact timing measurement at
`gs://yeto-exp2-52-model-training-497007/exp2-54-smoke-r5h-directional-worktree-harness-canary`.
It executed repo/harness pin
`e8c7d28a228023a8bf8963aad1f53a80b07cc191`; the checked-in r5h spec is commit
`34d27feadfbb2208b4c6d25f14e76b4eff09e064`.
Both arms committed the exact canonical sequence 1--16 with fragment order
`[0,1,2,3]` in each of four fixed-H waves. All 16 authoritative transcript
joins, four joins per fragment, exact probe payloads, final syncer checkpoint,
six-file adapter export, and eval loss/token `1.4166344593414884` passed. The
ON learner retained 32 verified artifacts totaling `259,962,886` bytes. The
bounded writer accepted and completed all 32 jobs/`328,076,192` reserved bytes,
with zero block, abandonment, queued, reserved, or in-flight residue at close.

The 2% performance gate nevertheless failed. The exact commit-1-to-commit-16
interval was `18.421168652 s` OFF and `19.888230758 s` ON: **7.964001% capture
overhead**. The A100 manifest measured 52/52 coalesced CUDA snapshots totaling
`281,309,184` bytes and `0.157208758 s`; 16 endpoint-identity hashes over
`86,556,672` bytes took `0.263190275 s`; retained endpoint clones took
`0.009378067 s`; and structural freeze took `0.019477676 s`. Worker-side
payload hashing, serialization, and publication consumed `0.852979217 s`,
`0.183581429 s`, and `1.369562928 s`, respectively. Queue blocking and duplicate
endpoint reads are no longer observed bottlenecks. The end-to-end capture path
still exceeds the frozen budget; these component counters identify substantial
remaining CPU, memory, and publication work but do not uniquely attribute the
interval delta because the operations overlap. r5h is an engineering failure,
not optimizer evidence.

The harness abandonment record is retained with SHA-256
`f284969aceb426b1143f34d3c17e4d81cd0e9bcd7b5d4fef7086ba0900535c0f`.
Exact instance ID `1879903009492427199` and auto-delete disk ID
`2154987517248266687` both return 404. The four-A100 r6 direction-only draft
was not launched: scaling a failed cost path would still not create the full
model/optimizer/RNG/data restore state required for a causal loss comparison.
r5h evaluated no optimizer candidate and produced no optimizer-loss evidence.
No candidate in the completed hunt has established superiority over SGD-0.28;
PTI remains direction-only motivation for a future full capture-v2 CRN test.

### Final branch verification

After the r5h teardown and CGC kernel commit `f9f9a37`, the release checks are:

- repository-wide Python: 1,186 passed, one skipped, six dependency
  deprecation warnings; the skip is the CUDA-only coalesced-transfer unit test,
  while r5h's A100 manifest independently records 52/52 coalesced CUDA calls;
- Rust syncer: 173 passed, zero failed;
- focused optimizer-harness suite: 67 passed;
- direct CTTN golden-trace program: all cases passed;
- targeted Ruff lint and format: all 11 selected changed/critical files passed;
- Lean aggregate build: 8,569 jobs completed successfully, including CGC's
  norm/cap/stationary theorems and reversal counterexample;
- r5h spec validation: passed with exact repo/harness pin `e8c7d28`;
- `git diff --check`: passed.

Repository-wide Pytest had initially collected four numerical helper functions
from the directly executable `scripts/test_cttn.py` as fixture-based tests.
That script now declares itself non-collectable without changing direct
execution; the full clean count above is the post-fix rerun, not a filtered
failure count.

### PTI-SGD: next temporal candidate, not yet an empirical result

A separate agent review proposed **Prequential Transverse Interlock SGD
(PTI-SGD)** to address the most consistent prior failure: strong same-round
held-worker directions becoming negative on the next same-fragment boundary.
The retained-tape development screen extracted the previous production
direction's component orthogonal to the current unit direction and evaluated
the fixed signed coefficient grid `{0, ±1/32, ±1/16, ±1/8, ±1/4}`. Every
candidate was norm-grafted back to the exact SGD-0.28 pseudo-gradient norm, so
its maximum possible rotation was `atan(1/4) ≈ 14.04°`.

Every signed coefficient was shadow-scored on every valid boundary whether or
not it was eligible. The development router could use a nonzero coefficient
only when that coefficient's three most recent valid **counterfactual shadow
scores** all improved the sealed next same-fragment cosine. The subsequent
Amendment 1 freeze selected the sole live coefficient `-1/4`; the current PTI
state machine performs no coefficient search or tie-break. A nonpositive score
disables application but does not stop later shadow scoring, so three new
consecutive positive scores can restore eligibility. Missing continuity,
degenerate transverse geometry, nonfinite state, or a hash mismatch returns
bit-identical SGD. Unlike CPER-SGD, which compared two large fixed full-vector
experts and abstained 76/76 times, PTI uses a small signed transverse turn and
a recent causal interlock rather than a lifetime cumulative winner. It is
killed if it acts on fewer than 25% of all predefined valid post-warm-up
boundaries; integrity exclusions and warm-up are fixed before outcomes, so the
policy cannot shrink the denominator by declaring difficult cases ineligible.

This is only a falsifiable proposal. Its shadow gate requires at least 32
post-warm-up decisions, eight per fragment, mean sealed next-fragment cosine
gain above 0.001 with a positive block-bootstrap lower bound, at least three
positive fragments, and nontrivial action. The checked-in fresh PTI
preregistration defines one mechanism/finite-loss gate and explicitly keeps
PTI outside EXP2-54's two-candidate Holm family. The Lean file
`PrequentialTransverseInterlock.lean` proves only the normalized alignment
identity and its stationary-direction counterexample: every nonzero
orthogonal turn is worse when the next direction is unchanged. It proves no
training-loss or convergence superiority. The direct module check and complete
Lean build pass with no `sorry`, `admit`, or project axiom; the audited theorem
dependencies are only `propext`, `Classical.choice`, and `Quot.sound`.

### 2026-07-14 PTI implementation, CRN substrate, and GCP status

Four foundational commits turn parts of the PTI design into executable
substrate, without creating a PTI-versus-SGD loss result:

- `eb6b3bd` freezes the exploratory online GCP sequence and its E1/E2 resource,
  integrity, and teardown gates. It predates the production PTI wiring and is
  still deliberately labeled draft, blocked, and non-launchable; it contains
  no JSON launch specification.
- `f294828` adds a single-process PyTorch capture-v2 live publisher/restorer.
  It snapshots fp32 trainable fragments, supported optimizer tensor/integer
  state and parameter-group topology, model buffers, JSON-representable
  scheduler/scaler state, train/eval mode, Python/NumPy/Torch CPU and visible
  CUDA RNG state, input provenance, and real pre-materialized future groups.
  Restore verifies the CAS objects, target topology/devices, every represented
  value, and a digest of all restored mutable state. Unsupported state fails
  closed before an endpoint is published.
- `2323657` makes the policy-agnostic capture-v2 CRN orchestration executable.
  For one pre-authorized boundary it performs four fresh restores in A/B then
  B/A order, applies the exact sealed stock or candidate action, evaluates at
  k=0, consumes the same eight sealed future groups, evaluates at k=8, checks
  order invariance and branch isolation, and atomically publishes an isolation
  attestation plus paired outcome. It also adds the PTI-specific closed-campaign
  finite-loss arithmetic over authority-verified outcomes.
- `73ed593` adds `pti-sgd` to the production Rust syncer, adds matched
  `pti_m1_stock`/`pti_m1_candidate` and
  `pti_m4_stock`/`pti_m4_candidate` presets to `compare_diloco.py`, extends the
  syncer checkpoint format with per-fragment PTI state, and adds PTI evidence
  fields to the JSONL event tape. Tests assert that each online pair differs
  only in the outer direction selector.

The frozen online arithmetic is exactly the Amendment-1 `-1/4` policy. For
current and preceding same-fragment stock pseudo-gradients `G_t` and
`G_{t-1}`, it computes

\[
u=G_t/\lVert G_t\rVert,\qquad
v=G_{t-1}/\lVert G_{t-1}\rVert,
\]

\[
p=\frac{v-\langle v,u\rangle u}
        {\lVert v-\langle v,u\rangle u\rVert},\qquad
C_t=\lVert G_t\rVert
    \frac{u-\tfrac14p}{\lVert u-\tfrac14p\rVert}.
\]

All products, sums, square roots, divisions, and coordinate writes follow the
frozen sequential f32 operation order. The shadow sealed at boundary `t-1` is
resolved only when factual same-fragment `G_t` arrives:

\[
z_{t-1}=\cos(C_{t-1},G_t)-\cos(G_{t-1},G_t).
\]

The candidate is selected only when the three most recent **already resolved**
scores are all strictly positive. A valid candidate is still sealed for the
next shadow when the interlock is closed. Warm-up, a closed interlock, or
invalid/degenerate geometry applies the stock direction; invalid causal state
is cleared rather than repaired from an invented observation. The selected
direction then feeds the same outer Nesterov implementation as the stock arm
with outer LR exactly `0.28` and momentum positive zero, i.e. this campaign's
memoryless outer SGD-0.28 kernel. The paired presets otherwise retain the same
weighted RDA merge, `merge_alpha=0`, f32 wire, no delta correction, strict
quorum, deterministic commit order, data order, seed, evaluation work, and
matched inner AdamW. Thus “SGD-0.28” here names the outer step; it does not
rename the matched inner optimizer.

The Rust candidate kernel is checked against frozen Python little-endian f32
bit fixtures, including nontrivial three- and four-coordinate cases. PTI state
is carried inside an immutable action preview and installed only by
`commit_preview`; repeated previews are equal and do not resolve a shadow or
advance the interlock twice. The checkpoint extension persists the preceding
stock direction, pending candidate, and at most three resolved scores for each
fragment. A test splits a causal direction stream at a checkpoint and proves
that resumed statistics, parameters, outer buffer, and PTI state remain
bit-identical to uninterrupted execution. Python export validates and skips
that optimizer-only extension while still recovering the model checkpoint.

Each PTI event-tape row now reports the resolved shadow score, score count,
interlock/action/clear flags, reason code, and SHA-256 identities for the
current stock, previous stock, candidate, and selected action. Those are
per-record vector identities, not yet a chained PTI ledger: `73ed593` does not
write the hash-chain predecessor or a separately sealed final ledger head
required by the E1 plan. Follow-up commit `905328f` exercises the causal path
through a real syncer process and checks its emitted tape rather than only a
direct state-machine call. Follow-up commit `8d774bf` adds a wrapper that runs
one exact stock/PTI setting pair and then a fail-closed validator. The
validator binds finite terminal losses to exactly 32 matched commit/responder
schedules, four balanced fragments, both final checkpoints, causal
same-fragment hash continuity, three-positive interlock decisions, at least
one non-stock action, and exact stock-fallback action hashes; it atomically
writes PASS or FAIL evidence and labels the scope exploratory/non-CRN. It does
not turn the per-row hashes into the missing chain or bind a live cloud
runtime/image/input manifest. No immutable GCP JSON spec exists yet. Thus the
online code and local evidence validator are executable, but the branch still
does not satisfy the complete E1 cloud launch gate.

The live capture-v2 bridge and evaluator similarly close interfaces, not the
end-to-end live CRN path. The bridge intentionally does not choose a causally
safe learner boundary or advance/materialize a production data iterator. The
current training program does not yet pause a real learner and join its exact
endpoint/future-eight bundle to the corresponding syncer boundary, and no
production `IsolatedCRNBackend` yet performs fresh model construction,
endpoint restore, sealed-action application, exact eight-group training, and
fixed-object evaluation. A real campaign must still precommit a closed set of
at least 32 balanced boundaries, eight for each of four fragments, before any
callbacks or outcomes run. The finite-loss analyzer reports the separately
frozen action-rate check and enforces the k0/k8 sample, gain, bootstrap,
fragment, positive-boundary, and worst-regression statistics, but it
deliberately does not accept the direction-score confidence or matched runtime
overhead evidence needed for PTI promotion. Fake-backend and toy-PyTorch tests
are contract tests, not live Qwen loss evidence.

The GCP plan remains a sequential two-stage screen using exact READY image ID
`7290368630472593484`, a 250 GB `pd-ssd` boot disk, Spot provisioning with
delete-on-termination, a 3,600-second provider runtime limit, and a
campaign-wide ceiling of eight active accelerators:

1. E1 reserves `exp2-pti-online-e1-m1-canary` on one
   `a2-highgpu-1g` A100. It pairs the m1 stock and PTI arms at H4 for 32
   syncer commits, 32,768 training tokens, eight evaluation rows, and seed
   `271271`. It is only an engineering canary; its loss sign is descriptive,
   while exact fallback, at least one action, final checkpoint/evidence,
   finite-loss, equal-work, checksum, and below-2%-overhead checks determine
   whether engineering passed.
2. E2 reserves `exp2-pti-online-e2-m4-screen` on one
   `a2-highgpu-4g` four-A100 VM. It is conditional on a reviewed, checksummed
   E1 pass and keeps the treatment semantics fixed while moving to four
   learners, H16, 700,000 tokens, 64 evaluation rows, and 32 syncer commits.
   The two stages may not overlap; permission to use up to eight A100s is a
   campaign ceiling, not a reason to skip E1 or allocate eight GPUs to either
   frozen stage.

The GCP authentication block was resolved later on 2026-07-14 with the
`admin-codex@model-training-497007.iam.gserviceaccount.com` service account in
an isolated gcloud configuration. The live project reports a Spot A100 quota
of 16 but an A2 CPU quota of 48, so the largest feasible A2 allocation is one
four-A100 `a2-highgpu-4g`, not eight A100s at once. The campaign's eight-GPU
number remains a ceiling; it is not available capacity and does not override
the sequential E1 gate.

The first immutable online launch,
`exp2-pti-online-e1-m1-canary-r1`, allocated Spot VM ID
`3181859973759747432` and boot-disk ID `3875931301588223335` from exact image
ID `7290368630472593484`. Its verified input manifests passed, but the
optimizer command never started. The local CLI entry point had resolved the
editable virtual environment's stale `/Users/shou/yeto` harness instead of the
experiment worktree's newer harness. The stale controller emitted the already
known unsafe nested-single-quote runner, producing
`checksum: line 4: is: command not found` and an unexpected end of file. This
was controller module drift, not a corrupt checksum or PTI failure.

The r1 run directory, command, spec, logs, and input provenance were synced to
`gs://yeto-exp2-52-model-training-497007/exp2-pti-online-e1-m1-canary-r1`.
The harness wrote and round-tripped an abandonment record with SHA-256
`557494d6b641dc49c2341d07897c979eb424102eab2b0e7f499176d403e1b4fe`,
then deleted only the authenticated VM and auto-delete boot disk. Independent
instance and disk lookups both return 404. The r1 identity and prefix are
quarantined and will not be reused.

Commit `426425aaf09068608ddf2900c71f6d50f1bb5fdb` makes the controller prepend
and verify its sibling checkout before importing `yeto.optimizer_harness`.
Its executable regression places a decoy stale harness first on `PYTHONPATH`,
runs the real entry point from an unrelated directory, and requires the
base64-safe detached renderer instead of the stale quote-vulnerable renderer.
The focused harness/spec suite passes 72 tests. Commit `be1bb1f` adds the fresh
r2 immutable JSON spec, which pins that repair commit and uses new VM, run,
checkout, artifact, and GCS identities throughout.

Fresh `exp2-pti-online-e1-m1-canary-r2` then ran on Spot VM ID
`2050450023308312795` with boot-disk ID `2921173287301791963`. The exact
model/data manifests passed. The roughly 143--146-second start-to-training
delay was dominated by serial local SHA-256 over the approximately 16.678 GiB
Qwen 9B model on the 250 GB `pd-ssd`, an effective roughly 117--119 MiB/s; no
model ingress or network download occurred. Because the loader immediately
reads the same model again, this is a redundant integrity pass. A future image
may replace it only with an explicit fail-closed `fs-verity`/signed-attestation
mode bound to the exact numeric source-image ID; mutable timestamps or an
unsigned “already verified” marker are not acceptable substitutes.

Both online arms completed exactly 32 commits, eight per fragment. The stock
arm used 34 local steps, took 81.5 seconds, and evaluated at
`1.3319227656615953` loss/token. The PTI arm used the same 34 local steps, took
87.9 seconds, and evaluated at the identical
`1.3319227656615953`. This equality is expected because PTI constructed 28
candidate directions and resolved 24 delayed scores, 11 positive and 13
negative, but its three-strictly-positive interlock never opened and it applied
zero non-stock actions. It is therefore not evidence of a tie between two
optimizers.

The fail-closed validator rejected r2 with the sole error
`PTI tape selected no nonstock actions`. Its FAIL artifact has SHA-256
`ccac85e0447549c2c2b9321bdd52b9c637f754d8aca71cd3a80e6ce51f65b3ce`
and explicitly labels the evidence `exploratory_online_non_crn`. All run data,
checkpoints, adapters, tapes, logs, provenance, validation, checksums, and
results are preserved under
`gs://yeto-exp2-52-model-training-497007/exp2-pti-online-e1-m1-canary-r2`.
The abandonment record SHA-256 is
`c2785e27e245d1b56548f50be200d0404dcc4826e4cc4963e5f93d5dba407235`.
The harness deleted exactly that authenticated VM and auto-delete disk, and
both provider lookups return 404. E2 remains blocked and was not launched.
Under the frozen gate, PTI is killed as inert at this E1 cell; weakening the
interlock or increasing the run only after seeing this failure would define a
new candidate and require fresh preregistration and evidence.

Finally, the two evidence classes must not be conflated. The planned online
stock/PTI arms are matched and paired by seed, but once PTI first takes a
non-stock action their model, optimizer, and subsequent pseudo-gradient states
diverge. Their trajectory or terminal-loss difference is useful exploratory
screening only. The preregistered capture-v2 CRN test instead restores both
actions from the same complete boundary state, runs each in both A/B and B/A
order, and consumes byte-identical future groups through k=8. Only that latter
same-state design can contribute the frozen causal finite-loss evidence; no
result from either design has yet been acquired.

### CGC-SGD: bounded angular continuation kernel

A later failure-synthesis agent proposed **Causal Geodesic-Continuation SGD
(CGC-SGD)**. It preserves PTI's historically favored opposite-transverse sign
but replaces PTI's fixed turn with a one-step angular-velocity estimate. For
unit current direction `u`, previous direction `p`, acute cosine
`rho=<u,p>`, and normalized backward tangent

\[
b = \frac{p-\rho u}{\sqrt{1-\rho^2}},
\]

the candidate uses

\[
\lambda=\min\!\left(\frac{\sqrt{1-\rho^2}}{\rho},\frac14\right),
\qquad
q=\frac{u-\lambda b}{\sqrt{1+\lambda^2}},
\]

then grafts `q` to the exact stock pseudo-gradient norm. Below the cap this is
constant-angular-velocity continuation; at zero angular velocity it degenerates
exactly to stock rather than applying PTI's fixed stationary-direction penalty.
The cap keeps the turn at most `atan(1/4) ≈ 14.04°`.

Commit `f9f9a37f7f5f3b2106ae30aa12825e91eaa88dae` implements only the bounded,
state-free f32 kernel. It consumes exact little-endian current/previous bytes;
malformed evidence is an error, while stationary, nonacute, degenerate,
nonfinite, or f32-inert geometry returns the identical stock `bytes` object.
It contains no causal interlock, live optimizer hook, loss-dependent router, or
claim that the output should be applied. Focused CGC/PTI/adapter regression
testing passed 46 tests.

`CausalGeodesicContinuation.lean` proves norm preservation under orthonormal
stock/tangent geometry, exact stock degeneracy at zero angular velocity, exact
constant-angular continuation below the cap, the current-direction cosine,
and the squared-cosine lower bound `16/17`. It also proves an explicit reversal
counterexample in which continuation is strictly worse than stock. The formal
result therefore rules out unconditional dominance; it does not establish
that language-model pseudo-gradients have constant angular velocity or that
CGC improves loss. CGC is a preregisterable shadow/CRN candidate only after the
same full capture-v2 engineering gate required by PTI.

### CPLG-SGD: two-turn phase-locked continuation

The PTI E1 failure gives a narrow, useful negative result: a fixed `-1/4`
transverse candidate produced both positive and negative delayed scores, and
the three-positive causal interlock correctly kept it inert. It does not
justify weakening that interlock. A distinct follow-up proposal, **Causal
Phase-Locked Geodesic SGD (CPLG-SGD)**, instead requires two consecutive
same-fragment rotations to have coherent phase and makes the turn shrink to
zero under stationarity or reversal.

For previous/current unit stock directions `v = u_{t-1}` and `u = u_t`, let

\[
\rho=\langle u,v\rangle,\qquad
d=\frac{\rho u-v}{\sqrt{1-\rho^2}},
\]

where `d` is the **forward** continuation tangent at `u`. Existing PTI/CGC
code constructs the opposite backward tangent
`b=(v-rho*u)/sqrt(1-rho^2)=-d`; reusing `b` without negating it would reverse
the proposed action. For the preceding forward tangent `h` at `v`, shortest
great-circle transport on the artificial unit sphere of direction vectors is

\[
\operatorname{PT}_{v\rightarrow u}(h)
=h-\frac{\langle h,u\rangle}{1+\rho}(v+u).
\]

A tangent-plane projection is not this transport. With current and preceding
turn angles `theta_t` and `theta_{t-1}`, phase coherence and the candidate are

\[
c=\max(0,\langle d,\operatorname{PT}(h)\rangle),\qquad
\phi=c\min(\theta_t,\theta_{t-1},\arctan(1/4)),
\]

\[
q=\cos(\phi)u+\sin(\phi)d,\qquad C_t=\lVert G_t\rVert q.
\]

This angle semantics is part of the algorithm identity. The existing
normalized-linear CGC helper accepts a tangent ratio whose actual angle is
`atan(alpha)`. Feeding it `phi`, or using
`c*min(tan(theta_t),tan(theta_{t-1}),1/4)`, defines a different optimizer. A
correct helper reuse would require `alpha=tan(phi)`; the planned f32 kernel
instead materializes pinned `cosf(phi)` and `sinf(phi)` directly and
renormalizes before norm grafting. The initial Lean ratio-proxy draft was
caught by an independent audit before being described as the exact CPLG
algorithm. Corrective commit `8690a7a83a899b106fa98260261ea0aed616cd77`
now formalizes the angle construction and exact sphere-transport formula. It
proves transport tangency, ideal constant-rotation transport/coherence and
recovery, exact stock degeneration, unit norm, the actual angular cap, and a
reversal non-dominance theorem. It also proves that the rejected nested-arctan
ratio interpretation is strictly different. The full Lean build passes with
no `sorry`, `admit`, or added axioms; these are real-geometry safety theorems,
not f32, loss, convergence, or superiority proofs.

CPLG remains causal and fail-closed. Per fragment it needs the previous stock
direction, forward tangent and angle, one pending shadow candidate with source
identity, and at most three resolved scores. Preview is pure; only commit may
install the next state. Sequence/layout/hash failure, nonfinite or nonacute
geometry, degenerate transport, or a checkpoint invariant failure returns the
original stock bytes and clears eligibility. The three already-resolved
strictly-positive score interlock remains unchanged.

Commit `26b985a` adds an independent Python exact-f32 reference and
`scripts/replay_cplg_shadow.py`. It fixes sequential product-then-add reduction
order without FMA, dot overshoot tolerance `2^-20`, norm-squared floor `2^-40`,
strict acute geometry, the forward-tangent sign, shortest-sphere transport,
reprojection/normalization, literal `sin(phi)`/`cos(phi)` angle semantics, and
the cap bits `0x3e7adbb0`. Golden vectors distinguish 30-degree forward
continuation from the wrong 10-degree sign, literal angle from `atan(phi)`, the
cap, a nonplanar transport, and a reversal fallback. The replay accepts only a
strict, hashed full-vector stock tape. The retained PTI r2 event tape contains
vector identities but not those vectors, so it is correctly rejected as
unidentifiable and cannot be recycled into CPLG evidence.

Commit `538f328` adds the action-capable production Rust selector
`--outer-optimizer cplg-sgd`. It freezes the treatment as exact f32 outer LR
`0.28` (bits `0x3e8f5c29`), exact positive-zero momentum, and no per-fragment
LR overrides. It pins Rust `libm` 0.2.15, implements the same guarded geometry,
opens no earlier than the sixth same-fragment visit after three strictly
positive already-resolved shadow scores, and installs state only on commit.
The `CPL1` checkpoint extension round-trips per-fragment phase/interlock state;
conditional event-tape fields and action seals bind the geometry, vector
hashes, fallback reason, and selected action without changing non-CPLG/PTI
tape records. The Rust suite passes 190/190 tests, including the old PTI golden
and checkpoint tests, and the 57-test Python CPLG/CGC/PTI subset passes. The
post-CPLG repository-wide Python run passes 1,252 tests with one CUDA-only skip
and six dependency deprecation warnings. Targeted Ruff lint/format, Rust
format, and the direct corrected Lean module also pass; the complete corrected
Lean build previously completed all 8,570 jobs.

Python currently calls the host C `atan2f`, `sinf`, and `cosf`, whereas the
production selector calls the pinned Rust `libm`. Exact raw-f32 cross-runtime
fixtures in commit `316cc6c` confirm that this is a real, not hypothetical,
gap. Constant-phase geometry, transport, candidate, and shadow score are
bit-identical. The nonplanar final candidate also matches, although two angle
scalars differ by one ULP. At the cap, Darwin C `sinf` returns `0x3e785b43`
while pinned Rust `libm` returns `0x3e785b42`, changing the first candidate
coordinate from Python `0x40838c9a` to Rust `0x40838c9b`; the second coordinate
matches. No tolerance or algorithm was changed to hide the mismatch.

The Python report therefore correctly continues to say
`UNSATISFIED_UNTIL_RUST_FIXTURE_MATCH`. A single portable trig implementation
or Rust-produced authoritative replay is required before cross-runtime
evidence can be sealed. This numerical provenance gap, a fresh immutable
full-vector stock tape, the fixed shadow-analysis gate, and a chained final
ledger remain launch gates. No CPLG GPU VM or loss arm has been run.

The first admissible experiment is shadow-only on a fresh stock trajectory:
exactly 32 ordered boundaries, eight per fragment, two phase warm-ups per
fragment, 24 sealed shadows, 20 resolved shadows, and four declared unresolved
tails. The fixed gate requires at least eight simulated actions among all 32
boundaries, mean resolved directional gain above `0.001`, a positive
fragment-stratified moving-block lower endpoint, at least three positive
fragment means, exact fallback/integrity, and below-2% overhead. Only a fresh,
separately preregistered capture-v2 campaign may then test finite loss from the
same state and future-eight groups. Sphere-geometry and nonanticipation proofs
cannot establish language-model loss improvement or unconditional dominance;
the checked-in rotation-reversal theorem explicitly shows why no such general
claim is available.

### CRP-SGD: causally resolved residual pulses

Another distinct proposal is **Causally Resolved Residual-Pulse SGD
(CRP-SGD)**. This is a working name and falsifiable design, not a novelty or
performance claim. For one separately declared proposal source, it seals the
residual `r_t = Q_t - G_t` at boundary `t`, applies exact SGD-0.28, and only at
the next same-fragment boundary measures whether that tiny residual improved
directional agreement. A residual may enter an eight-boundary FIFO only when
its relative norm is strictly between zero and `1/20`, its provenance chain is
intact, and the predeclared delayed score is positive. A nonpositive score,
missing continuity, nonfinite state, or hash failure clears the whole fragment
bank; CRP never reverses a bad residual after seeing its outcome.

Once at least two admitted residuals have accumulated, CRP sums them, removes
the component parallel to the current factual direction, and acts only when
the remaining norm reaches `1/20` of the factual direction. The pulse is never
scaled upward and is clipped down at `1/8`; after emission the bank clears.
Thus an action is orthogonal to stock, has a predeclared relative norm in
`[0.05, 0.125]`, and increases total direction norm by at most
`sqrt(1 + (1/8)^2) ≈ 1.00778`. Every no-pulse branch returns `None` to the
unchanged SGD-0.28 caller with the original factual gradient buffer. Auxiliary
bank state cannot enter stock optimizer state.

CRP is designed to make the earlier tiny-action and negative-next-boundary
failures decisive rather than cosmetically positive. Coherent tiny residuals
must supply enough real mass for a measurable bounded pulse; incoherent,
negative, or stale residuals clear, cancel, or expire. The cheap causal-tape
screen requires at least 32 post-warm-up opportunities, eight per fragment,
pulses on at least 25% of all predefined opportunities, a mean sealed delayed
gain above `0.001` with positive moving-block lower endpoint, at least three
positive fragments, and at least 60% positive pulses. Failure retains
SGD-0.28; it is not permission to lower the threshold, extend the delay, flip
signs, or choose a different proposal source after inspection.

The Lean-sized proof target is deliberately narrow. Bank accounting records
every admitted, emitted, and discarded residual via
`e' = e + r - a - d`, hence `e' + a + d = e + r`; a telescoped version proves
the bank creates no correction mass. A scalar quadratic identity exposes the
conditional loss gap as `λ/2 * (a² - 2a(x-b))`. Neither theorem makes delayed
cosine a safety proof or establishes convergence. CRP must pass byte-parity and
CPU quadratic property tests before it may consume an isolated CRN bundle, and
each proposal generator is a separate hypothesis. The current v1 evidence is
not sufficient to run that loss replay, so no GPU acquisition is justified
merely to discover whether the bank fires.

#### Retained-tape audit and PTI direction screen

The CPU-only `scripts/replay_crp_sgd.py` freezes a distinct **Causally Resolved
Residual-Pulse SGD (CRP-SGD)** state machine before reading retained outcomes.
At a boundary it resolves the previous same-fragment shadow residual against
the current factual direction, clears the bank on every failed resolution,
forms a transverse pulse only from residuals admitted before the current
boundary, and admits the newly resolved residual only after that pulse
decision. Thus a residual sealed at `t` can be resolved at `t+1` but cannot be
emitted before `t+2`. Individual residuals must have strict norm ratio below
`1/20`; at least two admitted residuals are required; source age is at most
eight same-fragment boundaries; and a bank projection acts only at ratio at
least `1/20`, with downward-only clipping at `1/8`. Every abstention returns
the original stock f32 byte object without re-encoding.

The exact replay contract is `crp_exact_vectors_v1`: each chronological row
must hash exact little-endian f32 bytes for both factual stock direction `G_t`
and proposal direction `Q_t`, with event, sequence, fragment, and length. The
engine rejects or clears on checksum, shape, finiteness, norm, continuity, or
resolution failure. Twenty-nine tests cover causal ordering, strict threshold
behavior, bank clearing and expiry, no upward scaling, maximum clipping,
NaN-payload/negative-zero bit-identical fallback, deterministic repeated
replay, exact checkpoint materialization, and equivalence of the analytic PTI
score to direct vector construction. The exact reader additionally rejects
duplicate, missing, and unexpected fields; non-standard JSON constants;
booleans, strings, or floats in integer fields; noncanonical or uppercase
SHA-256 text; and absolute, escaping, non-regular, or symlink-containing vector
and checkpoint paths. A global sequence gap clears every fragment bank and
consumes a bit-identical stock fallback. Vector/hash and shape integrity
fallbacks are separately counted in the output ledger. Reports are fsynced to
a same-directory temporary file, atomically replaced and rehashed, after which
an atomically replaced `.sha256` file is published as the completion marker.

The retained `exp2-53a2` BCMP tape does **not** satisfy that contract. Across
four learner JSONLs it contains 458 shadow events and 1,368 joined candidate
resolutions (456 each for ray, slab, and reset), but only scalar norms,
residual/future-gradient dots, and cosines. It has neither `G_t` nor `Q_t`
bytes, tensor layout/accumulation order, residual-residual cross terms, later
stock vectors for projection, nor a merged production-boundary order. The
audit therefore returns `CRP: UNIDENTIFIABLE`; it reconstructs no CRP action
and reports no empirical CRP score. Descriptively, all 456 slab residuals were
below the frozen `1/20` norm ratio and 142 also had positive future residual
dot. That dot is not CRP's normalized cosine gain `z`, so these 142 records
cannot be called admissions or pulses. Ray and reset each had only two tiny
residuals and zero records satisfying both proxy conditions.

Three older syncer-current captures do contain exact checkpoint bytes at
consecutive same-fragment boundaries. Version checks prove that subtracting
each current checkpoint fragment from the next checkpoint for that fragment
materializes its realized factual f32 displacement. This is enough for a
narrow historical PTI geometry screen: 621 valid scores, balanced as 208 for
fragment 3, 207 for fragment 7, and 206 for fragment 11. It is not enough for
CRP because the BCMP proposal/residual vectors are absent, and it is not enough
for MSTP because no joined anchor/H/2/H arrays, exact midpoint/end Adam moment
and metric arrays, clocks, LR/decay accounting, or production-RDA parity proof
exist locally.

The PTI coefficient screen found the following exact historical direction
statistics:

| coefficient | mean next-direction cosine gain | positive-score fraction | post-warm-up three-positive eligible fraction | positive fragment means |
| ---: | ---: | ---: | ---: | ---: |
| `-1/4` | `0.018742651964765277` | `0.966183574879227` | `0.930976430976431` | `3/3` |
| `-1/8` | `0.006805254846032888` | `0.9452495974235104` | `0.8754208754208754` | `3/3` |
| `-1/16` | `0.002709298347969893` | `0.9049919484702094` | `0.7861952861952862` | `3/3` |
| `-1/32` | `0.0011771748717649948` | `0.8727858293075684` | `0.7222222222222222` | `3/3` |
| `+1/32` | `-0.0008177389738578069` | `0.22705314009661837` | `0.06060606060606061` | `0/3` |
| `+1/16` | `-0.0012747033566810307` | `0.2914653784219002` | `0.10437710437710437` | `0/3` |
| `+1/8` | `-0.0011166459386409112` | `0.428341384863124` | `0.1835016835016835` | `0/3` |
| `+1/4` | `0.0032522881997281102` | `0.6666666666666666` | `0.4225589225589226` | `3/3` |

The pooled `-1/4` result is also descriptive-positive in each exact source
capture independently:

| source capture | scores | mean cosine gain | positive-score fraction | post-warm-up interlock eligible fraction |
| --- | ---: | ---: | ---: | ---: |
| `equal-token-late-smollm2-p4de-seed53-syncer-current-6m` | 207 | `0.01895086235250181` | `0.9565217391304348` | `0.9040404040404041` |
| `equal-token-late-smollm2-p4de-seed67-syncer-current-6m` | 202 | `0.01914589422492158` | `0.9900990099009901` | `0.9689119170984456` |
| `equal-token-late-smollm2-p4de-seed79-syncer-current-6m` | 212 | `0.018155130800552848` | `0.9528301886792453` | `0.9211822660098522` |

Their exact index SHA-256 / derived factual-direction-chain SHA-256 pairs are:

- seed 53: `8b291187a33711970d6e42ccc1af6d3f2f77bb6b46b0794af7eb97a25e04660e`
  / `da959b31aeeb79ec2885e18cdb3bfba093f636e9f52829bc6a47e8b970a8f883`;
- seed 67: `cc472c93c54c274c10e039aa53ccbbe034c2b634c8b5026a2bb66ae56a6b7115`
  / `dff5e018e1bf22a890baffaeecfec347c3153d2cc4c36762aeb98e71ab851b8c`;
- seed 79: `a2c68cfb38eb9d51fc85a8a1686a40a1a7d7e6b56a561a78300e93e04855a564`
  / `2f5fea96cf3f325ff1626be3be3fe42e5f10af2270684ebf02ac8abb24d4980c`.

This per-capture agreement is a robustness description of historical
direction geometry. It is not a new promotion gate and does not change
`DIRECTION_SCREEN_ONLY`.

For `-1/4`, the fragment means were `0.01739135170016281` (fragment 3),
`0.016985339645211022` (fragment 7), and `0.021872914611294623` (fragment
11). This sign asymmetry is useful mechanism evidence, not evidence that PTI
beats SGD-0.28. The directions become off-policy after the first hypothetical
non-stock action, and the source tapes contain no sealed k=0/k=8 loss bundle.
At the time of this retained-tape audit the grid screen had no predeclared
composite tie-break, so the audit reported each coefficient separately and did
not retroactively apply the later frozen `-1/4` policy. `MSTP` remains
`UNIDENTIFIABLE`, never a zero-action result.

After this development-only screen, the fresh PTI protocol was frozen in
`experiments/optimizer/pti-sgd-fresh-confirmation-prereg.md`. Its initial
post-development SHA-256 was
`9c7127d18c0657500debb3273ff8e550903966653947b7fbf3de3eb83346909c`.
Before any fresh acquisition, adversarial implementation review found that the
text had not fully specified per-operation f32 rounding and had asked an
online policy to retroactively replace an already applied action when a stream
later ended. Amendment 1 fixes those specification defects. Its current
SHA-256 is
`693f190c14ca0f31b6cbe65a4fa829ff15797b2f2a4f72e0dfa4aa531c43fe58`;
the checked-in sidecar binds this current digest. The amendment defines
left-to-right sequential f32 arithmetic and an explicit non-action tail-close
record; it does not change the selected coefficient, empirical gates, or use
any fresh outcome.
It fixes the only non-stock coefficient at `-1/4`, so there is no live
tie-break or coefficient search; the seed-53/67/79 tapes are permanently
ineligible for confirmation. It requires the three-positive causal interlock,
bit-identical stock fallback, complete capture-v2 capability checks, isolated
A/B and B/A k0/k8 replay, one fresh mechanism/finite-loss gate, five fresh
paired online seeds, and then H/model/inner-optimizer breadth before any
generalization claim. This is a post-development freeze for new data, not a
retroactive preregistration of the retained result.

Commit `30620b7020b82801547baea612d59716e49a8c77` implements that fixed
direction policy as a reusable causal state machine. It consumes hash-bound
little-endian f32 stock bytes in a gap-free global event order; allows strictly
increasing global fragment versions rather than incorrectly assuming `+1`;
resolves only the previously sealed shadow; updates the last-three score
interlock; and then seals the current action and next shadow. It always shadows
the fixed candidate while valid, even when the interlock is closed. Gaps,
equal/decreasing versions, corruption, shape changes, nonfinite data, and
degenerate geometry clear the affected causal state and return the identical
input stock bytes object. The deterministic action and ledger contain no loss
or outcome field. This makes PTI executable for future exact replay, but does
not upgrade the historical direction screen into a loss result.

Commit `c2010ed` bridges that state machine to the stored capture-v2
syncer-boundary and sealed-action schema. Syncer schema v2 owns the exact stock
pseudo-gradient CAS object, canonical flat-f32 numel/layout identity, and equal
pre/stock/post/broadcast byte lengths. It
independently checks the memoryless f32 outer update. The adapter accepts no
caller-supplied PTI event or relabel: sequence, fragment, version, shape, and
stock bytes come only from the verified boundary. It requires production RDA,
exact Nesterov LR `0.28`/positive-zero momentum, the declared policy
capabilities, and causal session/sequence/version/layout continuity. It then
publishes a CAS decision proof and a generic sealed action; unsupported or
inconsistent evidence returns `UNIDENTIFIABLE` without advancing policy state.
The stock path preserves exact stock and post-fragment object identities, and
the non-stock path deterministically materializes the f32 result. Commit
`4b145a9` maps the adapter into the closed action/fallback reason vocabulary
while retaining the detailed, policy-generated PTI reason only inside the
hash-bound decision proof. This is boundary/action authority only. It does not
provide the live full-state producer, branch restore, future-group execution,
or A/B then B/A evaluator required for a finite-loss outcome, and it is not a
fresh empirical PTI result.

The machine-readable report is
`docs/optimizer-reports/crp-retained-evidence-report.json`. Its frozen-policy
digest is
`7b8fb30ca98f4b0916f4158824c98246799c61c08d416bfb6bb37d5b2e022710`;
the recorded analyzer-source digest is
`a0c9645ce9cde5b8084497af6c681a2d13c830159150883e078fa8048c24fccb`;
the report SHA-256 is
`90efb32abb93ef57185824e45464a7d6a7b52ee791ffad51211948894d7d52ea`;
the same digest and basename are recorded in
`crp-retained-evidence-report.json.sha256`. Two complete invocations produced
byte-identical reports. No cloud resource, model execution, live outcome,
parameter retuning, post-hoc gate, or push was used.

### CFLX-SGD: cross-fitted lookahead candidate

A later independent review proposed **Cross-Fitted Lookahead Extragradient SGD
(CFLX-SGD)**. At a sparse fragment-balanced boundary it computes the exact
stock trial point `Y = theta - 0.28 G`, evaluates an auxiliary gradient at
`Y`, and extracts only that gradient's component transverse to the stock
direction. It tests the positive grid `{1/32, 1/16, 1/8, 1/4}`, never rotates
past the observed lookahead direction, and norm-grafts the result back to
`||G||`, capping the turn at `atan(1/4) ≈ 14.04°`.

The proposal gradient cannot certify itself. A disjoint, presealed eight-block
validation batch must prefer the rotated trial point with a one-sided 99.5%
Student-t lower bound above zero and relative mean gain above `2e-4`; a third
disjoint stream is audit-only. Both live arms must pay the same probe cost and
buffer the same asynchronous arrivals. Any rejection uses the originally
captured production `G` bytes and the stock SGD-0.28 kernel. Shadow discovery
requires at least 96 scheduled boundaries, action on at least 25% of all valid
boundaries, positive independent finite-loss and next-direction lower bounds,
and three of four positive fragments before a separately preregistered exact
CRN campaign may reuse the numerical k0/k8 and two-seed/product thresholds.
CFLX is not an EXP2-54 hypothesis and changes the capture and multiplicity
design. This is a preregisterable proposal, not an
empirical result or novelty claim. `CrossFittedLookahead.lean` proves the
narrow scalar inequality underlying the proposed normalized-alignment argument
and a separate quadratic counterexample where probe preference is opposite
true-loss preference. The vector/orthonormal cosine interpretation remains an
informal corollary, not an encoded theorem. The direct and current 8,569-job
aggregate builds pass without `sorry`, `admit`, or project axioms; these
results do not prove finite loss, population transfer, convergence, or
superiority over SGD.

## EXP2-52/53 operator history

The exp2-52 run, analysis, matched control, image creation, canary, family
promotion, and exact-ID teardown are complete. Full SCAFFOLD and identity
shuffle are not triggered because `G_res < 0.018`.

The first full BC-MP discovery attempt, `exp2-53a`, was intentionally not
analyzed. All four learners completed their 1,368-step cap and disconnected at
outer commit 316 while the strict-quorum syncer was configured for 340. In
non-barrier pipeline execution, broadcast arrival resets fragment windows, so
the naive `floor(1368/16)*4=340` arithmetic is not a liveness guarantee. With
zero connected learners, commits 317--340 were unreachable. The harness
preserved an explicit incomplete-run record at
`gs://yeto-exp2-52-model-training-497007/exp2-53a`, then deleted exact VM ID
`6940191863701259021` and disk ID `7297892197699437325`; both return 404. Its
316-commit tape is negative provenance, not a completed discovery.

The replacement runner started at `2026-07-14T09:59:32Z` on Spot instance
`exp2-53a2-shadow`, exact VM ID `4065267032092554960`, in `us-central1-c`.
Its exact boot disk was ID `3451429984025687760`, a 250 GB `pd-ssd` sourced
from exact image ID `7290368630472593484` and attached with auto-delete. The
stored argv passed `--learner-max-steps 1600`; pinned commit `9f9edad...` has a
display-only bug whose startup banner still reports the token-derived
`1368 steps/learner`.

The liveness repair worked. The syncer reached exactly outer commit 340 at
`2026-07-14T10:32:57.983Z`; learners 0--3 stopped cleanly at local steps
1,451, 1,459, 1,462, and 1,448 and all saved their adapters. Their 5,820 total
microsteps at 128 tokens give exactly **744,960 aggregate raw local tokens**.
Final evaluation completed and the runner exited zero at
`2026-07-14T10:35:52.895157Z`. The descriptive shadowed-model eval loss was
`1.3601954652806376`, versus the untrained model's `1.5593921925059278`; this
is not a candidate-vs-stock optimizer comparison because no counterfactual
policy was applied.

The run manifest retains the original pre-run family digest
`ccde253df57bcada7ee450d05d32c31b0a3c806228f85e275778088ea32242c3`.
After the run, the reusable metadata guard was hardened to require the larger
of the ideal-relative `1360+128=1488` bound and the empirical
`1471+129=1600` bound. That metadata-only hardening leaves the executed command
unchanged but produces the current family canonical hash
`030b1893c1ace11642acf4da51b5996c6a5fd34ba5e6bb098ebb8951158f7430`
and raw formatted file hash
`118a182436c4379dd738ee6610355aeda7147dc293421c9a78d5682cf23db1a1`.
Do not rewrite historical run provenance to the post-run family hash.

The amended, score-blind analyzer froze `NO_GO` at
`2026-07-14T10:38:00.306615Z`. The raw selection-manifest SHA-256 is
`b4ab816540002e3156b6a2e0a44b9279ad48b9ba2bc93abcbb509d33885b6d04`;
its embedded self-hash is
`cdd65b88eead63728ce8f237d84b8cd3dae1567d9ae16141638f71f3e79212e6`.
No confirmation instance was launched and the committed hidden seed was not
used for training.

All completion data, checksums, the run manifest, the exact amended analyzer,
config, amendment note, and selection manifest are preserved under
`gs://yeto-exp2-52-model-training-497007/exp2-53a2`. After the final sync, the
harness deleted exact VM ID `4065267032092554960`; its auto-delete boot disk ID
`3451429984025687760` is also gone. Independent instance and disk describes
both return 404. There is no active EXP2-53 GCP VM. The retained custom image
remains the reusable launch artifact; future GPU experiments should use exact
image ID `7290368630472593484` and `execution.source_mode=checkout`.

The current EXP2-54 state is the r5h result above: deterministic one-A100
correctness parity passed, the 2% overhead gate failed at 7.964001%, the exact
VM and disk were deleted, and the obsolete r6 draft was not launched. The PTI
online E1 r2 engineering canary also completed and was killed for zero
non-stock actions; its conditional four-A100 E2 was not launched. There is no
optimizer-harness GCP VM or disk, and current A100/A2 usage is zero. The only
running project VM is the unrelated `instance-20260526-guiflow`, which remains
out of scope. No optimizer promotion experiment is authorized until a new
candidate passes its immutable shadow gate and the full live capture-v2
restore/evaluation path passes its cost qualifier.

### 2026-07-14 continuation audit

The optimizer hunt resumed from the checked-out
`experiment/optimizer-state-capture-round3` worktree at commit
`16afcbe6601cfd6bc9aef01dc5f14684eba1141d`. Historical attempts were treated
as retained evidence rather than rerun suggestions. The local Git worktree
inventory contains the earlier barrier qualifier, BC-MP shadow, capture-v2
object-store, committed-boundary index, CRN authority, CRP replay, async
capture, Lean/bake-off, and curvature branches. Their branch tips and durable
documents were cross-referenced against this log. None overrides the frozen
verdicts above: PTI remains killed at E1 for zero actions; BC-MP and the
round-two inventions remain killed or unidentifiable under their stated
gates; and r5h remains a correctness pass but cost-gate failure. Unmerged
historical worktree code is not itself new evidence and will not be silently
promoted into a fresh experiment.

The strongest implemented but unrun candidate remains CPLG-SGD. Its first
admissible new evidence is the already specified fresh, stock-trajectory,
full-vector shadow screen. A GPU launch is still blocked at the implementation
audit boundary: the independent Python evaluator uses Darwin host C
`atan2f`/`sinf`/`cosf`, while production uses pinned Rust `libm = 0.2.15`.
At the angular cap, `sinf` differs by one ULP and one candidate coordinate also
differs by one ULP. The checked-in report therefore correctly says
`UNSATISFIED_UNTIL_RUST_FIXTURE_MATCH`. The planned repair is to keep the
Python geometric/state reference independent while supplying its f32
transcendentals from a small, hash-pinned Rust-libm helper. Rust then becomes
the sole authoritative transcendental evaluator before any fresh outcome is
opened; raw-bit fixtures must prove full candidate parity. This is a numerical
contract repair, not coefficient or gate tuning.

The prior 7.964001% r5h result measured the much broader live exact-state
capture path and does not authorize weakening its 2% gate. CPLG's earlier
shadow stage needs only the exact ordered stock pseudo-gradient vectors and
their hashes, not learner Adam/RNG/data-iterator restore bundles. A separate
low-overhead syncer-owned full-vector tape is therefore the next capture
design to audit. If implemented, it must preserve exact f32 bytes, sequence and
fragment identity, atomic/checksummed publication, a closed writer ledger, and
the frozen 32-boundary/eight-per-fragment denominator. It cannot be described
as capture-v2 finite-loss authority, and a shadow pass would still leave the
full restore/evaluate cost qualifier and same-state CRN stage outstanding.

The resumed Stage-2 audit found additional blockers that were not explicit in
the earlier summary. First, the Rust treatment validator compared momentum
numerically with `0.0`, which admits IEEE-754 negative zero even though CPLG
and PTI require exact positive-zero identity. It must compare raw f32 bits and
gain a negative-zero regression. Second, the existing Python CPLG replay is a
stateless triple-vector construction rather than an independent causal state
machine. It retains history through nonacute/invalid geometry even though
production clears the fragment state; it seals a pending shadow only for a
non-stock vector even though production seals a valid byte-stock zero-phase
candidate; and it neither advances the last-three resolved-score window nor
simulates the three-positive interlock. These differences can change later
eligibility and action counts even when a single candidate vector matches.
The reference must model preview/commit, clearing, valid stock-shadow
resolution, and checkpoint/resume before Stage 2 can pass.

The shadow analyzer also does not yet implement the frozen experiment verdict.
It currently lacks enforcement of 32 ordered boundaries with exactly eight
per fragment, 24 sealed and 20 resolved shadows plus four declared tails, at
least eight simulated actions, the `>0.001` mean score, the specified
fragment-stratified moving-block lower endpoint, three positive fragment
means, the chained final ledger, and matched below-2% overhead. The exact
bootstrap seed/draw/block/multiplicity semantics and the CPLG E2 advancement
gate still require a dedicated immutable dossier. These are launch blockers,
not details to fill after a GPU outcome.

A separate completion audit also found that PTI r2's zero-action failure is
genuine and sufficient to kill advancement, but its validator did not cover
all other frozen E1 requirements: it did not enforce the below-2% interval
overhead or create/verify the planned predecessor-chained action ledger. Its
81.5-second versus 87.9-second descriptive arm times are not a valid proof of
that interval gate. The PTI plan header is chronologically stale, and its old
confirmation ordering predates the active SOP's H16/H64/H256 Stage-6 product
gate. None of these omissions rescues PTI; they narrow the evidence claim and
must not be copied into a future validator.

The live provider audit used the isolated campaign gcloud configuration and
recomputed the latest remote checksums. There is no optimizer VM or disk in
GCP, no pending/running EC2 instance in any enabled region of AWS account
`533462777468`, and no local optimizer/training process at the final sample.
All ten lifecycle records in this worktree are `ABANDONED`; exact provider
lookups return not found for every recorded VM and disk. The PTI r2 validation
digest recomputes as
`ccac85e0447549c2c2b9321bdd52b9c637f754d8aca71cd3a80e6ce51f65b3ce`
and its abandonment digest as
`c2785e27e245d1b56548f50be200d0404dcc4826e4cc4963e5f93d5dba407235`.
The r5h parity digest recomputes as
`2afac80f24bb8274bc79c2fc465cc928389f60016c6d3e9471d6908b6b6ccf4b`
and its abandonment digest as
`f284969aceb426b1143f34d3c17e4d81cd0e9bcd7b5d4fef7086ba0900535c0f`.
No collection or teardown remains for those attempts.

The one live GCP VM is the protected unrelated CPU workload
`instance-20260526-guiflow`, numeric ID `3908640733128066700`, in
`asia-east2-c`; it remains out of scope. Exact image ID
`7290368630472593484` remains intentionally `READY` and consumes no GPU.
Current `us-central1` quota is 16 Spot A100s but only 48 A2 CPUs, so the
quota-feasible allocation is at most one four-A100 `a2-highgpu-4g`, not eight
concurrent A100s. Zonal Spot placement is not guaranteed. Scientific capacity
remains zero until the CPU and shadow prerequisites pass. A stale main-worktree
`exp2-53-smoke` ledger says running even though exact VM ID
`1744019134829053908` and its named disk are absent; its incomplete prefix is
negative provenance and needs only local lifecycle reconciliation, never ID or
prefix reuse.

The first continuation implementation pass now closes the previously listed
selector/reference defects on CPU. The Rust treatment boundary compares the
momentum bits and rejects negative zero. The independent Python reference is a
causal per-fragment preview/commit machine: it resolves the pending shadow
before current geometry, retains only the last three scores, seals valid
zero-phase byte-stock candidates, clears the complete state on nonacute or
invalid geometry, and selects a non-stock action only after three already
resolved strictly positive scores. The replay implements the frozen exact
32-boundary round-robin accounting, 24/20/4 seal/resolve/tail counts, eight-
action minimum, mean and fragment gates, and the 20,000-draw SplitMix64
fragment-stratified circular block bootstrap. It also rejects noncanonical or
escaping vector paths, duplicate JSON fields, discontinuous versions, broken
predecessor chains, incomplete writer accounting, manifest/sidecar mismatch,
and any missing full vector rather than shrinking the denominator.

Production now has an opt-in syncer-owned stock-vector writer rather than the
failed broad learner-state capture path. The capture path independently
materializes the deterministic weighted-RDA stock pseudo-gradient before the
legacy stock commit recomputes the same aggregate from the same sorted inputs;
the matched overhead gate therefore measures this extra capture computation as
well as I/O. It transfers the owned f32 vector through a bounded FIFO, writes
atomic little-endian vector objects, hashes a canonical
predecessor-chained JSONL row for every commit, and closes an atomic manifest
plus checksum sidecar. Capture is accepted only for fresh deterministic
token-weighted stock SGD-0.28 with f32 wire, exact positive-zero momentum and
delta-norm reference, no fragment LR override, and an expected row count equal
to the syncer total. A writer queue/I/O error or incomplete expected count is
fatal. This is direction-shadow evidence only; it contains no Adam, RNG,
future-group, restore, or finite-loss authority.

Pinned Rust `libm = 0.2.15` is now available to the independent Python
reference through a persistent closed-schema raw-bit helper. The exact raw-f32
constant-phase, angular-cap, nonplanar, and reversal fixtures all match the
production Rust expectations when this helper is authoritative. In
particular, the cap candidate is now the production bytes
`9b8c8340954db540`, while the host-Darwin diagnostic remains the deliberately
different `9a8c8340954db540`. The helper executable SHA-256 is reported by the
analyzer, and the frozen shadow gate refuses to pass without it.

The current focused audit passes 38 Python reference/replay tests, including a
fully passing synthetic shadow gate, cost-only failure, raw-bit parity, causal
state, and tape corruption cases. Rust passes all 193 main tests plus both
helper tests; exact vector bytes, ledger head, manifest completion,
preview/commit, checkpoint/resume, legacy behavior, and signed-zero
regressions are included. The complete Lean build passes all 8,570 jobs; the
existing style warnings are unchanged and no Lean source was modified in this
continuation. Targeted Ruff and Rust format checks pass. On this workstation,
bare `uv run` selected CPython 3.14 and could not install pinned Torch 2.8;
verification must explicitly use the available CPython 3.13 and
`PYTHONPATH=.` because the editable package metadata does not expose the
top-level `yeto` package. The cloud argv already has to pin the sibling
checkout/interpreter/import path, so this local discovery is a reproducibility
constraint rather than optimizer evidence. The generated untracked `uv.lock`
is not an experiment artifact and must not enter the implementation commit.

One additional binding gap emerged before finalizing the runner. The current
capture CLI accepts a caller-supplied `layout_sha256`, but the syncer has not
yet proved that value against the actual layout received in HELLO. Stable rows
alone would therefore bind an assertion, not the live layout. Before the
immutable spec can be frozen, the producer must derive a canonical digest from
the established wire layout and exact layout metadata (or verify the supplied
digest against it), and the OFF/ON pair must publish equal exact initial-state
hashes and unrounded monotonic interval evidence. No GPU is authorized while
those producer/run-evidence bindings remain open.

Dry-rendering the inherited M=1/H4 command exposed a separate work-accounting
error before launch. With `--syncer-total-steps 32`, the syncer shuts the
learner down after the same 34 terminal local steps observed in PTI E1. At
sequence length 128 and microbatch one, that is exactly 4,352 raw local
training tokens. The old `--token-budget 32768` label only computed a 256-step
default ceiling that was overridden by `--learner-max-steps 96` and never
reached; it did not make the run consume 32,768 tokens. The CPLG shadow
preregistration now uses the truthful `--token-budget 4352`, retains 96 only
as strict-quorum headroom, and gates the observed 34 steps. This is acceptable
for a stock-direction acquisition screen, not for an active E1 claiming the
SOP's standard 32,768-token profile. Any later CPLG E1 must preregister a
schedule that actually executes its declared token budget; it may not copy the
historical label as proof of equal work.

The frozen CPLG wrapper was tightened further before cloud use. It now hashes
the actual separate scientific configuration file and rejects a compare argv
whose declared configuration digest differs, leaving the immutable cloud spec
to pin the source commit without a self-referential file hash. The analyzer
requires the overhead JSON's basename-bound SHA-256 sidecar, and exact export
tree hashing rejects symlinked directories as well as symlinked files. These
are pre-acquisition integrity controls, not favorable optimizer evidence.
The resulting frozen scientific configuration is
`experiments/optimizer/cplg-sgd-shadow-direction-r1-config.json`, SHA-256
`6636a9a537cfac495c2f4c22f641b2a325f4e79e78da3d876ff70595b49f79d8`;
it contains no observed CPLG direction outcome.

The first pre-outcome consistency scan caught and corrected a draft-only JSON
mismatch: it had transcribed the activity threshold as four simulated actions
and included an extra per-fragment action minimum, while the dossier and
executable gate both freeze eight total actions. No direction outcome existed;
the corrected config removes the extra field, requires eight, and has the
final digest above. The superseded draft digest was never launchable.

The post-hardening local audit passed 45 focused CPLG Python tests, all 196
`yeto-syncer` tests, both `cplg_libm_oracle` tests, Rust formatting, and Ruff
check/format over the seven changed Python files. A first full Python command
using only the `dev` test dependency produced 1,255 passes, two skips, and 12
unrelated failures because the isolated environment omitted the declared
`launcher`/`nava` extras (`skypilot[aws]` and `boto3`). Repeating the same full
suite with the project-declared `dev`, `launcher`, and `nava` extras passed
1,267 tests with two skips. The failed dependency-incomplete invocation is
retained here so the green result is not presented without its environment
qualification.

The final wrapper audit also made the frozen configuration's checksum sidecar
mandatory at runtime and closed the comparison-results cardinality/order
check: exactly one OFF row followed by exactly one ON row is accepted.
Duplicate, extra, or reordered result rows fail before overhead analysis.
The wrapper also builds both Rust binaries with `cargo build --locked` before
acquisition, publishes the clean source commit, Cargo.lock digest, and exact
helper executable digest in a checksummed preflight receipt, and rejects any
post-pair helper digest change. Thus the authoritative transcendental binary
is fixed before either stock arm produces an outcome. Its explicit expected
40-character commit must match HEAD, and Git status must have no tracked or
untracked nonignored files; a fresh remote checkout proves the commit was
available from the pinned repository.

A latest-state audit then found that a correct configuration digest could be
stamped onto an argv with different model/data, seeds, LoRA, or workload
inputs. Before outcomes, the wrapper was changed to parse the checksummed JSON
and compare every frozen scientific input to exactly one explicit compare
option. The profile also made the previously source-defaulted learner choices
explicit: dense-model all-linear LoRA, AdamW defaults (betas 0.9/0.999,
epsilon 1e-8), weight decay 0.01, ten-step linear warmup then constant LR,
clip norm 1.0, gradient checkpointing off, DDP, preload tokenization, and
assistant-token cross entropy. The corrected final configuration digest is
`6636a9a537cfac495c2f4c22f641b2a325f4e79e78da3d876ff70595b49f79d8`;
the preceding digest was never used to launch or inspect an outcome.

The final pre-outcome config revision also caps the whole campaign at one A100
and closes the retry table: scientific `FAIL` has no retry; `INCONCLUSIVE` or
`UNIDENTIFIABLE` requires a newly reviewed protocol and identity; only one
fresh-identity `INFRA_FAILURE` retry is allowed after evidence preservation,
exact-ID teardown, and a documented root-cause repair. Its final digest is the
`6636a9a5...9f79d8` value above.

The wrapper now has a distinct terminal supervisor artifact. It atomically
publishes checksummed `PASS` or `FAIL` after a valid analyzer result,
`INCONCLUSIVE` for post-acquisition evidence-validation failure, and
`INFRA_FAILURE` for pre-acquisition/configuration/build/compare failure. A
nonfresh terminal path prevents execution, so a traceback or missing analyzer
report can no longer masquerade as an unclassified completed attempt.

A subsequent pre-outcome implementation audit found four evidence-quality
gaps: the two runtimes did not yet consume one full causal raw-bit trace; the
standalone analyzer could replace a stale report; unexpected ordinary
exceptions could escape the terminal supervisor; and the documentation
incorrectly said the capture aggregate was materialized only once. All four
were corrected before acquisition. Both runtimes now consume the same
11-boundary fixture and compare action, candidate, complete causal state,
theta, delayed-score window, clear/recovery behavior, and resumed continuation
by raw f32 bits. It includes a first selected non-stock action, a nonacute
clear, a negative-zero coordinate, byte-stock zero-phase sealing, and a later
exact positive-zero score. The fixture exposed one real diagnostic-only drift:
Rust wrote `non_acute_turn` while Python wrote `nonacute_turn`. Rust was changed
to the already frozen analyzer vocabulary; the numerical state and action bytes
already agreed. Analyzer and wrapper publication is now no-clobber even for
dangling symlinks, the outer supervisor catches and records unexpected
exceptions as `INFRA_FAILURE`, and the capture description now truthfully says
that capture rematerializes the deterministic aggregate before the legacy
stock path recomputes it. The overhead gate includes that extra work.

The same closure pass made `device=cuda` and `gpu_slots=1` explicit scientific
fields and changed the no-outcome configuration digest to
`6636a9a537cfac495c2f4c22f641b2a325f4e79e78da3d876ff70595b49f79d8`.
The validator now enforces the complete frozen mechanism, analysis, gate, and
lifecycle constants, rechecks clean Git/Cargo/helper identity after acquisition
and after analysis, and requires the analyzer's helper digest to equal the
checksummed preflight digest. Final local verification passed 89 focused Python
tests, 197 syncer tests, two helper tests, the locked release build, Ruff and
Rust formatting checks, and 1,281 Python tests with two skips under CPython
3.13 plus the declared `dev`, `launcher`, and `nava` extras. No direction
outcome existed during any of these corrections.

The complete implementation was then committed cleanly as
`a4122c5f6caffb03636e114609e2feb136a7f5ae` (`Add frozen CPLG stock-shadow
acquisition`) and pushed to
`origin/experiment/optimizer-state-capture-round3`; an independent
`git ls-remote` returned that exact 40-character SHA. The configured HTTPS
credential was expired and the first push failed before remote mutation, so
the already authenticated GitHub SSH transport was used without changing the
commit or repository settings. From that exact clean commit, the full Python
suite again passed 1,281 tests with two skips, Rust passed 197 syncer plus two
helper tests, the locked release build succeeded, Ruff and both format checks
passed, and `lake build LeanMechanism.CausalPhaseLockedGeodesic` completed all
2,190 jobs with only the pre-existing short-copyright-header warnings.

The immutable acquisition specification is
`experiments/optimizer/exp2-cplg-shadow-direction-r1.json`, basename-bound
SHA-256
`5fb637265f0b2abd6e3ee784247473da0aa4f85804c9f715d0522cd2ef5416e7`.
It pins the pushed implementation commit, exact image ID, one Spot A100 with a
campaign maximum of one, isolated checkout/run/prefix identities, complete
argv and environment, config/dossier/Cargo identities, all input manifests,
the terminal/analyzer/overhead/preflight receipts, both arm receipts and
stock artifacts, and all 32 deterministically named full-vector files. Three
spec contract tests pass. `optimizer_experiment.py validate` passed, rendered
the intended OFF-then-ON stock-only command, and the provider-aware `doctor`
under `/private/tmp/yeto-gcloud-admin-codex` reported cloud prerequisites OK at
`2026-07-14T20:42:32Z`. No GPU, VM, disk, run prefix, or direction outcome had
been created or opened at that point.

### 2026-07-14 CPLG stock-shadow acquisition r1: preserved `INCONCLUSIVE`

After the independent gate and resource audits reported no blocker, the exact
frozen r1 spec was launched on one Spot A100. The harness owned VM
`exp2-cplg-shadow-direction-r1` / provider ID
`1959194496687873622`, boot disk ID `1112489218094603862`, ownership nonce
`d8b6d70ab54f6b04`, and retained source-image ID `7290368630472593484`.
The runtime preflight bound implementation commit `a4122c5f...`, config digest
`6636a9a5...`, `Cargo.lock` digest `2f5e404a...`, and Linux Rust-libm helper
digest `72fa765d43b71463753024adff3afb2d86284ea2422a427e1b91347ad6360bbb`.

Both stock arms reached exactly 34 local steps and 32 outer commits. Their
canonical layout digest was
`85cdff92c745f0f3e221c573b60a39f0cb14cd35e4313e4f5086428cf7d6fdd9`
and their exact initial-state digest was
`3b398acd56d9df580416856f8c7d86a97f34590671faf8617504687c3adca931`.
OFF and ON evaluation losses were bit-identical at
`1.3319227656615953`; the untrained base row was `1.4244070067489154`.
The ON writer closed with 32 accepted/completed items, no dropped or
abandoned bytes, 173,113,344 vector bytes, tape digest
`b27eb88d82f8fe1608e577f90b3b171e28b40bbfeb7eeb5c7633bcb9c7dcba62`,
ledger head
`07acb144dc935d24d39921c74474ddd593367e9bf53a80868668cdf4aa751fec`,
and manifest digest
`203e83d48331ea3ad7267eec62bb1685d3bfa8aea850beb28697b2a759346a0b`.
The unrounded producer intervals were 46,887,555,224 ns OFF and
47,205,244,120 ns ON, a diagnostic capture overhead of approximately
0.67755%; the rounded full-arm wall rows were 80.5 s OFF and 80.3 s ON.

These observations are diagnostics only. Before the replay analyzer opened a
CPLG direction verdict, the post-acquisition validator rejected
`results.jsonl`: it expected exactly `[OFF, ON]`, whereas
`compare_diloco.py --skip-baseline` truthfully and always writes
`[base (untrained), OFF, ON]`; `--skip-baseline` omits the separately trained
synchronous baseline, not the untrained evaluation row. The checksummed
terminal therefore records `INCONCLUSIVE`, stage
`post_acquisition_evidence_validation`, error class `EvidenceError`, and the
runner exited 3. This attempt is not a direction PASS or FAIL, does not
authorize active E1, and will not be retrospectively reclassified or analyzed
to tune the frozen mechanism.

The complete failed attempt was explicitly synced to
`gs://yeto-exp2-52-model-training-497007/exp2-cplg-shadow-direction-r1`
before teardown: 103 objects totaling 450,179,942 bytes were present,
including both tapes/checkpoints/exports/receipts, all 32 vector objects,
preflight, terminal receipt, results, logs, and provenance. Exact-ID
`abandon` wrote abandonment record digest
`03cc82f3fbb14b89a83cf86500a50feb8703dc7c4a8abca4a2138ffc823a239b`
and completed at `2026-07-14T21:01:23Z`. Subsequent provider lookups returned
not-found for both owned VM and disk. The protected unrelated CPU VM
`instance-20260526-guiflow` remained RUNNING with ID
`3908640733128066700`.

The failed-run teardown check exposed a second pre-retry lifecycle defect:
clean checkouts produced zero-byte `git-status.txt` and `git-diff.patch`, but
the harness requires every declared completion path to be nonempty before
`delete`. Thus even a runner-zero PASS would have been undeletable through the
successful-run path. The repair writes explicit nonempty clean-checkout and
no-diff attestations after the existing full tracked/untracked cleanliness
guard. The comparison validator repair accepts exactly the real closed schema
`[base (untrained), OFF, ON]`, validates all three finite losses, and still
requires bit-identical OFF/ON losses. A new attempt requires a reviewed
protocol, source commit, run ID, capture UUID, artifact prefix, immutable spec,
and fresh audit; r1 evidence may be used only to diagnose these controller
defects.

The fresh r2 controller protocol is now frozen before any r2 direction
outcome. It accepts only the real closed result order
`[base (untrained), cplg_shadow_off, cplg_shadow_on]`, validates all three
finite losses, and still requires OFF/ON bit identity. The harness writes
nonempty clean-checkout and no-diff attestations only after its existing full
tracked/untracked cleanliness rejection. R2 uses run ID
`exp2-cplg-shadow-direction-r2`, capture UUID
`667f5de8-6d6d-4ce0-9344-efc239583abf`, configuration
`cplg-sgd-shadow-direction-r2-config.json`, and configuration SHA-256
`fb7d4c0539cc8760058e0f0b20101bde7fcbac9224c8b27ca69d9724180aaf96`.
The new dossier explicitly excludes all r1 vectors and diagnostic values from
r2 replay, bootstrap, gates, and tuning. Before commit, 131 focused
CPLG/replay/wrapper/spec/harness tests passed; the full Python suite passed
1,285 tests with two skips and five unchanged dependency warnings, and Ruff
format/check passed. No cloud resource or r2 artifact prefix existed while
these controls were written.

The immutable r2 cloud launch contract is
`experiments/optimizer/exp2-cplg-shadow-direction-r2.json`, SHA-256
`b39ec7d4faec2691895a2ea41d94df0950e130212a26ad8abbe80d5efe989efd`.
It pins repaired runtime source commit
`eb6d21146011112ffe8df5cb518c985e8c0297bd`, retained image ID
`7290368630472593484`, one `a2-highgpu-1g` Spot VM in `us-central1-c`, a
3,600-second maximum run, and the fresh r2 artifact prefix. The contract test
also binds the truthful three-row result schema; its focused spec/wrapper run
passed 23 tests. Harness validation and command rendering passed before the
control-plane commit and launch.

## Primary references used in the audit

- SCAFFOLD: <https://proceedings.mlr.press/v119/karimireddy20a.html>
- Mime and coordinated optimizer statistics:
  <https://arxiv.org/abs/2008.03606>
- Original DiLoCo: <https://arxiv.org/abs/2311.08105>
- ReLoRA optimizer-state reset precedent: <https://arxiv.org/abs/2307.05695>
- AdamP momentum projection precedent: <https://arxiv.org/abs/2006.08217>
- FedGaLore optimizer-state/subspace alignment:
  <https://arxiv.org/abs/2602.01746>
- DES-LOC desynchronized optimizer-state timescales:
  <https://arxiv.org/abs/2505.22549>
- FedAdamW state-reset/second-moment precedent:
  <https://arxiv.org/abs/2510.27486>
- MT-DAO multi-timescale momentum: <https://arxiv.org/abs/2510.05361>
- Distributed adaptive optimization with local updates:
  <https://arxiv.org/abs/2409.13155>
- Local SGD versus minibatch SGD:
  <https://proceedings.mlr.press/v119/woodworth20a.html>
- Riemannian Preconditioned LoRA: <https://arxiv.org/abs/2402.02347>
- LoRA-RITE: <https://arxiv.org/abs/2410.20625>
- Google Cloud Spot lifecycle:
  <https://docs.cloud.google.com/compute/docs/instances/create-use-spot>
- Google Cloud custom image creation:
  <https://docs.cloud.google.com/compute/docs/images/create-custom>
- Google Cloud image family promotion (`gcloud compute images update`):
  <https://docs.cloud.google.com/sdk/gcloud/reference/compute/images/update>
