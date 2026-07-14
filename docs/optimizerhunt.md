# Optimizer hunt: experiment log, harness, and formal audit

Last updated: 2026-07-14 (America/Los_Angeles)

This is a working operator handoff, not yet a durable repository source of
truth. The document, harness, analyzers, specs, tests, and new Lean sources are
untracked, and supporting tracked files—including `scripts/compare_diloco.py`,
the Lean root import/README, and `.gitignore`—also have uncommitted changes on
`lean-anchor-drift@4a43c39b88f9ae174c3741d75586f3f8cbc3fbd8`. A clean checkout
cannot reproduce the local harness/analyzer/Lean claims until the complete
change set is reviewed, committed, and pushed. The BC-MP shadow code used by
`exp2-53a` and `exp2-53a2` is independently pinned to pushed commit
`9f9edadf3e8a39c91ce5d64d5ac0a93ca22424c4` on
`origin/experiment/bcmp-shadow-round2`.

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

Local state is written under `.optimizer-harness/state/` and ignored by Git.
It contains no cloud or Hugging Face credential.

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
still a preflight
sanity check, not a proof of liveness under arbitrary non-barrier asynchronous
scheduling.

The current harness suite passes 52 focused tests. The BC-MP analyzer passes
11 tests plus three subtests, the de-confound analyzer passes five tests, and
the combined focused harness/analyzer suite passes 68 tests plus three
subtests. Remaining desirable work is a fully staged runner in which
each arm has an immutable completion manifest and can restart independently
after Spot preemption. The current `compare_diloco.py` is still monolithic and
deletes its work directory at startup, so the harness does not pretend it can
resume a partially completed arm.

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

AWS is not currently viable for this four-A100 job:

- us-east-1 P-Spot quota is 128, with 96 consumed by the protected p4de node;
- the increase request to 256 remains `CASE_OPENED` as of 2026-07-14;
- other US p4d placement scores were poor;
- eu-west-1 had capacity but zero P-Spot quota.

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

Compact reports for these retained-tape screens are mirrored under
`docs/optimizer-reports/`; the hashed preregistrations, decisions, summaries,
and executables still live under `/tmp`. The full CPER-SGD artifact set is also
archived at
`gs://yeto-exp2-52-model-training-497007/optimizer-screens/cper-sgd-20260714`;
the earlier screens do not yet have equivalent durable object prefixes.
Neither working-tree location is committed. Preserve the complete evidence
before treating this table as a reproducible repository result record.

These results leave exact outer SGD-0.28 as the production optimizer. They also
sharpen capture requirements: HMC needs bias-corrected `exp_avg_sq`, clocks, and
first post-broadcast gradients; NG-TR needs exact full-f32 H/2 and H endpoints
with identity metadata. Neither quantity may be fabricated from endpoint norms
or adjacent rounds.

The corresponding local algebraic mechanisms are implemented in the untracked
working-tree file `lean-mechanism/LeanMechanism/OptimizerRound2.lean`:
Richardson cancels a
quadratic-in-time scalar displacement term, harmonic scalar preconditioner
consensus preserves the weighted immediate step under a common numerator and
eliminates metric-only pairwise dispersion, and delete-one jackknife cancels
the first-order `a/M` term in the standard bias expansion. Local `lake build`
and direct checks pass without `sorry`; these artifacts are not durable until
committed and pushed. The replay results, not these local identities, determine
the optimizer decisions.

## Formal results and limits

Lean 4.31 with Mathlib was used for small mechanism results. In the current
local working tree, `lake build` succeeds and no `sorry`, `admit`, or newly
declared axiom was found. The new files `BCMPAdamW.lean`,
`TransverseWin.lean`, and `OptimizerRound2.lean` and their imports are not yet
committed or pushed; treat them as local audit artifacts, not checked-in
proofs. Their `#print axioms` output has Mathlib's conventional `propext`,
`Classical.choice`, and `Quot.sound` dependencies. None is a theorem about
language-model loss.

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

### Qualifier and GCP sequence

The first GPU spend is a behavior-preservation qualifier, not a candidate
score. One command runs matched four-learner `capture_m4_off` and
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

Only a passing qualifier can unlock seed-223 H16 **state acquisition**. Its
minimum is 32 complete committed boundaries, at least eight per fragment.
It does not unlock candidate scoring. Before MTRF/MSTP development can begin,
the joined-bundle materializer and full CRN capture/restore/evaluation layer
must be implemented, independently tested, and frozen against the existing
schema. Only then do the preregistered mean-k8, Holm-adjusted lower-bound, and
action/safety gates apply. Seed 239 remains locked until one immutable winner,
configuration, and analysis hash are selected. There is no honest
winner ETA while full CRN state is absent. The qualifier is expected to finish
in about 30 minutes but now has a one-hour VM safety envelope so checkout,
validation, evaluation, upload, and teardown are not racing a 30-minute
provider auto-delete. Development and a separately authorized confirmation
retain two-hour envelopes.

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
`experiment/optimizer-state-capture-round3`. The current pre-commit tree passed
755 Python tests, 171 Rust tests, Ruff lint/format, the replay self-test, and all
8,567 Lean build jobs. Its reproducible commit identity is recorded below after
the tree is frozen. Independent launch audits found and closed the reconnect,
cold-start timing, portable parity-input, image-input provenance, and too-short
VM-envelope holes. No EXP2-54 GPU instance may be created until the code is
committed and pushed, the launch derivative pins that exact code commit, and
the newly quota-aware doctor is green. Draft and confirmation specs remain
`adopt_only` until their stage gates pass.

### PTI-SGD: next temporal candidate, not yet an empirical result

A separate agent review proposed **Prequential Transverse Interlock SGD
(PTI-SGD)** to address the most consistent prior failure: strong same-round
held-worker directions becoming negative on the next same-fragment boundary.
For unit current and previous production directions, it extracts only the
previous direction's component orthogonal to the current direction and tests a
fixed signed coefficient grid
`{0, ±1/32, ±1/16, ±1/8, ±1/4}`. Every candidate is norm-grafted back to the
exact SGD-0.28 pseudo-gradient norm, so its maximum possible rotation is
`atan(1/4) ≈ 14.04°`.

Every signed coefficient is shadow-scored on every valid boundary whether or
not it is selected. The router may use a nonzero coefficient only when that
coefficient's three most recent valid **counterfactual shadow scores** all
improved the sealed next same-fragment cosine. A nonpositive score disables
application but does not stop later shadow scoring, so three new consecutive
positive scores can restore eligibility. Missing continuity,
degenerate transverse geometry, nonfinite state, or a hash mismatch returns
bit-identical SGD. Unlike CPER-SGD, which compared two large fixed full-vector
experts and abstained 76/76 times, PTI tests small signed transverse turns,
uses a recent causal interlock rather than a lifetime cumulative winner, and
is killed if it acts on fewer than 25% of all predefined valid post-warm-up
boundaries; integrity exclusions and warm-up are fixed before outcomes, so the
policy cannot shrink the denominator by declaring difficult cases ineligible.

This is only a falsifiable proposal. Its shadow gate requires at least 32
post-warm-up decisions, eight per fragment, mean sealed next-fragment cosine
gain above 0.001 with a positive block-bootstrap lower bound, at least three
positive fragments, and nontrivial action. A separate PTI preregistration may
reuse the numerical fresh paired SGD-0.28 k0/k8 thresholds, but PTI is not a
third EXP2-54 hypothesis and cannot inherit its two-candidate Holm/ranking rule.
The Lean file
`PrequentialTransverseInterlock.lean` proves only the normalized alignment
identity and its stationary-direction counterexample: every nonzero
orthogonal turn is worse when the next direction is unchanged. It proves no
training-loss or convergence superiority. The direct module check and complete
Lean build pass with no `sorry`, `admit`, or project axiom; the audited theorem
dependencies are only `propext`, `Classical.choice`, and `Quot.sound`.

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
informal corollary, not an encoded theorem. The direct and 8,567-job aggregate builds pass
without `sorry`, `admit`, or project axioms; these results do not prove finite
loss, population transfer, convergence, or superiority over SGD.

## Current operator sequence

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
