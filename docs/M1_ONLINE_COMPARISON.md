# Milestone-1 online comparison evidence

`python -m yeto.rl.m1_online_comparison COMPARISON.json` verifies the closed
Qwen3.5-4B full-parameter online comparison. It does not launch training and it
does not manufacture missing evidence.

The comparison has two paired arms for each of seeds 17, 29, and 43:

- one centralized native Miles learner using four trainer GPUs and four TP1
  inference workers;
- two dense `H=1` learner islands, each using two trainer GPUs and two TP1
  inference workers.

Each global round gives the centralized learner two groups in one optimizer
step. The dense arm gives each island one group in one local optimizer step,
then performs one full-roster dense merge. Thus both arms use four training and
four inference GPUs and the same two groups and six trajectories per round,
while their truthful optimizer-receipt counts differ: one centralized receipt
versus two concurrent island-local receipts.

## Closed study contract

The comparison manifest precommits, and hashes before loading results:

- the immutable model and tokenizer revisions and initial full-policy identity;
- the full-parameter layout, container, Yeto source, and Miles source;
- train, held-out, TaskPack, prompt-schedule, reward, sampling, optimizer, and
  evaluation contracts;
- policy rounds, accepted groups and trajectories, trained tokens, per-arm
  optimizer receipts, and the fixed four-training/four-inference GPU topology;
- seeds `17,29,43`; and
- the pass@1 and result non-inferiority rule and absolute margins.

Every arm record binds its launch, accounting, optimizer, publication,
held-out, and final-policy artifacts by relative path and SHA256. The verifier
rejects non-canonical JSON, duplicate keys, missing or extra fields, symlinked
artifact paths, changed hashes, incomplete seed/arm matrices, non-contiguous
policy versions, unreconciled learner/global totals, missing update norms,
unpaired prompt or generation-seed schedules, unpaired evaluation seeds, and
v0 metric differences.

A valid comparison can still return `not_demonstrated`. It returns
`non_inferior` only when the dense-minus-centralized held-out improvement is
within each predeclared absolute margin for every seed and for the paired mean.

## Canonical producer adapter and current gap

The canonical adapter target is
`yeto-qwen35-m1-online-arm-result/v1`. An adapter may construct that record only
by parsing and validating real producer artifacts; operators must not hand-fill
the summary fields.

For the dense arm, the planned source is the direct launcher's canonical
`yeto-m1-dense-final-report-v1`, backed by its launch manifest, both island
evaluation summaries, learner event tapes, Miles metric histories, syncer
ledger, trajectory-accounting aggregate, and policy-publication receipts. A
dense adapter must derive the ordered global-round roster and bind every
derived field to those source artifacts.

That adapter is intentionally unsupported at present. The current dense
producer retains per-island input-batch hashes but does not retain a canonical
combined prompt manifest or training generation-seed manifest. Its held-out
summary retains exact policy identity, scalar metrics, and sample count, but
not the evaluation generation-seed, sample-manifest, or evidence hashes needed
to prove pairing. Relabeling existing hashes would be false evidence.

There is also no same-compute centralized full-parameter SecRLEnv producer.
`scripts/benchmark_rl.py` is the existing LoRA benchmark, and the direct M1
launcher currently has only a four-GPU one-round gate and the eight-GPU
two-island dense final mode. The missing producer surface is therefore:

1. persist canonical combined prompt and rollout-generation seed manifests in
   both online arms, plus canonical evaluation seed/sample/evidence manifests;
2. add the eight-GPU centralized-native full-parameter SecRLEnv arm with the
   same initial inputs, four-training/four-inference topology, two groups per
   round, and the same terminal report evidence;
3. freeze the dense final-report schema, implement strict dense and centralized
   adapters, and run both arms for all three seeds.

Until those producer gaps are closed, the verifier is an executable exit
contract, not evidence that Milestone 1 has passed.
