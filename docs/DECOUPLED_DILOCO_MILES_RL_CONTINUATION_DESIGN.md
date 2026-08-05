# Decoupled Miles RL Phase Continuation

## Status and Scope

This document defines one narrow RL capability: starting a new Decoupled
DiLoCo Miles RL phase from the final PEFT adapter produced by an earlier
completed phase.

This is a policy warm start, not an in-place extension of a terminal run. It
does not preserve the previous inner optimizer, scheduler position, outer
optimizer momentum, syncer checkpoint, fragment versions, rollout IDs, or
partially completed groups.

The following remain outside scope:

- changing a learner budget while a run is active;
- reopening or extending a finalized syncer session;
- exact optimizer-state continuation across phases;
- non-Decoupled, SFT, full-parameter, actor/critic, or Diffusion training;
- Hub or object-store parent adapters;
- changes to Miles, the Rust syncer, or the wire protocol.

## Contract

A continuation phase accepts two new public inputs:

```text
--rl-initial-adapter PATH
--rl-initial-adapter-sha256 SHA256
```

`PATH` must be a local directory exported by Yeto's Decoupled RL exporter. The
launcher computes its directory SHA256 and compares an explicitly supplied
digest when present. It then mounts the same immutable directory read-only at
every logical island and passes the computed digest to each learner.

The learner fails before the first rollout unless all of the following hold:

1. the mounted directory digest equals the launch-time digest;
2. `adapter_config.json` declares causal-LM LoRA with the requested rank,
   alpha, zero dropout, no bias, base model, immutable model revision, no
   RS-LoRA/DoRA/fan-in-fan-out modifier, and no rank or alpha pattern;
3. `adapter_model.safetensors` contains exactly the canonical PEFT names,
   shapes, and finite f32-compatible values expected by the current model;
4. `yeto_rl_provenance.json` identifies a Decoupled RL export and its recorded
   policy hash equals the loaded tensor-policy hash.

Legacy pickle weights are never loaded.

## Initialization Flow

Miles still creates the Megatron actor, optimizer, and SGLang engines normally.
At the existing external-policy-sync initialization boundary, Yeto exports the
fresh actor only to discover and verify its canonical tensor contract. When a
parent adapter is configured, Yeto loads that adapter as policy version zero
instead of using the freshly initialized LoRA values.

Learner 0 initializes a fresh syncer from that canonical parent policy. Every
island receives the resulting complete version-zero cut, verifies its tensor
hash, applies it to Megatron with a fresh optimizer state, and publishes it to
all SGLang engines before rollout zero. The normal Decoupled lifecycle then
continues unchanged.

The new phase therefore has this identity:

```text
parent policy tensors: preserved exactly
local optimizer state: fresh
outer optimizer state: fresh
rollout/optimizer step: 0
fragment versions:     all 0
```

## Recovery and Provenance

The parent adapter directory path is operational and may change after a
restart; it is not checkpoint identity. Its directory SHA256 is immutable and
is included in a phase checkpoint only when warm start is enabled. Existing
non-warm-start checkpoint payloads remain unchanged.

Each island records one `rl_initial_adapter` event containing the parent
directory SHA256 and canonical tensor-policy hash. On a fresh phase, the
version-zero syncer cut must have that same policy hash. A mismatch is fatal
before rollout.

Normal in-phase checkpoint recovery remains unchanged: the learner restores
its local optimizer-step counter, receives the committed syncer cut, resets
the process-local optimizer as already required by recovery, and continues the
same phase. It does not reload progress from the parent adapter.

Each fresh phase uses a new Yeto `--cluster-prefix`. Reusing an existing run
identity retains the existing in-phase syncer and island recovery semantics;
it does not create another fresh phase.

## Validation

Automated coverage must prove:

- a valid final Decoupled adapter loads as canonical policy version zero;
- digest, model revision, LoRA contract, tensor layout, non-finite tensor, and
  provenance-policy mismatches fail closed;
- the Decoupled initialization hook seeds the syncer from the parent and
  rejects a different version-zero cut;
- launcher validation accepts only Decoupled RL with a local adapter, computes
  the digest, mounts one directory on every island, and passes both learner
  flags;
- existing runs without a parent adapter retain their current arguments and
  checkpoint configuration.
