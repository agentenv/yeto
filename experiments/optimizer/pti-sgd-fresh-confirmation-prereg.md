# PTI-SGD fresh-confirmation freeze

Freeze date: 2026-07-14 (America/Los_Angeles)

Status: post-development protocol frozen after inspecting the retained
seed-53/67/79 direction tapes and before acquiring or scoring any fresh PTI
capture. Those three tapes are development evidence only and are permanently
ineligible for confirmation.

## Fixed optimizer action

At a valid boundary for fragment `f`, let `G_t` be the exact production
pseudo-gradient and let `G_prev` be the immediately preceding, hash-chained
production pseudo-gradient for the same fragment. All dot products, norms,
normalization, and final materialization use the declared deterministic f32
kernel and tensor order; no BLAS-dependent reduction is permitted.

For nonzero finite directions, define

```text
u = G_t / ||G_t||
v = G_prev / ||G_prev||
p_raw = v - dot(v, u) * u
p = p_raw / ||p_raw||
c = -1/4
C_raw = u + c * p
C_t = ||G_t|| * C_raw / ||C_raw||
```

The only non-stock coefficient is exactly `-1/4`. The other coefficients from
the retained-tape screen are not eligible alternatives, so there is no live
tie-break, coefficient search, sign flip, or adaptive magnitude. The maximum
turn from stock is `atan(1/4)`, approximately 14.04 degrees, and the final
norm is exactly grafted to the production norm before the unchanged outer
SGD-0.28 kernel is called.

The policy applies `C_t` only when all of the following were true before the
action was sealed:

1. `G_prev` is the unique immediately preceding same-fragment direction in a
   gap-free event/hash chain;
2. every required byte object, tensor layout, version, fragment identity, and
   causal boundary identity verifies;
3. both directions and every intermediate scalar are finite and nondegenerate;
4. the three most recent valid, already-resolved shadow scores for coefficient
   `-1/4` on this fragment are each strictly positive; and
5. none of those three records is from the boundary currently being decided.

The sealed score for boundary `t` is resolved only when the next valid factual
same-fragment direction becomes available:

```text
z_t = cos(C_t, G_next) - cos(G_t, G_next)
```

It is written even when the live interlock was closed, so eligibility cannot
hide adverse counterfactual scores. A zero or negative resolved score closes
the interlock until three newer consecutive positive scores exist. Missing
continuity, integrity failure, nonfinite or degenerate geometry, missing
history, or an open tail clears the three-score window and returns the
original `G_t` bytes to SGD-0.28 without re-encoding. It is never coded as a
zero vector or as a newly encoded copy of stock.

## Capability and causal requirements

Every scored boundary must pass the capture-v2 contracts for exact learner
endpoint restore, exact syncer pre/post boundary restore, ordered responders
and f64 weight bits, complete next-eight group IDs, all RNG streams, data
iterator position, fixed evaluation object, source/image/model/data/config
digests, and reconstruction of the factual production broadcast bytes. Any
missing capability yields `UNIDENTIFIABLE`, not an abstention, zero action, or
failed outcome.

For each complete boundary, a policy-agnostic evaluator must restore isolated
baseline and PTI branches from the same immutable state. It applies one exact
outer action, evaluates the same fixed object at `k=0`, consumes the same next
eight actual update groups, and evaluates again at `k=8`. It runs both A/B and
B/A arm orders; state hashes, batch hashes, and within-arm losses must be
identical across order. Any cross-arm mutation or order effect invalidates the
boundary. Equal probe/evaluation work is charged to both arms.

The primary paired effect is

```text
D_t,k = NLL_SGD-0.28,t,k - NLL_PTI,t,k
```

so positive is better. The direction score is a secondary mechanism measure
and cannot replace finite-loss evidence.

## Fresh development-replication gate

One fresh, precommitted capture seed must provide at least 32 complete CRN
boundaries, balanced at least eight per fragment. PTI advances only if all of
the following pass without exclusions chosen after outcomes are opened:

- action on at least 25% of all predefined valid post-warm-up opportunities;
- mean sealed next-same-fragment direction gain above `0.001`, a positive
  fragment-stratified moving-block 95% lower endpoint, and at least three of
  four positive fragment means;
- mean paired `k=8` NLL gain above `0.002`, a positive fragment-stratified
  boundary-bootstrap 95% lower endpoint, at least three of four positive
  fragment means, and at least 60% positive individual boundaries;
- mean `k=0` gain not below `-0.001`, with its fifth percentile above `-0.01`;
- no individual NLL regression worse than `0.05` at either horizon;
- exact stock reconstruction and bit-identical fallback on every boundary;
  no integrity failure is excluded from the denominator; and
- candidate compute plus required capture/replay overhead below 2% of the
  matched stock path, measured on the same committed interval.

The bootstrap uses 20,000 deterministic replicates with seed `5318008`, with
boundaries as clusters and fragment-stratified resampling. PTI is the only
hypothesis in this fresh campaign; MTRF, MSTP, CRP, CFLX, and every alternative
PTI coefficient are separate families and cannot be substituted after results
are visible.

## Confirmation and breadth

Passing the fresh gate authorizes exactly one frozen PTI candidate for five
new paired online seeds against simultaneous SGD-0.28 controls. Each pair must
match code, exact image, initialization, data/order, H, token budget, LoRA
shape, inner optimizer, quorum, RDA/merge path, evaluation objects, and compute
cost. Promotion requires a positive paired aggregate confidence interval, at
least four of five seed effects positive, no seed regression worse than
`0.009`, and mean gain above the campaign's existing practical threshold
`0.018` on at least one core workload with a plausible second core workload.

Even that result is a campaign pass, not a general optimizer claim. A claim of
generalization additionally requires a separately frozen matrix spanning H16,
H64, and H256, at least one additional model family, and both SGD and AdamW
inner optimizers. Failures retain exact SGD-0.28. Thresholds, coefficient,
interlock length, eligibility denominator, horizons, and workloads cannot be
changed in response to the fresh outcomes.

## Formal limit

The existing Lean PTI result proves only a normalized alignment identity and
a stationary-direction counterexample. It does not prove lower loss,
convergence, or superiority. This protocol makes the empirical claim causal
and falsifiable; it does not turn it into a theorem.
