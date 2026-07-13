#!/usr/bin/env python3
"""Golden-trace Lean<->Rust consistency check for the anchor-drift formalization.

Purpose (see docs/ANCHOR_DRIFT_CONTROL.md, docs/OPTIMIZER_SEMANTICS.md):
build one tiny trace (2 param dims, 2 workers, 3 commits, explicit base model /
local endpoints / server current / Nesterov buffer) and run it through THREE
merge semantics, in EXACT rational arithmetic so the same numbers can be
asserted bit-for-bit in (1) the real Rust syncer functions (merge_avg +
nesterov_step; see syncer/src/tests/golden_trace.rs) and (2) the Lean model
(lean-mechanism/LeanMechanism/*.lean).

Production ground truth being checked (merge.rs / state.rs):
  * per-worker pseudo-gradient  delta_m = anchor - upload_m   (anchor MINUS learner)
  * anchor is self.params[fid], the syncer's CURRENT global fragment  <-- current-anchor
  * merge_avg = weighted MEAN (not sum) of per-worker deltas, weight w_m
  * Nesterov: b_t = mu b_{t-1} + delta_t ; d_t = delta_t + mu b_t ; theta -= lr d_t
  * buffer init b_0 = 0

The three semantics differ ONLY in which `anchor` each worker's delta is taken
against:
  barrier              : anchor = worker's base global, and (by the barrier) the
                         base equals the server current at merge => anchor_drift=0
  streaming vmatched   : anchor = worker's declared base global (base_version tag)
  streaming curr-anchor : anchor = server current global at merge (PRODUCTION)

All trace quantities are chosen to be exactly representable in f32 (dyadic
rationals) so f32 == exact and Rust asserts hold with zero tolerance.
"""
from __future__ import annotations
from fractions import Fraction as F
from dataclasses import dataclass, field
import json, sys

Vec = tuple  # (F, F)

def vadd(a, b): return (a[0] + b[0], a[1] + b[1])
def vsub(a, b): return (a[0] - b[0], a[1] - b[1])
def smul(s, a): return (s * a[0], s * a[1])

def merge_avg(anchor, uploads, weights):
    """Exact analogue of merge::merge_avg: weighted mean of (anchor - upload)."""
    wsum = sum(weights)
    out = (F(0), F(0))
    for up, w in zip(uploads, weights):
        wf = F(w) / F(wsum)
        out = vadd(out, smul(wf, vsub(anchor, up)))
    return out

def nesterov_step(theta, buf, delta, lr, mu):
    """Exact analogue of merge::nesterov_step. Returns (theta', buf')."""
    buf2 = vadd(smul(mu, buf), delta)          # b_t = mu b_{t-1} + delta_t
    direction = vadd(delta, smul(mu, buf2))    # d_t = delta_t + mu b_t
    step = smul(lr, direction)                 # eta d_t
    theta2 = vsub(theta, step)                 # theta -= eta d_t
    return theta2, buf2, direction, step

@dataclass
class Commit:
    uploads: list         # per-worker theta_m
    weights: list         # per-worker weight
    bases: list           # per-worker declared base global (for vmatched)

def run(commits, lr, mu, theta0, anchor_mode):
    """anchor_mode in {'current','vmatched'}.
    'current' == production (anchor = server current);
    'vmatched' == anchor = each worker's declared base (per-worker delta).
    Barrier is 'vmatched' with the invariant bases==current (anchor_drift=0)."""
    theta = theta0
    buf = (F(0), F(0))
    steps = []
    for k, c in enumerate(commits):
        if anchor_mode == 'current':
            delta = merge_avg(theta, c.uploads, c.weights)
        else:  # vmatched: each worker vs its own declared base
            wsum = sum(c.weights)
            delta = (F(0), F(0))
            for up, w, base in zip(c.uploads, c.weights, c.bases):
                wf = F(w) / F(wsum)
                delta = vadd(delta, smul(wf, vsub(base, up)))
        theta2, buf2, direction, step = nesterov_step(theta, buf, delta, lr, mu)
        steps.append(dict(
            commit=k,
            delta=delta, buf_before=buf, buf_after=buf2,
            direction=direction, step=step, theta_after=theta2,
            anchor_drift=[vsub(theta, b) for b in c.bases],  # current - base per worker
        ))
        theta, buf = theta2, buf2
    return steps

def f2s(v):
    return [str(v[0]), str(v[1])]

def main():
    lr, mu = F(1, 2), F(1, 2)
    theta0 = (F(0), F(0))
    # Commit 1: fresh, equal weights, both bases = theta0
    c1 = Commit(uploads=[(F(1), F(0)), (F(0), F(1))], weights=[1, 1],
                bases=[(F(0), F(0)), (F(0), F(0))])
    # Commit 2: UNEQUAL weights 1:3 (tests weighting), both bases = theta1 (no lag)
    theta1 = (F(3, 8), F(3, 8))
    c2 = Commit(uploads=[(F(1, 2), F(1, 2)), (F(1, 4), F(1, 2))], weights=[1, 3],
                bases=[theta1, theta1])
    # Commit 3: worker B LAGS (base = theta1 while server is at theta2) -> anchor_drift
    theta2 = (F(25, 64), F(17, 32))
    c3 = Commit(uploads=[(F(1, 2), F(1, 2)), (F(1, 2), F(1, 2))], weights=[1, 1],
                bases=[theta2, theta1])
    commits = [c1, c2, c3]

    cur = run(commits, lr, mu, theta0, 'current')
    vm = run(commits, lr, mu, theta0, 'vmatched')

    # sanity: theta1/theta2 declared above equal the current-anchor trajectory
    assert cur[0]['theta_after'] == theta1, (cur[0]['theta_after'], theta1)
    assert cur[1]['theta_after'] == theta2, (cur[1]['theta_after'], theta2)

    out = {'lr': str(lr), 'mu': str(mu), 'theta0': f2s(theta0),
           'current_anchor': [], 'version_matched': []}
    for tag, tr in (('current_anchor', cur), ('version_matched', vm)):
        for s in tr:
            out[tag].append(dict(
                commit=s['commit'],
                delta=f2s(s['delta']),
                buf_before=f2s(s['buf_before']),
                buf_after=f2s(s['buf_after']),
                direction=f2s(s['direction']),
                step=f2s(s['step']),
                theta_after=f2s(s['theta_after']),
            ))
    # divergence at commit 3 (current-anchor vs version-matched)
    out['commit3_current_delta'] = f2s(cur[2]['delta'])
    out['commit3_vmatched_delta'] = f2s(vm[2]['delta'])
    out['commit3_workerB_anchor_drift'] = f2s(vsub(theta2, theta1))
    print(json.dumps(out, indent=2))
    # human-readable
    sys.stderr.write("\n=== current-anchor (PRODUCTION) ===\n")
    for s in cur:
        sys.stderr.write(f"c{s['commit']}: delta={f2s(s['delta'])} buf'={f2s(s['buf_after'])} theta'={f2s(s['theta_after'])}\n")
    sys.stderr.write("=== version-matched ===\n")
    for s in vm:
        sys.stderr.write(f"c{s['commit']}: delta={f2s(s['delta'])} buf'={f2s(s['buf_after'])} theta'={f2s(s['theta_after'])}\n")
    sys.stderr.write(f"commit3 divergence: current={out['commit3_current_delta']} vmatched={out['commit3_vmatched_delta']} (workerB anchor_drift={out['commit3_workerB_anchor_drift']})\n")

if __name__ == '__main__':
    main()
