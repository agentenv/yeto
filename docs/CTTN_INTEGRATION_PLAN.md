# CTTN Integration Plan (Rust syncer + sidecar wire)

The Python CTTN math is done + validated (`yeto/cttn.py`, `cttn_torch.py`,
`cttn_sidecar.py`; tests in `scripts/test_cttn*.py`). This wires it through the
action-probe sidecar so the syncer can commit a CTTN step. Plan from a Plan-agent
pass over the actual files, 2026-07-13.

## Architectural decision
Ship CTTN as a new **commit policy `ProbeCttnV1`** + a new **sidecar verb
`cttn_step`**, NOT an `OuterOptimizer` enum variant. Rationale: `apply_outer_step`
/`materialize_applied_step` (merge.rs:138/206) are pure, synchronous, torch-free
— CTTN's `d`/`b_new` need the async HVP sidecar, which only `perform_merge`
(server.rs:1382, already async, already holds `ActionProbeClient`) can trigger.
CTTN uniquely *inverts* the "preview built in Rust, sidecar only selects"
contract: the sidecar *computes* `d`,`b_new` that become state. `--outer-optimizer`
stays the fallback/control committed if the probe is unavailable/fails.

## Experiment-design decision (RESOLVED)
CTTN's `mu` must be 0.9 (it damps momentum) while the SGD-0.28 control AND the
failure fallback must be memoryless (`mu=0`). Same `--outer-momentum` knob can't
serve both, so add a **separate `--cttn-mu` (default 0.9)**; `--outer-momentum`
stays the fallback/control momentum (0 for the pre-registered experiment).

## Top 3 risks
1. **Fragment ordering** — flat order of `g`/`b` (Rust fragment tensor order) ==
   `request.fragment_names` == sidecar HVP param-tuple order. Guarded by the one
   reused validation gate (action_probe.py:1397) + one serialization loop.
2. **`g` sign/definition** — `g` = the POST-renormalization merged delta
   (state.rs:799), sign `anchor − upload`; the exact vector plain-nesterov feeds
   as `delta`. Wrong pre-renorm/sign breaks `q^Td==||g||`. Centralize in a new
   `state.rs::cttn_inputs`. Rust-side defensive check: recompute `q·d` vs `||g||`.
3. **Preview/materialize parity** — the dedicated CTTN commit path builds
   `ApplyPreview` directly (`applied_step=lr·d`, `resulting_params=params−lr·d`,
   `resulting_optimizer_buffer=b_new`); it must NOT route through
   `apply_outer_step`/`materialize_applied_step` (the norm-equality assert at
   state.rs:881 crashed a prior rho-adaptive mismatch). Parity exact by
   construction (same `d`).

## File-by-file (implement in this order)
1. **yeto/action_probe.py** — extract `_parse_state_block` helper; add
   `CttnRequest` + `parse_cttn_request` (state block + `g`,`b` f32 payloads +
   `mu`,`rho`,`block_steps`, per-tensor sha256, unclaimed-bytes check); add
   `build_cttn_result_frame` (header type `cttn_result` with `d`,`b_new` specs +
   `diagnostics` object + payload `d‖b_new`). Reuse frame/_slice_f32/digest.
2. **yeto/action_probe_server.py** — `attn_implementation="eager"` on model load
   (141-146); `ActionProbeReplica.cttn_step` (re-validate identity + fragment
   order 1397; `_apply_state`; `params=tuple(self.params[n] for n in
   request.fragment_names)`; call `cttn_sidecar.cttn_sidecar_step(model, params,
   self.panels, g, b, mu, rho, block_steps)`; return d/b_new/diagnostics); worker
   dispatch `op=="cttn_step"` branch; `WorkerPoolBackend.cttn_step` (single
   worker, no A0-A4 fan-out); refactor `handle`→`(header,payload)` + a
   `cttn_step` branch + `_serve_connection` send(header,payload).
3. **syncer/src/action_probe.rs** — `CommitPolicy::ProbeCttnV1` (Display/FromStr
   `cttn_v1`, requires_probe true, is_shadow/is_loo false, no multipliers);
   `build_cttn_request` (reuse state loop, append `g`,`b` payload tensors in
   fragment order, `cttn` header object); `cttn_step(...)`→`verify_cttn_response`
   (parse `cttn_result`, slice+sha256 `d`,`b_new`, finite, **q·d≈||g|| f64
   check**) → `VerifiedCttn{d,b_new,diagnostics,request_digest}`.
4. **syncer/src/state.rs** — `cttn_inputs(aggregate,…)->{g(post-renorm),
   b=momentum[fid].clone(), outer_lr, mu}`; `commit_cttn_step(aggregate,
   target_version, d, b_new, outer_lr)`: build `applied_step=lr·d`,
   `resulting_params=params−applied_step`, `resulting_optimizer_buffer=b_new`,
   carry all other optimizer state unchanged, fingerprint + `commit_preview`
   /`install_preview`; NO `apply_outer_step`.
5. **syncer/src/server.rs** — `Config.cttn_rho`,`Config.cttn_mu` + `--cttn-rho`
   /`--cttn-mu` flags; `perform_merge` arm `else if ProbeCttnV1`: build aggregate,
   `cttn_inputs`, baseline fallback preview; if probe unavailable→fallback; else
   `client.cttn_step(...).await` → Ok:`commit_cttn_step` / Err:warn+baseline;
   thread `v.diagnostics` into `CommitDecision`+tape (bind,tau,retention,
   e_before,e_after,budget,n_modes_90,ritz_max,loss).
6. **scripts/compare_diloco.py** — `"cttn"`→OUTER_OPTIMIZERS, `"cttn_v1"`→
   COMMIT_POLICIES; when `outer_optimizer=="cttn"` emit `--commit-policy cttn_v1`
   + `--outer-optimizer <fallback>` + `--cttn-mu 0.9`; add `--cttn-rho` (0.10).
   Anchor manifest already IS the held-out HVP source — no new data wiring.
7. **Local parity/smoke** on the tiny LoRA-Llama before GPU.

## Watch items
- `allow_unused=False` in `make_hvp`: a fragment param that doesn't influence the
  loss on the sampled panels → `autograd.grad` raises → fail-closed to baseline
  (safe, but expected in logs). Consider `allow_unused=True` + zero-fill.
- Single-worker CTTN leaves other replicas idle (HVP is the cost) — acceptable.

## Pre-registered experiment (recap; see docs/CTTN_DESIGN.md)
2 H16 workloads × 4 seeds × {SGD-0.28, CTTN} + scalar-HVP control + H256
sentinel. Success: CTTN > SGD by >0.018 on both H16 workloads, correct sign every
seed, ≤0.009 worse on H256, AND beats the scalar-HVP control by ≥0.009. Else bank
SGD-0.28 and write the negative result.
