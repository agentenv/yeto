# Outer-muP program v1 — registered closure record (2026-07-25)

Program verdict: **STOP_G1_NOT_EVALUABLE_AUDIT_PAPER** (per the frozen contract's
closed vocabulary; contract: outer-mup-2day-prereg @ ace8978).

## G1 readout (frozen analyzer b8a3d542..., post-freeze None-guard 22d33355... in
## secondary-only block; readout: h200-n1:/root/g1-readout.json)
- D_obs(H16)  = 1.0266 [1.0133, 1.0416]  (10k paired seed bootstrap)  — (1-mu) law holds
- D_obs(H512) = 2.5116 [2.4550, 2.5685]                              — 2.5x deviation, CI excludes 1
- eta*(mu.9) UNBRACKETED at H64 and H256 (ladders centered on the first-order prior;
  true optima above top rung) -> requirement "all required eta optima interior" false.
- rho_hat lag-1 = None at H64/H256 despite present telemetry (analyzer aggregation
  defect, under forensic analysis); defined at H16 (0.6923) and H512 (0.6163).
- Registered D_pred from raw lag-1 rho: 2.95 (H16) vs observed 1.03 — the raw lag-1
  cosine is noise-compressed and is not a valid estimator of effective rho.
- Verdict basis: unbracketed required optima + undefined rho => NOT_EVALUABLE.
  Per contract: "No one-sided or outcome-aware extension is allowed" -> STOP honored,
  no in-flight ladder extension, no estimator swap inside v1.

## What v1 established (pilot yield, all evidence sealed on-node + hash-recorded)
1. H-dependent breakdown of the (1-mu) equivalence confirmed with seed CIs at the
   two instrumented-valid H values; direction and magnitude match the A100-era
   single-seed campaign.
2. Instrument defect A: eta ladders for mu.9 at intermediate H must be centered on
   an H-dependent prior, not the first-order prior.
3. Instrument defect B: raw lag-1 pseudo-gradient cosine underestimates effective
   rho (noise compression); a noise-corrected estimator is required.

v2 (confirmatory, fresh seeds, corrected instruments) is registered separately and
treats every v1 quantity as pilot/calibration input, disclosed as such.
