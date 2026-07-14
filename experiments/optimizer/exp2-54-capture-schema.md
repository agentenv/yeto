# Capture contract for MTRF/MSTP replay

Each line of the campaign index references one immutable boundary bundle:

```json
{
  "schema": "exact_adam_midpoint_capture_v1",
  "boundary_id": "seed223-frag0-local12",
  "seed": 223,
  "fragment": 0,
  "worker_ids": [0, 1, 2, 3],
  "responder_order": [2, 3, 1, 0],
  "h": 16,
  "accepted_mid_steps": [8, 8, 8, 8],
  "accepted_end_steps": [16, 16, 16, 16],
  "state_npz": "boundaries/.../state.npz",
  "optimizer_metadata_json": "boundaries/.../optimizer.json",
  "crn_manifest_json": "boundaries/.../crn.json",
  "source_commit": "...",
  "image_digest": "sha256:...",
  "model_digest": "sha256:...",
  "data_digest": "sha256:...",
  "analysis_config_digest": "sha256:..."
}
```

`state.npz` minimally contains arrays:

```text
theta0, theta_mid, theta_end          float32/master [workers, coordinates]
exp_avg_mid, exp_avg_end              optimizer dtype [workers, coordinates]
metric_mid, metric_end                exact vhat numerator [workers, coordinates]
step_mid, step_end                    int64 [workers]
weights                               float64 [workers]
bounds                                int64 [tensor_count + 1]
lr_mass_first, lr_mass_second         float64 [workers]
baseline_direction                    float32 [coordinates]
next_direction                        float32 [coordinates], optional sealed target
loss_baseline_k0/loss_mtrf_k0/loss_mstp_k0   scalar, post-evaluation
loss_baseline_k8/loss_mtrf_k8/loss_mstp_k8   scalar, post-evaluation
```

The compact CPU skeleton assumes weight-decay movement has already been
removed from `a` and `b` for MTRF. A production recorder must additionally
store raw parameters and the exact decay decomposition so the analyzer can
verify this rather than trust it. `metric_*` must be the state actually used
by Adam: `max_exp_avg_sq` for AMSGrad, otherwise `exp_avg_sq`.

`optimizer.json` records beta values, epsilon placement, step convention,
AdamW/AMSGrad flags, per-parameter-group scheduler/decay, clipping/scaler
events, dtype/master-weight/fused behavior, and exact state hashes.

`crn.json` identifies the full model/buffer and optimizer restore points, CPU
and CUDA RNG states, sampler state, next eight batch-group IDs and hashes,
fixed evaluation microbatch, deterministic arm order, and sealed action hashes.

The recorder must write capture inputs before candidate evaluation and append
loss/timing outcomes through a checksummed result object; it must never rewrite
the input bundle after a loss becomes visible.
