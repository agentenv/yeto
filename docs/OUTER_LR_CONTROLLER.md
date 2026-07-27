# Age-aware outer-LR controller

The syncer can optionally multiply its configured outer learning rate by a
scalar chosen from the outer-momentum age, the planned per-fragment horizon,
and an optional probe-measured spectral sketch. The controller changes only the
learning-rate scalar. It does not change the Nesterov buffer recursion or the
merged pseudo-gradient.

The default remains disabled. With no controller flag, the syncer follows the
pre-controller path byte for byte.

## Modes and indexing

For momentum `mu`, one-indexed per-fragment age `t`, and planned per-fragment
horizon `T_planned`, the three modes are:

1. `transient`: apply the code-true direction-multiplier normalization

   ```text
   q_transient(mu, t) = 1 / (1 - mu^(t+1)).
   ```

   This is the exact rule used by the existing `--outer-bias-correction` flag.
   The legacy flag remains available as a compatibility alias.

2. `measured-drift`: multiply the transient scale by a versioned factorial
   surface supplied as JSON:

   ```text
   q(mu, t, T_planned, sketch)
     = q_transient(mu, t) * 2^surface(mu, t, T_planned, sketch).
   ```

   The JSON output is a direct log2 LR multiplier. A negative surface value
   lowers the LR; the syncer does not invert it. When all surface coefficients
   are zero, the drift multiplier is exactly `1.0` and the complete output is
   bit-identical to `transient` mode.

3. `oracle`: return the scheduled scale for `(fragment, t)` exactly. No
   transient normalization is added.

The syncer derives `t` from the checkpointed round-robin fragment version. For
fragment `fid` among `F` fragments, its scheduled global steps are
`fid+1, fid+1+F, ...`. `T_planned` is the count of those steps not exceeding
`--total-steps`, so uneven final rounds are represented exactly. A resumed run
therefore resumes at the same controller age.

## Syncer flags

```text
--outer-lr-controller transient

--outer-lr-controller measured-drift \
--outer-lr-drift-surface /path/to/surface.json \
--outer-lr-spectral-sketch /path/to/sketch.json

--outer-lr-controller oracle \
--outer-lr-oracle-schedule /path/to/oracle.json
```

`--outer-lr-spectral-sketch` is optional when no nonzero surface term consumes
a spectral feature. `transient` and `measured-drift` require the Nesterov outer
optimizer. All explicit modes currently require `token_weighted` commits so a
probe/CTTN action cannot bypass or compound the controller silently.

`--outer-bias-correction` and `--outer-lr-controller` are mutually exclusive.
Surface, sketch, and oracle flags are rejected unless their matching controller
mode is selected.

## Drift surface JSON

The mechanism lane produces this strict interface:

```json
{
  "schema": "yeto.outer_lr_drift_surface.v1",
  "output": "log2_lr_multiplier",
  "features": [
    {
      "name": "u",
      "source": "mu",
      "center": 0.0,
      "scale": 0.9
    },
    {
      "name": "q",
      "source": "T_planned",
      "center": 5.0,
      "scale": 10.0
    },
    {
      "name": "h",
      "source": "spectral.window_steps",
      "transform": "log2",
      "center": 9.0,
      "scale": 1.0
    }
  ],
  "terms": [
    {"coefficient": -0.2033958365, "powers": {"u": 1}},
    {"coefficient": -0.4691937641, "powers": {"q": 1, "u": 1}},
    {"coefficient": -0.1068146276, "powers": {"h": 1, "u": 1}}
  ],
  "drift_scale_bounds": {"min": 0.125, "max": 8.0}
}
```

Each named feature is evaluated as

```text
feature = (transform(source) - center) / scale.
```

`transform` is optional and is one of `identity` (default), `log2`, or `ln`.
`center` defaults to `0.0`; `scale` defaults to `1.0` and must be nonzero.

Built-in sources are:

| source | value |
|---|---|
| `mu` | configured outer momentum |
| `t` | one-indexed current per-fragment age |
| `T_planned` | planned per-fragment update count |
| `age_fraction` | `t / T_planned` |
| `remaining_steps` | `T_planned - t` |
| `remaining_fraction` | `(T_planned - t) / T_planned` |
| `spectral.NAME` | named finite scalar from the spectral sketch |

Each term contributes
`coefficient * product(feature_name ^ integer_power)` to the log2 multiplier.
An empty `powers` object is an intercept. Powers must be in `1..=16`. Terms are
evaluated in array order, so a frozen producer should emit a deterministic
order. A term with coefficient exactly zero is skipped before resolving its
features; this is the defined zero-surface reduction behavior.

`drift_scale_bounds` is required, must be positive and ordered, and must contain
`1.0`. Bounds clamp the surface in log2 space before exponentiation, then the
bounded drift multiplier is multiplied by the transient factor.

## Spectral sketch JSON

Probe output is a map of arbitrary named finite scalars. Fragment features
override global features with the same name.

```json
{
  "schema": "yeto.outer_lr_spectral_sketch.v1",
  "global_features": {
    "window_steps": 1024.0,
    "lambda_max": 0.03125,
    "trace": 1.75
  },
  "fragment_features": {
    "0": {"lambda_max": 0.027},
    "3": {"lambda_max": 0.041}
  }
}
```

The schema does not assign semantics to names. The probe producer and surface
producer own that contract together. At startup, the syncer rejects every
active nonzero surface term whose required spectral value is missing for any
scheduled fragment.

## Oracle schedule JSON

Oracle arrays are one-indexed by convention and addressed as `scales[t-1]`.
A fragment override replaces the default array.

```json
{
  "schema": "yeto.outer_lr_oracle_schedule.v1",
  "scales": [1.0, 0.9, 0.8],
  "fragment_scales": {
    "0": [1.25, 1.0, 0.75]
  }
}
```

Every scheduled fragment must resolve to an array whose length is exactly its
derived `T_planned`. Every scale must be finite and positive. Extra fragment
ids, missing schedules, and short or long arrays are startup errors.

## Reference and telemetry

The CPU reference is `yeto.outer_lr_controller`. Its public
`outer_lr_scale(...)` function implements the same scalar boundary as the Rust
module, and its JSON classes validate the same schemas.

When active, each event-tape row appends:

```text
outer_lr_controller_mode
outer_lr_scale
outer_lr_transient_scale
outer_lr_drift_scale
```

The last two values are `null` when they do not apply. The legacy
`outer_bias_correction` tape field remains present for modes with transient
normalization, preserving existing analysis consumers. No controller fields are
emitted on the default path.
