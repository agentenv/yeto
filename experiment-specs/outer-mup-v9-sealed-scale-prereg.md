# Outer-muP V9 sealed scale preregistration

Status: **SEALED BEFORE ANY V9 VERIFICATION CELL**.

`MECHANISM: SURFACE-FALLBACK`

The referee confirmed no spectral lane on full data. The point predictions therefore use the G6-selected empirical surfaces: raw F3 and corrected F1, with coefficients copied from the complete mechfit refits after exact equality to G6. This registration supports an empirical transport claim only; it does not support a static-spectral-mechanism claim.

## Frozen predictions

| model | arm | predicted eta* | verification eta ladder | error band (bits) |
|---|---:|---:|---|---:|
| SmolLM2-1.7B | raw | `0.00164330670357` | `[0.0008216533517871457, 0.0013042933949801124, 0.0020704367072666044, 0.003286613407148583]` | 0.625 |
| SmolLM2-1.7B | corrected | `0.000794068520673` | `[0.00039703426033673177, 0.000630252602525944, 0.001000463644255379, 0.001588137041346927]` | 0.625 |
| Qwen2.5-7B | mu0 | `0.0110450931863` | `[0.00656745170151272, 0.011045093186307909, 0.018575558533019097]` | 0.500 |
| Qwen2.5-7B | raw | `0.00303047740489` | `[0.0018019326458734598, 0.0030304774048872667, 0.005096635172554164]` | 0.500 |

Sealed prediction SHA-256: `97e02dcad63782978ac51b320621e5a681236518cb0d5db19454b8981549ca9c`.

The 1.7B T=10/S=5120/H=512 raw and corrected coordinates were never run before this seal. The 7B stage is T=5/S=2560/H=512 with seeds {901,907}, M=4, and one GPU per learner.

## Prospective gates

- G9A uses bands `{'corrected': 0.625, 'raw': 0.625}`, near-bracket allowance `0.5` bits, and at least `7500` valid refits.
- G9B uses bands `{'mu0': 0.5, 'raw': 0.5}`, near-bracket allowance `0.5` bits, and at least `7500` valid refits.
- The exact two-seed 10,000-draw bootstrap groups are copied from the pre-outcome gatesim. No post-seal recentering, ladder edit, band edit, target substitution, or analyzer edit is permitted.

## Execution order

1. Push the exact registration commit.
2. Run and drain all 16 1.7B cells.
3. Run the registered Qwen one-step admission smoke, then run and drain all 12 7B cells in four 4-GPU queues.
4. Run the frozen analyzer and publish G9 without outcome-aware edits.
