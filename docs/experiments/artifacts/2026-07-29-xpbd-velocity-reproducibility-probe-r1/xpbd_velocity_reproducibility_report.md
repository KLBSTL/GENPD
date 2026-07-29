# XPBD 128x128 Velocity Reproducibility Probe

## Scope

This is a rendered trajectory diagnostic, not a performance benchmark. Every run uses atomic `gpu-xpbd-jacobi`, 32 iterations/frame, a 1600x900 rendered viewport, GPU synchronization, disabled VSync, 20 warm-up frames, and 120 measured frames. State checkpoints are exported every 10 frames and are excluded from any timing claim.

The design distinguishes two sources of variation:

- **Within-condition repeats:** `resident-a:resident-b` and `forced-a:forced-b` measure ordinary run-to-run variation with the same state-management condition.
- **Across-condition pairs:** all four resident/forced combinations measure the effect associated with crossing the finalized position/velocity state through CPU between frames.
- **Scene control:** `hanging` contains no collision primitives; `moving-sphere` includes the moving sphere and plane. A much larger moving-sphere cross-condition result implicates collision/contact amplification, but does not prove one individual shader instruction is at fault.

## Checkpoint Comparison

| Scene | Pair type | Pair | Matched checkpoints | Position P95 relative L2 | Velocity P95 relative L2 | Velocity-difference P95 RMS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| hanging | within-condition-repeat | resident-a:resident-b | 12 | 6.452e-05 | 9.689e-03 | 8.224e-04 |
| hanging | within-condition-repeat | forced-a:forced-b | 12 | 4.112e-05 | 5.429e-03 | 4.608e-04 |
| hanging | across-condition | resident-a:forced-a | 12 | 7.807e-05 | 1.213e-02 | 1.029e-03 |
| hanging | across-condition | resident-a:forced-b | 12 | 9.181e-05 | 1.349e-02 | 1.145e-03 |
| hanging | across-condition | resident-b:forced-a | 12 | 4.277e-05 | 5.888e-03 | 4.997e-04 |
| hanging | across-condition | resident-b:forced-b | 12 | 4.633e-05 | 7.339e-03 | 6.228e-04 |
| moving-sphere | within-condition-repeat | resident-a:resident-b | 12 | 8.021e-05 | 1.273e-02 | 1.083e-03 |
| moving-sphere | within-condition-repeat | forced-a:forced-b | 12 | 3.941e-05 | 5.876e-03 | 4.998e-04 |
| moving-sphere | across-condition | resident-a:forced-a | 12 | 5.415e-05 | 9.125e-03 | 7.761e-04 |
| moving-sphere | across-condition | resident-a:forced-b | 12 | 6.299e-05 | 1.003e-02 | 8.531e-04 |
| moving-sphere | across-condition | resident-b:forced-a | 12 | 4.825e-05 | 7.389e-03 | 6.285e-04 |
| moving-sphere | across-condition | resident-b:forced-b | 12 | 3.785e-05 | 7.363e-03 | 6.263e-04 |

## Quality Sanity

| Scene | Condition | P95 mean stretch | P95 max stretch | Maximum penetration |
| --- | --- | ---: | ---: | ---: |
| hanging | resident-a | 0.005109 | 1.019820 | 0.000000 |
| hanging | resident-b | 0.005103 | 1.019790 | 0.000000 |
| hanging | forced-a | 0.005107 | 1.019750 | 0.000000 |
| hanging | forced-b | 0.005108 | 1.019810 | 0.000000 |
| moving-sphere | resident-a | 0.005121 | 1.019800 | 0.001000 |
| moving-sphere | resident-b | 0.005122 | 1.019840 | 0.001000 |
| moving-sphere | forced-a | 0.005121 | 1.019820 | 0.001000 |
| moving-sphere | forced-b | 0.005122 | 1.019790 | 0.001000 |

## Interpretation

- `hanging`: largest within-condition velocity P95 is `9.689e-03`; median across-condition velocity P95 is `9.733e-03`; classification: **repeat-scale**.
- `moving-sphere`: largest within-condition velocity P95 is `1.273e-02`; median across-condition velocity P95 is `8.257e-03`; classification: **repeat-scale**.
- The moving-sphere cross-condition scale is comparable to the hanging control. This probe does not isolate moving contact as the main amplifier.
- A `condition-sensitive` result is evidence against calling the 128x128 value a single-run accident. It still does not prove transfer corruption: the counterfactual preserves scalar state values but changes host/device crossings and next-frame buffer upload timing. XPBD also uses floating-point atomic accumulation, and moving collision introduces discontinuous projection branches; both can amplify a small state perturbation over many frames.
- A `repeat-scale` result instead means the observed magnitude is comparable to ordinary run-to-run variation, so the prior resident/forced velocity discrepancy cannot be attributed confidently to the roundtrip condition alone.

## Evidence Boundary

This probe establishes only the source scale of the observed 128x128 velocity difference. It does not establish bitwise deterministic XPBD, an equal-quality performance ranking, or a general resolution-independent conclusion. The existing formal R3 study remains the source for rendered 300-frame residency timing and state-traffic measurements.

## Artifacts

- `manifest.json`: exact protocol, commit, GPU, and driver.
- `planned_runs.csv`: the eight rendered trajectory runs.
- `run_quality_summary.csv`: finite-state and physical-quality checks for every run.
- `trajectory_pair_errors.csv`: all matched checkpoint errors.
- `reproducibility_summary.csv`: compact per-pair P50/P95/max metrics.
- `runs/<scene>/<condition>/`: raw CSVs, metadata, logs, and checkpoint binaries.
