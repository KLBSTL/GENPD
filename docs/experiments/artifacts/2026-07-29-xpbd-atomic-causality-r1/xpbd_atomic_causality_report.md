# XPBD Atomic-Scatter Causality Probe

## Protocol

- Commit `f2e62687704160d4903990a964b34c3a11fc3e7b`; GPU `NVIDIA GeForce RTX 3070 Laptop GPU`; driver `581.57`.
- Moving-sphere cloth, 128x128, rendered 1600x900; 20 warm-up + 120 measured frames; checkpoint stride 10.
- All runs use 32 XPBD iterations/frame, the same collision settings, GPU synchronization, disabled VSync, and no forced CPU state roundtrip.
- A/B/B/A process order: atomic Jacobi, fixed-order vertex gather, fixed-order vertex gather, atomic Jacobi.
- This is a trajectory diagnostic, not a timing comparison.

## Repeat Results

| Pair | Checkpoint SHA-256 all equal | Position P95 relative L2 | Velocity P95 relative L2 | Velocity-difference P95 RMS |
| --- | --- | ---: | ---: | ---: |
| atomic-a:atomic-b | no | 6.101e-05 | 9.650e-03 | 8.208e-04 |
| gather-a:gather-b | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## Physical Comparability

| Metric | Atomic mean/max | Gather mean/max | Difference | Gate |
| --- | ---: | ---: | ---: | ---: |
| P95 mean stretch | 0.005120 | 0.005120 | 8.301e-05 relative | <= 2.000e-02 |
| P95 max stretch | 1.019825 | 1.019830 | 4.903e-06 relative | <= 2.000e-02 |
| Max penetration | 0.001000 | 0.001000 | 1.200e-07 absolute | <= 1.000e-05 |

Physical comparability: **PASS**.

## Verdict

**strong-support-for-atomic-scatter-causality**

The atomic repeat is nonidentical and above the registered velocity signal, while the no-atomic vertex-gather repeat is checkpoint-hash identical and physically comparable. This is strong causal evidence that the float atomic scatter, rather than CPU roundtrip or moving contact alone, is the dominant source of the observed 128x128 run-to-run divergence.

## Artifacts

- `manifest.json`: pre-registered controls and conclusion rule.
- `run_quality_summary.csv`: finite-state and physical-quality checks.
- `trajectory_pair_errors.csv`: matched checkpoint errors and exact SHA-256 equality flags.
- `causality_summary.csv` and `causality_summary.json`: compact verdict inputs.
- `runs/<id>/`: raw rendered profiles, metadata, logs, and checkpoint binaries.
