# Minimal Optimizer-Effects Recheck

## Protocol

- Commit `0e0fa79` on `NVIDIA GeForce RTX 3070 Laptop GPU` (driver `581.57`).
- Rendered 1600x900 end-to-end timing with GPU synchronization, vsync disabled, 150 measured + 30 warm-up frames, and six interleaved process repetitions.
- Baseline: 386^2 moving sphere, default physics, one NCG iteration per frame. Stress: the same scene/mesh/physics with eight NCG iterations per frame.
- Quality is a separate 120 measured + 20 warm-up rendered run against the archived CPU-NCG reference checkpoints; every condition must satisfy P95 position relative L2 <= 1e-3 with no invalid frame.
- A treatment is a material benefit only when its paired mean frame-time improvement is at least 3%, the bootstrap 95% interval is entirely positive, and it is faster in at least five of six blocks. The symmetric rule marks a material regression; intervals contained in +/-3% are no-material-effect; all other outcomes are inconclusive. A comparison with either condition failing the quality gate is reported as raw timing only and receives no performance verdict.

## Condition Summary

| Suite | Condition | Frame ms | Gradient+stats ms | P95 position error | Gate | K | History | Armijo failures |
|---|---|---:|---:|---:|---|---:|---|---:|
| baseline | edge-scatter-k8 | 12.6157 +/- 0.6958 | 0.2002 | 5.506e-04 | pass | 8 | none | 0.0 |
| baseline | gather-fusion-k8 | 12.7853 +/- 0.5703 | 0.4833 | 5.505e-04 | pass | 8 | none | 0.0 |
| baseline | vertex-gather-k8 | 12.8603 +/- 0.6600 | 0.4978 | 5.505e-04 | pass | 8 | none | 0.0 |
| stress | adaptive-k4-frame-history | 9.1156 +/- 0.2591 | 0.0006 | 2.163e-03 | fail | 4 | frame | 0.0 |
| stress | adaptive-k4-no-history | 9.2491 +/- 0.1338 | 0.0007 | 2.163e-03 | fail | 4 | none | 0.0 |
| stress | edge-scatter-k8 | 15.9315 +/- 0.3332 | 0.1909 | 2.164e-03 | fail | 8 | none | 0.0 |
| stress | fixed-k4 | 6.5388 +/- 0.2598 | 0.0007 | 2.163e-03 | fail | 4 | none | 0.0 |
| stress | gather-fusion-k8 | 18.2909 +/- 0.7901 | 0.4876 | 2.163e-03 | fail | 8 | none | 0.0 |
| stress | vertex-gather-k8 | 18.5286 +/- 0.8878 | 0.4942 | 2.165e-03 | fail | 8 | none | 0.0 |

## Paired Comparisons

Positive improvement means the treatment is faster than the control. Comparisons marked `not-qualified-quality` are diagnostic raw timings only.

| Comparison | Treatment improvement | 95% bootstrap CI | Faster blocks | Quality | Verdict |
|---|---:|---:|---:|---|---|
| baseline-gather-vs-scatter | -2.34% | [-8.85%, +5.10%] | 2/6 | pass | inconclusive |
| baseline-fusion-vs-gather | +0.49% | [-2.72%, +2.92%] | 5/6 | pass | no-material-effect |
| baseline-fusion-vs-scatter | -1.59% | [-7.04%, +3.21%] | 2/6 | pass | inconclusive |
| stress-gather-vs-scatter | -16.39% | [-21.75%, -11.31%] | 0/6 | fail | not-qualified-quality |
| stress-fusion-vs-gather | +1.14% | [-2.74%, +5.03%] | 3/6 | fail | not-qualified-quality |
| stress-fusion-vs-scatter | -14.89% | [-19.35%, -10.48%] | 0/6 | fail | not-qualified-quality |
| stress-adaptive-no-history-vs-fixed | -41.59% | [-44.89%, -38.30%] | 0/6 | fail | not-qualified-quality |
| stress-adaptive-frame-history-vs-fixed | -39.54% | [-43.45%, -35.66%] | 0/6 | fail | not-qualified-quality |
| stress-frame-history-vs-no-history | +1.44% | [-0.51%, +3.48%] | 4/6 | fail | not-qualified-quality |

## Line-Search Diagnostics

| Condition | Full searches | Candidate evaluations | History uses | First-batch accepts | Second-batch accepts | Fallbacks | Rejections |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-k4 | 960 | 0 | 0 | 0 | 0 | 0 | 0 |
| adaptive-k4-no-history | 960 | 3840 | 0 | 960 | 0 | 0 | 0 |
| adaptive-k4-frame-history | 960 | 3840 | 960 | 960 | 0 | 0 | 0 |

## Interpretation Boundary

This recheck tests one retained paper workload and one pre-registered higher-workload point. It can establish whether the current implementation has a material rendered end-to-end effect at these two points; it cannot establish a universal result across GPU architectures, mesh topologies, material parameters, or all Armijo workloads.
