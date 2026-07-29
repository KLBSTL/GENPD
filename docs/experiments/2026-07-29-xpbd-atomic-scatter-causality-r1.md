# XPBD Atomic-Scatter Causality R1

## Question

The previous 128x128 XPBD probe found a velocity relative-L2 difference of
roughly `1e-2` between independent runs, but its resident/forced-roundtrip
comparison could not identify one cause. This experiment tests whether the
float atomic constraint scatter is the dominant source of that variation.

## Controlled A/B/B/A Protocol

- Commit: `f2e62687704160d4903990a964b34c3a11fc3e7b`.
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU; driver 581.57.
- Scene: moving-sphere cloth, 128x128 vertices, 32 XPBD iterations/frame.
- Rendering: 1600x900, actual rendering enabled, `--uncapped`, `--sync-gpu`,
  and disabled VSync. No run uses `--no-render` or forced CPU state roundtrip.
- Trajectory: 20 warm-up plus 120 measured frames; position and velocity are
  checkpointed every 10 frames, producing 12 matched measured checkpoints.
- Process order: `atomic-a`, `gather-a`, `gather-b`, `atomic-b`.

The atomic condition is `gpu-xpbd-jacobi`. Its constraint shader scatters
corrections with floating `atomicAdd` operations. The control is the existing
`gpu-xpbd-vertex-gather` implementation: each constraint writes one
edge-owned correction record, then each vertex consumes the static CSR
incidence list in order. It removes atomic accumulation while retaining the
same Jacobi iteration count, pin handling, collision configuration, and state
residency policy.

The result was pre-registered as strong support only when all conditions held:

1. atomic repeat velocity P95 is at least `1e-3` and its checkpoint files are
   not all hash-identical;
2. gather repeat velocity P95 is at most `1e-7` and every matching checkpoint
   file has the same SHA-256 digest;
3. mean/max stretch and maximum penetration are within the registered quality
   gates between the two implementations.

## Result

| Repeat pair | Checkpoint SHA-256 all equal | Position P95 relative L2 | Velocity P95 relative L2 | Velocity-difference P95 RMS |
| --- | --- | ---: | ---: | ---: |
| atomic-a : atomic-b | no | 6.101e-05 | 9.650e-03 | 8.208e-04 |
| gather-a : gather-b | yes | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Atomic checkpoints first differ at frame 40, then grow monotonically in this
audit to the reported final/P95 difference. The two fixed-order gather runs
are byte-identical at every one of the 12 checkpoint files, including both
position and velocity payloads.

| Quality metric | Atomic | Gather | Difference | Registered gate |
| --- | ---: | ---: | ---: | ---: |
| P95 mean stretch | 0.00511970 | 0.00512012 | 8.301e-05 relative | <= 2.000e-02 |
| P95 max stretch | 1.019825 | 1.019830 | 4.903e-06 relative | <= 2.000e-02 |
| Maximum penetration | 0.00100024 | 0.00100036 | 1.200e-07 absolute | <= 1.000e-05 |

All quality gates pass; all frames are finite and non-exploded. The result is
therefore **strong support for atomic-scatter causality** under this protocol.

## Interpretation

This is a qualitative causal result for the current implementation, not merely
a correlation:

- CPU roundtrip is absent in all four runs.
- The moving scene, render path, initial conditions, solver budget, collision,
  pins, GPU, and driver are matched.
- Replacing the correction-accumulation path removes all observed independent-
  run trajectory variation while preserving the reported physical metrics.

The practical mechanism is the non-associativity of floating addition. In the
atomic implementation, multiple constraints add to each vertex delta with no
fixed arithmetic reduction order (`shaders/xpbd_constraints.comp`, lines
45--51). The vertex-gather implementation materializes one correction per edge
and sums it serially inside one vertex invocation over the static CSR order
(`shaders/xpbd_constraints_gather.comp` and `shaders/xpbd_apply_gather.comp`).
Its repeat-exact result rules out CPU roundtrip and moving-contact branching as
necessary explanations for the observed 128x128 divergence.

This does not prove that every GPU, driver, mesh, or use of `atomicAdd` will
produce the same magnitude. It also does not make vertex gather a performance
winner: prior stage and rendered pilots show that its edge-correction buffer
and irregular CSR reads cost more than atomic Jacobi. A separate single-kernel
delta-buffer hash test would be a lower-level confirmation, but is no longer
necessary to establish the long-horizon implementation-level cause.

## Paper Consequence

Describe atomic XPBD as physically stable but not repeat-exact at 128x128 on
the tested platform. Keep the GPU-resident state-traffic claim separate from
this result: the resident/roundtrip counterfactual is still valid for timing,
but strict 128x128 velocity equivalence cannot be claimed for atomic scatter.
Use fixed-order gather only as a deterministic diagnostic control unless a
future optimization closes its measured performance gap.

## Artifacts

- `scripts/run_xpbd_atomic_causality_probe.ps1`: rendered A/B/B/A protocol.
- `scripts/analyze_xpbd_atomic_causality_probe.py`: artifact validation,
  checkpoint hashing, physical comparability, and verdict generation.
- `results/diagnostic-20260729-xpbd-atomic-causality-r1/`: local raw root
  (96 files, 22.3 MiB) with profiles, metadata, logs, and checkpoints.
- `docs/experiments/artifacts/2026-07-29-xpbd-atomic-causality-r1/`: compact
  Git-tracked manifest and result summaries.
