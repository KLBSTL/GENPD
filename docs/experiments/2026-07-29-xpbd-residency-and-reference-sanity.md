# XPBD Residency and CPU Reference Sanity Protocol

## Goal

This phase adds two narrow causal checks to the GenPD evidence chain:

1. Compare GPU-resident XPBD state against the same XPBD solver with a forced
   per-frame CPU position/velocity roundtrip.
2. Test whether the CPU NCG reference using 100 iterations per frame is close
   to 200- and 400-iteration CPU reference trajectories.

All timing runs are rendered. CPU continues to dispatch iterations and collect
the small control/statistics readbacks; the intended claim is **simulation
state GPU-resident**, not GPU-autonomous simulation.

## XPBD Residency Protocol

- Variant: `gpu-xpbd-jacobi` with the existing Jacobi constraint, vertex apply,
  and external sphere collision passes.
- Case: 256 x 256 moving-sphere cloth, 32 XPBD iterations per frame. The
  iteration count is the qualified historical R2 operating point for this
  variant and resolution.
- Timing: 1600 x 900 rendered end-to-end timing, 30 warm-up plus 300 measured
  frames, three repetitions per condition.
- Conditions: `resident` and `forced-cpu-state-roundtrip`.
- Controlled difference: the forced condition reads finalized position and
  velocity state to CPU, invalidates resident state, and uploads it on the next
  frame. XPBD constraint/apply/collision dispatch counts must remain equal.
- Trajectory audit: a separate rendered 20 warm-up plus 120 frame run exports
  checkpoints every 10 frames for both conditions. Position and velocity P95
  relative L2 errors must not exceed `1e-3`, the same numerical-consistency
  scale used by the equal-quality reference protocol. This gate allows small
  accumulated float/host-scalar roundoff from the deliberately forced
  representation crossing; it does not claim bitwise-identical trajectories.

The archived `paper-20260729-xpbd-residency-r1` raw runs used an overly strict
`1e-6` gate. Their P95 position/velocity errors were `6.19e-6` and `8.69e-4`:
they reject exact-equivalence but remain below the quality-consistency scale.
R2 is the separately rerun, pre-registered protocol used for evidence.

The timing run never enables quality/checkpoint readback. State traffic,
trajectory checking, and screenshots are separate from the main timing path.

## CPU Reference Sanity Protocol

- CPU NCG references use 100, 200, and 400 iterations per frame.
- Two 256 x 256 rendered cases are checked: hanging cloth and moving sphere.
- Each run uses 5 warm-up plus 30 measured frames and exports one checkpoint
  per frame.
- The 400-iteration trajectory is the comparison target. The primary decision
  is P95 position relative L2 for 100 versus 400 at matched measured frames;
  the pre-registered threshold is `1e-3`. Velocity error is reported as a
  diagnostic rather than a separate quality gate.
- Any invalid frame, missing checkpoint, inconsistent scene/configuration, or
  unrendered frame rejects the study.

These checks validate the reference adequacy and state-residency mechanism.
They do not replace the full equal-quality performance matrix or claim that
XPBD and NCG are physically identical solvers.

## Formal Results

Formal R2 output roots are `results/paper-20260729-xpbd-residency-r2` and
`results/paper-20260729-cpu-reference-sanity-r1`. Both runs use commit
`87f6b6c`, an NVIDIA GeForce RTX 3070 Laptop GPU, driver `581.57`, 1600 x 900
rendering, GPU synchronization, and disabled VSync.

### XPBD Residency

For 256 x 256 moving-sphere cloth at 32 XPBD iterations per frame, three
30-warm-up plus 300-measured-frame repetitions gave:

| Condition | Rendered frame time (ms) | State H2D / D2H (MiB/frame) | State upload / readback calls | XPBD dispatches / frame |
|---|---:|---:|---:|---:|
| GPU-resident | 20.497 +/- 0.913 | 0.000 / 0.000 | 0 / 0 | 96 |
| Forced CPU roundtrip | 21.938 +/- 0.094 | 1.500 / 1.500 | 2 / 2 | 96 |

The forced state roundtrip is `1.07x` slower in rendered end-to-end time;
equivalently, retaining the state on GPU is about `6.6%` faster for this
specific case. The comparison has identical XPBD shader dispatch counts, so
the difference is attributable to the deliberately introduced state traffic
and its synchronization consequences, not a different constraint schedule.

The independent 20-warm-up plus 120-frame checkpoint audit had P95 relative
L2 errors of `8.46e-6` for position and `7.83e-4` for velocity. Both meet the
R2 `1e-3` consistency gate. This is quality-consistent rather than bitwise
identical: the forced path crosses device float buffers and host scalar
vectors every frame, so small accumulated roundoff is expected.

### CPU Reference Sanity

For both 256 x 256 hanging cloth and moving-sphere cloth, each CPU NCG run
used 5 warm-up plus 30 measured rendered frames and wrote one checkpoint per
frame. The 100-vs-400 and 200-vs-400 P95 position and velocity relative L2
errors were all exactly `0.0` at stored float precision. Therefore the
100-iteration CPU NCG reference is sufficient for these two measured cases;
the 400-iteration trajectory provides no changed checkpoint for the tested
window.

This conclusion is deliberately local to the two 256 x 256 scenes, current
material settings, 35-frame window, and current commit. It must be rerun if
the solver, stiffness, timestep, scene, or reference precision changes.
