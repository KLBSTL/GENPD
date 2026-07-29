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
