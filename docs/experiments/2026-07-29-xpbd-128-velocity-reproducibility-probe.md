# XPBD 128x128 Velocity Reproducibility Probe

## Answer

The `1.21e-2` (R2) and `1.30e-2` (R3) 128x128 moving-sphere velocity
relative-L2 values are not evidence that the forced CPU roundtrip uniquely
causes a velocity error. They are reproducible in the weaker but more useful
sense that an independent run at the same 128x128 configuration reaches the
same order of difference. The new probe shows that this is **repeat-scale
trajectory variation**: two runs under the same GPU-resident condition differ
by as much as, or more than, a resident/forced pair.

This is not a claim that the discrepancy occurs at every resolution or under
every XPBD configuration. It is established only for the current 128x128,
32-iteration atomic-Jacobi XPBD implementation, on the recorded RTX 3070
Laptop GPU and driver 581.57, over the 140-frame rendered audit.

## Probe Protocol

- Raw result root:
  `results/diagnostic-20260729-xpbd-velocity-repro-r1/`.
- Simulation commit: `241fe67be66467207540bcac6c39877870bfeaa9`.
- Solver: `gpu-xpbd-jacobi`, 32 iterations/frame, separate constraint,
  apply, and collision passes.
- Rendering: 1600x900, `--uncapped`, `--sync-gpu`, VSync disabled. No run
  uses `--no-render`.
- Trajectory: 20 warm-up plus 120 measured frames; position and velocity
  checkpoints every 10 frames, yielding 12 matched measured checkpoints.
- Two scenes: `hanging` (`scenes/test_scene.xml`, no collision primitive) and
  `moving-sphere` (`scenes/moving_sphere_cloth.xml`).
- Four independent process launches per scene: `resident-a`, `resident-b`,
  `forced-a`, and `forced-b`. The forced condition alone reads finalized
  position and velocity to CPU, invalidates GPU state, and uploads the state
  before the next frame.

The analysis compares two same-condition pairs and all four resident/forced
pairs. It is a trajectory diagnostic only; checkpoint and quality readback are
excluded from any performance claim.

## Results

| Scene | Largest same-condition velocity P95 | Median cross-condition velocity P95 | Cross-condition range | Largest position P95 across all pairs |
| --- | ---: | ---: | ---: | ---: |
| hanging | 9.689e-03 | 9.733e-03 | 5.888e-03 to 1.349e-02 | 9.181e-05 |
| moving-sphere | 1.273e-02 | 8.257e-03 | 7.363e-03 to 1.003e-02 | 8.021e-05 |

The moving-sphere same-condition pair `resident-a:resident-b` reaches
`1.273e-02` velocity relative L2, exceeding every resident/forced pair in the
same scene. Its final-checkpoint velocity-difference RMS is `1.083e-03` while
the resident velocity RMS is `8.506e-02`; the result is therefore not solely
an artifact of dividing by an almost-zero velocity norm.

All eight runs are finite, rendered, and use the requested iteration count.
Physical metrics match at the displayed precision: the moving-sphere P95 mean
stretch is about `0.00512`, P95 maximum stretch is about `1.0198`, and maximum
penetration is `0.00100024` for resident versus `0.00100036` for forced. The
hanging control has zero penetration. Thus the observed trajectory variation
does not coincide with a failure, explosion, or material-quality difference.

## Interpretation

1. **Not a roundtrip-specific effect.** In both scenes the resident/forced
   discrepancies are no larger than ordinary independent-run discrepancies.
   The previous single resident/forced comparison cannot identify the CPU
   state crossing as the cause of the 128x128 velocity number.
2. **Not a moving-contact-specific effect.** The collision-free hanging scene
   has the same error scale as moving sphere. Contact projection may still
   contribute during a particular run, but this small probe does not identify
   it as the primary amplifier.
3. **Most plausible implementation mechanism: atomic accumulation order.**
   The XPBD constraint shader scatters each edge correction with four floating
   `atomicAdd` operations (`shaders/xpbd_constraints.comp`, lines 45--51 and
   88--89). The order in which different constraints accumulate at one vertex
   is not a deterministic arithmetic reduction order. Floating-point addition
   is non-associative, so small process-to-process differences can seed a
   trajectory difference that grows over frames. The state finalization then
   forms velocity from a position difference divided by the timestep
   (`shaders/cs2State.comp`, lines 71--83), making its relative velocity metric
   more sensitive than position.
4. **The forced path is still a valid timing counterfactual.** It reads the
   two state buffers with `glGetBufferSubData` and uploads them on the next
   frame with `glBufferSubData` (`source/simulation.cpp`, lines 2146--2159 and
   2329--2340); default `ScalarType` is float. The probe does not find a
   unique quality regression associated with that transfer path. Its R3
   timing/state-traffic result remains a causal performance measurement, but
   it must not be presented as strict velocity-equivalent trajectory evidence
   at 128x128.

## Paper Consequence

Keep 128x128 velocity relative L2 as a diagnostic, not an eligibility gate for
the residency timing claim. Report the R3 position, strain, penetration, and
finite-state checks, and explicitly say that CPU schedules all frame dispatches
while simulation state remains GPU-resident between frames. Do not claim
bitwise determinism or strict velocity equivalence for 128x128 atomic XPBD.

If strict reproducibility becomes necessary, the next implementation study is
a deterministic per-vertex reduction or graph-colored constraint pass. That is
a solver-design change with possible performance cost, not a small correction
to the CPU roundtrip code. A lighter validation extension is to repeat this
probe on a second GPU/driver and add more independent launches before making a
cross-device reproducibility claim.

## Artifacts

- `scripts/run_xpbd_velocity_reproducibility_probe.ps1`: rendered eight-run
  protocol.
- `scripts/analyze_xpbd_velocity_reproducibility_probe.py`: artifact validator
  and pairwise checkpoint aggregation.
- `results/diagnostic-20260729-xpbd-velocity-repro-r1/manifest.json`: exact
  configuration and hardware metadata.
- `results/diagnostic-20260729-xpbd-velocity-repro-r1/reproducibility_summary.csv`:
  compact numeric evidence.
- `results/diagnostic-20260729-xpbd-velocity-repro-r1/xpbd_velocity_reproducibility_report.md`:
  generated raw-result report.
