# CPU Reference Convergence Audit R2

## Protocol

This audit was rerun at current commit `50c1693` after the XPBD residency
matrix. It uses rendered CPU NCG checkpoint runs on 256x256 hanging and
moving-sphere cloth:

- iteration caps: 100, 200, and 400;
- 5 warm-up + 30 measured frames per run;
- 1600x900 rendering, GPU synchronization, and disabled VSync;
- one stored reference checkpoint per frame.

The 400-cap trajectory is the comparison target. All stored 100-vs-400 and
200-vs-400 position and velocity relative-L2 errors are exactly zero at the
exported float precision.

## Convergence Behavior

| Scene | Cap | Actual iterations mean [min, max] | Frames converged before cap | P95 gradient norm | Final gradient norm | Final objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hanging | 100 | 2.63 [2, 21] | 30 / 30 | 1.349e-04 | 1.401e-04 | 3.414e-05 |
| hanging | 200 | 2.63 [2, 21] | 30 / 30 | 1.349e-04 | 1.401e-04 | 3.414e-05 |
| hanging | 400 | 2.63 [2, 21] | 30 / 30 | 1.349e-04 | 1.401e-04 | 3.414e-05 |
| moving-sphere | 100 | 2.63 [2, 21] | 30 / 30 | 1.349e-04 | 1.401e-04 | 3.414e-05 |
| moving-sphere | 200 | 2.63 [2, 21] | 30 / 30 | 1.349e-04 | 1.401e-04 | 3.414e-05 |
| moving-sphere | 400 | 2.63 [2, 21] | 30 / 30 | 1.349e-04 | 1.401e-04 | 3.414e-05 |

Every profile row has `termination_reason=none`, which denotes no invalid or
exploded termination. Successful early stopping is derived from
`converged=1` together with an observed iteration count below the configured
cap; all 180 measured rows have this outcome.

Thus the identical stored states are explained by convergence before the
smallest 100-iteration cap, rather than by a broken higher-cap reference
path. This conclusion is local to these two scenes, 256x256 resolution,
material/timestep settings, and 35-frame window. It must be rerun after any
solver, material, timestep, or checkpoint-format change.

Raw and derived artifacts are under
`results/paper-20260729-cpu-reference-sanity-r2/`, including
`cpu_reference_sanity_summary.csv` and
`cpu_reference_convergence_summary.csv`.
