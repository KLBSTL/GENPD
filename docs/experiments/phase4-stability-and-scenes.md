# Phase 4 Stability and Scene Matrix

## Scope

This phase supplies a reproducible stability sweep, scene input, and per-run
metadata. It does not make the solver GPU autonomous: the CPU still controls
frame iteration, line search, and compute dispatch. The simulation state can
remain GPU-resident for the persistent GPU variants.

The external-collision scope is limited to primitives represented by the current
scene path. `moving_sphere_cloth.xml` exercises a plane plus a translating
sphere. The current implementation does not include cloth self-collision;
paper text must describe the method as supporting the present external and
moving-obstacle constraints only, not as a general garment simulator.

## Runtime Matrix

The following flags override the loaded config for one run and are recorded in
`run_metadata.json` under `experiment_overrides`:

- `--timestep FLOAT`
- `--stretch-stiffness FLOAT`
- `--bending-stiffness FLOAT`
- `--cloth-dimension N` for a square cloth grid
- `--scene PATH` relative to project root or absolute
- `--quality-metrics` to record finite state, strain, energy, and penetration
  without requiring a reference checkpoint

Suggested paper matrix:

| Axis | Values |
| --- | --- |
| timestep | 0.01665, 0.0333, 0.05 |
| stretch stiffness | 40, 80, 160 |
| bending stiffness | 10, 20, 40 |
| cloth dimension | 64, 128, 256 |
| scene | `scenes/test_scene.xml`, `scenes/moving_sphere_cloth.xml` |

Run one material/mesh/scene slice with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_stability_sweep.ps1 `
  -RunLabel paper-stability-moving-128 `
  -ClothDimension 128 `
  -BendingStiffness 20 `
  -Scene scenes/moving_sphere_cloth.xml `
  -Frames 300 -Warmup 30

powershell -ExecutionPolicy Bypass -File scripts/plot_stability_heatmap.ps1 `
  -InputCsv results/paper-stability-moving-128/stability_sweep.csv
```

Repeat for every requested cloth resolution, bending stiffness, scene, and
solver variant. Each sweep writes per-case profiles in `runs/`, then writes
`stability_sweep.csv`; the SVG uses green for a finite, non-exploded case and
red for a failed or incomplete case.

## Stability Criterion

A case is marked stable only when every measured profile record is finite and
not exploded, and every measured quality record has `finite=1`. The CSV retains
the number of exploded frames, non-finite records, failure rate, final gradient
norm, maximum position magnitude, and maximum external penetration depth.

This is an operational robustness criterion. It does not replace the Phase 3
reference-error comparison: report the stability heatmap together with
position/velocity error and energy/strain quality at a comparable visual or
numerical target.
