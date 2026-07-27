# PeriDyno PPM External Operating Point

## Scope

This is an external same-hardware operating-point baseline for *Projective
Peridynamic Modeling of Hyperelastic Membranes With Contact* (PPM, 2024), using
the authors' PeriDyno implementation. It replaces the planned Wang 2021 source
package because the official Wang archive contains only three reference files
and explicitly omits the build system, entry point, scenes, and dependencies.

The result is not an equal-model or equal-quality ranking. GenPD uses a dynamic
mass-spring cloth with attachment constraints and NCG. PPM uses a triangular
hyperelastic peridynamic membrane with the authors' contact pipeline. Report it
as operating-point context only.

## Provenance

- Paper: Lu et al., *Projective Peridynamic Modeling of Hyperelastic Membranes
  With Contact*, IEEE TVCG, 2024.
- Official source: `https://github.com/peridyno/peridyno`.
- Pinned local revision: `12a5f00dd95d5f3594493b7d8bff27d20dee13cf`.
- Third-party source remains in `D:\GenPD_test_p\external\peridyno-ppm` and
  is not copied into this repository.
- The tracked adapter is in `external_baselines/peridyno_ppm/`. The runner
  overlays it into the local PeriDyno example tree before configuring CMake.
- `external_baselines/peridyno_ppm/peridyno-ppm-genpd-operating-point.patch`
  is applied idempotently by the runner. It is the complete local change to
  the pinned PeriDyno revision: safe no-self-contact control flow plus a
  public one-frame GLFW draw/swap and BMP capture interface.

## Scene Mapping

The adapter preserves GenPD's regular cloth topology and its two top-corner
attachments. It applies a uniform `0.1` position scale to make the PeriDyno
material parameters numerically well-conditioned.

| GenPD scene | PPM adapter |
| --- | --- |
| `scenes/test_scene.xml` | Hanging triangular membrane with two fixed top corners and no active obstacle. |
| `scenes/moving_sphere_cloth.xml` | Same cloth plus a sphere initialized at `(0, 10, -2.5)`, radius `2`, velocity `(0, -1.2, 0)`, all scaled by `0.1`. |

The moving-sphere path calls `BasicShapeToVolume::reset()` once per PPM
substep, after refreshing the `SphereModel` mesh, so the rendered sphere and
PPM SDF share the same position rather than leaving a static obstacle in place.
The renderer uses PeriDyno's official `GLSurfaceVisualNode` cross-node pattern
for the sphere mesh; its output is verified by a non-black hanging-cloth and
moving-sphere BMP smoke. This moving-SDF refresh cost is intentionally included
in the reported PPM timing.

The adapter disables PPM self-contact. This matches GenPD's current stated
scope (no self-collision) while retaining the official PPM external SDF contact
path. The tracked patch makes `setSelfContact(false)` skip only the upstream
self-contact broad phase and Jacobi loop, whose position buffer is not
connected in that mode. It does not disable `VolumeBoundary` external SDF
contact.

## Protocol

The initial smoke protocol is `64 x 64` vertices, 8 warm-up frames, 30 measured
frames, `dt=0.001`, 33 substeps per nominal GenPD frame, and 10 PPM solver
iterations. The default run opens a `1600 x 900` GLFW context, disables VSync,
and calls the patched one-frame graphics update, draw, and buffer-swap path
once per warm-up and measured frame. It writes `frame_profile.csv` and
`run_metadata.json`. `frame_host_ms` includes simulation and the rendered
presentation; `simulation_gpu_ms` is retained only as a solver-side CUDA-event
diagnostic and excludes the OpenGL draw.

Run it from the GenPD project root:

```powershell
scripts\run_peridyno_ppm_operating_point.ps1 -Scene hanging
scripts\run_peridyno_ppm_operating_point.ps1 -Scene moving-sphere
scripts\run_peridyno_ppm_operating_point.ps1 -Scene moving-sphere -CaptureOutput results\external-ppm-proof.bmp
```

`--headless` remains available only for diagnosis and short regression. It must
not supply manuscript timing. The screen capture is deliberately written after
the timing loop, so readback does not affect `frame_host_ms`.

PeriDyno's GLFW destructor can block after a bounded non-interactive loop, even
after its renderer is explicitly terminated. Therefore the adapter deliberately
retains the GLFW window object until process exit after all files are flushed.
Windows reclaims that one-shot process resource; startup and process teardown
are outside the reported per-frame interval. This is an adapter lifecycle
workaround, not a solver timing optimization.

These records are still not an equal-quality PPM-versus-GenPD speed ranking.
They become paper-eligible external operating-point evidence only after the
rendered output is visually checked and a documented cross-model quality
protocol is satisfied. Until then, they establish that the external 2024 solver
runs on the same GPU and supports the mapped current-project contact scenario.

## Required Reporting Boundary

Any manuscript table must include mesh size, PPM substeps, solver iterations,
the moving-SDF refresh policy, GPU model, driver, PeriDyno revision, and this
scope limitation. It must not claim an equal-quality speedup over PPM unless a
separate cross-model quality protocol is implemented and passed.
