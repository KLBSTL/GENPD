# XPBD Residency, CPU Context, and Signed-Gather Scaling (R1)

## Purpose

This rendered diagnostic extends the prior 256x256 XPBD residency check across
three mesh sizes. It measures four conditions in the existing moving-sphere
scene:

1. `cpu-ncg`: CPU NCG timing context.
2. `gpu-xpbd-jacobi`: the normal GPU-resident atomic XPBD path.
3. `gpu-xpbd-jacobi --force-cpu-state-roundtrip`: the same XPBD path, but the
   finalized position/velocity state is read to CPU and uploaded again before
   the next frame.
4. `gpu-xpbd-vertex-gather`: the current signed-incidence gather XPBD path.

The only intentional difference in the resident/roundtrip pair is the full
position/velocity state transfer. This is therefore a causal residency test.
CPU NCG is not algorithmically or quality-equivalent to XPBD, and is reported
only to show its observed timing context.

## Protocol

- Commit that produced the raw runs: `e312dd6`.
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU; driver 581.57.
- Scene: `scenes/moving_sphere_cloth.xml`.
- Sizes: 128x128, 256x256, and 386x386 vertices.
- Rendered viewport: 1600x900; `--uncapped`, `--sync-gpu`, and
  `--disable-vsync` enabled.
- Each case: 30 warm-up frames and 300 measured, actually rendered frames.
- XPBD variants: exactly 32 iterations/frame and 96 XPBD dispatches/frame
  (32 constraint + 32 apply + 32 collision).
- CPU NCG: 32-iteration maximum, with early stopping enabled. Its actual mean
  was 2.0, 2.0, and 2.1 iterations/frame at 128x128, 256x256, and 386x386;
  the 386x386 sequence included a rare 32-iteration frame.
- Quality readback was disabled while timing. All 12 cases finished 300
  measured frames with finite state, `frame_valid=1`, and
  `termination_reason=none`.

## Results

`mean/P50/P95` are rendered `frame_wall_ms`; `solver` is the application's
mean `optimization_ms`. State traffic is per-frame H2D/D2H MiB.

| mesh | condition | mean / P50 / P95 (ms) | solver (ms) | state MiB | notes |
| --- | --- | ---: | ---: | ---: | --- |
| 128x128 | CPU NCG | 9.658 / 9.407 / 12.570 | 4.943 | 0 / 0 | 32 max, 2.0 actual iter |
| 128x128 | XPBD resident | 4.189 / 4.048 / 5.543 | 1.736 | 0 / 0 | atomic |
| 128x128 | XPBD forced roundtrip | 5.542 / 5.293 / 7.261 | 1.898 | 0.375 / 0.375 | atomic, forced state copies |
| 128x128 | XPBD signed gather | 4.955 / 4.798 / 6.101 | 1.469 | 0 / 0 | vertex-owned apply |
| 256x256 | CPU NCG | 85.270 / 69.025 / 113.847 | 55.343 | 0 / 0 | 32 max, 2.0 actual iter |
| 256x256 | XPBD resident | 22.904 / 15.276 / 74.803 | 5.629 | 0 / 0 | atomic |
| 256x256 | XPBD forced roundtrip | 26.243 / 22.072 / 49.101 | 6.434 | 1.500 / 1.500 | atomic, forced state copies |
| 256x256 | XPBD signed gather | 21.178 / 20.787 / 23.688 | 6.725 | 0 / 0 | vertex-owned apply |
| 386x386 | CPU NCG | 164.408 / 155.934 / 178.366 | 111.377 | 0 / 0 | 32 max, 2.1 actual iter |
| 386x386 | XPBD resident | 34.565 / 34.131 / 39.718 | 10.057 | 0 / 0 | atomic |
| 386x386 | XPBD forced roundtrip | 42.485 / 41.850 / 47.329 | 13.262 | 3.410 / 3.410 | atomic, forced state copies |
| 386x386 | XPBD signed gather | 37.890 / 37.748 / 41.173 | 14.239 | 0 / 0 | vertex-owned apply |

## Controlled Residency Result

| mesh | resident mean (ms) | forced mean (ms) | resident speedup | forced extra state traffic |
| --- | ---: | ---: | ---: | ---: |
| 128x128 | 4.189 | 5.542 | 1.32x | 0.750 MiB/frame |
| 256x256 | 22.904 | 26.243 | 1.15x | 3.000 MiB/frame |
| 386x386 | 34.565 | 42.485 | 1.23x | 6.820 MiB/frame |

The resident path eliminated the intended per-frame full-state copies while
keeping the same atomic XPBD solver, 32-iteration budget, and 96 XPBD
dispatches. This supports a narrow claim: retaining the simulation state on
GPU avoids a measurable 15--32% rendered-frame cost for this XPBD pipeline.
The CPU still schedules simulation iterations and dispatches; this is
**simulation state GPU-resident**, not GPU-autonomous simulation.

## Signed-Gather Result

| mesh | gather / atomic mean frame | gather / atomic P50 frame | gather / atomic solver |
| --- | ---: | ---: | ---: |
| 128x128 | 1.18x | 1.19x | 0.85x |
| 256x256 | 0.92x | 1.36x | 1.19x |
| 386x386 | 1.10x | 1.11x | 1.42x |

Ratios larger than 1 mean signed gather is slower. The 256x256 mean-frame
ratio is distorted by long-tail presentation stalls in the atomic run (P95
74.803 ms, including frames 314, 323, and 325 at 176.289, 132.650, and
118.485 ms). Its P50 and solver time both favor atomic XPBD. Across all three
sizes, signed gather has no consistent rendered or solver-time improvement.
It should remain a negative implementation result, not a performance claim.

This is compatible with the implementation: the signed adjacency removes the
raw gather prototype's full edge fetch from the vertex pass, but retains the
same atomic constraint-scatter pass and adds a vertex gather pass. The earlier
256x256 short checkpoint diagnostic established atomic/gather state equality
for the same initial state. This R1 test establishes 300-frame validity, but
does not replace a long-horizon trajectory comparison at all resolutions.

## Boundary and Follow-up

This is one 300-frame sequence per case, so it is reproducible internal
evidence rather than a final multi-repetition paper result. It must not be
used to report an equal-quality CPU-NCG versus XPBD speedup: the methods use
different updates, and CPU NCG early-stops well below its 32-iteration cap.
Before a paper claim, run repeated reference-calibrated comparisons and report
quality metrics along with mean, variance, P50, and P95.

Raw artifacts are intentionally ignored by Git and remain under:
`results/diagnostic-20260729-xpbd-residency-scaling-r1/`. The raw result folder
contains `manifest.json`, `planned_runs.csv`, all per-case profiles and
metadata, `xpbd_residency_scaling_summary.csv`,
`xpbd_residency_scaling_comparisons.csv`, and an automatically generated copy
of this report.
