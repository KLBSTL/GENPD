# Fixed K=8 Batched Armijo Revalidation R2

## Protocol

This is a current-commit (`50c1693`) revalidation of the useful Armijo result,
not a rerun of the rejected adaptive-history variants.

- Methods: persistent serial Armijo and persistent fixed batched Armijo,
  `K=8`, `beta=0.5`, non-descent NCG restart.
- Cases: hanging and moving-sphere cloth at 256x256 and 386x386.
- Timing: rendered 1600x900, 30 warm-up + 300 measured frames, three
  repetitions per method, GPU synchronization enabled and VSync disabled.
- Quality: separate 120-frame rendered traces use each case's archived CPU
  reference checkpoints. Timing has no quality readback.
- All eight traces are finite, have zero invalid frames and zero Armijo
  failures, and pass the registered P95 position relative-L2 gate of `1e-3`.

## Results

| Case | Serial mean +/- std ms | Fixed K=8 mean +/- std ms | Fixed / serial frame speedup | Serial / fixed line-search time | P95 position relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hanging 256x256 | 1.260 +/- 0.061 | 0.933 +/- 0.060 | 1.35x | 1.40x | 7.233e-04 |
| hanging 386x386 | 2.258 +/- 0.059 | 1.764 +/- 0.080 | 1.28x | 1.54x | 4.894e-04 |
| moving sphere 256x256 | 1.263 +/- 0.044 | 1.003 +/- 0.036 | 1.26x | 1.47x | 7.233e-04 |
| moving sphere 386x386 | 2.368 +/- 0.022 | 1.816 +/- 0.117 | 1.30x | 1.51x | 4.894e-04 |

Fixed K=8 improves rendered frame time in all four cases while retaining the
same measured P95 position error and no Armijo failures. On this current
implementation, it is therefore supported as the NCG line-search variant to
carry forward. The conclusion is intentionally narrower than “batched Armijo
is universally faster”: it covers this fixed schedule, these two scenes, the
selected equal-quality iteration budgets, and the tested GPU.

Raw artifacts are stored in
`results/paper-20260729-fixed-batched-armijo-r2/`, with per-case profiles,
metadata, decision traces, quality metrics, and combined summary CSVs.
