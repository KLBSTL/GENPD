# R2 Protocol-4 Preflight

## Scope And Provenance

This is a rendered protocol preflight, not paper performance evidence.

- Git commit: `59b8ff2`.
- Run root: `results/r2-preflight-v4-20260724-h128-59b8ff2`.
- Scope: hanging cloth at `128^2` only, marked
  `protocol-preflight-not-paper-evidence` in the manifest.
- Rendering: `1600x900`, `--sync-gpu`, `--disable-vsync`, and no
  `--no-render` runs.
- Hardware recorded in every run: NVIDIA GeForce RTX 3070 Laptop GPU,
  driver `581.57`.

## Gate Audit

The original absolute XPBD maximum-strain condition was rejected during a
rendered protocol-2 preflight because the CPU NCG reference itself reached
P95 maximum stretch strain `0.98095805`, above the old `0.10` limit.

Protocol 3 changed the maximum-strain condition to reference-relative, but
still compared all XPBD frames to reference checkpoint frames. Its preflight
was retained as historical audit evidence and not used for selection.

Protocol 4 samples both current and reference XPBD strain only at the same
reference checkpoint frames. The current gate is:

- P95 checkpoint mean strain `<= 0.02`.
- P95 checkpoint maximum strain `<= 1.10 *` reference P95 maximum strain.
- Maximum penetration depth `<= 0.02`.

The amendment and the rationale are in
`docs/experiments/2026-07-24-xpbd-quality-gate-amendment.md`.

## Calibration Results

The preflight completed all `7 x 13 = 91` calibration candidates with
`20` warm-up and `120` measured rendered frames. All candidates had valid
extended-frame records and no timeout or non-finite termination.

| Variant | Selected iterations/frame | Quality result |
| --- | ---: | --- |
| CPU NCG | 2 | P95 position relative L2 `8.69e-5` |
| GPU edge-scatter NCG | 2 | P95 position relative L2 `8.77e-5` |
| GPU gather no fusion | 2 | P95 position relative L2 `8.74e-5` |
| GPU gather fusion | 2 | P95 position relative L2 `8.74e-5` |
| GPU gather fusion plus batched LS | 2 | P95 position relative L2 `8.74e-5` |
| GPU gather fusion plus persistent buffers | 2 | P95 position relative L2 `8.67e-5` |
| GPU Jacobi XPBD | 24 | checkpoint max-strain ratio `1.094 < 1.10`, zero penetration |

## Rendered Timing Preflight

Each selected case ran `30` warm-up plus `300` measured rendered frames,
repeated three times. `preflight_performance_summary.csv` records the
aggregate and remains preflight-only.

| Variant | Iterations/frame | Frame wall time mean (ms) | Std (ms) |
| --- | ---: | ---: | ---: |
| CPU NCG | 2 | 7.706 | 0.137 |
| GPU edge-scatter NCG | 2 | 1.673 | 0.027 |
| GPU gather no fusion | 2 | 1.717 | 0.049 |
| GPU gather fusion | 2 | 1.631 | 0.070 |
| GPU gather fusion plus batched LS | 2 | 1.605 | 0.043 |
| GPU gather fusion plus persistent buffers | 2 | 0.692 | 0.008 |
| GPU Jacobi XPBD | 24 | 4.325 | 0.033 |

These values must not be quoted as the paper's scalability or aggregate
performance result because they cover one scene and one resolution only.

## Visual Check

Both rendered captures contain a visible, nonempty cloth with two top fixed
points:

- `captures-preflight/persistent/frame-120.png`
- `captures-preflight/xpbd/frame-120.png`

## Next Formal Step

Launch a fresh complete protocol-4 root, `results/paper-20260724-r2`, at a
fixed commit. It must cover both scenes, all three square resolutions, all
seven variants, three timing repetitions, the stability matrix, and the
capture matrix. The formal summary and figure scripts reject this partial
preflight by construction.
