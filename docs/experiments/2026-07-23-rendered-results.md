# Rendered Paper Results: 2026-07-23

## Provenance

- Code commit: `c3e36ba`
- Run root: `results/paper-20260723-rendered/`
- Platform: Intel Core i7-11800H, NVIDIA GeForce RTX 3070 Laptop GPU, driver 581.57, OpenGL 4.6, Windows 11
- Primary metric: rendered end-to-end `frame_wall_ms`, including simulation, drawing, swap, and an explicit GPU completion point
- Resolution: 1600x900; vertical synchronization disabled
- Performance protocol: 30 warm-up frames, 300 measured frames, three repetitions

The result files contain a per-run `run_metadata.json`, raw frame CSV files, the manifest, calibration selections, summary CSV files, stability records, and current-commit captures. Screenshots were captured separately and are excluded from timing.

## Equal-Quality Gate

Each CPU and GPU variant was calibrated against CPU NCG at 100 iterations per frame. The reference uses 20 warm-up and 120 measured frames, with checkpoints every ten frames. Candidate iteration budgets are `1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32`. A candidate qualifies only when its P95 relative position L2 error is at most `1e-3`, it has reference samples, and it has no non-finite state or explosion.

There are 32 qualified scene-resolution-variant selections. A post-hoc audit found that the following four `386^2` hanging-cloth gather variants were not ordinary quality-gate misses: every tested budget terminated with an `E0: frame-0 explosion`. They have no equal-quality performance ranking:

- `gpu-gather-no-fusion`
- `gpu-gather-fusion`
- `gpu-gather-fusion-batched-ls`
- `gpu-gather-fusion-batched-ls-persistent`

All qualified selected configurations have calibration failure rate zero. The legacy plots and any `NQ` labels are historical only and must not be used as positive performance evidence. The replacement validity matrix labels these cases `E0`, rather than concealing an invalid state behind a generic non-qualification marker.

## Stability Result

The final persistent configuration was tested on the `256^2` moving-sphere scene for 30 warm-up plus 300 measured rendered frames, repeated three times per cell. Bending stiffness was 20. All repetitions were stable for timestep `1/60` and `0.0333` at stretch stiffness `40`, `80`, and `160`, and for timestep `0.05` at stretch stiffness `40`. Timestep `0.05` with stretch stiffness `80` or `160` was unstable in all three repetitions.

## Scope

These results compare CPU NCG and internal GPU variants only. They do not establish an external GPU PD/contact comparison, self-collision support, or a broad line-search parameter sensitivity claim. The simulation state is GPU-resident during ordinary playback; the CPU continues to control iterations, line search, and OpenGL dispatch.
