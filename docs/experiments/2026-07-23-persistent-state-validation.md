# Persistent GPU-State Validation

The `gpu-gather-fusion-batched-ls-persistent` variant now keeps simulation
positions and velocities in GPU buffers for every cloth resolution, including
`386^2`, and supports the existing external plane/sphere collision shader.
CPU code still schedules each NCG and line-search dispatch.

## Validity Protocol

The persistent path does not read the full position or velocity buffers during a
performance frame. After finalization it reads only one pair of maximum
position/displacement values per compute workgroup from `csStateStatsID`. This
small status readback is included in the frame timing and increments
`host_readbacks`; it prevents a false `frame_valid=1` when the GPU state has
already become non-finite.

The NCG gradient/statistics buffer is not used as a finite-state gate in this
mode because it is intentionally not read back on the persistent route. The
position/velocity statistics are the authoritative validity gate.

## Smoke Evidence Only

`results/smoke-persistent-sphere-h386-final` used actual rendering with
`386^2`, 16 iterations/frame, and `moving_sphere_cloth.xml`:

- `persistent_buffers_active=1` for all five frames;
- `persistent_collision_dispatches=1` for all five frames;
- all frames had `frame_valid=1`, `exploded=0`, and finite max position values.

`results/smoke-persistent-hanging-h386-final` exercised the corresponding
empty-scene hanging-cloth route at the same resolution and iteration budget.
All five rendered frames were valid with `persistent_buffers_active=1`; the
saved capture shows a finite suspended cloth and its two pinned vertices.

This is an implementation smoke test, not a quality-calibrated or ranked paper
result. The formal R2 manifest must retain the reported readback/synchronization
cost rather than treating persistent state as zero-synchronization execution.
