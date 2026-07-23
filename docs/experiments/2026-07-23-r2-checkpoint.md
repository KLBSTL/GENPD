# R2 Checkpoint

This checkpoint records the state before launching any R2 calibration or
performance experiment. Historical `paper-20260723-rendered` data remains
unchanged and is not positive performance evidence.

## Completed Infrastructure

- `d6c4a9d` makes the persistent gather path active at `386^2` with external
  collision, checks GPU-state finite values through a bounded workgroup
  readback, and records the synchronization cost.
- `61024b0` upgrades the formal protocol to rendered R2 measurements, adds
  the internal GPU Jacobi XPBD baseline, writes `validity_matrix.csv`, and
  blocks downstream performance output if a required `386^2` hanging gather
  case is invalid or misses its quality gate.
- Nsight and benchmark wrappers now use actual rendering for paper-labelled
  runs. `--no-render` is rejected for a `paper-*` label.

## Pending Resume Point

The current commit adds `--profile-line-search-decisions`. When enabled, the
extended frame CSV records Armijo rejections/failures/fallbacks and accepted
candidate indices. This tracing deliberately adds decision readbacks and is
for the line-search study only; it must not be used for the R2 timing runs.

Before any long sweep, run a short rendered smoke for both
`gpu-gather-fusion` (serial Armijo) and
`gpu-gather-fusion-batched-ls-persistent` with this flag, then inspect
`frame_profile_extended.csv` and `run_metadata.json`. No formal R2 reference,
calibration, performance, stability, material-matrix, or line-search sweep
has been launched at this checkpoint.
