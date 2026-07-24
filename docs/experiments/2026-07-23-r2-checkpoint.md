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
- `f567a39` makes the R2 manifest self-describing for its full calibration
  budget set and validity-matrix schema. Figure generation distinguishes an
  invalid frame from a finite result that misses its quality gate.

## Line-Search Smoke Evidence

The decision-tracing contract was exercised with actual rendering, not with
`--no-render`:

- `results/smoke-ls-serial-h128` verifies serial Armijo. Every one of its
  eight solver iterations per frame accepted candidate index zero.
- `results/smoke-ls-batched-direct-accept` verifies the batched CPU-state
  direct-accept return path. It records four accepted candidates for four
  iterations per frame.
- `results/smoke-ls-persistent-gradient-semantics` verifies persistent batched
  Armijo with the moving sphere. Frames are valid, and the extended profile
  marks its un-read GPU gradient norm as unavailable rather than non-finite.

## Pending Resume Point

The current commit adds `--profile-line-search-decisions`. When enabled, the
extended frame CSV records Armijo rejections/failures/fallbacks and accepted
candidate indices. This tracing deliberately adds decision readbacks and is
for the line-search study only; it must not be used for the R2 timing runs.

The smoke prerequisite is complete. Before a long line-search sweep, generate
the CPU-NCG reference checkpoints for its chosen scene/resolution and pass
them to `run_line_search_sweep.ps1 -RequireReference`. No formal R2 reference,
calibration, performance, stability, material-matrix, or line-search sweep
has been launched at this checkpoint.

The line-search runner now uses protocol 2: each timing configuration defaults
to three actual-render repetitions, while one separate rendered trace collects
the Armijo/restart diagnostics and quality metrics. `ProcessTimeoutSeconds` and
an inter-run delay prevent a stalled OpenGL process from blocking the full
matrix. `results/smoke-line-search-repeats-2` verified two repetitions for both
serial and batched variants, with valid frames and no residual process. Its
one-frame measurements are plumbing evidence only, not paper data. The
evidence-gated `generate_line_search_figures.py` refuses any run missing the
pre-registered 300+30/3 timing protocol, 120+20 trace protocol, or CPU quality
reference.

## Scene/Material Smoke Evidence

`results/smoke-scene-material-rect` exercised the separate material matrix
pipeline on a `512x128` rectangular cloth (65,536 vertices), hanging scene,
stretch stiffness 80, and bending stiffness 20. It produced a rendered CPU
reference, a quality comparison, and an untraced rendered timing run. Its two
frames only validate the square/rectangle and metric plumbing; they are not
formal material-matrix evidence.

`results/smoke-scene-material-r2-schema` repeats the same kind of plumbing
check under the protocol-2 runner: rendered reference and quality passes plus
two protected timing repetitions. It confirms that the summary contains the
per-repetition frame budget, total sample count, mean/std fields, commit, GPU,
and driver metadata. Its one-frame configuration remains smoke-only. The
formal matrix must use both scenes, both mesh shapes, all nine material pairs,
the 120+20 reference/quality protocol, and three 300+30 timing repetitions.
`generate_scene_material_figures.py` rejects any smaller or incomplete matrix.

## Gather Stability Repair

`results/gradient-diagnostics-r2-preflight` is diagnostic-only (no rendering
and never paper evidence). It verifies the CSR and initial CPU/edge-scatter/
gather gradient on both scenes at `128^2`, `256^2`, and `386^2`. All six cases
passed the `1e-4` relative threshold; gather relative L2 ranged from about
`5.3e-7` to `1.3e-6`. The historical E0 issue was therefore not an adjacency
or initial-gradient mismatch.

The rendered regression then found an uninitialised-descent defect in
`shaders/descent.comp`: its initial update evaluated `-g + 0*d_old`, and IEEE
`0 * NaN` propagated an uninitialised GPU buffer value into the position update.
The fix makes `update_mode == 0` write `d = -g` without reading old descent.
`frame_profile_extended.csv` now includes `gradient_dot_descent` for this
diagnostic boundary.

The repaired, commit-aligned actual-render regression is
`results/gather-regression-h386-0243d0b`. It covers all four gather variants at
`386^2` hanging cloth for every R2 candidate budget
`{1,2,4,6,8,10,12,16,20,24,32,48,64}`. All 52 cases record commit `0243d0b`
and completed three measured frames after one warm-up frame with no invalid
state, no `process_timeout`, and no simultaneous `converged`/`exploded` flag.
These are stability regressions, not equal-quality performance measurements;
R2 calibration remains pending.
