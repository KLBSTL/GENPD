# Adaptive Line Search and External Baseline Scope

## Development Baseline

Development baseline: `1cbd66f00cb5dd005069e102a22dc3ff092e0867`.

This is the current `main` merge commit. Its second parent is the tip of the
merged `paper-experiment-20260722` branch, so it contains the rendered R2
protocol, persistent-state roundtrip validation, vertex-owned dataflow
diagnostics, quality gates, and the existing internal GPU XPBD baseline.

All new experiments in this phase must record a descendant of this commit.
Historical R2 data remain read-only and must not be reinterpreted to support
the adaptive-line-search contribution.

## Frozen Infrastructure

The following infrastructure is already present and is consumed rather than
rebuilt:

- seven existing solver variants and their command-line metadata contract;
- protocol-v4 rendered timing, quality-reference checkpoints, and validity
  gates;
- K/beta/restart sweeps and on-demand Armijo decision tracing;
- GPU-resident persistent state, forced CPU-state-roundtrip validation, and
  vertex-owned gradient diagnostics;
- the internal GPU Jacobi XPBD baseline.

## New Work Scope

This checkpoint permits only the following additions:

1. one `gpu-gather-fusion-adaptive-ls-persistent` variant;
2. an isolated persistent SSBO for adaptive Armijo history;
3. temporal-history scopes (`none`, `iteration`, and `frame`) and trace-only
   telemetry;
4. formal residency and vertex-owned microstudies using the established
   rendered/quality protocol;
5. a provenance-preserving, same-hardware operating-point reproduction of
   Wang 2021 without copying third-party source into this repository unless
   its license permits redistribution;
6. protocol-v5, manuscript integration, and a claim-evidence audit driven by
   the resulting data.

Adaptive Armijo is a conditional contribution. It may appear as a headline
contribution only after the pre-registered Go conditions are satisfied;
otherwise it remains an implementation optimization with its negative or
neutral result reported.

## Baseline Verification

On 2026-07-24, the following contracts passed from this baseline:

- `scripts/test_solver_variants.ps1`
- `scripts/test_solver_variant_metadata.ps1`
- `scripts/test_line_search_sweep_contract.ps1`
- `scripts/test_fused_gradient_stats.ps1`
- `scripts/test_gather_regression_contract.ps1`
- `scripts/test_paper_experiment_contract.ps1`

`scripts/build_release.ps1` also produced `Release|x64` successfully.

## Evidence Boundaries

- Paper performance data must remain rendered end-to-end measurements; do not
  use `--no-render` as paper timing evidence.
- CPU control of iteration, line search, and dispatch remains explicit. The
  accurate phrase is "simulation state GPU-resident", not GPU autonomous.
- Wang 2021 is external same-hardware operating-point context, not an
  equal-quality solver ranking unless a separately documented compatible case
  is achieved.
