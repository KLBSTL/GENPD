# GPU XPBD Baseline Status

Commit-level baseline: `gpu-xpbd-jacobi` is an internal, standard Jacobi XPBD
comparison implementation. It is not a reimplementation of an external paper.

## Algorithm Contract

- One compute invocation processes one spring or attachment constraint.
- Spring constraints retain one XPBD lambda per constraint for the current frame
  and atomically accumulate Jacobi position corrections.
- A vertex compute pass averages accumulated corrections and reapplies attachment
  targets as exact pins.
- The existing OpenGL collision resolve shader runs as a separate pass after each
  XPBD projection iteration when the scene contains supported plane or sphere
  primitives.
- XPBD lambdas reset at the beginning of each simulation frame. CPU code still
  controls the iteration count and every compute dispatch.

## Smoke Evidence Only

These rendered smoke runs validate the implementation path; they are not paper
performance results or quality-qualified rankings.

| Run | Scene | Mesh | Iterations | Result |
| --- | --- | --- | --- | --- |
| `results/smoke-xpbd-128` | hanging | `128^2` | 16 | 5/5 valid frames; constraint and vertex passes dispatched 16 times/frame. |
| `results/smoke-xpbd-sphere-128` | moving sphere | `128^2` | 16 | 5/5 valid frames; the separate collision pass dispatched 16 times/frame. |
| `results/smoke-xpbd-sphere-386` | moving sphere | `386^2` | 16 | 5/5 valid frames; the same three-pass solver path executed. |

The 120-frame `128^2` moving-sphere stability smoke remained finite with no
invalid frames, but its maximum stretch strain reached about `0.95`. It is
therefore not yet a quality-qualified XPBD configuration. The R2 calibration
must choose XPBD iteration budgets against predeclared strain and penetration
thresholds before it appears in any performance ranking.

## Remaining Requirements

- Add a dedicated XPBD quality gate to the R2 calibration manifest.
- Run 300 measured + 30 warm-up rendered frames with three repetitions.
- Report strain, penetration, failure rate, memory, dispatch counts, and timing
  alongside the NCG family; do not compare XPBD through the CPU-NCG position
  error gate.
