# Rendered Paper Protocol

The aborted `paper-20260723` no-render outputs are implementation diagnostics only. They are excluded from the paper evidence chain.

Formal runs use `paper-20260723-rendered` and require all of the following:

- Rendered OpenGL frames at a fixed `1600 x 900` viewport.
- `--sync-gpu` so the presentation measurement includes submitted GPU work.
- `--disable-vsync` to avoid display-refresh capping; the run log records whether WGL accepted it.
- No screenshot readback during reference, calibration, performance, or stability timing.
- `frame_presentation.csv` with one row per simulated frame. `frame_wall_ms` is the primary end-to-end metric; `render_and_present_wall_ms` is reported separately.
- Existing `frame_profile.csv` remains a solver-breakdown record and is not labeled as rendered frame time.

The manifest, summarizer, and figure generator reject a formal run when it is no-render, lacks GPU synchronization, uses a different viewport, has missing presentation rows, or contains non-finite values.
