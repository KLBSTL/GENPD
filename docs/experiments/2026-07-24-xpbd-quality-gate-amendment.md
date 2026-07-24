# XPBD Quality-Gate Amendment

## Status

This amendment supersedes the XPBD quality gate in the unlaunched formal R2
matrix. The runner now emits protocol version `4`. Historical protocol-2 and
protocol-3 preflights
directories remain immutable and are not eligible for the formal paper
summary or figures.

## Evidence That Invalidated The Old Gate

Rendered preflight `results/r2-preflight-20260724-h128-8cd665c` used the
same hanging `128^2` scene, `20` warm-up frames, `120` measured frames, and
the CPU NCG `100`-iteration reference for every candidate.

- The reference P95 mean stretch strain was `0.002991472`.
- The reference P95 maximum stretch strain was `0.98095805`.
- The reference maximum penetration depth was `0`.
- GPU XPBD at 64 iterations was finite, had P95 mean strain `0.0045677535`,
  P95 maximum strain `0.89483955`, and zero penetration.

The old absolute XPBD maximum-strain limit of `0.10` would reject the
reference physical state itself. It could therefore not serve as a quality
gate for this pinned-cloth setup.

## Protocol-4 Gate

For `gpu-xpbd-jacobi`, a candidate must be finite with no invalid frame and
must satisfy all of:

- P95 mean stretch strain at reference checkpoint frames `<= 0.02`.
- P95 maximum stretch strain at the same reference checkpoint frames
  `<= 1.10 *` the CPU-reference P95 maximum stretch strain.
- Maximum external-contact penetration depth `<= 0.02`.

The calibration CSV and validity matrix record both full-frame reporting
metrics and the checkpoint-only current/reference strain values used for the
decision. This keeps the XPBD comparison in a strain/penetration quality space
while retaining a threshold compatible with the scene's physical reference
solution and a symmetric sampling rule.
