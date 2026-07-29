# XPBD Vertex-Gather Prototype

## Decision

Do not promote `gpu-xpbd-vertex-gather` to a performance baseline or paper
claim. It is a correctness-preserving diagnostic prototype that identifies the
cost moved out of the atomic constraint pass.

The route was selected after the rendered stage profile
`diagnostic-20260729-xpbd-stage-profile-r1` measured the legacy constraint
scatter at 81.0% of the XPBD GPU time. The prototype implements a two-pass
Jacobi update:

1. one constraint invocation computes lambda and writes one correction record;
2. one vertex invocation traverses the existing CSR adjacency and gathers the
   incident correction records before applying the averaged correction.

Pins and collision retain their existing semantics. The legacy
`gpu-xpbd-jacobi` path is unchanged.

## Evidence

All new code is at commit `a955299`.

- Study root: `results/diagnostic-20260729-xpbd-vertex-gather-r1/`
- Machine: NVIDIA GeForce RTX 3070 Laptop GPU, driver 581.57
- Scene: moving sphere cloth, 256 x 256 vertices, 32 XPBD iterations/frame
- Quality check: 12 actual rendered frames, 960 x 540, atomic XPBD checkpoints
  as the reference
- Timing: one actual rendered 1600 x 900 pilot per variant, 5 warm-up and 60
  measured frames, uncapped, VSync disabled, GPU synchronized

The gather trajectory was bit-identical to the atomic reference over the short
quality check: P95/max relative L2 were zero for both position and velocity;
all frames were finite and non-exploded. This validates the prototype's
Jacobi/pin semantics for this check, not a general equivalence proof.

| Variant | Constraint GPU stage | Vertex-side GPU stage | XPBD GPU stage total | Rendered frame pilot |
| --- | ---: | ---: | ---: | ---: |
| Atomic Jacobi | 3.269 ms | 0.758 ms | 4.027 ms | 14.830 ms |
| Vertex gather | 2.232 ms | 7.096 ms | 9.329 ms | 24.607 ms |

The constraint pass improves, but the vertex-side CSR gather more than absorbs
that saving. The gather path also replaces a 1,048,576-byte delta buffer with
a 5,210,160-byte edge-correction buffer. This is consistent with the CSR
walk's irregular edge-record loads and its higher memory traffic.

## Interpretation Boundary

The two GPU stage numbers come from GL timer queries whose result is read after
each substage. They identify where GPU work goes; they are not end-to-end frame
timing evidence. The 60-frame rendered frame-time result confirms the same
direction with those queries disabled, but has one repetition and is only an
engineering pilot. It is not suitable for a paper table, speedup claim, or
general claim that vertex ownership improves XPBD.

The raw CSV, metadata, and generated report remain at the study root. They can
be re-validated with:

```powershell
E:\Anaconda\envs\DL\python.exe scripts\analyze_xpbd_vertex_gather_prototype.py `
  --study-root results\diagnostic-20260729-xpbd-vertex-gather-r1
```

## Next Action

Keep the prototype as a negative result. Any further XPBD optimization should
avoid materializing a `vec4` per edge and avoid an irregular per-vertex CSR
walk. A different proposal needs a separate quality gate and a fresh rendered,
multi-repeat comparison before it enters the paper evidence chain.
