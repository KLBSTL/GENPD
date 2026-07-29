# XPBD Signed-Incidence Vertex Gather

## Change

`gpu-xpbd-vertex-gather` previously dereferenced a full `Edge` record for every
CSR incidence merely to recover the endpoint sign and filter attachments. The
new path builds one static `uint` buffer alongside the existing CSR:

- low 31 bits: spring edge index;
- high bit: current vertex is the second endpoint;
- `0xffffffff`: attachment incidence.

The gather shader now reads the packed incidence and edge correction directly.
It does not load `Edge` in the hot gather loop. The cached hard-pin path and
the cache-disabled compatibility scan preserve the prior pin behavior.

## Validation

- Implementation commit: `afeff1a`
- Scene: moving-sphere cloth, 256 x 256 vertices, 32 XPBD iterations/frame
- Quality protocol: 12 actual rendered frames at 960 x 540 against atomic
  XPBD checkpoints from the same commit
- Timing protocol: 5 warm-up plus 60 actual rendered 1600 x 900 frames,
  uncapped, VSync disabled, GPU synchronized

The signed-incidence path was bit-identical to atomic XPBD over the short
quality check: P95/max relative L2 were zero for both position and velocity;
all frames were finite and non-exploded.

| Variant | Constraint GPU stage | Vertex-side GPU stage | XPBD GPU stage total | Rendered frame pilot |
| --- | ---: | ---: | ---: | ---: |
| Atomic Jacobi | 3.269 ms | 0.758 ms | 4.027 ms | 14.830 ms |
| Raw CSR gather | 2.232 ms | 7.096 ms | 9.329 ms | 24.607 ms |
| Signed-incidence gather | 2.279 ms | 3.648 ms | 5.927 ms | 16.465 ms |

The packed representation adds a static 2,605,072-byte buffer for the
651,268 CSR incidences. It reduces raw-gather vertex-stage time by 1.945x and
total XPBD stage time by 1.574x. This confirms that full edge-record fetches
were a major bottleneck.

## Decision

Keep signed incidence as the correct implementation of the vertex-gather
prototype. It is substantially better than raw CSR gather, but it remains
slower than atomic Jacobi: 1.472x in XPBD GPU-stage time and 1.110x in the
one-repeat rendered frame pilot. It is therefore not a paper performance
claim, nor a replacement for the GPU XPBD baseline.

The remaining gap is the per-edge correction materialization plus the
per-vertex CSR walk. The next low-effort candidate remains static
spring-degree normalization for the atomic path, which removes contribution
count atomics without introducing this gather traffic.

Raw output and the generated report are at
`results/diagnostic-20260729-xpbd-signed-incidence-r1/`. Re-run the aggregation
with:

```powershell
E:\Anaconda\envs\DL\python.exe scripts\analyze_xpbd_signed_incidence_study.py `
  --raw-root results\diagnostic-20260729-xpbd-vertex-gather-r1 `
  --signed-root results\diagnostic-20260729-xpbd-signed-incidence-r1
```
