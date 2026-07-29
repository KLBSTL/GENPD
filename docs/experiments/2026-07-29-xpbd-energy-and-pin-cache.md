# XPBD Energy Audit and Pin-Cache Study

## Scope

This narrow implementation audit checks the energy equations and one XPBD
optimization without changing the XPBD iteration budget or rendered workload.
CPU still controls frame stepping and compute dispatch; the accurate description
is **simulation state GPU-resident**, not GPU-autonomous.

## Energy Definitions

For implicit-Euler NCG, the objective is

\[
F(x) = \tfrac{1}{2}(x-y)^T M(x-y) + h^2 E_C(x),
\]

where \(y=x_n+h v_n+h^2M^{-1}f_{ext}\), and \(E_C\) is the spring plus finite
attachment penalty. External force is already in \(y\), so it must not be
subtracted from this objective again.

The diagnostic records, independently, the implicit objective, raw constraint
energy, inertia, physical elastic-plus-external potential \(E_C-f_{ext}^Tx\),
kinetic energy, mechanical sum, constraint residuals, and a central
finite-difference directional derivative. `energy_for_linesearch.comp` is thus
an optimization-objective evaluator, not a physical-potential evaluator.
`evaluatePotentialEnergyCS` now reads the raw constraint term and subtracts
external work only. Sphere contact remains a nonsmooth projection and is not
represented by a smooth potential.

XPBD applies attachment vertices as exact hard pins after its Jacobi correction,
whereas CPU NCG has finite attachment penalties. This audit verifies formulas
within each implementation; it does not claim identical XPBD/NCG energy models.

### Formula Result

`results/xpbd-energy-fusion-20260729-r1/energy_audit/` used commit `4f8cefa`,
256 x 256 moving-sphere cloth, 32 XPBD iterations, 2 warm-up plus 8 measured
frames, actual 1600 x 900 rendering, GPU synchronization, and disabled VSync.
All 10 logged frames were finite.

| Check | Maximum error |
|---|---:|
| CPU/GPU implicit objective relative error | `2.450e-05` |
| CPU/GPU raw constraint energy relative error | `8.557e-05` |
| CPU/GPU inertia relative error | `3.477e-05` |
| Directional derivative absolute error | `2.355e-08` |

The largest directional-derivative relative error was `2.355e-02`, but both
derivatives were near zero in single precision. The absolute criterion remains
well below `1e-6`; the audit accepts the formula implementation. It performs
readbacks and is excluded from performance timing.

## Apply/Collision Fusion: Negative Result

The first local change fused vertex apply with sphere collision. It preserves
the order constraint scatter, vertex correction and hard pinning, then collision
projection. A rendered 12-checkpoint test was an exact byte match.

At 32 iterations, dispatches fell from 96 to 64 per frame. Yet GPU-query solver
time changed only from `8.931 ms` to `8.872 ms` (`0.66%`). Three rendered
30+300-frame repetitions gave `19.442 +/- 1.273 ms` unfused and
`20.171 +/- 2.557 ms` fused. Fusion is not a performance claim. Constraint
scatter atomics and the per-iteration barrier, rather than launch count, remain
the dominant work.

## Hard-Pin Cache

The old XPBD vertex pass scanned every vertex's CSR list in every iteration only
to find a hard attachment. Commit `69be4e2` precomputes the first attachment per
vertex as a `vec4` pin record when CSR is rebuilt. The legacy scan remains
available through `--xpbd-cached-pins 0`; cached pins are the default.

The cache does not change constraints, lambda persistence, iteration count, or
collision order. A rendered 12-checkpoint comparison at 256 x 256, 32
iterations was an exact byte match.

### Rendered Results

`results/xpbd-pin-cache-20260729-r1/` uses commit `69be4e2`, RTX 3070 Laptop,
driver `581.57`, 256 x 256 moving-sphere cloth, 32 XPBD iterations, 1600 x 900
rendering, GPU synchronization, disabled VSync, 30 warm-up plus 300 measured
frames, and three repetitions. All 1,800 measured frames were valid. Fusion
was enabled in both conditions, so pin lookup is the sole controlled difference.

| Condition | Frame-time mean +/- std. (ms) | P50 (ms) | P95 (ms) | XPBD dispatches/frame |
|---|---:|---:|---:|---:|
| Legacy CSR pin scan | `17.377 +/- 1.998` | `16.692` | `20.557` | 64 |
| Cached pin record | `14.400 +/- 3.415` | `12.875` | `19.628` | 64 |

The sample means are `17.1%` lower and all-frame median is `22.9%` lower with
cached pins. Repetition variability is large and these first three repetitions
were blocked by condition, so this is evidence consistent with a benefit, not a
final high-confidence paper speedup estimate.

The separate 60-frame rendered GPU-query diagnostic measured `7.922 ms` for
legacy XPBD and `3.546 ms` cached, a `55.2%` solver-loop reduction. Its
end-to-end values were `17.234 ms` and `12.522 ms`. Query readback perturbs
timing, so it localizes the bottleneck rather than replacing the main table.

## Interpretation

The earlier state-residency causal test gained `6.6%`: eliminating roughly
3 MiB/frame of state traffic leaves the 32 iterative passes unchanged. Caching
pins attacks repeated vertex-side CSR loads paid 32 times per frame, hence the
larger solver effect. The next structural target is still constraint-owned
atomic scatter and its dependency barrier, not additional CPU/GPU state copies.

## Reproduction

```powershell
& .\scripts\run_xpbd_energy_fusion_study.ps1
```

Pin-cache A/B uses the same rendered runner with
`--xpbd-fuse-apply-collision 1 --xpbd-cached-pins 0|1`. Rerun the two conditions
in interleaved order before using the cache timing in a publication figure.
