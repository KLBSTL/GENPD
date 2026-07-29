# Formal XPBD Residency Scaling R3

## Scope and Protocol

This is the current formal XPBD residency matrix, produced at commit
`426c37f62b9874f53120aa16dd2e5ac97a4e5f1d` on an NVIDIA GeForce RTX 3070
Laptop GPU with driver `581.57`.

- Scene: moving-sphere cloth.
- Meshes: 128x128, 256x256, and 386x386.
- XPBD: atomic Jacobi, 32 iterations/frame, 96 compute dispatches/frame
  (constraint, apply, collision each execute 32 times).
- Conditions: ordinary GPU-resident state and forced CPU position/velocity
  roundtrip before the next frame.
- Timing: rendered 1600x900, `--uncapped`, `--sync-gpu`, VSync disabled;
  30 warm-up + 300 measured frames, three independent repetitions.
- Trajectory/quality audit: a separate rendered 20 warm-up + 120 frame run,
  checkpointed every 10 frames. It is excluded from timing.

The forced condition is a causal counterfactual: it retains the same XPBD
dispatch sequence and solver budget, then reads final position/velocity state
to CPU and reuploads it for the next frame. CPU still controls dispatch; the
claim is **simulation state GPU-resident**, not GPU-autonomous simulation.

## Rendered Timing and State Traffic

| Mesh | Condition | Mean +/- std ms | P50 / P95 ms | H2D / D2H MiB/frame | Upload / readback calls | XPBD dispatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128x128 | resident | 4.266 +/- 0.819 | 4.059 / 5.741 | 0 / 0 | 0 / 0 | 96 |
| 128x128 | forced roundtrip | 4.697 +/- 0.240 | 4.590 / 5.863 | 0.375 / 0.375 | 2 / 2 | 96 |
| 256x256 | resident | 17.011 +/- 2.036 | 17.268 / 20.072 | 0 / 0 | 0 / 0 | 96 |
| 256x256 | forced roundtrip | 18.892 +/- 1.643 | 18.903 / 22.333 | 1.500 / 1.500 | 2 / 2 | 96 |
| 386x386 | resident | 32.682 +/- 0.151 | 32.348 / 35.709 | 0 / 0 | 0 / 0 | 96 |
| 386x386 | forced roundtrip | 40.421 +/- 0.340 | 40.136 / 43.777 | 3.410 / 3.410 | 2 / 2 | 96 |

| Mesh | Resident speedup by repetition mean | Resident speedup by pooled P50 |
| --- | ---: | ---: |
| 128x128 | 1.10x | 1.13x |
| 256x256 | 1.11x | 1.09x |
| 386x386 | 1.24x | 1.24x |

The timing direction is consistent across all three resolutions: eliminating
the deliberately injected full-state transfers reduces rendered frame time.
The absolute benefit grows with state size, while the relative benefit remains
smaller than a compute-light NCG residency comparison can show.

## Trajectory and Physical Audit

| Mesh | P95 position relative L2 | P95 velocity relative L2 | P95 mean stretch, resident / forced | P95 max stretch, resident / forced | Max penetration, resident / forced |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128x128 | 8.993e-05 | 1.301e-02 | 0.005122 / 0.005121 | 1.01977 / 1.01979 | 0.00100024 / 0.00100036 |
| 256x256 | 2.339e-06 | 3.218e-04 | 0.004576 / 0.004576 | 2.08798 / 2.08798 | 0.00100036 / 0.00100036 |
| 386x386 | 1.608e-06 | 1.737e-04 | 0.003586 / 0.003586 | 3.23649 / 3.23652 | 0.00100036 / 0.00100036 |

All six trajectory runs are finite, all pass the registered position P95
relative-L2 gate of `1e-3`, and resident/forced strain and penetration values
match at the shown precision. Velocity relative L2 is reported rather than
used as the pass/fail gate. At 128x128 it reaches `1.301e-02`, reproducibly
above the earlier `1e-3` velocity threshold, despite the small position error
and matched constraint-quality metrics.

This is a real limitation of the counterfactual, not a result to suppress:
full host/device state representation crossings can accumulate a velocity
difference over the 140-frame audit. Therefore R3 supports a controlled
position/constraint-quality and data-traffic claim, but **does not support a
strict bitwise or velocity-equivalent trajectory claim at every resolution**.
Any final manuscript should state this boundary or investigate the 128x128
velocity divergence further before using stronger wording.

## Artifact Contract

The ignored raw root
`results/paper-20260729-xpbd-residency-scaling-r3/` contains the parent
manifest, 24 rendered run directories, all profiles/metadata, trajectory
checkpoints/quality metrics, and the generated CSV/report. The prior aborted
R2 root is retained separately as an audit record of the original strict
velocity-gate failure.
