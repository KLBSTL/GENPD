# Persistent GPU-Resident State Study

## Protocol

- Commit `7d82a84`; rendered 1600x900 timing with 30 warm-up + 300 measured frames and three repetitions.
- Conditions differ only by the per-frame position/velocity CPU roundtrip; timing excludes quality readback and capture.
- Separate rendered quality runs use the archived CPU-NCG reference checkpoints from `c73d2bb`.

## Results

| Scene | Grid | Resident ms | Roundtrip ms | Ratio | H2D/D2H resident MB | H2D/D2H roundtrip MB | Dispatches | P95 position error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hanging | 256^2 | 0.953 +/- 0.043 | 3.155 +/- 0.130 | 3.31x | 0.000/0.000 | 1.500/1.500 | 6 | 4.851e-04 |
| Hanging | 386^2 | 1.790 +/- 0.106 | 9.703 +/- 0.227 | 5.42x | 0.000/0.000 | 3.410/3.410 | 6 | 4.829e-04 |
| Moving sphere | 256^2 | 1.024 +/- 0.030 | 2.940 +/- 0.139 | 2.87x | 0.000/0.000 | 1.500/1.500 | 6 | 4.851e-04 |
| Moving sphere | 386^2 | 1.821 +/- 0.118 | 9.118 +/- 0.530 | 5.01x | 0.000/0.000 | 3.410/3.410 | 6 | 4.829e-04 |

The result supports an implementation-level claim about avoiding the measured full-state host roundtrip in these configurations. It does not claim zero synchronization or a cross-API comparison.
