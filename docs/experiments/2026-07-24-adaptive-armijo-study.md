# Adaptive Armijo Study

## Protocol

- Commit `c8831ec`; all timing is rendered 1600x900 with 30 warm-up + 300 measured frames and three repetitions.
- A separate rendered 100-iteration CPU-NCG reference produces checkpoints for the 120-frame quality/decision traces. Timing excludes quality readback and decision tracing.
- Core: serial persistent Armijo, fixed K=8 persistent, and adaptive K=4 persistent with no/iteration/frame history. Sensitivity: moving sphere at 256^2 and 386^2, K={2,4,8}, beta={0.25,0.5,0.75}, and three history modes.

## Core results

| Scene | Grid | Method | Frame ms | LS ms | Candidates/search | History use | P95 position error |
|---|---:|---|---:|---:|---:|---:|---:|
| hanging | 128 | Serial Armijo | 0.963 +/- 0.035 | 0.530 | 0.000 | 0.000 | 8.67e-05 |
| hanging | 128 | Fixed K=8 | 0.671 +/- 0.020 | 0.453 | 8.000 | 0.000 | 8.67e-05 |
| hanging | 128 | Adaptive K=4 | 0.781 +/- 0.019 | 0.981 | 4.000 | 0.000 | 8.67e-05 |
| hanging | 128 | Adaptive + iter. hist. | 0.810 +/- 0.044 | 0.962 | 4.000 | 0.500 | 8.67e-05 |
| hanging | 128 | Adaptive + frame hist. | 0.811 +/- 0.012 | 0.759 | 4.000 | 1.000 | 8.67e-05 |
| hanging | 256 | Serial Armijo | 1.233 +/- 0.072 | 0.781 | 0.000 | 0.000 | 7.23e-04 |
| hanging | 256 | Fixed K=8 | 0.927 +/- 0.047 | 0.464 | 8.000 | 0.000 | 7.23e-04 |
| hanging | 256 | Adaptive K=4 | 1.081 +/- 0.013 | 0.900 | 4.000 | 0.000 | 7.23e-04 |
| hanging | 256 | Adaptive + iter. hist. | 1.091 +/- 0.039 | 0.985 | 4.000 | 0.000 | 7.23e-04 |
| hanging | 256 | Adaptive + frame hist. | 1.093 +/- 0.030 | 0.952 | 4.000 | 1.000 | 7.23e-04 |
| hanging | 386 | Serial Armijo | 2.333 +/- 0.293 | 1.156 | 0.000 | 0.000 | 4.89e-04 |
| hanging | 386 | Fixed K=8 | 1.756 +/- 0.123 | 0.731 | 8.000 | 0.000 | 4.89e-04 |
| hanging | 386 | Adaptive K=4 | 2.074 +/- 0.058 | 1.640 | 4.000 | 0.000 | 4.89e-04 |
| hanging | 386 | Adaptive + iter. hist. | 2.118 +/- 0.136 | 1.459 | 4.000 | 0.000 | 4.89e-04 |
| hanging | 386 | Adaptive + frame hist. | 2.067 +/- 0.102 | 1.462 | 4.000 | 1.000 | 4.89e-04 |
| moving-sphere | 128 | Serial Armijo | 0.999 +/- 0.028 | 0.501 | 0.000 | 0.000 | 8.67e-05 |
| moving-sphere | 128 | Fixed K=8 | 0.731 +/- 0.033 | 0.343 | 8.000 | 0.000 | 8.67e-05 |
| moving-sphere | 128 | Adaptive K=4 | 0.860 +/- 0.033 | 0.854 | 4.000 | 0.000 | 8.67e-05 |
| moving-sphere | 128 | Adaptive + iter. hist. | 0.828 +/- 0.002 | 0.928 | 4.000 | 0.500 | 8.67e-05 |
| moving-sphere | 128 | Adaptive + frame hist. | 0.839 +/- 0.047 | 0.841 | 4.000 | 1.000 | 8.67e-05 |
| moving-sphere | 256 | Serial Armijo | 1.275 +/- 0.019 | 0.675 | 0.000 | 0.000 | 7.23e-04 |
| moving-sphere | 256 | Fixed K=8 | 1.027 +/- 0.046 | 0.437 | 8.000 | 0.000 | 7.23e-04 |
| moving-sphere | 256 | Adaptive K=4 | 1.165 +/- 0.042 | 0.899 | 4.000 | 0.000 | 7.23e-04 |
| moving-sphere | 256 | Adaptive + iter. hist. | 1.163 +/- 0.039 | 0.977 | 4.000 | 0.000 | 7.23e-04 |
| moving-sphere | 256 | Adaptive + frame hist. | 1.182 +/- 0.041 | 0.914 | 4.000 | 1.000 | 7.23e-04 |
| moving-sphere | 386 | Serial Armijo | 2.321 +/- 0.068 | 1.177 | 0.000 | 0.000 | 4.89e-04 |
| moving-sphere | 386 | Fixed K=8 | 1.801 +/- 0.122 | 0.723 | 8.000 | 0.000 | 4.89e-04 |
| moving-sphere | 386 | Adaptive K=4 | 2.188 +/- 0.039 | 1.594 | 4.000 | 0.000 | 4.89e-04 |
| moving-sphere | 386 | Adaptive + iter. hist. | 2.087 +/- 0.005 | 1.485 | 4.000 | 0.000 | 4.89e-04 |
| moving-sphere | 386 | Adaptive + frame hist. | 2.139 +/- 0.089 | 1.516 | 4.000 | 1.000 | 4.89e-04 |

## Go / no-go

- [x] hanging: adaptive candidate reduction: `0.5`
- [x] hanging: adaptive quality: `0.000489363`
- [x] hanging: adaptive valid/no Armijo failures: `0`
- [x] moving-sphere: adaptive candidate reduction: `0.5`
- [x] moving-sphere: adaptive quality: `0.000489364`
- [x] moving-sphere: adaptive valid/no Armijo failures: `0`
- [ ] at least one 386 case improves end-to-end by >=5%: `-0.176946`
- [ ] other 386 case regresses by <=3%: `-0.187876`
- [ ] frame history reduces line-search time by >=5% in both 386 cases: `0.0489293`
- [ ] adaptive is faster in >=2/3 repetitions for both 386 cases: `0`

Result: **NO-GO**.
- 386^2 end-to-end improvements (hanging, moving sphere): -17.69%, -18.79%.
- Frame-history line-search reductions (hanging, moving sphere): 10.89%, 4.89%.
- Faster adaptive repetitions (hanging, moving sphere): 0/3, 0/3.

The sensitivity CSV is `adaptive_armijo_sensitivity.csv`; Figure: `results/paper-20260724-adaptive-armijo-r1/adaptive_armijo.pdf`.
