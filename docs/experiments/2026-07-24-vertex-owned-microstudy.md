# Vertex-Owned Gather Microstudy

## Protocol

- Commit `7d82a84`; rendered 1600x900 timing with 30 warm-up + 300 measured frames and three repetitions.
- The three variants share the same scene, topology, iteration budget, rendering, GPU synchronisation, and CPU-NCG reference checkpoints. Quality readback is a separate rendered run.
- The scalar atomic count is structural: `6 * spring constraints + 3 * attachment constraints` for edge scatter, and zero for gather. It is not a hardware transaction counter.

## Results

| Scene | Grid | Variant | Frame ms | Gradient+stats ms | Dispatches | Barriers | Atomics (M) | P95 position error |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| hanging | 256 | Edge scatter | 3.958 +/- 0.035 | 0.095 | 3.0 | 3.0 | 1.954 | 1.46e-04 |
| hanging | 256 | Vertex gather | 3.915 +/- 0.019 | 0.229 | 2.0 | 2.0 | 0.000 | 1.46e-04 |
| hanging | 256 | Gather + fusion | 3.884 +/- 0.128 | 0.227 | 1.0 | 1.0 | 0.000 | 1.46e-04 |
| hanging | 386 | Edge scatter | 8.982 +/- 0.256 | 0.193 | 3.0 | 3.0 | 4.451 | 1.01e-04 |
| hanging | 386 | Vertex gather | 9.291 +/- 0.104 | 0.492 | 2.0 | 2.0 | 0.000 | 1.01e-04 |
| hanging | 386 | Gather + fusion | 9.344 +/- 0.233 | 0.483 | 1.0 | 1.0 | 0.000 | 1.01e-04 |
| moving-sphere | 256 | Edge scatter | 4.876 +/- 0.047 | 0.093 | 3.0 | 3.0 | 1.954 | 1.46e-04 |
| moving-sphere | 256 | Vertex gather | 4.929 +/- 0.126 | 0.223 | 2.0 | 2.0 | 0.000 | 1.46e-04 |
| moving-sphere | 256 | Gather + fusion | 4.884 +/- 0.026 | 0.228 | 1.0 | 1.0 | 0.000 | 1.46e-04 |
| moving-sphere | 386 | Edge scatter | 11.874 +/- 0.280 | 0.198 | 3.0 | 3.0 | 4.451 | 1.01e-04 |
| moving-sphere | 386 | Vertex gather | 12.192 +/- 0.218 | 0.491 | 2.0 | 2.0 | 0.000 | 1.01e-04 |
| moving-sphere | 386 | Gather + fusion | 12.411 +/- 0.617 | 0.481 | 1.0 | 1.0 | 0.000 | 1.01e-04 |

## Interpretation boundary

This isolates the gradient dataflow. It demonstrates that gather removes the edge-scatter atomic accumulation and fusion removes the standalone stats pass; it does not by itself attribute every frame-time difference to atomics because rendering, reductions, and line search remain in the end-to-end timing scope.

Figure: `results/paper-20260724-vertex-owned-r1/vertex_owned_microstudy.pdf`.
