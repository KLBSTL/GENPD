# PeriDyno PPM r1 正式运行点结果说明

**结果根目录：** `results/paper-20260729-ppm-r1/`  
**记录日期：** 2026-07-29  
**用途：** 外部 GPU 可变形模拟基线的同硬件、实际渲染运行点（operating point）记录。

## 1. 结论与使用边界

本轮已成功将 Lu et al. 的 2024 PPM（Projective Peridynamic Modeling of
Hyperelastic Membranes With Contact）作者实现 PeriDyno 接入当前项目的两类
场景：悬挂布料和运动球外部障碍物。PPM 在同一 GPU 上完成了实际窗口渲染、
有限性检查和三次重复计时。

这组结果**可以**用于说明：外部的 2024 GPU membrane/contact 方法能够在
当前映射场景与同一硬件上运行，并给出其完整 rendered end-to-end operating
point。

这组结果**不能**用于说明 GenPD 相对 PPM 的等质量加速比，原因如下：

- 两者的物理模型不同：GenPD 是 mass-spring cloth + attachment + NCG；PPM
  是三角形 hyperelastic peridynamic membrane。
- 材料参数、离散能量、收敛标准和误差度量尚未校准为同一质量门槛。
- 当前只报告 PPM 的 128 x 128 运行点，未完成跨模型的质量对齐或全分辨率
  扫描。

论文中应将它标为 “same-hardware external operating point”，不能放进
“equal-quality speedup”排序图，也不应从中推导 GenPD 的 FPS 倍率。

## 2. 可追溯性

| 项目 | 记录值 |
| --- | --- |
| GenPD commit | `8c23d8318a885fed87332212575720e97b9a4ed3` |
| PeriDyno revision | `12a5f00dd95d5f3594493b7d8bff27d20dee13cf` |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |
| NVIDIA driver | 581.57 |
| GPU memory | 8192 MiB |
| PPM adapter | `external_baselines/peridyno_ppm/GenPD_PPM_OperatingPoint.cpp` |
| 正式编排脚本 | `scripts/run_peridyno_ppm_formal_operating_point.ps1` |
| 完整配置 | `results/paper-20260729-ppm-r1/manifest.json` |
| 汇总 CSV | `results/paper-20260729-ppm-r1/operating_point_summary.csv` |
| 单次重复 CSV | `results/paper-20260729-ppm-r1/per_repetition_summary.csv` |

`manifest.json` 还记录了正式编排脚本、基础 runner、PPM adapter 和 PeriDyno
compatibility patch 的 SHA-256，因此该记录可以定位到具体实验源码。

## 3. 协议

- 网格：`128 x 128 = 16,384` vertices。
- 场景：`hanging` 与 `moving-sphere`；均使用规则网格和两个顶部固定点。
- PPM 时间步：`dt = 0.001 s`；每个名义 GenPD frame 前进 33 个 PPM substeps。
- PPM solver：每 substep 10 iterations。
- 碰撞范围：启用 PPM external SDF sphere contact；禁用 self-contact，与当前
  GenPD 无自碰撞的适用范围一致。
- moving-sphere：每个 PPM substep 刷新 SphereModel 和 `BasicShapeToVolume`
  SDF，刷新开销包含在计时中。
- 呈现：`1600 x 900` GLFW graphics update、draw 和 buffer swap；VSync 关闭。
- 计时：每场景 30 warm-up frames，随后 3 次独立重复，每次 300 measured
  frames；每场景共 900 measured frames。
- 主指标：`frame_host_ms`，包含 PPM substeps 以及 graphics update/draw/swap；
  不含启动、销毁和截图读回。
- 辅助指标：`simulation_gpu_ms` 是 CUDA-event solver-side 诊断，不包含
  OpenGL draw，不能替代主指标。

截图仅在每个场景的第一重复完成计时后写出，因此不会污染 frame timing：

- `results/paper-20260729-ppm-r1/hanging/128x128/rep01/proof.bmp`
- `results/paper-20260729-ppm-r1/moving-sphere/128x128/rep01/proof.bmp`

## 4. 汇总结果

均值和标准差按三次 repetition mean 计算；P50/P95 由每场景全部 900 帧池化
计算。单位均为毫秒。

| 场景 | repetitions | measured frames | `frame_host_ms` mean +/- std | P50 | P95 | `simulation_gpu_ms` mean +/- std | 最终有限 | moving-SDF refresh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| hanging | 3 | 900 | 91.70 +/- 1.67 | 91.46 | 96.39 | 90.23 +/- 1.69 | 是 | 否 |
| moving-sphere | 3 | 900 | 171.37 +/- 2.81 | 168.06 | 191.83 | 169.71 +/- 2.81 | 是 | 是 |

单次重复均值如下，便于检查批间波动：

| 场景 | rep 1 `frame_host_ms` | rep 2 | rep 3 | rep 1 `simulation_gpu_ms` | rep 2 | rep 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hanging | 92.34 | 92.95 | 89.81 | 90.94 | 91.45 | 88.30 |
| moving-sphere | 172.47 | 168.18 | 173.46 | 170.82 | 166.51 | 171.78 |

## 5. 审计结果

以下条件均已通过：

- 六个 run（2 scenes x 3 repetitions）都包含恰好 300 条 measured frame
  records。
- 每条 frame record 标记为 `rendered=1`，元数据为 1600 x 900 渲染且
  `final_state_finite=true`。
- 所有 run 的 `dt`、substeps、solver iterations、warm-up 和呈现分辨率与
  manifest 一致。
- PPM 正式 runner 对缺失 CSV、帧数不匹配、非有限 timing、未渲染记录或
  非有限最终状态会拒绝汇总。
- 已运行 `scripts/test_peridyno_ppm_contract.ps1` 与
  `scripts/test_peridyno_ppm_formal_contract.ps1`；二者均通过。

## 6. 对结果的解释

运动球场景的主指标相对 hanging 高 `79.67 ms`（约 `86.9%`）。这说明在该
特定 PPM operating point 下，激活的外部 obstacle/contact 路径连同每
substep 的 moving-SDF refresh 带来明显端到端成本。

该差值不是通用的“接触成本定律”：它还受 PPM 的网格更新、SDF 构建、场景
几何和当前 material/solver setting 影响。它更不能用来推导 GenPD 对接触
场景的相对优势。

截图验证了两个场景均被实际绘制。最终 moving-sphere 截图对应第 300 个
measured frame 后的状态，球已在布料下方；它是运行和呈现证据，不应单独
作为某一碰撞瞬间的物理质量图。

## 7. 论文可用写法

可以使用如下保守表述：

> We additionally report a same-hardware rendered operating point for the
> authors' PeriDyno implementation of PPM (2024) on the mapped hanging and
> moving-obstacle scenes. Because PPM and GenPD use different constitutive
> models, material parameters, and quality criteria, these measurements are
> reported as external operating-point context and are not used for an
> equal-quality speedup comparison.

表格标题可写为：`External PPM rendered operating point (not an
equal-quality comparison)`。表中至少保留 mesh size、substeps、iterations、
GPU、driver、three-repetition statistics 和 no-self-contact 的范围说明。

## 8. 复现命令

在 `D:\GenPD_test_p\GenPD` 下运行：

```powershell
scripts\test_peridyno_ppm_contract.ps1
scripts\test_peridyno_ppm_formal_contract.ps1
scripts\run_peridyno_ppm_formal_operating_point.ps1 -RunLabel paper-20260729-ppm-r1
```

最后一条命令会复用已经完整的结果目录；使用新的 `-RunLabel` 时会执行完整
的 2 scenes x 3 repetitions 正式矩阵。

## 9. 后续工作

外部 PPM operating point 已完成。主线的下一项是 GPU XPBD 的 resident
state 与 forced CPU roundtrip 因果对照；该实验能直接回答“simulation state
GPU-resident”带来的端到端收益，而不把不同物理模型混入同一个速度结论。
