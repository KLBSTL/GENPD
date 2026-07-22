# GenPD 论文实验改进目标

状态文件创建时间：2026-07-22  
主开发仓库：`D:\GenPD_test_p\GenPD`  
实验分支：`paper-experiment-20260722`  
baseline tag：`baseline-20260722` -> `8433064`

## 写作与术语修正

- [ ] 全文将“GPU-resident”改成更准确的“simulation state GPU-resident”。
- [ ] 明确 CPU 仍负责每轮迭代、line search 和 kernel dispatch 的控制流，不暗示完全 GPU autonomous。
- [ ] 清理所有占位符、错误引用、未验证术语和过度泛化表述。
- [ ] 每个核心 claim 必须登记 run id、commit、配置和结果路径；没有实验支持的说法删除或降级为限制说明。

## Phase 1：Git、路径与 Nsight 基础设施

- [x] 本地仓库分支和 baseline tag 可追溯。
- [x] CLI 支持 `--project-root PATH`、`--output-dir PATH`、`--run-label NAME`、`--profile-gpu-queries`、`--print-paths`。
- [x] 所有 shader、config、texture、输出文件均从 project root 或 output dir 解析。
- [x] benchmark 输出到 `results/<run-label>/frame_profile.csv`，同时生成 `run_metadata.json`。
- [x] 提供 `scripts/build_release.ps1`、`scripts/run_benchmark.ps1`、`scripts/run_nsys.ps1`、`scripts/run_ncu.ps1`。
- [x] OpenGL compute dispatch 包含可选 debug group；CSV 继续保留现有 GL timer query 字段。
- [x] Nsight Compute 若无法分析 OpenGL GLSL compute shader，脚本写入原因文件。

验收入口：

- [x] 从 `D:\GenPD_test_p\GenPD` 运行 benchmark。
- [x] 从 exe 目录运行同一 exe，并显式传 `--project-root D:\GenPD_test_p\GenPD`。
- [x] 从临时目录运行同一 exe，并显式传 project root 与 output dir。
- [x] 用 `scripts/run_nsys.ps1 -RunLabel smoke-nsys` 生成 `.nsys-rep` 或记录工具不可用/运行失败原因。

## Phase 2：内部基线与消融

运行时变体清单：

- [x] `cpu-ncg`：CPU 约束遍历 NCG。
- [x] `gpu-edge-scatter`：GPU edge/constraint scatter + atomic NCG。
- [x] `gpu-gather-no-fusion`：GPU vertex gather，但 gradient、norm、gdotd 分 kernel。
- [x] `gpu-gather-fusion`：当前主方法，vertex gather + fused gradient/stat partials。
- [x] `gpu-gather-fusion-batched-ls`：融合版本加 batched line search。
- [x] `gpu-gather-fusion-batched-ls-persistent`：进一步保留 persistent buffers。
- [x] Runtime CLI: `--solver-variant`; each run writes `solver_variant` to `run_metadata.json` and emits `frame_profile_experiment.csv` without changing the legacy CSV schema.

每个变体记录：

- [ ] 帧时间、optimization time、transfer time。
- [ ] dispatch 数、同步/读回开销。
- [ ] GPU 显存占用和主要 buffer 大小。
- [ ] `converged`、`exploded`、`gradient_norm`。

外部/相关基线候选：

- [ ] CPU NCG 原始实现。
- [ ] GPU PD 或迭代式求解方法的可复现实验。
- [ ] Subspace-Preconditioned GPU Projective Dynamics with Contact, SIGGRAPH Asia 2023: https://wanghmin.github.io/publication/li-2023-spg/
- [ ] Projective Peridynamic Modeling of Hyperelastic Membranes With Contact, TVCG 2023: https://lausr.org/dashboard/?doi=10.1109%2Ftvcg.2023.3271511

比较原则：

- [ ] 所有方法按相同误差或相同视觉质量比较，固定迭代数 FPS 只作为补充。
- [ ] 每个图表给出 run id、commit、配置和统计范围。

## Phase 3：物理质量与参考解

- [x] 增加可配置高迭代 CPU NCG 参考解导出。
- [x] 计算位置相对误差。
- [x] 计算速度相对误差。
- [x] 计算约束能量相对误差与平均/最大拉伸应变。
- [x] 记录最大外部碰撞穿透深度。
- [x] 统计失败率、NaN/Inf、exploded frame。
- [x] 输出独立 `quality_metrics.csv` 和汇总 CSV。

实现入口：

- `--iterations-per-frame N`、`--reference-export-dir PATH`、`--quality-reference-dir PATH`、`--quality-checkpoint-stride N`。
- `scripts/run_reference.ps1` 固定使用 `cpu-ncg`，默认 100 次迭代/帧，并按检查点步长写入 `reference_state_<frame>.bin`；二进制头包含 magic、版本、帧号、标量字节数和向量维度。
- 对比运行在同帧检查点存在时计算位置/速度相对 L2、约束能量相对误差、平均/最大弹簧应变和当前/参考最大外部穿透深度。`finite` 与 `exploded` 字段用于失败率。
- `scripts/summarize_quality_metrics.ps1` 输出误差、能量/应变、穿透和失败率的 count、mean、stddev、P50、P95、minimum、maximum。误差项只汇总 `has_reference=1` 的帧。
- 对 persistent GPU state，质量记录前会显式回读当前位置和速度，保证 CSV 不读取陈旧 CPU 镜像；该回读是质量测量开销，不应混入关闭质量记录的性能 profile。

当前限定：约束能量和应变由当前 `Mesh::my_edge` 的 spring/attachment 表示计算，适合当前 cloth GPU 路径；tet 材料能量和参考迭代数的收敛标定仍需在正式实验前单独验证。CPU 仍负责每帧迭代、line-search 和 dispatch 的控制流。

## Phase 4：稳定性与场景矩阵

- [x] 扫描 `dt × stiffness` 并输出稳定性 CSV：`scripts/run_stability_sweep.ps1`。
- [x] 生成稳定性热图：`scripts/plot_stability_heatmap.ps1` 生成 SVG。
- [x] 加入运动障碍物场景：`scenes/moving_sphere_cloth.xml`。
- [x] 加入不同网格分辨率：`--cloth-dimension N` 只覆盖本次运行。
- [x] 加入不同材质/刚度配置：`--stretch-stiffness FLOAT` 和 `--bending-stiffness FLOAT` 只覆盖本次运行。
- [x] 增加更多约束类型时同步说明适用范围：当前质量能量/应变仍只覆盖 `Mesh::my_edge` 的 spring/attachment 表示。
- [x] 若未实现自碰撞，论文明确限定为外部/运动障碍物接触和当前约束集合。详见 `docs/experiments/phase4-stability-and-scenes.md`。

Phase 4 的代码和 smoke run 已完成；正式论文表格仍应从固定 commit 的完整扫描结果生成，不能把 smoke run 当作论文数据。

## Phase 5：线搜索实验与 claim audit

- [ ] 扫描 batched line-search `K`。
- [ ] 扫描 Armijo `rho/beta`。
- [ ] 记录 Armijo 失败率。
- [ ] 记录 fallback 比例和 fallback 触发原因。
- [ ] 与串行 backtracking 比较质量/时间权衡。
- [ ] 加入 NCG restart 开关，或解释 FR 在多帧求解中的可靠性边界。
- [ ] 逐条审查论文 claim，确保每条对应实验记录。

## 复现记录模板

每个正式实验 run 至少登记：

- run id：
- commit：
- branch：
- executable：
- config：
- scene：
- GPU：
- driver：
- OS：
- timer range：
- warm-up frames：
- measured frames：
- metric CSV：
- metadata JSON：
- summary CSV/table：
- mean/std/P50/P95：
- failure rate：
- quality/error target：
- notes：
