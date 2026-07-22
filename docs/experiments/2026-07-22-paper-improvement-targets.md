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

- [ ] 增加高精度参考解导出。
- [ ] 计算位置相对误差。
- [ ] 计算速度相对误差。
- [ ] 计算能量或应变误差。
- [ ] 记录最大穿透深度。
- [ ] 统计失败率、NaN/Inf、exploded frame。
- [ ] 输出 `frame_profile_extended.csv` 或独立质量指标 CSV。

## Phase 4：稳定性与场景矩阵

- [ ] 扫描 `dt × stiffness` 并输出稳定性 CSV。
- [ ] 生成稳定性热图。
- [ ] 加入运动障碍物场景。
- [ ] 加入不同网格分辨率。
- [ ] 加入不同材质/刚度配置。
- [ ] 增加更多约束类型时同步说明适用范围。
- [ ] 若未实现自碰撞，论文明确限定为外部/运动障碍物接触和当前约束集合。

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
