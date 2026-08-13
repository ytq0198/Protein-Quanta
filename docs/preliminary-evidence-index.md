# 初赛人工撰写用证据索引（非提交稿）

> 初赛模板明确禁止使用 AI 作答。本文件只索引已经运行的实验事实、代码和图表，不能直接作为参赛答卷提交。最终表述、取舍和署名必须由团队成员独立完成。

## 历史冻结方案与当前有效状态

> **2026-08-13 新版结论：** published NeuralMD 是当前严格可复现主基线；epoch-5 只是未晋升的多指标研究候选。以下旧冻结方案使用过旧 T1 `2→18`，全部作为历史证据，不得写成当前提交候选。

- 历史方法：竞赛 T1/T2/T3 场景感知 checkpoint selection + 随时间衰减的 Static residual anchor；
- 模型：NeuralMD，seed 42，epoch 5；
- 后处理：`β=1`，decay scale 98 frames；
- 机器可读配置：`configs/frozen_candidate.json`；
- checkpoint SHA256：`0e7d5150aa5f305499f17663d3a74b1063b0591733f534e676303ce11de50b8c`；
- checkpoint 不进入 Git；已从实验运行目录复制到 manifest 所列专用归档目录，源文件与归档文件 SHA256 一致。

2026-08-13 的 bond-aware validation 显示，`beta=1` 相对未锚定 epoch-5 的 T1/T2 键长 MAE 分别恶化 8.99%/13.69%，超过预注册 5% 防线，因此该组合已 **demote/no-go**，不得再作为方向二整体候选。新版 T1 重评后，published NeuralMD 保持当前主基线；未锚定 epoch-5 只保留为非晋升研究候选。后续直接比较表明 epoch-5 改善 T3 结构分布与多数平均键长诊断，但 T1/T2 Dyn 未整体改善且 T3 极端键长事件更高；E17 位移损失也未改变动态幅度。因此当前仅能主张“多时间尺度 checkpoint selection 的可行性与明确局限”，不能主张方向二综合提升。

## 与评分维度的证据映射

| 维度 | 可核查证据 | 不应夸大的边界 |
|---|---|---|
| 技术（45%） | 官方 20 帧窗口精确复现；T1/T2/T3 独立初始化与评估；场景条件策略有严格解析/校验与本地/服务器回归；checkpoint/报告哈希；补丁栈可从 pinned upstream 应用 | 本地没有官方归一化评分器，不能报告伪总分 |
| 科学（30%） | 发布 checkpoint 与纠正复现 `<1e-3` 一致；20 帧训练与 100 帧闭环错配；Pair/zero 同 epoch 因果消融；三 seed 早停复核；冻结测试 | MISATO 每个 split 只有 10 个复合物；不能外推到全部蛋白–配体体系 |
| 创新（20%） | 用多时间尺度场景指标选择 checkpoint，并通过 E15/E16/E17 因果实验定位“表面稳定、共价几何、动态幅度”之间的冲突 | Static anchor、Pair loss、低容量门控和位移 loss 均为 no-go；不能包装成成功创新，当前有效方法仅是 checkpoint selection |
| 开源（5%） | GitHub 阶段提交、README 最短复现命令、第三方 commit/patch 注册、数据与权重不入库 | NeuralMD 仓库该 commit 缺少独立 license 文件，复用范围需谨慎描述 |

### Phys/Dyn 证据边界（必须保留）

- 已完成“距离小于共价半径之和”的项目碰撞代理审计；配体—蛋白重叠比例极低，且锚定前后没有一致升高。
- 配体内部代理在 6 个 validation/test 场景比较中有 5 个小幅升高（`+0.060` 至 `+0.230` 个百分点；冻结测试 T3 为 `-0.044`）。
- E13 后从 RCSB PDB 显式 `CONECT` 保守恢复 validation 9/10 拓扑，已能报告键长与排除 1/2-hop 的非键碰撞，但覆盖不完整、没有能量/立体化学或官方 evaluator。
- epoch-5 相对 published 的平均键长 MAE在 T1/T2/T3 改善 1.69%/2.22%/55.60%，但 T3 极端事件率更高（0.2666% vs 0.1082%），不得写成综合 Phys 提升。
- epoch-5 的 T3 Rg/原子对 W1 改善 8.48%/11.93%，但 T1/T2 Dyn 未整体改善，且模型步幅仅为真值约 1%；不得写成综合 Dyn 提升。

## 最重要的数字

以下均相对发布 checkpoint，误差类负值为改善。

| 数据 | 场景 | 坐标 RMSE | Matching | Stability | RMSF MAE |
|---|---|---:|---:|---:|---:|
| Validation | T1 | -0.65% | -1.79% | +0.39 点 | -2.68% |
| Validation | T2 | -0.87% | -1.51% | +0.15 点 | -3.34% |
| Validation | T3 | +1.54% | -7.19% | +2.17 点 | -1.17% |
| Frozen internal test | T1 | -0.11% | -2.48% | +0.51 点 | -3.51% |
| Frozen internal test | T2 | -0.35% | -2.27% | +0.42 点 | -1.60% |
| Frozen internal test | T3 | +0.36% | -4.99% | +1.61 点 | -0.36% |

### 未晋升的场景条件锚定

`T1=8, T2=8, T3=1` 在 validation 改善 T1/T2 全部四项指标；冻结测试也改善 T1/T2 的 Matching、Stability 和 RMSF，但 T1 坐标 RMSE 相对 global `β=1` 上升 2.295%，超过预注册的 2% 防线。因此它是有方向信号但未通过晋升门槛的 no-go，当前冻结方案不变。不得只引用其几何改善而省略坐标失败。

### 未晋升的不确定性门控

按复合物 LOOCV 的六维 SE(3) 不变逻辑门控在 aggregate T1/T2 指标上有改善，但 safe/useful 标签 balanced accuracy 只有 0.40，未达到预注册的 0.60；因此没有访问 test。若人工材料提到该实验，必须同时披露分类失败，不能只引用聚合改善。

## 推荐引用的仓库产物

- 总实验账：`reports/experiment-progress-report.md`；
- 科研设计：`docs/experiment-design-and-research-roadmap.md`；
- 新版长序列机制证据链：`reports/reproduction/2026-08-13-temporal-updated-guide.md`、`reports/reproduction/2026-08-13-temporal-noise-updated-guide.md`、`reports/reproduction/2026-08-13-proar-antidrift-updated-guide.md`、`reports/reproduction/2026-08-13-temporal-closed-loop-updated-guide.md`；
- 新版机制对比图：`reports/figures/temporal_mechanism_chain_updated_guide.png`；
- train-only 多时间尺度闭环：`reports/reproduction/2026-08-13-temporal-multiscale-train-only.md`；
- 一次性 validation 确认：`reports/reproduction/2026-08-13-temporal-multiscale-official-validation-confirmation.md`；
- 场景基准图：`reports/figures/neuralmd_scenario_baselines.png`；
- 最终组合图：`reports/figures/neuralmd_earlystop_anchor1_tradeoff.png`；
- 场景条件锚定 no-go：`reports/reproduction/2026-08-12-scenario-conditioned-anchor.md`、`reports/reproduction/scenario_anchor881_decision.json`、`reports/figures/neuralmd_scenario_anchor881_tradeoff.png`；
- 不确定性门控 LOOCV no-go：`reports/reproduction/2026-08-12-uncertainty-gate-loocv.md`、`reports/reproduction/uncertainty_gate_loocv_val.json`、`reports/figures/uncertainty_gate_loocv.png`；
- E15 bond-aware Phys no-go：`reports/reproduction/2026-08-13-bond-aware-phys-validation.md`；
- E16 Dyn 分布验证：`reports/reproduction/2026-08-13-dynamics-distribution-validation.md`；
- E17 位移损失 no-go：`reports/reproduction/2026-08-13-e17-displacement-feasibility.md`、`reports/reproduction/e17_displacement_gate.json`；
- 长序列架构筛选：`reports/reproduction/2026-08-13-temporal-architecture-screen.md`、`reports/reproduction/temporal_architecture_screen_val.json`、`reports/figures/temporal_architecture_screen.png`；
- E6 失败与早停/组合因果链：`reports/reproduction/2026-08-12-neuralmd-pair-loss-and-earlystop.md`；
- 最终验证/测试原始报告：`reports/reproduction/neuralmd_earlystop_anchor1_val.json`、`reports/reproduction/neuralmd_earlystop_anchor1_test.json`；
- Phys 初筛：`reports/reproduction/2026-08-12-collision-proxy-audit.md`、`reports/reproduction/neuralmd_earlystop_anchor1_collision_val.json`、`reports/reproduction/neuralmd_earlystop_anchor1_collision_test.json`；
- 竞赛场景实现：`protein_quanta/scenarios.py`、`scripts/evaluate_neuralmd_scenarios.py`；
- 最终后处理：`protein_quanta/anchoring.py`、`scripts/evaluate_anchor_scenarios.py`；
- 训练复现与补丁：`protein_quanta/neuralmd_training.py`、`third_party/patches/`。

## 人工提交前必须确认

1. 对照最新指导手册和模板再次确认截止时间、文件名、页数及上传格式；
2. 由团队成员自行决定是否把 Pair no-go 放入正文或附录；
3. 不把 proxy 指标写成官方分数，不自行拼接 T1/T2/T3 总分；
4. 不把 MISATO 内部 test 称为比赛隐藏测试；
5. 检查所有图表标题、单位、baseline 和“越高/越低越好”；
6. 确认 checkpoint 能从 manifest 所列服务器路径读取，并另外保留可恢复备份；
7. 最终文字由团队人工撰写并完成真实性、署名和合规检查。
