# 初赛人工撰写用证据索引（非提交稿）

> 初赛模板明确禁止使用 AI 作答。本文件只索引已经运行的实验事实、代码和图表，不能直接作为参赛答卷提交。最终表述、取舍和署名必须由团队成员独立完成。

## 当前冻结方案

- 方法：竞赛 T1/T2/T3 场景感知 checkpoint selection + 随时间衰减的 Static residual anchor；
- 模型：NeuralMD，seed 42，epoch 5；
- 后处理：`β=1`，decay scale 98 frames；
- 机器可读配置：`configs/frozen_candidate.json`；
- checkpoint SHA256：`0e7d5150aa5f305499f17663d3a74b1063b0591733f534e676303ce11de50b8c`；
- checkpoint 不进入 Git；已从实验运行目录复制到 manifest 所列专用归档目录，源文件与归档文件 SHA256 一致。

## 与评分维度的证据映射

| 维度 | 可核查证据 | 不应夸大的边界 |
|---|---|---|
| 技术（45%） | 官方 20 帧窗口精确复现；T1/T2/T3 独立初始化与评估；场景条件策略有严格解析/校验与本地/服务器回归；checkpoint/报告哈希；补丁栈可从 pinned upstream 应用 | 本地没有官方归一化评分器，不能报告伪总分 |
| 科学（30%） | 发布 checkpoint 与纠正复现 `<1e-3` 一致；20 帧训练与 100 帧闭环错配；Pair/zero 同 epoch 因果消融；三 seed 早停复核；冻结测试 | MISATO 每个 split 只有 10 个复合物；不能外推到全部蛋白–配体体系 |
| 创新（20%） | 把创新从单纯改网络转为“两层风险控制”：多时间尺度 checkpoint selection + Static residual uncertainty decay；验证和冻结测试方向一致 | Pair loss no-go，不能作为成功创新主张；anchor 是后处理而非新主干网络 |
| 开源（5%） | GitHub 阶段提交、README 最短复现命令、第三方 commit/patch 注册、数据与权重不入库 | NeuralMD 仓库该 commit 缺少独立 license 文件，复用范围需谨慎描述 |

### Phys 证据缺口（必须保留）

- 已完成“距离小于共价半径之和”的项目碰撞代理审计；配体—蛋白重叠比例极低，且锚定前后没有一致升高。
- 配体内部代理在 6 个 validation/test 场景比较中有 5 个小幅升高（`+0.060` 至 `+0.230` 个百分点；冻结测试 T3 为 `-0.044`）。
- 当前预处理没有共价键表，正常成键近邻也被计入，因此该指标不是化学有效的 clash rate，更不是官方 Phys 分数。
- 在补齐键长、键角、立体化学和能量/有效性检查前，不得写“物理合理性得到提升”。碰撞审计见 `reports/reproduction/2026-08-12-collision-proxy-audit.md`。

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

## 推荐引用的仓库产物

- 总实验账：`reports/experiment-progress-report.md`；
- 科研设计：`docs/experiment-design-and-research-roadmap.md`；
- 场景基准图：`reports/figures/neuralmd_scenario_baselines.png`；
- 最终组合图：`reports/figures/neuralmd_earlystop_anchor1_tradeoff.png`；
- 场景条件锚定 no-go：`reports/reproduction/2026-08-12-scenario-conditioned-anchor.md`、`reports/reproduction/scenario_anchor881_decision.json`、`reports/figures/neuralmd_scenario_anchor881_tradeoff.png`；
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
