# 当前有效证据索引（2026-08-14）

> 供团队人工撰写初赛材料时核对事实。模板明确禁止 AI 作答，因此本文件不是参赛答卷，不提供可直接提交的方案表述。

## 当前结论

- **复现主基线**：published NeuralMD。它已按新版 T1 10→10、T2 80→20、T3 20→80 重新评估；没有被当前研究候选全面超过。
- **最强正机制证据**：12 维时序 proxy 上，`[5,10,20,40]` 多时间尺度闭环在 train-only 64/16 与一次冻结 validation 确认中改善长程 T3；这证明“训练 horizon 覆盖”值得迁移，但不是三维模型成绩。
- **三维正确性证据**：最小 dense E(3) 等变径向场通过 5/5 correctness gate，共同刚体误差 `8.58×10⁻⁶ Å`。
- **三维效果结论**：64/16 三 seed 多尺度效应 gate **no-go**。新增损失改变参数，但宏 RMSE 相对恶化 `3.42×10⁻⁹`，0/3 seed 胜出；安全项通过，效果项失败。
- **当前科学瓶颈**：最小等变场 step-amplitude ratio 约 `0.009`，即只恢复真值约 1% 的运动，属于近静态解。下一阶段应先恢复动力学振幅，再谈长程抗漂移。

## 可引用证据与边界

| 目的 | 首选证据 | 必须同时披露的边界 |
|---|---|---|
| 说明题目口径 | `docs/2026-08-13-updated-manual-and-template-audit.md` | 新版未公开 T1/T2/T3 精确权重，不拼接官方总分 |
| 说明基线可复现 | `reports/reproduction/2026-08-13-neuralmd-updated-guide-validation.md` | MISATO-100，不是完整 MISATO |
| 说明长序列机制 | `reports/reproduction/2026-08-13-temporal-multiscale-official-validation-confirmation.md` | 12 维 proxy，不是原子轨迹模型 |
| 说明严格等变实现 | `reports/reproduction/2026-08-14-dense-equivariant-correctness-pilot.md` | correctness 通过不等于预测有效 |
| 说明正式三维消融 | `reports/reproduction/2026-08-14-dense-equivariant-effect-gate.md` | train-only 64/16、样本互斥但非同源/scaffold 分组、结论为 no-go |
| 说明全过程审计 | `reports/reproduction/2026-08-14-dense-effect-gate-live-audit.md` | 首次评估有序列化失败，已限定热修复并独立复算 |
| 说明原始数字 | `reports/reproduction/evidence/dense-equivariant-effect-summary.json` | 内部项目代理，不是官方分数 |

## 不再用于主论证的历史结果

- 所有旧 T1 `2→18` 结果、旧版权重合成和由此产生的候选排序；
- anchor、scenario-conditioned anchor、uncertainty gate、位移损失等已判 no-go 的方案；
- 只在 validation/test 上表现局部改善但未通过综合门槛的 epoch-5 研究候选；
- 未完成许可证、训练 ID、同源和配体泄漏审计的 PVB/EPT/DPLM/ProTDyn 权重迁移。

## 下一执行 gate

1. 完整 MISATO 文件达到 `132,841,014,019` 字节且 `.aria2` 消失后，核对官方 MD5 `9bc6446922cd80e0f2f3f69349bf88ed`；再做 HDF5、去多肽 split 和有限坐标审计。
2. 仅用 development 设计 step-amplitude recovery gate，同时要求 E(3)、finite、bond safety；未显著高于当前约 1% 时不访问新 holdout。
3. 模型升级优先显式速度/时间、局部多层高阶几何与合规结构预训练；不在已解封 64/16 holdout 上扫损失系数。
4. 团队人工确认官网具体截止时刻、ZIP 命名/大小和上传字段；团队自行完成模板文字、真实性核验与签字。
