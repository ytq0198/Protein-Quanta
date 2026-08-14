# 当前有效证据索引（2026-08-14）

> 供团队人工撰写初赛材料时核对事实。模板明确禁止 AI 作答，因此本文件不是参赛答卷，不提供可直接提交的方案表述。

## 当前结论

- **复现主基线**：published NeuralMD。它已按新版 T1 10→10、T2 80→20、T3 20→80 重新评估；没有被当前研究候选全面超过。
- **最强正机制证据**：12 维时序 proxy 上，`[5,10,20,40]` 多时间尺度闭环在 train-only 64/16 与一次冻结 validation 确认中改善长程 T3；这证明“训练 horizon 覆盖”值得迁移，但不是三维模型成绩。
- **三维正确性证据**：最小 dense E(3) 等变径向场通过 5/5 correctness gate，共同刚体误差 `8.58×10⁻⁶ Å`。
- **三维效果结论（范围已收窄）**：64/16 三 seed 多尺度效应 gate 在“ODE 时间 `/100`、初始帧差 `×1`”协议下 **no-go**。新增损失改变参数，但宏 RMSE 相对恶化 `3.42×10⁻⁹`，0/3 seed 胜出；安全项通过，效果项失败。该结果不能用于否定单位一致协议下的多尺度监督。
- **当前科学瓶颈**：最小等变场 step-amplitude ratio 约 `0.009`。解析审计发现，时间 `/100` 而初始帧差不换算会使零加速度第一步位移正好只剩 `1%`；因此近静态既可能反映模型能力，也受数值单位约定强烈影响。下一阶段先做 development-only 单位一致性 gate，再谈长程抗漂移。
- **单位一致性 gate**：64 个 development 复合物的零加速度解析对照已通过。`initial_velocity_scale=100` 将第二观测帧 RMSE 降至 `3.6–5.6×10⁻⁸ Å`，三场景步幅比恢复到 `0.916–1.028`；但恒速 T3 RMSE 达 `33.88 Å`，所以这只是数值正确性证据，不是性能提升。
- **帧时间 learnability gate**：48/16 development-only 配对预实验 **fail**。帧时间候选保持步幅 `0.944` 且有限，但 5 epoch 后 T3 仅由 `34.1267` 降至 `33.9024 Å`（改善 `0.6575%`，门槛 10%）；240 次更新中 124 次裁剪。当前位置-only 径向场不能晋升。
- **速度感知正确性 gate**：10,243 参数速度感知 E(3) 场在一个真实 train 复合物上 7/7 通过；旋转/反射误差 `3.81×10⁻⁵/3.05×10⁻⁵ Å`，速度敏感度范数 `0.568`。这只授权新的 development-only 可学习性实验，不是成绩证据。
- **速度感知效果 gate**：36/12 新 development 内部划分上，候选将 T1/T2/T3 RMSE 相对位置-only 降低 `33.1%/45.9%/80.5%`，是首个强效果信号；但 T3 步幅仅 `0.1135`、裁剪率 `98.3%`，综合 gate **fail**。必须做有界阻尼和数值归一化，不能以低 RMSE 掩盖过度刹车。

## 可引用证据与边界

| 目的 | 首选证据 | 必须同时披露的边界 |
|---|---|---|
| 说明题目口径 | `docs/2026-08-13-updated-manual-and-template-audit.md` | 新版未公开 T1/T2/T3 精确权重，不拼接官方总分 |
| 说明基线可复现 | `reports/reproduction/2026-08-13-neuralmd-updated-guide-validation.md` | MISATO-100，不是完整 MISATO |
| 说明长序列机制 | `reports/reproduction/2026-08-13-temporal-multiscale-official-validation-confirmation.md` | 12 维 proxy，不是原子轨迹模型 |
| 说明严格等变实现 | `reports/reproduction/2026-08-14-dense-equivariant-correctness-pilot.md` | correctness 通过不等于预测有效 |
| 说明正式三维消融 | `reports/reproduction/2026-08-14-dense-equivariant-effect-gate.md` | train-only 64/16、样本互斥但非同源/scaffold 分组；no-go 仅适用于时间 `/100`、初速度 `×1` 协议 |
| 说明全过程审计 | `reports/reproduction/2026-08-14-dense-effect-gate-live-audit.md` | 首次评估有序列化失败，已限定热修复并独立复算 |
| 说明近静态机制与单位修正 | `reports/reproduction/2026-08-14-velocity-time-consistency-audit.md` | 零加速度解析控制；恢复步幅但长程误差恶化，不是学习模型成绩 |
| 说明单位一致模型可学习性 | `reports/reproduction/2026-08-14-frame-time-learnability-preflight.md` | 48/16 development-only；gate fail，不访问旧 holdout/官方 validation |
| 说明速度感知严格等变实现 | `reports/reproduction/2026-08-14-velocity-equivariant-correctness.md` | 单真实 train 复合物、随机初始化；correctness 通过不等于效果提升 |
| 说明速度感知初期效果 | `reports/reproduction/2026-08-14-velocity-equivariant-effect-preflight.md` | RMSE 大幅改善但幅度/裁剪 gate 失败；不是可晋升候选 |
| 说明原始数字 | `reports/reproduction/evidence/dense-equivariant-effect-summary.json` | 内部项目代理，不是官方分数 |

## 不再用于主论证的历史结果

- 所有旧 T1 `2→18` 结果、旧版权重合成和由此产生的候选排序；
- anchor、scenario-conditioned anchor、uncertainty gate、位移损失等已判 no-go 的方案；
- 只在 validation/test 上表现局部改善但未通过综合门槛的 epoch-5 研究候选；
- 未完成许可证、训练 ID、同源和配体泄漏审计的 PVB/EPT/DPLM/ProTDyn 权重迁移。

## 下一执行 gate

1. 完整 MISATO 文件达到 `132,841,014,019` 字节且 `.aria2` 消失后，核对官方 MD5 `9bc6446922cd80e0f2f3f69349bf88ed`；再做 HDF5、去多肽 split 和有限坐标审计。
2. 速度感知效果已有强信号但综合 gate 失败；先实现有界阻尼、速度不变量归一化与耗散功率诊断的 correctness/stability 测试。现有 64 development 已多次拆分，不再继续用其 diagnostic 调参；等待完整 MISATO 后建立按同源/scaffold 分组的新 split。
3. 模型升级优先显式速度/时间、局部多层高阶几何与合规结构预训练；不在已解封 64/16 holdout 上扫损失系数。
4. 团队人工确认官网具体截止时刻、ZIP 命名/大小和上传字段；团队自行完成模板文字、真实性核验与签字。
# 完整数据状态更新（2026-08-14）

- `MD.hdf5` 精确字节数：`132,841,014,019`；MD5 `9bc6446922cd80e0f2f3f69349bf88ed` 与公开值一致；HDF5 可打开，包含 16,972 个复合物。
- 16,972 个 group ID 与官方三份原始 split ID 并集逐 ID 完全相等。
- 多肽过滤后 `13,066/1,357/1,357` 全部存在于 HDF5；确定性分层抽样 48/48 schema 与 finite 检查通过。
- 此证据解锁完整数据的训练准备，不解锁 public test 调参；同源泄漏和真实 ligand scaffold 审计仍分别列为待完成项。
- 机器证据：`reports/reproduction/full_misato_acceptance.json`；解释报告：`reports/reproduction/2026-08-14-full-misato-acceptance.md`。

## 完整数据训练接口

- 完整 train 的流式数据入口计数为 13,066，首个过滤样本已在服务器 GPU 0 完成 bounded + normalized velocity-aware 单步前向/反向；梯度 finite nonzero。
- 首次前反向 7.934 s 是 CUDA 冷启动，后续同步稳态为 0.009288 s；此前据冷启动做的 28.8 小时/epoch 外推已撤回。
- Top-k 16/32/64/128 全部比 dense 慢，预注册 sparse gate no-go；保留 dense bounded+normalized 候选。validation/test 仍未访问。
- 证据：`reports/reproduction/full_misato_stream_smoke.json` 与 `reports/reproduction/2026-08-14-full-misato-streaming-smoke.md`。
