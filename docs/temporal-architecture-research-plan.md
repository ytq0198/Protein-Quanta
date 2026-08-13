# 长序列时序架构比较计划

> 版本：v1.0（2026-08-13）  
> 性质：内部科研与实验计划，不是初赛答卷；最终参赛材料必须由团队独立撰写。

> **新版覆盖：** T1 已修正为观察 10 帧、预测 10 帧；T2/T3 为 `80→20/20→80`。新版未公布场景权重。本文以下涉及 `2→18` 或 50% 的文字只属于旧协议研究动机，当前结果与决策以 `reports/reproduction/2026-08-13-temporal-updated-guide.md` 为准。

## 1. 为什么值得做，但不能直接把坐标送入 RNN/Transformer

方向二同时包含三种历史/预测长度：T1 观察 10 帧并预测 10 帧，T2 观察 80 帧并预测 20 帧，T3 观察 20 帧并预测 80 帧。因此长记忆结构主要可能在 T2 的历史压缩和 T3 的长程误差控制中发挥作用，不能假设它天然改善局部 T1。当前场景权重未知。

直接对绝对三维坐标训练普通时序网络存在三个混杂因素：不同配体原子数不一致；旋转/平移会被错误学习；自回归坐标可能破坏共价几何。公平比较应保持空间几何建模不变，只替换 temporal core。

## 2. 两级验证

### Level A：不变量时序筛选

每帧提取 12 维刚体运动不变量：Rg、原子对距离 10/25/50/75/90% 分位数、距离均值/标准差、一步位移幅度均值/标准差/50/90% 分位数。使用 train split 训练，validation split 比较：

- last-observation Static；
- 无记忆 MLP；
- vanilla RNN；
- LSTM；
- GRU；
- causal Transformer。

所有神经模型少于 10k 参数，使用相同 epoch、batch、优化器、随机种子和更新次数。主要指标为按比赛 T1/T2/T3 权重计算的 standardized rollout RMSE；同时分别报告结构特征与步长特征误差。

Level A 只能回答“历史记忆结构是否对 MISATO 动态摘要有预测价值”，不能作为比赛 Geo/Phys/Dyn/Stab 成绩。

### Level B：等变三维小试验

仅当某个记忆模型同时满足以下条件才进入：

1. 加权 rollout RMSE 优于 MLP；
2. 至少两个场景优于 Static；
3. T2 或 T3 至少一个相对 MLP 改善 5%；
4. 全部 rollout 有限且无灾难性误差。

Level B 共享 BindingNet 或等变图空间编码器，把标量/向量表示沿时间送入获胜的 recurrent core 和 Transformer，输出速度/位移而非绝对坐标。此时才运行完整 Geo/Phys/Dyn/Stab validation gate。RNN、LSTM、GRU 中只选择 Level A 最优者，避免重复大预算训练。

## 3. 科研假设

- 若 LSTM/GRU 在 T2 明显改善而 T1 无变化，说明长历史压缩有价值，但不能单独成为全场景方案。
- 若 Transformer 只在训练 teacher-forcing 有优势而 rollout 退化，核心问题仍是 exposure bias，不是注意力容量。
- 若所有记忆模型都不优于 MLP/Static，说明现有 100 帧轨迹的时间分辨率或 80 个训练复合物不足以辨识长记忆；下一步应优先改状态表示、时间降采样或随机动力学建模。
- 若 SSM 后续加入，使用 S4D/轻量状态空间模型作为长序列扩展；它不是本轮首批模型，避免额外 CUDA kernel/优化器差异破坏公平性。

## 4. 文献与开源参考

- NeuralMD：固定同一 BindingNet 骨干比较不同轨迹生成范式，说明空间骨干一致是公平对照的关键。
- Tsai 等：LSTM 可在低维随机分子轨迹上恢复多时间尺度统计。
- Zeng 等：高维分子轨迹中，时间分辨率和状态划分可能比 LSTM/Transformer 架构差异更重要，Transformer 未必优于 LSTM。
- EST/ESTAG：更相关的 3D 路线是 E(3) 等变时空图注意力，而不是普通坐标 Transformer。
- S4：适合长序列且有官方开源实现，但依赖和优化设置与普通 PyTorch core 差异较大，放到第二批比较。

## 5. 执行边界

- 不覆盖当前 seed-42 epoch-5 安全基线；
- 不访问 internal test 做架构选择；
- 不把 Level A 指标写成方向二官方 Dyn 分数；
- Level A 若无模型晋升，不启动五套 3D 大模型；
- 初赛截止前若时间不足，只报告已完成的小预算架构证据与后续等变路线，不宣称尚未运行的结果。
