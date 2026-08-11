# NeuralMD 稳定训练 E3：全局梯度裁剪 1.0

## 结论

本实验是严格单变量消融：相对 seed 42 的“从首帧开始、随机最长 99 帧”压力训练，只增加全局 L2 梯度裁剪 `1.0`、裁剪前梯度范数日志和非有限更新保护。实验完整运行，但验证集结果未达到预注册条件，判定 **no-go**。因此未使用统一评估器新增测试集结果，也不继续扫描裁剪阈值。后续复核确认该对照组遗漏官方 `--no_NeuralMD_Binding_start_with_first_frame`，所以本报告保留为长跨度压力实验，不代表官方 20 帧训练配置。

## 配置与环境

| 项目 | 值 |
|---|---|
| 数据 | MISATO-100，官方 80/10/10 划分 |
| 训练/决策 | train 80；val 10；test 不参与本实验决策 |
| GPU | NVIDIA RTX A6000，GPU 0 |
| PyTorch / wall time | `2.6.0+cu124` / 约 6 分 54 秒（日志创建至完成） |
| seed / epochs | 42 / 100 |
| batch / workers | 8 / 0 |
| 优化器 | Adam，学习率 `1e-4` |
| 训练跨度 | 从首帧开始，随机终点 1–99（压力设定；非官方 20 帧窗口） |
| ODE | Euler，step 5，scaling 100 |
| 模型 | NeuralMD Binding01，100 radial bases，关闭 velocity refinement |
| 基础损失 | 位置 MSE |
| 新增项 | global L2 clip `1.0`；跳过并统计非有限 loss/gradient |
| 上游 NeuralMD | commit `a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe` |
| 训练输出 | `/mnt/localDisk3/weizian/runs/protein-quanta/stability/neuralmd-seed42-clip1/` |

完整启动参数由 `scripts/run_neuralmd_stable_training.sh` 固化。数据、原始日志和权重不进入 Git。

## 预检

1 epoch 预检成功：10 个训练 batch 全部应用，裁剪 0 次，非有限跳过 0 次；mean/max pre-clip norm 为 `0.002605/0.009852`。`model.pth` 与 `model_final.pth` 均成功保存。

## 100 epoch 训练诊断

| 诊断 | 数值 |
|---|---:|
| 完成 epoch | 100 |
| 至少一次裁剪的 epoch | 68 |
| 裁剪 batch 总数 | 408 |
| 非有限跳过总数 | 0 |
| 最大 epoch 平均位置损失 | 3244.37839（epoch 47） |
| 最大单 batch 裁剪前梯度范数 | 403,650,528（epoch 47） |
| epoch 48 最大梯度范数 | 13,917,789 |
| 上游 best checkpoint | epoch 35（按验证 coordinate MAE） |
| 最高日志验证 Stability | 81.09654%（epoch 5） |

epoch 47 的极端事件发生后，梯度范数长期维持高波动，后半程多数 epoch 都有裁剪。虽然没有 NaN/Inf，验证 Stability 到 epoch 100 仍降至 45.93%（上游日志口径）。

## 统一验证集比较

统一评估协议：每个复合物用前 2 帧初始化，Euler 100 帧推演，评价后 98 帧；10 个复合物先逐样本计算、再无权平均。以下是复现实验代理诊断，不是比赛官方评分。

| 权重 | Coord RMSE（Å，↓） | Matching（Å，↓） | Stability（%，↑） | Aligned RMSD（Å，↓） | Rg MAE（Å，↓） | RMSF MAE（Å，↓） | Contact（↑） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 未裁剪 best | 2.4683 | 0.6743 | 66.64 | 0.9669 | 0.2317 | 2.2242 | 0.9407 |
| clip-1 best | 2.4714 | 0.7638 | 59.69 | 1.0036 | 0.3079 | 2.1766 | 0.9286 |
| 未裁剪 final | 2.6924 | 1.2298 | 52.27 | 1.6365 | 0.3254 | 2.0266 | 0.8794 |
| clip-1 final | 2.9152 | 1.4080 | 45.52 | 1.7112 | 0.4141 | 1.5954 | 0.8688 |

预注册条件为：clip-1 final 相对未裁剪 final 的 Stability 至少提高 10 个百分点，且坐标 RMSE 恶化不超过 2%。实际为 Stability **下降 6.75 个百分点**、RMSE **恶化约 8.3%**。best 对比也未改善：Stability 下降 6.95 个百分点，Matching 恶化约 13.3%。

## 科研解释与决策

1. 在长跨度压力训练中梯度爆炸客观存在，监控与防护代码有保留价值。
2. 固定阈值裁剪只能限制更新长度，不能消除不同随机跨度目标的高方差，也没有直接约束配体内部距离或长程滚动稳定性。
3. 上游用 coordinate MAE 选出的 epoch 35 并不是 Stability 最佳点，模型选择目标也需要与比赛强调的几何/动力学质量对齐。
4. 停止该压力设定下的阈值扫描和三 seed；先纠正官方窗口采样并重新建立 baseline，再研究 `L_pair` 与长程目标。只有单 seed 验证通过后才进行多 seed。
5. C1 固定 Static-anchor residual 已在冻结测试集取得正结果，继续作为当前初赛候选；E3 仅作为严谨负消融写入报告。

## 权重与产物

- clip-1 best：SHA256 `2f33bb71231b935a427fdd8861ac3baf3d181ff57b0c624f1c25572ed3da83e6`
- clip-1 final：SHA256 `98b5ea5d004b814543916dc435e53843bcd0206b92f61d932a2263875a43bd28`
- 未裁剪 final：SHA256 `9222dcd8ca33a6a7b6b5985234830324e67ca5c53fef0cc30ea8fc7a9fef08aa`
- JSON：`neuralmd_misato100_clip1_best_val.json`、`neuralmd_misato100_clip1_final_val.json`、`neuralmd_misato100_retrained_final_val.json`
- 图：`../figures/neuralmd_clip1_training_comparison.png`
