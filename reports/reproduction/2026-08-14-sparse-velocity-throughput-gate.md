# Protein Top-k 稀疏化稳态吞吐门：no-go

## 结论

在实验前冻结的同权重、同样本、warm-up 后成对基准中，Top-k 16/32/64/128 均未达到 2× 加速门槛，且全部慢于 dense。稀疏 gate 判定 **fail**，不进入 32-complex learnability gate。

更重要的是，dense 热身后前向/反向仅需 9.29 ms，证明前一份 smoke 的 7.93 s 主要是 CUDA 冷启动/首次 kernel 开销。此前据 7.93 s 外推的“约 28.8 小时/epoch”无效，现已在进度报告和原 smoke 报告中明确撤回。

## 预注册协议

- 数据：完整 MISATO 官方 train 排除 peptides 后的首样本 `5WIJ`；不访问 validation/test。
- 模型：bounded + normalized velocity-aware，所有候选与 dense 使用完全相同权重。
- 候选：按同一复合物内 C-alpha 欧氏距离选 Top-k 16/32/64/128。
- 计时：3 次 warm-up，随后 10 次 forward/backward；每段前后同步 CUDA。
- 通过门：相对 dense 至少 2×；相对 acceleration L2 误差不超过 2%；loss/梯度 finite nonzero。

## 结果

| 模型 | 秒/前反向 | dense speedup | 相对 acceleration L2 误差 | 峰值显存 | 通过 |
|---|---:|---:|---:|---:|---:|
| dense | **0.009288** | 1.000× | 0 | 23.76 MB | 对照 |
| Top-k 16 | 0.010741 | 0.865× | 8.866% | 19.46 MB | 否 |
| Top-k 32 | 0.013230 | 0.702× | 4.659% | 19.86 MB | 否 |
| Top-k 64 | 0.013028 | 0.713× | 2.137% | 20.60 MB | 否 |
| Top-k 128 | 0.013322 | 0.697× | **0.744%** | 22.04 MB | 否 |

全部模型 loss 和梯度均 finite nonzero。Top-k 128 满足近似误差门，但速度更慢；Top-k 16 显存减少约 18%，但误差和速度同时失败。没有候选满足全门。

## 科研解释

该样本只有 31 个配体重原子和 274 个蛋白残基，dense pair tensor 并不大。Top-k 路径额外引入 `cdist`、mask、排序和 gather；在这个尺度上，减少 MLP pair 数不足以抵消邻居选择开销。由于 274 不是极端大蛋白，仅凭此单样本不能断言所有体系都如此，但预注册 gate 已失败，不能事后更换大体系挑出有利结论。

## 决策

1. 冻结 Top-k 路线为 no-go，不扫描 k、半径或阈值。
2. 保留 dense bounded+normalized velocity-aware 作为下一正式候选。
3. 下一阶段不再用首个 CUDA 运行估算吞吐；所有训练预算使用 warm-up 后、同步 CUDA 的测量。
4. 下一实验为完整 train 内部新 ID 的 32-train/8-diagnostic 单 seed paired gate：unbounded velocity-aware 对照 vs bounded+normalized 候选，保持同初始化、同 schedule、同窗口；仍不访问 validation/test。

机器可读结果见 `reports/reproduction/sparse_velocity_throughput_gate.json`；冻结配置见 `configs/sparse_velocity_throughput_gate.json`。
