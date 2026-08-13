# 多时间尺度闭环训练：train-only 完整 gate 通过

## 设计与防泄漏协议

本实验在结果产生前冻结。MISATO-100 官方 train 的 80 个复合物通过 salted SHA256 确定性划分为 64 个 development 与 16 个 holdout；官方 validation 和 internal test 均未读取。该划分样本互斥、可复现，但尚未做到蛋白同源和配体 scaffold 分组，因此只能作为小规模机制证据。

普通 causal Transformer 与多时间尺度闭环候选使用相同 64/16 数据、模型、200 epoch、final epoch、seeds `0/42/123` 和优化器。候选在一步 MSE 外加入权重 `0.25` 的可微闭环损失，每个 batch 从 horizons `[5,10,20,40]` 均匀抽取一个，并在合法区间抽取 prefix。没有按 holdout 选择 epoch。

## 结果

| 指标 | 一步训练 Transformer | 多时间尺度闭环 | 相对变化 |
|---|---:|---:|---:|
| 等权宏平均 RMSE | 0.60415 ± 0.00777 | **0.57935 ± 0.02971** | **-4.10%** |
| T1 RMSE | 0.63052 | **0.62161** | **-1.41%** |
| T2 RMSE | 0.48104 | **0.47459** | **-1.34%** |
| T3 RMSE | 0.70088 | **0.64186** | **-8.42%** |
| 配对 seed 胜数 | — | 2/3 | 要求 ≥2/3 |

T3 的逐 seed 配对差为 `-0.05990/-0.05647/-0.06070`，三个种子一致改善，且超过预注册的 5% 长程门槛。T1 防线、总体均值、种子胜数与有限值也全部通过，因此完整 gate 为 **pass**。

## 科研意义与严格边界

固定 10 步闭环在官方 validation proxy 上主要修复 T2、未明显修复 T3；本实验加入更长的 20/40 步训练暴露后，在完全 train-only 的新 holdout 上取得一致 T3 收益。这支持“训练 horizon 必须覆盖目标误差时间尺度”的因果假设，比继续扫一个损失系数更有机制意义。

但这仍不是可提交的三维模型：输入是 12 维不变量，未显式预测配体原子坐标，也没有直接优化 Geo/Phys/Dyn/Stab。下一步只能做一次预注册的官方 validation 确认；若不能复现 train-only 方向，则停止，不据 validation 回调 horizons 或权重。只有确认后才值得设计等变原子坐标迁移。

## 可复核产物

- 预注册：`configs/temporal_multiscale_closed_loop_preregistration.json`
- 划分清单：`temporal_train_only_split_manifest.json`
- 原始逐 epoch/seed 数据：`temporal_multiscale_train_only_val.json`
- 实现：`scripts/train_temporal_multiscale_internal.py`
- `official_validation_accessed=false`、`official_test_accessed=false` 已写入原始报告。
