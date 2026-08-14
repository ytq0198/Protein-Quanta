# 完整 train 新 ID：bounded + normalized velocity gate no-go

## 结论

在完整 MISATO filtered train 中选取仓库历史未使用的 32-train/8-diagnostic ID，按预注册同 schedule 对照 unbounded velocity-aware 与 bounded+normalized 候选。候选虽把 T3 step-amplitude 从 control 的 0.0763 提升至 0.6014，但 T3 RMSE 从 6.239 Å 恶化至 23.586 Å，T1/T2 同样恶化，且候选裁剪率 85.42% 超过 75% 门槛。完整 gate 判定 **fail**。

这是一条有机制意义的负结果：bounded 小阻尼解决了“几乎不动”，却产生了过强、方向或相位不准确的运动；当前实现不能把“幅度恢复”转化为“轨迹准确”。固定 `bounded_damping_max=0.05`、`initial_fraction=0.05` 与当前 normalization 组合冻结，不事后扫参。

## 数据与防泄漏

- 数据源：官方 train 13,066（排除 NeuralMD peptides）。
- 先从 tracked configs/reports 中恢复已出现的 official train ID，共排除 120 个。
- 对剩余 12,946 个 ID 按 `SHA256(salt:id)` 排序，固定前 32/后 8 为 train/diagnostic。
- 选择序列 SHA256：`60ccd79d26a64450152ba6e43424cb6658092d1ecb5360a8cf64cb9bff35a152`。
- official validation 与 public test 均未访问；diagnostic 不用于选择 epoch，固定 final epoch。

## 训练协议

- seed `20260814`，3 epochs，96 updates/model；
- frame-time `scaling=1`、`initial_velocity_scale=1`、Euler step 1；
- local horizon 最大 20，加 `[5,10,20,40]` multiscale Smooth-L1，系数 0.25；
- learning rate `1e-4`，gradient clip 1.0；
- 两模型所有非 damping 权重相同；候选 damping head 使用预注册的小 bounded 初始化。

## 结果

### 训练前 diagnostic

| 模型 | T1 RMSE | T2 RMSE | T3 RMSE | T3 amplitude |
|---|---:|---:|---:|---:|
| control | 7.241 | 6.715 | 7.079 | 0.0990 |
| candidate | 8.469 | 10.445 | 24.055 | 0.6185 |

候选在训练前已经表现为“更会动、但轨迹偏差更大”。因此最终对照不能解释为训练才造成 overshoot；这是该机制初始化/参数化的直接行为。

### final epoch diagnostic

| 模型 | T1 RMSE | T2 RMSE | T3 RMSE | T1 amplitude | T2 amplitude | T3 amplitude |
|---|---:|---:|---:|---:|---:|---:|
| control | **6.964** | **6.186** | **6.239** | 0.702 | 0.403 | 0.0763 |
| candidate | 8.460 | 10.409 | 23.586 | 1.163 | 0.961 | **0.6014** |

候选相对 control：

- T1 RMSE 恶化 21.49%；
- T2 RMSE 恶化 68.26%；
- T3 RMSE 恶化 278.06%；
- T3 amplitude 进入预注册 `[0.2, 2.0]`，但不能抵消坐标失败。

### 优化稳定性

| 模型 | clipped / updates | clipping rate | epoch gradient mean | epoch gradient max |
|---|---:|---:|---|---|
| control | 92/96 | 95.83% | 322.77 / 98.12 / 435.88 | 7,223.9 / 2,406.3 / 11,675.2 |
| candidate | 82/96 | 85.42% | 4.44 / 5.63 / 7.67 | 16.63 / 16.00 / 29.18 |

bounded 候选显著压低梯度尺度，但仍高频触发 clip。两模型 loss/梯度/rollout 均 finite，无 NaN/Inf。

## Gate 判定

| 子门 | 结果 |
|---|---:|
| 全 updates/rollouts finite | 通过 |
| candidate T3 RMSE 至少改善 5% | **失败** |
| candidate T1 恶化不超过 10% | **失败** |
| candidate T3 amplitude 在 `[0.2,2.0]` | 通过 |
| candidate clipping rate ≤75% | **失败** |

整体：**fail**。

## 科研决策与下一创新构思

当前证据否定的是“仅靠固定小 bounded damping + invariant normalization 即可同时修复长程幅度和坐标准确性”，不是 velocity-aware 架构本身。

下一构思转向**幅度校准的残差动力学**，而不是继续扫 `damping_max`：

1. 保留 control 的强阻尼轨迹作为稳定锚点；
2. 候选只预测一个 E(3)-equivariant residual acceleration；
3. 用推理时可观测的 invariant state 产生 `[0,1]` gate，初始严格接近 0，使初始行为匹配 control，而非从 ballistic 候选起步；
4. 训练目标同时约束坐标 Smooth-L1、step-amplitude log-ratio 和 residual energy，避免只追求“会动”；
5. 先做零初始化一致性、等变性和 8-train/4-diagnostic 小门；只有 T3 RMSE 与 amplitude 同时改善才扩大。

该方向的可证伪点明确：若 gate 长期贴 0，则 residual 没贡献；若 amplitude 改善但 RMSE 再次恶化，则“稳定锚点 + gated residual”也 no-go。暂不访问 validation/test。

机器结果：`reports/reproduction/full_train_bounded_velocity_gate.json`；预注册：`configs/full_train_bounded_velocity_gate.json`。
