# 速度感知 E(3) 场效果预实验：误差大幅下降但动力学仍塌缩

> 结论：**综合 gate fail，但获得首个强机制信号。** 速度感知候选相对位置-only 对照将新 12-complex diagnostic 的 T1/T2/T3 RMSE 分别降低 `33.1%/45.9%/80.5%`，然而 T3 step-amplitude ratio 只有 `0.1135`，且 `98.3%` 更新触发梯度裁剪；因此不能晋升为比赛性能候选。

## 隔离设计

本实验只使用上一轮尚未作为 diagnostic 的 48 个训练复合物，再以新 salt 确定性分为 36 train / 12 diagnostic。此前看过的 frame-time 16-complex diagnostic、旧 16-complex holdout 与官方 validation/test 均未访问。

两组统一使用帧时间参数化、相同 seed、数据顺序、局部/多尺度窗口、5 epoch、Adam `1e-4`、clip `1.0` 与 local MSE + `0.25 ×` multiscale Smooth-L1：

- 对照：8,930 参数 position-only E(3) 径向场；
- 候选：10,243 参数 velocity-aware E(3) 场，以 `|v|²`、`v·r` 调制径向消息，并加入 `−softplus(γ)v` 阻尼。

预注册综合门槛要求：候选 T3 RMSE 至少改善 5%；T1 回退不超过 10%；T3 步幅比在 `[0.2,2.0]`；候选裁剪率不超过 75%；所有更新和 rollout 有限。

## 冻结结果

| 场景 | Position-only RMSE (Å) | Velocity-aware RMSE (Å) | 候选改善 | 对照步幅比 | 候选步幅比 |
|---|---:|---:|---:|---:|---:|
| T1 | 4.73589 | 3.16650 | **33.14%** | 1.1736 | 0.6327 |
| T2 | 8.58497 | 4.64814 | **45.86%** | 0.9630 | 0.3560 |
| T3 | 32.75498 | 6.37885 | **80.53%** | 1.0497 | **0.1135** |

两组均无非有限更新或帧。T1、T3 RMSE 和有限性检查通过；T3 动态幅度与裁剪率检查失败，所以综合结论必须是 fail。

## 为什么这仍是重要的初期可行性证据

候选不是靠偶然初始化得到低误差：从自身随机初始化到训练后，T1/T2/T3 又分别改善 `10.46%/13.73%/24.04%`，而位置-only 只改善 `0.13%/0.32%/0.58%`。同时候选训练损失相对对照明显下降，说明速度通道与阻尼提供了当前模型真正缺少的动力学自由度。

但 `softplus(γ)` 无上界，候选在训练中继续压低速度：T3 步幅从初始化的 `0.1681` 降至 `0.1135`。这会通过“过度刹车”获取坐标 RMSE 收益，尚不能恢复真实动态分布。180 次候选更新有 177 次被裁剪；高达 `3567.6` 的最大梯度也说明输入和阻尼尺度需要显式归一化。

## 下一创新突破的具体构思

下一候选不放宽阈值，而是做结构性约束：

1. 将阻尼从无界 `softplus` 改为有界 `γ_max·sigmoid(raw)`，并把初始 bias 设为小正值，避免随机初始化已经大幅刹车；
2. 对 `|v|²` 与 `v·r` 使用按训练集统计冻结的尺度或解析饱和变换，降低 4–5 个数量级的梯度尺度差；
3. 分离保守径向力和耗散速度力，记录两者范数与功率 `a_damp·v≤0`，防止模型靠不可解释的任意速度项取巧；
4. 在新的、尚未看过的 development 子集上同时要求 RMSE、step amplitude、裁剪率与键长安全，不能只优化坐标误差。

当前 64 development 已被多轮机制实验逐步消耗，继续拆分会造成统计功效过低。更可靠的下一步应等待完整 MISATO 验收后，在官方 train 大样本上按蛋白同源/配体 scaffold 分组建立全新 development split；在此之前只实现 bounded-damping correctness 和合成稳定性测试，不继续看现有 diagnostic。

## 证据

- 预注册：`configs/velocity_equivariant_effect_preflight.json`
- 原始报告：`reports/reproduction/evidence/velocity-equivariant-effect-preflight.json`
- 原始报告 SHA-256：`31d7714af29c0bc0f78f568d43ebe17289ea7357246cceec01dc4ee309970ad3`
- 运行日志 SHA-256：`244f19ef6b974f379b7d254a71df037142a18c8849ef4ef11fb47201a78cf3ee`
- checkpoints 不进入 Git；其 SHA-256 记录于原始报告。
