# 帧时间参数化可学习性预实验：恢复振幅后仍未学会抗漂移

> 结论：**learnability gate fail。** 帧时间参数化解决了 `1%` 步幅塌缩，并产生显著梯度，但当前仅依赖位置的 8,930 参数等变径向场在 5 epoch 后只使 T3 RMSE 改善 `0.6575%`，未达到预注册的 `10%`；本实验族停止，不据诊断集调学习率或阈值。

## 严格实验设计

- 范围：原冻结 64-complex development 再以 salted SHA-256 分成 48 train / 16 diagnostic；不复用之前已解封的 16-complex holdout，不访问官方 validation/test。
- 配对变量仅为时间参数化：
  - 上游对照：`scaling=100, initial_velocity_scale=1, Euler step=0.025`；
  - 帧时间候选：`scaling=1, initial_velocity_scale=1, Euler step=1`。
- 两组使用相同模型初始化、seed 42、样本顺序、局部窗口、多尺度窗口、5 epoch、Adam `1e-4`、梯度裁剪 `1.0`，目标均为 local MSE + `0.25 ×` multiscale Smooth-L1。
- final epoch only；diagnostic 只在训练前后各评估一次，不参与选 checkpoint。
- 通过条件：所有更新/rollout 有限；帧时间 T3 相对自身初始化至少改善 10%；最终 T3 step-amplitude ratio 在 `[0.2,2.0]`。

## 结果

| 协议 | 场景 | 初始 RMSE (Å) | 训练后 RMSE (Å) | 相对改善 | 最终步幅比 |
|---|---|---:|---:|---:|---:|
| 上游缩放 | T1 | 1.6304564 | 1.6304560 | 0.000019% | 0.01030 |
| 上游缩放 | T2 | 2.3546468 | 2.3546454 | 0.000059% | 0.00946 |
| 上游缩放 | T3 | 2.5899622 | 2.5899388 | 0.000907% | 0.00958 |
| 帧时间 | T1 | 5.3853921 | 5.3776514 | 0.1437% | 1.02618 |
| 帧时间 | T2 | 8.6899397 | 8.6627436 | 0.3130% | 0.94030 |
| 帧时间 | T3 | 34.1267424 | 33.9023511 | **0.6575%** | 0.94434 |

有限性和步幅 gate 通过，T3 改善 gate 失败。帧时间模型没有通过性能可学习性预实验，不能晋升到正式多种子或官方 validation。

## 优化诊断与机制解释

| 协议 | 更新数 | 裁剪数 | 裁剪率 | epoch-1 平均梯度范数 | epoch-5 平均梯度范数 |
|---|---:|---:|---:|---:|---:|
| 上游缩放 | 240 | 0 | 0% | 7.48×10⁻⁵ | 3.04×10⁻⁵ |
| 帧时间 | 240 | 124 | 51.67% | 4.0068 | 0.6101 |

时间单位修正确实把学习信号放大约 4–5 个数量级，因此旧协议的近零响应不是偶然；但帧时间下随机 horizon 的损失高度波动，超过一半更新被裁剪。仅凭该 diagnostic 结果不能选择更小学习率或不同 clip，因为那将构成同一诊断集上的调参。

更具机制意义的瓶颈是当前 `DenseEquivariantAcceleration` 只以坐标、原子/残基类型和质量构造径向力，不读取速度。蛋白—配体在相似位置但不同运动方向时，模型无法产生方向相关的制动、旋转或耗散项。下一候选应在新的 development 内部划分上加入严格等变的速度通道，例如以不变量 `|v_i|²`、`v_i·r_ij` 调制沿 `r_ij` 的力，并加入沿 `v_i` 的标量阻尼；这保持 E(3) 等变，同时能表达抗漂移。

## 证据

- 预注册：`configs/frame_time_learnability_preflight.json`
- 原始报告：`reports/reproduction/evidence/frame-time-learnability-preflight.json`
- 原始报告 SHA-256：`a25daf7ffdffd98b5d4ec4182b2b87d76bb3fd26cb631670b374a719f742556b`
- 运行日志：`reports/reproduction/evidence/frame-time-learnability-preflight.log`
- checkpoints 不进入 Git；其 SHA-256 已记录在原始报告中。
