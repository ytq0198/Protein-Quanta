# 速度感知 E(3) 等变加速度：真实复合物正确性 gate

> 结论：**7/7 correctness gate 通过；尚未证明预测性能。** 10,243 参数最小模型在一个真实 MISATO-100 train 复合物上对 5/10/20/40 帧均保持有限前向、有限非零反向，并通过旋转、反射、平移和速度敏感性检查。

## 机制设计

上一帧时间预实验表明，单位一致参数化恢复运动振幅，但当前位置-only 径向场无法有效制动或转向。新模型将速度作为 E(3) 一阶向量输入，同时只让 MLP 读取标量不变量：`|v_i|²`、`|r_ij|²` 与 `v_i·r_ij`。输出向量限制为相对位移方向和速度方向的线性组合：

`a_i = Σ_j φ_ij(invariants) r_ij + Σ_p ψ_ip(invariants) r_ip − softplus(γ_i) v_i`

点积与平方范数在旋转和反射下不变，相对位移与速度按同一正交矩阵变换，平移在相对位移中消去；`softplus(γ_i)≥0` 使最后一项初始即为耗散方向。该设计借鉴 EGNN 对向量型速度输入的处理思想，但本阶段是项目内最小可审计实现，不声称复现完整 EGNN/EGNO/TrajCast。

## 真实 train 复合物结果

| Horizon | Smooth-L1 | 参数梯度范数 | 有限前向/反向 |
|---:|---:|---:|---|
| 5 | 1.243786 | 0.261922 | 是/是 |
| 10 | 2.324360 | 0.919717 | 是/是 |
| 20 | 3.620314 | 2.580120 | 是/是 |
| 40 | 5.071603 | 5.326225 | 是/是 |

其余 gate：

- 蛋白条件坐标位级不变：通过；
- 40 帧终点到初始位置梯度有限非零，范数 `45.9621`；
- 共同旋转+平移最大误差：`3.8147×10⁻⁵ Å`；
- 共同反射+平移最大误差：`3.0518×10⁻⁵ Å`；
- 将真实初速度改为零会改变加速度，差值范数 `0.56755`；
- 两个刚体误差均低于预注册 `10⁻⁴ Å`；汇总 7/7 通过。

## 边界与下一步

随机初始化损失和梯度只能证明实现可微、速度通道有效且对称性正确，不能与 NeuralMD 或位置-only 模型比较成绩。下一实验必须使用尚未查看结果的新 development 内部诊断划分，从头配对比较位置-only 与速度感知模型；已查看的 frame-time 16-complex diagnostic、旧 16-complex holdout 和官方 validation/test 均不得用于选择结构或超参数。

## 证据

- 实现：`protein_quanta/velocity_equivariant_dynamics.py`
- 合成单元测试：`tests/test_velocity_equivariant_dynamics.py`
- 预注册：`configs/velocity_equivariant_correctness_gate.json`
- 原始报告：`reports/reproduction/evidence/velocity-equivariant-correctness.json`
- 原始报告 SHA-256：`f0cc7862a42d796604a8349ed467383d2ab8600e67bd269419d557f135abfd01`
