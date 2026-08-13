# 稠密径向 E(3) 等变动力学正确性 pilot

> 状态：单复合物实现正确性 `5/5` 通过；尚未证明训练有效或比赛指标提升。  
> 数据边界：一个 MISATO-100 train 复合物；不访问 validation/test。  
> 因果边界：该 pilot 用于替代未通过旋转等变门槛的 FrameNet 空间路径，不与已训练 NeuralMD 比较预测效果。

## 1. 设计依据与取舍

EGNN 的核心坐标更新由不变量标量消息乘相对位移构成，因此在平移、旋转、反射和同类节点置换下结构性等变。官方实现采用 MIT 许可证：<https://github.com/vgsatorras/egnn>；原论文为 Satorras、Hoogeboom 与 Welling，ICML 2021：<https://proceedings.mlr.press/v139/satorras21a.html>。

NequIP 与 PaiNN 能表达更丰富的高阶几何信息，并已有原子势能/动力学证据；NequIP 的 Nature Communications 论文尤其报告了小数据条件下的高数据效率：<https://www.nature.com/articles/s41467-022-29939-5>。但它们需要球谐、不可约表示或完整原子势生态，当前赛前迁移成本和输入契约风险更高。因此本阶段选择一个 8,930 参数的最小径向场来先验证正确性，不将其包装为最终架构。

模型只预测配体加速度：

`a_i = Σ_j φ_LL(h_i,h_j,||x_j-x_i||²)(x_j-x_i) + Σ_r φ_LP(h_i,h_r,||p_r-x_i||²)(p_r-x_i)`

其中 `φ` 只接收原子/残基嵌入和平方距离，输出标量；再乘相对位移得到等变向量。蛋白 `p_r` 仅作固定条件。第一版使用稠密配对与平滑指数包络，避免 radius cutoff、边集合和边顺序引入额外变量。

## 2. 冻结实现

- 空间场：`protein_quanta/dense_equivariant_dynamics.py`；
- ODE：与 NeuralMD 一致的二阶状态 `(velocity, position)`；
- Euler 内部步长：`0.025`，确保 5 帧跨度至少包含两个积分更新并获得网络梯度；
- horizons：`5/10/20/40`；
- 蛋白：逐调用固定；
- 本阶段随机初始化，仅做前向/反向和对称性 gate。

## 3. 真实 train 复合物结果

| horizon | Smooth-L1 | 参数梯度范数 | 有限前向/反向 |
|---:|---:|---:|---|
| 5 | 0.624791 | 2.93e-8 | 是/是 |
| 10 | 0.681339 | 1.14e-6 | 是/是 |
| 20 | 0.607108 | 5.82e-6 | 是/是 |
| 40 | 0.571817 | 2.42e-5 | 是/是 |

其余 gate：

- 蛋白坐标位级不变：通过；
- 40 帧末帧到初始配体位置梯度：有限非零，范数 `1.603200`；
- 共同旋转/平移 40 帧最大误差：`8.5831e-6 Å`，低于预注册 `1e-4 Å`；
- 汇总：`5/5` 通过。

这些随机初始化 loss 与发布 NeuralMD 数值接近没有性能含义：短跨度坐标主要由初始速度支配，且两个模型的权重状态不同，禁止据此比较优劣。

## 4. 下一步与阻塞

下一阶段只允许做冻结 development 中 10 个复合物、seed 42、5 epoch 的运行性 preflight。MISATO 处理对象不保留显式配体共价键，因此不能用距离猜键来计算极端键长事件；必须把处理顺序映射回 PDB ID，并复用已有 RCSB `CONECT` 键图审计。完成前，preflight 即使 loss/gradient/显存全部稳定，也只能判“运行性部分通过、Phys 阻塞”，不能进入 64/16 效果 gate。

机器证据：`reports/reproduction/evidence/e18a/dense-equivariant-pilot.json`。
