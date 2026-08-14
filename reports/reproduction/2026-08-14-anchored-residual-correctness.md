# Control-anchored gated residual：真实复合物 correctness 通过

## 结论

针对 bounded+normalized 候选“幅度恢复但坐标 ballistic”的失败，已实现 control-anchored gated E(3) residual 原型，并在完整 MISATO 真实训练复合物 `4K6V` 与前一实验冻结 control checkpoint 上通过 7/7 correctness gates。

该结果证明创新构思具备三个必要条件：初始行为严格继承稳定 anchor、存在可训练的非零梯度、更新后仍满足 E(3) 等变且修正幅度受控。它不是效果提升证据；按预注册只能解锁后续 8-train/4-diagnostic train-only effect gate。

## 构思与实现

模型输出为：

`a = a_anchor + gate(z, |v|²) · tanh(s) · a_residual`

- `a_anchor`：上一完整 train gate 中训练得到的 unbounded velocity-aware control，参数全部冻结；
- `a_residual`：bounded+normalized velocity-aware E(3) acceleration；
- `gate`：只依赖配体类型与 `log(1+|v|²)` 的 invariant gate，范围 `[0,0.25]`；
- `s`：全局 residual scale logit，初始化为 0，因此 `tanh(s)=0`，初始输出严格等于 anchor；
- `d tanh(s)/ds = 1` at zero，因此 residual scale 保留一阶可训练梯度，不会因精确零初始化而完全锁死。

这直接针对前一失败：不再让小阻尼候选从一开始就以错误幅度自由运动，而是从较稳定 control 起步，只逐步学习受门控修正。

## 冻结输入

- 样本：`4K6V`，来自上一 32 个 training IDs，不复用 diagnostic；
- anchor checkpoint SHA256：`766629ac7a2d3064a410cd85f224baaa21c61072df5fb66a685556acb594542c`；
- maximum gate：0.25；initial gate fraction：0.05，实际初始 gate 0.0125；
- 只做一次 `lr=1e-3` 更新；不访问 diagnostic、validation 或 public test。

## 结果

| 门禁 | 数值 | 结果 |
|---|---:|---:|
| 初始输出 vs anchor max abs error | **0.0** | 通过 |
| residual scale 初始梯度 | `-3.3353e-6` | finite nonzero |
| anchor frozen + bitwise unchanged | true | 通过 |
| 单步后 `tanh(scale)` | `9.9701e-4` | 非零 |
| correction / anchor 相对 L2 | `6.5883e-8` | ≤5% |
| proper rotation equivariance max error | `2.0862e-7` | ≤1e-4 |
| reflection equivariance max error | `2.6822e-7` | ≤1e-4 |

7/7 全部通过。

## 科研边界与下一门

当前只证明结构正确、可训练、初始安全。极小的单步 correction 不保证经过多次更新后有效，也可能出现 gate/scale 长期贴近 0 的“无贡献”失败。

下一 effect gate 固定为：

1. 从此前 32 个 training IDs 内取 8-train/4-diagnostic，不新增数据访问；
2. frozen anchor vs anchored residual，同 schedule；
3. residual 目标加入 coordinate Smooth-L1、T3 step-amplitude log-ratio 与 correction-energy penalty；
4. 必须同时满足 T3 RMSE 改善、T3 amplitude 向 1 靠近、T1 不恶化、correction ratio 有界；
5. 若 scale/gate 近零且无改善，判“结构可行但机制无效”；若幅度改善但 RMSE 恶化，判重复 ballistic failure；两者都不扫参救结果。

机器结果：`reports/reproduction/anchored_residual_correctness_gate.json`；冻结配置：`configs/anchored_residual_correctness_gate.json`。
