# E6 成对距离损失与竞赛场景早停实验

日期：2026-08-12
状态：E6/E6b 作为独立创新 **no-go**；竞赛场景感知早停 **go**，成为当前初赛模型选择候选。

## 1. 问题与实验纪律

正确复现表明，官方 20 帧随机窗口训练的 epoch 15 权重能复现发布 checkpoint，但训练损失有限并不能防止 100 帧闭环外推发散。比赛又分别考察 T1/T2/T3，因此本实验不再只按官方 coordinate MAE 选择模型。

所有方法选择仅使用 10 个验证复合物。测试集在冻结 `seed=42, epoch=5, λ_pair=0` 后评估一次，未用于回调。所有表格是项目代理指标，不是官方归一化分数。

## 2. E6 设计

位置损失之外增加：

\[
L = L_{pos}+\lambda_d L_{pair}
\]

`L_pair` 对每个复合物分别取无序、非对角重原子对距离的 Smooth-L1（`beta=0.5 Å`），先在复合物内平均，再在 batch 内平均。该损失对整体旋转和平移不变。

10 个确定性训练批次分别测量 `L_pos` 与 `L_pair` 的参数梯度范数，并按

\[
\lambda_d=0.1\,\mathrm{median}\left(\frac{\|g_{pos}\|}{\max(\|g_{pair}\|,10^{-12})}\right)
\]

冻结得到 `λ_pair=0.67725090936284`。其中一个仅一步外推的批次两项参数梯度均为 0，按预先实现的公式原样保留，没有事后剔除。校准 JSON SHA256 为 `e8a842847f9b99b99232b3d7fc10fc71739897ad4670b15931975540afaf3d57`。

## 3. 工程等价性

新训练补丁在 `λ_pair=0` 时完全绕过成对损失计算。为了在关闭开发期测试集评估后仍复现训练集 shuffle，代码只消费与 DataLoader 迭代器相同的随机 seed，不读取测试样本。

与旧的官方 5-epoch preflight 比较，新旧权重最大绝对参数差为 `1.19×10^-7`，满足数值等价。补丁同时支持每 5 epoch 保存权重和训练期禁用测试集评估。

## 4. E6 结果与因果消融

### 4.1 相对发布 checkpoint 的 20/100 epoch 结果

- epoch 5/10 满足原场景门槛：T3 Matching 约改善 6.01%，Stability 约提升 2.02 点，RMSF 恶化约 0.84%；
- epoch 15 开始失败；epoch 20 的 T3 Matching 恶化 92.16%，Stability 下降 27.87 点；
- 完整 100 epoch 中仅 5/10 通过，15–100 全部失败；epoch 100 的 T3 Matching 恶化约 849%，没有第二个恢复窗口；
- seeds 0/42/123 的早期“通过”现象均出现。

### 4.2 Pair 对同 epoch 的纯位置基线

| Epoch | T1 Matching | T2 Matching | T3 Matching | T3 Stability | 解释 |
|---:|---:|---:|---:|---:|---|
| 5 | 约 0% | 约 0% | -0.0002% | -0.0001 点 | 无可归因作用 |
| 10 | 约 0% | -0.0001% | -0.0005% | +0.0005 点 | 无可归因作用 |
| 15 | +0.0027% | -0.0040% | +0.9789% | -0.6125 点 | T3 略差 |
| 20 | -4.3529% | -6.6989% | -41.5280% | +11.0603 点 | 显著缓解纯位置训练崩坏，但仍差于发布基线 |

关键结论：epoch 5/10 相对发布 checkpoint 的收益来自 checkpoint 时机，而不是 `L_pair`。10% 辅助梯度的 Pair 到后期才产生明显保护作用，但不足以把模型拉回可竞赛区域。因此 E6 不能作为已验证创新对外主张。

## 5. E6b 机制验证

把辅助梯度目标提高到 50%，即 `λ_pair=3.3862545468142`，只运行一次 seed 42 / 20 epoch：

- epoch 5/10 仍与纯位置基线几乎相同；
- epoch 15 的 T3 几何比 E6 更接近发布基线，但 ligand collision 相对初始上升约 28%；
- epoch 20 的 T3 Matching 仍恶化约 13.5%、Stability 下降约 7.76 点。

E6b 同样 no-go，不继续扫描权重。它支持“更强 Pair 会延迟几何崩坏”，但没有形成合格竞赛模型。

## 6. 真正通过的结果：竞赛场景感知早停

纯位置损失 `λ_pair=0` 的 epoch 5 在 seeds 0/42/123 上给出几乎相同的验证变化：T1/T2 基本不变，T3 Matching 改善约 6.01%、Stability 提升约 2.02 点、RMSF 代价约 0.84%。因此冻结 seed 42 / epoch 5。

### 验证集与冻结测试集（相对发布 checkpoint）

| 数据 | 场景 | 坐标 RMSE | Matching | Stability | RMSF MAE |
|---|---|---:|---:|---:|---:|
| Validation | T1 | +0.02% | -0.02% | -0.01 点 | +0.02% |
| Validation | T2 | +0.02% | +0.05% | -0.02 点 | +0.04% |
| Validation | T3 | +0.12% | **-6.01%** | **+2.02 点** | +0.84% |
| Frozen test | T1 | +0.02% | -0.11% | +0.01 点 | +0.03% |
| Frozen test | T2 | +0.01% | -0.01% | +0.00 点 | +0.02% |
| Frozen test | T3 | +0.09% | **-5.45%** | **+1.53 点** | +0.64% |

测试集复现了验证集方向，没有用于改变 epoch 或 seed。selected checkpoint SHA256 为 `0e7d5150aa5f305499f17663d3a74b1063b0591733f534e676303ce11de50b8c`。

## 7. 科研解释与下一步

官方按全轨迹 coordinate MAE 选 epoch 15；比赛场景评估显示 epoch 5 对 T3 的内部几何更好，而坐标与动力学代价很小。这说明局部训练目标与竞赛多时间尺度目标之间的错配不仅体现在 loss，也体现在 checkpoint selection。

当前可以可靠主张的是“竞赛多场景验证驱动的模型选择”，不是“Pair loss 已成功”。下一步优先验证 epoch-5 权重与已通过的 Static 锚点残差能否形成互补 Pareto 改善；若不能，则保留 epoch 5 与 C1 为两个独立候选。复赛研究可进一步设计随 rollout 风险自适应的 loss schedule 或门控，而不是继续盲扫常数 `λ_pair`。

## 8. 产物与限制

- `reports/reproduction/neuralmd_pair_gradient_calibration.json`
- `reports/reproduction/neuralmd_pair20_scenarios_val.json`（E6 epoch 20 失败样本）
- `reports/reproduction/neuralmd_earlystop_epoch005_val.json`
- `reports/reproduction/neuralmd_earlystop_epoch005_test.json`
- `reports/reproduction/neuralmd_scenarios_published_test.json`
- `reports/figures/neuralmd_earlystop_tradeoff.png`

限制：每个 split 只有 10 个复合物；本地没有官方归一化评分代码；epoch 5 的优势主要集中在 T3；Pair 的负结论只覆盖当前网络、窗口、Smooth-L1 与两档梯度占比，不能外推为所有几何正则均无效。
