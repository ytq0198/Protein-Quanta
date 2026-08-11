# Protein-Quanta 实验进展报告

> 这是持续更新的实验总账。详细 JSON、单次复现说明和图片保留在 `reports/reproduction/` 与 `reports/figures/`。

## 项目状态

| 项目 | 当前状态 |
|---|---|
| 数据 | MISATO-100 已下载并校验；100/100 复合物审计通过 |
| 统一评估器 | Static、Linear、NeuralMD checkpoint 已接入 |
| 官方 checkpoint | 测试集 10 个复合物、100 帧完成 |
| 从头训练 | seed 42、100 epoch 完成；发现训练爆炸 |
| 创新实验 | Static 锚点残差 C1 已通过；进入训练稳定化与可学习门控 |
| 测试 | 本地 39 项通过；服务器同步验证待本阶段结束执行 |

## 2026-08-10 至 2026-08-11：复现基础设施

### 关键决策

1. 比赛指导手册没有公开 `Geo/Phys/Dyn/Stab` 权重或 `T1/T2/T3` 定义，因此所有新增指标只标为项目代理诊断。
2. 数据和模型权重不进入 Git；所有 checkpoint 记录 SHA256。
3. 使用 NeuralMD 作者维护的 condition-aware `torchdiffeq` fork，固定 commit `3d7c7ec8c534a9b18b8b7c7d1fea0c235e6468d0`。
4. 发布权重结构与上游 parser 默认值不同；从 state dict 识别 100 个 radial bases 和关闭 velocity refinement，并执行严格加载。

### 已完成成果

- Static/Linear 基线、MISATO 审计器、统一代理指标、NeuralMD 单样本与分割批量评估器。
- 发布 checkpoint SHA256：`364404a7ce61ec1180fa3800c0dcbddf377b672d22f915fc6b19d202710a4a5a`；missing/unexpected keys 均为 0。
- 10 个测试复合物的官方预处理与项目预处理最大绝对差均小于 `1e-6 Å`。

## 2026-08-11：发布 checkpoint 基线

### 测试集 100 帧

| 模型 | 坐标 RMSE（Å，↓） | Matching（Å，↓） | Stability（%，↑） | 对齐 RMSD（Å，↓） | Rg MAE（Å，↓） | RMSF MAE（Å，↓） | 接触一致率（↑） |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeuralMD | 2.2286 | 0.4503 | 79.69 | 0.7465 | 0.1277 | 2.3419 | 0.9615 |
| Static | 2.2761 | 0.4329 | 82.94 | 0.7932 | 0.0900 | 2.6565 | 0.9696 |
| Linear | 37.3079 | 50.2041 | 6.83 | 36.9511 | 34.7432 | 26.6246 | 0.6593 |

结论：NeuralMD 捕获了非平凡动态并改善坐标/RMSF，但 Static 在 Matching、Stability、Rg 和接触图上仍更强。Linear 长程发散，不再作为可竞争方案。

### 数值积分消融

验证集 100 帧下，Euler 步长 5/2.5/1 的坐标 RMSE 分别为 2.4804/2.4802/2.4801 Å，Stability 分别为 76.38/76.14/76.00%。减小步长没有解决几何漂移，保留官方步长 5。

![NeuralMD 相对 Static 的代理指标变化](figures/neuralmd_test_static_improvement.png)

![逐复合物坐标与稳定性](figures/neuralmd_test_per_complex.png)

## 2026-08-11：从头训练复现

### 配置

- seed 42，100 epochs，batch size 8，Adam，学习率 `1e-4`；
- 随机最长 20 帧训练片段；Euler step 5，scaling 100；
- 原始位置 MSE，无梯度裁剪，无几何正则。

### 关键观察

- 最优验证 checkpoint 约在 epoch 40。
- epoch 48 的平均位置损失出现约 `4531` 的尖峰。
- 尖峰后 Stability 长期停在约 52.6%，说明训练未自行恢复。

| 权重 | 测试坐标 RMSE（Å，↓） | Matching（Å，↓） | Stability（%，↑） | Rg MAE（Å，↓） |
|---|---:|---:|---:|---:|
| 发布 checkpoint | 2.2286 | 0.4503 | 79.69 | 0.1277 |
| 从头训练 best | 2.2566 | 0.5668 | 69.05 | 0.1980 |
| 从头训练 final | 2.5403 | 1.2492 | 52.21 | 0.2731 |

### 权重

- best SHA256：`ef33b335ae1ef8488015f11ceed4af4dca24eb46446a7b5f88f05190d636aa72`
- final SHA256：`9222dcd8ca33a6a7b6b5985234830324e67ca5c53fef0cc30ea8fc7a9fef08aa`

### 阶段结论

原始坐标 MSE 能得到接近发布权重的坐标 RMSE，但不能稳定保持内部几何；单次异常梯度可破坏后续全部训练。下一阶段先做梯度控制，再做几何感知损失。

## 创新突破路线与当前假设

详见 `docs/experiment-design-and-research-roadmap.md`。当前按以下顺序推进：

1. C1：Static 锚点残差的验证/测试可行性实验；
2. E3：梯度裁剪 1.0 与完整梯度日志；
3. E5：稳定训练三随机种子；
4. E6：成对距离 Smooth-L1；
5. E7/E8：速度/Rg 辅助监督与可学习锚点门控。

每个阶段完成后，本报告追加实验编号、Git commit、配置、数字、图表和 go/no-go 决策。

## 2026-08-12：C1 Static 锚点残差可行性

### 假设与预注册

把 NeuralMD 预测写成相对最后观测构象的动态残差，并按

\[
\hat{x}_t=x_{static}+\exp(-\beta t/98)(x_t^{NMD}-x_{static})
\]

随时间抑制漂移。验证集只扫描 `β ∈ {0, 0.25, 0.5, 1, 2, 4, 8}`。候选必须满足：RMSF MAE 不超过 `β=0` 的 105%，坐标 RMSE 不超过 102%；在合格候选中最大化 Stability，再以 Matching 和较小 β 打破并列。测试集不重新扫描。

### 验证集决策

预注册规则选择 `β=4.0`。

| 指标 | NeuralMD `β=0` | Anchor `β=4` | 变化 |
|---|---:|---:|---:|
| 坐标 RMSE（Å，↓） | 2.4804 | 2.4726 | 改善 0.3% |
| Matching（Å，↓） | 0.5742 | 0.5218 | 改善 9.1% |
| Stability（%，↑） | 76.38 | 81.36 | +4.99 个百分点 |
| 对齐 RMSD（Å，↓） | 0.8959 | 0.8839 | 改善 1.3% |
| Rg MAE（Å，↓） | 0.1578 | 0.1254 | 改善 20.5% |
| RMSF MAE（Å，↓） | 2.3249 | 2.3953 | 恶化 3.0% |
| 接触一致率（↑） | 0.9614 | 0.9696 | 改善 0.9% |

达到 go 条件：Stability 提升超过 2 个百分点、Matching 改善超过 5%，且两个防塌缩约束均满足。

### 冻结测试集结果

`β=4.0` 由验证集冻结；测试集只比较 `β=0` 与 `β=4` 一次。

| 指标 | NeuralMD `β=0` | Anchor `β=4` | Static | 结论 |
|---|---:|---:|---:|---|
| 坐标 RMSE（Å，↓） | 2.2286 | 2.2556 | 2.2761 | 较 NeuralMD 恶化 1.2%，仍优于 Static |
| Matching（Å，↓） | 0.4503 | 0.4189 | 0.4329 | 较 NeuralMD 改善 7.0%，也优于 Static |
| Stability（%，↑） | 79.69 | 83.47 | 82.94 | +3.78 个百分点，也优于 Static |
| 对齐 RMSD（Å，↓） | 0.7465 | 0.7685 | 0.7932 | 较 NeuralMD 恶化 2.9%，仍优于 Static |
| Rg MAE（Å，↓） | 0.1277 | 0.0912 | 0.0900 | 改善 28.6%，略差于 Static |
| RMSF MAE（Å，↓） | 2.3419 | 2.4075 | 2.6565 | 恶化 2.8%，仍优于 Static |
| 接触一致率（↑） | 0.9615 | 0.9700 | 0.9696 | 改善 0.9%，略优于 Static |

### 科研结论

固定时间衰减已经在未参与选择的测试集复现了验证集方向：它显著修复 Matching、Stability、Rg 和接触图，同时保留大部分坐标与 RMSF 动态信息。Anchor `β=4` 在七项指标中六项优于 Static，仅 Rg MAE 略差。这支持以下更强假设：不同复合物、不同时间点需要不同的动态残差置信度；固定 β 应升级为保持 SE(3) 性质的可学习门控，而不是继续手调全局 β。

![验证选择与冻结测试的锚点权衡](figures/anchor_residual_tradeoff.png)

### 产物

- `reports/reproduction/anchor_residual_val_scan.json`
- `reports/reproduction/anchor_residual_test_frozen.json`
- `reports/figures/anchor_residual_tradeoff.png`
- `protein_quanta/anchoring.py`
- `scripts/evaluate_anchor_residual.py`
