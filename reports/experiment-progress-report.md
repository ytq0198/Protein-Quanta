# Protein-Quanta 实验进展报告

> 2026-08-13 新版手册校准：正式方向二数据口径已更新为去多肽后的 MISATO
> `13,066/1,357/1,357`，T1/T2/T3 固定为 `10→10 / 80→20 / 20→80`，并新增可选 T4。
> 新版暂不公开 Geo/Phys/Dyn/Stab 与 T1/T2/T3 权重；本报告此前所有加权结果均降级为
> MISATO-100 内部 feasibility proxy，不是官方成绩。详见
> `docs/2026-08-13-updated-manual-and-template-audit.md`。

> **2026-08-13 T1 协议审计**：项目此前实际执行的 T1 是旧解释 `2→18`，不是新版 `10→10`。
> 因此下文所有旧 T1 数字统一降级为“历史 T1-2→18”；T2/T3 与复现工程证据仍保留。
> 活动候选已撤销；纠正后的 validation 已完成但没有形成全指标获胜候选。详见
> `docs/2026-08-13-t1-protocol-correction.md`。

纠正后的 validation 已完成：published NeuralMD 在新版 T1 的坐标 RMSE 为 `1.3880 Å`，Static 为 `1.3605 Å`；但 published 的 Matching/Stability/RMSF/误差斜率均优于 Static。epoch 5 在 T1/T2 与 published 几乎相同，在 T3 改善 Matching `6.01%`、Stability `+2.0185` 点，同时坐标/RMSF/斜率轻微回退。因新版未公布权重，published 保持严格主基线，epoch 5 仅保留为未晋升研究候选。详见 `reports/reproduction/2026-08-13-neuralmd-updated-guide-validation.md`。

## 2026-08-13：新版材料、模型迁移与增强实验校准

完成新版 38 页手册与旧版 36 页手册、最新算法赛 Word 模板的全文和重点页面核验。正式方向二数据口径改为去多肽后的 MISATO `13,066/1,357/1,357`；T1/T2/T3 固定为 `10→10 / 80→20 / 20→80`，新增 T4。新版不再公布旧版两层权重，故全部历史加权 proxy 已降级为内部筛选工具。

完成 ProTDyn、ProAR、PVB、EPT、dynamics-aware DPLM 的任务适配与合规审计。PVB/EPT 的预训练均与 PDB/PDBBind 有关系，在无法拿到训练 ID 并排除官方测试/近同源体系前，不能进入合规候选；ProAR 的 anti-drifting 机制优先于大模型直接迁移。

在 commit `7143837` 预注册后完成三 seed 训练态噪声增强。`σ=0.02` 的标准化特征噪声使 T1 RMSE 改善 `1.03%`，但 T2/T3 恶化 `3.87%/1.84%`，整体 proxy 恶化 `0.98%`，仅 1/3 seeds 获胜，判定 no-go。完整报告见 `reports/reproduction/2026-08-13-temporal-noise-augmentation.md`。

> 这是持续更新的实验总账。详细 JSON、单次复现说明和图片保留在 `reports/reproduction/` 与 `reports/figures/`。

## 项目状态

| 项目 | 当前状态 |
|---|---|
| 数据 | MISATO-100 已下载并校验；100/100 复合物审计通过 |
| 统一评估器 | Static、Linear、NeuralMD checkpoint 及 T1/T2/T3 场景评估已接入 |
| 官方 checkpoint | 测试集 10 个复合物、100 帧完成 |
| 从头训练 | 官方配置已纠正；seed 42 best 几乎逐位复现发布 checkpoint |
| 创新实验 | 普通 Transformer 机制筛选通过；10 步闭环总体/T2 改善但未过 T3 门槛；anchor、Pair、噪声与 ProAR proxy no-go |
| 测试 | 当前本地 145 项通过，另有 29 个子测试通过 |

## 2026-08-10 至 2026-08-11：复现基础设施

### 关键决策

1. 本段记录旧手册阶段决策：旧版曾给出 `Geo/Phys/Dyn/Stab = 40%/25%/25%/10%` 和 `T1/T2/T3 = 50%/30%/20%`；新版已经撤下这些权重。当前只报告各场景原始代理与相对基线，不拼接伪“总分”。
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

## 2026-08-11：从首帧开始的长跨度压力训练（后验更正）

### 配置

- seed 42，100 epochs，batch size 8，Adam，学习率 `1e-4`；
- 从第 0 帧开始、随机终点 1–99 的可变长训练片段；Euler step 5，scaling 100；
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

### 阶段结论与配置更正

原始坐标 MSE 能得到接近发布权重的坐标 RMSE，但不能稳定保持内部几何；单次异常梯度可破坏后续全部训练。2026-08-12 复核官方 `hyperparameter.txt` 后确认，本次命令遗漏了 `--no_NeuralMD_Binding_start_with_first_frame`，导致 `frame_num=20` 未生效。因此本节结果重新定义为**最长 99 帧的压力实验**，不再声称是官方训练配置复现。

## 创新突破路线与当前假设

详见 `docs/experiment-design-and-research-roadmap.md`。当前按以下顺序推进：

1. C1：Static 锚点残差的验证/测试可行性实验（已通过）；
2. R2：纠正官方窗口采样并建立 T1/T2/T3 对齐评估（训练已完成）；
3. E3-L：长跨度压力训练下的梯度裁剪 1.0（已完成，no-go）；
4. E6：成对距离 Smooth-L1 与长跨度泛化的受控实验；
5. E7：速度/Rg 辅助监督；
6. E8：可学习锚点门控。

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

## 2026-08-12：E3-L 长跨度压力训练（梯度裁剪 1.0）

### 受控设计

只增加全局 L2 梯度范数裁剪 `1.0`、裁剪前范数日志和非有限更新保护。数据划分、seed 42、网络、位置 MSE、从第 0 帧开始的随机长跨度、Euler 配置、Adam 与学习率均保持不变。有限但很大的损失仍参与反向传播；只有非有限 loss/gradient 才跳过。该实验与上一节压力配置严格对照，但不是官方 20 帧窗口配置。

1 epoch 预检成功保存 best/final checkpoint，且日志字段完整。随后在 A6000 GPU 0 完成 100 epoch。权重与原始日志保存在服务器 Git 仓库外。

### 训练现象

- 100 epoch 中有 68 个 epoch 至少发生一次裁剪，共裁剪 408 个 batch；非有限跳过数为 0。
- 最大单 epoch 平均位置损失为 `3244.37839`（epoch 47）。
- 最大单 batch 裁剪前梯度范数为 `403,650,528`（epoch 47）；epoch 48 仍达到 `13,917,789`。
- 梯度裁剪阻止了非有限更新，却没有阻止验证几何质量持续退化。
- 上游按验证坐标 MAE 保存的 best 位于 epoch 35，而最高验证 Stability 出现在 epoch 5；这进一步证明单一坐标误差不是可靠的模型选择准则。

### 统一 100 帧验证集结果

以下均为同一统一评估器的 10 个验证复合物无权平均；未运行新的统一测试集评估。

| 权重 | 坐标 RMSE（Å，↓） | Matching（Å，↓） | Stability（%，↑） | 对齐 RMSD（Å，↓） | Rg MAE（Å，↓） | RMSF MAE（Å，↓） | 接触一致率（↑） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 未裁剪 best | 2.4683 | 0.6743 | 66.64 | 0.9669 | 0.2317 | 2.2242 | 0.9407 |
| 裁剪 best | 2.4714 | 0.7638 | 59.69 | 1.0036 | 0.3079 | 2.1766 | 0.9286 |
| 未裁剪 final | 2.6924 | 1.2298 | 52.27 | 1.6365 | 0.3254 | 2.0266 | 0.8794 |
| 裁剪 final | 2.9152 | 1.4080 | 45.52 | 1.7112 | 0.4141 | 1.5954 | 0.8688 |

相对未裁剪 final，裁剪 final 的 Stability 下降 6.75 个百分点，坐标 RMSE 恶化约 8.3%；相对未裁剪 best，裁剪 best 的 Stability 下降 6.95 个百分点。预注册 go 条件失败，因此在该长跨度压力设定下明确 **no-go**，不继续扫描裁剪阈值，也不做三 seed 扩展。此结论不得外推为“官方 20 帧配置下裁剪必然无效”。

### 决策与下一假设

该负结果否定的是“单独梯度裁剪足以恢复长程质量”，不是梯度监控本身。数据表明异常批次会产生极大的梯度，但固定裁剪仍允许模型在高方差随机跨度目标下逐步偏离几何有效区域。下一阶段优先处理目标错配：以成对距离损失直接对准 Matching/Stability，并比较更受控的跨度采样；C1 固定锚点继续作为当前最稳健的初赛候选，不因 E3 失败而撤回。

![裁剪训练的损失、梯度与验证稳定性](figures/neuralmd_clip1_training_comparison.png)

### 产物与权重溯源

- best SHA256：`2f33bb71231b935a427fdd8861ac3baf3d181ff57b0c624f1c25572ed3da83e6`
- final SHA256：`98b5ea5d004b814543916dc435e53843bcd0206b92f61d932a2263875a43bd28`
- `reports/reproduction/neuralmd_misato100_clip1_best_val.json`
- `reports/reproduction/neuralmd_misato100_clip1_final_val.json`
- `reports/reproduction/neuralmd_misato100_retrained_final_val.json`
- `reports/reproduction/2026-08-12-neuralmd-stable-training.md`

## 2026-08-12：R2 官方 20 帧窗口训练纠正与精确复现

### 根因与修正

官方 Hugging Face `hyperparameter.txt` 明确包含 `--no_NeuralMD_Binding_start_with_first_frame --NeuralMD_Binding_frame_num=20`。上游代码在该开关为 false 时，才会令 `start=max(0,end-20)`；此前遗漏开关导致窗口长度参数被旁路。现在由 `protein_quanta.neuralmd_training.build_training_command` 显式生成所有布尔参数，并有回归测试防止再次遗漏。

1 epoch 预检触发上游边界缺陷：`print_every_epoch=5` 时没有任何验证记录，却在结束处读取 best 数组而报 `IndexError`。最小有效预检改为 5 epoch 后完成，未修改模型逻辑。

### 复现结果

纠正配置完成 seed 42、100 epoch。上游按验证 coordinate MAE 选择 epoch 15 为 best。统一验证集 100 帧结果如下：

| 权重 | 坐标 RMSE（Å，↓） | Matching（Å，↓） | Stability（%，↑） | 对齐 RMSD（Å，↓） | Rg MAE（Å，↓） | RMSF MAE（Å，↓） | 接触一致率（↑） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 发布 checkpoint | 2.4804332 | 0.5741583 | 76.3756 | 0.8958692 | 0.1577934 | 2.3248793 | 0.9614264 |
| 纠正训练 best | 2.4804328 | 0.5741685 | 76.3746 | 0.8958735 | 0.1577995 | 2.3248730 | 0.9614258 |
| 纠正训练 final | 95106.2867 | 146546.2032 | 25.8781 | 103624.8019 | 103622.7072 | 54972.4733 | 0.7321 |

best 与发布 checkpoint 的坐标 RMSE 差约 `4.2×10^-7 Å`，Stability 只差 `0.00095` 个百分点，可视为精确复现。训练期位置损失始终有限（最大 epoch 均值 20.7566），最大批梯度范数仅 27.01；然而 final 的 100 帧 rollout 灾难性发散。这把根因从“单纯梯度爆炸”推进为更具体的证据：**20 帧局部训练目标不能约束 100 帧闭环外推，模型选择必须显式观察 T1/T2/T3 的不同时间窗。**

### 产物

- best SHA256：`fc9092027d05af9a9a40159177d091a94ebef847f5a4dedd9ed0ef1a80bf92ec`
- final SHA256：`d7a2ee0bfa12a63b4fca0d462a39ca37f45d4bbc67bc53cfac34235af75c4362`
- `reports/reproduction/neuralmd_misato100_official_corrected_best_val.json`
- `reports/reproduction/neuralmd_misato100_official_corrected_final_val.json`
- `reports/reproduction/2026-08-12-neuralmd-official-corrected.md`

## 2026-08-12：T1/T2/T3 竞赛场景验证基准

### 协议冻结

- T1：观测帧 0–1，预测帧 2–19（18 帧）；
- T2：观测帧 0–79，以帧 78–79 初始化，预测帧 80–99（20 帧）；
- T3：观测帧 0–19，以帧 18–19 初始化，预测帧 20–99（80 帧）。

每个场景都把最后两个观测帧转换为位置/速度初值，随后使用与论文一致的 Euler 配置独立滚动。表中为 10 个验证复合物的无权平均原始代理指标，不是官方归一化分数。

| 场景 | 方法 | 坐标 RMSE（Å，↓） | Matching（Å，↓） | Stability（%，↑） | RMSF MAE（Å，↓） | RMSE 增长率（Å/帧，↓） |
|---|---|---:|---:|---:|---:|---:|
| T1 | NeuralMD | 1.5506 | 0.4855 | 81.90 | 1.5034 | 0.0652 |
| T1 | Static | **1.5095** | **0.4535** | **83.73** | 1.5585 | 0.0774 |
| T2 | NeuralMD | 1.3510 | **0.4284** | **83.31** | **1.2887** | **0.0470** |
| T2 | Static | **1.3455** | 0.4607 | 82.27 | 1.3500 | 0.0479 |
| T3 | NeuralMD | **2.2139** | 0.4763 | 82.08 | **2.1020** | 0.0150 |
| T3 | Static | 2.3390 | **0.4590** | **83.54** | 2.4968 | **0.0149** |

发布 checkpoint 与纠正训练的 best checkpoint 在全部场景、全部标量指标上的绝对差均小于 `1e-3`，因此场景评估也复现一致。两份报告 SHA256 分别为 `5fe9ad31d4dc3f25e69647babf90329c31afeed0137978807d06af9637dd71d2` 与 `f5fa89cfff98d972e22a57b9ce2067e8e0334dfd41fa4fa392362534e401ce48`。

### 对创新方向的约束

结果否定了“NeuralMD 在所有时间尺度都自然优于静态基线”的假设。T1 中 Static 同时赢得坐标、Matching 与 Stability；T2 是 NeuralMD 最均衡的窗口；T3 中 NeuralMD 保留更真实的坐标变化和 RMSF，但牺牲了 Matching/Stability。由此冻结 E6 的目标：首先用 `L_pair` 修复 T1/T3 的内部几何，同时用坐标、RMSF 和误差增长率防止模型退化成 Static。若只改善 Stability 而损害 T3 动态性，则判为失败。

![T1/T2/T3 验证基准](figures/neuralmd_scenario_baselines.png)

### 产物

- `reports/reproduction/neuralmd_scenarios_published_val.json`
- `reports/reproduction/neuralmd_scenarios_corrected_best_val.json`
- `reports/figures/neuralmd_scenario_baselines.png`

## 2026-08-12：E6 Pair loss 的 no-go 与场景感知早停突破

### E6/E6b

按 10 个训练批次的梯度尺度把 `L_pair` 初始贡献校准到 10%，冻结 `λ_pair=0.67725090936284`。零权重 5-epoch preflight 与旧官方 preflight 的最大参数差仅 `1.19×10^-7`。完成 seed 42 / 100 epoch 及 seeds 0/123 / 20 epoch，并逐个评估周期 checkpoint。

虽然 Pair 运行的 epoch 5/10 相对发布权重满足 T3 门槛，但同 epoch 的 `λ=0` 因果对照显示差异只有 `10^-6` 量级；早期收益并非 Pair 造成。Pair 到 epoch 20 才相对纯位置训练显著改善 T3 Matching/Stability，但两者都已经差于发布基线。把梯度目标提高到 50% 也只延迟崩坏并导致碰撞率上升。故 E6/E6b 不作为已验证创新，不继续扫参。

### 场景感知早停

真正稳定复现的结果是纯位置训练 epoch 5。验证集 seeds 0/42/123 的变化几乎一致；冻结 seed 42 / epoch 5 后，测试集只评估一次：

| 数据 | T1 坐标/Matching | T2 坐标/Matching | T3 坐标 | T3 Matching | T3 Stability | T3 RMSF |
|---|---:|---:|---:|---:|---:|---:|
| Validation | +0.02% / -0.02% | +0.02% / +0.05% | +0.12% | **-6.01%** | **+2.02 点** | +0.84% |
| Frozen test | +0.02% / -0.11% | +0.01% / -0.01% | +0.09% | **-5.45%** | **+1.53 点** | +0.64% |

这里百分比均相对发布 checkpoint，误差类负值为改善。结果说明官方 coordinate-MAE checkpoint selection 与比赛多场景目标不一致。当前初赛候选升级为“场景感知 epoch-5 checkpoint”，selected checkpoint SHA256：`0e7d5150aa5f305499f17663d3a74b1063b0591733f534e676303ce11de50b8c`。

![场景感知早停的验证与冻结测试结果](figures/neuralmd_earlystop_tradeoff.png)

完整因果消融、E6/E6b 失败数据与限制见 `reports/reproduction/2026-08-12-neuralmd-pair-loss-and-earlystop.md`。

### 初赛首选组合更新

固定 epoch-5 后，在验证集检查 C1 锚点：`β=4/2` 分别因 T3 坐标代价 3.72%/2.54% 失败；`β=1` 把代价控制为 1.54%，同时改善 T1/T2 全部四项指标。冻结组合测试后，相对发布 checkpoint：

- T1：坐标 -0.11%、Matching -2.48%、Stability +0.51 点、RMSF -3.51%；
- T2：坐标 -0.35%、Matching -2.27%、Stability +0.42 点、RMSF -1.60%；
- T3：坐标 +0.36%、Matching -4.99%、Stability +1.61 点、RMSF -0.36%。

因此当前首选为 `seed 42 / epoch 5 / anchor β=1`。除 T3 坐标轻微代价外，冻结测试其余 11 项比较全部改善。

![epoch-5 与 β=1 组合的验证和冻结测试](figures/neuralmd_earlystop_anchor1_tradeoff.png)

## 2026-08-12：Phys 碰撞代理审计

指导手册把 Phys 设为每个场景的 25%，因此对冻结组合补做了原子重叠初筛。规则为“原子间距离小于两者共价半径之和”；配体内部只统计非对角、无重复原子对，配体—蛋白统计所有跨分子原子对。验证集用于审计，冻结内部测试不回调参数。

| 数据 | 场景 | 配体内部：Anchor−NeuralMD（百分点） | 配体—蛋白：Anchor−NeuralMD（百分点） |
|---|---|---:|---:|
| Validation | T1 | +0.187458 | +0.000000 |
| Validation | T2 | +0.229778 | -0.000354 |
| Validation | T3 | +0.060177 | +0.001135 |
| Frozen test | T1 | +0.170025 | +0.000000 |
| Frozen test | T2 | +0.090809 | -0.000120 |
| Frozen test | T3 | -0.044150 | -0.000027 |

配体—蛋白重叠比例极低，且没有一致升高；配体内部代理则在 6 个比较中有 5 个小幅升高。关键限制是当前 NeuralMD/MISATO 预处理没有共价键表，正常成键近邻也被统计，因此绝对比例不是化学有效的 clash rate。决策是：保留 `epoch 5 + β=1` 为当前多指标候选，但不宣称 Phys 得到改善，也不构造本地总分；将 bond-aware 键长、键角、立体化学及能量/有效性检查列为后续硬门槛。

完整协议和原始数据见 `reports/reproduction/2026-08-12-collision-proxy-audit.md`。

## 2026-08-12：初赛材料合规复核

重新逐页核验指导手册与算法赛初赛模板，确认初赛截止日期为 8 月 16 日、Top 40 进入复赛、算法赛跨方向权重为 45%/30%/20%/5%，方向二内部权重为 `Geo/Phys/Dyn/Stab=40/25/25/10` 与 `T1/T2/T3=50/30/20`。手册为规划版本且没有写具体截止时刻，因此官网/群公告仍需人工最终复核。

模板明确禁止使用 AI 作答。项目据此只输出可检查的实验事实、原始数据、图表、复现入口和缺口清单，不生成可直接提交的答卷文字。逐模板字段的证据映射、P0/P1 风险和 8 月 12-16 日三人分工见 `docs/preliminary-submission-compliance-checklist.md`。当前 P0 包括：具体提交时刻/入口待人工核实、项目级 LICENSE 尚未选择、NeuralMD 上游许可状态待澄清、Phys 只有无键表碰撞初筛。

## 2026-08-12：服务器环境证据归档

新增白名单式环境采集器，只记录 Python、操作系统内核/架构、关键 Python 包、PyTorch/CUDA 与 GPU 型号/显存；明确不采集用户名、主机名、路径或环境变量。服务器实测为 Python 3.10.20、PyTorch 2.6.0+cu124、CUDA 12.4、4 张 NVIDIA RTX A6000（每张可见显存 47.402 GiB）。完整包版本见 `reports/reproduction/server-environment.json`。新增 3 项回归测试后，本地和服务器均为 77 项通过，服务器 4 项可选绘图测试跳过。

## 2026-08-12：官网公开信息复核

GOAI 官网 AI for Research 页面公开内容与本地手册一致：8 月 16 日初赛截止；Top 40 由算法赛与开放探索赛各 20 队组成；复赛后 Top 20 由两类各 10 队组成。公开页面仍未给具体截止时刻或算法赛文件格式/大小。官网链接的 Datawhale 方向二 baseline 教程页摘要另确认：每队只能选择一个算法方向；官网作品最多提交 3 次，以最后一次为准。上述公开入口与未确认项已写入 `docs/preliminary-submission-compliance-checklist.md`。

继续读取方向二教程正文及其赛题解读 PDF 后，确认上传流程为“人工完成 Word（手动补第六部分团队介绍）→ 压缩为 ZIP → 官网填写并上传”。Datawhale 截图打卡属于 Token Plan 激励，不等同于赛事提交。发现 7 月解读 PDF 的 Top 50/Top 15 已与当前官网 Top 40/Top 20 冲突；按来源优先级采用当前官网，并在 `reports/reproduction/2026-08-12-official-materials-audit.md` 留存冲突与未决项。

## 2026-08-12：公开发布自动审计

新增只读发布审计器，对 Git 跟踪文件检查 checkpoint/轨迹/HDF5 等禁入后缀、5 MiB 大文件、常见 token 前缀、manifest 证据缺失及“官方得分”误称。首轮审计通过，唯一警告为项目级 LICENSE 尚未由团队选择。新增 3 项回归测试后，本地与服务器均为 80 项通过，服务器 4 项可选绘图测试跳过。

## 2026-08-12：场景条件锚定 no-go

在查看本实验 frozen-test 前提交预注册设计 `e5dba58`：由 validation 固定 `T1=8, T2=8, T3=1`，并以当前 `global beta=1` 为比较对象。validation 上 T1/T2 四项指标全部改善，T3 保持 beta=1；完整性与 T3 坐标防线通过后，执行一次 frozen-test。

冻结结果中，T1 Matching/Stability/RMSF 分别改善 3.98%/+1.17 点/12.03%，T2 分别改善 4.24%/+0.70 点/5.01%，但 T1 coordinate RMSE 恶化 2.295%，超过预注册的 2% 上限。因此该策略判定 **no-go**，不修改 `configs/frozen_candidate.json`，也不依据 test 回调 beta。结果说明场景 ID 本身不足以决定锚定强度；后续 C2 必须加入样本/状态不确定性并在 leave-one-complex-out 验证中显式保护坐标误差。完整报告见 `reports/reproduction/2026-08-12-scenario-conditioned-anchor.md`。

## 2026-08-12：低容量不确定性门控 LOOCV no-go

在不访问 test 的前提下，对 validation 的 10 个复合物×3 场景执行 grouped leave-one-complex-out。固定六维 SE(3) 不变输入、balanced logistic regression、L2=10 和阈值 0.5；标签要求 `beta=8` 相对 `beta=1` 同时通过坐标/RMSF 防线并改善 Matching/Stability。预注册提交为 `a45a7ab`。

OOF 聚合相对 global beta=1 改善 T1/T2 Matching（-2.57%/-1.91%）、Stability（+0.79/+0.55 点）和 RMSF（-6.27%/-5.83%），所有 aggregate guard 通过；但正类召回 0.30、负类召回 0.50、balanced accuracy 仅 0.40，低于 0.60 防线。因此判定 validation **no-go**，不访问内部 test、不修改冻结候选、不在本轮调正则或阈值。完整数据与解释见 `reports/reproduction/2026-08-12-uncertainty-gate-loocv.md`。

## 2026-08-13：评分对齐重排与 bond-aware Phys no-go

按照指导手册 `Geo/Phys/Dyn/Stab=40/25/25/10` 与 `T1/T2/T3=50/30/20` 重排 E13-E17：先补 Phys，再补 Dyn 分布，最后开展 T1 优先的小预算创新。预注册门槛与执行顺序见 `docs/scoring-aligned-execution-plan.md`。

E13 没有按距离猜键，而是从 RCSB PDB 显式 `CONECT` 恢复键图，并要求重原子元素顺序与 MISATO 完全一致、PDB/MISATO frame-0 内部距离 MAE `<=2.0 A`、全部重原子有键覆盖。validation/test 覆盖率为 9/10、8/10；特殊 altloc、共价肽和多残基糖体系被严格拒绝。

E14 在覆盖 validation 子集实现了同帧键长 MAE、20% 违例率、极端键长事件，以及排除图距离 1/2 后的非键碰撞。E15 结果显示 `beta=1` 相对未锚定 epoch-5 的 T1/T2/T3 键长 MAE变化为 `+8.99%/+13.69%/-1.67%`，前两项超过预注册 5% 防线；T1/T2 还新增极少量 extreme event。因此在 validation 即判 **no-go**，不运行新的 bond-aware test、不回调 beta。历史 manifest 已标记 `active_for_submission=false`，当前安全基线回退到未锚定 seed-42 epoch 5，等待相对 published NeuralMD 的直接 Phys/Dyn 对照。完整报告见 `reports/reproduction/2026-08-13-bond-aware-phys-validation.md`。

## 2026-08-13：E16 动态分布验证与创新问题收敛

重新生成 published checkpoint 的 validation 三场景轨迹后，对它与未锚定 epoch-5 同时计算 RMSF Pearson/Spearman、Rg/原子对/步长 Wasserstein 距离、位移幅度比和速度自相关。epoch-5 在三个场景的 RMSF 形状相关性均提高，T3 的 Rg/原子对分布距离改善 `8.48%/11.93%`；但 T1/T2 分布距离略有回退，未达到“两场景整体改善”的 E16 门槛，因此仍为 **no-go**，不访问新的 internal test 分布结果。

新的机制证据是：epoch-5 和 published NeuralMD 的逐帧位移幅度都只有真值约 `0.9%-1.4%`。这说明模型虽然不是完全 Static，却存在强烈欠动力学；后续 E17 不再泛泛处理长程漂移，而是以训练期局部位移分布/速度动态为直接目标，并以 bond-aware 指标作安全门槛。直接 Phys 对照中，epoch-5 的 T1/T2/T3 平均键长 MAE相对 published 分别改善 `1.69%/2.22%/55.60%`，但 T3 rare extreme event 为 `0.2666%`、高于 published 的 `0.1082%`，所以只保留为安全工作基线，不声称综合 Phys 提升。完整报告见 `reports/reproduction/2026-08-13-dynamics-distribution-validation.md`。

## 2026-08-13：E17 局部位移损失 feasibility no-go

在 commit `b393d38` 预注册单变量设计后，以 seed 42、官方 20 帧随机窗口训练 5 epoch，只增加 coefficient=1、beta=0.5 的连续位移 Smooth-L1。训练无非有限更新，且 `--no_eval_test_during_training` 使 test 字段保持 `nan`。validation 的 20 项联合 gate 中 19 项安全检查通过；唯一失败项为核心机制目标：T1 step-amplitude ideal-gap ratio 为 `0.9999999998`，要求 `<=0.95`，说明候选与 epoch-5 基线在数值上几乎完全相同。按预注册判 **no-go**，不扫系数、不访问新 internal test、不修改活动候选。完整报告见 `reports/reproduction/2026-08-13-e17-displacement-feasibility.md`。

## 2026-08-13：长序列时序架构 Level-A 筛选

在 commit `7c1ad03` 预注册后，对 12 维刚体运动不变量状态，在同一 80-complex train、10-complex validation、参数量 <10k 和 200 epoch 预算下比较 MLP/RNN/LSTM/GRU/causal Transformer，未访问 internal test。MLP 的方向权重代理 RMSE 最低（`0.38409`），Transformer 为 `0.38453`，整体差 0.11%；但 Transformer 相对 MLP 在 T2/T3 改善约 `8.5%/10.3%`，T1 恶化约 `11.4%`。其余 recurrent core 没有整体优势。五个模型均未通过完整 gate，因此不直接启动 3D 大模型；结果支持一个需要新验证设计的后续假设：T1 使用局部短记忆路径，T2/T3 使用 causal attention，并与等变空间编码器结合。完整报告见 `reports/reproduction/2026-08-13-temporal-architecture-screen.md`。

## 2026-08-13：新版 MISATO 划分审计通过

对 MISATO 官方公开 train/validation/test ID 和 NeuralMD `peptides.txt` 逐 ID 审计。原始 `13,765/1,595/1,612` 分别排除 `699/238/255` 个多肽后，精确得到新版手册要求的 `13,066/1,357/1,357`；所有源文件无重复行，三个划分无交集。审计保存源文件及过滤集合 SHA256，不复制数据 ID。完整报告见 `reports/reproduction/2026-08-13-updated-misato-split-audit.md`。

## 2026-08-13：ProAR 式交替抗漂移 proxy no-go

在 commit `4556a31` 预注册后，按 ProAR 的双阶段训练和交替采样思想构造 12 维确定性最小机制实验。一次端点预测和交替修正共享相同的插值器/预测器权重，只改变推理调用顺序；seeds `0/42/123`、最终 epoch、validation-only，不访问 internal test。

一次预测与交替修正的 T1/T2/T3 宏平均 RMSE 分别为 `0.41027` 与 `0.42621`；交替策略逐种子胜出 `0/3`，T1 恶化 `6.25%`，T3 仅改善 `0.34%`，远低于 5% 门槛。按预注册判 **no-go**，不移植到三维模型、不围绕块长调参。结果说明仅改变调用顺序不足以复现 ProAR 的抗漂移收益，完整方法中的预测分布训练、结构化概率性、等变局部条件和松弛可能是必要组成。完整报告见 `reports/reproduction/2026-08-13-proar-antidrift-proxy.md`。

## 2026-08-13：新版 T1 与未公开权重口径纠正

新版指导手册明确规定 T1 为 `10→10`，而旧实现为 `2→18`；同时新版手册说明 T1–T3 精确权重后续发布，项目旧脚本却使用了自定 `0.5/0.3/0.2`。现已完成两项纠正：所有新场景统一为 `T1 10→10 / T2 80→20 / T3 20→80`；当前研究汇总使用三场景等权宏平均，不再把历史权重称为比赛权重。旧原始文件保留以维持审计链，但旧 T1 数字、旧加权总量和由此产生的候选结论全部降级为历史证据。

## 2026-08-13：新版协议长序列架构与 RoPE

固定 200 epoch、seeds `0/42/123`、最终 epoch、validation-only 后，普通 Transformer 的等权宏平均为 `0.41778±0.03279`，相对 MLP `0.47707±0.01361` 改善 `12.43%`，三个种子全部胜出；T1/T3 分别改善 `2.58%/26.65%`，T2 恶化 `2.47%`。RoPE 为 `0.42593±0.03160`，相对 MLP 改善 `10.72%`，但没有超过普通 Transformer，故只保留为位置编码消融。GRU 三种子均胜 MLP，但长期改善不足 5%，作为低算力备份。完整报告见 `reports/reproduction/2026-08-13-temporal-updated-guide.md`。

## 2026-08-13：新版协议噪声增强 no-go

按固定 `σ=0.02`、三 seed、最终 epoch 重跑训练态噪声。噪声模型等权宏平均为 `0.42426±0.03190`，无噪声参考为 `0.41778±0.03279`，三个 seed 全部恶化。T1 改善 `0.82%`，但 T2/T3 恶化 `3.87%/1.84%`。因此不扫噪声强度，下一项创新转为直接在训练目标中加入可微多步闭环 rollout，检验是否能从根因上缓解 exposure bias。完整报告见 `reports/reproduction/2026-08-13-temporal-noise-updated-guide.md`。

## 2026-08-13：可微闭环训练的部分可行证据

在 commit `0a67359` 预注册后，为普通 Transformer 加入权重 `0.25` 的 10 步可微自回归损失；固定 200 epoch、final epoch 与三 seed。等权宏平均从一步训练的 `0.41778±0.03279` 改善到 `0.39723±0.00717`（`4.92%`），2/3 种子胜出；T1/T2/T3 分别改善 `1.05%/15.31%/0.46%`。闭环训练还把 T2 的 seed SD 从 `0.09780` 降到 `0.00533`，支持其缓解中程 rollout 和种子不稳定性的机制价值。

但 T3 改善远低于预注册 5% 门槛，完整 gate 判 **不通过**，不晋级三维主模型，也不立即扫描相同权重或单一 horizon。科研结论收敛为：10 步闭环能修复 T2，却不足以约束 80 步 T3；下一步应设计多时间尺度闭环目标，并通过 train 内部 grouped selection 解决训练后段回退，不能事后用 validation 挑 epoch。完整报告见 `reports/reproduction/2026-08-13-temporal-closed-loop-updated-guide.md`。

## 2026-08-13：train-only 多时间尺度闭环 gate 通过

为避免继续适应官方 validation，从 MISATO-100 的 80 个官方训练样本用 salted SHA256 确定性划出 64/16 development/holdout；官方 validation/test 未读取。在相同三 seed、200 epoch、final epoch 对照下，多时间尺度 `[5,10,20,40]` 闭环把等权宏平均从 `0.60415±0.00777` 改善到 `0.57935±0.02971`（`4.10%`），2/3 seed 胜出。T1/T2/T3 分别改善 `1.41%/1.34%/8.42%`；T3 三个 seed 的配对改善都在约 `0.056-0.061`，完整预注册 gate 通过。

该结果支持“训练 horizon 覆盖决定长期误差控制”的机制假设，但仍只是 12 维 proxy。按预注册访问规则，现允许一次冻结的官方 validation 确认；确认前不改 horizons、权重或 epoch，确认后也不据结果回调。完整报告见 `reports/reproduction/2026-08-13-temporal-multiscale-train-only.md`。

## 2026-08-13：多时间尺度闭环一次性 validation 确认通过

严格按“训练并保存全部 6 份 final checkpoint 后统一读取 validation 一次”的代码路径执行。候选与一步训练对照都只使用同一 64-complex development split，不加入 train-holdout 或 validation，未访问 internal test。多时间尺度候选的宏平均为 `0.39609±0.00757`，一步对照为 `0.40821±0.00600`，改善 `2.97%`，2/3 seed 胜出；T1/T2/T3 分别改善 `1.92%/0.08%/5.59%`，T3 三 seed 全部同向改善，确认 gate 通过。

这使 `[5,10,20,40]` 多时间尺度可微闭环成为当前最强的创新机制证据：train-only T3 改善 `8.42%`，一次 validation 确认为 `5.59%`。但它仍是 12 维 proxy，不是原子轨迹模型或官方成绩；当前主基线仍是 published NeuralMD。下一阶段是设计 E(3) 等变原子坐标迁移与 Geo/Phys/Dyn/Stab 联合门槛，不因本次 validation 结果调整现有 horizons/权重。完整报告见 `reports/reproduction/2026-08-13-temporal-multiscale-official-validation-confirmation.md`。

## 2026-08-14：三维多时间跨度 E18a 首轮 gate 失败

在接入真实 NeuralMD 前重新审查上游训练入口，纠正了一个关键因果叙述：NeuralMD 基线本身已经对随机最长 20 帧片段做连续可微二阶 ODE 训练，不是一步 teacher forcing。三维候选因此收窄为“保留原随机 1–20 帧 ODE 损失，再额外等概率采样 `[5,10,20,40]` 时间跨度”，而不是伪造一层自回归闭环。零权重时直接返回原 loss 且不额外消耗随机数。

使用一个 MISATO-100 train 复合物和发布 NeuralMD 权重做真实 E18a：10/20/40 帧前向与参数反向有限非零，蛋白坐标位级不变，40 帧末帧到初始位置的梯度范数为 `1.6045`；但 5 帧参数梯度为 `0`，共同旋转/平移最大等变误差 `6.2084e-4 Å` 超过预注册 `1e-4 Å`，组合 gate 仅 `3/5` 通过，故不启动 E18b。

后续诊断确认 5 帧零梯度来自积分离散化：复现配置 Euler 步长 `0.05` 与 5 帧缩放跨度相同，单个 Euler 位置更新不依赖网络加速度；步长减半后梯度恢复到 `1.3302e-5`。等变误差主要来自旋转（`6.4468e-4 Å`），纯平移仅 `3.0518e-5 Å`。float64 复测仍为纯旋转 `6.4317e-4 Å`、共同刚体 `6.2359e-4 Å`，排除了 float32 精度是主因，指向上游 FrameNet/ODE 的实现级偏差。当前不放宽门槛、不训练候选，只做单次导数与误差增长定位。完整审计见 `reports/reproduction/2026-08-14-neuralmd-multiscale-e18a.md`。

最终定位显示单次 acceleration 旋转误差已为 `9.1233e-3`，坐标误差从 5 帧 `3.8147e-6 Å` 累积到 40 帧 `6.4468e-4 Å`；分块误差主要出现在 protein residue 表示（`2.3729e-2`）与 complex vector（`1.0992e-1`），而 ligand vector 为 `5.0545e-5`。三个 radius 图的边集合旋转前后完全一致，且未触发 32 邻居上限，排除了邻居集合变化。由此 E18a 冻结失败，后续先修正/替换蛋白—复合物空间编码路径，再重新进入多时间跨度训练。

追加边顺序审计确认三张图的完整 `edge_index` 张量逐元素一致，排除浮点误差来自邻居顺序改变，故不实施无效的确定性排序补丁。当前缺陷范围冻结为相同图上的蛋白标量化/复合物消息路径。
