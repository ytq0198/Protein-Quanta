# Protein-Quanta 实验设计与科研路线

> 版本：v1.6（2026-08-13）
> 目标：在 2026-08-16 初赛截止前形成可信、可复现、有定量提升的方案，同时为复赛保留清晰的创新升级路线。

> 评分对齐执行以 `docs/scoring-aligned-execution-plan.md` 为准。该文件把方向二的 `T1/T2/T3` 与 `Geo/Phys/Dyn/Stab` 双层权重转化为 E13-E17 的硬门槛；本路线图中的早期探索顺序若与其冲突，以评分对齐计划为准。

> **最新决策覆盖早期结论：** E13-E15 已恢复可追溯共价拓扑并完成 validation gate。历史候选 `epoch 5 + beta=1` 因 T1/T2 键长 MAE 分别恶化 `8.99%/13.69%` 且新增极端键长事件，被降级为 no-go；当前活动安全基线为未锚定 `epoch 5`。下文关于 anchor 的“go/首选/冻结基线”只记录当时基于 Geo/Dyn/Stab 代理的阶段判断，不再代表当前主方案。

## 1. 科学问题与当前证据

任务是根据蛋白–配体复合物的初始状态预测配体长时间轨迹。当前 NeuralMD 使用 SE(3) 等变的多粒度 BindingNet 与二阶神经微分方程，在 MISATO 上预测配体动力学。Nature Communications 论文将坐标 MAE/RMSE 作为重建指标，将原子对距离 Matching 与阈值为 0.5 Å 的 Stability 作为有效性指标，并指出长时推演的灾难性误差累积仍是核心困难。

我们的复现已经得到八个直接证据：

1. Static 是很强的基线；NeuralMD 的优势主要体现在坐标、对齐 RMSD 与 RMSF，而不是所有几何指标。
2. 把 Euler 步长从 5 降至 2.5 或 1 没有改善稳定性，因此主要问题不是积分过粗。
3. 官方 20 帧随机窗口配置纠正后，seed 42 的 best 权重在验证集七项指标上几乎逐位复现发布 checkpoint，训练与评估流水线可信。
4. 同一纠正训练的 final 在短窗口训练损失保持有限时，100 帧 rollout 却发散五个数量级，直接证明局部训练和长程闭环推理之间存在目标错配。
5. 从首帧开始的长跨度压力实验在 epoch 47–48 出现极端损失/梯度；全局梯度裁剪 1.0 限制了 408 个 batch 仍未改善最终验证结果。因此梯度问题和长程几何问题相关，但单独裁剪不是充分解。
6. 竞赛场景验证显示时间尺度的取舍并不相同：T1 的 Static 在坐标、Matching、Stability 上均更强；T2 的 NeuralMD 在 Matching、Stability、RMSF 上更强；T3 的 NeuralMD 改善坐标与 RMSF，却落后于 Static 的 Matching/Stability。因此创新目标必须按场景约束，而不能只优化单一 100 帧均值。
7. 梯度校准的 Pair loss 在早期没有可测因果作用，后期虽能缓解纯位置训练崩坏，但仍不能恢复到发布基线；提高到 50% 梯度占比也造成碰撞退化。因此常数权重 `L_pair` 当前 no-go。
8. 纯位置训练 epoch 5 相对发布 checkpoint 在验证集 T3 Matching/Stability 改善约 6.01%/+2.02 点，并在三 seed 复现；冻结测试集仍改善约 5.45%/+1.53 点，T1/T2 与坐标/RMSF 代价很小。这证明 checkpoint selection 本身存在多时间尺度目标错配。

## 2. 文献依据

- [NeuralMD](https://www.nature.com/articles/s41467-025-67808-z)：确认多轨迹 80/10/10 划分、四项核心指标，以及长时间误差累积和 Stability 的重要性。
- [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html)：梯度范数裁剪是控制爆炸梯度的直接方法。
- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405)：长滚动性能受误差累积影响；在输入中注入模拟滚动误差的噪声可提高长期表现。
- [Scheduled Sampling](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)：训练时逐步暴露模型自身状态，缓解训练和推理状态分布不一致。
- [Forces Are Not Enough](https://arxiv.org/abs/2210.07237)：单点误差不能代表真实模拟质量，必须使用稳定性和模拟派生指标验证。
- [BioMD](https://arxiv.org/abs/2509.02642)：在更近期的蛋白–配体动力学生成中使用碰撞、键长和中心约束组成几何正则，并采用分层预测缓解长程误差。

上述文献不是“直接可复制的答案”。我们的推断是：梯度裁剪适合先解决已观察到的训练爆炸；几何损失适合对准 Matching/Stability；误差扰动和多跨度监督适合解决 20 帧训练、100 帧推理的不一致。

## 3. 三层研究路线

### 路线 A：训练稳定性控制（初赛必做）

保持数据、网络、ODE、位置 MSE 和随机种子不变，只增加：

- 每批记录未裁剪梯度范数、裁剪后范数和位置损失；
- `clip_grad_norm_`，首轮候选阈值为 1.0；
- 仅当 loss/gradient 非有限时跳过该批次并计数；有限但很大的损失不能静默丢弃；
- 保存 last、best-coordinate、best-stability 三类 checkpoint；选择模型时只使用验证集。

成功标准：seed 42 的 100 epoch 不出现非有限值或灾难性参数劣化；最终权重相对原始最终权重的验证 Stability 至少提高 10 个百分点，同时坐标 RMSE 不恶化超过 2%。若成立，再用 seeds 0、42、123 验证。

**阶段结果（2026-08-12）：长跨度压力设定下 E3 no-go。** 裁剪版 final 的验证 Stability/坐标 RMSE 为 45.52%/2.9152 Å，未裁剪 final 为 52.27%/2.6924 Å；两个成功条件均失败。最高裁剪前梯度范数达到 `4.04×10^8`，说明诊断有效，但固定阈值无法解决目标高方差和长程几何漂移。该设定后验确认不是官方 20 帧窗口配置，因此不把结论外推到官方训练；保留裁剪和日志作为安全设施，不把它作为创新贡献。

### 路线 B：几何感知、多时间尺度训练（初赛主提升）

场景基准已冻结为 T1 `(0–1)→(2–19)`、T2 `(0–79)→(80–99)`、T3 `(0–19)→(20–99)`。发布权重与纠正训练 best 在所有场景标量指标上的差异均小于 `1e-3`。最直接的缺口是 T1/T3 的 Matching 与 Stability，而最必须保护的是 T2/T3 的 RMSF 及 T3 坐标优势；这使 `L_pair` 成为比继续调积分器或单独裁剪更贴合竞赛 `Geo/Phys/Stab` 要求的下一项单变量实验。

基础损失为：

\[
L=L_{pos}+\lambda_dL_{pair}+\lambda_vL_{vel}+\lambda_{rg}L_{rg}
\]

- `L_pos`：原始坐标 MSE。
- `L_pair`：每个复合物内部的重原子成对距离 Smooth-L1；按复合物平均，避免大配体支配 batch。
- `L_vel`：相邻预测帧位移与真实位移的 Smooth-L1，防止模型通过趋近 Static 获得表面稳定。
- `L_rg`：预测与真实回转半径差，针对已观察到的尺度漂移。

先只加 `L_pair`，再逐项增加，任何组合都必须与同 seed、同预算的路线 A 对照。权重根据首个 batch 各损失梯度尺度初始化，而不是随意指定；随后只在验证集做小网格。

成功标准：验证 Matching 下降且 Stability 上升，坐标 RMSE 不恶化超过 2%；至少两个核心指标同时改善才进入测试集。若只提升坐标而损害 Stability，则判定失败。

**阶段结果（2026-08-12）：E6/E6b no-go。** 10% 梯度占比的 Pair 在 epoch 5/10 相对同 epoch 零权重对照几乎无差异，到 epoch 20 才缓解崩坏但仍落后于发布基线；50% 版本延迟了 T3 退化，却使 ligand collision 在 epoch 15 上升约 28%。因此不把 Pair 写成有效创新，也不继续常数权重扫描。该结果提示复赛若重启几何损失，应采用 rollout-risk-aware schedule、归一化重设计或与 checkpoint gate 联合训练。

### 路线 D：竞赛场景感知模型选择（当前初赛主候选）

不改变网络和训练目标，每 5 epoch 在验证集运行 T1/T2/T3，并用预注册 guard 选择 checkpoint。seed 42 的 epoch 5 通过，epoch 15–100 失败；seeds 0/123 的 epoch 5 方向一致。冻结测试结果复现 T3 Matching/Stability 改善，同时 T1/T2 基本不变。

该方法的科学主张是：**短窗口 coordinate-MAE 最优 checkpoint 不等于竞赛多时间尺度动力学最优 checkpoint。** 它属于训练/模型选择算法改进，而非新网络。

**历史阶段结果：代理 gate 曾判 go，但已被 E15 否决。** 固定 epoch 5 后，仅在验证集检查锚点强度；当时 `β=1` 在 Geo/Dyn/Stab 代理上通过并冻结，frozen internal test 也呈现 Matching/Stability 等改善。然而后续 bond-aware Phys validation 显示 T1/T2 键长 MAE 分别恶化 `8.99%/13.69%`，且出现未锚定模型没有的极端键长事件。依据统一评分门槛，anchor 已降级，不再是初赛首选；该失败说明“向 Static 收缩”不能代替化学有效的局部动力学。

### 路线 C：Static 锚点残差动力学（创新突破）

#### C1：无需训练的可行性验证

NeuralMD 轨迹写成静态参考与动态残差：

\[
\hat{x}_t=x_0+w(t)(x_t^{NMD}-x_0), \quad w(t)=\exp(-\beta\tau_t)
\]

其中 `β=0` 等价于 NeuralMD，较大的 β 随时间抑制漂移。先在验证集扫描 β，寻找 NeuralMD 与 Static 之间的 Pareto 改善；冻结 β 后只运行一次测试集。这不是最终模型，而是验证“动态残差需要随不确定性受限”是否成立。

**历史阶段结果（2026-08-12）：C1 通过代理 gate；2026-08-13 被 Phys gate 覆盖。** 验证集当时按 Geo/Dyn/Stab 代理选择 `β=4.0`，冻结后的测试集呈现 Stability、Matching 与 Rg 改善，因此曾支持进入 C2。bond-aware Phys 证据现已证明这类 Cartesian 收缩会扭曲共价几何；固定 β 不再作为初赛稳健基线，仅保留为解释“表面稳定与物理合法性可能冲突”的负消融。

#### C2：可学习锚点门控

若 C1 成立，把固定 `w(t)` 升级为依赖时间、配体统计量和模型状态的门控 `w_θ(t,z)`。门控只输出 `[0,1]` 标量或逐原子标量，保持旋转/平移等变性。训练时增加防塌缩约束，使门控不能全部趋近 0。

**场景条件中间实验（2026-08-12）：no-go。** 在 frozen-test 前预注册 `T1=8, T2=8, T3=1`。它在 validation 上改善 T1/T2 全部四项代理指标，在 frozen-test 上也改善 T1/T2 的 Matching、Stability 和 RMSF；但 T1 coordinate RMSE 相对 global beta=1 恶化 2.295%，超过 2% 晋升防线。因此场景 ID 不能替代可学习不确定性门控，冻结方案仍为 global beta=1。C2 的下一版必须使用推理时可观测、SE(3) 不变的样本/状态特征，采用 leave-one-complex-out 验证，并把坐标风险写入门控目标或硬约束；在交叉验证通过前不再使用 test。

**低容量不确定性门控（2026-08-12）：validation no-go。** 六维 SE(3) 不变输入配合固定 L2=10 的 balanced logistic regression，在 grouped LOOCV 的 aggregate T1/T2 指标上优于 global beta=1，坐标/RMSF 防线也通过；但对“`beta=8` 安全且有益”标签的 balanced accuracy 仅 0.40（要求 `>=0.60`）。因此没有访问 test，也没有继续调特征、阈值或正则。该结果说明当前 10 个独立复合物不足以支持可靠的个体风险门控；复赛若继续，需扩充独立训练复合物，并把二分类改为风险/收益双输出加保守拒绝，而非在同一 validation 上继续搜参。

#### C3：可微几何校正

MISATO 当前预处理没有显式键表，因此第一版不声称“严格键约束”。可先使用重原子成对距离与回转半径构造一个小步、可微的坐标校正；若后续可靠恢复化学键，再只对共价键施加更强投影。该模块必须比较校正前后坐标误差、Matching、Stability、RMSF 与接触图，防止把轨迹压成静态构象。

创新主张应限定为：**等变 Neural ODE 上的静态锚点残差门控与几何感知长程训练**。只有 C1/C2 数据支持后，才在初赛材料中表述为有效方法。

## 4. 实验矩阵与顺序

| 阶段 | 实验 | 训练预算 | 数据决策 | 进入下一阶段条件 |
|---|---|---:|---|---|
| E0 | 发布 checkpoint、Static、Linear | 已完成 | test 仅基线固定 | 形成统一评估协议 |
| E1 | 从首帧开始的长跨度压力训练 | 已完成 100 epoch | val 选 best | 暴露梯度失败模式；非官方配置 |
| R2 | 官方 20 帧窗口 seed 42 | 已完成 100 epoch | val 选 best | best 精确复现发布权重 |
| E2 | 锚点残差 β 扫描 | 无训练 | val 选 β，test 一次 | 出现 Pareto 改善 |
| E3 | 长跨度压力设定梯度裁剪 1.0 | 已完成 1×100；no-go | val | 未恢复稳定性，停止独立扩展 |
| E4 | 裁剪阈值 0.5/5.0 | 取消 | val | E3 方向性失败，避免无效扫参 |
| E5 | 三随机种子稳定训练 | 延后 | val 汇总 | 仅对出现 val 改善的新目标执行 |
| E6 | `L_pair` 10%/50% 梯度占比 | 已完成；no-go | val | 有后期保护但未胜发布基线 |
| E7 | `L_vel`、`L_rg` 逐项消融 | 2–4×100 | val | 多指标 Pareto 改善 |
| E8 | 可学习锚点门控原型 | 小规模 + 1×100 | val | 优于固定锚点和 E7 |
| E9 | 场景感知 epoch-5 冻结方案 | 已完成 1 次 test | test | T3 改善在 test 复现 |
| E10 | 冻结候选的碰撞代理审计 | 已完成 val + 1 次 frozen test | 不选参 | 配体—蛋白无一致恶化；配体内部信号需键表复核 |
| E11 | 场景条件锚点 `T1/T2/T3=8/8/1` | 已完成 val + 1 次 frozen test；no-go | val 冻结映射，test 不回调 | T1 坐标 +2.295% 超过 2% 防线 |
| E12 | 六维不确定性门控 grouped LOOCV | 已完成 validation OOF；no-go | 10-fold group by complex；未访问 test | balanced accuracy 0.40，低于 0.60 防线 |
| E13 | 化学拓扑可用性审计 | 无训练 | 数据/schema/上游来源 | 可靠键表及来源可追溯，否则保持 Phys 未验证 |
| E14 | bond-aware Phys evaluator | 无训练 | 合成单测 + validation | 覆盖非键碰撞、键长误差和极端事件 |
| E15 | 冻结候选 Phys gate | 无训练 | validation；test 不回调 | 三场景通过预注册 Phys 防线 |
| E16 | Dyn 分布 evaluator | 无训练 | 合成单测 + validation | 能区分真实涨落与 Static/过平滑轨迹 |
| E17 | T1 优先的局部闭环训练实验 | 小预算同预算对照 | validation | T1 Geo/Phys/Dyn 联合改善，T2/T3 不明显回退 |

E13-E15 已完成，`epoch 5 + β=1` 为 Phys no-go。当前优先级更新为 E16 Dyn 分布证据 → 未锚定 epoch-5 与 published NeuralMD 的直接联合比较 → E17 T1 优先小预算创新 → 初赛证据冻结。活动安全基线为未锚定 `seed 42 / epoch 5`；只有 E16 与基础 Phys 都不回退后才能晋升。E3、E6/E6b、E11、E12、E15 的负结果保留为完整消融；初赛截止前不再用现有 validation 调 anchor/门控。E7/E8 转为复赛路线。



## 5. 统一评估与防止数据泄漏

- 超参数和 checkpoint 只由训练集/验证集决定。
- 测试集用于已冻结方案的一次最终评价，不因测试结果回调参数。
- 主表至少包含 coordinate MAE/RMSE、Matching、Stability；项目代理包含 aligned RMSD、Rg MAE、RMSF MAE、contact agreement。
- Phys 初筛可报告不含自环和重复对的碰撞代理，但在没有键表时必须注明成键原子未排除，不能称为化学有效 clash rate 或官方 Phys 分数。
- 报告均给出 10 个复合物的无权均值和逐复合物结果；三 seed 阶段报告均值与标准差。
- 图表必须注明“复现实验代理诊断，不是比赛官方评分”，除非获得官方评分脚本。

## 6. 工程与科研可追溯性

- 数据、checkpoint、预测轨迹放在服务器仓库外；Git 只保存代码、配置、小型 JSON、Markdown 和图表。
- 每个阶段记录：代码提交、上游 commit、数据哈希/划分、命令、环境、GPU、随机种子、运行时间、失败与结论。
- 阶段性提交采用 `reproduce: ...`、`experiment: ...`、`report: ...` 等明确消息；未经验证的实验不标为完成。
- 本设计的执行进度统一写入 `reports/experiment-progress-report.md`。

## 7. 设计自检

- 无待定占位符；每条路线都有可量化成功/失败条件。
- 初赛路线和复赛创新有依赖关系，但 C1 可立即低成本验证，不需要等待新模型。
- 不使用测试集选择超参数。
- 不把没有键表的成对距离约束描述成化学键约束。
- 使用指导手册公开的单场景与三场景权重解释优先级，但在没有官方归一化代码时不构造本地复合总分。
