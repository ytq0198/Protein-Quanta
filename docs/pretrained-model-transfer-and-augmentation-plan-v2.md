# 方向二预训练模型、抗漂移与数据增强路线 v2

> 依据：2026-08-13 新版指导手册、最新初赛模板、公开论文与官方代码。  
> 目标：同时服务竞赛可验证性与科研意义；先过泄漏/许可证/任务适配门槛，再看模型规模。

## 1. 优先级结论

| 候选 | 任务接近度 | 可直接运行性 | 最大价值 | 主要阻塞 | 当前决策 |
|---|---:|---:|---|---|---|
| ProAR | 5/5 | 2/5 | 双网络概率预测与 anti-drifting sampling，直接针对长程漂移 | 未找到官方代码/权重；ATLAS 是蛋白自身而非配体轨迹；每轮 Amber relaxation 极慢 | **A：先复现机制，不复制整模** |
| PVB | 5/5 | 3/5 | 静态结构预训练后用有限轨迹微调；官方含 MISATO 推理入口 | PDB/PDBBind/MISATO 权重存在测试重叠风险；原始训练数据不公开；需要完整 topology | **A-：仅 pretrain 权重通过审计后迁移** |
| EPT | 4/5 | 3/5 | 30M+ E(3) 等变 3D encoder；block-level rotation/translation denoising | 5.89M 预训练集含 PDB/PDBBind；仓库未发现 LICENSE；不是动力学模型 | **B：优先借鉴架构/增强，权重暂缓** |
| ProTDyn | 3/5 | 2/5 | 多时间尺度 Transformer、动态 inpainting | 1.4B、约 16GB 权重、依赖 gated ESM3；蛋白构象而非蛋白-配体；1-100 ns 与本题帧尺度不匹配 | **B-：读设计，不在初赛前迁移** |
| dynamics-aware DPLM | 3/5 | 3/5 | 只需蛋白序列即可提供动态感知条件 embedding | 输出蛋白级/残基级静态表示，不预测配体轨迹；是 2026 预印本 | **B：可作低成本条件特征消融** |

说明：此处 DPLM 指 Yuexu Jiang 等人的 dynamics-aware Protein Language Model，而不是 ByteDance 的 Diffusion Protein Language Model。

## 2. 为什么 ProAR 机制先于大模型迁移

ProAR 把一个长跨度 `h` 的生成拆成两个角色：

1. interpolator 在当前帧与候选未来帧之间预测带结构协方差的中间帧；
2. forecaster 用越来越接近未来的中间状态反复修正跨度末端；
3. 交替插值与预测，降低单次外推误差在自回归 rollout 中累积。

这与本项目已观测到的“20 帧局部训练无法约束 100 帧闭环、步幅欠动力学”直接对应。初赛前最合理的低成本试验不是复刻全部 SE(3) 模型，而是在当前 NeuralMD 上做单变量机制验证：

- 冻结或复用同一 NeuralMD forecaster；
- 对 `h=4` 或 `h=8` 的 endpoint forecast，加入一个轻量中间帧校正器；
- 训练时只使用 train 轨迹的端点与内部帧；
- 与同 seed、同 epoch、同计算量的普通 autoregressive rollout 比较；
- 先看 T3 drift、位移幅度与 Dyn 分布，同时用 T1 Geo/Phys 防线约束。

完整 ProAR 的 Amber relaxation 不作为首轮方案：公开论文的后续基准显示结构 relaxation 可占大部分推理时间，不适合先做主线。

## 3. PVB/EPT 迁移学习的合规门

新版允许预训练模型，但禁止外部资源包含测试集复合物或近同源体系。PVB/EPT 都与 PDB/PDBBind 有数据关系，因此必须满足：

1. 获得预训练数据的精确 PDB/complex ID 清单；
2. 获得比赛 train/validation/test 的 PDB ID 与去多肽 split；
3. 做 exact ID、蛋白序列 identity、配体 scaffold/similarity 三层重叠审计；
4. 预先确定阈值并保存 manifest；
5. 若权重无法“反训练”移除冲突数据，则只可作为不可用于正式候选的研究对照。

### PVB

- 优点：论文任务设计最接近“单结构预训练 → 稀缺轨迹微调”；MIT；代码直接有 `infer_complex` 与 MISATO 路径。
- 限制：作者未公开原始训练数据；`pvb_misato.ckpt` 显然已经在 MISATO 微调，不能在未知 split 重叠下用于正式验证；`pvb_pretrain.ckpt` 也需查明预训练 ID。
- 决策：只下载和审计 `pretrain` 权重；`misato` 权重仅用于接口复现/上界研究，禁止作为当前合规候选。

### EPT

- 优点：E(3) 等变；atom/block 双层表示；对 block 做随机旋转和平移并恢复，和小数据增强高度相关。
- 限制：预训练含 600k PDB 条目和 22,295 PDBBind 口袋；GitHub 当前 checkout 未发现 LICENSE 文件；下游是 affinity/property，不是轨迹预测。
- 决策：重写/借鉴“block denoising + 等变 encoder”思想；在许可证与 ID 审计解决前不复制代码、不加载正式候选权重。

## 4. ProTDyn 与 DPLM 的正确定位

### ProTDyn

ProTDyn 是大规模蛋白热力学/动力学生成模型，包含 Transformer、多时间步与 inpainting，但模型约 1.4B，依赖 ESM3，首次权重约 16GB，且其动力学输入输出没有蛋白-配体口袋条件。优先提取三条设计思想：

- timestep conditioning；
- coarse trajectory + inpainting 的多尺度生成；
- 结构与序列联合条件。

若后续进入复赛，可把它作为蛋白动态先验或多尺度课程设计，而不是直接预测本题配体坐标。

### dynamics-aware DPLM

DPLM 通过序列与 MD trajectory embedding 对比学习，把动态信息蒸馏进蛋白序列 embedding。最可行的消融是：

- 冻结 DPLM；
- 提取每个蛋白的 per-sequence/per-residue embedding；
- 作为不随时间变化的条件输入加入 ligand dynamics model；
- 与同架构、不含 DPLM 条件的模型多 seed 比较。

它可能提升“未见蛋白”泛化，但不会自行解决 ligand drift，也不能替代 3D pocket geometry。

## 5. 数据增强优先级

### A 级：可在 train-only 上严格验证

1. **状态噪声/denoising**：对输入坐标或位移施加小 Gaussian noise，target 保持 clean；缓解 teacher forcing 与 rollout 状态分布错配。先在标准化状态上预注册 `sigma=0.02` feasibility，再升级到坐标空间的物理尺度噪声。
2. **E(3) 刚体变换**：对整个复合物做相同随机旋转和平移；非等变模型可获数据增益，等变模型用于严格 equivariance 测试。不能只旋转配体而不旋转静态蛋白口袋。
3. **随机时间窗与跨度课程**：从 train trajectory 抽取 T1/T2/T3 对齐窗口，并显式输入 horizon/Δt；避免只训练 20 帧。
4. **block-level 扰动恢复**：借鉴 EPT，对配体块或蛋白残基块施加小刚体扰动，训练等变 encoder 恢复平移/旋转；只用官方 train。

### B 级：后续研究

- masked-frame/temporal inpainting 自监督预训练；
- 多尺度 coarse-to-fine rollout；
- ProAR 风格 endpoint + intermediate refinement；
- 只在完整 topology 可用时加入 bond-aware projection/relaxation。

### 拒绝或谨慎

- 不做无物理依据的 trajectory MixUp；
- 不做简单 time reversal，除非速度、热浴和条件全部正确变换；
- 不把验证/测试轨迹用于自监督预训练；
- 不以强 anchor/平滑换取低 RMSD 而牺牲 Dyn；
- 不在同一轮同时改 encoder、loss、noise、sampling，避免无法归因。

## 6. 预注册实验队列

| ID | 单变量 | 数据 | seeds | 主观察 | 通过条件 |
|---|---|---|---|---|---|
| A1 | causal Transformer + train-state noise `σ=0.02` | MISATO-100 train/val feasibility | 0/42/123 | weighted proxy、T1/T2/T3 | 平均优于无噪声且 ≥2/3 seeds；T1 不恶化 >2% |
| A2 | NeuralMD + short anti-drift refinement | MISATO-100 train/val | 0/42/123（先 42 preflight） | T3 drift、Dyn、T1 safeguards | T3 至少两类指标改善，T1 Geo/Phys 无明显回退 |
| A3 | DPLM frozen conditioning | 全量 train 内去泄漏 val | ≥3 | 未见蛋白泛化 | 对无 DPLM 同预算对照稳定改善 |
| A4 | EPT/PVB encoder transfer | 仅审计通过的数据/权重 | ≥3 | 全四维代理 | 权重来源与 split overlap audit 通过后才可运行 |

## 7. 可形成的科研主线

“面向未见蛋白-配体复合物的多尺度等变抗漂移动力学模型”：静态等变结构先验学习局部化学与口袋几何；causal temporal block 学历史依赖；endpoint-intermediate refinement 抑制长程 drift；train-only denoising 暴露模型于 rollout 状态；Geo/Phys/Dyn/Stab 联合门槛阻止塌缩。

这条线同时具有：

- 竞赛意义：直接覆盖 T1/T2/T3、物理合法性和长时稳定；
- 科研意义：检验预训练结构先验与抗漂移采样如何共同改善未见复合物的轨迹泛化；
- 可证伪性：每一模块都有同预算、同 seed 的单变量消融和明确 no-go 条件。

## 8. 来源与版本留档

- ProTDyn official GitHub：`Harrydirk41/ProTDyn`，本地审计 commit `94efb78...`，MIT。
- ProAR：AAAI 2026 论文；截至本次检索未找到作者官方代码仓库，因此只按论文重实现机制。
- PVB official GitHub：`yaledeus/PVB`，本地审计 commit `c08e5e3...`，MIT。
- EPT official GitHub：`jiaor17/EPT`，本地审计 commit `481a550...`；checkout 未发现 LICENSE，需作者/仓库进一步澄清。
- dynamics-aware DPLM official GitHub：`yuexujiang/DPLM_release`，本地审计 commit `f3e5c3b...`，MIT。
- ESTAG official GitHub：`GLAD-RUC/ESTAG`，checkout 未发现 LICENSE；仅用于论文级架构参考。

