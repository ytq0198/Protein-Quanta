# 多时间尺度闭环的 E(3) 等变三维迁移设计

> 状态：基于 train-only gate 与一次性 validation confirmation 后冻结的下一阶段设计；尚未声称三维实验完成。
> 边界：内部科研设计与实验门槛，不是初赛答卷。最终提交文字由团队成员独立完成。

## 1. 已被数据支持的创新假设

现有证据已经区分了四种可能机制：

1. causal Transformer 相对 MLP 的新版等权宏平均改善 `12.43%`，说明时间注意力有独立信号；
2. RoPE 没有超过普通 Transformer，排除“只换位置编码即可解决长序列”的解释；
3. iid 训练噪声和推理期交替修正均失败，说明间接模拟 exposure bias 不足；
4. 固定 10 步闭环主要修复 T2，而 `[5,10,20,40]` 多时间尺度闭环在 train-only T3 改善 `8.42%`，一次性 validation T3 确认改善 `5.59%`。

因此三维创新不定义为“换一个更大的 Transformer”，而定义为：**在 E(3) 等变的配体动力学模型中，让训练期自身预测暴露覆盖局部、中程和长程时间尺度，并用物理合法性门槛阻止长期 rollout 通过坍缩或非物理构象获取表面误差收益。**

## 2. 最小三维候选 E18

### 2.1 空间与时间分工

- 空间编码沿用已精确复现的 NeuralMD BindingNet/ODE 路线，保留蛋白—配体、多粒度和 E(3) 等变性；不直接把展平坐标送入普通 Transformer。
- 蛋白在输入与所有预测帧中固定，只更新配体重原子位置，严格符合新版 semi-flexible 设定。
- 第一版不引入未经数据/许可证审计的 PVB/EPT/DPLM 权重，避免把机制因果证据与预训练收益混在一起。
- 时间创新只改变训练 rollout：复现审计确认 NeuralMD 基线本身已用 ODE 对随机局部片段进行连续可微训练，并非普通的一步 teacher forcing。公平对照 B 保留原始随机 1–20 帧位置损失；候选 C 在同一空间网络上额外加入等概率 `[5,10,20,40]` 时间跨度的 ODE rollout 损失，其他参数、数据和训练预算配对。

### 2.2 可微闭环

对每个候选 batch 从 `H={5,10,20,40}` 均匀抽取 horizon `h`，再在合法范围内均匀抽取 prefix。由观测位置与相邻帧速度初始化二阶 ODE，并连续积分 `h` 步；ODE 状态天然使用自身预测连续演化，中间状态不得 `detach`。这不是在既有 NeuralMD 外再套一层虚假的自回归，而是把原先最多 20 帧的局部时间覆盖扩展到 40 帧。采用梯度 checkpointing 控制显存时不得截断跨步梯度。

冻结总损失：

`L = L_local-ODE + 0.25 * L_multiscale-ODE + λ_bond * L_bond + λ_clash * L_clash`

- `L_local-ODE`：与精确复现基线一致的随机 1–20 帧配体坐标 MSE；
- `L_multiscale-ODE`：四种 horizon 均按原子、时间步和 xyz 维平均的坐标 Smooth-L1，避免长 horizon 仅因项数多而获得更高权重；
- `L_bond`：只使用可追溯的显式共价图，不按距离猜键；以相对 frame-0 键长变化的 Smooth-L1 约束闭环帧；
- `L_clash`：排除共价图 1-hop/2-hop 后的非键软排斥，加上配体—固定蛋白的软排斥；
- `λ_bond/λ_clash` 不在 E18 主实验前随意指定。先在 train-only 的 8-complex calibration 子集上，以各损失对共享层梯度范数达到位置损失 `5%/2%` 为确定性标定目标，然后冻结；该标定不访问 16-complex holdout 或官方 validation。

### 2.3 为什么不是简单集成

创新变量具有明确的反事实对照：

- A：published NeuralMD checkpoint；
- B：相同代码从头训练的原始随机 1–20 帧 ODE 位置损失；
- C：B + `[5,10,20,40]` 多时间跨度可微 ODE；
- D：C + 显式 bond/clash safety loss。

只有 `B→C` 能检验扩大时间跨度 exposure 的因果收益；`C→D` 检验物理门槛是否减少极端事件，而不是把所有变化归因于 Transformer、预训练或更多参数。初赛阶段可以展示 A/B 与低维机制证据，不能把尚未运行的 C/D 写成结果。早期文档中“NeuralMD 是一步 teacher forcing”的表述已被源码审计否定，禁止继续使用该因果叙述。

## 3. 数据与访问协议

### 当前 MISATO-100 pilot

- 沿用已冻结的 64 development / 16 train-holdout，不改变 salt；
- 官方 validation 只允许在 E18 train-only gate 完整通过后进行一次确认；
- internal test 禁止用于选择模型、horizon、损失权重或 epoch；
- 所有归一化和物理阈值只从 development 计算。

### 全量 MISATO

全量 `MD.hdf5` 完成后必须先通过：精确字节数 `132,841,014,019`、MD5 `9bc6446922cd80e0f2f3f69349bf88ed`、HDF5 可读、去多肽划分 `13,066/1,357/1,357`、ID 覆盖与有限坐标审计。任何一项未通过时不启动全量训练。

全量开发划分必须按 protein sequence cluster 与 ligand scaffold 联合分组；当前 64/16 hash split 只能用于 MISATO-100 机制 pilot，不能冒充结构分组 OOD 评估。

## 4. 分阶段执行与停止条件

### E18a：实现正确性（1 个复合物）

必须全部通过：

- horizon 5/10/20/40 前向与反向有限；
- 随机共同旋转/平移后，预测坐标按相同变换变化，最大等变误差 `<1e-4 Å`；
- 蛋白坐标逐帧位级不变；
- horizon=1 且闭环权重=0 时，与一步训练路径参数梯度最大差 `<1e-6`；
- 中间预测没有 detach，首步状态能接收来自末步误差的非零梯度。

任一失败则停止训练，先修实现。

**2026-08-14 源码审计修订：** 上述 `horizon=1` 检查保留为组合损失的零权重单元测试，但真实基线一致性必须以“候选权重为 0 时不额外采样随机数、训练入口和原随机 1–20 帧 ODE 路径完全相同”为准。原因是 NeuralMD 原训练目标已经是局部多步 ODE，而非一步模型。真实 E18a 使用发布权重和一个 train 复合物，对 5/10/20/40 帧分别执行前向、参数反向、末帧到初始位置反向、固定蛋白与共同刚体变换检查。

### E18b：10-complex preflight

运行 seed 42、5 epoch，仅 development 数据。停止条件：非有限 loss/gradient；显存溢出；梯度裁剪触发率超过 50%；极端键长事件相对一步对照增加超过 `0.1` 个百分点；蛋白发生移动。

该阶段只判断可运行性，不从中选 horizon 或权重。

### E18c：64/16 三 seed train-only gate

固定 final epoch，不按 holdout 挑 checkpoint。相对配对一步三维模型必须同时满足：

- 三场景等权 coordinate proxy 改善；
- 至少 2/3 seed 宏平均胜出；
- T1 coordinate RMSE 不恶化超过 2%；
- T3 coordinate RMSE 改善至少 3%，且 RMSE 随时间斜率改善至少 5%；
- T3 RMSF/接触/Rg 分布至少两项改善，防止静态坍缩；
- 平均键长 MAE 不恶化超过 5%，极端键长事件不增加超过 `0.05` 个百分点；
- 非键碰撞和配体—蛋白碰撞均不过预注册防线。

完整通过后才允许一次官方 validation 确认。未通过则保留负结果，不访问 test、不围绕 validation 扫参。

## 5. 与比赛四维评价的对应

| 设计组件 | 主要作用 | 必须联合检查的反作弊指标 |
|---|---|---|
| 多时间尺度闭环 | Geo、Stab | T3 误差斜率、有限帧比例 |
| 等变 BindingNet/ODE | Geo、跨复合物泛化 | 旋转/平移等变误差 |
| 显式 bond loss | Phys | 键长 MAE、极端事件，不只看平均值 |
| 非键/蛋白 clash loss | Phys | 排除 1/2-hop 后的碰撞与合法帧比例 |
| 多 horizon 与动态分布审计 | Dyn | RMSF、Rg、接触分布、速度自相关 |

不构造本地伪总分。新版权重未知时逐场景、逐维报告原始值、均值/SD、配对差和失败项。

## 6. 工程实现建议

- 在 NeuralMD 的训练封装外增加 `rollout_loss.py`，不直接侵入上游第三方源码；
- 使用 gradient checkpointing 包裹每个积分区间；horizon 40 若 A6000 显存不足，先减 batch 并做梯度累积，不能改变 horizon 集合；
- 三 seed 可分配 GPU 0/1/2，第 3 张用于 baseline 或评估；启动前动态检查占用；
- 每个 checkpoint 保存代码 commit、数据/划分哈希、seed、epoch、配置和 SHA256；
- 原子坐标、全量数据与权重不进 Git，仓库只保存配置、指标 JSON、图表和复现入口。

## 7. 当前最诚实的项目结论

已经证明的是：多时间尺度闭环在小规模低维机制代理上跨 train-only 与一次 validation 都改善 T3，并且优于固定 10 步、噪声、RoPE 和推理期交替修正。尚未证明的是：它能在原子坐标层面同时改善 Geo/Phys/Dyn/Stab，或在全量 MISATO 与隐藏评测上取得更高成绩。

下一阶段的科研价值正在于检验这条机制能否从低维不变量迁移到等变原子动力学，而不是把低维提升直接包装成比赛模型结果。
