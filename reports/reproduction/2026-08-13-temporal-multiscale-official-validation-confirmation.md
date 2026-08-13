# 多时间尺度闭环：一次性官方 validation 机制确认通过

## 访问与冻结协议

该确认仅在 train-only 64/16 实验完整 gate 通过后启动。模型仍只使用 64 个 development 复合物训练；16 个 train-holdout 和 10 个官方 validation 均不进入训练或归一化。普通一步 Transformer 与多时间尺度候选按相同模型、200 epoch、final epoch、seeds `0/42/123` 从头重训，并先把全部 6 份 final checkpoint 保存到服务器；之后代码才统一读取官方 validation 一次。未访问 internal test。

冻结候选仍使用 horizons `[5,10,20,40]`、闭环权重 `0.25`，没有因 train-only 结果或 validation 结果改动超参数，也不按 validation 选择 epoch。

## 结果

| 指标 | 一步训练 Transformer | 多时间尺度闭环 | 相对变化 |
|---|---:|---:|---:|
| 等权宏平均 RMSE | 0.40821 ± 0.00600 | **0.39609 ± 0.00757** | **-2.97%** |
| T1 RMSE | 0.41029 | **0.40243** | **-1.92%** |
| T2 RMSE | 0.30894 | **0.30868** | -0.08% |
| T3 RMSE | 0.50540 | **0.47716** | **-5.59%** |
| 宏平均配对 seed 胜数 | — | 2/3 | 要求 ≥2/3 |

T3 的逐 seed 配对差为 `-0.02885/-0.05013/-0.00573`，三个种子全部同向改善；强门槛 5% 也通过。T1 防线、总体均值、种子胜数、T3 方向与有限值均通过，因此一次性 confirmation 为 **pass**。

## 机制结论

train-only holdout 的 T3 改善为 `8.42%`，官方 validation 一次确认仍为 `5.59%`，方向跨划分一致。固定 10 步闭环此前只改善 T3 `0.46%`，而 `[5,10,20,40]` 多时间尺度训练超过 5%，支持“训练暴露的时间尺度覆盖决定长程 rollout”的机制解释。T2 在确认中近乎持平，说明收益集中于 T1 与 T3，不应夸大成全任务普遍提升。

## 当前创新定位与下一步

多时间尺度可微闭环是当前唯一同时通过 train-only gate 和一次性 validation confirmation 的创新机制。它可以进入三维等变原子坐标模型的设计阶段，但尚不能替换 published NeuralMD 主基线：现有输入是 12 维轨迹不变量，未生成配体原子坐标，也未直接接受 Geo/Phys/Dyn/Stab 评估。

三维迁移必须满足：蛋白固定、配体坐标 E(3) 等变；一步原子损失与 `[5,10,20,40]` 闭环损失配对消融；bond-aware Phys、Dyn 分布与 Stab 同时检查；不访问 internal test 选择模型。初赛只能将本结果描述为“机制可行性证据与明确的后续实现路线”，不能写成官方得分或完整算法成绩。

## 权重归档（不进入 Git）

服务器目录：`/mnt/localDisk3/weizian/checkpoints/protein-quanta/multiscale-confirmation/`

| 权重 | SHA256 |
|---|---|
| baseline seed 0 | `ab3f8cf4421d85304aa78a472edc89e01b4a340207404b23e3d6a10c5a61f07b` |
| baseline seed 42 | `e0c8c5340a90a10808572d0f840428437fc5cff8da94cff3db21b4e5e2d2019f` |
| baseline seed 123 | `b539598b0dce48968ccc76845ad26cced88493855f5d0bba96478eb3907f0f47` |
| multiscale seed 0 | `aafc6cfcfe6c176c633c159c680a617d5070c0a86f9a4b512139c92e88107db4` |
| multiscale seed 42 | `749c48597f3f6935618dae434f26c2c192b364790cf6e317df9c7f6efb30d36e` |
| multiscale seed 123 | `870d326a1bcde5f89d0e0c81463a2fbc35fb45e3d28074997b1d0dc6fb738c2b` |

原始报告：`temporal_multiscale_official_validation_confirmation.json`。
