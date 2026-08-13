# NeuralMD 新版 T1/T2/T3 validation 纠正重评

## 协议

本次重评在协议修正 commit `aece457` 后运行，报告内部场景元数据已核对：T1 `0–9 → 10–19`，T2 `0–79 → 80–99`，T3 `0–19 → 20–99`。数据为 MISATO-100 validation 10 个复合物；未访问 internal test。指标为项目代理，不是官方分数。

比较对象：

- published NeuralMD，checkpoint SHA256 `364404a7ce61ec1180fa3800c0dcbddf377b672d22f915fc6b19d202710a4a5a`；
- 纠正训练的 seed 42 / epoch 5，SHA256 `0e7d5150aa5f305499f17663d3a74b1063b0591733f534e676303ce11de50b8c`；
- Static 是统一评估器内的最后观测帧恒定对照。

## 结果

| 场景 | 方法 | 坐标 RMSE Å ↓ | Matching Å ↓ | Stability % ↑ | RMSF MAE Å ↓ | RMSE 斜率 Å/帧 ↓ |
|---|---|---:|---:|---:|---:|---:|
| T1 | published | 1.3880 | **0.5251** | **82.2833** | **1.2581** | **0.07455** |
| T1 | epoch 5 | 1.3881 | 0.5252 | 82.2726 | 1.2581 | 0.07458 |
| T1 | Static | **1.3605** | 0.6026 | 81.0103 | 1.2901 | 0.07857 |
| T2 | published | **1.3510** | **0.4284** | **83.3124** | **1.2887** | **0.04700** |
| T2 | epoch 5 | 1.3513 | 0.4287 | 83.2887 | 1.2892 | 0.04704 |
| T3 | published | **2.2139** | 0.4763 | 82.0841 | **2.1020** | **0.01501** |
| T3 | epoch 5 | 2.2165 | **0.4477** | **84.1025** | 2.1196 | 0.01509 |

epoch 5 相对 published：T1/T2 所有变化在约 `0.00–0.05%` 或 `0.024` Stability 点内；T3 Matching 改善 `6.01%`、Stability 增加 `2.0185` 点，但坐标 RMSE、RMSF 与误差斜率分别恶化 `0.117%`、`0.841%`、`0.543%`。

## 决策

1. 新版 T1 下，Static 仍能获得略低的坐标 RMSE，但 NeuralMD 明显改善 Matching、Stability、RMSF 与误差斜率。不能仅用坐标误差选模型，必须保留动态与塌缩对照。
2. epoch 5 的 T3 几何/稳定收益仍存在，但并非所有维度一致改善；新版手册未公开权重，因此不能重新声明它综合优于 published。
3. published NeuralMD 是当前严格可复现主基线；epoch 5 保留为多指标研究候选。`active_for_submission` 继续为 false，直到形成基于新版协议的明确候选规则。
4. 旧 T1-2→18 报告不改写，但不再进入新版得分证据。

## 产物

- `neuralmd_updated_guide_published_val.json`，SHA256 `af25150eacb7a34b473415746d1f965590db39dd000e8779f892225070e4de09`
- `neuralmd_updated_guide_epoch5_val.json`，SHA256 `da32c5adbe49242a69e5e49a0c154602c1c730cc72a768e5c6b018a667a71809`
