# 新版 T1 协议纠正与历史证据降级

## 纠正结论

新版指导手册第 20 页明确规定：T1 给定前 10 帧、预测后 10 帧；T2 为 80→20；T3 为 20→80。

项目代码此前沿用了旧版对“T1 短程预测”的实现性解释：观察帧 0–1、预测 2–19（2→18）。这不是新版的 10→10。T2/T3 没有偏差。现已把统一场景定义纠正为：

| 场景 | 观察 | 初始化末两帧 | 目标 |
|---|---|---|---|
| T1 | 0–9 | 8–9 | 10–19 |
| T2 | 0–79 | 78–79 | 80–99 |
| T3 | 0–19 | 18–19 | 20–99 |

## 影响范围

凡调用 `competition_scenarios()` 生成的旧 T1 数字，都必须标为“历史旧协议 T1-2→18”，不能用于新版指导手册对齐的评分论证。包括此前的 NeuralMD 场景基准与早停、anchor、碰撞、bond-aware Phys、dynamics distribution、长序列架构、噪声与 ProAR 式机制实验。

这些实验对 T2/T3、实现正确性、失败机制和工程复现仍有研究价值，但任何跨 T1/T2/T3 加权或“当前候选”结论均失效。旧 JSON 和报告保留以维持审计链，不改写历史原始数据。

## 即时安全措施

1. `configs/active_candidate.json` 暂时设为 `active_for_submission=false`，直到新版 T1 validation 重跑完成。
2. 旧权重从候选清单降级为历史非官方字段；新版说明精确权重随复赛评测材料发布。
3. 场景单元测试固定检查 T1 initializer `8,9` 与 target `10..19`。
4. 后续先重算 validation，不因新 T1 结果访问或回调 internal test。
5. 初赛材料只引用纠正后的数字；历史研究过程必须明确指出协议纠正。

## 重跑顺序

1. Static、Linear、published NeuralMD 与 epoch-5 checkpoint 的新版 T1 validation；
2. 时序 MLP/Transformer 多种子参考与已完成的 RoPE/noise/ProAR 机制；
3. bond-aware Geo/Phys/Dyn/Stab 代理只对仍有竞争力的候选重算；
4. 所有候选依据新版 T1/T2/T3 validation 重新决定，不继承旧候选身份。
