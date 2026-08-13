# 方向二新版执行计划（2026-08-13 起）

## 总原则

初赛材料只需要证明：问题理解正确、路线合理、已有可复核验证、后续工作可执行且合规。距离 8 月 16 日很近，不用 1.4B 模型或 133GB 全量训练制造“看起来很大”的结果；先冻结小规模严谨证据，同时启动全量数据准备作为复赛 readiness。

## 工作包与顺序

### P0：新版合规与材料边界（立即完成）

- 新旧手册差异、模板字段、AI 禁用作答边界：已审计。
- 把旧权重公式全部标为历史口径，不作为当前官方评分。
- 所有 MISATO-100 结果统一标为 feasibility / internal proxy。
- 外部预训练模型建立数据来源、许可证、权重、精确 ID/近同源审计表。
- 团队人工完成最新 Word 模板正文；项目提供证据索引与事实核验。

### P1：初赛可展示的阶段验证（8 月 13-14 日）

1. 固化 NeuralMD 复现与关键失败机制：短训练窗口、100 帧闭环 drift、静态塌缩风险。
2. 固化 RNN/LSTM/GRU/Transformer/RoPE 三 seed 结果：普通 Transformer 只获得“长程机制有效”的结论，不称完整 3D 改进。
3. 完成 train-state noise 单变量增强试验。
4. 若时间允许，完成 ProAR anti-drift 极小型 preflight；否则只保留设计和预注册，不编造结果。
5. 更新图表、原始 JSON、复现命令、负结果与限制。

### P2：全量数据 readiness（8 月 13 日开始，后台进行）

- 从 Zenodo 获取 132.8GB `MD.hdf5`，核对官方 MD5 `9bc6446922cd80e0f2f3f69349bf88ed`。
- 获取 `train_MD.txt`、`val_MD.txt`、`test_MD.txt` 与 NeuralMD `peptides.txt`。
- 生成去多肽 split，并强制计数 13,066/1,357/1,357。
- 审计 HDF5 ID 覆盖、100 帧规范、有限坐标、atom/residue schema。
- 构建 train 内 protein-sequence + ligand-scaffold 去泄漏 validation；官方 validation 仅用于阶段反馈。
- 不在初赛前承诺完成全量模型训练。

### P3：预训练迁移（复赛 readiness）

按以下门槛顺序：

1. PVB pretrain 数据 ID/近同源审计；不使用 `pvb_misato.ckpt` 作为合规候选。
2. EPT 权重和代码许可证审计；若训练 ID 无法排除重叠，转为 train-only block-denoising 自监督。
3. DPLM 动态条件 embedding 的冻结消融。
4. ProTDyn 只在 P0-P3 其他项完成且时间/显存允许时做接口研究。

### P4：正式创新主线

等变空间编码器 + causal temporal Transformer + ProAR-style anti-drift refinement + train-only denoising。每个模块依次加入，至少三 seed；同时报告 Geo/Phys/Dyn/Stab 原始代理与 T1/T2/T3，正式 evaluator 发布后再替换。

## 三人分工

| 成员 | 8 月 13-16 日主责 | 复赛 readiness |
|---|---|---|
| 魏子安 | 新版材料事实核对；人工撰写模板；Git/实验总协调；全量数据状态 | 模型集成、实验决策、提交管理 |
| 熊润 | 全量 MISATO 下载/切分/预处理；A6000 训练；PVB/EPT 工程兼容 | 等变模型、分布式训练、性能优化 |
| 耿健尧 | 论文证据表、结果图表人工复核、模板合规/引用/许可证检查 | benchmark 整理、消融表与复现审核 |

魏子安与熊润均可在工作日做代码与实验，因此关键路径采用双人互备；耿健尧不承担会阻塞训练的唯一任务。

## 初赛前停止条件

- 不因全量下载未完成而延迟材料冻结；如实写“全量训练待开展”。
- 不把未审计预训练权重写为已使用方案。
- 不把 proxy RMSE 写成官方得分。
- 不用 test/validation 结果反复调参。
- 不由 AI 填写最终提交模板正文。

