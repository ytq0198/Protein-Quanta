# 完整 MISATO 流式数据与 GPU 训练接口门禁

## 结论

完整 MISATO 已首次通过从官方 train split、NeuralMD peptide 排除、132 GB HDF5 流式解析，到 bounded + normalized velocity-aware 模型单 batch 前向/反向的端到端门禁。此次只访问第一个过滤训练样本，不访问 validation 或 public test。

该结果证明“正式数据能够进入当前候选模型并产生有限非零梯度”，但同时表明 dense ligand–protein 计算若逐样本直接扩展到 13,066 个体系，时间成本不可接受；正式 paired effect gate 前必须先做稀疏邻域化与吞吐门。

## 最终结果

| 项目 | 结果 |
|---|---:|
| 过滤后 train 计数 | **13,066** |
| 首样本 | `5WIJ` |
| 配体重原子 | 31 |
| 蛋白骨架原子 | 822 |
| 轨迹帧 | 100 |
| 候选模型参数 | 10,243 |
| 单步 acceleration Smooth-L1 | 0.857021 |
| 参数梯度范数 | **0.001287（finite, nonzero）** |
| dataset 构造 + 首 batch | 1.901 s |
| GPU 前向 + 反向 | 7.934 s |
| 峰值新增 CUDA 显存 | 23,481,856 bytes（约 22.4 MiB） |

所有门禁均通过：过滤计数一致、数据与模型输出 finite、梯度 finite 且非零。

## 实现决策

上游 `DatasetMISATOSemiFlexibleMultiTrajectory` 是 `InMemoryDataset`，会先解析整个 split 并写入 PyG 缓存。为避免在截止日前因全量预处理、峰值内存或巨型缓存失败，本项目新增 `StreamingMISATODataset`：

- 复用 NeuralMD 官方 `parse_MISATO_data`，不另造不同的原子/残基预处理口径；
- split ID 统一大写并显式排除官方 NeuralMD `peptides.txt`；
- HDF5 handle 在每个进程首次访问时延迟打开，序列化/worker 创建时不携带 handle；
- 样本 ID 保存在 dataset 索引表，不注入 PyG tensor fields；
- 使用 NeuralMD 的 `DataLoaderMISATO` 生成其模型约定的 `batch_ligand` 与 `batch_residue`。

## 失败与修复记录

1. 首次运行在 CUDA peak-memory reset 处失败：PyTorch 2.6 环境不接受该处传入的 `torch.device` 形式。改为显式 `set_device(integer_index)` 后解决。
2. 第二次运行发现通用 PyG DataLoader 不生成 NeuralMD 约定字段。替换为上游 `DataLoaderMISATO`。
3. 第三次运行发现字符串 `sample_id` 被 PyG collate 当成 tensor 拼接。将 ID 移出 Data 对象、保留在 dataset 索引表。
4. 最终同协议运行通过。以上失败都发生在接口/设备层，没有被包装成模型效果证据。

## 扩展性判断

单个体系的一次前向/反向约 7.93 秒。即使忽略数据读取、优化器、多时间尺度 rollout 与评估，按 13,066 个样本顺序执行一次也约需 28.8 小时；真实训练会更长。因此：

- **不启动** 当前 dense 版本的全量 epoch；
- 下一门是将 ligand–protein 全连接交互替换为固定半径/Top-k 邻域，同时保持 E(3) 等变与 velocity invariants；
- 在 32 个过滤 train 样本上比较 dense 与 sparse 的单步输出偏差、forward/backward 时间、显存、finite/梯度；
- 只有 sparse 吞吐至少提升 5 倍且相对加速度误差可控，才进入 bounded damping 的多 seed paired effect gate；
- public validation/test 继续不访问。

机器可读证据见 `reports/reproduction/full_misato_stream_smoke.json`。
