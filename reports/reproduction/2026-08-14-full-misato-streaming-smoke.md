# 完整 MISATO 流式数据与 GPU 训练接口门禁

## 结论

完整 MISATO 已首次通过从官方 train split、NeuralMD peptide 排除、132 GB HDF5 流式解析，到 bounded + normalized velocity-aware 模型单 batch 前向/反向的端到端门禁。此次只访问第一个过滤训练样本，不访问 validation 或 public test。

该结果证明“正式数据能够进入当前候选模型并产生有限非零梯度”。其中 7.934 s 是首次 CUDA 前反向，不能作为稳态训练吞吐；后续预注册 warm-up 基准测得 dense 为 0.009288 s/前反向，已撤回基于冷启动值的 28.8 小时/epoch 外推。详见 `2026-08-14-sparse-velocity-throughput-gate.md`。

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

## 扩展性判断（后续证据纠正）

首次 7.93 s 主要是 CUDA 冷启动。后续 3 次 warm-up + 10 次同步计时测得 dense 为 9.29 ms/前反向；Top-k 16/32/64/128 反而只有 dense 的 0.70–0.86×，稀疏 gate no-go。因此当前保留 dense；正式 rollout 训练的实际成本仍需用真实 horizon 另测，不能再由单步冷启动或单步稳态直接外推。

机器可读证据见 `reports/reproduction/full_misato_stream_smoke.json`。
