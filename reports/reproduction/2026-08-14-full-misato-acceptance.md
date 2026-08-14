# 完整 MISATO 数据验收与正式实验解锁

## 结论

完整 `MD.hdf5` 已通过本项目的数据成员关系、更新版比赛划分计数和分层抽样数值完整性门禁，可以取代 MISATO-100 进入正式训练准备阶段。该结论不等同于已经完成同源蛋白/配体骨架泄漏审计，也不代表已经访问比赛隐藏测试集。

## 可复核结果

| 检查项 | 结果 |
|---|---:|
| 本地 HDF5 字节数 | 132,841,014,019 |
| MD5 | **9bc6446922cd80e0f2f3f69349bf88ed（与公开值一致）** |
| HDF5 complex groups | 16,972 |
| 官方原始 train/val/test ID 并集 | 16,972 |
| HDF5 groups 与原始 split 并集 | **完全一致** |
| 排除 NeuralMD 多肽后的 train | **13,066** |
| 排除 NeuralMD 多肽后的 validation | **1,357** |
| 排除 NeuralMD 多肽后的 public test | **1,357** |
| 过滤后 ID 缺失于 HDF5 | **0** |
| 确定性抽样 | train/val/test 各 16，共 48 |
| 抽样 schema/finite 检查 | **48/48 通过** |
| 抽样帧数 | 全部 100 帧 |

HDF5 group ID 集合与官方三份原始 split 并集的 SHA256 均为：

`1e9cbdca2be7162581520317728aad8f133ada1b8115f3db976b8708c9b39132`

这比只核对计数更强：不仅数量相等，而且规范化后的具体 ID 集合相等。

## 抽样设计

没有使用可变随机种子。对每个过滤后的 split 按规范化 ID 排序，固定选择首尾与等距分位点，共 16 个；抽样 ID 集合 SHA256 为：

`5995a5f1fb6ba195a75bd565a07a1ac65eb14cb8a7ffd6f390de056bebdec256`

对 48 个样本检查：

- 六个必需字段是否存在；
- 坐标是否为 `(frames, atoms, 3)`；
- 原子属性、能量数组的长度是否与轨迹一致；
- 配体起点是否合法；
- 以 25 帧为块读取全部 100 帧，坐标是否全部有限；
- 配体及全体系原子数量是否非空。

抽样覆盖 1,320–27,836 个全体系原子、13–106 个配体原子。坐标观测范围为 -26.461–149.439 Å；未发现 NaN/Inf。

## 仍未完成、不能过度声称的部分

1. **48 个样本的 finite 检查不是 16,972 个体系的逐坐标穷举。** 全量成员关系已穷举；数值数组采用确定性分层抽样，避免在 MD5 已全盘读取一次后再次读取约 132 GB 全部数据。
2. **不能从此 HDF5 单独完成严格配体 scaffold 审计。** HDF5 有原子类型、残基类型和坐标，但没有化学键图或 SMILES。Bemis–Murcko scaffold 必须补充可追溯 PDB/CCD 化学图；不得用元素组成或原子类型直方图冒充 scaffold。
3. **同源蛋白审计尚未完成。** `atoms_residue` 能帮助恢复残基类型序列，但仍需明确链边界、序列提取规则与固定阈值的聚类实现，并将结果只用于泄漏审计/补充 robustness split，不得改写官方主评测协议。

## 决策影响

- MISATO-100 diagnostics 自此继续冻结，只保留为开发历史证据。
- 正式模型比较必须使用过滤后的 `13,066/1,357/1,357` 数据入口；模型选择仅使用 validation。
- public test 只在候选模型冻结后访问一次，不据其调参。
- 下一阶段先完成训练侧的流式索引/缓存与一个小规模 full-data pipeline smoke；在此基础上再做 bounded damping 对照。这样可以把“架构改进”与“数据加载/单位错误”分开。

## 复现

```bash
PYTHONPATH=. python scripts/audit_full_misato_acceptance.py \
  --h5 /path/to/MISATO_full/raw/MD.hdf5 \
  --train .external/misato-dataset/data/MD/splits/train_MD.txt \
  --val .external/misato-dataset/data/MD/splits/val_MD.txt \
  --test .external/misato-dataset/data/MD/splits/test_MD.txt \
  --peptides .external/NeuralMD/NeuralMD/datasets/MISATO/utils/peptides.txt \
  --samples-per-split 16 \
  --chunk-frames 25 \
  --observed-md5 9bc6446922cd80e0f2f3f69349bf88ed \
  --expected-md5 9bc6446922cd80e0f2f3f69349bf88ed \
  --output reports/reproduction/full_misato_acceptance.json
```

机器可读证据见 `reports/reproduction/full_misato_acceptance.json`。
