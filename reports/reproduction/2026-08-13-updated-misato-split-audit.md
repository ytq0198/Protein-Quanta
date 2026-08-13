# 新版 MISATO 划分与多肽排除审计

## 结论

新版指导手册给出的 `13,066 / 1,357 / 1,357` 不是重新随机切分得到的数字，而是对 MISATO 官方公开 train/validation/test 划分逐 ID 排除 NeuralMD `peptides.txt` 后的精确结果。本次审计没有读取轨迹坐标，也没有访问比赛隐藏评测数据。

| 划分 | MISATO 原始 ID | 排除多肽 | 保留 ID | 重复行 | 与其他划分交集 |
|---|---:|---:|---:|---:|---:|
| train | 13,765 | 699 | **13,066** | 0 | 0 |
| validation | 1,595 | 238 | **1,357** | 0 | 0 |
| test | 1,612 | 255 | **1,357** | 0 | 0 |

`peptides.txt` 含 1,431 个互异 ID；其中 1,192 个出现在三份 MISATO 原始划分中，另外 239 个不在这些划分内。三个划分过滤后仍完全互斥。新版指导手册的三个计数均通过自动检查。

## 可复现性

- MISATO 划分来源修订：`7b06d532e2ed0719411fcc1b3ac39743db4ca10d`
- NeuralMD 多肽列表来源修订：`a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe`
- 原始文件、规范化保留集合和排除集合的 SHA256 已写入 `updated_misato_split_audit.json`。
- 仓库只保存计数与哈希，不复制外部数据集的 ID 列表。

复现命令：

```bash
PYTHONPATH=. python scripts/audit_updated_misato_splits.py \
  --train /path/to/misato-dataset/data/MD/splits/train_MD.txt \
  --val /path/to/misato-dataset/data/MD/splits/val_MD.txt \
  --test /path/to/misato-dataset/data/MD/splits/test_MD.txt \
  --peptides /path/to/NeuralMD/NeuralMD/datasets/MISATO/utils/peptides.txt \
  --output reports/reproduction/updated_misato_split_audit.json
```

## 对正式实验的约束

1. 全量训练、验证和后续可公开测试必须以该过滤规则为唯一入口，不能使用 MISATO 原始计数直接训练。
2. 模型选择只使用 validation；公开 test 在候选冻结后访问一次，不能据其回调超参数。
3. MISATO-100 仍只属于 feasibility proxy，不可冒充上述全量划分结果。
4. PVB、EPT 等外部预训练权重必须另做复合物 ID 与近同源体系泄漏审计；本审计仅证明比赛基础划分自身互斥。
