# 训练态小噪声增强试验：no-go

## 结论

在结果产生前已通过 commit `7143837` 预注册：固定 causal Transformer、200 epoch、seeds `0/42/123`，仅在标准化 12D invariant input 上加入 `σ=0.02` Gaussian noise，预测 clean next state。结果未通过升级门槛。

噪声模型的历史加权代理 RMSE 为 `0.40425 ± 0.02032`，无噪声参考为 `0.40032 ± 0.02126`，平均恶化 `0.00393`（约 `0.98%`），仅 seed 0 改善。T1 平均改善 `1.03%`，但 T2/T3 分别恶化 `3.87%/1.84%`。因此判定 **no-go**，不在看到结果后修改 `σ` 或继续扫描。

## 结果

| 指标 | 无噪声 Transformer | `σ=0.02` 噪声 | 相对变化 | 判定 |
|---|---:|---:|---:|---|
| 历史加权 proxy RMSE | 0.40032 ± 0.02126 | 0.40425 ± 0.02032 | +0.98% | 失败 |
| T1 RMSE | 0.39112 ± 0.01287 | 0.38708 ± 0.01983 | -1.03% | 通过 T1 guard |
| T2 RMSE | 0.36010 ± 0.09780 | 0.37403 ± 0.09874 | +3.87% | 恶化 |
| T3 RMSE | 0.48363 ± 0.01453 | 0.49251 ± 0.02034 | +1.84% | 恶化 |
| 配对 seed 胜数 | — | 1/3 | 要求 ≥2/3 | 失败 |

负变化代表误差改善。所有数值均有限；没有访问 internal test。

## 解释边界

该试验只说明“统一的、标准化特征空间小噪声”不能稳定改善本 proxy 的长程 rollout。它不是原子坐标增强，也没有 E(3) 刚体结构，因此不能否定以下更物理的方案：

- 对整个复合物施加共同旋转/平移；
- 对 ligand/residue block 做可逆刚体扰动并预测恢复量；
- 按 rollout 时真实误差分布校准坐标噪声；
- ProAR 风格 endpoint-intermediate refinement。

科学上更重要的信号是 T1 改善而 T2/T3 恶化：局部 denoising 正则可能降低短程过拟合，但独立同分布噪声没有模拟自回归误差的时间相关结构，不能替代显式抗漂移机制。

## 可复核产物

- 预注册：`configs/temporal_noise_augmentation_preregistration.json`
- 原始逐 seed 结果：`reports/reproduction/temporal_noise_augmentation_val.json`
- 运行入口：`scripts/train_temporal_noise_augmentation.py`
- 参考结果：`reports/reproduction/temporal_multiseed_rope_val.json`

