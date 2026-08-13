# 新版协议训练态噪声增强实验：no-go

## 协议

在看到新版协议结果前已冻结：普通 causal Transformer、200 epoch、seeds `0/42/123`、最终 epoch，不按 validation 选择 checkpoint；仅在训练期标准化 12 维不变量输入上加入 `σ=0.02` Gaussian noise，目标仍是 clean next state。T1/T2/T3 为新版 `10→10 / 80→20 / 20→80`，未访问 internal test。

`σ=0.02` 继承自旧协议预注册值，没有根据新版结果调参。由于官方场景权重尚未公布，主指标为三场景等权宏平均。

## 结果

| 指标 | 无噪声 Transformer | `σ=0.02` 噪声 | 相对变化 |
|---|---:|---:|---:|
| 等权宏平均 RMSE | **0.41778 ± 0.03279** | 0.42426 ± 0.03190 | +1.55% |
| T1 RMSE | 0.40961 | **0.40623** | -0.82% |
| T2 RMSE | **0.36010** | 0.37403 | +3.87% |
| T3 RMSE | **0.48363** | 0.49251 | +1.84% |
| 配对 seed 胜数 | — | 0/3 | 要求 ≥2/3 |

负变化代表改善。三个噪声模型的宏平均均差于同 seed 无噪声参考，所有数值有限。

## 决策与机制解释

实验判定 **no-go**，不继续扫描噪声强度。独立同分布的小扰动能轻微正则化 T1，却没有模拟自回归误差的时间相关累积，因此同时伤害 T2/T3。这将创新问题进一步收敛为：训练目标需要直接暴露于自身预测形成的闭环状态，并用多步误差约束 drift；仅做输入噪声或仅改变推理调用顺序都不够。

该结论仍仅适用于 12 维机制代理，不能否定原子坐标上的刚体旋转/平移、局部构象扰动或等变 denoising。

## 可复核产物

- 预注册：`configs/temporal_noise_augmentation_updated_guide_preregistration.json`
- 原始逐 seed 结果：`temporal_noise_augmentation_updated_guide_val.json`
- 无噪声参考：`temporal_multiseed_rope_updated_guide_val.json`
- 运行入口：`scripts/train_temporal_noise_augmentation.py`
