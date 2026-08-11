# NeuralMD 官方 20 帧窗口配置纠正与精确复现

## 摘要

官方 Hugging Face 超参数包含 `--no_NeuralMD_Binding_start_with_first_frame`。此前命令遗漏该布尔开关，导致 `--NeuralMD_Binding_frame_num=20` 未生效，训练实际从第 0 帧滚动到随机终点。本轮在代码、dry-run 与实际日志三层显式验证开关为 false，重新完成 seed 42、100 epoch。纠正后的 best checkpoint 在统一验证集上与官方发布权重几乎逐位一致，正式建立可信训练 baseline。

## 根因证据链

1. 官方 `hyperparameter.txt`：包含 `--no_NeuralMD_Binding_start_with_first_frame --NeuralMD_Binding_frame_num=20`。
2. 上游训练代码：开关为 true 时固定 `start_traj_idx=0`；只有 false 时才执行 `max(0,end-20)`。
3. 旧日志：`NeuralMD_Binding_start_with_first_frame=True`。
4. 纠正后日志：`NeuralMD_Binding_start_with_first_frame=False`。
5. 回归保护：`tests/test_neuralmd_training_command.py` 要求 canonical command 显式包含 false 开关、窗口 20 和 `--no_MLP_velocity`。

## 配置

| 项目 | 值 |
|---|---|
| 数据/划分 | MISATO-100，80 train / 10 val / 10 test |
| seed / epoch | 42 / 100 |
| batch / optimizer / lr | 8 / Adam / `1e-4` |
| 训练窗口 | 随机终点；最多回看 20 帧；可从轨迹中部开始 |
| ODE | Euler，step 5，scaling 100 |
| 网络 | NeuralMD Binding01，100 radial bases，关闭 velocity refinement/MLP velocity |
| 损失 | 原始位置 MSE |
| 梯度 | 记录全局范数；不裁剪（`max_grad_norm=0`） |
| 上游 commit | `a2ae030838c6ea0251eb6a29bfe99dc9d8ee1cfe` |
| PyTorch | `2.6.0+cu124` |

## 预检失败与处理

1 epoch 训练数值有限，但上游每 5 epoch 才执行验证，结束时却无条件读取 best 数组，因此报 `IndexError`。这是入口脚本的短运行边界缺陷，不是模型失败。未修改上游逻辑，改用最小可评估的 5 epoch 预检；该预检完成 checkpoint 保存且非有限更新为 0。

## 训练与统一验证结果

- 训练位置损失的最大 epoch 均值：20.75661（epoch 41）。
- 最大单 batch 梯度范数：27.010654（epoch 91）。
- 非有限更新：0。
- 按上游验证 coordinate MAE，best 位于 epoch 15。
- 日志中最高验证 Stability 位于 epoch 10，提示 coordinate 与 stability 选择目标并不一致。

| 权重 | Coord RMSE（Å，↓） | Matching（Å，↓） | Stability（%，↑） | Aligned RMSD（Å，↓） | Rg MAE（Å，↓） | RMSF MAE（Å，↓） | Contact（↑） |
|---|---:|---:|---:|---:|---:|---:|---:|
| 官方发布 | 2.4804332 | 0.5741583 | 76.3756 | 0.8958692 | 0.1577934 | 2.3248793 | 0.9614264 |
| 纠正训练 best | 2.4804328 | 0.5741685 | 76.3746 | 0.8958735 | 0.1577995 | 2.3248730 | 0.9614258 |
| 纠正训练 final | 95106.2867 | 146546.2032 | 25.8781 | 103624.8019 | 103622.7072 | 54972.4733 | 0.7321 |

best 与官方发布权重的差异小于数值噪声量级，说明训练复现成功。final 的短窗口训练损失仍正常，但 100 帧 rollout 已发散五个数量级，表明局部监督无法保证长时程闭环稳定。这与比赛指导手册 T1/T2/T3 和 Stab 模块的设计直接吻合。

## 科研决策

1. 正式 baseline 采用纠正训练 best 与官方发布 checkpoint；此前两个实验重新标记为长跨度压力设定。
2. 不使用 final checkpoint，不以训练 MSE 单独选择模型。
3. 在新损失实验前，先建立 T1/T2/T3 场景化验证，分别评估局部、长观测短预测和少观测长预测。
4. 下一训练变量优先为直接对准 Phys/Dyn 的成对距离目标，并保存周期 checkpoint；多 seed 只在单 seed 通过验证后启动。

## 产物与权重

- best SHA256：`fc9092027d05af9a9a40159177d091a94ebef847f5a4dedd9ed0ef1a80bf92ec`
- final SHA256：`d7a2ee0bfa12a63b4fca0d462a39ca37f45d4bbc67bc53cfac34235af75c4362`
- 服务器目录：`/mnt/localDisk3/weizian/runs/protein-quanta/reproduction/neuralmd-misato100-seed42-official-corrected/`
- JSON：`neuralmd_misato100_official_corrected_best_val.json`、`neuralmd_misato100_official_corrected_final_val.json`
