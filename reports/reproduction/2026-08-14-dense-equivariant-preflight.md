# 稠密径向等变动力学 10-complex preflight

> 状态：运行性通过；8/10 显式拓扑覆盖子集 Phys 门槛通过；不是效果 gate。  
> 数据边界：冻结 development 的 10 个 train 复合物；不访问 validation/test。  
> 决策边界：允许设计 64/16 train-only 效果 gate，不代表候选有效或可提交。

## 1. 预注册与样本

预注册提交：`c453756`（首次 runtime）与 `a06c3b8`（配对 Phys）。十个 ID 为冻结 development 清单的前十个，并按原 `train_MD.txt` 处理顺序映射到索引 `0–8,10`。训练均使用 seed 42、5 epoch、batch 1、Adam `1e-4`、Euler `0.025`、final epoch。

首次 runtime 候选为 `L_local + 0.25 L_multiscale`，50 个更新全部有限，梯度裁剪 `0/50`，蛋白坐标不变，峰值 CUDA 分配约 `334 MiB`。final checkpoint SHA256 为 `bbc23e9b35058ae121b1bef3861eec9f7971ccd2ac52ebdf6e4e7dba9a899bb0`。

## 2. 拓扑覆盖

RCSB PDB 显式 `CONECT` 经过重原子元素顺序、frame-0 内部距离和全原子键覆盖检查：

- 通过：`5KBQ, 5X27, 4DRU, 6JAW, 6F3B, 1W5X, 1ZSH, 1V2S`；
- 未匹配/不完整：`3SNC, 3MP6`；
- 覆盖率：`8/10 = 80%`。

未匹配样本不按距离猜键、不参与 Phys 均值，并作为覆盖缺口报告。

## 3. 配对 Phys preflight

首次配对尝试因直接索引 PyG 样本缺少 DataLoader 创建的 `batch_ligand/batch_residue` 而在训练前失败，没有产生 checkpoint。提交 `2d250b2` 只补单复合物全零 batch 向量，从头重跑。

控制与候选从完全相同初始化出发，共用预生成的 batch 顺序、局部窗口和多尺度窗口。控制闭环系数为 0，候选为 0.25。两者各 40 个训练更新均无非有限值、无梯度裁剪。

final checkpoint 从 frame 0 推演 40 帧：

| 指标（8 样本均值） | 控制 | 候选 | 候选−控制 |
|---|---:|---:|---:|
| coordinate RMSE | 1.854857929 | 1.854857840 | -8.94e-8 |
| bond-length MAE (Å) | 0.037807966 | 0.037807946 | -1.98e-8 |
| extreme bond event (%) | 0.0 | 0.0 | 0.0 pp |

预注册 Phys 防线为 extreme event 增量不超过 `0.1` 个百分点，故通过。控制/候选 checkpoint SHA256 分别为 `331522e8d97557e0d75ec04d72251fc83d5164cc2da263f30dd6e764b440306e` 与 `97de91af6062dcdb409d165493d3134a10986eaad17f8683062e7e01e6c8c95d`。

## 4. 科研解释

该结果证明候选在小预算下可训练、显存安全、蛋白固定，并未新增极端键长事件；但候选与控制几乎数值相同，**没有多尺度效果证据**。原因可能是加速度场初始化尺度小、5 epoch 更新幅度不足，或局部 MSE 主导。不能据此调大闭环权重，因为 `0.25` 已冻结且小样本 preflight 不用于选参。

下一阶段若执行 64/16 三 seed gate，必须保持权重 `0.25`、Euler `0.025` 和 final epoch，并直接判断 T1/T2/T3 坐标、T3 误差斜率、RMSF/Rg/接触、键长与碰撞。若依然无效，则该轻量 EGNN 路线 no-go，转向预训练 PaiNN/NequIP/EPT，而不是在 holdout 上扫尺度。

机器证据：

- `reports/reproduction/evidence/dense-equivariant-preflight.json`
- `reports/reproduction/evidence/dense-equivariant-preflight-topology.json`
- `reports/reproduction/evidence/dense-equivariant-paired-phys.json`
