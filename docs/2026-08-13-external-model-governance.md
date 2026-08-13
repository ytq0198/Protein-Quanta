# 外部预训练模型合规边界

新版方向二允许使用预训练模型和外部 MD 数据，但明确要求不得包含测试复合物或近同源体系。由此，是否“能下载、能运行”不能作为可用标准。本项目采用默认拒绝策略：许可、逐体系训练清单或重叠审计任一未知，就不得进入正式候选。

当前核查结论：

| 资源 | 许可 | 精确训练 ID | 正式候选 | 当前允许用途 |
|---|---|---|---|---|
| PVB | MIT | 未公开/未取得 | 阻塞 | 架构与接口研究；权重需审计后再议 |
| EPT | checkout 未见 LICENSE | 未取得 | 阻塞 | block denoising/E(3) 思想参考 |
| dynamics-aware DPLM | MIT | 未取得 | 等待审计 | 全量 split 落地后研究 frozen conditioning |
| ProTDyn | MIT | 未取得 | 初赛前延期 | 多时间尺度/inpainting 设计参考 |
| ESTAG | checkout 未见 LICENSE | 不适用/未核实 | 阻塞 | 论文级等变时空设计参考 |

特别说明：已下载的 `pvb_misato.ckpt` 和 `pvb_pdbbind.ckpt` 只作为接口与来源审计材料保存在 Git 外。MISATO 权重很可能直接见过公开 MISATO 体系，PDBBind 权重也存在复合物重叠风险；二者均不进入正式 validation/test。未完成的 `pvb_pretrain` 文件既不是有效 checkpoint，也不能绕过训练 ID 缺失问题。

机器可检查的仓库修订、LICENSE 哈希、权重哈希与阻塞原因保存在 `configs/external_model_resource_manifest.json`。后续若取得训练清单，审计顺序固定为 exact complex ID、蛋白序列近同源、配体 scaffold/similarity；通过前不加载正式候选权重。
