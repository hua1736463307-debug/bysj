# ISIC 真实数据 1 epoch GPU smoke（2026-10-28）

> **这不是论文性能结果。** 1 epoch、随机初始化、单 seed，只用于证明
> "数据 → 冻结四集 → GPU 训练 → checkpoint → 评价" 这条工程链路可用。
> 任何数值都不得进入论文表格，也不构成模型有效性证据。

## 1. 环境与代码

| 项 | 值 |
|---|---|
| 项目 | `bysj-drill2-f004-20261027`（隔离项目，运行项目未参与） |
| 代码 | `63bff5f4da7cade2a5962653997f59965445d094` |
| 镜像 | `bysj-api:isic-63bff5f` = `sha256:5104dd8b849d6ffe572cdf2bf3866fed943cae3244fd878c033f8bb912bb802b`<br>`bysj-worker:isic-63bff5f` = `sha256:a9ef8869eddba8eb715a903454dcc3cd49899c6dd055042b54a7ed838c50b2b7` |
| GPU | RTX 4060 Laptop 8188 MiB，容器内 `cuda_available=true`、`device_count=1` |
| 运行时 | torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / CUDA 12.4 / python 3.12.14 / numpy 2.4.4 |
| 显存峰值 | 1.211 GB（`peak_gpu_allocated_bytes=1,211,269,632`） |

## 2. 数据

官方训练集（10,015 张）导入后按 lesion 分组派生四集（`split_seed=42`，70/10/10/10）：

| split | 图数 |
|---|---|
| train | 7,109 |
| val | 933 |
| calibration | 967 |
| test | 1,006 |

- `class_gate` 通过，`rare_class_warnings` 已记录；
- **训练与评价只读 train / val**；calibration 与 test 未参与本次 smoke，也未用于调参；
- 数据来源 SHA-256 见 `docs/datasets/ISIC-SPLIT-PLAN.md`。

## 3. 训练

| 参数 | 值 |
|---|---|
| epochs | 1 |
| batch | 16 |
| image_size | 224 |
| lr / weight_decay | 3e-4 / 1e-4 |
| class_weight | balanced |
| amp | true |
| seed | 42 |
| device | cuda:0 |

结果：

| 项 | 值 |
|---|---|
| 作业状态 | `completed`，`best_epoch=1` |
| train_loss | 1.691221 |
| val accuracy | 0.501608 |
| val macro-F1 | 0.256190 |
| val balanced accuracy | 0.313120 |
| val macro precision | 0.262454 |
| val top-1 ECE | 0.074524 |
| checkpoint | `model_id=158acb2986604b69bc67d463f559bce3`，sha256 `607d39dd914fd7d8fc67514ff6818ce800fa124b862d7afb8d0d7837be444c6f` |
| 元数据 | `research_valid=true`（真实训练，非 dummy）、`label_schema=isic2018-task3-v1`、`class_order` 七类固定、`preprocess_schema=isic-preprocess-v1`、`dataset_manifest_sha256` 与 `seed` 均已落库 |

## 4. 评价（val，未读 test）

独立评价作业（重新加载 checkpoint 推理）：

| 项 | 值 |
|---|---|
| split / 样本数 | val / 933 |
| accuracy | 0.500536 |
| macro-F1 | 0.254263 |
| 混淆矩阵 | 7×7 已落库 |

逐类 F1：

| 类 | support | F1 | 备注 |
|---|---|---|---|
| mel | 102 | 0.414201 | |
| nv | 641 | 0.687747 | 多数类，1 epoch 即可学到 |
| bcc | 43 | 0.179104 | |
| akiec | 27 | 0.100719 | |
| bkl | 93 | 0.311111 | |
| df | 7 | **0.0** | reason=`zero_precision_and_recall`（有真值但一次都没预测对） |
| vasc | 20 | 0.086957 | |

> 评价指标与训练时的 val 指标有 0.001 量级差异，原因是训练时用 AMP 半精度前向、
> 评价时从 checkpoint 重新以 fp32 推理。**两者都不是论文指标。**

## 5. 取消与租约释放（同一环境）

提交一个 2 epoch 训练，运行中取消：

| 检查 | 结果 |
|---|---|
| 取消前 | `status=running`，`gpu:0` holder = `trainer-026033b063f3…` |
| 取消接口 | 202 |
| 终态 | `cancelled` |
| 租约 | `gpu:0` holder 变为空 |
| 部分模型 | 该作业登记模型数 = **0**（不会留下半成品） |
| 子进程 | 取消后容器内无 `isic_cli` 进程；GPU 显存回到系统基线 |

**过程中发现并修复一个真实缺陷**：`ctx.should_stop()` 在收到取消时是**抛 `JobCancelled`**
而不是返回 `True`，导致 `child_runner` 里"写 stop.flag → SIGTERM → 宽限"的优雅终止分支
**永远不会执行**，子进程被 `finally` 里的 SIGKILL 直接杀掉。
已在 `63bff5f` 之后修正为显式接住异常并走优雅路径，并新增
`backend/tests/test_child_runner.py`（3 项）固定该行为。

> **口径说明**：上面的取消测试跑在修正前的镜像（`63bff5f`）上，所以观测到的是
> "直接 SIGKILL、无 stop.flag" 的旧行为；修正后的优雅路径目前只有单元测试证据
> （假进程断言 `terminate()` 被调用、`kill()` 未被调用、stop.flag 已写），
> **尚未在真实 GPU 作业上端到端重验**——需要下一次按最终提交重建镜像后再跑一次。

## 6. 结论

| 项 | 结论 |
|---|---|
| 数据 → 导入 → 冻结 | PASS（10,015 张，四集非空，类别门禁通过） |
| GPU 训练 | PASS（真实 DenseNet121，1 epoch，显存 1.21 GB，无 OOM） |
| checkpoint | PASS（含 arch/schema/class_order/image_size/state_dict，可被适配器重新加载） |
| 评价链路 | PASS（逐类 P/R/F1 + macro-F1 + 混淆矩阵 + ECE 全部落库） |
| 取消与租约 | PASS（终态 cancelled、租约释放、无半成品模型） |
| 论文性能 | **不适用**——1 epoch 随机初始化，不得引用 |

## 7. 未覆盖

1. 多 epoch 收敛、学习率调度、早停、类别不平衡策略对比。
2. calibration 集拟合（温度缩放）与 test 集评价（本次明确未读）。
3. 多 seed 重复与置信区间。
4. 外部留出集（官方 val/test）评价——因无分组，只能作图像级对照。
5. 断点续训、跨机迁移、更大 batch 的显存与吞吐。

## 8. 研究资格归类（2026-10-31 补记，不改写任何历史测量值）

本节只是在本记录上补充元数据说明，原始数值与结论保持原样。

| 项 | 值 |
|---|---|
| `experiment_purpose` | `smoke`（真实数据、真实 GPU，但 1 epoch、随机初始化、单 seed） |
| `research_valid` | `true`（非 dummy 产物；历史字段语义不变） |
| `paper_eligible` | **`false`**：smoke 不具备论文资格，即使 `research_valid=true` |
| 原因码 | `experiment_purpose_smoke`；历史模型记录无 purpose 时另为 `experiment_purpose_missing` |

含义：`research_valid=true` 只说明"不是 dummy"，**不说明能进论文**。自 2026-10-31 起由
`backend/app/domains/isic/experiment_purpose.py` 统一计算 `paper_eligible`，
`GET /api/v1/isic/results/paper-summary` 会把这些 smoke 记录列在 `excluded` 且不返回指标；
smoke 结果仍可在 `/models`、`/evaluations` 查询。正式实验须通过
`docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md` 的冻结协议门禁。
