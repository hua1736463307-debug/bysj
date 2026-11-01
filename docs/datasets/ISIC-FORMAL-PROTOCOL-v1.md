# ISIC 正式实验协议 v1（`isic-formal-v1`）

> 状态：**冻结候选（2026-10-31 编制，待人工确认后生效）**。
> 本文件是正式（论文）实验的唯一协议来源；**本轮未执行任何正式训练，也未读取 test 集**。
> 代码侧门禁：`backend/app/domains/isic/experiment_purpose.py`，通过
> `FORMAL_PROTOCOL_ID` / `FORMAL_PROTOCOL_SHA256` 绑定本文件的字节内容。
> 协议一旦冻结，任何修改都必须新建 `isic-formal-v2`，不得原地改写。

## 1. 实验资格（先于一切数值）

| 字段 | 含义 | 取值 |
|---|---|---|
| `experiment_purpose` | 提交训练时显式声明 | `smoke` / `formal`；缺失即"未分类" |
| `research_valid` | 历史字段：产物是否来自真实后端（非 dummy） | 布尔，语义不变 |
| `paper_eligible` | 是否能进入论文汇总/导出 | 由门禁函数计算，不可人工直接置位 |

`paper_eligible=true` 必须同时满足：`experiment_purpose=formal`、`research_valid=true`、
`synthetic!=true`、`backend!=dummy`、`protocol_id=isic-formal-v1` 且
`protocol_sha256` 与本文件一致、`dataset_manifest_sha256` 与第 2 节一致、
`init_weights_sha256` 已记录。

**默认拒绝**：历史记录（含 2026-10-28 的 smoke 模型 `158acb2986604b69`）没有
`experiment_purpose`，按"未分类"处理 → `paper_eligible=false`。因此**不需要改写历史
元数据**，也不会因为漏标而把 smoke 混进论文；smoke 结果仍可在接口查询。

`dummy` / `synthetic` 产物无论声明什么 purpose，永远 `paper_eligible=false`。

## 2. 数据（冻结版本，不可变）

| 项 | 值 |
|---|---|
| dataset_id | `5dcae51a-d374-4c88-9376-0f6563363ee0` |
| dataset_version_id | `87e0eb4f-f112-4992-be69-7a1c57812aa6` |
| manifest_sha256 | `8f5b609c6197792adf7cc71f79236338fd11c47b6fab85bc3fe51d20b2c563ae` |
| split_manifest_hash | `4083c441af21397e18db9ec31e2115600f8de3e4f034df2b4f8f1f9dcfd7f32f` |
| 划分 | 仅用官方 training（10,015 图）按 lesion 分组派生，`split_seed=42`，比例 0.7/0.1/0.1/0.1 |
| 资产数 | 10,015 |

来源 ZIP（`docs/datasets/ISIC-SPLIT-PLAN.md` 第 1 节实测）：

| 文件 | SHA-256 |
|---|---|
| `Training_Input.zip` | `4151b6de5e31eb21586732d2dc0df047737501d5f5333cb52b071ad704364cc0` |
| `Validation_Input.zip` | `a6517ff488d7ac46c799bfe39718ec9ea6bdab78283c6d75e468b511b1d469a8` |
| `Test_Input.zip` | `aa90bd5308259bf5fdcc6048b09dd70416d527d1200e5148aef8ae94f8678b60` |

各集支持数与分组（`artifacts/isic-smoke/freeze-job.json` 落库结果）：

| split | 图数 | 病变组 | mel | nv | bcc | akiec | bkl | df | vasc |
|---|---|---|---|---|---|---|---|---|---|
| train | 7,109 | 5,284 | 788 | 4,768 | 367 | 239 | 773 | 81 | 93 |
| val | 933 | 701 | 102 | 641 | 43 | 27 | 93 | 7 | 20 |
| calibration | 967 | 730 | 108 | 632 | 50 | 26 | 124 | 11 | 16 |
| test | 1,006 | 755 | 115 | 664 | 54 | 35 | 109 | 16 | 13 |

稀有类限制（必须随任何指标一起报告）：

- `df` 在 val 仅 7 张、test 16 张；`vasc` val 20 张。逐类指标对这两类只能作描述，
  不得据单类 F1 下结论；某类 support=0 时该指标 `null` + `reason`，**不允许因此重新抽样**。
- 分组隔离是 **lesion 级**，不是患者级：同一患者的不同病变可能跨集，不能声称"患者独立"。
- 外部官方 validation(193)/test(1,512) 无官方分组表，只能作为"来源不同的图像级对照"，
  **不得称为独立患者外部验证**。

重复与近重复（历史实测，本轮未复跑）：

- 字节级与解码像素级全量比对 11,720 张：重复 2 组 / 4 张，全部落在 training 内部且同属
  一个 lesion（`ISIC_0024366`=`ISIC_0029861`、`ISIC_0024777`=`ISIC_0029938`）；
  跨子集重复 0 组。证据：`docs/datasets/ISIC-SPLIT-PLAN.md` 第 4 节、
  `artifacts/isic-groups/isic-group-audit.json`。
- 近重复：dHash 距离 ≤6 候选 157,078 对（跨子集 38,462 对），距离 0 的跨子集候选 39 对，
  只能作为复核清单，**不构成重复结论**；裁剪/旋转/不同视野的同病变近重复未检测。

## 3. 模型与初始化

| 项 | 值 |
|---|---|
| 结构 | `torchvision.models.densenet121`，`num_classes=7`，softmax 单标签 |
| 类别顺序 | `mel, nv, bcc, akiec, bkl, df, vasc`（`isic2018-task3-v1`，落盘后不得重排） |
| 初始化 | **必须**提供 ImageNet 预训练 backbone（登记为 `isic_backbone`），并在模型元数据记录 `init_weights_sha256` |
| 随机初始化 | 仅允许 `experiment_purpose=smoke`；formal 预检会因缺 `base_model_id` 失败 |
| 预处理 | `isic-preprocess-v1`：EXIF 纠正 → 等比缩放（bilinear，短边到 224）→ 居中灰底填充至 224×224 → `/255` → ImageNet mean/std |
| image_size | 224 |

## 4. 训练预算（预登记，不得临场加大）

| 参数 | 值 |
|---|---|
| epochs | 30（上限，早停实际生效） |
| batch | 16 |
| lr | 3e-4 |
| weight_decay | 1e-4 |
| patience | 8（按 val macro-F1） |
| 数据增强 | 仅随机水平翻转（p=0.5），训练集内 |
| AMP | 训练开启（`amp=true`） |
| num_workers | 0 |
| 设备 | 单卡 `cuda:0` |

## 5. 对照臂

两个臂都按本协议完整训练并报告，**主臂在开训前固定为 `class_weight=balanced`**，
`class_weight=none` 为对照臂；不得在看到 test 结果后调换主臂。

| 臂 | class_weight | seeds |
|---|---|---|
| A（主） | balanced | 42 / 43 / 44 |
| B（对照） | none | 42 / 43 / 44 |

## 6. 模型选择规则

- 选择集：**仅 val**；指标 `macro_f1`（对 split 内真实出现过的类别取平均，与实现的
  `macro_f1_scope` 一致），并列时取更低的 val cross-entropy，再并列取更早 epoch。
- `calibration` 与 `test` 永不参与选择、早停或调参。
- 每个 (臂, seed) 只登记一个 `best_epoch` checkpoint；全部训练历史保留。

## 7. 校准

- 仅用 `calibration` split 拟合；`val` 不参与拟合（避免双重使用）。
- 方法：多类**温度缩放**——对 logits 除以单一标量 T，用 L-BFGS 最小化 calibration 的 NLL；
  不是 CXR 的逐标签 sigmoid 校准。
- 报告校准前后 Brier / NLL / ECE（10 bin，与 `isic-metrics-v1` 实现一致）与 T 值。
- **实现状态：温度缩放尚未实现**（`backend/app/ml/isic_cli.py` 目前只输出未校准概率），
  属 P3 施工项；在实现并通过同 checkpoint 复算前，不得声称"已校准"。

## 8. 评价精度

- 正式评价一律 **fp32**（`eval` 路径不启用 autocast），预处理与训练一致（224、`isic-preprocess-v1`）。
- 训练侧 AMP 与评价侧 fp32 的差异目前只是解释，未经验证：正式实验前必须用**同一
  checkpoint、同一输入**分别以 AMP 与 fp32 推理一次并记录差值，才能归因；否则不得把
  smoke 中 0.001 量级的 val 差异写成"精度导致"。

## 9. test 访问约束

1. 每 `(protocol_id, 臂, seed)` 只允许**一次** test 评价；重复执行必须由管理员在
   `EvaluationRun.purpose` 中写明理由并留痕。
2. test 评价必须由管理员发起，且 `purpose` 字段必须包含 `protocol_id`、`seed`、臂名与授权人。
3. test 结果出来后**不得**回流修改模型、超参、选择规则或类别权重；若必须修改，需新建
   `isic-formal-v2` 并重新预登记，原 test 结果作废并如实记录。
4. 访问记录：`EvaluationRun`（`split=test`、`purpose`、`created_at`）即审计轨迹，导出时一并保留。
5. 本轮状态：**test 从未被读取**（smoke 只用 train/val）。

## 10. 指标与汇总

- 主指标：**macro-F1**（test，主臂）。次要：balanced accuracy、逐类 P/R/F1 + support、
  7×7 混淆矩阵、accuracy。
- 校准：Brier、NLL、ECE（校准前后）。
- 区间：按 **lesion 组件**做 cluster bootstrap（1,000 次，固定重采样 seed 记录在结果元数据），
  CI 的估计对象是"组件（病变）重采样下的指标变异"，**不得**用图像独立重采样冒充。
- seed 变异：报告 3 个 seed 的逐值 + 均值 + 标准差，不与 bootstrap CI 混为一谈。
- 缺类/退化：指标 `null` + `reason`，不因分数不好而重新抽样或改口径。

## 11. 交付与禁止事项

- 每个正式 run 必须记录：`experiment_purpose=formal`、`protocol_id`、`protocol_sha256`、
  `dataset_manifest_sha256`、`init_weights_sha256`、`seed`、`effective_args`、`code_sha`、镜像 ID。
- 禁止：把 smoke 数值写进论文表格；用 test 调参；未过本协议门禁就启动长训练；
  改写已冻结 manifest；把 lesion 隔离说成患者隔离。

## 12. 未验证项（如实列出）

1. 温度缩放未实现（第 7 节）。
2. AMP 与 fp32 评价差异未实测归因（第 8 节）。
3. 近重复只有候选清单，裁剪/旋转近重复未检测（第 2 节）。
4. 外部官方 val/test 无分组，患者级独立不可证明（第 2 节）。
5. 本协议尚未经人工确认冻结；`FORMAL_PROTOCOL_SHA256` 需在确认后重新计算并写入代码常量。
