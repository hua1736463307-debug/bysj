# ISIC 2018 Task 3 数据核验与四集划分方案

> 目的：在冻结任何四集之前，先回答"分组是否可靠、重复是否跨集、两种划分方案哪个可用"。
> 结论：**推荐「仅用官方训练集按 lesion 分组派生四集」**，官方 validation/test 降级为外部留出集。
> 依据脚本：`scripts/audit_isic_groups.py`（分组与重复）、`scripts/prepare_isic.py`（标签审计）。

## 1. 下载与完整性

官方 S3 直链，经 7890 代理下载；无账号、无协议。

| 文件 | 字节 | 官方 Content-Length | 一致性 | 条目数 | `unzip -t` |
|---|---|---|---|---|---|
| `Training_Input.zip` | 2,771,732,744 | 2,771,732,744 | 一致 | 10,018 | 无错误 |
| `Validation_Input.zip` | 53,292,381 | 53,292,381 | 一致 | 196 | 无错误 |
| `Test_Input.zip` | 420,586,171 | 420,586,171 | 一致 | 1,515 | 无错误 |
| `Training_GroundTruth.zip` | 36,681 | — | — | 1 CSV + LICENSE + ATTRIBUTION | 已解压 |
| `Validation_GroundTruth.zip` | 7,619 | — | — | 同上 | 已解压 |
| `Test_GroundTruth.zip` | 12,202 | — | — | 同上 | 已解压 |
| `Training_LesionGroupings.csv` | — | — | — | 10,015 行 | 已读取 |

SHA-256：

| 文件 | SHA-256 |
|---|---|
| `Training_Input.zip` | `4151b6de5e31eb21586732d2dc0df047737501d5f5333cb52b071ad704364cc0` |
| `Validation_Input.zip` | `a6517ff488d7ac46c799bfe39718ec9ea6bdab78283c6d75e468b511b1d469a8` |
| `Test_Input.zip` | `aa90bd5308259bf5fdcc6048b09dd70416d527d1200e5148aef8ae94f8678b60` |

> 同一份哈希也记录在审计 JSON 的 `subsets.*.sha256`。三个 ZIP 均通过 `unzip -t`，无错误。

## 2. 标签与图像一致性

| 子集 | 标签行 | ZIP 内图像 | 缺图 | 无标签图 |
|---|---|---|---|---|
| training | 10,015 | 10,015 | 0 | 0 |
| validation | 193 | 193 | 0 | 0 |
| test | 1,512 | 1,512 | 0 | 0 |

每行**恰好一个** `1.0`（脚本强校验，出现零个或多个直接报错）；表头固定为
`image,MEL,NV,BCC,AKIEC,BKL,DF,VASC`，不重排。

类别分布：

| 子集 | NV | MEL | BKL | BCC | AKIEC | DF | VASC |
|---|---|---|---|---|---|---|---|
| training (10,015) | 6,705 | 1,113 | 1,099 | 514 | 327 | 115 | 142 |
| validation (193) | 123 | 21 | 22 | 15 | 8 | **1** | 3 |
| test (1,512) | 909 | 171 | 217 | 93 | 43 | 44 | 35 |

## 3. 分组信息核查

| 项 | 结果 |
|---|---|
| 官方分组文件覆盖 | **仅 training**：10,015 行 = 10,015 图，覆盖率 **100%** |
| 病变组数 | **7,470** |
| 每组图数 | 1 张 5,514 组；2 张 1,423；3 张 490；4 张 34；5 张 5；6 张 4 |
| 最大组 | 6 张 |
| validation / test 分组 | **官方未提供** |
| validation/test 图像出现在 training 分组表中的数量 | **0 / 0** |
| 子集间图像 ID 重叠 | training&validation 0、training&test 0、validation&test 0 |
| 确认方式 | histopathology 5,340；serial imaging 3,704；expert consensus 902；confocal 69 |

结论：**只有训练集能按病变分组**。任何把官方 validation/test 当成"独立测试集"的说法，
都只能声称"图像级不重叠"，**不能声称患者级/病变级独立**。

## 4. 重复图像风险

字节级 SHA-256 全量比对（三个子集共 **11,720** 张图像，100% 覆盖）：

| 项 | 结果 |
|---|---|
| 重复组 | **2 组 / 4 张**，全部在 training 内部 |
| 组内关系 | `ISIC_0024366` = `ISIC_0029861`（同属 lesion `HAM_0002300`，同类 histopathology）<br>`ISIC_0024777` = `ISIC_0029938`（同属 lesion `HAM_0001895`，同类 histopathology） |
| 跨子集重复 | **0**（train&val、train&test、val&test 均为 0 组） |
| 最大重复组 | 2 张 |

**关键结论**：两对重复都落在**同一个 lesion 内**，按 lesion 分组切分时会被整体分到同一集，
不会造成跨集泄漏。这正面支持第 5 节的方案 B。

**限制**：本检查只识别字节完全相同；重新编码、裁剪、旋转后的同病变图像**不会被发现**。
因此不能声称"已排除所有重复"，只能说"未发现字节级重复，且已知重复均在同组内"。

## 5. 两种四集方案对比

### 方案 A：直接使用官方三个子集

| split | 图数 | NV | MEL | BKL | BCC | AKIEC | DF | VASC | 分组 |
|---|---|---|---|---|---|---|---|---|---|
| train | 10,015 | 6,705 | 1,113 | 1,099 | 514 | 327 | 115 | 142 | lesion（100%） |
| val | 193 | 123 | 21 | 22 | 15 | 8 | **1** | 3 | 无 |
| test | 1,512 | 909 | 171 | 217 | 93 | 43 | 44 | 35 | 无 |
| calibration | **0** | — | — | — | — | — | — | — | — |

问题：
1. **没有 calibration 集**——温度缩放与 Youden 阈值只能从 val 拟合，而 val 同时被当作模型选择集，会双重使用。
2. **val 只有 193 张，DF 仅 1 张**——任何基于 val 的小类指标（DF、VASC=3、AKIEC=8）都不可用。
3. **val/test 无分组**——只能声称图像级不重叠；且 1,512 张 test 的病变可能与 train 共享患者（同一患者不同病变），无法排除。

### 方案 B：仅用训练集按 lesion 分组派生四集（70/10/10/10，`split_seed=42`）

| split | 图数 | 病变组 | NV | MEL | BKL | BCC | AKIEC | DF | VASC |
|---|---|---|---|---|---|---|---|---|---|
| train | 7,068 | 5,293 | 4,786 | 781 | 781 | 330 | 210 | 81 | 99 |
| val | 965 | 706 | 617 | 116 | 108 | 69 | 30 | 11 | 14 |
| calibration | 981 | 725 | 641 | 108 | 97 | 51 | 52 | 17 | 15 |
| test | 1,001 | 746 | 661 | 108 | 113 | 64 | 35 | 6 | 14 |

> 分配由 `assign_groups()` 的 `sha256(namespace|seed|group)` 决定；实际 `namespace` 含数据集 ID，
> 因此导入后具体数字会有少量浮动，但量级与类别覆盖不会变。上表用 `isic:planning-drill` 模拟。

优点：
1. **四集都有病变分组**——同一 lesion 只进一集，患者级隔离在训练集范围内**可证明**（同一 lesion 的 6 张图不会跨集）。
2. **有 calibration 集**——温度缩放与阈值只从 calibration 拟合，与 val 分离。
3. **val 覆盖小类**：DF 11、VASC 14、AKIEC 30，远好于官方 val 的 DF=1。
4. 已发现的两对字节重复天然落在同组，不会跨集。

代价与限制：
1. **放弃 1,705 张外部图像**（官方 val 193 + test 1,512）作为训练数据——但它们可作为**外部留出集**单独报告。
2. 四集各约 10%，test 仅 1,001 张、DF 6 张 → 小类指标的置信区间很宽，必须报告区间，不得只报点估计。
3. 该方案下"患者级独立"仅在**训练集内部**成立；外部留出集仍只有图像级证据。

## 6. 推荐

**主方案 B**：仅用官方训练集，按 `lesion_id` 分组派生 train / val / calibration / test（70/10/10/10，`split_seed=42`）。

**官方 validation + test 作为外部留出集**，用途严格限定：

- 只用于最终**一次**外部一致性报告，不参与训练、模型选择、温度拟合、阈值选择；
- 报告中必须标注"外部集无病变分组，仅图像级不重叠"；
- 若外部集表现与内部 test 差异明显，如实报告，不调参重跑。

**不推荐方案 A** 作为主方案，原因是没有 calibration 集、官方 val 的 DF 仅 1 张、且 val/test 无分组。

## 7. 冻结前的门禁（必须全部满足）

1. 三个图像 ZIP 均通过 `unzip -t`，字节数与官方一致，SHA-256 已记录。
2. 标签与图像 1:1，无缺图、无多正类（已满足）。
3. 方案 B 的四集在每个 split 上记录类别分布与病变组数，写入 manifest。
4. 类别门禁：train 每类 ≥ 1 个正例（已满足）；val/calibration/test 的小类（DF、VASC）在摘要中显式标注样本量。
5. manifest 记录来源、版本、许可证（CC BY-NC 4.0）、文件摘要、`split_seed`、分组级别。
6. 报告与页面固定注明：**皮肤镜专用、非商业许可、分类概率不等于确诊**。

## 8. GPU 训练 smoke 计划

### 阶段 A：导入与冻结（CPU，可立即执行）

```bash
# 1) 解压三份图像到数据根（保留官方目录名）
cd D:/毕业设计/datasets/isic2018
unzip -q -o ISIC2018_Task3_Training_Input.zip -d .
unzip -q -o ISIC2018_Task3_Validation_Input.zip -d .
unzip -q -o ISIC2018_Task3_Test_Input.zip -d .

# 2) 让服务认识该只读来源（部署环境变量，不写进仓库）
#    DATASET_SOURCES={"isic2018-local":"D:/毕业设计/datasets/isic2018"}

# 3) 导入（管理员，Idempotency-Key 必需）
curl -X POST "$API/api/v1/isic/datasets/import" -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: isic-import-1" -H 'Content-Type: application/json' \
  -d '{"source_id":"isic2018-local","split":"train","max_images":null}'

# 4) 冻结四集
curl -X POST "$API/api/v1/isic/datasets/$DATASET/versions" -H "Authorization: Bearer $TOKEN" \
  -H "Idempotency-Key: isic-freeze-1" -H 'Content-Type: application/json' \
  -d '{"split_seed":42,"ratios":[0.7,0.1,0.1,0.1]}'

# 5) 读摘要（类别门禁、分组计数、稀有类警告）
curl "$API/api/v1/isic/datasets/$DATASET/versions/$VERSION/summary" -H "Authorization: Bearer $TOKEN"
```

验收：`counts` 四集非空；`class_counts` 与第 5 节表一致（允许 namespace 导致的浮动）；
`group_level=lesion`；`class_gate.passed=true`；`rare_class_warnings` 如实列出 DF/VASC。

### 阶段 B：GPU smoke（**当前阻塞**）

**阻塞原因**：平台目前只有 `isic_import` / `isic_freeze` 两类作业，
**没有 ISIC 训练、评价、报告链路**；`cxr_adapter` 固定五路 sigmoid 并明确禁止 softmax，
不能直接复用为七类单标签。因此"在平台上跑 ISIC GPU smoke"目前**无法执行**。

两条可选路径：

| 路径 | 内容 | 产出性质 |
|---|---|---|
| B1（推荐） | 实现 `isic_train`：softmax 适配器（7 输出）、多类交叉熵、逐类指标（P/R/F1、macro-F1、混淆矩阵、逐类 ECE）、`isic_train`/`isic_evaluate` 作业与 API | 平台功能，可作为论文流程 |
| B2（临时） | 平台外脚本：读冻结 manifest，DenseNet121 7 类训练 1–2 epoch，只验证数据与 GPU 通路 | **不入平台、不产指标、不得引用** |

无论哪条路径，smoke 的验收条件是**工程性**的，不是性能性的：

1. 训练在 GPU 上完成（`nvidia-smi` 可见进程，`torch.cuda.max_memory_allocated` 有值）；
2. 1–2 epoch 内 loss 有限且下降趋势，无 NaN/Inf；
3. 产出 checkpoint 文件，记录 SHA-256 与 arch/label_schema/image_size；
4. 单卡显存峰值 < 8 GB（RTX 4060 Laptop 8188 MiB）；
5. 作业可被正常取消并释放 GPU 租约；
6. 全流程不读 test/外部留出集。

**smoke 结果一律不得作为论文性能指标**：它是 1–2 epoch 的工程连通性验证，
不构成收敛、不构成可比结果，也不进入任何汇总表。

## 9. 已完成与未完成

**数据侧已完成**：三个图像 ZIP 全部下载、字节数与官方一致、`unzip -t` 无错误、SHA-256 已记录；
标签与图像 1:1；跨子集字节重复 0；分组覆盖与重复风险已量化。

**未完成 / 未验证**：

1. **ISIC 训练链路未实现**（阶段 B 阻塞），因此没有任何 ISIC 模型、评价或报告。
2. **外部留出集的病变独立性未证明**（官方无分组），只能声称图像级不重叠。
3. **重编码/裁剪重复未检测**（仅字节级）。
4. **方案 B 的实际四集数字待导入后确认**（namespace 含数据集 ID，会小幅浮动）。
5. **许可证限制**：CC BY-NC 4.0，仅非商业研究；引用须含 HAM10000 论文与 ViDIR Group 署名。
