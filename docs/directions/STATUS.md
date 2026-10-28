# 多方向实施状态（2026-10-27）

> 方向注册表：`backend/app/directions/`，对外只读接口 `GET /api/v1/directions`。
> 数据盘点与实测：`docs/datasets/DATASET_INVENTORY.md`。

## 0. 一句话结论

**架构已从"单方向胸片"改为"方向注册 + 任务语义分离"**，三个方向共存互不污染：
胸片多标签（已实现，真实数据受限）、皮肤镜单标签（数据通路已实现，标签已审计、图像部分下载）、
血细胞检测（历史模块保留）。**没有任何真实训练结果**，所有模型结论仍为空。

## 1. 方向矩阵

| 方向 key | 任务语义 | 输出 | 标签 | 分组 | 代码 | 真实数据 |
|---|---|---|---|---|---|---|
| `cxr` | 多标签分类 | 逐标签独立评分（sigmoid） | `cxr-five-v1`，5 项 + 四态 | patient_component | **已实现** | **BLOCKED_ACCESS** |
| `isic2018` | 单标签分类 | softmax 分布（和为 1） | `isic2018-task3-v1`，7 类 | lesion（无分组时单图） | **已实现** | **PARTIAL**：标签 100%，验证集图像下载中 |
| `bccd_detect` | 目标检测 | bbox + 置信度 | `bccd-detect-v1`，3 类 | image_cluster | **已实现（历史）** | 旧工作区已有，未在本轮验证 |

任务语义分离是硬约束：注册表 `validate()` 会拒绝"单标签任务配多标签输出"这类错配；
`/api/v1/cxr` 与 `/api/v1/isic` 交叉访问同一数据集一律 `409 task_mismatch`（有测试）。

## 2. 皮肤镜方向已落地的东西

| 层 | 内容 | 证据 |
|---|---|---|
| 契约 | `isic2018-task3-v1` 七类，顺序与官方 CSV 一致 | `directions/isic.py` |
| 导入 | 官方目录布局解析、one-hot 强校验、缺失图像即拒绝、复制后 hash 复核 | `domains/isic/import_service.py` |
| 冻结 | 按 lesion 分组四集、单标签类别门禁、稀有类警告 | `domains/isic/split_service.py` |
| 表 | `isic_samples`（Alembic `4a6b1222f004`） | 空库 28 表 |
| 任务 | `isic_import` / `isic_freeze` 挂到 dataset 角色 worker | `workers/isic_handlers.py` |
| API | `/api/v1/isic/{schema,datasets,import,versions,summary}` | 6 个端点 |
| 测试 | 11 项（注册表、导入正/反例、冻结、跨方向拒绝） | `backend/tests/isic/` |
| 审计 | 真实标签表审计：train 10,015 / val 193 / test 1,512；7,470 个 lesion 组 | `scripts/prepare_isic.py` |

## 3. 真实数据审计（2026-10-27，实际运行）

```
python scripts/prepare_isic.py --source-root <ISIC 根目录> --output artifacts/isic-audit
```

| split | 行数 | 最大类占比 | 最小类 | 分组 |
|---|---|---|---|---|
| train | 10,015 | NV 6,705（66.9%） | DF 115 | 7,470 个 lesion，覆盖 100% |
| val | 193 | NV 123（63.7%） | **DF 1** | 无官方分组表 |
| test | 1,512 | — | — | 无官方分组表 |

train 的确认方式分布：histopathology 5,340 / serial imaging 3,704 / expert consensus 902 / confocal 69；
1,956 个 lesion 含多张图（最多 6 张）。这直接决定：**必须按 lesion 分组切分**，否则同病变跨集。

**限制（必须随结果一起报告）**：validation/test 无官方分组表，患者级隔离不可证明；
val 中 DF 仅 1 例，任何基于 val 的类别指标对小类都不可靠。

## 4. 本轮未做（明确）

- **没有真实训练**：ISIC 训练集图像 2.77 GB 因带宽（实测 9–12 KB/s）未下完；CXR 真实数据不可达。
- **没有 ISIC 模型/分析/报告页面**：本轮只接通数据与冻结；模型推理沿用 CXR 的 DenseNet 适配器
  （`isic` 方向已声明 `softmax` 输出，但尚未接入训练/评价/报告链路）。
- **没有 NIH 适配器**：NIH 数据未取得，不写"看起来能用"的空实现。
- 所有合成 fixture / dummy 后端产物带 `synthetic` / `research_valid=false`，不得进入论文。

## 5. 下一步

1. 用更好带宽下载 ISIC `Training_Input.zip`（2.77 GB）→ 冻结真实四集 → 1 epoch GPU smoke。
2. NIH 数据到位后新建 `nih-chestxray14-v1` 适配器（14 列布尔，不复用四态）。
3. ISIC 训练/评价/报告链路接入（softmax 指标：逐类 P/R/F1、macro-F1、混淆矩阵、ECE）。
