# 多方向实施状态（2026-10-27）

> 方向注册表：`backend/app/directions/`，对外只读接口 `GET /api/v1/directions`。
> 数据盘点与实测：`docs/datasets/DATASET_INVENTORY.md`。
> 隔离备份恢复演练：`docs/directions/RESTORE-DRILL-20261027.md`。
> 生产迁移窗口方案（**未执行**）：`docs/directions/MIGRATION-PLAN-f004.md`。
> 迁移前补充验证（BCCD 功能、GPU 取消、PG 降级、备份保护）：`docs/directions/MIGRATION-READINESS-20261028.md`。
> ISIC 数据核验与四集方案：`docs/datasets/ISIC-SPLIT-PLAN.md`。
> 本轮（2026-11-01）P0/P2/P4 复核与未验证项：`docs/directions/VERIFICATION-20261101.md`。
> 分支 `graduation/cxr-assist-20261027`，草稿 PR #2（base `graduation/unblock-20261022`，未合并）。

## 0. 一句话结论

**架构已从"单方向胸片"改为"方向注册 + 任务语义分离"**，三个方向共存互不污染：
胸片多标签（已实现，真实数据受限）、皮肤镜单标签（数据通路已实现，标签已审计、训练集已下载）、
血细胞检测（历史模块保留）。

**当前研究状态（2026-10-31 修订）：真实工程 smoke 已完成；正式实验未完成。**
ISIC 用真实数据跑过 1 epoch 的工程 smoke（`docs/directions/ISIC-SMOKE-20261028.md`），
它验证链路可用，**不构成论文指标**；没有任何正式（多 seed、冻结协议门禁）训练结果。

工程侧已完成：分支已推送、CI 全绿；f004 备份恢复演练在隔离环境全链路通过（运行项目仍为 f002、未被改动）；
ISIC 三个子集图像已全部下载并校验；生产迁移方案已写好但**未执行**，等待窗口确认。

**ISIC 训练链路已实现**（`63bff5f`）：七类 softmax 适配器、`isic_train` / `isic_evaluate` 作业、
逐类 P/R/F1 + macro-F1 + 混淆矩阵 + top-1 ECE，走既有 trainer 角色与 GPU 租约。
取消路径修复（`899444d`）目前只有单元测试证据（`c966f7c`），**修复后的真实 GPU 取消未验收**。

## 1. 方向矩阵

| 方向 key | 任务语义 | 输出 | 标签 | 分组 | 代码 | 真实数据 |
|---|---|---|---|---|---|---|
| `cxr` | 多标签分类 | 逐标签独立评分（sigmoid） | `cxr-five-v1`，5 项 + 四态 | patient_component | **已实现** | **BLOCKED_ACCESS** |
| `isic2018` | 单标签分类 | softmax 分布（和为 1） | `isic2018-task3-v1`，7 类 | lesion（无分组时单图） | **已实现** | **已完整取得**：标签 + 三个子集图像，均校验 |
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
| API | `/api/v1/isic/{schema,datasets,import,versions,summary,training,evaluations,models,results/paper-summary}` | 11 个端点 |
| 测试 | 注册表、导入正/反例、冻结、跨方向拒绝 + 实验用途/论文资格门禁 | `backend/tests/isic/` |
| 恢复演练 | 合成 ISIC 4 类记录随 f004 备份恢复，新代码可读 | `RESTORE-DRILL-20261027.md` |
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

## 4. 交付与验证状态（2026-10-27 收尾）

| 项 | 状态 | 证据 |
|---|---|---|
| 分支推送 | **PASS** | `e96a424`→`82ccf42` 已推送；`github.com` 经 7890 代理可写，7897 会 TLS 失败 |
| PR #2 | **OPEN / DRAFT** | head `c966f7c`，base `graduation/unblock-20261022`，未合并 |
| CI（`c966f7c`） | **PASS（有边界）** | backend / frontend / compose 均 success；gpu-acceptance **skipped**（无 GPU runner），**不等于 GPU 验收** |
| f004 备份恢复演练 | **PASS** | 两套隔离项目，28 表 / 14,122 行，文件哈希全一致，新代码读取 200 |
| 运行中项目 | **未改动** | 仍 `4a6b1222f002`、22 表、5,255 资产、115 任务；容器未重启 |
| ISIC 数据完整性 | **PASS** | 三个 ZIP 字节数与官方一致、`unzip -t` 无错误、SHA-256 已记录；标签↔图像 1:1 |
| ISIC 分组/重复审计 | **PASS（有边界）** | 7,470 lesion、训练覆盖 100%；11,720 张全量比对，跨子集字节重复 **0**；重编码重复未检测 |
| 生产迁移方案 | **READY（未执行）** | 门禁、停写验证、备份、迁移、验证、回滚、清单均已写明；等窗口确认 |
| PG 降级演练 | **PASS（有代价）** | 隔离库 f004→f002 成功，旧表零变化；但丢新表数据并留 14 条孤立资产 |
| 旧 BCCD 功能 | **部分 PASS** | 推理作业 3/3 完成、71 框落库；但旧训练权重 0 框（退化），只能验到管线级 |
| 备份保护 | **未完成** | 未加密、仅本机；方案见 READINESS 第 5 节，等你决定 |

## 5. 本轮未做（明确）

- **没有正式训练**：ISIC 训练链路已实现并跑过 1 epoch 工程 smoke；CXR 真实数据不可达。
- **旧 BCCD 权重不可用**：现有 `best.pt` 在 conf=0.01 下仍 0 框，属既有数据问题，未修复。
- **生产迁移未执行**：方案已就绪，运行库仍 f002。
- **没有 ISIC 模型分析/报告页面**：训练/评价/模型登记链路已接通（`63bff5f`），但前端仍无
  ISIC 专用页面（模型选择、单图分析、人工复核、可追溯报告），不得把 CXR 的五项文案直接复用。
- **没有 NIH 适配器**：NIH 数据未取得，不写"看起来能用"的空实现。
- 所有合成 fixture / dummy 后端产物带 `synthetic` / `research_valid=false`，不得进入论文。

### 5.1 P0 论文资格语义（2026-10-31 新增）

`research_valid` 只区分"是否 dummy"，不能回答"能否进论文"。自本轮起：

| 字段 | 来源 | 语义 |
|---|---|---|
| `experiment_purpose` | 提交训练时**显式声明**（`smoke` / `formal`） | 缺省即"未分类"，默认拒绝 |
| `research_valid` | 历史字段，语义不变 | 产物非 dummy |
| `paper_eligible` | 门禁函数**重新计算**，不读取已存布尔值 | 能否进入论文汇总/导出 |

formal 还须绑定冻结协议（`docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md` 的 id + 字节 sha256）、
冻结数据版本 manifest 哈希、预训练权重 `init_weights_sha256` 与预登记 seed(42/43/44)。

- 门禁实现：`backend/app/domains/isic/experiment_purpose.py`；
- 提交期校验：`training_service.preflight`（无 purpose → warn 且 `paper_eligible=false`）；
- 落库：`workers/isic_handlers.py` 写入模型元数据 `paper_eligible` / `paper_ineligible_reasons`；
- 读取/汇总：`GET /api/v1/isic/models`、`GET /api/v1/isic/evaluations` 返回 `paper_eligibility`；
  `GET /api/v1/isic/results/paper-summary`（管理员）只收符合资格的 test 结果，其余列原因且不给指标。

历史记录未改写：2026-10-28 的 smoke 模型（含 `158acb2986604b69`）没有 `experiment_purpose`，
按"未分类"处理，仍可查询但被正式汇总排除。

## 6. 下一步

1. ISIC 正式实验：按 `docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md` 执行（多 epoch、seed 42/43/44、
   calibration 温度缩放、test 只读一次、外部留出集只作来源不同的对照）；smoke 已完成但不作论文指标。
2. 生产迁移：等窗口确认；按最终提交重建镜像并重新验证（见 `MIGRATION-PLAN-f004.md`）。
3. NIH 数据到位后新建 `nih-chestxray14-v1` 适配器（14 列布尔，不复用四态）。
4. ISIC 前端页面（模型选择、单图分析、人工复核、可追溯报告）拆独立施工单。
