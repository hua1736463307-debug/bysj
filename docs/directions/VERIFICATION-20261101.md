# P0/P1/P2/P4 复核记录（2026-11-01）

> 本轮只做**复核与准备**：不执行正式训练、不读取 test、不执行生产迁移、不清理资源、不合并 PR。
> 结论先行：**P0 与 P2 的门禁与协议在本提交下可复现；P1 的真实 GPU 取消验收与 P4 的加密第二份
> 备份仍未完成（外部条件缺失）**。不得把本文件当作 GPU 验收或论文结果。

## 0. 基线与执行环境

| 项 | 值 |
|---|---|
| 仓库 | `D:\毕业设计\bysj-unblock` |
| 分支 | `graduation/cxr-assist-20261027` |
| HEAD | `02123fc735f20d622cff312caec7549f32c43cb7` |
| 工作区 | 1 个未提交改动：`deploy/compose.yml`（协作测试服务 `hub-test`，`profiles: [hub-test]`，本轮未提交、未删除） |
| 测试镜像 | `deploy/compose.yml` 中 `hub-test`（`sha256:14dce85f8197…`，`network_mode: none`，`/tmp` 可写，源码只读挂载） |
| 测试库 | `sqlite:////tmp/hub-test.db`，`DATA_ROOT=/tmp/data`、`MODEL_ROOT=/tmp/models`（与运行库无关） |
| GPU | **本轮执行环境不提供 GPU**；未执行任何训练或 AMP 对照 |
| 模型/推理配置 | 启动器请求 `deepseek-v4.1-flash` / `high`；客户端未回报可核验的实际配置，记为**未确认** |

## 1. P0 论文资格与状态语义（复核通过）

命令（固定测试入口，工作目录 `/workspace`）：

```
hub_docker(action="run", service="hub-test",
  command=["python","-m","pytest","-p","no:cacheprovider",
           "backend/tests/test_child_runner.py","backend/tests/isic","-q"])
```

实际结果：`40 passed, 1 skipped, 2 warnings in 14.78s`（exit 0）。

- 唯一 skip：`test_protocol_document_hash_matches_constant`（`protocol document not mounted`）。
  该断言本轮用宿主机命令单独补验：
  `sha256sum docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md` →
  `40064e0639e7e66727e94c2c90133253da991be55dc963615dd0f8f5b0266c60`，
  与 `experiment_purpose.FORMAL_PROTOCOL_SHA256` 完全一致。
- 门禁语义（`backend/app/domains/isic/experiment_purpose.py`）：`paper_eligible` 由门禁函数**重新计算**，
  不读取已存布尔；缺 `experiment_purpose` 即默认拒绝；`smoke` / `synthetic` / `dummy` 永不通过。
- 汇总排除：`GET /api/v1/isic/results/paper-summary`（管理员）只把 `paper_eligible=true` 的
  `split=test` 评价放入 `included`，其余进 `excluded` **且不返回指标**；
  对应用例 `test_paper_summary_excludes_smoke_run_but_keeps_it_queryable`、
  `test_paper_summary_includes_run_that_passes_all_gates` 本轮均通过。
- smoke 仍可查询：`/evaluations`、`/models` 返回 `paper_eligibility` 供审阅，历史元数据未改写。

**残留缺口（未在代码中强制）**：协议第 9.2 节要求 test 评价的 `EvaluationRun.purpose` 写明
`protocol_id` / `seed` / 臂名 / 授权人，但 `paper_summary` 目前只按模型元数据判定资格，
不校验该字段格式。属流程约束，建议在 P3 施工单里决定是否改为强校验（改动需同步
`FORMAL_PROTOCOL_SHA256`）。

## 2. P2 数据门禁与冻结协议（材料复核通过）

本轮只读核对既有产物，未改动已冻结 manifest：

| 核对项 | 来源 | 结果 |
|---|---|---|
| `manifest_sha256` | `artifacts/isic-smoke/freeze-job.json` | `8f5b609c6197792adf7cc71f79236338fd11c47b6fab85bc3fe51d20b2c563ae`，与协议第 2 节及 `FORMAL_DATASET_MANIFEST_SHA256` 一致 |
| `split_manifest_hash` | 同上 | `4083c441af21397e18db9ec31e2115600f8de3e4f034df2b4f8f1f9dcfd7f32f` |
| 四集图数 | 同上 | train 7,109 / val 933 / calibration 967 / test 1,006，合计 10,015 |
| 逐类支持数 | 同上 | 与协议第 2 节表格逐格一致（例：val `df`=7、`vasc`=20；test `df`=16） |
| 分组数 | 同上 | 5,284 / 701 / 730 / 755（lesion 级，非患者级） |
| 字节/像素重复 | `artifacts/isic-groups/isic-group-audit.json` | 11,720 张全量比对：重复 2 组 4 张，**全部在 training 内部**，`cross_subset_groups={}`、`pixel_cross_subset_groups={}` |
| 近重复 | 同上 | dHash≤6 候选 157,078 对（跨子集 38,462 对，距离 0 者 39 对），仅作复核清单，不构成重复结论 |

协议覆盖核对（`docs/datasets/ISIC-FORMAL-PROTOCOL-v1.md`）：数据版本与哈希、模型结构与
`init_weights_sha256`、预处理 `isic-preprocess-v1`、主指标 macro-F1 + lesion 组件 bootstrap、
两对照臂（`balanced` 主 / `none` 对照）、预算（30 epoch/16/3e-4/patience 8）、模型选择仅 val、
seed 42/43/44、温度缩放（仅用 calibration，单一标量 T）、评价精度 fp32、test 一次性访问约束。

精度相关只读核对：`backend/app/ml/isic_cli.py` 中 `torch.cuda.amp.autocast()` 仅出现在训练步
（第 221–222 行），`evaluate` 路径无 autocast → 评价为 fp32；但**同 checkpoint/同输入的
AMP vs fp32 差值未实测**（无 GPU），协议第 8 节仍标注为未验证。

## 3. P1 真实 GPU 取消验收（未完成，外部条件缺失）

- 本轮环境无 GPU，未提交任何训练作业，因此**没有**本提交下的取消时间线、退出码、
  子进程消失、租约释放与后续短任务复用证据；不得引用 `63bff5f` 时期的
  `artifacts/gpu-cancel/summary.json`（那是旧提交 + 旧镜像的证据）。
- 可在无 GPU 环境完成的回归已执行：`backend/tests/test_child_runner.py` 与
  `backend/tests/isic` 全绿（见第 1 节），说明取消路径的单元级行为未回退；
  这不等于 GPU 验收。
- 候选镜像：`02123fc` 的 api/worker 镜像**未构建**，ID 未知（见 `MIGRATION-PLAN-f004.md` 第 5 节）。
- 待用户提供：获准的 GPU 隔离验收环境与操作通道、可用窗口、现有任务/租约协调方式。

## 4. P4 迁移准备（文档已更新，密钥与备份未就绪）

- `MIGRATION-PLAN-f004.md` 已把目标代码从 `abfd310` 更正为 `02123fc…`，并新增
  "候选镜像未构建、`isic-63bff5f` 不含 P0/P2 改动、构建前需处理未提交的 `compose.yml`"声明。
- 已写入的既有结论保持不变：备份脚本会恢复服务 → 必须再次停写并做持续停写检查；
  PG `downgrade` 会**永久丢失**新表数据并留下孤立资产，`降级通过` ≠ 无损回滚；
  首选隔离新项目恢复备份，并记录切回时新写入数据的处理边界。
- **未完成**：第二份加密备份的目的地与密钥保管/调用安排未定（不擅自上传、不在聊天传密钥）；
  加密备份与密钥不得共同进 Git；前端镜像 `bysj-frontend:20261022` 是否含新页面仍未核验。

## 5. 未验证项清单（不得当作已通过）

1. 本提交下真实 GPU 训练取消 / 进程回收 / 租约释放 / 取消后短任务复用（无 GPU）。
2. AMP 与 fp32 在**同一 checkpoint、同一输入**下的差值归因（无 GPU）。
3. 温度缩放实现（`isic_cli.py` 尚未实现，协议第 7 节标注为 P3 施工项）。
4. 近重复的裁剪/旋转/不同视野判定；外部官方 val/test 的患者级独立性（不可证明）。
5. 候选 api/worker/前端镜像的构建与镜像 ID。
6. test 评价 `purpose` 字段的代码级强制（第 1 节残留缺口）。
7. 协议尚未经人工确认冻结（当前为"冻结候选"，哈希已绑定本文件字节）。
