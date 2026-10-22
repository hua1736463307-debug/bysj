# 数据库设计

## 1. 复用与新增

沿用既有物理表：`users`、`dataset_resources`、`dataset_versions`、`dataset_assets`、
`annotation_drafts`、`annotation_draft_assets`、`job_tasks`、`job_events`、`model_registry`。
新增表用于版本治理、评估、推理、复核与血缘，见下表。

| 表 | 用途 | 关键约束 |
|---|---|---|
| `auth_sessions` | JWT 会话与登出撤销 | `jti` 唯一 |
| `dataset_imports` | 导入记录与报告 | `dataset_id` 索引 |
| `evaluation_runs` | val/test 评估 | `model_id`,`dataset_version_id`,`split` |
| `inference_runs` | 推理批次 | `job_id` 索引 |
| `inference_items` | 分项状态与计时 | unique(`run_id`,`asset_id`,`attempt`) |
| `inference_predictions` | 原始预测（不可被复核覆写） | `item_id` 索引 |
| `review_records` | 图像级审阅 | `state`,`revision` |
| `review_items` | 逐条结论 | `prediction_id` 可空（漏检） |
| `feedback_exports` | 回流导出 | `idempotency_key` 唯一 |
| `experiment_runs` | 实验索引与可复现元数据 | `training_job_id` |
| `artifact_records` | 受控产物 | 相对路径 + sha256 |
| `resource_leases` | GPU/worker 租约 | 主键 `resource` |

## 2. 关键字段

### dataset_versions

| 字段 | 说明 |
|---|---|
| `state` | draft / validating / frozen / failed |
| `manifest_sha256` | 规范化 manifest 的 hash（不含时间戳） |
| `split_manifest_hash` | 三集内容 hash 的聚合 |
| `class_map_hash` | 类别表 hash |
| `split_config_json` | seed/ratios/group_level；回流子版本记录父版本 |
| `parent_version_id` | 子版本血缘 |

### dataset_assets

`content_hash`（SHA256）+ `source_group` + `group_level` + `split` +
`root_asset_id/original_asset_id/transform_json`。三集隔离在
`source_group / root_asset / content_hash` 三个层面同时成立。

### job_tasks

在既有列上补充 `owner_user_id`、`idempotency_key`、`request_hash`、`attempt`、
`lease_owner`、`lease_expires_at`。唯一键 `(owner_id,type,idempotency_key)`。

## 3. 索引

- 列表：`(owner_user_id, deleted_at)`、`created_at`；
- 任务：`(type,status,created_at)`；
- 资产：`(dataset_version_id, relative_path)` 唯一；
- 事件：`(task_id, sequence)` 唯一，单次查询上限 200；
- 预测：`inference_predictions(item_id)`。

## 4. 迁移策略

`backend/migrations/versions/304f405af95f_initial_medical_platform_schema.py` 在空库创建
全部 21 张业务表（实测）。

已有 bysj 实验库升级顺序：

1. 备份数据库副本（禁止直接操作生产库）；
2. `alembic stamp 304f405af95f`（表已存在，只登记版本）；
3. 如需字段补齐，单独编写增量迁移（nullable → 回填 → 加约束）；
4. 空库直接 `upgrade head`。

不得在生产对象上 `create_all`/`drop_all`；行业表不进入新应用 metadata，删除另列迁移。

## 5. 不变量

- 冻结版本内容 hash 不变；
- 一次回流导出只产生一个子版本；
- 预测原始记录不可被 review 覆写；
- 终态任务不可回到运行态；
- 删除被模型/训练/子版本引用的版本返回 409。
