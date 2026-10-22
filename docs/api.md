# API 说明（v1）

Base URL：`/api/v1`（同源；前端通过 nginx 代理）。
OpenAPI：`GET /api/openapi.json`，交互文档 `GET /api/docs`。

## 1. 统一 envelope

成功：

```json
{"code":"ok","message":"操作成功","data":{},"request_id":"9f0c..."}
```

失败：`code` 为稳定业务字符串，HTTP 状态保持真实语义；`data` 可含字段错误。

| HTTP | 典型 code | 语义 |
|---|---|---|
| 400 | `invalid_request` | 请求结构外的业务无效 |
| 401 | `unauthenticated` / `token_expired` / `session_revoked` | 未认证/过期/已登出 |
| 403 | `forbidden` / `approval_required` | 角色不足 |
| 404 | `not_found` | 不存在或无权（统一避免枚举） |
| 409 | `revision_conflict` / `version_frozen` / `resource_referenced` / `idempotency_conflict` | 冲突 |
| 413 | `file_too_large` / `zip_ratio` | 上传超限 |
| 422 | `validation_error` / `label_out_of_range` / `class_map_mismatch` | 字段/标签/几何 |
| 429 | `queue_limit` | 待执行任务超限 |
| 503 | `gpu_busy` / `not_ready` | 依赖不可用 |

列表统一 `{items,total,page,page_size}`，`page_size` 默认 50、上限 200。

## 2. 端点

### 认证与用户

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/auth/login` | 返回 `token/token_type/expires_at/user` |
| GET | `/auth/me` | 当前用户 |
| POST | `/auth/logout` | 撤销当前 jti |
| POST | `/auth/users` | 管理员创建 researcher/reviewer |
| GET | `/auth/users` | 管理员列出用户 |

### 健康

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health/live` | 进程存活，不依赖 DB/GPU |
| GET | `/health/ready` | DB/worker 分项；DB 不可用返回 503 |

### 数据集

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/datasets/import` | multipart（files + 来源字段）→ 202 job |
| GET | `/datasets` | 归属过滤 + 分页 |
| GET | `/datasets/{id}` | 详情 + 最新版本 |
| GET | `/datasets/{id}/assets` | `version_id/page/page_size` |
| GET | `/datasets/{id}/versions` | 版本列表 |
| POST | `/datasets/{id}/versions` | 冻结任务（202） |
| GET | `/dataset-versions/{id}` | 版本状态/hash/统计 |
| DELETE | `/datasets/{id}` / `/assets/{id}` | 软删除；被引用 409 |
| POST | `/datasets/{id}/restore` / `/assets/{id}/restore` | 管理员恢复 |

### 标注

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/annotations/{image_id}` | 草稿/标注/revision/可编辑性 |
| PUT | `/annotations/{image_id}` | `expected_revision` CAS；冲突 409 返回 `server_revision` |
| POST | `/annotations/export` | YOLO/JSON 导出任务（202） |

### 任务

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/jobs` / `/jobs/{id}` | 归属过滤；`status_url` |
| GET | `/jobs/{id}/events` | `after_sequence/limit≤200` |
| POST | `/jobs/{id}/cancel` | pending 立即取消；running 转 cancelling |
| POST | `/jobs/{id}/resume` | interrupted 重新排队（新 attempt） |

### 训练 / 模型 / 评估

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/training/preflight` | 返回 checks 明细与规范化参数 |
| POST | `/training/jobs` | 支持 `Idempotency-Key`；202 |
| GET | `/training/jobs/{id}` | 仅训练类型 |
| POST | `/training/jobs/{id}/cancel` | 训练别名 |
| GET | `/models` / `/models/{id}` | 归属过滤；不含服务器路径 |
| POST | `/models/register` | 仅管理员；受控目录 + sha256 |
| GET | `/models/{id}/download` | 加载前二次 hash 校验 |
| DELETE | `/models/{id}` | 被评估引用 409 |
| POST | `/evaluations` | val/test；test 需管理员 + purpose |
| GET | `/evaluations` / `/{id}` | 指标缺失为 null + reason |

### 推理与产物

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/inference/images` | multipart 或 `asset_ids`；202 |
| GET | `/inference/runs` / `/{id}` | 分项状态/预测/计时 |
| GET | `/inference/runs/{id}/export?format=json|csv` | schema_version/hash；CSV 防公式注入 |
| GET | `/artifacts/{id}` | 归属 + 目录边界 |

### 复核与实验

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/reviews` | 图像级审阅；漏检可无 prediction_id |
| GET | `/reviews` / `/{id}` | 归属/角色过滤 |
| PATCH | `/reviews/{id}` | `expected_revision` CAS |
| POST | `/reviews/{id}/submit` | draft → submitted |
| POST | `/reviews/{id}/decision` | reviewer/admin 审批 |
| POST | `/reviews/export-version` | 回流子版本任务（202） |
| GET | `/experiments` / `/{id}` | 同口径实验记录 |

## 3. 示例

训练：

```http
POST /api/v1/training/jobs
Idempotency-Key: exp-1f27a4d9-seed42
Content-Type: application/json

{"dataset_version_id":"...","model_id":"...","task":"detect",
 "params":{"epochs":80,"batch":8,"imgsz":640,"seed":42,"deterministic":true,"device":"cuda:0"}}
```

```json
{"code":"ok","message":"训练任务已接收","data":{"job_id":"...","status":"pending",
 "status_url":"/api/v1/jobs/..."},"request_id":"..."}
```

推理结果：

```json
{"run_id":"...","model_sha256":"...","items":[{"status":"completed",
 "width":640,"height":480,"predictions":[{"class_id":0,"class_name":"RBC",
 "confidence":0.87,"bbox_xyxy":[12.0,20.0,48.0,60.0]}],"timing":{"total_ms":23.4}}]}
```
