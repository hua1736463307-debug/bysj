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
| POST | `/datasets/{id}/train-tile-versions` | 已冻结 parent_version_id → 202 job；必需 Idempotency-Key；仅 train 切片，val/test 原图保持 |
| GET | `/datasets/{id}/assets` | `version_id/page/page_size` |
| GET | `/datasets/{id}/versions` | 版本列表 |
| POST | `/datasets/{id}/versions` | 冻结任务（202） |
| GET | `/dataset-versions/{id}` | 版本状态/hash/统计 |
| DELETE | `/datasets/{id}` / `/assets/{id}` | 软删除；被引用 409 |
| POST | `/datasets/{id}/restore` / `/assets/{id}/restore` | 管理员恢复 |

导入的 `group_level` 默认 original_image（按文件内容哈希分组）；若需要保守合并不同文件的相似图片，传 `group_level=image_cluster` 与 `source_groups` JSON 对象，键为本次图像规范相对路径，值为非空来源组名。映射必须精确覆盖全部图像，不能有遗漏/多余项；完全相同内容不得分到不同组。原始模式携带显式映射会拒绝，不静默忽略。image_cluster 不能宣称患者或载玻片级独立。已有冻结版本不会被此接口修改。

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

上例数字仅演示字段，不是性能验收。`total_ms` 为整批受理到该图写入阶段的累计时间，随批内位置增长；`processing_ms` 为本图预测及产物处理区间，`model_ms` 为框架同步后的推理区间。`queue_ms` 是领取前等待，`gpu_wait_ms` 是领取后 GPU 租约等待；不能把累计 `total_ms` 解释成单图 GPU 耗时。完整定义和实测样本见 [E4 性能报告](experiments/2026-10-22/PERFORMANCE.md)。

正式训练应显式固定 `optimizer/lr0/lrf/momentum/weight_decay/warmup_epochs`；当前只接受 AdamW 或 SGD，不接受随迭代规模改变行为的 auto。有效参数、增强、早停与锁/源码摘要保存在训练结果中。被取消的训练不能经 `/resume` 复活；工具的 `--retry-cancelled` 是显式创建新任务并保留旧证据。

## 9. 胸片五标签辅助分析（`/api/v1/cxr`）

新主线接口，与 detect 契约并存但互不混用。标签契约 `cxr-five-v1`，任务 `cxr_multilabel`，顺序固定为
`atelectasis / cardiomegaly / consolidation / edema / pleural_effusion`。

| 方法 | 路径 | 权限 | 说明 |
|---|---|---|---|
| GET | `/cxr/schema` | 登录 | 固定 key/中文名、四态、四集、格式、borderline margin |
| GET | `/cxr/dashboard` | 登录 | 真实统计（无数据为 0） |
| POST | `/cxr/datasets/import` | 管理员 | 202 `cxr_import`；`Idempotency-Key` 必需；只接受已配置 `source_id` |
| GET | `/cxr/datasets` | 登录 | 仅 `task_type=cxr_multilabel` |
| POST | `/cxr/datasets/{id}/versions` | 登录 | 202 `cxr_freeze`；四集 70/10/10/10，`split_seed=42` |
| GET | `/cxr/datasets/{id}/versions/{vid}/summary` | 登录 | 四集图数/患者/component、四态覆盖、重复候选、训练门禁 |
| POST | `/cxr/training/preflight` | 登录 | 独立 `CxrTrainingParams` 白名单与门禁 |
| POST | `/cxr/training/jobs` | 登录 | 202 `cxr_train`；`Idempotency-Key` 必需 |
| GET | `/cxr/models` | 登录 | `task=cxr_multilabel\|cxr_backbone` |
| POST | `/cxr/evaluations` | 登录 | 202 `cxr_evaluate`；`calibrate` 只能 calibration；`test` 需 admin + purpose |
| GET | `/cxr/evaluations` | 登录 | 摘要指标（不含 map50） |
| POST | `/cxr/analysis-batches` | 登录 | 202 `cxr_analyze`；`asset_ids`（≤100，不截断） |
| POST | `/cxr/analysis-batches/upload` | 登录 | 同上，multipart JPG/PNG |
| GET | `/cxr/analysis-batches`、`/{id}` | 登录 | 批次与逐项状态，不返回服务器路径 |
| GET | `/cxr/analyses/{id}`、`/{id}/image` | 登录 | 五项完整结果；原图回读 |
| POST | `/cxr/analyses/{id}/explanations` | 登录 | 202 `cxr_explain`；`label_key` 必需 |
| POST | `/cxr/reviews`、PATCH `/{id}` | 登录 | 逐标签意见；PATCH 需 `expected_revision`（CAS，409 保草稿） |
| POST | `/cxr/reviews/{id}/submit`、`/decision` | 登录 / 复核权限 | `draft→submitted→approved\|rejected`；五项未填全或含 indeterminate 不可批准 |
| POST | `/cxr/reports`、GET `/{id}`、`/{id}/export` | 登录 | 不可变快照；`format=json\|html`；无批准复核只出草稿 |

单项分析输出示例（字段语义固定，数值仅示意）：

```json
{"analysis_id":"...","status":"completed",
 "calibration":{"status":"not_calibrated","threshold_source":"default_no_calibration"},
 "labels":[
  {"label_key":"atelectasis","display_name":"肺不张","raw_score":0.4123,"calibrated_score":null,
   "threshold":0.5,"flag":"below_threshold","calibration_status":"not_calibrated"},
  {"label_key":"cardiomegaly","display_name":"心影增大","raw_score":0.7302,"calibrated_score":null,
   "threshold":0.5,"flag":"above_threshold","calibration_status":"not_calibrated"}
 ],
 "quality":{"flags":["low_resolution"],"definition":"工程提示，不是经临床验证的质量评估"},
 "explanation":{"score_semantics":"模型评分，不是临床确诊概率；未超过阈值不等于排除该征象或其他疾病"}}
```

`flag` 取值 `above_threshold / below_threshold / borderline / not_available`，永不返回 `confirmed_disease` 或 `healthy`。
`raw_uncertain`（来源报告不确定）、`borderline`（预测接近阈值）、`not_calibrated`（校准数据不足）是三个不同字段，不合并成"置信度低"。

任务类型保护：对 `task_type != detect` 的版本调用 BBox 冻结/切片/标注接口返回 `409 task_mismatch`；
用 `cxr_backbone` 或 detect 模型做胸片分析返回 `422 cxr_backbone_not_analyzable` / `422 model_task_mismatch`。
