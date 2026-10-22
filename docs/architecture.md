# 架构说明

## 1. 目标与边界

本平台用于教学与实验：登录 → 导入公开图像 → BBox 标注 → 冻结三集版本 → 异步训练 →
注册模型 → 独立评估 → 单图/批量推理 → 人工复核 → 新训练版本 → 再训练对比。
只做 `detect` 任务；分割、姿态、监控、人脸、录机、告警不进入新入口。

## 2. 模块职责

| 模块 | 职责 |
|---|---|
| `app/main.py` | 应用工厂；lifespan 只做建目录/DB 探测/首个管理员；路由白名单 |
| `app/core/config.py` | 环境变量与旧变量兼容映射；生产环境强制 JWT 密钥 |
| `app/core/auth.py` | bcrypt 口令、JWT 签发与校验、角色映射（user→researcher） |
| `app/core/errors.py` | `{code,message,data,request_id}` envelope 与异常映射 |
| `app/core/storage.py` | 路径规范化、根目录边界、原子写入、图像探测 |
| `app/api/v1/*` | 仅 HTTP 编排与 schema；不直接操作文件系统 |
| `app/domains/*` | 业务服务：数据集、标注、任务、训练、模型、评估、推理、复核 |
| `app/ml/*` | 适配器与纯几何/切片函数；API 进程不加载 torch |
| `app/workers/*` | 独立进程执行导入/冻结/训练/评估/推理；数据库领取任务 |
| `app/db/models.py` | 单一 SQLAlchemy metadata，复用既有物理表名 |

## 3. 关键设计

### 3.1 任务账本而非内存队列

`job_tasks` 是权威状态，七态为 `pending/running/cancelling/interrupted/completed/
failed/cancelled`。领取使用 `UPDATE ... WHERE status='pending'` 条件更新，配合
`(owner_id,type,idempotency_key)` 唯一键实现幂等。API 只入队并返回 202，不执行长任务。
同一时间只允许 `type=training` 或 `type=inference` 的任务持有 `gpu:0` 租约。

### 3.2 不可变数据版本

导入创建 `draft` 版本；冻结时按来源组分配三集、写 `manifest.json` 并计算
`manifest_sha256 / split_manifest_hash / class_map_hash`。冻结后像素、标签、类别、split
均不可改；修改需创建子版本。复核回流只改 train，且必须继承父版本 val/test 逐资产快照。

### 3.3 信任边界

- 客户端只提交 `asset_id / model_id / artifact_id`，从不提交磁盘路径；
- 模型加载前校验受控目录 + sha256，缺失/篡改标记 unavailable；
- 普通用户不能上传/登记 `.pt`，管理员登记基础权重也需固定 hash；
- 上传经 staging、二次计数、ZIP 反穿越与压缩比限制后再原子发布。

### 3.4 前端

`frontend/src/api/client.js` 是唯一 HTTP 入口：同源 `/api/v1`、token 注入、401 处理、
envelope 解包、AbortController。`main.js` 的全局 axios/fetch 兜底保留到所有页面迁移完成
（PR-09 再删除）。

## 4. 运行图

```text
浏览器 ──同源 /api/v1──> nginx(frontend) ── /api/ ──> api(FastAPI)
                                                     │
                                        ┌────────────┼─────────────┐
                                        ▼            ▼             ▼
                                   PostgreSQL     DATA_ROOT     MODEL/ARTIFACT
                                        ▲
                        ┌───────────────┼────────────────┐
                   dataset-worker   trainer-worker   inference-worker
                   (无 GPU)         (gpu:0 租约)      (gpu:0 租约)
```

只有 `--profile workers` 才启动 worker。GPU override 单独文件，缺 GPU 的机器不得传入。

## 5. 与既有代码的关系（重要）

当前仓库同时存在两套运行路径，必须区分“新入口”和“整个仓库”：

| 层面 | 事实 |
|---|---|
| 代码导入 | `backend/app` **不导入** `yolo_api` / `database` / `dataset_versioning` / `annotation_drafts` / `job_store` / `model_registry` |
| 旧文件 | 上述旧模块相对基线 `c47950b` **零改动**，旧入口 `start_api.py` 仍可启动旧应用 |
| 数据契约 | 新模型沿用 9 张既有表名：`users`、`dataset_resources`、`dataset_versions`、`dataset_assets`、`annotation_drafts`、`annotation_draft_assets`、`job_tasks`、`job_events`、`model_registry` |
| 新增表 | 12 张：`auth_sessions`、`dataset_imports`、`evaluation_runs`、`inference_runs/items/predictions`、`review_records/items`、`feedback_exports`、`experiment_runs`、`artifact_records`、`resource_leases` |
| 规模 | 新 `app/` 6,845 行；旧 `yolo_api.py` 16,292 行、`dataset_versioning.py` 1,887 行 |
| 前端 | 7 个页面中只有 `Login.vue` 走新 `api/client.js`；Dataset/Annotation/Training/Inference/ModelHub/DatasetManager 仍调用旧端点 |
| 测试 | 新 68 个后端测试只覆盖新实现；旧 `backend/tests/*` 仍针对旧实现，未做交叉回归 |

**结论**：v1 后端是**平行重写**（re-architecture），不是“基本无关联”，也**不是**在旧代码上做增量改造。
它与旧代码共享数据库 schema 和部分算法语义（组级三集划分、revision CAS、安全解压），
但实现是独立的；仓库整体和当前前端运行路径仍以旧代码为主。

这与 `START_HERE.md` 中“优先复用已经核验的源码符号，不从头开放式重写”存在偏差，代价是：

1. 版本划分、草稿冲突、解压安全、鉴权各存在两套实现，行为可能漂移；
2. 旧测试覆盖旧实现、新测试覆盖新实现，缺少跨实现一致性回归；
3. 表名相同但**未用旧数据库副本做迁移演练**，直接指向旧库的行为尚未验证；
4. PR-09 要求的“旧入口退出最终运行图”尚未完成。

可接受的前提是完成：前端页面全量迁移 → 删除旧入口/行业模块 → 旧库副本迁移演练 →
对同一份数据跑一遍新旧行为对比。在上述完成前，不能声称“重构已完成”或“已替换旧系统”。

## 6. 已知限制（截至 PR-08）

- 训练/推理子进程管理在 worker 内实现（`train_cli` 子进程 + 取消终止），
  但本轮未在真实 GPU 上验证 OOM/进程被 kill 的恢复路径。
- 推理计时未采样峰值显存；E4 报告保留 `null` 并注明。
- SQLite 测试只证明契约，不证明 PostgreSQL 行锁/advisory lock 竞争。
