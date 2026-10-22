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

## 5. 已知限制（截至 PR-08）

- 训练/推理子进程管理在 worker 内实现（`train_cli` 子进程 + 取消终止），
  但本轮未在真实 GPU 上验证 OOM/进程被 kill 的恢复路径。
- 推理计时未采样峰值显存；E4 报告保留 `null` 并注明。
- SQLite 测试只证明契约，不证明 PostgreSQL 行锁/advisory lock 竞争。
