# 生产迁移窗口方案：f002 → f004（2026-10-27 准备稿）

> 状态：**仅准备，未执行**。本文只做只读核对与方案设计；迁移、停服、切换镜像均等待确认窗口。
> 依据：`docs/directions/RESTORE-DRILL-20261027.md`（隔离演练已全链路通过）。

## 1. 目标与范围

把运行中的 `bysj-restore2-20261022` 从

- 代码 `195d34c`（镜像 `bysj-api:grouped-resume-20261023-0215`）
- 数据库 `4a6b1222f002`（22 表，无 CXR / ISIC 表）

升级到

- 代码 `abfd310`（分支 `graduation/cxr-assist-20261027` 当前 HEAD）
- 数据库 `4a6b1222f004`（28 表，新增 5 张 CXR 表 + 1 张 ISIC 表）

**不改**：数据卷内容（除新增文件外）、BCCD 数据集与模型、既有 API 契约、前端路由（旧页保留）。

## 2. 当前基线（只读核对，2026-10-27 实测）

| 项 | 值 |
|---|---|
| 项目 | `bysj-restore2-20261022`，6 个容器全部 Up |
| api / dataset-worker | `bysj-api:grouped-resume-20261023-0215` = `sha256:bbc2418b04b3…` |
| worker / inference-worker | `bysj-worker:grouped-resume-20261023-0215` = `sha256:e129b551c0b4…` |
| frontend | `bysj-frontend:20261022` = `sha256:fb022afabd00…` |
| db | `postgres:16-alpine` = `sha256:57c72fd2a128…` |
| 代码 SHA | `BYSJ_CODE_SHA=195d34c0977f44d7e17209e195715d1e62ac070a` |
| 数据库版本 | `4a6b1222f002` |
| 在途任务 | **0**（pending / running / cancelling / interrupted 均为空） |
| GPU 租约 | `gpu:0` holder = NULL，generation 68 |
| 健康 | `/health/live` 200、`/health/ready` 200（三角色齐全） |
| 业务基线 | 数据集 23、版本 33、活跃模型 22、资产 5,255、作业 115 |
| 卷 | `bysj-restore2-20261022-{db-data,data,models,artifacts}`，data ≈ 972 MB / 11,906 文件 |
| 磁盘 | D: 剩余 62 GB（备份需 ≈1 GB） |
| worker GPU | `HostConfig.DeviceRequests` = nvidia count 1（compose.gpu.yml 生效） |

## 3. 前置门禁（窗口开始前必须全部为真）

```bash
# 1) 无在途任务：必须返回空
docker exec bysj-restore2-20261022-db-1 psql -U bysj -d bysj -tAc \
 "select type,status,count(*) from job_tasks
  where status in ('pending','running','cancelling','interrupted') group by 1,2;"

# 2) GPU 无持有者：holder 必须为空
docker exec bysj-restore2-20261022-db-1 psql -U bysj -d bysj -tAc \
 "select resource,coalesce(holder,'NULL'),generation from resource_leases where resource='gpu:0';"

# 3) 无外部计算进程占用
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# 4) 磁盘余量 ≥ 5 GB
df -h /d

# 5) 目标镜像已构建并验证（见第 5 节）
docker image inspect bysj-api:<tag> --format '{{.Id}}'

# 6) 当前版本确认
docker exec bysj-restore2-20261022-db-1 psql -U bysj -d bysj -tAc "select version_num from alembic_version;"
```

任一项不满足 → **不进入窗口**。有在途任务时先等其完成或按既有流程取消，不得直接停服。

## 4. 停机范围与影响

| 服务 | 动作 | 影响 |
|---|---|---|
| frontend / api | 停止 → 换镜像 → 启动 | 界面与 API 不可用（预计 8–10 分钟） |
| worker / inference-worker / dataset-worker | 停止 → 换镜像 → 启动 | 训练/推理/导入不可提交（与 api 同时段） |
| db | **不停止**，仅执行迁移与 dump | 无中断；迁移是事务性 DDL，秒级 |

**预计停服**：实测同规模备份 2m45s + 迁移 <10s + 启动 30–60s + 抽查 3–5 分钟 ≈ **8–10 分钟**；
建议申请 **30 分钟窗口**（含回滚余量）。

**窗口内禁止**：提交训练/推理/导入任务、手工改库、重启 Docker、执行其他迁移。

## 5. 镜像准备（窗口前完成，不占停服时间）

在干净工作区、`HEAD=abfd310` 上构建：

```bash
cd D:/毕业设计/bysj-unblock
git status --short            # 必须为空
SHA=$(git rev-parse HEAD)
docker build -f deploy/backend.Dockerfile --target api \
  --build-arg BYSJ_CODE_SHA=$SHA -t bysj-api:f004-$SHA .
docker build -f deploy/backend.Dockerfile --target worker \
  --build-arg BYSJ_CODE_SHA=$SHA -t bysj-worker:f004-$SHA .
```

构建后必须记录并核对：

```bash
docker image inspect bysj-api:f004-$SHA    --format '{{.Id}}'
docker image inspect bysj-worker:f004-$SHA --format '{{.Id}}'
docker run --rm --entrypoint python bysj-api:f004-$SHA \
  -c "import app.version; print(app.version.build_info())"     # code_sha 必须等于 $SHA
docker run --rm --entrypoint python bysj-worker:f004-$SHA \
  -c "import torch,torchvision;print(torch.__version__,torchvision.__version__,torch.version.cuda)"
```

已有证据（同代码构建，`6b06e5b`，演练用）：

| 镜像 | ID | 关键依赖 |
|---|---|---|
| `bysj-api:drill-20261027` | `sha256:f73aa98680f6011cc074ccfb1c883190732386885bd55ddb9efe13013a55f457` | fastapi 0.136.1 / sqlalchemy 2.0.49 / psycopg 3.3.4 / numpy 2.4.4 |
| `bysj-worker:drill-20261027` | `sha256:cfa03bd9320e1b0971fb2e218a5b8fec11b9f2c0f76b7b220ee4a02446c90989` | torch 2.6.0+cu124 / torchvision 0.21.0+cu124 |

已构建并记录的候选（**都不含最终代码，只能作为构建可行性证据**）：

| 镜像 | ID | 代码 | 用途 |
|---|---|---|---|
| `bysj-api:candidate-ddce908` | `sha256:e9e88b31f18f…` | `ddce908` | 候选构建验证 |
| `bysj-worker:candidate-ddce908` | `sha256:d84039e38b19…` | `ddce908` | 同上 |
| `bysj-api:isic-63bff5f` | `sha256:5104dd8b849d…` | `63bff5f` | ISIC smoke |
| `bysj-worker:isic-63bff5f` | `sha256:a9ef8869eddb…` | `63bff5f` | ISIC smoke |

生产迁移仍**必须按最终提交重新构建并记录新 ID**，避免"镜像 SHA 与 HEAD 不符"的追溯缺口。
worker 重建要重新下载 ML 依赖（实测 34–53 分钟，随网络波动），**建议窗口前一天完成构建**。

前端镜像 `bysj-frontend:20261022` 无需重建（前端已包含 `/cxr/*` 页面，见 E2E 证据）。

## 6. 迁移步骤（窗口内）

### 6.1 一致性备份（停写后）

```bash
cd D:/毕业设计/bysj-unblock
PYTHONPATH=scripts python scripts/backup_project.py \
  --project bysj-restore2-20261022 \
  --helper-image bysj-api:f004-$SHA \
  --output "D:/毕业设计/backups/$(date +%Y%m%d-%H%M)-pre-f004"
```

脚本会：停 frontend/api/worker（保留 db）→ 复查无在途任务 → dump → 打包三卷（含逐文件 SHA）→ **自动重启被停的服务** → 写 `backup.json`。
要求 `status=complete`、`restart_failures=[]`。

> **注意：备份脚本会在 `finally` 里自动 `docker start` 被停的容器**（源码 `backup_project.py` 第 103–108 行）。
> 也就是说备份完成后服务会**重新对外提供服务**，此时到迁移之间可能出现新写入。
> 因此备份完成后必须**立即再次停写并核对行数**（见 6.2），不能把"备份完成"当成"已经停写"。

独立复核（不依赖脚本自述）：

```bash
python - <<'PY'
import hashlib, json, pathlib
root = pathlib.Path("D:/毕业设计/backups/<刚生成的目录>")
m = json.loads((root/"backup.json").read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/"database.dump") == m["database_sha256"]
for k, e in m["archives"].items():
    assert sha(root/f"{k}.tar.gz") == e["archive_sha256"]
print("backup verified", m["status"], m["alembic_revision"], len(m["row_counts"]))
PY
```

### 6.2 停写验证（不可跳过）

```bash
# a) 再次停掉所有业务进程（db 保持运行）
docker stop bysj-restore2-20261022-api-1 \
  bysj-restore2-20261022-dataset-worker-1 \
  bysj-restore2-20261022-worker-1 \
  bysj-restore2-20261022-inference-worker-1 \
  bysj-restore2-20261022-frontend-1

# b) 确认没有活跃业务连接（除本机的 psql 外应为空）
docker exec bysj-restore2-20261022-db-1 psql -U bysj -d bysj -tAc \
 "select count(*) from pg_stat_activity
  where datname='bysj' and pid <> pg_backend_pid() and state <> 'idle';"

# c) 再取一次行数，与备份报告的 row_counts 逐表比对
docker exec bysj-restore2-20261022-db-1 psql -U bysj -d bysj -tAc \
 "select 'job_tasks='||count(*) from job_tasks union all
  select 'dataset_assets='||count(*) from dataset_assets union all
  select 'model_registry='||count(*) from model_registry;"
```

与 `backup.json` 的 `row_counts` **必须一致**；不一致说明备份后又有写入，**重新备份**再继续。
只有在"业务进程已停 + 无活跃连接 + 行数与备份一致"三条同时成立时，才允许执行迁移。

### 6.3 迁移数据库（只跑迁移，不启动业务）

```bash
docker run --rm --network bysj-restore2-20261022_default \
  --env-file .runtime/restore2-20261022/.env \
  -e DATABASE_URL="postgresql+psycopg://bysj:<DB_PASSWORD>@db:5432/bysj" \
  --entrypoint python bysj-api:f004-$SHA \
  -m alembic -c /app/backend/alembic.ini upgrade head
```

预期输出：`Running upgrade 4a6b1222f002 -> 4a6b1222f003` 与 `4a6b1222f003 -> 4a6b1222f004`。

### 6.4 迁移后、启动前的校验

```bash
DB=bysj-restore2-20261022-db-1
docker exec $DB psql -U bysj -d bysj -tAc "select version_num from alembic_version;"          # 4a6b1222f004
docker exec $DB psql -U bysj -d bysj -tAc "select count(*) from pg_tables where schemaname='public';"  # 28
docker exec $DB psql -U bysj -d bysj -tAc "select tablename from pg_tables where schemaname='public' and (tablename like 'cxr%' or tablename like 'isic%') order by 1;"  # 6 行
docker exec $DB psql -U bysj -d bysj -tAc "select count(*) from pg_constraint where contype='f' and connamespace='public'::regnamespace and not convalidated;"  # 0
```

旧表行数必须与备份报告逐表一致（演练已验证：迁移前后零差异）。

### 6.5 切换镜像并启动

```bash
cd D:/毕业设计/bysj-unblock/.runtime/restore2-20261022
# 编辑 .env：BYSJ_API_IMAGE / BYSJ_WORKER_IMAGE / BYSJ_CODE_SHA
docker compose --env-file .env -p bysj-restore2-20261022 \
  -f compose.yml -f compose.formal.yml -f compose.gpu.yml --profile workers up -d
```

`migrate` 服务会再跑一次 `upgrade head`（已是 head，幂等）。

## 7. 启动后验证清单

| 类别 | 检查 | 通过标准 |
|---|---|---|
| 健康 | `GET /health/live`、`/health/ready` | 均 200，ready 三角色非零 |
| 版本 | `GET /api/v1/version` 或 `docker exec api python -c "import app.version;print(app.version.build_info())"` | code_sha = `$SHA` |
| 旧 BCCD 数据 | `GET /api/v1/datasets` | 23 条，与迁移前一致 |
| 旧 BCCD 版本 | `GET /api/v1/datasets/{id}` 与版本摘要 | 资产数、四集计数与迁移前一致（抽查 1–2 个版本） |
| 旧 BCCD 模型 | `GET /api/v1/models` | 22 条活跃模型 |
| 旧 BCCD 页面 | 浏览器打开 `/raw-dataset`、`/model-hub`、`/inference-service` | 可加载、列表非空 |
| 新方向 | `GET /api/v1/directions`、`/api/v1/cxr/schema`、`/api/v1/isic/schema` | 200，3 个方向、5 标签、7 类 |
| 新方向空态 | `GET /api/v1/cxr/datasets`、`/api/v1/isic/datasets` | 200 且为空列表（真实数据未导入） |
| 库结构 | 版本 f004、28 表、6 张新表、0 未校验外键 | 同上 |
| 任务系统 | 提交一个最小 BCCD 推理任务（可选） | 状态 completed，GPU 租约正常释放；**会产生新记录** |

可选功能 smoke 若执行，需在报告中标注其产生的记录 ID，避免与历史结果混淆。

## 8. 失败处理与回滚

**决策点**：迁移失败 / 启动后健康不通过 / 旧功能抽查失败 → 立即回滚，不做现场调试。

| 方案 | 步骤 | 代价 | 适用 |
|---|---|---|---|
| **A. 备份恢复（推荐）** | 用 6.1 的备份 `restore_project.py` 恢复到新项目 `bysj-restore3-<ts>`（新卷、新端口），验证后把前端/反代指向新项目 | 约 3–5 分钟；旧项目保留只读 | 任何失败 |
| **B. 降级 + 旧镜像** | `alembic downgrade 4a6b1222f002` → 改回旧镜像 → 启动 | 约 2 分钟；**会永久丢失 CXR/ISIC 数据** | 仅当新表数据可丢弃 |

**首选 A**：不依赖降级正确性，且旧库原样保留。

**B 的实测结果（2026-10-28，隔离库 PG）**：`downgrade 4a6b1222f002` 成功，
22 张旧表行数**逐表零差异**、6 张新表被删除、外键约束归零；再 `upgrade head` 也成功。
但降级**不可逆地丢弃新表数据**（`cxr_samples`/`isic_samples` 归 0），并留下
**14 条 `dataset_assets` 与 14 个数据卷文件成为孤立资产**（不会自动清理）。

因此 B 只在"新表数据可丢弃、且接受孤立资产人工清理"时使用；
需要保留 CXR/ISIC 数据时**只能用 A**。详见 `MIGRATION-READINESS-20261028.md` 第 4 节。

回滚后必须：恢复服务 → `/health/ready` 200 → 抽查数据集/模型数量 → 记录失败原因与镜像 ID。

## 9. 风险与仍未满足的条件

1. **GPU 未参与演练**：演练栈未挂 GPU，迁移后 GPU 训练/推理的实际可用性仍属未验证（代码路径在目标镜像内单独测过）。
2. **PG 降级已演练**（2026-10-28）：可用，但会丢新表数据并留下孤立资产，故仍以方案 A 为主。
3. **备份未加密、仅本机**：窗口前应确认磁盘余量，必要时复制到第二块盘。
4. **真实数据未就绪**：CXR 无数据、ISIC 图像未齐（见第 10 节），迁移后新方向只有空态页面，不产生任何指标。
5. **浏览器 E2E 针对隔离环境**：生产切换后建议手工走一遍 CXR 空态页面。
6. **前端镜像不重建**：`bysj-frontend:20261022` 已含新页面，但若前端代码在 `abfd310` 之后有改动需重新构建。

## 10. 建议窗口

| 项 | 建议 |
|---|---|
| 时长 | 30 分钟（预计实际 8–10 分钟） |
| 时段 | 无他人使用、可连续值守的时段；不建议在演示/答辩前 24 小时内 |
| 前置 | 第 3 节门禁全绿 + 镜像已构建验证 + 备份目录可写 |
| 顺序 | 备份 → 迁移 → 校验 → 切镜像 → 启动 → 抽查 → 记录 |
| 失败即停 | 任一步不符合预期，直接按第 8 节 A 回滚，不现场改代码 |

## 11. 执行清单（勾选后执行）

- [ ] 确认窗口时间，通知使用者
- [ ] 第 3 节门禁 6 项全部通过并截图/记录
- [ ] 新镜像已构建，ID 与 `BYSJ_CODE_SHA` 已记录
- [ ] `backup_project.py` 输出 `complete` 且独立复核哈希一致
- [ ] `alembic upgrade head` 输出 f002→f003→f004
- [ ] 版本 = f004、28 表、6 新表、0 未校验外键
- [ ] 旧表行数与备份报告逐表一致
- [ ] 服务启动，`/health/ready` 200
- [ ] 旧 BCCD：数据集 23 / 模型 22 / 抽查版本计数一致
- [ ] 新方向：directions、cxr/schema、isic/schema 200，数据集空列表
- [ ] 记录新镜像 ID、迁移耗时、验证结果到 `docs/directions/`
- [ ] 保留迁移前备份，24 小时内不删除
