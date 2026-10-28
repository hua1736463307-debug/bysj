# f004 隔离备份恢复演练（2026-10-27）

> 目的：证明包含旧 BCCD 数据结构与 CXR / ISIC 新表的 **f004** 数据库及相关文件，
> 能够被备份、恢复，并被新代码读取。
>
> 运行中的项目 `bysj-restore2-20261022` **未停止、未重启、未迁移、未改卷**，
> 全部动作发生在两套独立项目、独立数据库、独立卷中。

## 1. 结论

| 判定 | 内容 |
|---|---|
| **PASS** | f002 备份 → 隔离恢复 → 迁移 f004 → 合成记录 → 一致性备份 → 第二套恢复 → 新代码读取，全链路通过 |
| **PASS** | 迁移只新增 6 张表；迁移前后 22 张既有表行数逐表比对 **零差异** |
| **PASS** | 两次恢复的文件清单、数量与 SHA-256 全部一致（11,922 / 24 / 489） |
| **PASS** | 第二套恢复库版本为 `4a6b1222f004`，行数与备份报告逐表一致（28 表 / 14,122 行） |
| **PASS** | 合成记录全程 `synthetic=true` / `research_valid=false`；库内 `research_valid=true` 计数为 **0** |
| **NOT_RUN** | 未在演练环境执行任何训练 / 推理 / 评价作业，因此未验证 BCCD 功能级可用性，只验证了结构与数据 |
| **NOT_RUN** | 运行中的项目仍未迁移，仍为 f002（本次明确不做） |

## 2. 基线（演练开始前）

| 项 | 值 |
|---|---|
| Git HEAD | `e96a4240a217cef10a2ed40b534b5d3e2eb0cf39`（演练中追加 2 个提交，见第 3 节） |
| 工作区 | clean |
| 运行项目 | `bysj-restore2-20261022`（6 个容器，均 healthy） |
| 运行镜像 | api/dataset-worker `bysj-api:grouped-resume-20261023-0215` = `sha256:bbc2418b04b3…`；worker/inference `bysj-worker:grouped-resume-20261023-0215` = `sha256:e129b551c0b4…`；frontend `bysj-frontend:20261022` = `sha256:fb022afabd00…`；db `postgres:16-alpine` = `sha256:57c72fd2a128…` |
| 数据库迁移版本 | `4a6b1222f002` |
| 在途作业 | 0（`pending/running/cancelling/interrupted` 均为空） |
| GPU 租约 | `gpu:0` holder = NULL，generation 68（空闲） |

## 3. 镜像（从干净提交构建）

| 项 | api | worker |
|---|---|---|
| 构建上下文 | `6b06e5ba33ee458f5feec9e3b8dfb9995f6c57f0`（工作区 clean） | 同 |
| 镜像 | `bysj-api:drill-20261027` | `bysj-worker:drill-20261027` |
| 镜像 ID | `sha256:f73aa98680f6011cc074ccfb1c883190732386885bd55ddb9efe13013a55f457` | `sha256:cfa03bd9320e1b0971fb2e218a5b8fec11b9f2c0f76b7b220ee4a02446c90989` |
| 自声明 SHA | `BYSJ_CODE_SHA=6b06e5b…`（镜像 ENV 与 `version.json.code_sha` 一致） | 同 |
| 关键依赖 | fastapi 0.136.1 / sqlalchemy 2.0.49 / psycopg 3.3.4 / numpy 2.4.4 / pillow 13.0.0 | torch 2.6.0+cu124 / torchvision 0.21.0+cu124（CUDA 12.4 编译） |
| 构建耗时 | 7m34s | 34m48s（ML 依赖下载为主） |

构建命令：

```bash
docker build -f deploy/backend.Dockerfile --target api \
  --build-arg BYSJ_CODE_SHA=$(git rev-parse HEAD) -t bysj-api:drill-20261027 .
docker build -f deploy/backend.Dockerfile --target worker \
  --build-arg BYSJ_CODE_SHA=$(git rev-parse HEAD) -t bysj-worker:drill-20261027 .
```

**构建参数是本轮补的**：`deploy/backend.Dockerfile` 之前丢失了 `BYSJ_CODE_SHA` 构建参数，
而 `scripts/restore_project.py` 只接受自带 40 位提交号的镜像，等于新构建的镜像永远无法用于恢复。
提交 `6b06e5b` 补回该参数，并让 `version.json.code_sha` 取自构建参数而不是仓库里的历史值。

> 镜像构建于 `6b06e5b`；随后提交 `82ccf42` 新增 `scripts/seed_drill_synthetic.py`。
> 该文件位于 `scripts/`，Dockerfile 只 `COPY backend` 与 `deploy/version.json`，
> 因此**镜像内容与 `6b06e5b` 完全一致**，两者差异仅为仓库中的演练脚本。

## 4. 源备份完整性（独立复核）

`D:/毕业设计/backups/20261027-before-cxr`，`status=complete`，`alembic_revision=4a6b1222f002`。

| 对象 | 数量 | 字节 | 复核结果 |
|---|---|---|---|
| `database.dump` | — | 1,456,779 | sha256 `956caf597257d5a8…` 与清单一致 |
| `data.tar.gz` | 11,906 文件 | 972,118,537 | 归档与文件清单 sha256 均一致 |
| `models.tar.gz` | 23 文件 | 125,593,352 | 同上 |
| `artifacts.tar.gz` | 489 文件 | 28,543,025 | 同上 |

行数合计 14,081 行 / 22 表（`dataset_assets` 5,255、`inference_predictions` 7,101、`job_tasks` 115 等）。

## 5. 第一套隔离环境：恢复 f002 并迁移

项目 `bysj-drill-f004-20261027`，端口 db 55447 / api 18104 / frontend 18184，
卷 `bysj-drill-f004-20261027-{db-data,data,models,artifacts}`。

```bash
PYTHONPATH=scripts python scripts/restore_project.py \
  --backup "D:/毕业设计/backups/20261027-before-cxr" \
  --output ".runtime/drill-f004-20261027" \
  --source-env ".runtime/restore2-20261022/.env" \
  --project bysj-drill-f004-20261027 \
  --api-image bysj-api:drill-20261027 --worker-image bysj-worker:drill-20261027 \
  --frontend-image bysj-frontend:20261022 \
  --db-port 55447 --api-port 18104 --frontend-port 18184
```

结果 `data_verified_runtime_acceptance_pending`：22 表、11,906 / 23 / 489 个文件哈希逐一校验，
48 条引用关系 0 孤儿，外键 0 未校验，无在途作业。

迁移（仅隔离库）：

```bash
docker compose --env-file .runtime/drill-f004-20261027/.env \
  -p bysj-drill-f004-20261027 \
  -f .runtime/drill-f004-20261027/compose.yml \
  -f .runtime/drill-f004-20261027/compose.formal.yml run --rm --no-deps migrate
```

`4a6b1222f002 → 4a6b1222f003 → 4a6b1222f004`。

| 校验 | 结果 |
|---|---|
| 迁移前 22 表行数 vs 迁移后 | **逐表零差异**（`comm` 比对为空） |
| 新增表 | `cxr_samples` / `cxr_analysis_batches` / `cxr_analyses` / `cxr_reviews` / `cxr_reports` / `isic_samples` |
| 总表数 | 22 → **28** |
| 外键约束 | 7 条，全部 validated |

## 6. 合成记录（明确标记，不作为研究结果）

```bash
docker run --rm --env-file .runtime/drill-f004-20261027/.env \
  -e DATABASE_URL="postgresql+psycopg://bysj:***@db:5432/bysj" \
  --network bysj-drill-f004-20261027_default \
  -v bysj-drill-f004-20261027-data:/data -v bysj-drill-f004-20261027-models:/models \
  -e BYSJ_DRILL_PROJECT=bysj-drill-f004-20261027 -e PYTHONPATH=/repo/backend:/app/backend \
  -v "D:/毕业设计/bysj-unblock:/repo:ro" --entrypoint python bysj-api:drill-20261027 \
  /repo/scripts/seed_drill_synthetic.py \
  --expect-project bysj-drill-f004-20261027 --confirm SYNTHETIC-DRILL --report /report/seed-1.json
```

脚本四重防护：`BYSJ_DRILL_PROJECT` 必须匹配、库必须已在 f004、不得重复播种、`--confirm` 必须为标记字面量。

写入内容：

| 对象 | 数量 | 说明 |
|---|---|---|
| CXR 数据集/版本 | 1 / 1 | `task_type=cxr_multilabel`，`state=frozen`，manifest 自洽 |
| CXR 样本 | 10 | train 6 / val 1 / calibration 2 / test 1；覆盖 positive、negative、uncertain、unmentioned 四态 |
| CXR 模型 | 1 | `metadata_json.synthetic=true`、`research_valid=false` |
| 分析链 | 1 批 / 1 分析 / 1 复核(approved) / 1 报告(final) | 五项评分齐全，报告快照含 synthetic 标记 |
| ISIC 数据集/版本 | 1 / 1 | `task_type=isic_multiclass`，`state=frozen` |
| ISIC 样本 | 4 | nv / mel / bcc / akiec 各 1，单标签 |
| 文件 | 15 | 10 张 CXR + 4 张 ISIC + 1 个 dummy 模型文件 |

训练门禁 `training_gate` 返回空数组（每个标签在 train 均有正例与负例），说明 manifest 结构可被真实逻辑消费。

## 7. 隔离环境一致性备份

```bash
PYTHONPATH=scripts python scripts/backup_project.py \
  --project bysj-drill-f004-20261027 \
  --helper-image bysj-api:drill-20261027 \
  --output "D:/毕业设计/backups/20261027-drill-f004"
```

| 项 | 值 |
|---|---|
| status | `complete`，`restart_failures=[]` |
| alembic_revision | `4a6b1222f004` |
| 表 / 行 | 28 表 / 14,122 行 |
| 新表行数 | `cxr_samples` 10、`cxr_analysis_batches` 1、`cxr_analyses` 1、`cxr_reviews` 1、`cxr_reports` 1、`isic_samples` 4 |
| data | 11,922 文件 / 972,272,434 B（较源备份 +16：10+4 图 + 2 个 manifest） |
| models | 24 文件 / 125,593,447 B（+1 合成模型） |
| artifacts | 489 文件 / 28,543,025 B（未变） |
| database.dump | sha256 `77b942d4e33be548…` |

## 8. 第二套全新环境：恢复并验证

项目 `bysj-drill2-f004-20261027`，端口 db 55457 / api 18114 / frontend 18194。

```bash
PYTHONPATH=scripts python scripts/restore_project.py \
  --backup "D:/毕业设计/backups/20261027-drill-f004" \
  --output ".runtime/drill2-f004-20261027" \
  --source-env ".runtime/drill-f004-20261027/.env" \
  --project bysj-drill2-f004-20261027 \
  --api-image bysj-api:drill-20261027 --worker-image bysj-worker:drill-20261027 \
  --frontend-image bysj-frontend:20261022 \
  --db-port 55457 --api-port 18114 --frontend-port 18194
```

逐项核对：

| 要求 | 结果 |
|---|---|
| 数据库版本为 f004 | `4a6b1222f004` |
| 旧表存在且行数一致 | 28 表集合与备份完全一致，行数差异 **none**，合计 14,122 |
| CXR / ISIC 新表存在 | 6 张全部存在 |
| synthetic 记录与关联一致 | 分析→复核(approved)→报告(final) 链完整，5 项评分，四态取值保留（`uncertain` / `unmentioned`） |
| 文件清单与 SHA-256 | 11,922 / 24 / 489 逐一校验通过；15 个合成文件独立复核 0 不一致 |
| 新代码能读取 | `/cxr/…/summary`、`/cxr/analyses/{id}`、`/cxr/analyses/{id}/image`、`/cxr/reports/{id}/export`、`/isic/…/summary`、`/cxr/models` 全部 **200**；图像字节 sha256 与播种记录一致 |
| 未误标 research_valid=true | 全库 `model_registry.metadata_json->>'research_valid'='true'` 计数为 **0**；模型/版本/分析/报告四处均为 `synthetic=true` / `research_valid=false` |
| 外键未校验数 | 0 |

`/health/ready` 返回 503（`workers` 缺失）——第二套只启动了 db/api/frontend，未启动三个 worker，
这是**预期**的降级，不是缺陷；`/health/live` 为 200，数据库组件为 ok。

## 9. 报告与证据位置

| 文件 | 内容 |
|---|---|
| `D:/毕业设计/backups/20261027-before-cxr/backup.json` | 源备份清单（f002） |
| `D:/毕业设计/backups/20261027-drill-f004/backup.json` | 隔离环境备份清单（f004） |
| `D:/毕业设计/bysj-unblock/.runtime/drill-f004-20261027/restore.json` | 第一套恢复报告 |
| `D:/毕业设计/bysj-unblock/.runtime/drill2-f004-20261027/restore.json` | 第二套恢复报告 |
| `D:/毕业设计/bysj-unblock/artifacts/drill-20261027/seed-1.json` | 合成记录清单与文件哈希 |
| `D:/毕业设计/bysj-unblock/artifacts/drill-20261027/api-read.json` | 第一套 API 读取结果 |
| `D:/毕业设计/bysj-unblock/artifacts/drill-20261027/restore2/read-summary.json` | 第二套 API 读取结果 |

`.runtime/` 与 `artifacts/` 均在 `.gitignore` 内，不进入仓库。

## 10. 未验证 / 未执行

1. **未在演练环境执行任何作业**：BCCD 训练、推理、评价均未运行，因此只证明结构与数据完整，
   未证明恢复后功能级可用。
2. **运行中的项目未迁移**：仍为 f002；本次只验证隔离路径，生产迁移需另立窗口。
3. **GPU 未参与演练**：worker 镜像带 CUDA 12.4 依赖，但演练栈未挂 GPU，未验证 GPU 作业。
4. **第二套未启动 worker**：`/health/ready` 因此为 503（预期）。
5. **未做备份加密与异地存放**：备份含私有数据，当前仅本机磁盘。
6. **恢复耗时未做容量规划**：数据卷约 1 GB 时约 1 分钟；更大数据量未测。

## 11. 演练资源处置

两套隔离项目的容器已停止（`docker compose stop`），**卷保留**以便复核；
未删除任何数据库、卷、备份、模型、数据集或运行服务。如需彻底移除：

```bash
docker compose --env-file .runtime/drill-f004-20261027/.env -p bysj-drill-f004-20261027 \
  -f .runtime/drill-f004-20261027/compose.yml -f .runtime/drill-f004-20261027/compose.formal.yml down -v
docker compose --env-file .runtime/drill2-f004-20261027/.env -p bysj-drill2-f004-20261027 \
  -f .runtime/drill2-f004-20261027/compose.yml -f .runtime/drill2-f004-20261027/compose.formal.yml down -v
```
