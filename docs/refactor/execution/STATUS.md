# 执行状态

> **历史快照提示（2026-10-22 续作）**：本文件下文保留原 DS 交接事实。Docker、PG、GPU、前端、清退与备份恢复已经有后续验证；当前执行状态、精确代码与实验进度以 [续作状态](unblock/STATUS.md) 为准。不要再据下文的旧 BLOCKED 判定当前环境。

> 文档性质：当前实施状态快照，不是完成报告。
> 更新时间：2026-10-22 01:35（America/Los_Angeles）
> 分支：`graduation/pr-00-baseline`　基线：`c47950b`
> 结论：**新入口可运行、核心链路有真实证据；但容器、PostgreSQL、GPU、正式实验、旧代码清退未完成，不能宣称项目完成或 P0 全绿。**

---

## 1. 快速事实

| 项目 | 状态 |
|---|---|
| 新后端 `backend/app` | 6,845 行，可导入、可启动、68 个测试通过 |
| 旧后端核心文件 | 相对基线**零改动**，旧入口仍可独立启动 |
| 数据库表 | 沿用 9 张旧表 + 新增 12 张表；Alembic 空库建 21 表通过 |
| 依赖 | 43 个包锁定 sha256，空 venv `--require-hashes` 安装通过 |
| 部署配置 | Compose（含 GPU override）`config` 通过；**镜像构建/容器启动未验证** |
| 数据 | BCCD 364 图 / 4,888 框 / 291-37-36，完整性检查通过 |
| 训练 | dummy 全链路通过；**真实 CPU 1 epoch 冒烟通过**；GPU 未验证 |
| 前端 | 仅 Login 迁移到新客户端；其余 6 个页面仍走旧端点 |
| 正式实验 | E1–E4 全部未执行 |

---

## 2. 阶段状态

| 阶段 | 状态 | 说明 | 记录 |
|---|---|---|---|
| PR-00A | **PASS**（GPU 锁 PENDING） | 基线 186 文件校验、环境探测、配置映射 | [PR-00A](PR-00A.md) |
| PR-00B | **PARTIAL** | 锁/迁移/Compose 配置/前端构建 PASS；镜像与容器 BLOCKED | [PR-00B](PR-00B.md) |
| PR-01 | **PASS** | 应用工厂、鉴权、envelope、路由白名单、前端 client | [PR-01](PR-01.md) |
| PR-02 | **PASS（契约）** | 导入/去重/三集/冻结/引用保护；PG 竞争 BLOCKED | [PR-02](PR-02.md) |
| PR-03 | **PARTIAL** | 标注 CAS/分页/导出 PASS；真实浏览器未验 | [PR-03](PR-03.md) |
| PR-04 | **PARTIAL** | 队列/幂等/取消/中断恢复 PASS；CPU 真实训练冒烟；GPU BLOCKED | [PR-04](PR-04.md) |
| PR-05 | **PARTIAL** | 模型注册/血缘/评估契约 PASS；真实指标 BLOCKED | [PR-05](PR-05.md) |
| PR-06 | **PARTIAL** | 推理持久化/部分失败/导出 PASS；GPU 计时未测 | [PR-06](PR-06.md) |
| PR-07 | **PARTIAL** | BCCD 数据准备、P0 形态链路 PASS；E1 NOT_RUN | [PR-07](PR-07.md) |
| PR-08 | **PARTIAL** | 复核/审批/回流隔离 PASS；E3 NOT_RUN | [PR-08](PR-08.md) |
| PR-09 | **未完成** | 旧入口/行业模块未清退，前端未全量迁移 | [PR-09](PR-09.md) |

---

## 3. 已实测证据

| # | 检查 | 命令 | 退出码 | 结果 |
|---|---|---|---|---|
| 1 | 基线一致性 | `python scripts/verify_refactor_baseline.py` | 0 | 186 个受保护文件与固定 blob 一致 |
| 2 | 后端测试 | `pytest backend/tests/medical backend/tests/test_app_factory_auth.py -q` | 0 | **68 passed** |
| 3 | 前端测试 | `node --test tests/*.test.mjs src/utils/*.test.mjs` | 0 | **27 passed** |
| 4 | 前端构建 | `npm run build` | 0 | `dist/` 产物，JS 1.4 MB（gzip 456 kB） |
| 5 | 锁安装 | `pip install --require-hashes -r backend/requirements.lock.txt`（空 venv） | 0 | 43 个包安装成功 |
| 6 | 空库迁移 | `alembic -c backend/alembic.ini upgrade head`（SQLite） | 0 | 21 张业务表 |
| 7 | Compose 配置 | `docker compose ... config --quiet`（含 GPU override） | 0 | 变量/卷/端口合法 |
| 8 | BCCD 转换 | `python scripts/import_bccd.py ...` | 0 | 364 图 / 4,888 框 / 291-37-36 |
| 9 | 划分完整性 | `python scripts/check_split_integrity.py ...` | 0 | 364 个文件重算 hash，无交集 |
| 10 | 端到端冒烟 | `python scripts/smoke_medical.py ... --images data/prepared/bccd/images` | 0 | 登录→导入→标注→冻结→训练→评估→推理→复核回流 |
| 11 | 真实训练（CPU） | 经 API 提交，ultralytics 8.4.183 / torch 2.14.1+cpu | completed | 模型 sha `9c98eee5…`；P=0.713 / R=0.311 / mAP50=0.250 |

### 真实训练说明（避免误读）

- 12 张图、1 epoch、batch=2、imgsz=160、seed=42、`device=cpu`；
- val 只有 **1 张图 / 17 个实例**，上述指标**不具统计意义**；
- 这是链路冒烟，不是 E1，也不是 GPU 证据；正式实验仍需固定协议与加速设备。

---

## 4. 新旧代码关系（重要）

| 层面 | 事实 |
|---|---|
| 代码导入 | `backend/app` **不导入** `yolo_api` / `database` / `dataset_versioning` / `annotation_drafts` / `job_store` / `model_registry` |
| 旧文件改动 | 上述旧模块相对 `c47950b` **零改动**；`start_api.py` 仍可启动旧应用 |
| 数据契约 | 沿用 9 张表：`users`、`dataset_resources`、`dataset_versions`、`dataset_assets`、`annotation_drafts`、`annotation_draft_assets`、`job_tasks`、`job_events`、`model_registry` |
| 新增表 | 12 张：`auth_sessions`、`dataset_imports`、`evaluation_runs`、`inference_runs/items/predictions`、`review_records/items`、`feedback_exports`、`experiment_runs`、`artifact_records`、`resource_leases` |
| 前端 | 7 个页面中只有 `Login.vue` 使用 `api/client.js`；其余仍调用旧端点 |
| 测试 | 新 68 个测试只覆盖新实现；旧测试只覆盖旧实现，无交叉回归 |

**定性**：v1 后端是**平行重写**，不是增量改造，也不是“基本无关联”——它共享数据库 schema 与部分算法语义
（组级三集划分、revision CAS、安全解压），但代码独立；仓库整体与当前前端运行路径仍以旧代码为主。

这偏离了任务书“优先复用已核验源码符号”的要求，代价与关闭条件见 `docs/architecture.md` 第 5 节。

---

## 5. 未完成与阻塞

| 优先级 | 项目 | 状态 | 解除条件 |
|---|---|---|---|
| P0 | Docker 镜像构建与 `compose up` | BLOCKED | 启动 Docker 守护进程；`build --no-cache` + `up -d` |
| P0 | PostgreSQL 迁移与并发验证 | BLOCKED | 独立 PG 16；V10/V13 |
| P0 | GPU 训练/推理与 ML 依赖锁 | BLOCKED | CUDA 环境；生成 `requirements-ml.lock.txt` |
| P0 | E1 正式实验（80 epoch / seeds） | NOT_RUN | 固定评估协议 + GPU |
| P0 | 旧入口/行业模块清退 | NOT_DONE | 前端全量迁移后删除，补 E2E |
| P1 | 前端 6 个页面迁移 | NOT_DONE | Dataset/Annotation/Training/Inference/ModelHub/DatasetManager |
| P1 | 真实浏览器 20 图 / 3 页流程 | NOT_RUN | 安装 Playwright 或人工录屏 |
| P1 | E2 切片、E3 反馈对比、E4 性能 | NOT_RUN | 依赖 E1 与 GPU |
| P1 | 旧库副本迁移演练 | NOT_RUN | 需旧数据库 dump（禁止生产直连） |
| P2 | 空环境备份恢复演练 | NOT_RUN | 需容器环境 |
| P2 | 论文正式实验章节/图表 | NOT_RUN | 依赖 E1–E4 真实结果 |
| — | 学校成果认定/展示授权 | PENDING | 项目所有者确认，AI 不代签 |

---

## 6. 提交记录

| 提交 | 内容 |
|---|---|
| `c47950b` | 任务书/基线（原始交付） |
| `d72ae2f` | PR-00A 环境契约与探测脚本 |
| `de4863d` | PR-00B 依赖锁、部署栈、Alembic 基线、CI |
| `aa4ec41` | 医学应用主体（数据/标注/训练/推理/复核）与测试 |
| `8919506` | 架构/数据库/API/运行手册/论文提纲/阶段报告 |
| `537d0c4` | 构建元数据刷新 |
| `fd2387f` | 记录新旧代码关系与偏差 |

> 说明：应用代码以“一个可导入的整体”提交，未按 PR 逐卡拆分 commit；各卡证据在
> `docs/refactor/execution/PR-XX.md`，不能据此做逐提交 bisect。

---

## 7. 下一步（按依赖顺序）

1. **启动 Docker** → 构建 api/frontend → `compose up` → 补 V08/V09；用独立 PG 跑 V10。
2. **生成 ML 锁**（GPU 镜像内）→ 跑真实 CUDA 训练/推理 → 补 PR-04/05/06 的 GPU 证据。
3. **冻结评估协议** → E1（seed 42/43/44，或明确单 seed 限制）→ E3 → E2/E4。
4. **迁移剩余 6 个前端页面**到 `api/client.js` → 移除 `main.js` 全局补丁 → 删除旧入口/行业模块。
5. **旧库副本演练**：`alembic stamp` + 增量迁移，核对行数与 hash。
6. **空环境恢复演练** + 最终 release PR（commit/镜像 digest/lock hash/数据 manifest/模型 hash）。

---

## 8. 复现命令

```bash
# 基线
python scripts/verify_refactor_baseline.py

# 后端测试
PYTHONPATH=backend python -m pytest backend/tests/medical \
  backend/tests/test_app_factory_auth.py -q

# 前端
cd frontend && npm ci && node --test tests/*.test.mjs src/utils/*.test.mjs && npm run build

# 数据准备与完整性
python scripts/import_bccd.py --source <BCCD>/BCCD --output data/prepared/bccd \
  --config configs/experiments/bccd-baseline.json
python scripts/check_split_integrity.py --manifest data/prepared/bccd/manifest.json

# 端到端冒烟（需先启动 API，账号从环境变量读取）
python scripts/smoke_medical.py --base-url http://127.0.0.1:8000 \
  --config configs/experiments/bccd-smoke.json --evidence artifacts/smoke \
  --images data/prepared/bccd/images
```

---

## 9. 诚实声明

- 本文件中的数字均来自实际执行，未填猜测值；未执行项明确标为 BLOCKED / NOT_RUN。
- 没有用 mock、CPU 冒烟或旧报告替代真实 GPU/医学验收。
- 在 P0 容器、PG、GPU、E1 与旧代码清退完成前，**不得宣称项目完成**。
