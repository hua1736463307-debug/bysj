# PR-00B 执行记录：构建、部署与迁移基线

状态：**部分通过**（锁/Compose 配置/迁移/前端构建 PASS；镜像构建与容器启动 BLOCKED）
执行时间：2026-10-22 00:10 — 00:50（America/Los_Angeles）
基线：`graduation/pr-00a` 之后的 `c47950b` + 本阶段变更

## 1. 依赖锁

| 命令 | 退出码 | 证据 |
|---|---|---|
| `python -m pip install --dry-run --ignore-installed --report .locks/report-core.json -r backend/requirements.in` | 0 | 43 个包 |
| `python scripts/generate_lock.py --report ... --output backend/requirements.lock.txt` | 0 | `packages=43 failures=[]` |
| `python -m venv .venv-locktest && pip install --require-hashes -r backend/requirements.lock.txt` | 0 | `lock install ok 0.159.1` |

锁文件含直接与传递依赖，所有发行文件 sha256 均收录，Linux/Windows 均可校验。

ML 锁：`backend/requirements-ml.in` 已建立；`requirements-ml.lock.txt` 未生成
（GPU 依赖需在目标镜像内解析，本机 Docker 不可用）。worker target 在缺少 ML 锁时
会显式报警并只能运行 dummy 后端，不伪装 GPU 就绪。

## 2. Compose 配置

| 命令 | 退出码 |
|---|---|
| `docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml config --quiet` | 0 |
| 追加 `-f deploy/compose.gpu.yml config --quiet` | 0 |

`deploy/compose.yml` 提供 db/api/frontend，worker/dataset-worker/inference-worker 使用
`profiles: ["workers"]`；卷名统一 `bysj-*`；端口默认绑定 `127.0.0.1`。
GPU 设备保留只在 `compose.gpu.yml` 追加，基础文件不含设备声明。

## 3. 数据库迁移

| 命令 | 环境 | 退出码 | 结果 |
|---|---|---|---|
| `PYTHONPATH=backend python -m alembic -c backend/alembic.ini upgrade head` | 空 SQLite | 0 | 21 张业务表 + `alembic_version` |
| 同上 | 空 PostgreSQL 16 | NOT_RUN | BLOCKED：本机无 PG/Docker |

迁移脚本：`backend/migrations/versions/304f405af95f_initial_medical_platform_schema.py`。
已有实验库升级需 `alembic stamp` 后按增量迁移，见 `docs/database.md`。

## 4. 前端

| 命令 | 退出码 | 结果 |
|---|---|---|
| `npm ci` | 0 | 依赖安装完成 |
| `node --test tests/*.test.mjs src/utils/*.test.mjs` | 0 | 27 passed / 0 failed |
| `npm run build` | 0 | `dist/index.html` + assets，1.4 MB JS（gzip 456 kB） |

## 5. 镜像与容器

| 命令 | 状态 |
|---|---|
| `docker compose build --no-cache` | BLOCKED：Docker 守护进程未启动 |
| `docker compose up -d db api frontend` | BLOCKED：同上 |
| `/health/live`、`/health/ready` | PASS（非容器，SQLite 实例，见 PR-01/PR-07 冒烟） |

`deploy/backend.Dockerfile`（api/worker 双 target）、`deploy/frontend.Dockerfile`、
`deploy/nginx.conf`、`deploy/.env.example`、`deploy/version.json`（由
`scripts/write_version.py` 生成）已就绪。

## 6. CI

新增 `.github/workflows/medical-ci.yml`：PostgreSQL service + 锁安装 + Alembic +
契约测试；前端 Node 测试与构建；Compose config；密钥/绝对路径扫描；GPU 验收为
手动触发 job。本轮未在 GitHub 执行，状态 NOT_RUN。

## 7. 已知问题

1. 容器构建未验证，`version.json` 的 code_sha 为生成时提交，构建后需重新生成。
2. 无 GPU ML 锁，worker 镜像当前只能跑 dummy 后端。
3. `deploy/.env` 为本地生成的测试值，已加入 `.gitignore`（`.env.*` 规则覆盖）。

## 8. 回退

停止本阶段容器（`compose stop`，不删除卷），恢复上一提交；数据库使用备份副本。
禁止把 `down -v` 作为默认回退步骤。
