# 环境矩阵（PR-00A 实测）

采集时间：2026-10-21（America/Los_Angeles），脚本 `scripts/environment_probe.py`，
原始 JSON：[`refactor/execution/env-probe-pr00a.json`](refactor/execution/env-probe-pr00a.json)。

本文件记录**实际探测结果**，不是设计假设。缺项明确标注，不填猜测值。

## 1. 宿主与设备

| 项目 | 实测值 | 说明 |
|---|---|---|
| 操作系统 | Windows 11 (10.0.26200) | 开发宿主；部署目标为 Linux/WSL2 |
| 架构 | AMD64 | |
| CPU | 24 逻辑核 | |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，CC 8.9 | `nvidia-smi` 可用 |
| 驱动 | 617.14 | 仅证明驱动存在，不代表 CUDA 容器可用 |
| Docker CLI | 29.4.1 | |
| Docker Compose | v5.1.3 | |
| Docker 守护进程 | **CLI_ONLY（未启动）** | 本轮未执行镜像构建与容器启动 |

## 2. 语言与依赖

| 项目 | 实测值 | 目标/说明 |
|---|---|---|
| Python（本机隔离 venv） | 3.13.14 | 设计目标 3.11（镜像 `python:3.11-slim`） |
| Node | 22.22.2 | 与设计一致 |
| npm | 10.9.7 | |
| 核心依赖锁 | `backend/requirements.lock.txt`，43 个包带 sha256 | 空 venv `--require-hashes` 安装通过 |
| ML 依赖 | `backend/requirements-ml.in` | 是否生成哈希锁见 PR-00B 记录 |
| PostgreSQL | 本机未安装 | 由 `deploy/compose.yml` 提供 16-alpine |
| GPU 训练依赖 | 本机 venv 未预装 torch/ultralytics | 真实训练需 worker 镜像或 ML venv |

## 3. 依赖解析结果（核心）

| 包 | 锁定版本 |
|---|---|
| fastapi | 0.159.1 |
| pydantic | 2.14.1 |
| sqlalchemy | 2.1.6 |
| alembic | 1.20.1 |
| psycopg | 3.3.6 |
| uvicorn | 0.54.0 |
| python-jose | 3.5.0 |
| passlib | 1.7.4 |
| bcrypt | 4.0.1（passlib 1.7.4 兼容上限） |
| pillow | 13.0.0 |
| numpy | 2.5.4 |

安装验证命令与结果：

```bash
# 空环境安装（真实执行，退出码 0）
python -m venv .venv-locktest
.venv-locktest/Scripts/python.exe -m pip install --require-hashes -r backend/requirements.lock.txt
.venv-locktest/Scripts/python.exe -c "import fastapi,sqlalchemy,alembic,jose;print('ok')"
# -> lock install ok 0.159.1
```

## 4. 环境结论

- **CPU/契约工作可执行**：核心依赖、Alembic、API 测试、前端构建均在本机实测通过。
- **GPU 训练门禁未完成**：本机 venv 未安装 torch，且 Docker 守护进程未启动；
  `GPU_PENDING` 不得写成通过。解阻命令：
  ```bash
  docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml \
    -f deploy/compose.gpu.yml --profile workers build worker
  ```
- **PostgreSQL 并发证据缺失**：本机无 PG 实例，SQLite 测试不能证明 advisory lock/
  行锁竞争；V13 标记 BLOCKED，需在有 Docker 或独立 PG 的环境补跑。
- 配置变量统一为 `DATABASE_URL / JWT_SECRET_KEY / DATA_ROOT / MODEL_ROOT /
  ARTIFACT_ROOT / PUBLIC_ORIGIN`；旧变量只在 `app/core/config.py` 做兼容映射。
