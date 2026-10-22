# PR-00A 执行记录：环境契约与冻结证据

状态：**PASS（CPU/契约部分）；GPU 锁 GPU_PENDING**
执行时间：2026-10-21 23:54 — 2026-10-22 00:10（America/Los_Angeles）
基线提交：`c47950b7a08565003730435653b3fe1daf78b6df`（分支 `graduation/pr-00a`）

## 1. 基线校验

| 命令 | cwd | 退出码 | 结论 |
|---|---|---|---|
| `python scripts/verify_refactor_baseline.py` | 仓库根 | 0 | `status=PASS`，186 个受保护文件与固定 blob 一致 |

原始输出：

```json
{"status":"PASS","checked":186,"expected_parent":"401cef475718e6adb00007b5e4d1b1c921ed91c4","errors":[],
 "scope":"Initial protected files only; not a functional or security test."}
```

## 2. 环境探测

| 命令 | 退出码 | 证据 |
|---|---|---|
| `python scripts/environment_probe.py --output docs/refactor/execution/env-probe-pr00a.json` | 0 | [env-probe-pr00a.json](env-probe-pr00a.json) |

关键结果：

- Windows 11 / AMD64 / 24 逻辑核；
- RTX 4060 Laptop 8188 MiB，驱动 617.14，CC 8.9（`GPU_PENDING`：未验证容器内可用）；
- Docker CLI 29.4.1 / Compose v5.1.3，**守护进程未启动（CLI_ONLY）**；
- Python 3.13.14（隔离 venv），Node 22.22.2，npm 10.9.7；
- 本机无 PostgreSQL 实例。

## 3. 依赖解析

新增 `backend/requirements.in`（16 个直接依赖）与 `backend/requirements-ml.in`。
在隔离 venv 中解析得到 43 个包，写入 `backend/requirements.lock.txt`（带 sha256）。

| 检查 | 结果 |
|---|---|
| `pip install --dry-run --report` | 退出 0，43 个包 |
| 锁定版本可安装 | 是（见 PR-00B V06） |
| 是否连接外部业务资源 | 否；仅访问 PyPI 元数据 |

## 4. 配置变量映射

| 用途 | 固定变量 | 兼容别名（仅在 config.py） |
|---|---|---|
| 数据库 | `DATABASE_URL` | 无 |
| 鉴权 | `JWT_SECRET_KEY` | 无 |
| 数据根 | `DATA_ROOT` | `DEEPVISION_DATA_DIR` / `DEEPVISION_PROJECT_ROOT` |
| 模型根 | `MODEL_ROOT` | `TRAINING_LOG_ROOT` |
| 产物根 | `ARTIFACT_ROOT` | 无 |
| 对外地址 | `PUBLIC_ORIGIN` | 无 |

生产环境缺少 `JWT_SECRET_KEY` 或长度 <32 时，`Settings` 直接抛错；不生成默认弱口令。

## 5. 未执行项与阻塞

| 项目 | 状态 | 原因/解阻 |
|---|---|---|
| GPU 容器可见性 | GPU_PENDING | Docker 守护进程未启动；需 `docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi` |
| GPU 依赖锁 | NOT_RUN | 需在 GPU worker 镜像内解析；CPU 锁不可冒充 |
| 生产环境变量核对 | NOT_RUN | 无生产访问权限，且任务书禁止操作公司环境 |

## 6. 回退

本阶段只新增文档、`requirements.in`、探测脚本与执行记录；`git revert <commit>` 即可，
不影响既有应用代码。
