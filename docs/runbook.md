# 运行手册

## 1. 前置

- Docker Engine + Compose v2（或 WSL2 + Docker Desktop）
- 可选 GPU：NVIDIA 驱动 + nvidia-container-toolkit
- 端口：`8080`（前端）、`8000`（API）、`5433`（DB）默认只绑定 `127.0.0.1`

## 2. 首次启动

```bash
cp deploy/.env.example deploy/.env
# 必填：DB_PASSWORD、JWT_SECRET_KEY（≥32 字符）、BOOTSTRAP_ADMIN_PASSWORD
python -c "import secrets;print(secrets.token_urlsafe(48))"

docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml config --quiet
docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml build
docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml up -d db api frontend
```

API 容器启动时自动执行 `alembic upgrade head`；空库会创建 21 张业务表。

验证：

```bash
curl -s http://127.0.0.1:8000/api/v1/health/live
curl -s http://127.0.0.1:8000/api/v1/health/ready
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"<BOOTSTRAP_ADMIN_PASSWORD>"}'
```

前端：<http://127.0.0.1:8080>，子路由刷新由 nginx `try_files` 回落。

## 3. worker

```bash
# 无 GPU：只跑导入/冻结/导出
docker compose ... --profile workers up -d dataset-worker

# GPU：训练与推理（确认 nvidia-smi 在容器可见）
docker compose ... -f deploy/compose.gpu.yml --profile workers up -d worker inference-worker
```

worker 领取数据库任务；不需要 Redis。停止 worker 不会丢任务，重启后
`mark_stale_interrupted` 会把心跳超时的 running 标为 interrupted。

## 4. 停止与清理

```bash
docker compose ... stop            # 停止，保留数据卷
docker compose ... down            # 移除容器，保留卷
# 禁止把 down -v 作为常规步骤
```

## 5. 备份与恢复

```bash
# 备份
docker compose ... exec -T db pg_dump -U bysj bysj > backup-$(date +%F).sql
docker run --rm -v bysj-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/data-$(date +%F).tar.gz -C /data .

# 恢复（空卷）
docker compose ... stop api worker inference-worker dataset-worker
docker compose ... exec -T db psql -U bysj bysj < backup-2026-10-21.sql
docker run --rm -v bysj-data:/data -v "$PWD":/backup alpine \
  tar xzf /backup/data-2026-10-21.tar.gz -C /data
docker compose ... start api frontend
```

恢复后核对：`manifest_sha256`、模型 `sha256`、`inference_runs` 计数与备份前一致。

## 6. 本地开发（不用 Docker）

```bash
python -m venv .venv
.venv/Scripts/pip install --require-hashes -r backend/requirements.lock.txt
export PYTHONPATH=backend DATABASE_URL=sqlite:///./.data/bysj.db
export JWT_SECRET_KEY=dev-only-secret-key-0123456789abcdef
export INLINE_JOBS=true ML_BACKEND=dummy
python -m uvicorn app.main:app --reload --port 8000
```

`INLINE_JOBS=true` 仅用于测试/单机演示：任务仍先落库再执行，但会阻塞请求。

## 7. 常见问题

| 现象 | 排查 |
|---|---|
| 启动报 `JWT_SECRET_KEY` | 生产环境必须 ≥32 字符，且不能使用默认值 |
| `ready` 503 | 看 `components.database`；确认 DB 健康检查通过 |
| 训练任务 `gpu_busy` | 有其它任务持有 `gpu:0` 租约；检查 `resource_leases` |
| 训练 `ml_dependency_missing` | worker 镜像未安装 ML lock；重建 worker target |
| 上传 413 | 检查 `MAX_*` 配置；ZIP 压缩比上限 100 |
| 标注 409 | 其它会话已更新 revision；重新加载或另存子版本 |
| 回流 409 `test_leak_blocked` | 复核资产属于 test/val 或与之同源，禁止回流 |
| 模型 409 `model_tampered` | 文件被替换；重新登记或恢复备份 |
| 前端 404 子路由 | nginx 未应用 `try_files`，检查 `deploy/nginx.conf` |

## 8. 证据与报告

每个阶段记录写入 `docs/refactor/execution/PR-XX.md`；大文件放 `artifacts/`（不入库）。
报告必须包含命令、退出码、时间、环境和失败分类，不以“已通过”代替原始输出。
