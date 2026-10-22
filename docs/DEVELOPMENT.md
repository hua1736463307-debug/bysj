# 开发说明

## 当前入口与环境

后端唯一入口 `app.main:app`；`python backend/start_api.py` 是同一入口的本机启动包装。先执行 Alembic，再分别启动 dataset/trainer/inference worker。API 不导入 Torch，也不启动行业模块。

正式目标为 Linux amd64 / Python 3.12。`requirements-core-linux-py312.lock.txt` 用于 API，`requirements-test-linux-py312.lock.txt` 包含测试工具，`requirements-ml-linux-py312-cu124.lock.txt` 用于真实 GPU worker。锁是实际 Linux pip report 的完整图和 wheel SHA256；Windows 本机环境不等价于正式 Linux 验收。

```sh
python -m pip install --require-hashes -r backend/requirements-test-linux-py312.lock.txt
python -m pip check
PYTHONPATH=backend APP_ENV=test python -m pytest backend/tests/medical backend/tests/test_app_factory_auth.py scripts/tests -q
```

旧行为对照从 Git 固定提交 `113181f498bac6d16988f97261ad628c9cdc8ea8` 读取纯函数，CI 需要完整历史。离线容器可先执行 `python scripts/export_legacy_fixtures.py --output .runtime/legacy-fixtures`，再把 `BYSJ_LEGACY_FIXTURES` 指向导出目录；逐文件 hash 校验，不加载旧应用启动逻辑，不把历史源码装入运行镜像。

PostgreSQL 集成测试独立于 SQLite fixture：设置专用本地 `BYSJ_TEST_DATABASE_URL`，数据库名必须以 `bysj_test_` 开头，然后执行 `PYTHONPATH=backend python -m pytest tests/integration -q`。每次用随机 schema，Alembic 初始化，不得指向共享库。

前端使用 Node 22：在 `frontend/` 执行 `npm ci`、`npm test`、`npm run build`。浏览器用例需要真实独立后端、数据及用户，配置 `BYSJ_E2E_URL`、`BYSJ_E2E_API`（完整 `/api/v1` URL）、`BYSJ_E2E_USERNAME`、`BYSJ_E2E_PASSWORD`、`BYSJ_E2E_DATA` 后执行 `npm run test:e2e`；凭据仅放进程环境。当前正式环境分别为 `http://127.0.0.1:18084` 与 `http://127.0.0.1:18004/api/v1`，不要误用用例的旧默认 API 端口 18000。浏览器验收会创建样本与 GPU 任务，应在正式实验空闲时执行。

Windows 沙箱若无法创建系统临时文件，先把 `TEMP/TMP` 设置为本工作区专用临时目录再运行 pytest；不得把临时目录权限故障当作业务断言失败，也不要为解决它更改系统目录权限。

## 本机开发

根目录 `.env.example` 用于非 Docker 开发；Compose 使用 `deploy/.env.example`。不要把容器内 `/data` 等路径照搬到 Windows 本机。生产配置必须显式 PostgreSQL、足够长的非开发 JWT 密钥、`INLINE_JOBS=false`。测试模式不读取本地部署 dotenv。

数据、权重、下载缓存、数据库副本、认证信息和 Playwright trace 均不提交。实验配置、代码 SHA、锁 hash、数据版本与初始权重 hash 必须可追溯；正式协议冻结后不能按 test 指标调参。
