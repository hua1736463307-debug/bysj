# 验证记录（2026-10-27 第二轮）

> 目标：完成上一轮遗留的隔离环境、PostgreSQL、浏览器 E2E、备份验证。
> 全部为本机当场执行；未执行项单列并写明原因。

## 1. 隔离环境（已完成）

| 项 | 结果 |
|---|---|
| 数据/权重隔离 | 数据集放 `D:/毕业设计/datasets/`、备份放 `D:/毕业设计/backups/`，均在 Git 仓库之外 |
| 仓库内产物 | `artifacts/`、`frontend/dist/`、`.runtime/` 均被 `.gitignore` 覆盖，未进提交 |
| 指标对照环境 | 独立 venv 安装 scikit-learn 1.5.2，仅用于对照测试，不进镜像 |
| 磁盘 | C 197.8 GB / D 85.7 GB 可用；ISIC 全量 3.2 GB、备份 0.9 GB 均在余量内 |

## 2. PostgreSQL（已完成）

运行中的 `bysj-restore2-20261022-db-1`（PostgreSQL 16，宿主端口 55437）内新建**专用**测试角色与库：

```sql
CREATE ROLE bysj_test_agent LOGIN PASSWORD '***';
CREATE DATABASE bysj_test_agent OWNER bysj_test_agent;   -- 名称以 bysj_test_ 开头
```

```
BYSJ_TEST_DATABASE_URL=postgresql+psycopg://bysj_test_agent:***@127.0.0.1:55437/bysj_test_agent
pytest tests/integration -q
```

**15 passed**（含 job 并发、资源租约、worker 存活、旧库迁移、标注并发）。

首轮 14 passed / 1 failed：`test_legacy_migration` 的 `EXPECTED_NEW_TABLES` 未含新增的
`isic_samples`。这正说明该清单守卫有效——加表未更新清单会立刻失败；补齐后 18 张表一致，全部通过。

## 3. 浏览器 E2E（已完成）

环境：本地 API（`uvicorn`，SQLite + `INLINE_JOBS=true` + `CXR_ML_BACKEND=dummy`）+ 静态站点与
`/api` 反向代理（127.0.0.1:5199）+ Playwright（Edge 通道）+ 合成 CXR fixture（40 患者，显式 synthetic）。

```
BYSJ_E2E_URL=http://127.0.0.1:5199 BYSJ_E2E_API=http://127.0.0.1:5199/api/v1 \
BYSJ_E2E_CXR_SOURCE_ID=e2e-cxr npx playwright test e2e/cxr-flow.spec.cjs
```

**1 passed（10.9s）**，覆盖：登录 → 来源导入 → 四集冻结 → 病例分析（五项评分 + 阈值 + 提示）→
关注区域 → 逐标签复核 → 提交 → 批准 → 签发最终报告 → 导出 HTML（断言 `/export` 返回 200 + `text/html`）。

过程中修掉 4 个真实问题（都不是"测试写错"了事）：

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | 浏览器空白页 | 静态服务器把 `.js` 标成 `text/plain`（Windows 注册表），浏览器拒绝执行 module | 显式设置 `extensions_map` |
| 2 | 导入作业 failed | CXR 仍只读旧 `CXR_SOURCES`，未接通用 `DATASET_SOURCES` | 抽出 `app/core/sources.py` 统一解析，CXR 保留 `cxr_` 错误码前缀 |
| 3 | `approved` 断言超时 | 表格单元格与标题同时含该词，strict mode 多重匹配 | 断言限定 `h2` |
| 4 | 导出下载事件超时 | `saveBlob` 走 blob URL，不触发 Playwright download 事件 | 改为断言导出接口状态与 MIME |

## 4. 备份（已完成）

```
python scripts/backup_project.py --project bysj-restore2-20261022 \
  --helper-image bysj-api:grouped-resume-20261023-0215 \
  --output D:/毕业设计/backups/20261027-before-cxr
```

| 项 | 结果 |
|---|---|
| 前置条件 | 无 pending/running/cancelling 作业（查询为 0） |
| 状态 | `complete`，`restart_failures: []` |
| 产物 | `database.dump` 1.46 MB、`data.tar.gz` 902 MB、`models.tar.gz`、`artifacts.tar.gz` 24.6 MB，各卷附 `*-files.json` 清单 |
| 表数 | 22（当前运行库在 `4a6b1222f002`，**不含 CXR/ISIC 新表**） |
| 服务恢复 | api / worker / dataset-worker / inference-worker 重启后 healthy；db 未重启（up 3h） |

## 5. 未执行项（如实记录）

| 项 | 状态 | 原因 |
|---|---|---|
| 备份**恢复**到全新项目/卷 | **NOT_RUN** | 当前运行库停在 `f002`，不含 CXR/ISIC 表；此时恢复只能验证旧 BCCD 数据，无法验证本轮新增结构。应在"新镜像 + 迁移到 f004"之后执行，命令与前置条件已备（`scripts/restore_project.py`，需全新项目名、卷与空闲端口） |
| ISIC 训练集图像下载 | **PARTIAL** | 实测带宽 9–12 KB/s，2.77 GB 需约 10 小时；验证集 53 MB 仍在下载 |
| 真实 GPU 训练 smoke | **NOT_RUN** | 依赖上一项 |
| 100 图性能测试 | **NOT_RUN** | 依赖真实模型与数据 |

## 6. 可复现命令汇总

```bash
# 契约与方向测试
pytest backend/tests/cxr backend/tests/isic -q
# PG 集成
BYSJ_TEST_DATABASE_URL=... pytest tests/integration -q
# 数据审计
python scripts/prepare_isic.py --source-root <ISIC> --output artifacts/isic-audit
# E2E（需先起 API 与静态站点）
BYSJ_E2E_URL=... BYSJ_E2E_API=... npx playwright test e2e/cxr-flow.spec.cjs
# 备份
python scripts/backup_project.py --project <p> --helper-image <img> --output <dir>
```
