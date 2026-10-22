# 执行状态

更新时间：2026-10-22 04:10（America/Los_Angeles）
执行分支：`graduation/pr-00-baseline`
基线提交：`c47950b7a08565003730435653b3fe1daf78b6df`

| 阶段 | 状态 | 出口记录 |
|---|---|---|
| PR-00A | **PASS**（GPU 锁 PENDING） | [PR-00A.md](PR-00A.md) |
| PR-00B | **PARTIAL**（锁/迁移/Compose 配置/前端 PASS；镜像与容器 BLOCKED） | [PR-00B.md](PR-00B.md) |
| PR-01 | **PASS** | [PR-01.md](PR-01.md) |
| PR-02 | **PASS（契约）**；PG 竞争 BLOCKED | [PR-02.md](PR-02.md) |
| PR-03 | **PARTIAL**（API/契约 PASS；真实浏览器未验） | [PR-03.md](PR-03.md) |
| PR-04 | **PARTIAL**（队列/幂等/取消/中断恢复 PASS；CPU 真实训练冒烟；GPU BLOCKED） | [PR-04.md](PR-04.md) |
| PR-05 | **PARTIAL**（注册/血缘/评估契约 PASS；真实指标 BLOCKED） | [PR-05.md](PR-05.md) |
| PR-06 | **PARTIAL**（推理持久化/导出 PASS；GPU 计时未测） | [PR-06.md](PR-06.md) |
| PR-07 | **PARTIAL**（BCCD 数据准备与 P0 形态链路 PASS；E1 正式实验 NOT_RUN） | [PR-07.md](PR-07.md) |
| PR-08 | **PARTIAL**（复核回流契约 PASS；E3 NOT_RUN） | [PR-08.md](PR-08.md) |
| PR-09 | **未完成** | [PR-09.md](PR-09.md) |

## 已实测的关键数字

- 后端新增测试：**68 passed**（`backend/tests/medical` + `test_app_factory_auth.py`）。
- 前端 Node 测试：**27 passed**；`npm run build` 退出 0。
- 核心依赖锁：43 个包，空环境 `--require-hashes` 安装通过。
- Alembic 空库迁移：21 张业务表，退出 0。
- Compose config（含 GPU override）：退出 0。
- BCCD：364 图 / 4888 框 / 291-37-36，`check_split_integrity` 364 文件 hash 校验通过。
- 真实 CPU 训练：1 epoch，模型 sha256 `9c98eee5...`，指标 P=0.713/R=0.311/mAP50=0.250。

## 未关闭的必交缺口

1. Docker 镜像构建与容器启动（守护进程未启动）。
2. PostgreSQL 并发/行锁验证与 GPU 真实训练/推理。
3. E1—E4 正式实验与固定 test 协议。
4. 旧入口/行业模块清退与前端页面全量迁移。
5. 空环境备份恢复演练与最终 release PR。

未满足上述项前，不得宣称项目完成或 P0 全绿。
