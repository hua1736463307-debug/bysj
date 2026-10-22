# 项目现状（一页版）

> **当前状态入口**：[停止状态书](docs/refactor/execution/unblock/STATEBOOK-20261022.md)。已按用户指令停止实施；工程主链路已有证据，旧 E1/E2 因跨集近重复风险停止续跑。分组修复已部署，分组包尚未真实导入，最终最新备份待补。下文保留原 DS 交接历史。

> 更新时间：2026-10-22 01:35（America/Los_Angeles）
> 分支：`graduation/pr-00-baseline`　基线：`c47950b`
> 详细证据与逐阶段记录：[docs/refactor/execution/STATUS.md](docs/refactor/execution/STATUS.md)

## 结论

**新入口可运行、核心链路有真实证据；但容器、PostgreSQL、GPU、正式实验与旧代码清退未完成，不能宣称项目完成。**

## 已完成

- 新后端 `backend/app`：应用工厂、v1 契约、JWT 鉴权、七态任务账本、数据版本、标注 CAS、训练队列、模型注册、独立评估、推理持久化、人工复核回流。
- 部署基线：43 个包哈希锁、Alembic 空库 21 表、Compose（含 GPU override）配置、nginx 同源代理、CI 工作流。
- 数据与训练：BCCD 364 图 / 4,888 框 / 291-37-36；真实 HTTP 全链路冒烟通过；真实 CPU 1 epoch 训练产出模型（指标不具统计意义）。
- 文档：架构、数据库、API、运行手册、数据卡、论文提纲、演示脚本、PR-00A—PR-09 执行记录。

## 未完成（阻塞）

| 项目 | 状态 |
|---|---|
| Docker 镜像构建 / 容器启动 | BLOCKED（守护进程未启动） |
| PostgreSQL 迁移与并发验证 | BLOCKED（本机无 PG） |
| GPU 训练 / 推理 / ML 依赖锁 | BLOCKED（仅 CPU torch） |
| E1–E4 正式实验 | NOT_RUN |
| 旧入口与行业模块清退 | NOT_DONE |
| 剩余 6 个前端页面迁移 | NOT_DONE |
| 旧库副本迁移演练 / 备份恢复演练 | NOT_RUN |

## 关键数字

| 指标 | 值 |
|---|---|
| 后端测试 | 68 passed |
| 前端测试 / 构建 | 27 passed / exit 0 |
| 依赖锁 | 43 包（sha256） |
| Alembic 空库 | 21 张表 |
| BCCD | 364 图 / 4,888 框 |
| 真实 CPU 训练 | 1 epoch，模型 sha `9c98eee5…`，P=0.713 / R=0.311 / mAP50=0.250 |

## 新旧代码关系

- 新 `backend/app` **不导入**任何旧模块；旧核心文件相对基线**零改动**。
- 数据层面沿用 9 张旧表名并新增 12 张表，属于**平行重写 + 共享 schema**，不是增量改造。
- 前端 7 个页面中只有 Login 走新客户端，其余仍调用旧端点。

## 下一步

1. 启动 Docker → 构建并 `compose up` → 独立 PG 跑迁移。
2. 生成 GPU ML 锁 → 真实 CUDA 训练/推理。
3. 冻结评估协议 → E1 → E3 → E2/E4。
4. 迁移剩余页面 → 删除旧入口/行业模块 → 旧库迁移演练 → release PR。
