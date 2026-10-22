# 续作实施状态（持续更新）

执行目录：`D:\毕业设计\bysj-unblock`；分支：`graduation/unblock-20261022`。
起点：`113181f498bac6d16988f97261ad628c9cdc8ea8`。原 DS 目录保持不变。

用户停止条件：周限额剩余 <=20% 时先写状态书，再停止。起始剩余 72%；最近检查剩余 67%。尚未达到停止条件。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| U00 | PASS | 独立 worktree；补齐被 ignore 漏提交的 7 个现存源码；68 个原测试通过 |
| U01 | PARTIAL | 锁生成器 10 测试通过；候选 GPU 镜像构建及 pip check 通过；Linux core/test 锁正在洁净验证 |
| U02 | PARTIAL | PG 空 schema 迁移与重复升级通过；独立 Compose API/三 worker 启动通过；还需迁移失败负向门禁与 CI |
| U03 | PARTIAL | 原子租约、持久 generation、失联 GPU 隔离、独立 job/worker 心跳、attempt 写入保护已实现；9 个 PG 集成测试通过；真实取消/恢复待验 |
| U04 | CANDIDATE_PASS | 真实 GPU 训练/评估/推理/复核；真实 epoch 取消后 GPU 可再次推理；见 U04.md |
| U05 | IN_PROGRESS | 六页面已改接 v1，首次 build 通过；草稿异步保护、图像鉴权、真实标签导出、冻结物化已修；70 后端测试通过，浏览器 E2E 待执行 |
| U06 | NOT_RUN | 实验工具与 E1–E4 |
| U07 | NOT_RUN | 旧库合成演练、行为对照、旧入口清退 |
| U08 | NOT_RUN | 正式镜像、备份恢复、交付 |

本次新增资源：

- 候选镜像 `bysj-unblock-candidate:20261022`，仅本机测试使用。
- PG 容器 `bysj-test-pg-20261022`；新卷 `bysj-unblock-20261022-pg-tests`；本机端口 55432；库 `bysj_test_unblock`。
- 本地测试配置 `.runtime/pg.env`、`.runtime/pg-test.env`，均在 Git ignore 中，不提交密码。
- PostgreSQL 集成测试在 `tests/integration/`，刻意避开 `backend/tests/conftest.py` 的 SQLite 强制覆盖；只接受本地 bysj_test_ 库，每次新建独立随机 schema。

本文件为进行中记录，不能作为最终验收报告。
