# 续作实施状态（持续更新）

执行目录：`D:\毕业设计\bysj-unblock`；分支：`graduation/unblock-20261022`。
起点：`113181f498bac6d16988f97261ad628c9cdc8ea8`。原 DS 目录保持不变。

用户停止条件：周限额剩余 <=20% 时先写状态书，再停止。起始剩余 72%；最近检查剩余 58%。尚未达到停止条件。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| U00 | PASS | 独立 worktree；补齐被 ignore 漏提交的 7 个现存源码；68 个原测试通过 |
| U01 | PARTIAL | Linux core/test 锁在无网络空 venv 下安装与 pip check 通过；87 测试通过；干净 core-deps 镜像已建。ML 完整 86 包已解析，官方 wheel 校验下载中；正式镜像待构建 |
| U02 | LOCAL_PASS | PG 15 测试；空库/重复升级与独立 Compose 通过；迁移失败时 API/三 worker 均未启动；CI 文件已更新，远端未运行 |
| U03 | CANDIDATE_PASS | PG 12 测试；真实 epoch 取消、worker SIGKILL、数据库断联、隔离、排队取消、精确恢复与 GPU 再用通过；见 U03.md |
| U04 | CANDIDATE_PASS | 真实 GPU 训练/评估/推理/复核；真实 epoch 取消后 GPU 可再次推理；见 U04.md |
| U05 | CANDIDATE_PASS | 六页面 v1、24 图三页闭环及四项失败/取消用例，最终组合 5 passed；清退后重跑进行中；正式前端镜像待构建 |
| U06 | TOOL_PASS | 109 项后端/工具回归；真实 seed=0 训练、独立 test、幂等重跑、性能工具通过；正式 E1–E4 尚未运行；见 U06.md |
| U07 | LOCAL_PASS_PENDING_E2E | 13 项同输入对照、PG 合成迁移 3 项、163 个旧文件按引用清退；125 后端/工具回归及 PG 15 项通过；真实旧库未验证 |
| U08 | NOT_RUN | 正式镜像、备份恢复、交付 |

本次新增资源：

- 候选镜像 `bysj-unblock-candidate:20261022`，仅本机测试使用。
- PG 容器 `bysj-test-pg-20261022`；新卷 `bysj-unblock-20261022-pg-tests`；本机端口 55432；库 `bysj_test_unblock`。
- 本地测试配置 `.runtime/pg.env`、`.runtime/pg-test.env`，均在 Git ignore 中，不提交密码。
- PostgreSQL 集成测试在 `tests/integration/`，刻意避开 `backend/tests/conftest.py` 的 SQLite 强制覆盖；只接受本地 bysj_test_ 库，每次新建独立随机 schema。

本文件为进行中记录，不能作为最终验收报告。
