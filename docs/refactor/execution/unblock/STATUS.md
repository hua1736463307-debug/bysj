# 续作实施状态（持续更新）

执行目录：`D:\毕业设计\bysj-unblock`；分支：`graduation/unblock-20261022`。
起点：`113181f498bac6d16988f97261ad628c9cdc8ea8`。原 DS 目录保持不变。

用户停止条件：周限额剩余 <=20% 时先写状态书，再停止。起始剩余 72%；2026-10-22 08:36（America/Los_Angeles）最近检查剩余 51%。尚未达到停止条件。

当前正式运行源码：`bc3602e62e3b73b4aaeec4456c184dfe1bfc4a13`。运行项目 `bysj-restore2-20261022`，网页 http://127.0.0.1:18084，API http://127.0.0.1:18004。原候选 API/三 worker 已停，卷和容器保留。正式实验执行中，不得重启 trainer 或启动另一套 GPU worker。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| U00 | PASS | 独立 worktree；补齐被 ignore 漏提交的 7 个现存源码；68 个原测试通过 |
| U01 | LOCAL_PASS | 86 ML wheel 实物 SHA256 核验、无网络安装/pip check、干净 API/worker/Node22+Nginx 镜像、CUDA/NMS 与历史 CPU 权重加载通过；138 项洁净 Linux 回归 |
| U02 | LOCAL_PASS | PG 15 测试；空库/重复升级与独立 Compose 通过；迁移失败时 API/三 worker 均未启动；CI 文件已更新，远端未运行 |
| U03 | CANDIDATE_PASS | PG 12 测试；真实 epoch 取消、worker SIGKILL、数据库断联、隔离、排队取消、精确恢复与 GPU 再用通过；见 U03.md |
| U04 | LOCAL_PASS | 候选故障注入通过；正式干净镜像实际训练/评估/推理/复核、完整 BCCD 短预试通过 |
| U05 | LOCAL_PASS | 六页面 v1；候选五项边界通过；正式 Nginx 前端 24 图三页/切片/训练/评估/推理/复核闭环 1 passed，2.8 分钟 |
| U06 | RUNNING | 完整 BCCD 与 train-only tiles 核验通过；显式 AdamW 预试通过；E1/E2 协议已冻结，E1 seed42 训练中；E3 人工证据待补；E4 脚本待实测 |
| U07 | LOCAL_PASS | 13 项同输入对照、PG 合成迁移 3 项、163 个旧文件按引用清退；正式 E2E 通过；真实旧库未验证 |
| U08 | RESTORE_PASS_DELIVERY_PENDING | 22 表、1823 文件校验恢复成功，登录/历史/新 GPU 推理通过；正式实验后需补最终备份与交付状态 |

本次新增资源：

- 候选镜像 `bysj-unblock-candidate:20261022`，仅本机测试使用。
- PG 容器 `bysj-test-pg-20261022`；新卷 `bysj-unblock-20261022-pg-tests`；本机端口 55432；库 `bysj_test_unblock`。
- 本地测试配置 `.runtime/pg.env`、`.runtime/pg-test.env`，均在 Git ignore 中，不提交密码。
- PostgreSQL 集成测试在 `tests/integration/`，刻意避开 `backend/tests/conftest.py` 的 SQLite 强制覆盖；只接受本地 bysj_test_ 库，每次新建独立随机 schema。

正式实验：E1 `3c6b56d39497202c8e1babc9370f0993e2eb13632c29837987022d0cf763ebb1`；E2 `528ff6b86e2b07d2e7f4686bd210ff3085415cc89888dc729f628d6348baee38`；共同对比冻结 `c98d12602dc5c8512fc365f9450fc5a70d14d858d45154e5ed8084f851f37ff5`。记录位于忽略的 artifacts/formal-experiments，每 seed 独立状态，失败不得计零分。

本文件为进行中记录，不能作为最终验收报告。真实旧库副本、真实人工复核质量、学校/导师认可均不能由工程测试代替。
