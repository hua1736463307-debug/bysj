# 续作实施状态（持续更新）

执行目录：`D:\毕业设计\bysj-unblock`；分支：`graduation/unblock-20261022`。
起点：`113181f498bac6d16988f97261ad628c9cdc8ea8`。原 DS 目录保持不变。

接手步骤、精确命令和额度停线流程见 [CONTINUATION](CONTINUATION.md)。

用户停止条件：周限额剩余 <=20% 时先写状态书，再停止。起始剩余 72%；2026-10-22 09:48（America/Los_Angeles）最近检查剩余 33%。尚未达到停止条件。

当前正式运行源码：`bc3602e62e3b73b4aaeec4456c184dfe1bfc4a13`。运行项目 `bysj-restore2-20261022`，网页 http://127.0.0.1:18084，API http://127.0.0.1:18004。原候选 API/三 worker 已停，卷和容器保留。正式实验执行中，不得重启 trainer 或启动另一套 GPU worker。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| U00 | PASS | 独立 worktree；补齐被 ignore 漏提交的 7 个现存源码；68 个原测试通过 |
| U01 | LOCAL_PASS | 86 ML wheel 实物 SHA256 核验、无网络安装/pip check、干净 API/worker/Node22+Nginx 镜像、CUDA/NMS 与历史 CPU 权重加载通过；138 项洁净 Linux 回归 |
| U02 | PASS | PG 15 测试；空库/重复升级与独立 Compose 通过；迁移失败时 API/三 worker 均未启动；远端 CI 后端/PG/Compose/前端均通过 |
| U03 | CANDIDATE_PASS | PG 12 测试；真实 epoch 取消、worker SIGKILL、数据库断联、隔离、排队取消、精确恢复与 GPU 再用通过；见 U03.md |
| U04 | LOCAL_PASS | 候选故障注入通过；正式干净镜像实际训练/评估/推理/复核、完整 BCCD 短预试通过 |
| U05 | LOCAL_PASS | 六页面 v1，推理资产分页；最新正式 Nginx 五项浏览器验收全部通过，1.6 分钟，见 artifacts/browser/formal-final-20261022 |
| U06 | RUNNING | E1 三 seeds 完成；E4 性能完整测量通过；E3 12 图/259 预测待真实审核；E2 固定 train-only tiles 三种子执行中 |
| U07 | LOCAL_PASS | 13 项同输入对照、PG 合成迁移 3 项、163 个旧文件按引用清退；正式 E2E 通过；真实旧库未验证 |
| U08 | RESTORE_PASS_DELIVERY_PENDING | 22 表、1823 文件校验恢复成功，登录/历史/新 GPU 推理通过；正式实验后需补最终备份与交付状态 |

本次新增资源：

- 候选镜像 `bysj-unblock-candidate:20261022`，仅本机测试使用。
- PG 容器 `bysj-test-pg-20261022`；新卷 `bysj-unblock-20261022-pg-tests`；本机端口 55432；库 `bysj_test_unblock`。
- 本地测试配置 `.runtime/pg.env`、`.runtime/pg-test.env`，均在 Git ignore 中，不提交密码。
- PostgreSQL 集成测试在 `tests/integration/`，刻意避开 `backend/tests/conftest.py` 的 SQLite 强制覆盖；只接受本地 bysj_test_ 库，每次新建独立随机 schema。

正式实验：E1 `3c6b56d39497202c8e1babc9370f0993e2eb13632c29837987022d0cf763ebb1`；E2 `528ff6b86e2b07d2e7f4686bd210ff3085415cc89888dc729f628d6348baee38`；共同对比冻结 `c98d12602dc5c8512fc365f9450fc5a70d14d858d45154e5ed8084f851f37ff5`。记录位于忽略的 artifacts/formal-experiments，每 seed 独立状态，失败不得计零分。

本文件为进行中记录，不能作为最终验收报告。真实旧库副本、真实人工复核质量、学校/导师认可均不能由工程测试代替。

E1 已完成 68/77/78 epoch（预设早停）；三 seed test mAP50=0.9398±0.0150，mAP50–95=0.6518±0.0155，样本标准差。详见 docs/experiments/2026-10-22/RESULTS.md。E4 见同目录 PERFORMANCE.md。E2 当前 job 151bbfb6-a33e-48fc-830d-28a0c81518fb，运行时间明显更长；约 09:12 同机另一项目启动（无 GPU 请求），保留其运行状态，E2 耗时不作纯粹切片因果归因。

已推送独立分支，草稿 PR：https://github.com/hua1736463307-debug/bysj/pull/1 。远端提交 9acf611 的 medical-ci 39809041778 success：后端 114、PG 15、工具 36 项通过，前端测试/build 与 Compose 通过；远端 GPU job 按配置 skipped，本机另有实际 GPU 证据。后续新提交需看各自 CI。

09:48 后续证据：d6b54f1 的 medical-ci 39816096597 success，后端 114、PG 15、工具 41 项；前端和 Compose 均成功，GPU job skipped。新增恢复库 48 项标量引用只读检查全部通过；数据库没有声明 SQL 外键，不能把原先 FK 查询为空解释为全部关系完整。详见 U08 与 database-links-20261022.json。

E3 本地待审 ZIP：`D:\毕业设计\bysj-E3人工待审包-20261022.zip`，16 文件，SHA256 `8464c39815b98f35cee0f892212e20eacf546a06a68326588978d2e1c2efb765`，CRC 和逐项内容 hash 已核验；人工字段保持空白。
