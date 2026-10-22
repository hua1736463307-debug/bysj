# 后续执行入口与证据导航

本文件用于接手当前实施，最新是否停线、额度、正在执行的任务见 [STATUS](STATUS.md)。不能根据历史任务书里的旧路径重新覆盖本轮成果。

## 基线与运行位置

- 当前工作区：`D:\毕业设计\bysj-unblock`，分支 `graduation/unblock-20261022`。
- 原 DS 工作区 `D:\智能体\bysj` 保持原样；它不是本轮最新成果所在目录。
- 草稿 PR：[bysj #1](https://github.com/hua1736463307-debug/bysj/pull/1)。保持 Draft，未合并 main。
- 本机正式项目 `bysj-restore2-20261022`：网页 `http://127.0.0.1:18084`、API `http://127.0.0.1:18004`、PG 55437。
- 后端/worker 源码固定 `bc3602e62e3b73b4aaeec4456c184dfe1bfc4a13`；后续提交为前端、工具、证据和文档。正式实验中不重建/热替换后端。
- Compose 文件和私密 `.env` 在 `.runtime/restore2-20261022/`。用户名/密码从该文件读取，不写入报告、命令参数、Git 或聊天。
- 候选项目的 GPU 服务已经停止；不要同时启动另一套数据库控制的 GPU worker。同机其他项目由用户管理，不得替本任务停用。

## 已完成证据

| 项目 | 结果 | 本地证据 |
|---|---|---|
| 依赖与 GPU | 锁定 wheel、洁净安装、CUDA/NMS、历史权重加载通过 | U01.md；artifacts/formal-runtime/clean-gpu-probe.json |
| PG 与迁移 | 12 项并发/租约 + 3 项合成旧库迁移通过；迁移失败阻止业务启动 | U02.md；U07.md；artifacts/migration-negative.json |
| 真实故障 | 训练取消、worker 崩溃、数据库断联后隔离与恢复通过 | U03.md；artifacts/gpu-cancel、gpu-crash、database-disconnect |
| 正式前端 | 5 项浏览器验收通过，含 409 草稿保留、401/404、部分失败、训练取消、24 图分页闭环 | artifacts/browser/formal-final-20261022 |
| E1 | 42/43/44 三种子及独立 test 全部完成 | docs/experiments/2026-10-22/RESULTS.md；artifacts/formal-experiments/E1 |
| E4 | 导入/标注/训练提交/推理与资源采样完成 | docs/experiments/2026-10-22/PERFORMANCE.md；performance/ |
| 备份恢复 | 新卷恢复 22 表、1823 文件，登录/历史/新 GPU 推理通过 | U08.md；artifacts/backups/pre-formal-20261022；artifacts/formal-runtime/restore-acceptance.json |

浏览器 trace、认证配置、图片、权重和数据库仅留本机；Git 内为源码、脱敏汇总和图表。CI 中 GPU job 跳过不能当作 GPU 通过，本机 GPU 证据单列。

## E2 继续顺序

冻结协议 `528ff6b86e2b07d2e7f4686bd210ff3085415cc89888dc729f628d6348baee38`；原图父版本 `8f32546d-11ec-412d-a1c5-2c6aa41a1fa7`，切片版本 `a4d2a2b5-2465-58ff-8e20-6126a2e0d3ef`，train/val/test=1746/37/36。只切训练集，val/test 用原图整图评估。

1. 先查当前 runner 进程、任务 API 和 STATUS。已有活跃 runner 时只观察，不启动第二份。
2. 如果 runner 已停但任务仍 active，普通重跑会用相同幂等键重新接入。不要另建协议目录或改 seeds 以“跳过”已完成记录。
3. interrupted 先核对真实 owner 是否停止、GPU 是否隔离，再按 U03 精确恢复，显式调用该任务的 resume；不要直接改数据库状态。
4. cancelled 不支持 resume。用户明确继续后，使用 `--retry-cancelled` 为该 seed 从同一基础权重开始新 job；旧取消记录、日志和 checkpoint 保留。它不是从取消 checkpoint 接着训练。
5. failed 应先依据错误定位问题，不能当成取消自动重试；timeout 只表示 runner 等待超时，不能推断计算已停止。
6. 本工具的取消重试仅针对训练；若是 test evaluation 被取消，应先保留训练模型和评估记录，再实现并审查单独的幂等评估重试，不能重复训练或覆盖历史 test。

在已安全设置 `BYSJ_USER/BYSJ_PASSWORD` 的 PowerShell 中执行（不在脚本中硬编码凭据）：

```powershell
Set-Location 'D:\毕业设计\bysj-unblock'
$env:PYTHONUTF8 = '1'
$env:TEMP = (Resolve-Path '.runtime/tmp').Path
$env:TMP = $env:TEMP
& 'D:\智能体\bysj\.venv\Scripts\python.exe' scripts/run_experiments.py --config artifacts/formal-experiments/E2.json --output artifacts/formal-experiments/E2 --base-url http://127.0.0.1:18004 --timeout 14400
```

只有已确认 cancelled 且获继续执行指令时，才在末尾添加 `--retry-cancelled`。普通重跑读取已有新 key，不重复创建同一次重试。

参数保持 80 epoch 上限、batch=8、imgsz=640、AdamW、原 seeds/增强/早停。不能因为 E2 较慢而中途改 AMP、batch、workers、数据或优化器并继续称为同一冻结实验。E2 训练样本数为 E1 六倍，当前同机 CPU/内存共享，耗时须披露环境负载，不能声称等计算量对比。

每完成一个 seed 即核对独立 test evaluation、model/version/hash、代码/锁和实际参数，再生成汇总：

```powershell
& 'D:\智能体\bysj\.venv\Scripts\python.exe' scripts/summarize_experiments.py --e1 artifacts/formal-experiments/E1/3c6b56d39497202c8e1babc9370f0993e2eb13632c29837987022d0cf763ebb1 --e2 artifacts/formal-experiments/E2/528ff6b86e2b07d2e7f4686bd210ff3085415cc89888dc729f628d6348baee38 --output docs/experiments/2026-10-22
```

随后用 `scripts/plot_experiments.py --results <results.json> --output <figures>` 在已有 matplotlib 环境生成 PNG/SVG，目视检查。未完成种子不计零分；不足三种子保持 partial，不宣称显著性或临床效果。名义训练 batch 数是推导值，实际 optimizer step 数未仪表记录，仍为 null/reason。

## E3 和外部待补项

E3 见 [人工待审说明](E3-HUMAN-HANDOFF.md)；本地包 `artifacts/E3-human-review/` 已有 12 张叠图、来源标注、预测和空白人工字段。需要真实审核者填写身份、依据、时间及纠错，并走提交/审批流程。不得把来源标注、AI 判断或合成测试当成人工审批。

真实旧库副本尚未提供；当前合成迁移和本轮新库恢复分别通过，不能据此签收未知旧库。学校模板、导师认定和展示授权由用户确认。以上不妨碍完成已有授权的工程工作，但应在最终验收中保留边界。

## 达到额度停线或全部可执行工作完成时

1. 先停止本任务 runner，防止它自动提交下一 seed；再读取真实 job 状态。
2. 仅对本轮仍在途的具体训练/评估 job 经 API cancel；若已完成，直接收集结果，不重复取消。等待 worker 确认子进程停止、GPU holder 释放。
3. 保留原协议与 seed JSON，附加取消原因和任务终态；记录最后 epoch/耗时/日志路径。未完成模型不进入正式指标。
4. 无在途任务且无 GPU holder 后，用 `scripts/backup_project.py` 对 `bysj-restore2-20261022` 创建唯一名称的最终备份；核对 report complete 和 restart_failures 为空。不要覆盖 pre-formal 备份。
5. 写状态书：停止原因、额度读数/时间、提交/镜像、任务与数据 IDs、已完成结果、CI、备份位置和下一条执行命令。上传仅本任务源码/文档，保持草稿 PR，确认分支干净。
6. 停止新任务提交和继续迭代。服务是否保留运行应明确记录；不删除容器/卷，不干扰其他项目。
