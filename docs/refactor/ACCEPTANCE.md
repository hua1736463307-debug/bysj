# 验证计划、命令与最终验收

本文列的是执行者要完成的验证，不是本次已通过结果。状态必须分为 PASS / FAIL / BLOCKED / NOT_RUN，跳过 GPU 或真实数据不算通过。历史验证单独列引用，不计入本次执行通过数。

## 1. 证据文件约定

每个 PR 建 `docs/refactor/execution/PR-XX.md`，大日志/截图/指标放受控 artifacts 并在记录中保存 hash；只提交不含敏感信息的小证据/索引。记录命令、cwd、shell、时间、环境、退出码、原始输出引用、断言、失败分类。

失败分类：本次回归 / 既有问题 / 环境阻塞 / 方案假设失效。修复后只重跑受影响检查；最终整体回归在 PR-09 执行。不要为文档变化重跑 GPU 训练。

## 2. 当前已存在、可直接定位的命令

以下命令语法和目标路径已确认，应用相关命令在**隔离环境**准备好后运行；本次未安装应用依赖，因此没有声称它们通过。

| ID | 工作目录/shell | 命令 | 前置 | 通过信号/能证明的范围 |
|---|---|---|---|---|
| V00 | 仓库根/Python+Git，Bash或PowerShell | `python scripts/verify_refactor_baseline.py` | 本交付脚本、初始源码；Git或保留原字节快照 | 全部 protected 文件与固定 blob 一致；只证明初始源码未漂移 |
| V01 | frontend/Bash | `npm ci` | PR-00A 的 Node/npm；网络/缓存可用 | 退出0；lock 未被重写 |
| V02 | frontend/Bash | `npm run build` | V01 | 退出0、dist 产物；不证明 API 可用 |
| V03 | frontend/Bash | `node --test tests/*.test.mjs src/utils/*.test.mjs` | V01 | 既有 Node tests 退出0；真实浏览器仍需 V20 |
| V04 | 仓库根/隔离后端 Bash | `PYTHONPATH=backend python -m pytest backend/tests/test_annotation_draft_completeness.py backend/tests/test_annotation_session_isolation.py backend/tests/test_annotation_geometry_filter.py backend/tests/test_image_size_processing.py -q` | 已锁测试环境、临时数据根、外网断开 | 核心草稿/几何回归；SQLite 不代表 PG锁 |
| V05 | 仓库根/隔离后端 Bash | `PYTHONPATH=backend python -m pytest backend/tests -q` | PR-00B 已准备完整隔离环境 | 既有测试全量退出0；无法收集亦算失败/阻塞，不能忽略 |

PowerShell 下执行 V03，可用 `node --test (Get-ChildItem tests/*.test.mjs,src/utils/*.test.mjs | ForEach-Object FullName)`；后端环境变量 PowerShell 为 `$env:PYTHONPATH='backend'`，但原模块有固定 /app 路径，优先在容器/WSL2 隔离环境运行，不能通过创建真实宿主业务目录绕过失败。

## 3. 待创建命令的交付契约

下表文件/脚本在当前基线大多不存在，必须先由所属任务卡实现。执行前检查 `--help`、错误退出码及真实断言；命令不得只打印 PASS。

| ID | 创建阶段 | 根目录/Bash 的目标命令 | 必须验证 |
|---|---|---|---|
| V06 | 00B | `python -m pip install --require-hashes -r backend/requirements.lock.txt` | 空环境安装；GPU lock 另执行一次并记录 |
| V07 | 00B | `docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml config --quiet` | 变量/网络/卷/端口合法；deploy/.env 本地创建且忽略 |
| V08 | 00B | `docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml build --no-cache` | 前后端洁净构建与版本元数据 |
| V09 | 00B/01 | `docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml up -d db api frontend` | 空库迁移、live、ready、登录与前端路径（01补完） |
| V10 | 00B | `PYTHONPATH=backend python -m alembic -c backend/alembic.ini upgrade head` | 独立空 PG 和升级副本分别通过；环境从安全文件注入 |
| V11 | 02 | `PYTHONPATH=backend python -m pytest backend/tests/medical/test_dataset_versions.py backend/tests/medical/test_split_integrity.py backend/tests/medical/test_upload_safety.py -q` | 版本/来源/几何/上传恶意边界 |
| V12 | 03 | `PYTHONPATH=backend python -m pytest backend/tests/medical/test_annotation_revision.py -q` | 同版并发保存409、分页草稿保留、事务一致 |
| V13 | 04 | `PYTHONPATH=backend python -m pytest backend/tests/integration/test_training_lifecycle.py backend/tests/integration/test_job_concurrency.py -q` | 真实 PG、多连接/worker、取消/中断；fixture 创建测试 schema |
| V14 | 05 | `PYTHONPATH=backend python -m pytest backend/tests/medical/test_model_lineage.py backend/tests/medical/test_evaluation_contract.py -q` | 模型来源/可信加载/指标 schema/越权 |
| V15 | 06 | `PYTHONPATH=backend python -m pytest backend/tests/integration/test_inference_lifecycle.py -q` | 持久化、部分失败、幂等、GPU互斥（真实 GPU 项单标） |
| V16 | 07 | `python scripts/import_bccd.py --source data/raw/bccd --output data/prepared/bccd --config configs/experiments/bccd-baseline.json` | XML转换、文件hash、类映射、拒绝坏框；无网络隐式下载 |
| V17 | 02/07 | `python scripts/check_split_integrity.py --manifest data/prepared/bccd/manifest.json` | source/root/hash 无交集、派生继承；错误退出非0 |
| V18 | 08 | `PYTHONPATH=backend python -m pytest backend/tests/medical/test_feedback_roundtrip.py -q` | test/派生/改名重复回流拒绝、审批/幂等/原预测不变 |
| V19 | 07/08 | `python scripts/smoke_medical.py --base-url http://127.0.0.1:8000 --config configs/experiments/bccd-smoke.json --evidence artifacts/smoke` | 真 API 数据→标注→训练→模型→推理；账号从env安全读取 |
| V20 | 03/09 | `npm --prefix frontend run test:e2e` | 本阶段新增 Playwright 与脚本；真实浏览器路由和编辑流程 |
| V21 | 07/09 | `python scripts/run_experiments.py --config configs/experiments/bccd-baseline.json --output artifacts/experiments` | 任务提交/状态等候/真实结果导出；不自行手填指标 |
| V22 | 09 | `python scripts/benchmark_medical.py --config configs/experiments/performance.json --output artifacts/performance` | 100图导入/推理、实际计时/资源/失败率 |
| V23 | 09 | `PYTHONPATH=backend python -m pytest backend/tests/medical backend/tests/integration -q` | 完整医学回归，真实 PG；GPU skip 单独报告 |

worker 启动命令由 PR-00B 新增 profile `workers` 固定为 `docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml --profile workers up -d`；GPU 环境额外 `-f deploy/compose.gpu.yml`，并确认可见设备。不向环境缺 GPU 的机器传入 GPU override。API URL/前端端口需与 deploy/.env 保持一致；上述 8000 是毕业版设计默认。

测试账户、JWT、数据库口令只从安全环境读取，不能作为命令参数打印到日志。测试数据库连接串必须指向当前 Compose db 或明确独立测试库；测试 fixture 验证数据库名/项目标识，禁止使用任何生产 DSN。

## 4. 行为验收矩阵

| 用例ID | 场景与操作 | 通过标准 | 阶段/证据 |
|---|---|---|---|
| A01 | 全新安装、空库启动、刷新前端子路由 | 迁移成功、无默认弱口令、可登录、静态与API代理正确 | 00/01；构建和HTTP记录 |
| A02 | 未登录、token过期、跨用户dataset/job/model/result/artifact | 401或统一404/403；无敏感路径；无请求绕过 | 01—09；权限矩阵 |
| A03 | 导入正常/空标签/损坏图/未知类/越界框 | 成功与失败可定位；空标签按配置有效，不把坏标注当空标签 | 02；导入报告 |
| A04 | ZIP ../、绝对路径、盘符、symlink、重复路径、压缩炸弹、超像素 | 明确4xx；正式目录/外部目录未写入；内存有界 | 02；哨兵文件与目录快照 |
| A05 | 同内容不同文件名、同患者/slide、切片/增强跨集合 | 冻结拒绝或归同组；三个层级交集均空 | 02/07；manifest检查 |
| A06 | 冻结后编辑/删除；被模型引用删除 | 冻结内容hash不变；生成子版；引用删除409 | 02；前后hash |
| A07 | 画框缩放/平移、换页、撤销/重做、删除/恢复、导出/回读 | 原图坐标一致，类别ID一致，未加载页不丢 | 03；UI截图与标签比较 |
| A08 | 两标签页同revision保存，慢请求后切换dataset | 旧版409、草稿未覆盖；迟到响应不污染新会话 | 03；并发测试/UI |
| A09 | 相同幂等键重复提交训练、两worker争抢 | 一任务、一执行者、事件序号不重复 | 04；PG事务与PID/lease证据 |
| A10 | pending取消、running取消、训练完成与取消竞争 | 状态合法；确认进程结束才cancelled；终态不回退 | 04；进程/状态记录 |
| A11 | API重启、worker重启、强制中断本测试子进程 | 不重复训练，无幽灵running；resume策略明确 | 04；前后任务记录 |
| A12 | CPU、无GPU、GPU OOM、坏模型、错误geometry | 队列/明确失败；错误可定位、不崩API | 04；分别记录，不替代真实GPU |
| A13 | 模型注册/缺失/篡改/非信任上传/跨用户下载 | hash/血缘真实；不可信权重不load；权限正确 | 05；加载spy及真实产物 |
| A14 | 独立val/test评估、类别不一致、指标缺失 | split准确；错误422；缺指标null+原因 | 05；evaluation记录 |
| A15 | 单图、100张批量、损坏一张、取消、重启、刷新 | 成功项不丢；分项错误正确；结果可持久读回 | 06；数据库/文件/API |
| A16 | 训练与推理同时排队、模型切换 | 单GPU租约不重叠；释放确认；计时分离 | 06；租约与显存记录 |
| A17 | 预测图、JSON、CSV回读 | 原图坐标和类一致、含hash/参数、CSV无公式执行 | 06；导出校验 |
| A18 | 五类review，含无prediction漏检 | 审阅完整、原预测不变、审批留痕 | 08；review状态链 |
| A19 | test原图/派生/改名/同组反馈回流 | 拒绝、test/val manifest逐项不变 | 08；泄漏反例 |
| A20 | 反馈导出重复、审批冲突、中途失败 | 幂等一个子版、409、不出现可训练半版本 | 08；事务证据 |
| A21 | BCCD完整正式训练与反馈再训练 | 两组模型及完整实验记录、固定独立test对比 | 07/08；真实指标/配置 |
| A22 | 最终启动图、路由、依赖与UI检查 | 无监控/告警/人脸/录机/姿态入口或启动任务 | 09；OpenAPI/导入图/UI |
| A23 | 100图导入推理和标注保存性能 | 有硬件/规模/并发/p50/p95/失败率/峰值资源，无异常退出 | 09；原始计时与报告 |
| A24 | 空环境备份恢复演练 | 同版本数据/模型引用与hash一致，恢复后闭环成功 | 09；备份和演练记录 |

## 5. 实验验收

E1：固定版本正式训练；E2：原图与通用切片/增强；E3：反馈前后；E4：系统性能。配置在运行前提交，报告所有失败/无收益结果。优先 3 seeds 的均值和标准差；仅一 seed 必须披露。不可用测试集反复筛选最佳结果，不采用“mAP必须提高”作为工程完成门槛。

实验数据卡记录实际样本/类别分布、来源/许可、来源组可得性、三集hash、近重复检查和局限。原图test与tile test不同统计单位时不可直接比较。曲线图与统计表由原始 metrics 文件生成，保存作图脚本与输入hash。

## 6. 最终 Definition of Done

- P0 数据→标注→冻结→训练→模型→评估→推理→导出完整实测。
- 人工复核→新训练版本→再训练→固定test比较完整实测。
- 洁净依赖、Compose、迁移、独立PostgreSQL、GPU训练/推理和恢复分别有证据。
- 权限、上传安全、模型信任、来源组/测试集隔离、标注冲突无未关闭必交失败。
- 行业功能已从最终运行图退出，不只是菜单隐藏；旧代码删改有清单与回退记录。
- E1—E4、代码/数据/模型/依赖hash与真实统计可复现；不伪造精度/延迟。
- README/架构/ER/API/运行/备份恢复/实验/论文提纲/演示和贡献说明对应同一提交。
- 项目所有者对代码展示范围、学校成果认定与公开数据使用记录已确认；AI 不代签。

未满足任一必交项时，报告“部分完成/阻塞”，列明已有证据和缺口；不以任务卡打勾数量代替完成标准。
