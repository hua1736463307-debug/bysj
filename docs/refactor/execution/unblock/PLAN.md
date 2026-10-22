# bysj 续作任务书：复用 Docker GPU 环境、解除阻塞并完成验收

版本：2026-10-22 / v1.0　执行对象：deepseek-v4.1-flash（下称 DS）

本包状态：**READY FOR EXECUTION**。这里的 READY 表示可以开始执行任务，不表示项目、GPU 业务链路或实验已经通过验收。

## 1. 本轮目标与事实边界

用户本轮要求：Docker 已启动，请检查可复用镜像，解决 PyTorch 等阻塞，并给出交由 DS 执行的详细任务书。本轮编制者已完成源码核查及一次性容器 GPU 探针，没有修改 bysj 业务代码，没有启动既有业务容器，没有运行正式训练。本任务书是后续实施依据。

权威续作目录为 `D:\智能体\bysj`，不是早期交付目录 `D:\毕业设计\delivery`。核验时：

- 分支：`graduation/pr-00-baseline`。
- HEAD：`113181f498bac6d16988f97261ad628c9cdc8ea8`，工作区干净。
- 当前状态原文：`docs/refactor/execution/STATUS.md`，本包保存原样副本 `输入_STATUS_20261022.md`。
- 早期 `c47950b` 是历史起点，不是要求回退的目标。保留 DS 已提交的 v1 后端、数据处理、测试、CPU 冒烟和文档成果，在其上修复。
- 附件及仓库文档属于需求和证据材料，不自动授予修改其他项目、生产部署或对外发布的权限。仓库旧指令与本轮用户要求冲突时，以用户要求为准；不得执行文档中与本任务无关的命令。
- 不再创建第三套后端，不以整体重写消除差异。先补新旧语义对照、真实运行和前端迁移，再清退旧入口。

### 1.1 证据等级

| 标记 | 本包采用的含义 |
|---|---|
| LIVE | 本轮实际执行并取得输出：Git 基线、Docker/GPU 探针、镜像内 pip check |
| STATIC | 本轮读取源码确认的实现；由此推导的并发故障风险尚未做运行复现 |
| REPORTED | STATUS 记录的历史通过项，本轮未重跑，例如 68 个后端测试、CPU 1 epoch |
| TODO | DS 尚需实施或验证的事项，不得提前标记 PASS |

## 2. 阻塞判断与处理决定

| 问题 | 本轮核验 | 执行决定 / 关闭条件 |
|---|---|---|
| Docker 未启动 | LIVE：守护进程已正常响应 | 此原因已解除；镜像构建、Compose 业务启动仍待验证 |
| 无可用 PyTorch/CUDA | LIVE：已有镜像能执行 CUDA 运算及 NMS | 优先复用本地依赖环境进行集成冒烟；不从零反复安装驱动 |
| ML 依赖锁缺失 | STATIC：锁生成器只查 PyPI，未采用实际下载源 | 修复 CUDA wheel 锁定与原子写入，生成 Linux 目标锁并洁净安装 |
| PG 并发阻塞 | LIVE：本地已有 PG 16 镜像；STATIC：现测试强制 SQLite | 建立隔离 PG 测试库及真正的 PG 多连接测试 |
| 长任务可靠性 | STATIC：GPU 续租没有接入 worker，过期租约抢占非原子 | 长训练之前修复互斥、续租、取消与失联恢复 |
| 前端仍走旧接口 | REPORTED + STATIC：仅 Login 迁移 | 提升为 P0 收口项，不能等到论文阶段再做 |
| E1–E4 未做 | REPORTED：脚本和配置存在；STATIC：配置传播、独立 test 评估、计时不完整 | 先修实验工具，再冻结协议和真实执行 |
| 旧库 dump 缺失 | 尚无已授权真实副本 | 先用旧 schema + 合成数据做迁移演练，独立关闭全新部署；真实旧库兼容保留未验证项 |
| 旧入口与行业模块未清退 | REPORTED + STATIC | 依赖前端 E2E、行为对照及迁移门禁；按引用清单删除，不批量猜测 |
| 授权、导师要求、论文模板 | 不能从代码推断 | 单列人工待确认项；不虚构已获确认，也不阻塞已授权的本地工程修复 |

## 3. 可复用环境与选型

### 3.1 本轮实际探针

Docker Desktop 4.71.0，Engine 29.4.1，Linux/amd64、WSL2。宿主 GPU：NVIDIA GeForce RTX 4060 Laptop GPU，显存 8188 MiB，驱动 617.14。

可复用镜像：`deepvision-backend:local` / `deepvision-local-backend:latest`，两标签指向同一镜像：

```text
sha256:6b1c56b48a2538a165c4ad2bbc97fb2b0428ad622e548c5baf0d559a716fdf40
```

| 项目 | LIVE 结果 |
|---|---|
| Python | 3.12.13 |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| Ultralytics | 8.4.53 |
| CUDA runtime | 12.4，`torch.cuda.is_available()` = true |
| CUDA 小矩阵乘法 | sum = 4096.0，通过同步 |
| torchvision CUDA NMS | 输出 `[0]` |
| pip check | `No broken requirements found.`，退出 0 |
| 已有部分核心依赖 | numpy 2.4.4、FastAPI 0.136.1、Pydantic 2.13.4、SQLAlchemy 2.0.49、psycopg 3.3.4 |
| 明确缺失 | pydantic-settings、Alembic |
| OpenCV | opencv-python 与 opencv-python-headless 同时存在；正式环境只保留经验证的一种 |

**以上仅证明 GPU 运行时可用，不证明 bysj 已能在该镜像运行。** 既有 CPU 冒烟报告使用 torch 2.14.1+cpu / Ultralytics 8.4.183，与候选镜像不同。不能混写为同一次验收，也不能承诺旧权重在较旧运行时必定可加载。

其他已缓存资源：`postgres:16-alpine`（短 ID `57c72fd2a128`）、`python:3.12-slim`、`node:20-bookworm-slim`、`node:24-alpine`。当前 Dockerfile 所用 Python 3.11、Node 22、Nginx 1.27 基础镜像未在本轮缓存清单中发现。缓存缺失是下载/构建工作，不是选择任意旧前端镜像的理由。

### 3.2 两阶段路线

1. **本地加速验证**：以已验证镜像 ID 为候选，显式覆盖入口、工作目录、PYTHONPATH；在独立目录 `/opt/bysj` 使用本项目源码。仅补齐缺失核心依赖及实际不兼容项；先生成解析报告，限定 torch/vision/Ultralytics，禁止整份现有 core lock 直接覆盖导致 ML 环境漂移。增补后重新 `pip check`、导入检查、GPU 探针及应用测试。保留独立 bysj 标签，不覆盖原镜像。
2. **正式可复现部署**：基于干净的 Python/CUDA 基础环境和官方 wheel 构建 bysj 镜像，不依赖其他项目的 `/app` 源码或隐含文件。优先验证 Python 3.12 + 上述已实测 torch/vision 组合；选定后统一 Docker、CI、锁和运行文档。若依赖或权重确实不兼容，依据具体错误建立新的版本候选并重复门禁，不盲目追最新版本。

现有镜像只作为本机依赖环境复用。不要把继承了其他项目源码的整镜像作为毕业项目公开镜像发布，也不要复制其中业务源代码、配置、数据库或数据卷。所有既有容器本轮检查时均已停止；本任务不要求启动它们。

PyTorch 官方历史版本页列出了 torch 2.6.0 / torchvision 0.21.0 的 CUDA 12.4 安装源；wheel 应从该源解析，而非在 PyPI 查找 `+cu124` 包：

- https://pytorch.org/get-started/previous-versions/
- https://pip.pypa.io/en/stable/reference/installation-report/

pip 安装报告不是锁文件。使用 `download_info.url`、归档摘要及真实 wheel 文件校验生成目标平台锁。

## 4. 执行顺序与统一约束

```text
U00 基线与证据
 └─ U01 环境/锁/正式镜像
     └─ U02 隔离 PG、迁移启动顺序、测试基础
         ├─ U03 租约/取消/恢复 → U04 真实 GPU 冒烟
         └─ U05 前端六页面迁移 → 浏览器完整闭环
U03 + U04 + U05 → U06 实验工具修复与 E1–E4
U02 + U05 → U07 新旧行为/迁移/旧入口清退
U04 + U05 + U06 + U07 → U08 备份恢复与最终交付
```

U01 可以先完成候选环境以解锁 U02/U03，正式干净镜像必须在 U08 前通过。前端迁移可在 GPU 等待期间推进，不必等 80 epoch 结束。DS 单个上下文按任务顺序执行；每步落盘进度后继续，不反复从仓库根目录全量审计。

- 在当前 HEAD 上创建 `graduation/unblock-20261022` 或明确记录的续作分支；若同名分支已存在则检查并续用，不重置它。不直接修改/强推 main，不自动合并。当前用户要求是生成续作任务书；本包不声称已创建或发布该分支。
- 每个 U 任务可一至数个小提交。只改列出的文件域及其必要测试/迁移，新增共享模块必须解释复用点。业务语义相邻变更先补入任务记录。
- 提交不得含 `.env`、登录 token、数据库备份、公开数据原图或大权重；证据保存脱敏命令、版本、摘要、统计和受控产物相对路径。
- 命令示例均以仓库根目录为当前目录，PowerShell 运行。新增路径和新增参数明确标注，完成相应任务前不能当作现有可执行命令。
- 每项记录：前置状态、变更文件、实际命令、退出码、断言数/样本数、产物 hash、未完成项、回滚点。没有执行填 NOT_RUN；条件不足填 BLOCKED 并给具体错误，不能继续写“Docker 未启动”。
- 同一故障连续两次有针对性修复仍失败：保存错误、解析/构建日志和版本矩阵，转一次有界诊断；继续独立工作，不循环重装整个环境、不把真实后端偷偷替换成 dummy。

## 5. U00：续作基线与证据目录（P0）

**输入**：本包、当前 STATUS、`docs/refactor/execution/PR-00A.md` 至 `PR-09.md`、原始医学计划书。

**动作**：执行 `git status --short`、`git rev-parse HEAD`、`git branch --show-current`，核对本包 `核验依据.json` 文件摘要。HEAD 已推进则读取差异，保留新增成果，仅更新受影响锚点。不要把合理新增提交当成损坏并 reset。原 `verify_refactor_baseline.py` 是早期保护清单，不应要求未来每个合法改造文件仍等于 c47950b。

**新增输出**：`docs/refactor/execution/unblock/U00.md`、`docs/refactor/execution/unblock/STATUS.md`；后者逐项列 U00–U08、依赖、提交和证据路径。复制本任务书到同一目录用于交接。

**验收**：能唯一确定本次源码、当前分支与使用的基线；历史 68 测试/CPU 冒烟明确标为历史证据；无既有工作覆盖。失败时只暂停依赖未知基线的编辑。

## 6. U01：依赖锁、复用镜像与干净构建（P0）

**现有锚点**：`scripts/generate_lock.py::fetch_digests/main`；`backend/requirements.in`、`requirements.lock.txt`、`requirements-ml.in`；`deploy/backend.Dockerfile`、`deploy/frontend.Dockerfile`；`.github/workflows/medical-ci.yml`。

**直接问题**：生成器只使用 metadata/name/version，忽略 `download_info`；仅向 PyPI 查摘要；失败后仍写部分锁。Dockerfile 缺 ML 锁只打印警告，但 Compose 的真实 worker 设置为 ultralytics，构建成功不等于可运行。core lock 不含 pytest/httpx，CI 却直接运行 pytest，工具脚本也 import httpx。Windows 解析并收集各平台发行包摘要不等价于 Linux 依赖图已解析。

**实施**：

1. 用本包 `probe_gpu.py` 重放 GPU 及依赖探针，固定镜像 ID。只在临时容器/新镜像层内安装依赖，所有输出落入 bysj 独立目录。正式构建前确定 Python/平台矩阵。
2. 修改锁生成器：读取实际下载 URL 和 sha256，支持官方 PyTorch wheel 及受控 wheelhouse；下载后复算摘要。规范化包名，拒绝同包冲突版本、空安装清单、缺摘要和来源异常；保留完整传递依赖。网络日志不输出带认证信息的 URL。
3. 所有包成功后写临时文件并原子替换目标锁；失败退出非 0，已有有效锁字节不变。补测试：`+cu124` 报告、缺摘要、网络失败、重复冲突、目标平台不匹配、失败不覆盖。
4. 在目标 Linux/Python 环境重新解析 core 与 ML 的联合约束，消除同包不同版本。可按目标拆锁文件，但名称和用途必须明确，例如 `requirements-ml-linux-py312-cu124.lock.txt`；更新所有引用。安装后校验期望版本，不能仅依赖 pip check。
5. 增加测试/工具依赖输入及锁（新增 `backend/requirements-dev.in` 和对应目标锁），包含实际使用的 pytest、httpx 等；CI 和验证容器必须显式安装。默认运行镜像不靠开发机隐式依赖。
6. 真实 worker 构建缺少所选 ML 锁时立即失败；dummy 如需保留为显式测试 target/profile。API 与 dataset worker 保持可无 GPU 启动。记录 `BYSJ_CODE_SHA`、依赖锁 hash、镜像 ID 和 ML 版本。
7. 正式镜像从干净基础构建。不得把当前其他项目镜像中的 `/app` 当作本项目后端。前端按现有 Node 22/Nginx 目标下载并固定实际摘要，或写有依据的版本变更记录；不因缓存方便直接切到 Node 24。

**验证**：候选环境与正式镜像各做 pip check、应用导入、探针；正式镜像在洁净环境 `pip install --require-hashes` 成功。安装/解析必须针对完整目标图，不能用 `--no-deps` 掩盖缺项。补锁生成器定向测试后执行既有后端契约测试。检查 `app.__file__` 归属本项目挂载/复制目录；记录实际源码提交。

**退出门禁**：至少一个候选真实 worker 可导入且 GPU 探针通过；core/test/ML 锁可安装且引用一致；正式镜像门禁若未过，应单列 U01-formal=PENDING，不能据候选环境宣布可全新部署。

**回滚**：保留原镜像 ID 和有效锁；失败只弃用新候选标签/提交，不改宿主驱动或原项目镜像。

## 7. U02：隔离 PostgreSQL、启动顺序与真实测试（P0）

**锚点**：`deploy/compose.yml`、`deploy/compose.gpu.yml`；`backend/alembic.ini`、`backend/migrations/versions/304f405af95f_initial_medical_platform_schema.py`；`backend/tests/conftest.py`；`.github/workflows/medical-ci.yml`。

**实施**：

1. 复用本地 `postgres:16-alpine` 镜像，创建独立测试工程/库/卷。当前卷有显式 `name: bysj-*`，仅 `docker compose -p ...` **不会隔离这些卷**。增加卷前缀变量或独立验证 override，最终渲染的 DB/data/models/artifacts 都必须指向 `bysj-unblock-20261022-*` 独立名称。端口采用确认空闲的本机端口，不占用其他工程。
2. 新增一次性 `migrate` 服务执行 Alembic；API 和三个 worker 都依赖其 `service_completed_successfully`。当前 worker 只等 db healthy，可能先于 API 建表。迁移失败时 API/worker 不应继续工作。所有业务服务显式 `INLINE_JOBS=false`，按角色启动。
3. 保留原 SQLite 快速测试。新增独立 PG 集成测试路径，例如 `backend/tests/integration/`，同时修改根 conftest 的路由选择：当前根 conftest 无条件覆盖 DATABASE_URL、ML_BACKEND、INLINE_JOBS；单纯在 shell 设置 PG URL 或增加子 conftest 不够。
4. PG 测试模式需在导入应用前选定；使用专用库/schema 白名单及随机运行 ID，断言实际 dialect 是 postgresql。禁止把原有每测 `drop_all` fixture 指向共享库。空库结构必须由 Alembic 生成，不以 create_all 替代迁移验收。
5. CI 分开 SQLite、PG、GPU 门禁。修正测试依赖安装。现 GPU job 的 `-k gpu` 在目标训练生命周期测试文件中没有匹配用例；增加真实 GPU 用例和 marker，显式检查 collected > 0，跳过/全未选中不算 PASS。没有 GPU runner 时允许 GPU job 显示 NOT_RUN，附本机独立验收证据。

**验证**：PG 空库 upgrade head 成功、重复 upgrade 无新增破坏；迁移失败注入时业务不启动；三个 worker 注册且数据库连接为 PG；串行 SQLite 测试仍通过。PG 至少验证：同时提交同幂等键返回同一 job、并发 claim 只有一个成功、标注相同 revision 竞争一成功一 409、冻结/引用保护一致。GPU 互斥专项在 U03 完成。

**产物**：新增隔离验证 override、PG 测试入口与 README、迁移日志/表结构摘要、CI 定向结果。只保存脱敏配置摘要，不把展开密码的 Compose 全量配置写入公共报告。

**回滚**：停止本次新建服务；如需清理仅根据 U00 记录的资源清单处理。本任务无须删除任何既有卷；不要使用全局 prune。

## 8. U03：GPU 租约、取消、心跳与恢复（P0，长训练前必做）

**锚点**：`backend/app/domains/lease_service.py::acquire_lease/renew_lease/release_lease`；`domains/job_service.py::claim_job/heartbeat/finish_job/cancel_job/mark_stale_interrupted`；`workers/dispatcher.py::JobContext/run_claimed_job`；`workers/trainer_worker.py::_acquire_gpu/handle_training/handle_evaluation/run_forever`；`workers/inference_worker.py`；`domains/training/service.py::_run_ultralytics/_run_dummy/JobStopped`；`db/models.py::ResourceLease`；`api/v1/health.py::ready`。

**静态事实与风险**：

- 首次插入租约有主键冲突保护，但已存在过期租约的更新是读后写，无原子条件；两个连接可能都报告抢占成功。
- `renew_lease` 未接入 worker；默认 GPU TTL 600 秒，长训练超过该时间可被误认为可抢占。worker 的 30 秒存活租约只在外层循环刷新。
- `JobContext.progress` 只刷新 job；阻塞式评估/推理长调用没有持续进度时，任务可能过期。`_acquire_gpu` 默认等 120 秒后失败，不适配 80 epoch 的正常排队。
- 训练取消抛 `JobStopped`，dispatcher 只专门捕获 `JobCancelled`，因此存在取消落入通用失败分支的路径；dummy 训练也需验证取消语义。
- `claim_job` 已有 pending 条件更新，不应无依据重写；终态写入、过期清扫、旧 worker 回写仍需并发条件校验。
- `health/ready` 只依据数据库判断就绪，worker 数量为 0 也可报告 ok。

**实施**：

1. GPU 资源抢占采用数据库原子条件更新/UPSERT RETURNING 或等效行锁事务；续租、释放绑定 resource + holder + job + attempt/fencing token，返回是否实际更新。新增必要迁移。token 跨释放/重建不能回退；若释放删除行，应另有持久递增序列，否则保留租约行标为释放。
2. worker 存活、job、GPU 各有独立续租策略；在阻塞计算期间仍定时执行，周期不超过各 TTL 的 1/3。心跳线程/协程必须创建独立 DB session，不能跨线程共享业务 session。
3. 续租失败/失去归属即停止该 attempt 的计算和写回；仅当前 attempt 可写进度、终态、注册模型和产物。数据库 fencing 只能阻止陈旧写入，**不能终止仍运行的 CUDA 进程**：对失联的旧计算进程实行受控终止/确认退出，未确认释放时将资源置为待恢复，不能仅因 TTL 到期立刻启动第二个 GPU 任务。
4. 单卡先实现训练/评估/推理共享同一互斥资源、串行用卡。等待期间保持心跳、支持取消、展示 waiting_gpu；正常等待不因固定 120 秒直接失败。等待上限可配置并与执行失败区分；队列顺序可解释，避免一直抢不到资源。
5. 统一取消异常及状态。pending/cancelling/interrupted 的取消语义与状态机一致；正常取消最终为 cancelled。对子进程执行 terminate → 有界等待 → 必要时 kill → wait/reap，确认本任务子进程树退出后释放 GPU；不要结束不属于本任务的 PID。worker 被杀后孤儿训练不得继续占卡又启动新任务。
6. 终态/清扫使用条件更新和 attempt 校验；cancelling 失联也有恢复路线。训练已完成但注册之前崩溃时，重试注册不能生成重复模型。推理在 finally 释放 handle；单张坏图归入 item 失败而非误终止所有正常项，重试不重复计数。
7. health 分角色报告 dataset/trainer/inference 的存活、是否必需、最近心跳和 ML 能力。API liveness 保持轻量；完整部署 readiness 缺必需角色时降级，不能以“有任意一个 worker”代表全部就绪。

**强制用例（新增 PG 集成 + 进程级测试）**：

| 编号 | 刺激 | 必须观察到的结果 |
|---|---|---|
| L1 | 两独立连接同时抢首次/过期 GPU 租约，使用 barrier | 每轮仅一个有效持有者，重复至少 20 轮 |
| L2 | 任务持续超过 2 倍缩短测试 TTL | 续租有效，第二任务排队，不重叠执行 |
| L3 | 旧 token 在新 holder 生效后续租/释放/写终态 | 全部拒绝，不覆盖新 holder |
| L4 | 运行中训练取消、排队推理取消、重复取消 | cancelled；无遗留子进程；后续任务可正常用卡 |
| L5 | kill 本次测试 worker、断开其 DB、模拟长阻塞评估 | 状态准确、陈旧结果不落库；确认旧计算退出后才恢复用卡 |
| L6 | 完成/取消/清扫同时发生 | 只有合法终态；事件和注册记录不重复 |
| L7 | trainer 在线但 inference 离线，或长任务仍活跃 | readiness 按角色正确反映，长任务不被误报离线 |

缩短 TTL 的逻辑测试不能替代真实 GPU 演练。U04 至少实际验证一次运行中取消和后续复用。失败时禁止启动正式 80 epoch；允许继续 U05。

## 9. U04：真实 GPU 业务闭环冒烟（P0）

**前置**：U01 候选环境、U02、U03 通过。修改范围限 `backend/app/ml/`、训练/评估/推理域及其 worker 必要兼容修复，`scripts/smoke_medical.py` 和新增 GPU 集成测试。不得为通过冒烟绕开队列、鉴权、模型可信来源或版本冻结。

**执行**：

1. 建立独立冒烟数据版本，使用公开 BCCD 小子集。保持类别映射 RBC=0、WBC=1、Platelets=2，并验证本地 manifest；不要覆盖正式 291/37/36 版本。设置 1 epoch、batch=2、imgsz=160、seed=42，实际 device=cuda:0。
2. 先核对已登记基础模型 SHA256，再测试其在候选运行时的加载；历史 CPU 训练产物另测兼容性。不能以全局关闭安全加载检查来兼容任意权重。若 CPU 产物不兼容，保留原证据，使用可信基础模型在 GPU 环境重新训练产生新产物。
3. 由 HTTP API 提交 202，独立 worker 消费，数据库记录排队→运行→完成；返回模型、数据版本、代码和依赖 hash、真实 epochs_completed。API 进程不得执行训练。
4. 从注册模型发起独立 val 评估、单张/批量推理、导出和人工复核链路。评估响应当前为 `data.evaluation` + `data.job`，客户端按实际契约取任务 ID。冒烟指标仅用于确认非空、有限、来源正确，不作统计结论。
5. 再启动一个可运行足够时间的训练任务并取消，观察子进程退出、状态 cancelled、显存/租约释放；随后提交推理并成功完成。采集容器内 GPU 进程/利用率证据，不能仅依据 torch import 声称训练在 GPU。

**验收**：训练及评估/推理真实 backend=ultralytics，无 dummy；`.pt` 文件 hash 与注册记录匹配；API→队列→worker→模型→评估→推理完整闭环；取消后再次可用。若真实训练报 CUDA OOM，先降低冒烟 batch；正式实验预算另在 U06 冻结，不混改既有实验参数。

## 10. U05：六页面迁移与真实浏览器验收（P0）

**锚点**：`frontend/src/views/{Dataset,DatasetManager,Annotation,Training,ModelHub,Inference}.vue`；`frontend/src/api/` 已有各领域客户端及 `client.js`；`frontend/src/composables/useAnnotationDraft.js`、`useAnnotationHistory.js`、`useTrainingJobs.js`；必要的路由/组件及对应 v1 API。

逐个页面先列“旧请求→新请求→字段→状态→权限→失败显示”的映射表，再修改。不能只替换 import 或把旧路径前面加 `/api/v1`。已有客户端优先使用，字段以现有后端 OpenAPI/序列化实现为准。

| 页面 | 核心迁移与验收 |
|---|---|
| Dataset / DatasetManager | 导入 202 轮询、数据集与版本 ID 分离、分页资产列表、来源与 split 统计、冻结和引用保护；别把整数 asset_id 与 UUID dataset_id 混用 |
| Annotation | expected_revision CAS、boxes/deleted_ids、原图坐标、三页翻页、草稿恢复、撤销重做、冲突提示；图像读取走受控资源接口，不拼接本机绝对路径 |
| Training | 训练预检、模型/版本选择、进度事件、waiting_gpu、取消与中断提示；异步请求成功不等于训练完成 |
| ModelHub | 登记模型、血缘与哈希、独立 val/test 评估、真实指标/缺失原因及产物下载；失败不显示成零精度 |
| Inference | 单张/批量提交、持久化 run、分页结果/部分失败、可视化及 JSON/CSV 导出、复核提交和回流可达入口 |

若现有资产图片/缩略图或产物 API 不满足页面，补最小受鉴权下载接口，验证归属和路径边界；不要恢复旧应用的公开磁盘目录。实验/反馈功能优先接入已有页面与 client，不扩建另一套前端。

**测试**：保留现有前端 node 测试和 build；新增 Playwright 测试依赖、配置及 `test:e2e` 脚本（当前 package.json 无此脚本）。对真实部署跑至少 20 图、3 页：登录→导入→标注修改/删除/撤销→跨页→保存→刷新复核→冻结→训练/取消→模型评估→批量推理→导出→反馈。另测 401、其他用户资产 403/404、409 冲突、超时、坏图部分失败。

**证据**：截图/trace、请求路径清单、浏览器控制台错误、服务版本、job ID 和导出 hash。要求核心网络请求进入 v1；不因旧服务仍在后台而获得假通过。只在全部页面迁移后移除全局 axios/fetch 兼容补丁，并再次定向回归。UI 尚未完成时不允许先删除旧代码。

## 11. U06：实验工具修正与 E1–E4（P0/P1）

### 11.1 先修工具，后跑长实验

**锚点**：`scripts/run_experiments.py::REQUIRED/main/wait_job`、`scripts/benchmark_medical.py`、`configs/experiments/bccd-baseline.json`；`backend/app/api/v1/training.py::TrainingParams`；`domains/training/service.py::_normalize_params`；`ml/train_cli.py::run`、`ml/yolo_adapter.py::environment_snapshot`；评估域与 `workers/trainer_worker.py::handle_evaluation`；推理 worker 计时。

1. 训练参数目前仅传播 epochs/batch/imgsz/seed/deterministic/device；配置中的 augmentation、early_stopping 没有传到训练入口。增加明确白名单及类型/范围验证，逐层传递并保存最终生效的 Ultralytics args。patience 等映射清楚；不支持的 monitor/增强项应 422 或预检失败，不能忽略。测试请求参数与实际训练参数一致；支持 seed=0，不使用真假值判断有效配置。
2. 实验预检必须验证非空 seeds、已登记可信基础模型 ID/hash、数据版本与三集 hash、类别映射、训练预算、代码 SHA、依赖锁 hash、设备/框架版本；模型为空不允许静默使用随机初始化或任意网络下载。
3. 冻结文件采用按 config hash 的独立目录、写后只读语义；同目录配置变更时拒绝，不覆盖历史 `config.frozen.json`。幂等键包含实验协议/seed/任务类型，重跑从数据库恢复并补齐缺少的评估，不复用不同配置结果。
4. 当前脚本只收集训练返回的 val 指标。必须新增明确的独立 test 评估步骤：训练完成→取注册 model_id→POST evaluations(split=test,purpose=预登记用途)→轮询 data.job→读取评估记录。interrupted/failed/cancelled/超时全部单独记录；空 runs、缺 metric、NaN/Inf 不得算成功。所有 seed 数量不足时报告 partial，不填 0 补齐。
5. 评估输出真实 per-class 指标、混淆矩阵和可用曲线产物，关联哈希、split、模型和版本。当前适配器的 confusion_matrix/curves 占位 null 不能算实现完成；若某曲线本次后端不能提供，必须明确 reason 和缺项，不伪造。P0 要求的混淆矩阵应真实补齐。
6. 修计时：当前 queue_ms=0 等常量不是测量值。记录入队、领取、预处理、GPU 执行、后处理、存储/返回时刻；CUDA 区间使用正确同步/事件；各项定义和单位明确，不让总耗时错误等于某一子项。GPU 子进程内记录 allocated/reserved 峰值，结合采样的设备显存；说明采样间隔和其他进程影响。报告 p50/p95、吞吐、失败率和冷/热启动，未测项 null+reason。
7. `BYSJ_CODE_SHA` 在运行时/构建时准确注入，不能为 unknown 或借用镜像原项目 SHA。GPU 环境版本、锁 hash、基线模型 hash、数据 manifest/test hash 一并写入每个实验。

### 11.2 冻结协议与执行顺序

| 实验 | 执行要求 | 交付 |
|---|---|---|
| E1 基线 | 验证 BCCD 来源与本地 manifest；采用既有划分的真实冻结版本。预定 80 epoch、imgsz=640、seeds=[42,43,44]、轻量检测模型；batch=8 只是候选，先显存预试再冻结全部实验一致的可行 batch。实际早停 epoch 如实记录 | 三 seed 训练记录、独立 test 评估、每类 P/R/mAP、均值/样本标准差、混淆矩阵、环境与耗时 |
| E2 处理对比 | 从 E1 的 train/val/test 来源组出发，先划分后切片；同组派生图不跨集。切片推理要回映原图并统一去重，在共同的原图 test 目标上比较，不能把切片指标当原图指标。冻结相同模型、训练预算和 seeds；增强/切片差异单列 | 类别特别是小目标 Recall/mAP、性能代价、坐标/组隔离测试和处理配置 |
| E3 反馈闭环 | 使用预留训练侧样本池或明确的训练集纠错方案，审批后生成子版本；不能从 test 选样/调阈值/回流。对照相同初始权重、预算及 seeds；记录样本数、修正类型和新增信息 | 原预测与修正记录、审核记录、父子版本差异、固定 test hash、再训练/评估对比 |
| E4 性能 | 使用固定输入及并发，记录 100 图导入、标注保存、队列等待、训练任务接口、单/批图推理、资源峰值；冷启动与热运行分开，预热样本不混入 | 原始逐次测量、硬件、样本规模、p50/p95、吞吐与失败率、显存记录 |

**测试集使用纪律**：在 E1–E3 协议冻结前用 val 进行可行性与参数选择；协议冻结后只按预登记方案读取 test，不因测试结果较低再修改方案并继续报告同一确认实验。多个预登记对比可以使用同一固定 test，但不得用其反馈训练。

正式实验只用单卡排队，不同时堆叠多个 GPU 作业。8 GB 显存不保证 batch=8 一定可行。若 OOM 需要变更参数，生成新协议 hash，并对比较组统一重跑；不能暗中逐 seed 改 batch。真实反馈人工审核质量不足时，E3 标为 pending，功能闭环可用合成反馈验证但不能宣称真实人工效果。

**门禁**：工具回归先通过（参数丢弃、空 seed、缺模型、重复运行、失效 token、任务失败、缺指标）；再 GPU 冒烟；最后执行正式 E1–E4。允许如实报告无改善；不把小数据集指标解释为临床能力。

## 12. U07：旧库迁移、新旧行为对照与旧入口清退（P0/P1）

**锚点**：`backend/database.py`、`dataset_versioning.py`、`annotation_drafts.py`、`job_store.py`、`model_registry.py`、`start_api.py`、`yolo_api.py`；新域对应实现；Alembic 初始 revision；`docs/architecture.md` 和启动/构建脚本。

### 12.1 不依赖真实旧库 dump 的推进方式

现初始 revision 一次创建 21 张表；旧库已有 9 张共享表。**不能对未知旧库直接 stamp head 后称迁移完成**：stamp 不创建新增表，也不保证共享字段/约束兼容；直接 upgrade 又可能因表已存在失败。

1. 在隔离 PG 中依据旧 schema 建立合成旧库，加入用户、版本、资产、草稿 revision、job/event、模型注册的最小关联样本及受控文件；导入旧模块可能有副作用，应使用受控 schema fixture/DDL，不连接旧生产配置。
2. 导出新旧结构差异：列类型/可空/默认、主外键、唯一约束、索引及数据语义。选择明确的 additive 迁移或离线拷贝转换路径，记录旧 schema 指纹。只有严格识别到兼容旧 schema 并已有实际迁移方案时才允许采用明确 revision 标记，不做 blanket stamp。
3. 新增 12 表及必要共享字段/约束，保留 ID、归属、版本 hash、草稿修订、模型文件引用；修正路径映射的副本，不能写回旧源。验证行数、关系、文件 hash 与 API 可读取性；迁移失败可回退到演练快照。
4. 未拿到真实授权旧库副本时，结果写“合成旧 schema 演练通过；真实旧库数据兼容未验证”。全新部署可独立验收，不因缺 dump 把整个项目判为无法继续；也不伪称真实迁移已通过。

### 12.2 对照与清退

建立同一组输入驱动旧/新核心纯逻辑的对照：组级三集划分不变量、BBox 导出/边界、revision CAS、任务状态、模型 hash/血缘、安全解压。旧实现若与医学规范冲突，记录差异和采用的新规则，不要求错误行为也一致。不得将两套各自单测都通过当成行为等价。

U05 E2E、核心对照和新入口启动全部通过后，生成旧入口/行业模块的“文件→引用→运行用途→删除理由”清单，逐项清退。更新 Docker、CI、README、开发启动脚本、路由与发布元数据；毕业版只有一套权威运行入口。保留 Git 历史及来源说明，不用提交大批源码副本作为备份。每批删除后跑必要引用检查、构建与核心 E2E；不要因为文件名有行业词就删除仍被通用算法引用的逻辑。

## 13. U08：备份恢复、证据与最终交付（P0 收尾）

**前置**：新入口/正式镜像通过，核心功能与实验状态真实明确。

1. 在本次独立演示部署执行一致性备份：暂停新任务提交并等待/记录在途任务，导出 PG、受控文件清单与文件包；包含用户、版本、草稿、模型、评估、推理和复核血缘。密码/连接凭据不写入证据。
2. 恢复到另一套新建卷/库，验证数据行数、关键外键、模型/图像 hash、登录和已完成记录可读取；提交一个新的推理任务。恢复时处理旧 running/cancelling 任务为可解释状态，不能显示幽灵任务。
3. 更新部署文档：实际 Python/torch/vision/CUDA/Ultralytics、基础镜像摘要、锁用途、首次启动顺序、GPU 前置条件、备份恢复命令、故障排查、实验复现与数据准备。
4. 更新顶层 `docs/refactor/execution/STATUS.md`，逐项引用 U 证据。分别列“工程 P0”“E1”“E2–E4”“旧库真实副本”“人工授权/导师确认”，不把部分通过折叠成全完成。
5. 提交源码/测试/脱敏证据与可审阅差异。当前包不要求 DS 自动合并或发布到其他项目；若沿用用户已授权的 GitHub 交付流程，可准备独立分支/PR，并在交付中给出真实链接，不能把未推送的本地提交说成已上传。

**最终门禁**：全新正式镜像 + PG + 三类 worker + 前端能按文档启动；P0 核心 E2E 通过；真实 GPU 训练/取消/评估/推理通过；并发/恢复测试有 PG 证据；E1 有完整冻结协议及真实结果；E2–E4/人工项逐一说明状态；备份恢复通过；旧入口清退有证据。

## 14. 最小验证命令与运行方式

以下是已有或本包提供的检查命令；只读探针不挂载旧数据卷、不启动旧入口。`$BysjPacket` 改为本包实际位置。

```powershell
Set-Location -LiteralPath 'D:\智能体\bysj'
git status --short
git rev-parse HEAD
git branch --show-current
docker version
docker image inspect 6b1c56b48a25 --format '{{.Id}}'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

$BysjPacket = 'D:\毕业设计\DS续作任务包_20261022'
$BysjProbe = Get-Content -LiteralPath (Join-Path $BysjPacket 'probe_gpu.py') -Raw
docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=256m --cap-drop ALL --security-opt no-new-privileges --gpus all --workdir /tmp --entrypoint python 6b1c56b48a25 -c $BysjProbe
docker run --rm --pull never --network none --read-only --tmpfs /tmp:rw,size=256m --cap-drop ALL --security-opt no-new-privileges --workdir /tmp --entrypoint python 6b1c56b48a25 -m pip check
```

安装好 U01 对应测试依赖后，既有契约测试命令：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'backend')
python -m pytest backend/tests/medical backend/tests/test_app_factory_auth.py -q
Push-Location frontend
node --test tests/*.test.mjs src/utils/*.test.mjs
npm run build
Pop-Location
```

以上后端命令在原配置下是 SQLite/dummy，不是 PG/GPU 验收。U02/U03 新增测试入口应在各自记录中给出可直接运行命令、隔离库选择与 collected 数量；不要把不存在的脚本当作现有能力。Compose 在 U02 调整并验证卷隔离前不要直接 up；调整后记录最终 `config --quiet`、build、up、health、PG 测试、GPU 测试和 E2E 的完整命令。正式镜像至少进行一次无缓存构建验收，不需要每次小修改都重建全部依赖。

## 15. 交给用户的结果格式

每阶段用下表，不用“基本完成”代替证据：

| 字段 | 必填内容 |
|---|---|
| 任务 / 状态 | U 编号；PASS / PARTIAL / BLOCKED / NOT_RUN |
| 源码 | 分支、提交 SHA、变更文件、所用任务书版本 |
| 实际环境 | 镜像 ID、Python/依赖锁 hash、PG、GPU/驱动 |
| 测试 | 实际命令、退出码、通过/失败/跳过/选中数量 |
| 业务证据 | job/model/version/evaluation ID、产物相对路径与 hash |
| 实验声明 | smoke 或 formal；train/val/test；真实或 dummy |
| 剩余工作 | 明确阻塞条件、已试方法及下一步，不笼统归因 PyTorch |
| 回滚 | 提交/镜像/备份与本次资源清单 |

优先把 U00–U05 做到可交互、可恢复的真实 GPU 闭环；随后完成实验与清退。不要一开始就直接启动三组 80 epoch，再用训练等待时间掩盖工程缺陷。
