# 医学图像训推平台重构详细任务书 V1.0

执行对象：deepseek-v4.1-flash。以 `bysj` 当前交付提交为执行基线，上游固定为 `xtythpt@aa331df0657ad4ffcf3c105f9ba52b7e159deb2a`。本任务书不是功能完成报告。

## 1. 目标、完成标准与设计选择

完成一个供教学与实验使用、可本地部署的医学图像检测平台：登录 → 导入公开图像 → BBox 标注 → 冻结三集版本 → 异步训练 → 注册模型 → 独立评估 → 单图/批量推理 → 人工复核 → 新训练版本 → 再训练对比。主案例为 BCCD 血细胞检测；结果只表述为研究输出。

必交范围包括原文档 P0、P1-a/P1-b 反馈闭环及 E1—E4 实验；P1 分割与 P2 暂不实施。先交付一个检测模型适配器；YOLOE/pose 的历史支持不强制作为医学版界面功能。保留原始源码来源与许可，不把迁移代码宣称为独立原创算法。

选定技术路线：Vue 3/Element Plus/Vite；FastAPI/Pydantic；PostgreSQL/SQLAlchemy/Alembic；PyTorch/Ultralytics；独立训练进程与独立批量推理 worker；Compose。任务状态仍以现有 JobTask/JobEvent 为权威，不同时引入 Redis/Celery。单卡时训练与 GPU 推理使用共享互斥租约，API 不等待长推理。模型推理缓存可先采用单模型，切换必须释放旧模型后更新状态。

初始运行环境候选为 Linux/WSL2、Python 3.11、Node 22、PostgreSQL 16；这是本任务包的设计选择，不是现有部署事实。PR-00A 通过实际安装/导入决定 Python 补丁版本、Torch/CUDA/Ultralytics 精确版本和基础镜像 digest。若候选不兼容，仅围绕报错依赖调整并记录 ADR，不盲目升级全栈。GPU 不可用不妨碍 CPU 静态/契约工作，但 GPU 里程碑不得假通过。

## 2. 允许范围、保护项与目录落点

- 工作目录：用户克隆的 `bysj`；本文命令默认在仓库根目录的 Bash/WSL2 执行，另有注明除外。当前编制资料位于 Windows `D:\毕业设计\delivery`，不是服务器部署路径。
- 允许修改：各任务卡列出的 backend/frontend/deploy/tests/scripts/docs 文件；新增目录必须有实际调用者和测试，禁止先建大量空壳。
- 保护：LICENSE、原始需求副本、历史来源与验证记录；未被本卡迁移的代码行为；已有草稿完整性/会话隔离；数据归属、任务取消与模型血缘。
- 禁止：修改上游仓库、公司服务器/数据、恢复生产环境配置、上传权重/数据/密钥、全仓格式化、强推历史；不把原文档中的生产路径视为执行目标。

目标结构采用下列职责，文件名标注为“新增”的均需本阶段创建，当前并不存在：

```text
backend/app/main.py                  应用工厂、lifespan、路由白名单
backend/app/core/{config,auth,errors,logging,storage}.py
backend/app/db/{base,session,models}.py + migrations/
backend/app/api/v1/{auth,datasets,annotations,training,models,evaluations,inference,reviews,experiments,health}.py
backend/app/domains/{datasets,annotation,training,models,inference,experiments,feedback}/
backend/app/ml/{yolo_adapter,geometry,tiling,metrics}.py
backend/app/workers/{dataset_worker,trainer_worker,inference_worker}.py
frontend/src/api/{client,auth,datasets,annotations,training,models,inference,reviews,experiments}.js
frontend/src/composables/{useAnnotationDraft,useAnnotationHistory,useTrainingJobs}.js
frontend/src/components/{annotation,training,inference}/
deploy/{compose.yml,compose.gpu.yml,backend.Dockerfile,frontend.Dockerfile,nginx.conf,.env.example}
scripts/{smoke_medical.py,import_bccd.py,run_experiments.py,check_split_integrity.py,benchmark_medical.py}
docs/{architecture.md,database.md,api.md,runbook.md,experiments/}
```

数据库物理表先复用现有名称；上述 `models.py` 可按域拆文件，但不能复制一套不同 Base 导致重复建表。`backend/app` 对旧模块的调用只能经显式服务桥接；新入口不得 `from yolo_api import app` 后挂载全部业务。

## 3. 执行顺序与工作量

| 卡 | 周次 | 前置 | 产物/阶段出口 | 估算有效开发日 |
|---|---|---|---|---|
| PR-00A | W1 | 基线核验 | 环境矩阵、源/需求确认、可复现验证入口 | 1–2 |
| PR-00B | W2 | 00A | 锁依赖、独立 Compose/CI、洁净启动证据 | 2–3 |
| PR-01 | W2–W3 | 00B | 新应用工厂、兼容路由、权限基架 | 2–3 |
| PR-02 | W3 | 01 | 医学导入、版本/分组/通用几何 | 3–4 |
| PR-03 | W4 | 02 | 可恢复 BBox 标注、组件拆分 | 2–3 |
| PR-04 | W5 | 02/03 | 数据库训练队列、取消/恢复与记录 | 3–4 |
| PR-05 | W6 | 04 | 可信模型注册、独立评估、实验索引 | 2–3 |
| PR-06 | W7 | 05 | 持久化单图/批量推理与导出 | 3–4 |
| PR-07 | W8 | 02–06 | BCCD 端到端 P0、E1/数据准备 | 2–3 |
| PR-08 | W9 | 07 | 复核回流再训练、E3 | 3–4 |
| PR-09 | W10–W12 | 08 | 清退耦合、完整验收、E2/E4、论文资料 | 4–6 |

以上是计划估算，不是已经测量的执行耗时或模型性能承诺。W8 必须形成 P0；延期先取消可选项。各卡可在一张 PR 中用多个小提交，但不能同时横跨多个未验收领域。首次代码核验脚本不通过时先报告差异，不覆盖当前文件。

## 4. PR-00A：确认可执行环境与冻结证据

**状态：READY；无需 GPU 即可开展。** 只核对固定清单，不重做全仓审计。

**路径/锚点：** `backend/requirements.txt` 的 ultralytics/torch 范围；`frontend/package.json` 的 scripts/dependencies；`backend/training_worker.py:patched_torch_load`；`backend/database.py:init_db`；`backend/start_training_worker.py`；历史 `docs/VALIDATION.json`。

**操作：**
1. 在干净分支执行基线校验，记录 `git rev-parse HEAD`、`git status --short` 与环境版本，不输出 secret。遇到已有变更先隔离分支和记录，不重置文件。
2. 新增 `docs/refactor/execution/PR-00A.md` 与 `docs/environment-matrix.md`，记录宿主、CPU/GPU、驱动、Docker/Compose、Python/Node、镜像来源。无 GPU 时明确 `GPU_PENDING`。
3. 列出根 `.env.example`、database.py、auth_utils.py、路径常量和 worker 启动变量实际使用关系；建立统一变量映射表。配置设计固定为 DATABASE_URL、JWT_SECRET_KEY、DATA_ROOT、MODEL_ROOT、ARTIFACT_ROOT、PUBLIC_ORIGIN，兼容别名仅在 config.py 中处理，不要求修改用户系统全局环境。
4. 在临时隔离环境解析依赖并记录解析结果；保留原 requirements 作为对照。先验证 Python/Torch/Ultralytics 的导入与一个受控模型加载，不运行原 yolo_api 启动钩子。没有可用模型则记录加载门禁未完成。
5. 记录原需求 G0 与学校成果界限的项目所有者说明；不自行填写“已获授权”或变更仓库公开性。展示证据缺失不阻塞独立文档、配置、测试设计，只阻塞对外毕设展示就绪声明。

**允许新增/修改：** 环境说明、执行记录、`backend/requirements.in`、测试依赖输入、环境探测脚本；暂不搬业务代码。

**通过：** 精确版本候选可安装或有具体冲突报告、配置变量覆盖全部进程、运行测试不会连接外部业务资源。**异常：** 缺 Docker/GPU 不能在报告中填构建通过；先完成可离线部分，留明确解阻命令。**回退：** 本阶段为文档/配置输入新增，按提交反向撤销即可。

## 5. PR-00B：毕业版构建、CI 与隔离启动

**前置：** 00A 环境矩阵 CPU 部分通过；GPU 锁如依赖 CUDA 必须独立生成并验证，不冒充 CPU 锁通用。

**路径/锚点：** 原 requirements.txt、database.py 的 DATABASE_URL/init_db/seed、start_training_worker.py、frontend/vite.config.js；新建 deploy 文件与 CI。

**操作：**
1. 将已验证版本写入 `backend/requirements.lock.txt` 和 `backend/requirements-dev.lock.txt`，采用带哈希的解析产物；CPU/GPU 差异用单独 lock 和 Docker ARG 指定。锁住直接与传递依赖，不能只冻结当前杂乱宿主环境。
2. `deploy/compose.yml` 新增 db/api/frontend，worker 使用 profiles 分离；所有卷属于独立 `bysj` project，数据库非生产地址，端口默认本机绑定。GPU override 只追加设备配置。训练 API 未就绪前可先测核心服务，不把 worker profile 未启动算全通过。
3. 后端容器临时入口保持旧代码的兼容路径，但在首次任何原 app 启动前隔离外网并禁用业务副作用。更稳妥的顺序是先建立最小 `/health/live` bootstrap 应用；PR-01 接上核心域后，00B 的登录/核心读回归作为追加门禁完成。不要通过加载整个旧 app 来“快速让 health 变绿”。
4. 新增 Alembic 配置和迁移基线。先支持独立空 PostgreSQL；已有 bysj 实验数据仅在备份副本执行迁移演练；原生产数据库不属于迁移输入。不得在生产对象上 `create_all` 或 `drop_all`。
5. 前端 `npm ci`、build 采用现有 package-lock；服务静态文件、`/api/v1` 代理和资源请求采用同源，禁止硬编码私网源站。记录源码 SHA、lock hash、构建时间到新建 `version.json`，不伪造旧 release.json。
6. 新建 `.github/workflows/medical-ci.yml`：独立 PostgreSQL service、Python 基础/契约测试、Node 既有测试、构建、Compose config、凭据/数据路径扫描。GPU 验收列为专门 job 或手动证据，普通 runner 跳过时标注 `not_run`。

**通过：** 空缓存锁定安装、前后端镜像构建、Compose config、数据库迁移、API live、前端页面可访问；登录和核心域在 PR-01 合并前补验。构建元数据与当前提交一致。**失败：** 只修冲突依赖或新增部署配置，不能复制原 ops/runtime-sources。**回退：** 停止本项目容器但不删除数据卷；恢复上一提交及数据库副本。禁止 `down -v` 作为默认步骤。

## 6. PR-01：新入口、鉴权和渐进迁移框架

**前置：** 00B 基础服务可运行。**旧锚点：** `yolo_api.py:startup_auth_system`、`shutdown_behavior_retention`、`app.include_router`；`auth_routes.py` 的 router；`dataset_access.py:require_dataset_read/require_dataset_manage`；database.py 的 SessionLocal/Base。

**操作：**
1. 新建 `app/main.py:create_app`，lifespan 只处理毕业版资源、基础用户和任务协调；模块导入时不建目录、不连数据库、不加载 GPU。把日志/异常/request_id/配置/受控存储放 core。
2. 在 `app/db` 复用同一个 SQLAlchemy Base/session；原 database.py 暂作兼容导出。先保持数据表名称和 ID 类型，别一次性替换所有 ORM。
3. 抽取登录、权限和 health；新路由白名单只包含 auth/datasets/annotations/training/models/evaluations/inference/reviews/experiments/health。禁止直接复用原 startup_auth_system、完整 admin_router 或 system_admin_router。
4. 建立 v1 schema 与错误转换；旧端点从独立兼容 router 调服务，旧 JSON 不强行包新 envelope。新旧适配层同测归属控制，HTTP 401/403/409/422/503 不吞成 200。
5. 对尚未迁移的域，新入口返回明确未启用状态或暂不挂载，不为“页面不报错”返回空成功。该过渡阶段不声称 P0 可用；每个后续域完成后才切其前端调用。
6. 新增前端 `api/client.js`，限定同源 base URL、timeout、AbortController 和单处 token/401 处理。先迁移 Login；其它页面在各卡迁移，全部完成后再移除 main.js 全局补丁。

**验证：** 新应用 import 无副作用；live 不依赖 GPU；ready 显示 DB 与所需 worker 可用性；登录、登出、过期 token、研究用户越权拒绝；OpenAPI 中无监控/行为/人脸/录机路径。00B 登录门禁在此完成。

**允许：** yolo_api.py 中本卡装配/鉴权片段、database/auth/dataset_access 必需兼容改动、新 core/db/api/health、Login/main/client 和对应测试。**回退：** 保留旧入口文件但仅在隔离环境使用；恢复本卡兼容映射，不删除新数据表。

## 7. PR-02：医学数据、不可变版本和通用几何

**前置：** 01。**旧锚点：** `upload_dataset`、`upload_training_dataset`、`list_dataset_version_assets`、`save_dataset_annotation_version`；`dataset_versioning.py:validate_dataset/persist_manifest_version/_asset_source_group/_grouped_three_way_split/export_training_version`；`image_size_processing.py:plan_annotation_tiles/render_annotation_tile`；`utils/archive_utils.py:safe_extract_zip`。

**操作：**
1. 提取 datasets service/repository，新增导入记录与医学来源字段；用 DatasetResource 表保留归属，不另造第二个 datasets 主实体。API 接收 ZIP/文件列表和 source_url/license/source_version/modality，不接受任意服务端绝对路径。
2. 上传写隔离 staging，经文件数/字节数/压缩比/像素/后缀与实际解码/标签验证后原子发布目录。复用安全解压，失败清理仅限本次 staging，数据库事务与文件移动配套补偿，不删除已有资产。
3. 图像 SHA 去重；外部 patient/slide/study 信息映射 source_group 并记录 group_level。缺失时使用 original_image 级组，重复内容归同组；目录/命名启发式只作为辅助标记。编写跨组重复冲突检查。
4. 从原 geometry/tiling 抽纯函数，保留坐标转换测试。医学默认禁用 phone/badge profile；类别直接来自冻结 class_map。派生资产保存原图 ID、矩阵、裁剪/pad 参数、来源组和 split。
5. 版本 draft→validating→frozen/failed；生成规范 manifest、sha256、split_manifest_hash、class_map_hash。先分组再派生；比例默认为 80/10/10，按组而非按图片硬凑。组不足/类别覆盖不可满足时阻止冻结并给出明细。
6. 冻结版本内容只读；修改标注创建子版本。删除使用软删除与引用保护；被训练/模型引用返回 409，不能沿用旧 `delete_dataset_version_records` 清空模型关联来绕过保护。
7. 新增 `app/workers/dataset_worker.py`，以独立非 GPU 领取函数执行 import/version/export，复用 JobTask/JobEvent；本阶段创建通用 `/jobs/{id}` 查询与取消接口，不能等 PR-04 才能轮询。大导出复用 `run_annotation_export_task` 的通用逻辑，worker 重启失败/恢复策略与临时目录清理明确。
8. 迁移 Dataset.vue、DatasetManager.vue，分页资产接口保留 total/page/page_size；默认 50，最大 200；不全量加载图片/base64。添加来源、版本、验证报告、split 分布。

**新增表/字段与 API：** 按 CONTRACTS C02/C03，新增迁移和 `tests/medical/test_dataset_versions.py`、`test_split_integrity.py`、`test_upload_safety.py`。

**验收：** 正常/空标签/越界/错误格式/恶意 ZIP 的错误定位到文件和行；三集来源组与内容哈希无交集；同 seed 同输入 manifest 相同；冻结后写入被拒；切片坐标回投偏差在浮点容差内；引用删除返回 409。**回退：** 保留迁移前副本、失败 staging 清单；不原地改冻结文件。

## 8. PR-03：BBox 标注与请求客户端迁移

**前置：** 02。**旧锚点：** Annotation.vue 的 saveHistory/undo/redo/buildAnnotationDraftPayload/scheduleAnnotationDraftSave/saveAnnotationDraft/loadImageToCanvas/exportDataset；`annotation_drafts.py:get_draft_for_update/upsert_draft`；`annotation_export.py:frontend_annotation_to_manifest`；yolo_api.py:put_annotation_draft。

**操作：**
1. 抽出 `components/annotation/ImageCanvas.vue`、ClassEditor、ExportDialog；`useAnnotationHistory` 按 image_id 保存有界历史；`useAnnotationDraft` 管 owner/dataset/version/geometry/model 会话键、节流和卸载保存。
2. 内部坐标统一原图归一化 BBox；展示缩放/平移只改变视图。保留人工标签与预测分层；新增漏检框允许不依赖预测 ID。YOLO 标签严格 class cx cy w h。
3. PUT 必带 expected_revision；事务锁内 compare-and-swap，冲突 409 返回服务器 revision。用户选择重新载入/另存子版本，禁止静默覆盖。成功后更新 revision；失败保持本地未保存状态。
4. 分页草稿只 upsert 提交资产，明确删除列表；保留未加载页。切换数据集时中止旧请求，迟到响应不得覆盖新会话。撤销/重做后自动保存正确版本；离开页面显示未保存提示。
5. 删除框可撤销；删除资产走回收与引用保护。视频采帧、摄像头和原行业类别不进入医学标注 UI。Polygon 保留数据可读，非本次训练路径，不静默转成 BBox。
6. 将标注 API 移至 `api/annotations.js` 和 client；后台大导出返回任务 ID，不能在前端全量 JSZip 拼巨量文件。

**验证：** 既有两组 annotation completeness/session isolation 测试必须保留其行为断言；新增修订冲突、跨页删除/保留、慢响应切换、缩放后坐标、空标签、撤销重做和导出回读。真实浏览器完成 20 张图 3 页操作，刷新后数据一致。

**允许：** 本卡组件/composables/API/annotation 源码及 schema、草稿迁移、测试。**回退：** 兼容旧草稿字段、保留完整原 JSON；前端可回切上一组件，不覆盖新 revision。

## 9. PR-04：持久化训练与可复现记录

**前置：** 02/03；PostgreSQL 可用；GPU 测试阶段需要合法可用权重和设备。

**旧锚点：** `TrainingConfig/start_training/_build_training_preflight`；`job_store.py:create_job/update_job/request_job_cancel/claim_pending_training_job/fail_orphaned_training_jobs`；`TrainingWorkerService._acquire_singleton_lease/run`；`training_worker.py:get_training_mp_context/training_worker_process/stop_training_process`；Training.vue:buildTrainingRequest/runTrainingPreflight/startTraining。

**操作：**
1. 新请求只接 dataset_version_id、approved model_ref、task=detect、epochs/batch/imgsz/seed/deterministic/device；校验冻结、class_map、BBox、所有者和受控权重。v1 的 imgsz 映射旧 img_size 仅在 adapter 中完成。
2. 保留 JobTask/JobEvent 七态；新增幂等键 `(owner_id,type,key)` 与请求 hash。相同键相同请求返回原任务，不重复建 workspace；相同键不同请求返回 409。
3. API 事务入队返回 202，不运行训练。原“第二个任务直接 409”改为排队；限制每用户待执行数量（默认 5，可配置），满额返回 429。训练 worker 单实例租约与 job 领取事务分开，避免重复持有行锁。
4. 移出训练进程启动与监控到 worker，保持 spawn、心跳、日志分段/游标。取消 pending 即 cancelled；运行任务先 cancelling，宽限期后清理自己创建的子进程并确认 GPU 释放，再标 cancelled。
5. 中断不声称恢复了训练 epoch：已有 checkpoint 且版本/hash 一致才可显式 resume；否则标 interrupted 并提供重新排队新 attempt。两 worker 竞争、API 重启和 worker 重启都不得重复训练。
6. 去除 deterministic=False 硬编码。保存随机状态、实际 seed、锁 hash、code SHA、Torch/CUDA/驱动/GPU、基础权重 hash、冻结数据/划分 hash、增强参数。非确定算子异常明确记录，禁止悄悄关闭设置。
7. 抽 Training.vue 到 TrainingForm/TaskTable/TaskLog/useTrainingJobs，去掉医学流程中的 PoseTrainingPanel；轮询有退避、离开停止、终态停止。迁移 API client。

**验证：** PostgreSQL 两连接并发同键提交只一条任务；双 worker 只一方执行；成功/非法标签/无 GPU/OOM/取消/进程被终止/模型缺失分别有状态证据。至少一次小规模真实训练成功，记录真实 output/hash；CPU 快测单独标识，不代替 GPU。

**回退：** 先停止接收新任务、等待或有序取消本分支作业，再回滚 worker/API；保留原日志/模型/任务记录。不得数据库回滚丢弃已完成任务。

## 10. PR-05：模型注册、独立评估与实验中心

**前置：** 04。**旧锚点：** `model_registry.py:register_model_file/build_model_lineage/filter_available_model_records/reconcile_models`；`database.py:ModelRegistry`；yolo_api.py:list_models/upload_model/download_registered_model；frontend ModelHub.vue。

**操作：**
1. 复用 model_registry 表，新增 owner_id、model_task、trusted_source、training_job_id/版本外键或迁移现有 task_id 语义；创建明确 v1 输出 schema，去掉 path 与任意 metadata 覆盖。
2. 只注册管理员审查的基础权重和本平台成功任务产物；普通用户上传 .pt 返回 403/405。下载通过 model_id 及归属校验；加载前再次 hash 校验，丢失/篡改标 unavailable，不自动扫描任意磁盘目录。
3. 新建 evaluation_runs 与 experiment_runs，评估复用 JobTask 的 evaluation 类型；任务明确 split=val/test，独立于 train 返回指标。test 运行记录目的/参数/审批角色，不用于在线调参反馈。
4. adapter.validate 规范化 P/R/mAP50/mAP50-95、per_class、混淆矩阵和曲线 artifact；没有指标使用 null+原因，不填 0 假装实际评估。数据集类别不兼容返回 422。
5. 模型与数据版本/训练配置/代码/依赖/设备形成可点击血缘，实验页按同任务、相同测试 manifest 与指标口径对比；不可比标明原因。

**验证：** 完成一个模型注册和 val 评估；独立 test 仅在最终协议下触发；用户 A 不能读 B 的模型/指标/下载；权重缺失/sha 变更阻止加载；不可信文件不触发 torch.load；引用删除受保护。

**允许：** models/evaluations/experiments 域、API、worker job 分派、迁移、ModelHub/Experiments 页面和测试。**回退：** 保留原模型文件与血缘记录，恢复上一版本读接口；不删除新实验结果。

## 11. PR-06：独立图像推理与结果持久化

**前置：** 05。**旧锚点：** yolo_api.py:predict_image/predict_batch/resolve_registered_model_path；`inference_service/client.py/router.py/catalog.py`；Inference.vue:getPayload/extractDetections/downloadResults/loadAvailableModels。

**关键事实：** bysj 不含 inference_service/app.py。上游该文件可在固定提交阅读，但携带 pose/tracking/多模型业务；本卡不整文件移入。新增 `app/ml/yolo_adapter.py` 与 `app/workers/inference_worker.py`，复用当前检测结果解析中与业务无关的逻辑。

**操作：**
1. adapter 定义 load/predict/validate/release；task=detect，输出原图像素 xyxy，类别严格使用注册 class_map。通过模型 ID 获取可信路径，用户不能提交服务器 path。
2. 推理先持久化输入资产与配置；单图也通过同一任务链返回 202，UI 等待即可，避免两套同步/异步一致性问题。批量一次最多 100 张，上传限额可配置；API 不持有 GPU 模型。
3. JobTask type=inference，子项各自 pending/running/completed/failed/cancelled；结果保存 inference_runs/predictions。批量部分失败不丢成功项；整体完成且 failed_count>0 显示“部分失败”，不另改 JobTask 终态枚举。
4. 单卡训练和推理共用 PostgreSQL GPU 租约，训练使用期间推理排队；模型释放并确认后移交。记录 queue_ms/load_ms/preprocess_ms/model_ms/postprocess_ms/total_ms，GPU 测量前后 synchronize。不要只用显存观察推断并发安全。
5. 结果图与 JSON/CSV 原子保存，路径通过鉴权资源接口提供；保留原预测，不直接让人工修改覆盖。导出 CSV 防公式注入，JSON 带 schema_version、模型/图像 hash 与参数。
6. 前端移除 getPayload 多种包裹猜测，新 API 严格 schema；图层区分预测与人工框；列表分页、取消、失败重试新 attempt、刷新后恢复记录。迁移所有推理请求至 api/inference.js。

**验证：** 单张、100 张、损坏图片、取消批次、worker 重启、GPU 被训练占用、模型切换/缺失/越权；图像尺寸/坐标/类别/JSON/CSV 一致；刷新仍能读；合法用户之外无法访问生成图片；进程重启不重复落库。

**回退：** 停止新接单，保留结果目录与数据库；恢复上个 worker/API 协议版本；不得靠删除结果使重试通过。

## 12. PR-07：BCCD 主案例与 P0 闭环

**前置：** 02—06。**新增：** scripts/import_bccd.py、scripts/check_split_integrity.py、docs/experiments/dataset-card.md、configs/experiments/bccd-baseline.json、公开数据 smoke fixtures。

**操作：**
1. 从原需求指定的 [BCCD 源仓库](https://github.com/Shenggan/BCCD_Dataset) 获取固定 revision，保存许可副本/URL/日期、原始图片和 XML hash 清单；实际数量重新统计，不把约 364 张写成下载验收事实。
2. 固定类别 ID：0=RBC、1=WBC、2=Platelets；XML 名称映射显式表。VOC 坐标转换按数据实际约定确定 1-based/边界语义，以人工已知框 fixture 验证；不能盲目对所有字段减 1。未知类别拒绝，越界有可追踪报告。
3. 没有患者/载玻片元数据则 group_level=original_image，记录限制；重复和近似图像生成检查报告，人工确认的近重复必须合并组。先固化 split，后续 E1/E2/E3 复用同一 test manifest。
4. 基础权重默认选择受当前锁支持的轻量 detect 模型，在模型配置中固定权重 SHA 和来源；候选 yolo11n，不为追逐模型编号升级全栈；如需换模型写 ADR 并重新定义基线。
5. 冒烟使用少量样本/1 epoch，只证明链路；正式 E1 固定 imgsz=640、batch=8（OOM 时记录调整）、epochs=80、seed=42、早停与 deterministic。配置先冻结，再训练；记录实际结束 epoch。
6. 以两角色完成登录/导入/标注/冻结/训练/取消/模型/评估/推理/导出；保留真实操作截图和产物 hash。P0 全部通过才进入正式反馈实验。

**通过：** 合规公开数据与所有来源字段齐全、split 检查通过、至少一轮正式真实实验、有可重放命令和结果；数据不可访问时只完成转换 fixture，标注真实实验 BLOCKED。**回退：** 导入新 dataset_id，不覆盖旧版本；删除只允许本卡明确生成且未引用的 staging。

## 13. PR-08：医学人工复核与再训练

**前置：** 07；P1-a/b 在本任务包内必交。**旧代码只读：** behavior/feedback_mapping.py 与 BehaviorReviewSample 仅说明旧业务不可直接套用；不得把其告警真伪当图像标签。

**新增：** app/domains/feedback、api/v1/reviews.py、review_records/review_items 迁移、Reviews.vue、api/reviews.js、反馈测试与 E3 配置。

**操作：**
1. 图像级审阅关联 inference_run、image_asset、预测快照与模型版本；verdict 为 correct/false_positive/wrong_class/missed_object/unreviewable，漏检允许 prediction_id=null。bbox 修正另存，不覆盖预测。
2. 状态 draft→submitted→approved/rejected；expected_revision 防并发覆盖，管理员或复核角色审批；研究用户可提交自己的样本，不可提升自己权限。
3. 回流事务只选择 approved、非 test 来源、非隔离/坏图的项。检查完整 root_asset/source_group/content_hash/test_manifest 血缘，而不是只检查当前记录 split 字符串。重复审批/导出同键幂等。
4. 创建 parent_version_id 指向原训练版的子版本；只追加/修正 train，保留 val/test 的逐资产 hash 与标签快照。test 或其同源派生图、近重复合并组不允许绕过新文件名进入 train；不改原版本。
5. 复核导出完毕必须重新运行 split/标签/类别验证，冻结成功才允许再训练。导出失败无半成品可训练版本。
6. E3 用同一基础模型、预算、参数、seed 和固定独立 test 比较初始模型与回流模型；审阅选择只用 train/unlabeled pool 或 val 开发集且不能把 val 再当独立最终评价。记录人工标注量和成本，不承诺改进必然为正。

**验证：** 错类修正、误检删除、漏检新增、不可判读排除、审阅越权、修订冲突、重复导出、test 原图/切片/改名副本回流均被拒；完成真实回流→再训练→同 test 对比。**回退：** 保留不可变 review/prediction/版本审计；撤销新版本可见状态而非覆写原数据。

## 14. PR-09：清退行业耦合、总体测试与毕业交付

**前置：** 所有必交域已验收。**操作：**
1. 从新入口导入图、OpenAPI、lifespan、SQLAlchemy metadata、依赖和前端路由出发确认剩余行业引用。逐组解除 behavior/face_capture/recorder/continuous_monitor/alarm_push/pose_training；共享工具先迁移到 core/ml。只有零运行引用、相关测试迁移、E2E 通过后才删除旧模块。
2. 删除旧 yolo_api 巨型业务入口或缩为不注册业务的兼容启动桥；禁止新 main 反向导入它。统一路径配置，删除固定 `/app/static`、`/app/runs` 业务假设。所有页面改用 client 后移除 main.js 全局 axios/fetch 补丁及废弃 API 文件。
3. 清除医学 UI 的 phone/badge/视频/姿态语义；代码历史/原需求中的文字可以保留，不能用全仓字符串计数为零代替运行边界验收。解释许可信息不做“全部改成 MIT”。
4. 完成 ACCEPTANCE 的功能/权限/安全/恢复/性能矩阵；E2 原图 vs 切片严格同原图 test，切片预测必须映射回原图合并后评估；E4 报告硬件与 p50/p95、吞吐、失败率、峰值内存显存。尽可能 seeds=42/43/44，不能完成 3 seed 时明确样本量限制。
5. 整理 architecture/ER/API/部署/恢复/数据卡/模型卡/测试报告/实验表与作图脚本；论文六章提纲和 8 分钟演示脚本追溯到真实 evidence。记录新增代码贡献、复用边界和 AI 辅助使用；学校格式未提供时产出可适配正文提纲，不声称符合特定校模板。
6. 用新机器或空卷完成从 clone 到导入/训练/推理/反馈的最终演练；由项目所有者确认展示记录和学校成果边界。提交最终 release PR，列出最终 commit/镜像 digest/lock hash/数据 manifest/模型 hash。

**通过：** P0 和反馈必交验收均有真实证据，无未解释的必交失败；原业务端点/启动任务不在最终运行图；数据无已知泄漏；仓库无真实业务资产。精度无强制“好看”门槛，真实负结果允许。**回退：** 使用最后验收通过提交和数据库备份，文件仅撤销本 PR 删除清单；不回写源仓库或生产环境。

## 15. 每张任务卡统一结束协议

每卡报告：基线 SHA → 本卡提交 SHA、允许/实际文件、迁移记录、逐项命令/退出码/证据路径、行为验收、未执行项及原因、与上一阶段是否有回归、回退方法、下一卡状态。报告模板见 execution/REPORT_TEMPLATE.md。

无新证据不重复运行同一失败命令；同类失败两轮后转为有界诊断：记录失败函数/参数/最小样例/日志，再修正文档中受影响的假设。基线漂移只暂停受影响编辑，保护其它可完成工作。不得通过删测试、放宽安全断言、将错误响应改成假成功来通过验收。

成功必须同时满足功能、数据、运行与证据门禁。静态语法、mock 测试、构建、health、真实医学训练分别证明不同层面，不能互相代替。
