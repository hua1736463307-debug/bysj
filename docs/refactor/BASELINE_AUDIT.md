# 当前基线审查与事实修正

## 证据范围

本次通过 GitHub 连接器读取两个私有仓库的 main、完整递归树，以及 30 个关键源码/配置/筛选文件，核对代码符号和调用链。读取的是仓库源码，不是运行中的服务。用户要求的源地址是 `hua1736463307-debug/xtythpt`，不擅自替换为带前导连字符的旧名称。

| 项目 | 本次确认 |
|---|---|
| 上游提交 | `aa331df0657ad4ffcf3c105f9ba52b7e159deb2a`，481 个 blob，递归树未截断 |
| bysj 原有提交 | `401cef475718e6adb00007b5e4d1b1c921ed91c4`，192 个 blob，原有根提交 |
| 文件关系 | 173 个同路径同 blob；14 个适配文件；5 个新增文档 |
| bysj 已适配 | `.env.example`、`.gitignore`、README、database.py、alarm_push.py、一个鉴权测试、前端导航/入口/依赖及 Dataset/Inference/Login |
| 核心文件规模 | yolo_api.py 16292 行；Annotation.vue 5199 行；Training.vue 2381 行；database.py 1356 行（以实际 splitlines 计） |
| 原附件 SHA256 | `f6d01882e050b4ee4b159ca94d0bdb97197debee92f193943f2df34877ff2695` |
| 本次改动策略 | 仅更新任务文档及新增验证脚本；既有应用代码逐 blob 保留 |

完整对象关系见 [baseline-manifest.json](baseline-manifest.json)。旧 `docs/source-selection.json` 是历史筛选记录，部分文本哈希记录与最终 blob 关系可能不同；本次文件是否相同以远端 Git blob 为准，不修改旧记录来制造一致性。

## R01—R10 复核

| 原编号 | 当前证据与结论 | 本次决策 |
|---|---|---|
| R01 | requirements.txt 多数为 `>=` 范围；bysj 没有后端锁文件 | PR-00 建立经真实解析和安装验证的锁，不恢复未知旧锁 |
| R02 | 上游 README 明确旧锁/旧发布 CI 已移除；bysj 没有 Dockerfile、Compose、CI | 新建毕业版独立构建部署链，不把旧 release.json 当作现有必须修复的文件 |
| R03 | yolo_api.py 仍集中实现数据、标注、训练、推理及行业业务 | PR-01 起按域渐进迁移，PR-09 清退旧入口 |
| R04 | Annotation.vue 草稿/画布/导出集中；Training.vue 混入 PoseTrainingPanel | 拆 composables 与组件，保留草稿完整性/隔离测试 |
| R05 | dataset_versioning.py 引入手机/工牌切片常量；image_size_processing.py 含专用 profile | 通用 geometry/tiling 单独抽取；医学默认原图，切片只作受控实验 |
| R06 | 上游 inference_service/app.py 的 ModelSpec.kind 是 detect/pose；bysj 中该 app.py **不存在**，只有 client/router/catalog | PR-06 新建毕业版推理 worker 与适配器；不假定服务已可启动，也不把客户端当服务端 |
| R07 | 医学 review_records/inference_runs 等持久化业务未形成，原反馈是行为事件体系 | PR-08 新建图像审阅，不复用告警 verdict 或视频剪辑表 |
| R08 | training_worker.py 约 578 行硬编码 deterministic=False | PR-04 参数显式传递与环境快照，不以 seed 一项保证完全复现 |
| R09 | main.js 同时全局 axios 拦截和 window.fetch 包装；两套 API 文件并存 | 分阶段统一受控客户端，最后移除兜底，所有请求验证归属和失败行为 |
| R10 | bysj 已排除 runtime-sources、ops、scripts 与生产 Compose | 继续沿用筛选边界，禁止为凑齐启动配置从原仓库整目录恢复 |

## 影响计划的新发现

1. `startup_auth_system` 不只初始化登录：还迁移行为反馈、重放行为事件并启动保留期清理。`include_router` 注册姿态、告警、录机；`register_face_capture` 注册人脸路由。新应用必须使用明确路由白名单和自身 lifespan，禁止直接挂载完整旧 app。
2. `job_store.ALLOWED_TRANSITIONS` 实际有 pending/running/cancelling/interrupted/completed/failed/cancelled 七态。保留中断和取消阶段；终态不可转回运行。
3. `TrainingWorkerService._acquire_singleton_lease` 使用 PostgreSQL advisory lock；单元 SQLite 不能证明队列竞争正确。`claim_pending_training_job` 和领取事务必须用真实 PostgreSQL 验证。
4. `_asset_source_group` 在没有显式来源时会按目录或文件名启发式分组。这不等于患者/载玻片隔离。医学导入必须记录来源级别、缺失情况，禁止把启发式组宣称为患者独立。
5. `_grouped_three_way_split` 已有组级三集分配与类别覆盖约束；需复用并增加重复内容跨组校验、固定测试集标识和新版本继承规则，不重新随机划分反馈版本。
6. `AnnotationDraft.revision`、`upsert_draft` 及 `put_annotation_draft` 内的 revision 冲突检查已存在，不能把它们报告为完全缺失或再建一套无关联草稿。v1 把请求字段规范为 expected_revision，继续在更新事务内比较；前端仅提交增量仍需保留未加载页的资产。
7. `ModelRegistry.to_dict` 当前暴露 path 且末尾合并 metadata；v1 响应使用显式 schema，禁止元数据覆盖受保护字段或泄露服务器绝对路径。
8. 训练主入口 `start_training` 当前接受 dataset_path，GPU 并发达到 1 时拒绝第二个任务。毕业版改为受控 version_id、数据库排队、单活跃 GPU 任务；该行为属于有意变更，不能误报为完全等价。
9. 原 torch.load 兼容补丁、模型上传及按路径下载需要收敛为受信任模型目录与 model_id；不因“只读权重”就假设反序列化安全。
10. 前端 package.json 没有 `test` 脚本，已有 Node 原生测试，不应写成“运行 npm test 已通过”；Vitest/Playwright 若引入须新建依赖、脚本和首个真实用例。

## 不覆盖的事实与运行限制

未访问原服务器、数据库、摄像头、训练作业或用户资料。未安装应用依赖或运行应用导入，以免原入口启动钩子产生副作用。Git 直接克隆在此环境分别遇到 Schannel close_notify 和 OpenSSL EOF；随后使用连接器读取固定提交，未修改全局 Git/TLS 配置。

历史 `docs/VALIDATION.json` 记录 97 个后端测试、24 个前端测试及构建通过；这是此前的隔离筛选证据，本次未重新验证。真实 GPU、模型兼容版本、洁净 PostgreSQL 与 Compose 的可运行性均留给 PR-00/04/06 的明确门禁。

## 原文档与当前任务的关系

原文档是产品需求和旧审查材料。本次用户明确要求“根据指定仓库与附件制作任务书并上传新基线”，且确认包含现有源码。保留其医学研究定位、数据治理、真实性和阶段验收要求；修正旧路径和静态判断。代码使用/展示授权与学校认定由项目所有者提供记录，AI 不声称已核验；这不作为停止本次已授权的文档与现有基线发布的理由。
