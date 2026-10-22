# API、数据、权限与任务契约

本文是待实现的目标契约。除明确注明“已有”的表/函数外，不代表当前源码已经具备。所有新增 migration/endpoint/schema 在相应 PR 建立并测试。

## C01：接口统一与兼容

新接口统一 `/api/v1`；JSON 成功 envelope 为 `{code:"ok",message:"...",data:{...},request_id:"uuid"}`。失败同样 envelope，code 为稳定业务字符串，data 可包含字段错误；HTTP 使用实际语义状态，不以业务 code 代替 HTTP。

- 200 读取/更新、201 同步创建；202 异步接收并返回 job_id/status/status_url；204 无内容不返回 envelope。
- 400 请求结构以外的业务无效；401 未认证；403 无角色权限；404 不存在或无可见权资源（统一避免 ID 枚举）；409 修订/幂等/不可变/引用冲突；413 上传超限；422 字段/标签/几何错误；429 队列或速率超额；503 worker/必要依赖不可用。
- 文件下载为流式响应，不套 JSON，失败仍统一错误；不返回绝对文件路径、连接串或 traceback。request_id 写入日志。
- 列表统一 `items,total,page,page_size`，稳定排序 created_at + id；默认 page_size=50，上限 200。分页 SQL 必须在后端执行。
- 创建长任务支持 `Idempotency-Key`；唯一键 owner/type/key，保存 canonical request hash。同请求返回同一 job，冲突 409。
- 旧路径与旧返回只在 legacy adapter 存在；新页面各域切 v1 后再退旧路径。禁止同时让全局 axios 和 v1 client 重复 unwrap。

### API 端点清单与旧实现映射

| 方法与 v1 路径 | 关键输入/返回 | 旧锚点/新实现 | 阶段 |
|---|---|---|---|
| POST /auth/login、POST /auth/logout、GET /auth/me | token/session、用户角色；不记录密码 | auth_routes.py；新增显式响应 | 01 |
| GET /health/live、GET /health/ready | live 进程；ready 按 DB/worker 分项 | 新 core health，不导入业务单体 | 01 |
| GET /system/resources | 当前实验机 CPU/RAM/GPU 与队列摘要 | 原 system/resources 仅取通用采集逻辑 | 04 |
| POST /datasets/import | multipart files + source metadata →202 job_id | upload_dataset/safe_extract_zip | 02 |
| GET /datasets、GET /datasets/{id} | 所有者过滤、分页 | list_datasets + DatasetResource | 02 |
| GET /datasets/{id}/assets | page/page_size/version_id →图片元信息 | list_dataset_images/list_dataset_version_assets | 02 |
| GET /datasets/{id}/versions | 版本与统计 | list_dataset_versions | 02 |
| POST /datasets/{id}/versions | parent_id/annotation_revision/split_seed/ratios →202 | save_dataset_annotation_version/persist_manifest_version | 02 |
| GET /dataset-versions/{id} | state/manifest/hash/split/class map | v1 schema 显式字段 | 02 |
| DELETE /datasets/{id}、DELETE /assets/{id} | 软删；引用时409 | 原 delete_dataset/delete_dataset_asset 需收敛 | 02 |
| POST /datasets/{id}/restore、POST /assets/{id}/restore | 管理权限、恢复隔离对象 | 新恢复 service | 02 |
| GET /annotations/{image_id} | draft/annotations/revision/source_version | annotation_drafts | 03 |
| PUT /annotations/{image_id} | expected_revision/boxes/deleted_ids →新 revision | put_annotation_draft/upsert_draft | 03 |
| POST /annotations/export | dataset_id/version_id/format →202 | annotation/export-training-dataset/start | 03 |
| POST /training/preflight | 冻结版本/受信任模型/参数 →校验明细 | _build_training_preflight | 04 |
| POST /training/jobs | version/model_ref/task/epochs/batch/imgsz/seed →202 | start_training；禁止 dataset_path 输入 | 04 |
| GET /jobs、GET /jobs/{id} | type/state/progress/stage/错误/结果引用 | JobTask | 04（通用部分02先建） |
| GET /jobs/{id}/events | after_sequence/limit →有界事件 | JobEvent | 04 |
| POST /jobs/{id}/cancel | 归属+状态检查 →202或已有终态 | request_job_cancel | 04 |
| GET /training/jobs/{id}、POST /training/jobs/{id}/cancel | 通用 job 的 training 类型别名 | 不能暴露其它类型作业 | 04 |
| GET /models、GET /models/{id} | hash/task/lineage/available | ModelRegistry | 05 |
| POST /models/register | 管理员受控 source/ref/hash | register_model_file；不接任意路径 | 05 |
| GET /models/{id}/download | 权限+可信目录下受控文件流 | download_registered_model | 05 |
| POST /evaluations | model_id/version_id/split/params →202 | 新 evaluation job | 05 |
| GET /evaluations/{id} | 指标/类指标/artifact/eval context | 新 evaluation_runs | 05 |
| POST /inference/images | multipart 图片或有权限 asset_ids + model_id →202 | predict_image/predict_batch 改任务链 | 06 |
| GET /inference/runs、GET /inference/runs/{id} | 分页、分项状态/预测/错误 | 新持久化结果 | 06 |
| GET /inference/runs/{id}/export?format=json/csv | schema_version/模型/图像/参数/hash | 新 exporter | 06 |
| GET /artifacts/{id} | 权限+引用解析+目录边界 | 新资源服务 | 02/05/06 |
| POST /reviews、GET /reviews、GET /reviews/{id} | image/run/verdict/correction/revision | 新医学审阅 | 08 |
| PATCH /reviews/{id} | expected_revision/审阅内容 | compare-and-swap | 08 |
| POST /reviews/{id}/submit、POST /reviews/{id}/decision | 审批角色+状态校验 | submitted→approved/rejected | 08 |
| POST /reviews/export-version | approved ids/parent_version →202 | 创建新 train 版，test 永不回流 | 08 |
| GET /experiments、GET /experiments/{id} | 同口径记录与可复现元数据 | experiment_runs | 05/08 |

通用 `/jobs` 必须在 PR-02 支持 import/version/export 等非训练 job，不等到 PR-04 才能轮询。PR-04 扩展其训练状态与 worker 分派；不让导入任务调用 GPU 领取函数。非 GPU 作业采用单独数据库领取函数与后台 worker，禁止用失去进程就丢失的内存 BackgroundTasks 作为持久作业权威。

### 示例（仅契约，非实验数据）

```json
{"dataset_version_id":"uuid","model_ref":"approved-detect-id","task":"detect","epochs":80,"batch":8,"imgsz":640,"seed":42,"deterministic":true,"device":"cuda:0"}
```

异步响应 data 最少包含 `job_id,status,status_url`。推理完成 data 包含 `run_id,model_id,model_sha256,image_id,image_sha256,image_width,image_height,params,timing,predictions`；prediction 为 `{id,class_id,class_name,confidence,bbox_xyxy}`。空预测是合法空数组，错误预测不能伪装为空数组。

## C02：数据库复用与迁移

保留当前物理表及已有 ID 类型，用新 schema 映射产品术语，避免一次性建第二套 datasets/training_jobs/models 数据源。

| 现有类/表 | 目标处理 | 关键字段/约束 |
|---|---|---|
| User / users，角色与会话表 | 保留、收敛角色 | 保留当前用户主键类型，其他表 owner_id 用实际同型 FK；迁移前规范历史字符串 owner_id |
| DatasetResource / dataset_resources | 作为 datasets 主实体 | 补 name/source_url/source_version/license/modality/task_type/group_level/deleted_at，现有归属不丢 |
| DatasetVersion / dataset_versions | 保留 parent_version_id 等 | 新 state、frozen_at、manifest_sha256/split_manifest_hash/class_map_hash（或对现有 content_hash 明确兼容映射），FK 与引用保护 |
| DatasetAsset / dataset_assets | 作为 image_assets | 补 root_asset_id/source_group/group_level/split/width/height/hash、派生变换、original_asset_id；复用 annotations_json |
| AnnotationDraft/AnnotationDraftAsset | 可编辑区 | revision compare-and-swap；每 owner/dataset/source_version/geometry 会话唯一；完整记录删除 tombstone |
| JobTask/JobEvent | 通用任务账本 | type/owner/input/result/status/stage/heartbeat/attempt/idempotency；(task_id,sequence) 唯一，领取事务持锁后提交 |
| ModelRegistry / model_registry | 模型产物 | 保留 sha256/task_id/dataset_version_id；补可信来源/task/owner/available；模型路径仅内部可见 |
| OperationLog 等通用审计 | 保留核心操作 | 事件、操作者、资源、前后版本、时间、request_id；不保存 token/密码 |

新增：

| 表 | 关键字段与外键 | 不变量 |
|---|---|---|
| dataset_imports | id,owner_id,job_id,source_meta,staging_ref,status,report_ref | 失败导入不得变成 frozen 版本 |
| evaluation_runs | id,owner_id,job_id,model_id,dataset_version_id,split,params_json,metrics_json,artifacts,context_hash | 指标保留口径与实际 split；追加不可覆写 |
| inference_runs | id,owner_id,job_id,model_id,model_hash,params_json,status,counts,timestamps | 提交参数冻结、所有子项归同 owner |
| inference_items | id,run_id,asset_id,image_hash,status,timing_json,error,artifact_id | unique(run_id,asset_id)，重试有独立 attempt |
| inference_predictions | id,item_id,class_id,confidence,bbox_json,raw_json | 原始预测不可被 review 覆写 |
| review_records | id,owner_id,reviewer_id,inference_item_id,state,revision,reason,created/decided_at | 图像级，允许无预测的漏检审阅 |
| review_items | id,review_id,prediction_id nullable,verdict,original_json,corrected_json | missed_object 可无 prediction；纠正坐标须验证 |
| feedback_exports | id,job_id,parent_version_id,child_version_id,review_ids_hash,idempotency_key | 一次成功导出唯一子版本，失败不可见 |
| experiment_runs | id,owner_id,training_job_id,model_id,dataset_version_id,code_sha,env_json,config_json,artifact_refs | 环境/版本/权重/指标可追溯 |
| artifact_records | id,owner_id,resource_type,resource_id,relative_path,sha256,mime,size | 路径相对受控根，资源引用与权限一致 |

以上为最小列集；字段类型、索引与外键必须在 migration/ER 文档明确。不强制将所有原 annotations_json 立即拆成单框表，避免双写；冻结版本内标签通过 manifest 固化，草稿经 transaction 导出子版本。

已核验 ID 映射：User.id 与 DatasetAsset.id 是 BigInteger；DatasetResource.id 为 UUID 字符串，但业务身份是 `(dataset_id,dataset_role)`；DatasetVersion.id/JobTask.id/ModelRegistry.id 为 UUID 字符串。v1 dataset_id 统一使用 DatasetResource.id，内部 repository 映射回旧 dataset_id+role，禁止把两种 ID 混传。新增 owner_id FK 用 BigInteger；JobTask 旧字符串 owner_id 先保留兼容并补 owner_user_id，回填校验后再切换，不无校验强转。现有 role=user 映射为 researcher 的产品角色，历史角色映射须记录；新用户不再生成可逆 password_ciphertext，旧副本中该字段不得通过 API 暴露。

迁移顺序：先 Alembic 识别既有表 → 添加 nullable 字段 → 显式校验/回填 → 加约束 → 切应用读取。空库用 upgrade head；已有实验副本验证升级与保留行数、hash/引用；无备份不做破坏性 downgrade。行业表先不映射进新应用，PR-09 在证据和备份完整后另列删除迁移，不默默删表。

## C03：版本、分组与测试集治理

1. 数据集来源字段强制记录 URL、许可文本/说明、源版本、下载日期、原始文件 hash。未知许可/来源不能填默认 MIT；无来源数据不能纳入正式实验。
2. 规范 manifest 排序、相对 POSIX 路径、稳定浮点编码，hash 不包含生成时间等易变字段。manifest hash 与 split_manifest_hash 分开。
3. 一张原图及所有裁剪/切片/增强归同 root_asset/source_group/split；有患者/slide 元数据按其最高可得来源级别分组。相同内容哈希跨组时先合并或阻止冻结，不能只检查组字符串交集。
4. `train∩val=train∩test=val∩test=∅` 同时对来源组、root_asset 和内容 hash 成立；近重复审查是附加报告，不声称普通 SHA 可检测近重复。
5. frozen 的像素、标签、类别表、split 不可更新。修订创建新 parent_version_id；复核版本继承原 test/val 快照，只调整 train。下载/导出中不得悄悄重新随机划分。
6. 测试集不是调参池。评估 test 需冻结评估协议，并记录每次使用目的；最终 E1/E2/E3 可按预先固定协议评估一次各模型，禁止看结果后反复选择训练参数。反馈导入追溯全血缘拒绝 test 及同源/重复资产。
7. 标注规范：BBox 归一化中心点/宽高用于训练；UI/API 以明确 geometry_type、坐标空间字段表示。推理 xyxy 为原图像素；转换函数单一权威，round-trip 用已知尺寸 fixtures。
8. 切片实验的 test 始终以原图对象计分；还原坐标、合并跨 tile 重复后再算指标，记录 merge/NMS 参数。禁止把 tile 指标直接和原图指标并表。

## C04：任务状态与资源所有权

复用现有七态与合法转换表；不删 cancelling/interrupted：

```text
pending -> running | cancelling | cancelled | failed | interrupted
running -> cancelling | completed | failed | cancelled | interrupted
cancelling -> cancelled | failed | interrupted
interrupted -> pending | running | failed | cancelled
completed / failed / cancelled -> 无后继
```

更严格的 worker 条件：仅持有当前 lease/attempt 的执行者可写进度/终态；陈旧 worker 更新应拒绝。API 重启不篡改活跃 worker 所有权；心跳超时进入 interrupted 并核验进程/租约，再由显式恢复策略处理。取消与完成并发时事务比较状态，不能把已完成任务改为取消成功。

待运行 GPU 作业可显示 `stage=waiting_gpu`，不篡改 API 层状态集合。单 GPU 只有一个训练或推理执行者持有资源租约；worker 退出释放租约并清理它创建的子进程，不能 kill 其它应用进程。进度是工作进度估计，不作为持有 GPU 的证据。

事件日志分页，单次最多 200 条；长文件日志存文件/对象，DB 只存索引和简要事件。WebSocket 若保留，鉴权和资源归属与 REST 相同；页面卸载断开，不持续堆内存。

## C05：权限与文件边界

角色最小 admin/researcher；reviewer 可新增。管理员管理用户、受信任模型、复核审批和实验配置；研究用户只能访问自己的数据/作业/模型/结果；共享数据须显式授权记录，不能以公共静态目录实现共享。

每个资源执行认证→资源读取→归属/角色校验→文件解析。客户端提供 asset/model/artifact ID，不提供磁盘路径。resolve 后必须留在 DATA_ROOT/MODEL_ROOT/ARTIFACT_ROOT 且防 symlink 逃逸；导出名与文件名清洗，不能通过编码变体绕过。

上传建议初始限制（设计默认，可配置并在 UI 展示）：单文件图像 20 MiB、单次总量 512 MiB、ZIP 最多 10000 文件、解压总量 2 GiB、最大压缩比 100、单图最多 40M 像素；批量推理最多 100 张。这些不是测量结果，PR-02 按实验机内存验证，任何调整写配置和测试。服务器实际解压/累计字节必须二次计数，不只信 header/ZIP 声明大小。

普通用户不直接上传并加载 .pt。管理员登记也要限定受信任来源/目录与 hash；hash 证明内容一致，不证明文件无恶意。不得全局设置任意反序列化放行来消除报错。

## C06：可复现记录与性能

每次训练/评估保存 code_sha、输入模型 hash、dataset/version/manifest/split/class_map hash、精确依赖锁 hash、超参、augmentation、seed、deterministic 实际值、Python/Torch/CUDA/驱动/GPU、开始结束时间、完成/失败原因、产物 hash。不可用字段填 null 和原因。

E1 固定单配置；E2 只改变通用切片/增强策略；E3 只改变审阅后的训练样本；E4 记录系统负载。训练预算定义 epochs/样本量/优化步数，切片后样本数变化要额外报告实际步数与耗时，不能仅“同 epochs”就声称等计算预算。

性能统计明确排队、冷启动和热推理，记录 warm-up 数、实测样本数、分辨率、并发、成功/失败、计时边界、p50/p95、吞吐、峰值 RSS/显存。GPU 计时同步；不预设 mAP 或延迟成功值。

## C07：文档、证据与毕业交付

docs/architecture.md：模块职责与部署图；database.md：真实 ER/索引/迁移；api.md：OpenAPI 与错误例；runbook.md：洁净安装、变量、启动/停止、备份恢复；experiments/：数据卡、模型卡、配置、原始指标、统计/作图脚本、失败案例。

论文资料采用原需求六章结构；真实代码迁移/新增功能、复用与 AI 辅助贡献分开说明。演示 8 分钟使用公开图像和预先完成的真实训练记录，现场短流程可演示任务提交与推理。没有完成训练时不得用手填指标或模拟 UI 截图替代。
