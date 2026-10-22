# U05 页面迁移契约

沿用现有 Vue 应用、六个路由、v1 客户端、坐标与有界历史工具；六个视图按医学检测范围改造。旧文件由 Git 历史保留，暂不删除仍未核查的行业模块。

| 页面 | 原请求/语义 | v1 请求/字段 | 状态、权限、错误 |
|---|---|---|---|
| Dataset | upload/dataset、旧 datasets/path | POST datasets/import multipart files → job_id；GET datasets 分页，dataset_id 为 UUID | 202 只表示接收；JobProgress 轮询；401 统一退出、其他错误显示 request_id |
| DatasetManager | training-datasets、路径型训练集 | datasets/{id}/versions、assets(version_id,page,page_size)、POST versions 冻结 | 展示 draft/validating/frozen、来源与 split；409 引用保护原文展示 |
| Annotation | annotation/旧草稿与静态目录 | assets/{int}/image 认证 blob；GET/PUT annotations/{int}；expected_revision/boxes/deleted_ids | 同图保存串行；切图前 flush；本地草稿按用户+资产隔离；409 保留草稿并显式重载/导出 |
| Training | train、status/{id}、tasks | training/preflight、training/jobs、jobs/{id}/events/cancel/resume | 选择冻结版本和登记模型；waiting_gpu 不当作失败；interrupted 可恢复 |
| ModelHub | models/路径下载、export | models、models/{id}/download、evaluations(split=val/test) | SHA/血缘；基础登记仅 admin；test 用途必须填写；缺指标显示未提供 |
| Inference | predict、predict/batch、内存结果 | inference/images → run_id + job；runs 分页；artifacts/{id}；reviews 提交/审批/export-version | 持久化结果，部分失败逐项展示；资产关联结果可复核；审批按角色，回流为新任务 |

补充后端闭环：受控图像接口、JSON/YOLO ZIP 实际导出产物、数据集/版本关联校验；标注保存同步权威资产 JSON，冻结时物化 YOLO 标签文件，防止训练使用旧标签。

验收必须来自真实 PG + 三 worker + 新 API，不能仅依据静态编译。
