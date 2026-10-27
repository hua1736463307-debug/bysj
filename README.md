# 面向多病种的医学影像辅助诊断系统（研究原型）

**当前主线：胸部 X 光五标签辅助分析**（`cxr_multilabel` / `cxr-five-v1`）。
匿名病例影像 → 输入与模型匹配校验 → 五标签评分 → 关注区域与解释 → 人工复核 → 可追溯的辅助报告。
默认研究用途、必须人工复核；不输出确诊，不提供用药或治疗建议。

- 范围与红线：[docs/cxr/SCOPE.md](docs/cxr/SCOPE.md)
- 实施状态与分层验收：[docs/cxr/STATUS.md](docs/cxr/STATUS.md)
- 逐阶段记录：[docs/cxr/execution/](docs/cxr/execution/)
- 新接口：[docs/api.md](docs/api.md) 第 9 节；新表：[docs/database.md](docs/database.md) 第 8 节

五项观察：肺不张、心影增大、肺实变、肺水肿、胸腔积液。保留 `positive/negative/uncertain/unmentioned` 四态；
监督只用明确 positive/negative，`uncertain` 与空值不参与训练、也不当阴性。按患者 component 划分
train/val/calibration/test（70/10/10/10），同患者与已识别重复不跨集。

> 真实数据闭环与研究结论 **尚未完成**：本机无 CheXpert 授权副本、无 GPU、无 torch；
> 工程闭环用合成 fixture 与 dummy 后端验证契约，不产生任何可引用的研究指标。详见 STATUS。

---

# 历史研究模块：BCCD 检测链路

用于公开医学图像的教学与实验：导入、BBox 标注、不可变三集版本、异步训练、独立评估、模型管理、批量推理、人工复核与训练集反馈。

当前权威入口为 `backend/app/main.py`；胸片五标签页面在 `frontend/src/views/cxr/`，旧检测页面保留原路由与 deep-link。
继续实施的状态、证据和未完成项见 [CXR 状态](docs/cxr/STATUS.md) 与 [旧续作状态](docs/refactor/execution/unblock/STATUS.md)；本文件不代表全部毕业实验已经完成。

- [运行手册](docs/runbook.md)：部署、独立 worker、健康检查及故障恢复。
- [开发说明](docs/DEVELOPMENT.md)：Python 3.12、目标平台锁、测试和前端构建。
- [架构](docs/architecture.md) / [API](docs/api.md) / [数据库](docs/database.md)。
- [E1/E2 结果与图表](docs/experiments/2026-10-22/RESULTS.md) / [E4 性能实测](docs/experiments/2026-10-22/PERFORMANCE.md) / [E3 人工待审说明](docs/refactor/execution/unblock/E3-HUMAN-HANDOFF.md)。
- [续作计划](docs/refactor/execution/unblock/PLAN.md) / [原始任务书](docs/refactor/TASK_BOOK.md) / [原始需求](docs/requirements/医学信息系统开发_v1.0.md)。

当前已实测干净 GPU 镜像、PG 并发、取消/崩溃/断库恢复、正式浏览器闭环和新卷备份恢复。E1 三种子、E2 首种子已取得真实数值，但事后发现跨集近重复风险，已停止旧协议续跑，不能作为通过独立性验收的泛化结果；见 [数据质量说明](docs/experiments/2026-10-22/DATA-QUALITY-REVIEW.md)。E4 工程性能测量完成，E3 等待真实人工审核。合成旧 schema 迁移已通过；真实旧库副本兼容性尚未验证。

源码源于 [xtythpt](https://github.com/hua1736463307-debug/xtythpt) 的筛选基线，再沿 bysj 历史连续重构。保留许可证、来源记录和历史提交；本项目不连接原生产系统，不包含患者数据或模型权重。原始资料和历史执行记录中的旧路径仅供追溯，启动以本手册为准。
