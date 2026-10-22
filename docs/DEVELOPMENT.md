> 历史筛选阶段记录，保留追溯。当前医学方向、实施范围和详细任务已由 [重构执行入口](refactor/START_HERE.md) 与 [详细任务书](refactor/TASK_BOOK.md) 明确；下文的“题目未定/后续确定”等描述不再作为当前计划。

# 开发与环境边界

本次目标是代码筛选与保存，尚未完成独立部署重构。前端为 Vue 3/Vite 4，后端为 FastAPI、SQLAlchemy、PyTorch、Ultralytics；保留原依赖声明，不宣称已锁定可复现版本。

创建独立 Python 环境后安装 `backend/requirements.txt`，另装 pytest 才能运行测试。前端使用 `npm ci`。PyTorch/CUDA 版本应按实际开发机单独确定；本仓库没有模型权重或数据集。

`.env.example` 仅列配置名称；复制为 `.env` 后自行设置密钥及独立 PostgreSQL 连接。代码不自动加载所有进程的根目录 `.env`，启动方式需要显式传入环境。空数据库首次初始化要求管理员账号和非默认密码。训练任务还需要独立训练 worker，入口为 `backend/start_training_worker.py`；增强入口为 `backend/start_dataset_augmentation_worker.py`。当前不会仅因启动前端就完成训练任务。

原单体后端仍带额外模块路由和启动钩子，需要按重构计划解除。部分工具使用固定 `/app/static`、`/app/runs` 或相对工作目录；隔离测试提供空临时目录，不代表这些路径已经可移植。现阶段不提供可能误导的生产 Compose 或“一键可部署”承诺。

验证使用完全独立的只读候选源码、临时 SQLite、空数据/输出目录、CPU、无网络；依赖来自已有兼容镜像，结果不能替代新环境安装、PostgreSQL worker、真实训练和 GPU 验收。测试报告与来源清单在 `docs/`。
