# 运行手册

## 目标与验收状态

Linux amd64 / Python 3.12；API 核心依赖、测试依赖和 GPU 依赖分别使用带目标平台的 SHA256 锁。GPU 组合为 torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / Ultralytics 8.4.53，单卡串行领取任务。候选环境通过不能代替正式镜像通过；当前状态见 [续作 STATUS](refactor/execution/unblock/STATUS.md)。

## 首次部署

将 deploy/.env.example 复制为 deploy/.env，填写独立数据库密码、32 字符以上 JWT 密钥、初始管理员密码、BYSJ_CODE_SHA（实际干净提交）、独立卷前缀 BYSJ_VOLUME_PREFIX 及空闲端口。不要复用其他工程的卷。默认端口仅绑定 127.0.0.1。

```sh
docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml config --quiet
docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml --profile workers build
docker compose --env-file deploy/.env -p bysj -f deploy/compose.yml -f deploy/compose.gpu.yml --profile workers up -d
```

启动顺序为 db healthy → migrate upgrade head 成功 → API/worker → frontend。迁移是独立一次性服务；迁移失败时业务不能启动。空库会创建 21 张业务表，重复 upgrade head 应无破坏。不得把未知旧库直接 stamp head。

无 GPU 的导入/标注开发可以仅启动 db/migrate/api/frontend/dataset-worker；真实训练/评估/推理必须在具备可用 NVIDIA 驱动与 Docker GPU 支持的 worker 上运行。正式运行保持 INLINE_JOBS=false；dataset worker 无需安装 Torch。

```sh
curl --fail http://127.0.0.1:8000/api/v1/health/live
curl --fail http://127.0.0.1:8000/api/v1/health/ready
```

`live` 表示 API 进程存活；`ready` 验证数据库和配置要求的 worker 心跳。完整部署应配置 REQUIRED_WORKER_ROLES 为 JSON 数组 ["dataset","trainer","inference"]。浏览器打开 8080 端口，六页面通过同源 /api/v1 访问；Nginx 代理在 deploy/nginx.conf。

## 依赖下载与离线构建

锁必须来自目标 Linux/Python 的完整 pip report（--ignore-installed --only-binary=:all:），不是开发机 pip freeze。scripts/generate_lock.py 校验官方源、目标、冲突、摘要，全部成功后原子写入。CUDA 在线锁保留官方直链。

网络断续时可使用 scripts/download_report_wheels.py --report <report.json> --directory <wheelhouse>。工具按有界 Range 续传并校验完整 SHA256；--nvidia-mirror 只将 NVIDIA 包转至其官方 pypi.nvidia.com，同一摘要不能更改。代理只设置在当前下载进程，不改宿主全局配置。

离线安装要先用 generate_lock.py --wheelhouse <wheelhouse> --offline 生成经过全部 wheel 实物校验的离线锁，再使用 --no-index --find-links --require-hashes。在线 Torch URL 锁即使传 --no-index 仍可能访问直链，不能把这当成离线验收。

## 作业取消和故障恢复

取消由 API 请求、worker 确认；训练子进程停止并回收后才释放 GPU。排队 waiting_gpu 可取消。不要手工将数据库任务改成 completed，也不要仅因 GPU 租约超时便清空 holder。

异常退出后，旧租约进入隔离状态。先确认具体 owner 容器、该容器已经停止且 PID=0，再运行 scripts/recover_gpu_lease.py，默认只检查；传 --apply 才对匹配的 holder/generation/job 做 CAS 恢复。只允许本次 bysj Compose 项目。命令参数见 --help；恢复完按明确任务选择 resume 或 cancel，不批量重置状态。

数据库断连应使计算停止并避免模型注册；恢复数据库后仍需要处理隔离租约。此路径已有真实 GPU 故障注入证据，见 U03。停止服务使用 compose stop；常规停止不使用 down -v，不执行全局 prune。

## 备份与旧库迁移

一致性备份必须暂停新提交、等待在途任务结束并停止 API/worker 写入，同时备份 PostgreSQL、data、models、artifacts 和文件 SHA256 清单。只备份 data 不足以恢复模型和血缘。备份含用户和认证数据，禁止提交 Git。

恢复必须在另一套新卷/数据库进行，核对表计数、版本/模型/资产 hash、登录和已完成记录，再提交一次新推理；恢复前持有的 running/cancelling 作业必须转为可解释状态，不能幽灵运行。实际备份脚本与演练结果以 U08 为准，尚未验收时不声称可恢复。

旧库迁移工具 scripts/migrate_legacy_snapshot.py 目前为经测试的离线拷贝函数：只读 REPEATABLE READ 源、严格识别旧 schema、已 Alembic 初始化且空的目标库。源文件不改写。合成演练通过，真实旧库尚未提供授权副本，不能直接用于未知生产库。

## 常见问题

| 现象 | 排查 |
|---|---|
| ready 503 | 检查 components.database 和缺少角色心跳；live 正常不代表业务就绪 |
| waiting_gpu | 等待共享单 GPU 租约；确认 holder 的任务与隔离状态，勿抢占 |
| ml_dependency_missing | 检查真实 worker 镜像、ML 锁与 pip check |
| 上传 413/422 | 检查数量/尺寸/压缩比、重复 ZIP 路径、标签格式 |
| 标注 409 | 保留本地草稿，明确加载服务器版本后再编辑，不静默覆盖 |
| test_leak_blocked | val/test 及同源资产禁止回流训练 |
| model_tampered | 文件摘要与登记不符，恢复可信备份或重新登记 |
| 子路由 404 | 核对 Nginx try_files 与 /api 代理 |

开发测试见 [DEVELOPMENT](DEVELOPMENT.md)。原始日志/大模型/原图/trace 放忽略的 artifacts 或 .runtime，公开证据只写脱敏统计与 hash。
