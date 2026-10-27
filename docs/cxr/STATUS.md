# CXR 五标签辅助分析：实施状态（一页版）

> 更新时间：2026-10-27
> 分支：`graduation/cxr-assist-20261027`（基于 `d22a16d`，未合并 main / 未合并旧 PR）
> 范围声明：[SCOPE.md](SCOPE.md)　逐阶段记录：[execution/](execution/)

## 结论

**工程闭环已实现并通过契约测试；真实数据闭环、研究对比与人工/临床证据均未完成。**

| 验收层级 | 状态 | 依据 |
|---|---|---|
| 工程 MVP | **PASS（工程）** | 67 项 CXR 后端测试 + 7 项 CLI 测试通过；旧 detect 回归 184 项通过；前端 13 项单测 + `npm run build` 通过 |
| 真实数据闭环 | **BLOCKED_REAL_DATA** | 本机无 CheXpert 授权副本；`CXR_IMPORT_ROOT` 未配置；真实训练/校准/评价 NOT_RUN |
| 研究交付（三 seed 对比） | **NOT_RUN** | 无真实数据、无 GPU、torch/torchvision 未安装；runner 与协议冻结已就绪 |
| 人工/临床证据 | **NOT_AVAILABLE** | 无真实审核者；系统仅出"待复核研究草稿" |

不能因为工程测试通过而签收研究或临床结论。

## 已实现（M00–M08）

| 阶段 | 内容 | 关键路径 |
|---|---|---|
| M00 | 新分支、范围声明、旧 E1/E2 标延期 | `docs/cxr/SCOPE.md` |
| M02 | 5 张 CXR 表、`cxr-five-v1` schema、`/api/v1/cxr` 21 个端点、Alembic `4a6b1222f003` | `backend/app/db/models.py`、`schemas/cxr.py`、`api/v1/cxr.py`、`domains/cxr/` |
| M03 | 受控来源导入、患者 component 四集隔离、重复候选合并、冻结 | `domains/cxr/import_service.py`、`split_service.py`、`freeze_service.py`、`scripts/prepare_chexpert.py` |
| M04 | 6 个 cxr job 类型挂到既有三角色 worker，GPU 租约/fencing 复用 | `workers/cxr_handlers.py`、`domains/cxr/runner.py`、`ml/cxr_cli.py` |
| M05 | DenseNet121 五输出 + preprocess-v1 + masked BCE + pos_weight 对比臂 | `ml/cxr_adapter.py`、`ml/cxr_preprocess.py`、`domains/cxr/training_service.py` |
| M06 | 逐标签 AUROC/AUPRC/Brier/ECE、温度缩放、Youden 阈值、bootstrap 区间 | `ml/cxr_metrics.py`、`domains/cxr/evaluation_service.py` |
| M07 | 六个前端页面 + 单一 API client | `frontend/src/views/cxr/`、`api/cxr.js` |
| M08 | 不可变报告快照（JSON/HTML）、Grad-CAM 解释产物 | `domains/cxr/report_service.py`、`analysis_service.py` |

## 验证证据（本机，2026-10-27）

| 项 | 命令 | 结果 |
|---|---|---|
| 初始基线 | `python verify_baseline.py` | `PASS_INITIAL_BASELINE`，49 文件一致 |
| CXR 契约/流程 | `pytest backend/tests/cxr -q` | 67 passed |
| CXR CLI | `pytest scripts/tests/test_cxr_tools.py -q` | 7 passed |
| 旧回归 | `pytest backend/tests/medical backend/tests/test_app_factory_auth.py scripts/tests -q` | 184 passed（无回归） |
| 迁移 | `alembic upgrade head` / `downgrade 4a6b1222f002` / 再 upgrade | 21→27 表；回退只删 5 张 CXR 表，detect 表保留 |
| 前端 | `npm test` / `npm run build` | 13 passed / exit 0 |

未执行：真实 GPU 训练、真实数据校准/评价、100 图性能测试、PG 并发、Docker 镜像构建、浏览器 E2E（`frontend/e2e/cxr-flow.spec.cjs` 已编写但 NOT_RUN）。

## 阻塞与外部待补

| 项 | 状态 | 说明 |
|---|---|---|
| CheXpert 访问与许可 | **BLOCKED_REAL_DATA** | 需用户完成账号/条款；DS 不代签、不绕过门禁 |
| Docker daemon / PG / GPU | **BLOCKED_ENV** | 本会话 `docker version` 无 Server；与 2026-10-27 审查一致 |
| torch 2.6 / torchvision 0.21 | **NOT_INSTALLED** | 本机 venv 无 torch；真实训练/推理不可执行，dummy 仅验证契约 |
| `requirements-cxr` 锁 | **NOT_CREATED** | 未新增第三方依赖（指标改纯 numpy），故未生成新锁；如需 sklearn 需另行 Linux/Py3.12 锁 |
| 真实审核者 | 外部待补 | 系统记录实际操作账号，不冒用医生署名 |
| 临床有效性 / 外部验证 | 未验证 | 不得声称具备临床诊断能力 |

## 与旧工作的边界

- BCCD 数据、模型、备份、代码、结果全部保留；旧风险协议 BLOCKED 仍有效。
- BCCD 分组 E1/E2 标 `DEFERRED_BY_SCOPE_CHANGE`，未追加训练。
- 旧 E4 性能、旧 mAP、旧复核记录**不得**转写为胸片模型性能或人工诊断证据。
- 旧检测链路（BBox 冻结/切片/标注/推理）接口与数据未改义；CXR 版本调用旧 BBox 接口返回 `task_mismatch`。

## 下一步（按优先级）

1. 取得授权数据 → 配置 `CXR_IMPORT_ROOT` → 运行 `prepare_chexpert.py --dry-run` 审计。
2. 恢复 Docker + PG → 迁移演练 → 备份（`scripts/backup_project.py`）。
3. 安装 CXR 运行镜像（torch 2.6 / torchvision 0.21 / CUDA 12.4）→ 真实 2 epoch smoke。
4. smoke 通过后按 `configs/cxr/pilot.json` 冻结协议 → seed 42 闭环 → 43/44 → test 一次性评价。
5. 真实审核者完成复核 → 最终报告 → 全量备份与恢复演练。
