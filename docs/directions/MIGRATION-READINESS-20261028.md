# 迁移前补充验证（2026-10-28）

> 目的：在不动运行库的前提下，补齐"迁移方案里被当作已验证、但其实没验过"的部分。
> 所有操作都在隔离项目中进行；运行项目 `bysj-restore2-20261022` 未停止、未迁移、未换镜像。

## 1. 运行基线（只读核对）

| 项 | 值 |
|---|---|
| 项目 | `bysj-restore2-20261022`，6 容器 Up |
| api / dataset-worker | `bysj-api:grouped-resume-20261023-0215` = `sha256:bbc2418b04b3…` |
| worker / inference-worker | `bysj-worker:grouped-resume-20261023-0215` = `sha256:e129b551c0b4…` |
| frontend / db | `bysj-frontend:20261022` = `sha256:fb022afabd00…`；`postgres:16-alpine` = `sha256:57c72fd2a128…` |
| 代码 SHA | `BYSJ_CODE_SHA=195d34c0977f44d7e17209e195715d1e62ac070a` |
| 数据库版本 | `4a6b1222f002` |
| 在途任务 / GPU 租约 | 0 / `gpu:0` holder 为空（generation 68） |
| 健康 | `/health/live` 200、`/health/ready` 200 |
| 磁盘 | D: 剩余约 60 GB |

## 2. 候选镜像（ddce908，仅隔离使用）

| 项 | api | worker |
|---|---|---|
| 镜像 | `bysj-api:candidate-ddce908` | `bysj-worker:candidate-ddce908` |
| 镜像 ID | `sha256:e9e88b31f18fc751f3eaf4efaf80be3c691dc1ef362d6a6cecc70fa82e299e63` | `sha256:d84039e38b19c59f427d6e70ed4ed614d614c487b70fb62dcbc0185856720fa3` |
| 自声明 SHA | `ddce908377eaa47d68e7671ddb617a32e138ac8f`（`version.json.code_sha` 一致） | 同 |

**限制**：`ddce908` **不包含**随后的 ISIC 训练链路（`63bff5f`）。最终迁移候选必须按最终提交重建，
本镜像只能证明"该提交可构建、可自声明 SHA"，不能宣称包含后续代码。

## 3. 旧 BCCD 功能验证（隔离项目 `bysj-drill2-f004-20261027`）

只读接口返回 200 不算功能验证，因此实际提交了一次推理作业并检查了落库结果。

| 项 | 内容 |
|---|---|
| 数据 | `dataset_assets` id 1–3（`BloodImage_00000/1/2.jpg`），版本 `6579bad5-cf2f-4576-b7b6-bde3c5b37960` |
| 模型 A | `c478ab94d6c44b7691b79fd75132ebfd`（`gpu-smoke-base`，sha `0ebbc80d4a76…`） |
| 模型 B | `ff87503ec5164736a0e176408c07c412`（训练产物，sha `a214959717df…`） |
| 环境 | GPU 已挂载（`cuda_available=True`），`gpu:0` 租约空闲 |
| 作业 | `inference` × 3 次，均 `completed`，3/3 项成功、0 失败 |
| 结果 A | **71 个预测框**（`bbox_xyxy` + `confidence`），但 `class_id=54`、`class_name=null` |
| 结果 B | **0 个预测框**（conf 降到 0.01 仍为 0） |
| 租约 | 作业结束后 `gpu:0` holder 为空，generation 68 → 70 |

**结论（必须如实记录）**：

1. **推理链路本身可用**：作业调度、GPU 租约、预测落库、API 回读都正常，`bbox_xyxy` 与置信度是真实数值。
2. **旧 BCCD 训练权重退化**：容器内直接用 ultralytics 复现，`best.pt`（训练产物）在 conf=0.01 下也是 0 框，
   说明问题在权重本身（旧训练轮次/子集不足），不是适配器或阈值。
3. **元数据不一致**：`gpu-smoke-base` 的 `class_map_json` 写的是 `{RBC, WBC, Platelets}`，
   但权重文件是 COCO `yolo11n`（names 为 person/bicycle/…，54=donut），因此 `class_name` 为 null。
   这是**备份里既有的数据问题**，本次未修改。
4. 因此"旧 BCCD 功能"只能验证到**管线级**（作业→GPU→预测→落库→回读），
   **无法用现有权重验证"输出 BCCD 三类框"这一语义级功能**。

## 4. PostgreSQL 迁移与降级演练（隔离库）

在 `bysj-drill-f004-20261027` 的库上实际执行（该库此前为 f004，含 14 条合成记录）。

| 步骤 | 结果 |
|---|---|
| 降级前 | f004、28 表、旧表行数基线已记录 |
| `alembic downgrade 4a6b1222f002` | 成功：`f004 → f003 → f002` |
| 降级后 | 22 表、6 张新表消失、**22 张旧表行数逐表零差异**、外键约束 0 |
| `alembic upgrade head` | 成功：`f002 → f003 → f004`，28 表 |
| 旧数据 | `dataset_assets` 5,269 行（含 14 条合成资产），未受影响 |
| **新表数据** | `cxr_samples` / `isic_samples` 均为 **0 行** → 降级**不可逆地丢弃新表数据** |
| **残留** | 14 条 `dataset_assets`（`transform_json.synthetic=true`）与 14 个 `syn_drill_*.png` 文件仍留在库和卷里，但对应的样本行已随表删除 → **孤立资产** |

**结论**：

- `downgrade` 在 **PostgreSQL 上已验证可用**，只删 6 张新表，不动旧表数据。
- 但降级会**永久丢失 CXR/ISIC 记录**，并留下 `dataset_assets` 与数据卷文件的**孤立残留**（不会自动清理）。
- 因此**回滚仍以「用迁移前备份恢复到新项目」为首选**；"降级后切回旧镜像"只能作为
  在"新表确实为空、且接受残留资产"前提下的次要方案，且必须同时清理孤立资产，否则需人工核对。

## 5. 备份访问控制与加密方案

现状：`D:/毕业设计/backups/20261027-before-cxr`（997 MB）与 `20261027-drill-f004`（997 MB）
**未加密、仅存本机、未纳入访问控制**，内含私有数据（数据集文件、模型权重、用户与会话行）。

**已完成**：

- 备份输出目录位于 Git 之外（`.runtime/`、`artifacts/`、`backups/` 均被忽略），不会误提交。
- 每个备份自带 `backup.json`：数据库 SHA-256、三卷归档 SHA-256、逐文件清单与哈希、行数、迁移版本。
- 恢复脚本在恢复前**强制校验**上述哈希，不一致直接拒绝。

**建议（未执行，需你决定）**：

| 措施 | 做法 | 说明 |
|---|---|---|
| 访问控制 | 备份目录 ACL 仅当前用户可读写；`icacls` 移除继承 | 阻止同机其他账户读取 |
| 加密副本 | `age`/`gpg` 对称加密后另存（密钥不入库、不与备份同盘） | 加密后哈希需重新记录，恢复流程要增加解密步骤 |
| 异地保存 | 复制到第二块物理盘或受控对象存储（私有桶、服务端加密） | 避免单盘故障 |
| 保留策略 | 迁移前备份保留 ≥ 7 天；演练备份可随时重建 | 明确删除条件 |
| 校验 | 每份备份恢复演练一次（已有 `restore_project.py` 可直接用） | 防止"备份存在但不可恢复" |

**不做**：不上传 GitHub、不进镜像、不写入仓库。

## 6. 新版本 GPU 作业、取消与租约释放

在含 ISIC 训练链路的镜像（`bysj-api:isic-63bff5f` / `bysj-worker:isic-63bff5f`，代码 `63bff5f`）上，
于隔离项目 `bysj-drill2-f004-20261027` 实测：

| 检查 | 结果 |
|---|---|
| GPU 作业可运行 | `isic_train` 1 epoch 完成，真实 DenseNet121，显存峰值 1.211 GB |
| 取消 | 2 epoch 作业运行中取消 → `status=cancelled` |
| 租约释放 | `gpu:0` holder 由 `trainer-026033b063f3…` 变为空 |
| 半成品模型 | 该作业登记模型数 0 |
| 子进程 | 取消后容器内无 `isic_cli` 残留进程，显存回到系统基线 |

**发现并修复的缺陷**：`ctx.should_stop()` 收到取消时抛 `JobCancelled` 而非返回 True，
使 `child_runner` 的"stop.flag → SIGTERM → 宽限"分支永不执行，子进程被 SIGKILL 直接杀掉。
已修正并加 `backend/tests/test_child_runner.py`（3 项）。

细节见 `docs/directions/ISIC-SMOKE-20261028.md` 第 5 节。

## 7. 迁移候选镜像的口径

| 用途 | 镜像 | 代码 |
|---|---|---|
| 当前候选（仅隔离验证） | `bysj-api:candidate-ddce908` / `bysj-worker:candidate-ddce908` | `ddce908` |
| 含 ISIC 训练链路（smoke 用） | `bysj-api:isic-63bff5f` / `bysj-worker:isic-63bff5f` | `63bff5f` |

**最终迁移候选必须按最终提交重新构建并重新验证**，不能把上述任一镜像当作"包含最终代码"。
构建命令与验收见 `MIGRATION-PLAN-f004.md` 第 5 节。
