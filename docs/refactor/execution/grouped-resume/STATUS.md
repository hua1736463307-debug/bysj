# 分组续作状态（2026-10-23 02:15–03:00 America/Los_Angeles）

> 本轮标识：`grouped-resume-20261023-0215`
> 基线：`55faa94bda046de5b93883b0dddd6b5c31267498`（clean）
> 代码提交：`195d34c0977f44d7e17209e195715d1e62ac070a`（层级修复）、`77a046c`（导入工具）
> 运行项目：`bysj-restore2-20261022`（API 18004 / 网页 18084 / PG 55437）
> 结论：**工程续作 PARTIAL**；分组层级修复、真实导入/冻结/切片、最终备份与恢复、真实 GPU 推理已完成；
> E1/E2 正式训练与 E3 人工审核未完成。

## 阶段状态

| 阶段 | 状态 | 关键证据 |
|---|---|---|
| R00 基线 | PASS | 验证器 35 文件，HEAD/分支/clean 全部一致 |
| R01 Docker/状态/备份 | PASS | 无在途任务、GPU 空；before 备份 complete，48 引用 0 findings |
| R02 层级修复 | PASS | 28 定向 / 175 全量 / 15 PG；新镜像 code_sha=195d34c；migrate exit 0 |
| R03 导入工具 | PASS | `scripts/run_grouped_revision.py`，9 个 MockTransport 测试 |
| R04 真实导入/冻结 | PASS | 364 图 / 4,888 框 / 363 组；291/37/36；verify problems=[] |
| R05 切片 | PASS | child 93b62897…；1746/37/36；val/test 73 未变；group_level=image_cluster |
| R06 事后协议 | **PARTIAL** | E1/E2 prepare-only 通过；80 epoch×3 seeds×2 组未运行 |
| R07 E3 | **PENDING_HUMAN_REVIEW** | 无真实审核人；未生成新待审包（缺新 seed42 模型） |
| R08 最终备份/恢复 | PASS | final 备份 complete；新卷恢复 data_verified；真实 GPU 推理 completed |
| R09 文档 | PASS | R00–R09 + 本文件 + 旧状态入口指针 |

## 关键对象

| 对象 | ID / 摘要 |
|---|---|
| 新数据集 | `77c5c85e-3ea8-4300-9e5a-77c2d914cf95` |
| 新原图版本 | `fd962cbc-451d-47c2-9697-f0ed2530dbcb`（291/37/36，manifest `0f505df6…`） |
| 新切片版本 | `93b62897-ed34-5428-b9da-d1f6e8dae0c4`（1746/37/36） |
| 共同 test 摘要 | `2e58c3252040eaa45a0b6ec3b3e9f46ab1d435446990324084f9de8daac6dec0` |
| 基础模型 | `ebc82da059d24213b1b47965587204b7`，sha `0ebbc80d…44ee1` |
| E1 协议 | `08851496534b3f1fdc52f9f91790c4259a3c73700ee7266fd43797d069c9b039` |
| E2 协议 | `7b126b8731f06d54ccd2e800346db6228af26b60994eb9f65321fe7695977f17` |
| before 备份 | `artifacts/backups/grouped-resume-20261023-0215-before` |
| final 备份 | `artifacts/backups/grouped-resume-20261023-0215-final`（db sha `a5e9c5f6…`） |
| 恢复项目 | `bysj-grouped-restore-20261023-0215`（18005/18085/55438，已停容器、保留卷） |
| 恢复 GPU 推理 | model `630dab9f…`，run completed，inference 42.43 ms，peak GPU 64.2 MiB |

## 未完成与下一动作

1. **E1/E2 正式训练**：6 次 80-epoch 训练超出本轮预算；配置、协议、预检已就绪，直接串行执行 runner。
2. **E3**：等 E1 seed42 模型后再生成新待审包；真实审核前保持 PENDING_HUMAN_REVIEW。
3. **浏览器 E2E**：恢复环境未跑；需独立工程数据，避免把合成复核当质量实验。
4. **长期接管**：当前唯一 GPU 权威仍是源项目；恢复项目仅作验收，容器已停。

## 不得宣称

- 不得写“E1/E2 已完成”“mAP 提升”“test 未见过”；
- 不得把 `profile=formal` 解释为统计学确认性；
- 不得宣称临床有效性或跨患者泛化。
