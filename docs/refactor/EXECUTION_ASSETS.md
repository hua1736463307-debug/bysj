# 执行者需要创建的脚本、配置与文档

这些是交付接口规范，**不是当前已可运行的文件**；对应任务卡创建并验证后，ACCEPTANCE 中的命令才可执行。现有唯一随本任务包新增且已做小型验证的脚本是 `scripts/verify_refactor_baseline.py`。

## 配置规范

`configs/experiments/bccd-baseline.json`（PR-07 创建）：包含 schema_version、source_revision、source_license_ref、class_map、group_level、split_seed、train_ratio/val_ratio/test_ratio、approved_model_id、base_weight_sha256、dataset_version_id、test_manifest_sha256、epochs、batch、imgsz、seeds、deterministic、augmentation、early_stopping、evaluation_protocol。尚未取得的 hash/ID 以 null 表示且正式运行须拒绝；不得把示例 UUID 当有效输入。

默认 class_map 为 `{"0":"RBC","1":"WBC","2":"Platelets"}`；seed 列表默认 `[42,43,44]`；首轮主基线 seed=42；其它实验保持相同划分与基础权重。smoke 配置为独立文件，明确 `purpose=smoke`、1 epoch、小子集；生成结果带 smoke 标签，不进入正式论文统计。

`configs/experiments/performance.json`（PR-09）：base_url、model_id、input_manifest、count=100、concurrency、warmup_count、timeout_seconds、cold_start、device、repeat_count；只保存非敏感配置。密码和 token 不得写入 JSON。

`deploy/.env.example`（PR-00B）：变量名称、单位和说明；DB/JWT/admin password 为空占位，启动时缺值失败。将 `deploy/.env`、运行数据、artifacts、模型、缓存加入忽略列表；保留 .env.example。数据根与 model/artifact 根使用可配置挂载，不复制原服务器地址。

## 脚本职责

| 脚本 | 输入和执行流程 | 输出/退出码 | 阶段 |
|---|---|---|---|
| import_bccd.py | --source 已下载VOC目录、--output 受控新目录、--config；校验源版本/许可记录、逐图/XML转换、来源组划分、hash | manifest.json、labels、dataset.yaml、validation-report.json；0成功，2参数/数据不合规，1执行失败；输出非空不得覆盖 | 07 |
| check_split_integrity.py | --manifest；解析原图/源组/hash/split/派生继承；固定test与父manifest可比 | JSON交集明细、0通过/1发现泄漏/2输入无效；检查实际文件hash不只读清单文本 | 02/07 |
| smoke_medical.py | --base-url 不带/api/v1（脚本统一追加）、--config、--evidence；从env读取两角色凭据；调用真实API并轮询有超时 | 各环节请求id、状态、输入/结果hash、summary.json；无完成则非0；反馈阶段增加回流步骤 | 07/08 |
| run_experiments.py | --config、--output；冻结配置hash，按seed提交任务/评估，收集真实产物；断点重跑以幂等键复用原job | 原始metrics、运行清单、均值标准差、失败表与图；缺必需真实结果非0，不填伪数值 | 07/09 |
| benchmark_medical.py | --config、--output；先warm-up，分冷/热/排队测量，采样CPU/RAM/GPU；保存每请求原始时间 | raw-timings.csv、summary.json、hardware.json；失败率保留，缺采样维度注明而不填0 | 09 |
| 各脚本 --help | 显示参数、默认值、路径边界和必需env名称 | 0；不连网、不加载模型、不创建业务数据 | 对应阶段 |

所有脚本：main guard；输出目录受控；dry-run 如实现须标注；不隐式下载模型/依赖/数据；网络失败不无限重试。数值指标保存量纲和版本；时间戳 ISO8601 含时区。脚本自身失败不能包装为“实验结果为0”。

## 测试 fixtures

PR-02 建 `backend/tests/fixtures/medical/`：可程序生成的彩色矩形图、空标签、合法bbox、越界/零面积/NaN/未知类别标签；3组以上图像及同源派生、同hash改名、冲突源组；恶意ZIP由测试临时生成，勿把真实病人图当fixture。

PR-03：多页草稿与两个用户的隔离数据、两个并发 revision、画布缩放前后基准框。PR-04：独立PostgreSQL、多worker租约和受控可终止子进程。PR-05/06：可信dummy adapter供纯契约测试，另标真实Ultralytics/GPU测试；dummy与真实证据分别统计。PR-08：test派生/改名/同组反馈反例。

## 最终文档落点

- `docs/architecture.md`、`docs/database.md`、`docs/api.md`、`docs/runbook.md`。
- `docs/experiments/dataset-card.md`、`model-card.md`、`experiment-report.md`、`performance-report.md`。
- `docs/thesis/outline.md`：六章内容提纲、每章证据索引、原创/复用/AI贡献说明；学校模板取得后再适配。
- `docs/thesis/defense-demo.md`：8分钟脚本、所用真实数据/模型/任务ID、离线演示准备、故障应对。
- `docs/refactor/execution/PR-XX.md`：每阶段执行记录，不能覆盖本次 PACKET_VALIDATION 的文档准备结果。

外部参考以原始资料为准：本次核对 [BCCD 原仓库](https://github.com/Shenggan/BCCD_Dataset) 的 README 标注 VOC 整理格式与 MIT 许可；正式导入仍固定 revision 并保存许可。训练 seed/deterministic 等参数对照 [Ultralytics 官方训练文档](https://docs.ultralytics.com/modes/train/)，实际兼容性以 PR-00 锁定版本和实测为准。
