# 皮肤镜数据集适配说明

完整盘点与实测见 [DATASET_INVENTORY.md](DATASET_INVENTORY.md)。本文件记录 `isic2018-task3-v1`
的语义与切分规则。

## 1. ISIC 2018 Task 3（`isic2018-task3-v1`，已取得）

- 标签：7 类**单标签互斥**，顺序与官方 GroundTruth CSV 列顺序一致
  `MEL, NV, BCC, AKIEC, BKL, DF, VASC`。
- 每行恰好一个 `1.0`；导入时强校验，出现多正类或非法取值直接拒绝。
- 输出语义：softmax 分布（和为 1），**不是**胸片那种多标签独立 sigmoid；
  不得输出"多项同时超阈值"。
- 许可：CC BY-NC 4.0（HAM10000 / ViDIR Group, Medical University of Vienna），
  仅非商业研究用途；引用须含 HAM10000 论文。
- 切分：
  - `train / val / calibration / test` 由**官方 training 集**按 `lesion_id` 分组切分（70/10/10/10）；
  - 官方 `validation`(193) 与 `test`(1512) 作为**外部留出集**单独报告，不参与训练、校准与阈值选择；
  - validation/test **无 lesion 分组表**，只能按单图分组 → 患者级隔离**不可证明**，必须标注。

## 2. ISIC 2019（备选，未取得）

- 8 类单标签（含 SCC、UNK），训练集 25,331 图，9.10 GB。
- 无官方 LesionGroupings 文件；如改用，须另行论证分组来源，或明确标为"无分组"。

## 3. 禁止的做法

- 把七类当作"七种疾病的确诊"输出；报告须写明"分类概率，确诊需组织病理"。
- 用总体准确率替代逐类指标（NV 占多数会掩盖小类失败）。
- 把皮肤镜模型用于临床照片或其他成像方式。
