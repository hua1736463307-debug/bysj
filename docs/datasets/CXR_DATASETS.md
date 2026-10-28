# 胸部 X 光数据集适配说明

完整盘点与实测见 [DATASET_INVENTORY.md](DATASET_INVENTORY.md)。本文件只记录**语义规则**，
即"不同胸片数据源的标签如何映射、哪些不能混用"。

## 1. CheXpert（`cxr-five-v1` 四态）

- 来源列：`Atelectasis`、`Cardiomegaly`、`Consolidation`、`Edema`、`Pleural Effusion`。
- 原始取值 `1 / 0 / -1 / 空` 一一映射为 `positive / negative / uncertain / unmentioned`。
- 监督只用 `positive/negative`；`uncertain/unmentioned` 置 mask=0，但**保留原状态**。
- 这是**CheXpert 专属**语义，不得套到其他数据源。

## 2. NIH ChestX-ray14（拟接入，未实现）

- 标签为报告 NLP 抽取的 14 个观察项，**没有 uncertain 态**，只有出现/未出现。
- 因此：
  - 不得把 NIH 的"未出现"直接当作 CheXpert 的 `negative`——两者标注机制不同；
  - 不得把 NIH 数据喂给 `cxr-five-v1` 的四态解析器；
  - 必须新建 `nih-chestxray14-v1` 标签 schema 与独立适配器，`No Finding` 单独处理，
    不得与其他标签同时置正。
- 分组：使用官方 `Patient ID` 做患者级切分；`No Finding` 与其余标签的互斥关系以官方定义为准。
- 首期只映射与 `cxr-five-v1` 同名的五项观察，其余 9 项保留但不进入首期输出。

## 3. 禁止的做法

- 把不同来源的标签拼成同一个"通用疾病列表"；
- 把"未提及"当成阴性；
- 在没有分组字段时声称患者级隔离已证明。
