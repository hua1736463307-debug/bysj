# 数据集卡：BCCD 血细胞检测

## 1. 来源与许可

| 项目 | 值 |
|---|---|
| 仓库 | https://github.com/Shenggan/BCCD_Dataset |
| 固定 revision | `d272fb14cdff6e473fafeeeba32aba5f560e9e43` |
| 许可 | MIT（随仓库 LICENSE，导出的副本见 `data/prepared/bccd/LICENSE.source`） |
| 下载日期 | 2026-10-22（America/Los_Angeles） |
| 获取方式 | `git clone --depth 1`，未使用平台隐式下载 |

## 2. 内容统计（实际转换结果）

| 项目 | 数量 |
|---|---|
| 图像 | 364 |
| 标注框 | 4888 |
| RBC（class 0） | 4155 |
| WBC（class 1） | 372 |
| Platelets（class 2） | 361 |
| train / val / test | 291 / 37 / 36 |

manifest sha256：`1f27a4d97179dfd787e7a8504766a84882ec6453acc1c246062700d409a33ac8`
split_manifest_hash：`87b4200c6b5f50ee8a10f48158fbf329f30362634d21655695db56d51bc1c52a`
完整性检查：364 个文件逐个重算 sha256，无交集（`scripts/check_split_integrity.py` 退出 0）。

续作正式部署通过 v1 导入后，原图父版本为 `8f32546d-11ec-412d-a1c5-2c6aa41a1fa7`，服务器 manifest SHA256 为 `32988ed7845c5933d58a4c139c76b4e6f4e21dce38392c9f1b9b5df233887042`；上面的 manifest 是原始 prepared 文件摘要，两者包含的版本元数据不同。三集划分摘要一致。

E2 训练切片子版本 `a4d2a2b5-2465-58ff-8e20-6126a2e0d3ef` 包含 1746 train / 37 val / 36 test。仅训练原图按 320 边长、0.2 重叠切片，标注最小可见比例 0.5；val/test 图像和标注摘要完全保持。固定原图 test 上使用整图推理。本次对比不评估测试时切片/拼接的收益。

## 3. 类别映射

| class_id | 名称 | XML 名称 |
|---|---|---|
| 0 | RBC | RBC |
| 1 | WBC | WBC |
| 2 | Platelets | Platelets |

未知类别直接拒绝；坐标按 VOC 1-based 左闭/右闭转换为像素 xyxy，再归一化为
`cx,cy,w,h`，转换函数单一（`app/ml/geometry.py`）。

## 4. 来源组与划分

- BCCD 没有患者/载玻片元数据，`group_level=original_image`，每张原图自成一源组；
- 相同内容 hash 自动合并组，跨组重复直接阻止冻结；
- split 按组分配，seed=42，比例 80/10/10（291/37/36）；
- 近重复检测：**未实现**（SHA 不能代替），需人工审查；这是已知限制。

## 5. 已知限制

1. 无患者级隔离，不能宣称跨患者泛化；
2. 图像数量较少，test 仅 36 张；三 seed 标准差只描述训练随机性，未计算患者级置信区间；
3. 单一数据源、单一染色/设备域，不构成临床结论；
4. 结果只表述为研究输出，不用于诊断。
