# 公开数据集盘点与选型（2026-10-27）

> 核验原则：**以本机实际 HTTP 探测为准**，不以搜索结果认定"已取得数据"。
> 探测命令见文末第 6 节；每条结论标注"现场核验 / 官方页面文本 / 二手来源"。
> 不绕过访问限制，不代签需要本人同意的条款。

## 0. 结论

| 方向 | 推荐 | 状态 | 理由 |
|---|---|---|---|
| 胸部 X 光 | **NIH ChestX-ray14**（首选）、CheXpert（备选） | **BLOCKED_ACCESS**（本网络无法访问官方 Box；Kaggle 需账号） | 无 DUA、有 Patient ID、规模大；但下载入口在本机不可达 |
| 皮肤镜 | **ISIC 2018 Task 3**（首选）、ISIC 2019（备选） | **已取得（部分）**：全部标签表 + 验证集图像；训练/测试图像受带宽限制未完成 | 官方直链可下载、CC BY-NC 4.0、官方 lesion 分组表、四集可派生 |

**本机条件**：D 盘剩余 85.7 GB、C 盘 197.8 GB；RTX 4060 Laptop 8 GB（容器内 torch 2.6.0+cu124 已验证）；
到 ISIC S3 实测带宽峰值 74 KB/s、持续 9–12 KB/s（决定了下文"能不能下"的结论）。

## 1. 胸部 X 光：NIH ChestX-ray14（首选）

| 项 | 内容 | 核验 |
|---|---|---|
| 官方链接 | 数据：`https://nihcc.app.box.com/v/ChestXray-NIHCC`；论文：`https://arxiv.org/abs/1705.02315`；Kaggle 镜像：`https://www.kaggle.com/datasets/nih-chest-xrays/data` | 论文页 200；Box 两个域名均 **000（代理 502 / DNS 占位）**；Kaggle 网页 200、API 需登录 |
| 数据版本 | ChestX-ray8 论文版（2017）→ 发布版 v2020（`Data_Entry_2017_v2020.csv`） | 论文摘要现场读取；v2020 文件名属二手来源 |
| 任务类型 | 多标签分类（14 个观察项），可做弱监督定位 | 论文摘要（8 类版） |
| 标签及来源 | 报告文本经 NLP 抽取的 14 个观察项；无"不确定"态，只有出现/未出现 | 论文摘要 + 官方说明（二手） |
| 样本量 | 论文版 108,948 图 / 32,717 患者；发布版常引 112,120 图 | 108,948/32,717 为**现场核验**；112,120 为二手 |
| 下载方式 | 官方 12 个 `images_0NN.tar.gz` + CSV，经 Box；或 Kaggle CLI/网页 | 本机两种官方途径均不可行（见上） |
| 账号/协议 | 无 DUA、无需签署；Kaggle 途径需 Kaggle 账号 | 官方发布说明（二手） |
| 许可证/引用 | NIH Clinical Center 发布，研究用途；引用 Wang et al. CVPR 2017 | 论文页 200；许可全文未在本机取得 |
| 文件大小 | 图像约 42–45 GB（12 个 tar 包）；CSV 约 7 MB | **未核验**（入口不可达，无法 HEAD） |
| 分组字段 | `Patient ID`（每行自带），可做患者级隔离 | 二手（CSV 列名），未取得文件 |
| 已知限制 | NLP 标签噪声（原文报告约 90% 准确率量级）；类别极不平衡；无"不确定"态；`No Finding` 与其他标签互斥关系需按官方定义处理 | 论文与社区共识 |
| 适配成本 | **中**：需要新建 NIH 适配器（14 列布尔 → 多标签；`No Finding` 单独处理），四态**不能**复用 CheXpert 语义；`Patient ID` 可直接做患者分组 | 基于现有 CXR 代码评估 |

**为什么仍推荐它**：不需要签署个人协议、有患者 ID、规模足够做三 seed 对比，且与现有 CXR 五标签中
Atelectasis / Cardiomegaly / Consolidation / Edema / Pleural Effusion 五项同名可直接映射（另 9 项保留但不进入首期五标签）。

## 2. 胸部 X 光：CheXpert（备选）

| 项 | 内容 | 核验 |
|---|---|---|
| 官方链接 | `https://stanfordmlgroup.github.io/competitions/chexpert/`；数据入口 Stanford AIMI | 说明页 **200**（全文已读）；AIMI **403**（拒绝 curl） |
| 数据版本 | v1.0（小版 v1.0-small 另发） | 页面文本 |
| 任务类型 | 多标签分类 + 不确定标签；官方竞赛 5 项 | 页面文本 |
| 标签及来源 | 14 个观察项，规则式标注器从报告抽取；`1/0/-1/空白` 四态 | **现场核验**（页面明确写出 0/-1/1/blank） |
| 样本量 | 224,316 张胸片 / 65,240 患者；验证 200 研究（3 位放射科共识）；测试 500 研究 / 500 位未见患者（8 位放射科） | **现场核验** |
| 下载方式 | Stanford AIMI 注册后下载；测试集标签与图像链接在 GitHub | 页面文本 |
| 账号/协议 | **需要注册并接受研究使用条款**（本人操作） | 页面文本；AIMI 入口本机 403 |
| 许可证/引用 | Stanford 研究用途条款；引用 Irvin & Rajpurkar et al. | 页面文本 |
| 文件大小 | 未核验（需登录后可见） | — |
| 分组字段 | 路径含 `patientXXXXX/studyN`，可解析患者与检查 | 既有实现已支持 |
| 已知限制 | 单中心（Stanford）；不确定标签需策略选择；竞赛已关闭 | 页面文本 |
| 适配成本 | **低**：现有 `cxr_multilabel` 适配器即为 CheXpert 四态设计 | 已实现 |

**为什么是备选**：四态语义最贴合现有实现，但需要用户本人注册并接受条款；本机 AIMI 入口 403，无法代劳。

## 3. 皮肤镜：ISIC 2018 Task 3（首选，已取得）

| 项 | 内容 | 核验 |
|---|---|---|
| 官方链接 | `https://challenge.isic-archive.com/data/`；文件直链 `https://isic-archive.s3.amazonaws.com/challenges/2018/...` | 数据页 **200**；直链 **200**（HEAD 有 Content-Length） |
| 数据版本 | ISIC 2018 Challenge, Task 3（HAM10000 派生） | 页面 + `ATTRIBUTION.txt` 现场读取 |
| 任务类型 | **单标签 7 类分类**（互斥），非多标签 | GroundTruth CSV 每行恰好一个 1（已校验 10,015 行） |
| 标签及来源 | 组织病理/随访确认的 7 类：MEL/NV/BCC/AKIEC/BKL/DF/VASC | CSV 表头现场读取 |
| 样本量 | train 10,015；validation 193；test 1,512 | `wc -l` 现场统计 |
| 下载方式 | 官方 S3 直链，**无需账号、无需签署** | 已实际下载标签表与验证集 |
| 许可证/引用 | **CC BY-NC 4.0**（Attribution-NonCommercial 4.0）；须署名 ViDIR Group, Medical University of Vienna 与 HAM10000 论文 | `LICENSE.txt` / `ATTRIBUTION.txt` 现场读取 |
| 文件大小 | train 2.77 GB、val 53 MB、test 421 MB、标签表 <1 MB | HEAD 现场核验 |
| 分组字段 | **官方 `ISIC2018_Task3_Training_LesionGroupings.csv`**：`image, lesion_id, diagnosis_confirm_type`（10,015 行） | 现场读取 |
| 已知限制 | 仅 training 有 lesion 分组，validation/test **无分组**→患者级隔离不可证明；类别极不平衡（NV 占多数）；皮肤镜专用，不适用临床照片；非商业许可 | 已写入注册表 `report_limitations` |
| 适配成本 | **中**：新建单标签适配器 + lesion 分组切分（已实现，见 `app/domains/isic/`） | 已实现 |

**已实际下载**（2026-10-27）：
`ISIC2018_Task3_Training_GroundTruth.zip`、`Training_LesionGroupings.csv`、
`Validation_GroundTruth.zip`、`Test_GroundTruth.zip`、`Validation_Input.zip`（进行中）。
**未完成**：`Training_Input.zip`（2.77 GB）、`Test_Input.zip`（421 MB）——受 9–12 KB/s 带宽限制，
按实测速度需约 10 小时 / 1.6 小时。本轮结束时 `Validation_Input.zip` 已下载 13.6 MB / 53 MB（26%），
`Training_Input.zip` 未开始。

断点续传命令（已下载部分会被 `-C -` 跳过）：

```bash
BASE=https://isic-archive.s3.amazonaws.com/challenges/2018
for f in ISIC2018_Task3_Validation_Input.zip ISIC2018_Task3_Training_Input.zip ISIC2018_Task3_Test_Input.zip; do
  curl -L --retry 3 -C - -o "$f" "$BASE/$f"
done
```

## 4. 皮肤镜：ISIC 2019（备选）

| 项 | 内容 | 核验 |
|---|---|---|
| 官方链接 | 同数据页；`.../2019/ISIC_2019_Training_Input.zip` | 200，Content-Length 9,771,618,190（9.10 GB） |
| 任务类型 | 单标签 8 类（含 SCC 与 UNK） | 二手（2019 规则），未下标签表 |
| 样本量 | 25,331 训练图 | 二手 |
| 分组字段 | 有 `ISIC_2019_Training_Metadata.csv`（1.21 MB，HEAD 200），含 lesion 相关列，但**无官方 LesionGroupings 文件** | 部分核验 |
| 缺点 | 9.10 GB 在本机带宽下约 30 小时；无官方分组表 | 现场核验大小 |

## 5. 其它候选与本机可达性（排除理由）

| 候选 | 本机探测 | 结论 |
|---|---|---|
| MIMIC-CXR / VinDr-CXR（PhysioNet） | `physionet.org` **403** | 需 PhysioNet 认证 + 签署 DUA，排除 |
| PadChest / JSRT | 未探测 | 需机构协议/注册，排除 |
| OpenI (NLM) | API **200** | 可用但标签为 MeSH 术语、样本小，仅作 CXR 应急备选 |
| Hugging Face 镜像 | **000** | 本网络不可达，排除 |
| Academic Torrents | **000** | 排除 |

## 6. 探测命令（可复现）

```bash
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"
curl -sS -o /dev/null -w '%{http_code} %{size_download}\n' -L --max-time 20 -A "$UA" -r 0-1023 <URL>
curl -sSI --max-time 20 <URL> | grep -i content-length
nslookup nihcc.app.box.com     # 返回 2001::1 / 59.24.3.173 → DNS 占位，代理拦截
```

| URL | 结果 |
|---|---|
| `https://challenge.isic-archive.com/data/` | 200 |
| `https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip` | 200，2,771,732,744 B |
| `https://stanfordmlgroup.github.io/competitions/chexpert/` | 200 |
| `https://aimi.stanford.edu/datasets/chexpert-chest-x-rays` | 403 |
| `https://www.nih.gov/news-events/...chest-x-ray-datasets...` | 403 |
| `https://nihcc.app.box.com/v/ChestXray-NIHCC` | 000（CONNECT tunnel 502） |
| `https://nihcc.box.com/v/ChestXray-NIHCC` | 000 |
| `https://physionet.org/files/mimic-cxr-jpg/2.1.0/` | 403 |
| `https://huggingface.co/` | 000 |
| `https://arxiv.org/abs/1705.02315` | 206 |

## 7. 选型建议与备选

1. **胸片**：NIH ChestX-ray14。需要你在能访问 Box 的网络下载，或用 Kaggle 账号执行
   `kaggle datasets download -d nih-chest-xrays/data`；拿到后放入 `DATASET_SOURCES` 指向的只读目录，
   再运行 NIH 适配器（尚未实现，见 `docs/directions/STATUS.md`）。
2. **皮肤镜**：ISIC 2018 Task 3。标签与验证集已到手，训练/测试图像需继续下载（约 10 小时 / 1.6 小时）；
   在带宽更好的环境下载后可立即冻结四集并做真实训练。
3. **若两者都受限**：CXR 侧继续用合成 fixture 做工程验证（明确标注 synthetic），
   皮肤镜侧用已取得的 193 张验证图做真实图像链路 smoke，但**不产生任何论文指标**。
