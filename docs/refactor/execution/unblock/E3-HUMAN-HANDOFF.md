# E3 人工反馈：证据准备完成，质量实验仍待人工

本轮准备了 12 张固定 train 原图、259 条真实模型预测，使用 E1 seed=42 模型；未创建、提交或批准任何复核记录。样本按文件名排序选取，不使用 test 选样或调阈值。

- 父版本：8f32546d-11ec-412d-a1c5-2c6aa41a1fa7。
- 模型：e98aeb6fb7c746e98eebeec6c99ea640。
- 推理运行：2eb09e2d-c909-46d0-9f60-9f9911c2e2b8。
- 本地材料：artifacts/E3-human-review/README.md、evidence.json、review-worksheet.csv，以及 12 张预测叠图。
- 生成人工材料的可复跑工具：scripts/prepare_feedback_review.py；不自动填审核字段，不重复提交状态不明的请求。

真正审核者需要逐项检查图像、记录正确/误检/漏检或无法判定，填写身份、依据和日期，再按平台流程提交与审批。之后才生成训练侧子版本、核对固定 val/test 摘要并冻结 E3 方案。原始来源标注和预测都只是输入证据，不是人工纠错结论。

E1/E2 的 test 已按各自预登记协议使用，后续 E3 应披露登记时间和测试使用历史，不把它描述为完全未见 test 的首次确认实验。当前 E3 是 **PENDING_HUMAN_REVIEW**，不是 FAIL，也不能写 PASS；功能闭环的合成复核测试与此处真实质量证据严格区分。
