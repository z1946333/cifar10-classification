# 实际评估结果

正式训练完成后，自动使用最佳验证模型评估官方 10000 张测试图片。
每次运行有独立的 metrics.json、report.md 和 confusion_matrix.png。
latest.json 指向最近完成的正式实验；如果不存在，则尚无完成的正式实验。

smoke 开头的目录只包含小规模验证集流程检查，不能当成测试集成绩。
