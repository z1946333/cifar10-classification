# 实验记录

每次运行自动建立独立目录。`smoke_` 开头的记录仅验证程序流程，不能作为正式测试集成绩。

- config.json：实际使用的参数
- environment.json：环境、设备和版本
- source_hashes.json：实验代码的 SHA-256
- split_indices.json：训练、验证样本索引，支持检查数据无重叠
- history.csv：每轮训练和验证指标，准确率为 0 到 1
- curves.png：曲线，每轮刷新
- status.json：运行状态（不提交 Git）

原始数据经 torchvision 官方 MD5 检查。训练增强只作用于训练集。
