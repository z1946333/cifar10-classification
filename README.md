# CIFAR-10 图像分类实验

使用 PyTorch / torchvision 的 ResNet18 完成 CIFAR-10 十分类。针对 32×32 图像，将第一层改为 3×3、步长 1，并去掉初始最大池化；不使用预训练权重。

## 新手从这里开始

本机 Python 环境：`D:\conda_envs\yahui_torch\python.exe`。

1. 在 VS Code 打开此项目文件夹，选择以上解释器。
2. 双击 `RUN_SMOKE.cmd`：小规模 CPU 流程检查，**不是正式实验结果**。
3. 双击 `START_TRAIN.cmd`：启动后台正式 GPU 训练，若显存不足则排队。重复点击不会启动第二个由此启动器管理的任务。
4. 双击 `SHOW_STATUS.cmd`：查看正在等待、训练到第几轮、是否完成或报错。
5. 完成后在 `evaluation/<实验名称>/report.md` 查看报告；`metrics.json` 为原始指标。

训练期间保持电脑开机且不休眠；关闭 VS Code 不会停止后台任务。系统重启会停止任务。程序保存每轮权重，但当前版本没有断点续训；重新启动会建立新实验。

状态含义：`waiting_for_gpu` 等待显卡；`loading_data` 准备数据；`training` 训练；`evaluating_best_checkpoint` 最终评估；`completed` 完成；`failed` 失败（查看日志）。进程已退出时，遗留状态不能证明任务仍在运行。

## 环境

已使用本机 torch 2.5.0+cu118 / torchvision 0.20.0+cu118。不会修改已有环境。
换电脑时先创建独立 Python 环境，并按显卡驱动选择 PyTorch 版本。例如 CUDA 11.8：

```powershell
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

启动器与 VS Code 设置中的 Python 路径只适用于本机。其他电脑修改路径，或直接使用下面的命令。

## 运行命令（项目根目录）

```powershell
# 下载并验证官方 CIFAR-10，不关闭 HTTPS 证书检查
python -m tools.download_data

# 官方服务器单连接较慢时，可选择分段下载；请勿同时启动两个下载器
python -m tools.download_parallel

# CPU 小规模检查：128 张训练、64 张验证，1 轮，不评估测试集
python -m scripts.train --smoke --device cpu

# 正式 GPU 训练：30 轮，显存不足 1800 MiB 时等待
python -m scripts.train --device cuda --wait-for-gpu

# CPU 正式训练（速度较慢）
python -m scripts.train --device cpu

# 重新评估正式训练的最佳权重（将路径替换成实际实验名称）
python -m scripts.evaluate --checkpoint checkpoints/<实验名称>/best.pt

# 核心测试，不需要下载数据或占用 GPU
python -m unittest discover -s tests -v
```

在 PowerShell 中如果 `python` 指向错误环境，将命令开头的 `python` 换为：
`& 'D:\conda_envs\yahui_torch\python.exe'`。

## 数据与实验设置

- 官方训练集 50000 张，seed=42 固定划分为 45000 训练 / 5000 验证；具体索引会保存。
- 官方测试集 10000 张，只在训练完成、按验证准确率选出最佳模型后评估。
- 训练增强：随机裁剪（padding=4）、水平翻转。验证和测试没有随机增强。
- SGD，初始学习率 0.1，momentum=0.9，weight decay=0.0005，余弦学习率调度。
- batch=64，workers=0，CUDA AMP；参数保存在 `configs/resnet18.json`。
- 交叉熵损失；模型直接输出 logits。
- 原始压缩包与数据文件通过 torchvision 官方 MD5 校验。固定种子不承诺不同设备/版本间逐位一致。

## 目录

```text
configs/       训练配置
data/          下载数据，不提交 Git
datasets/      数据加载代码
docs/          使用和实验说明
evaluation/    真实测试指标、混淆矩阵和自动报告
experiments/   参数、环境、样本划分、每轮 CSV 和曲线
losses/        损失函数说明
models/        CIFAR-10 版 ResNet18
scripts/       训练和测试入口
tools/         下载、绘图、后台启动和状态查看
checkpoints/   最佳和最后权重，不提交 Git
tests/         数据划分、指标计算等回归测试
```

## 结果

此文档不预填准确率。只有正式运行完成后，`evaluation/latest.json` 才指向可报告的完整测试结果。
打开对应 `report.md` 查看准确率、训练时间和图表。所有 `smoke_` 目录都是流程检查，不可作为正式实验成绩。

## 上传 GitHub

1. GitHub Desktop → File → Add local repository，选择本项目目录（它是独立的本地 Git 仓库）。
2. 查看 Changes，提交代码和生成的 CSV/JSON/PNG/报告；数据、权重和日志已忽略。
3. Summary 填写“添加 CIFAR-10 分类实验”，点击 Commit。
4. Publish repository，名称建议 `cifar10-classification`。根据课程要求选择公开或私有。
5. 训练完成后，再提交结果并 Push origin。
6. View on GitHub 打开仓库，复制浏览器地址。私有仓库需要给老师授权。

本程序不会自动上传到 GitHub，也不会停止其他训练进程。

## 参考

- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- torchvision ResNet18: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- ResNet: https://arxiv.org/abs/1512.03385

模型通过已安装的 torchvision 调用，没有把第三方源码复制到本项目。数据集和依赖的使用须遵守各自适用条款。
