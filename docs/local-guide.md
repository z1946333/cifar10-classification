# Windows 本机操作指南

本文用于作者本机操作；项目方法、正式结果及通用复现命令见 [README](../README.md)。

## 打开项目

在 VS Code 中打开 `D:\GitHub\cifar10-classification`，选择解释器 `D:\conda_envs\yahui_torch\python.exe`。

其他电脑需要修改 `.vscode/settings.json` 和 `.cmd` 启动文件中的解释器路径，或在自己的环境中执行 README 的通用命令。

## 双击运行

- `RUN_SMOKE.cmd`：CPU 小规模流程检查，不生成正式测试成绩。
- `START_TRAIN.cmd`：后台正式 GPU 训练，空闲显存不足 1800 MiB 时等待；重复点击不会重复启动由此启动器管理的运行中任务。
- `SHOW_STATUS.cmd`：显示当前任务状态和日志末尾。

保持电脑开机且不休眠。关闭 VS Code 不会停止后台进程，但重启系统会停止任务；当前脚本没有断点续训功能。已有实验不会被覆盖，新启动会建立新的目录。

状态含义：`waiting_for_gpu` 等待显卡；`loading_data` 准备数据；`training` 训练；`evaluating_best_checkpoint` 最终测试；`completed` 完成；`failed` 失败。进程已退出时，遗留的 training 状态不能证明它仍在运行。

## 本次正式结果

30 轮实验已完成，正式测试准确率 92.89%。已完成的结果不需要重复训练，直接查看 [报告](../evaluation/resnet18_20261012_093403_514565/report.md)。

## 更新 GitHub

1. GitHub Desktop 左上角选择 `cifar10-classification`。
2. 在 Changes 中检查需要上传的修改。
3. 填写 Summary，点击 Commit to main。
4. 点击 Push origin，再用 Repository → View on GitHub 检查网页。

数据、日志和其他实验权重已忽略；本次正式实验的 best.pt 已单独加入版本管理，CSV、JSON、曲线及报告也应提交。私有仓库需要给导师访问权限，或按课程要求改为公开。
